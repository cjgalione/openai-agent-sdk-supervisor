#!/usr/bin/env python3
"""Run a single supervisor query with configurable routing/eval tracing options."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import braintrust
from agents import RunConfig, Runner, set_trace_processors
from braintrust import api_conn, init_logger
from braintrust.wrappers.openai import BraintrustTracingProcessor
from dotenv import load_dotenv

# Add project root to path so local imports work when run from scripts/.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_graph import get_supervisor
from src.config import AgentConfig
from src.helpers import extract_query_from_input, serialize_run_result

DEFAULT_PROJECT = "openai-agent-sdk-supervisor"
DEFAULT_ORG_NAME = "Braintrust Demos"
DEFAULT_SUPERVISOR_MODEL = "gpt-4.1-mini"


# ----------------------------
# Metadata parsing helpers
# ----------------------------
def _coerce_value(value: str) -> Any:
    """Best-effort parse for metadata values (JSON primitives/objects)."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_metadata(
    metadata_json: str | None,
    metadata_kv: list[str] | None,
) -> dict[str, Any]:
    """Parse metadata from JSON blob + repeated key=value pairs."""
    metadata: dict[str, Any] = {}

    if metadata_json:
        parsed = json.loads(metadata_json)
        if not isinstance(parsed, dict):
            raise ValueError("--trace-metadata-json must be a JSON object")
        metadata.update(parsed)

    for pair in metadata_kv or []:
        if "=" not in pair:
            raise ValueError(f"Invalid --trace-metadata entry (expected key=value): {pair}")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --trace-metadata entry (empty key): {pair}")
        metadata[key] = _coerce_value(raw_value)

    return metadata


# ----------------------------
# Trace context loading
# ----------------------------
def _extract_text_from_message_content(content: Any) -> str | None:
    """Extract text from common message content encodings."""
    if isinstance(content, str):
        text = content.strip()
        return text if text else None

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
            # Handle alternate chat-part schema variants.
            if item.get("type") == "text" and isinstance(item.get("content"), str):
                raw = item["content"].strip()
                if raw:
                    parts.append(raw)
        joined = " ".join(parts).strip()
        return joined if joined else None

    return None


def _extract_query_from_trace_input(trace_input: Any) -> str:
    """Extract user query from common root span input shapes."""
    if isinstance(trace_input, list):
        for item in trace_input:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role != "user":
                continue
            text = _extract_text_from_message_content(item.get("content"))
            if text:
                return text
        raise ValueError("Could not find user content in list-shaped trace input")

    if isinstance(trace_input, dict):
        return extract_query_from_input(trace_input)

    if isinstance(trace_input, str) and trace_input.strip():
        return trace_input

    raise ValueError("Unsupported trace input shape for query extraction")


def _resolve_project_id(project_name: str, org_name: str, api_key: str) -> str:
    """Resolve project UUID from project name via Braintrust logger metadata."""
    logger = init_logger(project=project_name, api_key=api_key, org_name=org_name)
    return logger.project.id


def _load_root_span_context(
    *,
    trace_id: str,
    project_name: str,
    org_name: str,
    api_key: str,
) -> dict[str, Any]:
    """Fetch root span row for a trace and return extracted context."""
    braintrust.login(api_key=api_key, org_name=org_name)
    project_id = _resolve_project_id(project_name, org_name, api_key)
    btql = (
        f"from: project_logs('{project_id}') "
        f"| filter: root_span_id='{trace_id}' and is_root=true "
        "| select: input, metadata, span_attributes "
        "| limit: 1"
    )
    rows = api_conn().post_json("btql", {"query": btql}).get("data", [])
    if not rows:
        raise ValueError(f"Trace not found or has no root span in project: {trace_id}")

    row = rows[0]
    trace_input = row.get("input")
    metadata = row.get("metadata") or {}
    span_attributes = row.get("span_attributes") or {}
    query_text = _extract_query_from_trace_input(trace_input)

    selected_model = None
    if isinstance(metadata, dict):
        selected_model = metadata.get("selected_model")

    workflow_name = None
    if isinstance(span_attributes, dict):
        workflow_name = span_attributes.get("name")

    return {
        "query": query_text,
        "selected_model": selected_model,
        "workflow_name": workflow_name,
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


# ----------------------------
# Main run logic
# ----------------------------
async def _run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("BRAINTRUST_API_KEY", "")
    trace_context: dict[str, Any] = {}
    if args.trace_id:
        if not api_key:
            raise RuntimeError(
                "BRAINTRUST_API_KEY is required when using --trace-id to fetch trace context."
            )
        trace_context = _load_root_span_context(
            trace_id=args.trace_id,
            project_name=args.project,
            org_name=args.org_name,
            api_key=api_key,
        )

    query = args.query or trace_context.get("query")
    if not query:
        raise ValueError("Provide --query, or use --trace-id that resolves to a root user query.")

    model_hint = trace_context.get("selected_model")
    supervisor_model = args.supervisor_model or model_hint or DEFAULT_SUPERVISOR_MODEL
    research_model = args.research_model or supervisor_model
    math_model = args.math_model or supervisor_model

    workflow_name = args.workflow_name or trace_context.get("workflow_name") or "routing-retest"

    metadata = _parse_metadata(args.trace_metadata_json, args.trace_metadata)
    if args.trace_id:
        metadata.setdefault("source_trace_id", args.trace_id)
    metadata.setdefault("selected_model", supervisor_model)

    logger = None
    if not args.no_braintrust:
        if not api_key:
            raise RuntimeError(
                "BRAINTRUST_API_KEY is missing. Set it in environment/.env or use --no-braintrust."
            )
        logger = init_logger(
            project=args.project,
            api_key=api_key,
            org_name=args.org_name,
        )
        set_trace_processors([BraintrustTracingProcessor(logger)])
    else:
        set_trace_processors([])

    config = AgentConfig(
        supervisor_model=supervisor_model,
        research_model=research_model,
        math_model=math_model,
    )
    supervisor = get_supervisor(config=config, force_rebuild=True)

    try:
        result = await Runner.run(
            starting_agent=supervisor,
            input=query,
            run_config=RunConfig(
                workflow_name=workflow_name,
                trace_metadata=metadata,
            ),
        )

        messages = serialize_run_result(result, user_query=query)
        print(f"FINAL: {getattr(result, 'final_output', '')}")
        print("MESSAGES:")
        print(json.dumps(messages, indent=2, ensure_ascii=False))
    finally:
        if logger is not None:
            logger.flush()
            print("Traces flushed to Braintrust.")


# ----------------------------
# CLI args
# ----------------------------
def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run one retest query through the supervisor.")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        help="User query to run through the supervisor.",
    )
    query_group.add_argument(
        "--trace-id",
        default=None,
        help=(
            "Root trace ID to replay. The script fetches the original root user query "
            "and metadata from Braintrust."
        ),
    )
    parser.add_argument(
        "--supervisor-model",
        default=None,
        help=(
            "Model for supervisor agent. If omitted with --trace-id, uses trace "
            "metadata selected_model when available, else gpt-4.1-mini."
        ),
    )
    parser.add_argument(
        "--research-model",
        default=None,
        help="Model for research agent (default: same as --supervisor-model).",
    )
    parser.add_argument(
        "--math-model",
        default=None,
        help="Model for math agent (default: same as --supervisor-model).",
    )
    parser.add_argument(
        "--workflow-name",
        default=None,
        help=(
            "Workflow name to log in traces. If omitted with --trace-id, uses the "
            "source root span name."
        ),
    )
    parser.add_argument(
        "--trace-metadata-json",
        default=None,
        help='JSON object for trace metadata, e.g. \'{"retest_case":"mean_variance_climate"}\'.',
    )
    parser.add_argument(
        "--trace-metadata",
        action="append",
        default=[],
        help="Repeatable key=value trace metadata entries.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_PROJECT),
        help=f"Braintrust project name (default: env BRAINTRUST_PROJECT or {DEFAULT_PROJECT}).",
    )
    parser.add_argument(
        "--org-name",
        default=os.environ.get("BRAINTRUST_ORG_NAME", DEFAULT_ORG_NAME),
        help=f"Braintrust org name (default: env BRAINTRUST_ORG_NAME or {DEFAULT_ORG_NAME}).",
    )
    parser.add_argument(
        "--no-braintrust",
        action="store_true",
        help="Disable Braintrust tracing/logging for this run.",
    )
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
