"""Supervisor evaluation for the OpenAI Agents SDK implementation."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

# Ensure project root is on sys.path so `src` package can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents import RunConfig, Runner, set_trace_processors  # noqa: E402
from braintrust import Eval, init_dataset, init_function, init_logger  # noqa: E402
from braintrust.oai import wrap_openai  # noqa: E402
from braintrust.wrappers.openai import BraintrustTracingProcessor  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from evals.braintrust_parameter_patch import apply_parameter_patch  # noqa: E402
from evals.parameters import (  # noqa: E402
    MathAgentPromptParam,
    MathModelParam,
    ResearchAgentPromptParam,
    ResearchModelParam,
    SUPERVISOR_PROMPT_UI_PARAM,
    SupervisorModelParam,
    SupervisorPromptSlugParam,
    SupervisorPromptVersionParam,
    SystemPromptParam,
)
from src.agents.deep_agent import get_supervisor  # noqa: E402
from src.config import AgentConfig  # noqa: E402
from src.helpers import (  # noqa: E402
    extract_query_from_input,
    serialize_run_result,
)

load_dotenv()
apply_parameter_patch()

DEFAULT_BRAINTRUST_PROJECT = "openai-agent-sdk-supervisor"
DEFAULT_BRAINTRUST_DATASET = "OpenAI Agent SDK Supervisor Dataset"

client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


def unwrap_parameters(params: dict) -> dict:
    """Extract raw parameter values from Braintrust parameter objects."""
    import inspect

    from pydantic import BaseModel

    result: dict[str, Any] = {}
    for key, param in params.items():
        if param is None:
            continue

        if inspect.isclass(param) and issubclass(param, BaseModel):
            param_instance = param()
            result[key] = getattr(param_instance, "value", param_instance)
        elif isinstance(param, BaseModel):
            result[key] = getattr(param, "value", param)
        else:
            result[key] = param

    return result


def _extract_system_prompt_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    first_nonempty: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            text = "\n".join(parts).strip()
        else:
            text = str(content).strip() if content is not None else ""
        if not text:
            continue
        if first_nonempty is None:
            first_nonempty = text
        if role == "system":
            return text
    return first_nonempty


def _extract_ui_prompt_selection(prompt_param: Any) -> tuple[str | None, dict[str, str]]:
    """Parse optional `supervisor_prompt` UI parameter into text/slug metadata."""
    if prompt_param is None:
        return None, {}

    metadata: dict[str, str] = {}
    prompt_data: Any = prompt_param

    # Braintrust Prompt objects can expose `.build()`.
    build_fn = getattr(prompt_param, "build", None)
    if callable(build_fn):
        try:
            built = build_fn()
            prompt_data = built if built is not None else prompt_param
        except Exception:
            prompt_data = prompt_param

    # Surface slug/version/id when present so task can prefer latest by slug.
    for key in ("slug", "version", "id"):
        value = None
        if isinstance(prompt_param, dict):
            value = prompt_param.get(key)
        else:
            value = getattr(prompt_param, key, None)
        if value:
            metadata[f"ui_prompt_{key}"] = str(value)

    if isinstance(prompt_data, dict):
        # Common built chat prompt shape.
        text = _extract_system_prompt_from_messages(prompt_data.get("messages"))
        if text:
            return text, metadata

        # Fallback if prompt data is nested.
        nested = prompt_data.get("prompt")
        if isinstance(nested, dict):
            text = _extract_system_prompt_from_messages(nested.get("messages"))
            if text:
                return text, metadata
            content = nested.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip(), metadata

    return None, metadata


async def run_supervisor_task(input: dict, hooks: Any = None) -> dict[str, list]:
    """Run a single task through the supervisor and return serialized messages."""
    try:
        params = hooks.parameters if hooks and hasattr(hooks, "parameters") else {}
        config_params = unwrap_parameters(params)
        ui_prompt_param = params.get("supervisor_prompt") if isinstance(params, dict) else None
        ui_prompt_text, ui_prompt_meta = _extract_ui_prompt_selection(ui_prompt_param)

        # Not an AgentConfig field; handled separately.
        config_params.pop("supervisor_prompt", None)

        # Precedence:
        # 1) UI selected saved prompt slug -> load latest via load_prompt
        # 2) UI inline prompt text
        # 3) explicit slug/version config or default local prompt
        ui_slug = ui_prompt_meta.get("ui_prompt_slug")
        if ui_slug:
            config_params["supervisor_prompt_slug"] = ui_slug
            config_params["supervisor_prompt_version"] = ""
        elif ui_prompt_text:
            config_params["system_prompt"] = ui_prompt_text

        config = AgentConfig(**config_params) if config_params else None

        supervisor = get_supervisor(config=config, force_rebuild=True)
        query = extract_query_from_input(input)

        trace_metadata = {"eval_type": "supervisor"}
        if config is not None:
            trace_metadata.update(config.supervisor_prompt_trace_metadata())
        trace_metadata.update(ui_prompt_meta)

        result = await Runner.run(
            starting_agent=supervisor,
            input=query,
            run_config=RunConfig(
                workflow_name="openai-agent-sdk-supervisor-eval-supervisor",
                trace_metadata=trace_metadata,
            ),
        )
        serialized_messages = serialize_run_result(result, user_query=query)

        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update(
                {
                    "final_output": str(getattr(result, "final_output", "")),
                    "num_messages": len(serialized_messages),
                }
            )

        return {"messages": serialized_messages}
    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"messages": [{"error": str(e)}]}


class RoutingAccuracyOutput(BaseModel):
    """Structured output for routing accuracy evaluation."""

    choice: Literal["A", "B", "C", "D"]
    reasoning: str


ROUTING_ACCURACY_PROMPT = """
You are an expert evaluator of AI agent routing systems. Your task is to determine whether a user question was correctly routed to the appropriate agents.

The system has the following specialized agents:
1. **MathAgent**: Should handle mathematical calculations, arithmetic, equations, numerical problems, and any query requiring computation with specific numbers.
2. **ResearchAgent**: Should handle factual questions, information lookup, current events, geography, history, statistics, and any query requiring external knowledge or web search.

The supervisor can:
- Route to a single agent
- Route to multiple agents (if the query requires both research and math)
- Answer directly without routing (for simple greetings, conversational queries, or ambiguous questions)

**User Question**: {input}

**Agents Called**: {agents_called}

**Task**: Evaluate the routing decision and respond with your reasoning, then select ONE of these options:

(A) CORRECT
(B) MOSTLY_CORRECT
(C) PARTIALLY_WRONG
(D) INCORRECT
"""


def _infer_agents_from_tool_name(tool_name: str) -> set[str]:
    lowered = tool_name.lower()
    agents: set[str] = set()

    if any(
        key in lowered
        for key in (
            "research",
            "tavily",
            "delegate_to_research_agent",
            "request_research_subtask",
        )
    ):
        agents.add("ResearchAgent")

    if any(
        key in lowered
        for key in (
            "math",
            "delegate_to_math_agent",
            "request_math_subtask",
            "add",
            "subtract",
            "multiply",
            "divide",
        )
    ):
        agents.add("MathAgent")

    return agents


async def _collect_agents_called(trace: Any, output: Any) -> list[str]:
    """Infer called agents from trace spans and serialized tool call messages."""
    found: set[str] = set()

    # Primary source: spans (task/function/llm) emitted during the run.
    spans: list[Any] = []
    try:
        spans = await trace.get_spans(span_type=["task", "function", "llm"])
    except Exception:
        spans = []

    for span in spans:
        span_name = str(getattr(span, "span_attributes", {}).get("name", "") or "")
        lowered = span_name.lower()
        if span_name in {"MathAgent", "ResearchAgent"}:
            found.add(span_name)
        else:
            found.update(_infer_agents_from_tool_name(lowered))

    # Fallback source: serialized run output includes tool call names.
    if isinstance(output, dict):
        messages = output.get("messages", [])
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tool_name = str(tc.get("name", "") or "")
                    found.update(_infer_agents_from_tool_name(tool_name))

    ordered = [name for name in ["ResearchAgent", "MathAgent"] if name in found]
    return ordered


async def routing_accuracy_scorer(input, output, expected, metadata, trace):
    choice_map = {"A": 1.0, "B": 0.7, "C": 0.3, "D": 0.0}
    agents_called = await _collect_agents_called(trace, output)

    agents_called_str = (
        ", ".join(agents_called)
        if agents_called
        else "None (supervisor answered directly)"
    )

    prompt = ROUTING_ACCURACY_PROMPT.format(
        input=input,
        agents_called=agents_called_str,
    )
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        text_format=RoutingAccuracyOutput,
    )
    parsed = response.output_parsed
    if parsed is None:
        return {
            "name": "Routing Accuracy",
            "score": 0.0,
            "metadata": {
                "agents_called": agents_called_str,
                "reasoning": "No output",
                "choice": "D",
            },
        }

    return {
        "name": "Routing Accuracy",
        "score": choice_map.get(parsed.choice, 0.0),
        "metadata": {
            "agents_called": agents_called_str,
            "reasoning": parsed.reasoning,
            "choice": parsed.choice,
        },
    }


response_quality_prompt = """
You are an expert evaluator of AI assistant responses.

User Question: {{input}}
AI Response: {{output}}

Evaluate the response based on:
1. ACCURACY
2. COMPLETENESS
3. CLARITY
4. RELEVANCE

Scoring guidance:
- For pure arithmetic questions, a concise correct numeric answer is acceptable.
- For compound questions that ask for both a factual lookup and a calculation,
  the response must include both the factual answer and the computed result.
- Do not mark a response incorrect merely for brevity if it fully answers the question.

Respond with:
EXCELLENT
GOOD
FAIR
POOR
"""

class ResponseQualityOutput(BaseModel):
    """Structured output for response quality scoring."""

    choice: Literal["EXCELLENT", "GOOD", "FAIR", "POOR"]
    reasoning: str


async def response_quality_scorer(input, output, expected, metadata, trace):
    """Score response quality with structured output parsing."""
    del expected, metadata, trace

    messages = output.get("messages", []) if isinstance(output, dict) else []
    assistant_response = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            assistant_response = str(msg["content"])
            break

    if isinstance(input, dict):
        try:
            normalized_input = extract_query_from_input(input)
        except Exception:
            normalized_input = str(input)
    else:
        normalized_input = str(input)

    prompt = response_quality_prompt.replace("{{input}}", normalized_input).replace(
        "{{output}}",
        assistant_response or str(output),
    )

    response = client.responses.parse(
        model=os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o"),
        input=[{"role": "user", "content": prompt}],
        text_format=ResponseQualityOutput,
    )
    parsed = response.output_parsed
    if parsed is None:
        return {
            "name": "Response Quality",
            "score": 0.0,
            "metadata": {"choice": "POOR", "reasoning": "No parsed output"},
        }

    score_map = {"EXCELLENT": 1.0, "GOOD": 0.75, "FAIR": 0.5, "POOR": 0.0}
    return {
        "name": "Response Quality",
        "score": score_map.get(parsed.choice, 0.0),
        "metadata": {"choice": parsed.choice, "reasoning": parsed.reasoning},
    }


async def step_efficiency_scorer(output):
    """Score based on number of serialized messages.

    In some remote-eval contexts Braintrust may pass a plain string output,
    so we defensively normalize non-dict outputs.
    """
    max_steps = 8
    if isinstance(output, dict):
        num_steps = len(output.get("messages", []))
    elif isinstance(output, str):
        # Treat a plain final response as a single-step completion.
        num_steps = 1 if output.strip() else 0
    else:
        num_steps = 0

    if num_steps <= max_steps:
        return 1.0
    return max(0.0, 1.0 - (num_steps - max_steps) / max_steps)


def load_local_dataset() -> list[dict[str, Any]]:
    """Load eval data from local dataset.jsonl for deterministic local execution."""
    dataset_path = project_root / "dataset.jsonl"
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_eval_data(project_name: str):
    """Choose local dataset by default, with optional remote dataset override."""
    use_remote = os.environ.get("BRAINTRUST_USE_REMOTE_DATASET", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if use_remote:
        return init_dataset(
            project=project_name,
            name=os.environ.get("BRAINTRUST_DATASET", DEFAULT_BRAINTRUST_DATASET),
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            org_name=os.environ.get("BRAINTRUST_ORG_NAME", "Braintrust Demos"),
        )
    return load_local_dataset()


project_name = os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT)
logger = init_logger(
    project=project_name,
    api_key=os.environ.get("BRAINTRUST_API_KEY"),
    org_name=os.environ.get("BRAINTRUST_ORG_NAME", "Braintrust Demos"),
)
set_trace_processors([BraintrustTracingProcessor(logger)])

use_published_step_scorer = (
    os.environ.get("USE_PUBLISHED_STEP_SCORER", "1").lower() in {"1", "true", "yes"}
)
step_efficiency_score = (
    init_function(project_name=project_name, slug="step-efficiency")
    if use_published_step_scorer
    else step_efficiency_scorer
)

Eval(
    project_name,
    experiment_name="supervisor",
    data=get_eval_data(project_name),
    task=run_supervisor_task,
    scores=[
        response_quality_scorer,
        routing_accuracy_scorer,
        step_efficiency_score,
    ],  # type: ignore
    parameters={
        "system_prompt": SystemPromptParam,
        "supervisor_prompt_slug": SupervisorPromptSlugParam,
        "supervisor_prompt_version": SupervisorPromptVersionParam,
        "research_agent_prompt": ResearchAgentPromptParam,
        "math_agent_prompt": MathAgentPromptParam,
        "supervisor_model": SupervisorModelParam,
        "research_model": ResearchModelParam,
        "math_model": MathModelParam,
    },
)

# Separate eval variant that exposes a prompt-type supervisor parameter in UI.
Eval(
    project_name,
    experiment_name="supervisor-prompt-ui",
    data=get_eval_data(project_name),
    task=run_supervisor_task,
    scores=[
        response_quality_scorer,
        routing_accuracy_scorer,
        step_efficiency_score,
    ],  # type: ignore
    parameters={
        "supervisor_prompt": SUPERVISOR_PROMPT_UI_PARAM,
        "supervisor_prompt_slug": SupervisorPromptSlugParam,
        "supervisor_prompt_version": SupervisorPromptVersionParam,
        "supervisor_model": SupervisorModelParam,
        "system_prompt": SystemPromptParam,
    },
)
