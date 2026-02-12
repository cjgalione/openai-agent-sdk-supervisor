"""Runtime helpers for OpenAI Agents SDK runs and eval serialization."""

from __future__ import annotations

import json
from typing import Any

from agents import ItemHelpers


def extract_query_from_input(input_payload: dict[str, Any]) -> str:
    """Extract a user query from eval input payloads."""
    if "query" in input_payload and input_payload["query"]:
        return str(input_payload["query"])

    messages = input_payload.get("messages", [])
    if isinstance(messages, list) and messages:
        first_message = messages[0]
        if isinstance(first_message, dict):
            content = first_message.get("content")
            if isinstance(content, str):
                return content

    raise ValueError("Could not extract user query from input payload")


def _parse_args(raw_args: Any) -> Any:
    if raw_args is None:
        return {}
    if isinstance(raw_args, (dict, list)):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return raw_args
    return str(raw_args)


def _serialize_item(item: Any) -> dict[str, Any] | None:
    item_type = getattr(item, "type", "")

    if item_type == "message_output_item":
        return {"role": "assistant", "content": ItemHelpers.text_message_output(item)}

    if item_type == "tool_call_item":
        raw_item = getattr(item, "raw_item", None)
        tool_name = ""
        args: Any = {}

        if isinstance(raw_item, dict):
            tool_name = str(raw_item.get("name", ""))
            args = _parse_args(raw_item.get("arguments"))
        elif raw_item is not None:
            tool_name = str(getattr(raw_item, "name", ""))
            args = _parse_args(getattr(raw_item, "arguments", None))

        if not tool_name:
            tool_name = str(getattr(item, "name", ""))

        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": tool_name, "args": args}],
        }

    if item_type == "tool_call_output_item":
        output_value = getattr(item, "output", None)
        if output_value is None:
            output_value = getattr(item, "raw_output", None)
        return {
            "role": "tool",
            "content": output_value if isinstance(output_value, str) else str(output_value),
        }

    return None


def serialize_run_result(result: Any, user_query: str | None = None) -> list[dict[str, Any]]:
    """Convert a RunResult object into JSON-serializable messages."""
    messages: list[dict[str, Any]] = []

    if user_query:
        messages.append({"role": "user", "content": user_query})

    for item in getattr(result, "new_items", []) or []:
        serialized = _serialize_item(item)
        if serialized is not None:
            messages.append(serialized)

    final_output = getattr(result, "final_output", None)
    if isinstance(final_output, str) and final_output.strip():
        has_assistant_output = any(
            msg.get("role") == "assistant" and msg.get("content")
            for msg in messages
        )
        if not has_assistant_output:
            messages.append({"role": "assistant", "content": final_output})

    return messages
