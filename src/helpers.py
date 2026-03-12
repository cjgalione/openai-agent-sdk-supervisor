"""Runtime helpers for OpenAI Agents SDK runs and eval serialization."""

from __future__ import annotations

import json
from typing import Any

from agents import ItemHelpers


def _extract_text(content: Any) -> str | None:
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue

            if not isinstance(item, dict):
                continue

            if isinstance(item.get("text"), str) and item["text"].strip():
                parts.append(item["text"].strip())
                continue

            nested = _extract_text(item.get("content"))
            if nested:
                parts.append(nested)

        if parts:
            return "\n".join(parts)
        return None

    if isinstance(content, dict):
        if isinstance(content.get("text"), str) and content["text"].strip():
            return content["text"].strip()
        return _extract_text(content.get("content"))

    return None


def _extract_query_from_messages(messages: list[Any]) -> str | None:
    # Prefer the latest user message with text content.
    for message in reversed(messages):
        if isinstance(message, str):
            text = message.strip()
            if text:
                return text
            continue

        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "") or "")
        if role and role not in {"user", "system"}:
            continue

        text = _extract_text(message.get("content"))
        if text:
            return text

    # Fallback: any message-like item with content.
    for message in messages:
        if isinstance(message, dict):
            text = _extract_text(message.get("content"))
            if text:
                return text
    return None


def extract_query_from_input(input_payload: Any) -> str:
    """Extract a user query from eval input payloads."""
    if isinstance(input_payload, str):
        text = input_payload.strip()
        if text:
            return text

    if isinstance(input_payload, list):
        extracted = _extract_query_from_messages(input_payload)
        if extracted:
            return extracted

    if isinstance(input_payload, dict):
        query_text = _extract_text(input_payload.get("query"))
        if query_text:
            return query_text

        messages = input_payload.get("messages")
        if isinstance(messages, list):
            extracted = _extract_query_from_messages(messages)
            if extracted:
                return extracted

        nested_input = input_payload.get("input")
        if nested_input is not None and nested_input is not input_payload:
            extracted = _extract_text(nested_input)
            if extracted:
                return extracted
            try:
                return extract_query_from_input(nested_input)
            except ValueError:
                pass

    raise ValueError(
        f"Could not extract user query from input payload (type={type(input_payload).__name__})"
    )


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
