"""Parameter definitions for Braintrust evals."""

from typing import Any

from braintrust.prompt import PromptChatBlock, PromptData, PromptMessage
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_MATH_AGENT_PROMPT,
    DEFAULT_MATH_MODEL,
    DEFAULT_RESEARCH_AGENT_PROMPT,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SUPERVISOR_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)

class PromptModificationParam(BaseModel):
    """Append-only prompt modification parameter for supervisor and subagents."""

    value: str = Field(
        default="",
        description=(
            "Optional append-only modification applied to supervisor, research, and math "
            "agent prompts. Use this for output-style or policy tweaks without replacing "
            "the full base prompts."
        ),
    )


class SupervisorModelParam(BaseModel):
    """Supervisor model selection parameter."""

    value: str = Field(
        default=DEFAULT_SUPERVISOR_MODEL,
        description="Model to use for the supervisor agent (e.g., gpt-4o-mini, gpt-4o).",
    )


class ResearchModelParam(BaseModel):
    """Research model selection parameter."""

    value: str = Field(
        default=DEFAULT_RESEARCH_MODEL,
        description="Model to use for the research agent (e.g., gpt-4o-mini, gpt-4o).",
    )


class MathModelParam(BaseModel):
    """Math model selection parameter."""

    value: str = Field(
        default=DEFAULT_MATH_MODEL,
        description="Model to use for the math agent (e.g., gpt-4o-mini, gpt-4o).",
    )


def _chat_prompt_default(prompt_text: str, model: str) -> dict[str, Any]:
    """Build a Braintrust prompt-object default for Playground editing."""
    return PromptData(
        prompt=PromptChatBlock(
            messages=[PromptMessage(role="system", content=prompt_text)]
        ),
        options={"model": model},
    ).as_dict()


SystemPromptParam = {
    "type": "prompt",
    "name": "supervisor_prompt",
    "default": _chat_prompt_default(DEFAULT_SYSTEM_PROMPT, DEFAULT_SUPERVISOR_MODEL),
    "description": "Supervisor prompt object, including its model selection.",
}

ResearchAgentPromptParam = {
    "type": "prompt",
    "name": "research_agent_prompt",
    "default": _chat_prompt_default(
        DEFAULT_RESEARCH_AGENT_PROMPT,
        DEFAULT_RESEARCH_MODEL,
    ),
    "description": "Research agent prompt object, including its model selection.",
}

MathAgentPromptParam = {
    "type": "prompt",
    "name": "math_agent_prompt",
    "default": _chat_prompt_default(
        DEFAULT_MATH_AGENT_PROMPT,
        DEFAULT_MATH_MODEL,
    ),
    "description": "Math agent prompt object, including its model selection.",
}
