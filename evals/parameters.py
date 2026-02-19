"""
Parameter definitions for Braintrust evals.

These parameters can be configured in the Braintrust playground UI.
Uses single-field Pydantic models with a 'value' field, which the
patched Braintrust SDK will unwrap properly.
"""

from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_MATH_AGENT_PROMPT,
    DEFAULT_MATH_MODEL,
    DEFAULT_RESEARCH_AGENT_PROMPT,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SUPERVISOR_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)

# Define parameters as single-field Pydantic models
# The patched SDK will extract the 'value' field's schema and default


class SystemPromptParam(BaseModel):
    """System prompt parameter for supervisor agent."""

    value: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="Custom system prompt for the supervisor agent.",
    )


class ResearchAgentPromptParam(BaseModel):
    """Research agent prompt parameter."""

    value: str = Field(
        default=DEFAULT_RESEARCH_AGENT_PROMPT,
        description="Custom system prompt for the research agent.",
    )


class MathAgentPromptParam(BaseModel):
    """Math agent prompt parameter."""

    value: str = Field(
        default=DEFAULT_MATH_AGENT_PROMPT,
        description="Custom system prompt for the math agent.",
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


class SupervisorPromptSlugParam(BaseModel):
    """Braintrust prompt slug for supervisor instructions."""

    value: str = Field(
        default="",
        description=(
            "Optional Braintrust prompt slug for supervisor instructions. "
            "When set, evals load prompt text from Braintrust via load_prompt()."
        ),
    )


class SupervisorPromptVersionParam(BaseModel):
    """Optional Braintrust prompt version for supervisor instructions."""

    value: str = Field(
        default="",
        description=(
            "Optional Braintrust prompt version (xact ID or version) to pin "
            "the supervisor prompt loaded from slug."
        ),
    )


SUPERVISOR_PROMPT_UI_PARAM = {
    "type": "prompt",
    "description": (
        "Optional prompt selector for supervisor instructions. "
        "If a saved prompt slug is selected, eval loads latest version by slug."
    ),
    "default": {
        "messages": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT.strip()}],
        "model": DEFAULT_SUPERVISOR_MODEL,
    },
}
