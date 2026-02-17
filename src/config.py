"""Configuration for the OpenAI Agents SDK supervisor and subagents."""

from datetime import datetime
import os
from typing import Any

from pydantic import BaseModel, ConfigDict

# Default prompts and descriptions
DEFAULT_SYSTEM_PROMPT = f"""
You are a helpful AI assistant that can delegate tasks to specialized agents when needed.

You have access to the following specialized agents:
- Research Agent: For web searches and finding information online
- Math Agent: For mathematical calculations and arithmetic

IMPORTANT INSTRUCTIONS:
- For simple greetings, small talk, or general conversational responses, respond directly yourself
- ALWAYS delegate to the Research Agent for:
  * Factual questions about real-world events, people, places, or statistics
  * Questions asking "who", "what", "when", "where" about specific facts
  * Historical records, achievements, or data points
  * ANY question where accurate, verified information is important
  * Questions that could benefit from current or verified information
- Delegate to the Math Agent for:
  * Queries requiring calculations with specific numbers
  * Statistical or quantitative methodology questions (e.g., mean, variance, standard deviation, regression)
  * Step-by-step mathematical procedures, even when no concrete numbers are provided
- For domain-coupled quantitative questions (research/study context + math/statistics), DO NOT answer directly.
  * Delegate to one specialist first (usually Research Agent for context/definitions/current practices),
    then rely on specialist-to-specialist handoff for the quantitative procedure.
- Delegate using the available handoff tools when specialized work is needed
- Use at most ONE handoff tool call per turn
- For compound requests that need both research and math, delegate to one specialist first; that specialist can hand off to the other specialist if needed
- When in doubt about whether to research something, USE THE RESEARCH AGENT - it's better to verify facts than to rely on potentially outdated information
- For compound questions, your final response MUST include:
  * The key factual value(s) found
  * The calculation result
  * A concise explanation linking them
- For research-backed answers, include at least one source URL in the final response

IMPORTANT INFORMATION:
- The current date is {datetime.now().strftime("%Y-%m-%d")}.

In order to complete the objective that the user asks of you, you have access to specialized agents.
"""

DEFAULT_RESEARCH_AGENT_DESCRIPTION = (
    "Research agent with web search capabilities. "
    "Use this agent for: web searches, finding information online, "
    "looking up current events, researching topics, gathering data from the internet, "
    "answering questions that require external knowledge or real-time information."
)

DEFAULT_MATH_AGENT_DESCRIPTION = (
    "Math calculation agent with arithmetic tools. "
    "Use this agent for: mathematical calculations, arithmetic operations, "
    "addition, subtraction, multiplication, division, numerical computations, "
    "solving math problems, performing calculations."
)

DEFAULT_RESEARCH_AGENT_PROMPT = (
    "You are a research agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with research-related tasks, DO NOT do any math\n"
    "- If a task requires a math computation after research, hand off to the Math Agent once with the computed numeric inputs\n"
    "- Use at most ONE handoff tool call per turn\n"
    "- Provide links to sources of your information in the response\n"
    "- If no additional handoff is needed, provide a concise factual answer with source URLs\n"
    "- When returning values needed for downstream math, include both the factual context and the raw numeric values."
)

DEFAULT_MATH_AGENT_PROMPT = (
    "You are a math agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with math-related tasks\n"
    "- If a task is missing a factual value, hand off to the Research Agent once to fetch it\n"
    "- Use at most ONE handoff tool call per turn\n"
    "- If no additional handoff is needed, provide a concise answer that includes both the calculation and the final numeric result\n"
    "- For compound tasks, preserve factual context in the final answer (do not return only a bare number)."
)

# Default model names
DEFAULT_SUPERVISOR_MODEL = "gpt-4o-mini"
DEFAULT_RESEARCH_MODEL = "gpt-4o-mini"
DEFAULT_MATH_MODEL = "gpt-4o-mini"


def _content_to_text(content: Any) -> str:
    """Normalize prompt content variants into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    return str(content)


def _extract_prompt_text(prompt_obj: Any) -> str:
    """Extract supervisor instructions from a Braintrust prompt object."""
    prompt_block = getattr(prompt_obj, "prompt", None)
    if prompt_block is None:
        return ""

    prompt_type = getattr(prompt_block, "type", None)
    if prompt_type == "completion":
        return _content_to_text(getattr(prompt_block, "content", ""))

    if prompt_type == "chat":
        messages = getattr(prompt_block, "messages", None) or []
        system_chunks: list[str] = []
        first_chunk = ""

        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", "")).lower()
                content = _content_to_text(message.get("content"))
            else:
                role = str(getattr(message, "role", "")).lower()
                content = _content_to_text(getattr(message, "content", None))

            if not content:
                continue
            if not first_chunk:
                first_chunk = content
            if role == "system":
                system_chunks.append(content)

        if system_chunks:
            return "\n\n".join(system_chunks).strip()
        return first_chunk

    return ""


class AgentConfig(BaseModel):
    """Configuration for the supervisor and subagents.

    All fields are optional with sensible defaults.
    """

    # Supervisor/System prompt
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    supervisor_prompt_slug: str | None = os.environ.get(
        "BRAINTRUST_SUPERVISOR_PROMPT_SLUG"
    )
    supervisor_prompt_version: str | None = os.environ.get(
        "BRAINTRUST_SUPERVISOR_PROMPT_VERSION"
    )
    supervisor_prompt_project: str | None = os.environ.get(
        "BRAINTRUST_SUPERVISOR_PROMPT_PROJECT", os.environ.get("BRAINTRUST_PROJECT")
    )
    supervisor_prompt_project_id: str | None = os.environ.get(
        "BRAINTRUST_SUPERVISOR_PROMPT_PROJECT_ID"
    )

    # Subagent prompts
    research_agent_prompt: str = DEFAULT_RESEARCH_AGENT_PROMPT
    math_agent_prompt: str = DEFAULT_MATH_AGENT_PROMPT

    # Subagent routing descriptions (used by SubAgentMiddleware)
    research_agent_description: str = DEFAULT_RESEARCH_AGENT_DESCRIPTION
    math_agent_description: str = DEFAULT_MATH_AGENT_DESCRIPTION

    # Model selections
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL
    research_model: str = DEFAULT_RESEARCH_MODEL
    math_model: str = DEFAULT_MATH_MODEL

    # Runtime metadata (set during prompt resolution)
    resolved_supervisor_prompt_source: str = "local_default"
    resolved_supervisor_prompt_slug: str | None = None
    resolved_supervisor_prompt_version: str | None = None
    resolved_supervisor_prompt_id: str | None = None

    def resolve_supervisor_prompt(self) -> str:
        """Resolve supervisor instructions from Braintrust prompt slug or local default."""
        slug = (self.supervisor_prompt_slug or "").strip()
        version = (self.supervisor_prompt_version or "").strip() or None
        if not slug:
            self.resolved_supervisor_prompt_source = "local_default"
            self.resolved_supervisor_prompt_slug = None
            self.resolved_supervisor_prompt_version = None
            self.resolved_supervisor_prompt_id = None
            return self.system_prompt

        try:
            from braintrust import load_prompt

            prompt_obj = load_prompt(
                project=self.supervisor_prompt_project,
                project_id=self.supervisor_prompt_project_id,
                slug=slug,
                version=version,
                api_key=os.environ.get("BRAINTRUST_API_KEY"),
                org_name=os.environ.get("BRAINTRUST_ORG_NAME"),
            )
            resolved_text = _extract_prompt_text(prompt_obj)
            if not resolved_text:
                raise ValueError(
                    "Loaded Braintrust prompt but could not extract non-empty text."
                )

            self.resolved_supervisor_prompt_source = "braintrust_prompt"
            self.resolved_supervisor_prompt_slug = slug
            self.resolved_supervisor_prompt_version = str(
                getattr(prompt_obj, "version", self.supervisor_prompt_version) or ""
            ) or None
            self.resolved_supervisor_prompt_id = getattr(prompt_obj, "id", None)
            return resolved_text
        except Exception as exc:
            print(
                f"Warning: failed to load Braintrust supervisor prompt slug={slug!r}: {exc}. "
                "Falling back to local system_prompt."
            )
            self.resolved_supervisor_prompt_source = "local_fallback_after_prompt_load_error"
            self.resolved_supervisor_prompt_slug = slug
            self.resolved_supervisor_prompt_version = version
            self.resolved_supervisor_prompt_id = None
            return self.system_prompt

    def supervisor_prompt_trace_metadata(self) -> dict[str, Any]:
        """Emit trace metadata describing which supervisor prompt source was used."""
        return {
            "supervisor_prompt_source": self.resolved_supervisor_prompt_source,
            "supervisor_prompt_slug": self.resolved_supervisor_prompt_slug,
            "supervisor_prompt_version": self.resolved_supervisor_prompt_version,
            "supervisor_prompt_id": self.resolved_supervisor_prompt_id,
        }

    model_config = ConfigDict(arbitrary_types_allowed=True)
