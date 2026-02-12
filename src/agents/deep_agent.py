"""OpenAI Agents SDK supervisor with handoffs to specialized subagents."""

from agents import Agent, handoff

from src.agents.math_agent import get_math_agent
from src.agents.research_agent import get_research_agent
from src.config import AgentConfig


def get_deep_agent(config: AgentConfig | None = None) -> Agent:
    """Create the supervisor agent and wire handoffs to subagents."""
    resolved_config = config or AgentConfig()

    research_agent = get_research_agent(
        system_prompt=resolved_config.research_agent_prompt,
        model=resolved_config.research_model,
    )
    math_agent = get_math_agent(
        system_prompt=resolved_config.math_agent_prompt,
        model=resolved_config.math_model,
    )

    return Agent(
        name="Supervisor Agent",
        model=resolved_config.supervisor_model,
        instructions=resolved_config.system_prompt,
        handoffs=[
            handoff(
                agent=research_agent,
                tool_name_override="delegate_to_research_agent",
                tool_description_override=resolved_config.research_agent_description,
            ),
            handoff(
                agent=math_agent,
                tool_name_override="delegate_to_math_agent",
                tool_description_override=resolved_config.math_agent_description,
            ),
        ],
    )


_cached_deep_agent: Agent | None = None


def get_supervisor(config: AgentConfig | None = None, force_rebuild: bool = False) -> Agent:
    """Get a cached or newly built supervisor agent."""
    global _cached_deep_agent

    if config is not None:
        return get_deep_agent(config)

    if force_rebuild or _cached_deep_agent is None:
        _cached_deep_agent = get_deep_agent()
    return _cached_deep_agent
