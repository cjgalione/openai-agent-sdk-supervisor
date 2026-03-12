import os

import pytest

os.environ["BRAINTRUST_DISABLE_AUTO_EVAL"] = "1"

from evals.eval_supervisor import run_supervisor_task, unwrap_parameters
from evals.parameters import PromptModificationParam
from src.config import AgentConfig


class _Hooks:
    def __init__(self, parameters):
        self.parameters = parameters
        self.metadata = {}


class _FakeResult:
    def __init__(self, final_output: str):
        self.final_output = final_output
        self.new_items = []


def test_render_prompts_append_modification_for_all_agents():
    config = AgentConfig(prompt_modification="Rispondi sempre in italiano.")

    supervisor_prompt = config.render_supervisor_prompt()
    research_prompt = config.render_research_prompt()
    math_prompt = config.render_math_prompt()

    assert "Rispondi sempre in italiano." in supervisor_prompt
    assert "Rispondi sempre in italiano." in research_prompt
    assert "Rispondi sempre in italiano." in math_prompt

    assert "core routing/safety constraints" in supervisor_prompt
    assert "core role/safety constraints" in research_prompt
    assert "core role/safety constraints" in math_prompt


def test_render_prompts_unchanged_when_no_modification():
    config = AgentConfig()

    assert config.render_supervisor_prompt() == config.system_prompt
    assert config.render_research_prompt() == config.research_agent_prompt
    assert config.render_math_prompt() == config.math_agent_prompt


def test_unwrap_parameters_extracts_single_value_models():
    params = {"prompt_modification": PromptModificationParam(value="in italiano")}

    unwrapped = unwrap_parameters(params)

    assert unwrapped["prompt_modification"] == "in italiano"


@pytest.mark.asyncio
async def test_run_supervisor_task_forwards_prompt_modification(monkeypatch):
    captured = {}

    def fake_get_supervisor(config=None, force_rebuild=False):
        captured["config"] = config
        captured["force_rebuild"] = force_rebuild
        return object()

    async def fake_runner_run(starting_agent, input, run_config):
        captured["input"] = input
        return _FakeResult(final_output="Ciao dal test.")

    monkeypatch.setattr("evals.eval_supervisor.get_supervisor", fake_get_supervisor)
    monkeypatch.setattr("evals.eval_supervisor.Runner.run", fake_runner_run)

    hooks = _Hooks(parameters={"prompt_modification": "Return every final answer in Italian."})
    result = await run_supervisor_task({"query": "hello"}, hooks=hooks)

    assert captured["force_rebuild"] is True
    assert captured["config"].prompt_modification == "Return every final answer in Italian."
    assert captured["input"] == "hello"
    assert result["messages"][-1]["content"] == "Ciao dal test."
    assert hooks.metadata["final_output"] == "Ciao dal test."
