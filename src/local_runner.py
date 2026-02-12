"""Local CLI runner for the OpenAI Agents SDK supervisor."""

import asyncio
import getpass
import os
import uuid

from agents import RunConfig, Runner, SQLiteSession, set_trace_processors
from braintrust import init_logger
from braintrust.wrappers.openai import BraintrustTracingProcessor
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.agent_graph import get_supervisor

DEFAULT_BRAINTRUST_PROJECT = "openai-agent-sdk-supervisor"


def _set_if_undefined(var: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


async def _run_chat() -> None:
    load_dotenv()
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")

    console = Console()
    logger = None

    if os.environ.get("BRAINTRUST_API_KEY"):
        logger = init_logger(
            project=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            org_name=os.environ.get("BRAINTRUST_ORG_NAME", "Braintrust Demos"),
        )
        set_trace_processors([BraintrustTracingProcessor(logger)])
    else:
        set_trace_processors([])

    supervisor = get_supervisor()
    session = SQLiteSession(session_id=f"local-{uuid.uuid4().hex}")

    welcome_text = Text("OpenAI Agents SDK Supervisor Chat", style="bold cyan")
    welcome_panel = Panel(
        welcome_text,
        subtitle="Type 'quit' or 'q' to exit",
        border_style="cyan",
    )
    console.print(welcome_panel)
    console.print()

    try:
        while True:
            user_input = Prompt.ask("[bold green]You[/bold green]", console=console)

            if user_input.lower() in {"q", "quit", "exit"}:
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break

            if not user_input.strip():
                continue

            with console.status("[bold blue]Processing...", spinner="dots"):
                result = await Runner.run(
                    starting_agent=supervisor,
                    input=user_input,
                    session=session,
                    run_config=RunConfig(
                        workflow_name="openai-agent-sdk-supervisor-local",
                        trace_metadata={"surface": "local_runner"},
                    ),
                )

            final_output = getattr(result, "final_output", "")
            console.print(
                Panel(
                    str(final_output) if final_output else "(No response generated)",
                    title="Assistant",
                    border_style="blue",
                )
            )
            console.print()
    finally:
        if logger is not None:
            logger.flush()


def main() -> None:
    asyncio.run(_run_chat())


if __name__ == "__main__":
    main()
