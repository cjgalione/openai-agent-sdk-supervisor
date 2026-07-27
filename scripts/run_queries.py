#!/usr/bin/env python3
"""Generate test questions and run them through the supervisor concurrently."""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

from agents import RunConfig, Runner, set_default_openai_client, set_trace_processors
from braintrust import init_logger
from braintrust.wrappers.openai import BraintrustTracingProcessor
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

DEFAULT_BRAINTRUST_PROJECT = "openai-agent-sdk-supervisor"
DEFAULT_BRAINTRUST_GATEWAY_URL = "https://gateway.braintrust.dev"

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import AgentConfig

load_dotenv()

MODEL_POOL = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
QUESTION_BANK = [
    "What is 37 * 24?",
    "Who won the first modern Olympic Games and in what year?",
    "If a supernova releases 10^44 joules, how many 60W lightbulb-hours is that?",
    "What's the capital of Japan and what is 18% of 250?",
    "When was the Eiffel Tower completed?",
    "What is the population of Canada and what is 2% of that number?",
    "Can you summarize what a quasar is in one sentence?",
    "Who discovered penicillin and in what year?",
    "What is (48 + 72) / 6?",
    "What recent developments has OpenAI announced?",
]


def _init_braintrust_logger():
    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        set_trace_processors([])
        return None

    logger = init_logger(
        project=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
        api_key=api_key,
        org_name=os.environ.get("BRAINTRUST_ORG_NAME", "Braintrust Demos"),
    )
    try:
        # init_logger resolves project metadata lazily; force it now so auth/config
        # failures fail the scheduled job before query results are silently dropped.
        _ = logger.project.id
    except Exception as exc:
        raise RuntimeError(
            "BRAINTRUST_API_KEY is set, but Braintrust logger initialization failed. "
            "Refresh the GitHub Actions BRAINTRUST_API_KEY secret."
        ) from exc

    set_trace_processors([BraintrustTracingProcessor(logger)])
    return logger


def _openai_client() -> OpenAI:
    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        raise RuntimeError("Missing BRAINTRUST_API_KEY in environment")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("BRAINTRUST_GATEWAY_URL", DEFAULT_BRAINTRUST_GATEWAY_URL),
    )


def _configure_gateway_client() -> None:
    """Route OpenAI Agents SDK calls through the Braintrust Demos gateway."""
    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        return
    set_default_openai_client(
        AsyncOpenAI(
            api_key=api_key,
            base_url=os.environ.get("BRAINTRUST_GATEWAY_URL", DEFAULT_BRAINTRUST_GATEWAY_URL),
        ),
        use_for_tracing=False,
    )


def _fallback_questions(num_questions: int, rng: random.Random) -> list[str]:
    questions = QUESTION_BANK.copy()
    rng.shuffle(questions)
    if num_questions <= len(questions):
        return questions[:num_questions]
    result: list[str] = []
    while len(result) < num_questions:
        result.extend(questions[: num_questions - len(result)])
        rng.shuffle(questions)
    return result


def _preflight_failure_category(exc: Exception) -> str:
    text = str(exc).lower()
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return "quota"
    if any(
        marker in text
        for marker in ("authentication", "invalid api key", "incorrect api key", "unauthorized", "401")
    ):
        return "authentication"
    if any(marker in text for marker in ("429", "timeout", "connection", "temporarily")):
        return "transient"
    return "provider"


def _run_preflight() -> dict[str, str]:
    missing = [
        name
        for name in ("BRAINTRUST_API_KEY", "EXA_API_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    from src.agents.research_agent import _search_exa

    for attempt in range(1, 4):
        try:
            _openai_client().responses.create(
                model="gpt-4o-mini",
                input="Reply with exactly: OK",
            )
            _search_exa(query="Braintrust", max_results=1)
            return {"braintrust": "ok", "model": "ok", "exa": "ok"}
        except Exception as exc:
            category = _preflight_failure_category(exc)
            if category == "transient" and attempt < 3:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"Provider preflight failed ({category}).") from exc

    raise RuntimeError("Provider preflight failed (transient).")


def _write_summary(
    path: str | None,
    *,
    preflight: dict[str, str],
    total: int,
    successes: int,
    failures: int,
) -> None:
    if not path:
        return
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "preflight": preflight,
                "total": total,
                "successes": successes,
                "failures": failures,
            },
            indent=2,
        )
        + "\n"
    )


def generate_questions(num_questions: int, seed: Optional[int] = None) -> list[str]:
    """Generate realistic, varied questions with natural language variation."""
    rng = random.Random(seed)
    client = _openai_client()

    prompt = f"""Generate exactly {num_questions} realistic user questions that test an AI multi-agent system.

Create a diverse mix of:
- Pure math questions
- Pure research questions
- Hybrid questions (research + math)
- Edge cases (ambiguous, conversational, frustrated)

Output requirements:
- Return ONLY a valid JSON array of strings
- No markdown, no explanation
- Keep each question under 200 characters
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
    )
    text = response.output_text.strip()

    try:
        questions = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Question generator returned non-JSON output: {text[:300]}") from exc

    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise RuntimeError("Question generator did not return a JSON array of strings")

    rng.shuffle(questions)
    return questions[:num_questions]


async def run_question(question: str) -> tuple[str, bool]:
    """Run one question through the supervisor with a random model assignment."""
    from src.agent_graph import get_supervisor

    selected_model = random.choice(MODEL_POOL)
    config = AgentConfig(
        supervisor_model=selected_model,
        research_model=selected_model,
        math_model=selected_model,
    )
    supervisor = get_supervisor(config=config, force_rebuild=True)

    try:
        result = await Runner.run(
            starting_agent=supervisor,
            input=question,
            run_config=RunConfig(
                workflow_name="openai-agent-sdk-supervisor-batch",
                trace_metadata={
                    "source": "daily_job",
                    "job_name": "run_queries",
                    "customer_id": f"customer_{random.randint(1000, 9999)}",
                    "selected_model": selected_model,
                },
            ),
        )
        print(f"✅ {question[:80]} -> {str(getattr(result, 'final_output', ''))[:80]}")
        return question, True
    except Exception as exc:
        print(f"❌ {question[:80]} -> {exc}")
        return question, False


async def main_async(args: argparse.Namespace) -> None:
    preflight = {} if args.skip_preflight else _run_preflight()
    if args.preflight_only:
        _write_summary(
            args.summary_path,
            preflight=preflight,
            total=0,
            successes=0,
            failures=0,
        )
        print("Provider preflight passed.")
        return

    num_questions = args.num_questions if args.num_questions is not None else random.randint(1, 100)
    questions = (
        _fallback_questions(num_questions=num_questions, rng=random.Random(args.seed))
        if args.question_source == "bank"
        else generate_questions(num_questions=num_questions, seed=args.seed)
    )

    print(f"Generated {len(questions)} questions")
    print(f"Running with concurrency={args.concurrency}")
    print(f"Model pool: {', '.join(MODEL_POOL)}")
    print(f"Question source: {args.question_source}")
    print("=" * 80)

    successes = 0
    failures = 0

    for i in range(0, len(questions), args.concurrency):
        batch = questions[i : i + args.concurrency]
        results = await asyncio.gather(*(run_question(q) for q in batch))
        for _, ok in results:
            if ok:
                successes += 1
            else:
                failures += 1
        print()

    print("=" * 80)
    print(f"Completed. successes={successes} failures={failures}")
    print("=" * 80)
    _write_summary(
        args.summary_path,
        preflight=preflight,
        total=len(questions),
        successes=successes,
        failures=failures,
    )

    if args.fail_on_error and failures > 0:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random questions and run through supervisor locally"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "3")),
        help="Number of concurrent questions to process (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Exact number of questions to generate (default: random 1-100)",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any request fails",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify the configured model and Exa adapter without running questions",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip provider preflight after a separate successful preflight step",
    )
    parser.add_argument(
        "--question-source",
        choices=("generated", "bank"),
        default=os.environ.get("QUESTION_SOURCE", "generated"),
        help="Question source: generated or deterministic bank",
    )
    parser.add_argument(
        "--summary-path",
        default=os.environ.get("QUERY_SUMMARY_PATH", ""),
        help="Optional path for a JSON query result summary artifact",
    )
    args = parser.parse_args()

    logger = _init_braintrust_logger()
    _configure_gateway_client()

    try:
        asyncio.run(main_async(args))
    finally:
        if logger is not None:
            print("\nFlushing traces to Braintrust...")
            logger.flush()
            print("✅ Traces sent")


if __name__ == "__main__":
    main()
