"""Math agent with arithmetic capabilities."""

from agents import Agent, function_tool

from src.config import DEFAULT_MATH_AGENT_PROMPT


@function_tool
def add(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b


@function_tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


@function_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


@function_tool
def divide(a: float, b: float) -> float:
    """Divide a by b and return the quotient.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


def get_math_agent(system_prompt: str | None = None, model: str = "gpt-4o-mini") -> Agent:
    """Create the math agent with optional custom prompt and model."""
    prompt = system_prompt if system_prompt is not None else DEFAULT_MATH_AGENT_PROMPT

    return Agent(
        name="Math Agent",
        model=model,
        instructions=prompt,
        tools=[add, subtract, multiply, divide],
    )
