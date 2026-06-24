"""CLI for testing forecast configs against the Anthropic API."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

from forecast_arena import Agent
from forecast_arena.configs.single import SingleLLM

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a forecast through the Single LLM config."
    )
    parser.add_argument(
        "question",
        help="The forecasting question (e.g. 'Will SpaceX land humans on Mars by 2030?')",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="Anthropic model ID (default: claude-haiku-4-5)",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "Error: ANTHROPIC_API_KEY is not set. Export it or source your .env file."
        )

    config = SingleLLM(Agent("anthropic", args.model))
    result = asyncio.run(config.run(args.question))
    forecast = result.forecast

    print()
    print(f"Question:   {args.question}")
    print(f"Model:      {args.model}")
    print()
    print(f"Prediction: {forecast.prediction:.2f}")
    print(f"Confidence: {forecast.confidence:.2f}")
    print(f"Reasoning:  {forecast.reasoning}")
    print()
    print(f"Tokens:     {result.total_input_tokens} in / {result.total_output_tokens} out")
    print(f"Time:       {result.processing_time_ms:.0f} ms")
    print(f"Cost:       ${result.total_cost_usd:.6f}")


if __name__ == "__main__":
    main()
