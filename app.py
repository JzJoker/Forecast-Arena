"""CLI for testing forecast configs against the Anthropic API."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

from forecast_arena import Agent
from forecast_arena.configs.single import SingleLLM
from forecast_arena.configs.swarm import VotingSwarm
from forecast_arena.forecast import ForecastConfig

load_dotenv()


PROVIDERS = ["anthropic", "openai", "google", "moonshot", "alibaba", "deepseek", "meta"]

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4.1-mini",
    "google": "gemini-2.5-flash",
    "moonshot": "kimi-k2.6",
    "deepseek": "deepseek-chat",
    "alibaba": "qwen-plus",
    "meta": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
}


def build_config(args: argparse.Namespace) -> tuple[ForecastConfig, str]:
    if args.swarm > 1:
        agents = [Agent(args.provider, args.model) for _ in range(args.swarm)]
        label = f"voting_swarm ({args.swarm}x {args.provider}/{args.model})"
        return VotingSwarm(agents), label
    return (
        SingleLLM(Agent(args.provider, args.model)),
        f"single_llm ({args.provider}/{args.model})",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a forecast through a single LLM or a voting swarm."
    )
    parser.add_argument(
        "question",
        help="The forecasting question (e.g. 'Will SpaceX land humans on Mars by 2030?')",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=PROVIDERS,
        help="Model provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID. If omitted, uses the cheap workhorse for --provider.",
    )
    parser.add_argument(
        "--swarm",
        type=int,
        default=1,
        metavar="N",
        help="Run an N-agent voting swarm of --model (default: 1, runs single LLM)",
    )
    args = parser.parse_args()

    if args.swarm < 1:
        sys.exit("Error: --swarm must be >= 1")
    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]

    config, label = build_config(args)
    result = asyncio.run(config.run(args.question))
    forecast = result.forecast

    print()
    print(f"Question:   {args.question}")
    print(f"Config:     {label}")
    print()
    print(f"Prediction: {forecast.prediction:.2f}")
    print(f"Confidence: {forecast.confidence:.2f}")
    print(f"Reasoning:")
    for line in forecast.reasoning.split("\n"):
        print(f"  {line}")
    print()
    print(f"Tokens:     {result.total_input_tokens} in / {result.total_output_tokens} out")
    print(f"Time:       {result.processing_time_ms:.0f} ms")
    print(f"Cost:       ${result.total_cost_usd:.6f}")


if __name__ == "__main__":
    main()
