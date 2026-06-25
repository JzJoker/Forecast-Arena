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

DEFAULT_SWARM_PROVIDERS = ["anthropic", "google", "openai", "moonshot", "deepseek"]


def build_config(args: argparse.Namespace) -> tuple[ForecastConfig, str]:
    if args.swarm_providers:
        providers = args.swarm_providers
        agents = [Agent(p, DEFAULT_MODELS[p]) for p in providers]
        label = f"voting_swarm ({len(providers)} agents: {', '.join(providers)})"
        return VotingSwarm(agents), label
    if args.swarm:
        agents = [Agent(p, DEFAULT_MODELS[p]) for p in DEFAULT_SWARM_PROVIDERS]
        label = f"voting_swarm (default mix: {', '.join(DEFAULT_SWARM_PROVIDERS)})"
        return VotingSwarm(agents), label
    return (
        SingleLLM(Agent(args.provider, args.model)),
        f"single_llm ({args.provider}/{args.model})",
    )


def _parse_provider_list(raw: str) -> list[str]:
    providers = [p.strip() for p in raw.split(",") if p.strip()]
    if not providers:
        raise argparse.ArgumentTypeError("--swarm-providers must list at least one provider")
    unknown = [p for p in providers if p not in PROVIDERS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown provider(s): {', '.join(unknown)}. Choose from: {', '.join(PROVIDERS)}"
        )
    return providers


async def _spinner(message: str) -> None:
    frames = "|/-\\"
    i = 0
    try:
        while True:
            sys.stderr.write(f"\r{message} {frames[i % len(frames)]}")
            sys.stderr.flush()
            i += 1
            await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        sys.stderr.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stderr.flush()
        raise


async def _run_with_indicator(config: ForecastConfig, question: str, message: str):
    spinner = asyncio.create_task(_spinner(message))
    try:
        return await config.run(question)
    finally:
        spinner.cancel()
        try:
            await spinner
        except asyncio.CancelledError:
            pass


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
        action="store_true",
        help=(
            "Run the default 5-provider voting swarm "
            f"({', '.join(DEFAULT_SWARM_PROVIDERS)})."
        ),
    )
    parser.add_argument(
        "--swarm-providers",
        type=_parse_provider_list,
        default=None,
        metavar="P1,P2,...",
        help=(
            "Comma-separated provider list for the swarm. Count = list length; "
            "repeats add multiplicity (e.g. openai,openai for 2 OpenAI agents). "
            "Overrides --swarm if both are set."
        ),
    )
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]

    config, label = build_config(args)
    is_swarm = isinstance(config, VotingSwarm)
    indicator = "Thinking and Voting..." if is_swarm else "Thinking..."
    result = asyncio.run(_run_with_indicator(config, args.question, indicator))
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
