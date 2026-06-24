from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

Provider = Literal["anthropic", "openai", "google", "moonshot", "alibaba", "deepseek", "meta"]


@dataclass
class AgentResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model: str
    provider: Provider
    raw: dict = field(default_factory=dict)


@dataclass
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float


PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-7": ModelPricing(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-6": ModelPricing(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-sonnet-4-6": ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0),
    "claude-haiku-4-5": ModelPricing(input_per_mtok=1.0, output_per_mtok=5.0),
}


class Agent:
    def __init__(
        self,
        provider: Provider,
        model: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        if provider != "anthropic":
            raise NotImplementedError(
                f"Provider {provider!r} not yet wired up. Only 'anthropic' is supported."
            )
        if model not in PRICING:
            raise ValueError(
                f"Model {model!r} has no pricing entry. "
                f"Known models: {sorted(PRICING)}. Add a ModelPricing to PRICING in agent.py."
            )
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def run(self, prompt: str) -> AgentResponse:
        return asyncio.run(self.arun(prompt))

    async def arun(self, prompt: str) -> AgentResponse:
        from forecast_arena.providers.anthropic import call_anthropic

        started = time.perf_counter()
        text, input_tokens, output_tokens, raw = await call_anthropic(
            model=self.model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            max_tokens=self.max_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        pricing = PRICING[self.model]
        cost_usd = (
            input_tokens * pricing.input_per_mtok
            + output_tokens * pricing.output_per_mtok
        ) / 1_000_000

        return AgentResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            model=self.model,
            provider=self.provider,
            raw=raw,
        )
