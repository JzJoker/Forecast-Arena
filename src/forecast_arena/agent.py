from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

Provider = Literal["anthropic", "openai", "google", "moonshot", "alibaba", "deepseek", "meta"]

_SUPPORTED_PROVIDERS: set[str] = {
    "anthropic",
    "openai",
    "google",
    "moonshot",
    "alibaba",
    "deepseek",
    "meta",
}


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


# Prices are USD per 1M tokens. Best-effort from public docs; verify against
# the provider's current dashboard before relying on cost numbers.
PRICING: dict[str, ModelPricing] = {
    # --- Anthropic ---
    "claude-opus-4-7": ModelPricing(5.0, 25.0),
    "claude-opus-4-6": ModelPricing(5.0, 25.0),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "claude-haiku-4-5": ModelPricing(1.0, 5.0),
    # --- OpenAI ---
    "gpt-5": ModelPricing(1.25, 10.0),
    "gpt-5-mini": ModelPricing(0.25, 2.0),
    "gpt-4.1": ModelPricing(2.0, 8.0),
    "gpt-4.1-mini": ModelPricing(0.4, 1.6),
    "o3": ModelPricing(2.0, 8.0),
    "o4-mini": ModelPricing(1.1, 4.4),
    # --- Google Gemini ---
    "gemini-2.5-pro": ModelPricing(1.25, 10.0),
    "gemini-2.5-flash": ModelPricing(0.3, 2.5),
    "gemini-2.0-flash": ModelPricing(0.1, 0.4),
    # --- Moonshot (Kimi) ---
    "kimi-k2-0905-preview": ModelPricing(0.6, 2.5),
    # --- DeepSeek ---
    "deepseek-chat": ModelPricing(0.27, 1.10),
    "deepseek-reasoner": ModelPricing(0.55, 2.19),
    # --- Alibaba Qwen (via DashScope) ---
    "qwen-max": ModelPricing(1.6, 6.4),
    "qwen-plus": ModelPricing(0.4, 1.2),
    # --- Meta Llama (via Together AI) ---
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelPricing(0.18, 0.18),
}


class Agent:
    def __init__(
        self,
        provider: Provider,
        model: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        if provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}. Supported: {sorted(_SUPPORTED_PROVIDERS)}"
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
        started = time.perf_counter()

        if self.provider == "anthropic":
            from forecast_arena.providers.anthropic import call_anthropic
            text, input_tokens, output_tokens, raw = await call_anthropic(
                model=self.model,
                system_prompt=self.system_prompt,
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
        else:
            from forecast_arena.providers.openai_compatible import call_openai_compatible
            text, input_tokens, output_tokens, raw = await call_openai_compatible(
                provider=self.provider,
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
