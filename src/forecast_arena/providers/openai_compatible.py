from __future__ import annotations

import os
from typing import Optional

from openai import AsyncOpenAI

PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "alibaba": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
    },
    "meta": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
}

_clients: dict[str, AsyncOpenAI] = {}


def _get_client(provider: str) -> AsyncOpenAI:
    if provider not in _clients:
        cfg = PROVIDER_CONFIG[provider]
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"Set {cfg['api_key_env']} in your environment to use provider {provider!r}."
            )
        _clients[provider] = AsyncOpenAI(api_key=api_key, base_url=cfg["base_url"])
    return _clients[provider]


async def call_openai_compatible(
    *,
    provider: str,
    model: str,
    system_prompt: Optional[str],
    prompt: str,
    max_tokens: int,
) -> tuple[str, int, int, dict]:
    client = _get_client(provider)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    text = response.choices[0].message.content or ""
    usage = response.usage
    return (
        text,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        response.model_dump(),
    )
