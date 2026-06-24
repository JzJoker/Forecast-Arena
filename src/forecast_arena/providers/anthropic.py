from __future__ import annotations

from typing import Optional

import anthropic

_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


async def call_anthropic(
    *,
    model: str,
    system_prompt: Optional[str],
    prompt: str,
    max_tokens: int,
) -> tuple[str, int, int, dict]:
    client = _get_client()

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    async with client.messages.stream(**kwargs) as stream:
        message = await stream.get_final_message()

    text = "".join(block.text for block in message.content if block.type == "text")
    return (
        text,
        message.usage.input_tokens,
        message.usage.output_tokens,
        message.model_dump(),
    )
