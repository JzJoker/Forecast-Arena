from __future__ import annotations

import json
import re
import time

from forecast_arena.agent import Agent
from forecast_arena.forecast import ConfigResult, Forecast, ForecastConfig

SYSTEM_PROMPT = """You are a calibrated forecaster. Given a yes/no question, output a probability that the answer is YES.

Respond with ONLY a JSON object — no preamble, no explanation outside the JSON:
{
  "prediction": <float between 0.0 and 1.0>,
  "confidence": <float between 0.0 and 1.0 — your self-assessed confidence in the prediction>,
  "reasoning": "<2-4 sentences explaining your key reasoning>"
}"""


class SingleLLM(ForecastConfig):
    name = "single_llm"

    def __init__(self, agent: Agent):
        if agent.system_prompt is None:
            agent.system_prompt = SYSTEM_PROMPT
        self.agent = agent

    async def run(self, question: str) -> ConfigResult:
        started = time.perf_counter()
        response = await self.agent.arun(question)
        forecast = _parse_forecast(response.text)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ConfigResult.aggregate(
            config_name=self.name,
            forecast=forecast,
            agent_responses=[response],
            processing_time_ms=elapsed_ms,
        )


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_forecast(text: str) -> Forecast:
    cleaned = _FENCE_RE.sub("", text.strip())
    data = json.loads(cleaned)
    return Forecast(
        prediction=float(data["prediction"]),
        confidence=float(data["confidence"]),
        reasoning=str(data["reasoning"]),
    )
