from __future__ import annotations

import asyncio
import time
from typing import Optional

from forecast_arena.agent import Agent
from forecast_arena.configs.single import SYSTEM_PROMPT, SingleLLM
from forecast_arena.forecast import ConfigResult, Forecast, ForecastConfig


class VotingSwarm(ForecastConfig):
    name = "voting_swarm"

    def __init__(
        self,
        agents: list[Agent],
        weights: Optional[list[float]] = None,
    ):
        if not agents:
            raise ValueError("VotingSwarm needs at least one agent")
        if weights is not None and len(weights) != len(agents):
            raise ValueError(
                f"weights length ({len(weights)}) must match agents length ({len(agents)})"
            )

        for agent in agents:
            if agent.system_prompt is None:
                agent.system_prompt = SYSTEM_PROMPT

        if weights is None:
            self.weights = [1.0 / len(agents)] * len(agents)
        else:
            total = sum(weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive number")
            self.weights = [w / total for w in weights]
        self.agents = agents

    async def run(self, question: str) -> ConfigResult:
        started = time.perf_counter()
        sub_configs = [SingleLLM(agent) for agent in self.agents]
        sub_results = await asyncio.gather(*(c.run(question) for c in sub_configs))
        elapsed_ms = (time.perf_counter() - started) * 1000

        forecast = _aggregate(sub_results, self.weights)
        agent_responses = [resp for sr in sub_results for resp in sr.agent_responses]

        return ConfigResult.aggregate(
            config_name=self.name,
            forecast=forecast,
            agent_responses=agent_responses,
            processing_time_ms=elapsed_ms,
        )


def _aggregate(sub_results: list[ConfigResult], weights: list[float]) -> Forecast:
    prediction = sum(sr.forecast.prediction * w for sr, w in zip(sub_results, weights))
    confidence = sum(sr.forecast.confidence * w for sr, w in zip(sub_results, weights))
    reasoning = "\n\n".join(
        f"[{sr.agent_responses[0].model} | weight={w:.2f} | p={sr.forecast.prediction:.2f}] "
        f"{sr.forecast.reasoning}"
        for sr, w in zip(sub_results, weights)
    )
    return Forecast(prediction=prediction, confidence=confidence, reasoning=reasoning)
