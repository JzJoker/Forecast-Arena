from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from forecast_arena.agent import AgentResponse


@dataclass
class Forecast:
    prediction: float
    confidence: float
    reasoning: str


@dataclass
class ConfigResult:
    config_name: str
    forecast: Forecast
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    processing_time_ms: float
    agent_responses: list[AgentResponse] = field(default_factory=list)

    @classmethod
    def aggregate(
        cls,
        config_name: str,
        forecast: Forecast,
        agent_responses: list[AgentResponse],
        processing_time_ms: float,
    ) -> "ConfigResult":
        return cls(
            config_name=config_name,
            forecast=forecast,
            total_input_tokens=sum(r.input_tokens for r in agent_responses),
            total_output_tokens=sum(r.output_tokens for r in agent_responses),
            total_cost_usd=sum(r.cost_usd for r in agent_responses),
            processing_time_ms=processing_time_ms,
            agent_responses=agent_responses,
        )


class ForecastConfig(ABC):
    name: str

    @abstractmethod
    async def run(self, question: str) -> ConfigResult: ...
