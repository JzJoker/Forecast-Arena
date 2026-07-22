from __future__ import annotations

import asyncio
import json
import random
import re
import time

from forecast_arena.agent import Agent, AgentResponse
from forecast_arena.configs.single import SingleLLM
from forecast_arena.forecast import ConfigResult, Forecast, ForecastConfig

CRITIQUE_SYSTEM_PROMPT = """You are a forecasting reviewer. You will see multiple anonymized forecasts for the same yes/no question. Score each on the QUALITY OF ITS REASONING — evidence, calibration, whether it addresses key considerations — not whether the prediction matches your own view.

Use an integer scale from 1 (unfounded, hand-wavy, contradictory) to 10 (well-supported, calibrated, thorough).

Respond with ONLY a JSON object mapping each forecast's letter label to its score:
{"A": <int 1-10>, "B": <int 1-10>, ...}"""


_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMCouncil(ForecastConfig):
    name = "llm_council"

    def __init__(self, agents: list[Agent]):
        if len(agents) < 2:
            raise ValueError("LLMCouncil needs at least 2 agents (peers must critique each other)")
        if len(agents) > len(_LABELS):
            raise ValueError(f"LLMCouncil supports at most {len(_LABELS)} agents")
        self.agents = agents

    async def run(self, question: str) -> ConfigResult:
        started = time.perf_counter()

        forecast_configs = [SingleLLM(a) for a in self.agents]
        forecast_results = await asyncio.gather(
            *(c.run(question) for c in forecast_configs)
        )

        critic_shuffles: list[list[int]] = []
        critique_tasks = []
        for critic_idx, agent in enumerate(self.agents):
            shuffle = list(range(len(self.agents)))
            random.shuffle(shuffle)
            critic_shuffles.append(shuffle)
            prompt = _build_critique_prompt(question, forecast_results, shuffle)
            critic = Agent(
                provider=agent.provider,
                model=agent.model,
                system_prompt=CRITIQUE_SYSTEM_PROMPT,
                max_tokens=agent.max_tokens,
            )
            critique_tasks.append(critic.arun(prompt))

        critique_responses = await asyncio.gather(*critique_tasks, return_exceptions=True)

        weights = _compute_weights(
            n_agents=len(self.agents),
            critic_shuffles=critic_shuffles,
            critique_responses=critique_responses,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        agent_responses: list[AgentResponse] = [
            r for fr in forecast_results for r in fr.agent_responses
        ]
        for resp in critique_responses:
            if isinstance(resp, AgentResponse):
                agent_responses.append(resp)

        forecast = _aggregate(forecast_results, weights)

        return ConfigResult.aggregate(
            config_name=self.name,
            forecast=forecast,
            agent_responses=agent_responses,
            processing_time_ms=elapsed_ms,
        )


def _build_critique_prompt(
    question: str,
    forecast_results: list[ConfigResult],
    shuffle: list[int],
) -> str:
    lines = [f"Question: {question}", "", "Forecasts to score:"]
    for pos, agent_idx in enumerate(shuffle):
        fr = forecast_results[agent_idx]
        lines.append("")
        lines.append(
            f"[{_LABELS[pos]}] prediction={fr.forecast.prediction:.2f} "
            f"confidence={fr.forecast.confidence:.2f}"
        )
        lines.append(f"Reasoning: {fr.forecast.reasoning}")
    last_label = _LABELS[len(shuffle) - 1]
    lines.append("")
    lines.append(
        f'Return ONLY a JSON object with integer scores 1-10 for every label A..{last_label}, '
        f'e.g. {{"A": 7, "B": 8, ...}}.'
    )
    return "\n".join(lines)


def _parse_scores(text: str, n_labels: int) -> dict[str, int]:
    cleaned = _FENCE_RE.sub("", text.strip())
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        return {}
    scores: dict[str, int] = {}
    for i in range(n_labels):
        label = _LABELS[i]
        for key in (label, label.lower()):
            if key in data:
                try:
                    score = int(float(data[key]))
                except (TypeError, ValueError):
                    continue
                if 1 <= score <= 10:
                    scores[label] = score
                break
    return scores


def _compute_weights(
    *,
    n_agents: int,
    critic_shuffles: list[list[int]],
    critique_responses: list,
) -> list[float]:
    received_totals = [0.0] * n_agents
    received_counts = [0] * n_agents

    for critic_idx, response in enumerate(critique_responses):
        if not isinstance(response, AgentResponse):
            continue
        try:
            scores = _parse_scores(response.text, n_agents)
        except json.JSONDecodeError:
            continue
        shuffle = critic_shuffles[critic_idx]
        for pos, agent_idx in enumerate(shuffle):
            if agent_idx == critic_idx:
                continue
            label = _LABELS[pos]
            if label in scores:
                received_totals[agent_idx] += scores[label]
                received_counts[agent_idx] += 1

    raw = [
        (received_totals[i] / received_counts[i]) if received_counts[i] > 0 else 0.0
        for i in range(n_agents)
    ]
    total = sum(raw)
    if total <= 0:
        return [1.0 / n_agents] * n_agents
    return [w / total for w in raw]


def _aggregate(forecast_results: list[ConfigResult], weights: list[float]) -> Forecast:
    prediction = sum(fr.forecast.prediction * w for fr, w in zip(forecast_results, weights))
    confidence = sum(fr.forecast.confidence * w for fr, w in zip(forecast_results, weights))
    reasoning = "\n\n".join(
        f"[{fr.agent_responses[0].model} | peer-weight={w:.2f} | p={fr.forecast.prediction:.2f}] "
        f"{fr.forecast.reasoning}"
        for fr, w in zip(forecast_results, weights)
    )
    return Forecast(prediction=prediction, confidence=confidence, reasoning=reasoning)
