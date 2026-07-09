from __future__ import annotations

from dataclasses import dataclass

from .eval import STRATEGIES, evaluate_strategy, summarize
from .synthetic import SyntheticBenchmark


@dataclass(frozen=True)
class RewardAblationRow:
    strategy: str
    accuracy: float
    exact_fail_rate: float
    avg_tool_calls: float
    unnecessary_memory_call_rate: float
    reward_with_u: float
    reward_without_u: float
    usage_bonus_delta: float


def reward_ablation(bench: SyntheticBenchmark, strategies: list[str] | None = None) -> list[RewardAblationRow]:
    names = strategies or list(STRATEGIES)
    rows: list[RewardAblationRow] = []
    for name in names:
        metrics = summarize(evaluate_strategy(bench, name))
        rows.append(
            RewardAblationRow(
                strategy=name,
                accuracy=metrics["accuracy"],
                exact_fail_rate=metrics["exact_fail_rate"],
                avg_tool_calls=metrics["avg_tool_calls"],
                unnecessary_memory_call_rate=metrics["unnecessary_memory_call_rate"],
                reward_with_u=metrics["reward_with_u"],
                reward_without_u=metrics["reward_without_u"],
                usage_bonus_delta=metrics["reward_with_u"] - metrics["reward_without_u"],
            )
        )
    return rows


def format_ablation(rows: list[RewardAblationRow]) -> str:
    lines = [
        f"{'strategy':<14}{'acc':>6}{'exact!':>8}{'calls':>8}{'unnec%':>8}{'R+U':>7}{'R':>7}{'delta':>8}",
        "-" * 66,
    ]
    for row in rows:
        lines.append(
            f"{row.strategy:<14}"
            f"{row.accuracy:>6.2f}"
            f"{row.exact_fail_rate:>8.2f}"
            f"{row.avg_tool_calls:>8.2f}"
            f"{row.unnecessary_memory_call_rate:>8.2f}"
            f"{row.reward_with_u:>7.2f}"
            f"{row.reward_without_u:>7.2f}"
            f"{row.usage_bonus_delta:>8.2f}"
        )
    return "\n".join(lines)
