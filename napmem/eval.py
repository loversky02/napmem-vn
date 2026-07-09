from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .reward import napmem_reward
from .schema import ToolTrace
from .synthetic import QAExample, SyntheticBenchmark
from .tools import MemoryTools


@dataclass
class NavigationRun:
    qid: str
    strategy: str
    answer: str
    correct: bool
    evidence_hit: bool
    support_found: tuple[str, ...]
    support_required: tuple[str, ...]
    tool_calls: int
    memory_used: bool
    answer_mode: str
    exactness_fail: bool
    reward_with_u: float
    reward_without_u: float
    trace: list[ToolTrace] = field(default_factory=list)


Strategy = Callable[[SyntheticBenchmark, QAExample], tuple[set[str], list[ToolTrace]]]


def _support_from_records(rows: list[dict]) -> set[str]:
    return {f"record:{row['record_id']}" for row in rows}


def _support_from_messages(rows: list[dict]) -> set[str]:
    return {f"message:{row['message_id']}" for row in rows}


def _support_from_file(name: str) -> set[str]:
    return {f"file:{name}"}


def no_memory(bench: SyntheticBenchmark, example: QAExample) -> tuple[set[str], list[ToolTrace]]:
    return set(), []


def passive_topk(bench: SyntheticBenchmark, example: QAExample) -> tuple[set[str], list[ToolTrace]]:
    tools = MemoryTools(bench.pyramid)
    rows = tools.search_records(example.question)
    return _support_from_records(rows), tools.trace


def records_only(bench: SyntheticBenchmark, example: QAExample) -> tuple[set[str], list[ToolTrace]]:
    tools = MemoryTools(bench.pyramid)
    rows = tools.search_records(example.question)
    ids = [row["record_id"] for row in rows]
    fetched = tools.get_records(ids)
    return _support_from_records(fetched), tools.trace


def upper_first(bench: SyntheticBenchmark, example: QAExample) -> tuple[set[str], list[ToolTrace]]:
    tools = MemoryTools(bench.pyramid)
    found: set[str] = set()
    tools.read_file("profile.md")
    found |= _support_from_file("profile.md")
    for name in sorted(bench.pyramid.topic_tracks):
        tools.read_file(name)
        found |= _support_from_file(name)
    return found, tools.trace


def drilldown(bench: SyntheticBenchmark, example: QAExample) -> tuple[set[str], list[ToolTrace]]:
    tools = MemoryTools(bench.pyramid)
    found: set[str] = set()
    rows = tools.search_records(example.question)
    found |= _support_from_records(rows)
    ids = [row["record_id"] for row in rows]
    fetched = tools.get_records(ids)
    found |= _support_from_records(fetched)
    source_ids = sorted({mid for row in fetched for mid in row["source_message_ids"]})
    found |= _support_from_messages(tools.get_conversation(source_ids))

    # Cheap topic/profile pass: read upper layers when low-level evidence may need
    # interpretation or stable summary context.
    if any(word in example.question.lower() for word in ("state", "style", "summary", "prefer")):
        tools.read_file("profile.md")
        found |= _support_from_file("profile.md")
        for name in sorted(bench.pyramid.topic_tracks):
            tools.read_file(name)
            found |= _support_from_file(name)
    return found, tools.trace


def oracle(bench: SyntheticBenchmark, example: QAExample) -> tuple[set[str], list[ToolTrace]]:
    tools = MemoryTools(bench.pyramid)
    found: set[str] = set()
    for support in example.support:
        kind, value = support.split(":", 1)
        if kind == "record":
            found |= _support_from_records(tools.get_records([value]))
        elif kind == "message":
            found |= _support_from_messages(tools.get_conversation([value]))
        elif kind == "file":
            tools.read_file(value)
            found |= _support_from_file(value)
    return found, tools.trace


STRATEGIES: dict[str, Strategy] = {
    "no_memory": no_memory,
    "passive_topk": passive_topk,
    "records_only": records_only,
    "upper_first": upper_first,
    "drilldown": drilldown,
    "oracle": oracle,
}


def evaluate_strategy(bench: SyntheticBenchmark, name: str) -> list[NavigationRun]:
    strategy = STRATEGIES[name]
    runs: list[NavigationRun] = []
    for example in bench.examples:
        found, trace = strategy(bench, example)
        required = set(example.support)
        correct = required.issubset(found)
        exactness_fail = example.answer_mode == "exact_string" and not correct
        runs.append(
            NavigationRun(
                qid=example.qid,
                strategy=name,
                answer=example.answer if correct else "",
                correct=correct,
                evidence_hit=correct,
                support_found=tuple(sorted(found)),
                support_required=example.support,
                tool_calls=len(trace),
                memory_used=bool(trace),
                answer_mode=example.answer_mode,
                exactness_fail=exactness_fail,
                reward_with_u=napmem_reward(True, correct, bool(trace), use_usage_bonus=True),
                reward_without_u=napmem_reward(True, correct, bool(trace), use_usage_bonus=False),
                trace=list(trace),
            )
        )
    return runs


def summarize(runs: list[NavigationRun]) -> dict[str, float]:
    n = len(runs) or 1
    memory_runs = [run for run in runs if run.support_required]
    non_memory_runs = [run for run in runs if not run.support_required]
    mem_n = len(memory_runs) or 1
    non_mem_n = len(non_memory_runs) or 1
    return {
        "accuracy": sum(run.correct for run in runs) / n,
        "memory_accuracy": sum(run.correct for run in memory_runs) / mem_n,
        "evidence_hit": sum(run.evidence_hit for run in runs) / n,
        "exact_fail_rate": sum(run.exactness_fail for run in runs) / n,
        "avg_tool_calls": sum(run.tool_calls for run in runs) / n,
        "memory_call_rate": sum(run.memory_used for run in runs) / n,
        "unnecessary_memory_call_rate": sum(run.memory_used for run in non_memory_runs) / non_mem_n,
        "reward_with_u": sum(run.reward_with_u for run in runs) / n,
        "reward_without_u": sum(run.reward_without_u for run in runs) / n,
    }


def evaluate_all(bench: SyntheticBenchmark, strategy_names: list[str] | None = None) -> dict[str, dict[str, float]]:
    names = strategy_names or list(STRATEGIES)
    return {name: summarize(evaluate_strategy(bench, name)) for name in names}
