from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .prompted import answer_correct, route_hint
from .reward import napmem_reward
from .synthetic import QAExample, SyntheticBenchmark


@dataclass(frozen=True)
class GRPORow:
    qid: str
    prompt: str
    gold_answer: str
    answer_mode: str
    requires_memory: bool
    support_layer: str
    route_hint: str
    support: tuple[str, ...]
    reward_note: str


def grpo_system_prompt() -> str:
    return (
        "You are training a memory-navigation policy. Return JSON actions only. "
        "Use memory tools only when needed; skip memory for non-memory questions. "
        "Final answers must use {\"answer\":\"...\",\"evidence_quote\":\"...\",\"reason\":\"...\"}."
    )


def build_grpo_row(example: QAExample, available_files: list[str]) -> GRPORow:
    prompt = "\n".join([
        grpo_system_prompt(),
        f"Question: {example.question}",
        f"Answer mode: {example.answer_mode}",
        f"Support layer: {example.tag}",
        f"Routing hint: {route_hint(example)}",
        "Available files: " + ", ".join(available_files),
    ])
    return GRPORow(
        qid=example.qid,
        prompt=prompt,
        gold_answer=example.answer,
        answer_mode=example.answer_mode,
        requires_memory=example.requires_memory,
        support_layer=example.tag,
        route_hint=route_hint(example),
        support=example.support,
        reward_note="Compare F+C+U against F+C; U can penalize correct memory skipping.",
    )


def build_grpo_rows(bench: SyntheticBenchmark) -> list[GRPORow]:
    files = ["profile.md", *sorted(bench.pyramid.topic_tracks)]
    return [build_grpo_row(example, files) for example in bench.examples]


def write_grpo_jsonl(bench: SyntheticBenchmark, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_grpo_rows(bench)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=True, sort_keys=True) + "\n")
    return out


def _completion_text(candidate: Any) -> Any:
    """Normalize a completion to a JSON string or dict.

    TRL hands completions as plain strings (base LM) or as chat message lists
    ([{"role":..,"content":..}]); reduce both to the final content payload.
    """
    if isinstance(candidate, list):
        if candidate and isinstance(candidate[-1], dict):
            return candidate[-1].get("content", "")
        return ""
    return candidate


def score_completion(
    candidate: Any,
    gold_answer: str,
    answer_mode: str = "exact_string",
    use_usage_bonus: bool = True,
) -> float:
    """Terminal reward for one candidate completion.

    A valid candidate is JSON with an `answer` and an optional `tool_calls` list;
    the reward is the paper's F+C+U (or F+C when `use_usage_bonus=False`).
    """
    text = _completion_text(candidate)
    try:
        payload = json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError:
        return -1.0
    if not isinstance(payload, dict) or "answer" not in payload:
        return -1.0
    tool_calls = payload.get("tool_calls", [])
    if tool_calls is None:
        tool_calls = []
    if not isinstance(tool_calls, list):
        return -1.0
    correct = answer_correct(str(payload.get("answer", "")), gold_answer, answer_mode)
    return napmem_reward(True, correct, bool(tool_calls), use_usage_bonus=use_usage_bonus)


def reward_candidate(candidate: str | dict[str, Any], row: GRPORow, use_usage_bonus: bool = True) -> float:
    return score_completion(candidate, row.gold_answer, row.answer_mode, use_usage_bonus=use_usage_bonus)


def make_grpo_reward_fn(use_usage_bonus: bool = True):
    """Build a TRL-GRPO reward function closing over the U-term choice.

    TRL calls `reward_fn(completions, **columns)` with each dataset column passed
    as a list aligned with `completions`; the seed JSONL supplies `gold_answer`
    and `answer_mode`. Returns one float reward per completion.
    """

    def napmem_grpo_reward(completions, gold_answer, answer_mode, **kwargs):
        return [
            score_completion(completion, gold, mode, use_usage_bonus=use_usage_bonus)
            for completion, gold, mode in zip(completions, gold_answer, answer_mode)
        ]

    napmem_grpo_reward.__name__ = "napmem_reward_FCU" if use_usage_bonus else "napmem_reward_FC"
    return napmem_grpo_reward
