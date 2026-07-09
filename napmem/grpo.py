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


def reward_candidate(candidate: str | dict[str, Any], row: GRPORow, use_usage_bonus: bool = True) -> float:
    try:
        payload = json.loads(candidate) if isinstance(candidate, str) else candidate
    except json.JSONDecodeError:
        return -1.0
    if not isinstance(payload, dict) or "answer" not in payload:
        return -1.0
    tool_calls = payload.get("tool_calls", [])
    if tool_calls is None:
        tool_calls = []
    if not isinstance(tool_calls, list):
        return -1.0
    correct = answer_correct(str(payload.get("answer", "")), row.gold_answer, row.answer_mode)
    return napmem_reward(True, correct, bool(tool_calls), use_usage_bonus=use_usage_bonus)
