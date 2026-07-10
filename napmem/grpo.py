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
    # Single-turn GRPO: the supporting evidence (if any) is given in the prompt so
    # the answer is achievable in one pass. The only free behaviour is whether the
    # policy declares a memory tool call. The instruction is deliberately NEUTRAL
    # about *when* to call memory — it only describes the output format — so the
    # tool-call decision is driven purely by the reward, isolating the usage term U
    # (including whether U teaches unnecessary calls on non-memory questions).
    return (
        "Answer the user's question. "
        "Output ONLY one compact JSON object on a single line and nothing else:\n"
        '{"answer": "<answer>", "tool_calls": [{"tool": "search_records"}]}\n'
        'tool_calls lists the memory tools you used; use an empty list [] if you used none.'
    )


def _evidence_for(example: QAExample, bench: SyntheticBenchmark) -> str:
    """Resolve an example's support ids to the underlying memory text."""
    parts: list[str] = []
    for support in example.support:
        kind, _, value = support.partition(":")
        try:
            if kind == "record" and value in bench.pyramid.records:
                parts.append(bench.pyramid.records[value].content)
            elif kind == "message" and value in bench.pyramid.messages:
                parts.append(bench.pyramid.messages[value].content)
            elif kind == "file":
                parts.append(bench.pyramid.read_file(value))
        except (FileNotFoundError, KeyError):
            continue
    return "\n".join(parts).strip()


def build_grpo_row(example: QAExample, evidence: str = "") -> GRPORow:
    # Memory questions carry their supporting evidence so a correct answer is
    # reachable in one pass. Non-memory questions are shown plainly, with NO hint
    # that they need no memory — the policy must decide whether to call a tool, so
    # the usage term U can (or cannot) teach unnecessary calls here.
    lines = [grpo_system_prompt(), f"Question: {example.question}"]
    if example.requires_memory and evidence:
        lines.append(f"Memory evidence:\n{evidence}")
    lines.append("JSON:")
    return GRPORow(
        qid=example.qid,
        prompt="\n".join(lines),
        gold_answer=example.answer,
        answer_mode=example.answer_mode,
        requires_memory=example.requires_memory,
        support_layer=example.tag,
        route_hint=route_hint(example),
        support=example.support,
        reward_note="Compare F+C+U against F+C; U rewards a non-empty tool_calls list.",
    )


def build_grpo_rows(bench: SyntheticBenchmark) -> list[GRPORow]:
    return [build_grpo_row(example, _evidence_for(example, bench)) for example in bench.examples]


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


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} block, tolerating surrounding prose/fences."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json_payload(candidate: Any) -> dict | None:
    """Best-effort parse of a completion into a JSON dict, tolerating extra text."""
    text = _completion_text(candidate)
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        block = _first_json_object(text)
        if block is None:
            return None
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


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
    payload = parse_json_payload(candidate)
    if payload is None or "answer" not in payload:
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
