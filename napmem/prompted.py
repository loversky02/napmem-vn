from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import Backend
from .schema import ToolTrace
from .synthetic import QAExample, SyntheticBenchmark
from .tools import MemoryTools

SYSTEM_PROMPT = """You are a personal assistant using a four-layer user memory bank.
Return only JSON. At each step choose either:
{"tool":"search_records","query":"..."}
{"tool":"search_conversations","query":"..."}
{"tool":"get_records","ids":["..."]}
{"tool":"get_conversation","ids":["..."]}
{"tool":"read_file","name":"profile.md"}
{"answer":"...","evidence_quote":"...","reason":"..."}

Use memory only when needed. Stop once evidence is sufficient. Do not guess user facts.
If answer_mode is exact_string, the answer must be an exact phrase copied from a tool
observation or from common non-memory knowledge. If a record is too vague, use its
source_message_ids with get_conversation before answering. Do not replace exact
ingredients, preferences, or style phrases with broader categories.
"""


@dataclass
class PromptedResult:
    answer: str
    reason: str
    evidence_quote: str = ""
    trace: list[ToolTrace] = field(default_factory=list)
    raw_steps: list[str] = field(default_factory=list)


def _json_from_text(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj = _first_json_object(text)
        if obj is None:
            raise
        return json.loads(obj)


def _first_json_object(text: str) -> str | None:
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


class PromptedNavigator:
    """Small tool-call loop for 9router/OpenAI-compatible models.

    This intentionally uses plain JSON instead of provider-native tool schemas so it
    works with the same minimal stdlib client used in AutoMem-VN.
    """

    def __init__(self, backend: Backend, max_tool_turns: int = 4):
        self.backend = backend
        self.max_tool_turns = max_tool_turns

    def answer(self, bench: SyntheticBenchmark, example: QAExample) -> PromptedResult:
        tools = MemoryTools(bench.pyramid)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {example.question}\n"
                    f"Answer mode: {example.answer_mode}\n"
                    "Available high-level files: profile.md, "
                    f"{', '.join(sorted(bench.pyramid.topic_tracks))}\n"
                    'Final answer JSON must be {"answer":"...","evidence_quote":"...","reason":"..."}'
                ),
            },
        ]
        raw_steps: list[str] = []
        for _ in range(self.max_tool_turns + 1):
            raw = self.backend(messages)
            raw_steps.append(raw)
            try:
                action = _json_from_text(raw)
            except json.JSONDecodeError:
                return PromptedResult(
                    answer="",
                    reason="invalid JSON from model",
                    evidence_quote="",
                    trace=list(tools.trace),
                    raw_steps=raw_steps,
                )
            if "answer" in action:
                return PromptedResult(
                    answer=str(action.get("answer", "")),
                    reason=str(action.get("reason", "")),
                    evidence_quote=str(action.get("evidence_quote", "")),
                    trace=list(tools.trace),
                    raw_steps=raw_steps,
                )
            if len(tools.trace) >= self.max_tool_turns:
                break
            observation = self._call_tool(tools, action)
            messages.append({
                "role": "assistant",
                "content": "Assistant action JSON: " + json.dumps(action, ensure_ascii=False),
            })
            messages.append({
                "role": "user",
                "content": "Tool observation:\n" + json.dumps(observation, ensure_ascii=False),
            })
        return PromptedResult(
            answer="",
            reason="tool budget exhausted",
            evidence_quote="",
            trace=list(tools.trace),
            raw_steps=raw_steps,
        )

    @staticmethod
    def _call_tool(tools: MemoryTools, action: dict):
        tool = action.get("tool")
        if tool == "search_records":
            return tools.search_records(str(action.get("query", "")))
        if tool == "search_conversations":
            return tools.search_conversations(str(action.get("query", "")))
        if tool == "get_records":
            return tools.get_records([str(x) for x in action.get("ids", [])])
        if tool == "get_conversation":
            return tools.get_conversation([str(x) for x in action.get("ids", [])])
        if tool == "read_file":
            return tools.read_file(str(action.get("name", "profile.md")))
        return {"error": f"unknown tool {tool!r}"}


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def answer_correct(predicted: str, gold: str, answer_mode: str = "exact_string") -> bool:
    pred = _norm(predicted)
    target = _norm(gold)
    if not target:
        return not pred
    if answer_mode == "exact_string":
        return pred == target or target in pred
    return target in pred


def quote_supports_answer(evidence_quote: str, gold: str, answer_mode: str = "exact_string") -> bool:
    if answer_mode != "exact_string":
        return True
    return _norm(gold) in _norm(evidence_quote)
