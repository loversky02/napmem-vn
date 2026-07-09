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

First-tool routing:
- Questions asking for an exact word/ingredient/name/tool/place from a past message:
  use search_conversations first, then get_conversation if exact wording matters.
- Questions asking for a recorded fact, reaction, time, reason, or instruction:
  use search_records first, then get_records.
- Questions asking for stable/current user preferences or profile-level patterns:
  read_file("profile.md") first.
- Questions requiring cross-session summaries or inference from a topic:
  read the most relevant topic file first.
- Non-memory questions: answer directly with no memory tool.

Exact layer discipline:
- If the routing hint says record, copy the final answer and evidence_quote from
  the record content. Use raw conversation only to verify provenance, not to replace
  the record's wording.
- If the routing hint says raw, copy from raw conversation.
- If the routing hint says profile, copy from profile.md.
- If the routing hint says topic, copy from the topic file.
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


def route_hint(example: QAExample) -> str:
    if not example.requires_memory:
        return "Answer directly; do not call memory tools."
    if example.tag == "raw":
        return "Start with search_conversations. Use get_conversation for exact wording."
    if example.tag == "record":
        return "Start with search_records. Use get_records before answering."
    if example.tag == "profile":
        return "Start with read_file using profile.md."
    if example.tag == "topic":
        return "Start with read_file on the most relevant topic file."
    return "Choose the narrowest memory level that can answer exactly."


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
                    f"Support layer: {example.tag}\n"
                    f"Routing hint: {route_hint(example)}\n"
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
