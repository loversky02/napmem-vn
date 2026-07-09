from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RecordType = Literal["fact", "event", "instruction", "preference"]


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    session_id: str
    role: str
    content: str
    timestamp: str


@dataclass
class MemoryRecord:
    record_id: str
    user_id: str
    record_type: RecordType
    content: str
    created_at: str
    updated_at: str
    source_message_ids: list[str]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    score: float
    text: str
    source: str


@dataclass
class ToolTrace:
    tool: str
    query_or_id: str
    n_results: int
