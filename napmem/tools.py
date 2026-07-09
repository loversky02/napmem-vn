from __future__ import annotations

from dataclasses import asdict

from .pyramid import MemoryPyramid
from .schema import ToolTrace


class MemoryTools:
    """Paper-faithful five-tool interface over the memory pyramid."""

    def __init__(self, pyramid: MemoryPyramid, top_k: int = 5):
        self.pyramid = pyramid
        self.top_k = top_k
        self.trace: list[ToolTrace] = []

    def search_records(self, query: str) -> list[dict]:
        ids = self.pyramid.search_records(query, self.top_k)
        self.trace.append(ToolTrace("search_records", query, len(ids)))
        return [asdict(self.pyramid.records[rid]) for rid in ids]

    def search_conversations(self, query: str) -> list[dict]:
        ids = self.pyramid.search_conversations(query, self.top_k)
        self.trace.append(ToolTrace("search_conversations", query, len(ids)))
        return [asdict(self.pyramid.messages[mid]) for mid in ids]

    def get_records(self, record_ids: list[str]) -> list[dict]:
        rows = [asdict(self.pyramid.records[rid]) for rid in record_ids if rid in self.pyramid.records]
        self.trace.append(ToolTrace("get_records", ",".join(record_ids), len(rows)))
        return rows

    def get_conversation(self, message_ids: list[str]) -> list[dict]:
        rows = [asdict(self.pyramid.messages[mid]) for mid in message_ids if mid in self.pyramid.messages]
        self.trace.append(ToolTrace("get_conversation", ",".join(message_ids), len(rows)))
        return rows

    def read_file(self, name: str) -> str:
        text = self.pyramid.read_file(name)
        self.trace.append(ToolTrace("read_file", name, 1))
        return text

    def used_memory(self) -> bool:
        return bool(self.trace)
