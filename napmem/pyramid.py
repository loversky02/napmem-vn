from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .schema import MemoryRecord, Message
from .search import rank_keyword, reciprocal_rank_fusion


class MemoryPyramid:
    """Four-layer user memory pyramid.

    L1 raw conversations and L2 records are structured stores. L3 topic tracks and
    L4 profile are markdown files. This is intentionally local and deterministic:
    the LLM extraction/reconciliation writers can be swapped in without changing
    the tool interface.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.record_dir = self.root / "records"
        self.files_dir = self.root / "files"
        for path in (self.raw_dir, self.record_dir, self.files_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.messages: dict[str, Message] = {}
        self.records: dict[str, MemoryRecord] = {}
        self.topic_tracks: dict[str, str] = {}
        self.profile = ""

    def append_message(self, message: Message) -> None:
        if not message.content.strip():
            return
        self.messages[message.message_id] = message
        self._write_jsonl(self.raw_dir / "messages.jsonl", asdict(message))

    def add_record(self, record: MemoryRecord) -> None:
        missing = [mid for mid in record.source_message_ids if mid not in self.messages]
        if missing:
            raise ValueError(f"record {record.record_id} references missing messages: {missing}")
        self.records[record.record_id] = record
        self._rewrite_records()

    def upsert_topic_track(self, name: str, markdown: str) -> None:
        safe = self._safe_file(name)
        self.topic_tracks[safe] = markdown
        (self.files_dir / safe).write_text(markdown, encoding="utf-8")

    def update_profile(self, markdown: str) -> None:
        self.profile = markdown
        (self.files_dir / "profile.md").write_text(markdown, encoding="utf-8")

    def search_records(self, query: str, limit: int = 5) -> list[str]:
        keyword = rank_keyword(query, ((rid, r.content) for rid, r in self.records.items()), limit * 4)
        # Placeholder second ranking mirrors keyword order until embeddings are wired.
        fused = reciprocal_rank_fusion([[rid for rid, _ in keyword], [rid for rid, _ in keyword]])
        return [rid for rid, _ in sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:limit]]

    def search_conversations(self, query: str, limit: int = 5) -> list[str]:
        keyword = rank_keyword(query, ((mid, m.content) for mid, m in self.messages.items()), limit * 4)
        fused = reciprocal_rank_fusion([[mid for mid, _ in keyword], [mid for mid, _ in keyword]])
        return [mid for mid, _ in sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:limit]]

    def read_file(self, name: str) -> str:
        safe = self._safe_file(name)
        path = self.files_dir / safe
        if not path.exists():
            raise FileNotFoundError(safe)
        return path.read_text(encoding="utf-8")

    def _rewrite_records(self) -> None:
        path = self.record_dir / "records.jsonl"
        path.write_text("", encoding="utf-8")
        for record in self.records.values():
            self._write_jsonl(path, asdict(record))

    @staticmethod
    def _write_jsonl(path: Path, row: dict) -> None:
        import json

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _safe_file(name: str) -> str:
        if not name or "/" in name or "\\" in name or ".." in name:
            raise ValueError(f"unsafe memory file name: {name!r}")
        return name
