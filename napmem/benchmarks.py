from __future__ import annotations

import json
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LOCOMO_PLUS_RAW_URL = "https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/main/data/locomo_plus.json"


@dataclass(frozen=True)
class BenchmarkSubsetRow:
    qid: str
    source: str
    question: str
    answer: str
    support_text: str
    support_layer: str
    relation_type: str
    metadata: dict[str, Any]


def load_json_source(path_or_url: str, verify_ssl: bool = True) -> Any:
    if path_or_url.startswith(("http://", "https://")):
        context = None if verify_ssl else ssl._create_unverified_context()
        with urllib.request.urlopen(path_or_url, timeout=120, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    return json.loads(Path(path_or_url).read_text(encoding="utf-8"))


def locomo_plus_to_rows(items: list[dict[str, Any]], limit: int = 16) -> list[BenchmarkSubsetRow]:
    rows: list[BenchmarkSubsetRow] = []
    for idx, item in enumerate(items[:limit], start=1):
        cue = str(item.get("cue_dialogue", "")).strip()
        trigger = str(item.get("trigger_query", "")).strip()
        relation = str(item.get("relation_type", "unknown")).strip() or "unknown"
        if not cue or not trigger:
            continue
        rows.append(
            BenchmarkSubsetRow(
                qid=f"locomo_plus_{idx:04d}",
                source="Locomo-Plus",
                question=(
                    "Which earlier memory should constrain this later user message? "
                    f"Later message: {trigger}"
                ),
                answer=cue,
                support_text=cue,
                support_layer="raw_or_record",
                relation_type=relation,
                metadata={
                    "time_gap": item.get("time_gap", ""),
                    "model_name": item.get("model_name", ""),
                    "final_similarity_score": item.get("final_similarity_score", None),
                    "raw_index": idx - 1,
                },
            )
        )
    return rows


def write_subset_jsonl(rows: list[BenchmarkSubsetRow], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=True, sort_keys=True) + "\n")
    return out


def format_subset_markdown(rows: list[BenchmarkSubsetRow], source_url: str) -> str:
    by_relation: dict[str, int] = {}
    for row in rows:
        by_relation[row.relation_type] = by_relation.get(row.relation_type, 0) + 1
    lines = [
        "# Benchmark Subset Artifact",
        "",
        f"Source: {source_url}",
        f"Rows: {len(rows)}",
        "",
        "## Relation Mix",
        "",
        "| relation | count |",
        "|---|---:|",
    ]
    for relation, count in sorted(by_relation.items()):
        lines.append(f"| {relation} | {count} |")
    lines += [
        "",
        "## Format",
        "",
        "Each row turns a LoCoMo-Plus cue/trigger pair into a NapMem-style evidence navigation item.",
        "The expected answer is the earlier cue dialogue; evaluation should focus on evidence recovery,",
        "not free-form answer exactness.",
        "",
    ]
    return "\n".join(lines)


def write_subset_markdown(rows: list[BenchmarkSubsetRow], path: str | Path, source_url: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_subset_markdown(rows, source_url), encoding="utf-8")
    return out
