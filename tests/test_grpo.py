import json

from napmem.grpo import build_grpo_rows, reward_candidate, write_grpo_jsonl
from napmem.synthetic import build_synthetic_benchmark


def test_grpo_rows_cover_synthetic_benchmark(tmp_path):
    bench = build_synthetic_benchmark(tmp_path / "mem")
    rows = build_grpo_rows(bench)

    assert len(rows) == 40
    assert any(not row.requires_memory for row in rows)
    assert "skip memory" in next(row.prompt for row in rows if not row.requires_memory)


def test_grpo_reward_exposes_usage_bonus_penalty(tmp_path):
    bench = build_synthetic_benchmark(tmp_path / "mem")
    row = next(row for row in build_grpo_rows(bench) if row.qid == "q_non_memory_math")
    candidate = {"answer": "4", "tool_calls": []}

    assert reward_candidate(candidate, row, use_usage_bonus=True) == 0.0
    assert reward_candidate(candidate, row, use_usage_bonus=False) == 1.0


def test_write_grpo_jsonl(tmp_path):
    bench = build_synthetic_benchmark(tmp_path / "mem")
    out = write_grpo_jsonl(bench, tmp_path / "grpo.jsonl")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 40
    assert rows[0]["qid"] == "q_raw_nate"
