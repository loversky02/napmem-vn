import json

from napmem.artifacts import build_offline_report, write_live_artifacts, write_offline_artifacts
from napmem.synthetic import build_synthetic_benchmark


def test_offline_report_captures_usage_bonus_finding(tmp_path):
    bench = build_synthetic_benchmark(tmp_path / "mem")
    report = build_offline_report(bench)

    assert report["benchmark"]["num_examples"] == 40
    assert report["finding"]["oracle_delta"] < 0
    assert report["finding"]["passive_topk_delta"] > 0


def test_write_offline_artifacts(tmp_path):
    bench = build_synthetic_benchmark(tmp_path / "mem")
    json_path, md_path = write_offline_artifacts(bench, tmp_path / "artifacts")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")

    assert payload["results"]["oracle"]["accuracy"] == 1.0
    assert "Usage bonus can penalize correct memory skipping" in markdown


def test_write_live_artifacts(tmp_path):
    rows = [
        {
            "qid": "q",
            "question": "What is 2 + 2?",
            "gold": "4",
            "answer": "4",
            "evidence_quote": "",
            "correct": True,
            "quote_ok": True,
            "tool_calls": 0,
            "memory_used": False,
            "requires_memory": False,
            "answer_mode": "exact_string",
            "reason": "basic arithmetic",
            "backend_error": False,
        }
    ]
    summary = {
        "accuracy": 1.0,
        "avg_tool_calls": 0.0,
        "unnecessary_memory_call_rate": 0.0,
        "exact_fail_rate": 0.0,
        "quote_fail_rate": 0.0,
        "error_rate": 0.0,
        "reward_with_u": 0.0,
        "reward_without_u": 1.0,
        "usage_bonus_delta": -1.0,
    }

    json_path, md_path = write_live_artifacts(rows, summary, tmp_path / "artifacts", "q")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert payload["summary"]["usage_bonus_delta"] == -1.0
    assert "Live Prompted Artifact" in markdown
