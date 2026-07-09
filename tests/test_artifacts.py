import json

from napmem.artifacts import build_offline_report, write_offline_artifacts
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
