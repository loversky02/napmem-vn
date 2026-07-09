import json

from napmem.benchmarks import locomo_plus_to_rows, write_subset_jsonl, write_subset_markdown


def test_locomo_plus_rows_preserve_cue_and_trigger(tmp_path):
    rows = locomo_plus_to_rows(
        [
            {
                "relation_type": "causal",
                "cue_dialogue": "A: I avoid late meetings because sleep matters.\nB: That is a clear boundary.",
                "trigger_query": "A: I accepted a midnight sync and feel awful.",
                "time_gap": "one week later",
                "model_name": "fixture",
                "final_similarity_score": 0.5,
            }
        ],
        limit=1,
    )

    assert rows[0].relation_type == "causal"
    assert "midnight sync" in rows[0].question
    assert "sleep matters" in rows[0].answer

    jsonl = write_subset_jsonl(rows, tmp_path / "subset.jsonl")
    md = write_subset_markdown(rows, tmp_path / "subset.md", "fixture://locomo")

    assert json.loads(jsonl.read_text(encoding="utf-8"))["qid"] == "locomo_plus_0001"
    assert "Relation Mix" in md.read_text(encoding="utf-8")
