import json

from napmem.grpo import (
    build_grpo_rows,
    make_grpo_reward_fn,
    parse_json_payload,
    reward_candidate,
    score_completion,
    write_grpo_jsonl,
)
from napmem.synthetic import build_synthetic_benchmark


def test_grpo_rows_cover_synthetic_benchmark(tmp_path):
    bench = build_synthetic_benchmark(tmp_path / "mem")
    rows = build_grpo_rows(bench)

    assert len(rows) == 40
    assert any(not row.requires_memory for row in rows)
    assert "tool_calls" in next(row.prompt for row in rows if not row.requires_memory)


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


def test_score_completion_reward_variants():
    good = '{"answer":"BWP-","tool_calls":[{"tool":"search_records"}]}'
    assert score_completion(good, "BWP-", "exact_string", use_usage_bonus=True) == 1.0

    # correct but no tool: penalized under F+C+U, full under F+C
    no_tool = '{"answer":"BWP-","tool_calls":[]}'
    assert score_completion(no_tool, "BWP-", "exact_string", use_usage_bonus=True) == 0.0
    assert score_completion(no_tool, "BWP-", "exact_string", use_usage_bonus=False) == 1.0

    # malformed JSON -> -1
    assert score_completion("not json", "BWP-") == -1.0

    # wrong answer + tool -> -0.5 under F+C+U
    wrong = '{"answer":"nope","tool_calls":[{"tool":"x"}]}'
    assert score_completion(wrong, "BWP-", "exact_string", use_usage_bonus=True) == -0.5

    # chat-list completion is reduced to its final content
    chat = [{"role": "assistant", "content": good}]
    assert score_completion(chat, "BWP-", "exact_string", use_usage_bonus=True) == 1.0


def test_parse_json_payload_tolerates_prose_and_fences():
    # clean JSON
    assert parse_json_payload('{"answer":"x","tool_calls":[]}')["answer"] == "x"
    # JSON wrapped in prose / markdown fence + trailing text
    wrapped = 'Sure! ```json\n{"answer":"Tampa beach","tool_calls":[{"tool":"search_records"}]}\n``` done'
    payload = parse_json_payload(wrapped)
    assert payload["answer"] == "Tampa beach"
    assert payload["tool_calls"]
    # no JSON at all
    assert parse_json_payload("no json here") is None


def test_grpo_rows_put_evidence_in_memory_prompts(tmp_path):
    bench = build_synthetic_benchmark(tmp_path / "mem")
    rows = {row.qid: row for row in build_grpo_rows(bench)}

    # a record-layer memory question carries its supporting evidence text
    mem_row = rows["q_record_instruction"]
    assert "Memory evidence:" in mem_row.prompt
    assert mem_row.gold_answer.lower() in mem_row.prompt.lower()

    # a non-memory question is shown plainly: no evidence block and no skip hint,
    # so the tool-call decision is left to the policy
    non_row = rows["q_non_memory_math"]
    assert "Memory evidence:" not in non_row.prompt
    assert "no memory" not in non_row.prompt.lower()
    assert non_row.prompt.rstrip().endswith("JSON:")


def test_grpo_reward_fn_flags_tool_spam():
    # TRL-shaped call: (completions, gold_answer, answer_mode) as aligned lists
    completions = [
        '{"answer":"4","tool_calls":[]}',              # non-memory correct, no tool (ideal)
        '{"answer":"4","tool_calls":[{"tool":"x"}]}',  # non-memory correct, tool spam
    ]
    gold = ["4", "4"]
    mode = ["semantic", "semantic"]

    fcu = make_grpo_reward_fn(use_usage_bonus=True)(completions, gold, mode)
    fc = make_grpo_reward_fn(use_usage_bonus=False)(completions, gold, mode)

    # Under F+C+U the usage bonus makes tool spam outscore the ideal no-tool policy.
    assert fcu[1] > fcu[0]
    # Under the F+C ablation both correct answers tie.
    assert fc[0] == fc[1] == 1.0
