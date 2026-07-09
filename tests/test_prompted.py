from napmem.prompted import PromptedNavigator, answer_correct, quote_supports_answer, route_hint
from napmem.synthetic import build_synthetic_benchmark


def test_prompted_navigator_uses_tools_until_answer(tmp_path):
    bench = build_synthetic_benchmark(tmp_path)
    example = next(ex for ex in bench.examples if ex.qid == "q_record_instruction")
    replies = iter([
        '{"tool":"search_records","query":"research plans language first"}',
        '{"tool":"get_records","ids":["r_linh_language"]}',
        '{"answer":"Vietnamese","evidence_quote":"Vietnamese first","reason":"record r_linh_language says this instruction."}',
    ])
    nav = PromptedNavigator(lambda messages: next(replies))
    result = nav.answer(bench, example)
    assert answer_correct(result.answer, example.answer, example.answer_mode)
    assert quote_supports_answer(result.evidence_quote, example.answer, example.answer_mode)
    assert [step.tool for step in result.trace] == ["search_records", "get_records"]


def test_prompted_navigator_can_skip_memory_for_non_memory_question(tmp_path):
    bench = build_synthetic_benchmark(tmp_path)
    example = next(ex for ex in bench.examples if ex.qid == "q_non_memory_math")
    nav = PromptedNavigator(lambda messages: '{"answer":"4","evidence_quote":"","reason":"basic arithmetic."}')
    result = nav.answer(bench, example)
    assert answer_correct(result.answer, example.answer, example.answer_mode)
    assert result.trace == []


def test_prompted_navigator_returns_backend_error(tmp_path):
    bench = build_synthetic_benchmark(tmp_path)
    example = next(ex for ex in bench.examples if ex.qid == "q_non_memory_math")

    def fail(_messages):
        raise TimeoutError("provider stalled")

    result = PromptedNavigator(fail).answer(bench, example)

    assert result.answer == ""
    assert result.reason.startswith("backend error: TimeoutError")


def test_exact_string_scoring_rejects_soft_generalization():
    assert not answer_correct("nuts", "almonds", "exact_string")
    assert not answer_correct("Jo prefers concise weekly summaries.", "crisp bullets", "exact_string")
    assert quote_supports_answer("Please avoid almonds in snack suggestions.", "almonds", "exact_string")


def test_exact_string_scoring_normalizes_unicode_digits():
    assert answer_correct("H₂O", "H2O", "exact_string")
    assert quote_supports_answer("The formula is H₂O.", "H2O", "exact_string")


def test_semantic_scoring_accepts_paraphrase_but_exact_stays_strict():
    assert answer_correct("Research-plan answers should begin in Vietnamese.", "Vietnamese first", "semantic")
    assert not answer_correct("Research-plan answers should begin in Vietnamese.", "Vietnamese first", "exact_string")


def test_route_hint_matches_memory_layer(tmp_path):
    bench = build_synthetic_benchmark(tmp_path)
    hints = {example.qid: route_hint(example) for example in bench.examples}
    assert "search_conversations" in hints["q_raw_nate"]
    assert "search_records" in hints["q_record_nate"]
    assert "profile.md" in hints["q_profile_nate"]
    assert "topic file" in hints["q_topic_state"]
    assert "do not call" in hints["q_non_memory_math"]
