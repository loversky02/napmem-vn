from napmem.prompted import PromptedNavigator, answer_correct, quote_supports_answer
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


def test_exact_string_scoring_rejects_soft_generalization():
    assert not answer_correct("nuts", "almonds", "exact_string")
    assert not answer_correct("Jo prefers concise weekly summaries.", "crisp bullets", "exact_string")
    assert quote_supports_answer("Please avoid almonds in snack suggestions.", "almonds", "exact_string")
