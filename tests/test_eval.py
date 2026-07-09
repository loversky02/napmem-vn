from napmem.eval import evaluate_all, evaluate_strategy
from napmem.synthetic import build_synthetic_benchmark
from napmem.ablation import reward_ablation


def test_synthetic_benchmark_covers_all_memory_layers(tmp_path):
    bench = build_synthetic_benchmark(tmp_path)
    assert len(bench.examples) == 40
    assert {example.tag for example in bench.examples} == {
        "raw",
        "record",
        "topic",
        "profile",
        "non_memory",
    }


def test_active_navigation_beats_passive_topk_on_synthetic(tmp_path):
    bench = build_synthetic_benchmark(tmp_path)
    results = evaluate_all(bench)
    assert results["passive_topk"]["memory_accuracy"] < results["drilldown"]["memory_accuracy"]
    assert results["oracle"]["memory_accuracy"] == 1.0
    assert results["drilldown"]["avg_tool_calls"] > results["passive_topk"]["avg_tool_calls"]


def test_usage_reward_can_prefer_wrong_memory_call_over_no_memory(tmp_path):
    bench = build_synthetic_benchmark(tmp_path)
    passive = evaluate_strategy(bench, "passive_topk")
    wrong_with_memory = next(run for run in passive if not run.correct and run.memory_used)
    assert wrong_with_memory.reward_with_u == -0.5
    assert wrong_with_memory.reward_without_u == -1.0


def test_no_memory_control_exposes_unnecessary_memory_calls(tmp_path):
    bench = build_synthetic_benchmark(tmp_path)
    results = evaluate_all(bench)
    assert results["no_memory"]["unnecessary_memory_call_rate"] == 0.0
    assert results["drilldown"]["unnecessary_memory_call_rate"] == 1.0


def test_reward_ablation_exposes_usage_bonus_delta(tmp_path):
    bench = build_synthetic_benchmark(tmp_path)
    rows = {row.strategy: row for row in reward_ablation(bench)}
    assert rows["oracle"].usage_bonus_delta < 0
    assert rows["passive_topk"].usage_bonus_delta > 0
