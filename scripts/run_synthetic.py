from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from napmem.ablation import format_ablation, reward_ablation
from napmem.eval import evaluate_all
from napmem.llm import client_from_env
from napmem.prompted import PromptedNavigator, answer_correct, quote_supports_answer
from napmem.reward import napmem_reward
from napmem.synthetic import build_synthetic_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline NapMem synthetic benchmark.")
    parser.add_argument("--root", type=Path, default=None, help="Optional memory-bank output directory.")
    parser.add_argument("--live", action="store_true", help="Also run the prompted 9router navigator.")
    parser.add_argument("--model", default=None, help="Override NAPMEM_MODEL/AUTOMEM_MODEL for live mode.")
    parser.add_argument("--insecure-ssl", action="store_true", help="Disable SSL verification for live mode.")
    parser.add_argument("--live-timeout", type=float, default=120.0, help="Per-request live LLM timeout in seconds.")
    parser.add_argument("--ablation", action="store_true", help="Print F+C+U vs F+C reward ablation.")
    parser.add_argument("--limit", type=int, default=0, help="Limit live prompted examples for quick smoke.")
    parser.add_argument("--qids", default="", help="Comma-separated live qids to run instead of the default order.")
    args = parser.parse_args()

    if args.root:
        bench = build_synthetic_benchmark(args.root)
        print_table(evaluate_all(bench))
        if args.ablation:
            print("\nreward_ablation")
            print(format_ablation(reward_ablation(bench)))
        if args.live:
            print_live(bench, args.model, not args.insecure_ssl, args.limit, args.live_timeout, args.qids)
        return

    with TemporaryDirectory() as tmp:
        bench = build_synthetic_benchmark(tmp)
        print_table(evaluate_all(bench))
        if args.ablation:
            print("\nreward_ablation")
            print(format_ablation(reward_ablation(bench)))
        if args.live:
            print_live(bench, args.model, not args.insecure_ssl, args.limit, args.live_timeout, args.qids)


def print_table(results: dict[str, dict[str, float]]) -> None:
    print(
        f"{'strategy':<14}{'acc':>6}{'memacc':>8}{'hit':>6}{'exact!':>8}{'calls':>8}"
        f"{'mem%':>7}{'unnec%':>8}{'R+U':>7}{'R':>7}"
    )
    print("-" * 79)
    for name, row in results.items():
        print(
            f"{name:<14}"
            f"{row['accuracy']:>6.2f}"
            f"{row['memory_accuracy']:>8.2f}"
            f"{row['evidence_hit']:>6.2f}"
            f"{row['exact_fail_rate']:>8.2f}"
            f"{row['avg_tool_calls']:>8.2f}"
            f"{row['memory_call_rate']:>7.2f}"
            f"{row['unnecessary_memory_call_rate']:>8.2f}"
            f"{row['reward_with_u']:>7.2f}"
            f"{row['reward_without_u']:>7.2f}"
        )


def print_live(
    bench,
    model: str | None,
    verify_ssl: bool,
    limit: int = 0,
    timeout_s: float = 120.0,
    qids: str = "",
) -> None:
    navigator = PromptedNavigator(client_from_env(model, verify_ssl=verify_ssl, timeout_s=timeout_s))
    total = correct = calls = unneeded = exact_fail = quote_fail = errors = 0
    reward_with_u = reward_without_u = 0.0
    examples = select_live_examples(bench.examples, limit, qids)
    print("\nlive_prompted")
    print("-" * 55)
    for example in examples:
        result = navigator.answer(bench, example)
        failed = result.reason.startswith("backend error:")
        ok = answer_correct(result.answer, example.answer, example.answer_mode)
        quote_ok = quote_supports_answer(result.evidence_quote, example.answer, example.answer_mode)
        total += 1
        correct += int(ok)
        errors += int(failed)
        calls += len(result.trace)
        used_memory = bool(result.trace)
        reward_with_u += napmem_reward(not failed, ok, used_memory, use_usage_bonus=True)
        reward_without_u += napmem_reward(not failed, ok, used_memory, use_usage_bonus=False)
        if example.answer_mode == "exact_string" and not ok:
            exact_fail += 1
        if example.answer_mode == "exact_string" and not quote_ok:
            quote_fail += 1
        if not example.requires_memory and result.trace:
            unneeded += 1
        print(
            f"{example.qid:<22} ok={int(ok)} calls={len(result.trace)} "
            f"quote_ok={int(quote_ok)} answer={result.answer!r} quote={result.evidence_quote!r}"
            f"{' error=' + result.reason if failed else ''}",
            flush=True,
        )
    exact_n = max(1, sum(e.answer_mode == "exact_string" for e in examples))
    print(
        f"summary acc={correct / total:.2f} avg_calls={calls / total:.2f} "
        f"unnecessary={unneeded / max(1, sum(not e.requires_memory for e in examples)):.2f} "
        f"exact_fail={exact_fail / exact_n:.2f} quote_fail={quote_fail / exact_n:.2f} "
        f"error_rate={errors / total:.2f} "
        f"R+U={reward_with_u / total:.2f} R={reward_without_u / total:.2f} "
        f"delta={(reward_with_u - reward_without_u) / total:.2f}"
    )


def select_live_examples(examples, limit: int = 0, qids: str = ""):
    wanted = [qid.strip() for qid in qids.split(",") if qid.strip()]
    if wanted:
        by_id = {example.qid: example for example in examples}
        missing = [qid for qid in wanted if qid not in by_id]
        if missing:
            raise SystemExit(f"Unknown qids: {', '.join(missing)}")
        return [by_id[qid] for qid in wanted]
    return examples[:limit] if limit > 0 else examples


if __name__ == "__main__":
    main()
