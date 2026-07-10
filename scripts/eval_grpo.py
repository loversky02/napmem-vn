"""Evaluate a (base or LoRA) checkpoint on the NapMem seed — the money-plot metric.

Loads the model in-process and generates one action per seed prompt, then scores
accuracy and, crucially, the unnecessary-memory-call rate on non-memory questions.
No model server needed. Run once per checkpoint (F+C+U and F+C) and compare.

    python scripts/eval_grpo.py --base Qwen/Qwen3-4B --adapter runs/fcu --label fcu --out results/grpo_eval_fcu.json
    python scripts/eval_grpo.py --base Qwen/Qwen3-4B --adapter runs/fc  --label fc  --out results/grpo_eval_fc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from napmem.grpo import parse_json_payload
from napmem.prompted import answer_correct


def load_seed_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_completion(text: str) -> tuple[str | None, bool, bool]:
    """Return (answer, has_tool_call, is_malformed) from a raw completion string."""
    payload = parse_json_payload(text)
    if payload is None or "answer" not in payload:
        return None, False, True
    tool_calls = payload.get("tool_calls") or []
    has_tool = isinstance(tool_calls, list) and len(tool_calls) > 0
    return str(payload.get("answer", "")), has_tool, False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir; omit to eval the base model")
    parser.add_argument("--seed", type=Path, default=Path("data/grpo_seed.jsonl"))
    parser.add_argument("--label", default="model")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (0 = all)")
    parser.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    parser.add_argument("--n-samples", type=int, default=1, help="samples per question (probabilistic rates)")
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = load_seed_rows(args.seed)
    if args.limit:
        rows = rows[: args.limit]

    do_sample = args.sample or args.n_samples > 1
    k = max(1, args.n_samples)
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "num_return_sequences": k}
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    samples = []
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        completions = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Per-question rates, averaged over the k samples (probabilistic when sampling).
        n_correct = n_tool = n_bad = 0
        for completion in completions:
            answer, has_tool, malformed = parse_completion(completion)
            n_tool += int(has_tool)
            n_bad += int(malformed)
            n_correct += int((not malformed) and answer_correct(answer or "", row["gold_answer"], row["answer_mode"]))
        m = len(completions) or 1
        samples.append({
            "qid": row["qid"],
            "requires_memory": bool(row["requires_memory"]),
            "support_layer": row["support_layer"],
            "has_tool": n_tool / m,
            "malformed": n_bad / m,
            "correct": n_correct / m,
            "completion": completions[0][:400],
        })

    n = len(samples) or 1
    mem = [s for s in samples if s["requires_memory"]]
    non = [s for s in samples if not s["requires_memory"]]
    metrics = {
        "label": args.label,
        "n": len(samples),
        "accuracy": sum(s["correct"] for s in samples) / n,
        "memory_recall": sum(s["correct"] for s in mem) / (len(mem) or 1),
        "malformed_rate": sum(s["malformed"] for s in samples) / n,
        "memory_call_rate": sum(s["has_tool"] for s in samples) / n,
        "unnecessary_memory_call_rate": sum(s["has_tool"] for s in non) / (len(non) or 1),
    }
    report = {"metrics": metrics, "samples": samples}

    print(json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
