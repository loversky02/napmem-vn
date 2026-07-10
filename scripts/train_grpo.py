"""GRPO training entrypoint for the NapMem navigation policy (RunPod / GPU).

This is the money-plot run: train the same base model twice — once with the
paper reward F+C+U and once with the F+C ablation — then compare unnecessary
memory-call rate. The offline reward smoke (`scripts/grpo_reward_smoke.py`)
already predicts F+C+U will inflate tool use; this confirms it after training.

Heavy deps (torch / transformers / trl / datasets) are imported lazily inside
`main` so the module stays importable on a CPU-only box for tests and
`py_compile`. Run it on a GPU pod; see `docs/GRPO_MINIRUN.md` for the env pins.

    python scripts/train_grpo.py --model Qwen/Qwen3-4B --out runs/fcu --max-steps 60
    python scripts/train_grpo.py --model Qwen/Qwen3-4B --out runs/fc --max-steps 60 --no-usage-bonus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from napmem.grpo import make_grpo_reward_fn


def load_seed_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--seed", type=Path, default=Path("data/grpo_seed.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/napmem_grpo"))
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--group-size", type=int, default=4, help="GRPO generations per prompt")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-completion-length", type=int, default=200)
    parser.add_argument("--no-lora", action="store_true", help="full fine-tune instead of LoRA (needs a big GPU)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--no-usage-bonus",
        action="store_true",
        help="train the F+C ablation (drop the U term) for the money-plot comparison",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Lazy GPU imports so this file stays importable on CPU-only machines.
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    rows = load_seed_rows(args.seed)
    dataset = Dataset.from_list(
        [
            {"prompt": row["prompt"], "gold_answer": row["gold_answer"], "answer_mode": row["answer_mode"]}
            for row in rows
        ]
    )

    reward_fn = make_grpo_reward_fn(use_usage_bonus=not args.no_usage_bonus)

    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    config = GRPOConfig(
        output_dir=str(args.out),
        per_device_train_batch_size=args.group_size,
        num_generations=args.group_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=True,
        logging_steps=1,
        save_steps=args.max_steps,
        report_to=[],
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.out))
    print(f"saved GRPO model ({'F+C' if args.no_usage_bonus else 'F+C+U'}) to {args.out}")


if __name__ == "__main__":
    main()
