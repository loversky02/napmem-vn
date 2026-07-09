from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from napmem.grpo import write_grpo_jsonl
from napmem.synthetic import build_synthetic_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NapMem synthetic GRPO seed data.")
    parser.add_argument("--out", type=Path, default=Path("data/grpo_seed.jsonl"))
    parser.add_argument("--root", type=Path, default=None, help="Optional memory-bank output directory.")
    args = parser.parse_args()

    if args.root:
        bench = build_synthetic_benchmark(args.root)
        out = write_grpo_jsonl(bench, args.out)
        print(f"wrote {out}")
        return

    with TemporaryDirectory() as td:
        bench = build_synthetic_benchmark(td)
        out = write_grpo_jsonl(bench, args.out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
