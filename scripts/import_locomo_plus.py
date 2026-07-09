from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from napmem.benchmarks import (
    LOCOMO_PLUS_RAW_URL,
    load_json_source,
    locomo_plus_to_rows,
    write_subset_jsonl,
    write_subset_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a small LoCoMo-Plus subset into NapMem rows.")
    parser.add_argument("--input", default=LOCOMO_PLUS_RAW_URL, help="Local JSON path or URL.")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("data/locomo_plus_subset.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("results/locomo_plus_subset.md"))
    parser.add_argument("--insecure-ssl", action="store_true", help="Disable SSL verification for URL inputs.")
    args = parser.parse_args()

    payload = load_json_source(args.input, verify_ssl=not args.insecure_ssl)
    if not isinstance(payload, list):
        raise SystemExit("LoCoMo-Plus input must be a JSON list.")
    rows = locomo_plus_to_rows(payload, limit=args.limit)
    jsonl = write_subset_jsonl(rows, args.out)
    report = write_subset_markdown(rows, args.report, args.input)
    print(f"wrote {len(rows)} rows: {jsonl} {report}")


if __name__ == "__main__":
    main()
