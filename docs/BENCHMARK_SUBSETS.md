# Benchmark Subsets

## LoCoMo / LoCoMo-Plus

LoCoMo evaluates very long-term conversational memory across QA, summarization,
and dialogue-generation tasks. LoCoMo-Plus extends it with cognitive memory
items: cue dialogues paired with later trigger queries where the surface words
may not match the earlier constraint.

Local importer:

```bash
../automem-vn/.venv/bin/python scripts/import_locomo_plus.py --limit 16 --out data/locomo_plus_subset.jsonl --report results/locomo_plus_subset.md --insecure-ssl
```

The importer does not vendor the full benchmark. It normalizes a small subset
into JSONL rows containing:

- later trigger question
- earlier cue dialogue as expected evidence
- relation type and source metadata

This is an evidence-navigation subset rather than a final leaderboard.

Current artifact: `results/locomo_plus_subset.md`. It is a first-16 smoke slice
from the public LoCoMo-Plus JSON, currently all `causal` relation rows because
the upstream file is grouped. A later stratified subset should sample across
relation types.

Sources:

- LoCoMo paper: https://arxiv.org/abs/2402.17753
- LoCoMo-Plus repo/data pipeline: https://github.com/xjtuleeyf/Locomo-Plus
- LongMemEval paper/code pointer: https://arxiv.org/abs/2410.10813
