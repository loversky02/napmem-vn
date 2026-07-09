# NapMem-VN Offline Synthetic Artifact

Generated: 2026-07-09
Examples: 40

## Strategy Results

| strategy | acc | memory acc | exact fail | calls | unnecessary | R+U | F+C |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_memory | 0.20 | 0.00 | 0.68 | 0.00 | 0.00 | -0.80 | -0.60 |
| passive_topk | 0.40 | 0.25 | 0.47 | 1.00 | 1.00 | 0.10 | -0.20 |
| records_only | 0.40 | 0.25 | 0.47 | 2.00 | 1.00 | 0.10 | -0.20 |
| upper_first | 0.60 | 0.50 | 0.40 | 9.00 | 1.00 | 0.40 | 0.20 |
| drilldown | 0.70 | 0.62 | 0.20 | 4.12 | 1.00 | 0.55 | 0.40 |
| oracle | 1.00 | 1.00 | 0.00 | 0.80 | 0.00 | 0.80 | 1.00 |

## Reward Usage Ablation

| strategy | acc | calls | unnecessary | R+U | F+C | delta |
|---|---:|---:|---:|---:|---:|---:|
| no_memory | 0.20 | 0.00 | 0.00 | -0.80 | -0.60 | -0.20 |
| passive_topk | 0.40 | 1.00 | 1.00 | 0.10 | -0.20 | 0.30 |
| records_only | 0.40 | 2.00 | 1.00 | 0.10 | -0.20 | 0.30 |
| upper_first | 0.60 | 9.00 | 1.00 | 0.40 | 0.20 | 0.20 |
| drilldown | 0.70 | 4.12 | 1.00 | 0.55 | 0.40 | 0.15 |
| oracle | 1.00 | 0.80 | 0.00 | 0.80 | 1.00 | -0.20 |

## Finding

**Usage bonus can penalize correct memory skipping**

The paper-style F+C+U reward gives no usage credit when the agent correctly answers non-memory questions without calling memory. In the synthetic oracle, F+C+U is lower than F+C despite perfect accuracy.

Oracle delta: -0.20
Passive top-k delta: 0.30
