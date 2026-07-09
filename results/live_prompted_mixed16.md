# NapMem-VN Live Prompted Artifact

Generated: 2026-07-09
Selection: q_raw_nate,q_record_nate,q_topic_state,q_profile_nate,q_raw_allergy,q_record_mira,q_topic_linh,q_profile_style,q_non_memory_math,q_non_memory_capital,q_non_memory_water,q_non_memory_week,q_non_memory_color,q_non_memory_planet,q_non_memory_square,q_non_memory_author
Examples: 16

## Summary

- Accuracy: 1.00
- Average tool calls: 0.75
- Unnecessary memory calls: 0.00
- Exact fail: 0.00
- Quote fail: 0.00
- Provider error rate: 0.00
- R+U: 0.50
- F+C: 1.00
- Delta: -0.50

## Runs

| qid | ok | calls | memory | quote ok | answer |
|---|---:|---:|---:|---:|---|
| q_raw_nate | 1 | 2 | 1 | 1 | Tampa beach |
| q_record_nate | 1 | 2 | 1 | 1 | Nate visited Tampa beach on a turtle trip for peace and relaxation. |
| q_topic_state | 1 | 1 | 1 | 1 | Florida |
| q_profile_nate | 1 | 1 | 1 | 1 | peaceful beach trips |
| q_raw_allergy | 1 | 2 | 1 | 1 | almonds |
| q_record_mira | 1 | 2 | 1 | 1 | throat irritation |
| q_topic_linh | 1 | 1 | 1 | 1 | Research-plan answers for Linh should begin in Vietnamese. |
| q_profile_style | 1 | 1 | 1 | 1 | crisp bullets |
| q_non_memory_math | 1 | 0 | 0 | 1 | 2 + 2 = 4 |
| q_non_memory_capital | 1 | 0 | 0 | 1 | Paris |
| q_non_memory_water | 1 | 0 | 0 | 1 | H₂O |
| q_non_memory_week | 1 | 0 | 0 | 1 | There are 7 days in a week. |
| q_non_memory_color | 1 | 0 | 0 | 1 | Pink |
| q_non_memory_planet | 1 | 0 | 0 | 1 | Mars |
| q_non_memory_square | 1 | 0 | 0 | 1 | 81 |
| q_non_memory_author | 1 | 0 | 0 | 1 | Jane Austen wrote Pride and Prejudice. |

## Finding

**Prompted trajectories expose the usage-bonus tradeoff**

When prompted navigation correctly skips memory for non-memory questions, F+C can exceed F+C+U because the usage term withholds credit for correct no-memory answers.
