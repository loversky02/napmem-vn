# NapMem-VN Live Prompted Artifact

Generated: 2026-07-09
Selection: all
Examples: 40

## Summary

- Accuracy: 0.97
- Average tool calls: 1.15
- Unnecessary memory calls: 0.00
- Exact fail: 0.04
- Quote fail: 0.00
- Provider error rate: 0.00
- R+U: 0.76
- F+C: 0.95
- Delta: -0.19

## Runs

| qid | ok | calls | memory | quote ok | answer |
|---|---:|---:|---:|---:|---|
| q_raw_nate | 1 | 2 | 1 | 1 | Tampa beach |
| q_record_nate | 1 | 2 | 1 | 1 | Nate visited Tampa beach on a turtle trip for peace and relaxation. |
| q_topic_state | 1 | 1 | 1 | 1 | Florida |
| q_profile_nate | 1 | 1 | 1 | 1 | peaceful beach trips |
| q_raw_allergy | 1 | 2 | 1 | 1 | almonds |
| q_record_mira | 1 | 2 | 1 | 1 | throat irritation |
| q_topic_mira | 1 | 1 | 1 | 1 | Mira avoids almonds in snack planning |
| q_profile_mira | 1 | 1 | 1 | 1 | almonds |
| q_raw_linh | 1 | 2 | 1 | 1 | research plans |
| q_record_instruction | 1 | 2 | 1 | 1 | Vietnamese |
| q_topic_linh | 1 | 1 | 1 | 1 | Research-plan answers for Linh should begin in Vietnamese. |
| q_profile_linh | 0 | 1 | 1 | 1 | Vietnamese |
| q_raw_jo | 1 | 3 | 1 | 1 | fluff |
| q_record_jo | 1 | 2 | 1 | 1 | concise weekly summaries |
| q_topic_jo | 1 | 1 | 1 | 1 | crisp bullets |
| q_profile_style | 1 | 1 | 1 | 1 | crisp bullets |
| q_raw_ava | 1 | 2 | 1 | 1 | my baby |
| q_record_ava | 1 | 2 | 1 | 1 | 11 pm |
| q_topic_ava | 1 | 1 | 1 | 1 | Ava keeps notifications quiet late at night to avoid disturbing her baby's sleep. |
| q_profile_ava | 1 | 1 | 1 | 1 | quiet-notification hours after 11 pm |
| q_raw_bao | 1 | 2 | 1 | 1 | BWP- |
| q_record_bao | 1 | 1 | 1 | 1 | BWP- |
| q_topic_bao | 1 | 1 | 1 | 1 | Build with Paper consulting |
| q_profile_bao | 1 | 1 | 1 | 1 | BWP- |
| q_raw_cy | 1 | 2 | 1 | 1 | cilantro |
| q_record_cy | 1 | 1 | 1 | 1 | it makes soup taste soapy |
| q_topic_cy | 1 | 1 | 1 | 1 | soup |
| q_profile_cy | 1 | 1 | 1 | 1 | cilantro |
| q_raw_diego | 1 | 2 | 1 | 1 | daily |
| q_record_diego | 1 | 1 | 1 | 1 | Google Docs |
| q_topic_diego | 1 | 1 | 1 | 1 | Diego's workflow from lab notes to final documents involves taking lab notes in Obsidian, then drafting final documents in Google Docs. |
| q_profile_diego | 1 | 1 | 1 | 1 | Obsidian |
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
