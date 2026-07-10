# GRPO Reward-Signal Smoke (offline, $0)

Seed rows: 40

| archetype | R(F+C+U) | R(F+C) | F+C+U memory | F+C+U non-memory |
|---|---:|---:|---:|---:|
| oracle | 0.80 | 1.00 | 1.00 | 0.00 |
| tool_spam | 1.00 | 1.00 | 1.00 | 1.00 |
| no_tool | 0.00 | 1.00 | 0.00 | 0.00 |
| wrong | -0.60 | -1.00 | -0.50 | -1.00 |

## Finding

The U (usage) term rewards tool calls for their own sake: on non-memory questions the target policy (oracle, no tool) scores 0.00 under F+C+U while tool_spam scores 1.00; the no_tool policy that is always correct still averages 0.00 under F+C+U vs 1.00 under F+C. GRPO under F+C+U is therefore trained to call memory even when it should not.
