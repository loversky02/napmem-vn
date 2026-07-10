"""Offline verification of the GRPO training signal ($0, no GPU).

Before spending GPU on a GRPO run we check that the reward function actually
shapes the policy the way we intend. We synthesize candidate completions for four
policy archetypes over the real 40-row seed and score them under both reward
variants (F+C+U and the F+C ablation). This both smoke-tests the TRL-shaped
reward function and quantifies the honest finding: the usage bonus U rewards
calling a memory tool for its own sake.

Archetypes (all "correct" ones copy the gold answer):
  oracle     correct; calls a tool iff the question needs memory  (the target policy)
  tool_spam  correct; ALWAYS calls a tool, even on non-memory questions
  no_tool    correct; NEVER calls a tool
  wrong      wrong answer; calls a tool iff the question needs memory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from napmem.grpo import make_grpo_reward_fn

TOOL = [{"tool": "search_records"}]


def _candidate(answer: str, use_tool: bool) -> str:
    return json.dumps({"answer": answer, "tool_calls": TOOL if use_tool else []})


def build_candidates(rows: list[dict]) -> dict[str, list[str]]:
    archetypes: dict[str, list[str]] = {"oracle": [], "tool_spam": [], "no_tool": [], "wrong": []}
    for row in rows:
        gold = row["gold_answer"]
        needs = bool(row["requires_memory"])
        archetypes["oracle"].append(_candidate(gold, use_tool=needs))
        archetypes["tool_spam"].append(_candidate(gold, use_tool=True))
        archetypes["no_tool"].append(_candidate(gold, use_tool=False))
        archetypes["wrong"].append(_candidate("__wrong__", use_tool=needs))
    return archetypes


def mean(values: list[float]) -> float:
    return sum(values) / (len(values) or 1)


def evaluate(rows: list[dict]) -> dict:
    gold = [row["gold_answer"] for row in rows]
    mode = [row["answer_mode"] for row in rows]
    mem_mask = [bool(row["requires_memory"]) for row in rows]
    reward_fcu = make_grpo_reward_fn(use_usage_bonus=True)
    reward_fc = make_grpo_reward_fn(use_usage_bonus=False)

    candidates = build_candidates(rows)
    out: dict[str, dict[str, float]] = {}
    for name, completions in candidates.items():
        r_fcu = reward_fcu(completions, gold, mode)
        r_fc = reward_fc(completions, gold, mode)
        mem = [r for r, m in zip(r_fcu, mem_mask) if m]
        non = [r for r, m in zip(r_fcu, mem_mask) if not m]
        out[name] = {
            "reward_FCU": mean(r_fcu),
            "reward_FC": mean(r_fc),
            "FCU_memory": mean(mem),
            "FCU_non_memory": mean(non),
        }
    return out


def build_report(rows: list[dict], table: dict) -> dict:
    return {
        "name": "grpo-reward-signal-smoke",
        "n_rows": len(rows),
        "archetypes": table,
        "finding": (
            "The U (usage) term rewards tool calls for their own sake: on non-memory "
            "questions the target policy (oracle, no tool) scores {oracle_non:.2f} under "
            "F+C+U while tool_spam scores {spam_non:.2f}; the no_tool policy that is "
            "always correct still averages {notool_fcu:.2f} under F+C+U vs "
            "{notool_fc:.2f} under F+C. GRPO under F+C+U is therefore trained to call "
            "memory even when it should not."
        ).format(
            oracle_non=table["oracle"]["FCU_non_memory"],
            spam_non=table["tool_spam"]["FCU_non_memory"],
            notool_fcu=table["no_tool"]["reward_FCU"],
            notool_fc=table["no_tool"]["reward_FC"],
        ),
    }


def format_markdown(report: dict) -> str:
    lines = [
        "# GRPO Reward-Signal Smoke (offline, $0)",
        "",
        f"Seed rows: {report['n_rows']}",
        "",
        "| archetype | R(F+C+U) | R(F+C) | F+C+U memory | F+C+U non-memory |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in report["archetypes"].items():
        lines.append(
            f"| {name} | {stats['reward_FCU']:.2f} | {stats['reward_FC']:.2f} | "
            f"{stats['FCU_memory']:.2f} | {stats['FCU_non_memory']:.2f} |"
        )
    lines += ["", "## Finding", "", report["finding"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, default=Path("data/grpo_seed.jsonl"))
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--artifact-stem", default="grpo_reward_smoke")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.seed.read_text(encoding="utf-8").splitlines() if line.strip()]
    table = evaluate(rows)
    report = build_report(rows, table)

    print("archetype    R(F+C+U)  R(F+C)  FCU-mem  FCU-nonmem")
    for name, stats in table.items():
        print(
            f"{name:<11}  {stats['reward_FCU']:>7.2f}  {stats['reward_FC']:>6.2f}  "
            f"{stats['FCU_memory']:>7.2f}  {stats['FCU_non_memory']:>9.2f}"
        )
    print("\n" + report["finding"])

    if args.artifacts:
        out = args.artifacts
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.artifact_stem}.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out / f"{args.artifact_stem}.md").write_text(format_markdown(report), encoding="utf-8")
        print(f"\nwrote artifacts: {out / (args.artifact_stem + '.json')} {out / (args.artifact_stem + '.md')}")


if __name__ == "__main__":
    main()
