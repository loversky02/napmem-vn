from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from .ablation import reward_ablation
from .eval import evaluate_all
from .synthetic import SyntheticBenchmark


def build_offline_report(bench: SyntheticBenchmark) -> dict[str, Any]:
    results = evaluate_all(bench)
    ablation = [asdict(row) for row in reward_ablation(bench)]
    return {
        "generated_on": date.today().isoformat(),
        "benchmark": {
            "name": "napmem-vn synthetic",
            "num_examples": len(bench.examples),
            "layers": ["raw", "record", "topic", "profile", "non_memory"],
        },
        "results": results,
        "reward_ablation": ablation,
        "finding": {
            "title": "Usage bonus can penalize correct memory skipping",
            "summary": (
                "The paper-style F+C+U reward gives no usage credit when the agent "
                "correctly answers non-memory questions without calling memory. In "
                "the synthetic oracle, F+C+U is lower than F+C despite perfect accuracy."
            ),
            "oracle_delta": results["oracle"]["reward_with_u"] - results["oracle"]["reward_without_u"],
            "passive_topk_delta": results["passive_topk"]["reward_with_u"]
            - results["passive_topk"]["reward_without_u"],
        },
    }


def write_offline_artifacts(bench: SyntheticBenchmark, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_offline_report(bench)
    json_path = out / "offline_synthetic.json"
    md_path = out / "offline_synthetic.md"
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_offline_markdown(report), encoding="utf-8")
    return json_path, md_path


def build_live_report(
    rows: list[dict[str, Any]],
    summary: dict[str, float],
    selection: str,
) -> dict[str, Any]:
    return {
        "generated_on": date.today().isoformat(),
        "benchmark": {
            "name": "napmem-vn prompted synthetic",
            "selection": selection,
            "num_examples": len(rows),
        },
        "summary": summary,
        "runs": rows,
        "finding": {
            "title": "Prompted trajectories expose the usage-bonus tradeoff",
            "summary": (
                "When prompted navigation correctly skips memory for non-memory "
                "questions, F+C can exceed F+C+U because the usage term withholds "
                "credit for correct no-memory answers."
            ),
        },
    }


def write_live_artifacts(
    rows: list[dict[str, Any]],
    summary: dict[str, float],
    out_dir: str | Path,
    selection: str,
    stem: str = "live_prompted",
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_live_report(rows, summary, selection)
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_live_markdown(report), encoding="utf-8")
    return json_path, md_path


def format_offline_markdown(report: dict[str, Any]) -> str:
    rows = report["results"]
    ablation = report["reward_ablation"]
    lines = [
        "# NapMem-VN Offline Synthetic Artifact",
        "",
        f"Generated: {report['generated_on']}",
        f"Examples: {report['benchmark']['num_examples']}",
        "",
        "## Strategy Results",
        "",
        "| strategy | acc | memory acc | exact fail | calls | unnecessary | R+U | F+C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['accuracy']:.2f} | {row['memory_accuracy']:.2f} | "
            f"{row['exact_fail_rate']:.2f} | {row['avg_tool_calls']:.2f} | "
            f"{row['unnecessary_memory_call_rate']:.2f} | {row['reward_with_u']:.2f} | "
            f"{row['reward_without_u']:.2f} |"
        )
    lines += [
        "",
        "## Reward Usage Ablation",
        "",
        "| strategy | acc | calls | unnecessary | R+U | F+C | delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ablation:
        lines.append(
            f"| {row['strategy']} | {row['accuracy']:.2f} | {row['avg_tool_calls']:.2f} | "
            f"{row['unnecessary_memory_call_rate']:.2f} | {row['reward_with_u']:.2f} | "
            f"{row['reward_without_u']:.2f} | {row['usage_bonus_delta']:.2f} |"
        )
    lines += [
        "",
        "## Finding",
        "",
        f"**{report['finding']['title']}**",
        "",
        report["finding"]["summary"],
        "",
        f"Oracle delta: {report['finding']['oracle_delta']:.2f}",
        f"Passive top-k delta: {report['finding']['passive_topk_delta']:.2f}",
        "",
    ]
    return "\n".join(lines)


def format_live_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# NapMem-VN Live Prompted Artifact",
        "",
        f"Generated: {report['generated_on']}",
        f"Selection: {report['benchmark']['selection']}",
        f"Examples: {report['benchmark']['num_examples']}",
        "",
        "## Summary",
        "",
        f"- Accuracy: {summary['accuracy']:.2f}",
        f"- Average tool calls: {summary['avg_tool_calls']:.2f}",
        f"- Unnecessary memory calls: {summary['unnecessary_memory_call_rate']:.2f}",
        f"- Exact fail: {summary['exact_fail_rate']:.2f}",
        f"- Quote fail: {summary['quote_fail_rate']:.2f}",
        f"- Provider error rate: {summary['error_rate']:.2f}",
        f"- R+U: {summary['reward_with_u']:.2f}",
        f"- F+C: {summary['reward_without_u']:.2f}",
        f"- Delta: {summary['usage_bonus_delta']:.2f}",
        "",
        "## Runs",
        "",
        "| qid | ok | calls | memory | quote ok | answer |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["runs"]:
        answer = str(row["answer"]).replace("|", "\\|")
        lines.append(
            f"| {row['qid']} | {int(row['correct'])} | {row['tool_calls']} | "
            f"{int(row['memory_used'])} | {int(row['quote_ok'])} | {answer} |"
        )
    lines += [
        "",
        "## Finding",
        "",
        f"**{report['finding']['title']}**",
        "",
        report["finding"]["summary"],
        "",
    ]
    return "\n".join(lines)
