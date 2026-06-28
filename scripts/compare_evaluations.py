"""Compare isolated eval_bilingual.py result directories."""

import argparse
import datetime
import json
import os
import re

from scripts.eval_bilingual import CHAT_TASKS, atomic_write_json, atomic_write_text


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run(path):
    path = os.path.abspath(os.path.expanduser(path))
    manifest_path = os.path.join(path, "manifest.json")
    status_path = os.path.join(path, "status.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing evaluation manifest: {manifest_path}")
    metrics = {}
    metrics_dir = os.path.join(path, "metrics")
    if os.path.isdir(metrics_dir):
        for filename in os.listdir(metrics_dir):
            if filename.endswith(".json"):
                metrics[filename[:-5]] = load_json(os.path.join(metrics_dir, filename))
    return {
        "path": path,
        "manifest": load_json(manifest_path),
        "status": load_json(status_path) if os.path.exists(status_path) else {},
        "metrics": metrics,
    }


def run_label(run):
    model = run["manifest"]["model"]
    return f"{model['source']}/{model['tag']}/{model['step']}"


def nested_get(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def collect_rows(runs):
    rows = []
    for run in runs:
        metrics = run["metrics"]
        row = {
            "label": run_label(run),
            "run_id": run["manifest"]["run_id"],
            "path": run["path"],
            "state": run["status"].get("state"),
            "language": {},
            "chat": {},
            "bpb": {},
            "core": {},
        }
        language = metrics.get("language", {})
        row["language"] = {
            "zh_response_rate": nested_get(language, "summary", "zh_response_rate"),
            "en_response_rate": nested_get(language, "summary", "en_response_rate"),
            "replacement_character_responses": nested_get(
                language, "summary", "replacement_character_responses"
            ),
            "zh_distinct_2": nested_get(
                language, "generation_quality", "zh", "mean_distinct_2"
            ),
            "zh_prompt_copy_responses": nested_get(
                language, "generation_quality", "zh", "prompt_copy_responses"
            ),
        }
        chat = metrics.get("chat", {})
        row["chat"] = {
            task: nested_get(chat, "tasks", task, "accuracy")
            for task in CHAT_TASKS
        }
        row["chat"].update({
            "chatcore": nested_get(chat, "aggregates", "chatcore"),
            "chatcore_without_spellingbee": nested_get(
                chat, "aggregates", "chatcore_without_spellingbee"
            ),
            "partial_chatcore": nested_get(chat, "aggregates", "partial_chatcore"),
        })
        bpb = metrics.get("bpb", {})
        row["bpb"] = {
            name: {
                "train": result.get("train_bpb"),
                "val": result.get("val_bpb"),
                "status": result.get("status"),
            }
            for name, result in bpb.get("datasets", {}).items()
        }
        row["core"] = {
            "core_metric": nested_get(metrics.get("core", {}), "core_metric"),
        }
        rows.append(row)
    return rows


def format_metric(value, percentage=False, digits=4):
    if value is None:
        return "-"
    if percentage:
        return f"{value:.2%}"
    return f"{value:.{digits}f}"


def render_markdown(rows):
    lines = [
        "# Bilingual Evaluation Comparison",
        "",
        f"Generated: {datetime.datetime.now().astimezone().isoformat()}",
        "",
        "## Runs",
        "",
        "| Model | Run ID | State |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['label']} | `{row['run_id']}` | {row['state']} |")

    if any(any(value is not None for value in row["language"].values()) for row in rows):
        lines.extend([
            "",
            "## Language And Generation",
            "",
            "| Model | Chinese rate | English rate | Replacement | ZH distinct-2 | ZH prompt copy |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            language = row["language"]
            lines.append(
                f"| {row['label']} "
                f"| {format_metric(language['zh_response_rate'], percentage=True)} "
                f"| {format_metric(language['en_response_rate'], percentage=True)} "
                f"| {language['replacement_character_responses'] if language['replacement_character_responses'] is not None else '-'} "
                f"| {format_metric(language['zh_distinct_2'])} "
                f"| {language['zh_prompt_copy_responses'] if language['zh_prompt_copy_responses'] is not None else '-'} |"
            )

    if any(any(value is not None for value in row["chat"].values()) for row in rows):
        lines.extend([
            "",
            "## Chat Evaluation",
            "",
            "| Model | ARC-E | ARC-C | MMLU | GSM8K | HumanEval | SpellingBee | ChatCORE | No SpellingBee |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            chat = row["chat"]
            lines.append(
                f"| {row['label']} "
                f"| {format_metric(chat['ARC-Easy'], percentage=True)} "
                f"| {format_metric(chat['ARC-Challenge'], percentage=True)} "
                f"| {format_metric(chat['MMLU'], percentage=True)} "
                f"| {format_metric(chat['GSM8K'], percentage=True)} "
                f"| {format_metric(chat['HumanEval'], percentage=True)} "
                f"| {format_metric(chat['SpellingBee'], percentage=True)} "
                f"| {format_metric(chat['chatcore'])} "
                f"| {format_metric(chat['chatcore_without_spellingbee'])} |"
            )

    bpb_names = sorted({name for row in rows for name in row["bpb"]})
    if bpb_names:
        lines.extend(["", "## Validation BPB", ""])
        lines.append("| Model | " + " | ".join(bpb_names) + " |")
        lines.append("|---|" + "|".join("---:" for _ in bpb_names) + "|")
        for row in rows:
            values = [
                format_metric(nested_get(row, "bpb", name, "val"), digits=6)
                for name in bpb_names
            ]
            lines.append(f"| {row['label']} | " + " | ".join(values) + " |")

    if any(row["core"]["core_metric"] is not None for row in rows):
        lines.extend(["", "## Base CORE", "", "| Model | CORE |", "|---|---:|"])
        for row in rows:
            lines.append(
                f"| {row['label']} | {format_metric(row['core']['core_metric'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare bilingual evaluation runs")
    parser.add_argument("runs", nargs="+", help="Evaluation result directories")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("Provide at least two evaluation run directories")

    runs = [load_run(path) for path in args.runs]
    rows = collect_rows(runs)
    if args.output_dir is None:
        stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(
            os.path.commonpath([run["path"] for run in runs]),
            f"comparison-{stamp}",
        )
    else:
        output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if os.path.exists(output_dir):
        raise FileExistsError(f"Comparison output already exists: {output_dir}")
    os.makedirs(output_dir)
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "runs": rows,
    }
    atomic_write_json(os.path.join(output_dir, "comparison.json"), payload)
    atomic_write_text(os.path.join(output_dir, "comparison.md"), render_markdown(rows))
    print(f"Comparison written to {output_dir}")


if __name__ == "__main__":
    main()
