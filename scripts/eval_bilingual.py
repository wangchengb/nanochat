"""
Unified, resumable bilingual evaluation for NanoChat checkpoints.

Each run writes to an isolated directory and persists every completed group:

  manifest.json
  status.json
  metrics/language.json
  metrics/chat.json
  metrics/bpb.json
  metrics/core.json
  summary.md

Existing base_eval.py, chat_eval.py, and zh_eval.py entrypoints remain unchanged.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import traceback

import torch

from nanochat.checkpoint_manager import find_last_step, load_model, tokenizer_dir_from_meta
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.engine import Engine
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes
from scripts.base_eval import evaluate_core
from scripts.chat_eval import run_chat_eval
from scripts.zh_eval import evaluate_language_responses


CHAT_TASKS = [
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
    "SpellingBee",
]
CHAT_BASELINES = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "MMLU": 0.25,
    "GSM8K": 0.0,
    "HumanEval": 0.0,
    "SpellingBee": 0.0,
}
VALID_GROUPS = {"language", "chat", "bpb", "core"}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def local_run_stamp():
    return datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def atomic_write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary_path, path)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_slug(text):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return slug.strip("-") or "model"


def command_output(command):
    try:
        return subprocess.check_output(
            command,
            cwd=os.getcwd(),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_metadata():
    status = command_output(["git", "status", "--porcelain"])
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"]) or "unknown",
        "branch": command_output(["git", "branch", "--show-current"]) or "unknown",
        "dirty": bool(status),
    }


def checkpoint_paths(base_dir, source, model_tag, step):
    source_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    checkpoint_dir = os.path.join(base_dir, source_dir, model_tag)
    loaded_step = step if step is not None else find_last_step(checkpoint_dir)
    return {
        "directory": checkpoint_dir,
        "model": os.path.join(checkpoint_dir, f"model_{loaded_step:06d}.pt"),
        "meta": os.path.join(checkpoint_dir, f"meta_{loaded_step:06d}.json"),
        "step": loaded_step,
    }


def tokenizer_metadata(base_dir, tokenizer, checkpoint_meta):
    tokenizer_dir = tokenizer_dir_from_meta(checkpoint_meta)
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "path": tokenizer_path,
    }
    if os.path.exists(tokenizer_path):
        metadata["sha256"] = file_sha256(tokenizer_path)
    return metadata


def data_manifest_metadata(base_dir):
    path = os.path.join(base_dir, "zh_experiment", "manifest.json")
    if not os.path.exists(path):
        return None
    return {
        "path": path,
        "sha256": file_sha256(path),
    }


def parse_named_paths(values):
    datasets = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH for --bpb-dataset, got: {value}")
        name, path = value.split("=", 1)
        name = safe_slug(name)
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"BPB dataset directory not found: {path}")
        datasets[name] = path
    return datasets


def default_bpb_datasets(base_dir):
    candidates = {
        "english": os.path.join(base_dir, "base_data_climbmix"),
        "chinese": os.path.join(base_dir, "zh_experiment", "pretrain_zh_eval"),
        "mixed": os.path.join(base_dir, "zh_experiment", "pretrain"),
    }
    return {name: path for name, path in candidates.items() if os.path.isdir(path)}


def ngram_diversity(text, n):
    if len(text) < n:
        return 1.0 if text else 0.0
    ngrams = [text[index:index + n] for index in range(len(text) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def normalize_prompt_for_copy(prompt):
    return re.sub(r"[\s。！？!?.,，；;：:]+", "", prompt)


def response_words(text):
    return re.findall(r"[A-Za-z']+|[\u3400-\u9fff]", text)


def longest_repeated_item_run(items):
    best = 0
    current = 0
    previous = None
    for item in items:
        if item == previous:
            current += 1
        else:
            previous = item
            current = 1
        best = max(best, current)
    return best


def max_repeated_ngram_count(text, n):
    compact = normalize_prompt_for_copy(text)
    if len(compact) < n:
        return 1 if compact else 0
    counts = {}
    for index in range(len(compact) - n + 1):
        ngram = compact[index:index + n]
        counts[ngram] = counts.get(ngram, 0) + 1
    return max(counts.values()) if counts else 0


def has_long_loop(text):
    words = response_words(text)
    return (
        longest_repeated_item_run(words) >= 20
        or max_repeated_ngram_count(text, 8) >= 20
    )


def generation_quality(records):
    if not records:
        return {
            "responses": 0,
            "mean_characters": 0.0,
            "mean_distinct_2": 0.0,
            "mean_distinct_3": 0.0,
            "prompt_copy_responses": 0,
            "empty_responses": 0,
            "long_loop_responses": 0,
            "max_repeated_word_run": 0,
            "max_repeated_8gram_count": 0,
        }
    prompt_copies = 0
    max_repeated_word_run = 0
    max_repeated_8gram_count = 0
    long_loop_responses = 0
    for record in records:
        prompt = normalize_prompt_for_copy(record["prompt"])
        response = normalize_prompt_for_copy(record["response"])
        prompt_copies += int(bool(prompt) and prompt in response)
        max_repeated_word_run = max(
            max_repeated_word_run,
            longest_repeated_item_run(response_words(record["response"])),
        )
        max_repeated_8gram_count = max(
            max_repeated_8gram_count,
            max_repeated_ngram_count(record["response"], 8),
        )
        long_loop_responses += int(has_long_loop(record["response"]))
    return {
        "responses": len(records),
        "mean_characters": sum(len(record["response"]) for record in records) / len(records),
        "mean_distinct_2": sum(ngram_diversity(record["response"], 2) for record in records) / len(records),
        "mean_distinct_3": sum(ngram_diversity(record["response"], 3) for record in records) / len(records),
        "prompt_copy_responses": prompt_copies,
        "empty_responses": sum(not record["response"].strip() for record in records),
        "long_loop_responses": long_loop_responses,
        "max_repeated_word_run": max_repeated_word_run,
        "max_repeated_8gram_count": max_repeated_8gram_count,
    }


def enrich_language_report(report):
    records = report["records"]
    report["generation_quality"] = {
        language: generation_quality([
            record for record in records if record["language"] == language
        ])
        for language in ("zh", "en")
    }
    return report


def centered_chat_scores(results):
    centered = {
        task: (accuracy - CHAT_BASELINES[task]) / (1.0 - CHAT_BASELINES[task])
        for task, accuracy in results.items()
    }
    payload = {
        "centered": centered,
        "partial_chatcore": sum(centered.values()) / len(centered) if centered else None,
    }
    without_spelling = {
        task: score for task, score in centered.items() if task != "SpellingBee"
    }
    payload["partial_chatcore_without_spellingbee"] = (
        sum(without_spelling.values()) / len(without_spelling)
        if without_spelling else None
    )
    if all(task in results for task in CHAT_TASKS):
        payload["chatcore"] = sum(centered[task] for task in CHAT_TASKS) / len(CHAT_TASKS)
        payload["chatcore_without_spellingbee"] = sum(
            centered[task] for task in CHAT_TASKS if task != "SpellingBee"
        ) / (len(CHAT_TASKS) - 1)
    return payload


def exception_payload(error):
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "at": utc_now(),
    }


def render_summary(manifest, status, metrics):
    model = manifest["model"]
    lines = [
        f"# Bilingual Evaluation: {model['source']}/{model['tag']}/{model['step']}",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Created: `{manifest['created_at']}`",
        f"- Git: `{manifest['git']['commit']}`"
        + (" (dirty)" if manifest["git"]["dirty"] else ""),
        f"- Device: `{manifest['evaluation']['device_type']}`",
        f"- Status: `{status['state']}`",
        "",
    ]
    language = metrics.get("language")
    if language:
        summary = language["summary"]
        lines.extend([
            "## Language Control",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Chinese response rate | {summary['zh_response_rate']:.2%} |",
            f"| English response rate | {summary['en_response_rate']:.2%} |",
            f"| Replacement-character responses | {summary['replacement_character_responses']} |",
            "",
        ])
        for language_name, label in [("zh", "Chinese"), ("en", "English")]:
            quality = language["generation_quality"][language_name]
            lines.extend([
                f"### {label} Generation",
                "",
                f"- Mean characters: {quality['mean_characters']:.2f}",
                f"- Distinct-2: {quality['mean_distinct_2']:.4f}",
                f"- Distinct-3: {quality['mean_distinct_3']:.4f}",
                f"- Prompt-copy responses: {quality['prompt_copy_responses']}",
                f"- Empty responses: {quality['empty_responses']}",
                "",
            ])
    chat = metrics.get("chat")
    if chat:
        lines.extend([
            "## Chat Evaluation",
            "",
            "| Task | Accuracy | Status |",
            "|---|---:|---|",
        ])
        for task in CHAT_TASKS:
            task_result = chat["tasks"].get(task)
            if task_result and task_result.get("status") == "completed":
                lines.append(f"| {task} | {task_result['accuracy']:.2%} | completed |")
            elif task_result:
                lines.append(f"| {task} | - | failed |")
            else:
                lines.append(f"| {task} | - | pending |")
        aggregates = chat.get("aggregates", {})
        lines.extend([
            "",
            f"- ChatCORE: {aggregates.get('chatcore')}",
            f"- ChatCORE without SpellingBee: {aggregates.get('chatcore_without_spellingbee')}",
            f"- Partial ChatCORE: {aggregates.get('partial_chatcore')}",
            "",
        ])
    bpb = metrics.get("bpb")
    if bpb:
        lines.extend([
            "## BPB",
            "",
            "| Dataset | Train | Validation | Status |",
            "|---|---:|---:|---|",
        ])
        for name, result in bpb["datasets"].items():
            if result.get("status") == "completed":
                lines.append(
                    f"| {name} | {result['train_bpb']:.6f} | "
                    f"{result['val_bpb']:.6f} | completed |"
                )
            else:
                lines.append(f"| {name} | - | - | failed |")
        lines.append("")
    core = metrics.get("core")
    if core:
        lines.extend([
            "## Base CORE",
            "",
            f"- CORE metric: {core.get('core_metric')}",
            "",
        ])
    errors = status.get("errors", {})
    if errors:
        lines.extend(["## Errors", ""])
        for name, error in errors.items():
            lines.append(f"- `{name}`: {error['type']}: {error['message']}")
        lines.append("")
    return "\n".join(lines)


class EvaluationRun:
    def __init__(self, output_dir, manifest, resume):
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.status_path = os.path.join(output_dir, "status.json")
        self.summary_path = os.path.join(output_dir, "summary.md")
        if os.path.exists(output_dir) and not resume:
            raise FileExistsError(
                f"Evaluation output already exists: {output_dir}. "
                "Use --resume or choose another --run-id."
            )
        os.makedirs(self.metrics_dir, exist_ok=True)
        existing_manifest = read_json(self.manifest_path)
        if existing_manifest is not None:
            for key in ("source", "tag", "step"):
                if existing_manifest["model"][key] != manifest["model"][key]:
                    raise ValueError(f"Resume model mismatch for {key}")
            self.manifest = existing_manifest
        else:
            self.manifest = manifest
            atomic_write_json(self.manifest_path, manifest)
        self.status = read_json(self.status_path, {
            "state": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "groups": {},
            "errors": {},
        })
        self.metrics = {}
        for group in VALID_GROUPS:
            payload = read_json(self.metric_path(group))
            if payload is not None:
                self.metrics[group] = payload
        self.save_status()

    def metric_path(self, group):
        return os.path.join(self.metrics_dir, f"{group}.json")

    def save_metric(self, group, payload):
        self.metrics[group] = payload
        atomic_write_json(self.metric_path(group), payload)
        self.status["groups"][group] = {
            "state": "completed",
            "updated_at": utc_now(),
        }
        for error_name in list(self.status["errors"]):
            if error_name == group or error_name.startswith(f"{group}/"):
                self.status["errors"].pop(error_name)
        self.save_status()

    def save_partial_metric(self, group, payload):
        self.metrics[group] = payload
        atomic_write_json(self.metric_path(group), payload)
        self.status["groups"][group] = {
            "state": "running",
            "updated_at": utc_now(),
        }
        self.save_status()

    def record_error(self, name, error):
        payload = exception_payload(error)
        self.status["errors"][name] = payload
        group = name.split("/", 1)[0]
        self.status["groups"][group] = {
            "state": "failed",
            "updated_at": utc_now(),
        }
        self.save_status()

    def save_status(self):
        self.status["updated_at"] = utc_now()
        atomic_write_json(self.status_path, self.status)
        atomic_write_text(
            self.summary_path,
            render_summary(self.manifest, self.status, self.metrics),
        )

    def finish(self):
        failed = bool(self.status["errors"])
        self.status["state"] = "completed_with_errors" if failed else "completed"
        self.save_status()


def evaluate_bpb_group(
    model,
    tokenizer,
    device,
    datasets,
    split_tokens,
    batch_size,
    initial_payload=None,
    progress_callback=None,
    tokenizer_dir=None,
):
    token_bytes = get_token_bytes(device=device, tokenizer_dir=tokenizer_dir)
    sequence_len = model.config.sequence_len
    tokens_per_step = batch_size * sequence_len
    adjusted_tokens = (split_tokens // tokens_per_step) * tokens_per_step
    if adjusted_tokens <= 0:
        raise ValueError(
            f"--bpb-split-tokens must be at least {tokens_per_step} for this model and batch size"
        )
    steps = adjusted_tokens // tokens_per_step
    results = initial_payload or {
        "requested_split_tokens": split_tokens,
        "evaluated_split_tokens": adjusted_tokens,
        "device_batch_size": batch_size,
        "datasets": {},
    }
    for name, data_dir in datasets.items():
        if results["datasets"].get(name, {}).get("status") == "completed":
            continue
        dataset_result = {"data_dir": data_dir, "status": "running"}
        results["datasets"][name] = dataset_result
        try:
            split_results = {}
            for split_name in ("train", "val"):
                loader = tokenizing_distributed_data_loader_bos_bestfit(
                    tokenizer,
                    batch_size,
                    sequence_len,
                    split_name,
                    device=device,
                    data_dir=data_dir,
                )
                split_results[split_name] = evaluate_bpb(model, loader, steps, token_bytes)
            dataset_result.update({
                "status": "completed",
                "train_bpb": split_results["train"],
                "val_bpb": split_results["val"],
            })
        except Exception as error:
            dataset_result.update({
                "status": "failed",
                "error": exception_payload(error),
            })
        if progress_callback is not None:
            progress_callback(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified bilingual checkpoint evaluation")
    parser.add_argument("-i", "--source", choices=["base", "sft", "rl"], required=True)
    parser.add_argument("-g", "--model-tag", required=True)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated groups: language,chat,bpb,core. "
        "Default: language,chat for sft/rl; bpb,core for base.",
    )
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default="")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--language-max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat-tasks", default="|".join(CHAT_TASKS))
    parser.add_argument("--chat-max-problems", type=int, default=128)
    parser.add_argument("--chat-max-new-tokens", type=int, default=512)
    parser.add_argument("--chat-num-samples", type=int, default=1)
    parser.add_argument("--chat-batch-size", type=int, default=8)
    parser.add_argument(
        "--bpb-dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable BPB dataset. Defaults to available english/chinese/mixed directories.",
    )
    parser.add_argument("--bpb-split-tokens", type=int, default=131072)
    parser.add_argument("--bpb-device-batch-size", type=int, default=1)
    parser.add_argument("--core-max-per-task", type=int, default=128)
    parser.add_argument(
        "--core-tasks",
        default="",
        help="Optional | separated CORE task labels. Default: all tasks.",
    )
    args = parser.parse_args()

    groups = (
        {"bpb", "core"} if args.source == "base" else {"language", "chat"}
    ) if args.groups is None else {
        group.strip() for group in args.groups.split(",") if group.strip()
    }
    invalid_groups = groups - VALID_GROUPS
    if invalid_groups:
        parser.error(f"Invalid groups: {sorted(invalid_groups)}")
    if not groups:
        parser.error("--groups must select at least one evaluation group")
    if "core" in groups and args.source != "base":
        parser.error("The core group currently requires --source=base")
    requested_tasks = [task for task in args.chat_tasks.split("|") if task]
    requested_core_tasks = [task for task in args.core_tasks.split("|") if task]
    unknown_tasks = set(requested_tasks) - set(CHAT_TASKS)
    if unknown_tasks:
        parser.error(f"Unknown chat tasks: {sorted(unknown_tasks)}")

    base_dir = get_base_dir()
    paths = checkpoint_paths(base_dir, args.source, args.model_tag, args.step)
    device_type = args.device_type or autodetect_device_type()
    _, _, _, world_size, device = compute_init(device_type)
    if world_size != 1:
        parser.error("eval_bilingual currently supports a single process only")

    model, tokenizer, checkpoint_meta = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=paths["step"],
    )
    run_id = args.run_id or (
        f"{args.source}-{safe_slug(args.model_tag)}-step{paths['step']:06d}-{local_run_stamp()}"
    )
    output_root = os.path.abspath(os.path.expanduser(
        args.output_root or os.path.join(base_dir, "evaluations")
    ))
    output_dir = os.path.join(output_root, safe_slug(run_id))
    command = " ".join(shlex.quote(argument) for argument in sys.argv)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "command": command,
        "git": git_metadata(),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "model": {
            "source": args.source,
            "tag": args.model_tag,
            "step": paths["step"],
            "checkpoint_dir": paths["directory"],
            "model_path": paths["model"],
            "model_sha256": file_sha256(paths["model"]),
            "meta_path": paths["meta"],
            "meta_sha256": file_sha256(paths["meta"]),
            "checkpoint_meta": checkpoint_meta,
        },
        "tokenizer": tokenizer_metadata(base_dir, tokenizer, checkpoint_meta),
        "data_manifest": data_manifest_metadata(base_dir),
        "evaluation": {
            "groups": sorted(groups),
            "device_type": device_type,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "language_max_new_tokens": args.language_max_new_tokens,
            "chat_tasks": requested_tasks,
            "chat_max_problems": args.chat_max_problems,
            "chat_max_new_tokens": args.chat_max_new_tokens,
            "chat_num_samples": args.chat_num_samples,
            "chat_batch_size": args.chat_batch_size,
            "bpb_split_tokens": args.bpb_split_tokens,
            "bpb_device_batch_size": args.bpb_device_batch_size,
            "core_max_per_task": args.core_max_per_task,
            "core_tasks": requested_core_tasks,
        },
    }
    run = EvaluationRun(output_dir, manifest, args.resume)
    print0(f"Evaluation output: {output_dir}")

    try:
        if "language" in groups and run.status["groups"].get("language", {}).get("state") != "completed":
            try:
                report = evaluate_language_responses(
                    model,
                    tokenizer,
                    max_new_tokens=args.language_max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=args.seed,
                    progress_callback=lambda index, total, record: print0(
                        f"[language {index:03d}/{total}] {record['language']} "
                        f"cjk={record['metrics']['cjk_ratio']:.2f}"
                    ),
                )
                run.save_metric("language", enrich_language_report(report))
            except Exception as error:
                run.record_error("language", error)

        if "chat" in groups and run.status["groups"].get("chat", {}).get("state") != "completed":
            engine = Engine(model, tokenizer)
            chat_payload = run.metrics.get("chat", {
                "settings": {
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "num_samples": args.chat_num_samples,
                    "max_new_tokens": args.chat_max_new_tokens,
                    "max_problems": args.chat_max_problems,
                    "batch_size": args.chat_batch_size,
                },
                "tasks": {},
                "aggregates": {},
            })
            for task in requested_tasks:
                if chat_payload["tasks"].get(task, {}).get("status") == "completed":
                    continue
                try:
                    accuracy = run_chat_eval(
                        task,
                        model,
                        tokenizer,
                        engine,
                        batch_size=args.chat_batch_size,
                        num_samples=args.chat_num_samples,
                        max_new_tokens=args.chat_max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        max_problems=args.chat_max_problems,
                    )
                    chat_payload["tasks"][task] = {
                        "status": "completed",
                        "accuracy": accuracy,
                        "updated_at": utc_now(),
                    }
                    run.status["errors"].pop(f"chat/{task}", None)
                except Exception as error:
                    chat_payload["tasks"][task] = {
                        "status": "failed",
                        "error": exception_payload(error),
                        "updated_at": utc_now(),
                    }
                    run.status["errors"][f"chat/{task}"] = chat_payload["tasks"][task]["error"]
                completed_results = {
                    name: result["accuracy"]
                    for name, result in chat_payload["tasks"].items()
                    if result.get("status") == "completed"
                }
                chat_payload["aggregates"] = centered_chat_scores(completed_results)
                run.save_partial_metric("chat", chat_payload)
            failed_tasks = [
                task for task in requested_tasks
                if chat_payload["tasks"].get(task, {}).get("status") != "completed"
            ]
            if failed_tasks:
                run.status["groups"]["chat"] = {
                    "state": "completed_with_errors",
                    "updated_at": utc_now(),
                }
                run.save_status()
            else:
                run.save_metric("chat", chat_payload)

        if "bpb" in groups and run.status["groups"].get("bpb", {}).get("state") != "completed":
            try:
                datasets = (
                    parse_named_paths(args.bpb_dataset)
                    if args.bpb_dataset else default_bpb_datasets(base_dir)
                )
                if not datasets:
                    raise FileNotFoundError("No BPB datasets were found")
                bpb_payload = evaluate_bpb_group(
                    model,
                    tokenizer,
                    device,
                    datasets,
                    args.bpb_split_tokens,
                    args.bpb_device_batch_size,
                    initial_payload=run.metrics.get("bpb"),
                    progress_callback=lambda payload: run.save_partial_metric("bpb", payload),
                    tokenizer_dir=tokenizer_dir_from_meta(checkpoint_meta),
                )
                failed_datasets = [
                    name for name, result in bpb_payload["datasets"].items()
                    if result.get("status") != "completed"
                ]
                if failed_datasets:
                    for name in failed_datasets:
                        run.status["errors"][f"bpb/{name}"] = (
                            bpb_payload["datasets"][name]["error"]
                        )
                    run.status["groups"]["bpb"] = {
                        "state": "completed_with_errors",
                        "updated_at": utc_now(),
                    }
                    run.save_status()
                else:
                    run.save_metric("bpb", bpb_payload)
            except Exception as error:
                run.record_error("bpb", error)

        if "core" in groups and run.status["groups"].get("core", {}).get("state") != "completed":
            try:
                core_payload = run.metrics.get("core", {
                    "results": {},
                    "centered_results": {},
                    "core_metric": None,
                })
                run.save_metric(
                    "core",
                    evaluate_core(
                        model,
                        tokenizer,
                        device,
                        max_per_task=args.core_max_per_task,
                        initial_payload=core_payload,
                        progress_callback=lambda payload: run.save_partial_metric(
                            "core", payload
                        ),
                        task_labels=requested_core_tasks,
                    ),
                )
            except Exception as error:
                run.record_error("core", error)
    finally:
        run.finish()
        compute_cleanup()

    print0(f"Summary written to {run.summary_path}")
    if run.status["state"] == "completed_with_errors":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
