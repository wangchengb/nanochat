"""
Build a cleaner Chinese SFT v2 dataset from an existing NanoChat JSONL SFT set.

The output format is the same as tasks.customjson.CustomJSON expects:
one JSON array of alternating user/assistant messages per line.
"""

import argparse
import difflib
import hashlib
import json
import os
import random
import re
from collections import Counter

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer


CJK_RE = re.compile(r"[\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text):
    return SPACE_RE.sub("", text.strip())


def cjk_language_ratio(text):
    cjk_count = len(CJK_RE.findall(text))
    latin_count = len(LATIN_RE.findall(text))
    return cjk_count / (cjk_count + latin_count) if cjk_count + latin_count else 0.0


def max_repeated_char_run(text):
    best = 0
    current = 0
    previous = None
    for char in text:
        if char.isspace():
            current = 0
            previous = None
            continue
        if char == previous:
            current += 1
        else:
            previous = char
            current = 1
        best = max(best, current)
    return best


def max_repeated_ngram_count(text, ngram_chars):
    compact = normalize_text(text)
    if len(compact) < ngram_chars:
        return 1
    counts = Counter(compact[i:i + ngram_chars] for i in range(0, len(compact) - ngram_chars + 1))
    return max(counts.values()) if counts else 1


def prompt_copy_ratio(user_text, assistant_text):
    user = normalize_text(user_text)
    assistant = normalize_text(assistant_text)
    if not user or not assistant:
        return 0.0
    match = difflib.SequenceMatcher(a=user, b=assistant, autojunk=False).find_longest_match(
        0, len(user), 0, len(assistant)
    )
    return match.size / len(user)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number}: expected a message list with at least two messages")
            rows.append(messages)
    return rows


def validate_messages(messages):
    if len(messages) % 2 != 0:
        return False
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            return False
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            return False
    return True


def analyze_conversation(messages, tokenizer, args):
    user_text = "\n".join(message["content"] for message in messages if message["role"] == "user")
    assistant_text = "\n".join(message["content"] for message in messages if message["role"] == "assistant")
    assistant_chars = len(assistant_text)
    rendered_ids, _ = tokenizer.render_conversation(
        {"messages": messages},
        max_tokens=args.render_check_max_tokens,
    )
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    expected_assistant_turns = sum(1 for message in messages if message["role"] == "assistant")
    assistant_end_count = sum(1 for token_id in rendered_ids if token_id == assistant_end)

    return {
        "assistant_chars": assistant_chars,
        "cjk_ratio": cjk_language_ratio(assistant_text),
        "max_char_run": max_repeated_char_run(assistant_text),
        "max_ngram_repeat": max_repeated_ngram_count(assistant_text, args.repeat_ngram_chars),
        "prompt_copy_ratio": prompt_copy_ratio(user_text, assistant_text),
        "rendered_tokens": len(rendered_ids),
        "assistant_end_count": assistant_end_count,
        "expected_assistant_turns": expected_assistant_turns,
        "assistant_text": assistant_text,
    }


def rejection_reason(messages, stats, seen_keys, args):
    if not validate_messages(messages):
        return "invalid_messages"
    if stats["assistant_chars"] < args.min_assistant_chars:
        return "assistant_too_short"
    if stats["assistant_chars"] > args.max_assistant_chars:
        return "assistant_too_long"
    if stats["cjk_ratio"] < args.min_cjk_ratio:
        return "low_cjk_ratio"
    if stats["rendered_tokens"] > args.max_rendered_tokens:
        return "too_many_tokens"
    if stats["assistant_end_count"] != stats["expected_assistant_turns"]:
        return "assistant_end_missing"
    if stats["max_char_run"] > args.max_repeated_char_run:
        return "char_repeat"
    if stats["max_ngram_repeat"] > args.max_ngram_repeat:
        return "ngram_repeat"
    if stats["prompt_copy_ratio"] > args.max_prompt_copy_ratio:
        return "prompt_copy"
    if args.drop_ai_refusals:
        assistant_text = stats["assistant_text"]
        if "人工智能助手" in assistant_text and ("无法" in assistant_text or "不能" in assistant_text):
            return "ai_refusal_template"
    key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    if key in seen_keys:
        return "duplicate"
    return None


def summarize(rows, tokenizer, args):
    if not rows:
        return {
            "rows": 0,
            "assistant_chars": 0,
            "rendered_tokens": 0,
            "rows_at_token_limit": 0,
        }
    assistant_chars = 0
    rendered_tokens = 0
    rows_at_token_limit = 0
    cjk_ratios = []
    for messages in rows:
        stats = analyze_conversation(messages, tokenizer, args)
        assistant_chars += stats["assistant_chars"]
        rendered_tokens += stats["rendered_tokens"]
        cjk_ratios.append(stats["cjk_ratio"])
        if stats["rendered_tokens"] >= args.max_rendered_tokens:
            rows_at_token_limit += 1
    return {
        "rows": len(rows),
        "assistant_chars": assistant_chars,
        "rendered_tokens": rendered_tokens,
        "mean_rendered_tokens": rendered_tokens / len(rows),
        "mean_assistant_cjk_ratio": sum(cjk_ratios) / len(cjk_ratios),
        "rows_at_token_limit": rows_at_token_limit,
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for messages in rows:
            handle.write(json.dumps(messages, ensure_ascii=False) + "\n")


def main():
    base_dir = get_base_dir()
    parser = argparse.ArgumentParser(description="Prepare a filtered Chinese SFT v2 dataset.")
    parser.add_argument("--source-train-jsonl", default=os.path.join(base_dir, "zh_experiment", "sft", "train.jsonl"))
    parser.add_argument("--source-val-jsonl", default=os.path.join(base_dir, "zh_experiment", "sft", "val.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(base_dir, "zh_experiment", "sft_v2"))
    parser.add_argument("--tokenizer-tag", default="bilingual-32k-han1-2b-v1")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-rows", type=int, default=1000)
    parser.add_argument("--max-rendered-tokens", type=int, default=513)
    parser.add_argument("--render-check-max-tokens", type=int, default=4096)
    parser.add_argument("--min-assistant-chars", type=int, default=20)
    parser.add_argument("--max-assistant-chars", type=int, default=700)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.75)
    parser.add_argument("--max-repeated-char-run", type=int, default=8)
    parser.add_argument("--repeat-ngram-chars", type=int, default=8)
    parser.add_argument("--max-ngram-repeat", type=int, default=8)
    parser.add_argument("--max-prompt-copy-ratio", type=float, default=0.85)
    parser.add_argument("--drop-ai-refusals", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.render_check_max_tokens < args.max_rendered_tokens:
        parser.error("--render-check-max-tokens must be >= --max-rendered-tokens")
    if args.val_rows < 1:
        parser.error("--val-rows must be positive")

    tokenizer = get_tokenizer(tokenizer_tag=args.tokenizer_tag)
    source_rows = load_jsonl(os.path.expanduser(args.source_train_jsonl))
    source_rows.extend(load_jsonl(os.path.expanduser(args.source_val_jsonl)))

    accepted = []
    rejection_counts = Counter()
    seen_keys = set()
    for messages in source_rows:
        stats = analyze_conversation(messages, tokenizer, args)
        reason = rejection_reason(messages, stats, seen_keys, args)
        if reason is not None:
            rejection_counts[reason] += 1
            continue
        seen_keys.add(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        accepted.append(messages)

    if len(accepted) <= args.val_rows:
        raise RuntimeError(
            f"Only {len(accepted)} rows passed filtering; need more than --val-rows={args.val_rows}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    val_rows = accepted[:args.val_rows]
    train_rows = accepted[args.val_rows:]

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "kind": "zh_sft_v2",
        "source_train_jsonl": os.path.abspath(os.path.expanduser(args.source_train_jsonl)),
        "source_val_jsonl": os.path.abspath(os.path.expanduser(args.source_val_jsonl)),
        "tokenizer_tag": args.tokenizer_tag,
        "seed": args.seed,
        "filters": {
            "max_rendered_tokens": args.max_rendered_tokens,
            "render_check_max_tokens": args.render_check_max_tokens,
            "min_assistant_chars": args.min_assistant_chars,
            "max_assistant_chars": args.max_assistant_chars,
            "min_cjk_ratio": args.min_cjk_ratio,
            "max_repeated_char_run": args.max_repeated_char_run,
            "repeat_ngram_chars": args.repeat_ngram_chars,
            "max_ngram_repeat": args.max_ngram_repeat,
            "max_prompt_copy_ratio": args.max_prompt_copy_ratio,
            "drop_ai_refusals": args.drop_ai_refusals,
        },
        "source_rows": len(source_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": sum(rejection_counts.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "train": summarize(train_rows, tokenizer, args),
        "validation": summarize(val_rows, tokenizer, args),
        "train_path": train_path,
        "val_path": val_path,
        "train_sha256": file_sha256(train_path),
        "val_sha256": file_sha256(val_path),
        "recommended_custom_train_token_ratio": 0.20,
        "notes": [
            "This dataset is derived from the existing Chinese SFT v1 JSONL files.",
            "Rows that would exceed the 512-token SFT context are removed so assistant_end remains visible.",
            "Use with existing English SFT tasks as replay; do not train this Chinese JSONL alone.",
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"source_rows={len(source_rows):,}")
    print(f"accepted={len(accepted):,} rejected={sum(rejection_counts.values()):,}")
    print(f"train={len(train_rows):,} val={len(val_rows):,}")
    print(f"rejection_counts={dict(sorted(rejection_counts.items()))}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
