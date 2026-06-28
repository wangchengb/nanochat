"""
Build SFT v4 data focused on concise answers and stopping behavior.

v4 starts from the filtered SFT v2 data, then adds a stronger set of short
Chinese/English examples. This is an SFT behavior experiment, not a knowledge
pretraining substitute.
"""

import argparse
import hashlib
import json
import os
import random

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from scripts.prepare_sft_v3_data import EN_TOPICS, ZH_TOPICS, strip_source


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def conversation(user, assistant):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def first_sentence(text, zh=False):
    delimiter = "。" if zh else "."
    return text.split(delimiter)[0].strip() + delimiter


def concise_zh_rows():
    rows = []
    for topic, answer in ZH_TOPICS.items():
        one = first_sentence(answer, zh=True)
        rows.extend([
            conversation(f"用一句中文介绍{topic}。", one),
            conversation(f"请简短回答：{topic}是什么？", one),
            conversation(f"不要展开，用中文说明{topic}。", one),
            conversation(f"用不超过40个中文字符回答：{topic}。", one),
            conversation(
                f"请列出关于{topic}的三个短要点。",
                f"1. {one}\n2. 与实际生活或学习有关。\n3. 解释时应具体，不要重复。",
            ),
        ])
    rows.extend([
        conversation("请用一句中文回答，不要重复。", "我会直接回答，并在表达清楚后停止。"),
        conversation("如果问题很简单，应该怎样回答？", "应简短、准确地回答，不需要反复展开。"),
        conversation("如果回答已经完整，下一步应该做什么？", "应该停止生成，而不是重复同一句话。"),
        conversation("请说明如何避免循环输出。", "先给结论，再给必要说明，最后及时停止。"),
        conversation("请用中文回答：信息不足时怎么办？", "说明缺少的信息，并提出一个具体的澄清问题。"),
    ])
    return rows


def concise_en_rows():
    rows = []
    for topic, answer in EN_TOPICS.items():
        one = first_sentence(answer, zh=False)
        rows.extend([
            conversation(f"Introduce {topic} in one English sentence.", one),
            conversation(f"Briefly answer: what is {topic}?", one),
            conversation(f"Do not expand. Explain {topic} in English.", one),
            conversation(f"Answer in under 25 words: {topic}.", one),
            conversation(
                f"List three short points about {topic}.",
                f"1. {one}\n2. It has practical or real-world importance.\n3. A good answer should be specific and avoid repetition.",
            ),
        ])
    rows.extend([
        conversation("Answer in one English sentence without repeating yourself.", "I will answer directly and stop when the point is clear."),
        conversation("How should you answer a simple question?", "A simple question should get a short, accurate answer without unnecessary expansion."),
        conversation("What should you do once an answer is complete?", "I should stop generating instead of repeating the same idea."),
        conversation("Explain how to avoid looping output.", "Give the conclusion, add only necessary context, and stop promptly."),
        conversation("What should you do if the question lacks details?", "I should state what is missing and ask one focused clarification question."),
    ])
    return rows


def synthetic_rows(repeats):
    base_rows = concise_zh_rows() + concise_en_rows()
    rows = []
    for _ in range(repeats):
        rows.extend(base_rows)
    return rows


def rendered_token_count(rows, tokenizer, max_tokens):
    total = 0
    rows_at_limit = 0
    for messages in rows:
        ids, _ = tokenizer.render_conversation({"messages": messages}, max_tokens=max_tokens)
        total += len(ids)
        if len(ids) >= max_tokens:
            rows_at_limit += 1
    return total, rows_at_limit


def main():
    base_dir = get_base_dir()
    parser = argparse.ArgumentParser(description="Prepare SFT v4 concise/stopping data.")
    parser.add_argument("--source-train-jsonl", default=os.path.join(base_dir, "zh_experiment", "sft_v2", "train.jsonl"))
    parser.add_argument("--source-val-jsonl", default=os.path.join(base_dir, "zh_experiment", "sft_v2", "val.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(base_dir, "zh_experiment", "sft_v4"))
    parser.add_argument("--tokenizer-tag", default="bilingual-32k-han1-2b-v1")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--synthetic-repeats", type=int, default=80)
    parser.add_argument("--synthetic-val-repeats", type=int, default=2)
    parser.add_argument("--max-rendered-tokens", type=int, default=513)
    args = parser.parse_args()

    tokenizer = get_tokenizer(tokenizer_tag=args.tokenizer_tag)
    source_train = [strip_source(row) for row in load_jsonl(os.path.expanduser(args.source_train_jsonl))]
    source_val = [strip_source(row) for row in load_jsonl(os.path.expanduser(args.source_val_jsonl))]
    train_synthetic = synthetic_rows(args.synthetic_repeats)
    val_synthetic = synthetic_rows(args.synthetic_val_repeats)

    train_rows = source_train + train_synthetic
    val_rows = source_val + val_synthetic
    random.Random(args.seed).shuffle(train_rows)

    train_tokens, train_limit_rows = rendered_token_count(train_rows, tokenizer, args.max_rendered_tokens)
    val_tokens, val_limit_rows = rendered_token_count(val_rows, tokenizer, args.max_rendered_tokens)

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "kind": "zh_sft_v4_concise_stop",
        "tokenizer_tag": args.tokenizer_tag,
        "seed": args.seed,
        "source_train_jsonl": os.path.abspath(os.path.expanduser(args.source_train_jsonl)),
        "source_val_jsonl": os.path.abspath(os.path.expanduser(args.source_val_jsonl)),
        "synthetic_base_rows": len(synthetic_rows(1)),
        "synthetic_repeats": args.synthetic_repeats,
        "synthetic_val_repeats": args.synthetic_val_repeats,
        "synthetic_train_rows": len(train_synthetic),
        "synthetic_val_rows": len(val_synthetic),
        "train": {
            "rows": len(train_rows),
            "rendered_tokens": train_tokens,
            "mean_rendered_tokens": train_tokens / len(train_rows),
            "rows_at_token_limit": train_limit_rows,
        },
        "validation": {
            "rows": len(val_rows),
            "rendered_tokens": val_tokens,
            "mean_rendered_tokens": val_tokens / len(val_rows),
            "rows_at_token_limit": val_limit_rows,
        },
        "recommended_custom_train_token_ratio": 0.22,
        "train_path": train_path,
        "val_path": val_path,
        "train_sha256": file_sha256(train_path),
        "val_sha256": file_sha256(val_path),
        "notes": [
            "SFT v4 is a behavior experiment for concise answers and stopping.",
            "It intentionally increases short bilingual examples relative to v3.",
            "Keep language eval max_new_tokens at 128 to expose remaining loops.",
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"train_rows={len(train_rows):,} val_rows={len(val_rows):,}")
    print(f"synthetic_train_rows={len(train_synthetic):,} synthetic_val_rows={len(val_synthetic):,}")
    print(f"train_tokens={train_tokens:,} val_tokens={val_tokens:,}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
