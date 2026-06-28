"""Build a small, isolated bilingual Math SFT dataset.

The training split combines the official GSM8K training rows with deterministic
English and Chinese arithmetic examples. A held-out slice of GSM8K train is
used for validation; the official GSM8K test split remains untouched for eval.
"""

import argparse
import hashlib
import json
import os
import random
import re

from datasets import load_dataset

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer


TOOL_RESULT_RE = re.compile(r"<<([^<>]+)>>")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_gsm8k_answer(answer):
    """Turn GSM8K calculator annotations into ordinary visible reasoning."""
    def replace_tool_result(match):
        expression = match.group(1)
        if "=" not in expression:
            return expression
        calculation, result = expression.rsplit("=", 1)
        return f"{calculation} = {result}"

    return TOOL_RESULT_RE.sub(replace_tool_result, answer).strip()


def conversation(question, answer):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def synthetic_example(index, language, rng):
    operation = index % 5
    if operation == 0:
        left = rng.randint(8, 180)
        right = rng.randint(5, 120)
        result = left + right
        if language == "zh":
            question = f"小明有{left}张卡片，又得到{right}张。他现在共有多少张卡片？"
            answer = f"把原有卡片和新得到的卡片相加：{left} + {right} = {result}。\n#### {result}"
        else:
            question = f"Mia has {left} cards and receives {right} more. How many cards does she have now?"
            answer = f"Add the original and new cards: {left} + {right} = {result}.\n#### {result}"
    elif operation == 1:
        result = rng.randint(5, 120)
        removed = rng.randint(3, 100)
        start = result + removed
        if language == "zh":
            question = f"仓库原有{start}个零件，用掉{removed}个后还剩多少个？"
            answer = f"用原有数量减去用掉的数量：{start} - {removed} = {result}。\n#### {result}"
        else:
            question = f"A store had {start} parts and used {removed}. How many parts remain?"
            answer = f"Subtract the used parts: {start} - {removed} = {result}.\n#### {result}"
    elif operation == 2:
        groups = rng.randint(2, 18)
        each = rng.randint(2, 24)
        result = groups * each
        if language == "zh":
            question = f"有{groups}个盒子，每个盒子装{each}支铅笔，一共有多少支铅笔？"
            answer = f"盒子数乘以每盒数量：{groups} × {each} = {result}。\n#### {result}"
        else:
            question = f"There are {groups} boxes with {each} pencils in each box. How many pencils are there?"
            answer = f"Multiply boxes by pencils per box: {groups} * {each} = {result}.\n#### {result}"
    elif operation == 3:
        result = rng.randint(2, 30)
        groups = rng.randint(2, 16)
        total = result * groups
        if language == "zh":
            question = f"把{total}块糖平均分给{groups}个人，每人能得到多少块？"
            answer = f"总数除以人数：{total} ÷ {groups} = {result}。\n#### {result}"
        else:
            question = f"If {total} candies are shared equally by {groups} people, how many does each person get?"
            answer = f"Divide the total by the number of people: {total} / {groups} = {result}.\n#### {result}"
    else:
        groups = rng.randint(2, 12)
        each = rng.randint(3, 20)
        extra = rng.randint(2, 40)
        result = groups * each + extra
        if language == "zh":
            question = f"学校买了{groups}包练习本，每包{each}本，后来又买了{extra}本。总共有多少本？"
            answer = f"先算成包购买的数量：{groups} × {each} = {groups * each}。再加上后来购买的{extra}本：{groups * each} + {extra} = {result}。\n#### {result}"
        else:
            question = f"A school buys {groups} packs of {each} notebooks and then buys {extra} more. How many notebooks are there in total?"
            answer = f"The packs contain {groups} * {each} = {groups * each} notebooks. Add the extra {extra}: {groups * each} + {extra} = {result}.\n#### {result}"
    return conversation(question, answer)


def rendered_length(tokenizer, messages):
    ids, _ = tokenizer.render_conversation({"messages": messages}, max_tokens=8192)
    return len(ids)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    base_dir = get_base_dir()
    parser = argparse.ArgumentParser(description="Prepare bilingual Math SFT data.")
    parser.add_argument("--output-dir", default=os.path.join(base_dir, "math_experiment", "sft_v1"))
    parser.add_argument("--tokenizer-tag", default="bilingual-32k-han1-2b-v1")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--synthetic-per-language", type=int, default=3000)
    parser.add_argument("--max-rendered-tokens", type=int, default=513)
    args = parser.parse_args()

    tokenizer = get_tokenizer(tokenizer_tag=args.tokenizer_tag)
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(gsm8k)))
    split_rng = random.Random(args.seed)
    split_rng.shuffle(indices)
    validation_indices = set(indices[:args.validation_rows])

    gsm_train = []
    gsm_validation = []
    filtered_long = 0
    for index, row in enumerate(gsm8k):
        messages = conversation(row["question"], flatten_gsm8k_answer(row["answer"]))
        if rendered_length(tokenizer, messages) > args.max_rendered_tokens:
            filtered_long += 1
            continue
        target = gsm_validation if index in validation_indices else gsm_train
        target.append(messages)

    synthetic_rows = []
    for language, offset in (("en", 0), ("zh", 1)):
        rng = random.Random(args.seed + offset)
        for index in range(args.synthetic_per_language):
            synthetic_rows.append(synthetic_example(index, language, rng))

    train_rows = gsm_train + synthetic_rows
    split_rng.shuffle(train_rows)
    train_tokens = sum(rendered_length(tokenizer, row) for row in train_rows)
    validation_tokens = sum(rendered_length(tokenizer, row) for row in gsm_validation)

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    validation_path = os.path.join(output_dir, "val.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, gsm_validation)

    manifest = {
        "kind": "bilingual_math_sft_v1",
        "seed": args.seed,
        "tokenizer_tag": args.tokenizer_tag,
        "source": {
            "dataset": "openai/gsm8k",
            "config": "main",
            "split": "train",
            "fingerprint": getattr(gsm8k, "_fingerprint", None),
            "rows": len(gsm8k),
        },
        "training": {
            "rows": len(train_rows),
            "gsm8k_rows": len(gsm_train),
            "synthetic_english_rows": args.synthetic_per_language,
            "synthetic_chinese_rows": args.synthetic_per_language,
            "rendered_tokens": train_tokens,
            "mean_rendered_tokens": train_tokens / len(train_rows),
        },
        "validation": {
            "rows": len(gsm_validation),
            "source": "held-out GSM8K training rows",
            "rendered_tokens": validation_tokens,
            "mean_rendered_tokens": validation_tokens / len(gsm_validation),
        },
        "filtered_over_token_limit": filtered_long,
        "max_rendered_tokens": args.max_rendered_tokens,
        "train_path": train_path,
        "val_path": validation_path,
        "train_sha256": file_sha256(train_path),
        "val_sha256": file_sha256(validation_path),
        "notes": [
            "The official GSM8K test split is not read or used during preparation.",
            "All answers end in the evaluator-compatible format: #### number.",
            "Synthetic rows teach elementary arithmetic in both English and Chinese.",
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"train_rows={len(train_rows):,} val_rows={len(gsm_validation):,}")
    print(f"gsm8k_train={len(gsm_train):,} synthetic={len(synthetic_rows):,} filtered_long={filtered_long:,}")
    print(f"train_tokens={train_tokens:,} val_tokens={validation_tokens:,}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
