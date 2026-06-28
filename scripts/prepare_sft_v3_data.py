"""
Build SFT v3 data by adding concise bilingual stability examples to SFT v2.

The purpose of v3 is narrow: keep the v2 Chinese filtering gains while adding
direct examples of short, complete answers that stop without looping.
"""

import argparse
import hashlib
import json
import os
import random

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer


ZH_TOPICS = {
    "北京": "北京是中国的首都，也是重要的政治、文化和交通中心。它有故宫、天坛和长城等著名历史文化景点。",
    "上海": "上海是中国东部的国际化城市，也是重要的金融、贸易和航运中心。它以外滩、陆家嘴和多元城市文化闻名。",
    "春天": "春天是一年中气温回升、植物生长加快的季节。人们常在春天踏青、赏花，并开始新的计划。",
    "机器学习": "机器学习是一种让计算机从数据中学习规律的方法。它常用于推荐系统、图像识别、语音识别和文本处理。",
    "数据库": "数据库用于有组织地存储、查询和管理数据。常见操作包括新增、读取、更新和删除数据。",
    "操作系统": "操作系统负责管理计算机硬件和软件资源。它为应用程序提供文件、进程、内存和网络等基础服务。",
    "历史": "历史研究过去发生的事件及其原因和影响。学习历史可以帮助人们理解社会变化和现实问题。",
    "健康饮食": "健康饮食强调食物多样、营养均衡和适量摄入。通常应多吃蔬菜水果，少吃高糖、高盐和高油食物。",
    "读书": "读书可以扩大知识面、训练思考能力，也能丰富语言表达。坚持阅读比偶尔大量阅读更容易形成长期收益。",
    "旅行": "旅行能帮助人们了解不同地方的自然环境和文化习俗。合理规划路线、预算和安全事项会让旅行更顺利。",
    "数学": "数学研究数量、结构、空间和变化。它既是基础学科，也广泛应用于工程、金融、计算机和科学研究。",
    "人工智能": "人工智能研究如何让机器完成需要智能的任务。它包括机器学习、自然语言处理、计算机视觉和规划等方向。",
}


EN_TOPICS = {
    "Beijing": "Beijing is the capital of China and a major center of politics, culture, and transportation. It is known for landmarks such as the Forbidden City, the Temple of Heaven, and nearby sections of the Great Wall.",
    "Paris": "Paris is the capital of France and a major center of art, fashion, and culture. It is known for the Eiffel Tower, the Louvre, and its historic neighborhoods.",
    "spring": "Spring is the season when temperatures rise and many plants begin to grow. People often associate it with flowers, outdoor activity, and new plans.",
    "machine learning": "Machine learning is a method that lets computers learn patterns from data. It is used in recommendation systems, speech recognition, image recognition, and text processing.",
    "databases": "A database stores and organizes data so it can be queried and updated efficiently. Common operations include creating, reading, updating, and deleting records.",
    "operating systems": "An operating system manages computer hardware and software resources. It provides basic services such as files, processes, memory, and networking.",
    "history": "History studies past events and their causes and effects. Learning history helps people understand social change and present-day problems.",
    "healthy eating": "Healthy eating means choosing varied foods in balanced amounts. A good diet usually includes vegetables, fruit, whole grains, protein, and limited sugar, salt, and oil.",
    "reading": "Reading expands knowledge, improves thinking, and strengthens language skills. A steady reading habit is usually more useful than occasional heavy reading.",
    "travel": "Travel helps people experience different places, environments, and cultures. Planning routes, budget, and safety details makes a trip smoother.",
    "mathematics": "Mathematics studies quantity, structure, space, and change. It is a foundation for science, engineering, finance, and computing.",
    "artificial intelligence": "Artificial intelligence studies how machines can perform tasks that require intelligence. It includes machine learning, language processing, computer vision, and planning.",
}


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


def conversation(user, assistant, source):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
    }


def strip_source(row):
    if isinstance(row, dict):
        return row["messages"]
    return row


def zh_examples():
    rows = []
    for topic, answer in ZH_TOPICS.items():
        rows.extend([
            conversation(f"请用中文简单介绍{topic}。", answer, "zh_concise"),
            conversation(f"请用两句话解释{topic}。", answer, "zh_concise"),
            conversation(f"围绕{topic}写一段简短说明。", answer, "zh_concise"),
            conversation(
                f"请列出关于{topic}的三个要点。",
                f"1. {answer.split('。')[0]}。\n2. 它和日常生活或学习有实际联系。\n3. 理解{topic}时应避免只重复概念，要结合具体场景。",
                "zh_list",
            ),
        ])
    rows.extend([
        conversation(
            "请用中文回答：如果信息不足，你应该怎么做？",
            "如果信息不足，我会说明缺少哪些关键信息，并提出一个简短的澄清问题。",
            "zh_stop",
        ),
        conversation(
            "请用中文给出一句话回答，不要展开。",
            "可以，我会用一句简短、完整的话回答。",
            "zh_stop",
        ),
        conversation(
            "请用中文说明回答问题时如何避免重复。",
            "回答时应先给出结论，再补充必要理由；如果已经表达清楚，就应停止，而不是重复同一句话。",
            "zh_stop",
        ),
    ])
    return rows


def en_examples():
    rows = []
    for topic, answer in EN_TOPICS.items():
        rows.extend([
            conversation(f"Briefly introduce {topic} in English.", answer, "en_concise"),
            conversation(f"Explain {topic} in two English sentences.", answer, "en_concise"),
            conversation(f"Write a short English description of {topic}.", answer, "en_concise"),
            conversation(
                f"List three key points about {topic} in English.",
                f"1. {answer.split('.')[0]}.\n2. It has practical uses or real-world importance.\n3. A good explanation should be specific and should stop once the answer is complete.",
                "en_list",
            ),
        ])
    rows.extend([
        conversation(
            "Answer in one concise English sentence: what should you do if details are missing?",
            "If details are missing, I should say what is unclear and ask one focused follow-up question.",
            "en_stop",
        ),
        conversation(
            "Answer briefly in English without repeating yourself.",
            "I will give a short, complete answer and stop when the point is clear.",
            "en_stop",
        ),
        conversation(
            "Explain how to avoid repetitive answers in English.",
            "State the answer directly, add only necessary context, and stop instead of repeating the same phrase.",
            "en_stop",
        ),
    ])
    return rows


def synthetic_rows(repeats):
    base_rows = zh_examples() + en_examples()
    rows = []
    for _ in range(repeats):
        rows.extend(base_rows)
    return rows


def rendered_token_count(rows, tokenizer, max_tokens):
    total = 0
    over_limit = 0
    for row in rows:
        messages = strip_source(row)
        ids, _ = tokenizer.render_conversation({"messages": messages}, max_tokens=max_tokens)
        total += len(ids)
        if len(ids) >= max_tokens:
            over_limit += 1
    return total, over_limit


def main():
    base_dir = get_base_dir()
    parser = argparse.ArgumentParser(description="Prepare SFT v3 data.")
    parser.add_argument("--source-train-jsonl", default=os.path.join(base_dir, "zh_experiment", "sft_v2", "train.jsonl"))
    parser.add_argument("--source-val-jsonl", default=os.path.join(base_dir, "zh_experiment", "sft_v2", "val.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(base_dir, "zh_experiment", "sft_v3"))
    parser.add_argument("--tokenizer-tag", default="bilingual-32k-han1-2b-v1")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--synthetic-repeats", type=int, default=40)
    parser.add_argument("--synthetic-val-repeats", type=int, default=2)
    parser.add_argument("--max-rendered-tokens", type=int, default=513)
    args = parser.parse_args()

    tokenizer = get_tokenizer(tokenizer_tag=args.tokenizer_tag)
    train_rows = load_jsonl(os.path.expanduser(args.source_train_jsonl))
    val_rows = load_jsonl(os.path.expanduser(args.source_val_jsonl))
    train_synth = synthetic_rows(args.synthetic_repeats)
    val_synth = synthetic_rows(args.synthetic_val_repeats)

    rng = random.Random(args.seed)
    merged_train = [strip_source(row) for row in train_rows] + [strip_source(row) for row in train_synth]
    merged_val = [strip_source(row) for row in val_rows] + [strip_source(row) for row in val_synth]
    rng.shuffle(merged_train)

    train_tokens, train_limit_rows = rendered_token_count(merged_train, tokenizer, args.max_rendered_tokens)
    val_tokens, val_limit_rows = rendered_token_count(merged_val, tokenizer, args.max_rendered_tokens)

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")
    write_jsonl(train_path, merged_train)
    write_jsonl(val_path, merged_val)

    manifest = {
        "kind": "zh_sft_v3_stability",
        "tokenizer_tag": args.tokenizer_tag,
        "seed": args.seed,
        "source_train_jsonl": os.path.abspath(os.path.expanduser(args.source_train_jsonl)),
        "source_val_jsonl": os.path.abspath(os.path.expanduser(args.source_val_jsonl)),
        "synthetic_repeats": args.synthetic_repeats,
        "synthetic_val_repeats": args.synthetic_val_repeats,
        "synthetic_base_rows": len(synthetic_rows(1)),
        "synthetic_train_rows": len(train_synth),
        "synthetic_val_rows": len(val_synth),
        "train": {
            "rows": len(merged_train),
            "rendered_tokens": train_tokens,
            "mean_rendered_tokens": train_tokens / len(merged_train),
            "rows_at_token_limit": train_limit_rows,
        },
        "validation": {
            "rows": len(merged_val),
            "rendered_tokens": val_tokens,
            "mean_rendered_tokens": val_tokens / len(merged_val),
            "rows_at_token_limit": val_limit_rows,
        },
        "recommended_custom_train_token_ratio": 0.20,
        "train_path": train_path,
        "val_path": val_path,
        "train_sha256": file_sha256(train_path),
        "val_sha256": file_sha256(val_path),
        "notes": [
            "SFT v3 keeps SFT v2 Chinese filtering and adds concise bilingual stability rows.",
            "Synthetic rows are intentionally short and complete to encourage stopping.",
            "This is not a factual-knowledge dataset; it targets repetition and answer termination.",
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"train_rows={len(merged_train):,} val_rows={len(merged_val):,}")
    print(f"synthetic_train_rows={len(train_synth):,} synthetic_val_rows={len(val_synth):,}")
    print(f"train_tokens={train_tokens:,} val_tokens={val_tokens:,}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
