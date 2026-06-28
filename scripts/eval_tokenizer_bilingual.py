"""Compare the default and a tagged tokenizer on fixed bilingual text."""

import argparse
import json
import os

from nanochat.common import get_base_dir
from nanochat.experiment import atomic_write_json, sha256_file
from nanochat.tokenizer import (
    RustBPETokenizer,
    get_tokenizer,
    resolve_tokenizer_dir,
)


SAMPLES = {
    "english_general": (
        "Machine learning enables computers to learn useful patterns from data "
        "without being explicitly programmed for every case."
    ),
    "english_science": (
        "Photosynthesis converts light energy into chemical energy while releasing oxygen."
    ),
    "chinese_general": "机器学习让计算机从数据中学习规律，并利用这些规律完成新的任务。",
    "chinese_knowledge": "北京是中国的首都，也是历史文化资源非常丰富的城市。",
    "mixed": "NanoChat 可以同时学习 English and 中文，并回答 bilingual questions.",
    "numbers_code": "for i in range(128): total += values[i] * 0.25",
}


def text_metrics(tokenizer, text):
    token_ids = tokenizer.encode(text)
    encoded_bytes = len(text.encode("utf-8"))
    cjk_chars = sum(0x3400 <= ord(char) <= 0x9FFF for char in text)
    return {
        "characters": len(text),
        "bytes": encoded_bytes,
        "tokens": len(token_ids),
        "bytes_per_token": encoded_bytes / len(token_ids),
        "tokens_per_cjk_character": (
            len(token_ids) / cjk_chars if cjk_chars else None
        ),
        "roundtrip": tokenizer.decode(token_ids) == text,
        "token_ids": token_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate-tag")
    candidate.add_argument("--candidate-dir")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_dir = get_base_dir()
    candidate_dir = (
        os.path.abspath(os.path.expanduser(args.candidate_dir))
        if args.candidate_dir
        else resolve_tokenizer_dir(tokenizer_tag=args.candidate_tag)
    )
    tokenizers = {
        "default": get_tokenizer(),
        "candidate": RustBPETokenizer.from_directory(candidate_dir),
    }
    payload = {
        "candidate_tag": args.candidate_tag,
        "candidate_dir": candidate_dir,
        "tokenizers": {},
        "samples": {},
    }
    for name, tokenizer in tokenizers.items():
        tokenizer_dir = (
            candidate_dir if name == "candidate" else resolve_tokenizer_dir()
        )
        payload["tokenizers"][name] = {
            "directory": tokenizer_dir,
            "vocab_size": tokenizer.get_vocab_size(),
            "sha256": sha256_file(os.path.join(tokenizer_dir, "tokenizer.pkl")),
        }
    for sample_name, text in SAMPLES.items():
        payload["samples"][sample_name] = {
            name: text_metrics(tokenizer, text)
            for name, tokenizer in tokenizers.items()
        }

    output = args.output or os.path.join(
        base_dir,
        "tokenizer_evaluations",
        f"{args.candidate_tag or os.path.basename(candidate_dir)}.json",
    )
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Evaluation written to {output}")


if __name__ == "__main__":
    main()
