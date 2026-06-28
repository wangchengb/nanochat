"""Build a bounded, language-balanced corpus for bilingual tokenizer training."""

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files
from nanochat.experiment import atomic_write_json, sha256_file


def language_counts(text):
    cjk = sum(0x3400 <= ord(char) <= 0x9FFF for char in text)
    latin = sum("a" <= char.lower() <= "z" for char in text)
    return cjk, latin


def read_documents(data_dir, split):
    paths = list_parquet_files(data_dir=data_dir)
    if len(paths) < 2:
        raise ValueError(
            f"Source must contain training shard(s) and one final validation shard: {data_dir}"
        )
    selected_paths = paths[:-1] if split == "train" else paths[-1:]
    for path in selected_paths:
        parquet_file = pq.ParquetFile(path)
        for row_group in range(parquet_file.num_row_groups):
            yield from parquet_file.read_row_group(
                row_group,
                columns=["text"],
            ).column("text").to_pylist()


def iter_language_documents(data_dir, split, language, doc_cap):
    for text in read_documents(data_dir, split):
        if not isinstance(text, str):
            continue
        text = text.strip()[:doc_cap]
        if not text:
            continue
        cjk, latin = language_counts(text)
        effective_chars = cjk if language == "zh" else latin
        other_chars = latin if language == "zh" else cjk
        if effective_chars == 0 or effective_chars < other_chars:
            continue
        yield text, effective_chars


def write_parquet(path, documents):
    pq.write_table(
        pa.table({"text": documents}),
        path,
        row_group_size=512,
        compression="zstd",
    )


def write_language_shards(
    *,
    data_dir,
    output_dir,
    language,
    target_chars,
    chars_per_shard,
    doc_cap,
    first_index,
):
    documents = []
    shard_effective_chars = 0
    total_effective_chars = 0
    total_raw_chars = 0
    total_documents = 0
    shards = []
    shard_index = first_index

    for text, effective_chars in iter_language_documents(
        data_dir,
        "train",
        language,
        doc_cap,
    ):
        remaining = target_chars - total_effective_chars
        if remaining <= 0:
            break
        documents.append(text)
        shard_effective_chars += effective_chars
        total_effective_chars += effective_chars
        total_raw_chars += len(text)
        total_documents += 1

        if shard_effective_chars >= chars_per_shard or total_effective_chars >= target_chars:
            path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
            write_parquet(path, documents)
            shards.append({
                "path": path,
                "language": language,
                "documents": len(documents),
                "effective_characters": shard_effective_chars,
                "raw_characters": sum(len(document) for document in documents),
                "sha256": sha256_file(path),
            })
            print(
                f"{language}: shard={os.path.basename(path)} "
                f"effective_chars={total_effective_chars:,}/{target_chars:,}",
                flush=True,
            )
            documents = []
            shard_effective_chars = 0
            shard_index += 2

    if total_effective_chars < target_chars:
        raise RuntimeError(
            f"{language} corpus ended at {total_effective_chars:,} effective characters, "
            f"below target {target_chars:,}"
        )
    return {
        "source_dir": os.path.abspath(os.path.expanduser(data_dir)),
        "documents": total_documents,
        "effective_characters": total_effective_chars,
        "raw_characters": total_raw_chars,
        "shards": shards,
    }


def collect_validation(data_dir, language, target_chars, doc_cap):
    documents = []
    effective_characters = 0
    raw_characters = 0
    for text, count in iter_language_documents(
        data_dir,
        "val",
        language,
        doc_cap,
    ):
        documents.append(text)
        effective_characters += count
        raw_characters += len(text)
        if effective_characters >= target_chars:
            break
    if effective_characters < target_chars:
        raise RuntimeError(
            f"{language} validation ended at {effective_characters:,} effective characters, "
            f"below target {target_chars:,}"
        )
    return documents, {
        "documents": len(documents),
        "effective_characters": effective_characters,
        "raw_characters": raw_characters,
    }


def main():
    base_dir = get_base_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        help="Legacy mixed-language source used for both Chinese and English",
    )
    parser.add_argument(
        "--zh-source-dir",
        default=os.path.join(base_dir, "zh_corpus", "cci3_hq_1b_local"),
    )
    parser.add_argument(
        "--en-source-dir",
        default=os.path.join(base_dir, "base_data_climbmix"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            base_dir,
            "bilingual_tokenizer_data",
            "cci3-climbmix-en1b-zh1b-v1",
        ),
    )
    parser.add_argument("--chars-per-language", type=int, default=1_000_000_000)
    parser.add_argument("--chars-per-shard", type=int, default=50_000_000)
    parser.add_argument("--validation-chars-per-language", type=int, default=5_000_000)
    parser.add_argument("--doc-cap", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.chars_per_language <= 0:
        parser.error("--chars-per-language must be positive")
    if args.chars_per_shard <= 0:
        parser.error("--chars-per-shard must be positive")
    if args.validation_chars_per_language <= 0:
        parser.error("--validation-chars-per-language must be positive")

    if args.source_dir:
        zh_source_dir = en_source_dir = args.source_dir
    else:
        zh_source_dir = args.zh_source_dir
        en_source_dir = args.en_source_dir

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite")
    if os.path.exists(output_dir) and not args.overwrite:
        existing = [
            name for name in os.listdir(output_dir)
            if name.endswith(".parquet")
        ]
        if existing:
            raise FileExistsError(
                f"{output_dir} already contains Parquet files; pass --overwrite"
            )
    os.makedirs(output_dir, exist_ok=True)
    if args.overwrite:
        for name in os.listdir(output_dir):
            if name.endswith(".parquet") or name == "manifest.json":
                os.remove(os.path.join(output_dir, name))

    languages = {
        "zh": write_language_shards(
            data_dir=zh_source_dir,
            output_dir=output_dir,
            language="zh",
            target_chars=args.chars_per_language,
            chars_per_shard=args.chars_per_shard,
            doc_cap=args.doc_cap,
            first_index=0,
        ),
        "en": write_language_shards(
            data_dir=en_source_dir,
            output_dir=output_dir,
            language="en",
            target_chars=args.chars_per_language,
            chars_per_shard=args.chars_per_shard,
            doc_cap=args.doc_cap,
            first_index=1,
        ),
    }

    validation_documents = []
    validation = {}
    for language, data_dir in (("zh", zh_source_dir), ("en", en_source_dir)):
        documents, stats = collect_validation(
            data_dir,
            language,
            args.validation_chars_per_language,
            args.doc_cap,
        )
        validation_documents.extend(documents)
        validation[language] = stats
    validation_path = os.path.join(output_dir, "shard_99999.parquet")
    write_parquet(validation_path, validation_documents)

    train_shards = sorted(
        (
            shard
            for stats in languages.values()
            for shard in stats["shards"]
        ),
        key=lambda shard: shard["path"],
    )
    manifest = {
        "schema_version": 2,
        "target_effective_characters_per_language": args.chars_per_language,
        "target_effective_characters_per_shard": args.chars_per_shard,
        "validation_effective_characters_per_language": (
            args.validation_chars_per_language
        ),
        "doc_cap": args.doc_cap,
        "languages": languages,
        "train": {
            "documents": sum(stats["documents"] for stats in languages.values()),
            "effective_characters": sum(
                stats["effective_characters"] for stats in languages.values()
            ),
            "raw_characters": sum(
                stats["raw_characters"] for stats in languages.values()
            ),
            "shards": train_shards,
        },
        "validation": {
            "path": validation_path,
            "documents": len(validation_documents),
            "languages": validation,
            "sha256": sha256_file(validation_path),
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
