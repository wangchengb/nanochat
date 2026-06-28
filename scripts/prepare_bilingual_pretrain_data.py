"""Build token-balanced bilingual pretraining shards with bounded documents."""

import argparse
import hashlib
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files
from nanochat.experiment import atomic_write_json, sha256_file
from nanochat.tokenizer import get_tokenizer, resolve_tokenizer_dir


def language_counts(text):
    cjk = sum(0x3400 <= ord(char) <= 0x9FFF for char in text)
    latin = sum("a" <= char.lower() <= "z" for char in text)
    return cjk, latin


def iter_source_documents(data_dir, split, language):
    paths = list_parquet_files(data_dir=data_dir)
    if len(paths) < 2:
        raise ValueError(
            f"Source must contain training shard(s) and one final validation shard: {data_dir}"
        )
    selected_paths = paths[:-1] if split == "train" else paths[-1:]
    for path in selected_paths:
        parquet_file = pq.ParquetFile(path)
        for row_group in range(parquet_file.num_row_groups):
            texts = parquet_file.read_row_group(
                row_group,
                columns=["text"],
            ).column("text").to_pylist()
            for text in texts:
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue
                cjk, latin = language_counts(text)
                if language == "zh" and cjk < latin:
                    continue
                if language == "en" and latin < cjk:
                    continue
                yield text


def decode_valid_utf8(tokenizer, token_ids, start, end):
    while end > start:
        try:
            text = tokenizer.enc.decode_bytes(token_ids[start:end]).decode("utf-8")
        except UnicodeDecodeError:
            end -= 1
            continue
        encoded = tokenizer.encode(text)
        if len(encoded) <= end - start:
            return text, encoded, end
        end -= max(1, len(encoded) - (end - start))
    raise RuntimeError("Unable to find a valid UTF-8 tokenizer chunk boundary")


def chunk_document(
    tokenizer,
    text,
    max_chunk_tokens,
    min_chunk_tokens,
    document_hash=None,
):
    token_ids = tokenizer.encode(text)
    document_hash = document_hash or hashlib.sha256(text.encode("utf-8")).hexdigest()
    start = 0
    chunk_index = 0
    while start < len(token_ids):
        end = min(start + max_chunk_tokens, len(token_ids))
        chunk_text, chunk_ids, end = decode_valid_utf8(
            tokenizer,
            token_ids,
            start,
            end,
        )
        if len(chunk_ids) >= min_chunk_tokens:
            yield {
                "text": chunk_text,
                "content_tokens": len(chunk_ids),
                "training_tokens": len(chunk_ids) + 1,
                "document_hash": document_hash,
                "chunk_index": chunk_index,
            }
        start = end
        chunk_index += 1


def iter_language_chunks(
    tokenizer,
    data_dir,
    split,
    language,
    max_chunk_tokens,
    min_chunk_tokens,
    excluded_document_hashes=None,
    seen_document_hashes=None,
):
    excluded_document_hashes = excluded_document_hashes or set()
    seen_document_hashes = seen_document_hashes if seen_document_hashes is not None else set()
    for text in iter_source_documents(data_dir, split, language):
        document_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if document_hash in excluded_document_hashes or document_hash in seen_document_hashes:
            continue
        seen_document_hashes.add(document_hash)
        yield from chunk_document(
            tokenizer,
            text,
            max_chunk_tokens,
            min_chunk_tokens,
            document_hash=document_hash,
        )


def take_chunks(iterator, target_training_tokens):
    chunks = []
    training_tokens = 0
    content_tokens = 0
    while training_tokens < target_training_tokens:
        try:
            chunk = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Source ended at {training_tokens:,} training tokens, "
                f"below target {target_training_tokens:,}"
            ) from exc
        chunks.append(chunk)
        training_tokens += chunk["training_tokens"]
        content_tokens += chunk["content_tokens"]
    return chunks, {
        "documents": len({chunk["document_hash"] for chunk in chunks}),
        "chunks": len(chunks),
        "content_tokens": content_tokens,
        "training_tokens": training_tokens,
    }


def write_parquet(path, chunks):
    pq.write_table(
        pa.table({
            "text": [chunk["text"] for chunk in chunks],
            "language": [chunk["language"] for chunk in chunks],
            "document_hash": [chunk["document_hash"] for chunk in chunks],
            "chunk_index": [chunk["chunk_index"] for chunk in chunks],
            "content_tokens": [chunk["content_tokens"] for chunk in chunks],
            "training_tokens": [chunk["training_tokens"] for chunk in chunks],
        }),
        path,
        row_group_size=512,
        compression="zstd",
    )


def add_language(chunks, language):
    for chunk in chunks:
        chunk["language"] = language


def write_eval_directory(path, chunks):
    os.makedirs(path, exist_ok=True)
    midpoint = max(1, len(chunks) // 2)
    train_path = os.path.join(path, "shard_00000.parquet")
    val_path = os.path.join(path, "shard_99999.parquet")
    write_parquet(train_path, chunks[:midpoint])
    write_parquet(val_path, chunks[midpoint:])
    return {
        "directory": path,
        "train": {
            "path": train_path,
            "chunks": midpoint,
            "sha256": sha256_file(train_path),
        },
        "validation": {
            "path": val_path,
            "chunks": len(chunks) - midpoint,
            "sha256": sha256_file(val_path),
        },
    }


def main():
    base_dir = get_base_dir()
    parser = argparse.ArgumentParser()
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
            "bilingual_pretrain_data",
            "han1-en60-zh40-200m-v1",
        ),
    )
    parser.add_argument("--tokenizer-tag", default="bilingual-32k-han1-2b-v1")
    parser.add_argument("--total-training-tokens", type=int, default=200_000_000)
    parser.add_argument("--english-token-ratio", type=float, default=0.60)
    parser.add_argument("--tokens-per-shard", type=int, default=10_000_000)
    parser.add_argument("--validation-tokens-per-language", type=int, default=2_000_000)
    parser.add_argument("--max-chunk-tokens", type=int, default=512)
    parser.add_argument("--min-chunk-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.english_token_ratio < 1:
        parser.error("--english-token-ratio must be between 0 and 1")
    if args.total_training_tokens <= 0 or args.tokens_per_shard <= 0:
        parser.error("Training token targets must be positive")
    if not 0 < args.min_chunk_tokens <= args.max_chunk_tokens:
        parser.error("--min-chunk-tokens must be in [1, --max-chunk-tokens]")

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(output_dir):
        existing = [
            name for name in os.listdir(output_dir)
            if name.endswith(".parquet") or name == "manifest.json"
        ]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already contains output files; pass --overwrite"
            )
        if args.overwrite:
            for name in existing:
                os.remove(os.path.join(output_dir, name))
    os.makedirs(output_dir, exist_ok=True)

    tokenizer_dir = resolve_tokenizer_dir(tokenizer_tag=args.tokenizer_tag)
    tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
    rng = random.Random(args.seed)

    sources = {
        "en": os.path.abspath(os.path.expanduser(args.en_source_dir)),
        "zh": os.path.abspath(os.path.expanduser(args.zh_source_dir)),
    }
    train_document_hashes = set()
    train_iterators = {
        language: iter_language_chunks(
            tokenizer,
            data_dir,
            "train",
            language,
            args.max_chunk_tokens,
            args.min_chunk_tokens,
            seen_document_hashes=train_document_hashes,
        )
        for language, data_dir in sources.items()
    }

    num_shards = (
        args.total_training_tokens + args.tokens_per_shard - 1
    ) // args.tokens_per_shard
    train_shards = []
    selected_train_document_hashes = set()
    totals = {
        language: {
            "chunks": 0,
            "content_tokens": 0,
            "training_tokens": 0,
        }
        for language in ("en", "zh")
    }
    for shard_index in range(num_shards):
        remaining_total = (
            args.total_training_tokens
            - sum(stats["training_tokens"] for stats in totals.values())
        )
        shard_target = min(args.tokens_per_shard, remaining_total)
        en_target = round(shard_target * args.english_token_ratio)
        targets = {"en": en_target, "zh": shard_target - en_target}

        chunks = []
        shard_languages = {}
        for language in ("en", "zh"):
            language_chunks, stats = take_chunks(
                train_iterators[language],
                targets[language],
            )
            add_language(language_chunks, language)
            chunks.extend(language_chunks)
            selected_train_document_hashes.update(
                chunk["document_hash"] for chunk in language_chunks
            )
            shard_languages[language] = stats
            for key in ("chunks", "content_tokens", "training_tokens"):
                totals[language][key] += stats[key]
        rng.shuffle(chunks)

        path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        write_parquet(path, chunks)
        train_shards.append({
            "path": path,
            "chunks": len(chunks),
            "training_tokens": sum(
                stats["training_tokens"] for stats in shard_languages.values()
            ),
            "languages": shard_languages,
            "sha256": sha256_file(path),
        })
        print(
            f"shard={shard_index:05d} "
            f"tokens={train_shards[-1]['training_tokens']:,} "
            f"en={shard_languages['en']['training_tokens']:,} "
            f"zh={shard_languages['zh']['training_tokens']:,}",
            flush=True,
        )

    validation_chunks = []
    validation = {}
    validation_document_hashes = set()
    selected_validation_document_hashes = set()
    for language in ("en", "zh"):
        iterator = iter_language_chunks(
            tokenizer,
            sources[language],
            "val",
            language,
            args.max_chunk_tokens,
            args.min_chunk_tokens,
            excluded_document_hashes=train_document_hashes,
            seen_document_hashes=validation_document_hashes,
        )
        chunks, stats = take_chunks(
            iterator,
            args.validation_tokens_per_language,
        )
        add_language(chunks, language)
        validation_chunks.extend(chunks)
        selected_validation_document_hashes.update(
            chunk["document_hash"] for chunk in chunks
        )
        validation[language] = {
            **stats,
            "eval": write_eval_directory(
                os.path.join(output_dir, f"eval_{language}"),
                chunks,
            ),
        }
    rng.shuffle(validation_chunks)
    validation_path = os.path.join(output_dir, "shard_99999.parquet")
    write_parquet(validation_path, validation_chunks)

    total_training_tokens = sum(
        stats["training_tokens"] for stats in totals.values()
    )
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "tokenizer": {
            "tag": args.tokenizer_tag,
            "directory": tokenizer_dir,
            "sha256": sha256_file(os.path.join(tokenizer_dir, "tokenizer.pkl")),
            "vocab_size": tokenizer.get_vocab_size(),
        },
        "sources": sources,
        "chunking": {
            "max_content_tokens": args.max_chunk_tokens,
            "min_content_tokens": args.min_chunk_tokens,
            "bos_tokens_per_chunk": 1,
        },
        "requested": {
            "total_training_tokens": args.total_training_tokens,
            "english_token_ratio": args.english_token_ratio,
            "tokens_per_shard": args.tokens_per_shard,
            "validation_tokens_per_language": args.validation_tokens_per_language,
        },
        "train": {
            "training_tokens": total_training_tokens,
            "unique_documents": len(selected_train_document_hashes),
            "languages": totals,
            "english_token_ratio": (
                totals["en"]["training_tokens"] / total_training_tokens
            ),
            "shards": train_shards,
        },
        "validation": {
            "path": validation_path,
            "unique_documents": len(selected_validation_document_hashes),
            "languages": validation,
            "sha256": sha256_file(validation_path),
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest["train"], ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
