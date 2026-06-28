"""Download a bounded Chinese corpus for bilingual tokenizer experiments."""

import argparse
import glob
import hashlib
import json
import os
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from nanochat.common import get_base_dir
from nanochat.experiment import atomic_write_json, sha256_file


def chinese_character_count(text):
    return sum(0x3400 <= ord(char) <= 0x9FFF for char in text)


def latin_character_count(text):
    return sum("a" <= char.lower() <= "z" for char in text)


def normalize_text(text):
    return "\n".join(line.strip() for line in text.replace("\r\n", "\n").splitlines()).strip()


def write_parquet(path, documents):
    pq.write_table(
        pa.table({"text": documents}),
        path,
        row_group_size=512,
        compression="zstd",
    )


def iter_hf_source_texts(repo_id, revision, text_column, file_prefix):
    api = HfApi()
    resolved_revision = api.dataset_info(
        repo_id,
        revision=revision,
        token=True,
    ).sha
    files = api.list_repo_files(
        repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        token=True,
    )
    source_files = sorted(
        path for path in files
        if path.endswith((".parquet", ".jsonl"))
        and (not file_prefix or path.startswith(file_prefix))
    )
    if not source_files:
        prefix_message = f" under prefix {file_prefix!r}" if file_prefix else ""
        raise RuntimeError(f"No Parquet or JSONL files found in {repo_id}{prefix_message}")

    print(
        f"Resolved {repo_id}@{resolved_revision}; "
        f"found {len(source_files):,} source files",
        flush=True,
    )
    fs = HfFileSystem(token=True)
    for index, path in enumerate(source_files, start=1):
        hf_path = f"datasets/{repo_id}@{resolved_revision}/{path}"
        print(f"Reading source file {index}/{len(source_files)}: {path}", flush=True)
        with fs.open(hf_path, "rb") as handle:
            if path.endswith(".parquet"):
                parquet_file = pq.ParquetFile(handle)
                available_columns = set(parquet_file.schema_arrow.names)
                selected_column = text_column
                if selected_column not in available_columns:
                    selected_column = next(
                        (
                            name for name in ("content", "raw_content")
                            if name in available_columns
                        ),
                        None,
                    )
                    if selected_column is None:
                        raise RuntimeError(
                            f"Text column {text_column!r} is not present in {path}; "
                            f"available columns: {sorted(available_columns)}"
                        )
                    print(f"Using text column {selected_column!r}", flush=True)
                for batch in parquet_file.iter_batches(
                    batch_size=256,
                    columns=[selected_column],
                ):
                    yield from batch.column(0).to_pylist()
                continue

            selected_column = None
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"Invalid JSONL data in {path} at line {line_number}"
                    ) from exc
                if selected_column is None:
                    selected_column = text_column
                    if selected_column not in row:
                        selected_column = next(
                            (name for name in ("content", "raw_content") if name in row),
                            None,
                        )
                        if selected_column is None:
                            raise RuntimeError(
                                f"Text column {text_column!r} is not present in {path}; "
                                f"available columns: {sorted(row)}"
                            )
                        print(f"Using text column {selected_column!r}", flush=True)
                yield row.get(selected_column)

    return resolved_revision


def iter_local_jsonl_texts(paths, text_column):
    for index, path in enumerate(paths, start=1):
        print(f"Reading local file {index}/{len(paths)}: {path}", flush=True)
        selected_column = None
        with open(path, "rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"Invalid JSONL data in {path} at line {line_number}"
                    ) from exc
                if selected_column is None:
                    selected_column = text_column
                    if selected_column not in row:
                        selected_column = next(
                            (name for name in ("content", "raw_content") if name in row),
                            None,
                        )
                        if selected_column is None:
                            raise RuntimeError(
                                f"Text column {text_column!r} is not present in {path}; "
                                f"available columns: {sorted(row)}"
                            )
                        print(f"Using text column {selected_column!r}", flush=True)
                yield row.get(selected_column)


def main():
    base_dir = get_base_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BAAI/CCI3-HQ")
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--file-prefix",
        help="Only read repository Parquet files whose paths start with this prefix",
    )
    parser.add_argument(
        "--local-jsonl",
        action="append",
        default=[],
        help="Read a local JSONL path or glob instead of streaming from Hugging Face",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(base_dir, "zh_corpus", "cci3_hq_1b"),
    )
    parser.add_argument("--target-cjk-chars", type=int, default=1_000_000_000)
    parser.add_argument("--cjk-chars-per-shard", type=int, default=50_000_000)
    parser.add_argument("--min-document-cjk-chars", type=int, default=200)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.60)
    parser.add_argument("--doc-cap", type=int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.target_cjk_chars <= 0:
        parser.error("--target-cjk-chars must be positive")
    if args.cjk_chars_per_shard <= 0:
        parser.error("--cjk-chars-per-shard must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite")
    os.makedirs(output_dir, exist_ok=True)

    if args.local_jsonl:
        local_paths = sorted({
            os.path.abspath(os.path.expanduser(path))
            for pattern in args.local_jsonl
            for path in glob.glob(os.path.expanduser(pattern))
        })
        if not local_paths:
            parser.error("--local-jsonl did not match any files")
        missing = [path for path in local_paths if not os.path.isfile(path)]
        if missing:
            parser.error(f"Local JSONL paths are not files: {missing}")
        resolved_revision = None
        source_texts = iter_local_jsonl_texts(local_paths, args.text_column)
    else:
        local_paths = []
        try:
            resolved_revision = HfApi().dataset_info(
                args.dataset,
                revision=args.revision,
                token=True,
            ).sha
            source_texts = iter_hf_source_texts(
                args.dataset,
                resolved_revision,
                args.text_column,
                args.file_prefix,
            )
        except Exception as exc:
            message = str(exc).lower()
            if not any(
                word in message
                for word in ("gated", "401", "403", "unauthorized", "forbidden")
            ):
                raise
            raise RuntimeError(
                f"{args.dataset} is gated and the active Hugging Face token cannot read it. "
                f"Open https://huggingface.co/datasets/{args.dataset} in the same account, "
                "submit/accept the access form, wait until the page confirms access, and "
                "ensure the token has read permission for public gated repositories."
            ) from exc

    train_documents = []
    validation_documents = []
    train_shards = []
    seen_hashes = set()
    cjk_characters = 0
    train_cjk_characters = 0
    validation_cjk_characters = 0
    current_shard_cjk_characters = 0
    accepted_documents = 0
    raw_rows = 0
    rejected_rows = 0
    duplicate_rows = 0
    for text in source_texts:
        raw_rows += 1
        if not isinstance(text, str):
            rejected_rows += 1
            continue
        text = normalize_text(text)[:args.doc_cap]
        cjk = chinese_character_count(text)
        latin = latin_character_count(text)
        if cjk < args.min_document_cjk_chars or cjk / max(cjk + latin, 1) < args.min_cjk_ratio:
            rejected_rows += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        if digest in seen_hashes:
            duplicate_rows += 1
            continue
        seen_hashes.add(digest)
        accepted_documents += 1
        cjk_characters += cjk

        # Hash partitioning keeps the split deterministic without retaining the
        # entire billion-character corpus in memory.
        split_value = int.from_bytes(digest[:8], "big") / 2**64
        if split_value < args.validation_fraction:
            validation_documents.append(text)
            validation_cjk_characters += cjk
        else:
            train_documents.append(text)
            train_cjk_characters += cjk
            current_shard_cjk_characters += cjk

        if current_shard_cjk_characters >= args.cjk_chars_per_shard:
            shard_path = os.path.join(
                output_dir,
                f"shard_{len(train_shards):05d}.parquet",
            )
            write_parquet(shard_path, train_documents)
            train_shards.append({
                "path": shard_path,
                "documents": len(train_documents),
                "cjk_characters": current_shard_cjk_characters,
                "sha256": sha256_file(shard_path),
            })
            train_documents = []
            current_shard_cjk_characters = 0

        if accepted_documents % 1000 == 0:
            print(
                f"accepted={accepted_documents:,} cjk_chars={cjk_characters:,} "
                f"raw_rows={raw_rows:,} shards={len(train_shards):,}",
                flush=True,
            )
        if cjk_characters >= args.target_cjk_chars:
            break

    if cjk_characters < args.target_cjk_chars:
        raise RuntimeError(
            f"Dataset ended at {cjk_characters:,} CJK characters, "
            f"below target {args.target_cjk_chars:,}"
        )

    if train_documents:
        shard_path = os.path.join(
            output_dir,
            f"shard_{len(train_shards):05d}.parquet",
        )
        write_parquet(shard_path, train_documents)
        train_shards.append({
            "path": shard_path,
            "documents": len(train_documents),
            "cjk_characters": current_shard_cjk_characters,
            "sha256": sha256_file(shard_path),
        })

    validation_path = os.path.join(output_dir, "shard_99999.parquet")
    write_parquet(validation_path, validation_documents)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "text_column": args.text_column,
            "file_prefix": args.file_prefix,
            "local_jsonl": local_paths,
        },
        "selection": {
            "target_cjk_characters": args.target_cjk_chars,
            "actual_cjk_characters": cjk_characters,
            "min_document_cjk_characters": args.min_document_cjk_chars,
            "min_cjk_ratio": args.min_cjk_ratio,
            "doc_cap": args.doc_cap,
            "cjk_characters_per_shard": args.cjk_chars_per_shard,
            "raw_rows": raw_rows,
            "accepted_documents": accepted_documents,
            "rejected_rows": rejected_rows,
            "duplicate_rows": duplicate_rows,
        },
        "train": {
            "documents": sum(shard["documents"] for shard in train_shards),
            "cjk_characters": train_cjk_characters,
            "shards": train_shards,
        },
        "validation": {
            "path": validation_path,
            "documents": len(validation_documents),
            "cjk_characters": validation_cjk_characters,
            "fraction": args.validation_fraction,
            "sha256": sha256_file(validation_path),
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(f"Downloaded corpus: {output_dir}")
    print(f"Accepted documents: {accepted_documents:,}")
    print(f"Effective CJK characters: {cjk_characters:,}")
    print(f"Training shards: {len(train_shards):,}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
