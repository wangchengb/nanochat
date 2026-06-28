"""Reproducibility helpers shared by NanoChat training entry points."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | os.PathLike[str], payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def git_metadata(repo_root: str | os.PathLike[str]) -> dict:
    repo_root = str(repo_root)

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def file_metadata(path: str | os.PathLike[str] | None) -> dict | None:
    if path is None:
        return None
    expanded = Path(path).expanduser().resolve()
    if not expanded.exists():
        return {"path": str(expanded), "exists": False}
    return {
        "path": str(expanded),
        "exists": True,
        "size": expanded.stat().st_size,
        "sha256": sha256_file(expanded),
    }


def directory_inventory(path: str | os.PathLike[str]) -> dict:
    """Record a cheap, deterministic inventory without hashing multi-GB datasets."""
    root = Path(path).expanduser().resolve()
    files = sorted(item for item in root.iterdir() if item.is_file())
    inventory = [
        {"name": item.name, "size": item.stat().st_size}
        for item in files
    ]
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    manifest_path = root / "manifest.json"
    return {
        "path": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in inventory),
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest": file_metadata(manifest_path) if manifest_path.exists() else None,
    }


def build_training_manifest(
    *,
    repo_root: str,
    command: list[str],
    user_config: dict,
    resolved_config: dict,
    output: dict,
    source_checkpoint: dict | None,
    tokenizer_dir: str,
    data_dir: str,
) -> dict:
    tokenizer_root = Path(tokenizer_dir).expanduser().resolve()
    tokenizer_files = {}
    for name in ("tokenizer.pkl", "tokenizer.json", "token_bytes.pt"):
        path = tokenizer_root / name
        if path.exists():
            tokenizer_files[name] = file_metadata(path)

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git": git_metadata(repo_root),
        "environment": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "output": output,
        "source_checkpoint": source_checkpoint,
        "tokenizer": {
            "path": str(tokenizer_root),
            "files": tokenizer_files,
        },
        "data": directory_inventory(data_dir),
        "arguments": user_config,
        "resolved": resolved_config,
    }


def checkpoint_source_metadata(
    checkpoint_dir: str,
    step: int,
) -> dict:
    root = Path(checkpoint_dir).expanduser().resolve()
    return {
        "directory": str(root),
        "step": step,
        "model": file_metadata(root / f"model_{step:06d}.pt"),
        "meta": file_metadata(root / f"meta_{step:06d}.json"),
    }


def ensure_fresh_checkpoint_dir(
    checkpoint_dir: str,
    *,
    resuming: bool,
    dry_run: bool,
) -> None:
    if resuming or dry_run:
        return
    root = Path(checkpoint_dir)
    if not root.exists():
        return
    existing = sorted(root.glob("model_*.pt"))
    if existing:
        raise FileExistsError(
            f"Refusing to write a fresh run into existing checkpoint directory: {root}. "
            "Choose a new --model-tag or use --resume-from-step."
        )
