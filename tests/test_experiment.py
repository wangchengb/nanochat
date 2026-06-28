import json

from nanochat.experiment import (
    atomic_write_json,
    directory_inventory,
    ensure_fresh_checkpoint_dir,
    sha256_file,
)


def test_sha256_and_atomic_json(tmp_path):
    payload_path = tmp_path / "payload.json"
    atomic_write_json(payload_path, {"value": "你好"})
    assert json.loads(payload_path.read_text()) == {"value": "你好"}
    assert len(sha256_file(payload_path)) == 64


def test_directory_inventory_is_deterministic(tmp_path):
    (tmp_path / "b.txt").write_text("bb")
    (tmp_path / "a.txt").write_text("a")
    first = directory_inventory(tmp_path)
    second = directory_inventory(tmp_path)
    assert first["file_count"] == 2
    assert first["total_bytes"] == 3
    assert first["inventory_sha256"] == second["inventory_sha256"]


def test_existing_checkpoint_requires_resume(tmp_path):
    checkpoint_dir = tmp_path / "model"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_000001.pt").write_bytes(b"checkpoint")
    try:
        ensure_fresh_checkpoint_dir(
            checkpoint_dir,
            resuming=False,
            dry_run=False,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected an existing checkpoint to be rejected")
    ensure_fresh_checkpoint_dir(checkpoint_dir, resuming=True, dry_run=False)
    ensure_fresh_checkpoint_dir(checkpoint_dir, resuming=False, dry_run=True)
