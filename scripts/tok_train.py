"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import json
import resource
import sys
import threading
import torch
from nanochat.tokenizer import (
    RustBPETokenizer,
    SPLIT_PATTERN_PROFILES,
    get_split_pattern,
)
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.experiment import atomic_write_json, directory_inventory, git_metadata, sha256_file

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
parser.add_argument('--data-dir', type=str, default=None, help='Optional parquet directory; final shard is excluded from training')
parser.add_argument('--tokenizer-tag', type=str, default=None, help='Write to $NANOCHAT_BASE_DIR/tokenizers/<tag>')
parser.add_argument('--output-dir', type=str, default=None, help='Explicit output directory (mutually exclusive with --tokenizer-tag)')
parser.add_argument('--overwrite', action='store_true', help='Allow replacing an existing tokenizer output')
parser.add_argument(
    '--split-pattern-profile',
    choices=sorted(SPLIT_PATTERN_PROFILES),
    default='default',
    help='Pretokenization profile; default preserves the original NanoChat behavior',
)
parser.add_argument(
    '--progress-every-seconds',
    type=float,
    default=30,
    help='Print a training heartbeat at this interval; 0 disables it',
)
args = parser.parse_args()
if args.tokenizer_tag is not None and args.output_dir is not None:
    parser.error("Specify only one of --tokenizer-tag or --output-dir")
if args.progress_every_seconds < 0:
    parser.error("--progress-every-seconds must be non-negative")
print(f"max_chars: {args.max_chars:,}", flush=True)
print(f"doc_cap: {args.doc_cap:,}", flush=True)
print(f"vocab_size: {args.vocab_size:,}", flush=True)
print(f"split_pattern_profile: {args.split_pattern_profile}", flush=True)
print(f"progress_every_seconds: {args.progress_every_seconds:g}", flush=True)
split_pattern = get_split_pattern(args.split_pattern_profile)

# -----------------------------------------------------------------------------
# Text iterator

progress = {
    "phase": "reading-corpus",
    "documents": 0,
    "characters": 0,
}


def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train", data_dir=args.data_dir):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            progress["documents"] += 1
            progress["characters"] = nchars
            yield doc_text
            if nchars > args.max_chars:
                progress["phase"] = "bpe-pair-counting-and-merges"
                return
    progress["phase"] = "bpe-pair-counting-and-merges"


def peak_rss_gib():
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return rss / (1024 ** 3)


def progress_heartbeat(stop_event, start_time):
    interval = args.progress_every_seconds
    while not stop_event.wait(interval):
        elapsed = time.monotonic() - start_time
        print(
            "[tokenizer-progress] "
            f"phase={progress['phase']} "
            f"elapsed={elapsed / 60:.1f}m "
            f"documents={progress['documents']:,} "
            f"characters={progress['characters']:,}/{args.max_chars:,} "
            f"peak_rss={peak_rss_gib():.2f}GiB",
            flush=True,
        )


text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
wall_start = time.monotonic()
stop_event = threading.Event()
heartbeat_thread = None
if args.progress_every_seconds > 0:
    heartbeat_thread = threading.Thread(
        target=progress_heartbeat,
        args=(stop_event, wall_start),
        name="tokenizer-progress",
        daemon=True,
    )
    heartbeat_thread.start()
try:
    tokenizer = RustBPETokenizer.train_from_iterator(
        text_iter,
        args.vocab_size,
        split_pattern=split_pattern,
    )
finally:
    stop_event.set()
    if heartbeat_thread is not None:
        heartbeat_thread.join(timeout=1)
train_time = time.monotonic() - wall_start
progress["phase"] = "complete"
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
if args.output_dir is not None:
    tokenizer_dir = os.path.abspath(os.path.expanduser(args.output_dir))
elif args.tokenizer_tag is not None:
    tokenizer_dir = os.path.join(base_dir, "tokenizers", args.tokenizer_tag)
else:
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.pkl")) and not args.overwrite:
    raise FileExistsError(
        f"Tokenizer already exists at {tokenizer_dir}; pass --overwrite or choose a new tag"
    )
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

manifest = {
    "schema_version": 1,
    "command": [sys.executable, "-m", "scripts.tok_train", *sys.argv[1:]],
    "git": git_metadata(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output": {
        "directory": tokenizer_dir,
        "tag": args.tokenizer_tag,
        "vocab_size": vocab_size,
        "tokenizer_sha256": sha256_file(os.path.join(tokenizer_dir, "tokenizer.pkl")),
        "token_bytes_sha256": sha256_file(token_bytes_path),
    },
    "training": {
        "max_chars": args.max_chars,
        "doc_cap": args.doc_cap,
        "train_time_seconds": train_time,
        "documents": progress["documents"],
        "characters": progress["characters"],
        "peak_rss_gib": peak_rss_gib(),
        "split_pattern_profile": args.split_pattern_profile,
        "split_pattern": split_pattern,
    },
    "data": directory_inventory(
        args.data_dir or os.path.join(base_dir, "base_data_climbmix")
    ),
}
atomic_write_json(os.path.join(tokenizer_dir, "manifest.json"), manifest)
print(f"Saved tokenizer manifest to {os.path.join(tokenizer_dir, 'manifest.json')}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
