# NanoChat Pipeline Map

Date: 2026-06-17

This note continues from `2026-06-17-sft-debug-summary.md` and maps the main
NanoChat training/evaluation path:

```text
dataset -> tokenizer -> base_train -> base_eval -> chat_sft -> chat_eval
```

## Artifact Root

Most generated artifacts live under `NANOCHAT_BASE_DIR`.

Default:

```text
$HOME/.cache/nanochat
```

Set in the run scripts:

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
```

Important subdirectories:

```text
$NANOCHAT_BASE_DIR/
  base_data_climbmix/          # downloaded pretraining parquet shards
  tokenizer/                   # tokenizer.pkl + token_bytes.pt
  base_checkpoints/<tag>/      # base model checkpoints
  chatsft_checkpoints/<tag>/   # SFT checkpoints
  eval_bundle/                 # CORE eval data
  base_eval/                   # base eval CSV outputs
  report/                      # report fragments
```

## 0. Dataset Download

Entry:

```bash
python -m nanochat.dataset -n 8
```

Script:

```text
nanochat/dataset.py
```

Purpose:

- Download ClimbMix parquet shards from Hugging Face.
- Always include the final shard as validation data.
- Store shards in `$NANOCHAT_BASE_DIR/base_data_climbmix`.

Split rule:

- Train split: all shards except the last.
- Validation split: last shard only.

Used by:

- `scripts/tok_train.py`
- `scripts/tok_eval.py`
- `scripts/base_train.py`
- `scripts/base_eval.py`

## 1. Tokenizer

Entry:

```bash
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval
```

Scripts:

```text
scripts/tok_train.py
scripts/tok_eval.py
nanochat/tokenizer.py
```

Input:

```text
$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet
```

Output:

```text
$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl
$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt
```

What happens:

- `tok_train.py` streams train documents through `parquets_iter_batched`.
- Each document is capped by `--doc-cap`.
- `RustBPETokenizer.train_from_iterator` trains GPT-4-style BPE with NanoChat special tokens.
- `tokenizer.pkl` stores the tiktoken-compatible encoding.
- `token_bytes.pt` stores byte lengths per token for bits-per-byte evaluation.

Special tokens:

```text
<|bos|>
<|user_start|> <|user_end|>
<|assistant_start|> <|assistant_end|>
<|python_start|> <|python_end|>
<|output_start|> <|output_end|>
```

Key idea:

- Base training uses normal text plus BOS.
- SFT/eval use conversation rendering with role/tool special tokens.

## 2. Base Training

Entry:

```bash
python -m scripts.base_train ...
```

Distributed reference:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
```

Local CPU/MPS demo:

```bash
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN
```

Scripts/modules:

```text
scripts/base_train.py
nanochat/dataloader.py
nanochat/gpt.py
nanochat/optim.py
nanochat/loss_eval.py
nanochat/checkpoint_manager.py
```

Input:

```text
$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl
$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt
$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet
```

Output:

```text
$NANOCHAT_BASE_DIR/base_checkpoints/<model-tag>/
  model_<step>.pt
  meta_<step>.json
  optim_<step>_rank<rank>.pt
```

Core flow:

1. Load tokenizer and token byte table.
2. Build a GPT config from `--depth`.
3. Derive model width, number of heads, training horizon, batch size, LR scaling, and weight decay.
4. Build a BOS-aligned best-fit pretraining dataloader.
5. Train next-token prediction on packed text rows.
6. Periodically evaluate validation BPB, optional CORE, optional samples.
7. Save checkpoint and metadata.

Important dataloader behavior:

- Every row starts with BOS.
- Documents are best-fit packed into `max_seq_len + 1` tokens.
- If no document fits, a document is cropped to fill the row.
- Base training has 100% token utilization and no target masking.

## 3. Base Evaluation

Entry:

```bash
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16
```

Reference:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
```

Script:

```text
scripts/base_eval.py
```

Input:

```text
$NANOCHAT_BASE_DIR/base_checkpoints/<model-tag>/
$NANOCHAT_BASE_DIR/tokenizer/
$NANOCHAT_BASE_DIR/eval_bundle/
```

Output:

```text
$NANOCHAT_BASE_DIR/base_eval/base_model_<step>.csv
report section: Base model evaluation
```

Eval modes:

```text
core,bpb,sample
```

What each mode means:

- `bpb`: bits per byte on train/val parquet splits.
- `core`: DCLM CORE-style in-context evaluation.
- `sample`: short generations from fixed prompts.

Checkpoint loading:

- `load_model("base", ...)` guesses the largest `d<number>` tag if no tag is given.
- It then loads the latest `model_<step>.pt` unless `--step` is provided.

## 4. Chat SFT

Entry:

```bash
python -m scripts.chat_sft ...
```

Local CPU/MPS demo:

```bash
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --chatcore-every=-1 \
    --run=$WANDB_RUN
```

Scripts/modules:

```text
scripts/chat_sft.py
nanochat/tokenizer.py
nanochat/checkpoint_manager.py
tasks/*.py
```

Input:

```text
$NANOCHAT_BASE_DIR/base_checkpoints/<model-tag>/
$NANOCHAT_BASE_DIR/tokenizer/
$NANOCHAT_BASE_DIR/identity_conversations.jsonl
HF/task datasets loaded by tasks/*
```

Output:

```text
$NANOCHAT_BASE_DIR/chatsft_checkpoints/<model-tag>/
  model_<step>.pt
  meta_<step>.json
  optim_<step>_rank<rank>.pt
```

Training mixture:

```text
SmolTalk train
identity_conversations.jsonl x2
MMLU auxiliary_train x --mmlu-epochs
GSM8K train x --gsm8k-epochs
SimpleSpelling
SpellingBee
```

Validation mixture:

```text
SmolTalk test
MMLU test subset
GSM8K test subset
```

Core flow:

1. Load base checkpoint with `load_model("base", phase="train")`.
2. Inherit `max_seq_len`, device batch size, total batch size, and base LRs unless overridden.
3. Optionally load optimizer state from the base checkpoint.
4. Build a deterministic `TaskMixture` of conversation datasets.
5. Render each conversation with `tokenizer.render_conversation`.
6. Pack full conversations into rows with padding when no conversation fits.
7. Train only on assistant target tokens.
8. Save an SFT checkpoint at the end.

SFT target masking:

- User messages: masked out.
- BOS and role boundary tokens before assistant text: mostly masked out.
- Assistant response text: trained.
- Python tool calls inside assistant messages: trained.
- Python tool outputs: masked out.
- Padding: masked out.

Important local stability fix:

```python
if (targets != -1).sum().item() == 0:
    continue
```

Without this guard, `F.cross_entropy(..., ignore_index=-1, reduction="mean")`
can return NaN on batches with no supervised assistant targets.

## 5. Chat Evaluation

Entry:

```bash
python -m scripts.chat_eval -i sft -b 8 -x 128 --device-type=mps
```

Script:

```text
scripts/chat_eval.py
```

Input:

```text
$NANOCHAT_BASE_DIR/chatsft_checkpoints/<model-tag>/
$NANOCHAT_BASE_DIR/tokenizer/
tasks/*
```

Output:

```text
console accuracies
report section: Chat evaluation sft
```

Default tasks:

```text
ARC-Easy
ARC-Challenge
MMLU
GSM8K
HumanEval
SpellingBee
```

Two eval styles:

- Categorical: ARC and MMLU batch prompts and score only answer-letter logits.
- Generative: GSM8K, HumanEval, and SpellingBee sample completions and call task-specific `evaluate`.

Prompt rendering:

- `render_for_completion` removes the final assistant answer from a task conversation.
- It renders the remaining conversation.
- It appends `<|assistant_start|>` to prime the model to answer.

ChatCORE:

- If all default tasks are evaluated, `chat_eval.py` computes a centered mean accuracy.
- Random baselines are 25% for categorical multiple-choice tasks and 0% for generative tasks.

## Full Pipeline In One View

```text
1. python -m nanochat.dataset -n 8
      |
      v
   base_data_climbmix/*.parquet
      |
      v
2. python -m scripts.tok_train
      |
      v
   tokenizer/tokenizer.pkl
   tokenizer/token_bytes.pt
      |
      v
3. python -m scripts.base_train
      |
      v
   base_checkpoints/d<depth>/model_<step>.pt
   base_checkpoints/d<depth>/meta_<step>.json
      |
      v
4. python -m scripts.base_eval
      |
      v
   CORE, BPB, samples
      |
      v
5. python -m scripts.chat_sft
      |
      v
   chatsft_checkpoints/d<depth>/model_<step>.pt
   chatsft_checkpoints/d<depth>/meta_<step>.json
      |
      v
6. python -m scripts.chat_eval -i sft
      |
      v
   ARC/MMLU/GSM8K/HumanEval/SpellingBee accuracies
```

## Mental Model

Tokenizer is the shared contract.

- Base training learns the language model from raw text.
- Base evaluation checks whether the pretrained model acquired broad next-token ability.
- SFT teaches the same base model the chat protocol, answer formats, tool-call format, and task habits.
- Chat evaluation checks instruction-following/task behavior under the chat protocol.

The main handoff artifacts are:

```text
tokenizer.pkl -> base_train/base_eval/chat_sft/chat_eval
token_bytes.pt -> BPB evaluation in base_train/base_eval/chat_sft
base_checkpoints -> chat_sft
chatsft_checkpoints -> chat_eval/chat_cli/chat_web
```

## Local Mac Notes

The `runs/runcpu.sh` path is intentionally educational, not capability-oriented.

For MPS/CPU stability:

- Use a small depth such as `--depth=6`.
- Use `--window-pattern=L`.
- Keep `--max-seq-len=512`.
- Keep `--total-batch-size` divisible by `device_batch_size * max_seq_len * world_size`.
- Disable full ChatCORE inside SFT with `--chatcore-every=-1` for local runs.

Recent local SFT fix branch:

```text
codex/fix-chat-sft-stability
```

Known good SFT verification from the previous note:

```text
step 00479 (31.93%) | loss: 2.524057 | lrm: 1.00 | dt: 2418.12ms | tok/sec: 6,775
```
