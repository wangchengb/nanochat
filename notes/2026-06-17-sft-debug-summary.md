# NanoChat SFT Debug Summary

Date: 2026-06-17

## Repository State

Repository path:

```text
/Users/tw/Documents/NanoChat/nanochat
```

Current clean fix branch:

```text
codex/fix-chat-sft-stability
```

Pushed remote branch:

```text
origin/codex/fix-chat-sft-stability
```

Core commit:

```text
5f83781 修复 Chat SFT 本地训练稳定性问题
```

The old local branch `codex/learning-improvements` was deleted.

## Problem

On a MacBook M4 with 16 GB memory, `scripts/chat_sft.py` produced `loss: nan` early in SFT training.

Observed behavior:

```text
step 00001-00003: finite loss
step 00004 or step 00006: loss becomes nan
```

Lowering learning rates alone did not solve the issue, which suggested the failure was not only LR explosion.

## Root Cause

SFT only computes loss on assistant response tokens. User tokens, special tokens, tool outputs, and padding are masked with `-1`.

If a batch has no valid assistant target tokens, all targets are `-1`. Then:

```python
F.cross_entropy(..., ignore_index=-1, reduction="mean")
```

computes the mean over an empty set and returns NaN.

This matches the upstream NanoChat discussion around:

```text
#590 SFT produces NaN on small batch sizes
#662 Skip all masked batch in SFT
#775 chat_sft --num-iterations should count optimizer steps, not micro-batches
```

## Implemented Fix

Changed `scripts/chat_sft.py`:

- Pass `max_tokens=row_capacity` into `tokenizer.render_conversation(...)`.
- Skip fully masked batches:

```python
if (targets != -1).sum().item() == 0:
    continue
```

- Make `--num-iterations` count optimizer steps instead of dataloader micro-batches.
- Clamp LR progress to `[0, 1]`.
- Update progress correctly during gradient accumulation.

Changed `runs/runcpu.sh`:

- Add this SFT argument:

```bash
--chatcore-every=-1
```

This avoids running full ChatCORE during training on local Mac hardware.

## Verification

Static checks:

```text
scripts/chat_sft.py passed Python AST parsing
runs/runcpu.sh passed bash -n
```

Actual training reached hundreds of steps without NaN.

Example stable log:

```text
step 00479 (31.93%) | loss: 2.524057 | lrm: 1.00 | dt: 2418.12ms | tok/sec: 6,775
```

## Chat Evaluation

Command:

```bash
python -m scripts.chat_eval -i sft -b 8 -x 128 --device-type=mps
```

Results:

```text
ARC-Easy       29/128 = 22.66%
ARC-Challenge  31/128 = 24.22%
MMLU           28/128 = 21.88%
GSM8K           0/128 = 0.00%
HumanEval       0/128 = 0.00%
SpellingBee   117/128 = 91.41%
```

Interpretation:

- SFT clearly worked for simple rule/instruction tasks such as SpellingBee.
- The d6 local model remains weak on broad knowledge, math, and code tasks.
- The main success was stabilizing SFT and confirming that the training pipeline works end to end on local Mac hardware.

## Git Notes

`origin` now uses SSH:

```text
git@github.com:wangchengb/nanochat.git
```

Recommended branch for the minimal SFT fix:

```text
codex/fix-chat-sft-stability
```

## Suggested Next Learning Topic

Continue by mapping the full NanoChat pipeline:

```text
tokenizer -> base_train -> base_eval -> chat_sft -> chat_eval
```

Recommended next-thread prompt:

```text
We continue studying NanoChat. Please first read notes/2026-06-17-sft-debug-summary.md, then help me map the full training pipeline: tokenizer -> base_train -> base_eval -> chat_sft -> chat_eval.
```
