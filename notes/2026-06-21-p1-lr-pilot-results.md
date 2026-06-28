# P1 English CPT Learning-Rate Pilot

Date: 2026-06-21

## Setup

```text
source checkpoint: base/d6/step5000
data: ClimbMix English
steps per run: 400
tokens per run: 6,553,600
optimizer: fresh
dataloader: fresh
warmup: 40 steps
warmdown: final 25%
```

## Results

All runs started from the same measured validation BPB:

```text
1.168731
```

| LR multiplier | Step 200 BPB | Step 400 BPB | Change from start |
|---:|---:|---:|---:|
| 0.10x | 1.170137 | 1.167517 | -0.001214 |
| 0.15x | 1.171707 | 1.168502 | -0.000229 |
| 0.20x | 1.173584 | 1.169849 | +0.001118 |
| 0.25x | 1.175588 | 1.171457 | +0.002726 |

The measured short-run optimum was 0.10x. Higher learning rates caused larger
early validation regressions, and 0.20x/0.25x did not recover below the starting
BPB by step 400.

The fixed base CORE-16 baseline is:

```text
CORE: 0.041618
```

This evaluation is suitable for detecting large regressions, but one example is
6.25 percentage points per task, so it is not evidence for small improvements.

## P2 Decision

The project owner chose not to run additional 0.05x/0.075x pilots or candidate
CORE evaluations. P2 will intentionally use the original 1.0x pretraining
learning rates:

```text
embedding: 0.3
unembedding: 0.008
matrix: 0.02
scalar: 0.5
```

This is an explicit experimental decision rather than the recommendation of the
P1 short-run measurements. P2 uses a new tag, fresh optimizer, intermediate
checkpoints, and periodic validation so the original checkpoints remain intact
and the run can be stopped if validation quality regresses.

