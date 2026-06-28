#!/bin/bash
set -euo pipefail

# Gated bilingual optimization runner. Expensive phases are intentionally
# separate: finishing one stage does not automatically launch the next.

cd "$(dirname "$0")/.."
source .venv/bin/activate

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
DEVICE_TYPE="${DEVICE_TYPE:-mps}"
P1_STEPS="${P1_STEPS:-400}"
P2_STEPS="${P2_STEPS:-6000}"
FORMAL_STEPS="${FORMAL_STEPS:-12000}"
STAGE="${1:-}"
SCALE="${2:-}"

usage() {
    echo "Usage:"
    echo "  bash runs/run_bilingual_optimization.sh p0-test"
    echo "  bash runs/run_bilingual_optimization.sh p1-dry-run 010|015|020|025"
    echo "  bash runs/run_bilingual_optimization.sh p1-train   010|015|020|025"
    echo "  bash runs/run_bilingual_optimization.sh p1-baseline-eval"
    echo "  bash runs/run_bilingual_optimization.sh p1-eval    010|015|020|025"
    echo "  bash runs/run_bilingual_optimization.sh p1-compare"
    echo "  bash runs/run_bilingual_optimization.sh p2-dry-run"
    echo "  bash runs/run_bilingual_optimization.sh p2-train"
    echo "  bash runs/run_bilingual_optimization.sh p2-eval [step]"
    echo "  bash runs/run_bilingual_optimization.sh tokenizer-data"
    echo "  bash runs/run_bilingual_optimization.sh tokenizer-pilot han1|han2|han4|han8"
    echo "  bash runs/run_bilingual_optimization.sh tokenizer-train"
    echo "  bash runs/run_bilingual_optimization.sh tokenizer-eval"
    echo "  bash runs/run_bilingual_optimization.sh pretrain-data"
    echo "  bash runs/run_bilingual_optimization.sh pretrain-smoke"
    echo "  bash runs/run_bilingual_optimization.sh pretrain-pilot"
    echo "  bash runs/run_bilingual_optimization.sh pretrain-train"
    echo "  bash runs/run_bilingual_optimization.sh pretrain-eval [step]"
    echo "  bash runs/run_bilingual_optimization.sh pretrain-baseline-eval"
    echo "  bash runs/run_bilingual_optimization.sh sft-train"
    echo "  bash runs/run_bilingual_optimization.sh sft-eval [step]"
    echo "  bash runs/run_bilingual_optimization.sh sft-v2-prepare"
    echo "  bash runs/run_bilingual_optimization.sh sft-v2-train"
    echo "  bash runs/run_bilingual_optimization.sh sft-v2-eval [step]"
    echo "  bash runs/run_bilingual_optimization.sh sft-v2-eval-full [step]"
    echo "  bash runs/run_bilingual_optimization.sh sft-v3-prepare"
    echo "  bash runs/run_bilingual_optimization.sh sft-v3-train"
    echo "  bash runs/run_bilingual_optimization.sh sft-v3-eval [step]"
    echo "  bash runs/run_bilingual_optimization.sh sft-v4-prepare"
    echo "  bash runs/run_bilingual_optimization.sh sft-v4-train"
    echo "  bash runs/run_bilingual_optimization.sh sft-v4-eval [step]"
    echo "  bash runs/run_bilingual_optimization.sh math-sft-prepare"
    echo "  bash runs/run_bilingual_optimization.sh math-sft-baseline-eval"
    echo "  bash runs/run_bilingual_optimization.sh math-sft-train"
    echo "  bash runs/run_bilingual_optimization.sh math-sft-eval [step]"
}

lr_args() {
    case "$1" in
        010) echo "0.03 0.0008 0.002 0.05" ;;
        015) echo "0.045 0.0012 0.003 0.075" ;;
        020) echo "0.06 0.0016 0.004 0.10" ;;
        025) echo "0.075 0.002 0.005 0.125" ;;
        *)
            echo "Unknown LR scale '$1'; expected 010, 015, 020, or 025" >&2
            exit 2
            ;;
    esac
}

run_p1() {
    local mode="$1"
    local scale="$2"
    local values
    values="$(lr_args "$scale")"
    read -r embedding_lr unembedding_lr matrix_lr scalar_lr <<< "$values"

    local dry_run_args=()
    local device_batch_size=32
    local total_batch_size=16384
    if [ "$mode" = "dry-run" ]; then
        dry_run_args=(--dry-run --no-compile)
        device_batch_size="${DRY_RUN_DEVICE_BATCH_SIZE:-1}"
        total_batch_size="${DRY_RUN_TOTAL_BATCH_SIZE:-512}"
    fi

    local command=(
        python -m scripts.base_train
        --device-type="$DEVICE_TYPE"
        --depth=6
        --aspect-ratio=64
        --head-dim=64
        --window-pattern=L
        --max-seq-len=512
        --device-batch-size="$device_batch_size"
        --total-batch-size="$total_batch_size"
        --num-iterations="$P1_STEPS"
        --embedding-lr="$embedding_lr"
        --unembedding-lr="$unembedding_lr"
        --matrix-lr="$matrix_lr"
        --scalar-lr="$scalar_lr"
        --warmup-steps=40
        --warmdown-ratio=0.25
        --final-lr-frac=0.10
        --eval-every=100
        --eval-tokens=131072
        --core-metric-every=-1
        --sample-every=100
        --save-every=200
        --init-from-model-tag=d6
        --init-from-step=5000
        --model-tag="d6-en-cpt-lr${scale}-pilot"
        --data-dir="$NANOCHAT_BASE_DIR/base_data_climbmix"
        --run=dummy
    )
    if [ "$mode" = "dry-run" ]; then
        command+=("${dry_run_args[@]}")
    fi
    "${command[@]}"
}

run_p2() {
    local mode="$1"
    local dry_run_args=()
    local device_batch_size=32
    local total_batch_size=16384
    if [ "$mode" = "dry-run" ]; then
        dry_run_args=(--dry-run --no-compile)
        device_batch_size="${DRY_RUN_DEVICE_BATCH_SIZE:-1}"
        total_batch_size="${DRY_RUN_TOTAL_BATCH_SIZE:-512}"
    fi

    local command=(
        python -m scripts.base_train
        --device-type="$DEVICE_TYPE"
        --depth=6
        --aspect-ratio=64
        --head-dim=64
        --window-pattern=L
        --max-seq-len=512
        --device-batch-size="$device_batch_size"
        --total-batch-size="$total_batch_size"
        --num-iterations="$P2_STEPS"
        --embedding-lr=0.3
        --unembedding-lr=0.008
        --matrix-lr=0.02
        --scalar-lr=0.5
        --warmup-steps=40
        --warmdown-ratio=0.65
        --final-lr-frac=0.05
        --eval-every=500
        --eval-tokens=131072
        --core-metric-every=-1
        --sample-every=500
        --save-every=1000
        --init-from-model-tag=d6
        --init-from-step=5000
        --model-tag=d6-en-cpt-1x-r1
        --data-dir="$NANOCHAT_BASE_DIR/base_data_climbmix"
        --run=dummy
    )
    if [ "$mode" = "dry-run" ]; then
        command+=("${dry_run_args[@]}")
    fi
    "${command[@]}"
}

case "$STAGE" in
    p0-test)
        python -m compileall -q nanochat scripts tasks tests
        if python -c "import pytest" >/dev/null 2>&1; then
            python -m pytest -q \
                tests/test_engine.py \
                tests/test_zh_experiment.py \
                tests/test_bilingual_eval.py \
                tests/test_experiment.py
        else
            python -m scripts.run_test_functions
        fi
        ;;
    p1-dry-run)
        run_p1 dry-run "$SCALE"
        ;;
    p1-train)
        run_p1 train "$SCALE"
        ;;
    p1-baseline-eval)
        python -m scripts.eval_bilingual \
            -i base \
            -g d6 \
            -s 5000 \
            --groups=language,bpb,core \
            --bpb-split-tokens=131072 \
            --core-max-per-task=128 \
            --language-max-new-tokens=64 \
            --device-type="$DEVICE_TYPE" \
            --run-id=p1-baseline-d6-step5000
        ;;
    p1-eval)
        python -m scripts.eval_bilingual \
            -i base \
            -g "d6-en-cpt-lr${SCALE}-pilot" \
            -s "$P1_STEPS" \
            --groups=language,bpb,core \
            --bpb-split-tokens=131072 \
            --core-max-per-task=128 \
            --language-max-new-tokens=64 \
            --device-type="$DEVICE_TYPE" \
            --run-id="p1-d6-en-cpt-lr${SCALE}-step${P1_STEPS}"
        ;;
    p1-compare)
        python -m scripts.compare_evaluations \
            "$NANOCHAT_BASE_DIR/evaluations/p1-baseline-d6-step5000" \
            "$NANOCHAT_BASE_DIR/evaluations/p1-d6-en-cpt-lr010-step${P1_STEPS}" \
            "$NANOCHAT_BASE_DIR/evaluations/p1-d6-en-cpt-lr015-step${P1_STEPS}" \
            "$NANOCHAT_BASE_DIR/evaluations/p1-d6-en-cpt-lr020-step${P1_STEPS}" \
            "$NANOCHAT_BASE_DIR/evaluations/p1-d6-en-cpt-lr025-step${P1_STEPS}"
        ;;
    p2-dry-run)
        run_p2 dry-run
        ;;
    p2-train)
        run_p2 train
        ;;
    p2-eval)
        P2_EVAL_STEP="${SCALE:-$P2_STEPS}"
        python -m scripts.eval_bilingual \
            -i base \
            -g d6-en-cpt-1x-r1 \
            -s "$P2_EVAL_STEP" \
            --groups=language,bpb,core \
            --bpb-split-tokens=131072 \
            --core-max-per-task=16 \
            --language-max-new-tokens=64 \
            --device-type="$DEVICE_TYPE" \
            --run-id="p2-d6-en-cpt-1x-step${P2_EVAL_STEP}"
        ;;
    tokenizer-data)
        python -m scripts.prepare_bilingual_tokenizer_data \
            --zh-source-dir="$NANOCHAT_BASE_DIR/zh_corpus/cci3_hq_1b_local" \
            --en-source-dir="$NANOCHAT_BASE_DIR/base_data_climbmix" \
            --output-dir="$NANOCHAT_BASE_DIR/bilingual_tokenizer_data/cci3-climbmix-en1b-zh1b-v1" \
            --chars-per-language=970000000 \
            --chars-per-shard=50000000 \
            --validation-chars-per-language=5000000 \
            --overwrite
        ;;
    tokenizer-train)
        python -m scripts.tok_train \
            --data-dir="$NANOCHAT_BASE_DIR/bilingual_tokenizer_data/cci3-climbmix-en1b-zh1b-v1" \
            --tokenizer-tag=bilingual-32k-han1-2b-v1 \
            --max-chars=2500000000 \
            --doc-cap=10000 \
            --vocab-size=32768 \
            --split-pattern-profile=bilingual-han1 \
            --progress-every-seconds=30
        ;;
    tokenizer-pilot)
        case "$SCALE" in
            han1|han2|han4|han8) ;;
            *)
                echo "Usage: bash runs/run_bilingual_optimization.sh tokenizer-pilot han1|han2|han4|han8" >&2
                exit 2
                ;;
        esac
        python -m scripts.tok_train \
            --data-dir="$NANOCHAT_BASE_DIR/bilingual_tokenizer_data/cci3-climbmix-en1b-zh1b-v1" \
            --tokenizer-tag="bilingual-32k-${SCALE}-250m-pilot" \
            --max-chars=250000000 \
            --doc-cap=10000 \
            --vocab-size=32768 \
            --split-pattern-profile="bilingual-${SCALE}" \
            --progress-every-seconds=10
        ;;
    tokenizer-eval)
        python -m scripts.eval_tokenizer_bilingual \
            --candidate-tag=bilingual-32k-han1-2b-v1
        ;;
    pretrain-data)
        python -m scripts.prepare_bilingual_pretrain_data \
            --zh-source-dir="$NANOCHAT_BASE_DIR/zh_corpus/cci3_hq_1b_local" \
            --en-source-dir="$NANOCHAT_BASE_DIR/base_data_climbmix" \
            --output-dir="$NANOCHAT_BASE_DIR/bilingual_pretrain_data/han1-en60-zh40-200m-v1" \
            --tokenizer-tag=bilingual-32k-han1-2b-v1 \
            --total-training-tokens=200000000 \
            --english-token-ratio=0.60 \
            --tokens-per-shard=10000000 \
            --validation-tokens-per-language=2000000 \
            --max-chunk-tokens=512 \
            --min-chunk-tokens=32 \
            --overwrite
        ;;
    pretrain-smoke)
        python -m scripts.base_train \
            --device-type="$DEVICE_TYPE" \
            --dry-run \
            --no-compile \
            --depth=6 \
            --aspect-ratio=64 \
            --head-dim=64 \
            --window-pattern=L \
            --max-seq-len=512 \
            --device-batch-size=1 \
            --total-batch-size=512 \
            --num-iterations=1 \
            --eval-every=-1 \
            --core-metric-every=-1 \
            --sample-every=-1 \
            --save-every=-1 \
            --model-tag=d6-bi-han1-smoke \
            --data-dir="${PRETRAIN_SMOKE_DATA_DIR:-$NANOCHAT_BASE_DIR/bilingual_pretrain_data/han1-en60-zh40-200m-v1}" \
            --tokenizer-tag=bilingual-32k-han1-2b-v1 \
            --run=dummy
        ;;
    pretrain-pilot)
        python -m scripts.base_train \
            --device-type="$DEVICE_TYPE" \
            --depth=6 \
            --aspect-ratio=64 \
            --head-dim=64 \
            --window-pattern=L \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=1000 \
            --embedding-lr=0.3 \
            --unembedding-lr=0.008 \
            --matrix-lr=0.02 \
            --scalar-lr=0.5 \
            --warmup-steps=40 \
            --warmdown-ratio=0.65 \
            --final-lr-frac=0.05 \
            --eval-every=200 \
            --eval-tokens=131072 \
            --core-metric-every=-1 \
            --sample-every=200 \
            --save-every=1000 \
            --model-tag=d6-bi-han1-en60-zh40-pilot \
            --data-dir="$NANOCHAT_BASE_DIR/bilingual_pretrain_data/han1-en60-zh40-200m-v1" \
            --tokenizer-tag=bilingual-32k-han1-2b-v1 \
            --run=dummy
        ;;
    pretrain-train)
        python -m scripts.base_train \
            --device-type="$DEVICE_TYPE" \
            --depth=6 \
            --aspect-ratio=64 \
            --head-dim=64 \
            --window-pattern=L \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations="$FORMAL_STEPS" \
            --embedding-lr=0.3 \
            --unembedding-lr=0.008 \
            --matrix-lr=0.02 \
            --scalar-lr=0.5 \
            --warmup-steps=40 \
            --warmdown-ratio=0.65 \
            --final-lr-frac=0.05 \
            --eval-every=1000 \
            --eval-tokens=131072 \
            --core-metric-every=-1 \
            --sample-every=1000 \
            --save-every=1000 \
            --model-tag=d6-bi-han1-en60-zh40-200m-v1 \
            --data-dir="$NANOCHAT_BASE_DIR/bilingual_pretrain_data/han1-en60-zh40-200m-v1" \
            --tokenizer-tag=bilingual-32k-han1-2b-v1 \
            --run=dummy
        ;;
    pretrain-eval)
        PRETRAIN_EVAL_STEP="${SCALE:-$FORMAL_STEPS}"
        PRETRAIN_DATA_DIR="$NANOCHAT_BASE_DIR/bilingual_pretrain_data/han1-en60-zh40-200m-v1"
        python -m scripts.eval_bilingual \
            -i base \
            -g d6-bi-han1-en60-zh40-200m-v1 \
            -s "$PRETRAIN_EVAL_STEP" \
            --groups=language,bpb,core \
            --language-max-new-tokens=64 \
            --bpb-dataset=english="$PRETRAIN_DATA_DIR/eval_en" \
            --bpb-dataset=chinese="$PRETRAIN_DATA_DIR/eval_zh" \
            --bpb-dataset=mixed="$PRETRAIN_DATA_DIR" \
            --bpb-split-tokens=131072 \
            --core-max-per-task=128 \
            --device-type="$DEVICE_TYPE" \
            --run-id="d6-bi-han1-en60-zh40-step${PRETRAIN_EVAL_STEP}"
        ;;
    pretrain-baseline-eval)
        PRETRAIN_DATA_DIR="$NANOCHAT_BASE_DIR/bilingual_pretrain_data/han1-en60-zh40-200m-v1"
        python -m scripts.eval_bilingual \
            -i base \
            -g d6 \
            -s 5000 \
            --groups=language,bpb,core \
            --language-max-new-tokens=64 \
            --bpb-dataset=english="$PRETRAIN_DATA_DIR/eval_en" \
            --bpb-dataset=chinese="$PRETRAIN_DATA_DIR/eval_zh" \
            --bpb-dataset=mixed="$PRETRAIN_DATA_DIR" \
            --bpb-split-tokens=131072 \
            --core-max-per-task=128 \
            --device-type="$DEVICE_TYPE" \
            --run-id=d6-step5000-bilingual-data-baseline
        ;;
    sft-train)
        python -m scripts.chat_sft \
            --device-type="$DEVICE_TYPE" \
            --model-tag=d6-bi-han1-en60-zh40-200m-v1 \
            --model-step=12000 \
            --output-model-tag=d6-bi-han1-en60-zh40-sft-v1 \
            --load-optimizer=0 \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=600 \
            --embedding-lr=0.3 \
            --unembedding-lr=0.008 \
            --matrix-lr=0.02 \
            --init-lr-frac=0.2 \
            --warmup-ratio=0.0 \
            --warmdown-ratio=0.5 \
            --final-lr-frac=0.0 \
            --eval-every=200 \
            --eval-tokens=131072 \
            --chatcore-every=-1 \
            --save-every=200 \
            --custom-train-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft/train.jsonl" \
            --custom-val-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft/val.jsonl" \
            --custom-train-token-ratio=0.30 \
            --run=dummy
        ;;
    sft-eval)
        SFT_EVAL_STEP="${SCALE:-600}"
        python -m scripts.eval_bilingual \
            -i sft \
            -g d6-bi-han1-en60-zh40-sft-v1 \
            -s "$SFT_EVAL_STEP" \
            --groups=language,chat \
            --language-max-new-tokens=128 \
            --chat-max-problems=128 \
            --chat-batch-size=8 \
            --device-type="$DEVICE_TYPE" \
            --run-id="d6-bi-han1-en60-zh40-sft-step${SFT_EVAL_STEP}"
        ;;
    sft-v2-prepare)
        python -m scripts.prepare_sft_v2_data \
            --source-train-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft/train.jsonl" \
            --source-val-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft/val.jsonl" \
            --output-dir="$NANOCHAT_BASE_DIR/zh_experiment/sft_v2" \
            --tokenizer-tag=bilingual-32k-han1-2b-v1
        ;;
    sft-v2-train)
        python -m scripts.chat_sft \
            --device-type="$DEVICE_TYPE" \
            --model-tag=d6-bi-han1-en60-zh40-200m-v1 \
            --model-step=12000 \
            --output-model-tag=d6-bi-han1-en60-zh40-sft-v2 \
            --load-optimizer=0 \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=600 \
            --embedding-lr=0.15 \
            --unembedding-lr=0.004 \
            --matrix-lr=0.01 \
            --init-lr-frac=0.2 \
            --warmup-ratio=0.0 \
            --warmdown-ratio=0.5 \
            --final-lr-frac=0.0 \
            --eval-every=100 \
            --eval-tokens=131072 \
            --chatcore-every=-1 \
            --save-every=100 \
            --custom-train-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v2/train.jsonl" \
            --custom-val-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v2/val.jsonl" \
            --custom-train-token-ratio=0.20 \
            --run=dummy
        ;;
    sft-v2-eval)
        SFT_EVAL_STEP="${SCALE:-600}"
        python -m scripts.eval_bilingual \
            -i sft \
            -g d6-bi-han1-en60-zh40-sft-v2 \
            -s "$SFT_EVAL_STEP" \
            --groups=language,chat \
            --language-max-new-tokens=128 \
            --chat-tasks=SpellingBee \
            --chat-max-problems=128 \
            --chat-batch-size=8 \
            --device-type="$DEVICE_TYPE" \
            --run-id="d6-bi-han1-en60-zh40-sft-v2-spelling-step${SFT_EVAL_STEP}"
        ;;
    sft-v2-eval-full)
        SFT_EVAL_STEP="${SCALE:-600}"
        python -m scripts.eval_bilingual \
            -i sft \
            -g d6-bi-han1-en60-zh40-sft-v2 \
            -s "$SFT_EVAL_STEP" \
            --groups=language,chat \
            --language-max-new-tokens=128 \
            --chat-max-problems=128 \
            --chat-batch-size=8 \
            --device-type="$DEVICE_TYPE" \
            --run-id="d6-bi-han1-en60-zh40-sft-v2-full-step${SFT_EVAL_STEP}"
        ;;
    sft-v3-prepare)
        python -m scripts.prepare_sft_v3_data \
            --source-train-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v2/train.jsonl" \
            --source-val-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v2/val.jsonl" \
            --output-dir="$NANOCHAT_BASE_DIR/zh_experiment/sft_v3" \
            --tokenizer-tag=bilingual-32k-han1-2b-v1
        ;;
    sft-v3-train)
        python -m scripts.chat_sft \
            --device-type="$DEVICE_TYPE" \
            --model-tag=d6-bi-han1-en60-zh40-200m-v1 \
            --model-step=12000 \
            --output-model-tag=d6-bi-han1-en60-zh40-sft-v3 \
            --load-optimizer=0 \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=600 \
            --embedding-lr=0.15 \
            --unembedding-lr=0.004 \
            --matrix-lr=0.01 \
            --init-lr-frac=0.2 \
            --warmup-ratio=0.0 \
            --warmdown-ratio=0.5 \
            --final-lr-frac=0.0 \
            --eval-every=100 \
            --eval-tokens=131072 \
            --chatcore-every=-1 \
            --save-every=100 \
            --custom-train-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v3/train.jsonl" \
            --custom-val-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v3/val.jsonl" \
            --custom-train-token-ratio=0.20 \
            --run=dummy
        ;;
    sft-v3-eval)
        SFT_EVAL_STEP="${SCALE:-600}"
        python -m scripts.eval_bilingual \
            -i sft \
            -g d6-bi-han1-en60-zh40-sft-v3 \
            -s "$SFT_EVAL_STEP" \
            --groups=language,chat \
            --language-max-new-tokens=128 \
            --chat-tasks=SpellingBee \
            --chat-max-problems=128 \
            --chat-batch-size=8 \
            --device-type="$DEVICE_TYPE" \
            --run-id="d6-bi-han1-en60-zh40-sft-v3-spelling-step${SFT_EVAL_STEP}"
        ;;
    sft-v4-prepare)
        python -m scripts.prepare_sft_v4_data \
            --source-train-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v2/train.jsonl" \
            --source-val-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v2/val.jsonl" \
            --output-dir="$NANOCHAT_BASE_DIR/zh_experiment/sft_v4" \
            --tokenizer-tag=bilingual-32k-han1-2b-v1
        ;;
    sft-v4-train)
        python -m scripts.chat_sft \
            --device-type="$DEVICE_TYPE" \
            --model-tag=d6-bi-han1-en60-zh40-200m-v1 \
            --model-step=12000 \
            --output-model-tag=d6-bi-han1-en60-zh40-sft-v4 \
            --load-optimizer=0 \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=600 \
            --embedding-lr=0.15 \
            --unembedding-lr=0.004 \
            --matrix-lr=0.01 \
            --init-lr-frac=0.2 \
            --warmup-ratio=0.0 \
            --warmdown-ratio=0.5 \
            --final-lr-frac=0.0 \
            --eval-every=100 \
            --eval-tokens=131072 \
            --chatcore-every=-1 \
            --save-every=100 \
            --custom-train-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v4/train.jsonl" \
            --custom-val-jsonl="$NANOCHAT_BASE_DIR/zh_experiment/sft_v4/val.jsonl" \
            --custom-train-token-ratio=0.22 \
            --run=dummy
        ;;
    sft-v4-eval)
        SFT_EVAL_STEP="${SCALE:-600}"
        python -m scripts.eval_bilingual \
            -i sft \
            -g d6-bi-han1-en60-zh40-sft-v4 \
            -s "$SFT_EVAL_STEP" \
            --groups=language,chat \
            --language-max-new-tokens=128 \
            --chat-tasks=SpellingBee \
            --chat-max-problems=128 \
            --chat-batch-size=8 \
            --device-type="$DEVICE_TYPE" \
            --run-id="d6-bi-han1-en60-zh40-sft-v4-spelling-step${SFT_EVAL_STEP}"
        ;;
    math-sft-prepare)
        python -m scripts.prepare_math_sft_data \
            --output-dir="$NANOCHAT_BASE_DIR/math_experiment/sft_v1" \
            --tokenizer-tag=bilingual-32k-han1-2b-v1 \
            --validation-rows=256 \
            --synthetic-per-language=3000
        ;;
    math-sft-baseline-eval)
        python -m scripts.eval_bilingual \
            -i sft \
            -g d6-bi-han1-en60-zh40-sft-v3 \
            -s 600 \
            --groups=chat \
            --chat-tasks=GSM8K \
            --chat-max-problems=128 \
            --chat-max-new-tokens=256 \
            --chat-batch-size=8 \
            --device-type="$DEVICE_TYPE" \
            --run-id=d6-bi-han1-en60-zh40-sft-v3-gsm8k-baseline
        ;;
    math-sft-train)
        python -m scripts.chat_sft \
            --device-type="$DEVICE_TYPE" \
            --model-source=sft \
            --model-tag=d6-bi-han1-en60-zh40-sft-v3 \
            --model-step=600 \
            --output-model-tag=d6-bi-han1-en60-zh40-math-sft-v1 \
            --load-optimizer=0 \
            --data-profile=custom-only \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=300 \
            --embedding-lr=0.15 \
            --unembedding-lr=0.004 \
            --matrix-lr=0.01 \
            --init-lr-frac=0.1 \
            --warmup-ratio=0.05 \
            --warmdown-ratio=0.5 \
            --final-lr-frac=0.0 \
            --eval-every=100 \
            --eval-tokens=131072 \
            --chatcore-every=-1 \
            --save-every=100 \
            --custom-train-jsonl="$NANOCHAT_BASE_DIR/math_experiment/sft_v1/train.jsonl" \
            --custom-val-jsonl="$NANOCHAT_BASE_DIR/math_experiment/sft_v1/val.jsonl" \
            --run=dummy
        ;;
    math-sft-eval)
        MATH_SFT_EVAL_STEP="${SCALE:-300}"
        python -m scripts.eval_bilingual \
            -i sft \
            -g d6-bi-han1-en60-zh40-math-sft-v1 \
            -s "$MATH_SFT_EVAL_STEP" \
            --groups=chat \
            --chat-tasks=GSM8K \
            --chat-max-problems=128 \
            --chat-max-new-tokens=256 \
            --chat-batch-size=8 \
            --device-type="$DEVICE_TYPE" \
            --run-id="d6-bi-han1-en60-zh40-math-sft-v1-gsm8k-step${MATH_SFT_EVAL_STEP}"
        ;;
    *)
        usage
        exit 2
        ;;
esac
