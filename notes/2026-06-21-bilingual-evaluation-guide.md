# NanoChat 统一中英评测指南

日期：2026-06-21

## 目标

`scripts/eval_bilingual.py` 将原来分散的评测整合到同一个可恢复入口，并为
每次运行建立独立目录。它不会替代或改变原有的：

```text
scripts/base_eval.py
scripts/chat_eval.py
scripts/zh_eval.py
```

每次运行输出：

```text
$NANOCHAT_BASE_DIR/evaluations/<run-id>/
  manifest.json
  status.json
  metrics/
    language.json
    chat.json
    bpb.json
    core.json
  summary.md
```

不存在的评测组不会生成对应文件。

## Manifest 内容

`manifest.json`记录：

- 完整命令。
- Git commit、branch 和 dirty 状态。
- Python、PyTorch 和系统信息。
- model source、tag、step。
- checkpoint路径和SHA256。
- checkpoint完整meta。
- tokenizer路径、词表大小和SHA256。
- 中文实验数据manifest路径和SHA256。
- 全部评测参数。

模型权重和数据不会复制到评测目录。

## 快速冒烟测试

只运行少量ARC-Easy题目：

```bash
python -m scripts.eval_bilingual \
  -i sft \
  -g d6-zh-cpt-sft-lrfix \
  -s 600 \
  --groups=chat \
  --chat-tasks=ARC-Easy \
  --chat-max-problems=8 \
  --device-type=mps \
  --run-id=smoke-d6-zh-cpt-sft-lrfix
```

## 标准Chat评测

```bash
python -m scripts.eval_bilingual \
  -i sft \
  -g d6-zh-cpt-sft-lrfix \
  -s 600 \
  --groups=language,chat \
  --chat-max-problems=128 \
  --language-max-new-tokens=128 \
  --device-type=mps
```

默认Chat任务：

```text
ARC-Easy
ARC-Challenge
MMLU
GSM8K
HumanEval
SpellingBee
```

每个任务结束后立即更新`metrics/chat.json`。HumanEval等单项失败不会丢失
已经完成的其他任务。

## Base评测

```bash
python -m scripts.eval_bilingual \
  -i base \
  -g d6 \
  -s 5000 \
  --groups=bpb,core \
  --bpb-split-tokens=131072 \
  --core-max-per-task=128 \
  --device-type=mps
```

未显式传入`--bpb-dataset`时，会评测本地存在的：

```text
english=$NANOCHAT_BASE_DIR/base_data_climbmix
chinese=$NANOCHAT_BASE_DIR/zh_experiment/pretrain_zh_eval
mixed=$NANOCHAT_BASE_DIR/zh_experiment/pretrain
```

也可以显式指定：

```bash
python -m scripts.eval_bilingual \
  -i base -g d6 -s 5000 \
  --groups=bpb \
  --bpb-dataset=english="$HOME/.cache/nanochat/base_data_climbmix" \
  --bpb-dataset=chinese="$HOME/.cache/nanochat/zh_experiment/pretrain_zh_eval"
```

## 断点续跑

指定同一个`--run-id`并增加`--resume`：

```bash
python -m scripts.eval_bilingual \
  -i sft \
  -g d6-zh-cpt-sft-lrfix \
  -s 600 \
  --groups=language,chat \
  --run-id=my-eval-run \
  --resume \
  --device-type=mps
```

已完成评测组和Chat子任务会跳过；失败或未完成任务会重试。

## 生成对比报告

```bash
python -m scripts.compare_evaluations \
  "$HOME/.cache/nanochat/evaluations/run-original" \
  "$HOME/.cache/nanochat/evaluations/run-zh-sft" \
  "$HOME/.cache/nanochat/evaluations/run-zh-cpt-sft"
```

输出一个独立的`comparison-<timestamp>`目录：

```text
comparison.json
comparison.md
```

对比报告包含：

- 中英文回答率。
- UTF-8 replacement character数量。
- 中文Distinct-2和提示复述数量。
- ChatCORE全部子任务。
- ChatCORE与去SpellingBee版本。
- 各数据集验证BPB。
- Base CORE。

## 结果解释

`partial_chatcore`只代表当前已经完成的任务，不应与完整ChatCORE直接比较。
只有六个默认任务全部完成时才会产生：

```text
chatcore
chatcore_without_spellingbee
```

固定中文提示目前主要衡量语言控制和生成稳定性，不代表完整中文知识评测。

## 失败处理

失败信息记录在：

```text
status.json
metrics/<group>.json
summary.md
```

运行存在失败项时进程返回非零状态，但已完成结果会保留。修复环境问题后使用
`--resume`继续即可。
