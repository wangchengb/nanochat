# NanoChat Math SFT 小实验

## 目标

在不覆盖当前最佳双语 SFT checkpoint 的前提下，验证独立 Math SFT
能否让 d6 模型在 GSM8K 上获得可测的非零结果。

## 实验设置

- 起点：`d6-bi-han1-en60-zh40-sft-v3/step600`
- 输出：`d6-bi-han1-en60-zh40-math-sft-v1`
- 设备：MPS，float32
- 训练步数：300
- 每步 token positions：16,384
- 数据模式：`custom-only`
- 数据：7,217 条 GSM8K train、3,000 条英文基础算术、3,000 条中文基础算术
- 验证集：从 GSM8K train 固定留出 256 条；官方 test 未用于训练或验证
- 答案格式：统一以 `#### number` 结尾

数据 manifest 位于：

```text
~/.cache/nanochat/math_experiment/sft_v1/manifest.json
```

## 训练结果

| Step | Validation BPB |
|---:|---:|
| 0 | 1.1697 |
| 100 | 0.6880 |
| 200 | 0.6236 |
| 300 | 0.6264 |

step200 的验证 BPB 最低，因此选择它进行 GSM8K 对比评测。

## GSM8K 结果

评测固定使用官方 test 前 128 题、temperature 0、最多生成 256 token。

| Checkpoint | Correct | Accuracy |
|---|---:|---:|
| `sft-v3/step600` | 1/128 | 0.78% |
| `math-sft-v1/step200` | 3/128 | 2.34% |

表面上正确数提升了 3 倍，但逐题检查发现三道命中题的推理过程均不正确：

- 出现 `10/2 = 55`、`4*8 = 1616`、`1+1 = 33` 等数字重复或拼接错误。
- 三题只是最终 `####` 后的数字碰巧与标准答案一致。
- 因此当前模型的可靠 GSM8K 推理准确率仍应视为接近 0%。

完整命中记录位于：

```text
~/.cache/nanochat/evaluations/
  d6-bi-han1-en60-zh40-math-sft-v1-gsm8k-step200/gsm8k_hits.jsonl
```

## 工程结论

- Math SFT 明显降低了数学验证 BPB，模型学会了题目语言、答案格式和部分数量模式。
- 仅看最终数值准确率会高估真实推理能力，后续评测应同时检查计算过程的一致性。
- 73M 级 d6 模型仍难以稳定完成多步文字题；下一轮应先加入可自动校验的基础算术课程，
  再逐级增加两步和多步题目，而不是直接重复更多 GSM8K。
- 当前通用主 checkpoint 仍保持 `d6-bi-han1-en60-zh40-sft-v3/step600`；
  Math SFT checkpoint 仅作为独立实验分支，不直接替换通用模型。

## 复现命令

```bash
bash runs/run_bilingual_optimization.sh math-sft-prepare
bash runs/run_bilingual_optimization.sh math-sft-baseline-eval
bash runs/run_bilingual_optimization.sh math-sft-train
bash runs/run_bilingual_optimization.sh math-sft-eval 200
```
