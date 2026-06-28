# NanoChat 中英文混合模型持续优化计划

日期：2026-06-21

## 1. 总目标

建立一套可以持续迭代、稳定归因、避免中英文能力相互破坏的 NanoChat
训练流程，使模型在相同硬件预算下逐步提升：

1. 英文基础语言建模、常识和指令能力。
2. 中文语言建模、中文指令理解和回答质量。
3. 数学、代码、工具调用等专项能力。
4. 中英文混合训练后的能力保持与灾难性遗忘控制。
5. 数据、tokenizer、预训练、SFT、评测和推理全链路的可复现性。

这里的“持续提升”不是某一个指标上涨，而是每个新模型必须满足：

```text
主要目标指标提升
+ 非目标语言不明显退化
+ 生成稳定性不变差
+ 训练过程可复现
+ 能够解释提升来自哪个变量
```

## 2. 当前状态

### 2.1 模型与硬件

```text
设备：Apple M4，16 GB
模型：d6
层数：6
隐藏维度：384
注意力头：6
上下文：512
词表：32768
总参数：73,531,646
Transformer matrices：约 10.62M
```

现有关键 checkpoint：

```text
base/d6/step5000
chatsft/d6/step1500
base/d6-zh-cpt/step1200
chatsft/d6-zh-sft-demo/step600
chatsft/d6-zh-cpt-sft/step600
chatsft/d6-zh-cpt-sft-lrfix/step600
```

### 2.2 已确认结论

1. 现有 tokenizer 可以无损表示中文，但中文通常需要每字 2-3 token。
2. 中文 SFT 可以将中文回答率提升到 100%，但不能单独建立可靠中文知识。
3. 中英混合 CPT 将纯中文验证 BPB 从 1.6263 降到 1.1463。
4. CPT 的语言建模收益尚未明显转化为更好的中文问答语义。
5. SFT 必须显式控制学习率，不能依赖不同来源 checkpoint 的自动继承。
6. 直接中文 SFT 会削弱 SpellingBee；CPT 后再 SFT 可以缓解部分遗忘。
7. ARC 和 MMLU 仍接近随机基线，GSM8K 和 HumanEval 仍为 0。
8. 原始预训练只有约 81.92M token，约为 3.53 token/scaling-param，
   低于脚本目标 12。
9. 当前 SFT 数据按训练行数看约四分之一是拼写任务；按 assistant loss
   token 计算，SpellingBee 的实际影响可能更高。

完整证据见：

```text
notes/2026-06-20-zh-training-experiment-summary.md
notes/2026-06-20-zh-experiment-guide.md
```

### 2.3 当前代码能力

当前分支已经支持：

- 自定义预训练 Parquet 目录。
- 从指定 base checkpoint 仅加载模型权重，重置 optimizer 和 dataloader。
- 自定义 SFT train/val JSONL。
- 按估算 token 比例混入自定义 SFT 数据。
- 独立输出 tag 和中间 checkpoint。
- CLI 累积 UTF-8 解码。
- 固定中英文提示语言评测。
- 中文数据 manifest、数据哈希和 tokenizer SHA。

## 3. 核心研发原则

### 3.1 一次只改变一个主要变量

每轮实验只能有一个主要问题：

- 数据量是否不足？
- 中英文比例是否不合理？
- tokenizer 是否限制中文效率？
- SFT 配比是否导致偏科？
- 学习率是否不匹配？
- 模型容量是否不足？

禁止同时更换 tokenizer、模型深度、数据比例和 SFT 配方后直接比较。

### 3.2 保留不可变基线

以下 checkpoint 永不覆盖：

```text
base/d6/step5000
chatsft/d6/step1500
chatsft/d6-zh-sft-demo/step600
base/d6-zh-cpt/step1200
chatsft/d6-zh-cpt-sft-lrfix/step600
```

所有新实验使用独立 tag，并在 manifest 中记录父 checkpoint。

### 3.3 同时报告分项和综合指标

ChatCORE 不能单独作为通用能力结论，因为当前结果主要由 SpellingBee
贡献。每次必须报告：

- 每个子任务原始准确率。
- 去掉 SpellingBee 后的综合结果。
- 中文与英文分别的 BPB。
- 中英文回答率。
- 重复、提示复述和非法字符指标。
- 固定提示的完整输出。

### 3.4 训练比较必须同条件

公平对比必须固定：

- tokenizer 和 tokenizer SHA。
- 父 checkpoint。
- 数据 revision、seed 和数据顺序。
- 总训练 token。
- batch size、context、optimizer 和学习率计划。
- SFT 数据 token 比例。
- 评测样本、temperature、top-k 和 max tokens。

## 4. 统一评测体系

在继续训练前，先建立统一评测入口。所有 checkpoint 都输出一份 JSON
结果，不再依赖会被覆盖的 Markdown 报告。

### 4.1 Base 模型评测

每个 base checkpoint 评测：

| 类别 | 指标 |
|---|---|
| 英文语言建模 | ClimbMix 固定验证集 BPB |
| 中文语言建模 | FineWeb2 中文固定验证集 BPB |
| 混合语言建模 | 固定中英混合验证集 BPB |
| 基础能力 | CORE 全部分项和 centered score |
| 事实补全 | 固定英文事实提示准确率 |
| 简单推理 | 反义词、日期、算术、关系推理 |
| 稳定性 | 重复 n-gram、有效终止、异常字符 |

### 4.2 Chat 模型评测

每个 SFT checkpoint 评测：

| 类别 | 指标 |
|---|---|
| 语言控制 | 中文回答率、英文回答率、语言串扰 |
| 中文质量 | 固定问答集、事实正确率、人工评分 |
| 英文能力 | ARC-Easy、ARC-Challenge、MMLU |
| 数学 | GSM8K |
| 代码 | HumanEval 或独立合法代码验证集 |
| 拼写 | SpellingBee |
| 综合 | ChatCORE、去 SpellingBee ChatCORE |
| 生成质量 | 重复率、提示复述率、平均有效长度 |

### 4.3 中文固定评测集需要升级

当前 100 条中文提示主要测语言选择。下一版拆成：

```text
语言选择：50 条
常识事实：50 条
解释与总结：50 条
格式遵循：30 条
简单数学：20 条
拒答和安全边界：20 条
中英切换：30 条
```

要求保存标准答案或评分规则。不能只看 CJK 字符比例。

### 4.4 评测门槛

任何候选模型进入下一阶段前至少满足：

```text
中文回答率 >= 95%
英文回答率 >= 95%
replacement character = 0
ChatCORE 相对当前可比基线下降 <= 0.02
SpellingBee 不低于可比基线 10 个百分点以上
英文验证 BPB 不恶化超过 2%
重复率不高于可比基线
```

对于新 tokenizer 从头训练的模型，不与旧模型使用训练 step 对齐，而使用：

- 相同训练 FLOPs。
- 相同训练 token。
- 相同 wall-clock 预算。

分别做三种口径比较。

## 5. 轨道 A：保留现有 Tokenizer 的近期优化

目标：用最低改动找出当前模型最主要的性能限制，并完善训练闭环。

### 阶段 A0：评测与记录基础设施

任务：

1. 新增统一评测脚本，例如 `scripts/eval_bilingual.py`。
2. 每次运行写独立 JSON，不覆盖旧报告。
3. 记录模型 tag、step、git SHA、tokenizer SHA、manifest SHA 和命令行。
4. 汇总生成 CSV/Markdown 对比表。
5. 增加重复率、prompt copy、语言串扰和非法字符指标。
6. 允许只运行某个评测组，避免 M4 上每次全量运行。

验收：

- 对三个已有模型重跑后能复现现有主要指标。
- 同一 checkpoint 重复两次确定性评测结果一致。
- 评测失败时保留已完成子任务结果。

### 阶段 A1：增强原始英文 Base

假设：当前通用能力差的第一原因是 base 欠训练，而不是 SFT step 不足。

实验：

```text
父模型：base/d6/step5000
数据：原始英文 ClimbMix
fresh optimizer
新 tag：d6-en-cpt
第一阶段增加约 98M token，约 6000 step
每 1000 step 保存
```

学习率：

- 不恢复已经 warmdown 的 optimizer。
- 从原始预训练学习率的 10%-25% 做小规模 LR range test。
- 先用 200-400 step 比较稳定性和 BPB 斜率，再确定完整训练 LR。

每个 checkpoint 测：

- 英文 val BPB。
- CORE 小样本。
- 固定事实与简单推理。
- 中文 val BPB，监控非目标语言变化。

停止条件：

- 英文 val BPB 连续两个 checkpoint 无改善。
- CORE 和固定事实没有任何正向趋势。
- 明显过拟合或生成重复恶化。

验收：

- 英文 val BPB 明显低于原始 d6。
- 固定英文事实补全和简单推理有可重复提升。
- CORE 不低于原始 base。

### 阶段 A2：中英文 CPT 比例消融

从同一个增强英文 base 出发，训练相同 token 预算：

```text
E0：100% English
E1：90% English / 10% Chinese
E2：70% English / 30% Chinese
E3：50% English / 50% Chinese
```

注意：之前的 70% 中文 / 30% 英文实验用于快速证明中文可学习，不应直接当作
长期最优比例。

每组先训练 10M-20M token，比较：

- 中文 BPB 改善速度。
- 英文 BPB 退化速度。
- CORE 变化。
- 语言生成稳定性。

选择 Pareto 最优比例，而不是只选中文 BPB 最低的模型。

推荐判定：

```text
中文 BPB 相对增强英文 base 至少下降 10%
英文 BPB 恶化不超过 1%
CORE 不下降超过 0.02
```

### 阶段 A3：重构 SFT 数据配比

问题：当前 SFT 按数据行混合，无法控制实际 assistant loss token；长
SpellingBee 回答会产生不成比例的影响。

需要实现：

1. 数据准备阶段统计每个任务的：
   - conversation token。
   - assistant loss token。
   - 平均长度。
   - 截断率。
2. SFT 按 assistant loss token 配比采样。
3. 每个任务设置最大 token 占比。
4. 训练日志输出实际累计任务 token 占比。

建议初始配比：

| 类型 | Assistant loss token 目标 |
|---|---:|
| 通用英文对话 | 35% |
| 通用中文对话 | 25% |
| 英文知识/选择题 | 10% |
| 中文知识/解释 | 10% |
| 数学 | 8% |
| 代码 | 5% |
| 工具调用 | 3% |
| 拼写与字符任务 | 3% |
| 身份与格式 | 1% |

这只是起始假设，必须通过消融验证。

关键实验：

```text
S0：当前配方
S1：SpellingBee 降到 5%
S2：S1 + 数学数据
S3：S2 + 合法代码指令数据
S4：S3 + 中英平衡通用指令
```

每组使用同一 base、相同 SFT token 和相同 LR。

### 阶段 A4：SFT 学习率与训练长度消融

固定最佳数据配比，测试：

```text
LR：0.1x、0.2x、0.4x initial fraction
Steps：200、400、600、800
```

保存所有 checkpoint，不以最低 SFT BPB 自动选择模型。

选择依据：

- 中英文任务分数。
- 重复率。
- 语言串扰。
- 通用能力保持。
- 人工样例质量。

### 阶段 A5：解码与重复控制

这一阶段不改变模型权重，只研究推理策略：

- temperature 0、0.2、0.5、0.8。
- top-k 20、50、100。
- repetition penalty 或重复 n-gram 阻断。
- EOS/assistant_end 学习和停止行为。

目标是区分：

```text
模型能力问题
vs.
确定性解码放大的循环问题
```

解码优化不能掩盖事实错误，只能作为独立报告项。

## 6. 轨道 B：中英 Tokenizer 与从头预训练

目标：解决中文 token 效率低、上下文利用率低和词表参数利用不均衡问题。

轨道 B 在轨道 A 的评测与数据统计基础设施完成后启动。

### 阶段 B0：Tokenizer 数据与候选方案

准备固定 tokenizer 训练语料：

```text
English：50%
Simplified Chinese：40%
code/math：10%
```

比例按字符/字节采样和按文档采样都要记录，避免英文长文档或中文 token
膨胀导致实际比例失真。

候选词表：

```text
16K
24K
32K
```

候选算法保持 NanoChat 现有实现优先，避免同时改变过多变量。

### 阶段 B1：Tokenizer 离线评测

每个候选 tokenizer 测：

- 英文 bytes/token。
- 中文字符/token 和 bytes/token。
- code、math、science 压缩率。
- 单字、常用词、专有名词的切分。
- 512 和 1024 token 可容纳的有效文本长度。
- 词表中英文、中文、字节回退 token 的分布。

目标建议：

```text
中文平均 token/汉字 <= 1.5
英文压缩率相对当前 tokenizer 下降 <= 5%
代码压缩率不明显恶化
所有 UTF-8 文本可无损往返
```

### 阶段 B2：小预算从头预训练对照

固定 d6 架构，分别训练：

```text
T0：当前 tokenizer
T1：最佳中英 tokenizer
```

要求：

- 相同数据。
- 相同 token 数。
- 相同 FLOPs。
- 相同 optimizer。
- 相同 checkpoint 间隔。

同时额外报告“看到的原始字符/字节总量”，因为相同 token 预算下新 tokenizer
会覆盖更多中文原文。

若 T1 在中文 BPB、英文 BPB和生成稳定性上形成 Pareto 改善，再扩大预算。

### 阶段 B3：完整中英 Base 预训练

使用阶段 A2 选出的中英文数据比例和阶段 B1 选出的 tokenizer。

训练预算分级：

```text
Smoke：5M token
Pilot：25M token
Main：100M-300M token
```

每级必须通过评测门槛后才能进入下一级。

不要一开始直接运行长训练；先验证：

- loss 是否稳定。
- checkpoint 是否可恢复。
- manifest 是否正确。
- 中英文 BPB 是否同步下降。
- 样本是否没有乱码和异常循环。

### 阶段 B4：架构容量实验

只有在 d6 数据充分训练后仍明显受限时再比较：

```text
d6
d8
d10
```

主要关注：

- Transformer matrices 占总参数比例。
- value embeddings 的收益与成本。
- 词表大小对 embedding/lm_head 参数的影响。
- 512 与 1024 context 的实际收益。

模型比较必须以相同 FLOPs 为主，不能只比较 step。

## 7. 数据治理任务

### 7.1 预训练数据

每个数据源必须记录：

- 数据集名称、config 和 revision。
- 许可证与使用限制。
- 下载日期。
- 文档数、字符数、token 数。
- 语言检测结果。
- 去重比例。
- 长度分布。
- 训练/验证切分方式和 SHA。

需要增加：

- 文档级 exact dedup。
- 可选 MinHash/近似去重。
- HTML、乱码、极短文档过滤。
- 中英文语言置信度过滤。
- code/math 单独分类和比例控制。

### 7.2 SFT 数据

每个样本检查：

- role 是否严格交替。
- assistant 是否非空。
- 截断后是否仍有 assistant target。
- 是否包含重复模板。
- 是否与验证集重叠。
- 是否包含测试集答案泄漏。
- 中英文语言标签是否正确。

严禁把 ARC、MMLU、GSM8K、HumanEval 的 test split 加入训练。

## 8. 训练基础设施优化

优先任务：

1. 统一 experiment config，减少长 CLI 复制错误。
2. 每次训练自动写 manifest：
   - git SHA。
   - 完整参数。
   - 父 checkpoint。
   - 数据 SHA。
   - tokenizer SHA。
   - 环境和 PyTorch 版本。
3. 自动生成独立 report 路径，禁止覆盖上次评测。
4. checkpoint 保存：
   - model。
   - optimizer。
   - dataloader state。
   - RNG state。
   - loop state。
5. 增加 dry-run：
   - 只加载数据。
   - 只构建一个 batch。
   - 只运行一个 optimizer step。
6. 增加训练前校验：
   - 输出 tag 不得覆盖已有模型。
   - 数据至少包含 train 和 val shard。
   - tokenizer SHA 与 checkpoint 相符。
   - 继承 LR 时打印醒目警告。

## 9. 实验命名规范

建议：

```text
base/d6-en-cpt-r1
base/d6-bi-cpt-en70-zh30-r1
base/d6-bivocab24k-pretrain-r1
chatsft/d6-bi-sft-mix01-r1
```

每个实验目录附带：

```text
manifest.json
metrics/*.json
samples/*.json
notes.md
model_*.pt
optim_*.pt
```

失败实验也保留 manifest 和失败原因，不必永久保留所有大 checkpoint。

## 10. Checkpoint 保留策略

M4 本地磁盘有限，建议：

- 永久保留所有基线和最终候选模型。
- 每次实验保留最佳、最后和一个中间 checkpoint。
- 明确失败且不再分析的 optimizer 文件可以删除。
- 删除前将 meta、metrics、sample 和命令归档进 Git。
- 不把模型权重、数据集或 Hugging Face cache 提交到 Git。

## 11. 决策树

### 情况 A：增加英文预训练后 CORE 提升

说明主要瓶颈是 base 欠训练。继续扩大中英混合预训练，再优化 SFT。

### 情况 B：BPB下降但 CORE和事实能力不提升

说明模型容量、数据质量或评测任务需要进一步分析。先检查数据，再考虑 d8。

### 情况 C：中文 BPB下降但中文问答不提升

说明缺少中文知识/指令数据、tokenizer 效率或模型容量。比较新 tokenizer
从头预训练，不继续盲目增加相同 SFT step。

### 情况 D：中文提升但英文持续退化

降低中文 CPT 比例，引入英文 replay，减少 SFT 参数更新强度，并检查
assistant-token 配比。

### 情况 E：专项任务很强但其他任务接近随机

这是数据偏科，不是通用能力。降低该专项 token 占比，增加覆盖面并按分项指标
选 checkpoint。

## 12. 建议执行顺序

按照以下顺序推进，避免过早进入成本更高的新 tokenizer 训练：

```text
P0 统一评测输出与实验 manifest（已完成）
P1 英文 base 继续预训练小规模 LR 实验
P2 英文 base 继续预训练约 6000 step
P3 中英文 CPT 比例消融
P4 SFT assistant-token 配比统计与重构
P5 数学、代码、通用中英 SFT 消融
P6 中英 tokenizer 候选训练和离线评测
P7 当前 tokenizer vs 中英 tokenizer 小预算从头预训练
P8 最佳 tokenizer + 最佳数据比例完整预训练
P9 d6/d8 容量比较
```

## 13. P0 完成状态与下一线程任务

P0 已于 2026-06-21 完成：

- 新增 `scripts/eval_bilingual.py`。
- 新增 `scripts/compare_evaluations.py`。
- 每次评测使用独立目录，不再覆盖旧报告。
- 自动记录 checkpoint、tokenizer、数据 manifest 和 Git SHA。
- 支持 language、chat、bpb、core 评测组。
- Chat任务和BPB数据集逐项持久化，失败后可使用 `--resume`。
- 增加中英文回答率、Distinct-2/3、提示复述和异常字符指标。
- 已对三个关键SFT checkpoint完成8题ARC-Easy smoke验证并生成统一对比表。

Smoke结果只用于验证工程闭环，不能作为模型性能结论。

下一线程不要立即运行完整长训练。先执行 P1 的学习率小实验：

> 从 `base/d6/step5000` 仅加载权重并重置 optimizer/data state，使用原始
> 英文 ClimbMix 数据，对原始预训练学习率的 0.10x、0.15x、0.20x、0.25x
> 做 200-400 step 短程对照。每组保存独立 tag，并使用统一评测器比较英文
> BPB、中文 BPB、CORE小样本、固定事实提示和重复率。根据 BPB下降斜率和
> 稳定性选择后续约6000 step英文继续预训练的学习率。

建议新线程首先读取：

```text
notes/2026-06-20-zh-training-experiment-summary.md
notes/2026-06-21-bilingual-model-improvement-plan.md
notes/2026-06-21-bilingual-evaluation-guide.md
runs/runzh.sh
scripts/eval_bilingual.py
scripts/compare_evaluations.py
scripts/zh_eval.py
scripts/base_eval.py
scripts/chat_eval.py
```

建议新线程提示词：

```text
请先读取 notes/2026-06-20-zh-training-experiment-summary.md、
notes/2026-06-21-bilingual-model-improvement-plan.md 和
notes/2026-06-21-bilingual-evaluation-guide.md。P0已经完成，请从P1开始：
设计并实现英文base继续预训练的短程学习率消融。先审查base_train.py的
学习率缩放、warmup/warmdown和weight-only初始化语义，为0.10x、0.15x、
0.20x、0.25x四组实验建立独立配置与tag；先做dry-run和200-400 step实验，
不要直接启动6000 step长训练。使用eval_bilingual.py生成可比较结果。
```

## 14. 阶段性成功标准

近期成功：

- 评测和训练产物不再互相覆盖。
- 每个实验可通过 manifest 完整复现。
- 找到比当前 d6 更好的英文 base。
- 找到中英文 BPB 的 Pareto 最优 CPT 比例。
- SFT 不再由 SpellingBee 单项主导。

中期成功：

- 中文问答不仅语言正确，而且事实和解释质量明显改善。
- 英文 BPB、ChatCORE 和固定事实能力不低于当前基线。
- GSM8K 或代码任务至少一个从 0 提升到稳定非零。
- 中英 tokenizer 在中文压缩率上显著改善，同时英文损失可控。

最终成功：

- 在固定硬件预算下，中英两种语言的综合能力持续形成 Pareto 改善。
- 每一次能力变化都能追溯到数据、tokenizer、训练或解码中的明确变量。
- 形成一套可以重复用于 d6、d8 和后续模型的完整训练方法。
