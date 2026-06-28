# NanoChat 全链路分析与优化路线

日期：2026-06-21

## 1. 项目目标

本项目不只是让一个小模型“能够聊天”，而是借助 NanoChat 建立一套可以
反复实验、准确归因的大模型训练方法：

```text
数据 -> tokenizer -> 预训练 -> base eval -> SFT -> chat eval
```

当前阶段的核心目标是：

1. 先确认每个环节是否正确、可复现、可比较。
2. 找到 d6 模型当前最主要的能力瓶颈。
3. 用短程消融选择训练方案，再投入长训练。
4. 同时提升中英文能力，避免用单项成绩掩盖整体退化。
5. 最终形成可复用于 d6、d8 和新 tokenizer 模型的实验闭环。

## 2. 当前项目总结

### 2.1 已跑通的完整流程

目前已经跑通：

```text
ClimbMix 数据下载
  -> 英文 tokenizer 训练
  -> d6 base 预训练
  -> base BPB / CORE / 固定生成
  -> 英文 SFT
  -> ChatCORE
  -> 中文 SFT
  -> 中英混合继续预训练
  -> CPT 后重新 SFT
  -> 中英文统一评测
```

主要产物位于：

```text
$NANOCHAT_BASE_DIR/
  base_data_climbmix/
  tokenizer/
  base_checkpoints/
  chatsft_checkpoints/
  zh_experiment/
  evaluations/
```

### 2.2 当前模型

```text
模型：d6
Transformer 层数：6
隐藏维度：384
注意力头：6
上下文长度：512
词表大小：32768
总参数：73,531,646
```

关键参数构成：

| 参数组 | 参数量 | 占比 |
|---|---:|---:|
| Token embedding | 12.58M | 17.1% |
| Value embeddings | 37.75M | 51.3% |
| LM head | 12.58M | 17.1% |
| Transformer matrices | 10.62M | 14.4% |

这个结构意味着总参数很多，但真正承担跨 token 计算和抽象变换的
Transformer matrices 只有约 10.62M。当前模型适合验证训练机制，不应把它
当作已经具有充分知识容量的通用助手。

### 2.3 关键 checkpoint

```text
base/d6/step5000
chatsft/d6/step1500
base/d6-zh-cpt/step1200
chatsft/d6-zh-sft-demo/step600
chatsft/d6-zh-cpt-sft-lrfix/step600
```

原始 base：

```text
训练 token：81.92M
scaling parameters：约 23.2M
token/scaling-parameter：约 3.53
英文验证 BPB：1.1653
```

脚本默认目标为 12 token/scaling-parameter，因此当前 base 最明确的问题是
欠训练。事实补全、简单推理和稳定生成也都表现较弱。

### 2.4 已确认的实验结论

1. 当前 tokenizer 可以无损编码中文，但通常需要每个汉字约 2-3 token。
2. 中文 SFT 可以把中文回答率提升到 100%，但回答仍可能空泛、重复或错误。
3. 中英混合 CPT 将纯中文验证 BPB 从 1.6263 降到 1.1463，说明中文语言建模
   确实增强。
4. BPB 改善没有自动转化为可靠知识、推理和问答质量。
5. 中文 SFT 会削弱原来过度强化的 SpellingBee；CPT 后再 SFT 可缓解部分遗忘。
6. 原始、中文 SFT、中文 CPT+SFT 的 ARC 和 MMLU 都接近随机区间。
7. GSM8K 和 HumanEval 都为 0，base 能力是比中文表达更基础的瓶颈。
8. ChatCORE 被 SpellingBee 明显主导，不能单独代表通用能力。
9. 训练来源、学习率和 optimizer 初始化方式必须显式记录；继承错误的低学习率
   已经造成过一次不可公平比较的实验。

## 3. 全链路优化原则

每轮实验遵守以下规则：

```text
一次只改变一个主要变量
固定父 checkpoint、数据、tokenizer 和评测集
用相同 token 或 FLOPs 比较训练
所有实验使用独立 tag
短程实验通过后才运行长训练
同时检查目标能力和非目标能力
```

模型晋级不能只看训练 loss 或综合分数。至少要同时满足：

- 目标验证集 BPB 有改善。
- 分项能力没有不可接受的退化。
- 生成重复、乱码和语言串扰没有恶化。
- manifest 能够说明模型由什么数据和配置产生。

## 4. 阶段一：数据

### 4.1 当前实现

预训练默认数据为 ClimbMix Parquet：

```text
nanochat/dataset.py
```

当前切分规则：

- 按文件名排序。
- 最后一个 shard 作为验证集。
- 其余 shard 作为训练集。

中文实验数据：

```text
SFT：shibing624/alpaca-zh
CPT 中文：FineWeb2 cmn_Hani
CPT 英文：现有 ClimbMix
```

已准备数据：

```text
中文 SFT train：45,518 行，约 17.24M 渲染 token
中文 SFT val：1,000 行
混合 CPT train：19.03M token
混合 CPT 比例：约 70% 中文 / 30% 英文
混合 CPT val：0.97M token
```

中文实验 manifest 已记录 seed、数据源、路径、SHA、文档数、字符数和 token 数。

### 4.2 当前问题

1. 原始 ClimbMix 只按 shard 划分，缺少内容级训练/验证去重证明。
2. 没有统一记录原始英文数据的 revision、下载时间和完整数据质量统计。
3. 中英文比例可以按生成后的 token 控制，但缺少统一的数据构建配置。
4. 还没有系统统计乱码、HTML、极短文档、重复文档和语言误判。
5. SFT 当前任务混合主要通过数据行重复或估算 token 实现，不是实际
   assistant loss token 调度。
6. 训练数据与 ARC、MMLU、GSM8K、HumanEval 等评测集的污染检查不足。

### 4.3 优化任务

第一步建立统一数据 manifest：

```text
dataset name / config / revision
license
downloaded_at
seed
source document count
accepted / rejected count
characters / bytes / tokenizer tokens
language distribution
length distribution
exact duplicate ratio
train / validation SHA
tokenizer SHA
```

预训练数据增加：

- 文档级 exact dedup。
- 可选 MinHash 近似去重。
- HTML、控制字符、乱码、极短文档过滤。
- 中英文语言标签和置信度。
- code、math、普通文本分类。
- 固定且不可变的英文、中文、混合验证集。

SFT 数据增加：

- role 顺序、空回答和截断后 target 检查。
- 每个任务的 conversation token、assistant loss token 和截断率统计。
- 训练集与验证集 exact/near dedup。
- 与标准评测集的污染检查。
- 实际累计 assistant loss token 占比日志。

### 4.4 验收门槛

数据进入训练前必须满足：

- train/val 没有 exact duplicate。
- 所有数据文件和 tokenizer 都有 SHA。
- 每个数据源有许可证和 revision。
- 训练配比按 token 报告，而不是只按行数报告。
- SFT 每个样本截断后至少保留一个 assistant target。
- 评测集没有直接进入训练 mixture。

## 5. 阶段二：Tokenizer

### 5.1 当前实现

入口：

```text
scripts/tok_train.py
scripts/tok_eval.py
nanochat/tokenizer.py
```

当前 tokenizer：

```text
训练数据：ClimbMix 英文
最大训练字符：2B
单文档上限：10,000 字符
词表：32,768
算法：RustBPE / tiktoken-compatible BPE
```

它包含 UTF-8 byte fallback，因此中文不会无法编码；问题是中文缺少合并后的
高频 token，编码效率很低。

### 5.2 当前问题

1. 中文 token/汉字约为 2-3，浪费上下文和训练计算。
2. 同一 512-token 窗口能容纳的中文信息远少于英文。
3. 32K 词表带来较大的 embedding、value embedding 和 LM head 参数成本。
4. `tok_eval.py` 主要测少量硬编码样本，缺少稳定的中英/code/math 测试集。
5. 当前 tokenizer 默认写入固定目录，训练候选时容易覆盖基线。
6. checkpoint 与 tokenizer 的兼容性目前主要靠实验纪律，而非强制校验。

### 5.3 优化任务

近期保持当前 tokenizer，不在 P1-P5 中更换，以便继续利用已有 checkpoint。

并行准备新 tokenizer 轨道：

```text
训练语料建议：
English 50%
Simplified Chinese 40%
code/math 10%

候选词表：
16K
24K
32K
```

每个候选使用独立 tag 和目录，离线评测：

- 英文 bytes/token。
- 中文 token/汉字、bytes/token。
- code/math/science bytes/token。
- 常用中英词、数字、标点和混合文本切分。
- 512/1024 token 可容纳的原始字符和字节数。
- UTF-8 roundtrip。
- 各类 token 在词表中的数量及训练频率。

### 5.4 验收门槛

候选 tokenizer 建议目标：

```text
中文平均 token/汉字 <= 1.5
英文 bytes/token 相对当前下降 <= 5%
代码压缩率不明显恶化
任意 UTF-8 样本均可无损往返
```

新 tokenizer 必须通过同架构、同数据、同 token 和同 FLOPs 的小预算从头
预训练，不能只凭压缩率决定替换。

## 6. 阶段三：预训练

### 6.1 当前实现

入口：

```text
scripts/base_train.py
```

当前已支持：

- 从零预训练。
- `--resume-from-step` 完整恢复训练。
- `--init-from-model-tag/step` 只加载权重并重置 optimizer 和 dataloader。
- `--data-dir` 使用独立 Parquet 数据。
- 独立 `--model-tag`。
- 中间 checkpoint。
- `--dry-run` 运行一次前向、反向和 optimizer step。
- 训练 manifest。
- 检查新实验是否会写入已有 checkpoint 目录。

### 6.2 当前问题

1. 原始 base 仅训练 81.92M token，明显低于当前脚本的推荐数据规模。
2. d6 的 Transformer 核心较小，知识和推理能力上限有限。
3. 当前英文继续预训练的最佳学习率还未确定。
4. 中英 CPT 只验证过 70% 中文 / 30% 英文，不能视为长期最优比例。
5. 训练 manifest 已开始实现，但尚未覆盖 SFT，且 tokenizer/checkpoint
   兼容性还未在加载时强制验证。
6. checkpoint 尚未完整记录所有 RNG 和恢复语义。

### 6.3 优化顺序

#### P1：英文 CPT 短程学习率消融

固定：

```text
父模型：base/d6/step5000
数据：原始 ClimbMix
训练步数：400
每步 token：16,384
fresh optimizer / fresh dataloader
```

比较原始预训练学习率的：

```text
0.10x
0.15x
0.20x
0.25x
```

先对四组执行 dry-run；再运行短程训练。比较 BPB 下降斜率、训练稳定性、
样本重复和小规模 CORE，而不是只选最终 loss 最低的一组。

#### P2：英文 base 增强

使用 P1 最佳学习率继续约 6,000 step：

```text
新增 token：约 98.3M
累计 token：约 180.2M
累计 token/scaling-parameter：约 7.8
```

建议每 1,000 step 保存并评测。若英文 BPB、CORE 和固定事实在后半程仍持续
改善，再决定是否向 12 token/scaling-parameter 靠近。

#### P3：中英文比例消融

从相同的增强英文 base 出发，使用相同 token 预算比较：

```text
100% English
90% English / 10% Chinese
70% English / 30% Chinese
50% English / 50% Chinese
```

选择中文 BPB、英文 BPB 和 CORE 的 Pareto 最优点，而不是只追求中文 BPB。

#### 后续：容量和上下文

只有在 d6 得到充分数据训练后仍然受限，才测试：

```text
d6 vs d8
512 vs 1024 context
当前 tokenizer vs 新中英 tokenizer
```

这些变量不能与数据比例或 SFT 配方在同一轮同时更换。

### 6.4 预训练晋级标准

短程候选进入长训练至少需要：

- 验证 BPB 稳定下降且无 loss spike。
- 生成没有新增乱码或严重循环。
- CORE 不低于父模型的可比结果。
- 非目标语言 BPB 退化在预设范围内。
- manifest、checkpoint 和评测结果完整。

中英 CPT 推荐门槛：

```text
中文 BPB 相对增强英文 base 至少下降 10%
英文 BPB 恶化不超过 1%
CORE 下降不超过 0.02
```

## 7. 阶段四：Base Eval

### 7.1 当前实现

原始入口：

```text
scripts/base_eval.py
```

已支持：

- train/val BPB。
- DCLM CORE。
- 固定提示 sample。
- 指定 checkpoint、step、数据目录和评测规模。

P0 新增统一入口：

```text
scripts/eval_bilingual.py
scripts/compare_evaluations.py
```

统一评测已经支持：

- 每次评测独立目录。
- manifest、状态和逐项结果。
- language、bpb、core、chat 分组。
- 失败后 resume。
- checkpoint、tokenizer、数据和 Git 信息。
- 多模型 JSON/Markdown 对比。

### 7.2 当前问题

1. 原始固定 sample 只能人工观察，缺少标准答案和自动评分。
2. 小样本 CORE 波动较大，不能过度解读几个百分点变化。
3. 当前统一评测 smoke 数据只验证工程闭环，还没有形成正式基线。
4. 中文 BPB、英文 BPB、混合 BPB 必须在固定数据上分别报告。
5. 需要区分知识不足、推理不足和解码循环。

### 7.3 优化任务

Base eval 固定为五组：

| 评测组 | 主要指标 |
|---|---|
| 英文 LM | 固定 ClimbMix val BPB |
| 中文 LM | 固定 FineWeb2 中文 val BPB |
| 混合 LM | 固定中英混合 val BPB |
| CORE | 每个子任务准确率、centered score、置信区间 |
| 固定能力集 | 事实、反义词、算术、日期、关系推理、重复率 |

正式基线至少评测：

```text
base/d6/step5000
base/d6-zh-cpt/step1200
P1 的四个短程 checkpoint
```

对准确率指标报告样本数和二项置信区间。小规模结果只用于筛选，不用于最终
能力结论。

### 7.4 晋级标准

base checkpoint 进入 SFT 前应满足：

- 英文、中文、混合 BPB 均有可追踪结果。
- CORE 分项没有明显整体退化。
- 固定事实/简单推理至少呈现可重复正向趋势。
- 重复率和异常字符不高于父模型。
- 两次确定性评测结果一致。

## 8. 阶段五：SFT

### 8.1 当前实现

入口：

```text
scripts/chat_sft.py
```

当前训练使用 assistant-only loss，并混合：

```text
SmolTalk
Identity
MMLU
GSM8K
SimpleSpelling
SpellingBee
可选自定义 JSONL
```

已支持：

- 指定 base checkpoint。
- fresh 或继承 optimizer。
- 显式学习率。
- 自定义 train/val JSONL。
- 自定义数据估算 token 比例。
- 独立输出 tag。
- 中间 checkpoint。

### 8.2 当前问题

1. 任务混合仍是静态 `TaskMixture`，实际 assistant loss token 比例不可控。
2. 拼写任务按行数已约占四分之一，按 assistant token 可能占比更高。
3. SFT validation BPB 是混合平均值，可能掩盖单任务退化。
4. MMLU/GSM8K 重复 epoch 会改变任务权重，但没有统一 manifest 解释。
5. SFT 默认可继承预训练学习率，来源 checkpoint 改变时容易形成不公平实验。
6. 当前中文数据能教会语言选择，但知识、推理和高质量解释覆盖不足。
7. 训练步数相同不等于 assistant target token 相同。

### 8.3 优化任务

#### P4：按 assistant loss token 重构 mixture

训练前统计每个任务：

```text
rows
conversation tokens
assistant loss tokens
average / p95 length
truncation rate
language distribution
```

训练时：

- 按 assistant loss token 设定目标占比。
- 设置单任务最大占比。
- 日志报告实际累计占比。
- 每个任务使用独立 validation 指标。

建议起始配比：

| 类型 | Assistant loss token |
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

该表只是初始假设，必须做消融。

#### P5：数据和学习率消融

固定同一个增强 base、相同 assistant token 预算：

```text
S0 当前配方
S1 降低 SpellingBee
S2 S1 + 合法数学指令
S3 S2 + 合法代码指令
S4 S3 + 中英平衡通用指令
```

在最佳配比上再比较：

```text
init LR fraction：0.1x / 0.2x / 0.4x
checkpoint：200 / 400 / 600 / 800 step
```

SFT manifest 应与预训练一致，记录父 checkpoint、数据 SHA、tokenizer SHA、
实际任务 token 比例和显式学习率。

### 8.4 晋级标准

SFT 模型进入正式 chat eval 前必须：

- 中文回答率和英文回答率都达到 95%。
- replacement character 为 0。
- 每个任务的实际 assistant token 比例接近目标。
- 通用能力不被某一专项任务完全主导。
- validation 各分项无明显异常退化。
- 固定样本中没有严重提示复述和无限重复。

## 9. 阶段六：Chat Eval

### 9.1 当前实现

入口：

```text
scripts/chat_eval.py
scripts/zh_eval.py
scripts/eval_bilingual.py
```

当前任务：

```text
ARC-Easy
ARC-Challenge
MMLU
GSM8K
HumanEval
SpellingBee
固定中英文语言提示
```

现有三组正式对比：

| Task | Original SFT | Chinese SFT | Chinese CPT+SFT |
|---|---:|---:|---:|
| ARC-Easy | 22.66% | 23.44% | 25.78% |
| ARC-Challenge | 24.22% | 25.78% | 26.56% |
| MMLU | 21.88% | 24.22% | 26.56% |
| GSM8K | 0% | 0% | 0% |
| HumanEval | 0% | 0% | 0% |
| SpellingBee | 91.41% | 65.62% | 75.78% |
| ChatCORE | 0.1385 | 0.1059 | 0.1350 |

### 9.2 当前问题

1. ChatCORE 被 SpellingBee 主导。
2. ARC/MMLU 接近随机时，小幅变化不能证明推理提升。
3. 中文固定提示主要测“是否输出中文”，不是事实正确性和回答质量。
4. GSM8K 和 HumanEval 为 0，需要检查能力、格式解析和评测协议三者。
5. 确定性解码可能放大重复，但采样不能修复知识错误。
6. 缺少统一的人评规则和错误类型统计。

### 9.3 优化任务

每次正式评测必须同时报告：

- 全部任务分项。
- 原 ChatCORE。
- 去掉 SpellingBee 的综合分。
- 中英文回答率和语言串扰。
- replacement character。
- 重复 n-gram、Distinct-2/3、提示复述率。
- 完整固定样本输出。

升级中文评测集：

```text
语言选择：50
常识事实：50
解释与总结：50
格式遵循：30
简单数学：20
拒答与边界：20
中英切换：30
```

每类提供标准答案、关键词、格式规则或 1-5 分人工评分说明。

对生成任务分别运行：

```text
确定性：temperature=0
稳健性：固定 seed 的少量采样
```

GSM8K 和 HumanEval 需要额外保存：

- 原始输出。
- 答案解析结果。
- 解析失败原因。
- 执行/测试失败原因。

这样才能判断是模型不会，还是输出格式未被评测器识别。

### 9.4 最终验收

近期候选模型建议满足：

```text
中文回答率 >= 95%
英文回答率 >= 95%
replacement character = 0
英文 BPB 不恶化超过 2%
ChatCORE 相对可比基线下降 <= 0.02
SpellingBee 相对基线下降不超过 10 个百分点
重复率不高于可比基线
```

对于新 tokenizer 从头训练的模型，必须按相同 token、相同 FLOPs和相同
wall-clock 三种口径分别比较。

## 10. 跨阶段基础设施

当前 P0 已完成：

- 独立评测目录。
- 统一评测 manifest。
- 分组运行和断点恢复。
- 多 checkpoint 对比报告。
- 预训练 dry-run。
- 预训练 manifest。
- 防止新实验覆盖已有 checkpoint。

下一步基础设施优先级：

1. 将训练 manifest 扩展到 SFT。
2. 加载 checkpoint 时校验 tokenizer SHA。
3. 统一 YAML/JSON 实验配置，减少长 CLI 复制错误。
4. 记录完整 RNG、optimizer、dataloader 和 loop state。
5. 每个 checkpoint 自动关联对应 metrics 和 sample。
6. 数据构建、训练、评测统一使用 run ID。
7. 增加正式基线的回归测试，不只做 8 题 smoke。

## 11. 推荐执行顺序

```text
P0 统一评测与预训练 manifest                  已完成
P1 英文 CPT 0.10x-0.25x 学习率短程实验        当前任务
P2 最佳学习率英文 CPT 约 6000 step
P3 中英文 CPT 比例消融
P4 SFT assistant loss token 配比重构
P5 数学、代码、通用中英 SFT 消融
P6 中英 tokenizer 16K/24K/32K 离线比较
P7 当前 tokenizer 与最佳中英 tokenizer 小预算对照
P8 最佳数据比例与 tokenizer 的完整预训练
P9 在数据充分后比较 d6/d8 和 512/1024 context
```

现在不应直接启动 6,000-step 长训练。当前最合理的下一步是：

1. 审查 P1 四组命令和 manifest。
2. 对 0.10x、0.15x、0.20x、0.25x 各做一次 dry-run。
3. 依次运行 400-step pilot，避免 M4 同时竞争内存。
4. 用统一评测器比较英文/中文/混合 BPB、CORE 和固定生成。
5. 选出最佳学习率后再制定 P2 长训练命令。

## 12. 如何判断真正进步

项目中需要始终区分三类能力：

```text
Tokenizer 效率：
同样文本需要多少 token。

Base 能力：
模型是否学会语言规律、知识和可迁移表示。

SFT 行为：
模型是否按要求组织和输出已有能力。
```

因此：

- “能输出中文”主要证明 tokenizer 可表示中文且 SFT 学会语言选择。
- “中文 BPB 降低”证明中文语言建模增强。
- “中文问题答对更多”才证明知识、推理和指令能力提升。
- “某个专项分数很高”不等于通用能力强。
- “综合分上涨”必须检查是否由单一任务贡献。

最终成功不是某次训练得到一个更高分，而是可以稳定回答：

```text
改了什么？
为什么预计有效？
哪个指标验证了它？
付出了什么代价？
下一轮只需要再改变哪个变量？
```

