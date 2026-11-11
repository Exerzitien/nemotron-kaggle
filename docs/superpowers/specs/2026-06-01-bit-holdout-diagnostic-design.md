# Bit 记忆隔离诊断仪 — 设计文档

**日期**: 2026-06-01
**状态**: 待用户审阅
**背景会话**: 分析 Taha "0.86 custom repo" notebook + V38≡amr 逐字节复刻发现

---

## 1. 问题陈述

我们卡在 LB 0.85（V34，#53080019）已久。复刻了多个 0.86 notebook（V36/V37/V38/V39），
结果全部更差或翻车。本会话查清的决定性事实：

- **V38 = amr "0.86" notebook 逐字节相同**（14 个 code cell `diff` 为空，6 个 dataSource
  `databundleVersionId` 全等），却只跑出 0.84。逐字节相同的输入不可能产出能力质不同的模型。
- **V36(0.38)/V37(0.44)/V39(0.77) 三次翻车，同一模式：val_950 不塌、LB 塌。**
- 根因不是缺杠杆，是**测量信号被记忆污染**：val_950 的题模型训练时背过
  （dgxchen 风格 64.8% / huikang 风格 88.3% 的题在 train 内），所以它对"风格/格式类改动"
  完全免疫。V39 那次模型学会无脑盖"✓"橡皮图章，val_950 看不出，LB −8 点。
- **huikang 本人**给 Taha（同样卡 0.86）的唯一建议：train on 90% / validate on 10%，
  或自造训练集、在全集验证——即"你 overfit 了，你测不准"。和我们的病一模一样。

**结论**：在装上可信测量之前，任何"风格类"深挖都是蒙眼下注。先造一台
**对记忆免疫的诊断仪**，再用它闸控对 bit（最大真杠杆）的深挖。

## 2. 目标与非目标

**目标**：
- 生成模型**证明性地从未训练过**的全新 bit 题，得到 V34 在私测同分布上的真实泛化准确率。
- 区分两种泛化能力：**规则应用泛化**（学过的规则用到新输入）vs **规则归纳泛化**
  （从样例归纳出没见过的规则）——后者才是私测真正考的。
- 用读数（提交结果前先写死阈值）判定 bit 93.5% 是真推理还是记忆，决定是否值得深挖 bit CoT。

**非目标**（本轮不做）：
- 不动训练、不烧训练配额（只 1 次 eval）。
- 不改 CoT、不生成训练数据、不提交 LB。
- 不碰 bit 以外的类（拿到读数后再议）。
- 不追求"刷高某个本地分"——本诊断仪的价值在**判别力**，不在分数高低。

## 3. 架构：三步管线

```
① bit_holdout_gen.py（本地，0 GPU）
     ├─ Layer A「规则见过·输入全新」~100 题
     │     从 research/majortom_results.csv 的 1602 条已解规则抽样，
     │     灌全新随机 8-bit 输入（样例 + 查询），用 solver 的 predictor 算答案。
     │     测「学过的规则能否用到新输入」= 应用泛化。
     └─ Layer B「规则全新·没见过」~100 题
           用 majortom_bit_solver 的语法生成 train 中不存在的 (expr, trans) 组合，
           灌随机输入，predictor 算答案。
           测「能否从样例归纳出没见过的规则」= 归纳泛化（私测真正考的）。
                         │
② Kaggle dataset: exerzitien/nemotron-bit-holdout（上传两层 CSV）
                         │
③ eval-v39 kernel 变体（换 df 数据源 → 我们的留出集，adapter 指向 V34）
     跑 1 次 → 每层准确率
```

## 4. 组件详述

### 4.1 `datagen/bit_holdout_gen.py`（新建）

**输入**：
- `research/majortom_results.csv`（现成，1602 条规则：`expr` 布尔函数 + `trans` 位变换 JSON）
- `research/train_with_type.csv`（现成，1602 道真实 bit 题，用于真值比对 + 唯一性过滤的参照）

**复用**：`research/majortom_bit_solver.py` 的 `get_source_bit`、`OPS`、语法生成器、
`parse_bit_problem`、predictor 构造逻辑（import，不重写）。

**Layer A 生成**（规则见过·输入全新）：
1. 从 majortom_results.csv 抽 ~100 条规则（按 nesting-depth 分布抽样，
   覆盖 depth 0/1/2/3/4，与真实分布 154/899/394/48/107 同比）。
2. 每条规则：随机生成 N 个（N∈{7,8,9,10}，匹配真实分布）输入 8-bit，
   用 predictor 算输出，组装成与真实格式逐字一致的 prompt
   （`"In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers..."`
   + 样例 + `"Now, determine the output for: <query>"`），answer = predictor(query)。
3. **关键**：query 输入必须保证模型未在该规则的任何 train 题里见过此 (规则,query) 对。
   因为输入是全新随机采样，天然满足。

**Layer B 生成**（规则全新·没见过）：
1. 用 solver 的语法枚举 (expr, trans) 组合，**排除** majortom_results.csv 里出现过的
   全部 1602 条规则（按 `(expr, trans)` 规范化字符串建 set 去重）。
2. 抽 ~100 条全新规则，同样灌随机输入 + 算答案 + 组装 prompt。
3. **唯一性过滤（Layer B 的命门）**：生成的题必须**唯一可解**——即在 solver 的搜索空间内，
   只有目标规则能重现所有样例。否则模型选了另一个等价规则不算错，污染读数。
   实现：对每道生成题，跑 solver 求解，确认它找回的规则与目标规则在所有 8-bit 输入上
   行为一致（不要求 expr 字符串相同，要求**外延等价**：对全部 256 个输入预测相同）。
   不唯一 / solver 找不到 → 丢弃重生成。

**输出**：`datagen/bit_holdout_A.csv` + `datagen/bit_holdout_B.csv`，
schema = `id,prompt,answer,type`（type 固定 `bit_manipulation`，与 eval 的 `detect_category` 兼容）。

**自校验**（生成后、上传前必跑）：
- 每道题 `parse_bit_problem(prompt)` 能解析出样例 + query。
- 每道题答案 = 8 位二进制串、与 predictor(query) 一致（round-trip）。
- Layer A 的每个 (规则, 全部样例输入, query) 不出现在 train_with_type.csv 的任何 bit 题里。
- Layer B 的每条规则的外延（256 输入的预测向量）不等于任何 train 规则的外延。
- Layer B 每题唯一可解（solver 找回外延等价规则）。

### 4.2 Kaggle dataset `exerzitien/nemotron-bit-holdout`

两个 CSV 上传为一个 dataset。metadata `id = exerzitien/nemotron-bit-holdout`。

### 4.3 eval kernel 变体 `kernels/eval-bit-holdout/`（从 eval-v39 派生）

**唯一改动**：cell 15 的数据源。eval-v39 当前读 `jiazhuang/nemotron-val-set-950`，
改为读我们的 `bit_holdout_A.csv` + `bit_holdout_B.csv`，并加一列 `layer`（A/B）。
cell 21 的 groupby 从 `category` 改为 `layer`（或同时按两者），输出每层准确率。

**adapter 来源**：kernel-metadata 指向 **V34** 的 submission.zip
（V34 是当前最佳，#53080019；其 adapter 资产位置见 baseline_strategy memory）。
eval cell 1 的 glob 逻辑会自动找到 submission.zip。

**其余全部不动**：vLLM 参数（temp=0, max_tokens=7680, max_model_len=8192, max_lora_rank=32）、
verify()/extract_final_answer()、PROMPT_SUFFIX——与打分器对齐，保持可比。

## 5. 数据流

```
majortom_results.csv (1602 规则) ─┐
train_with_type.csv (真值参照) ───┼─► bit_holdout_gen.py ─► A.csv + B.csv ─► Kaggle dataset
majortom_bit_solver.py (语法) ────┘                              │
                                                                 ▼
                              V34 adapter (#53080019) ─► eval-bit-holdout kernel ─► 每层准确率
```

## 6. 判决标准（**提交结果前写死，禁止事后找补**）

| 读数 | 解读 | 行动 |
|---|---|---|
| Layer A ≈ 93.5%（val_950 bit 水平） | 学过的规则能用到新输入 → 应用泛化 OK | 继续看 B |
| Layer A ≪ 93.5% | 连应用泛化都不行，93.5 几乎全是死记 query | bit 死路，立即改向 |
| Layer B ≥ ~80% | 归纳泛化也真 → **bit 是真杠杆**，93.5 有真 headroom 到 oracle 98.8% | 砸更好的 token 级 CoT 冲 bit |
| Layer B ≤ ~60% | 93.5 大半靠背规则，归纳能力弱 | **改向**，别再追 bit CoT 幻影 |
| 60% < Layer B < 80% | 部分泛化 | 据 A/B 差值细判，下个 spec 决定 |

阈值数值在**生成完成、eval 提交前**冻结写入本文件，eval 出分后只读不改。

### 已冻结阈值（生成完成于 2026-06-01，eval 出分前写死，出分后只读）

- 参照基线：V34 在 val_950 上 bit = **93.5%**。
- **Layer A 判据**：≥ 88%（= 93.5% 容 5.5pp 单次 eval 方差）→ 应用泛化成立；< 80% → 应用泛化失败，bit 死路。
- **Layer B 判据**：≥ 80% → 归纳泛化成立，bit 是真杠杆，值得砸更好 CoT；≤ 60% → 93.5% 大半靠背规则，改向；60–80% → 看 A−B 差值，下个 spec 决定。
- 记录：实际生成 Layer A = **100** 题、Layer B = **100** 题（bit_holdout_A.csv / bit_holdout_B.csv，各 100，自校验全过；A 层 100/100 规则 seen-in-train、B 层 100/100 规则 novel，200 题 prompt 全互异）。

## 7. 错误处理

- solver 对某 (expr,trans) 找不到外延等价规则 → 该题丢弃，不计入。
- 生成 N 道唯一题前耗尽尝试预算（如 Layer B 规则空间在唯一性约束下太稀疏）→
  log 实际生成数，不静默截断；若 < 50 题则报警，人工看是否放宽 depth。
- eval kernel adapter glob 命中多个 submission.zip → 显式 assert 命中的是 V34。

## 8. 测试

- `bit_holdout_gen.py` 自校验（§4.1）100% 通过才上传。
- 人工抽查 5 道 A + 5 道 B：prompt 格式与真题逐字对比、答案手验 1~2 道。
- eval kernel 先 `RUN_EVALUATION` 小样（如各层 10 题）跑通管线，再全量。

## 9. 风险

- **Layer B 唯一性是软约束**：solver 搜索空间有限，"唯一可解"是相对 solver 语法而言。
  若真实模型用了 solver 语法外的规则，仍可能误判。缓解：外延等价用全 256 输入验证，
  比"expr 字符串相同"严格得多。
- **样例输入随机 vs 真题分布**：真题输入是否真随机未完全验证（已看前 3 题像随机）。
  若真题输入有隐藏结构而我们用纯随机，分布偏移可能影响读数。缓解：生成时也从真题
  采样输入分布做对照（可选，列为下一轮）。
- **N=1 次 eval 的方差**：单次读数有噪声。但 Layer A/B 各 ~100 题，比"单题抖动"稳；
  且我们看的是**层间差值**（A vs B），共模噪声部分抵消。

## 10. 不做什么（YAGNI）

- 不做通用多类框架（只 bit；eq 等拿到读数再说）。
- 不做训练数据生成（这是诊断，不是改进）。
- 不做 GRPO / 多 adapter 对比（超范围）。
- 不做 LB 提交（本轮零 LB 消耗）。
