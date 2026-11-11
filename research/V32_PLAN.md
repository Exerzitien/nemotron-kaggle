# v32 设计 —— 单变量 cryptarithm CoT 替换

**目标 LB**：0.83 → **0.85+**（一跳上金牌区台阶）
**单变量改动**：dgxchen 训练数据里 781 行 `cryptarithm_{deduce,guess}` 的 `generated_cot` 字段，整列替换为基于 lkevincc 97.2% solver 结果重新生成的正确 CoT。
**其他**：训练参数、loss、optimizer、target_modules、所有非 cryptarithm 类数据 —— 100% 与 v30 baseline 相同（Adam, lr=2e-4, SFTTrainer, lm_head 手动注入）。

---

## 根因诊断（已核实，非假设）

### v30 cryptarithm 100% 错误的具体机制
从 `eval_results/run1/mistakes/cryptarithm_deduce.csv` 实测：模型对**每个**例题的算子都判定为"unknown"，然后说
> "As the question operator is unknown, we default to concatenation."

这不是模型笨 —— 这是它**精确学到了 dgxchen 训练数据里的 CoT**。dgxchen 的 cryptarithm CoT 来自 `huikang_ref/reasoners/cryptarithm.py`（164 行）—— 该 reasoner **只能识别 fwd/rev concat 两种规则**，对所有其他规则（算术变形）都说 "operator: unknown"。

### lkevincc solver 揭示真实分布（核实于 solver_results.parquet 823 行）
| solver_category | 占比 | huikang 能处理? |
|---|---|---|
| arithmetic（纯算术） | 38% (315) | ❌ |
| little_endian（小端算术） | 33% (276) | ❌ |
| mixed_concat（拼接+算术混合） | 13% (108) | 部分 |
| pure_concat | 7% (59) | ✅ |
| mixed_concat_little_endian | 3% (26) | ❌ |
| query_unseen_concat | 2% (16) | ❌ |

**huikang reasoner 实际只覆盖 ~10% 真题**。剩下 90% 都是算术变形（mul/add/absdiff/sub/mul_p1/add_m1/...），模型在这 90% 上必败。

### 算子语义分布（top 20，来自 solver_ops）
mul:394, add:383, absdiff:258, sub_signed:256, concat_fwd:110, add_m1:85, mul_m1:82, mul_p1:79, add_p1:78, rsub_signed:40, neg_absdiff:38, concat_rev:28, mod:15, gcd:14, rmod:10, lcm:4, ...

`_p1` = "结果再加 1"，`_m1` = "结果再减 1"。这种偏移操作是出题人特意混淆的。我们的 CoT 必须能教会模型推这些。

---

## 错例数据验证：ID `02902eb3`

| 来源 | 答案 |
|---|---|
| 真实 gold | `>/` (数字 81) |
| lkevincc solver | `>/` ✅ |
| huikang reasoner CoT | `!!$>` ❌ (default to concat) |
| 我们的 v23 模型输出 | `!!$>` ❌ (照抄了 huikang CoT 的错误结论) |

lkevincc 解出 `+` = `add_m1`（加完减 1），`*` = `mul`，`-` = `absdiff`，digit map 完整。

**这是单一变量替换就能修的问题。**

---

## 数学预期

`val_950`（950 道留出）当前：
- cryptarithm_deduce: 53 道，~50 错（94%）
- cryptarithm_guess: 20 道，~17 错（85%）
- 合计 cryptarithm 73 道，~67 错

如果 cryptarithm 准确率 0% → 50%：
- 多对 ~36 道 → val_950 += 3.8pp → **LB +0.038**

如果到 70%：
- 多对 ~51 道 → val_950 += 5.4pp → **LB +0.054**

**保守预测**：0.83 → 0.85（+0.02）
**乐观预测**：0.83 → 0.87（+0.04）

JK-Piece (rank 6, LB 0.87) 自己说 cryptarithm CoT 是 0.87→0.88 的剩余瓶颈 —— 这与我们预期一致。

---

## 风险与缓解

### R1：模板粘性反向作用
模型已经精确学了 huikang 的 box-notation 格式（v23 mistakes 完美复刻）。如果我们大幅改格式，模型可能不学。
- **缓解**：保留 box-notation `【】` 符号 + "We need to infer..." 开头 + `\boxed{}` 结尾。**只换 CoT 中间的推理内容**，从"unknown → concat"换为"arithmetic deduction"。

### R2：CoT 过长
算术推理 CoT 可能比 concat CoT 长。token 预算 ≤8192。
- **缓解**：生成 + tokenizer 实测；目标 <4000 tokens；模板限制到 <2000 tokens。

### R3：lkevincc 23 道未解
- **缓解**：这 23 道保留 dgxchen 原 CoT（最差情况无变化，不引入新错误）。

### R4：solver_ops 的算子语义教不会
模型可能学不会 `add_m1` 这种偏移。
- **缓解**：CoT 显式写"等式 = 标准算术 ± 1"。同一 CoT 里多次重复（一次声明、一次应用、一次验证）。

### R5：数据质量自检失败
- **缓解**：生成 → 100% round-trip（boxed 提取 == answer 列）；tokenize 100% <8192；前 20 道人工审。**所有 gate 不通过不进入训练**。

---

## 严格分步骤工作流（避免再翻车）

### Stage A: 0-GPU 生成 + 自检（今天）
1. 写 `datagen/cryptarithm_gen_v3.py`
2. 输出 `datagen/synth_dataset_v3/cryptarithm_v3.csv`（5 列：id, prompt, answer, type, generated_cot）
3. **Gate A1**：5 道样本人工审 —— CoT 必须可读、逻辑正确
4. **Gate A2**：所有 781 行 round-trip 提取 `\boxed{xxx}` 与 answer 列 100% 一致
5. **Gate A3**：所有 781 行 tokenize（用 Nemotron tokenizer）<8000 tokens
6. **Gate A4**：把 100 道随机抽样 + dgxchen 的对应 100 道并列对比，确认我们的 CoT 在格式上的相似度

### Stage B: 数据集打包 + push（30 分钟）
1. `kaggle datasets version` 上传到 `exerzitien/nemotron-cryptarithm-v3`
2. 更新 `kernels/train/nemotron-lora-sft.ipynb`：
   ```python
   SYNTH_DROP_TYPES = ['cryptarithm_deduce', 'cryptarithm_guess']
   SYNTH_ADD_GLOBS = ['cryptarithm_v3.csv']
   ```
3. `kernel-metadata.json` 加 `dataset_sources`
4. **不动其他任何 cell**

### Stage C: 训练（~7h GPU）
1. `kaggle kernels push`
2. Babysit 第一个 step：loss 应该在 normal 范围（不像 v29 那样爆炸）
3. 训练完成后 `kaggle kernels output --file-pattern submission.zip`

### Stage D: 评测（~1h GPU）
1. 跑 `nemotron-adapter-eval` v7 + val_950
2. **Gate D1**：cryptarithm_deduce ≥ 30%（baseline 0%）—— 若不过，说明 CoT 没学进去
3. **Gate D2**：overall val_950 ≥ 90%（baseline 88.5%）
4. **Gate D3**：其他 8 类没有显著退化（每类降 <2pp）

### Stage E: LB 提交（5 分钟，0 GPU）
1. 网页选择 train kernel 最新版本提交
2. 等 2 小时 LB 结果
3. **Gate E1**：LB ≥ 0.84（不低于 baseline）
4. **如果 Gate E1 pass 且 ≥0.85**：success，下一轮考虑加 Muon / GRPO / 更多类别
5. **如果 Gate E1 fail**：v32 立刻回退，分析失败模式

---

## 不做的事（重要 —— 避免散focus）

- ❌ 不加 Muon 优化器（Bobber 用，但是另一个变量）
- ❌ 不改 lr（保持 2e-4）
- ❌ 不加 prompt loss mask（v29 已证伪）
- ❌ 不动 lm_head（保持 v30 的手动注入，已经在）
- ❌ 不改 bit_manipulation 数据（Bobber 已证 +20K bit 数据不能突破 0.86）
- ❌ 不加 GRPO（Bobber 已证伪）
- ❌ 不动 equation_numeric_{deduce,guess}（先解决最大杠杆 cryptarithm，再看）
- ❌ 不并行启动多个实验

**如果 v32 成功：v33 才考虑加 equation_numeric_deduce/guess 类似处理。**

---

## 时间线

| 阶段 | 时间 | GPU | 累计 LB 风险 |
|---|---|---|---|
| Stage A | 今天 4-6h | 0 | 0 |
| Stage B | +30min | 0 | 0 |
| Stage C | +7h | ~7h | 0 |
| Stage D | +1h | ~1h | 0 |
| Stage E | +2h | 0 | 1 LB 名额 |

总 GPU 消耗：**~8h**（这周配额 30h 够用一次完整 run，且留出 22h 后备）。

---

## v32 之后的路线（如果 v32 +0.02 LB）

1. v33: 同样手法处理 equation_numeric_deduce/guess（596+136=732 行），用 huikang `equation_numeric.py` reasoner（已知 91.2% 覆盖 deduce）替换 dgxchen 的劣质版本
2. v34: 处理 bit_manipulation（用 MAJ0RT0M solver 99.4% 覆盖）
3. v35（如果还有时间）：加 Muon 优化器（Bobber 配方关键变量）
4. v36（终极）：GRPO with verify() reward

**截止 2026-06-15，剩 20 天 = 80h GPU 配额 = 6-8 个有效 run。**
