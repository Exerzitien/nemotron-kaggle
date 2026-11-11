# v32+ 完整路线图（2026-05-26 会话结尾）

本文档持久化所有决策 + 已完成 + 待办，给下次会话或被压缩后的恢复用。

## 已完成（高密度成果，下次会话不用重做）

### 调研
- ✅ 论坛过去 7 天 20+ 帖子拉完，确认 0.86 高原 / 0.87 frozen / Bobber 公开 / lkevincc 97.2% / Komil GRPO 修复 / metric 2026-05-08 改了
- ✅ Bobber notebook 122KB 全 5 cell dump 到 `research/bobber_086/`
- ✅ lkevincc github repo clone 到 `research/lkevincc_solver/`，含 823 题 ground truth parquet
- ✅ MAJ0RT0M 9200 字符代码从 forum 688461 comments 拉回，保存 `research/majortom_bit_solver.py`

### 生成器（可直接用，不需重写）
- ✅ `datagen/cryptarithm_gen_v3.py`：基于 lkevincc 输出生成 cryptarithm CoT，656/823 行覆盖
- ✅ `datagen/synth_dataset_v3/cryptarithm_v3.csv`：656 行 5 列与 dgxchen 兼容
- ✅ `research/majortom_bit_solver.py`：MAJ0RT0M 完整算法，跑 train 1602 道 98.88% 准确
- ✅ `research/majortom_results.csv`：每道 bit 的 (gold, pred, expr, trans) ground truth

### 文档
- ✅ `research/V32_PLAN.md`：完整 v32 设计 + 数学预期
- ✅ `research/V32_STATS.txt`：覆盖率 + token 统计
- ✅ `research/V32_SIDE_BY_SIDE_ARITHMETIC.txt`：5 道 dgxchen 错 vs v3 对的对比
- ✅ memory 文件全更新，下次会话可恢复完整上下文

## 待办（按优先级）

### P0：bit_gen_v3.py CoT 生成器
**目标**：把 1584 个 MAJ0RT0M solver 结果写成 CoT，输出 `datagen/synth_dataset_v3/bit_manipulation_v3.csv`

**关键设计决策**（下次会话需要先定）：
- 选项 A：完全仿 huikang `huikang_ref/reasoners/bit_manipulation.py` 格式（per-bit 列分析 + 354 组合枚举），但里面的"matched"用 MAJ0RT0M 的 3-var 规则替换。**优点**：format-locked 安全。**缺点**：3-var 规则塞不进 huikang 的 2-var narrative。
- 选项 B：紧凑全局 narrative — "We found the rule is `expr` with A=rot k1, B=rot k2, C=shl k3. Apply per-bit: ..."。**优点**：短、清晰、全准。**缺点**：format 与 huikang/model 学的不一致。
- 选项 C：只换 238 道（huikang 错的）为选项 B 格式，1364 道留 dgxchen 原样。**优点**：低改动面、低风险。**缺点**：bit 类 CoT 训练里 15% 是新格式 85% 是旧的，model 可能困惑。

**推荐**：选项 C 起步。Gate: 选项 B 也得能落地（token 长度 + round-trip 测）。

### P1：bit_gen_v3 实现 + Gate 验证
跟 cryptarithm_gen_v3 类似的 4 个 Gate:
- A1: 5 道样本人工审
- A2: round-trip boxed extraction 100% pass (post-May-8 metric extractor)
- A3: token max <8000
- A4: 与 huikang 同 ID CoT 并列对比，确认格式可识别为"同一族"

### P2：Bobber swap reconstruction
**目标**：把 v3 CoT 数据 → 嫁接到 Bobber notebook 里。需要把 CSV 改成 Bobber 期待的 `synthetic.json` 格式（per-PID 一个文件，含 tokens 数组 + mask 数组）。

**Bobber synthetic.json schema**（从 cell 4 代码逆向）：
```json
{
  "tokens": [int, int, ...],
  "mask": [0/1, 0/1, ...]   // 0=prompt loss skipped, 1=response loss applied
}
```

**实现步骤**：
1. 加载 Nemotron tokenizer（`metric/nemotron-3-nano-30b-a3b-bf16`）
2. 对每个 v3 CoT row:
   - chat template: user = prompt + PROMPT_SUFFIX, assistant = cot + `\n</think>\n\\boxed{answer}`
   - apply_chat_template + tokenize
   - 找 assistant 开始位置，建 mask
3. 写 `<dataset>/<pid>/synthetic.json`
4. 打包成 Kaggle dataset

**约束**：token 长度必须 ≤ Bobber 用的 MAX_SEQ_LEN = 8192。我们 cryptarithm_v3 max 537 OK；bit_v3 需要测。

### P3：嫁接 Bobber notebook
1. fork Bobber notebook (`research/bobber_086/vex506-reinst-kaggle-replica.ipynb`)
2. 改 SWAP_CRYPTARITHM_THK 指向我们的 v3 swap 数据集
3. 删除 SWAP_V6 / SWAP_BINARY_FMT / EXTRA_HYPREVISE_SYNTH 引用（私有拉不到，但 Bobber 代码已有 defensive fallback 到 huikang base）
4. 删除 Modal cells（cell 5, 6 的 IS_MODAL_* 分支）
5. 上传 kernel + push 跑

### P4：训练 + 评测 + LB
GPU 消耗：~8h 训练 + ~1h eval + 0 LB 名额

Gate:
- D1: cryptarithm 类 val_950 ≥ 40%（baseline 0%）
- D2: bit 类 val_950 ≥ 95%（baseline 90%）
- D3: overall val_950 ≥ 92%
- E1: LB ≥ 0.84（不退步）

### P5 (v33+)：迭代
- 如果 P4 LB 0.85-0.86：写 hypothesis-revise binary 13500 合成（补 Bobber 缺的最后一块）
- 如果 P4 LB ≥ 0.86：上 GRPO（用 Komil 修复后的 transformers 5.5.3 + verify reward）
- 如果 P4 LB < 0.84：回退到 v30，重新分析

## 关键文件路径速查

```
/Users/exerzitien/Desktop/nemotron-kaggle/
├── data/
│   └── train.csv (9500 行原始训练数据)
├── datagen/
│   ├── cryptarithm_gen_v3.py     ← v3 生成器（已就绪）
│   ├── _ref/problem_ids_matched.csv  ← dgxchen 7830 行 baseline
│   ├── huikang_dataset/huikang_cot.csv  ← huikang 8333 行 (v29 用过 -5 LB)
│   └── synth_dataset_v3/
│       └── cryptarithm_v3.csv     ← v3 cryptarithm 656 行（就绪）
├── huikang_ref/
│   ├── reasoners/                ← 9 类 huikang reasoner 源码
│   │   ├── bit_manipulation.py   ← huikang 85% 算法
│   │   ├── cryptarithm.py        ← huikang 只覆盖 concat (~10%)
│   │   └── ...
│   ├── reasoning/                ← 9500 道每道一个 CoT txt（huikang 输出）
│   └── problems.jsonl            ← 9500 行类别索引
├── research/
│   ├── V32_PLAN.md
│   ├── V32_STATS.txt
│   ├── V32_SIDE_BY_SIDE_ARITHMETIC.txt
│   ├── V32_ROADMAP.md            ← 本文件
│   ├── train_with_type.csv       ← train.csv 加 type 列
│   ├── majortom_bit_solver.py    ← MAJ0RT0M 算法
│   ├── majortom_results.csv      ← 1602 行 bit ground truth
│   ├── lkevincc_solver/data/solver_results.parquet  ← 823 cryptarithm ground truth
│   └── bobber_086/
│       ├── vex506-reinst-kaggle-replica.ipynb  ← Bobber 0.86 完整 notebook
│       └── bobber_cells.txt      ← cells 文本 dump
├── kernels/
│   └── train/nemotron-lora-sft.ipynb  ← 我们的 v30 train kernel
└── eval_results/run1/            ← v23 baseline holdout 评测结果
    └── mistakes/                 ← 各类错例 CSV
```

## 用户的硬性偏好（永记）

1. **不愿盲推 GPU**：需要"高度确定"才 push 训练
2. **取竞品精华去糟粕**：愿意嫁接 Bobber，不死守单变量纪律
3. **诚实风险评估**：不要承诺过高再失败
4. **不浪费配额**：每步推之前确认，宁可多花调研时间
5. **持久化所有状态**：怕额度耗尽要重启，所有关键信息要落到 memory + 文档

## 数学终态预测

| 整合阶段 | 中位 LB | 上限 | 下限 |
|---|---|---|---|
| v30 baseline (现状) | 0.83 | - | - |
| v32 cryptarithm 修复（dgxchen base + 我们 656 CoT） | 0.85 | 0.87 | 0.83 |
| v32 + bit MAJ0RT0M 修复（238 道补全） | 0.86 | 0.87 | 0.83 |
| v32 + bit + Muon 优化器 | 0.87 | 0.88 | 0.83 |
| **完整 Bobber 嫁接（+ huikang base + Muon + CCE + 我们的 swap）** | **0.87** | **0.89** | 0.82 |

**0.87 仍不能 100% 保证**（榜首 frozen for months）。但完整 Bobber 嫁接的 PATH 是当前最佳。
