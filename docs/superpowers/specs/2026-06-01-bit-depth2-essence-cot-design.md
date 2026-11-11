# Bit depth≥2 串联精华 CoT — 设计文档

**日期**: 2026-06-01
**状态**: 待用户审阅
**前置**: [[nemotron-bit-holdout-diagnostic]]（诊断仪已建，V34 读数 93.5/84/71，断点=depth≥2）

---

## 1. 问题陈述

诊断仪（2026-06-01）给出对记忆免疫的真泛化读数：V34 在 bit 上 val_950=93.5% → Layer A(规则见过·输入全新)=84% → **Layer B(规则全新)=71%**。按规则嵌套深度拆，断点精确定位：

| 规则深度 | train 规则占比 | Layer B 准确率（实测精确值） | 结构 |
|---|---|---|---|
| depth 0–1 | 66%（train） | **87.8%**（36/41，占 Layer B 41%） | 1–2 变量，单门 |
| depth ≥2 | 34%（train） | **59.3%**（35/59，占 Layer B 59%） | 3 变量复合 |

（注：「train 规则占比」是 1602 条已解 train 规则里的深度分布；准确率分母 41/59 是 bit-holdout Layer B 100 题里的深度分布——Layer B 用语法 mint 全新规则，天然偏深，故 depth≥2 占比更高。两者合计 36+35=71 = Layer B 整体 71%。）

**根因（零成本核实 V34 实际输出已证）**：V34 训练的 dgxchen/THK 风格 bit CoT 只有「每个输出位 = 相邻输入位对上的一个 2 输入门」这套词汇。遇到 depth≥2 复合规则，模型**没有「组合」的语言**——它沿用单门匹配，匹配得上的位用单门、匹配不上的位直接填 `default 1`/`C0` 常量瞎猜（见 mistakes_layer_B.csv 实证：`Matched: 0 none 1 none ... Selected: 0 default 1 1 default 1`）。**瓶颈是 CoT 表征对 35% 规则空间表达不完整，不是算力不足。**

## 2. 关键结构事实（全部本地实测验证，是整个方案的地基）

1. **depth-2 规则 = 两个 depth-1 门的串联**（40/40 复现 solver）：
   `XNOR(A, NOT_A_AND_B(B,C))` ≡ 先算 `g = NOT_A_AND_B(B', C')`（完整 8 位中间向量），再算 `out = XNOR(A', g)`。每一步都是模型已 87.8% 会的 depth-1 操作。
2. **任意深度规则 = depth-1 门在 8 位中间向量上的有向无环图 DAG**（depth-3+ 80/80 复现，门数最多 4）。
3. **规则空间小**：549 条 depth≥2 规则只坍缩成 **11 个布尔骨架**，top 6 覆盖 80%。
4. **长度是利好不是负担**（tokenizer 实测）：

   | CoT | tokens |
   |---|---|
   | dgxchen depth-1（V34 现在学的，87.8%） | 7316 |
   | 串联精华 CoT（depth-2，中道颗粒度） | ~600 |
   | 全 per-bit 展开版 | 2655 |

   dgxchen 7316 token 大半是**搜索日志 + bitsum-hash**（试遍所有门的噪声）。串联 CoT 砍掉搜索、只留「分解骨架 + 逐门计算 + 应用」，比 baseline 短 7–12×。

## 3. 目标与非目标

**目标**：对 depth≥2 的 bit 题生成「串联精华 CoT」，教模型把已会的 depth-1 门**接起来**，提升 Layer B depth≥2 分桶准确率（当前 59.3%），且不损伤已 87.8% 的 depth 0-1。

**非目标（本轮不做）**：
- 不碰 depth 0-1（保留 dgxchen 原版 CoT 不动）。
- 不碰 bit 以外的类。
- 不做 GRPO/多 adapter。
- 训练（Tier 2）只在前三道闸全过后才做，且单独决策。

## 4. 可学性三原则（回答"怎么提炼精华"）

1. **复用已会原子**：每个中间步 = depth-1 门（模型 Layer B 已 87.8% 会）。不教新操作，只教"接起来"。
2. **长度=砍搜索**：精华 = 删掉搜索日志/bitsum-hash，只留分解 + 逐门计算 + 应用。
3. **抗橡皮图章（避开 V39 坑）**：query 两步强制逐位展开（`bit0: g=NAB(1,0)=0, out=XNOR(1,0)=1`…）逼真算；样例向量级（`g=..., out=...`）保持紧凑。颗粒度下限不是最短，是**最少的、但强制真算的步骤**。

## 5. 架构

```
bit_essence_gen.py（本地，0 GPU）
  对 1754 道 dgxchen bit train 题，逐题跑 majortom solver 求 expr + depth：
   ├─ depth 0–1（~66%, ~1150题）→ 保留 dgxchen 原版 generated_cot 不动
   └─ depth ≥2（~34%, ~600题）→ 生成串联精华 CoT 替换：
        1. solver 求 (expr, trans)
        2. expr 解析成 depth-1 门的 DAG（parse_expr → 树）
        3. 拓扑序逐门算 8 位中间向量（eval_tree）
        4. 渲染 CoT：分解骨架 + 样例(向量级) + query(逐位强制) + \boxed
        5. Tier 0 自洽校验（逐门重算==answer 且 token<8192）；失败→回退原版
                          │
   输出混合 CSV：id,prompt,answer,type,generated_cot（depth0-1原版 + depth≥2串联）
                          │
   Kaggle dataset exerzitien/nemotron-bit-essence
                          │
   训练 kernel cell2 数据源换成它 → V40 adapter（Tier 1 通过后才训）
```

## 6. 组件

### 6.1 `datagen/bit_essence_lib.py`（新建，纯函数）
复用 `research/majortom_bit_solver.py`（OPS/get_source_bit/solve_dfs_trace_dynamic/parse_bit_problem）。
- `parse_expr(expr_str) -> tree`：把 `OR(NOT_A_AND_B({A},{B}), AND({C},{A}))` 解析成嵌套树 `('OP',name,[args])` / `('VAR','{A}')`。
- `eval_tree(tree, vecs) -> list[int]`：拓扑序对 8 位向量求值（NOT 一元，其余二元用 OPS）。
- `count_gates(tree) -> int`：门数（= 中间步数，驱动 CoT 长度上界）。
- `topo_steps(tree) -> list[(label, op, inputs)]`：返回从叶到根的有序门列表，供渲染。
- `render_essence_cot(ins, outs, query, expr, trans) -> str`：组装中道颗粒度 CoT。
- `selfcheck_cot(cot, query, answer) -> bool`：从 CoT 末尾解析 \boxed、且独立逐门重算 == answer。

### 6.2 `datagen/bit_essence_gen.py`（新建，编排）
逐题分流（depth 阈值），depth≥2 调 lib 生成 + Tier0 自洽，失败回退原版；输出混合 CSV；打印 depth 分布与替换/回退计数（不静默截断）。

### 6.3 `kernels/eval-incontext-bit/`（Tier 1 闸，新建）
- 派生一个 eval kernel：**基座模型、不挂 adapter**，把 2–3 条写满的串联 CoT 作为 few-shot 前缀，在 bit-holdout **Layer B depth≥2 子集**上解题。
- 复用诊断仪 eval 基建（vLLM 参数对齐 scorer），按 depth 分桶出分。

### 6.4 训练/最终 eval（Tier 2，复用现有）
训练 kernel `kernels/train/` cell2 数据源换 `nemotron-bit-essence`；最终用诊断仪 `kernels/eval-bit-holdout/` 测 Layer B depth≥2 分桶。

## 7. 四道闸（核心：每道比下一道便宜一个量级）

```
Tier0 自洽（0 GPU，秒级，强制 baked-in）
  每条 depth≥2 CoT 逐门重算==answer 且 token<8192，否则回退原版
        ↓
人工核 20 条（0 GPU）
  抽查 depth≥2 CoT：逐位推导真的可跟、骨架正确、答案自洽
        ↓
Tier1 in-context 可跟随（1h GPU，不训练）← 训练前的 GPU 闸
  基座模型 + few-shot 串联 CoT，测 Layer B depth≥2 分桶：
   · 明显涨过 59.3% → 格式可跟随 → 值得训练
   · 持平 → 照着示范都跟不动 → 砍，省 7h
        ↓
Tier2 真训练 + 诊断仪裁决（7h+1h GPU）
  V40 adapter → 诊断仪 Layer B depth≥2 分桶：涨才提交 LB
```

判据数值在生成完成、各 Tier 执行前冻结写入本文件。

### 已冻结 Tier1 判据（2026-06-01，Tier1 eval 提交前写死，出分后只读）

Tier1 在 base 模型上跑两遍 bit_holdout depth≥2 留出集（~150 题，Task 5 产出）：
- **A 遍**：zero-shot（无 demo），得 `acc_zero`。
- **B 遍**：few-shot（2 条精华 CoT demo 前缀），得 `acc_demo`。

判据（同时满足才算「格式可跟随、值得训练」）：
- `acc_demo ≥ 70%`，且
- `acc_demo − acc_zero ≥ +15pp`（demo 带来的真实提升，排除 base 模型本身会做的部分）。

任一不满足 → 格式连 in-context 都跟不动 → **砍掉，不进 Tier2 训练**，省 7h GPU。
（说明：此判据用 base 模型 with/without demo 的「提升量」，比直接对比 V34 的 59.3%（adapter 条件，apples-to-oranges）更干净。59.3% 仅作背景参照。）

## 8. 数据流

```
majortom_results.csv (规则) ─┐
train_with_type.csv (题+真值)─┼─► bit_essence_gen.py ─► 混合CSV ─► Kaggle dataset
majortom_bit_solver.py (eval)─┘                              │
                          ┌──────────────────────────────────┤
                   Tier1: 基座+few-shot ─► eval-incontext-bit ─► Layer B depth≥2 跟随率
                   Tier2: train(V40) ─► eval-bit-holdout ─► Layer B depth≥2 准确率
```

## 9. 错误处理
- solver 找不到规则的 depth≥2 题 → 保留原版 CoT（不强造）。
- CoT 自洽校验失败（\boxed≠gold 或逐门重算≠answer）→ 回退原版。
- token ≥ 8192 → log 并回退原版（depth-3+ 门多时可能触发）。
- 替换率 < 50%（大量回退）→ 报警，人工看渲染逻辑，别静默全量。

## 10. 测试
- `bit_essence_lib` 单元测试：parse_expr 往返、eval_tree 对全 256 输入 == solver predictor（外延等价）、count_gates、selfcheck True/False 两路。
- gen 自校验：每条 depth≥2 CoT 自洽 + token<8192；打印 depth 分布。
- 人工抽查 20 条（Tier 0 后、Tier 1 前）。
- Tier 1 in-context 跟随率 = 训练前 GPU 闸。
- Tier 2 诊断仪 Layer B depth≥2 = 最终裁决。

## 11. 风险
- **in-context≠SFT**：Tier1 可跟随是 SFT 可学性的强先行指标，非充分条件。缓解：它只用来**砍**（持平就停），不用来保证成功；最终仍靠 Tier2 诊断仪裁决。
- **两种格式混训**：depth0-1 原版（长、搜索式）与 depth≥2 串联（短、分解式）混在一训练集。风险=模型对 depth≥2 题误用原版搜索格式。缓解：两种 CoT 开头句不同（depth≥2 以"single-gate matching fails; 2-stage composition"起头），给模型可区分的触发信号；诊断仪分桶能测出是否串味。
- **depth≥2 子集样本量**：bit-holdout Layer B 里 depth≥2 约 59 题，分桶后单次 eval 方差偏大。缓解：看 depth≥2 整体桶而非更细分；必要时 Tier1/Tier2 前给 Layer B 补一批 depth≥2 全新题（复用诊断仪生成器，零 GPU）。

## 12. 不做什么（YAGNI）
- 不碰 depth 0-1、不碰其他类、不做 GRPO、Tier2 训练单独决策（前三闸全过才提）。退路永远 V34=0.85。
