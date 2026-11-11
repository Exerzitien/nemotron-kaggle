# v2 设计规格(代码前的检查点)

**状态**:✅ **已实现并部署**(2026-05-20)
- `datagen/equation_gen_v2.py`、`datagen/cryptarithm_gen_v2.py` 各 deduce/guess 模式
- 共 2600 行生成数据(1000+200+1200+200),5 项自校验全过
- 数据集 `exerzitien/nemotron-synth-cot` v2 已上传(4 CSVs)
- notebook cell 2/11 已更新(开关 + 训练参数)
- kernel v25 已 push、RUNNING

---

**目标**:修复 run#24 暴露的 4 个根因。

| 根因 | 来自 | 修法 |
|---|---|---|
| equation_deduce 77% 题被误归为 guess(走 abs-diff fallback) | dgxchen 模板没显式校验「query op 在不在示例里」 | 在 v2 CoT 里**显式列示例算符集合** + **显式比对 query op 是否在其中** |
| equation_guess 100% 默认 abs-diff(并污染 deduce) | 126 行训练 CoT 全是 "not found → abs diff" | v2 guess CoT 重写:先做与 deduce 一样的搜索 + **明确不同的措辞**,让模型能 distinguish |
| cryptarithm_deduce 0%(只覆盖 concat / reverse-concat) | dgxchen 模板候选集只有 2 种 | v2 把候选集扩到 **~12 种结构性位置图** |
| cryptarithm_guess 100% 默认 concat(并污染 deduce) | 154 行训练 CoT 全是 "default to concat" | 同上 + 改 fallback 措辞 |

**训练参数也调**:
- `max_grad_norm`: 1e9 → **1.0**(关键!初始 loss 26.86 + lr 上升期梯度可任意大,这是 14 道 signed magnitude 退化最可能的来源)
- `learning_rate`: 2e-4 → **1.5e-4**(温和)
- `weight_decay`: 0 → **0.01**(轻 L2)
- 其他不动

---

## 一、equation 模板(deduce + guess 共用骨架)

```
We need to infer the transformation rule from the examples.
I will put my final answer inside \boxed{}.

Examples:
  {a1}{op1}{b1} = {r1}
  ...

The inputs are {all numbers, comma-separated}

The outputs are {all RHS strings, comma-separated}
{Decoration sentence — see below}

Looking at the input of the examples
{aN}{opN}{bN} -> {opN}
...

The operators
{distinct ops, one per line}

Looking at the question
{qa}{qop}{qb} -> {qop}
{FORK SENTENCE — see below}

Looking at operator 【{qop}】 [{filtered examples list}]:
  Trying common operations on identity [{operands}] [{expected RHS for each}]:
    {op_name} f(a, b) = ... = X {match/mismatch}
    ...
  Trying common operations reversed operands [{reversed pairs}] [{expected for each}]:
    ...
  Trying rare operations on identity [...]:
    ...
  Trying rare operations reversed operands [...]:
    ...

Applying to {query}:
  {framing}
  {chosen_op} f({qa}, {qb}) = ... = {result_value}
  {Decoration application if needed}
  Result: 【{answer}】

I will now return the answer in \boxed{}
The answer in \boxed{–} is \boxed{{answer}}
</think>
\boxed{{answer}}
```

### 1.1 Decoration sentence

- 若示例所有 RHS 都是纯数字:`No outputs have a symbol prefix or suffix.`
- 若部分示例 RHS 含 leading 算符字符 X(等于该题的某算符):
  ```
  Some outputs have the operator symbol as prefix 【X】.
  We now consider the outputs to be {original_with_prefix_as_negative}.
  We will add back the operator prefix if our answer is negative.
  ```
  关键:**把算符前缀解释为「负号的伪装」**(dgxchen 原版的优雅设计,完整保留)。

### 1.2 FORK 句(deduce vs guess 关键区分点)

**deduce** (query op IS in example ops):
```
The question operator is 【{qop}】, which is in the example operators. We have direct examples constraining the rule.
```

**guess** (query op NOT in example ops):
```
The question operator is 【{qop}】, which is NOT in the example operators {set}. We have no direct constraint. We will investigate the most common example operator 【{common_op}】 to infer the puzzle's general rule family, then apply that family to the query.
```

**关键差异**:
- deduce 明确说 "is in" + "direct examples constraining"
- guess 明确说 "NOT in" + "no direct constraint" + "infer general family"

让模型在训练时**学到这两条措辞对应不同上下文条件**,从而 inference 时能严格区分。

### 1.3 搜索段

24 ops × 4 framings(identity/reversed operands × identity/reversed result),**完整保留 dgxchen 原版**(已验证强大)。

匹配判定**收紧**:必须**所有过滤后的示例都匹配**,只匹配 2/3 不算。

### 1.4 v2 改进点(相对 dgxchen)

- ✅ 显式 `is in / NOT in the example operators` 比对(dgxchen 没显式列示例算符集)
- ✅ 显式 "direct examples constraining" vs "no direct constraint" 措辞
- ✅ guess 不再 100% 默认 abs-diff —— **改为「研究最常见示例算符的规则,再套用」**,与 dgxchen 原版同方向但措辞更明确

### 1.5 数据量(精准重平衡 + 适度扩)

- equation_deduce: **1000 行**(从 658 扩):
  - 24 ops × 4 framings 均衡覆盖(每格 ~10 行)
  - 装饰约定:75% plain / **25% prefix-as-negative**(从 v1 的 ~7% 大幅提升)
- equation_guess: **200 行**(从 126 扩)

---

## 二、cryptarithm 模板(deduce + guess 共用骨架)

```
We need to infer the transformation rule from the examples.
I will put my final answer inside \boxed{}.

[For each example:]
【{LHS}】 = 【{RHS}】
  input: 【s1】【s2】【s3】【s4】【s5】
  left: 【s1】【s2】
  operator: 【s3】
  right: 【s4】【s5】
  output: 【o1】【o2】... (parsed RHS)
  {Candidate tests — see below}
  operator: 【s3】{conclusion}

Question【{query}】
  input: 【...】
  left: 【...】
  operator: 【qop】
  right: 【...】

{FORK SENTENCE — see below}

{Apply the chosen rule:}
  {rule_name}(left, right) = {result symbols}
  output: 【{answer}】 -> 【{{answer}}】

I will now return the answer in \boxed{}
The answer in \boxed{–} is \boxed{{answer}}
</think>
\boxed{{answer}}
```

### 2.1 候选规则列表(扩到 ~12 种)

每条都是输出 = 操作数符号的位置序列。`L1 L2` = 左操作数两个符号,`R1 R2` = 右,长度可变(2-4):

| name | output pattern |
|---|---|
| concatenation | L1 L2 R1 R2 |
| reverse concatenation | R1 R2 L1 L2 |
| left only | L1 L2 |
| right only | R1 R2 |
| left reversed | L2 L1 |
| right reversed | R2 R1 |
| interleave LR | L1 R1 L2 R2 |
| interleave RL | R1 L1 R2 L2 |
| left first only | L1 |
| right last only | R2 |
| left doubled | L1 L2 L1 L2 |
| left wraps right | L1 R1 R2 L2 |

(这 12 条覆盖 length 1-4,共 ~10% 真题分布;可根据生成的 train 错例进一步迭代。)

### 2.2 Candidate tests(每个示例的搜索段)

```
  concatenation: 【L1】【L2】【R1】【R2】 [match/mismatch]
  reverse concatenation: 【R1】【R2】【L1】【L2】 [match/mismatch]
  left only: 【L1】【L2】 [match/mismatch]
  right only: 【R1】【R2】 [match/mismatch]
  left reversed: 【L2】【L1】 [match/mismatch]
  right reversed: 【R2】【R1】 [match/mismatch]
  interleave LR: 【L1】【R1】【L2】【R2】 [match/mismatch]
  interleave RL: 【R1】【L1】【R2】【L2】 [match/mismatch]
  left first only: 【L1】 [match/mismatch]
  right last only: 【R2】 [match/mismatch]
  left doubled: 【L1】【L2】【L1】【L2】 [match/mismatch]
  left wraps right: 【L1】【R1】【R2】【L2】 [match/mismatch]
```

**conclusion**:在所有 match 的候选中,挑最简洁(优先 concat、reverse-concat、其后按上表顺序)→ "operator: 【s3】{candidate_name}"

如果所有候选都 mismatch → "operator: 【s3】unknown"

### 2.3 FORK 句(deduce vs guess 关键区分)

**deduce** (query op IS one of the example ops with a determined rule):
```
The question operator is 【{qop}】, which is in the example operators. From the examples, we determined: 【{qop}】 is {rule_name}.
```

**guess** (query op NOT in example ops with determined rules):
```
The question operator is 【{qop}】, which is NOT in the operators we determined from the examples {set}. We have no direct constraint. We will use the most common rule among the determined operators as our best guess.
```

**关键差异**:
- deduce 说 "is in" + "we determined"
- guess 说 "NOT in" + "most common rule as best guess"

不再 100% 默认 concat;guess 选示例里出现最多的规则当 fallback(也可能仍是 concat,但不是无脑默认)。

### 2.4 数据量(精准重平衡 + 适度扩)

- cryptarithm_deduce: **1200 行**(从 627 扩):
  - 12 候选规则各 ~100 行(等概率覆盖)
- cryptarithm_guess: **200 行**(从 154 扩)

### 2.5 答案符号约定

- 符号池排除 `{}` `\boxed` 相关字符,保证答案能干净放入 `\boxed{...}`
- 操作数 4 符号两两不同 + 算符符号也不同 → posmap 单例可推

---

## 三、生成与校验流程

### 每个生成器自校验

每生成一条都做:
1. **CoT 末尾**:`\boxed{answer}` 且 answer 不含 `{` `}`
2. **示例真实匹配**:CoT 标注 "match" 的候选必须真的等于该示例 RHS;标 "mismatch" 的必须真的不等
3. **结论真实**:CoT 声称的 rule/op 必须真能在所有该算符示例上产出对应 RHS
4. **Apply 真实**:Apply 段计算结果必须等于 answer
5. **FORK 一致**:deduce 一定声称 "is in",guess 一定声称 "NOT in",措辞用确切常量

### 端到端 dry-run

生成各类全量(658+126+627+154)→ 跑混合块 → 跑 `\boxed` 剥离 + 重提取 → 必须 100% 一致。

---

## 四、训练参数 v2

```python
# Mode Selection 格不变。training_args 改:
training_args = SFTConfig(
    ...
    learning_rate=1.5e-4,     # 2e-4 → 1.5e-4
    max_grad_norm=1.0,        # 1e9 → 1.0  (关键!)
    weight_decay=0.01,        # 0 → 0.01
    # 其他全部保留 dgxchen 配方
)
```

预期效应:
- **`max_grad_norm=1.0`**:压制 step 1-30 的剧烈梯度(初始 loss 26.86),保护 LoRA 初始化阶段的 signed magnitude 推理 —— 直接针对 run#24 那 14 道 signed 退化
- **`lr=1.5e-4`**:平均收益更温和,减少过拟合到少数派模式的概率
- **`weight_decay=0.01`**:LoRA 上极轻正则,与混合数据相得益彰
- **`num_train_epochs=1` 不动**:loss 已平台,1 epoch 够;不冒 14h GPU 风险

---

## 五、风险与回退

**主要风险**:
1. v2 CoT 生成器的搜索段实现错误 → 训练数据有自相矛盾的 CoT → 模型困惑
2. cryptarithm 12 候选规则中存在「同一示例多个候选都 match」的歧义 → 训练数据规则选择不一致
3. 训练参数 3 个一起改 → 出问题难归因

**缓解**:
1. 自校验 5 项 + 全量 dry-run + 至少抽样 50 条人工读 CoT
2. 候选规则按确定优先级排序(concat 优先),保证选择一致
3. 三个参数都是保守方向,理论上各自风险都低,但出问题用 backout 路径:回退 max_grad_norm

**回退路径**:
- 若 v2 跑出来比 run#24 还差 → 下下个 run 退回 run#24 的参数 + 重新审视生成器
- 若 v2 持平或微跌 → 看错例诊断哪个变量出问题
