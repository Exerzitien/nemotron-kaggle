# Bit 记忆隔离诊断仪 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成模型证明性地从未训练过的全新 bit 题（分两层：规则见过·输入全新 / 规则全新），用 V34 adapter 跑 1 次 eval，得到对记忆免疫的真实泛化读数，判定 bit 93.5% 是真推理还是记忆。

**Architecture:** 纯函数库 `bit_holdout_lib.py`（复用 `research/majortom_bit_solver.py` 的语法/求解器）负责单题生成与有效性校验；`bit_holdout_gen.py` 编排两层采样并写出 CSV + 自校验；eval kernel 变体把数据源从 val_950 换成我们的留出集，按 layer 分组出分。本地生成 + 测试全自主、零 GPU；上传/推送/跑 eval 是受 Kaggle 资源约束的门控步骤。

**Tech Stack:** Python 3.11, pytest 7.4, pandas 2.1；复用 majortom_bit_solver（动态语法 boolean-function 搜索）；Kaggle kernel + vLLM（已有 eval-v39 基建）。

**Spec:** `docs/superpowers/specs/2026-06-01-bit-holdout-diagnostic-design.md`

**关于 git（用户已推迟决定）：** 项目根目录当前不是 git 仓库，且 `huikang_ref/` 有嵌套 `.git`、存在 1.3GB `submission.zip` 与 14820 个 corpus 子目录——**禁止 `git add .`**。所有 commit 步骤只 add 指定的新文件。Task 0 提供一个带 `.gitignore` 的安全 `git init`；若不想用版本控制，跳过 Task 0 与每个任务的 commit 子步骤即可，其余照常执行。

---

## File Structure

| 文件 | 责任 | 新建/修改 |
|---|---|---|
| `.gitignore` | 排除重型/厂商目录，保护 `git add` | 新建（Task 0，可选） |
| `conftest.py` | 把 `datagen/` 与 `research/` 加入 sys.path 供 pytest 导入 | 新建 |
| `datagen/bit_holdout_lib.py` | 纯函数：语法加载、predictor 重建、256-签名、prompt 格式化、确定性/新颖性校验、单题生成 | 新建 |
| `datagen/bit_holdout_gen.py` | 编排：Layer A/B 采样循环、CSV 写出、自校验 | 新建 |
| `tests/test_bit_holdout_lib.py` | lib 单元测试 | 新建 |
| `datagen/bit_holdout_A.csv` / `_B.csv` | 生成产物（id,prompt,answer,type,layer） | 生成 |
| `kernels/eval-bit-holdout/build_kernel.py` | 从 eval-v39 派生留出集 eval notebook 的脚本 | 新建 |
| `kernels/eval-bit-holdout/nemotron-eval-bit-holdout.ipynb` | eval kernel（数据源=留出集，按 layer 分组） | 生成 |
| `kernels/eval-bit-holdout/kernel-metadata.json` | kernel 元数据（adapter=V34，dataset=留出集） | 新建 |

---

## Task 0: （可选）安全初始化 git

**Files:**
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/.gitignore`

- [ ] **Step 1: 写 .gitignore（排除重型目录，否则 git 会卡死/污染）**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/.gitignore`:

```gitignore
# 厂商/嵌套仓库/重型数据 —— 一律不纳入版本控制
huikang_ref/
corpus/
problems/
raw/
reasoning/
investigations/
augmentations/
submissions/
kernels/*/packages/
kernels/*/adapter/
*.zip
*.safetensors
datagen/augmented_bit.csv
datagen/synth_dataset*/
__pycache__/
*.pyc
.DS_Store
```

- [ ] **Step 2: 初始化仓库（不 add 任何东西）**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && git init && git status --short | head`
Expected: 仓库创建成功；`git status` 因 .gitignore 不列出 corpus/huikang_ref/zip。若仍列出海量文件，停下检查 .gitignore 路径。

- [ ] **Step 3: 首次提交仅 .gitignore + 已有 spec**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
git add .gitignore docs/superpowers/specs/2026-06-01-bit-holdout-diagnostic-design.md
git commit -m "chore: init repo with gitignore and bit-holdout spec"
```

---

## Task 1: 建 conftest + lib 基础（bits 转换 / 语法 / predictor / 签名）

**Files:**
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/conftest.py`
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_holdout_lib.py`
- Test: `/Users/exerzitien/Desktop/nemotron-kaggle/tests/test_bit_holdout_lib.py`

- [ ] **Step 1: 写 conftest 让 pytest 能 import lib 与 solver**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/conftest.py`:

```python
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("datagen", "research"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

- [ ] **Step 2: 写失败测试（bits 往返 / 签名长度 / predictor 对齐 CSV）**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/tests/test_bit_holdout_lib.py`:

```python
import csv
import json
import os

import bit_holdout_lib as L

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "research", "majortom_results.csv")
_TYPED = os.path.join(_ROOT, "research", "train_with_type.csv")


def test_bits_roundtrip():
    for v in (0, 1, 85, 170, 255):
        bits = L.int_to_bits(v)
        assert len(bits) == 8
        assert L.str_to_bits(L.bits_to_str(bits)) == bits


def test_signature_length_and_determinism():
    e2f = L.build_expr2func()
    pf = L.make_predictor("{A}", {"{A}": ("rot", 0)}, e2f)
    sig = L.signature(pf)
    assert len(sig) == 256 * 8
    assert L.signature(pf) == sig  # deterministic


def test_predictor_matches_solver_csv():
    """重建的 predictor 必须复现 solver 当初写进 CSV 的 query 预测。"""
    e2f = L.build_expr2func()
    rows = list(csv.DictReader(open(_RESULTS)))
    typed = list(csv.DictReader(open(_TYPED)))
    id2prompt = {r["id"]: r["prompt"] for r in typed if r["type"] == "bit_manipulation"}
    import majortom_bit_solver as S
    ok = tested = 0
    for r in rows[:60]:
        if not r["expr"]:
            continue
        trans = L.parse_trans(r["trans"])
        _, _, query = S.parse_bit_problem(id2prompt[r["id"]])
        if query is None:
            continue
        pred = L.make_predictor(r["expr"], trans, e2f)(query)
        tested += 1
        ok += (pred == r["pred"])
    assert tested >= 50
    assert ok == tested
```

- [ ] **Step 3: 运行测试确认失败**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_holdout_lib.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'bit_holdout_lib'`

- [ ] **Step 4: 写 lib 基础实现**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_holdout_lib.py`:

```python
"""bit_holdout_lib — 生成「模型没训练过」的 bit 题的纯函数库。

复用 research/majortom_bit_solver.py 的动态语法与求解器。
约定：8-bit 用 list[int]（index b = 整数的第 b 位，b=0 为最低位）表示，
字符串与之的转换由 bits_to_str/str_to_bits 互逆完成；只要内部一致，
MSB/LSB 标签不影响正确性（规则是按数组下标外延定义的）。
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.join(_HERE, "..", "research")
if _RESEARCH not in sys.path:
    sys.path.insert(0, _RESEARCH)

import majortom_bit_solver as S  # noqa: E402

HEADER = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
    "\n\nHere are some examples of input -> output:\n"
)
QUERY_PREFIX = "\n\nNow, determine the output for: "


def int_to_bits(v: int) -> list[int]:
    return [(v >> b) & 1 for b in range(8)]


def bits_to_str(bits: list[int]) -> str:
    return "".join(str(b) for b in bits)


def str_to_bits(s: str) -> list[int]:
    return [int(c) for c in s]


def parse_trans(trans_json: str) -> dict:
    return {k: tuple(v) for k, v in json.loads(trans_json).items()}


def build_expr2func() -> dict:
    d = {}
    for _tt, expr, func in S.generate_grammar_dynamically():
        d[expr] = func
    return d


def make_predictor(expr: str, trans: dict, expr2func: dict):
    func = expr2func[expr]

    def predict(q: list[int]) -> str:
        out = []
        for b in range(8):
            a = S.get_source_bit(q, b, trans.get("{A}", ("rot", 0)))
            bb = S.get_source_bit(q, b, trans.get("{B}", ("rot", 0)))
            c = S.get_source_bit(q, b, trans.get("{C}", ("rot", 0)))
            out.append(str(func(a, bb, c, 1)))
        return "".join(out)

    return predict


def signature(predict_fn) -> str:
    """规则的外延：对全部 256 个输入的预测拼成 2048 字符指纹。"""
    return "".join(predict_fn(int_to_bits(i)) for i in range(256))


def format_bit_prompt(examples: list[tuple[str, str]], query_str: str) -> str:
    body = "\n".join(f"{i} -> {o}" for i, o in examples)
    return HEADER + body + QUERY_PREFIX + query_str
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_holdout_lib.py -q`
Expected: PASS（3 passed）

- [ ] **Step 6: 提交（git 已初始化时）**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
git add conftest.py datagen/bit_holdout_lib.py tests/test_bit_holdout_lib.py
git commit -m "feat: bit holdout lib core (bits/predictor/signature/prompt)"
```

---

## Task 2: prompt 格式 byte-exact 锁定

**Files:**
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/tests/test_bit_holdout_lib.py`（追加测试）

- [ ] **Step 1: 追加 gold-format 失败测试（用真题逐字节比对）**

在 `tests/test_bit_holdout_lib.py` 末尾追加：

```python
def test_format_matches_real_prompt_byte_exact():
    """用真实 bit 题（id 00066667）的样例重建 prompt，必须逐字节相等。"""
    real = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
        "binary numbers. The transformation involves operations like bit shifts, "
        "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
        "\n\nHere are some examples of input -> output:\n"
        "01010001 -> 11011101\n00001001 -> 01101101\n00010101 -> 01010101\n"
        "11111111 -> 10000001\n10011101 -> 01000101\n00111011 -> 00001001\n"
        "10111101 -> 00000101\n00100110 -> 10110011"
        "\n\nNow, determine the output for: 00110100"
    )
    examples = [
        ("01010001", "11011101"), ("00001001", "01101101"),
        ("00010101", "01010101"), ("11111111", "10000001"),
        ("10011101", "01000101"), ("00111011", "00001001"),
        ("10111101", "00000101"), ("00100110", "10110011"),
    ]
    assert L.format_bit_prompt(examples, "00110100") == real
```

- [ ] **Step 2: 运行测试**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_holdout_lib.py::test_format_matches_real_prompt_byte_exact -q`
Expected: PASS（HEADER 与 QUERY_PREFIX 已按真题校准；若 FAIL 则 diff 报告，逐字符核对 §Task1 的常量）

- [ ] **Step 3: 提交**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
git add tests/test_bit_holdout_lib.py
git commit -m "test: lock bit prompt format byte-exact against real problem"
```

---

## Task 3: 确定性校验 + 训练签名集（新颖性）

**Files:**
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_holdout_lib.py`
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/tests/test_bit_holdout_lib.py`

- [ ] **Step 1: 追加失败测试（确定性 True/False + 训练签名集规模）**

在 `tests/test_bit_holdout_lib.py` 末尾追加：

```python
def test_is_determinate_false_when_underdetermined():
    """单个全零样例无法定规则 -> 不确定。"""
    e2f = L.build_expr2func()
    id_sig = L.signature(L.make_predictor("{A}", {"{A}": ("rot", 0)}, e2f))
    assert L.is_determinate([("00000000", "00000000")], id_sig, e2f) is False


def test_is_determinate_true_for_real_rule():
    """10 个随机样例足以定一个真实 2-var 规则。"""
    import random
    e2f = L.build_expr2func()
    expr, trans = "XOR({B}, {A})", {"{A}": ("shl", 1), "{B}": ("shl", 7)}
    pf = L.make_predictor(expr, trans, e2f)
    tgt = L.signature(pf)
    rng = random.Random(1)
    exs = []
    for v in rng.sample(range(256), 10):
        q = L.int_to_bits(v)
        exs.append((L.bits_to_str(q), pf(q)))
    assert L.is_determinate(exs, tgt, e2f) is True


def test_load_train_signatures():
    e2f = L.build_expr2func()
    sigs = L.load_train_signatures(_RESULTS, e2f)
    assert isinstance(sigs, set)
    assert 700 <= len(sigs) <= 900  # 实测 765 distinct
    assert all(len(s) == 256 * 8 for s in list(sigs)[:5])
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_holdout_lib.py -q -k "determinate or train_signatures"`
Expected: FAIL，`AttributeError: module 'bit_holdout_lib' has no attribute 'is_determinate'`

- [ ] **Step 3: 在 lib 追加实现**

在 `datagen/bit_holdout_lib.py` 末尾追加：

```python
import csv as _csv


def is_determinate(examples, target_sig, expr2func, time_limit=5.0) -> bool:
    """跑 solver（简单优先搜索）能否从样例恢复出与 target 外延等价的规则。

    sig 相等 <=> 在全部 256 输入上行为一致 <=> query 答案被唯一确定。
    """
    in_arrays = [str_to_bits(i) for i, _o in examples]
    out_arrays = [str_to_bits(o) for _i, o in examples]
    _trace, predictor, _expr, _trans = S.solve_dfs_trace_dynamic(
        in_arrays, out_arrays, len(in_arrays), time_limit=time_limit
    )
    if predictor is None:
        return False
    return signature(predictor) == target_sig


def load_train_signatures(results_csv_path, expr2func) -> set:
    """majortom_results.csv 里全部已解规则的外延签名集合。"""
    sigs = set()
    for r in _csv.DictReader(open(results_csv_path)):
        if not r["expr"]:
            continue
        trans = parse_trans(r["trans"])
        sigs.add(signature(make_predictor(r["expr"], trans, expr2func)))
    return sigs
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_holdout_lib.py -q`
Expected: PASS（全部 6 passed；`load_train_signatures` 会跑 ~1600 次 solve，约数十秒，可接受）

- [ ] **Step 5: 提交**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
git add datagen/bit_holdout_lib.py tests/test_bit_holdout_lib.py
git commit -m "feat: determinacy check and train-signature novelty set"
```

---

## Task 4: 单题生成器 `generate_one_problem`

**Files:**
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_holdout_lib.py`
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/tests/test_bit_holdout_lib.py`

- [ ] **Step 1: 追加失败测试（生成的题：格式可解析 + 答案 round-trip + 确定）**

在 `tests/test_bit_holdout_lib.py` 末尾追加：

```python
def test_generate_one_problem_valid():
    import random
    import majortom_bit_solver as S
    e2f = L.build_expr2func()
    rng = random.Random(7)
    prob = L.generate_one_problem("XOR({B}, {A})",
                                  {"{A}": ("shl", 1), "{B}": ("shl", 7)},
                                  rng, e2f)
    assert prob is not None
    # 格式可被 solver 解析
    ins, outs, query = S.parse_bit_problem(prob["prompt"])
    assert query is not None and 7 <= len(ins) <= 10
    # 答案是 8-bit、与样例同源
    assert len(prob["answer"]) == 8 and set(prob["answer"]) <= {"0", "1"}
    # query 不在样例输入里（确属"新输入"）
    ex_inputs = {L.bits_to_str(x) for x in ins}
    assert L.bits_to_str(query) not in ex_inputs
    # 该题确定可解
    assert L.is_determinate(list(zip([L.bits_to_str(x) for x in ins],
                                     [L.bits_to_str(x) for x in outs])),
                            prob["target_sig"], e2f) is True
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_holdout_lib.py::test_generate_one_problem_valid -q`
Expected: FAIL，`AttributeError: ... 'generate_one_problem'`

- [ ] **Step 3: 在 lib 追加实现**

在 `datagen/bit_holdout_lib.py` 末尾追加：

```python
def generate_one_problem(target_expr, target_trans, rng, expr2func,
                         time_limit=5.0):
    """造一道以 (target_expr,target_trans) 为真规则的全新 bit 题。

    成功返回 {prompt, answer, target_sig}；若样例未能唯一确定该规则返回 None（调用方重试）。
    """
    predict = make_predictor(target_expr, target_trans, expr2func)
    target_sig = signature(predict)
    n = rng.choice([7, 8, 9, 10])
    vals = rng.sample(range(256), n + 1)  # n+1 个互异输入，最后一个作 query
    inputs = [int_to_bits(v) for v in vals[:n]]
    query = int_to_bits(vals[n])
    examples = [(bits_to_str(x), predict(x)) for x in inputs]
    answer = predict(query)
    prompt = format_bit_prompt(examples, bits_to_str(query))
    if not is_determinate(examples, target_sig, expr2func, time_limit):
        return None
    return {"prompt": prompt, "answer": answer, "target_sig": target_sig}
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_holdout_lib.py -q`
Expected: PASS（7 passed）

- [ ] **Step 5: 提交**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
git add datagen/bit_holdout_lib.py tests/test_bit_holdout_lib.py
git commit -m "feat: generate_one_problem with determinacy gating"
```

---

## Task 5: 编排脚本 `bit_holdout_gen.py` + 生成两层 CSV

**Files:**
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_holdout_gen.py`
- 产物: `datagen/bit_holdout_A.csv`, `datagen/bit_holdout_B.csv`

- [ ] **Step 1: 写编排脚本**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_holdout_gen.py`:

```python
"""bit_holdout_gen — 造两层「模型没训练过」的 bit 留出集。

Layer A「规则见过·输入全新」：从 majortom_results.csv 的已解规则抽样，灌全新输入。
Layer B「规则全新·没见过」：用 solver 语法 mint 训练签名集外的规则，灌输入。
两层都过确定性校验（solver 能恢复目标外延），B 额外过新颖性（签名不在训练集）。

输出 schema: id,prompt,answer,type,layer（type 固定 bit_manipulation）。
所有数字写死、可复现（SEED 固定）；不静默截断，dropped 计数 log 出来。
"""
import csv
import os
import random

import bit_holdout_lib as L

SEED = 20260601
N_PER_LAYER = 100
MAX_TRIES_PER_ITEM = 40
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "research", "majortom_results.csv")
OUT_A = os.path.join(_HERE, "bit_holdout_A.csv")
OUT_B = os.path.join(_HERE, "bit_holdout_B.csv")


def _depth(expr: str) -> int:
    return expr.count("(")


def gen_layer_a(rng, e2f, n):
    """从 CSV 已解规则按 nesting-depth 分层抽样，每条灌全新输入。"""
    rows = [r for r in csv.DictReader(open(RESULTS)) if r["expr"]]
    by_depth = {}
    for r in rows:
        by_depth.setdefault(_depth(r["expr"]), []).append(r)
    total = sum(len(v) for v in by_depth.values())
    out, dropped = [], 0
    for d, group in sorted(by_depth.items()):
        quota = max(1, round(n * len(group) / total))
        picks = rng.sample(group, min(quota, len(group)))
        for r in picks:
            if len(out) >= n:
                break
            trans = L.parse_trans(r["trans"])
            prob = None
            for _ in range(MAX_TRIES_PER_ITEM):
                prob = L.generate_one_problem(r["expr"], trans, rng, e2f)
                if prob is not None:
                    break
            if prob is None:
                dropped += 1
                continue
            out.append((f"holdoutA_{len(out):03d}", prob["prompt"], prob["answer"]))
    print(f"[Layer A] generated {len(out)} (dropped {dropped})")
    return out


def gen_layer_b(rng, e2f, train_sigs, n):
    """mint 训练签名集外的全新规则，每条灌全新输入。"""
    exprs = list(e2f.keys())
    TR = L.S.TRANSFORMATIONS
    out, dropped, tries = [], 0, 0
    budget = n * MAX_TRIES_PER_ITEM * 4
    while len(out) < n and tries < budget:
        tries += 1
        expr = rng.choice(exprs)
        used = L.S.get_used_vars(expr)
        trans = {v: rng.choice(TR) for v in used}
        sig = L.signature(L.make_predictor(expr, trans, e2f))
        if sig in train_sigs:
            continue  # 不新颖
        prob = L.generate_one_problem(expr, trans, rng, e2f)
        if prob is None:
            dropped += 1
            continue
        out.append((f"holdoutB_{len(out):03d}", prob["prompt"], prob["answer"]))
    print(f"[Layer B] generated {len(out)} (dropped {dropped}, tries {tries})")
    return out


def _write(path, rows, layer):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "answer", "type", "layer"])
        for pid, prompt, answer in rows:
            w.writerow([pid, prompt, answer, "bit_manipulation", layer])


def main():
    rng = random.Random(SEED)
    e2f = L.build_expr2func()
    print("Loading train signatures (~1600 solves, may take ~30s)...")
    train_sigs = L.load_train_signatures(RESULTS, e2f)
    print(f"train distinct signatures: {len(train_sigs)}")

    a = gen_layer_a(rng, e2f, N_PER_LAYER)
    b = gen_layer_b(rng, e2f, train_sigs, N_PER_LAYER)

    if len(a) < 50:
        print(f"WARNING: Layer A only {len(a)} (<50) — investigate before upload")
    if len(b) < 50:
        print(f"WARNING: Layer B only {len(b)} (<50) — widen depth or budget")

    _write(OUT_A, a, "A")
    _write(OUT_B, b, "B")
    print(f"Wrote {OUT_A} ({len(a)}) and {OUT_B} ({len(b)})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行生成**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 datagen/bit_holdout_gen.py`
Expected: 打印两层生成数（各应 ≥ 50，目标 100），写出 A.csv / B.csv。若任一层 < 50 会 WARNING——停下，按 §Task6 调参（放宽 depth 抽样或加 budget）再跑。

- [ ] **Step 3: 写一次性自校验脚本并运行（生成后、上传前必过）**

Run（直接执行，不存盘）:

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -c "
import csv, sys
sys.path.insert(0, 'datagen'); sys.path.insert(0, 'research')
import bit_holdout_lib as L
import majortom_bit_solver as S
e2f = L.build_expr2func()
train_sigs = L.load_train_signatures('research/majortom_results.csv', e2f)
for path, layer in [('datagen/bit_holdout_A.csv','A'), ('datagen/bit_holdout_B.csv','B')]:
    rows = list(csv.DictReader(open(path)))
    assert rows, path+' empty'
    for r in rows:
        ins, outs, query = S.parse_bit_problem(r['prompt'])
        assert query is not None, 'unparseable '+r['id']
        # round-trip: 用样例求解, 复现 query 答案
        _, pred, _, _ = S.solve_dfs_trace_dynamic(ins, outs, len(ins), 5.0)
        assert pred is not None, 'unsolvable '+r['id']
        assert pred(query) == r['answer'], 'answer mismatch '+r['id']
        assert r['type']=='bit_manipulation' and r['layer']==layer
        if layer=='B':
            ex = list(zip([L.bits_to_str(x) for x in ins],[L.bits_to_str(x) for x in outs]))
            # B 层规则签名必须不在训练集
            # (用 solver 恢复的规则签名近似目标外延)
            sig = L.signature(pred)
            assert sig not in train_sigs, 'NOT novel: '+r['id']
    print(path, 'OK', len(rows), 'rows')
print('ALL SELF-CHECKS PASSED')
"
```
Expected: 两行 `... OK N rows` + `ALL SELF-CHECKS PASSED`。任一 assert 失败即停，修生成器。

- [ ] **Step 4: 人工抽查 5A + 5B（格式 + 手验 1~2 道答案）**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -c "
import csv
for p in ['datagen/bit_holdout_A.csv','datagen/bit_holdout_B.csv']:
    rows=list(csv.DictReader(open(p)))
    print('=====', p, '=====')
    for r in rows[:5]:
        print(r['id'], '| answer:', r['answer'])
        print(r['prompt'][:300]); print('...')
"`
Expected: prompt 开头与真题一致（"In Alice's Wonderland, a secret bit manipulation..."），样例行形如 `01010001 -> 11011101`，结尾 `Now, determine the output for: <8bit>`。

- [ ] **Step 5: 提交（CSV 是小文件可入库；augmented_bit.csv 已被 .gitignore）**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
git add datagen/bit_holdout_gen.py datagen/bit_holdout_A.csv datagen/bit_holdout_B.csv
git commit -m "feat: generate two-layer bit holdout set (memory-isolated diagnostic)"
```

---

## Task 6: （若某层 < 50）放宽抽样后重生成

**前置条件：仅当 Task 5 Step 2 出现 WARNING 才执行；否则跳过。**

**Files:**
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_holdout_gen.py`

- [ ] **Step 1: Layer A 不足 → 把分层配额改为"先按比例、不足再从全池补足"**

在 `gen_layer_a` 的 `for d, group ...` 循环之后、`return` 之前插入补足逻辑：

```python
    # 配额不足时从全池随机补到 n
    if len(out) < n:
        pool = [r for grp in by_depth.values() for r in grp]
        rng.shuffle(pool)
        for r in pool:
            if len(out) >= n:
                break
            trans = L.parse_trans(r["trans"])
            prob = None
            for _ in range(MAX_TRIES_PER_ITEM):
                prob = L.generate_one_problem(r["expr"], trans, rng, e2f)
                if prob is not None:
                    break
            if prob is not None:
                out.append((f"holdoutA_{len(out):03d}", prob["prompt"], prob["answer"]))
```

- [ ] **Step 2: Layer B 不足 → 调大 budget 系数**

把 `gen_layer_b` 里 `budget = n * MAX_TRIES_PER_ITEM * 4` 的 `* 4` 改为 `* 12`。

- [ ] **Step 3: 重跑生成 + 重跑 Task 5 Step 3 自校验**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 datagen/bit_holdout_gen.py`
Expected: 两层均 ≥ 50（目标 100），无 WARNING。然后重跑 Task 5 Step 3 自校验，必须 `ALL SELF-CHECKS PASSED`。

- [ ] **Step 4: 提交**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
git add datagen/bit_holdout_gen.py datagen/bit_holdout_A.csv datagen/bit_holdout_B.csv
git commit -m "fix: backfill layer sampling to reach target count"
```

---

## Task 7: 冻结判决阈值（写进 spec，eval 前不可改）

**Files:**
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/docs/superpowers/specs/2026-06-01-bit-holdout-diagnostic-design.md`

- [ ] **Step 1: 在 spec §6 表格下方追加"已冻结阈值"块**

在 spec 第 6 节表格之后追加：

```markdown
### 已冻结阈值（生成完成于 2026-06-01，eval 出分前写死，出分后只读）

- 参照基线：V34 在 val_950 上 bit = **93.5%**。
- **Layer A 判据**：≥ 88%（= 93.5% 容 5.5pp 单次 eval 方差）→ 应用泛化成立；< 80% → 应用泛化失败，bit 死路。
- **Layer B 判据**：≥ 80% → 归纳泛化成立，bit 是真杠杆，值得砸更好 CoT；≤ 60% → 93.5% 大半靠背规则，改向；60–80% → 看 A−B 差值，下个 spec 决定。
- 记录：实际生成 Layer A = ___ 题、Layer B = ___ 题（跑完 Task 5 填入）。
```

- [ ] **Step 2: 填入实际生成题数（从 Task 5 输出）**

把上一步的 `___` 替换为 A.csv / B.csv 实际行数（`wc -l` 减 1 表头）。

- [ ] **Step 3: 提交**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
git add docs/superpowers/specs/2026-06-01-bit-holdout-diagnostic-design.md
git commit -m "docs: freeze verdict thresholds before eval"
```

---

## Task 8: 构建 eval kernel 变体（派生 eval-v39，换数据源 + 按 layer 分组）

**Files:**
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/kernels/eval-bit-holdout/build_kernel.py`
- 产物: `kernels/eval-bit-holdout/nemotron-eval-bit-holdout.ipynb`
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/kernels/eval-bit-holdout/kernel-metadata.json`

- [ ] **Step 1: 写 notebook 构建脚本（程序化改 cell，避免手改 JSON 出错）**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/kernels/eval-bit-holdout/build_kernel.py`:

```python
"""从 eval-v39 notebook 派生留出集 eval：
- cell 15: 数据源从 val_950 换成我们的 A.csv + B.csv，df 带 'layer' 列。
- cell 21: groupby('category') 改为 groupby('layer')。
其余（vLLM 参数、verify、prompt 构造、adapter glob）全部不动，保持与打分器一致。
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, "..", "eval-v39", "nemotron-eval-v39.ipynb")
DST = os.path.join(_HERE, "nemotron-eval-bit-holdout.ipynb")

# 新 cell 15：加载两层留出集（Kaggle dataset 解包后的路径），合并 + 加 layer 列
NEW_CELL15 = r'''import pandas as pd
import glob as _g

def detect_category(prompt: str) -> str:
    if "secret bit manipulation rule transforms 8-bit binary numbers" in prompt:
        return "bit_manipulation"
    return "unknown"

# EVAL SET: exerzitien/nemotron-bit-holdout (two layers A/B)
_a = _g.glob("/kaggle/input/**/bit_holdout_A.csv", recursive=True)
_b = _g.glob("/kaggle/input/**/bit_holdout_B.csv", recursive=True)
assert _a and _b, ("bit_holdout_A/B.csv not found - add "
                   "`exerzitien/nemotron-bit-holdout` to dataset_sources")
_da = pd.read_csv(_a[0]); _db = pd.read_csv(_b[0])
df = pd.concat([_da, _db], ignore_index=True)
df["category"] = df["prompt"].apply(detect_category)
assert (df["category"] == "bit_manipulation").all(), "non-bit rows leaked in"
print(f"holdout loaded: A={len(_da)} B={len(_db)} total={len(df)}")
print(df["layer"].value_counts().sort_index())
df = df.reset_index(drop=True).copy()
'''

# 新 cell 21：按 layer 分组出分（同时保留 mistakes 落盘）
NEW_CELL21 = r'''if RUN_EVALUATION:
    import os
    os.makedirs("mistakes", exist_ok=True)
    for layer in df["layer"].unique():
        m = df[(df["layer"] == layer) & (~df["correct"])]
        if not m.empty:
            m.to_csv(f"mistakes/layer_{layer}.csv", index=False)
    stats = df.groupby("layer")["correct"].agg(correct="sum", total="count").sort_index()
    stats["correct"] = stats["correct"].astype("int")
    gt = stats["total"].sum()
    stats["percentage"] = (stats["correct"] / stats["total"] * 100).map("{:.1f}%".format)
    overall = stats["correct"].sum() / gt * 100
    print(stats.to_string())
    print(f"OVERALL: {stats['correct'].sum()}/{gt} = {overall:.1f}%")
    stats.to_csv("results.csv")
    df.to_csv("validation.csv", index=False)
'''

def main():
    nb = json.load(open(SRC))
    cells = nb["cells"]
    # 定位并替换：用稳定锚点而非硬下标，防 eval-v39 cell 漂移
    replaced15 = replaced21 = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "val_set_950" in src and "detect_category" in src:
            c["source"] = NEW_CELL15
            replaced15 += 1
        elif "groupby(\"category\")" in src and "results.csv" in src:
            c["source"] = NEW_CELL21
            replaced21 += 1
    assert replaced15 == 1, f"cell15 anchor matched {replaced15}x (expected 1)"
    assert replaced21 == 1, f"cell21 anchor matched {replaced21}x (expected 1)"
    json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
    print(f"Wrote {DST} (replaced cell15 x{replaced15}, cell21 x{replaced21})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行构建脚本**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 kernels/eval-bit-holdout/build_kernel.py`
Expected: `Wrote .../nemotron-eval-bit-holdout.ipynb (replaced cell15 x1, cell21 x1)`。若 anchor 匹配数 ≠ 1，停下看 eval-v39 是否结构变了。

- [ ] **Step 3: 校验生成的 notebook 是合法 JSON 且改对了**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -c "
import json
nb=json.load(open('kernels/eval-bit-holdout/nemotron-eval-bit-holdout.ipynb'))
srcs=[''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in nb['cells'] if c['cell_type']=='code']
assert any('bit_holdout_A.csv' in s for s in srcs), 'cell15 not applied'
assert any('groupby(\"layer\")' in s for s in srcs), 'cell21 not applied'
assert not any('val_set_950' in s for s in srcs), 'val_950 still present'
print('notebook OK:', len(nb[\"cells\"]), 'cells')
"`
Expected: `notebook OK: N cells`

- [ ] **Step 4: 写 kernel-metadata.json（adapter 源 = V34，照抄 eval-v39 的 model/docker/machine 字段）**

先读 eval-v39 的 metadata 作为蓝本：

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && cat kernels/eval-v39/kernel-metadata.json`

然后 Create `/Users/exerzitien/Desktop/nemotron-kaggle/kernels/eval-bit-holdout/kernel-metadata.json`，
在 eval-v39 metadata 基础上改三处：`id` → `exerzitien/nemotron-eval-bit-holdout`、`title` → `Nemotron Eval Bit Holdout`、`code_file` → `nemotron-eval-bit-holdout.ipynb`；
`dataset_sources` 在原有基础上**加** `exerzitien/nemotron-bit-holdout`；
**adapter 源保持指向 V34**（与 eval-v39 当前指向同一个产生 0.85 的 adapter——若 eval-v39 用 `kernel_sources` 拉训练 kernel 的 submission.zip，则保持该 `kernel_sources` 不变；若用 dataset，则保持该 dataset）。
其余 `model_sources` / `docker_image` / `machine_shape` / `enable_gpu` / `enable_internet` 逐字保留。

> ⚠️ 确认点：eval-bit-holdout 必须 eval **V34 的 adapter**（#53080019，当前最佳 0.85），不是 V39。运行前用 Step 5 核对。

- [ ] **Step 5: 提交（不含 .ipynb 重复进库——可选只提脚本+元数据）**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
git add kernels/eval-bit-holdout/build_kernel.py kernels/eval-bit-holdout/kernel-metadata.json kernels/eval-bit-holdout/nemotron-eval-bit-holdout.ipynb
git commit -m "feat: bit-holdout eval kernel (layer-grouped, V34 adapter)"
```

---

## Task 9: 【门控·消耗 Kaggle 资源】上传数据集 + 推 kernel + 取读数

**前置：以下步骤上传数据/跑 GPU eval，必须先经用户确认再执行。鉴权用 `KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token)`。**

- [ ] **Step 1: 建并上传留出集 Kaggle dataset**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
mkdir -p /tmp/bit-holdout-ds
cp datagen/bit_holdout_A.csv datagen/bit_holdout_B.csv /tmp/bit-holdout-ds/
cat > /tmp/bit-holdout-ds/dataset-metadata.json <<'EOF'
{"id": "exerzitien/nemotron-bit-holdout", "title": "nemotron-bit-holdout"}
EOF
KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token) \
  kaggle datasets create -p /tmp/bit-holdout-ds
```
Expected: dataset 创建成功，返回 URL。若 dataset 已存在改用 `kaggle datasets version -p /tmp/bit-holdout-ds -m "update"`。

- [ ] **Step 2: 核对 kernel-metadata adapter 源指向 V34，然后推送**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && cat kernels/eval-bit-holdout/kernel-metadata.json`
确认 adapter 源 = V34（#53080019 对应的 submission/dataset）。确认无误后：

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token) \
  kaggle kernels push -p kernels/eval-bit-holdout
```
Expected: kernel 推送成功，进入 RUNNING。

- [ ] **Step 3: 轮询状态直至 COMPLETE**

```bash
KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token) \
  kaggle kernels status exerzitien/nemotron-eval-bit-holdout
```
Expected: 最终 `COMPLETE`。eval ~1h。若 ERROR：取 log（见 [[kaggle-cli-bearer-output]]：curl Bearer + 预签名 URL，别用 `kernels output` 全量拉）。

- [ ] **Step 4: 取 results.csv（两层准确率），别整包下载**

按 [[kaggle-cli-bearer-output]]：
```bash
KGAT=$(tr -d '[:space:]' < ~/.kaggle/access_token)
curl -s -H "Authorization: Bearer $KGAT" \
  "https://www.kaggle.com/api/v1/kernels/output?userName=exerzitien&kernelSlug=nemotron-eval-bit-holdout" \
  -o /tmp/holdout_out.json
# 从 logNullable 直接读两层打印结果，或从 files 里取 results.csv 的预签名 url 点对点下
```
Expected：拿到 Layer A / Layer B 各自准确率。存到 `eval_results/bit-holdout/results.csv`。

- [ ] **Step 5: 对照 Task 7 冻结阈值判决，写结论 memory**

按 spec §6 冻结阈值判定（A≥88 应用泛化成立；B≥80 归纳泛化成立=bit 真杠杆；B≤60 改向）。
更新 memory（项目入口 `project_nemotron_baseline_strategy.md`）记录读数与判决——这是本轮唯一对未来有价值的产出。

---

## Self-Review

**1. Spec coverage（逐节核对）:**
- §3 三步管线 → Task 5（gen）+ Task 8（kernel）+ Task 9（run）✓
- §4.1 Layer A/B 生成 + 唯一性过滤 → Task 4（generate_one_problem 含 is_determinate）+ Task 5（gen_layer_a/b 含 train_sigs 新颖性）✓
- §4.1 自校验（parse/round-trip/novelty/determinacy） → Task 5 Step 3 ✓
- §4.2 Kaggle dataset → Task 9 Step 1 ✓
- §4.3 eval 变体（cell15 换源 / cell21 按 layer / adapter=V34 / vLLM 不动） → Task 8 ✓
- §6 判决阈值冻结 → Task 7 ✓
- §7 错误处理（找不到规则丢弃 / 不足报警不静默截断 / adapter glob 核对） → generate_one_problem 返回 None + gen 的 dropped 计数与 <50 WARNING（Task5 Step2）+ Task6 补足 + Task8 Step4 确认点 ✓
- §8 测试（自校验 / 抽查 / 小样跑通） → Task5 Step3-4；小样跑通由 eval kernel `RUN_EVALUATION` 控制，Task9 全量 ✓
- §10 YAGNI（只 bit / 不训练 / 不提交 LB） → 全程无训练、无 LB 提交 ✓

**2. Placeholder scan:** spec §6 的阈值"留空"是 Task 7 显式冻结动作，非占位符；plan 内无 TBD/TODO/"类似上文"。所有代码步给出完整代码。✓

**3. Type consistency:** `int_to_bits/bits_to_str/str_to_bits/build_expr2func/make_predictor/signature/format_bit_prompt/parse_trans/is_determinate/load_train_signatures/generate_one_problem` 在 Task1/3/4 定义，Task5 调用签名一致；`generate_one_problem` 返回 `{prompt,answer,target_sig}` 在 Task4 测试与 Task5 写出处一致；CSV schema `id,prompt,answer,type,layer` 在 Task5 写出与 Task8 cell15 读取一致；solver 函数 `generate_grammar_dynamically/get_source_bit/get_used_vars/parse_bit_problem/solve_dfs_trace_dynamic/TRANSFORMATIONS/OPS` 均来自已读过的 `majortom_bit_solver.py`。✓

一个已知软约束（已在 spec §9 记录）：Task5 Step3 自校验里 B 层新颖性用"solver 恢复规则的签名 ∉ train_sigs"近似目标外延新颖性——与生成时"目标签名 ∉ train_sigs"在确定性成立时等价（sig 相等⟺外延一致），逻辑自洽。
