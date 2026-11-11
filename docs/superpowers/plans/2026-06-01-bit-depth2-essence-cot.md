# Bit depth≥2 串联精华 CoT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 depth≥2 的 bit 题生成「串联精华 CoT」（把规则分解成 depth-1 门在 8 位中间向量上的 DAG，逐门计算），替换 dgxchen 的搜索式长 CoT；用 base 模型 + few-shot 的 in-context 跟随测试做训练前 GPU 闸。

**Architecture:** 纯函数库 `bit_essence_lib.py`（复用 `research/majortom_bit_solver.py` 的 OPS/求解器）负责 expr 解析、DAG 求值、CoT 渲染、自洽校验；`bit_essence_gen.py` 编排 1754 道 bit 题分流（depth0-1 保留原版 / depth≥2 换精华 CoT）输出混合训练集；一个补充脚本把 Layer B depth≥2 留出集扩到 ~150 去噪；一个 in-context eval kernel（base 模型、无 adapter、few-shot）做训练前闸。本地 + 测试全自主、零 GPU；只有最后一个上传/跑 eval 的任务消耗 Kaggle 资源。

**Tech Stack:** Python 3.11, pytest 7.4, pandas 2.1, tokenizers；复用 majortom_bit_solver（boolean-gate DAG）+ bit_holdout_lib（诊断仪，上一个 plan 产出）+ eval-bit-holdout kernel 基建。

**Spec:** `docs/superpowers/specs/2026-06-01-bit-depth2-essence-cot-design.md`

**已验证的地基（写代码可直接信）：**
- `S.OPS` 有 10 个二输入门：`AND OR XOR NAND NOR XNOR NOT_A_AND_B A_AND_NOT_B NOT_A_OR_B A_OR_NOT_B`，签名 `f(a,b) -> int`（需 `& 1`）。
- depth≥2 的 549 条规则**全是二输入门组合**：0 条含 bare `NOT(...)`，0 条含 `C0/C1` 常量。出现的算子：`AND A_AND_NOT_B NOR NOT_A_AND_B NOT_A_OR_B OR XNOR XOR`。
- `S.get_source_bit(in_bits_list, out_idx, trans_tuple)`：`in_bits_list`=8 个 int，`trans_tuple`=`('shl',1)` 等。
- `S.parse_bit_problem(prompt)` → `(in_arrays, out_arrays, query)`，均为 `list[list[int]]` / `list[int]`。
- `S.solve_dfs_trace_dynamic(in_arrays,out_arrays,n,time_limit)` → `(trace, predictor, expr, trans_dict)`；`trans_dict` 值是 tuple。
- dgxchen CSV `datagen/_ref/problem_ids_matched.csv` 表头：`﻿id,prompt,answer,type,generated_cot`（**`id` 列带 BOM**）；bit 行 1754 条。
- dgxchen 的 `generated_cot` 内容**不含开头 `<think>`**（chat 模板会加），以 `\n</think>\n\boxed{答案}` 结尾。
- `conftest.py`（上个 plan 建）已把 `datagen/` 和 `research/` 加进 sys.path，测试可直接 `import bit_essence_lib`。

**关于 git：** 用户已推迟版本控制。**不要运行任何 git 命令，跳过所有 commit 步骤**——只创建文件、跑测试。

---

## File Structure

| 文件 | 责任 | 新建/修改 |
|---|---|---|
| `datagen/bit_essence_lib.py` | 纯函数：parse_expr / eval_tree / count_gates / topo_steps / render_essence_cot / selfcheck_cot | 新建 |
| `tests/test_bit_essence_lib.py` | lib 单元测试 | 新建 |
| `datagen/bit_essence_gen.py` | 编排：分流 + 渲染 + 自洽 + 输出混合 CSV | 新建 |
| `datagen/bit_essence_mixed.csv` | 混合训练集产物（depth0-1 原版 + depth≥2 精华） | 生成 |
| `datagen/bit_holdout_depth2_supplement.py` | 扩 Layer B depth≥2 留出集到 ~150（去噪 Tier1） | 新建 |
| `datagen/bit_holdout_depth2.csv` | depth≥2 全新留出集产物 | 生成 |
| `kernels/eval-incontext-bit/build_kernel.py` | 从 eval-bit-holdout 派生 in-context base 模型 eval | 新建 |
| `kernels/eval-incontext-bit/nemotron-eval-incontext-bit.ipynb` | Tier1 闸 kernel | 生成 |
| `kernels/eval-incontext-bit/kernel-metadata.json` | kernel 元数据（base 模型、无 adapter） | 新建 |

---

## Task 1: bit_essence_lib —— expr 解析 + DAG 求值

**Files:**
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_essence_lib.py`
- Test: `/Users/exerzitien/Desktop/nemotron-kaggle/tests/test_bit_essence_lib.py`

- [ ] **Step 1: 写失败测试（parse_expr 结构 + eval_tree 外延等价 + count_gates）**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/tests/test_bit_essence_lib.py`:

```python
import csv
import json
import os

import bit_essence_lib as E

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "research", "majortom_results.csv")


def test_parse_expr_structure():
    t = E.parse_expr("XNOR({A}, NOT_A_AND_B({B}, {C}))")
    assert t == ("OP", "XNOR", [
        ("VAR", "{A}"),
        ("OP", "NOT_A_AND_B", [("VAR", "{B}"), ("VAR", "{C}")]),
    ])


def test_count_gates():
    assert E.count_gates(E.parse_expr("{A}")) == 0
    assert E.count_gates(E.parse_expr("XOR({A}, {B})")) == 1
    assert E.count_gates(E.parse_expr("XNOR({A}, NOT_A_AND_B({B}, {C}))")) == 2


def test_eval_tree_matches_solver_over_all_inputs():
    """parse_expr + eval_tree 必须在全部 256 输入上复现 solver predictor（外延等价）。"""
    import majortom_bit_solver as S
    rows = [r for r in csv.DictReader(open(_RESULTS))
            if r["expr"] and r["expr"].count("(") >= 2]
    tested = ok = 0
    for r in rows[:60]:
        expr = r["expr"]
        trans = {k: tuple(v) for k, v in json.loads(r["trans"]).items()}
        tree = E.parse_expr(expr)
        # 对每个 8-bit 输入：solver 的 predictor 用 trans 取源位再算；eval_tree 同理
        mism = 0
        for v in range(256):
            inp = [(v >> b) & 1 for b in range(8)]
            vecs = {var: [S.get_source_bit(inp, b, trans.get(var, ("rot", 0)))
                          for b in range(8)]
                    for var in ("{A}", "{B}", "{C}")}
            got = "".join(str(x) for x in E.eval_tree(tree, vecs))
            # solver predictor 重建：直接用 expr 的 func？改用已验证路径——比对 r['pred'] 不够（只 1 输入）
            # 这里用 solver 的 OPS 手算同一棵树作为 oracle 不行（就是被测对象）。
            # 改为：与 solver 在该题样例上恢复的 predictor 比对见下个断言；此处仅自一致性占位。
            pass
        tested += 1
    assert tested >= 50
```

> 注意：上面 `test_eval_tree_matches_solver_over_all_inputs` 的循环体是占位思路，**Step 3 要把它替换成真正的 oracle 比对**——用 solver 在该题真实样例上 `solve_dfs_trace_dynamic` 恢复的 `predictor`（已知正确）作为 oracle，对全 256 输入比对 `eval_tree`。见 Step 3 给出的最终版测试。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_essence_lib.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'bit_essence_lib'`

- [ ] **Step 3: 用最终版测试替换 Step 1 的占位测试**

把 `tests/test_bit_essence_lib.py` 整个替换为：

```python
import csv
import json
import os

import bit_essence_lib as E
import majortom_bit_solver as S

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "research", "majortom_results.csv")
_TYPED = os.path.join(_ROOT, "research", "train_with_type.csv")


def test_parse_expr_structure():
    t = E.parse_expr("XNOR({A}, NOT_A_AND_B({B}, {C}))")
    assert t == ("OP", "XNOR", [
        ("VAR", "{A}"),
        ("OP", "NOT_A_AND_B", [("VAR", "{B}"), ("VAR", "{C}")]),
    ])


def test_count_gates():
    assert E.count_gates(E.parse_expr("{A}")) == 0
    assert E.count_gates(E.parse_expr("XOR({A}, {B})")) == 1
    assert E.count_gates(E.parse_expr("XNOR({A}, NOT_A_AND_B({B}, {C}))")) == 2


def test_eval_tree_matches_solver_predictor():
    """eval_tree 必须复现 solver 在真实题上恢复的 predictor（对该题全部样例+query）。"""
    import re
    typed = {r["id"]: r["prompt"] for r in csv.DictReader(open(_TYPED))
             if r["type"] == "bit_manipulation"}
    rows = [r for r in csv.DictReader(open(_RESULTS))
            if r["expr"] and r["expr"].count("(") >= 2]
    tested = ok = 0
    for r in rows[:60]:
        if r["id"] not in typed:
            continue
        ins, outs, query = S.parse_bit_problem(typed[r["id"]])
        trace, predictor, expr, trans = S.solve_dfs_trace_dynamic(
            ins, outs, len(ins), 5.0)
        if predictor is None:
            continue
        tree = E.parse_expr(expr)
        # 对 query：用 trans 取源位向量，eval_tree 出结果，比对 solver predictor
        vecs = {var: [S.get_source_bit(query, b, trans.get(var, ("rot", 0)))
                      for b in range(8)]
                for var in ("{A}", "{B}", "{C}")}
        got = "".join(str(x) for x in E.eval_tree(tree, vecs))
        tested += 1
        ok += (got == predictor(query))
    assert tested >= 40
    assert ok == tested
```

- [ ] **Step 4: 写 lib 实现**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_essence_lib.py`:

```python
"""bit_essence_lib — 把任意深度 bit 规则渲染成「串联精华 CoT」。

核心事实（已实测）：任意 bit 规则 = depth-1 二输入门在 8 位中间向量上的 DAG。
depth≥2 规则全是二输入门组合（无 bare NOT、无 C0/C1 常量），为稳健仍支持一元 NOT / 常量。
复用 research/majortom_bit_solver.py 的 OPS / get_source_bit。
8-bit 用 list[int]，index b = 第 b 位。
"""
from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.join(_HERE, "..", "research")
if _RESEARCH not in sys.path:
    sys.path.insert(0, _RESEARCH)

import majortom_bit_solver as S  # noqa: E402

OP_ABBREV = {
    "AND": "AND", "OR": "OR", "XOR": "XOR", "NAND": "NAND", "NOR": "NOR",
    "XNOR": "XNOR", "NOT_A_AND_B": "NAB", "A_AND_NOT_B": "ANB",
    "NOT_A_OR_B": "NOB", "A_OR_NOT_B": "ONB",
}


def parse_expr(e: str):
    """expr 字符串 -> 树：('VAR','{A}') / ('CONST','C0') / ('OP',name,[args])。"""
    e = e.strip()
    if e in ("{A}", "{B}", "{C}"):
        return ("VAR", e)
    if e in ("C0", "C1"):
        return ("CONST", e)
    m = re.match(r"^([A-Z_]+)\((.*)\)$", e)
    if not m:
        raise ValueError(f"cannot parse expr: {e!r}")
    name, inner = m.group(1), m.group(2)
    depth = 0
    parts = []
    cur = ""
    for ch in inner:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return ("OP", name, [parse_expr(p) for p in parts])


def eval_tree(tree, vecs):
    """对 8 位向量求值。vecs 键为 '{A}'/'{B}'/'{C}'，值为已套 transform 的 8-bit list。"""
    kind = tree[0]
    if kind == "VAR":
        return vecs[tree[1]]
    if kind == "CONST":
        return [0] * 8 if tree[1] == "C0" else [1] * 8
    name = tree[1]
    args = [eval_tree(a, vecs) for a in tree[2]]
    if name == "NOT":
        return [(~args[0][i]) & 1 for i in range(8)]
    f = S.OPS[name]
    return [f(args[0][i], args[1][i]) & 1 for i in range(8)]


def count_gates(tree) -> int:
    if tree[0] in ("VAR", "CONST"):
        return 0
    return 1 + sum(count_gates(a) for a in tree[2])
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_essence_lib.py -q`
Expected: PASS（3 passed；`test_eval_tree_matches_solver_predictor` 会跑 solver 数十次，约 10–30s）

- [ ] **Step 6: 提交 —— SKIP（git deferred）。**

---

## Task 2: bit_essence_lib —— 渲染 + 自洽校验

**Files:**
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_essence_lib.py`
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/tests/test_bit_essence_lib.py`

- [ ] **Step 1: 追加失败测试（CoT 自洽 + 格式末尾匹配 dgxchen 包裹 + 含强制逐位）**

在 `tests/test_bit_essence_lib.py` 末尾追加：

```python
def _one_depth2(idn=None):
    """取一道真实 depth-2 题，返回 (ins, outs, query, expr, trans, answer)。"""
    typed = {r["id"]: r for r in csv.DictReader(open(_TYPED))
             if r["type"] == "bit_manipulation"}
    rows = [r for r in csv.DictReader(open(_RESULTS))
            if r["expr"] and r["expr"].count("(") == 2 and r["id"] in typed]
    r = rows[0]
    pr = typed[r["id"]]
    ins, outs, query = S.parse_bit_problem(pr["prompt"])
    trans = {k: tuple(v) for k, v in json.loads(r["trans"]).items()}
    answer = str(pr["answer"]).strip().zfill(8)
    return ins, outs, query, r["expr"], trans, answer


def test_render_and_selfcheck_consistent():
    ins, outs, query, expr, trans, answer = _one_depth2()
    cot = E.render_essence_cot(ins, outs, query, expr, trans)
    # 自洽：boxed==answer 且独立重算==answer
    assert E.selfcheck_cot(cot, query, answer, expr, trans) is True
    # boxed 内容确实等于真答案
    import re
    boxed = re.findall(r"\\boxed\{([01]{8})\}", cot)
    assert boxed and boxed[-1] == answer


def test_render_format_matches_dgxchen_wrapping():
    """CoT 内容须以 </think> + \\boxed 结尾，且不含开头 <think>（与 dgxchen 一致）。"""
    ins, outs, query, expr, trans, answer = _one_depth2()
    cot = E.render_essence_cot(ins, outs, query, expr, trans)
    assert not cot.lstrip().startswith("<think>")
    assert cot.rstrip().endswith("\\boxed{" + answer + "}")
    assert "</think>" in cot
    assert cot.count("</think>") == 1


def test_render_forces_query_per_bit():
    """query 段必须逐位展开（出现 bit0..bit7 的强制计算行）。"""
    ins, outs, query, expr, trans, answer = _one_depth2()
    cot = E.render_essence_cot(ins, outs, query, expr, trans)
    for i in range(8):
        assert f"bit{i}:" in cot


def test_selfcheck_rejects_wrong_answer():
    ins, outs, query, expr, trans, answer = _one_depth2()
    cot = E.render_essence_cot(ins, outs, query, expr, trans)
    bad = cot.replace("\\boxed{" + answer + "}", "\\boxed{00000000}")
    # 故意篡改 boxed，selfcheck 必须拒（除非真答案恰好是全0，极不可能）
    assert E.selfcheck_cot(bad, query, "00000000", expr, trans) is False
```

- [ ] **Step 2: 跑确认失败**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_essence_lib.py -q -k "render or selfcheck"`
Expected: FAIL，`AttributeError: ... 'render_essence_cot'`

- [ ] **Step 3: 在 lib 追加实现**

在 `datagen/bit_essence_lib.py` 末尾追加：

```python
def topo_steps(tree):
    """后序遍历 -> 有序门列表 [(out_label, op_name, [in_labels]), ...]，根标 'out'。
    叶变量标 A/B/C；中间门标 g1,g2,...。"""
    steps = []
    counter = [0]

    def label_of(t):
        if t[0] == "VAR":
            return t[1].strip("{}")
        if t[0] == "CONST":
            return t[1]
        child_labels = [label_of(c) for c in t[2]]
        counter[0] += 1
        lab = f"g{counter[0]}"
        steps.append((lab, t[1], child_labels))
        return lab

    root = label_of(tree)
    if steps:
        last_lab, last_op, last_in = steps[-1]
        steps[-1] = ("out", last_op, last_in)
    return steps, root


def _short(name):
    return OP_ABBREV.get(name, name)


def _tname(t):
    return f"{t[0]}{t[1]}"


def _vstr(v):
    return "".join(str(b) for b in v)


def _var_trans(trans):
    return {v.strip("{}"): trans.get(v, ("rot", 0)) for v in ("{A}", "{B}", "{C}")}


def _compute_env(inp, steps, var_trans):
    """算出该输入下每个 label 的 8 位向量。"""
    env = {v: [S.get_source_bit(inp, b, var_trans[v]) for b in range(8)]
           for v in ("A", "B", "C")}
    env["C0"] = [0] * 8
    env["C1"] = [1] * 8
    for lab, opname, inlabels in steps:
        a = env[inlabels[0]]
        if opname == "NOT":
            env[lab] = [(~a[i]) & 1 for i in range(8)]
        else:
            b = env[inlabels[1]]
            f = S.OPS[opname]
            env[lab] = [f(a[i], b[i]) & 1 for i in range(8)]
    return env


def render_essence_cot(ins, outs, query, expr, trans) -> str:
    """渲染串联精华 CoT。样例向量级、query 逐位强制；结尾 </think>+\\boxed，匹配 dgxchen 包裹。"""
    tree = parse_expr(expr)
    steps, _root = topo_steps(tree)
    vt = _var_trans(trans)
    used_vars = sorted({iv for _, _, ils in steps for iv in ils
                        if iv in ("A", "B", "C")})

    def disp(lab):
        return f"{lab}'" if lab in ("A", "B", "C") else lab

    L = ["Single-gate matching fails; the rule is a multi-stage composition."]
    for v in used_vars:
        L.append(f"{v}' = {v}.{_tname(vt[v])}")
    for lab, opname, inlabels in steps:
        args = ", ".join(disp(x) for x in inlabels)
        L.append(f"{lab} = {_short(opname)}({args})")
    L.append("")
    for k, inp in enumerate(ins):
        env = _compute_env(inp, steps, vt)
        parts = [f"in={_vstr(inp)}"]
        for v in used_vars:
            parts.append(f"{v}'={_vstr(env[v])}")
        for lab, _, _ in steps:
            if lab != "out":
                parts.append(f"{lab}={_vstr(env[lab])}")
        parts.append(f"out={_vstr(env['out'])}")
        L.append(f"ex{k}: " + " ".join(parts) + f" (want {_vstr(outs[k])})")
    L.append("")
    qenv = _compute_env(query, steps, vt)
    L.append(f"query: in={_vstr(query)}")
    for v in used_vars:
        L.append(f"{v}'={_vstr(qenv[v])}")
    for i in range(8):
        seg = []
        for lab, opname, inlabels in steps:
            if opname == "NOT":
                seg.append(f"{lab}=NOT({qenv[inlabels[0]][i]})={qenv[lab][i]}")
            else:
                seg.append(
                    f"{lab}={_short(opname)}("
                    f"{qenv[inlabels[0]][i]},{qenv[inlabels[1]][i]})={qenv[lab][i]}")
        L.append(f" bit{i}: " + " ".join(seg))
    ans = _vstr(qenv["out"])
    L.append(f"out={ans}")
    L.append("</think>")
    L.append(f"\\boxed{{{ans}}}")
    return "\n".join(L)


def selfcheck_cot(cot, query, answer, expr, trans) -> bool:
    """1) CoT 末尾 \\boxed == answer；2) 独立 eval_tree 重算 == answer。"""
    m = re.findall(r"\\boxed\{([01]{8})\}", cot)
    if not m or m[-1] != answer:
        return False
    tree = parse_expr(expr)
    vt = _var_trans(trans)
    vecs = {f"{{{v}}}": [S.get_source_bit(query, b, vt[v]) for b in range(8)]
            for v in ("A", "B", "C")}
    recomputed = "".join(str(x) for x in eval_tree(tree, vecs))
    return recomputed == answer
```

- [ ] **Step 4: 跑确认通过**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -m pytest tests/test_bit_essence_lib.py -q`
Expected: PASS（7 passed）

- [ ] **Step 5: 人工核一条渲染输出（打印，确认可读+逐位可跟）**

Run:
```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -c "
import sys; sys.path.insert(0,'datagen'); sys.path.insert(0,'research')
import csv, json
import bit_essence_lib as E
import majortom_bit_solver as S
typed={r['id']:r for r in csv.DictReader(open('research/train_with_type.csv')) if r['type']=='bit_manipulation'}
rows=[r for r in csv.DictReader(open('research/majortom_results.csv')) if r['expr'] and r['expr'].count('(')==2 and r['id'] in typed]
r=rows[0]; pr=typed[r['id']]
ins,outs,q=S.parse_bit_problem(pr['prompt'])
trans={k:tuple(v) for k,v in json.loads(r['trans']).items()}
print('RULE:', r['expr'], trans)
print(E.render_essence_cot(ins,outs,q,r['expr'],trans))
"
```
Expected: 打印一条完整 CoT；骨架（gN = OP(...)）、样例向量、query 逐位、`</think>`、`\boxed{}` 都在；人工确认每条 `biti:` 行的门计算自洽。

- [ ] **Step 6: 提交 —— SKIP（git deferred）。**

---

## Task 3: bit_essence_gen —— 输出混合训练集

**Files:**
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_essence_gen.py`
- 产物: `datagen/bit_essence_mixed.csv`

- [ ] **Step 1: 写编排脚本**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_essence_gen.py`:

```python
"""bit_essence_gen — 生成混合 bit 训练集。

非 bit 行 + bit 中 depth0-1 行：保留 dgxchen 原版 generated_cot。
bit 中 depth≥2 行：换成串联精华 CoT（自洽 + token<8192 才换，否则回退原版）。
输出 schema 同 dgxchen：id,prompt,answer,type,generated_cot。不静默截断，计数 log 出来。
"""
import csv
import os
import sys

csv.field_size_limit(sys.maxsize)

import bit_essence_lib as E
import majortom_bit_solver as S
from tokenizers import Tokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
DGX = os.path.join(_HERE, "_ref", "problem_ids_matched.csv")
TOKJSON = os.path.join(_HERE, "..", "huikang_ref", "tokenizer.json")
OUT = os.path.join(_HERE, "bit_essence_mixed.csv")
TOKEN_LIMIT = 8192


def _passthrough(r, idcol):
    return {"id": r[idcol], "prompt": r["prompt"], "answer": r["answer"],
            "type": r.get("type"), "generated_cot": r.get("generated_cot", "")}


def main():
    tok = Tokenizer.from_file(TOKJSON)
    rows = list(csv.DictReader(open(DGX, encoding="utf-8")))
    idcol = "﻿id" if "﻿id" in rows[0] else "id"

    out_rows = []
    n_nonbit = n_d01 = n_replaced = n_fallback = 0
    depth_dist = {}

    for r in rows:
        if r.get("type") != "bit_manipulation":
            out_rows.append(_passthrough(r, idcol))
            n_nonbit += 1
            continue
        ins, outs, query = S.parse_bit_problem(r["prompt"])
        if query is None:
            out_rows.append(_passthrough(r, idcol))
            n_fallback += 1
            continue
        _trace, predictor, expr, trans = S.solve_dfs_trace_dynamic(
            ins, outs, len(ins), 5.0)
        depth = expr.count("(") if expr else -1
        depth_dist[depth] = depth_dist.get(depth, 0) + 1
        if predictor is None or depth <= 1:
            out_rows.append(_passthrough(r, idcol))
            if predictor is not None and depth <= 1:
                n_d01 += 1
            else:
                n_fallback += 1
            continue
        # depth>=2 -> render essence
        answer = str(r["answer"]).strip().zfill(8)
        trans_t = {k: tuple(v) for k, v in trans.items()}
        cot = E.render_essence_cot(ins, outs, query, expr, trans_t)
        ntok = len(tok.encode(cot).ids)
        if E.selfcheck_cot(cot, query, answer, expr, trans_t) and ntok < TOKEN_LIMIT:
            out_rows.append({"id": r[idcol], "prompt": r["prompt"],
                             "answer": r["answer"], "type": "bit_manipulation",
                             "generated_cot": cot})
            n_replaced += 1
        else:
            out_rows.append(_passthrough(r, idcol))
            n_fallback += 1

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "type",
                                          "generated_cot"])
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"rows total: {len(out_rows)}")
    print(f"  non-bit passthrough: {n_nonbit}")
    print(f"  bit depth0-1 passthrough: {n_d01}")
    print(f"  bit depth>=2 replaced: {n_replaced}")
    print(f"  bit fallback (unsolved/selfcheck-fail/token>=limit): {n_fallback}")
    print(f"bit depth distribution: {dict(sorted(depth_dist.items()))}")
    n_bit = n_d01 + n_replaced + n_fallback
    if n_bit and n_replaced < 0.5 * (n_replaced + n_fallback):
        print("WARNING: depth>=2 replacement rate < 50% — inspect render logic")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行生成**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 datagen/bit_essence_gen.py`
Expected: 打印计数（depth≥2 replaced 应 ~500–600，fallback 应很小），写出 `bit_essence_mixed.csv`。若出现 `WARNING: replacement rate < 50%` 停下查渲染逻辑。

- [ ] **Step 3: 自校验脚本（混合 CSV 里每条精华 CoT 自洽 + token<8192 + depth0-1 原版未动）**

Run:
```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -c "
import sys; sys.path.insert(0,'datagen'); sys.path.insert(0,'research')
import csv; csv.field_size_limit(sys.maxsize)
import bit_essence_lib as E
import majortom_bit_solver as S
from tokenizers import Tokenizer
tok=Tokenizer.from_file('huikang_ref/tokenizer.json')
rows=list(csv.DictReader(open('datagen/bit_essence_mixed.csv')))
ess=[r for r in rows if 'multi-stage composition' in r['generated_cot']]
print('essence rows:', len(ess))
bad=0
for r in ess:
    ins,outs,q=S.parse_bit_problem(r['prompt'])
    _,pred,expr,trans=S.solve_dfs_trace_dynamic(ins,outs,len(ins),5.0)
    ans=str(r['answer']).strip().zfill(8)
    tr={k:tuple(v) for k,v in trans.items()}
    if not E.selfcheck_cot(r['generated_cot'], q, ans, expr, tr): bad+=1; print('SELFCHECK FAIL', r['id'])
    if len(tok.encode(r['generated_cot']).ids)>=8192: bad+=1; print('TOKEN OVER', r['id'])
    if expr.count('(')<2: bad+=1; print('NOT depth>=2', r['id'])
print('FAILURES:', bad)
assert bad==0, 'self-check failed'
print('ALL ESSENCE ROWS OK')
"
```
Expected: `essence rows: ~500-600` + `ALL ESSENCE ROWS OK`。任何 FAIL 停下修。

- [ ] **Step 4: 人工抽查 5 条精华 + 确认总行数==1754 bit + 其余类未动**

Run:
```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -c "
import csv,sys; csv.field_size_limit(sys.maxsize)
rows=list(csv.DictReader(open('datagen/bit_essence_mixed.csv')))
from collections import Counter
print('total rows:', len(rows), '| by type:', dict(Counter(r['type'] for r in rows)))
ess=[r for r in rows if 'multi-stage composition' in r['generated_cot']]
for r in ess[:3]:
    print('====', r['id'], 'answer', r['answer'], '===='); print(r['generated_cot'][:600]); print('...')
"
```
Expected: total 同 dgxchen 原始行数；type 分布与原始一致（bit_manipulation 1754）；精华 CoT 可读。

- [ ] **Step 5: 提交 —— SKIP（git deferred）。**

---

## Task 4: 冻结 Tier1 闸阈值（写进 spec）

**Files:**
- Modify: `/Users/exerzitien/Desktop/nemotron-kaggle/docs/superpowers/specs/2026-06-01-bit-depth2-essence-cot-design.md`

- [ ] **Step 1: 在 spec §7 末尾追加「已冻结 Tier1 判据」块**

在 spec 第 7 节「判据数值在生成完成、各 Tier 执行前冻结写入本文件。」这一行之后追加：

```markdown
### 已冻结 Tier1 判据（2026-06-01，Tier1 eval 提交前写死，出分后只读）

Tier1 在 base 模型上跑两遍 bit_holdout depth≥2 留出集（~150 题，Task 5 产出）：
- **A 遍**：zero-shot（无 demo），得 `acc_zero`。
- **B 遍**：few-shot（2 条精华 CoT demo 前缀），得 `acc_demo`。

判据（同时满足才算「格式可跟随、值得训练」）：
- `acc_demo ≥ 70%`，且
- `acc_demo − acc_zero ≥ +15pp`（demo 带来的真实提升，排除 base 模型本身会做的部分）。

任一不满足 → 格式连 in-context 都跟不动 → **砍掉，不进 Tier2 训练**，省 7h GPU。
（说明：此判据用 base 模型 with/without demo 的「提升量」，比直接对比 V34 的 59.3%（adapter 条件，apples-to-oranges）更干净。59.3% 仅作背景参照。）
```

- [ ] **Step 2: 提交 —— SKIP（git deferred）。**

---

## Task 5: 扩 Layer B depth≥2 留出集到 ~150（去噪 Tier1）

**Files:**
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_holdout_depth2_supplement.py`
- 产物: `datagen/bit_holdout_depth2.csv`

- [ ] **Step 1: 写补充生成脚本（复用 bit_holdout_lib，mint 全新 depth≥2 规则）**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/datagen/bit_holdout_depth2_supplement.py`:

```python
"""bit_holdout_depth2_supplement — mint ~150 道全新（规则不在 train）depth≥2 bit 题，
用于去噪 Tier1 in-context 闸（原 Layer B depth≥2 只 59 题，单次 eval 方差大）。
复用 bit_holdout_lib（上一个 plan 的诊断仪库）。schema: id,prompt,answer,type,layer。
"""
import csv
import os
import random

import bit_holdout_lib as L

SEED = 20260601
N_TARGET = 150
MAX_TRIES = N_TARGET * 200
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "research", "majortom_results.csv")
OUT = os.path.join(_HERE, "bit_holdout_depth2.csv")


def main():
    rng = random.Random(SEED)
    e2f = L.build_expr2func()
    print("loading train signatures (~30s)...")
    train_sigs = L.load_train_signatures(RESULTS, e2f)
    exprs = list(e2f.keys())
    TR = L.S.TRANSFORMATIONS

    out = []
    tries = 0
    while len(out) < N_TARGET and tries < MAX_TRIES:
        tries += 1
        expr = rng.choice(exprs)
        if expr.count("(") < 2:           # 只要 depth>=2
            continue
        used = L.S.get_used_vars(expr)
        trans = {v: rng.choice(TR) for v in used}
        sig = L.signature(L.make_predictor(expr, trans, e2f))
        if sig in train_sigs:             # 必须新颖
            continue
        prob = L.generate_one_problem(expr, trans, rng, e2f)
        if prob is None:                  # 样例须唯一确定该规则
            continue
        out.append((f"hd2_{len(out):03d}", prob["prompt"], prob["answer"]))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "answer", "type", "layer"])
        for pid, prompt, answer in out:
            w.writerow([pid, prompt, answer, "bit_manipulation", "B_d2"])
    print(f"generated {len(out)} depth>=2 novel problems (tries={tries})")
    if len(out) < 100:
        print("WARNING: <100 generated — raise MAX_TRIES or widen depth")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行生成**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 datagen/bit_holdout_depth2_supplement.py`
Expected: `generated 150 depth>=2 novel problems`，写出 `bit_holdout_depth2.csv`。若 <100 WARNING，调大 MAX_TRIES。

- [ ] **Step 3: 自校验（全 depth≥2、全新颖、答案 round-trip）**

Run:
```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 -c "
import sys; sys.path.insert(0,'datagen'); sys.path.insert(0,'research')
import csv
import bit_holdout_lib as L
import majortom_bit_solver as S
e2f=L.build_expr2func(); train=L.load_train_signatures('research/majortom_results.csv', e2f)
rows=list(csv.DictReader(open('datagen/bit_holdout_depth2.csv')))
print('rows:', len(rows))
bad=0
for r in rows:
    ins,outs,q=S.parse_bit_problem(r['prompt'])
    _,pred,expr,_=S.solve_dfs_trace_dynamic(ins,outs,len(ins),5.0)
    if pred is None or expr.count('(')<2: bad+=1; print('not depth>=2/unsolved', r['id']); continue
    if pred(q)!=r['answer']: bad+=1; print('answer mismatch', r['id'])
    if L.signature(pred) in train: bad+=1; print('NOT novel', r['id'])
print('FAILURES:', bad); assert bad==0; print('ALL OK')
"
```
Expected: `rows: ~150` + `ALL OK`。

- [ ] **Step 4: 提交 —— SKIP（git deferred）。**

---

## Task 6: 构建 in-context Tier1 eval kernel（base 模型 + few-shot）

**Files:**
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/kernels/eval-incontext-bit/build_kernel.py`
- 产物: `kernels/eval-incontext-bit/nemotron-eval-incontext-bit.ipynb`
- Create: `/Users/exerzitien/Desktop/nemotron-kaggle/kernels/eval-incontext-bit/kernel-metadata.json`

- [ ] **Step 1: 写 notebook 构建脚本（自包含写出，不派生 eval-bit-holdout 以避开 LoRA 逻辑）**

Create `/Users/exerzitien/Desktop/nemotron-kaggle/kernels/eval-incontext-bit/build_kernel.py`:

```python
"""构建 Tier1 in-context eval notebook：base 模型（无 adapter）+ few-shot 精华 demo，
在 bit_holdout_depth2 留出集上跑两遍（zero-shot / few-shot），按条件出准确率。
复用竞赛 scorer 的 extract_final_answer/verify 语义（内联）。"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(_HERE, "nemotron-eval-incontext-bit.ipynb")

CELLS = [
    # cell 0: install (与诊断仪一致，离线 wheels)
    r'''import os, subprocess, sys
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
if IS_KAGGLE:
    subprocess.run("pip install -q --no-index --find-links "
        "/kaggle/input/datasets/mayukh18/nemotron-packages/packages vllm",
        shell=True, check=False)
print("env ready")''',

    # cell 1: load model (base, no LoRA)
    r'''import kagglehub, torch
from vllm import LLM, SamplingParams
MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
llm = LLM(model=str(MODEL_PATH), tensor_parallel_size=1, max_num_seqs=64,
          gpu_memory_utilization=0.85, dtype="auto", max_model_len=8192,
          trust_remote_code=True, enable_prefix_caching=True,
          enable_chunked_prefill=True)
tokenizer = llm.get_tokenizer()
# Tier1 自有诊断，generation 上限设小（精华 CoT 目标 ~600 tok），省时
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048)
print("base model loaded (no adapter)")''',

    # cell 2: scorer (内联竞赛 extract/verify)
    r'''import re, math
def extract_final_answer(text):
    if text is None: return "NOT_FOUND"
    m = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if m:
        ne = [x.strip() for x in m if x.strip()]
        return ne[-1] if ne else m[-1].strip()
    return "NOT_FOUND"
def verify(stored, pred):
    stored, pred = str(stored).strip(), str(pred).strip()
    try:
        return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return pred.lower() == stored.lower()
print("scorer ready")''',

    # cell 3: load eval set + demos
    r'''import glob, pandas as pd
_ev = glob.glob("/kaggle/input/**/bit_holdout_depth2.csv", recursive=True)
assert _ev, "bit_holdout_depth2.csv not found — add exerzitien/nemotron-bit-holdout"
df = pd.read_csv(_ev[0])
print("eval set:", len(df), "depth>=2 novel problems")

_es = glob.glob("/kaggle/input/**/bit_essence_mixed.csv", recursive=True)
assert _es, "bit_essence_mixed.csv not found — add exerzitien/nemotron-bit-essence"
ess = pd.read_csv(_es[0])
ess = ess[ess["generated_cot"].astype(str).str.contains("multi-stage composition")]
# 2 条确定性 demo（按 id 排序取前 2），其 reasoning 去掉 </think> 当示范文本
demos = ess.sort_values("id").head(2).to_dict("records")
def demo_text(d):
    cot = str(d["generated_cot"]).replace("</think>\n", "").replace("</think>", "")
    return f"Problem:\n{d['prompt']}\nSolution:\n{cot}"
DEMO_BLOCK = "\n\n".join(demo_text(d) for d in demos)
print("demos chosen:", [d["id"] for d in demos])''',

    # cell 4: build prompts (zero-shot + few-shot) and generate
    r'''SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
def build(prompt, few_shot):
    if few_shot:
        user = ("Here are worked examples using the multi-stage composition method:\n\n"
                + DEMO_BLOCK
                + "\n\nNow solve this one using the same method:\n" + prompt + SUFFIX)
    else:
        user = prompt + SUFFIX
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)

def run(few_shot):
    prompts = [build(p, few_shot) for p in df["prompt"]]
    outs = llm.generate(prompts, sampling_params=sampling_params)
    preds = [extract_final_answer(o.outputs[0].text) for o in outs]
    correct = sum(verify(a, p) for a, p in zip(df["answer"], preds))
    return correct, len(df), preds

c0, n, _ = run(False)
c1, _, _ = run(True)
acc_zero = 100*c0/n; acc_demo = 100*c1/n
print(f"acc_zero (no demo):  {c0}/{n} = {acc_zero:.1f}%")
print(f"acc_demo (few-shot): {c1}/{n} = {acc_demo:.1f}%")
print(f"LIFT: {acc_demo-acc_zero:+.1f}pp")
import pandas as pd
pd.DataFrame([{"acc_zero":acc_zero,"acc_demo":acc_demo,"lift":acc_demo-acc_zero,"n":n}]).to_csv("results.csv", index=False)
print("GATE: pass if acc_demo>=70 and lift>=+15pp")''',
]


def main():
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": c} for c in CELLS],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                      "language": "python"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
    print(f"Wrote {DST} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行构建 + 校验 JSON 合法**

Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && python3 kernels/eval-incontext-bit/build_kernel.py && python3 -c "import json; nb=json.load(open('kernels/eval-incontext-bit/nemotron-eval-incontext-bit.ipynb')); print('cells:', len(nb['cells']))"`
Expected: `Wrote ...` + `cells: 5`

- [ ] **Step 3: 写 kernel-metadata.json**

先读诊断仪 metadata 作蓝本：
Run: `cd /Users/exerzitien/Desktop/nemotron-kaggle && cat kernels/eval-bit-holdout/kernel-metadata.json`

Create `/Users/exerzitien/Desktop/nemotron-kaggle/kernels/eval-incontext-bit/kernel-metadata.json`:

```json
{
  "id": "exerzitien/nemotron-eval-incontext-bit",
  "title": "Nemotron Eval Incontext Bit",
  "code_file": "nemotron-eval-incontext-bit.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "keywords": ["gpu"],
  "dataset_sources": [
    "mayukh18/nemotron-packages",
    "exerzitien/nemotron-bit-holdout",
    "exerzitien/nemotron-bit-essence"
  ],
  "kernel_sources": [],
  "competition_sources": [
    "nvidia-nemotron-model-reasoning-challenge"
  ],
  "model_sources": [
    "metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1"
  ],
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9fa0da194fad2241d3f01a80581cbecbd3a258b4d1b695e2cbbbc62a0fd205ac",
  "machine_shape": "NvidiaRtxPro6000"
}
```

> 注：`docker_image` 与 `model_sources` 照抄 eval-bit-holdout 的值（Step 3 的 cat 输出为准；若不一致以 cat 输出为准）。无 LoRA、无 adapter dataset。

- [ ] **Step 4: 提交 —— SKIP（git deferred）。**

---

## Task 7【门控·消耗 Kaggle 资源】上传数据集 + 跑 Tier1 + 取读数判决

**前置：以下步骤上传数据 + 跑 ~1h GPU eval，必须先经用户确认再执行。** 鉴权：`KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token)`。

- [ ] **Step 1: 把 bit_holdout_depth2.csv 加进 nemotron-bit-holdout 数据集（新版本）**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
rm -rf /tmp/bh-ds && mkdir -p /tmp/bh-ds
cp datagen/bit_holdout_A.csv datagen/bit_holdout_B.csv datagen/bit_holdout_depth2.csv /tmp/bh-ds/
cat > /tmp/bh-ds/dataset-metadata.json <<'EOF'
{"id": "exerzitien/nemotron-bit-holdout", "title": "nemotron-bit-holdout", "licenses": [{"name": "CC0-1.0"}]}
EOF
KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token) \
  kaggle datasets version -p /tmp/bh-ds -m "add depth2 supplement"
```
Expected: 新版本上传成功。

- [ ] **Step 2: 上传 bit_essence_mixed.csv 为新数据集 nemotron-bit-essence**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
rm -rf /tmp/ess-ds && mkdir -p /tmp/ess-ds
cp datagen/bit_essence_mixed.csv /tmp/ess-ds/
cat > /tmp/ess-ds/dataset-metadata.json <<'EOF'
{"id": "exerzitien/nemotron-bit-essence", "title": "nemotron-bit-essence", "licenses": [{"name": "CC0-1.0"}]}
EOF
KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token) \
  kaggle datasets create -p /tmp/ess-ds
```
Expected: 创建成功（若已存在改用 `datasets version -p /tmp/ess-ds -m update`）。

- [ ] **Step 3: 推 Tier1 kernel**

```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle
KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token) \
  kaggle kernels push -p kernels/eval-incontext-bit
```
Expected: 推送成功，进入 RUNNING。

- [ ] **Step 4: 轮询直至 COMPLETE**

```bash
KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token) \
  kaggle kernels status exerzitien/nemotron-eval-incontext-bit
```
Expected: 最终 `COMPLETE`。若 ERROR：取 log（Bearer + 预签名 URL，见 [[kaggle-cli-bearer-output]]），常见原因=few-shot prompt 超 max_model_len（调小 demo 数或 max_tokens）。

- [ ] **Step 5: 取 results.csv（acc_zero / acc_demo / lift）**

```bash
KGAT=$(tr -d '[:space:]' < ~/.kaggle/access_token)
curl -s -H "Authorization: Bearer $KGAT" \
  "https://www.kaggle.com/api/v1/kernels/output?userName=exerzitien&kernelSlug=nemotron-eval-incontext-bit" \
  -o /tmp/t1.json
python3 -c "
import json
d=json.load(open('/tmp/t1.json'))
log=d.get('log') or d.get('logNullable') or ''
if isinstance(log,list): log=''.join(x.get('data','') if isinstance(x,dict) else str(x) for x in log)
for l in log.splitlines():
    if 'acc_zero' in l or 'acc_demo' in l or 'LIFT' in l: print(l[-120:])
"
```
Expected：拿到 acc_zero / acc_demo / lift 三个数。

- [ ] **Step 6: 对冻结判据判决（Task 4 的阈值）+ 写 memory**

判据：`acc_demo ≥ 70%` 且 `acc_demo − acc_zero ≥ +15pp` → 格式可跟随，**进 Tier2 训练**（单独决策，下一个 plan）。否则 → 砍，不烧 7h。把读数 + 判决更新进 [[nemotron-bit-holdout-diagnostic]] / 总入口 memory。

---

## Self-Review

**1. Spec coverage（逐节核对）:**
- §2 关键结构事实（DAG/eval_tree） → Task 1（parse_expr/eval_tree/count_gates + 外延等价测试）✓
- §4 可学性三原则（复用原子/砍搜索/抗橡皮图章） → Task 2 render（样例向量级 + query 逐位强制）✓
- §5 架构（分流 depth0-1 保留 / depth≥2 换） → Task 3 gen ✓
- §6.1 lib 六函数 → Task 1（parse_expr/eval_tree/count_gates）+ Task 2（topo_steps/render_essence_cot/selfcheck_cot）✓
- §6.2 gen 编排 → Task 3 ✓
- §6.3 in-context kernel（base/无 adapter/few-shot/depth≥2 子集） → Task 6 ✓
- §6.4 Tier2 复用 → 不在本 plan（明确单独决策）✓
- §7 四道闸（Tier0 自洽 baked-in / 人工 20 条 / Tier1 GPU 闸 / Tier2 裁决） → Tier0=Task3 Step3；人工=Task2 Step5 + Task3 Step4；Tier1=Task6+Task7；Tier2=下一 plan ✓
- §7 阈值冻结 → Task 4 ✓
- §9 错误处理（unsolved/selfcheck-fail/token≥limit 回退；替换率<50% 报警） → Task 3 gen ✓
- §10 测试 → Task1/2 单元测试 + Task3 自校验 + 人工抽查 + Tier1 ✓
- §11 风险（样本量 → 补 ~150） → Task 5 ✓

**2. Placeholder scan:** Task 1 Step 1 是**故意的占位测试**，Step 3 显式整体替换为最终版——非残留占位符。其余无 TBD/TODO。Task 4 的阈值是显式冻结动作。✓

**3. Type consistency:** lib 函数签名贯穿一致——`parse_expr(str)->tree`、`eval_tree(tree, vecs{'{A}':...})`、`count_gates(tree)`、`topo_steps(tree)->(steps,root)`、`render_essence_cot(ins,outs,query,expr,trans)->str`、`selfcheck_cot(cot,query,answer,expr,trans)->bool`，在 Task1/2 定义、Task3 调用一致；`trans` 全程 `{'{A}':('shl',1)}` 形（tuple 值）；CSV schema `id,prompt,answer,type,generated_cot` 在 gen 写出与训练 kernel 读取一致；holdout schema `id,prompt,answer,type,layer` 与诊断仪一致。✓

**已知软约束（spec §11 已记）：** Tier1 in-context 可跟随是 SFT 可学性的强先行指标、非充分条件——只用来「砍」，最终仍靠 Tier2 诊断仪裁决。
