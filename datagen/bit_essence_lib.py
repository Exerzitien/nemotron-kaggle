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

    L = ["Single-gate matching fails; the rule composes shifted copies of the 8-bit input.",
         "(A', B', C' are all the SAME input, each transformed by one shift/rotate.)"]
    for v in used_vars:
        L.append(f"{v}' = input.{_tname(vt[v])}")
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
