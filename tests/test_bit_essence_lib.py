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
    """eval_tree 必须复现 solver 在真实题上恢复的 predictor（对该题 query）。"""
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
        vecs = {var: [S.get_source_bit(query, b, trans.get(var, ("rot", 0)))
                      for b in range(8)]
                for var in ("{A}", "{B}", "{C}")}
        got = "".join(str(x) for x in E.eval_tree(tree, vecs))
        tested += 1
        ok += (got == predictor(query))
    assert tested >= 40
    assert ok == tested


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
    assert E.selfcheck_cot(cot, query, answer, expr, trans) is True
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
    assert E.selfcheck_cot(bad, query, "00000000", expr, trans) is False
