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
