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
        # solver 回收的「最简」规则必须本身 depth>=2：有些 depth>=2 表达式在
        # 给定 transformation 下外延塌缩成更简单(depth<2)的规则，simple-first
        # 求解器会回收那个更简形。这类题对模型不是真正的 depth>=2 题，剔除。
        ins, outs, q = L.S.parse_bit_problem(prob["prompt"])
        _t, pred, rexpr, _tr = L.S.solve_dfs_trace_dynamic(
            ins, outs, len(ins), 5.0)
        if pred is None or rexpr.count("(") < 2:
            continue
        if pred(q) != prob["answer"]:     # 答案须可被回收规则复现
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
