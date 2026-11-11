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
