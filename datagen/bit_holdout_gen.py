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
