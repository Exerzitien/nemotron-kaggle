"""bit_generate_augmented.py — STAGE 2: 生成全量增强 bit CoT 数据集。

策略(承接 bit_verify_augment.py STAGE 1 已验证):
- 仅对"规则可重现全部样例(ok_ex)"的题做增强 -> verify 段全 match,信号干净。
- 这些干净题中按 ~SELF_CORRECT_RATIO 注入自纠错轨迹(仅限有 decoy 可注入的题)。
- 其余(规则未能重现全样例 / 解析失败 / round-trip 失败)保留原 CoT 不动 -> 不引入坏信号。
- 输出 schema 与训练 kernel 期望一致: id,prompt,answer,type,generated_cot; type='bit_manipulation'。
- 每条增强后做 round-trip 自校验(\\boxed{gold} 存在 且 apply_vector(query)==gold),失败则回退原 CoT。
"""
import random
import pandas as pd
from bit_verify_augment import (
    parse_prompt, parse_selected, apply_vector, augment_cot, find_decoy,
)

SEED = 42
SELF_CORRECT_RATIO = 0.30
IN = "datagen/_ref/problem_ids_matched.csv"
OUT = "datagen/augmented_bit.csv"


def main():
    rng = random.Random(SEED)
    df = pd.read_csv(IN)
    bit = df[df["type"] == "bit_manipulation"].reset_index(drop=True)

    # 第一遍: 分类每行 -> clean(可增强) / keep(保留原版)
    rows = []          # (idx, cot, gold, ins, outs, query, vec)  clean 候选
    keep_idx = []      # 保留原版的行 idx
    for i, r in bit.iterrows():
        cot = str(r["generated_cot"]); gold = str(r["answer"]).strip().zfill(8)
        ins, outs, query = parse_prompt(str(r["prompt"])); vec = parse_selected(cot)
        if vec is None or query is None:
            keep_idx.append(i); continue
        if apply_vector(query, vec) != gold:          # round-trip 失败
            keep_idx.append(i); continue
        if not all(apply_vector(ins[k], vec) == outs[k] for k in range(len(ins))):
            keep_idx.append(i); continue              # 规则不能重现全样例 -> 保留原版
        rows.append((i, cot, gold, ins, outs, query, vec))

    # 在 clean 行里挑 ~ratio 注入自纠错(仅限有 decoy 可注入的)
    injectable = []
    for (i, cot, gold, ins, outs, query, vec) in rows:
        sc_bit = next((j for j in range(8) if find_decoy(ins, outs, vec, j)), None)
        injectable.append(sc_bit)   # None = 不可注入
    inj_positions = [k for k, b in enumerate(injectable) if b is not None]
    rng.shuffle(inj_positions)
    n_sc = int(round(SELF_CORRECT_RATIO * len(rows)))
    sc_set = set(inj_positions[:n_sc])

    out_records = []
    n_aug = n_sc_used = n_verify_only = n_keep = n_rt_fallback = 0
    for k, (i, cot, gold, ins, outs, query, vec) in enumerate(rows):
        if k in sc_set:
            jbit = injectable[k]; decoy = find_decoy(ins, outs, vec, jbit)
            aug = augment_cot(cot, ins, outs, vec, with_self_correct_bit=jbit, decoy=decoy)
        else:
            aug = augment_cot(cot, ins, outs, vec)
        # round-trip 自校验: 失败回退原 CoT
        if f"\\boxed{{{gold}}}" not in aug and apply_vector(query, vec) != gold:
            aug = cot; n_rt_fallback += 1; keep_idx.append(i)
            continue
        r = bit.iloc[i]
        out_records.append({"id": r["id"], "prompt": r["prompt"], "answer": r["answer"],
                            "type": "bit_manipulation", "generated_cot": aug})
        n_aug += 1
        if k in sc_set: n_sc_used += 1
        else: n_verify_only += 1

    for i in keep_idx:
        r = bit.iloc[i]
        out_records.append({"id": r["id"], "prompt": r["prompt"], "answer": r["answer"],
                            "type": "bit_manipulation", "generated_cot": str(r["generated_cot"])})
        n_keep += 1

    out = pd.DataFrame(out_records)[["id", "prompt", "answer", "type", "generated_cot"]]
    assert len(out) == len(bit), f"row count drift {len(out)} != {len(bit)}"
    # 源数据本身有重复 id(1754 行/1354 唯一),训练 kernel 按行读取不要求 id 唯一;
    # 仅核对增强后唯一 id 数与源一致(没丢/没造行)。
    assert out["id"].nunique() == bit["id"].nunique(), \
        f"unique-id drift {out['id'].nunique()} != {bit['id'].nunique()}"
    out.to_csv(OUT, index=False)

    # 全量再校验: 每条最终 cot 的 \boxed 是否 == gold(增强行)
    final = pd.read_csv(OUT)
    box_ok = 0
    import re as _re
    for _, r in final.iterrows():
        gold = str(r["answer"]).strip().zfill(8)
        m = _re.search(r"\\boxed\{([01]{8})\}", str(r["generated_cot"]))
        if m and m.group(1) == gold: box_ok += 1
    print(f"输出 {OUT}: {len(out)} 行 (bit 总 {len(bit)})")
    print(f"  增强行: {n_aug} (自纠错 {n_sc_used} / 纯验证 {n_verify_only})")
    print(f"  保留原版: {n_keep}  (含 round-trip 回退 {n_rt_fallback})")
    print(f"  最终 \\boxed==gold: {box_ok}/{len(final)} ({100*box_ok/len(final):.1f}%)")
    print(f"  自纠错占增强行比例: {100*n_sc_used/max(n_aug,1):.1f}% (目标 {100*SELF_CORRECT_RATIO:.0f}%)")
    # token 粗估
    lens = [len(str(c))//4 for c in final["generated_cot"]]
    lens.sort()
    print(f"  token 估计 中位≈{lens[len(lens)//2]} max≈{lens[-1]}")


if __name__ == "__main__":
    main()
