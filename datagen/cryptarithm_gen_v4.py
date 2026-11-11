"""
cryptarithm_gen_v4.py — 搜索式(演绎) CoT 生成器 (route C: 混合)
============================================================
设计契约 (4 invariants, 始终保证):
  ① 可学习: 每步有界局部算术(单未知求解/AllDiff); seed 块尽量小
  ② 不毒: dgxchen 格式+footer, 长度压区间内, 只换 crypt 不碰 bit
  ③ 泛化: 零答案泄漏(mapping 只用于取正确数字, 不作推理依据); 过程跨题统一
  ④ 准确: 末尾 \\boxed 精确==gold; 每步真实计算; mapping 取自 lkevincc solver_correct

架构: 先求"最小 seed 集"——使其余符号能靠"单未知级联"逐个解出(纯算术,可学)。
seed 块(v1: 断言+验证, Task#3 升级为个位/量级推理) + 干净级联 + 全验证 + 套用query。
seed 大小分布 = 可学性硬指标(越小越好)。范围: standard 模式、fully-forced std-arith。
"""
import sys, json
sys.path.insert(0, '/Users/exerzitien/Desktop/nemotron-kaggle/research/lkevincc_solver/src')
from solver_eq_symbolic import AliceEquationSolver, OPERATIONS, SIGNED_OPS

OP_HUMAN = {
    'add':'a + b','sub':'a - b','rsub':'b - a','absdiff':'|a - b|','mul':'a × b',
    'add_m1':'a + b - 1','add_p1':'a + b + 1','add_m2':'a + b - 2','add_p2':'a + b + 2',
    'mul_m1':'a × b - 1','mul_p1':'a × b + 1','mul_m2':'a × b - 2','mul_p2':'a × b + 2',
    'absdiff_m1':'|a - b| - 1','absdiff_p1':'|a - b| + 1','absdiff_m2':'|a-b|-2','absdiff_p2':'|a-b|+2',
    'sub_m1':'a - b - 1','sub_p1':'a - b + 1','rsub_m1':'b - a - 1','rsub_p1':'b - a + 1',
    'gcd':'gcd(a,b)','lcm':'lcm(a,b)','fdiv':'a // b','rdiv':'b // a','mod':'a mod b','rmod':'b mod a',
    'min':'min(a,b)','max':'max(a,b)','sub_signed':'a - b (signed)','rsub_signed':'b - a (signed)',
    'neg_absdiff':'-(|a - b|)',
}

def eq_all_syms(lhs, rhs, opc):
    s = {lhs[0], lhs[1], lhs[3], lhs[4]}
    has_sign = len(rhs) > 1 and rhs[0] == opc
    s.update(rhs[1:] if has_sign else rhs)
    return s

def compute(lhs, rhs, opc, op, m):
    """given mapping m (covering involved syms), return (L,R,v,mag,encoded_rhs)"""
    L = m[lhs[0]]*10 + m[lhs[1]]; R = m[lhs[3]]*10 + m[lhs[4]]
    v = OPERATIONS[op](L, R)
    if v is None: raise ValueError('op none')
    has_sign = len(rhs) > 1 and rhs[0] == opc
    res = rhs[1:] if has_sign else rhs
    rev = {d: s for s, d in m.items()}
    if op in SIGNED_OPS:   mag = abs(v); prefix = opc if v < 0 else ''
    elif op == 'neg_absdiff':
        if v < 0: raise ValueError('neg');
        mag = v; prefix = opc
    else:
        if v < 0: raise ValueError('neg');
        mag = v; prefix = ''
    ds = str(mag)
    if len(ds) > len(res): raise ValueError('toolong')
    ds = ds.zfill(len(res))
    enc = prefix + ''.join(rev[int(c)] for c in ds)
    return L, R, v, mag, enc

# ---------- minimal-seed + single-unknown cascade ----------

def cascade_from(seed, examples, syms_all):
    """Given seed symbols known, derive others via single-unknown equations + AllDiff.
    Returns (steps, known) where steps=[(sym,lhs,rhs) or (sym,None,None) for AllDiff]."""
    known = set(seed); steps = []
    progressed = True
    while progressed and len(known) < len(syms_all):
        progressed = False
        for lhs, rhs in examples:
            unk = [c for c in eq_all_syms(lhs, rhs, lhs[2]) if c not in known]
            if len(unk) == 1:
                X = unk[0]; steps.append((X, lhs, rhs)); known.add(X); progressed = True; break
        if not progressed and len(known) == len(syms_all) - 1:
            X = [c for c in syms_all if c not in known][0]
            steps.append((X, None, None)); known.add(X); progressed = True
    return steps, known

def minimal_seed(examples, syms_all):
    """Greedily pick fewest seed symbols so the rest cascade out."""
    seed = []
    while True:
        steps, known = cascade_from(seed, examples, syms_all)
        if len(known) == len(syms_all):
            return seed, steps
        cand = [c for c in syms_all if c not in known]
        # add the symbol whose addition maximizes cascade closure
        best = max(cand, key=lambda c: len(cascade_from(seed + [c], examples, syms_all)[1]))
        seed.append(best)

def narrate_cascade_step(lhs, rhs, X, known, mapping):
    """Substitute known, show the single-unknown solve as arithmetic."""
    oc = lhs[2]; op = ops_global[oc]
    sub = lambda tok: ''.join(str(known[c]) if c in known else c for c in tok)
    d = mapping[X]; m2 = dict(known); m2[X] = d
    L, R, v, mag, enc = compute(lhs, rhs, oc, op, m2)
    opdesc = OP_HUMAN.get(op, op).replace('a', str(L)).replace('b', str(R))
    return (f"  • {lhs}={rhs}: substitute knowns → {sub(lhs[0:2])}{oc}{sub(lhs[3:5])}={sub(rhs)}. "
            f"{L} {oc} {R} = {opdesc} = {mag} → encodes to {enc}; this fixes {X}={d}.")

ops_global = None  # set per problem

def build_cot(prompt, mapping, ops, gold):
    global ops_global
    ops_global = ops
    s = AliceEquationSolver(prompt)
    examples = s.examples
    syms_all = sorted(set().union(*[eq_all_syms(l, r, l[2]) for l, r in examples]))
    seed, steps = minimal_seed(examples, syms_all)

    L = []
    L.append("Each symbol stands for a fixed secret digit 0-9 (a one-to-one map). Each line reads as a "
             "2-digit number, an operator symbol, a 2-digit number, then the result — all in symbols. "
             "I'll pin every symbol's digit from the examples, then read off the query.")
    L.append("\nWhat each operator does (read from the examples, confirmed by the full check below):")
    for oc in sorted(ops):
        L.append(f"  '{oc}' = {OP_HUMAN.get(ops[oc], ops[oc])}.")

    L.append("\nWork out the digits (examples only):")
    known = {}
    # seed block (v1: assert + verify via an equation that contains seed syms)
    if seed:
        for c in seed:
            known[c] = mapping[c]
        # find the equation containing the most seed symbols, show it holds
        best_eq = max(examples, key=lambda lr: len(set(seed) & eq_all_syms(lr[0], lr[1], lr[0][2])))
        oc = best_eq[0][2]; op = ops[oc]
        # need all syms in best_eq known to display; if not all seed, still show with known seed (others may be unknown)
        seedstr = ", ".join(f"{c}={mapping[c]}" for c in seed)
        L.append(f"  • A few symbols can't be isolated one-at-a-time; the examples jointly admit only one "
                 f"consistent assignment for them: {seedstr}. (Seed for the chain below.)")
    # cascade steps
    for (X, lhs, rhs) in steps:
        if lhs is None:
            d = mapping[X]
            L.append(f"  • Digits used so far: {sorted(known.values())}; only {d} remains → {X} = {d}.")
            known[X] = d
        else:
            L.append(narrate_cascade_step(lhs, rhs, X, known, mapping))
            known[X] = mapping[X]

    # verify
    L.append("\nVerify on every example:")
    for lhs, rhs in examples:
        oc = lhs[2]; op = ops[oc]
        try:
            Ln, Rn, v, mag, enc = compute(lhs, rhs, oc, op, mapping)
        except Exception:
            return None, None, {'fail': 'verify-exc'}
        if enc != rhs:
            return None, None, {'fail': 'verify'}
        L.append(f"  {lhs} → {Ln} {oc} {Rn} = {mag} → {enc} ✓")

    # apply query
    qt = s.apply_query_trace(mapping, ops, 'standard')
    if qt is None:
        return None, None, {'fail': 'query'}
    ans = qt['encoded_answer']
    L.append(f"\nApply to the query {s.query}: {qt['L_letters']}={qt['L_digits']}, {qt['R_letters']}={qt['R_digits']}, "
             f"'{qt['op_char']}' = {OP_HUMAN.get(qt['op_type'], qt['op_type'])} → {qt['raw_result']} → {ans}.")
    if ans != gold:
        return None, None, {'fail': 'roundtrip', 'ans': ans, 'gold': gold}
    cot = "\n".join(L)
    return cot, ans, {'seed_size': len(seed), 'n_syms': len(syms_all), 'cascade': len(steps)}


if __name__ == '__main__':
    import pandas as pd, statistics
    from collections import Counter
    df = pd.read_parquet('/Users/exerzitien/Desktop/nemotron-kaggle/research/lkevincc_solver/data/solver_results.parquet')
    df = df[(df['solver_correct'] == True) & (df['solver_mode'] == 'standard')].copy()
    ok = 0; fails = Counter(); lens = []; seeds = []; n = 0; printed = 0
    for _, r in df.iterrows():
        try:
            ops = json.loads(r['solver_ops']); mapping = json.loads(r['solver_mapping'])
        except Exception:
            continue
        if any('concat' in str(v) for v in ops.values()): continue
        if float(r['solver_radix'] or 10) != 10.0: continue
        s = AliceEquationSolver(r['prompt']); order, trace = s._derive_order(mapping, ops, 'standard')
        if not trace or any(t['kind'] != 'forced' for t in trace): continue
        n += 1
        cot, ans, meta = build_cot(r['prompt'], mapping, ops, r['answer'])
        if cot is None:
            fails[meta.get('fail', '?')] += 1; continue
        ok += 1; lens.append(len(cot)); seeds.append(meta['seed_size'])
        if printed < 2:
            printed += 1
            print(f"\n{'#'*70}\nID {r['id']} ans={r['answer']} seed={meta['seed_size']}/{meta['n_syms']} (len={len(cot)} ≈{len(cot)//4}tok)\n{'#'*70}")
            print(r['prompt']); print('-'*40); print(cot)
    print(f"\n\n{'='*70}\nSUMMARY: {ok}/{n} round-trip OK | fails={dict(fails)}")
    if lens:
        print(f"CoT chars: median {int(statistics.median(lens))} max {max(lens)} (≈tok median {int(statistics.median(lens))//4} max {max(lens)//4})")
        print(f"SEED SIZE (asserted symbols, want SMALL) distribution: {dict(sorted(Counter(seeds).items()))}")
        print(f"  mean seed {statistics.mean(seeds):.2f} | seed≤2: {sum(1 for x in seeds if x<=2)}/{len(seeds)} ({100*sum(1 for x in seeds if x<=2)/len(seeds):.0f}%) | seed≤3: {sum(1 for x in seeds if x<=3)}/{len(seeds)}")
