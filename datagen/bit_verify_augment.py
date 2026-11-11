"""bit_verify_augment.py — 在 THK 工作版 bit CoT 上原地追加"全样例自验证"(+可选自纠错轨迹)。
策略: 不重写格式(v4 那样会崩)。解析现有 CoT 的 'Selected' 段拿到每位规则,
用 THK 同款求值在 query 上 round-trip 验证解析正确, 再渲染验证段插到 \\boxed 前。
STAGE 1 (本文件当前): 仅 parse + round-trip 验证, 确认解析 100% 正确。
"""
import re, sys

FAMILIES = ["XOR-NOT", "OR-NOT", "AND-NOT", "XOR", "OR", "AND", "NOT", "I"]  # 长在前避免前缀冲突


def bit_not(b): return "1" if b == "0" else "0"


def eval_binary(a, b, fam):
    if fam in ("AND", "AND-NOT"): return "1" if a == "1" and b == "1" else "0"
    if fam in ("OR", "OR-NOT"):   return "1" if a == "1" or b == "1" else "0"
    if fam in ("XOR", "XOR-NOT"): return "1" if a != b else "0"
    raise ValueError(fam)


def eval_rule(bits, fam, p, s):
    """bits=8-char input; (fam,p,s)=该输出位规则; 返回输出位 '0'/'1'."""
    if fam == "0": return "0"
    if fam == "1" or fam == "DEFAULT": return "1"
    if fam == "I": return bits[p]
    if fam == "NOT": return bit_not(bits[p])
    a = bits[p]; b = bits[s]
    if "-NOT" in fam: b = bit_not(b)
    return eval_binary(a, b, fam)


def parse_token(tok):
    """ 'XOR-NOT45' -> ('XOR-NOT',4,5); 'OR-NOT07'->('OR-NOT',0,7); 'I3'->('I',3,None);
        'NOT5'->('NOT',5,None); '0'/'1'->const; 'default'->DEFAULT. """
    tok = tok.strip()
    if tok in ("0", "C0"): return ("0", None, None)
    if tok in ("1", "C1"): return ("1", None, None)
    if tok.lower().startswith("default"): return ("DEFAULT", None, None)
    for fam in FAMILIES:
        if tok.startswith(fam):
            digits = tok[len(fam):]
            ds = [int(c) for c in digits if c.isdigit()]
            if fam in ("I", "NOT"):
                return (fam, ds[0], None) if ds else (None, None, None)
            return (fam, ds[0], ds[1]) if len(ds) >= 2 else (None, None, None)
    return (None, None, None)


def parse_prompt(prompt):
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    ins = [p[0] for p in pairs]; outs = [p[1] for p in pairs]
    # query = 最后一个孤立 8-bit (出现在 'determine the output for' 之后, 无 '->')
    q = re.findall(r"(?:output for[:\s]*)\s*([01]{8})", prompt)
    query = q[-1] if q else None
    if query is None:  # 回退: 末尾孤立 8-bit
        allb = re.findall(r"[01]{8}", prompt)
        paired = set(ins) | set(outs)
        leftover = [b for b in allb if b not in paired]
        query = leftover[-1] if leftover else None
    return ins, outs, query


def parse_selected(cot):
    """抓 'Selected' 段后的 8 行 'i TOKEN'。返回 [(fam,p,s)]*8 或 None。"""
    m = re.search(r"\nSelected\n(.*?)(?:\n\n|\nMatched|\nApplying|\Z)", cot, re.S)
    if not m:
        # 回退: 'Matched' 段
        m = re.search(r"\nMatched\n(.*?)(?:\n\n|\nApplying|\Z)", cot, re.S)
    if not m: return None
    rules = {}
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*([0-7])\s+(.+)", line.strip())
        if mm:
            fam, p, s = parse_token(mm.group(2))
            if fam is None: return None
            rules[int(mm.group(1))] = (fam, p, s)
    if len(rules) != 8: return None
    return [rules[i] for i in range(8)]


def apply_vector(bits, vector):
    return "".join(eval_rule(bits, f, p, s) for (f, p, s) in vector)


def tok_str(fam, p, s):
    if fam in ("0", "1", "DEFAULT"): return {"0": "C0", "1": "C1", "DEFAULT": "C1"}[fam]
    if fam in ("I", "NOT"): return f"{fam}{p}"
    return f"{fam}{p}{s}"


def render_verify(ins, outs, vec):
    """全样例自验证段(紧凑): 套规则到每个输入, 确认重现输出。"""
    L = ["", "Verify the selected rule reproduces every example before answering:"]
    allok = True
    for inp, out in zip(ins, outs):
        got = apply_vector(inp, vec)
        ok = got == out
        allok &= ok
        L.append(f"{inp} -> {got}  (expected {out})  {'match' if ok else 'MISMATCH'}")
    L.append("All examples reproduced." if allok else "Some example failed -- rule is wrong, must revisit.")
    return L


def find_decoy(ins, outs, vec, j):
    """为输出位 j 找一个'近似但错'的规则: 在某个样例的 bit j 上出错。返回 (fam,p,s) 或 None。"""
    cf, cp, cs = vec[j]
    cands = []
    if cp is not None and cs is not None:
        for fam in ("XOR", "OR", "AND", "XOR-NOT", "OR-NOT", "AND-NOT"):
            if fam != cf: cands.append((fam, cp, cs))         # 换算符
        cands.append((cf, cp, (cs + 1) % 8))                  # 偏移挪一位
        cands.append((cf, (cp + 1) % 8, cs))
    for (fam, p, s) in cands:
        # 必须: 至少一个样例 bit j 出错(=能被验证抓到)
        if any(eval_rule(ins[k], fam, p, s) != outs[k][j] for k in range(len(ins))):
            return (fam, p, s)
    return None


def render_self_correct(ins, outs, vec, j, decoy):
    """自纠错轨迹: 先用 decoy(错) -> 验证抓到某样例 bit j 不符 -> 改回正确 -> 再验证 ✓。"""
    df, dp, ds = decoy
    # 找第一个出错样例
    kbad = next(k for k in range(len(ins)) if eval_rule(ins[k], df, dp, ds) != outs[k][j])
    got = eval_rule(ins[kbad], df, dp, ds); exp = outs[kbad][j]
    cf, cp, cs = vec[j]
    L = ["", f"Double-check bit {j}. First guess was {tok_str(df, dp, ds)}; test it on the examples:"]
    L.append(f"  example {ins[kbad]}: bit {j} -> {tok_str(df, dp, ds)} gives {got}, but expected {exp} -- MISMATCH.")
    L.append(f"  So bit {j} is not {tok_str(df, dp, ds)}. Re-derive from the columns: bit {j} = {tok_str(cf, cp, cs)}.")
    L.append(f"  Recheck bit {j} on every example: now all match.")
    return L


def augment_cot(cot, ins, outs, vec, with_self_correct_bit=None, decoy=None):
    """在 'Applying to' 前插入验证段(+可选自纠错)。返回新 cot。"""
    anchor = cot.find("Applying to ")
    if anchor < 0:
        anchor = cot.find("I will now return the answer")
    head = cot[:anchor].rstrip()
    tail = cot[anchor:]
    blocks = []
    if with_self_correct_bit is not None and decoy is not None:
        blocks += render_self_correct(ins, outs, vec, with_self_correct_bit, decoy)
    blocks += render_verify(ins, outs, vec)
    return head + "\n" + "\n".join(blocks) + "\n\n" + tail


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("datagen/_ref/problem_ids_matched.csv")
    bit = df[df["type"] == "bit_manipulation"].reset_index(drop=True)
    n = ok_parse = ok_rt = ok_ex = 0
    fails = []
    for _, r in bit.iterrows():
        n += 1
        cot = str(r["generated_cot"]); prompt = str(r["prompt"]); gold = str(r["answer"]).strip().zfill(8)
        ins, outs, query = parse_prompt(prompt)
        vec = parse_selected(cot)
        if vec is None or query is None:
            fails.append((r["id"], "parse")); continue
        ok_parse += 1
        # round-trip: 套用到 query == gold ?
        pred = apply_vector(query, vec)
        if pred == gold: ok_rt += 1
        else: fails.append((r["id"], f"rt {pred}!={gold}")); continue
        # 验证: 套用到每个样例 == 样例输出 ?
        if all(apply_vector(ins[i], vec) == outs[i] for i in range(len(ins))):
            ok_ex += 1
        else:
            fails.append((r["id"], "ex-mismatch"))
    print(f"bit rows: {n}")
    print(f"parse OK:        {ok_parse}/{n} ({100*ok_parse/n:.1f}%)")
    print(f"round-trip(query==gold): {ok_rt}/{n} ({100*ok_rt/n:.1f}%)")
    print(f"rule reproduces all examples: {ok_ex}/{n} ({100*ok_ex/n:.1f}%)")

    # ---- 增强后长度 + round-trip 不变 检查 (全量) ----
    import statistics
    orig_lens, aug_lens, aug_rt_ok, sc_possible = [], [], 0, 0
    for _, r in bit.iterrows():
        cot = str(r["generated_cot"]); gold = str(r["answer"]).strip().zfill(8)
        ins, outs, query = parse_prompt(str(r["prompt"])); vec = parse_selected(cot)
        if vec is None or query is None: continue
        aug = augment_cot(cot, ins, outs, vec)
        orig_lens.append(len(cot)); aug_lens.append(len(aug))
        if f"\\boxed{{{gold}}}" in aug: aug_rt_ok += 1
        for j in range(8):
            if find_decoy(ins, outs, vec, j): sc_possible += 1; break
    print(f"\n增强后: round-trip保持={aug_rt_ok}/{len(aug_lens)} | "
          f"orig tok中位≈{int(statistics.median(orig_lens))//4} 增强tok中位≈{int(statistics.median(aug_lens))//4} (max≈{max(aug_lens)//4})")
    print(f"可注入自纠错(至少一位有decoy)的题: {sc_possible}/{len(aug_lens)}")

    # ---- 打印 2 个样例 ----
    def show(idx, sc):
        r = bit.iloc[idx]; cot = str(r["generated_cot"]); gold = str(r["answer"]).strip().zfill(8)
        ins, outs, query = parse_prompt(str(r["prompt"])); vec = parse_selected(cot)
        decoy = jbit = None
        if sc:
            for j in range(8):
                d = find_decoy(ins, outs, vec, j)
                if d: jbit, decoy = j, d; break
        aug = augment_cot(cot, ins, outs, vec, with_self_correct_bit=jbit, decoy=decoy)
        # 只打印插入段附近(Selected 末尾 -> Applying 之前),省篇幅
        s = aug.find("\nSelected"); e = aug.find("Applying to ")
        print(f"\n{'#'*72}\n样例 {r['id']} ({'自纠错版' if sc else '纯验证版'}) gold={gold} 增强后≈{len(aug)//4}tok\n{'#'*72}")
        print("...[前面是 THK 原版搜索叙述]...")
        print(aug[s:e].rstrip())
        print("[然后是原版 Applying to query → \\boxed]")
        print("round-trip:", "✓" if f"\\boxed{{{gold}}}" in aug else "✗")
    # verify-only: 找一个能注入自纠错的反例之外的干净case; self-correct: 找有decoy的
    sc_idx = next((i for i in range(len(bit)) if (lambda v,a,b: v and any(find_decoy(a,b,v,j) for j in range(8)))(
        parse_selected(str(bit.iloc[i]['generated_cot'])), *parse_prompt(str(bit.iloc[i]['prompt']))[:2])), 0)
    show(0, sc=False)
    show(sc_idx, sc=True)
