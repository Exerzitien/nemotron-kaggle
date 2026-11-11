"""Programmatic generator for the `cryptarithm_deduce` category (structural ops).

A cryptarithm puzzle shows worked equations whose operands and results are
written in disguised digit-symbols, with the operators (+,-,*,/) each standing
for a hidden transformation. We confirmed (by exhaustive CSP search over the
real puzzles) that the operators are NOT standard arithmetic.

This generator covers the *structural* operators: transformations that
rearrange / select the operand symbols (concatenate, reverse, interleave,
pick, duplicate, ...). These are independent of any digit->symbol mapping, so
each example, the query, and the answer are exact by construction, the puzzle
is uniquely deducible from a single example, and the chain-of-thought is the
real deduction: align every output symbol to the operand slot it came from.

Arithmetic operators (variable-length output, e.g. `/`->3 then 4 symbols in the
same real puzzle) need the digit->symbol bijection solved and are left to a
separate, harder generator.

Output columns match dgxchen/nemotron-cot-tong (problem_ids_matched.csv):
    id, prompt, answer, type, generated_cot
so generated rows drop straight into the existing SFT pipeline.
"""

import argparse
import csv
import random

PROMPT_HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:"
)

# digit-symbols the disguised numbers are drawn from (no braces so \boxed{}
# stays unambiguous).
SYMBOL_POOL = list("@#%^&!?][()<>~`\"'$:;._")
# Operator alphabet. In the real puzzles the operator is the middle symbol of a
# 5-symbol LHS and is itself disguised: ~70% of the time it is one of + - * /,
# the rest can be any symbol. We weight the pool to match that distribution.
OP_POOL = list("+-*/") * 6 + SYMBOL_POOL
OPERAND_LEN = 2  # real cryptarithm operands are always 2 symbols

# A slot names one input symbol: (operand index 0|1, char index).
SLOT_NAMES = {
    (0, 0): "the first symbol of the left operand",
    (0, 1): "the second symbol of the left operand",
    (1, 0): "the first symbol of the right operand",
    (1, 1): "the second symbol of the right operand",
}
ALL_SLOTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _random_posmap(rng):
    """A structural rule = an ordered list of input slots forming the output.

    Length 2..4, must use >=2 distinct slots (otherwise it is a trivial
    constant) and must not be the plain left-then-right concatenation of the
    untouched operands (too easy / leaks nothing)."""
    while True:
        length = rng.randint(2, 4)
        posmap = [rng.choice(ALL_SLOTS) for _ in range(length)]
        if len({s for s in posmap}) < 2:
            continue
        if posmap == [(0, 0), (0, 1), (1, 0), (1, 1)]:
            continue
        return posmap


def _apply(posmap, left, right):
    operands = (left, right)
    return "".join(operands[op][idx] for op, idx in posmap)


def _make_operands(rng, forbidden):
    """Two operands with four pairwise-distinct symbols, none equal to an
    operator symbol of the puzzle (so each LHS char is unambiguous and each
    output symbol traces back to exactly one input slot)."""
    pool = [s for s in SYMBOL_POOL if s not in forbidden]
    s = rng.sample(pool, 4)
    return s[0] + s[1], s[2] + s[3]


def _pick_operators(rng, n):
    """n distinct operator symbols, weighted toward + - * / like the real data."""
    chosen = []
    while len(chosen) < n:
        c = rng.choice(OP_POOL)
        if c not in chosen:
            chosen.append(c)
    return chosen


def make_problem(rng):
    n_ops = rng.randint(2, 3)
    op_syms = _pick_operators(rng, n_ops)
    forbidden = set(op_syms)
    rules = {op: _random_posmap(rng) for op in op_syms}
    q_op = rng.choice(op_syms)

    examples = []  # (left, op, right, result)
    # query operator: 2-3 examples so the rule is over-determined
    for _ in range(rng.randint(2, 3)):
        l, r = _make_operands(rng, forbidden)
        examples.append((l, q_op, r, _apply(rules[q_op], l, r)))
    # filler examples from the other operators
    for op in op_syms:
        if op == q_op:
            continue
        for _ in range(rng.randint(1, 2)):
            l, r = _make_operands(rng, forbidden)
            examples.append((l, op, r, _apply(rules[op], l, r)))
    rng.shuffle(examples)

    ql, qr = _make_operands(rng, forbidden)
    answer = _apply(rules[q_op], ql, qr)

    lines = [f"{l}{op}{r} = {res}" for l, op, r, res in examples]
    prompt = PROMPT_HEADER + "\n" + "\n".join(lines)
    prompt += f"\nNow, determine the result for: {ql}{q_op}{qr}"

    cot = _build_cot(examples, q_op, rules[q_op], ql, qr, answer)
    prob = {"prompt": prompt, "answer": answer,
            "type": "cryptarithm_deduce", "generated_cot": cot}
    validate(prob, q_op, rules[q_op], examples, ql, qr)
    return prob


def _describe(posmap):
    return "; ".join(f"position {i + 1} = {SLOT_NAMES[s]}"
                     for i, s in enumerate(posmap))


def _build_cot(examples, q_op, posmap, ql, qr, answer):
    rel = [e for e in examples if e[1] == q_op]
    lines = [
        f"The query uses the operator '{q_op}'. Each operand is two symbols; I "
        f"call them (left = L1 L2, right = R1 R2). I deduce what '{q_op}' does "
        f"by tracing every output symbol back to an input slot."
    ]
    for l, op, r, res in rel:
        trace = []
        for i, s in enumerate(posmap):
            operands = (l, r)
            label = f"{'L' if s[0] == 0 else 'R'}{s[1] + 1}"
            trace.append(f"{res[i]}={label}")
        lines.append(f'  "{l}{op}{r} = {res}": output is {", ".join(trace)}.')
    lines.append(
        f"Every example agrees, so '{q_op}' rearranges the operands as: "
        f"{_describe(posmap)}."
    )
    qtrace = []
    for s in posmap:
        operands = (ql, qr)
        label = f"{'L' if s[0] == 0 else 'R'}{s[1] + 1}"
        qtrace.append(f"{label}={operands[s[0]][s[1]]}")
    lines.append(
        f"For the query {ql}{q_op}{qr}: L1L2={ql}, R1R2={qr}. "
        f"Taking {', '.join(qtrace)} gives \"{answer}\"."
    )
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def validate(prob, q_op, posmap, examples, ql, qr):
    """Self-check: prompt well-formed; boxed answer matches; and -- the property
    the puzzle's solvability depends on -- every query-operator example has four
    pairwise-distinct input symbols, so the posmap is uniquely deducible, and
    the posmap reproduces every example result and the query answer."""
    assert prob["prompt"].startswith("In Alice's Wonderland")
    assert "Now, determine the result for:" in prob["prompt"]
    assert prob["generated_cot"].rstrip().endswith("\\boxed{" + prob["answer"] + "}")
    assert "}" not in prob["answer"] and "{" not in prob["answer"]
    for l, op, r, res in examples:
        if op != q_op:
            continue
        assert len(set(l + r)) == 4, f"non-distinct input symbols: {l}{op}{r}"
        assert _apply(posmap, l, r) == res, f"example mismatch: {l}{op}{r}={res}"
    assert len(set(ql + qr)) == 4, f"non-distinct query symbols: {ql}{qr}"
    assert prob["answer"] == _apply(posmap, ql, qr)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        p = make_problem(rng)  # make_problem self-validates before returning
        p["id"] = f"cryptarithm_gen_{args.seed}_{i}"
        rows.append(p)

    lens = {}
    for p in rows:
        lens[len(p["answer"])] = lens.get(len(p["answer"]), 0) + 1
    print(f"generated + validated {len(rows)} cryptarithm_deduce problems (structural ops)")
    print(f"  answer-length distribution: {dict(sorted(lens.items()))}")
    for p in rows[: args.samples]:
        print("=" * 72)
        print("[PROMPT]\n" + p["prompt"])
        print("\n[GENERATED_COT]\n" + p["generated_cot"])
        print("\n[ANSWER]", repr(p["answer"]))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "type", "generated_cot"])
            w.writeheader()
            for p in rows:
                w.writerow(p)
        print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
