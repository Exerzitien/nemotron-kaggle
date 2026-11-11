"""v2 generator for `cryptarithm_deduce` and `cryptarithm_guess`.

run#24 diagnosis (see V2_DESIGN.md):
* dgxchen's cryptarithm CoT only tests two candidate rules per example:
  concatenation and reverse concatenation. Any operator whose true rule is
  something else is marked `unknown`, and on guess puzzles the model defaults
  to concatenation. Result: 0/120 on both holdout sub-categories in run#24.
* v2 keeps dgxchen's `input/left/op/right/output` parsing skeleton verbatim
  (the model already adopted it) but tests *twelve* structural candidate
  rules. The chosen rule is whichever single candidate matches all of the
  examples for that operator.
* The deduce vs guess split is gated on an explicit textual check of whether
  the query operator is in the example operators, mirroring the same fix in
  equation_gen_v2.

Output columns match dgxchen/nemotron-cot-tong (problem_ids_matched.csv):
    id, prompt, answer, type, generated_cot
"""

import argparse
import csv
import random
from collections import Counter

PROMPT_HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:"
)
PROMPT_FOR = "Now, determine the result for: "

# Digit-symbols. Braces excluded so the boxed answer is unambiguous; digits
# 0-9 excluded so symbol/digit confusion can't arise.
SYMBOL_POOL = list("@#%^&!?][()<>~`\"'$:;._")

# Operator-disguise symbols. Real puzzles bias ~70% to +-*/ and pick anything
# else for the remainder.
OP_POOL = list("+-*/") * 6 + SYMBOL_POOL


# ---------------------------------------------------------------------------
# Candidate structural rules. Each is a list of (operand_idx, char_idx) slots
# that defines what the output is composed of. Operand 0 = left, 1 = right.
# Order matters: when several candidates match, we pick the first one that
# does. The first candidates are dgxchen's two (concatenation, reverse-concat)
# so the generated CoT keeps the same priority the model already learnt.
# ---------------------------------------------------------------------------
CANDIDATES = [
    ("concatenation",         [(0, 0), (0, 1), (1, 0), (1, 1)]),
    ("reverse concatenation", [(1, 0), (1, 1), (0, 0), (0, 1)]),
    ("left only",             [(0, 0), (0, 1)]),
    ("right only",            [(1, 0), (1, 1)]),
    ("left reversed",         [(0, 1), (0, 0)]),
    ("right reversed",        [(1, 1), (1, 0)]),
    ("interleave LR",         [(0, 0), (1, 0), (0, 1), (1, 1)]),
    ("interleave RL",         [(1, 0), (0, 0), (1, 1), (0, 1)]),
    ("left first only",       [(0, 0)]),
    ("right last only",       [(1, 1)]),
    ("left doubled",          [(0, 0), (0, 1), (0, 0), (0, 1)]),
    ("left wraps right",      [(0, 0), (1, 0), (1, 1), (0, 1)]),
]


def _apply(posmap, left, right):
    operands = (left, right)
    return "".join(operands[op_idx][ch_idx] for op_idx, ch_idx in posmap)


def _pretty_pattern(posmap):
    """Render a posmap as 【L1】【L2】【R1】【R2】-style brackets for the CoT."""
    label = {(0, 0): "L1", (0, 1): "L2", (1, 0): "R1", (1, 1): "R2"}
    return "".join(f"【{label[s]}】" for s in posmap)


def _pretty_concrete(posmap, left, right):
    operands = (left, right)
    return "".join(f"【{operands[op_idx][ch_idx]}】" for op_idx, ch_idx in posmap)


# ---------------------------------------------------------------------------
# Operand and operator picking
# ---------------------------------------------------------------------------

def _pick_operators(rng, n):
    chosen = []
    while len(chosen) < n:
        c = rng.choice(OP_POOL)
        if c not in chosen:
            chosen.append(c)
    return chosen


def _make_operands(rng, forbidden):
    """Two 2-symbol operands with four pairwise-distinct digit-symbols, none
    equal to any operator symbol in the puzzle (so every LHS char is
    unambiguously parseable)."""
    pool = [s for s in SYMBOL_POOL if s not in forbidden]
    s = rng.sample(pool, 4)
    return s[0] + s[1], s[2] + s[3]


# ---------------------------------------------------------------------------
# CoT rendering pieces
# ---------------------------------------------------------------------------

HEAD = (
    "We need to infer the transformation rule from the examples.\n"
    "I will put my final answer inside \\boxed{}.\n"
)


def _example_parse_block(left, op_sym, right, result):
    """Render dgxchen-style 'input/left/op/right/output' parsing plus the
    twelve-candidate test for one example. Returns (text, matching_rule_name).
    matching_rule_name is the first candidate (by priority) that produces
    `result`; or None if none match (treated as 'unknown')."""
    inp = f"【{left[0]}】【{left[1]}】【{op_sym}】【{right[0]}】【{right[1]}】"
    out_chars = "".join(f"【{c}】" for c in result)
    lines = [
        f"【{left}{op_sym}{right}】 = 【{result}】",
        f"  input: {inp}",
        f"  left:【{left[0]}】【{left[1]}】",
        f"  operator: 【{op_sym}】",
        f"  right:【{right[0]}】【{right[1]}】",
        f"  output: {out_chars}",
    ]
    matching = None
    for name, posmap in CANDIDATES:
        got = _apply(posmap, left, right)
        tag = "match" if got == result else "mismatch"
        if got == result and matching is None:
            matching = name
        lines.append(f"  {name}: {_pretty_concrete(posmap, left, right)} {tag}")
    conclusion = matching if matching else "unknown"
    lines.append(f"  operator: 【{op_sym}】{conclusion}")
    return "\n".join(lines), matching


def _question_block(query_left, q_op_sym, query_right):
    return (
        f"\nQuestion【{query_left}{q_op_sym}{query_right}】\n"
        f"  input: 【{query_left[0]}】【{query_left[1]}】【{q_op_sym}】【{query_right[0]}】【{query_right[1]}】\n"
        f"  left:【{query_left[0]}】【{query_left[1]}】\n"
        f"  operator:【{q_op_sym}】\n"
        f"  right:【{query_right[0]}】【{query_right[1]}】"
    )


def _apply_block(rule_name, posmap, query_left, query_right, answer):
    applied_pattern = _pretty_pattern(posmap)
    applied_value = _pretty_concrete(posmap, query_left, query_right)
    return (
        f"\n  {rule_name}({_pretty_concrete([(0,0),(0,1)], query_left, query_right)}, "
        f"{_pretty_concrete([(1,0),(1,1)], query_left, query_right)}) = {applied_value}\n"
        f"  output: 【{answer}】-> 【{{{answer}}}】"
    )


def _tail(answer):
    return (
        f"\n\nI will now return the answer in \\boxed{{}}\n"
        f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}\n"
        f"</think>\n\\boxed{{{answer}}}"
    )


# ---------------------------------------------------------------------------
# Top-level: make_deduce / make_guess
# ---------------------------------------------------------------------------

def make_deduce(rng):
    """Query operator IS one of the example operators. Multiple ops; we
    generate >=2 examples for the query op and 1-2 for each filler op."""
    n_ops = rng.randint(2, 3)
    op_syms = _pick_operators(rng, n_ops)
    forbidden = set(op_syms)
    rules = {op: rng.choice(CANDIDATES) for op in op_syms}
    q_op_sym = rng.choice(op_syms)

    examples = []  # (left, op_sym, right, result)
    for _ in range(rng.randint(2, 3)):
        l, r = _make_operands(rng, forbidden)
        rname, posmap = rules[q_op_sym]
        examples.append((l, q_op_sym, r, _apply(posmap, l, r)))
    for s in op_syms:
        if s == q_op_sym:
            continue
        for _ in range(rng.randint(1, 2)):
            l, r = _make_operands(rng, forbidden)
            rname, posmap = rules[s]
            examples.append((l, s, r, _apply(posmap, l, r)))
    rng.shuffle(examples)

    ql, qr = _make_operands(rng, forbidden)
    q_rule_name, q_posmap = rules[q_op_sym]
    answer = _apply(q_posmap, ql, qr)

    prompt_lines = [f"{l}{s}{r} = {res}" for l, s, r, res in examples]
    prompt = PROMPT_HEADER + "\n" + "\n".join(prompt_lines) + f"\n{PROMPT_FOR}{ql}{q_op_sym}{qr}"

    # CoT body
    body_parts = [HEAD]
    distinct_ops = []
    for l, s, r, res in examples:
        text, _ = _example_parse_block(l, s, r, res)
        body_parts.append(text)
        if s not in distinct_ops:
            distinct_ops.append(s)

    body_parts.append(_question_block(ql, q_op_sym, qr))
    body_parts.append(
        f"\nThe question operator 【{q_op_sym}】 is in the example operators "
        f"{{{', '.join(distinct_ops)}}}. From the examples we determined: "
        f"【{q_op_sym}】 is {q_rule_name}."
    )
    body_parts.append(_apply_block(q_rule_name, q_posmap, ql, qr, answer))
    body_parts.append(_tail(answer))

    return {"prompt": prompt, "answer": answer,
            "type": "cryptarithm_deduce",
            "generated_cot": "\n".join(body_parts)}


def make_guess(rng):
    """Query operator is NOT among the example operators. Each example op has
    a determined rule; v2's guess CoT picks the *most common determined rule*
    instead of dgxchen's blind default-to-concatenation."""
    n_ops = rng.randint(2, 3)
    op_syms = _pick_operators(rng, n_ops + 1)
    forbidden = set(op_syms)
    example_ops = op_syms[:-1]
    query_op_sym = op_syms[-1]
    rules = {s: rng.choice(CANDIDATES) for s in example_ops}

    examples = []
    for s in example_ops:
        for _ in range(rng.randint(1, 2)):
            l, r = _make_operands(rng, forbidden)
            _, posmap = rules[s]
            examples.append((l, s, r, _apply(posmap, l, r)))
    rng.shuffle(examples)

    # query operands -- the rule v2 picks for the query is the most common
    # rule among the determined example operators (deterministic tiebreak by
    # candidate priority order).
    rule_counts = Counter()
    for s in example_ops:
        rule_counts[rules[s][0]] += 1
    most_common_rule_name = sorted(
        rule_counts.items(),
        key=lambda kv: (-kv[1], [c[0] for c in CANDIDATES].index(kv[0])),
    )[0][0]
    most_common_posmap = next(p for n, p in CANDIDATES if n == most_common_rule_name)
    ql, qr = _make_operands(rng, forbidden)
    answer = _apply(most_common_posmap, ql, qr)

    prompt_lines = [f"{l}{s}{r} = {res}" for l, s, r, res in examples]
    prompt = PROMPT_HEADER + "\n" + "\n".join(prompt_lines) + f"\n{PROMPT_FOR}{ql}{query_op_sym}{qr}"

    body_parts = [HEAD]
    distinct_ops = []
    for l, s, r, res in examples:
        text, _ = _example_parse_block(l, s, r, res)
        body_parts.append(text)
        if s not in distinct_ops:
            distinct_ops.append(s)

    body_parts.append(_question_block(ql, query_op_sym, qr))
    body_parts.append(
        f"\nThe question operator 【{query_op_sym}】 is NOT in the example operators "
        f"{{{', '.join(distinct_ops)}}}. We have no direct constraint. "
        f"Among the determined operators, the most common rule is {most_common_rule_name}; "
        f"we will use that as our best guess for the query."
    )
    body_parts.append(_apply_block(most_common_rule_name, most_common_posmap, ql, qr, answer))
    body_parts.append(_tail(answer))

    return {"prompt": prompt, "answer": answer,
            "type": "cryptarithm_guess",
            "generated_cot": "\n".join(body_parts)}


# ---------------------------------------------------------------------------
# Self-validation
# ---------------------------------------------------------------------------

def validate(prob):
    assert prob["prompt"].startswith(PROMPT_HEADER)
    assert PROMPT_FOR in prob["prompt"]
    cot = prob["generated_cot"]
    assert cot.rstrip().endswith("\\boxed{" + prob["answer"] + "}")
    assert "{" not in prob["answer"] and "}" not in prob["answer"], \
        f"answer has braces: {prob['answer']!r}"
    if prob["type"] == "cryptarithm_deduce":
        assert "is in the example operators" in cot
        assert "NOT in the example operators" not in cot
    else:
        assert "NOT in the example operators" in cot
        assert "is in the example operators" not in cot
    # output: 【answer】-> 【{answer}】 line
    assert f"output: 【{prob['answer']}】" in cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-deduce", type=int, default=20)
    ap.add_argument("--n-guess", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-deduce", default="")
    ap.add_argument("--out-guess", default="")
    ap.add_argument("--samples", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    deduce, guess = [], []
    for i in range(args.n_deduce):
        p = make_deduce(rng)
        validate(p)
        p["id"] = f"cryptarithm_gen_v2_d_{args.seed}_{i}"
        deduce.append(p)
    for i in range(args.n_guess):
        p = make_guess(rng)
        validate(p)
        p["id"] = f"cryptarithm_gen_v2_g_{args.seed}_{i}"
        guess.append(p)

    rule_counts = Counter()
    for p in deduce:
        for n, _ in CANDIDATES:
            if f"is {n}." in p["generated_cot"]:
                rule_counts[n] += 1
                break
    print(f"generated {len(deduce)} deduce + {len(guess)} guess")
    print("deduce rule distribution:")
    for n, _ in CANDIDATES:
        print(f"  {n:24s} {rule_counts.get(n, 0)}")
    if args.samples > 0 and deduce:
        print("\n========== DEDUCE SAMPLE ==========")
        p = deduce[0]
        print("[PROMPT]\n" + p["prompt"])
        print("\n[COT]\n" + p["generated_cot"])
        print("\n[ANSWER]", repr(p["answer"]))
    if args.samples > 0 and guess:
        print("\n========== GUESS SAMPLE ==========")
        p = guess[0]
        print("[PROMPT]\n" + p["prompt"])
        print("\n[COT]\n" + p["generated_cot"])
        print("\n[ANSWER]", repr(p["answer"]))

    for rows, path in [(deduce, args.out_deduce), (guess, args.out_guess)]:
        if path and rows:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "type", "generated_cot"])
                w.writeheader()
                for p in rows:
                    w.writerow({k: p[k] for k in ["id", "prompt", "answer", "type", "generated_cot"]})
            print(f"wrote {len(rows)} -> {path}")


if __name__ == "__main__":
    main()
