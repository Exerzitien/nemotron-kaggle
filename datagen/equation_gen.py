"""Programmatic generator for the `equation_numeric_deduce` category.

Each problem shows several worked "equations" using disguised operator symbols,
then asks for the result of one more. We own the rules, so every example, the
query, and the answer are exact by construction, and the chain-of-thought is the
real deduction: identify the query symbol's rule from its examples, then apply.

Why this category needs synthetic data: on the run-1 holdout the base adapter
got the magnitude right but botched the *answer formatting* on 30/53 failures
(dropped/added the operator char, wrong sign). So the generator deliberately
varies the answer-decoration convention and the CoT names it explicitly.

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

# Symbols used to disguise operators (none are real arithmetic operators that
# would leak the rule). Braces are excluded so a brace-free answer can always
# be wrapped unambiguously in \boxed{} for the SFT target.
SYMBOL_POOL = list("@#%^&*!?/\\|][()<>~`\"'$:;")

# --- operation library: each is (name, description, fn(a,b) -> int) ----------
OPERATIONS = [
    ("add", "add the two numbers", lambda a, b: a + b),
    ("sub", "subtract the second number from the first", lambda a, b: a - b),
    ("rsub", "subtract the first number from the second", lambda a, b: b - a),
    ("absdiff", "take the absolute difference of the two numbers", lambda a, b: abs(a - b)),
    ("mul", "multiply the two numbers", lambda a, b: a * b),
    ("digitsum", "add up every digit of both numbers", lambda a, b: sum(int(d) for d in f"{a}{b}")),
    ("maxv", "take the larger of the two numbers", lambda a, b: max(a, b)),
    ("minv", "take the smaller of the two numbers", lambda a, b: min(a, b)),
]

# --- answer-decoration conventions: how the integer result is rendered -------
# Each: (name, description, render(value, symbol) -> str)
def _plain(v, s):
    return str(v)

def _suffix(v, s):
    return f"{v}{s}"

def _prefix(v, s):
    return f"{s}{v}"

def _signsym(v, s):
    # negative results show the operator symbol in place of the minus sign
    return f"{s}{abs(v)}" if v < 0 else str(v)

CONVENTIONS = [
    ("plain", "written as-is", _plain),
    ("suffix", "written with the operator symbol appended after it", _suffix),
    ("prefix", "written with the operator symbol placed before it", _prefix),
    ("signsym", "written so that a negative result shows the operator symbol "
                "instead of a minus sign", _signsym),
]


def _operands(rng):
    return rng.randint(10, 99), rng.randint(10, 99)


# operations that can yield a negative result (so 'signsym' is deducible)
SIGNED_OPS = {"sub", "rsub"}


def _build_operator(rng, symbol):
    """An operator = symbol + operation + answer-decoration convention.

    'signsym' is only assigned to ops that can actually go negative, otherwise
    the convention could never be demonstrated by an example."""
    name, desc, fn = rng.choice(OPERATIONS)
    pool = CONVENTIONS if name in SIGNED_OPS else [c for c in CONVENTIONS if c[0] != "signsym"]
    cname, cdesc, render = rng.choice(pool)
    return {
        "symbol": symbol, "op_name": name, "op_desc": desc, "fn": fn,
        "conv_name": cname, "conv_desc": cdesc, "render": render,
    }


def _example_str(op, a, b):
    v = op["fn"](a, b)
    return f"{a}{op['symbol']}{b} = {op['render'](v, op['symbol'])}", v


def make_problem(rng):
    n_ops = rng.randint(2, 3)
    symbols = rng.sample(SYMBOL_POOL, n_ops)
    operators = [_build_operator(rng, s) for s in symbols]

    # the query operator must be demonstrated by >= 2 examples to be deducible;
    # if its convention is signsym, at least one of those must be negative.
    q_op = rng.choice(operators)

    # The query operator's examples must pin its rule unambiguously. For
    # direction-sensitive ops (sub/rsub/absdiff) we include both an a>b and an
    # a<b case; that also guarantees a negative example when the convention is
    # 'signsym'.
    # sub/rsub/absdiff need both orders to be distinguished from each other;
    # maxv/minv need both orders so they cannot be mistaken for "take the
    # first/second operand" (which would coincide on single-direction examples).
    examples = []  # list of (line, operator, a, b, value)
    q_examples = []
    need_both_orders = q_op["op_name"] in ("sub", "rsub", "absdiff", "maxv", "minv")
    target = rng.randint(2, 3)
    have_gt = have_lt = False
    for _ in range(800):
        if len(q_examples) >= target and (not need_both_orders or (have_gt and have_lt)):
            break
        a, b = _operands(rng)
        if a == b:
            continue
        if need_both_orders:
            if a > b and have_gt and len(q_examples) >= target:
                continue
            if a < b and have_lt and len(q_examples) >= target:
                continue
        line, v = _example_str(q_op, a, b)
        q_examples.append((line, q_op, a, b, v))
        have_gt |= a > b
        have_lt |= a < b

    examples.extend(q_examples)
    # filler examples from the other operators
    for op in operators:
        if op is q_op:
            continue
        for _ in range(rng.randint(1, 2)):
            a, b = _operands(rng)
            line, v = _example_str(op, a, b)
            examples.append((line, op, a, b, v))
    rng.shuffle(examples)

    # the query
    qa, qb = _operands(rng)
    if q_op["conv_name"] == "signsym":
        # bias toward producing a negative answer sometimes to exercise the rule
        if rng.random() < 0.5:
            for _ in range(200):
                qa, qb = _operands(rng)
                if q_op["fn"](qa, qb) < 0:
                    break
    q_val = q_op["fn"](qa, qb)
    answer = q_op["render"](q_val, q_op["symbol"])

    prompt = PROMPT_HEADER + "\n"
    prompt += "\n".join(e[0] for e in examples)
    prompt += f"\nNow, determine the result for: {qa}{q_op['symbol']}{qb}"

    cot = _build_cot(examples, q_op, qa, qb, q_val, answer)
    prob = {"prompt": prompt, "answer": answer, "type": "equation_numeric_deduce",
            "generated_cot": cot}
    validate(prob, q_op, examples, qa, qb)
    return prob


def _build_cot(examples, q_op, qa, qb, q_val, answer):
    sym = q_op["symbol"]
    rel = [e for e in examples if e[1] is q_op]
    lines = [
        f"The query uses the symbol '{sym}'. I deduce its rule only from the "
        f"examples that contain '{sym}'."
    ]
    for line, op, a, b, v in rel:
        lines.append(f"  {line}  -- here {a} and {b} give {v}.")
    lines.append(
        f"Across these, the rule for '{sym}' is: {q_op['op_desc']}. "
        f"The result is {q_op['conv_desc']}."
    )
    lines.append(
        f"Apply to the query {qa}{sym}{qb}: {q_op['op_desc']} gives {q_val}."
    )
    lines.append(
        f"Rendering that result ({q_op['conv_desc']}) yields: {answer}."
    )
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def validate(prob, q_op, examples, qa, qb):
    """Self-check: prompt well-formed; the boxed answer equals answer; and the
    stated operation+convention actually reproduces every relevant example line
    and the query answer (guards against any drift in rendering logic)."""
    assert prob["prompt"].startswith("In Alice's Wonderland")
    assert "Now, determine the result for:" in prob["prompt"]
    assert prob["generated_cot"].rstrip().endswith("\\boxed{" + prob["answer"] + "}")
    assert "}" not in prob["answer"] and "{" not in prob["answer"]
    # recompute every query-operator example from scratch
    for line, op, a, b, v in examples:
        if op is not q_op:
            continue
        expect = f"{a}{op['symbol']}{b} = {op['render'](op['fn'](a, b), op['symbol'])}"
        assert line == expect, f"example mismatch: {line!r} != {expect!r}"
    # recompute the query answer
    assert prob["answer"] == q_op["render"](q_op["fn"](qa, qb), q_op["symbol"])
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
        p["id"] = f"equation_gen_{args.seed}_{i}"
        rows.append(p)

    convs = {}
    for p in rows:
        for c in CONVENTIONS:
            if c[1] in p["generated_cot"]:
                convs[c[0]] = convs.get(c[0], 0) + 1
    print(f"generated + validated {len(rows)} equation_numeric_deduce problems")
    print(f"  convention coverage: {convs}")
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
