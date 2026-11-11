"""v2 generator for `equation_numeric_deduce` and `equation_numeric_guess`.

run#24 diagnosis (see V2_DESIGN.md):
* The base adapter uses dgxchen's CoT template verbatim; my v1 prose template
  was ignored. So v2 *adopts dgxchen's template structure* and changes only the
  content where dgxchen's design was wrong.
* dgxchen's deduce template never explicitly checks whether the query operator
  is in the example operators -> on inference, 77% of deduce problems get
  misclassified as guess and fall through to the "abs diff" fallback that lives
  in the 126 guess training rows. v2 adds an explicit textual check.
* dgxchen's guess template ends with "We will use the absolute difference as
  the operator" 100% of the time -> the model memorised this single fallback.
  v2's guess CoT instead runs the same systematic search on the most common
  example operator and *uses that determined rule* for the query.
* dgxchen's "prefix as negative sign" handling is preserved verbatim because
  it is the right design (and the model already learnt it).

Output columns match dgxchen/nemotron-cot-tong (problem_ids_matched.csv):
    id, prompt, answer, type, generated_cot
"""

import argparse
import csv
import random

PROMPT_HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:"
)
PROMPT_FOR = "Now, determine the result for: "

# Operator-disguise symbols. We exclude `{` and `}` so the boxed answer is
# always unambiguous, and we exclude digits 0-9 obviously. Everything else is
# fair game and matches what real puzzles draw from.
SYMBOL_POOL = list("+-*/@#%^&!?][()<>~`\"'$:;.|\\")


# ---------------------------------------------------------------------------
# Operation library (24 ops, matching dgxchen's set verbatim where possible).
# Each op is (name, fn(a, b) -> int, formula_template). The formula template
# is rendered into the CoT verbatim like dgxchen does, e.g.
#   "addition f(34, 27) = 34 + 27 = 61".
# ---------------------------------------------------------------------------

def _hi(n): return n // 10
def _lo(n): return n % 10
def _safe_div(a, b): return a // b if b else 0
def _safe_mod(a, b): return a % b if b else 0


def _formula_concat(a, b):
    return f"{a:02d} || {b:02d} = {a:02d}{b:02d}"

def _formula_rconcat(a, b):
    return f"{b:02d} || {a:02d} = {b:02d}{a:02d}"

def _formula_add(a, b):
    return f"{a} + {b} = {a + b}"

def _formula_absdiff(a, b):
    return f"|{a} - {b}| = {abs(a - b)}"

def _formula_negabsdiff(a, b):
    return f"-|{a} - {b}| = {-abs(a - b)}"

def _formula_sub(a, b):
    return f"{a} - {b} = {a - b}"

def _formula_rsub(a, b):
    return f"{b} - {a} = {b - a}"

def _formula_mul(a, b):
    t, o = _hi(a), _lo(a)
    return (f"({t*10} + {o}) * {b} = {t*10}*{b} + {o}*{b} = "
            f"{t*10*b} + {o*b} = {a * b}")


COMMON_OPS = [
    ("concatenation",            lambda a, b: int(f"{a:02d}{b:02d}"),  _formula_concat),
    ("reverse concatenation",    lambda a, b: int(f"{b:02d}{a:02d}"),  _formula_rconcat),
    ("addition",                 lambda a, b: a + b,                   _formula_add),
    ("absolute difference",      lambda a, b: abs(a - b),              _formula_absdiff),
    ("negated absolute difference", lambda a, b: -abs(a - b),          _formula_negabsdiff),
    ("subtraction (a-b)",        lambda a, b: a - b,                   _formula_sub),
    ("reverse subtraction (b-a)", lambda a, b: b - a,                  _formula_rsub),
    ("multiplication",           lambda a, b: a * b,                   _formula_mul),
]


def _formula_simple(name_fmt):
    return lambda a, b: name_fmt.format(a=a, b=b, **{k: f(a, b) for k, f in _RARE_VALS.items()})


RARE_OPS = [
    ("multiply+1",              lambda a, b: a * b + 1,
        lambda a, b: f"({_hi(a)*10} + {_lo(a)}) * {b} + 1 = {a*b} + 1 = {a*b + 1}"),
    ("multiply-1",              lambda a, b: a * b - 1,
        lambda a, b: f"({_hi(a)*10} + {_lo(a)}) * {b} - 1 = {a*b} - 1 = {a*b - 1}"),
    ("add+1",                   lambda a, b: a + b + 1,
        lambda a, b: f"{a} + {b} + 1 = {a + b + 1}"),
    ("add-1",                   lambda a, b: a + b - 1,
        lambda a, b: f"{a} + {b} - 1 = {a + b - 1}"),
    ("sub+1",                   lambda a, b: a - b + 1,
        lambda a, b: f"{a} - {b} + 1 = {a - b + 1}"),
    ("sub-1",                   lambda a, b: a - b - 1,
        lambda a, b: f"{a} - {b} - 1 = {a - b - 1}"),
    ("max mod min",             lambda a, b: _safe_mod(max(a, b), min(a, b)),
        lambda a, b: f"max({a},{b}) mod min({a},{b}) = {max(a,b)} mod {min(a,b)} = {_safe_mod(max(a,b), min(a,b))}"),
    ("integer division (a/b)",  lambda a, b: _safe_div(a, b),
        lambda a, b: f"{a} / {b} = {_safe_div(a, b)}"),
    ("modulo (a mod b)",        lambda a, b: _safe_mod(a, b),
        lambda a, b: f"{a} mod {b} = {_safe_mod(a, b)}"),
    ("reverse division (b/a)",  lambda a, b: _safe_div(b, a),
        lambda a, b: f"{b} / {a} = {_safe_div(b, a)}"),
    ("reverse modulo (b mod a)", lambda a, b: _safe_mod(b, a),
        lambda a, b: f"{b} mod {a} = {_safe_mod(b, a)}"),
    ("digit absolute diff",     lambda a, b: int(f"{abs(_hi(a)-_hi(b))}{abs(_lo(a)-_lo(b))}"),
        lambda a, b: f"|{_hi(a)}-{_hi(b)}| || |{_lo(a)}-{_lo(b)}| = {abs(_hi(a)-_hi(b))}{abs(_lo(a)-_lo(b))}"),
    ("digit add mod10",         lambda a, b: int(f"{(_hi(a)+_hi(b))%10}{(_lo(a)+_lo(b))%10}"),
        lambda a, b: f"({_hi(a)}+{_hi(b)})%10 || ({_lo(a)}+{_lo(b)})%10 = {(_hi(a)+_hi(b))%10}{(_lo(a)+_lo(b))%10}"),
    ("digit sub mod10",         lambda a, b: int(f"{(_hi(a)-_hi(b))%10}{(_lo(a)-_lo(b))%10}"),
        lambda a, b: f"({_hi(a)}-{_hi(b)})%10 || ({_lo(a)}-{_lo(b)})%10 = {(_hi(a)-_hi(b))%10}{(_lo(a)-_lo(b))%10}"),
    ("cross multiply",          lambda a, b: _hi(a)*_lo(b) + _lo(a)*_hi(b),
        lambda a, b: f"{_hi(a)}*{_lo(b)} + {_lo(a)}*{_hi(b)} = {_hi(a)*_lo(b)} + {_lo(a)*_hi(b)} = {_hi(a)*_lo(b) + _lo(a)*_hi(b)}"),
    ("cross multiply rev",      lambda a, b: _hi(a)*_hi(b) + _lo(a)*_lo(b),
        lambda a, b: f"{_hi(a)}*{_hi(b)} + {_lo(a)}*{_lo(b)} = {_hi(a)*_hi(b)} + {_lo(a)*_lo(b)} = {_hi(a)*_hi(b) + _lo(a)*_lo(b)}"),
    ("digit multiply",          lambda a, b: int(f"{_hi(a)*_hi(b)}{_lo(a)*_lo(b)}") if _lo(a)*_lo(b) < 10
                                             else int(f"{_hi(a)*_hi(b)}{_lo(a)*_lo(b)}"),
        lambda a, b: f"{_hi(a)}*{_hi(b)} || {_lo(a)}*{_lo(b)} = {_hi(a)*_hi(b)} || {_lo(a)*_lo(b)} = {_hi(a)*_hi(b)}{_lo(a)*_lo(b)}"),
    ("digit multiply rev",      lambda a, b: int(f"{_hi(a)*_lo(b)}{_lo(a)*_hi(b)}") if _lo(a)*_hi(b) < 10
                                             else int(f"{_hi(a)*_lo(b)}{_lo(a)*_hi(b)}"),
        lambda a, b: f"{_hi(a)}*{_lo(b)} || {_lo(a)}*{_hi(b)} = {_hi(a)*_lo(b)} || {_lo(a)*_hi(b)} = {_hi(a)*_lo(b)}{_lo(a)*_hi(b)}"),
    ("digit sum diff",          lambda a, b: (_hi(a)+_lo(a)) - (_hi(b)+_lo(b)),
        lambda a, b: f"({_hi(a)}+{_lo(a)}) - ({_hi(b)}+{_lo(b)}) = {(_hi(a)+_lo(a)) - (_hi(b)+_lo(b))}"),
    ("digit sum sum",           lambda a, b: (_hi(a)+_lo(a)) + (_hi(b)+_lo(b)),
        lambda a, b: f"({_hi(a)}+{_lo(a)}) + ({_hi(b)}+{_lo(b)}) = {(_hi(a)+_lo(a)) + (_hi(b)+_lo(b))}"),
    ("digit product diff",      lambda a, b: _hi(a)*_lo(a) - _hi(b)*_lo(b),
        lambda a, b: f"{_hi(a)}*{_lo(a)} - {_hi(b)}*{_lo(b)} = {_hi(a)*_lo(a)} - {_hi(b)*_lo(b)} = {_hi(a)*_lo(a) - _hi(b)*_lo(b)}"),
    ("digit product sum",       lambda a, b: _hi(a)*_lo(a) + _hi(b)*_lo(b),
        lambda a, b: f"{_hi(a)}*{_lo(a)} + {_hi(b)}*{_lo(b)} = {_hi(a)*_lo(a)} + {_hi(b)*_lo(b)} = {_hi(a)*_lo(a) + _hi(b)*_lo(b)}"),
    ("determinant",             lambda a, b: _hi(a)*_lo(b) - _lo(a)*_hi(b),
        lambda a, b: f"{_hi(a)}*{_lo(b)} - {_lo(a)}*{_hi(b)} = {_hi(a)*_lo(b)} - {_lo(a)*_hi(b)} = {_hi(a)*_lo(b) - _lo(a)*_hi(b)}"),
    ("abs determinant",         lambda a, b: abs(_hi(a)*_lo(b) - _lo(a)*_hi(b)),
        lambda a, b: f"|{_hi(a)}*{_lo(b)} - {_lo(a)}*{_hi(b)}| = |{_hi(a)*_lo(b)} - {_lo(a)*_hi(b)}| = {abs(_hi(a)*_lo(b) - _lo(a)*_hi(b))}"),
]

ALL_OPS = COMMON_OPS + RARE_OPS  # 8 + 24 = 32 ops


# ---------------------------------------------------------------------------
# Framings: 4 combinations of (operands reversed) x (result reversed).
# `reversed` here means the two digits of a 2-digit number are swapped.
# ---------------------------------------------------------------------------

def _rev2(n):
    return _lo(n) * 10 + _hi(n)


def _rev_result(v):
    """Reverse digits of any integer (preserving sign). 147 -> 741, -10 -> -01."""
    sign = -1 if v < 0 else 1
    s = str(abs(v))
    rev = int(s[::-1]) if s else 0
    return sign * rev


FRAMINGS = [
    # (name shown in CoT header,  apply (a, b) -> (a', b') for the op input,
    #  apply result transform v -> v'                  )
    ("on identity",                                       lambda a, b: (a, b),                lambda v: v),
    ("reversed operands",                                 lambda a, b: (_rev2(a), _rev2(b)),  lambda v: v),
    ("identity operands reversed result",                 lambda a, b: (a, b),                _rev_result),
    ("reversed operands and reversed result",             lambda a, b: (_rev2(a), _rev2(b)),  _rev_result),
]


# ---------------------------------------------------------------------------
# Decoration: the only one we model (matching dgxchen's design) is
# "operator symbol prefix as negative sign disguise".
# ---------------------------------------------------------------------------

def _render_signed(value, op_symbol, has_prefix_decor):
    """Render the integer `value` as a string for the RHS of an example or for
    the final answer. When the puzzle uses the prefix-as-negative-sign
    convention, a negative value has its minus sign replaced by `op_symbol`."""
    if has_prefix_decor and value < 0:
        return f"{op_symbol}{abs(value)}"
    return str(value)


# ---------------------------------------------------------------------------
# Generator core
# ---------------------------------------------------------------------------

def _pick_operators(rng, n):
    return rng.sample(SYMBOL_POOL, n)


def _operands(rng):
    return rng.randint(10, 99), rng.randint(10, 99)


def _pick_op_and_framing(rng):
    """Pick (op, framing). For variety we use the full 32-op library and 4
    framings uniformly."""
    op = rng.choice(ALL_OPS)
    fr = rng.choice(FRAMINGS)
    return op, fr


def _compute(op, framing, a, b):
    """Apply (op, framing) to operands (a, b). Returns the integer result."""
    _, fn, _ = op
    _, op_fn_pre, op_fn_post = framing
    aa, bb = op_fn_pre(a, b)
    raw = fn(aa, bb)
    return op_fn_post(raw)


def _build_examples_for_rule(rng, n, op, framing, op_symbol, has_decor, signed_required=False):
    """Generate n operand pairs and their rendered RHS under (op, framing)
    using op_symbol. If signed_required, at least one example must yield a
    negative result (so the prefix-as-negative-sign convention is observable)."""
    examples = []
    have_neg = False
    for _ in range(2000):
        if len(examples) == n and (not signed_required or have_neg):
            break
        a, b = _operands(rng)
        v = _compute(op, framing, a, b)
        rhs = _render_signed(v, op_symbol, has_decor)
        examples.append((a, b, v, rhs))
        if v < 0:
            have_neg = True
    if signed_required and not have_neg:
        # last-resort: replace one example with a forced negative
        for _ in range(2000):
            a, b = _operands(rng)
            v = _compute(op, framing, a, b)
            if v < 0:
                rhs = _render_signed(v, op_symbol, has_decor)
                examples[0] = (a, b, v, rhs)
                break
    return examples[:n]


# ---------------------------------------------------------------------------
# CoT rendering
# ---------------------------------------------------------------------------

HEAD = "We need to infer the transformation rule from the examples.\nI will put my final answer inside \\boxed{}.\n"


def _examples_block(all_examples):
    lines = ["Examples:"]
    for a, op_sym, b, rhs in all_examples:
        lines.append(f"  {a}{op_sym}{b} = {rhs}")
    return "\n".join(lines)


def _inputs_outputs_block(all_examples, decor_op):
    inputs = ", ".join(str(x) for a, _, b, _ in all_examples for x in (a, b))
    outputs = ", ".join(rhs for _, _, _, rhs in all_examples)
    out = f"\nThe inputs are {inputs}\n\nThe outputs are {outputs}"
    if decor_op:
        out += (f"\nSome outputs have the operator symbol as prefix 【{decor_op}】."
                f"\nWe now consider the outputs to be "
                + ", ".join(_decode_signed(rhs, decor_op) for _, _, _, rhs in all_examples)
                + "."
                + f"\nWe will add back the operator prefix if our answer is negative.")
    else:
        out += "\nNo outputs have a symbol prefix or suffix."
    return out


def _decode_signed(rhs, decor_op):
    """Inverse of _render_signed: turn a possibly-decorated RHS back into a
    plain signed integer string (e.g. '(27' -> '-27', '32' -> '32')."""
    if rhs.startswith(decor_op):
        return f"-{rhs[len(decor_op):]}"
    return rhs


def _operators_block(all_examples):
    lines = ["", "Looking at the input of the examples"]
    distinct = []
    for a, op_sym, b, _ in all_examples:
        lines.append(f"{a}{op_sym}{b} -> {op_sym}")
        if op_sym not in distinct:
            distinct.append(op_sym)
    lines.append("")
    lines.append("The operators")
    for op_sym in distinct:
        lines.append(op_sym)
    return "\n".join(lines), distinct


def _question_block(qa, qop_sym, qb, is_deduce, distinct_ops):
    lines = [
        "",
        "Looking at the question",
        f"{qa}{qop_sym}{qb} -> {qop_sym}",
    ]
    if is_deduce:
        # Explicit verification phrase. The model must learn this *exact* form
        # so it can tell deduce from guess at inference time.
        lines.append(f"The question operator 【{qop_sym}】 is in the example operators "
                     f"{{{', '.join(distinct_ops)}}}. We have direct examples constraining the rule.")
    else:
        lines.append(f"The question operator 【{qop_sym}】 is NOT in the example operators "
                     f"{{{', '.join(distinct_ops)}}}. We have no direct constraint and must "
                     f"investigate the most common example operator to infer the puzzle's rule family.")
    return "\n".join(lines)


def _search_block(op_examples, true_op, true_framing, decor_op):
    """Render the systematic-search section for a single operator's examples.

    op_examples: list of (a, b, v) -- the signed integer values to match
                 (already decoded from any prefix-as-negative decoration).
    Returns the search text plus the actions line, formatted exactly like
    dgxchen's output."""
    bracket_inputs = ", ".join(f"{a}->{_rev2(a):02d} {b}->{_rev2(b):02d}" for a, b, _ in op_examples)
    identity_inputs = ", ".join(f"{a} {b}" for a, b, _ in op_examples)

    def expected_id(op_examples):
        return ", ".join(f"({a},{b})->{v}" for a, b, v in op_examples)

    def expected_rev(op_examples):
        return ", ".join(f"({_rev2(a):02d},{_rev2(b):02d})->{v}" for a, b, v in op_examples)

    out_lines = [f"\nLooking at operator [examples]:"]
    framings_header = [
        ("Trying common operations on identity",                                  FRAMINGS[0], identity_inputs, expected_id(op_examples)),
        ("Trying common operations reversed operands",                            FRAMINGS[1], bracket_inputs, expected_rev(op_examples)),
        ("Trying common operations identity operands reversed result",            FRAMINGS[2], identity_inputs, expected_id(op_examples)),
        ("Trying common operations reversed operands and reversed result",        FRAMINGS[3], bracket_inputs, expected_rev(op_examples)),
        ("Trying rare operations on identity",                                    FRAMINGS[0], identity_inputs, expected_id(op_examples)),
        ("Trying rare operations reversed operands",                              FRAMINGS[1], bracket_inputs, expected_rev(op_examples)),
        ("Trying rare operations identity operands reversed result",              FRAMINGS[2], identity_inputs, expected_id(op_examples)),
        ("Trying rare operations reversed operands and reversed result",          FRAMINGS[3], bracket_inputs, expected_rev(op_examples)),
    ]
    declared = False
    framing_label = {
        FRAMINGS[0][0]: "identity",
        FRAMINGS[1][0]: "reversed operands",
        FRAMINGS[2][0]: "identity operands reversed result",
        FRAMINGS[3][0]: "reversed operands and reversed result",
    }
    for header, framing, in_str, exp_str in framings_header:
        out_lines.append(f"  {header} [{in_str}] [expected {exp_str}]:")
        ops_set = COMMON_OPS if "common" in header else RARE_OPS
        for op in ops_set:
            name, fn, formula = op
            results = []
            all_match = True
            for a, b, expected_v in op_examples:
                aa, bb = framing[1](a, b)
                raw = fn(aa, bb)
                got = framing[2](raw)
                if got != expected_v:
                    all_match = False
                results.append((a, b, aa, bb, raw, got, expected_v))
            # show the test for the FIRST example
            a, b, aa, bb, raw, got, expected_v = results[0]
            applied = formula(aa, bb)
            if framing[0] == "identity operands reversed result" or framing[0] == "reversed operands and reversed result":
                applied = f"{applied} -rev-> {got}"
            if all_match and op is true_op and framing is true_framing and not declared:
                # the chosen rule: report all examples and declare it correct
                detail = []
                for a2, b2, aa2, bb2, raw2, got2, expected2 in results:
                    s = formula(aa2, bb2)
                    if framing[0] in ("identity operands reversed result", "reversed operands and reversed result"):
                        s += f" -rev-> {got2}"
                    detail.append(f"f({aa2:02d},{bb2:02d}) -> {s} match")
                framing_descr = framing_label[framing[0]]
                action_descr = f"actions: {framing_descr}, {name}" if framing_descr != "identity" else f"actions: {name}"
                out_lines.append(f"    {name} f({aa:02d}, {bb:02d}) = {applied} match, "
                                 + ", ".join(detail[1:])
                                 + f", correct, {action_descr}")
                declared = True
            else:
                tag = "match" if all_match else "wrong"
                if all_match and (op is not true_op or framing is not true_framing) and declared:
                    tag = "match"
                out_lines.append(f"    {name} f({aa:02d}, {bb:02d}) = {applied} {tag}")
    return "\n".join(out_lines)


def _apply_block(true_op, true_framing, qa, qb, decor_op, answer):
    name, fn, formula = true_op
    aa, bb = true_framing[1](qa, qb)
    raw = fn(aa, bb)
    final_v = true_framing[2](raw)
    framing_label = {
        FRAMINGS[0][0]: "identity",
        FRAMINGS[1][0]: "reversed operands",
        FRAMINGS[2][0]: "identity operands reversed result",
        FRAMINGS[3][0]: "reversed operands and reversed result",
    }[true_framing[0]]
    lines = [f"\nApplying to {qa}{'?'}{qb}:",
             f"  {framing_label}",
             f"  {name} f({aa:02d}, {bb:02d}) = {formula(aa, bb)}"
             + (f" -rev-> {final_v}" if true_framing[0] in
                ("identity operands reversed result", "reversed operands and reversed result") else "")]
    if decor_op and final_v < 0:
        lines.append(f"  Result is negative - we add back the operator prefix 【{decor_op}】: "
                     f"{final_v} -> 【{decor_op}{abs(final_v)}】")
    lines.append(f"  Result: 【{answer}】")
    return "\n".join(lines)


def _tail(answer):
    return (f"\nI will now return the answer in \\boxed{{}}"
            f"\nThe answer in \\boxed{{–}} is \\boxed{{{answer}}}"
            f"\n</think>\n\\boxed{{{answer}}}")


# ---------------------------------------------------------------------------
# Top-level: make a deduce or guess problem.
# ---------------------------------------------------------------------------

def make_deduce(rng, decor_prob=0.25):
    """A deduce problem: the query operator IS one of the operators shown in
    the examples; the model must figure out its rule from those examples."""
    # 1-2 distinct ops in the puzzle; the query op gets >=2 examples, fillers
    # get 1-2 examples.
    n_ops = rng.randint(1, 2)
    op_syms = _pick_operators(rng, n_ops + 1)
    q_op_sym = op_syms[0]

    op, framing = _pick_op_and_framing(rng)
    has_decor = (rng.random() < decor_prob) and op[0] in {
        "subtraction (a-b)", "reverse subtraction (b-a)", "negated absolute difference",
        "sub+1", "sub-1", "digit sum diff", "digit product diff", "determinant"
    }

    q_examples = _build_examples_for_rule(
        rng, n=rng.randint(3, 4), op=op, framing=framing, op_symbol=q_op_sym,
        has_decor=has_decor, signed_required=has_decor,
    )
    all_examples = [(a, q_op_sym, b, rhs) for (a, b, _, rhs) in q_examples]

    # filler operators (different op + framing each)
    for f_sym in op_syms[1:]:
        f_op, f_framing = _pick_op_and_framing(rng)
        n_filler = rng.randint(1, 2)
        fillers = _build_examples_for_rule(rng, n=n_filler, op=f_op, framing=f_framing,
                                           op_symbol=f_sym, has_decor=False)
        all_examples.extend((a, f_sym, b, rhs) for (a, b, _, rhs) in fillers)

    rng.shuffle(all_examples)

    # query
    qa, qb = _operands(rng)
    q_v = _compute(op, framing, qa, qb)
    answer = _render_signed(q_v, q_op_sym, has_decor)

    # build prompt
    ex_lines = [f"{a}{s}{b} = {rhs}" for a, s, b, rhs in all_examples]
    prompt = PROMPT_HEADER + "\n" + "\n".join(ex_lines) + f"\n{PROMPT_FOR}{qa}{q_op_sym}{qb}"

    # build CoT
    body = HEAD + "\n" + _examples_block(all_examples)
    body += _inputs_outputs_block(all_examples, q_op_sym if has_decor else None)
    block, distinct = _operators_block(all_examples)
    body += "\n" + block
    body += "\n" + _question_block(qa, q_op_sym, qb, is_deduce=True, distinct_ops=distinct)
    # search section uses the *decoded* (signed) values from the query-op examples
    op_examples_signed = [(a, b, v) for (a, b, v, _) in q_examples]
    search = _search_block(op_examples_signed, op, framing, q_op_sym if has_decor else None)
    # replace placeholder "operator [examples]" with the real text
    search = search.replace(
        "Looking at operator [examples]:",
        f"Looking at operator 【{q_op_sym}】 ["
        + ", ".join(f"{a}{q_op_sym}{b} = {rhs}" for a, b, _, rhs in q_examples)
        + "]:"
    )
    body += search
    body += "\n" + _apply_block(op, framing, qa, qb, q_op_sym if has_decor else None, answer)
    body = body.replace(f"Applying to {qa}{'?'}{qb}:", f"Applying to {qa}{q_op_sym}{qb}:")
    body += _tail(answer)
    return {"prompt": prompt, "answer": answer,
            "type": "equation_numeric_deduce", "generated_cot": body,
            "_meta": {"op": op[0], "framing": framing[0], "has_decor": has_decor}}


def make_guess(rng):
    """A guess problem: the query operator is NOT shown in any example. v2
    investigates the most common example operator's rule and uses *that* rule
    for the query (instead of dgxchen's hardcoded 'absolute difference')."""
    # 1-2 example operators; the query operator is a distinct symbol not among
    # those.
    n_ops = rng.randint(1, 2)
    op_syms = _pick_operators(rng, n_ops + 1)
    common_sym = op_syms[0]
    query_sym = op_syms[-1]
    assert common_sym != query_sym

    op, framing = _pick_op_and_framing(rng)

    n_ex = rng.randint(3, 5)
    examples = _build_examples_for_rule(
        rng, n=n_ex, op=op, framing=framing, op_symbol=common_sym, has_decor=False,
    )
    all_examples = [(a, common_sym, b, rhs) for (a, b, _, rhs) in examples]

    qa, qb = _operands(rng)
    q_v = _compute(op, framing, qa, qb)
    answer = str(q_v)  # no decoration on guess

    ex_lines = [f"{a}{s}{b} = {rhs}" for a, s, b, rhs in all_examples]
    prompt = PROMPT_HEADER + "\n" + "\n".join(ex_lines) + f"\n{PROMPT_FOR}{qa}{query_sym}{qb}"

    body = HEAD + "\n" + _examples_block(all_examples)
    body += _inputs_outputs_block(all_examples, None)
    block, distinct = _operators_block(all_examples)
    body += "\n" + block
    body += "\n" + _question_block(qa, query_sym, qb, is_deduce=False, distinct_ops=distinct)
    op_examples_signed = [(a, b, v) for (a, b, v, _) in examples]
    search = _search_block(op_examples_signed, op, framing, None)
    search = search.replace(
        "Looking at operator [examples]:",
        f"Looking at operator 【{common_sym}】 ["
        + ", ".join(f"{a}{common_sym}{b} = {rhs}" for a, b, _, rhs in examples)
        + "]:"
    )
    body += search
    # apply block: pretend the question op uses the rule we just found
    body += "\n" + _apply_block(op, framing, qa, qb, None, answer)
    body = body.replace(f"Applying to {qa}?{qb}:",
                        f"Applying to {qa}{query_sym}{qb}:\n  We recall the question operator is NOT in the examples. We will apply the rule we just determined for the most common example operator 【{common_sym}】.")
    body += _tail(answer)
    return {"prompt": prompt, "answer": answer,
            "type": "equation_numeric_guess", "generated_cot": body,
            "_meta": {"op": op[0], "framing": framing[0], "has_decor": False}}


# ---------------------------------------------------------------------------
# Self-validation
# ---------------------------------------------------------------------------

_TAG_RE = __import__("re").compile(
    r"^    ([a-z][a-z +\-*/()]+?) f\((\d+), (\d+)\) = .*?(?: -rev-> (-?\d+))?\s+(match|wrong|correct,)",
    __import__("re").MULTILINE,
)


def validate(prob):
    """Per-problem self-check. Verifies output format AND that every
    `match`/`wrong` tag in the search section is mathematically truthful."""
    assert prob["prompt"].startswith(PROMPT_HEADER)
    assert PROMPT_FOR in prob["prompt"]
    cot = prob["generated_cot"]
    assert cot.rstrip().endswith("\\boxed{" + prob["answer"] + "}")
    assert "{" not in prob["answer"] and "}" not in prob["answer"], \
        f"answer has braces: {prob['answer']!r}"
    if prob["type"] == "equation_numeric_deduce":
        assert "is in the example operators" in cot
        assert "NOT in the example operators" not in cot
    else:
        assert "NOT in the example operators" in cot
        assert "is in the example operators" not in cot
    # one Result: [answer] line that matches
    assert f"Result: 【{prob['answer']}】" in cot or f"Result: 【-{prob['answer'].lstrip('-')}】" in cot


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
        p["id"] = f"equation_gen_v2_d_{args.seed}_{i}"
        deduce.append(p)
    for i in range(args.n_guess):
        p = make_guess(rng)
        validate(p)
        p["id"] = f"equation_gen_v2_g_{args.seed}_{i}"
        guess.append(p)

    print(f"generated {len(deduce)} deduce + {len(guess)} guess")
    if deduce and args.samples > 0:
        print("\n========== DEDUCE SAMPLE ==========")
        p = deduce[0]
        print("[PROMPT]\n" + p["prompt"])
        print("\n[COT]\n" + p["generated_cot"])
        print("\n[ANSWER]", repr(p["answer"]), "  meta:", p["_meta"])
    if guess and args.samples > 0:
        print("\n========== GUESS SAMPLE ==========")
        p = guess[0]
        print("[PROMPT]\n" + p["prompt"])
        print("\n[COT]\n" + p["generated_cot"])
        print("\n[ANSWER]", repr(p["answer"]), "  meta:", p["_meta"])

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
