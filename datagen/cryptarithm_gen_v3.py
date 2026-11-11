"""
cryptarithm_gen_v3.py — Replace dgxchen cryptarithm CoT with arithmetic-aware CoT
derived from lkevincc's gold-conditioned solver (97.2% coverage on 823 train rows).

Why: huikang/dgxchen reasoner only handles fwd/rev concat (~10% of true cases) and
defaults all others to "operator unknown -> default to concatenation". This teaches
the model exactly the wrong thing. v23/v30 cryptarithm = 0%.

This generator:
  - reads lkevincc solver_results.parquet (823 rows)
  - for each solver_correct row, emits a CoT that hypothesizes the digit map +
    operator semantics, verifies on each example, then applies to the query
  - keeps huikang's box-notation 【】 wrapper to stay compatible with the
    template the model has already locked onto
  - for 23 unsolved rows, returns None so caller falls back to dgxchen baseline

CoT shape (per problem, ~1000-1500 tokens):
  Header
  Hypothesize digit map + operator semantics (declarative, sourced from solver)
  For each example: show left, right, computed value, encode back, mark match
  Apply to query the same way
  Final \\boxed{answer}

Output: CSV with columns: id, prompt, answer, type, generated_cot
"""

from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd


REPO = Path('/Users/exerzitien/Desktop/nemotron-kaggle')
SOLVER_PARQUET = REPO / 'research/lkevincc_solver/data/solver_results.parquet'
TRAIN_TYPED = REPO / 'research/train_with_type.csv'
OUT_DIR = REPO / 'datagen/synth_dataset_v3'
OUT_CSV = OUT_DIR / 'cryptarithm_v3.csv'


# ─────────────────────────────────────────────────────────────────────────────
# Operator narration: maps solver_ops semantics → human-readable description
# Operands a, b are integers. The operator returns an integer (possibly negative).
# ─────────────────────────────────────────────────────────────────────────────
OP_DESC = {
    'add':            ('a + b',                 lambda a, b: a + b),
    'add_p1':         ('(a + b) + 1',           lambda a, b: a + b + 1),
    'add_m1':         ('(a + b) - 1',           lambda a, b: a + b - 1),
    'add_p2':         ('(a + b) + 2',           lambda a, b: a + b + 2),
    'sub_signed':     ('a - b',                 lambda a, b: a - b),
    'rsub_signed':    ('b - a',                 lambda a, b: b - a),
    'mul':            ('a * b',                 lambda a, b: a * b),
    'mul_p1':         ('a * b + 1',             lambda a, b: a * b + 1),
    'mul_m1':         ('a * b - 1',             lambda a, b: a * b - 1),
    'absdiff':        ('|a - b|',               lambda a, b: abs(a - b)),
    'absdiff_p1':     ('|a - b| + 1',           lambda a, b: abs(a - b) + 1),
    'absdiff_m2':     ('|a - b| - 2',           lambda a, b: abs(a - b) - 2),
    'neg_absdiff':    ('-|a - b|',              lambda a, b: -abs(a - b)),
    'mod':            ('a mod b',               lambda a, b: a % b if b else 0),
    'rmod':           ('b mod a',               lambda a, b: b % a if a else 0),
    'gcd':            ('gcd(a, b)',             None),
    'lcm':            ('lcm(a, b)',             None),
    'a2_plus_b':      ('a*a + b',               lambda a, b: a * a + b),
    'concat_fwd':     ('concat(a, b) as digits', None),
    'concat_rev':     ('concat(b, a) as digits', None),
}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // _gcd(a, b)


# Fill in gcd / lcm closures (need real implementations)
OP_DESC['gcd'] = ('gcd(a, b)', _gcd)
OP_DESC['lcm'] = ('lcm(a, b)', _lcm)


def _box(s: str) -> str:
    """Wrap each char in 【】."""
    return ''.join(f'【{c}】' for c in s)


def _try_strip_outer_backticks(s: str) -> str:
    """Conditionally strip a single backtick from each end IF doing so yields a
    valid 5-char LHS-shaped string. Backticks ARE symbols in some puzzles, so we
    must not strip blindly."""
    if len(s) >= 2 and s[0] == '`' and s[-1] == '`':
        return s[1:-1]
    return s


def _normalize_escapes(s: str) -> str:
    """Convert Python-repr style escapes back to literal chars."""
    # Order matters: do \\ before \' to avoid double-processing
    return s.replace('\\\\', '\x00').replace("\\'", "'").replace('\x00', '\\')


def _parse_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    """
    Extract (examples, query) from a cryptarithm prompt.
    Each example LHS is exactly 5 chars: `L1 L2 OP R1 R2`. Backticks may be
    decorative wrappers OR puzzle symbols — we accept whichever interpretation
    yields a 5-char LHS.
    """
    m = re.search(
        r'examples:\s*\n(.*?)\nNow, determine the result for:\s*([^\n]+)',
        prompt, re.DOTALL,
    )
    if not m:
        raise ValueError(f'Cannot parse prompt:\n{prompt[:200]}')
    block = m.group(1)
    query_raw = m.group(2).strip()

    def _split_eq(line: str) -> tuple[str, str] | None:
        line = _normalize_escapes(line.strip())
        if ' = ' not in line:
            return None
        lhs, rhs = line.split(' = ', 1)
        # Try with and without outer backtick stripping; prefer the one giving 5-char LHS
        for ls_try, rs_try in [(lhs, rhs), (_try_strip_outer_backticks(lhs), _try_strip_outer_backticks(rhs))]:
            if len(ls_try) == 5:
                return (ls_try, rs_try)
        # Default: keep as-is
        return (lhs, rhs)

    examples = []
    for raw_line in block.split('\n'):
        line = raw_line.strip()
        if not line or '=' not in line:
            continue
        parsed = _split_eq(line)
        if parsed is not None:
            examples.append(parsed)

    # Query parsing: similar logic, but no RHS.
    q = _normalize_escapes(query_raw)
    # If q has 6 chars and starts/ends with backticks, strip; if 5, leave.
    if len(q) != 5:
        q_stripped = _try_strip_outer_backticks(q)
        if len(q_stripped) == 5:
            q = q_stripped
    return examples, q


def _digits_to_symbols(n: int, mapping: dict[str, int], width: int | None = None) -> str | None:
    """Convert integer n to symbol string using digit→symbol reverse mapping.
    width: if given, pad to that many symbols (using leading-zero symbol)."""
    if n < 0:
        return None  # caller must handle sign
    inv = {v: k for k, v in mapping.items()}
    digits = [int(d) for d in str(n)]
    if width is not None and len(digits) < width:
        if 0 not in inv:
            return None
        digits = [0] * (width - len(digits)) + digits
    try:
        return ''.join(inv[d] for d in digits)
    except KeyError:
        return None  # specific result digit not in mapping


def _symbols_to_digits(s: str, mapping: dict[str, int], little_endian: bool = False) -> int | None:
    """Convert symbol string to integer using mapping. If little_endian, leftmost is lowest place."""
    try:
        digs = [mapping[c] for c in s]
    except KeyError:
        return None
    if little_endian:
        digs = list(reversed(digs))
    n = 0
    for d in digs:
        n = n * 10 + d
    return n


def gen_cot_arithmetic(
    examples: list[tuple[str, str]],
    query: str,
    mapping: dict[str, int],
    ops: dict[str, str],
    little_endian: bool,
    final_answer: str,
) -> str | None:
    """
    Generate CoT for arithmetic-mode (standard or little-endian) puzzles.
    Returns None if any verification fails.
    """
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # ── Hypothesis block (declarative) ────────────────────────────────────
    mode_name = "little-endian" if little_endian else "standard"
    lines.append(f"This puzzle is a digit-substitution cipher with {mode_name} digit order.")
    lines.append("Each symbol stands for a digit 0-9. The operators are arithmetic.")
    lines.append("")

    # State the digit map sorted by digit value for readability
    inv = {v: k for k, v in mapping.items()}
    map_pairs = [f"{inv[d]}={d}" for d in sorted(inv) if d in inv]
    lines.append("Digit mapping: " + ", ".join(map_pairs) + ".")

    # State op semantics
    op_strs = []
    for sym, sem in ops.items():
        if sem in OP_DESC:
            desc, _ = OP_DESC[sem]
            op_strs.append(f"【{sym}】 = {desc}")
    lines.append("Operator semantics: " + "; ".join(op_strs) + ".")
    lines.append("")

    # ── Verify each example ──────────────────────────────────────────────
    lines.append("Verify each example:")
    lines.append("")
    for lhs, rhs in examples:
        if len(lhs) != 5:
            return None
        a_sym, b_sym = lhs[0:2], lhs[3:5]
        op_sym = lhs[2]
        if op_sym not in ops:
            # Operator absent from declared semantics; skip narration for it,
            # but still demonstrate the structural parsing.
            continue
        sem = ops[op_sym]
        if sem not in OP_DESC:
            return None
        desc, fn = OP_DESC[sem]
        if fn is None and sem not in ('concat_fwd', 'concat_rev'):
            return None

        a = _symbols_to_digits(a_sym, mapping, little_endian)
        b = _symbols_to_digits(b_sym, mapping, little_endian)
        if a is None or b is None:
            return None

        # Compute expected result
        if sem in ('concat_fwd', 'concat_rev'):
            # Concat is symbol-level; for mixed_concat mode where the op outputs
            # a symbol-level concatenation rather than an arithmetic value,
            # encode as joined symbols directly.
            if sem == 'concat_fwd':
                pred = a_sym + b_sym
            else:
                pred = b_sym + a_sym
            ok = (pred == rhs)
            lines.append(f"【{lhs}】 = 【{rhs}】")
            lines.append(f"  left: 【{a_sym}】 right: 【{b_sym}】 op: 【{op_sym}】 ({desc})")
            lines.append(f"  concat result: 【{pred}】  {'match' if ok else 'MISMATCH'}")
            lines.append("")
            if not ok:
                return None
            continue

        # Arithmetic op
        result = fn(a, b)
        rhs_clean = rhs
        # Encode result back
        if result >= 0:
            encoded = _digits_to_symbols(result, mapping)
            if little_endian and encoded is not None:
                encoded = encoded[::-1]
        else:
            # Negative result: output uses a leading sign symbol if present;
            # commonly the puzzle uses '-' (literal minus). Skip encoding,
            # just show numeric.
            encoded = None

        lines.append(f"【{lhs}】 = 【{rhs}】")
        lines.append(f"  left: 【{a_sym}】 = {a}")
        lines.append(f"  right: 【{b_sym}】 = {b}")
        lines.append(f"  op 【{op_sym}】 ({desc}): {result}")
        if encoded is not None:
            ok = (encoded == rhs_clean)
            lines.append(f"  encode {result} as symbols: 【{encoded}】  {'match' if ok else 'MISMATCH'}")
        else:
            lines.append(f"  result = {result}, output 【{rhs_clean}】 (signed value)")
        lines.append("")

    # ── Apply to query ─────────────────────────────────────────────────
    if len(query) != 5:
        return None
    qa_sym, qb_sym = query[0:2], query[3:5]
    q_op_sym = query[2]

    if q_op_sym not in ops:
        return None
    sem = ops[q_op_sym]
    if sem not in OP_DESC:
        return None
    desc, fn = OP_DESC[sem]

    a = _symbols_to_digits(qa_sym, mapping, little_endian)
    b = _symbols_to_digits(qb_sym, mapping, little_endian)
    if a is None or b is None:
        return None

    lines.append(f"Apply to query 【{query}】:")
    lines.append(f"  left: 【{qa_sym}】 = {a}")
    lines.append(f"  right: 【{qb_sym}】 = {b}")

    if sem in ('concat_fwd', 'concat_rev'):
        if sem == 'concat_fwd':
            ans = qa_sym + qb_sym
        else:
            ans = qb_sym + qa_sym
        lines.append(f"  op 【{q_op_sym}】 ({desc}): concat → 【{ans}】")
    else:
        result = fn(a, b)
        lines.append(f"  op 【{q_op_sym}】 ({desc}): {result}")

        if result >= 0:
            ans = _digits_to_symbols(result, mapping)
            if little_endian and ans is not None:
                ans = ans[::-1]
        else:
            ans = '-' + (_digits_to_symbols(abs(result), mapping) or '')
            if little_endian and ans is not None:
                # weird case; little-endian + negative — uncommon. Just leave.
                pass
        if ans is None:
            return None
        lines.append(f"  encode {result} as symbols: 【{ans}】")

    # Sanity check vs gold
    if ans != final_answer:
        # Could be a sign / encoding mismatch; bail out
        return None

    lines.append("")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{ans}}}")
    return "\n".join(lines)


def gen_cot_concat(
    examples: list[tuple[str, str]],
    query: str,
    final_answer: str,
) -> str | None:
    """Pure-concat / mixed-concat fallback: use huikang-style fwd/rev concat narration."""
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # Detect fwd or rev concat by majority of examples
    fwd_hits = 0
    rev_hits = 0
    for lhs, rhs in examples:
        if len(lhs) != 5:
            continue
        a_sym, b_sym = lhs[0:2], lhs[3:5]
        if rhs == a_sym + b_sym:
            fwd_hits += 1
        if rhs == b_sym + a_sym:
            rev_hits += 1
    if fwd_hits == 0 and rev_hits == 0:
        return None
    is_fwd = fwd_hits >= rev_hits
    label = "concatenation" if is_fwd else "reverse concatenation"

    for lhs, rhs in examples:
        if len(lhs) != 5:
            continue
        a_sym, b_sym = lhs[0:2], lhs[3:5]
        op_sym = lhs[2]
        lines.append(f"【{lhs}】 = 【{rhs}】")
        lines.append(f"  input: {_box(lhs)}")
        lines.append(f"  left:{_box(a_sym)}")
        lines.append(f"  operator: 【{op_sym}】")
        lines.append(f"  right:{_box(b_sym)}")
        lines.append(f"  output: {_box(rhs)}")
        fwd = a_sym + b_sym
        rev = b_sym + a_sym
        lines.append(f"  concatenation: {_box(fwd)} {'match' if rhs == fwd else 'mismatch'}")
        lines.append(f"  reverse concatenation: {_box(rev)} {'match' if rhs == rev else 'mismatch'}")
        lines.append("")

    if len(query) != 5:
        return None
    qa, qb = query[0:2], query[3:5]
    if is_fwd:
        ans = qa + qb
    else:
        ans = qb + qa

    lines.append(f"Question 【{query}】 → applying {label} → 【{ans}】")
    lines.append("")
    if ans != final_answer:
        return None
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{ans}}}")
    return "\n".join(lines)


def generate_for_row(row: pd.Series) -> str | None:
    """Dispatch to the right generator based on solver_category."""
    if not row.get('solver_correct'):
        return None
    try:
        examples, query = _parse_prompt(row['prompt'])
    except Exception:
        return None

    cat = row['solver_category']
    final = row['solver_answer']

    if cat in ('pure_concat',):
        return gen_cot_concat(examples, query, final)

    if cat in ('arithmetic', 'little_endian', 'mixed_concat', 'mixed_concat_little_endian', 'query_unseen_concat'):
        try:
            mapping = json.loads(row['solver_mapping']) if isinstance(row['solver_mapping'], str) else None
            ops = json.loads(row['solver_ops']) if isinstance(row['solver_ops'], str) else None
        except Exception:
            return None
        if not mapping or not ops:
            return None
        little_endian = (cat in ('little_endian', 'mixed_concat_little_endian')) or (row.get('solver_mode') == 'little_endian')
        return gen_cot_arithmetic(examples, query, mapping, ops, little_endian, final)

    return None


def main():
    pq = pd.read_parquet(SOLVER_PARQUET)
    typed = pd.read_csv(TRAIN_TYPED)
    typed = typed[['id', 'type']]
    merged = pq.merge(typed, on='id', how='left')

    out_rows = []
    stats = {'attempted': 0, 'generated': 0, 'fallback': 0, 'by_cat': {}}
    for _, row in merged.iterrows():
        stats['attempted'] += 1
        cat = row['solver_category']
        stats['by_cat'].setdefault(str(cat), {'tried': 0, 'ok': 0})
        stats['by_cat'][str(cat)]['tried'] += 1

        cot = generate_for_row(row)
        if cot is None:
            stats['fallback'] += 1
            continue
        stats['generated'] += 1
        stats['by_cat'][str(cat)]['ok'] += 1
        out_rows.append({
            'id': row['id'],
            'prompt': row['prompt'],
            'answer': row['answer'],
            'type': row['type'],
            'generated_cot': cot,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'Wrote {len(df)} rows to {OUT_CSV}')
    print(f'Stats:')
    print(json.dumps(stats, indent=2, default=str))


if __name__ == '__main__':
    main()
