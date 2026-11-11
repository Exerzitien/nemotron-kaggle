"""
bit_gen_v3.py — Generate bit_manipulation CoT using MAJ0RT0M solver (98.88% on train).

Strategy (combines MAJ0RT0M solver + Donald compact template):
  1. Re-run MAJ0RT0M solver on each train bit_manipulation problem
  2. Take only rows where solver finds a rule AND predicts correctly
  3. Emit structured per-bit narration:
     - S1: type identification
     - S2: COLUMNS (input/output columns per bit position)
     - S3: SOLVE — global rule (expr + variable transforms)
     - S4: APPLY — per-bit value computation in serial (Donald's key insight)
     - S5: ANS + \boxed{}

Output format matches dgxchen baseline (5 cols: id, prompt, answer, type, generated_cot)
ready for SYNTH_ADD_GLOBS injection.

Coverage: 1584/1602 (98.88%); 18 rows with solver-wrong predictions skipped.
For those 18 + 0 unsolved, training kernel falls back to dgxchen original CoT
via 'bit_v3_fallback.csv' helper.
"""

from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path
import pandas as pd

REPO = Path('/Users/exerzitien/Desktop/nemotron-kaggle')
sys.path.insert(0, str(REPO / 'research'))
from majortom_bit_solver import (
    OPS, TRANSFORMATIONS, get_source_bit, evaluate_bit,
    generate_grammar_dynamically, solve_dfs_trace_dynamic,
    parse_bit_problem,
)

TYPED_CSV = REPO / 'research/train_with_type.csv'
DGXCHEN_CSV = REPO / 'datagen/_ref/problem_ids_matched.csv'
OUT_DIR = REPO / 'datagen/synth_dataset_v3'
OUT_CSV = OUT_DIR / 'bit_manipulation_v3.csv'
FALLBACK_CSV = OUT_DIR / 'bit_v3_fallback.csv'  # for unsolved + wrong predictions


def _source_pos_desc(trans):
    """Describe transformation as 'shl(k)' / 'shr(k)' / 'rot(k)'."""
    ttype, shift = trans
    return f"{ttype}({shift})"


def _source_bit_str(bit_idx, trans):
    """Describe source position for output bit_idx given transformation."""
    ttype, shift = trans
    if ttype == 'rot':
        return f"in[{(bit_idx + shift) % 8}]"
    elif ttype == 'shl':
        src = bit_idx + shift
        return f"in[{src}]" if 0 <= src < 8 else "0(oor)"
    elif ttype == 'shr':
        src = bit_idx - shift
        return f"in[{src}]" if 0 <= src < 8 else "0(oor)"


def _eval_var_at(in_bits, bit_idx, trans):
    """Get the actual value of a variable at output bit_idx."""
    return get_source_bit(in_bits, bit_idx, trans)


def _eval_expr_at_bit(predictor, query_in, bit_idx):
    """Use the solver's predictor to compute output[bit_idx]."""
    pred = predictor(query_in)
    return int(pred[bit_idx])


def gen_cot_for_row(in_arrays, out_arrays, query_in, expr, trans_dict, predictor):
    """Generate full CoT string for one bit problem."""
    n_ex = len(in_arrays)
    pred_query = predictor(query_in)
    answer = ''.join(pred_query)

    lines = []
    lines.append("We need to deduce the per-bit transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append("S1: This is a binary boolean decomposition. Each output bit is a boolean function "
                 "of input bits at fixed positions, derived from rotation/shift transformations.")
    lines.append("")

    # S2: COLUMNS
    lines.append("S2: COLUMNS (read input/output column at each bit position across all examples)")
    in_cols = []
    for bit in range(8):
        col = ''.join(str(in_arrays[ex][bit]) for ex in range(n_ex))
        in_cols.append(col)
    out_cols = []
    for bit in range(8):
        col = ''.join(str(out_arrays[ex][bit]) for ex in range(n_ex))
        out_cols.append(col)
    lines.append("  IN columns: " + " ".join(f"i{b}={in_cols[b]}" for b in range(8)))
    lines.append("  OUT columns: " + " ".join(f"o{b}={out_cols[b]}" for b in range(8)))
    lines.append("  TGT: " + ''.join(str(b) for b in query_in))
    lines.append("")

    # S3: SOLVE — describe the rule
    lines.append("S3: SOLVE — found global rule consistent with all examples:")
    lines.append(f"   expr = {expr}")
    if trans_dict:
        var_descs = []
        for var, trans in trans_dict.items():
            var_descs.append(f"{var}={_source_pos_desc(trans)}")
        lines.append(f"   var bindings: {', '.join(var_descs)}")
        lines.append(f"   for each output bit i: {expr} where each variable is read at a transformed position")
    else:
        lines.append("   (constant or simple identity; no variable bindings)")
    lines.append("")

    # Quick verify on first 2 examples to give the model evidence
    lines.append("S3b: VERIFY on first 2 examples (showing the rule applies):")
    n_verify = min(2, n_ex)
    for ex in range(n_verify):
        in_str = ''.join(str(b) for b in in_arrays[ex])
        out_str = ''.join(str(b) for b in out_arrays[ex])
        lines.append(f"  Example {ex}: in={in_str} -> out={out_str}")
        # Compute predicted from this input
        computed = predictor(in_arrays[ex])
        match = "✓ match" if computed == out_str else f"✗ predicted {computed}"
        lines.append(f"    rule predicts: {computed} {match}")
    lines.append("")

    # S4: APPLY to query — bit-serial computation
    lines.append("S4: APPLY to TGT (bit-serial computation, one bit at a time)")
    tgt_str = ''.join(str(b) for b in query_in)
    lines.append(f"  TGT bits: " + " ".join(f"t{i}={query_in[i]}" for i in range(8)))
    for bit_idx in range(8):
        # Compute values of each variable at this bit
        var_values = []
        for var, trans in (trans_dict or {}).items():
            val = _eval_var_at(query_in, bit_idx, trans)
            src = _source_bit_str(bit_idx, trans)
            var_values.append(f"{var}={src}={val}")
        result_bit = int(pred_query[bit_idx])
        if var_values:
            lines.append(f"  bit {bit_idx}: " + ", ".join(var_values)
                         + f"; {expr} = {result_bit}")
        else:
            lines.append(f"  bit {bit_idx}: constant rule -> {result_bit}")

    lines.append("")
    lines.append(f"S5: ANS={answer}")
    lines.append(f"\\boxed{{{answer}}}")
    return '\n'.join(lines)


def main():
    typed = pd.read_csv(TYPED_CSV)
    dgxchen = pd.read_csv(DGXCHEN_CSV)
    # We need original prompt from train.csv (typed contains id+type+...)
    # train.csv has id, prompt, answer
    train = pd.read_csv(REPO / 'data/train.csv')

    bit_ids = typed[typed['type'] == 'bit_manipulation']['id'].tolist()
    print(f"Bit_manipulation problems: {len(bit_ids)}")

    out_rows = []
    fallback_ids = []
    stats = {'attempted': 0, 'gen_ok': 0, 'solver_failed': 0, 'pred_wrong': 0}
    t0 = time.time()
    for idx, bid in enumerate(bit_ids):
        stats['attempted'] += 1
        row_train = train[train['id'] == bid].iloc[0]
        prompt = row_train['prompt']
        gold = str(row_train['answer']).strip().zfill(8)

        in_arrays, out_arrays, query_in = parse_bit_problem(prompt)
        if in_arrays is None:
            stats['solver_failed'] += 1
            fallback_ids.append(bid)
            continue

        trace, predictor, expr, trans_dict = solve_dfs_trace_dynamic(
            in_arrays, out_arrays, len(in_arrays), time_limit=5.0,
        )
        if predictor is None:
            stats['solver_failed'] += 1
            fallback_ids.append(bid)
            continue

        pred_query = predictor(query_in)
        if pred_query != gold:
            stats['pred_wrong'] += 1
            fallback_ids.append(bid)
            continue

        # Generate CoT
        cot = gen_cot_for_row(in_arrays, out_arrays, query_in, expr, trans_dict, predictor)
        out_rows.append({
            'id': bid,
            'prompt': prompt,
            'answer': gold,
            'type': 'bit_manipulation',
            'generated_cot': cot,
        })
        stats['gen_ok'] += 1

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{len(bit_ids)}: ok={stats['gen_ok']}, "
                  f"fallback={len(fallback_ids)}, elapsed={elapsed:.0f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_CSV, index=False)

    # Write fallback CSV with original dgxchen rows for IDs we didn't generate
    fallback_df = dgxchen[
        (dgxchen['type'] == 'bit_manipulation') & (dgxchen['id'].isin(fallback_ids))
    ]
    fallback_df.to_csv(FALLBACK_CSV, index=False)

    print(f"\n=== bit_gen_v3 done ===")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {len(df)} rows to {OUT_CSV}")
    print(f"Wrote {len(fallback_df)} fallback rows to {FALLBACK_CSV}")
    print(f"Total bit coverage: {len(df) + len(fallback_df)} / {len(bit_ids)}")

    # CoT length stats
    if len(df) > 0:
        df['cot_chars'] = df['generated_cot'].str.len()
        print(f"\nCoT chars: min={df.cot_chars.min()}, "
              f"p50={df.cot_chars.median():.0f}, "
              f"p95={df.cot_chars.quantile(0.95):.0f}, "
              f"max={df.cot_chars.max()}")


if __name__ == "__main__":
    main()
