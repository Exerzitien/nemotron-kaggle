"""
V35: huikang-style bit_manipulation CoT generator
==================================================
Strategy:
  1. Use huikang reasoning/*.txt for 1364 'rule_found' cases (huikang correct)
  2. For 238 huikang-failed cases:
     - If MAJ0RT0M solved (223 cases): replace huikang's last \\boxed{} with
       correct answer (huikang reasoning otherwise preserved — same format)
     - If MAJ0RT0M also failed (15 cases): keep huikang verbatim

Why this is better than V33's MAJ0RT0M Donald-style swap:
  - V33 replaced all 1584 bit_manipulation with Donald global expressions
    → completely different CoT format from dgxchen → model confused → LB -0.11
  - V35 preserves huikang's per-bit 354-binary-enumeration format
    → model sees same format as dgxchen baseline + correct answers

Output: bit_manipulation_v4.csv (1602 rows, 5 cols matching dgxchen schema)
"""
import os
import json
import re
import pandas as pd

HUIKANG_REASONING_DIR = '/Users/exerzitien/Desktop/nemotron-kaggle/huikang_ref/reasoning'
TRAIN_CSV = '/Users/exerzitien/Desktop/nemotron-kaggle/data/train.csv'
PROBLEMS_JSONL = '/Users/exerzitien/Desktop/nemotron-kaggle/huikang_ref/problems.jsonl'
MAJORTOM_CSV = '/Users/exerzitien/Desktop/nemotron-kaggle/research/majortom_results.csv'
DGX_REF = '/Users/exerzitien/Desktop/nemotron-kaggle/datagen/_ref/problem_ids_matched.csv'
OUT_CSV = '/Users/exerzitien/Desktop/nemotron-kaggle/datagen/synth_dataset_v3/bit_manipulation_v4.csv'


def main():
    print("Loading inputs...")
    train = pd.read_csv(TRAIN_CSV, dtype={'answer': str})
    train_by_id = dict(zip(train['id'], zip(train['prompt'], train['answer'])))

    probs = {}
    with open(PROBLEMS_JSONL) as f:
        for line in f:
            p = json.loads(line)
            probs[p['id']] = p

    majortom = pd.read_csv(MAJORTOM_CSV)
    maj_by_id = {row['id']: row for _, row in majortom.iterrows()}

    # Use dgxchen baseline as schema source (id/prompt/answer/type/generated_cot)
    dgx = pd.read_csv(DGX_REF)
    dgx_bit_ids = set(dgx[dgx['type'] == 'bit_manipulation']['id'])

    bit_ids = [pid for pid, p in probs.items() if p['category'] == 'bit_manipulation']
    print(f"Total bit_manipulation: {len(bit_ids)}")
    print(f"In dgxchen baseline: {len(set(bit_ids) & dgx_bit_ids)}")

    rows = []
    stats = {'huikang_correct': 0, 'majortom_fix': 0, 'both_failed': 0, 'no_majortom': 0}

    for pid in bit_ids:
        if pid not in train_by_id:
            continue
        prompt, gold_answer = train_by_id[pid]
        gold_answer = str(gold_answer)

        # Load huikang reasoning
        rfile = f'{HUIKANG_REASONING_DIR}/{pid}.txt'
        if not os.path.exists(rfile):
            continue
        with open(rfile) as f:
            cot = f.read()

        # Decide which CoT to use
        status = probs[pid]['status']

        if status == 'rule_found':
            # huikang got it right; use as-is
            stats['huikang_correct'] += 1
            base_cot = cot
            final_answer = gold_answer
        elif pid in maj_by_id and maj_by_id[pid]['correct']:
            # huikang failed but MAJ0RT0M solved; replace last \boxed with correct
            stats['majortom_fix'] += 1
            new_cot = re.sub(
                r'\\boxed\{[^}]*\}(\s*)$',
                f'\\\\boxed{{{gold_answer}}}\\1',
                cot,
                count=1
            )
            last_box = re.findall(r'\\boxed\{([^}]*)\}', new_cot)[-1]
            if last_box != gold_answer:
                new_cot = re.sub(r'\\boxed\{[^}]*\}\s*$', '', cot).rstrip()
                new_cot = new_cot + f'\n\\boxed{{{gold_answer}}}'
            base_cot = new_cot
            final_answer = gold_answer
        else:
            if pid in maj_by_id:
                stats['both_failed'] += 1
            else:
                stats['no_majortom'] += 1
            base_cot = cot
            last_box_matches = re.findall(r'\\boxed\{([^}]*)\}', cot)
            final_answer = last_box_matches[-1] if last_box_matches else gold_answer

        # CRITICAL: match dgxchen footer format
        # dgxchen baseline appended `\n</think>\n\\boxed{answer}` to huikang's
        # reasoning. We must do the same so kernel cell 11 pipeline produces
        # identical training text format as dgxchen baseline.
        final_cot = base_cot.rstrip() + f'\n</think>\n\\boxed{{{final_answer}}}'

        rows.append({
            'id': pid,
            'prompt': prompt,
            'answer': final_answer,
            'type': 'bit_manipulation',
            'generated_cot': final_cot,
        })

    df = pd.DataFrame(rows)
    print(f"\nGenerated: {len(df)} rows")
    print(f"Stats: {stats}")

    # Sanity check: how many rows have answer == train gold
    df['_gold'] = df['id'].map(lambda x: train_by_id[x][1] if x in train_by_id else None)
    correct = (df['answer'] == df['_gold']).sum()
    print(f"answer == train gold: {correct}/{len(df)} ({100*correct/len(df):.1f}%)")
    df = df.drop(columns=['_gold'])

    # Save with proper string handling for leading zeros
    import csv as _csv
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, quoting=_csv.QUOTE_NONNUMERIC)
    print(f"\n✅ Saved: {OUT_CSV}")
    print(f"   File size: {os.path.getsize(OUT_CSV):,} bytes")

    # Verify roundtrip — read back and check answer integrity
    reread = pd.read_csv(OUT_CSV, dtype={'answer': str})
    ans_lens = reread['answer'].str.len().value_counts().to_dict()
    print(f"   Answer length distribution: {ans_lens}")
    if 8 in ans_lens:
        print(f"   {ans_lens[8]} rows have 8-char answer (full bit pattern)")
    if any(L < 8 for L in ans_lens):
        print(f"   ⚠️  Some answers < 8 chars (leading zero stripped by upstream)")


if __name__ == '__main__':
    main()
