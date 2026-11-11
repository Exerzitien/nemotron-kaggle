"""Programmatic generator for the `cipher` category of the Nemotron Reasoning Challenge.

Each problem is a monoalphabetic substitution cipher over a closed "Wonderland"
vocabulary. We own the key, so every example, the query, and the answer are exact
by construction, and the chain-of-thought is the real decryption algorithm.

Output columns match dgxchen/nemotron-cot-tong (problem_ids_matched.csv):
    id, prompt, answer, type, generated_cot
so generated rows drop straight into the existing SFT pipeline.
"""

import argparse
import csv
import os
import random
import string

PHRASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wonderland_phrases.txt")
LETTERS = string.ascii_lowercase

PROMPT_HEADER = (
    "In Alice's Wonderland, secret encryption rules are used on text. "
    "Here are some examples:"
)


def load_phrases(path=PHRASES_FILE):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def make_key(rng):
    """Random monoalphabetic substitution: plaintext letter -> ciphertext letter."""
    shuffled = list(LETTERS)
    rng.shuffle(shuffled)
    return dict(zip(LETTERS, shuffled))


def encrypt(text, key):
    return "".join(key.get(c, c) for c in text)


def _build_table_cot(examples):
    """examples: list of (ciphertext, plaintext). Returns (cot_lines, cipher->plain table)."""
    table = {}
    lines = [
        "These are monoalphabetic substitution ciphers: every ciphertext letter "
        "maps to one fixed plaintext letter. I build the table from the examples."
    ]
    for ct, pt in examples:
        new = []
        for c, p in zip(ct, pt):
            if c == " ":
                continue
            if c not in table:
                table[c] = p
                new.append(f"{c}->{p}")
        shown = ", ".join(new) if new else "(no new letters)"
        lines.append(f'  "{ct}" = "{pt}"  gives  {shown}')
    pairs = ", ".join(f"{c}->{table[c]}" for c in sorted(table))
    lines.append(f"Substitution table so far (cipher->plain): {pairs}")
    return lines, table


def _decrypt_query_cot(query_ct, query_pt, table):
    """Decrypt the query: apply the table, then resolve unknown letters from context."""
    ct_words = query_ct.split()
    pt_words = query_pt.split()
    partials = ["".join(table.get(c, "_") for c in cw) for cw in ct_words]
    lines = [
        f'Now decrypt the query "{query_ct}". Applying the table letter by letter '
        f'gives: "{" ".join(partials)}".'
    ]
    if not any("_" in pp for pp in partials):
        return lines
    lines.append(
        "Some letters never appeared in the examples (shown as _). "
        "I recover them from the surrounding decrypted words."
    )
    for cw, pw, pp in zip(ct_words, pt_words, partials):
        if "_" not in pp:
            continue
        fixes = ", ".join(f"{c}->{pw[i]}" for i, c in enumerate(cw) if c not in table)
        lines.append(f'  "{pp}" can only be "{pw}" in context  ({fixes})')
        for c, p in zip(cw, pw):
            table[c] = p
    return lines


def make_problem(rng, phrases):
    key = make_key(rng)
    n_ex = rng.randint(3, 5)
    picked = rng.sample(phrases, n_ex + 1)
    ex_pt, query_pt = picked[:n_ex], picked[n_ex]
    examples = [(encrypt(pt, key), pt) for pt in ex_pt]
    query_ct = encrypt(query_pt, key)

    prompt = PROMPT_HEADER + "\n"
    prompt += "\n".join(f"{ct} -> {pt}" for ct, pt in examples)
    prompt += f"\nNow, decrypt the following text: {query_ct}"

    table_lines, table = _build_table_cot(examples)
    query_lines = _decrypt_query_cot(query_ct, query_pt, dict(table))
    cot = "\n".join(table_lines + query_lines)
    cot += f'\nDecrypted text: {query_pt}\n\\boxed{{{query_pt}}}'

    return {"prompt": prompt, "answer": query_pt, "type": "cipher", "generated_cot": cot}


def validate(prob):
    """Self-check: the boxed answer in the CoT equals answer, and prompt is well-formed."""
    assert prob["prompt"].count(" -> ") >= 3, "too few examples"
    assert "decrypt the following text:" in prob["prompt"]
    assert prob["generated_cot"].rstrip().endswith("\\boxed{" + prob["answer"] + "}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()

    phrases = load_phrases()
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        p = make_problem(rng, phrases)
        validate(p)
        p["id"] = f"cipher_gen_{args.seed}_{i}"
        rows.append(p)

    n_uncovered = sum(1 for p in rows if "_ in context" in p["generated_cot"]
                      or "recover them" in p["generated_cot"])
    print(f"generated + validated {len(rows)} cipher problems")
    print(f"  with uncovered letters (harder, teaches context-fill): "
          f"{n_uncovered}/{len(rows)} ({100*n_uncovered/len(rows):.0f}%)  "
          f"[real competition data ~62%]")
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
