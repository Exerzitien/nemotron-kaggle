"""
V35 Tinker SFT training entry
==============================

Train LoRA adapter via Tinker for V35 = dgxchen (minus cryptarithm + bit) +
km_cryptarithm + bit_v4. Adapted from huikang/train_sft.py minimally.

Usage:
    export TINKER_API_KEY=...  # or place in ~/.tinker_key
    python train_v35_tinker.py --version v35b --dry-run    # estimate cost
    python train_v35_tinker.py --version v35b              # actual run

Versions:
    v35a: dgxchen + km_cryptarithm only       (sanity: should match V34 = 0.85)
    v35b: v35a + bit_manipulation_v4 swap     (target: 0.86)
    v35c: dgxchen + bit_v4 only (no km swap)  (isolate bit_v4 effect)
"""
import argparse
import asyncio
import math
import os
import random
import re
import time
from pathlib import Path

import pandas as pd

# ============================================================
# Config
# ============================================================
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
LORA_RANK = 32
LR_PEAK = 2e-4              # Adam scale (huikang verified for Tinker)
NUM_EPOCHS = 1
BATCH_SIZE = 64
MAX_SEQ = 8192
SEED = 42

REPO_ROOT = Path(__file__).parent.parent
DGX_REF = REPO_ROOT / "datagen" / "_ref" / "problem_ids_matched.csv"
DATA_DIR = REPO_ROOT / "datagen" / "synth_dataset_v3"

VERSIONS = {
    "v35a": {
        "drop": ["cryptarithm_deduce", "cryptarithm_guess"],
        "add": ["kimberleyduran_cryptarithm.csv"],
        "desc": "V34 baseline (Adam) — sanity check Tinker reaches 0.85",
    },
    "v35b": {
        "drop": ["cryptarithm_deduce", "cryptarithm_guess", "bit_manipulation"],
        "add": ["kimberleyduran_cryptarithm.csv", "bit_manipulation_v4.csv"],
        "desc": "V35 main: V34 + bit_v4 swap (target 0.86)",
    },
    "v35c": {
        "drop": ["bit_manipulation"],
        "add": ["bit_manipulation_v4.csv"],
        "desc": "bit_v4 ONLY swap (no km cryptarithm) — isolates bit_v4 effect",
    },
}


# ============================================================
# Data prep
# ============================================================
def load_data(version):
    cfg = VERSIONS[version]
    dgx = pd.read_csv(DGX_REF, dtype={"answer": str})
    df = dgx[~dgx["type"].isin(cfg["drop"])].reset_index(drop=True)
    print(f"\n[{version}] {cfg['desc']}")
    print(f"  Baseline (after drop): {len(df)} rows")
    for fname in cfg["add"]:
        extra = pd.read_csv(DATA_DIR / fname, dtype={"answer": str})
        extra = extra[df.columns]
        df = pd.concat([df, extra], ignore_index=True)
        print(f"  Added {fname}: +{len(extra)} (total: {len(df)})")
    print(f"  Composition: {dict(sorted(df['type'].value_counts().items()))}")
    return df


def _extract_ids(result):
    """apply_chat_template returns BatchEncoding (Tinker) or list (HF wrapper).
    Normalize to list of int token IDs."""
    # BatchEncoding / Encoding (Tinker TokenizersBackend)
    if hasattr(result, "__getitem__") and len(result) > 0:
        first = result[0] if not isinstance(result, list) else result[0]
        if hasattr(first, "ids"):
            return list(first.ids)
    # Already a flat list of ints
    if isinstance(result, list) and (len(result) == 0 or isinstance(result[0], int)):
        return result
    # BatchEncoding dict access
    if hasattr(result, "get"):
        ids = result.get("input_ids")
        if ids is not None:
            return ids[0] if isinstance(ids[0], list) else list(ids)
    raise TypeError(f"Unknown tokenizer result type: {type(result).__name__}")


def build_examples(df, tokenizer):
    """Build training examples (tokens + weights mask)."""
    SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
    examples = []
    skipped = 0
    errors = 0

    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        cot = str(row["generated_cot"])
        cat = str(row["type"])

        if not cot or cot == "nan" or len(cot.strip()) < 5:
            skipped += 1
            continue

        user_msg = prompt + SUFFIX
        cot_cleaned = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
        assistant_msg = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"

        try:
            p_raw = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=True, add_generation_prompt=True, enable_thinking=True,
            )
            f_raw = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                tokenize=True, add_generation_prompt=False, enable_thinking=True,
            )
            prompt_tokens = _extract_ids(p_raw)
            full_tokens = _extract_ids(f_raw)
        except Exception as e:
            if errors < 3:
                print(f"  tokenization error on row {row['id']}: {e}")
            errors += 1
            skipped += 1
            continue

        if len(full_tokens) > MAX_SEQ:
            full_tokens = full_tokens[:MAX_SEQ]
        if len(full_tokens) <= len(prompt_tokens):
            skipped += 1
            continue

        n_prompt = min(len(prompt_tokens), len(full_tokens))
        weights = [0.0] * n_prompt + [1.0] * (len(full_tokens) - n_prompt)

        examples.append({"tokens": full_tokens, "weights": weights, "category": cat})

    print(f"\nBuilt {len(examples)} examples (skipped {skipped}, errors {errors})")
    return examples


def stratified_batches(examples, batch_size, rng):
    """Stratified by category, approximating huikang's strategy."""
    by_cat = {}
    for i, ex in enumerate(examples):
        by_cat.setdefault(ex["category"], []).append(i)
    for ids in by_cat.values():
        rng.shuffle(ids)

    n_batches = max(1, math.ceil(len(examples) / batch_size))
    batches = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)

    assigned = 0
    for cat in sorted(by_cat.keys()):
        for idx in by_cat[cat]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1

    for batch in batches:
        rng.shuffle(batch)
    return batches


# ============================================================
# Tinker training
# ============================================================
async def run_training(args):
    # Load data first (cheap, no Tinker needed)
    df = load_data(args.version)

    # Load API key
    if not os.environ.get("TINKER_API_KEY"):
        key_path = Path("~/.tinker_key").expanduser()
        if key_path.exists():
            os.environ["TINKER_API_KEY"] = key_path.read_text().strip()
    assert os.environ.get("TINKER_API_KEY"), \
        "Need TINKER_API_KEY env var or ~/.tinker_key file"

    # For dry-run, skip Tinker init (no cost). Use char-based token estimate.
    if args.dry_run:
        print(f"\n[DRY RUN] Char-based token estimate (skip Tinker init, no cost)...")
        tokenizer = None  # use char-based fallback below
        tc = None
    else:
        import tinker
        from tinker import types
        globals()["types"] = types  # used in loop below
        print(f"\nInitializing Tinker (model={MODEL}, rank={LORA_RANK})...")
        sc = tinker.ServiceClient()
        tc = await sc.create_lora_training_client_async(
            base_model=MODEL,
            rank=LORA_RANK,
            seed=SEED,
            user_metadata={"version": args.version, "epochs": str(args.epochs)},
        )
        tokenizer = tc.get_tokenizer()
        print("  ✓ Training client created")

    # Build examples (with whichever tokenizer)
    if tokenizer is None:
        # Char-based rough estimate
        examples = []
        for _, row in df.iterrows():
            cot = str(row["generated_cot"])
            if not cot or len(cot) < 5:
                continue
            est_tokens = int((len(str(row["prompt"])) + len(cot) + len(str(row["answer"])) + 200) / 4)
            examples.append({"tokens": [0] * est_tokens, "weights": [], "category": str(row["type"])})
    else:
        examples = build_examples(df, tokenizer)

    # Cost estimate
    total_tokens = sum(len(e["tokens"]) for e in examples)
    cost_full = total_tokens * args.epochs * 0.40 / 1e6
    cost_half = cost_full / 2
    print(f"\nTotal training tokens: {total_tokens:,} × {args.epochs} epoch(s) = {total_tokens * args.epochs:,}")
    print(f"Estimated Tinker cost: ${cost_full:.2f} full / ${cost_half:.2f} (50% promo)")

    if args.dry_run:
        print("\n[DRY RUN] Stopping before training.")
        return

    if cost_full > args.max_cost:
        print(f"\n❌ Cost exceeds --max-cost (${args.max_cost}). Aborting.")
        print(f"   Add --max-cost {math.ceil(cost_full)} to override")
        return

    # Training loop
    rng = random.Random(SEED)
    n_batches = max(1, math.ceil(len(examples) / BATCH_SIZE))
    total_steps = n_batches * args.epochs
    print(f"\nTraining: {n_batches} batches × {args.epochs} epochs = {total_steps} steps")
    print(f"Effective batch size: {BATCH_SIZE}")
    print(f"Peak LR: {LR_PEAK} (linear decay to 0)")

    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        batches = stratified_batches(examples, BATCH_SIZE, rng)
        for batch_idx, batch_indices in enumerate(batches):
            # Build Datum list
            data = []
            for idx in batch_indices:
                ex = examples[idx]
                tokens = ex["tokens"]
                weights = ex["weights"]
                target_tokens = tokens[1:]
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": types.TensorData(
                            data=target_tokens, dtype="int64", shape=[len(target_tokens)],
                        ),
                        "weights": types.TensorData(
                            data=weights[1:], dtype="float32", shape=[len(weights) - 1],
                        ),
                    },
                )
                data.append(datum)

            # Forward backward
            fb = await tc.forward_backward_async(data=data, loss_fn="cross_entropy")
            fb_result = await fb.result_async()

            # LR linear decay
            lr_now = LR_PEAK * max(0.0, 1.0 - step / max(1, total_steps))

            # Optim step
            opt = await tc.optim_step_async(types.AdamParams(learning_rate=lr_now))
            await opt.result_async()

            step += 1

            if step % args.log_every == 0 or step == total_steps:
                elapsed = time.time() - t_start
                eta = elapsed / step * (total_steps - step)
                # Try to extract loss
                loss_val = "n/a"
                try:
                    if hasattr(fb_result, "loss"):
                        loss_val = f"{fb_result.loss:.4f}"
                    elif hasattr(fb_result, "metrics"):
                        loss_val = str(fb_result.metrics)[:60]
                except Exception:
                    pass
                print(f"  step {step:>4d}/{total_steps}  lr={lr_now:.2e}  "
                      f"loss={loss_val}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

            # Periodic checkpoint
            if args.save_every > 0 and step % args.save_every == 0 and step < total_steps:
                ckpt_name = f"{args.version}-step{step}"
                print(f"  saving intermediate checkpoint: {ckpt_name}")
                ckpt = await tc.save_weights_for_sampler_async(name=ckpt_name)
                await ckpt.result_async()

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    final = await tc.save_weights_for_sampler_async(name=f"{args.version}-final")
    final_result = await final.result_async()
    print(f"  Tinker path: {final_result.path}")

    total_elapsed = time.time() - t_start
    print(f"\n✅ Training complete in {total_elapsed/60:.1f} min")
    print(f"   Tinker checkpoint: {final_result.path}")
    print(f"\nNext step: run tinker_to_kaggle.py to convert + upload to Kaggle dataset")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=list(VERSIONS.keys()), default="v35b")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--dry-run", action="store_true", help="estimate cost and stop")
    p.add_argument("--max-cost", type=float, default=20.0, help="abort if cost > $X (default 20)")
    p.add_argument("--log-every", type=int, default=10, help="log every N steps")
    p.add_argument("--save-every", type=int, default=0, help="save intermediate ckpt every N steps (0 = only final)")
    args = p.parse_args()
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
