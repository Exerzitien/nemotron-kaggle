"""
V33 adapter evaluation on local RTX 5090 (32GB VRAM)
=====================================================
Target: Diagnose WHY V33 LB dropped from 0.84 baseline to 0.73.
Output: per-category accuracy on jiazhuang/nemotron-val-set-950.

Requirements (5090 + WSL Ubuntu):
  - CUDA 12.4+ (Blackwell sm_120)
  - vLLM 0.7+ with bitsandbytes support
  - ~60GB disk for Nemotron base model
  - 32GB VRAM (INT4 quant + LoRA + KV cache)

Inputs expected at fixed paths (see SETUP block below):
  - Base model:     ~/models/Nemotron-3-Nano-30B-A3B-BF16/
  - V33 adapter:    ~/adapters/v33/{adapter_config.json, adapter_model.safetensors}
  - Val set:        ~/data/val_set_950.jsonl

Output:
  - ~/eval_results/v33_results.csv       (per-row id/category/gold/pred/correct/raw)
  - ~/eval_results/v33_per_class.txt     (per-category accuracy summary)

Usage:
  python eval_v33_on_5090.py
"""
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# ============================================================
# SETUP — adjust these paths if files are elsewhere
# ============================================================
HOME = Path.home()
BASE_MODEL = HOME / "models" / "Nemotron-3-Nano-30B-A3B-BF16"
LORA_PATH = HOME / "adapters" / "v33"
VAL_JSONL = HOME / "data" / "val_set_950.jsonl"
OUT_DIR = HOME / "eval_results"

MAX_LORA_RANK = 32
MAX_MODEL_LEN = 8192
MAX_TOKENS = 7680
TEMPERATURE = 0.0


# ============================================================
# Official metric/extract/verify (post-2026-05-08 spec)
# ============================================================
def extract_final_answer(text):
    if text is None:
        return "NOT_FOUND"
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pat in patterns:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored, predicted):
    s, p = str(stored).strip(), str(predicted).strip()
    try:
        return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return p.lower() == s.lower()


def detect_category(prompt):
    """Copied from kernels/eval/nemotron-eval.ipynb — proven category detector."""
    if "secret bit manipulation rule transforms 8-bit binary numbers" in prompt:
        return "bit_manipulation"
    if "secret encryption rules are used on text" in prompt:
        return "cipher"
    if "secret set of transformation rules is applied to equations" in prompt:
        after_header = prompt.split("Below are a few examples:\n", 1)[1]
        examples_text, rest = after_header.split("\nNow, determine the result for: ", 1)
        question_text = rest.strip()
        if any(c.isdigit() for c in examples_text):
            q_match = re.fullmatch(r"(\d+)(\D)(\d+)", question_text)
            if q_match and re.search(r"\d" + re.escape(q_match.group(2)) + r"\d", examples_text):
                return "equation_numeric_deduce"
            return "equation_numeric_guess"
        if len(question_text) == 5:
            q_op = question_text[2]
            for ex_line in examples_text.strip().splitlines():
                inp = ex_line.split(" = ")[0].strip()
                if len(inp) == 5 and inp[2] == q_op:
                    return "cryptarithm_deduce"
        return "cryptarithm_guess"
    if "gravitational constant has been secretly changed" in prompt:
        return "gravity"
    if "converted into a different numeral system" in prompt:
        return "numeral"
    if "secret unit conversion is applied to measurements" in prompt:
        return "unit_conversion"
    return "unknown"


# ============================================================
# Main eval
# ============================================================
def main():
    print(f"Python: {sys.version.split()[0]}")
    import torch
    print(f"Torch: {torch.__version__} CUDA: {torch.cuda.is_available()} Devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU 0: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Sanity checks
    for p, label in [(BASE_MODEL, "BASE_MODEL"), (LORA_PATH, "LORA_PATH"), (VAL_JSONL, "VAL_JSONL")]:
        assert p.exists(), f"{label} missing: {p}"
        print(f"  ✓ {label}: {p}")

    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load val_950
    val = []
    with open(VAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                val.append(json.loads(line))
    print(f"\nLoaded val_set_950: {len(val)} rows")

    # Categorize
    for item in val:
        item["category"] = detect_category(item["prompt"])
    from collections import Counter
    cat_dist = Counter(item["category"] for item in val)
    print(f"Categories: {dict(sorted(cat_dist.items()))}")

    # vLLM + LoRA + bitsandbytes 4bit
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("\nInitializing vLLM (this may take 3-5 min)...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=str(BASE_MODEL),
        quantization="bitsandbytes",      # 4-bit weight quant, 5090 32GB friendly
        load_format="bitsandbytes",
        tensor_parallel_size=1,
        max_num_seqs=4,                   # conservative on 32GB
        gpu_memory_utilization=0.92,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    print(f"  vLLM loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    tokenizer = llm.get_tokenizer()
    prompts = []
    for item in val:
        user_content = item["prompt"] + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            prompt = user_content
        prompts.append(prompt)

    print(f"\nRunning inference on {len(prompts)} prompts (LoRA = V33)...")
    t0 = time.time()
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("v33", 1, str(LORA_PATH)),
    )
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s ({len(prompts) / elapsed:.2f} prompts/s)")

    # Score
    results = []
    for item, output in zip(val, outputs):
        raw = output.outputs[0].text
        pred = extract_final_answer(raw)
        gold = str(item["answer"])
        is_correct = verify(gold, pred)
        results.append({
            "id": item["id"],
            "category": item["category"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "raw_len": len(raw),
            "raw_tail": raw[-200:],
        })

    # Save full results
    import pandas as pd
    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "v33_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-row results: {out_csv}")

    # Per-category accuracy
    summary_lines = ["V33 adapter evaluation on val_set_950", "=" * 50]
    summary_lines.append(f"Overall: {df['correct'].sum()}/{len(df)} ({100*df['correct'].mean():.2f}%)")
    summary_lines.append("")
    summary_lines.append(f"{'category':30s} {'correct':>8s} {'total':>8s} {'pct':>8s}")
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        c = sub["correct"].sum()
        t = len(sub)
        summary_lines.append(f"{cat:30s} {c:>8d} {t:>8d} {100*c/t:>7.2f}%")

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (OUT_DIR / "v33_per_class.txt").write_text(summary)

    # Mistakes by category (for diagnosis)
    mistakes_dir = OUT_DIR / "v33_mistakes"
    mistakes_dir.mkdir(exist_ok=True)
    for cat in df["category"].unique():
        wrong = df[(df["category"] == cat) & (~df["correct"])]
        if len(wrong) > 0:
            wrong.to_csv(mistakes_dir / f"{cat}.csv", index=False)
    print(f"\nMistake CSVs by category: {mistakes_dir}")
    print(f"\n✅ Done. Send back v33_per_class.txt for analysis.")


if __name__ == "__main__":
    main()
