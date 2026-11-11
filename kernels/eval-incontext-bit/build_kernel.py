"""构建 Tier1 in-context eval notebook —— 从被验证的 eval-v39 notebook 派生。

前两版手写 cell（pip install --no-index vllm、缺 TRANSFORMERS_NO_TF）触发
protobuf `'MessageFactory' object has no attribute 'GetPrototype'`（TF 被导入）而 ERROR。
本版改为像 eval-bit-holdout 一样从 eval-v39 锚点替换，继承全部环境 cell
（cell 6 metric utility / cell 11 TRANSFORMERS_NO_TF + vLLM import / cell 13 LLM+LoRA），
只换 5 个实验 cell：

  cell15  数据源    : val_set_950 -> bit_holdout_depth2.csv（depth>=2 全新留出）+ 从
                      bit_essence_mixed.csv 取 1 条精华 demo -> DEMO_BLOCK
  cell16  prompt    : 单 prompts -> prompts_zero（zero-shot）+ prompts_demo（1-shot 精华前缀）
  cell18  生成      : 单遍 -> 跑两遍 outputs_zero / outputs_demo（每条件独立 max_tokens）
  cell20  计分      : 单列 correct -> correct_zero / correct_demo
  cell21  出分      : 按 layer 分组 -> acc_zero / acc_demo / lift + results.csv + raw_outputs.csv

仪器=V34 adapter（cell 1-2 自动定位 raw adapter / submission.zip，cell 18 lora_path 不变）。
base 模型不产 boxed/不停（前两次 0/0 根因），故用 V34 当仪器，看 demo 是否拉高 depth>=2 acc。
判据：lift >= +15pp（绝对 acc 在 adapter 条件下 apples-to-oranges，lift 才是信号）。
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, "..", "eval-v39", "nemotron-eval-v39.ipynb")
DST = os.path.join(_HERE, "nemotron-eval-incontext-bit.ipynb")

# ── cell6 替换：metric utility 路径鲁棒化 + 自诊断 ──────────────────────
# 原 cell 硬编码 /kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script，
# CLI push 的 kernel_sources 偶发挂载到别处或没挂上 -> 改为 glob 动态定位；
# 若真没挂，打印 /kaggle/usr/lib 清单并在 ~3min 处快速失败（拿地面真相再 pivot）。
NEW_CELL6 = r'''"""Metric for NVIDIA (129716) — path-robust: locate ptxas by binary, not hardcoded dir.

Kaggle changed the utility-script mount path: was
/kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script (underscores, notebooks/owner/),
now /kaggle/usr/lib/nvidia-metric-utility-script (hyphens, flat). Glob for the ptxas
binary itself so we're agnostic to the spelling/layout, then extract that util root to /tmp.
"""
import glob, os, subprocess, sys

if RUN_EVALUATION:
    subprocess.run("uv pip uninstall torch torchvision torchaudio",
                   shell=True, check=False)

    # Prefer the metric utility's ptxas; fall back to any mounted utility's ptxas.
    _ptxas = sorted(
        glob.glob("/kaggle/usr/lib/**/triton/backends/nvidia/bin/ptxas", recursive=True),
        key=lambda p: ("metric" not in p, p))
    print("ptxas candidates:", _ptxas)
    assert _ptxas, ("no triton ptxas found under /kaggle/usr/lib — "
                    "metric utility script not mounted (check kernel_sources)")
    util_root = _ptxas[0].split("/triton/")[0]
    print("util_root:", util_root)
    # Extract util tree into /tmp so /tmp/triton/... matches cell 11's TRITON_PTXAS_PATH.
    subprocess.run(f"tar -cf - -C {util_root} . | tar -xf - -C /tmp",
                   shell=True, check=True)
    for _b in ("ptxas", "ptxas-blackwell"):
        _bp = f"/tmp/triton/backends/nvidia/bin/{_b}"
        if os.path.exists(_bp):
            os.chmod(_bp, 0o755)
            print("chmod +x", _bp)
sys.path.insert(0, "/tmp")
'''

# ── cell15 替换：Tier1 eval 集 + 精华 demo ──────────────────────────────
NEW_CELL15 = r'''import pandas as pd
import glob as _g

# ── Tier1 in-context gate ──────────────────────────────────────────────
# Eval set: depth>=2 NOVEL holdout (rules absent from train) — Layer-B style,
# ~150 problems (datagen/bit_holdout_depth2.csv).
_ev = _g.glob("/kaggle/input/**/bit_holdout_depth2.csv", recursive=True)
assert _ev, ("bit_holdout_depth2.csv not found - add "
             "`exerzitien/nemotron-bit-holdout` to dataset_sources")
df = pd.read_csv(_ev[0]).reset_index(drop=True).copy()
print(f"Tier1 eval set: {len(df)} depth>=2 novel problems")

# Few-shot demo: 1 deterministic essence CoT from the mixed training set.
_es = _g.glob("/kaggle/input/**/bit_essence_mixed.csv", recursive=True)
assert _es, ("bit_essence_mixed.csv not found - add "
             "`exerzitien/nemotron-bit-essence` to dataset_sources")
_ess = pd.read_csv(_es[0])
_ess = _ess[_ess["generated_cot"].astype(str).str.contains("composes shifted copies")]
assert len(_ess) > 0, "no essence rows matched marker — check bit_essence_mixed.csv version"
_demos = _ess.sort_values("id").head(1).to_dict("records")
def _demo_text(d):
    cot = str(d["generated_cot"]).replace("</think>\n", "").replace("</think>", "")
    return f"Example problem:\n{d['prompt']}\nWorked solution:\n{cot}"
DEMO_BLOCK = "\n\n".join(_demo_text(d) for d in _demos)
print("demos chosen:", [d["id"] for d in _demos])
'''

# ── cell16 替换：构造 zero-shot + 1-shot 两套 prompt ────────────────────
NEW_CELL16 = r'''PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
DEMO_INTRO = ("Here is a worked example showing how to decompose the rule into "
              "shifted copies of the input and gates:\n\n")
DEMO_OUTRO = "\n\nNow solve this one the same way:\n"

if RUN_EVALUATION:
    tokenizer = llm.get_tokenizer()
    def _wrap(user):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
    prompts_zero = [_wrap(p + PROMPT_SUFFIX) for p in df["prompt"]]
    prompts_demo = [_wrap(DEMO_INTRO + DEMO_BLOCK + DEMO_OUTRO + p + PROMPT_SUFFIX)
                    for p in df["prompt"]]
    print("demo tokens (first prompt):", len(tokenizer.encode(prompts_demo[0])),
          "| zero tokens (first):", len(tokenizer.encode(prompts_zero[0])))
'''

# ── cell18 替换：跑两遍（zero / 1-shot），每条件独立 max_tokens ──────────
NEW_CELL18 = r'''# Tier1: run generation twice — zero-shot vs 1-shot essence demo (V34 adapter).
lora_path = "/kaggle/working"

if RUN_EVALUATION:
    # zero-shot prompt is short -> full 7680; demo prompt carries ~1k-tok demo,
    # so cap at 6400 to stay within max_model_len (8192) and still leave boxed room.
    sp_zero = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)
    sp_demo = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=6400)
    _lora = LoRARequest("adapter", 1, lora_path)
    outputs_zero = llm.generate(prompts_zero, sampling_params=sp_zero, lora_request=_lora)
    outputs_demo = llm.generate(prompts_demo, sampling_params=sp_demo, lora_request=_lora)
'''

# ── cell20 替换：每条件计分 ────────────────────────────────────────────
NEW_CELL20 = r'''if RUN_EVALUATION:
    df["out_zero"] = [o.outputs[0].text for o in outputs_zero]
    df["out_demo"] = [o.outputs[0].text for o in outputs_demo]
    df["fin_zero"] = [o.outputs[0].finish_reason for o in outputs_zero]
    df["fin_demo"] = [o.outputs[0].finish_reason for o in outputs_demo]
    df["pred_zero"] = df["out_zero"].apply(extract_final_answer)
    df["pred_demo"] = df["out_demo"].apply(extract_final_answer)
    df["correct_zero"] = df.apply(
        lambda r: verify(str(r["answer"]), str(r["pred_zero"])), axis=1)
    df["correct_demo"] = df.apply(
        lambda r: verify(str(r["answer"]), str(r["pred_demo"])), axis=1)
'''

# ── cell21 替换：出 acc_zero / acc_demo / lift + 落盘 ───────────────────
NEW_CELL21 = r'''if RUN_EVALUATION:
    import pandas as pd
    n = len(df)
    c0 = int(df["correct_zero"].sum()); c1 = int(df["correct_demo"].sum())
    acc_zero = 100 * c0 / n; acc_demo = 100 * c1 / n; lift = acc_demo - acc_zero

    def _diag(tag, fincol, predcol, outcol):
        trunc = int((df[fincol] == "length").sum())
        nf = int((df[predcol] == "NOT_FOUND").sum())
        avg = df[outcol].astype(str).str.len().mean()
        print(f"[{tag}] truncated(length)={trunc}/{n}  NOT_FOUND={nf}/{n}  "
              f"avg_out_chars={avg:.0f}")
    _diag("zero", "fin_zero", "pred_zero", "out_zero")
    _diag("demo", "fin_demo", "pred_demo", "out_demo")

    print(f"acc_zero (V34, no demo):  {c0}/{n} = {acc_zero:.1f}%")
    print(f"acc_demo (V34, 1-shot):   {c1}/{n} = {acc_demo:.1f}%")
    print(f"LIFT: {lift:+.1f}pp")
    print("GATE: pass if lift >= +15pp (V34-instrument; lift is the signal, "
          "absolute acc is apples-to-oranges)")

    pd.DataFrame([{"acc_zero": acc_zero, "acc_demo": acc_demo,
                   "lift": lift, "n": n}]).to_csv("results.csv", index=False)
    df.to_csv("raw_outputs.csv", index=False)
    print("\n===== SAMPLE demo output [0] (first 1800 chars) =====")
    print(str(df["out_demo"].iloc[0])[:1800])
'''


def _src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def main():
    nb = json.load(open(SRC))
    # 锚点 -> 替换文本；每个必须恰好命中 1 个 code cell。
    # cell18 用顶层 `lora_path = "/kaggle/working"` 区分（cell 8 的 generate_predictions
    # 函数体内也有 outputs = llm.generate，但不含该顶层赋值），且绝不动 cell 8
    # （它定义 extract_final_answer / verify）。
    rules = [
        ("cell6", lambda s: "Metric for NVIDIA" in s and "nvidia_metric_utility_script" in s, NEW_CELL6),
        ("cell15", lambda s: "val_set_950" in s and "detect_category" in s, NEW_CELL15),
        ("cell16", lambda s: "PROMPT_SUFFIX" in s and "apply_chat_template" in s, NEW_CELL16),
        ("cell18", lambda s: 'lora_path = "/kaggle/working"' in s, NEW_CELL18),
        ("cell20", lambda s: 'df["output"]' in s and "extract_final_answer" in s, NEW_CELL20),
        ("cell21", lambda s: "groupby" in s and "results.csv" in s, NEW_CELL21),
    ]
    hits = {name: 0 for name, _, _ in rules}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = _src(c)
        for name, pred, repl in rules:
            if pred(s):
                c["source"] = repl
                hits[name] += 1
    for name in hits:
        assert hits[name] == 1, f"{name} anchor matched {hits[name]}x (expected 1)"

    json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
    print(f"Wrote {DST} (replaced " + ", ".join(f"{k}x{v}" for k, v in hits.items()) + ")")


if __name__ == "__main__":
    main()
