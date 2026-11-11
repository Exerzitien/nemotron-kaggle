"""从 eval-v39 notebook 派生留出集 eval：
- cell 15: 数据源从 val_950 换成我们的 A.csv + B.csv，df 带 'layer' 列。
- cell 21: groupby('category') 改为 groupby('layer')。
其余（vLLM 参数、verify、prompt 构造、adapter glob）全部不动，保持与打分器一致。
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, "..", "eval-v39", "nemotron-eval-v39.ipynb")
DST = os.path.join(_HERE, "nemotron-eval-bit-holdout.ipynb")

# 新 cell 15：加载两层留出集（Kaggle dataset 解包后的路径），合并 + 加 layer 列
NEW_CELL15 = r'''import pandas as pd
import glob as _g

def detect_category(prompt: str) -> str:
    if "secret bit manipulation rule transforms 8-bit binary numbers" in prompt:
        return "bit_manipulation"
    return "unknown"

# EVAL SET: exerzitien/nemotron-bit-holdout (two layers A/B)
_a = _g.glob("/kaggle/input/**/bit_holdout_A.csv", recursive=True)
_b = _g.glob("/kaggle/input/**/bit_holdout_B.csv", recursive=True)
assert _a and _b, ("bit_holdout_A/B.csv not found - add "
                   "`exerzitien/nemotron-bit-holdout` to dataset_sources")
_da = pd.read_csv(_a[0]); _db = pd.read_csv(_b[0])
df = pd.concat([_da, _db], ignore_index=True)
df["category"] = df["prompt"].apply(detect_category)
assert (df["category"] == "bit_manipulation").all(), "non-bit rows leaked in"
print(f"holdout loaded: A={len(_da)} B={len(_db)} total={len(df)}")
print(df["layer"].value_counts().sort_index())
df = df.reset_index(drop=True).copy()
'''

# 新 cell 21：按 layer 分组出分（同时保留 mistakes 落盘）
NEW_CELL21 = r'''if RUN_EVALUATION:
    import os
    os.makedirs("mistakes", exist_ok=True)
    for layer in df["layer"].unique():
        m = df[(df["layer"] == layer) & (~df["correct"])]
        if not m.empty:
            m.to_csv(f"mistakes/layer_{layer}.csv", index=False)
    stats = df.groupby("layer")["correct"].agg(correct="sum", total="count").sort_index()
    stats["correct"] = stats["correct"].astype("int")
    gt = stats["total"].sum()
    stats["percentage"] = (stats["correct"] / stats["total"] * 100).map("{:.1f}%".format)
    overall = stats["correct"].sum() / gt * 100
    print(stats.to_string())
    print(f"OVERALL: {stats['correct'].sum()}/{gt} = {overall:.1f}%")
    stats.to_csv("results.csv")
    df.to_csv("validation.csv", index=False)
'''

def main():
    nb = json.load(open(SRC))
    cells = nb["cells"]
    # 定位并替换：用稳定锚点而非硬下标，防 eval-v39 cell 漂移
    replaced15 = replaced21 = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "val_set_950" in src and "detect_category" in src:
            c["source"] = NEW_CELL15
            replaced15 += 1
        elif "groupby(\"category\")" in src and "results.csv" in src:
            c["source"] = NEW_CELL21
            replaced21 += 1
    assert replaced15 == 1, f"cell15 anchor matched {replaced15}x (expected 1)"
    assert replaced21 == 1, f"cell21 anchor matched {replaced21}x (expected 1)"
    json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
    print(f"Wrote {DST} (replaced cell15 x{replaced15}, cell21 x{replaced21})")


if __name__ == "__main__":
    main()
