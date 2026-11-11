# V35 完整工作流

## 数据资产（已就绪）
- `datagen/synth_dataset_v3/bit_manipulation_v4.csv` (1602 行, 99.1% 答案正确)
- `datagen/synth_dataset_v3/kimberleyduran_cryptarithm.csv` (1140 行, V34 用过)
- Kaggle dataset `exerzitien/nemotron-v32-data`（含上面两个文件）

## V35 设计（按版本）

| 版本 | 数据组成 | 目的 | 预期 LB |
|---|---|---|---|
| **v35a** | dgxchen + km_cryptarithm | Sanity check (= V34 配方 + Adam) | 0.85 |
| **v35b** | v35a + bit_v4 swap | 主实验：验证 bit_v4 增益 | **0.86** |
| v35c | dgxchen + bit_v4 only | 隔离 bit_v4 效果 | 0.85-0.86 |

## 训练流程

### 1. 估算成本（无 Tinker 调用，免费）
```bash
cd scripts/
python3 train_v35_tinker.py --version v35b --dry-run
```

输出：约 11M tokens × $0.40/M = **$4.50 full / $2.26 (50% off)**

### 2. 实际训练
```bash
# 设置 API key (一次)
export TINKER_API_KEY=$(cat ~/.tinker_key)

# 跑 v35b（主实验）
python3 train_v35_tinker.py --version v35b

# 可选：跑完后跑其他 ablation
python3 train_v35_tinker.py --version v35a   # baseline 验证
python3 train_v35_tinker.py --version v35c   # bit_v4 隔离
```

训练完成后输出 Tinker checkpoint 路径，如 `tinker://abc-uuid`。

### 3. 转换 + 上传到 Kaggle
```bash
python3 tinker_to_kaggle.py \
    --tinker-path tinker://abc-uuid \
    --dataset-id exerzitien/nemotron-v35b-adapter \
    --version-label "V35b: bit_v4 + km swap"
```

这个脚本：
- 下载 Tinker raw checkpoint
- 应用 huikang SVD 融合（Mamba in_proj 处理）
- 转换为 PEFT 格式
- 上传到 Kaggle dataset

### 4. Kaggle 提 LB
1. 修改 `kernels/eval/kernel-metadata.json` 加 `exerzitien/nemotron-v35b-adapter` 到 dataset_sources
2. 在 Kaggle 网页选 nemotron-lora-sft 提交（USE_PRETRAINED 模式会自动加载新 adapter）

## 成本上限

| 阶段 | 成本 |
|---|---|
| Tinker training (3 versions) | ~$6 |
| Conversion (free, 本地) | $0 |
| Kaggle upload (free) | $0 |
| Kaggle LB submission (不耗 GPU 配额) | $0 |
| **总计** | **~$6** |

## 已知风险

1. **SVD 25% 能量损失**: huikang 验证过 LB 影响 ≈ 0
2. **Adam vs Muon**: V34 用 Muon @ 0.85，V35 Tinker 用 Adam @ 2e-4。
   两者公开都达 0.85（pkuszboi Muon = huikang Adam）。
3. **bit_v4 模板差异**: 测试与 dgxchen 末尾格式 100% 一致（双 </think> + I will now return）
4. **MoE expert LoRA**: Tinker 自动注入 MoE，与 Kaggle Unsloth 行为一致

## 失败 fallback

如果 V35b LB < V34 0.85：
1. 检查 Tinker → PEFT 转换是否丢失了某些 keys（safetensors inspect）
2. 跑 v35a 验证 Tinker pipeline 本身能复现 0.85（排除 SVD/Adam 问题）
3. 如果 v35a < 0.85 → Tinker pipeline 有问题，回退 Kaggle
4. 如果 v35a = 0.85 但 v35b < 0.85 → bit_v4 数据有问题，需进一步调试

## 文件清单

| 文件 | 用途 |
|---|---|
| `train_v35_tinker.py` | Tinker 训练入口（3 个版本可选） |
| `tinker_to_kaggle.py` | Tinker → PEFT → Kaggle 上传 |
| `eval_v33_on_5090.py` | 本地 5090 V33 评测脚本 |
| `PROMPT_FOR_5090_AGENT.md` | 给 5090 WSL Claude Code 的 setup prompt |
| `../datagen/bit_gen_v4.py` | bit_v4 数据生成器（一次性运行） |
