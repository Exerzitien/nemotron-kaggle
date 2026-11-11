# NVIDIA Nemotron Model Reasoning Challenge（Kaggle，2026）

个人参赛项目。在 LoRA 秩 ≤ 32 的限制下，为 Nemotron-3-Nano-30B-A3B（MoE）训练适配器，解决 9 类逻辑推理题。

- 最终排名：1,304 / 4,185（前 32%）
- 公开榜最高：0.85（V34，提交 #53080019），当时榜首 0.87
- 时间：2026 年 3 月至 6 月，主要工作集中在 5 月 12 日至 6 月 2 日

## 先读这几份

| 文件 | 内容 |
| --- | --- |
| `RETROSPECTIVE.md` | v1–v27 每一版的改动、失败原因、环境踩坑、留出集评测、自造数据、Tinker |
| `research/V36_SESSION_LOG_2026-05-29.md` | 全部版本的公开榜分数、V34 分类基线、密码算术不可学的量化证明 |
| `research/V37_MUON_TIE_POSTMORTEM_2026-05-31.md` | Muon 优化器 × 绑定 MoE 专家导致 0.44 的复盘 |
| `docs/superpowers/specs/` | 对记忆免疫的 bit 诊断集、bit 串联 CoT 的设计文档（判据在出分前冻结） |
| `SPEC.md` | 比赛规则与约束 |

## 目录

| 目录 | 内容 |
| --- | --- |
| `kernels/` | Kaggle notebook：`train/`（主训练）、`eval*/`（各版评测）、`diagnostic/` |
| `datagen/` | 程序化出题器、bit 诊断集与串联 CoT 生成器，以及小体积的生成数据 |
| `research/` | 方案文档与复盘；`majortom_bit_solver.py` 取自 Kaggle 论坛（作者 MAJ0RT0M），供 `datagen/` 调用 |
| `eval_results/` | run1 与 V38 的分类汇总，bit 诊断集、Tier1 闸门的结果与错例 |
| `scripts/`、`tinker/` | Tinker 云端训练与 adapter 格式转换、RTX 5090 离线评测脚本 |
| `logs/` | v1–v21 训练日志 |
| `tests/` | 诊断集与串联 CoT 库的单元测试 |

## 代码来源

- `kernels/train/nemotron-lora-sft.ipynb` 移植自 dgxchen 的公开 notebook `dgxchen/training-with-unsloth-to-achieve-0-85-lb`，开头的致谢段落（Tong Hui Kang、konbu17）沿用原文；此后的版本迭代在它的基础上修改。
- 主训练 notebook 中的 Muon 优化器类代码约 80 行取自 pkuszboi 的公开 notebook（见 `research/V32_FINAL.md`）。
- `kernels/eval*/` 的评测 notebook 基于他人公开的验证 notebook，开头注明其验证思路取自 Kh0a 的 notebook。
- `research/majortom_bit_solver.py` 取自 Kaggle 论坛，作者 MAJ0RT0M。

## 没有放进仓库的内容

- **比赛原始数据**（`data/` 与由它整理出的 `train_with_type.csv`）：从 Kaggle 比赛页下载。
- **含比赛原题或标准答案的文件**：在原题上生成 CoT 的 `datagen/synth_dataset_v3/` 与 `kaggle_dataset_v32/` 数据、run1 的逐题评测结果与错例、`research/majortom_results.csv`（bit 题求解结果，`datagen/bit_holdout_gen.py` 读取它）。
- **基本照搬他人 notebook 的版本**：V36（基于 huikang 的 0.85 notebook）、V37（amr V4 流程，只换优化器）、V38（逐字照搬 amr V4）。它们的分数和复盘见 `research/V37_MUON_TIE_POSTMORTEM_2026-05-31.md`。
- **模型权重与提交包**：训练出的 adapter（约 3.5GB）、V36 与 Tinker 的 `submission.zip`。
- **他人的代码与数据**：huikang、lkevincc、Bobber、amr、konbu17、kimberleyduran 等公开仓库、notebook 与数据集的本地副本，以及 dgxchen CoT 数据集。
- **大于 5MB 的生成数据**：可以用 `datagen/` 里的脚本重新生成。
- 日志中出现过的访问令牌已替换为 `KGAT_<REDACTED>`。
