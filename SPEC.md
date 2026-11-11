# SPEC.md — NVIDIA Nemotron Model Reasoning Challenge

**URL**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
**Status**: 🔴 Needs Kaggle login to confirm deadline / prize / leaderboard scores

---

## 1. Competition Overview

| Field | Detail |
|-------|--------|
| Host | NVIDIA × Kaggle |
| Task | Fine-tune a LoRA adapter for Nemotron-Nano to solve logical reasoning puzzles |
| Model | `Nemotron-Nano-30B` (MoE ~30B params, BF16) — also `nvidia/OpenMath-Nemotron-14B-Kaggle` |
| Evaluation | **Accuracy** — answer must appear in `\boxed{}`, exact match or ±1e-2 tolerance |
| Submission type | **Model submission** — LoRA adapter weights (`.zip`) |
| Constraint | LoRA rank ≤ 32 |
| Deadline | **2026-06-15 23:59 UTC** (33 days left) |
| Prize | **$106,388 USD** total · 2983 teams competing |
| Our submissions | 0 (not yet submitted) |

---

## 2. Task Definition

### Input
- `train.csv`: columns `id`, `prompt`, `answer`
- `test.csv`: columns `id`, `prompt`

### Problem types
- Bit manipulation
- Algebraic equations
- Logical reasoning puzzles
- (Possibly more — confirm from data tab)

### Output
Model must generate answer wrapped in `\boxed{<answer>}`.

### Submission format
```
submission.zip
└── adapter/          ← LoRA weights (rank ≤ 32)
    ├── adapter_config.json
    └── adapter_model.safetensors (or .bin)
```

---

## 3. Top Approaches (from Public Solutions)

### 🥇 Progress Prize Winner — tonghuikang
**Source**: https://github.com/tonghuikang/nemotron

| Step | Detail |
|------|--------|
| 1. Problem analysis | Categorize each training problem by type; track per-problem loss |
| 2. Synthetic data gen | `reasoning.py` — extract patterns; `augmentation.py` — generate more examples |
| 3. Corpus building | `corpus.py` — JSONL with token-level masking strategies |
| 4. SFT with LoRA | `train_sft.py` — variable LR, epochs, batch size |
| 5. Evaluation | Per-problem logprob tracking; min-logprob monitoring by category |
| 6. Upload | Modal for adapter uploading |

**Key insight**: Per-problem loss analysis → categorize as "rules found / hypothesis / unknown" → focus augmentation on hard categories.

---

## 4. Strategy & Implementation Plan

### Phase 1 — Baseline (Day 1)
- [ ] Download data via `kaggle competitions download nvidia-nemotron-model-reasoning-challenge`
- [ ] Run base model (no fine-tuning) on test set → get baseline accuracy
- [ ] Understand problem distribution (types, difficulty)

### Phase 2 — Core Training (Day 2–4)
- [ ] Set up LoRA SFT pipeline (HuggingFace PEFT or NeMo)
- [ ] Train on `train.csv` with LoRA rank=16 first
- [ ] Track per-problem loss → identify hard categories
- [ ] Synthetic data generation for hard categories

### Phase 3 — Iteration (Day 5–7)
- [ ] Augment corpus with synthetic examples
- [ ] Sweep: LoRA rank (8/16/32), LR, epochs
- [ ] Ensemble / rank averaging if multiple checkpoints
- [ ] Package and submit

---

## 5. Technical Constraints

| Constraint | Value |
|-----------|-------|
| LoRA rank | ≤ 32 |
| Base model | Nemotron-Nano-30B (BF16) |
| Kernel mode | Commit mode required for test set access |
| Internet in kernel | Check competition rules |
| GPU | Kaggle T4×2 or A100 (check allocation) |

---

## 6. Environment Setup

```bash
# Data download
kaggle competitions download nvidia-nemotron-model-reasoning-challenge -p ./data/

# Pull top notebook with metadata
kaggle kernels pull ryanholbrook/nvidia-nemotron-submission-demo -p ./kernels/ -m

# Check submission status
kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge
```

```python
# Submission packaging (only adapter weights, NOT notebook)
import zipfile, os
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir('adapter/'):
        zf.write(f'adapter/{f}', f'adapter/{f}')
```

---

## 7. Score Stabilization Reminder

> ⚠️ Wait **4+ hours** after submission before making optimization decisions.
> Early scores (< 2h) are often inflated.

---

## 8. Success Metrics

| Milestone | Target | Timeline |
|-----------|--------|----------|
| Data download + EDA | — | Day 1 |
| Base model accuracy | ~0.60–0.70 (est.) | Day 1 |
| After SFT LoRA rank=16 | 0.80+ | Day 3 |
| After augmentation | 0.84+ | Day 5 |
| Final submission | **0.87 (top 1%)** | Day 7 |

**Leaderboard snapshot (2026-05-13)**:
- #1: 0.87 (MOONMOON, Researcher 7919, Lora is all you need)
- #4–17: 0.86 (dense competition)
- Top score is 0.87 — achievable with SFT + augmentation

---

## 9. Status

### Competition Facts (confirmed)
- Deadline: 2026-06-15 (33 days left)
- Prize: $106,388 USD total
- 2983 teams
- Top score: **0.87** (multiple teams)
- Data: 9500 train / 3 public test (hidden test evaluated server-side)

### Problem Categories (6, ~1600 each)
1. Bit manipulation (8-bit, XOR/AND/OR/NOT/shifts)
2. Gravitational constant (physics calc with secret G)
3. Unit conversion (secret conversion rules)
4. Cipher/encryption (text decode)
5. Numeral system (secret base conversion)
6. Equation transformation (secret algebraic rules)

### 提交记录
- v13-v20：环境调试期，逐步修复各类问题（详见 RETROSPECTIVE.md）
- v21（2026-05-14）：✅ 训练成功，❌ callback bug 导致 adapter 未保存
  - CoT 数据（dgxchen/nemotron-cot-tong，7830条）
  - packing=False，max_length=2048，60步，loss 24→3.2
- v22：⏳ 等待 GPU 配额重置（周一），仅修复 callback bug

### 下周首要任务
1. 周一：`kaggle kernels push -p .`（v22）
2. v22 完成后立即提交，获取第一个基线分
3. v23：去掉 MAX_STEPS，跑完整 1 epoch（~14.5h）
