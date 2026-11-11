# V32 完全体 — 最终设计（一次到位，猛冲 0.87+）

**生成时间**: 2026-05-27
**目标**: 0.83 → **0.87-0.90** (中位 0.88)
**实施周期**: 4-6h 代码 + 9h GPU + 2h LB 等待 = 16h 端到端

---

## 一行核心方程

```
V32 = dgxchen baseline
    + cryptarithm v3 swap (lkevincc 97.2%)
    + bit MAJ0RT0M v3 swap (98.88%, Donald 紧凑模板)
    + huikang 8463 augmentations (sub-skill curriculum)
    + Muon 优化器 (pkuszboi 公开 +0.01 LB)
    + Branch-weighted CE loss (huikang Progress Prize)
    + lm_head 手动注入 (v30 已有)
```

---

## 杠杆来源全表 + 单独验证

| 杠杆 | 来源 | 单独验证 LB | 我们的实现路径 | 风险 |
|---|---|---|---|---|
| dgxchen base | v23 verbatim | **0.84** | ✅ 已用 | 0 |
| cryptarithm v3 swap | lkevincc solver (97.2%) + huikang format | +0.02~0.04 推断 (Bobber +0.01 over kimberleyduran 55 道) | ✅ `datagen/synth_dataset_v3/cryptarithm_v3.csv` 656 行 (100% 自洽 post-May metric) | 低 |
| **bit MAJ0RT0M v3** | MAJ0RT0M solver (98.88%) + Donald 模板 | +0.005~0.015 推断 (bit 当前 90% 提到 98%) | 🟡 需写 `datagen/bit_gen_v3.py` (~4h, 草稿已验证 1000 chars/题答对) | 中 |
| **huikang 8463 augmentations** | huikang_ref/augmentations/* (本地) | huikang base 0.85 核心 (curriculum learning) | ✅ `datagen/synth_dataset_v3/huikang_augmentations.csv` 8463 行 | 中 (子技能与主任务格式不同,需特殊处理) |
| Muon 优化器 | pkuszboi 公开 notebook | **+0.01** (单独 isolated) | ✅ 类代码直接复制 (80 行 from pkuszboi cell 11) | 中 (要确保 lm_head 1D bias 等 LoRA params 不进 Muon) |
| **Branch-weighted CE loss** | huikang Progress Prize 主写作 | huikang 自述 "the technique" | 🟡 在线近似 (sketched, compatible with Muon via different SFTTrainer hook) | 中 (online 估计与 huikang offline 数学不等价,可能更弱或更强) |
| lm_head 注入 | v30 baseline | (已有) | ✅ kernel cell 9 已有 | 0 |

---

## 数据层完整组合

```
基础: dgxchen 7830 行
  - DROP: cryptarithm_deduce (627) + cryptarithm_guess (154) = -781
  - DROP: bit_manipulation (1754) = -1754  ← V32 新增 drop
  + ADD: cryptarithm_v3.csv (656 行)
  + ADD: bit_manipulation_v3.csv (~1584 行)  ← 待生成
  + ADD: huikang_augmentations.csv (8463 行)
= 14148 训练行

类别覆盖率检查:
  - bit: 1584 (MAJ0RT0M correct) + 18 (MAJ0RT0M 解但答错, 走 dgxchen fallback?)
    实际方案: 只取 MAJ0RT0M correct=True 的 1584 道，剩 18 + 0 unsolved 走 dgxchen 兜底
  - cryptarithm: 656 (v3 correct) + 167 (lkevincc 未解或 v3 gen 失败) → 留 dgxchen
  - 实际净增减: 看一下面整合策略
```

### 数据整合策略（避免冲突）

```python
SUB_SKILL_CATS = {'matching', 'splitting', 'concatenation', 'spelling', 'lstrip'}
SYNTH_DROP_TYPES = ['cryptarithm_deduce', 'cryptarithm_guess', 'bit_manipulation']
SYNTH_ADD_GLOBS = ['cryptarithm_v3.csv', 'bit_manipulation_v3.csv', 'huikang_augmentations.csv']

# cryptarithm_v3 不覆盖的 167 道 + bit MAJ0RT0M 不覆盖的 18 道
# → 这些 IDs 会从 dgxchen 完全消失（既 drop 了又没补回）
# 解决: bit_v3 + cryptarithm_v3 各加一个 "fallback to dgxchen" 列
#       或者训练数据简单点,接受这 185 道丢失 (1.2% of train)
```

**关键提醒**: 8463 augmentations 没有 `\boxed{}` 包装，训练 kernel cell 11 需要分类型处理：

```python
for _, row in train_df.iterrows():
    cot = str(row["generated_cot"])
    if row["type"] in SUB_SKILL_CATS:
        # Sub-skill: 完成即答案, 不加 \boxed wrap
        assistant_content = cot.rstrip()
    else:
        # 主任务: 加 </think>\boxed{answer}
        cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
        assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{row['answer']}}}"
```

---

## 训练层完整组合

```python
# ===== kernel cell 11 完整重写 =====
import torch, torch.nn as nn
from trl import SFTTrainer, SFTConfig
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict
import random, math, re

# 1) Muon optimizer (pkuszboi 原版)
@torch.compile()
def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                if p.ndim < 2: continue  # 安全护栏: skip 1D params
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=group['momentum'])
                g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-group['lr'])
        return loss

# 2) Stratified batching (pkuszboi 原版)
class PrecomputedOrderSampler(Sampler):
    def __init__(self, order): self.order = list(order)
    def __iter__(self): return iter(self.order)
    def __len__(self): return len(self.order)

def build_stratified_index_order(labels, batch_size, seed):
    by_label = defaultdict(list)
    for i, l in enumerate(labels): by_label[l].append(i)
    rng = random.Random(seed)
    for L in by_label.values(): rng.shuffle(L)
    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches = [[] for _ in range(n_batches)]
    bo = list(range(n_batches)); rng.shuffle(bo)
    a = 0
    for l in sorted(by_label.keys()):
        for i in by_label[l]:
            batches[bo[a % n_batches]].append(i); a += 1
    return [i for b in batches for i in b]

# 3) V32 trainer = Stratified + Muon + Branch-weighted CE
class V32Trainer(SFTTrainer):
    branch_logprob = 0.01
    
    def __init__(self, *args, stratified_order=None, **kw):
        super().__init__(*args, **kw)
        self.stratified_order = stratified_order
    
    def get_train_dataloader(self):
        # Stratified order (pkuszboi)
        if self.stratified_order is None:
            return super().get_train_dataloader()
        dl_kw = {
            'batch_size': self.args.per_device_train_batch_size,
            'sampler': PrecomputedOrderSampler(self.stratified_order),
            'collate_fn': self.data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory,
            'drop_last': self.args.dataloader_drop_last,
        }
        return DataLoader(self.train_dataset, **dl_kw)
    
    def create_optimizer(self):
        # Muon optimizer (pkuszboi)
        if self.optimizer is None:
            matrix_params = [p for n, p in self.model.named_parameters() 
                            if 'lora' in n and p.requires_grad and p.ndim >= 2]
            self.optimizer = Muon(matrix_params, lr=2e-3, momentum=0.95, backend_steps=5)
        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Branch-weighted CE (huikang online approx)
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        # weight = min(1, |logprob| / branch_logprob)
        weight = (ce.detach() / self.branch_logprob).clamp(max=1.0)
        valid = (shift_labels.view(-1) != -100).float()
        weight = weight * valid
        loss = (ce * weight).sum() / weight.sum().clamp(min=1.0)
        return (loss, outputs) if return_outputs else loss

# 4) SFTConfig (huikang/pkuszboi 共同验证)
training_args = SFTConfig(
    output_dir="/kaggle/working/sft_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch 16
    lr_scheduler_type="linear",  # 注: Muon 已有 LR decay 但 SFTTrainer 也会做调度,注意一致性
    warmup_steps=0,
    max_length=8192,
    max_grad_norm=1e9,  # no clip
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing=False,
    save_strategy="no",
    report_to="none",
    logging_steps=10,
)

# 5) Train
effective_batch_size = 16
stratified_order = build_stratified_index_order(record_types, effective_batch_size, SEED)
trainer = V32Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    stratified_order=stratified_order,
)
trainer.train()
```

---

## bit_gen_v3 设计草稿（已验证可行）

```python
# datagen/bit_gen_v3.py
import pandas as pd, re, sys
sys.path.insert(0, '/Users/exerzitien/Desktop/nemotron-kaggle/research')
from majortom_bit_solver import (
    solve_dfs_trace_dynamic, parse_bit_problem
)

# 对每道 MAJ0RT0M correct 的 bit 题：
# 1. parse 输入示例 + query
# 2. 用 MAJ0RT0M solver 拿 predictor + expr + trans_dict
# 3. 生成 Donald-style CoT (实测 ~1000 chars):
#    S1: 类型识别
#    S2: COLUMNS (input/output 按位列出)
#    S3: SOLVE - 用 expr 和 trans_dict 说明规则
#    S4: APPLY 每位计算
#    S5: ANS + \boxed{...}
# 4. 输出 cryptarithm_v3.csv 同格式: id, prompt, answer, type, generated_cot

# 已验证: 1 道样本 CoT 999 chars, 答案正确
# 全 1602 道预计 ~6 分钟 (solver) + 几秒 (gen)
```

---

## 预期 LB 数学（诚实区间）

| 累加层 | 增益区间 | 累计中位 LB |
|---|---|---|
| v23 baseline | - | 0.84 |
| + cryptarithm v3 (656 替换 781) | +0.02 ~ +0.04 | 0.86 |
| + bit MAJ0RT0M v3 (1584 替换 1754) | +0.005 ~ +0.015 | 0.87 |
| + 8463 augmentations | +0.005 ~ +0.015 | 0.875 |
| + Muon 优化器 | +0.005 ~ +0.015 | 0.88 |
| + Branch-weight loss | -0.01 ~ +0.015 | 0.88 (online 近似不确定) |

**最终预期**:
- **中位**: 0.87-0.88
- **上限**: 0.90 (突破 0.87 frozen 高原)
- **下限**: 0.82 (多变量风险叠加可能踩坑)

注: 多变量同时改导致风险被放大。如果训练出现明显异常（loss 爆炸 / val 跌穿），立即回退测试单变量。

---

## 风险评估 + 缓解清单

### R1: Muon 与 lm_head 大词表兼容性
- **风险**: lm_head LoRA 矩阵很大（vocab × rank），Muon NewtonSchulz 可能 OOM
- **缓解**: Muon class 加 `p.ndim < 2: continue` (1D 安全护栏 + LoRA params 应全是 2D，已加)
- **验证**: 训练首步监控 VRAM, 若 OOM 立即排除 lm_head from Muon

### R2: Muon LR 2e-3 + linear scheduler 双调度冲突
- **风险**: SFTTrainer linear scheduler 也调整 LR，与 Muon 内部状态不一致
- **缓解**: pkuszboi/Bobber 都用相同配置且 0.85-0.86 OK，无需额外处理

### R3: Branch-weight loss online 估计不等价于 huikang offline
- **风险**: online 估计可能不收敛或方向相反
- **缓解**: 训练首 50 步监控 loss 曲线，若 loss 不降立即切回标准 CE

### R4: 8463 augmentations 不带 \boxed 与训练 collator 冲突
- **风险**: SFTTrainer 期望统一格式
- **缓解**: cell 11 加 SUB_SKILL_CATS 分支 (已设计)

### R5: 多变量同时改导致问题归因困难
- **风险**: 若 LB 失败,不知道哪个变量出问题
- **缓解**: 用户选择"激进一步到位", 接受归因风险换取 8h GPU 一次拿到结果

### R6: 训练时间超 12h kernel 限制
- **估算**: 14148 行 × ~3500 tokens/行 / 16 batch ≈ 884 steps
  - pkuszboi 0.85 报告 5h on Kaggle
  - 我们 +75% 数据 → 估 7-9h
- **缓解**: 若超时,首先 drop augmentations (8463 行) 减负

### R7: cryptarithm/bit 167+18=185 道空洞 (drop 了 dgxchen 但没补)
- **缓解**: 写一个简单 fallback: 这些 ID 保留 dgxchen 原 CoT, 否则训练数据真空

---

## 实施工作量（从现状开始）

| 任务 | 时间 | 状态 |
|---|---|---|
| 1. 写 datagen/bit_gen_v3.py | 3-4h | 草稿已验证 |
| 2. 跑 bit_gen_v3 生成 ~1584 行 | 10min | - |
| 3. 修 cryptarithm/bit drop 留 fallback 逻辑 | 30min | - |
| 4. 修改 kernel cell 2 (SYNTH_DROP/ADD) | 5min | - |
| 5. 修改 cell 11 加 V32Trainer + Muon + branch-loss + sub-skill 分支 | 1h | 代码已写完整 |
| 6. 上传 3 个 v3 数据集到 Kaggle | 30min | - |
| 7. 更新 kernel-metadata dataset_sources | 5min | - |
| 8. 本地 dry-run (10 道样本端到端验证) | 30min | - |
| 9. push kernel → 训练 | 7-9h GPU | - |
| 10. eval val_950 | 1h GPU | - |
| 11. LB 提交 (1 daily slot) | 2h 等结果 | - |

**总: 5-6h 代码 + 9h GPU + 2h LB = ~17h 端到端**

---

## V33+ 留作下一轮 (V32 之后)

- v33: GRPO with verify() reward (Komil 修复后可行)
- v34: 进一步压 token (Donald template 全 9 类应用)
- v35: 多 epoch + offline branch logprob (huikang 真正的 winning loss)
- v36: 模型 ensemble / TTA (推理硬约束 temp=0 单次,可能没空间)

---

## 决定点

**当前 ready 资产** (再核实):
- ✅ dgxchen baseline CSV
- ✅ cryptarithm_v3.csv (656 行)
- ✅ huikang_augmentations.csv (8463 行)
- 🟡 bit_manipulation_v3.csv (待生成,代码草稿可用)
- ✅ Muon + V32Trainer 完整代码
- ✅ Branch-weight loss 代码

**全部就绪需要做的最后两件事**:
1. 写完 datagen/bit_gen_v3.py 跑出 csv (3-4h)
2. 整合所有改动到 train kernel cell 11 (1h)

然后就可以 push GPU。
