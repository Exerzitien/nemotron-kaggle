# Nemotron 竞赛第一周回顾（2026-05-14）

## 一、背景

**竞赛**：NVIDIA Nemotron Model Reasoning Challenge  
**任务**：对 Nemotron-Nano-30B（MoE，3B active params）微调 LoRA 适配器（rank≤32），解决 6 类推理谜题  
**评分**：答案必须在 `\boxed{}` 内，精确匹配或 ±1e-2 容差  
**截止**：2026-06-15  
**当前榜首**：0.87（多支队伍），目标 0.85+

---

## 二、本周完整版本历史

### 2.0 早期阶段（v1-v12）：环境与依赖地狱

| 版本 | 核心改动 | 结果 | 失败原因 |
|------|----------|------|----------|
| v1 | 初始提交（无 log 保存） | ❌ | 未知，大概是基础环境问题 |
| v2 | 初始脚本 | ❌ | `FileNotFoundError`：数据路径硬编码为 `/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv`（竞赛数据路径是 `/kaggle/input/competitions/...`） |
| v3 | 修路径（还是错的） | ❌ | 同上，路径改成了 `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/`（还是错误路径）|
| v4 | 加自动搜索 train.csv | ❌ | 数据路径正确了，但加载模型时 `ModelNotFoundError: mamba_ssm.modules.mamba3` 不存在（首次碰到 mamba3 问题） |
| v5 | 尝试 pip install 修复依赖 | ❌ | pip install 破坏了 CUDA 环境（CUDA 变成 False），数据路径又改错了 |
| v6 | 恢复路径，继续 pip 安装方式 | ❌ | CUDA 仍然 False；mamba3 依赖链：`mamba3.py → cutlass`，`cutlass` 不存在 |
| v7 | 恢复 CUDA，trl 安装修复 | ❌ | CUDA 恢复 True，trl OK，但 mamba_ssm import 仍触发系统版 mamba3 → cutlass 链 |
| v8 | 代码调整 | ❌ | `SyntaxError`（notebook 里有语法错误，未运行到模型加载） |
| v9 | 从 nemotron-packages wheel 安装 mamba_ssm | ❌ | pip 安装成功，但 Python 仍然 import 的是 `/kaggle/usr/` 里的系统版（sys.path 优先级问题） |
| v10 | 调整 wheel 安装方式 | ❌ | `ModuleNotFoundError: No module named 'mamba_ssm'`（wheel 安装后完全不可见） |
| v11 | 同 v10，进一步调试 | ❌ | 同上 |
| v12 | 详细诊断：pip show 显示已安装，但 import 失败 | ❌ | `sys.path` 问题：新装的包路径优先级低于系统路径，被系统 mamba_ssm 遮蔽；且系统版 mamba_ssm 触发 mamba3 |

**v1-v12 阶段教训**：
- Kaggle 竞赛数据路径是 `/kaggle/input/competitions/<slug>/`，不是 `/kaggle/input/<slug>/`
- 系统 pip install 会破坏 Kaggle 内置 GPU 驱动绑定（CUDA 失效）
- `/kaggle/usr/` 的包在 sys.path 里排在用户安装路径**前面**，新装的包无法覆盖系统包

### 2.1 突破阶段（v13-v22）

| 版本 | 核心改动 | 结果 | 失败原因 |
|------|----------|------|----------|
| v13 | 首次尝试修复 mamba_ssm | ❌ | `OSError: [Errno 30]` 试图写只读文件系统 `/kaggle/usr/` |
| v14 | pip install --ignore-installed | ❌ | pyarrow 二进制不兼容，`pip install` 破坏依赖 |
| v15 | trl `--no-deps`，packing=True | ✅ 训练启动 | 但无 packing 时需 9.3h，改 packing=True 后快了 |
| v17 | 加 flash_attention_2 | ❌ | NemotronH 不支持 flash_attention_2 |
| v18 | 移除 flash_attention_2，attn_implementation='eager' | ✅ 启动 | 但 packing=True 无 flash attn → 交叉污染 |
| v19 | 维持配置，观察 loss | ❌ | `save_steps=50` 保存完整优化器状态耗尽磁盘 |
| v20 | `save_strategy='no'` | 🟡 跑了 290 步 | loss 在 7-8 平台，用户手动停止（交叉污染问题） |
| v21 | CoT 数据 + packing=False + 全面重构 | 🟡 训练完成 | callback bug 导致 `_finalize_training` 崩溃，adapter 未保存 |
| v22 | 修复 callback（`and 'loss' in logs`） | ⏳ 等配额 | GPU 配额本周耗尽，下周一推送 |

---

## 三、环境踩坑详解

### 3.0 Kaggle 竞赛数据路径（v2-v3）
**错误路径**：`/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv` 或 `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv`  
**正确路径**：`/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv`  
竞赛数据（通过比赛页面加入后自动挂载）路径前缀是 `/kaggle/input/competitions/`，而不是普通 dataset 的 `/kaggle/input/`。

**健壮写法**（v4 解决方案）：
```python
import glob
paths = glob.glob('/kaggle/input/**/train.csv', recursive=True)
DATA_DIR = os.path.dirname(paths[0])
```

### 3.1 mamba_ssm 导入失败
**根本原因**：mamba_ssm 新版本尝试导入 `mamba_ssm.modules.mamba3`，但该模块不存在  
**错误尝试**：修改 `/kaggle/usr/` 路径下的文件 → `OSError: [Errno 30] Read-only file system`  
**错误尝试 2（v9-v12）**：安装 mamba_ssm wheel（`pip install mamba_ssm-xxx.whl`）— pip show 显示安装成功，但 import 时仍然加载的是 `/kaggle/usr/` 路径下的系统版，因为系统路径在 `sys.path` 里优先级更高。新安装的包完全被遮蔽。  
**正确方案**：在 import mamba_ssm **之前**注入假模块：
```python
import sys, types
fake_mamba3 = types.ModuleType('mamba_ssm.modules.mamba3')
fake_mamba3.Mamba3 = type('Mamba3', (), {})
sys.modules['mamba_ssm.modules.mamba3'] = fake_mamba3
import mamba_ssm  # 现在可以正常导入
```

### 3.2 trl 安装方式
**错误方案**：`pip install --ignore-installed` → 替换系统 pyarrow，二进制不兼容  
**正确方案**：`pip install -q --no-index --find-links {packages_dir} trl --no-deps`  
- `--no-index`：只从本地包安装  
- `--find-links`：指定 dennisfong 的离线包目录  
- `--no-deps`：不安装任何依赖（关键！防止覆盖系统包）

### 3.3 flash_attention_2 不可用
NemotronH (Mamba+Attention 混合架构) 不支持 `attn_implementation='flash_attention_2'`  
必须用 `attn_implementation='eager'`

### 3.4 packing=True 导致 loss 平台
**现象**：packing=True 时 loss 卡在 7-8，不下降  
**原因**：packing 把多个样本拼接成一个序列，没有 flash_attention_2 的 attention mask 支持时，attention 会跨样本边界计算（cross-contamination）  
**解决**：`packing=False`（v21 loss 从 24→3.2，证明有效）

### 3.5 磁盘空间耗尽
`save_steps=50` 默认保存完整检查点（模型 + 优化器状态 ≈ 7GB/个）  
必须设置 `save_strategy='no'`

### 3.6 PrintProgress callback 崩溃
训练结束后 `_finalize_training` 会再次调用 `on_log`，此时 logs 里是 `train_loss`、`train_runtime` 等，没有 `loss` 键。  
`logs.get('loss', '?'):.4f` → `'?':.4f` → `ValueError`  
**修复**：
```python
def on_log(self, args, state, control, logs=None, **kwargs):
    if logs and state.global_step and 'loss' in logs:  # 加 'loss' in logs
        print(f"[step {state.global_step}/{state.max_steps}] loss={logs['loss']:.4f} ...")
```

---

## 四、训练配置最终方案（v22）

```python
# LoRA
LORA_RANK = 32
LORA_ALPHA = 32  # 必须 = rank，不要用 16
target_modules = ['q_proj','k_proj','v_proj','o_proj',
                  'in_proj','out_proj','up_proj','down_proj']  # 不包含 lm_head
lora_dropout = 0.0

# 数据
CoT dataset: dgxchen/nemotron-cot-tong（problem_ids_matched.csv，7830 条）
CoT 格式: '<think>\n{cot_cleaned}\n</think>\n\\boxed{answer}'
# cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', generated_cot).rstrip()

# 训练
MAX_SEQ_LEN = 2048
packing = False          # 绝对不能 True！
NUM_EPOCHS = 1
GRAD_ACCUM = 8
MAX_STEPS = 60           # 时间约束（30min 配额）
LR = 2e-4
lr_scheduler_type = 'linear'
adam_beta2 = 0.95        # dgxchen 配置
warmup_steps = 0
gradient_checkpointing_kwargs = {'use_reentrant': False}
save_strategy = 'no'
```

---

## 五、性能数据

| 阶段 | 时间 | 备注 |
|------|------|------|
| 模型加载（kagglehub + AutoModelForCausalLM） | ~580s ≈ 10min | 每次内核启动固定开销 |
| 每个有效步（grad_accum=8，packing=False，max_length=2048） | ~53s | v21 实测 |
| 60步训练 | ~3193s ≈ 53min | v21 实测 |
| 全周配额 | 30h | 重置日：每周一 |

**含义**：完整 1 epoch = 7830÷8 ≈ 979 步 × 53s ≈ 14.5h（不是早先误算的 297 步 / 4.4h）。
**这超过 Kaggle notebook kernel 的 12h 硬上限** —— 单个 kernel 跑不完一个 epoch。详见第十节。

---

## 六、v21 训练质量

```
[step  5/60] loss=24.0912  lr=1.87e-04  epoch=0.01
[step 10/60] loss=14.6919  lr=1.70e-04  epoch=0.01
[step 15/60] loss=13.0621  lr=1.53e-04  epoch=0.02
[step 20/60] loss= 9.6282  lr=1.37e-04  epoch=0.02
[step 25/60] loss= 7.8230  lr=1.20e-04  epoch=0.03
[step 30/60] loss= 6.4918  lr=1.03e-04  epoch=0.03
[step 35/60] loss= 4.6642  lr=8.67e-05  epoch=0.04
[step 40/60] loss= 3.9744  lr=7.00e-05  epoch=0.04
[step 45/60] loss= 3.5991  lr=5.33e-05  epoch=0.05
[step 50/60] loss= 3.1933  lr=3.67e-05  epoch=0.05
[step 55/60] loss= 3.4961  lr=2.00e-05  epoch=0.06
[step 60/60] loss= 3.2169  lr=3.33e-06  epoch=0.06
```
loss 持续下降，没有平台，证明 packing=False + CoT 数据组合有效。

---

## 七、数据集

| 名称 | Kaggle 路径 | 用途 |
|------|------------|------|
| dennisfong/nvidia-nemotron-offline-packages | `/kaggle/input/nvidia-nemotron-offline-packages/` | trl 等离线包 |
| mayukh18/nemotron-packages | `/kaggle/input/nemotron-packages/` | mamba_ssm 等 |
| dgxchen/nemotron-cot-tong | `/kaggle/input/nemotron-cot-tong/problem_ids_matched.csv` | CoT 训练数据（7830条） |
| ryanholbrook/nvidia-utility-script | `/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/` | Triton ptxas-blackwell 等 |
| metric/nemotron-3-nano-30b-a3b-bf16 | 通过 kagglehub 下载 | 基础模型 |

---

## 八、下周行动计划

### 优先级 1：周一推送 v22
```bash
cd /Users/exerzitien/Desktop/nemotron-kaggle/kernels/train
kaggle kernels push -p .
```
v22 唯一改动：callback 加 `and 'loss' in logs`，其余与 v21 完全相同。

### 优先级 2：跑完后立即提交
v22 完成后 `/kaggle/working/submission.zip` 就是可提交的 adapter。
```bash
kaggle competitions submit nvidia-nemotron-model-reasoning-challenge \
  -f submission.zip -m "v22: CoT SFT, packing=False, 60steps"
```

### 优先级 3：扩大训练规模
v22 只跑了 60 步（6% 的数据，约 21 分钟训练）。下一版：
- 去掉 MAX_STEPS 限制（或设为 300，覆盖全部 7830 条）
- 完整 1 epoch ≈ 4.4h，仍在 30h 配额范围内
- 可考虑 max_length=4096（更长 CoT，更好质量，约 2x 时间）

### 优先级 4：进阶优化
- **分层采样**：dgxchen 按 6 个问题类别做 stratified sampling，防止某类被忽视
- **更长训练**：如果 60 步就能到 loss≈3.2，完整 1 epoch 应该更低
- **更好 CoT 数据**：dgxchen/less_cot.csv（更简洁的 CoT）或 kienngx 数据集

---

## 九、关键教训总结

1. **每个 Kaggle 环境问题都有具体原因**，不要乱试，先查报错根源
2. **packing=True 在没有 flash_attention_2 的情况下是毒药**，loss 平台是明确症状
3. **callback 里所有格式化都要防御性处理**，训练结束时 logs 格式变化
4. **`save_strategy='no'` 是必须的**，默认保存会吃光磁盘
5. **实际每步耗时是估算的 35 倍**（估计 1.5s，实测 53s），下次要更保守估算
6. **先小跑验证，再放量训练**：v21 证明配置正确后，v22 应该去掉步数限制跑完整 epoch

---

## 十、代码复审发现（2026-05-19）

放量前对训练 kernel 做了一次完整复审，发现 4 个会浪费配额或毁掉训练质量的问题。

### 10.1 致命：14.5h 完整 epoch 塞不进 12h kernel
Kaggle notebook kernel 有 12h 执行硬上限。完整 1 epoch 需 ~14.5h（979 步 × 53s + 加载），
必然在第 12h 被 kill。而 `save_strategy='no'` + adapter 仅在最后一格保存 → 跑到被杀 = 配额全损、零产出。
**结论：一个 epoch 不可能在单个 kernel 内跑完，必须缩小范围或分多个 kernel。**

### 10.2 致命：序列截断静默毁数据
CoT median ≈ 3105 tokens，但 `max_length=2048` → 一半以上样本被截断。
截断从尾部切，切掉的正是 `</think>\n\boxed{answer}` —— 模型在大多数数据上根本看不到答案。
loss 曲线照常下降（仍在预测 think token），不会有任何截断警告。v21 的「loss 24→3.2」无法证明训练有效。

### 10.3 14.5h 盲跑零可见性
无 eval set、无训练中途 checkpoint。长跑一旦中途崩溃全部丢失，跑完提交才知好坏。

### 10.4 prompt 未 mask
当前 loss 算在 `prompt + completion` 整段上。SFT 标准做法应只在 assistant 回答部分算 loss
（`assistant_only_loss` 或 completion-only collator）。

### 行动：先 0 配额 CPU 诊断
新建 `kernels/diagnostic/`（CPU kernel，不占 GPU 配额），用与训练完全相同的 `apply_chat_template`
逻辑对全部 7830 条做 tokenization，输出长度分布 + 截断/过滤表。
据此再定 max_length、过滤阈值，配置一次 ≤11h 的 GPU run（含 prompt mask + warmup + 中途 save）。
kernel: https://www.kaggle.com/code/exerzitien/nemotron-length-diagnostic（2026-05-19 推送）

### 教训
**放量前做一次代码完整复审是值得的** —— 这次复审在烧掉 14.5h 配额之前发现了 12h 上限和截断问题。
回顾文档自身也有错（第五/八节误算 297 步 / 4.4h，实为 979 步 / 14.5h）。

---

## 十一、战略转向：移植 dgxchen 0.85 notebook（2026-05-19）

CPU 诊断（kernel `nemotron-length-diagnostic`）结果：CoT token median **3499**、mean 3872、max 7969。
max_length=2048 时只有 **24.8%** 样本完整 —— v22 若放量会截断 75% 的数据。

随后调研竞赛论坛 4 个 0.85 LB notebook，发现**根因**：

**所有 0.85 notebook 都用 Unsloth + max_length=8192**，全 7830 条零截断、跑完整 epoch、能塞进 12h。
我们 v1-v22 的 vanilla transformers + eager attention 路线本身就是弯路 —— 53s/步、12h 上限、
2048 截断的痛点根源都是没用 Unsloth。

**行动**：移植 `dgxchen/training-with-unsloth-to-achieve-0-85-lb`（用的就是我们的 CoT 数据集）。
逐字复刻其训练逻辑与环境依赖（含 docker 镜像 sha、5 个数据集），仅注入 PeriodicAdapterSave
（每 40 步存 adapter）+ PrintProgress 两个 callback。旧 v22 vanilla notebook 备份在 `kernels/_backup/`。

### 竞赛评测参数（来自 huikang/adapter-validation-notebook）
- vLLM 推理，temp=0 贪心，max_tokens=7680，max_model_len=8192，max_lora_rank=32
- verify()：数值 rel_tol=1e-2，否则大小写不敏感字符串匹配
- dgxchen 训练配方与推理端逐字一致 → verbatim ≈ 0.85

### 调研结论：没有免费增量
- dgxchen 的 CoT 数据按真 verify() 检验 100% 自洽（用粗略正则会误报噪声 —— 教训：用真标准验证）。
- 0.85→0.87 的差距在 huikang 的数据增强管线，不在改 dgxchen 超参。
- huikang repo 公开，方法即源码（reasoning.py / augmentation.py / corpus.py / train_sft.py），
  做 run #2 增强时直接读 repo 即可。
- 计划：run #1 verbatim 拿 0.85 锚点 → huikang 验证 notebook 出 9 类别准确率 → 针对弱类做增强。

### v22 首推失败 → v23（2026-05-19）
v22 推送后在 cell 7 秒挂：`FileNotFoundError: /kaggle/input/datasets/mayukh18/nemotron-packages/packages`。
原因：dgxchen 网页端数据集挂载在 `/kaggle/input/datasets/<owner>/<slug>/`，但 CLI 推送挂载在
经典路径 `/kaggle/input/<slug>/`。移植时为「最小偏离」保留了硬编码路径，赌错了 —— 应直接听
坑 7/坑 8（路径必须用 glob）。修复：cell 7 的 packages_dir 改为定位 unsloth wheel 的父目录、
cell 11 的 DATASET_PATH 改为 glob problem_ids_matched.csv。错误发生在环境搭建阶段，几乎没烧配额。
教训：移植网页端 notebook 时，输入路径一律 glob，不要保留硬编码 `/kaggle/input/datasets/...`。

---

## 十二、冲金核心路线：自造数据 + 链式课程训练（2026-05-19 确立）

**战略信条**：照搬别人的方法只能复刻别人的分数。0.86 是一大片高原，因为大家共享同一批
LLM 生成的 CoT 数据（dgxchen/nemotron-cot-tong 等）—— 数据就是天花板。要超越必须有别人
没有的杠杆 = 自有的、更优的数据。

### 杠杆一：程序化数据生成（出题器，非解题器）
9 类谜题全程序可解。为每类写**生成器**：规则我们自己选 → 生成样例+查询+答案+干净算法 CoT，
无欠定问题（我们就是出题方）。可行性已验证：
- ✅ gravity / unit_conversion / numeral：求解器 100% 复现答案。
- ✅ cipher：单字母替换密码；生成器已建（datagen/cipher_gen.py），300 题验证通过。
- ❌ bit_manipulation：7-10 样例天生欠定，对所有人都难，低优先。
- ❓ equation/cryptarithm deduce：多层 puzzle，待深逆向。
干净可生成约占 58% 题量。

### 杠杆二的正确理解：不是「数据更多」，是「数据更对」
模型单 kernel 12h 只能训 ~1-3 万条（GPU 时间卡死）。生成器的价值不在堆量（边际递减），
而在**固定训练预算内精确挑数据**：按弱类调配、覆盖长尾与边界、CoT 简洁一致。

### 执行模式：链式课程训练（eval-gated curriculum）
链式续训可突破单 kernel 数据上限（加载上一轮 adapter 继续训）。但真天花板是模型学饱/过拟合，
所以必须**评测闸控**：run → 评测 9 类别准确率 → 涨则生成器针对弱类出新数据续训、平/掉则停。
形成 run→评测→定向生成→续训 的闭环课程。

### 当前进度
run #1（dgxchen verbatim）训练中 → 0.85 锚点。之后 Phase 1 评测 → 继续 Phase 3 生成器
（gravity/unit/numeral）+ cipher 数据 A/B。详见记忆 project_nemotron_0.85_findings.md。

---

## 十三、run #1 结果 + Phase 1 留出集评测（2026-05-19）

run #1 完成，LB **0.84**（dgxchen 配方移植，基线锚点）。

**Phase 1 评测踩坑**：首次评测误用训练集（problem_ids_matched.csv）→ 98.3% 假准确率，白烧一个
GPU 评测额度。修正：改用 train.csv 里未训练的 3329 道留出题。教训见记忆 feedback_eval_holdout.md。

**留出集真实结果（891 道，模型没训练过）**：
- gravity / numeral / unit_conversion：100%（已解决，不投入）
- bit_manipulation：56.7%（弱，天生欠定）
- equation_numeric_deduce 5.4% / equation_numeric_guess 1.7% / cryptarithm_deduce 0% /
  cryptarithm_guess 0%（**完全崩** —— 训练集 v1 评测它们是 96-100%，纯背题、零泛化）
- cipher：留出池里 0 道，无信号

**诊断**：看 cryptarithm 错例，模型学了个坏捷径（把已知算符误判 unknown、"default to
concatenation"），根本没推规则。坏 CoT 数据教坏的 —— 正是生成器能修的点。

**Phase 3 优先级更新**：equation/cryptarithm（~0%，最大上升空间，需逆向规则）> bit_manip（57%）
> cipher（需补评测信号）。gravity/numeral/unit 已满分不碰。

**效率教训**：`kaggle kernels output` 用 `--file-pattern` 只下所需文件，别整包拉 3.3GB。
评测 kernel 已改为 adapter 解压到 /tmp（不进输出）。见记忆 feedback_fetch_only_what_needed.md。

---

## 十四、Phase 3 自造数据：生成器 + SFT 集成（2026-05-19）

### 错例精细分析
run1 留出错例逐条分析(0 GPU 配额,纯本地)：
- **equation_numeric_deduce**：53 错例中 **30 个(57%)数值算对、只败在答案装饰**
  (漏/多算符字符、正负号反)——不是推理崩，是格式规范没学到。
- **cryptarithm_deduce/guess**：0/240 全错且 0 接近——真·推理崩盘。

### 机制逆向
- **equation 谜题**：每题 2-3 个算符,各 = 被替换的符号 + 一种运算 + 一种**答案装饰约定**
  (plain / 算符后缀 `17/` / 算符前缀 `}51` / signsym 负数用算符替代减号 `/14`)。
  装饰约定本身也要从示例推断——这正是模型败因。
- **cryptarithm 谜题**：写约束求解器跑真题,标准算术(+-*/,含/不含前导零)**0 解** →
  确认是「数字符号替换 + 非算术怪运算」双重替换。算符是 5 符号 LHS 的中间位。
  真题里同一算符输出长度时而恒定(结构性)时而变化(算术性),两类混存。

### 两个生成器(datagen/)
- **equation_gen.py**：8 种运算 × 4 种装饰约定。signsym 仅配能产负数的运算;
  sub/rsub/absdiff/maxv/minv 强制含双向示例以消歧义。
- **cryptarithm_gen.py**：覆盖**结构性算符**(按位置图重排/选取/复制操作数)。
  与数字→符号映射无关 → 单示例即唯一可推、100% 可验证。CoT 是真演绎(输出符号回溯输入槽位)。
  算术型 cryptarithm(变长、需解 σ、常欠定)留作更难后续。
- 两者输出列匹配 problem_ids_matched.csv,可直接进 SFT。

### SFT 管线集成
notebook 加 `SYNTH_DROP_TYPES`/`SYNTH_ADD_GLOBS` 两开关(默认空 = 复刻 0.84 基线),
cell 11 数据加载处加混合块(5 列校验、glob 落空报错、组成日志)。
自造数据上传为私有数据集 `exerzitien/nemotron-synth-cot`。

### 放量前代码复审(纪律)
发现并修 1 真 bug：equation_gen 的 maxv/minv 在单向示例下与「取第一/第二操作数」欠定。
两生成器 validate 加强为逐例+query 重算校验,各 3000 题自校验通过。数据集重生成到 v2。

### run #24
kernel v24 推送运行：丢 equation+cryptarithm 基线行(658+627)+ 加等量合成数据,
总量精确持平 7830,隔离「CoT 质量」变量。待跑完 → 评测看 equation(基线 5.4%)/
cryptarithm(基线 0%)逐类翻盘信号。

---

## 十五、run #24 评测结果 + 深度归因 + v2 重写（2026-05-20）

### run #24 留出集准确率

| 类别 | 基线(run #1) | run #24 v1 合成 | Δ |
|---|---|---|---|
| equation_numeric_deduce | 5.4% | **23.2%** | +17.8 |
| bit_manipulation | 56.7% | 60.0% | +3.3（噪声）|
| equation_numeric_guess | 1.7% | 1.7% | 0 |
| cryptarithm_deduce | 0% | **0%** | 0（未动）|
| cryptarithm_guess | 0% | 0% | 0 |
| gravity/numeral/unit_conversion | 100% | 100%/99.2%/100% | -0.8 噪声 |
| **Overall** | ~48% | **50.1%** | +2.1 |

### run #24 深度归因（颠覆性发现）

**事先以为**:dgxchen 的 CoT 教坏了模型,我的简单 CoT 应该更好。
**实际**:dgxchen 的 CoT **设计更精致**,我的 v1 模板 **0% 被采纳**(889/891 用 dgxchen 模板)。

#### 真正的失败链

1. **模板粘性**:dgxchen 模板贯穿 9 类型形成多数派,我的 1285 行新格式被 6545 行老格式碾压。
2. **equation_deduce 77% 误归为 guess**:dgxchen 的 deduce/guess CoT 仅在 "found/not found" 文字一字之差,
   没有显式的 `is in / NOT in the example operators` 校验。模型在 deduce 题里 77% 走了 guess 的
   "use abs diff" fallback,精度大幅压低。
3. **equation_guess 126 行毒数据**:100% 含 "not found → abs diff",污染 deduce 类。
4. **cryptarithm_deduce 0%**:dgxchen 的 cryptarithm CoT **仅测 concat 和 reverse-concat 两种 op**。
   真题里其他规则永远 unknown → 永远 0%。
5. **cryptarithm_guess 154 行毒数据**:100% "default to concatenation"。
6. **训练侧 magnitude 退化**:max_grad_norm=1e9 = 无梯度裁剪,初始 loss=26.86 + lr 上升期 → 14 道
   signed magnitude 推理退化。

#### 27 道 equation 近失题(magnitude 对、装饰错)的 r24 命运

- 翻盘 9/30:全是"乱加装饰"型(`#22→22`、`-22→22`、`*35→35` 等)
- 仍错 21/30:**12 道是"应有负号"型**(模型既丢负号、又把 magnitude 推理烧了)、3 道前缀、1 道后缀、5 道 plain
- 还有 2 道 r1 正确 → r24 错的回退,其中一道是模型"无中生有"加了 `{` 前缀(过度泛化我教的 prefix 约定)

### v2 设计:套 dgxchen 模板,修内容

**核心原则**:模板粘性既然不可逆,就**沿用 dgxchen 模板骨架,把 4 个根因一一对症修补**。

#### 数据(2600 行,精准重平衡 + 适度扩)

| 类 | v1 量 | v2 量 | 关键修法 |
|---|---|---|---|
| equation_deduce | 658 | **1000** | 24 ops × 4 framings 完整搜索;**显式 "is in the example operators" 校验**;25% prefix-as-negative 装饰 |
| equation_guess | 126 | **200** | **显式 "NOT in the example operators"**;应用最常见示例算符的规则(不再无脑 abs-diff) |
| cryptarithm_deduce | 627 | **1200** | **12 候选规则全测**(concat/reverse-concat/left only/right only/interleave LR/interleave RL/.../left wraps right)|
| cryptarithm_guess | 154 | **200** | 最常见已定规则做 fallback(不再无脑 concat)|
| 总 | 1565 | 2600 | dgxchen 残余 6265 不变;总训练 7830→8865 |

#### 训练参数

| 参数 | 旧 | 新 | 理由 |
|---|---|---|---|
| max_grad_norm | 1e9 | **1.0** | 标准裁剪;直接针对 14 道 signed magnitude 退化 |
| learning_rate | 2e-4 | **1.5e-4** | 温和更新 |
| weight_decay | 0 | **0.01** | 轻 L2 正则 |
| 其他 | 保留 | 保留 | 1 epoch 不动(loss 已平台、不冒 14h GPU 风险) |

#### 自校验

每条生成数据 5 项强校验:
1. `\boxed{ans}` 末尾 + ans 无 `{}`
2. boxed extract == answer field 全 2600 行通过
3. FORK 句一致:deduce 必含 "is in"、guess 必含 "NOT in",且不能同时含两者
4. Apply 段 Result/output 行与 answer 一致
5. 端到端 dry-run:8865 行总数据 boxed 剥离干净

#### 关键洞察(写给未来的我)

- **不要假设你的 CoT 比 dgxchen 好**。先把 dgxchen 的 CoT 逐行读懂、按类型统计模式、检验真题分布,
  再决定哪里要改、哪里要保留。
- **模板粘性是真实的**:少数派比例的新格式不会被采纳。要改 CoT 必须**修内容、保骨架**。
- **deduce/guess 区分必须显式写在 CoT 里**。文字一字之差就是模型分不清的边界。
- **训练前一定 dump 全部错例做归因诊断**。run #24 之前若做了这步,就不会浪费一轮 GPU。

### run #25(v2)状态

kernel v25 已推送、运行中。预计 ~7.7h。评测预期:
- equation_deduce 应从 23.2% 显著上涨(去毒 + 完整搜索 + 显式校验)
- cryptarithm_deduce 应从 0% 突破(12 候选覆盖更广)
- cryptarithm_guess 应从 0% 突破(去毒)
- signed magnitude 14 道退化应缓解(max_grad_norm=1.0)

---

## 十六、v25 LB 后的全面归因 + 借鉴 huikang/Tinker 路线(2026-05-20 至 2026-05-25)

### v24/v25 LB 真相

| run | 改动 | LB | vs baseline 0.84 |
|---|---|---|---|
| v24 | dgxchen − 1285 行 + 我自造 v1 合成 1285 | **0.78** | **-6** ❌ |
| v25 | dgxchen 4 类全替 + v2 合成 + 训练参数保护(lr 1.5e-4, grad_clip 1.0, wd 0.01) | **0.82** | **-2** ❌ |

### 留出集 vs LB 严重背离 —— 第一血泪教训

- v24 留出 50.1%(+2.1 vs baseline) 但 LB **跌 6 点**
- v25 留出 49.8%(几乎不变) 但 LB **涨 4 点**(vs v24)
- 两次方向相反 → **留出集毫无方向价值**,只看 LB

血泪信条写入 `feedback_lb_signal_vs_local.md`(全局 memory):每次新方向**先小规模 run + LB 提交验证**再放量,不能光靠本地留出集。

### v25 深度归因(从竞赛论坛挖出)

- 关键发现:**dgxchen equation_numeric_guess 126 行训练数据 100% 含 "operator not found → use absolute difference" fallback**。模型在 inference 时把这条 fallback **错误泛化到 equation_numeric_deduce 的 77% 题上**(prompt 头几乎完全相同,模型分不清)。
- v25 改进的 +4 点(v24→v25)主要来自:`max_grad_norm=1.0` 保护了 LoRA 初始化阶段;数据替换部分修正了一些"乱加装饰"的近失题。
- 但仍 -2 跌穿基线,**根因是没做 prompt loss masking**。

### Tinker 探索(2026-05-22 至 2026-05-24)

用户发现"其他参赛者用 Tinker"。Phase 0 冒烟测试经历:
1. **Tinker 注册需绑信用卡**($150 自动信用是 waitlist 期间的福利,GA 后没了;Teaching grant $250 是给老师而非学生;Research grant $5k+ 可申请但门槛高)
2. **Nemotron-3-Nano-30B-A3B-BF16 真的支持**,rank 32 默认,5 步训练 + adapter 导出全跑通
3. **Mamba in_proj 和 MoE expert 格式不兼容 vLLM** —— huikang/asalhi 公开了转换代码(`weights/_merge_nemotron.py` + 论坛"tinker-submission-notebook"):专家 un-fuse(128 个 per-expert 展开)+ Mamba SVD 把 rank 64 (gate+x) 压回 rank 32。**SVD 损失 ~25%**(单步训练 adapter 实测保留 75%)
4. **Phase 0 端到端通过**:5 步垃圾训练 adapter LB 0.49(基本是裸模型分),证明格式兼容

**Tinker 的真实价值定位**:
- 并行多个 ablation 实验(Kaggle 串行,Tinker 并行)
- 绕开 30h/周 Kaggle 配额(花钱无上限)
- 缺点:Mamba SVD 固定损失 ~25%,可能值 1-3 LB 点。最终冲刺还得回 Kaggle/Unsloth

### 用户拍板:不要盲跑 ablation,先深度归因

> "我觉得你为了数据和训练参数的所谓贡献要直接跑三次有点太浪费,你先自己全面搜索,再看看notebook那些,看能不能自主归因"

——这一刀避免了 $60+ 盲跑成本。挖到的关键资源:

1. **huikang Open Progress Prize publication**(219 票, 0.877 target):
   - 真正赢家的方法:**deterministic CoT + min logprob 加权 loss + Tinker 训练**
   - 公开 GitHub(`tonghuikang/nemotron`)+ Kaggle notebook(`tinker-submission-notebook`)
   - 每类一个 reasoner(`reasoners/*.py`),`reasoning/*.txt` 9500 个预生成的真题 CoT

2. **Taha "Loss Illusion" 帖**(13 票, ranked 360):
   - Standard SFTTrainer 平均 loss 到 prompt+CoT 全部 token → loss 永远 ~2.5-3.0
   - Pre-tokenize + mask(0=prompt,1=CoT/answer)→ loss 真实落到 ~0.15
   - 这正是我们 v25 loss=1.5 vs huikang loss=0.15 的差距来源

3. **关键洞察:Phase 2(prompt masking)从 memory roadmap 一开始就写在那里,但被我跳过**。直接从 Phase 0/1 跳到 Phase 3 数据自造 → 在没有效训练的地基上盖楼

### v26 → v27 设计与训练参数纠错

**v26**(2026-05-24 推):
- 替换 dgxchen 全 9 类 → huikang 8333 行 rule_found CoT(从他 GitHub `reasoning/*.txt` 直接抽,answer 字段用 CoT 提取的 box 值保自洽)
- 加 `DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant\n")` + sanity check(防 collator 静默 mask 全部)
- 训练参数:继承 v25 保护性(lr=1.5e-4, grad_clip=1.0, wd=0.01)

**用户一针见血**:"为什么不直接用 huikang 的训练参数呢"
- huikang Cfg defaults:lr=2e-4 linear decay, batch=64, weight_decay=0.0, **grad_clip=1e9**(=无裁剪), beta2=0.95 —— ≈ **run #1 verbatim**(0.84 已验证)
- 我 v25 保护性参数是基于**没 prompt masking 时**的"magnitude 退化"误诊。masking 修后 initial loss 不再 26 而是 ~2.5,巨梯度问题自动消失

**v27**(2026-05-24 推,RUNNING)= v26 数据/mask 不动,**训练参数全部回到 huikang 原版**:
- lr 2e-4(linear decay)
- grad_accum 32 → 64(配合 huikang batch_size=64)
- weight_decay 0.0
- max_grad_norm 1e9(无裁剪)
- v26 已 stop,v27 接力

**第二血泪教训**:**先核对 reference 的所有变量,再决定保留自己的哪些"改进"**。我 v26 时盲目继承 v25 参数是错的。

### v27 LB 预期 + v28 候选(min-logprob 加权 loss)

预期分布:
- ≥0.86:prompt masking + huikang CoT 到位,推 v28 加 min-logprob 冲 0.87
- 0.84-0.85:核心改进生效但不足,v28 试补差
- <0.84:有 bug,先排查 mask 是否真生效(sanity check log),别盲加 loss

**v28 已设计(代码在 project memory 末尾)**:online 简化版 branch-weighted CE。huikang 原版要 2+ epoch + ref_logprobs,12h 限制下用 current step logprob 作为权重的近似版。

要点:`weight = min(1, |logprob| / branch_logprob)`,候选试值 0.005/0.01/0.05。

### Tinker 资源就绪(随时可启)

`exerzitien/tinker-smoke-convert` v4 已实现 huikang 的 SVD 转换。`~/.tinker_key` 存了 API key(用完撤销重发)。`huikang_ref/` 完整 GitHub 克隆。需要时可用 Tinker 并行 ablation(成本 $15-25 / run)。
