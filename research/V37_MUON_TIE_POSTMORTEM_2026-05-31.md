# V37 复盘：为什么"只换了个优化器"分数从 0.86 崩到 0.44

> **日期**：2026-05-31
> **结论一句话**：`Muon × 绑定的 MoE expert (MoE-tie)` 是致命组合。Muon 单独不致命（V34=0.85），tie 单独不致命（0.86 ref），两者叠加才崩（V36=0.38 / V37=0.44）。机制 = 优化器隐式偏好（implicit bias）：训练 loss 相同 ≠ 解相同；Muon 把更新"谱白化"（放大噪声方向），tie 又把 128 个 expert 塌成单一共享方向，叠加后把噪声相干地砸进所有 expert → 完美拟合训练集但泛化崩。
>
> **这份文档供后期查漏学习**：包含全部硬证据、推导、我中途走错的两步（及为何错）、以及尚未切干净的边界。所有原始证据位置都标了，可自行复核。

---

## 0. 原始证据位置（可复核）

| 证据 | 位置 |
|---|---|
| V37 训练 log（Muon@2e-3, LB 0.44） | `~/Downloads/nemotron-v37-replay-math-0-86.log` |
| 0.86 参考训练 log（AdamW@3.5e-4） | `~/Downloads/log.txt`（= amr V4 的 Kaggle run） |
| V37 kernel 源 | `kernels/v37-replay-math/v37-replay-math.ipynb` |
| 0.86 原版 notebook（已拉，V4） | `research/amr_086_v4/nemotron-replay-data-0-86.ipynb` |
| V34 kernel 源（Muon, LB 0.85） | `kernels/train/nemotron-lora-sft.ipynb` |
| V34/baseline adapter（拆过结构） | `kernels/train/adapter/adapter_model.safetensors` |
| V38（修复版 = AdamW，RUNNING） | `kernels/v38-replay-086/` |

---

## 1. 事实层（全部已核实，非推断）

### 1.1 LB 战绩
| run | 数据/流水线 | 优化器 | MoE-tie | expert 结构 | LB |
|---|---|---|---|---|---|
| **V34** | dgxchen, Unsloth SFTTrainer | Muon@2e-3 | ❌ 无 | 128 个**独立 2D** 矩阵 | **0.85** ✓ |
| **V36** | huikang, 手写循环 | Muon@2e-3 | ✅ | 3D 批张量，绑定 | **0.38** ✗ |
| **V37** | huikang+replay, 手写循环 | Muon@2e-3 | ✅ | 3D 批张量，绑定 | **0.44** ✗ |
| **0.86 ref (amr V4)** | huikang+replay, 手写循环 | **AdamW@3.5e-4** | ✅ | 3D 批张量，绑定 | **0.86** |

→ **受控对比 = V37 vs 0.86 ref**：流水线逐项相同（huikang base + replay ~2.0M answer tok + MoE-tie 5888 参 + CCE + 同 9 模块含 lm_head + RESET + batch32/micro4 + 1ep ~254 步），**唯一区别 = 优化器**。

### 1.2 两份 log 的 loss 曲线几乎重合（关键！）
| step | 0.86 ref (AdamW) | V37 (Muon) |
|---|---|---|
| 1 | 0.4304 | 0.4267 |
| 20 | 0.0809 | 0.1289 |
| 100 | 0.0579 | 0.0445 |
| 200 | 0.0740 | 0.0120 |
| 末步 | **0.0014** | **0.0018** |
| 末30步均值 | 0.0383 | 0.0205 |

两条曲线起点都 ~0.43、末步都到 ~0.001（0.86 那条甚至更低）。**训练动力学看不出任何区别。** V37 干净跑完 260 步，无 NaN / 无 OOM / grad_norm 平滑 1.14→0.017。

### 1.3 优化器身份确认
- **0.86 ref log**：`lr=3.5e-4`，全文**无** `Muon` / `Newton` / `zeropower` / `Creating Muon optimizer` 字样；notebook 里 `optimizer = torch.optim.AdamW(...)` → **AdamW@3.5e-4**。
- **V37 log**：`lr=2e-3` + `Creating Muon optimizer for 12010 LoRA tensors` → **Muon@2e-3**。

### 1.4 V34 与 V37 的 expert 结构根本不同（拆 adapter 实测）
V34 adapter（12008 张量）里 expert 是**每个 expert 一个独立 2D 矩阵**：
```
experts.0.down_proj.lora_A  shape=(32, 1856)   ← 2D
experts.0.up_proj.lora_A    shape=(32, 2688)   ← 2D
... 共 11868 个 expert 张量
```
量级健康：所有 lora_A ≈ 3.3（≈ kaiming 初值，几乎没动），lora_B 小而有梯度（0.08~0.44，expert_B≈0.17）。

V37/V36/0.86 ref 里 expert 是**单个 3D 批张量 `[128, r, in]`，且 tie**（`_tie_grads` 把 128 片梯度求和广播 → 128 片强制相同）。

### 1.5 两份 Muon 实现不同（为 1.4 的结构差异配套）
| | V34 Muon（纯 2D 写法） | V37 Muon（批处理写法） |
|---|---|---|
| 归一化 | `X.norm()`（全局标量） | `X.norm(dim=(-2,-1))`（逐切片） |
| 转置 | `X.T`（反转所有维） | `X.transpose(-2,-1)`（批） |
| 正交化 matmul | `X @ X.T` | `X @ X.transpose(-2,-1)` |
| 缩放因子 | `max(1, g.size(0)/g.size(1))^0.5` | `max(1, g.size(-2)/g.size(-1))^0.5` |
| 选参 | `"lora" in name and ndim>=2`（含 lm_head） | 同（含 lm_head） |

V34 的 2D 写法只能处理 2D 矩阵——它对每个 expert 的独立 2D 矩阵各自正交化。V37 的批写法配合 tie，128 个 expert 走**同一个**正交化方向。

---

## 2. 推理层：为什么会这样

### 2.1 正确的问题框架
用户原话"就差了个优化器"——**这只对 V37 vs 0.86 ref 成立**。V34 是**另一台机器**（独立 2D 未绑定 expert + 不同 Muon 实现 + dgxchen 数据 + SFTTrainer）。所以"V34 用 Muon 没事"≠"Muon 对 V37 也该没事"。

真正受控的 2×2（固定其它，只看 优化器 × tie）：
```
                  无 tie / 独立 expert        有 tie / 共享 expert
  Muon              0.85 (V34) ✓               0.38/0.44 (V36/V37) ✗
  AdamW              —（没干净跑过）            0.86 (ref) ✓
```
→ **Muon 不是问题，tie 不是问题，`Muon × tie` 才是。**

### 2.2 为什么 loss 一样（都 0.001）分数却 0.86 vs 0.44 —— implicit bias
神经网络损失面有无数个 train-loss≈0 的极小点，泛化天差地别。**不同优化器把你推向不同的极小点**（隐式正则/隐式偏好）。这是"loss 不是 LB 判别器"的根本原因。
- **AdamW**：逐坐标 RMS 归一，更新 ≈ `lr·sign(g)`；在弱信号方向上更新很小 → 偏好平坦、泛化好的解。
- **Muon**：把梯度 `G=UΣVᵀ` 正交化成 `UVᵀ`，**所有奇异值拉成 1**（"谱白化"）。在第 32 个奇异方向（噪声）和第 1 个方向（主信号）上**使一样大的劲**。对普通隐藏层是优点（更快），但意味着**它不压低噪声方向**。

### 2.3 为什么偏偏 `Muon × tie` 爆
1. **tie 把 128 个 expert 塌成一个共享方向**，其梯度 = `Σ Gᵢ`（128 expert 求和）。这个和的低奇异方向≈"各 expert 的分歧/噪声"。
2. **Muon 把这个共享方向白化** → 把噪声方向**放大到满强度**，再**复制到全部 128 个 expert**、每步每层。
3. **没有多样性平均噪声**：所有 expert 同步走同一个被放大了噪声的方向 → 一个"谱平坦的相干扰动"反复推 MoE 计算 → 完美拟合训练集（loss→0.001），但权重被推到泛化崩的地方。

对照不爆的两个：
- **V34（Muon×独立 expert）**：128 个 expert 各用自己的梯度独立白化 → 方向各异、噪声相互抵消，每个 expert 只影响自己路由到的 token。无"相干放大"。✓
- **0.86（AdamW×tie）**：同样共享方向，但 AdamW 用 `sign(ΣGᵢ)`，不把噪声放大到满强度 → 落在泛化好的谷。✓

**核心机制：tie 制造"单一共享方向"，Muon 恰好"把所有方向（含噪声）一视同仁放大"——叠加 = 把噪声相干地砸进所有 expert。**

---

## 3. 诚实的边界（这题我错过两次，必须分清事实与假设）

### 3.1 我中途走错的两步（学习材料）
1. **第一次猜"Muon 发散/把权重训爆"** → 被 V37 log 证伪：loss 平滑下降、grad_norm 1.14→0.017、无 NaN。**不是发散。**
2. **第二次猜"Muon 把训练集背死了（loss→0.0018=记忆过拟合）"** → 被 0.86 ref log 证伪：**成功的 0.86 run 同样训到 0.0014 还泛化到 0.86**。**loss 根本不是判别器。**

**教训：别凭单条 loss 曲线推因。** 两次都是只看了 V37 自己的数据就下结论，直到拿到"对照组"（0.86 log）和"地面真相"（V34 adapter 结构）才纠正。**受控对照 + 拆真实产物**才是出路。

### 3.2 还没切干净的嫌疑：lm_head
Muon 的设计铁律是**不要用于输出头**——对 `lm_head_B [vocab, 32]` 会乘 `(vocab/32)^0.5 ≈ 68×` 放大，可能严重扭曲输出分布。V37 把 lm_head 也丢进了 Muon。
**但**：V34 的代码同样注入 lm_head 并丢进 Muon，却没爆 → **lm_head 大概率不是单独主因**（否则 V34 也该爆），更可能是 `Muon×tie` 为主、lm_head 放大为辅的叠加。仅凭现有数据无法把这一条完全切干净。
（注：本地 V34 adapter 的 config 是 8 模块**无 lm_head**，疑似 v23/baseline 旧产物，非 V34 本体 → 无法用它确认 V34 跑时 lm_head 的真实状态。这是个未闭合的小缺口。）

### 3.3 要 100% 钉死主次权重，需要的消融（各 1 个 GPU run）
- **① Muon + 不 tie 的 3D expert**（huikang 流水线）：直接测"tie"是不是命门。预测：若回到 ~0.85，则 tie 确认为主因。
- **② Muon + tie，但 lm_head 改用 AdamW**：测 lm_head 放大的贡献。
- **③ Muon@更低 lr（如 3.5e-4 或 5e-4）+ tie**：测是否"lr 太高"而非"Muon 类型"。
- **V38（AdamW@3.5e-4，已 RUNNING）**：确认"换回 AdamW 能回 0.86"——机制的修法侧闭环点。

---

## 4. 可迁移的元教训（写给未来的自己）

1. **训练 loss ≠ LB**。本案铁证：两个 run loss 曲线重合到 0.001，LB 却 0.86 vs 0.44。以后**别用 loss 收敛判断一个 run 好不好**，只认留出 eval / LB。（呼应 [[lb-signal-vs-local]]）
2. **优化器不是即插即换**。Muon ↔ AdamW 的隐式偏好不同；同一个 lr/数据下解的泛化能差 0.4 个 LB。换优化器 = 换实验，必须单独验证。
3. **把一个"0.85 配方"搬到新流水线时，同一个优化器可能行为完全变**——因为参数结构变了（独立 2D expert → 3D 批 + tie）。**移植时要核对参数的实际结构（拆 adapter / 看 named_parameters），不能假设"代码一样就行为一样"。**
4. **Muon 的两条已知锋利边**：① 不要作用于 embedding/lm_head（输出头）；② 对"被强制共享/低多样性"的参数（如 tied MoE）会把噪声方向相干放大。用 Muon 必须给这两类参数单独分组（通常用 AdamW）。
5. **诊断顺序**：先建受控对照（只差一个变量的两个 run）→ 看 log 的 loss/grad/报错 → 拆真实产物（adapter 张量结构与量级）→ 才下机制结论。本案就是靠这个顺序从两个错误假设里爬出来的。

---

## 5. 当前状态
- **V38（= 0.86 配方 AdamW@3.5e-4，逐字照搬 amr V4，仅改 id/title）已 RUNNING**（2026-05-31 推送）。预期 0.86 = 首破 0.85。
- 机制结论（第 2 节）是目前证据支持度最高的解释；精确主次权重待第 3.3 节消融。
- 相关记忆：[[nemotron-v37-run]]、[[nemotron-surya-086]]、[[nemotron-v34-training]]、[[lb-signal-vs-local]]、[[verify-dont-assume]]。
