# V36 会话日志 + 续作手册 (2026-05-29)

> **新窗口从这里开始读。** 本会话因额度不足中断在"开始写 cryptarithm_gen_v4.py"之前。
> 全部结论已核实（非臆测）。配套记忆见 `~/.claude/.../memory/` 的 MEMORY.md 索引。

---

## 0. 当前任务 = V36：cryptarithm 搜索式 CoT（用户已拍板）

目标：让模型**真正泛化**"只从样例推出数字映射"的演绎技能，把 cryptarithm 正确率从 9.4% 提上去。
**下一步动作（被中断处）= 写 `datagen/cryptarithm_gen_v4.py`**（规格见第 5 节）。全程 0-GPU，预检全过才 push。

四个待办（本会话用 TaskCreate 建过，但任务列表可能不跨窗口，以此为准）：
1. **写 cryptarithm_gen_v4.py**（in_progress，未开始写代码）
2. 本地验证：round-trip 100% + token<8192 + 人工读可学性
3. 迭代 narration 至最优可学性
4. 集成训练数据（只换 crypt、绝不碰 bit、保 V34 配方、dry-run 后 push）

---

## 1. Kaggle MCP 授权已修（重要：再坏了怎么办）

- 病根：`~/.claude.json` 里 project `/Users/exerzitien` 的 kaggle MCP 写死了**已失效** token `KGAT_<REDACTED>...`（initialize 能过、真实 tool call 返回 `Unauthenticated`）。
- 已改成**动态读取** `~/.kaggle/access_token`：
  ```json
  "kaggle": {"type":"stdio","command":"sh","args":["-c",
    "exec npx mcp-remote https://www.kaggle.com/mcp --header \"Authorization: Bearer $(cat ~/.kaggle/access_token)\""],"env":{}}
  ```
- **token 再失效**：只改 `~/.kaggle/access_token`（38字节小文件，当前有效值 `KGAT_<REDACTED>`）；**别用 `mcp__kaggle__authorize` 工具（会卡死）**；改完 `/mcp` reconnect 或重启 Claude Code。
- 拉 kernel 输出：`download_notebook_output(ownerSlug=exerzitien,kernelSlug=...,filePath=...,versionNumber=N)` 返回签名 URL，**当步立刻 curl**（会过期）。

---

## 2. 全部版本真实 LB（已从 Kaggle 核实，非记忆）

| 版本 | 配方 | LB |
|---|---|---|
| v23 | dgxchen verbatim + Adam | 0.84 |
| v24/v25 | 自造 v1/v2 CoT | 0.78/0.82 |
| v29 | huikang full + prompt mask | 0.79 |
| v30 | dgxchen + lm_head | 0.83 |
| **V33** | Muon + bit-Donald + crypt-lkevincc | **0.73 灾难** |
| **V34** | Muon + dgxchen bit + km-crypt | **0.85**（当前最好，退路锚点）|
| **V35** | V34 + bit_v4 swap (huikang格式) | **0.76 灾难** |

**真实战绩：优化器(Muon)+0.01 是唯一的胜；6 次数据 swap 全 ≤0，两次 bit swap 灾难。**

## 3. 已结案的关键诊断（别再走回头路）

- **V33/V35 灾难真因 = bit 单类崩塌**：V33 评测(nemotron-adapter-eval **v9**)显示 bit 93.5%→**42.6%**（≈86/90 失分），未动的类别(eq/gravity/numeral/unit)分毫不差。V35(bit_v4 保 huikang 格式)仍 −0.09 → **证伪"格式冲突"假设；动 bit 数据本身就毒**。→ **bit = 永久禁区。**
- **km cryptarithm 只有 95 unique**（=kimberleyduran 全部），V34 已用尽 = +0；"用更大 km 集"是伪杠杆。lkevincc crypt(V33) 反而更差(1.9%)。
- **min-logprob 是 ≥2-epoch 方法**（epoch0 纯CE记logprob，epoch1才加权）；我们 1-epoch 下=纯CE白加；online近似≠它且 V32 崩。**别再提 1-epoch min-logprob。**（详见记忆 [[nemotron-minlogprob-2epoch]]）
- **0.87 仍是绝对天花板**（实时排行榜 5-28：8 队 0.87、0 队 0.88）；0.86≈30+ 队；我们 0.85 已在公开 notebook 天花板。最近 notebook/讨论无新公开杠杆（那批 `cc9xp`/`pretrain-50b-wsd` 是某小组从零预训练，与本赛 LoRA 无关，噪声）。

## 4. V34 评测分类基线（val_950，nemotron-adapter-eval 最新版=89.5%）

```
bit 93.5% | cipher 99.4% | crypt_deduce 9.4%(53题) | crypt_guess 10%(20题)
eq_deduce 91.2% | eq_guess 28.6%(近信息上限25%) | gravity 99.5% | numeral 100% | unit 99.3%
```
错误预算：cryptarithm 66/100 = 最大错池（=V36 攻击目标）。bit/eq_deduce/简单四类近天花板。
**纪律：以后每次 eval 把 results.csv 拉到本地 eval_results/<版本>/**（见记忆 [[eval-save-per-class-local]]）。

---

## 5. cryptarithm_gen_v4.py 设计规格（核心 —— 照此实现）

### 5.1 可行性（已量化，0-GPU）
用 `research/lkevincc_solver/data/solver_results.parquet`（823行,800 solver_correct）：
- 类别：**standard-arith 315(39%)** / little_endian 276(35%) / concat 134(17%)。
- **std-arith 68.3% 完全 forced**（σ 只靠样例唯一确定、0猜测=纯演绎可学）；全步骤 94.5% forced。
- **v1 范围 = forced std-arith(~215) + concat(134) ≈ 349 题**。little_endian + 含猜测的(~85) 留 v2。

### 5.2 关键复用（别重写求解器）
`research/lkevincc_solver/src/solver_eq_symbolic.py` 已内置全套：
```python
import sys; sys.path.insert(0,'research/lkevincc_solver/src')
from solver_eq_symbolic import AliceEquationSolver, OPERATIONS, SIGNED_OPS
s = AliceEquationSolver(prompt)            # 解析 examples/query；mode='standard' 才有轨迹
order, trace = s._derive_order(mapping, ops, mode='standard')  # 只用样例的 CSP 演绎轨迹
qt = s.apply_query_trace(mapping, ops, 'standard')             # 套用到 query，qt['encoded_answer']
struct = s.narrow_op_candidates_structural()                   # 算符的 sign_pattern/result_lens(供算符ID justification)
```
- parquet 字段：id, prompt, answer, solver_mode, solver_ops(JSON), solver_mapping(JSON), solver_radix, solver_correct, solver_category。
- **75 行 concat 的 solver_ops JSON 含 `\"` 转义会让 json.loads 失败** → try/except 归为 concat 桶。
- trace 每步：`{sym,digit,domain(=[digit]),kind('forced'/'guess'),determined_before,per_example_feasibility}`。
  `per_example_feasibility['passes'][0]['examples'][i]` = `{lhs,rhs,op_char,op_type,feasible_per_sym{sym:[digits]}, intersected_after,...}`。
- **witness 已确认**：几乎每个 forced 步可归因到单个等式（已锁符号代入后把目标符号逼到唯一值）；首步偶尔需 2 等式求交；末步=AllDiff("只剩一个数字")。

### 5.3 CoT 模板（"展示搜索"，可学；不是断言）
按此 5 段渲染，**答案取自 qt['encoded_answer']（保证 round-trip）**：
1. **Setup**：每例 = 两位数 [算符] 两位数 = 结果，符号是 0-9 的固定双射；要从样例还原映射再解 query。
2. **算符识别**：对每个 op char，用 struct（result_lens/sign_pattern）做依据，陈述 ops[oc] 的运算（OP_HUMAN 见下），注"下方验证"。
3. **逐符号 forced 演绎（按 trace order 级联）**：每步把 determined 代入 binding 等式，展示该等式把 sym 收窄到 {集合}，（必要时与第二等式求交）→ 锁定 sym=digit。binding 等式只剩 1 个未知时**展示完整算术**。末步=AllDiff("已用数字{...}，只剩{d}→sym=d")。**这一段是可学性命门**：尽量让每步是"代入已知→对该等式解出唯一未知→验证一致"的可复现算术；bootstrap(det=0)步做**有界枚举**最强约束等式(重复符号/短结果)。
4. **验证全样例**：用映射逐例计算 LHS→结果，展示 == RHS（完整算术）。这段全可学、教模型自检一致性。
5. **套用 query**：用 qt 展示算术 → `\boxed{answer}`。
末尾 footer：`cot.rstrip() + "\n</think>\n\\boxed{" + answer + "}"`（dgxchen 风格）。

OP_HUMAN（补全常见）：add:a+b, sub:a−b, rsub:b−a, absdiff:|a−b|, mul:a×b, add_m1:a+b−1, add_p1:a+b+1, mul_m1:a×b−1, mul_p1:a×b+1, absdiff_m1/p1, sub_m1/p1, rsub_m1/p1, gcd, lcm, fdiv:a//b, rdiv:b//a, mod, rmod, min, max, sub_signed/rsub_signed(带符号前缀), neg_absdiff（符号前缀+|a−b|）。

mode：**v1 只做 standard（无数字反转）**。little_endian 需在 encode 时反转数字串（v2）。

concat（134题）：单独简单模板——"算符把左右两个两位数拼接(fwd/rev)"，结构可学，huikang 已覆盖该思路。

### 5.4 输出 + 验证门（Task #2）
- 输出 5 列 CSV：`id,prompt,answer,type,generated_cot`（type=cryptarithm_deduce/guess），存 `datagen/synth_dataset_v3/cryptarithm_v4.csv`。
- **round-trip 100%**：正则取最后一个 `\boxed{...}` == gold answer（用官方 extract_final_answer 逻辑更稳）。
- **token<8192**：用 `huikang_ref/tokenizer.json`（或 transformers AutoTokenizer 加载 Nemotron）实测；估算 chars/3.5。
- 人工读 5 道：确认每步是可验证算术、无答案泄漏（forced 步不能用到 gold）、连贯。

### 5.5 集成（Task #4）—— 严格单变量 vs V34
- 训练 kernel `kernels/train/nemotron-lora-sft.ipynb`，cell 2 开关：`SYNTH_DROP_TYPES=['cryptarithm_deduce','cryptarithm_guess']` + `SYNTH_ADD_GLOBS=[cryptarithm_v4.csv]`（替换，不动 bit/其它）。
- 其余 = V34 配方：Muon lr=2e-3 linear、standard CE、bs=1×grad_accum16、1 epoch、max_len8192、lora r32 含 lm_head。
- 上传 csv 到 Kaggle dataset（`exerzitien/nemotron-v32-data` 或新建），端到端 dry-run（boxed 不双框、总量、各类组成）→ 才 push（~7h GPU）。
- push 后 eval(val_950) 看 crypt_deduce 是否大涨 + total，再 LB 提交。

---

## 6. 诚实风险 + 决策树
- 演绎框架可学 ≠ SFT 一定教会执行（全竞赛卡住点）。命门是 5.3 第3段 narration 质量。
- 乐观：crypt_deduce 9.4%→30-45% ≈ **+0.01~0.02 LB**；成功率我估 ~15-25%。
- push 后：crypt_deduce 评测大涨 + LB≥0.86 → 成功，扩到 little_endian(v2)；评测涨但 LB 平 → 可能背题，查泛化；LB<0.85 → 回退 V34（crypt 是独立类、已很低，不应像 bit 那样灾难）。
- 退路永远是 **V34=0.85**（submission #53080019）。

## 7. GPU/截止
- 截止 2026-06-15；GPU 30h/周（周一重置，约 6/2）；一个 train+eval ≈ 9h → 每周~3 run，剩 ~7 run。
- 不盲推 GPU；高度确定（预检全过）再 push。

## 8. 关键文件
- 生成器(待写)：`datagen/cryptarithm_gen_v4.py`
- 求解器(复用)：`research/lkevincc_solver/src/solver_eq_symbolic.py` + `data/solver_results.parquet`
- 旧 v3(断言式,反面教材)：`datagen/cryptarithm_gen_v3.py` / `synth_dataset_v3/cryptarithm_v3.csv`
- 训练/评测 kernel：`kernels/train/`、`kernels/eval/`（slug `exerzitien/nemotron-adapter-eval`）
- 本日志 + 记忆 MEMORY.md 索引（auto-load）。

---

## 9. 进度更新 (2026-05-29 续) — cryptarithm_gen_v4.py v1 已写并测

**状态**：`datagen/cryptarithm_gen_v4.py` v1 完成。在 215 forced std-arith 上：
- **round-trip 10/215 OK**；205 失败 = `seed-ambiguous`。
- 成功的 CoT 质量好：~314 token（极短）、每步真实算术正确、verify 段干净、dgxchen 风格（footer 待加）。

**已定位的核心 gap + 修法**：
- v1 的 seed 用"单个等式"锁符号 → 单等式几乎定不住 → 205 失败。
- 真实 deduction = **跨等式约束传播**：贪心单未知级联(97%干净步) + 卡住时跑一次 propagation(迭代每等式可行集求交→锁单值)批量裂解 seed。
- **下一步**：在 build_cot 里加 `feasible_sets(eq, locked, domains)` + propagation 循环替换弱 seed。测 round-trip + "断言式 seed 锁掉的符号占比"(=可学性代价)。
- 然后 Task#3 迭代 seed narration：把"combined constraints 断言"升级为个位同余/量级/重复符号的真推理（③泛化命门）。

**4 invariants 落地检查清单**（每轮验证）：
- ④准确：round-trip 100%(末尾\boxed==gold，符号串精确)；每步 compute() 真算。
- ②不毒：token 压在 dgxchen 区间(v1 ~314 远低于上限✓)；只换 crypt；训练后全 9 类 eval。
- ①可学 ③泛化：cascade 步=单未知真算术(已达)；seed 步必须从断言升级为可复现推理(待做)；零答案泄漏(mapping 只取值不入推理依据)。

**关键事实**：cascade(单未知)97% 可学已验证；唯一难点=每题的 seed 批(跨等式)，其 narration 质量 = 整个方案可学性的命门。

---

## 10. 决定性发现 (2026-05-29) — cryptarithm 是"不可约多元联立", seed 是命门

`cryptarithm_gen_v4.py` v2(最小seed+单未知级联) 实测 215 forced std-arith：
- **round-trip 215/215 ✓ (④准确达成)**, CoT ~311 token (②长度无忧), dgxchen 风格。
- **但 seed(被断言、非推导的符号数) 太大**, 按"每步允许解 k 个未知"：
  - k=1: 平均 seed **5.80** (dist 4:7,5:59,6:119,7:29,8:1) — seed≤1 占 0%
  - k=2: 平均 **3.36** — seed≤1 占 1%
  - k=3: 平均 **2.12** — seed≤1 占 29% (但 3-未知步要枚举~720 组合, 模型难复现)
- **本质结论**: 这类题不存在"小 seed + 干净级联"分解; 是不可约多元联立。seed 那 ~2-6 符号靠"断言"=违反 ①可学 ③泛化 (模型学不会找种子→新题废)。这就是全竞赛卡死 cryptarithm 的量化根因。

**唯一可能破局杠杆 = 个位同余(+量级)**: 提供更便宜的 mod-10 约束, 有望把 seed 联立搜索剪到"可枚举少数候选", 让 seed 也成为可学的有界搜索(人类先定个位)。**未验证能否够; 是 ①③ 达标命门。**

**下一步(被中断处)**: 实测"个位+量级"能把 seed 候选集剪到多小:
- 对每题的 seed 符号(k=2 级联剩下的~3个), 算每等式的个位同余(mul: b×d≡r_units; add: b+d≡r_units mod10; absdiff 复杂)+ 结果长度量级界, 看联合候选数。
- 若剪到 ≤~5 → seed 可学(枚举给模型看) → 写进生成器, route C 成立。
- 若仍 ~100s → cryptarithm seed 可能不可学; 需告知用户、考虑 fallback。

**Fallback 选项(若个位剪枝不够)**: 只为"小 seed 子集"(k=3 下 seed≤1 的 ~29%≈60题)生成可学搜索CoT, 其余 crypt 保留 dgxchen 原版(不退步、不毒)。但 60 题可能不够撼动 LB。

**生成器现状**: `datagen/cryptarithm_gen_v4.py` = 最小seed+单未知级联+全验证+套用query, round-trip 100%, 但 seed 段是断言(待个位升级)。helpers(compute/eq_all_syms/cascade/minimal_seed) 可复用。

---

## 11. 🛑 铁结论 (2026-05-29) — cryptarithm 显式搜索 CoT 不可学, 死路

**决定性测量**：bootstrap 等式(开局最受约束的)经"个位同余+结果长度"双重剪枝后, 模型仍要枚举的候选数：
- **中位 3024 / 均值 6555 / 最小 20 / 最大 74400; 0% 的题 ≤40。**
- 即: 找数字映射这步是**数千候选的联立搜索**, 7680 token 装不下、模型学不会。

**因此**(满足事先约定"剪不够落退路", 且退路也不成立——bootstrap 普遍数千、无可学子集)：
- **cryptarithm deduce 无法用任何可学 CoT 解决**(显式搜索太长; 断言式=v3/km/lkevincc 已实测 1.9~9.4% 且 ≤baseline LB, 不泛化)。这定量证明了全竞赛(含0.87队)卡死 cryptarithm 的原因。
- **别再尝试 cryptarithm 显式/断言 CoT**。route C 在此类不成立。

**根因**：10符号→数字双射 + 多为乘法(不像加法能逐列分解), 信息分散, 廉价约束(个位/量级)只剪~50x, 远不够。

**全局含义(诚实)**：0.85 时剩余错池全部是——bit(碰即崩,禁区)/eq_deduce+简单四类(近天花板)/两个guess(近信息上限25%)/cryptarithm(本条:不可学)。**没有可学的数据杠杆剩下了**, 这就是 0.87 frozen 数月的真因, 我们 0.85 已在公开天花板。

**剩余唯一有据可试的杠杆 = min-logprob 2-epoch**(训练动力学, 非数据; huikang 获奖技术, 我们从未干净跑过; 需 2-epoch 链式; ~25-30% 可能 +0.01; 详见 [[nemotron-minlogprob-2epoch]])。否则锁定 V34=0.85。

cryptarithm 生成器 `datagen/cryptarithm_gen_v4.py` 保留作记录(round-trip 100% 但 seed 靠断言, 不泛化, 别上 GPU)。

---

## 12. 🎯 找到 0.02 杠杆的真相 (2026-05-29) — base 数据 (huikang CoT) 才是关键, 不是 swap

**推翻"无杠杆"悲观结论。** 深挖本地 Bobber 0.86 完整 notebook + 论坛 #701761 顶尖队披露：

**论坛硬事实(顶尖队亲口)**:
- huikang: public LB≈250题, 0.872=218/250 → **0.85→0.87 只差 ~5 题**, 是积累小分非单一大杠杆。
- Giovanny(榜上): "cryptarithm 唯一难点=设计模型真能学会的 CoT" → **证实 cryptarithm 全员未破, 不是 0.02 来源**。
- JK-Piece: bit 长 CoT 易在结尾出错 = 抠分点。
- Murugesan: 改一类 CoT → 暴跌 0.43 = CoT 改动极脆弱(印证我们 v33/v35)。

**Bobber 0.86 真实配方拆解(本地 vex506-reinst notebook)**:
- **1 epoch, completion-mask CE(非min-logprob!), Muon + CCE(cut-cross-entropy) + batch32(microbatch4)**。
- base = **huikang 的 CoT corpus**(huikang/huikang-nemotron-repository-snapshot 公开快照), **不是 dgxchen**。
- 叠 swap: cryptarithm-THK(kimberleyduran,+0.01) + V6 + binary-fmt + 13.5K hyp-revise synth。
- Bobber 原话: "单变量=叠55个cipher_digit CoT, 复现 +0.01 LB"。

**决定性洞察(解释我们所有失败)**:
> 同一个 km cryptarithm swap: 在 huikang-CoT base 上=+0.01(Bobber), 在我们 dgxchen-CoT base 上=+0(V34实测)。
> **差距在 base 数据(huikang CoT vs dgxchen CoT), 不在 swap。我们一直用 dgxchen base→swap不粘; 我们的 bit swap 也因 dgxchen base 脆弱而崩。**
- 我们从没干净跑过 huikang base: v29(huikang数据)=0.79 是因为**没Muon+加了prompt-mask+管线错**, 不是 huikang 数据本身差。错误地打成死路退回 dgxchen。

**V36 精确方案(验证过的 0.86 provenance)**: 复现 Bobber vex506 base = **huikang公开corpus + Muon + CCE + batch32 + 1epoch**(我们有他完整notebook), 再叠 **km cryptarithm swap(公开)**。预期 0.85→0.86。
- 风险: huikang数据在Unsloth上finicky(v29翻车), 必须照Bobber管线精确复现(用他notebook); km swap +0.01 是Bobber声称, 需LB验证。
- 资产: research/bobber_086/vex506-reinst-kaggle-replica.ipynb; huikang公开快照; datagen/huikang_dataset/huikang_cot.csv; kimberleyduran_v2 crypt。
- konbu17/kienngx 不在 top-40(非0.86), 记忆"kienngx=0.86"是臆测, 已排除。

---

## 13. ✅ Bobber 0.86 核实 + huikang-base 路线确认 (2026-05-29) — V36 执行计划

**Bobber 0.86 核实结果**:
- `bobber/vex506-reinst-kaggle-replica` **best_public_score=0.86 实锤**(Kaggle/Unsloth, 非仅Tinker)。
- 但用了**私有 swap 栈**(km-cryptarithm公开 + V6/binary-fmt/13.5K hyp-revise私有); public 版 swap 路径已空 → 照搬只到 huikang base≈0.85。
- Bobber 帖 #702447 亲述: 20K合成/GRPO/换CoT格式 **全部突破不了0.86**, "0.86是天花板"。→ **0.86→0.87 公开不可复现**(0.87队frozen+私有)。诚实目标=搏0.86。

**huikang 数据资产(已核实, 本地+Kaggle都有)**:
- Kaggle: `huikang/huikang-nemotron-repository-snapshot`(5.2GB,公开CC-BY,13kernel在用)。
- 本地: `huikang_ref/training/sft/04-08-16-14/` 有全套: `tokens/<pid>/synthetic.json={tokens,mask}`(7830 pid,0缺失) + `logprobs/index.jsonl`(训练顺序) + config.json。
- huikang config: **plain CrossEntropy, num_epochs=1, batch_size=32, lr=2e-4, rank32, train mlp/attn/unembed(含lm_head)**。(注: 非min-logprob/非2-epoch — 之前假设修正。)

**🔑 v29(0.79)翻车真因(高置信)**: v29 把 huikang_cot.csv **重新过 Unsloth tokenizer** → tokenization/mask 对不上 huikang chat-template → 死记/崩。**Bobber 直接喂 huikang 预tokenized token-id, 绕开 re-tokenization**。这是修法, 也是为何必须用 Bobber 手写训练循环(喂token-id+CCE手算loss), 不能用 SFTTrainer。

**V36 执行计划(增量, 全程先0-GPU)**:
- **Run1 = huikang-base only**: 改造 Bobber vex506 notebook → Kaggle-only(去Modal), RESET_WEIGHTS=True(fresh LoRA), 挂 huikang公开快照, **不加私有swap**(代码遇缺自动跳过), Muon+CCE(cut_cross_entropy)+batch32(micro4)+1epoch。预期~0.85, **验证管线(防v29)**。base跑不到0.85就停debug。
- **Run2 = +可复现swap**: km cryptarithm(公开,我们有) + 我们MAJ0RT0M自造 huikang-格式 binary-fmt bit, tokenize成{tokens,mask}同格式按pid覆盖。eval全9类防毒。目标0.86。
- 关键文件: research/bobber_086/vex506-reinst-kaggle-replica.ipynb(移植源); huikang_ref/training/sft/04-08-16-14/tokens(本地数据); CCE patch在bobber cells 526-562; Muon 675+; 数据加载 243-373; 训练循环 749-885。
- 风险诚实: huikang-base-on-Unsloth需精确复现Bobber; CCE离线包是否可用待确认(在读); km在huikang base上+0.01是Bobber声称未经我们验证; 私有swap无法照搬→可能到不了0.86。退路V34=0.85。

---

## 14. V36 Run1 已构建+验证 (2026-05-29) — huikang base + Muon (Muon-only)

**kernels/v36-run1-huikang-base/** 建好。基于 huikang 亲验 0.84-0.85 notebook (ref-085, = `huikang/end-to-end-finetuning-for-lb-0-85`, huikang本人评论区背书"0.84,0.85,0.84,0.84,0.84"), **唯一改动 = AdamW→Muon(lr 2e-3)**, 移植自 Bobber vex506 的 Muon(NS5+nesterov+scale)。
- 配置: CCE + Muon + MoE_TIE_WEIGHTS=True + 9模块(q/k/v/o/up/down/in/out/lm_head) + batch32/micro4 + 1epoch + RESET_WEIGHTS=True + huikang 预tokenized corpus(7830)。
- 校验全过: 数据组装7830例(≤8192,95%completion)、Muon移植10项标记校验、6 cell ast.parse 语法校验、Kaggle-safe(Modal guard/上传Modal-only/纯离线装包)。
- 测试假设: huikang更好数据 + Muon(验证+0.01) = 0.86? 零数据风险。

**km override 暂不纳入(我的建议)**: 风险——kimberleyduran 预tokenized 数据集**权限被拒**, 只能自 tokenize CSV = **v29 翻车机制**; 且低 EV(crypt swap 全员≤0、huikang称public过拟合、Taha称不可学)。Muon-only 更干净高EV。若 Muon-only 停在0.85, 再上经严格验证的 swap(优先 binary-fmt bit, 非km)作 Run B。

**待用户**: ① km 是否纳入此 run(倾向否); ② push GPU 绿灯(~7h, GPU周一重置)。push 前我会最后通读一遍。

**关键文件**: kernels/v36-run1-huikang-base/{v36-run1-huikang-base.ipynb, kernel-metadata.json(id=exerzitien/v36-run1-huikang-base, sources=huikang快照+mayukh18+llkh0a+comp+model)}。

---

## 15. V36 min-logprob run 已构建+三重验证 (2026-05-29) — huikang base + Muon + min-logprob 2-epoch

**kernels/v36-run1-huikang-base/v36-run1-huikang-base.ipynb** 完成。= 冠军 huikang 获奖法(min-logprob 2-epoch) **叠加** Muon。
- **配置**: huikang 预tokenized corpus(7830) + Muon(lr2e-3) + CCE + MoE-tie + 9模块 + batch32/micro4 + RESET_WEIGHTS=True。
- **2-epoch min-logprob** (faithful huikang): ep0 普通 completion-mask CE + 记录每 token ref-NLL(按样例 idx 对齐); ep1 权重 = mask × min(1, ref-NLL/0.01)(branch_logprob=0.01)。
- **🛡️ epoch-0 兜底**: ep0 结束先存 Muon-only submission.zip; ep1 完成后 post-loop save 覆盖成 min-logprob 结果。即使 ep1 超 12h 被砍也有 Muon-only 产物(=折叠了原 Run A)。
- **验证**: ① min-logprob 加权+对齐逻辑纯python验证 PASS; ② 6 cell ast.parse 全过; ③ 16 项标记全 PASS; ④ loop→print 边界肉眼确认。(注: 第一版有孤立')'被 AST gate 挡下, 已从 ref-085 干净重建修复。)
- **预期**: ~488 步(244×2), ~8-9h, 兜底防超时。冲 0.86-0.87(冠军 +0.02 法 + Muon +0.01)。
- **待**: 用户 GPU push 绿灯(quota 周一~6/1 重置, ~8-9h 是大块)。push 前已完成最终通读。
- 关键风险残留(仅GPU可验): Muon×min-logprob 是否真叠加增益; 2-epoch 是否塞进12h(有兜底); base-on-Unsloth 复现(huikang 背书+预tokenized 已大幅降risk)。

---

## 16. 🚀 V36 min-logprob run 已 PUSHED + RUNNING (2026-05-29) — 本周最后一搏

**状态**: kernel `exerzitien/v36-run1-huikang-base` version 1 **RUNNING** (Kaggle GPU, ~8-9h)。
链接: https://www.kaggle.com/code/exerzitien/v36-run1-huikang-base

**配方** = huikang base 数据(预tokenized,7830) + **Muon(2e-3)** + CCE + MoE-tie + 9模块(q/k/v/o/up/down/in/out/lm_head) + batch32/micro4 + **2-epoch min-logprob**:
- ep0: 普通 completion-mask CE + 记录每 token ref-NLL(按样例 idx 对齐)
- ep1: 权重 = mask × min(1, ref-NLL / 0.01)  (huikang branch_logprob=0.01)
- **epoch-0 兜底**: ep0 结束先存 Muon-only submission.zip(地板); ep1 完成覆盖成 min-logprob 最终。超时也不浪费。
= 冠军 huikang 获奖法(min-logprob 2ep) × pkuszboi/Bobber 的 Muon。冲 0.86-0.87。

**🔑 推送鉴权(新窗口照做)**: kaggle.json key(a1c82)已失效 → 用 env 覆盖:
`KAGGLE_USERNAME=exerzitien KAGGLE_KEY=$(tr -d '[:space:]' < ~/.kaggle/access_token) kaggle kernels push -p kernels/v36-run1-huikang-base/`
(998 KGAT token 对 CLI v2.1.2 有效; MCP 用同一 token。)

### 跑完→取结果→提交→评测 (步骤)
1. 查状态: MCP `get_notebook_session_status(userName=exerzitien, kernelSlug=v36-run1-huikang-base)` (RUNNING/COMPLETE/ERROR)。
2. 取日志+产物: MCP `list_notebook_session_output` + `download_notebook_output(ownerSlug=exerzitien, kernelSlug=v36-run1-huikang-base, filePath=submission.zip / 或日志)`; 签名URL当步立刻curl。kernel 日志含每步 `e0/e1 step .. loss:mean` + "SAFETY: epoch-0 ... saved" 行。
3. 提 LB: 代码竞赛→Kaggle 网页 Submit 页选该 kernel 版本(submission.zip)。
4. 评测: 跑 `exerzitien/nemotron-adapter-eval`(挂该 adapter) 拿全9类错率, **存到本地 eval_results/v36/**(见 [[eval-save-per-class-local]])。

### 健康检查(头~30-60min, 早止损保配额)
看: ① loss 像 V34 正常下降; ② grad_norm 有限; ③ 无 Muon "non-matrix param" 报错; ④ 无 OOM; ⑤ 单步 wall-time × 488 步 < 12h。任一异常→cancel。

### 结果决策树
- **≥0.86 突破**: → Run B = GRPO(ke10086路径) 或 binary-fmt bit swap(MAJ0RT0M+huikang格式) 冲 0.87。
- **~0.85 (地板守住,min-logprob没叠上)**: Run B 调 branch_logprob(试 0.05/0.1, 更激进下调) 或换 binary-fmt bit(JK-Piece 点名的真杠杆, 真演绎可学)。不退步、没浪费。
- **<0.85**: epoch-0 兜底 submission 仍可提; 查 huikang-base-on-Unsloth 复现是否有问题; 用 epoch0-eval 诊断。

### 残留风险(仅GPU可验, 已尽力降低)
Muon×min-logprob 是否真叠加(可能冗余→落地板); 2-epoch 是否塞进12h(兜底覆盖); branch_logprob=0.01 是 huikang 文档示例值非确认获奖值; huikang-base 首次复现(huikang背书+预tokenized 已大幅降 v29 风险)。

### 本会话(2026-05-29)完整弧线小结
从"接受0.85"被用户顶回 → 全网/本地深挖 → 证伪 cryptarithm(冠军 8.2% crypt 仍夺冠) → 锁定真杠杆=huikang数据+训练法 → 核实 Bobber 0.86 真实(私有swap无法照搬,base可复现~0.85) → 找到 v29 真因(re-tokenization) → 冠军分类表证 min-logprob 才是 +0.02 → 构建 huikang+Muon+min-logprob 2ep run, 多重 gate 验证(含修掉一个孤立')') → 终审清 → 推送 RUNNING。
