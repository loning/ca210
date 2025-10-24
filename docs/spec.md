# RCCA-Min v2 程序规范文档（Program Specification · 最完整版）

> 极简而不简化 · 纯信息论 · 可逆/可检/可复现
> 适用对象：实现与评测团队、第三方复现实验者、期刊/审稿复核

---

## 0. 范围（Scope）

本规范定义 **RCCA-Min v2** 的参考实现与验收流程，覆盖：

1. **Law 层（最小核）**：二阶可逆物理核 + **非退化**单一提案 + 单一守门 + 最小历史；
2. **Obs 层（观察/评估）**：时间序列、三分解（alias/bern/trunc）、选窗变分、非渐近预算残差；
3. **Search（人择）**：线性鸽尾调度 + Pareto 前沿；
4. **Wave-CI**：时/空波像、干涉 vs 源间距、时间窗增益；
5. **Agency-CI**：自决占优（`I(η;H_int|P)` vs `I(η;P|H_int)`）；
6. **数据模式/接口/CLI** 与 **合规验收**（S0/S4/S6/S8/S11 + 波像 + Agency）。

> 全文不引入任何物理常数；所有阈值为**协议级**无量纲相对量。

---

## 1. 术语（Glossary）

* **Universe**：环格系统状态容器。
* **Law（最小核）**：提案/守门/提交/物理核。
* **Window**：仅用于观察层的窗权，不回写 Law。
* **三分解**：`R = alias + bern + trunc`。
* **Dovetail**：线性鸽尾（串行"并行"）公平推进。
* **Pareto 前沿**：多目标非支配解集（无加权）。
* **Agency**：自决占优互信息判据。
* **CI**：合规检查（S 系列 + Wave + Agency）。

---

## 2. 文件结构（此实现已生成，可直接下载）

```
/mnt/data/rcca_min_v2/
  core.py      # Law 层：最小核（非退化提案器）
  eval.py      # Obs：时间序列/三分解/选窗/预算
  search.py    # Dovetail + Pareto
  README.md    # 使用说明（简）
```

可直接下载：

* [core.py](sandbox:/mnt/data/rcca_min_v2/core.py) · [eval.py](sandbox:/mnt/data/rcca_min_v2/eval.py) · [search.py](sandbox:/mnt/data/rcca_min_v2/search.py) · [README.md](sandbox:/mnt/data/rcca_min_v2/README.md)

实验产物：

* [baseline_timeseries.csv](sandbox:/mnt/data/rcca_min_v2_baseline_timeseries.csv)
* [dovetail_all.csv](sandbox:/mnt/data/rcca_min_v2_dovetail_all.csv)
* [pareto.csv](sandbox:/mnt/data/rcca_min_v2_pareto.csv)

---

## 3. Law 层（最小核）规范

### 3.1 状态（每格）

* `x_prev, x_cur ∈ {0,1}`：物理寄存器（二阶可逆）
* `a ∈ {0,1}`：共识寄存器
* `B ∈ {0,1}`：许可预算位
* `h = (eta, u) ∈ {0,1}^2`：最小历史（若不启用"硬前缀"，`u≡0`）

> 禁止：持久化端口键、`p` 寄存器写盘、冗余历史位（k/ρ/dB）、外部随机/实数权。

### 3.2 二阶可逆物理核

$$
x_{t+1}(i)=x_{t-1}(i)\ \oplus\ f\big(x_t(i-1),x_t(i),x_t(i+1)\big),
$$

随后交换 $(x_{t-1},x_t)\mapsto(x_t,x_{t+1})$；`f` 取 Rule-110。

### 3.3 **非退化**提案与单守门

* 邻域奇偶：$\theta = x_L \oplus x_C \oplus x_R$
* 时间增量：$dX = x_{\rm prev}\oplus x_{\rm cur}$
* **提案（一次 XOR，无参数）**：
  $$
  \boxed{\ \widehat p = x_L \oplus x_C \oplus x_R \oplus dX \oplus B\ }
  $$
* **守门（单门）**：
  $$
  \mathrm{allowed1}= B \wedge (\widehat p_L \vee \widehat p \vee \widehat p_R \vee \theta)
  $$
* **提交**：$\eta=\mathrm{allowed1}\wedge \widehat p,\ a \mathrel{\oplus=} \eta,\ h=(\eta,u)$（`u` 仅硬前缀时为 1）

### 3.4 编排与不变式

* 编排：**提交 → 物理**；并发避让可用棋盘二步（二者择一）。
* 不变式：
  I1 可逆（`step(); step_inverse()` 恒等）；I2 非干预；I3 Trap-free；I4 最小历史；I5 全布尔。

---

### 3.5 分层架构（原型，逐层建立）

为支持“元胞嵌套”，本版加入无反馈的宏层包装：

- 分组：将微胞按固定窗口 `g` 连续分组，得到宏胞索引；
- 聚合：宏 `x` 与 `a` 由组内比特通过 `parity`（或 `majority`）得到；
- 推进：仅微层推进（保持可逆），宏层仅读取聚合态，不写回 Law（信息不丢失、不干预）。

参考实现：`src/layer.py`

API：
- `Layered(U, MacroLayer(group_size=4, mode='parity'))`
- `Layered.macro_snapshot() -> (X_macro, A_macro)`
- `Layered.step(t)` / `Layered.step_inverse(t)` / `Layered.inverse_check(t)`
- `macro_timeseries(N,T,group_size,mode)` → DataFrame(`name,t,H3_x,H_a,MI_xa`)

备注：后续可在保持全局双射的前提下，引入“上行许可/下行门控”闭环；当前阶段以“无反馈”保证可逆性与信息守恒。

## 4. Obs 层（观察/评估）规范

### 4.1 时间序列（最小指标）

* $S_{\rm phys}(t)=H_3(x_t)$（三块熵）
* $S_{\rm cons}(t)=-H(a_t)$（共识负熵）
* $S_{\rm holo}(t)=\frac{1}{N}\sum_i(\eta_t(i)+u_t(i))$（历史记账）
* $I_{x:a}(t)$（经验互信息）
* **残差** $R_t=\Delta S_{\rm phys}+\Delta S_{\rm cons}+\Delta S_{\rm holo}-\Delta I_{x:a}$

### 4.2 三分解（S4）

`R = alias + bern + trunc`；能量闭合 $|R|_2^2=\sum|·|_2^2$。
EM 伯努利层：最低 `B2`，稳健可加 `B4/B6`。
API：`decompose(R, frac=0.3, use_B4B6=True)`。

### 4.3 选窗变分（S6）

窗族：Blackman / Kaiser12 / Exp12（至少一类规范窗）。
目标：`J(h)=alias_share(R⊙h)+λ·pen(h)`（λ 协议参数）。
窗仅用于**观测**，**不得回写 Law**。
API：`choose_window(R, lam=0.1)`。

### 4.4 预算闭合（非渐近恒等式）

输出 `recon_err` 与残差统计 $\varepsilon_t$：
$$\Delta S_{\rm phys}+\Delta S_{\rm cons}+\Delta S_{\rm holo}-\Delta I_{x:a}=\varepsilon_t$$

---

## 5. Search（线性鸽尾 + 人择筛选）规范

### 5.1 候选描述（确定性）

`candidate = (c, r ∈ {2,3,4}, ptype ∈ {x_cur,x_prev,B}, Gray mask)`
API：`enum_candidate(n, N)`；`apply_local_perturb(U, cand)`。

### 5.2 Dovetail 调度

层数 `layers`、步长 `delta`、每层候选数 `num_cand`。
API：`dovetail(N, layers, delta, num_cand) -> DataFrame`，字段含 `layer,T,c,r,ptype,mask` 与指标。

### 5.3 Pareto 前沿（无加权）

最大化 `{mi_gain, h3_gain}`，最小化 `{alias_share, recon_err}`。
API：`pareto_front(df, keys_max, keys_min)`。

---

## 6. 数据模式（CSV / JSON）

### 6.1 时间序列 `*_baseline_timeseries.csv`

```
t,H3_x,H_a,MI_xa,R
```

### 6.2 Dovetail 结果 `*_dovetail_all.csv`

```
layer,T,cand_id,c,r,ptype,mask,alias_share,alias_win,win_name,recon_err,mi_gain,h3_gain
```

### 6.3 Pareto 表 `*_pareto.csv`

```
cand_id,c,r,ptype,mask,alias_share,alias_win,recon_err,mi_gain,h3_gain
```

> 所有值为**无量纲**的信息量/相对量；不包含任何具体常数。

---

## 7. 公共 API（语言无关 + Python 参考）

```python
# Law
from rcca_min_v2.core import make_universe  # Universe(N, mode)
# Obs
from rcca_min_v2.eval import timeseries, decompose, choose_window
# Search
from rcca_min_v2.search import dovetail, pareto_front, enum_candidate, apply_local_perturb
```

---

## 8. 合规验收（S 系列 + Wave + Agency）

### 8.1 S 系列（协议阈值，均为相对量）

* **S0**：环移不变（H、Neff、N2 的相对误差 ≤ 阈值）；
* **S4**：三分解能量闭合（`recon_err ≤ 阈值`）；
* **S6**：选窗使 `alias_share` 显著下降；
* **S8**：Δ² 对 alias 的相对能量降幅 ≥ 阈值；
* **S11**：随窗带宽/EM 阶数增大，谱—轨道差值**收敛**（占位验证或图表）。

### 8.2 Wave-CI（波像/干涉）

* 时间：峰值占比↑、谱熵↓，Δ² 湮灭增益>0；
* 空间/2D：稳定 `k` 峰与"色散脊"；
* 干涉：`interference_rel(d)` 曲线存在显著峰。

### 8.3 Agency-CI（自决占优）

* 多数候选（或 Pareto top-K）满足
  $$
  I(\eta;H_{\rm int}\mid P) > I(\eta;P\mid H_{\rm int}).
  $$
* 解释为"**局部自决**主导选择，端口仅限集"。

---

## 9. 运行规范与再现性

* **复现实验**：确定性枚举，无随机；若加入随机初态，必须记录 seed（只在 Obs 层使用）。
* **资源**：Law 每步 O(N)；三分解 O(T log T)（FFT）；Dovetail 总步长 ~ O(num_cand · layers²)。
* **安全**：Law 层不得暴露写回回调；Obs 层只读；Search 只做初态扰动与调度。
* **版本**：此规范稳定面为 Law API、Obs 字段、Search 产物列名；扩展以协议参数方式注入（向下兼容）。

---

## 10. 示例工作流（最小）

```python
# 1) 基线
U = make_universe(N=64, mode="deterministic")
df_ts = timeseries(U, T=128)                       # 保存CSV
R = df_ts[df_ts["t"]>=1]["R"].values.astype(float)
dec = decompose(R, frac=0.3, use_B4B6=True)        # alias/bern/trunc/recon_err
win = choose_window(R, lam=0.1)                    # best_window / best_alias

# 2) 搜索 + 人择
df_all = dovetail(N=64, layers=4, delta=48, num_cand=32)
df_last = df_all[df_all["layer"]==df_all["layer"].max()]
pareto = pareto_front(df_last,
    keys_max=["mi_gain","h3_gain"], keys_min=["alias_share","recon_err"])
```

---

## 11. 交付清单（Deliverables）

1. **程序**：`core.py`（最小核，非退化提案器）、`eval.py`（三分解/选窗/预算）、`search.py`（Dovetail + Pareto）、README。
2. **实验产物**：baseline、dovetail_all、pareto（CSV）；可选附加：频谱/干涉图、Wave-CI/Agency-CI 报表。
3. **报告**（可选）：合并 S + Wave + Agency 的 `full_ci_report.md`（一键 PASS/FAIL）。

---

## 12. 附：法则伪代码（RCCA-Min v2）

```python
# state per site: (x_prev, x_cur), a, B, h=(eta,u)

# propose (non-degenerate, minimal, parameter-free)
theta = xL ^ xC ^ xR
dX    = x_prev[i] ^ x_cur[i]
p_hat = xL ^ xC ^ xR ^ dX ^ B[i]

# guard (single)
allowed1 = B[i] & (p_hatL | p_hat | p_hatR | theta)

# commit
eta = allowed1 & p_hat
a[i] ^= eta
u    = 1 if (HARD_PREFIX and allowed1 == 0) else 0
h[i] = (eta, u)

# physical (2nd-order reversible lift)
x_next = x_prev[i] ^ f(xL, xC, xR)  # f = Rule-110
x_prev[i], x_cur[i] = x_cur[i], x_next
```

---

> 如需，我可以把 **Wave-CI** 与 **Agency-CI** 的 CLI 脚本也加上，输出统一的 `full_ci_report.md`（含图/表/阈值判定），并把它们加入 `README.md` 的使用示例中。
