## CA210 物理层（P0）架构 · 极简版（等价元胞 · 树结构）

本文件仅定义“物理层”最小实现与接口，作为后续各层的稳定底座。其目标是：可逆（信息守恒）、可替换规则（默认 Rule110）、实现简单（数组/向量化友好）。

### 1. 等价元胞与结构
- 只有一个类型 `Cell`，统一接口；不区分 Universe/Cell。
- 通过 `make_ring([...])` 将若干子元胞串成环，根节点既是“最大元胞”。
- 叶子元胞持有最小物理状态；复合元胞可以再包含子环，实现无限层嵌套。复合节点自身只负责调度，不落盘额外状态。

### 2. 叶子状态（最小）
- 每个叶子仅维护二阶寄存器：x_prev, x_cur ∈ {0,1}。
- 不引入任何“共识/历史/端口/事件”位（这些属于更高层）。

### 3. 局部规则（Rule，共享）
- 接口（伪代码）：
  ```
  interface Rule { fn local(l:int, c:int, r:int) -> int }
  ```
- 所有元胞共享同一规则实例；默认规则为 Rule110（可热插拔替换）。

### 4. 可逆推进（XOR 提升）
```text
x_next[i] = x_prev[i] XOR Rule.local(x_cur[i-1], x_cur[i], x_cur[i+1])
(x_prev, x_cur) <- (x_cur, x_next)
```
- 边界条件：环（索引按 N 取模）。
- 该提升为双射：存在成对逆过程（见 §4）。

### 5. 接口（P0 · OO 版 · 统一 Cell）
```text
class Cell:
  children: Optional[List[Cell]]  # None 表示叶子
  x_prev: Optional[int]
  x_cur: Optional[int]

  def step(rule: Rule):
    if children is None:
      raise RuntimeError("leaf must live inside a ring")
    _step_ring(children, rule)

  def step_inverse(rule: Rule):
    if children is None:
      raise RuntimeError("leaf must live inside a ring")
    _step_inverse_ring(children, rule)
```
- `_step_ring` / `_step_inverse_ring` 两阶段屏障：
  1. **计算阶段**：遍历整个环，对每个元胞读取相邻 `x_cur` 与自身 `x_prev`，计算下一拍，暂存于 `_stage_next`（逆向时则暂存 `_stage_prev`）。
  2. **提交阶段**：统一交换寄存器 `(x_prev, x_cur)` 并清空暂存，确保同步更新。
- 复合元胞的递归策略：
  - 正向 `step`：先更新当前环，再对每个含子环的元胞递归调用 `_step_ring`；
  - 逆向 `step_inverse`：先递归撤销子环，再恢复当前环，保证与正向严格对称。
- 约束：仅叶子写入 `x_prev/x_cur`；组合节点不落盘其他状态。

辅助构造函数：
- `make_leaf(x_prev, x_cur)` 创建叶子；
- `make_node(x_prev, x_cur, children)` 创建拥有内部环的复合元胞；
- `make_ring(list_of_cells)` 创建根容器（不参与演化，仅调度 `children`）。

### 6. 测试与不变式
- 可逆性：`state == step_inverse(rule, step(rule, state))`。
- 环移一致性：环移初态后的推进，与推进后再环移等价（可选）。
- 性能建议：使用位数组/NumPy 向量化（实现阶段可权衡）。

### 7. 后续层级（仅做指引，不在 P0 范围）
- Agency（最小历史 h=(eta,u)）：在物理层之上增加提案/守门/提交，但不破坏可逆性。
- 观察层（只读）：时间序列、三分解、选窗，均不写回物理层。
- 嵌套（Nested）：通过分组聚合（parity/majority）定义宏层；默认无反馈。
