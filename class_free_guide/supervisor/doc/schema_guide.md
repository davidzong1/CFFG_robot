# Reward Schema 配置指南

本文档详细说明 `schema.yaml` 的结构、每个 reward term 参数的含义、如何为新的机器人/任务编写 Schema，以及 Schema 在 Supervisor 工作流中的作用。

---

## 目录

1. [Schema 概述](#schema-概述)
2. [参数详解](#参数详解)
3. [现有 Term 说明（Go2 速度跟踪任务）](#现有-term-说明go2-速度跟踪任务)
4. [Schema 在 Supervisor 中的角色](#schema-在-supervisor-中的角色)
5. [如何为新任务编写 Schema](#如何为新任务编写-schema)
6. [校验规则详解](#校验规则详解)
7. [完整代码定位参考](#完整代码定位参考)

---

## Schema 概述

**Schema** 定义了 Supervisor 可以修改的奖励项（reward terms）及其合法范围。它是 LLM 输出修补（patch）的硬约束边界——任何超出 `[min, max]` 范围的修补都会被 `Guardrails` 拒绝。

### 关键文件

| 文件 | 说明 |
|------|------|
| `config/schema.yaml` | 默认 Schema 文件 |
| `schema.py` | `RewardSchema` 数据类（加载 + 验证） |
| `config.py` | `RewardBound` 数据类（单 term 约束） |
| `guardrails.py` | 修补护栏（引用 Schema 进行校验） |
| `config/supervisor.yaml` | `schema_path` 字段指定使用的 Schema 文件 |

### 数据结构

```yaml
# Schema YAML 顶级结构
terms:                          # 必选 — 奖励项字典
  term_name_1:                  # 奖励项名称（需与 RewardManager 中的名称一致）
    default: 1.0                # 默认权重（任务启动时的原始值）
    min: 0.1                    # 权重下限
    max: 5.0                    # 权重上限
    description: "..."          # 该项的含义说明（注入 LLM 提示）
  term_name_2:
    ...
```

```python
# 对应的 Python 数据类
@dataclass
class RewardBound:
    min: float          # 权重下限
    max: float          # 权重上限
    default: float      # 默认值
    description: str    # 说明文本

@dataclass
class RewardSchema:
    bounds: dict[str, RewardBound]  # term_name → RewardBound
```

---

## 参数详解

### `terms`

| 字段 | 值 |
|------|-----|
| 类型 | `dict[str, dict]` |
| 必选 | 是 |

Schema 的唯一顶级字段。字典的 key 是奖励项名称（必须与环境的 `RewardManager` 中注册的名称一致），value 是该项的约束配置。

```yaml
terms:
  track_linear_velocity:
    ...
  body_orientation_l2:
    ...
```

### `terms.<name>.default`

| 字段 | 值 |
|------|-----|
| 类型 | `float` |
| 必选 | 是 |
| 对应代码 | `RewardBound.default` |

该奖励项的默认权重值。在 LLM 提示中作为参考值发送，告知 LLM 当前项的"正常"水平。回滚时也会用此值作为参考。

**建议**：设为任务启动时的原始权重（即 `task_cfg.py` 中定义的初始值）。

```yaml
default: 1.0      # 跟踪线性速度的默认权重
default: -1.0      # body_orientation_l2 是惩罚项，默认为负
```

### `terms.<name>.min`

| 字段 | 值 |
|------|-----|
| 类型 | `float` |
| 必选 | 是 |
| 对应代码 | `RewardBound.min` |

该奖励项允许的最小权重值。任何低于此值的修补都会被拒绝。

**制定原则**：

- **正向奖励**（default > 0）：`min ≈ 0.1 × default`，通常不为 0
- **惩罚项**（default < 0）：`min ≈ 5 × default`（更负），即放大了惩罚的幅度

```yaml
track_linear_velocity:
  default: 1.0
  min: 0.1        # 最多降到默认值的 0.1 倍
  max: 5.0

body_orientation_l2:
  default: -1.0
  min: -5.0        # 惩罚项 min 更小（允许更大的惩罚力度）
  max: 0.0
```

### `terms.<name>.max`

| 字段 | 值 |
|------|-----|
| 类型 | `float` |
| 必选 | 是 |
| 对应代码 | `RewardBound.max` |

该奖励项允许的最大权重值。

**制定原则**：

- **正向奖励**：`max ≈ 5 × default`
- **惩罚项**：`max = 0`（不让惩罚变正向）

```yaml
track_linear_velocity:
  default: 1.0
  max: 5.0         # 最多升到默认值的 5 倍

body_orientation_l2:
  default: -1.0
  max: 0.0         # 惩罚项最多为 0（即取消惩罚）
```

### `terms.<name>.description`

| 字段 | 值 |
|------|-----|
| 类型 | `str` |
| 必选 | 否（但不写会影响 LLM 判断质量） |
| 对应代码 | `RewardBound.description` |

该奖励项的文字说明。此字段会被打包进 LLM 提示中的 schema 摘要，帮助 LLM 理解每个项的含义和作用。

```yaml
description: "Linear velocity tracking reward. Primary task signal."
description: "Penalty for body tilt away from upright."
```

---

## 现有 Term 说明（Go2 速度跟踪任务）

下表列出了 `schema.yaml` 中定义的 Go2 速度跟踪任务所有奖励项及其含义：

| Term 名称 | 默认值 | 范围 | 类别 | 说明 |
|-----------|--------|------|------|------|
| `track_linear_velocity` | 1.0 | [0.1, 5.0] | 正向 | 线性速度跟踪奖励，**主要任务信号** |
| `track_angular_velocity` | 1.0 | [0.1, 5.0] | 正向 | 角速度跟踪奖励 |
| `body_orientation_l2` | -1.0 | [-5.0, 0.0] | 惩罚 | 身体倾斜偏离直立姿势的惩罚 |
| `pose` | 1.0 | [0.0, 5.0] | 正向 | 可变姿态奖励（取决于站立/行走/跑步状态） |
| `body_ang_vel` | -0.05 | [-0.5, 0.0] | 惩罚 | 身体角速度惩罚 |
| `angular_momentum` | -0.025 | [-0.25, 0.0] | 惩罚 | 身体角动量惩罚 |
| `is_terminated` | -20.0 | [-500.0, 0.0] | 惩罚 | 提前终止惩罚（摔倒/超时） |
| `joint_acc_l2` | -2.5e-07 | [-5e-06, 0.0] | 惩罚 | 关节加速度 L2 惩罚（动作平滑性） |
| `joint_pos_limits` | -10.0 | [-50.0, 0.0] | 惩罚 | 关节位置超限惩罚 |
| `action_rate_l2` | -0.05 | [-0.5, 0.0] | 惩罚 | 动作变化率惩罚（动作平滑性） |
| `foot_gait` | 0.5 | [0.0, 2.5] | 正向 | 匹配预设步态模式的奖励 |
| `foot_clearance` | -1.0 | [-5.0, 0.0] | 惩罚 | 摆动足高度不正确的惩罚 |
| `foot_slip` | -0.25 | [-2.0, 0.0] | 惩罚 | 接触时足部滑动惩罚 |
| `soft_landing` | -0.001 | [-0.1, 0.0] | 惩罚 | 着地时硬接触惩罚 |
| `stand_still` | -1.0 | [-5.0, 0.0] | 惩罚 | 零速指令时发生移动的惩罚 |

---

## Schema 在 Supervisor 中的角色

### 工作流概览

```
┌──────────────────────────────────────────────────────────┐
│                    Supervisor 周期                        │
│                                                          │
│  1. 采集数据 ──→ 2. LLM 诊断 ──→ 3. 早停检查              │
│                                     │                    │
│                              (未早停) ↓                   │
│                          4. LLM 提议 ──→ 5. Schema 校验   │
│                                              │           │
│                                     ┌── 拒绝 ─┼── 通过 ──→│
│                                     ↓         ↓          │
│                              记录 audit   6. 应用修补      │
│                              + 回滚检查       + 记录版本   │
└──────────────────────────────────────────────────────────┘
```

Schema 在步骤 5 中由 `Guardrails.evaluate()` → `RewardSchema.validate_patch()` 执行校验。

### 校验流程（`schema.py:38-72`）

`RewardSchema.validate_patch(patch, current_weights, max_rel_change)` 对 LLM 返回的 patch 逐项检查：

1. **已知性检查**：`name in self.bounds` —— 修补的项名必须在 schema 中定义
2. **数值性检查**：`float(new_val)` —— 值必须可转为数值
3. **范围检查**：`bound.contains(new_val_f)` —— 值必须在 `[min, max]` 内
4. **变化幅度检查**：`|new - old| / max(|old|, 1e-6) <= max_rel_change` —— 相对变化不超过限制

全部通过后返回 `(True, "ok", clamped_patch)`。

### Schema 注入 LLM 提示

每次 LLM 提议时，Supervisor 会发送当前允许的 bounds 摘要：

```python
# supervisor.py:_schema_summary
def _schema_summary(self) -> dict[str, dict[str, float]]:
    return {
        name: {"min": b.min, "max": b.max, "default": b.default}
        for name, b in self.schema.bounds.items()
        if name in self._known_terms
    }
```

提示模板指示 LLM：
> "Keep each new value inside `[min, max]` from the schema. Single-step relative change ≤ 30% per weight."

---

## 如何为新任务编写 Schema

### 步骤 1：确定奖励项列表

从环境的 `RewardManager` 中获取所有活跃的奖励项名称。例如在 Go2 速度跟踪任务中：

```python
env.unwrapped.reward_manager.active_terms
# → ['track_linear_velocity', 'track_angular_velocity', 'body_orientation_l2', ...]
```

### 步骤 2：确定每个项的初始值和合理范围

参考 `task_cfg.py` 中的定义：

```python
# 示例：从 task_cfg.py 获取默认权重
track_linear_velocity_weight = 1.0
body_orientation_l2_weight = -1.0
```

遵循范围制定原则：

| 项类型 | default 符号 | min 建议 | max 建议 |
|--------|-------------|---------|---------|
| 正向奖励 | > 0 | 0 ~ 0.5×default | 5×default |
| 惩罚项 | < 0 | 5×default | 0 |
| 零权重 | 0 | -5×|default| ∼ 5×|default| |

### 步骤 3：编写 YAML 文件

```yaml
# class_free_guide/supervisor/config/my_task_schema.yaml

terms:
  # 正向奖励 — 任务的核心信号
  my_task_reward:
    default: 1.0
    min: 0.1
    max: 5.0
    description: "Main task reward. The primary signal for success."

  # 惩罚项 — 防作弊 / 保持平滑
  my_regularization:
    default: -0.01
    min: -0.1
    max: 0.0
    description: "L2 penalty on joint torques for smooth motion."

  my_fall_penalty:
    default: -10.0
    min: -100.0
    max: 0.0
    description: "Penalty for falling or early termination."

  # ... 更多项
```

### 步骤 4：在 Supervisor 配置中指定 Schema

```yaml
# my_supervisor.yaml
schema_path: class_free_guide/supervisor/config/my_task_schema.yaml
# 或相对路径（相对于运行目录）
# schema_path: config/my_task_schema.yaml
```

或使用默认路径（设为 `null`），此时会加载 `class_free_guide/supervisor/config/schema.yaml`。

### 步骤 5：验证

```bash
# 检查文件可被正确加载
python -c "
from class_free_guide.supervisor.schema import RewardSchema
schema = RewardSchema.load('path/to/my_schema.yaml')
print('Terms:', list(schema.names()))
for name, bound in schema.bounds.items():
    print(f'  {name}: [{bound.min}, {bound.max}] default={bound.default}')
"
```

---

## 校验规则详解

### 完整校验链

```
LLM 返回 patch JSON
       │
       ▼
┌──────────────────────┐
│ Guardrails.evaluate  │  ← guardrails.py:33
│                      │
│ 1. killswitch 检查   │  检查 supervisor/PAUSE 文件
│ 2. cooldown 检查     │  iter - last_apply_iters >= cooldown_iters
│ 3. 结构检查          │  必须有 patch/rationale/expected_effect/rollback_if
│ 4. 非空检查          │  patch 不能为空
│ 5. 字段数检查         │  len(patch) <= max_patch_fields
│ 6. 黑名单检查         │  不能包含 blacklist 中的 term
│ 7. Schema 校验 ──────→ RewardSchema.validate_patch()
│                      │   ├── 已知性 — term 必须在 schema 中定义
│                      │   ├── 数值性 — 值必须可转为 float
│                      │   ├── 范围   — min <= val <= max
│                      │   └── 幅度   — |Δ|/|old| <= max_rel_change
└──────┬───────────────┘
       │
   ┌───┴───┐
   │ PASS  │  → 应用修补
   │ FAIL  │  → 写入 rejected audit + 回滚检查
   └───────┘
```

### 黑名单机制

当某个 term 连续触发 `max_consecutive_rollbacks` 次回滚后，`Guardrails.note_rollback()` 将其加入 `blacklist`。此后包含该 term 的任何修补都会被自动拒绝。

```python
# rollback.py:58-63
for name in last_record.patch.keys():
    n = self._consecutive.get(name, 0) + 1
    self._consecutive[name] = n
    if n >= self.guardrails.cfg.max_consecutive_rollbacks:
        self.guardrails.note_rollback([name])
```

### 回滚条件规则 DSL

LLM 在提案中指定的 `rollback_if` 字段使用简单 DSL：

```
<metric_tag> <op> <number>
```

支持的 op：`<`, `<=`, `>`, `>=`

示例：

- `"Train/mean_reward < 0.3"` — 平均奖励低于 0.3 时回滚
- `"Episode/mean_length <= 50"` — 平均 episode 长度 ≤ 50 时回滚
- `"body_orientation_l2 > -0.1"` — body orientation 惩罚太小（接近 0）时回滚

解析逻辑见 `rollback.py:17` 的 `_RULE_RE` 正则。

---

## 完整代码定位参考

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| `RewardBound` 数据类 | `config.py` | 81-94 | 单 term 的 min/max/default/description |
| `RewardSchema` 数据类 | `schema.py` | 14-33 | 完整 schema 的加载和内存表示 |
| `RewardSchema.load()` | `schema.py` | 20-33 | YAML 加载逻辑 |
| `RewardSchema.validate_patch()` | `schema.py` | 38-72 | 校验一个 patch 是否合法 |
| `Guardrails.evaluate()` | `guardrails.py` | 33-80 | 完整护栏校验链 |
| `Guardrails.note_rollback()` | `guardrails.py` | 85-87 | 将 term 加入黑名单 |
| `RollbackEvaluator.maybe_rollback()` | `rollback.py` | 32-64 | 回滚触发 + 黑名单计数 |
| `Supervisor._schema_summary()` | `supervisor.py` | 411-412 | 注入 LLM 提示的 schema 摘要 |
| `Supervisor.__init__()` | `supervisor.py` | 78-81 | schema 路径解析 |
| `schema_path` 配置 | `supervisor.yaml` | 75 | 自定义 schema 路径 |
