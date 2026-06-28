# Supervisor Configuration Guide

本文档详细说明 `supervisor.yaml`（默认配置）与 `ds_supervisor.yaml`（DeepSeek 示例配置）中每一个参数的含义、取值范围以及如何编写新的配置文件。配置由 `SupervisorConfig` 数据类（`config.py`）加载校验。

---

## 目录

1. [快速开始](#快速开始)
2. [周期控制参数](#周期控制参数)
3. [修补护栏参数](#修补护栏参数)
4. [早停机制参数](#早停机制参数)
5. [数据采集参数](#数据采集参数)
6. [LLM Provider 参数](#llm-provider-参数)
7. [目标与 Schema 路径](#目标与-schema-路径)
8. [IPC 共享内存发布](#ipc-共享内存发布)
9. [完整参数速查表](#完整参数速查表)
10. [如何编写新的 Supervisor 配置](#如何编写新的-supervisor-配置)
11. [如何指定配置使用](#如何指定配置使用)
12. [部署场景示例](#部署场景示例)

---

## 快速开始

最简单的启动方式（使用默认配置）：

```bash
python train.py Unitree-Go2-Flat --supervisor
```

此时 Supervisor 使用 `class_free_guide/supervisor/config/supervisor.yaml` 作为默认配置。使用自定义配置文件：

```bash
python train.py Unitree-Go2-Flat --supervisor --supervisor_config path/to/my_supervisor.yaml
```

---

## 周期控制参数

控制 Supervisor 主循环的触发频率和等待时机。

### `interval_min`

| 字段 | 值 |
|------|-----|
| 类型 | `float` |
| 默认值 | `15.0` |
| 单位 | 分钟 |
| 对应代码 | `SupervisorConfig.interval_min` |

两次分析周期之间的最小间隔。Supervisor 在一个周期结束后等待 `interval_min` 分钟再启动下一个周期。该值决定了 LLM 介入调整奖励权重的频率。

**建议范围**：`5.0` ~ `60.0`。短间隔适合快速调试，长间隔适合长时间稳定训练。

```yaml
interval_min: 15.0   # 每 15 分钟分析一次
```

### `cooldown_iters`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `200` |
| 单位 | training iterations |
| 对应代码 | `SupervisorConfig.cooldown_iters` |

两次成功应用修补之间的最小迭代间隔。如果上一次修补在迭代 `N` 生效，则直到 `N + cooldown_iters` 之前，所有新修补都会被拒绝。防止频繁修改导致训练不稳定。

```yaml
cooldown_iters: 200   # 两次修补至少间隔 200 个迭代
```

### `warmup_iters`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `200` |
| 单位 | training iterations |
| 对应代码 | `SupervisorConfig.warmup_iters` |

预热期：在训练迭代数达到 `warmup_iters` 之前，Supervisor 不会进行任何分析。避免在训练初期数据不稳定的情况下做出错误判断。

```yaml
warmup_iters: 200   # 训练 200 轮后再开始分析
```

---

## 修补护栏参数

防止 LLM 输出过于激进的修改，保障训练安全。

### `max_patch_fields`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `3` |
| 对应代码 | `SupervisorConfig.max_patch_fields` |

每个周期最多允许修改的奖励项数量。如果 LLM 返回的修补包含超过此数量的字段，整个修补将被拒绝。

```yaml
max_patch_fields: 3   # 每周期最多修改 3 个奖励项
```

### `max_rel_change`

| 字段 | 值 |
|------|-----|
| 类型 | `float` |
| 默认值 | `0.30` |
| 范围 | `(0, 1]` |
| 对应代码 | `SupervisorConfig.max_rel_change` |

每个奖励项的单步相对变化上限。计算公式为 `|new - old| / max(|old|, 1e-6)`，超过此比例的变化将被拒绝。例如 `max_rel_change=0.30` 意味着一个权重为 `1.0` 的项最多只能被修改为 `[0.7, 1.3]`。

```yaml
max_rel_change: 0.30   # 每周期每个项的变化不超过 ±30%
```

### `max_consecutive_rollbacks`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `2` |
| 对应代码 | `SupervisorConfig.max_consecutive_rollbacks` |

同一个奖励项连续触发回滚的次数上限。达到此上限后，该项将被加入黑名单（`Guardrails.blacklist`），后续所有涉及该项的修补将自动被拒绝。

```yaml
max_consecutive_rollbacks: 2   # 同一项连续回滚 2 次后拉黑
```

---

## 早停机制参数

Supervisor 可以在训练达到满意水平时自动停止训练。

### `early_stopping`

| 字段 | 值 |
|------|-----|
| 类型 | `bool` |
| 默认值 | `true` |
| 对应代码 | `SupervisorConfig.early_stopping` |

是否启用早停机制。设为 `false` 则完全禁用。

```yaml
early_stopping: true   # 启用早停
```

### `pass_score`

| 字段 | 值 |
|------|-----|
| 类型 | `float` |
| 默认值 | `75.0` |
| 范围 | `0` ~ `100` |
| 对应代码 | `SupervisorConfig.pass_score` |

LLM 在诊断阶段给出的 0-100 评分阈值。当 LLM 评分 ≥ `pass_score` **且** 当前迭代 ≥ `min_training_iters` 时，触发早停。

```yaml
pass_score: 75.0   # 评分达到 75 分即可停止
```

### `min_training_iters`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `1000` |
| 单位 | training iterations |
| 对应代码 | `SupervisorConfig.min_training_iters` |

早停的最少训练迭代数。即使 LLM 给出高分，也必须训练满此迭代数才能停止。防止训练初期数据稀疏导致误判。

```yaml
min_training_iters: 10000   # 至少训练 10000 轮才能早停
```

**早停触发流程**（代码见 `supervisor.py` 的 `_check_completion` 方法）：

1. LLM 诊断时给出 `score`（整数，0-100）
2. 检查 `early_stopping` 开关 → `score >= pass_score` → `current_iter >= min_training_iters`
3. 全部满足后：写入 `supervisor/EARLY_STOP` 标记文件 + 设置 `stopping_event`
4. Runner 中的 `early_stop_event` 检测到信号后会跳出训练循环

---

## 数据采集参数

控制每个周期向 LLM 发送的观测数据量和视频帧数。

### `metric_window`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `200` |
| 单位 | TensorBoard 数据点 |
| 对应代码 | `SupervisorConfig.metric_window` |

TensorBoard 指标的时间窗口大小，即取最近多少个数据点用于分析。

```yaml
metric_window: 200   # 取最近 200 个 TB 数据点
```

### `metric_downsample`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `50` |
| 对应代码 | `SupervisorConfig.metric_downsample` |

降采样后的数据点数量。原始 `metric_window` 个点经过降采样后保留约 `metric_downsample` 个点，以减少发送给 LLM 的数据量。

```yaml
metric_downsample: 50   # 最终保留约 50 个数据点给 LLM
```

### `clips_per_cycle`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `2` |
| 对应代码 | `SupervisorConfig.clips_per_cycle` |

每个周期从最新视频中采样的片段数量。每个片段对应训练过程中的一个时间点。

```yaml
clips_per_cycle: 2   # 每人周期发送 2 段视频片段
```

### `video_frames_per_clip`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `6` |
| 对应代码 | `SupervisorConfig.video_frames_per_clip` |

每个视频片段中提取的帧数。这些帧以 base64 编码的 PNG 图片形式发送给 LLM（支持视觉能力的模型，如 Claude）。

```yaml
video_frames_per_clip: 6   # 每段视频提取 6 帧
```

**总帧数** = `clips_per_cycle × video_frames_per_clip`。以默认配置为例，每个周期发送 `2 × 6 = 12` 张图片给 LLM。

---

## LLM Provider 参数

控制 Supervisor 调用哪个 LLM 以及如何鉴权。

### `provider`

| 字段 | 值 |
|------|-----|
| 类型 | `str` |
| 默认值 | `"anthropic"` |
| 可选值 | `"anthropic"`, `"openai"`, `"openrouter"`, `"stub"` |
| 对应代码 | `SupervisorConfig.provider` |

LLM 提供商选择：

| 值 | 说明 | SDK | 对应 Client 类 |
|----|------|-----|---------------|
| `anthropic` | Anthropic Claude API | `anthropic` | `ClaudeClient` |
| `openai` | OpenAI API | `openai` | `OpenAIClient` |
| `openrouter` | 任何兼容 OpenAI `/v1/chat/completions` 协议的服务 | `openai` | `OpenRouterClient` |
| `stub` | 离线测试桩，返回空修补 | 无需 | `StubClient` |

```yaml
provider: anthropic
```

### `model`

| 字段 | 值 |
|------|-----|
| 类型 | `str` |
| 默认值 | `"claude-opus-4-7"` |
| 对应代码 | `SupervisorConfig.model` |

LLM 模型名称。格式取决于 provider：

- **Anthropic**：`claude-opus-4-7`, `claude-opus-4-8`, `claude-sonnet-4-6` 等
- **OpenAI**：`gpt-4o`, `gpt-4.1` 等
- **OpenRouter**：使用 `provider/model` 格式，如 `anthropic/claude-opus-4-7`、`deepseek/deepseek-v4-pro`
- **自部署 vLLM**：任意模型名

```yaml
model: claude-opus-4-7
```

### `max_tokens`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `2048` |
| 对应代码 | `SupervisorConfig.max_tokens` |

LLM 响应的最大 token 数。对应 OpenAI SDK 的 `max_tokens` / Anthropic SDK 的 `max_tokens` 参数。JSON 诊断 + 修补响应通常不需要很多 tokens，`2048` 是合理的默认值。

```yaml
max_tokens: 2048
```

### `temperature`

| 字段 | 值 |
|------|-----|
| 类型 | `float` |
| 默认值 | `0.3` |
| 对应代码 | `SupervisorConfig.temperature` |

LLM 采样温度。**建议不要修改**（默认配置中已标明 `# dont change`）。较低的 temperature 确保输出更确定、更安全。

```yaml
temperature: 0.3
```

### `api_base`

| 字段 | 值 |
|------|-----|
| 类型 | `str` or `None` |
| 默认值 | `null` |
| 对应代码 | `SupervisorConfig.api_base` |

API 端点基础 URL。仅对 `provider=openai` 和 `provider=openrouter` 有效；`provider=anthropic` 忽略此字段。

**典型值**：

| 服务 | api_base |
|------|----------|
| 默认 OpenAI | 不设（`null`），SDK 使用默认端点 |
| OpenRouter | `https://openrouter.ai/api/v1` |
| DeepInfra | `https://api.deepinfra.com/v1/openai` |
| Together | `https://api.together.xyz/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| 本地 vLLM | `http://localhost:8000/v1` |
| 自定义网关 | 由运维提供 |

```yaml
api_base: https://openrouter.ai/api/v1
```

### `api_key`

| 字段 | 值 |
|------|-----|
| 类型 | `str` or `None` |
| 默认值 | `null` |
| 对应代码 | `SupervisorConfig.api_key` |

直接写在配置文件中的 API Key。**优先级高于 `api_key_env`**。如果不为空，Supervisor 直接使用此值；否则回退到 `api_key_env` 指定的环境变量。

> ⚠️ **安全提醒**：不要在公共仓库中提交包含 `api_key` 的配置文件。推荐使用 `api_key_env` + 环境变量的方式。

```yaml
# api_key: "sk-..."       # 直接写入（不推荐）
api_key_env: ANTHROPIC_API_KEY   # 从环境变量读取
```

### `api_key_env`

| 字段 | 值 |
|------|-----|
| 类型 | `str` |
| 默认值 | `"ANTHROPIC_API_KEY"` |
| 对应代码 | `SupervisorConfig.api_key_env` |

存储 API Key 的环境变量名称。Supervisor 通过 `os.environ.get(api_key_env)` 获取实际密钥。

- **Anthropic**：默认 `ANTHROPIC_API_KEY`
- **OpenAI**：回退尝试 `OPENAI_API_KEY`
- **OpenRouter**：回退尝试 `OPENROUTER_API_KEY` → `OPENAI_API_KEY`

```yaml
api_key_env: ANTHROPIC_API_KEY
```

**API Key 解析优先级**（代码见 `llm_client.py` 的 `_resolve_api_key` 函数）：

1. `api_key`（配置文件内联密钥）
2. `api_key_env` 指定的环境变量
3. 各 client 的回退环境变量（如 `OPENAI_API_KEY`、`OPENROUTER_API_KEY`）

### `extra_headers`

| 字段 | 值 |
|------|-----|
| 类型 | `dict[str, str]` |
| 默认值 | `{}` |
| 对应代码 | `SupervisorConfig.extra_headers` |

发送给 LLM API 的额外 HTTP 头。仅对 `provider=openai` 和 `provider=openrouter` 生效（通过 OpenAI SDK 的 `default_headers` 注入）。常用于：

- **OpenRouter ranking 信号**：`HTTP-Referer`、`X-Title`
- **自定义鉴权代理**：额外的认证头

```yaml
extra_headers:
  HTTP-Referer: "https://github.com/your/repo"
  X-Title: "Go2 Locomotion Supervisor"
```

---

## 目标与 Schema 路径

### `objective_path`

| 字段 | 值 |
|------|-----|
| 类型 | `str` |
| 默认值 | `"class_free_guide/supervisor/objectives/stable_walk.json"` |
| 对应代码 | `SupervisorConfig.objective_path` |

训练目标 JSON 文件的路径。支持三种格式：

1. **相对/绝对文件路径**（含 `/` 或以 `.json` 结尾）：如 `class_free_guide/supervisor/objectives/stable_walk.json`
2. **短名称**（不含 `/` 且不以 `.json` 结尾）：如 `fast_walk`，自动查找 `objectives/fast_walk.json`
3. **`None`**：使用内置默认目标

可通过 CLI `--objective` 参数覆盖：

```bash
python train.py Unitree-Go2-Flat --supervisor --objective class_free_guide/supervisor/objectives/fast_walk.json
```

```yaml
objective_path: class_free_guide/supervisor/objectives/stable_walk.json
```

### `schema_path`

| 字段 | 值 |
|------|-----|
| 类型 | `str` or `None` |
| 默认值 | `null` |
| 对应代码 | `SupervisorConfig.schema_path` |

自定义 reward schema YAML 文件路径。为 `null` 时自动回退到 `class_free_guide/supervisor/config/schema.yaml`。

```yaml
schema_path: null   # 使用默认 schema
# schema_path: path/to/my_schema.yaml  # 自定义 schema
```

---

## IPC 共享内存发布

Supervisor 可以通过 dzipc SHM（共享内存）向外发布状态，供外部监控工具订阅。

### `ipc_enabled`

| 字段 | 值 |
|------|-----|
| 类型 | `bool` |
| 默认值 | `true` |
| 对应代码 | `SupervisorConfig.ipc_enabled` |

是否启用共享内存状态发布。Runner 侧（如 `VelocityOnPolicyRunner`、`FpoOnPolicyRunner`）会创建对应的 SHM 订阅者来接收 Supervisor 状态。

```yaml
ipc_enabled: true
```

### `ipc_topic`

| 字段 | 值 |
|------|-----|
| 类型 | `str` |
| 默认值 | `"supervisor_status"` |
| 对应代码 | `SupervisorConfig.ipc_topic` |

SHM 发布的话题名称。发布者（Supervisor）和订阅者（Runner）必须使用相同的话题名。

```yaml
ipc_topic: supervisor_status
```

### `ipc_domain`

| 字段 | 值 |
|------|-----|
| 类型 | `int` |
| 默认值 | `1` |
| 对应代码 | `SupervisorConfig.ipc_domain` |

dzipc SHM 通信域 ID。同一台机器上运行多个训练任务时，需使用不同的 domain 以避免干扰。

```yaml
ipc_domain: 1
```

---

## 完整参数速查表

| 参数 | 类型 | 默认值 | 单位 / 范围 | 说明 |
|------|------|--------|-------------|------|
| `interval_min` | float | 15.0 | 分钟 | 分析周期间隔 |
| `cooldown_iters` | int | 200 | iterations | 两次修补最小间隔 |
| `warmup_iters` | int | 200 | iterations | 预热迭代数 |
| `max_patch_fields` | int | 3 | 1~N | 每周期最大修改字段数 |
| `max_rel_change` | float | 0.30 | (0, 1] | 单步相对变化上限 |
| `max_consecutive_rollbacks` | int | 2 | 1~N | 触发黑名单的回滚次数 |
| `early_stopping` | bool | true | true/false | 是否启用早停 |
| `pass_score` | float | 75.0 | 0~100 | 早停评分阈值 |
| `min_training_iters` | int | 1000 | iterations | 早停最小训练迭代 |
| `metric_window` | int | 200 | data points | TB 数据窗口大小 |
| `metric_downsample` | int | 50 | data points | 降采样后数据点数 |
| `clips_per_cycle` | int | 2 | 1~N | 每周期视频片段数 |
| `video_frames_per_clip` | int | 6 | 1~N | 每片段提取帧数 |
| `provider` | str | "anthropic" | 见上文枚举 | LLM 提供商 |
| `model` | str | "claude-opus-4-7" | — | 模型名称 |
| `max_tokens` | int | 2048 | — | 最大输出 token 数 |
| `temperature` | float | 0.3 | (0, 1] | 采样温度 |
| `api_base` | str/null | null | URL | API 端点 URL |
| `api_key` | str/null | null | — | 内联 API Key |
| `api_key_env` | str | "ANTHROPIC_API_KEY" | 环境变量名 | API Key 环境变量 |
| `extra_headers` | dict | {} | — | 额外 HTTP 头 |
| `objective_path` | str | 见上文 | 文件路径/短名 | 训练目标 JSON |
| `schema_path` | str/null | null | 文件路径 | 自定义 Schema |
| `ipc_enabled` | bool | true | true/false | SHM 发布开关 |
| `ipc_topic` | str | "supervisor_status" | — | SHM 话题名 |
| `ipc_domain` | int | 1 | — | SHM 域 ID |

---

## 如何编写新的 Supervisor 配置

### 步骤 1：复制模板

```bash
cp class_free_guide/supervisor/config/supervisor.yaml \
   class_free_guide/supervisor/config/my_supervisor.yaml
```

### 步骤 2：修改参数

根据需求调整关键参数。以下是一些典型场景的配置模板：

#### 场景 A：快速调试（激进干预）

```yaml
interval_min: 5.0            # 更频繁的周期
cooldown_iters: 100          # 更短的冷却期
warmup_iters: 100            # 更短的预热期
max_patch_fields: 5          # 允许更大幅度的修改
max_rel_change: 0.50          # 允许更大的单步变化
early_stopping: false         # 禁用手停（调试时可能不想早停）
```

#### 场景 B：保守生产训练

```yaml
interval_min: 30.0           # 更长的周期间隔
cooldown_iters: 500          # 更长的冷却期
warmup_iters: 1000           # 更长的预热期
max_patch_fields: 2          # 更保守的修改量
max_rel_change: 0.15          # 更小的单步变化
max_consecutive_rollbacks: 2
early_stopping: true
pass_score: 85.0
min_training_iters: 50000
```

#### 场景 C：使用 DeepSeek / 自定义网关

参考 `ds_supervisor.yaml`：

```yaml
provider: openrouter
model: deepseek/deepseek-v4-pro        # 或你使用的模型名
api_base: https://aiapi.lejurobot.com/v1   # 你的网关地址
api_key: sk-your-key-here              # 或使用 api_key_env
max_tokens: 4096                        # DeepSeek 支持更大的输出
```

### 步骤 3：验证配置

配置文件会由 `SupervisorConfig.load(path)` 在启动时加载。YAML 中的未知字段会被收集到 `cfg.extra` 字典中，不会报错。所有已知字段会与 dataclass 默认值合并（YAML 中提供的值会覆盖默认值）。

---

## 如何指定配置使用

有三种方式指定 Supervisor 使用哪个配置文件：

### 方式 1：CLI 参数（推荐）

```bash
python train.py Unitree-Go2-Flat --supervisor \
    --supervisor_config class_free_guide/supervisor/config/my_supervisor.yaml
```

```bash
python train_fpo.py Unitree-Go2-Flat-FPO --supervisor \
    --supervisor_config class_free_guide/supervisor/config/ds_supervisor.yaml
```

CLI 参数 `--supervisor_config` 覆盖 `train.py`/`train_fpo.py` 中的默认路径。

### 方式 2：同时覆盖目标文件

```bash
python train.py Unitree-Go2-Flat --supervisor \
    --supervisor_config configs/prod_supervisor.yaml \
    --objective class_free_guide/supervisor/objectives/fast_walk.json
```

> `--objective` CLI 参数会覆盖 `supervisor.yaml` 中 `objective_path` 的值。

### 方式 3：环境变量 + 配置文件

在 YAML 配置中使用 `api_key_env` 管理密钥：

```bash
# 设置环境变量
export MY_CUSTOM_API_KEY="sk-..."

# 启动训练
python train.py Unitree-Go2-Flat --supervisor \
    --supervisor_config class_free_guide/supervisor/config/ds_supervisor.yaml
```

对应的 `ds_supervisor.yaml` 中写 `api_key_env: MY_CUSTOM_API_KEY`。

---

## 部署场景示例

### 场景 1：Anthropic Claude + 本地训练

```yaml
provider: anthropic
model: claude-opus-4-7
max_tokens: 2048
temperature: 0.3
api_base: null
api_key: null
api_key_env: ANTHROPIC_API_KEY
```

启动：

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python train.py Unitree-Go2-Flat --supervisor
```

### 场景 2：内网 DeepSeek 网关

```yaml
provider: openrouter
model: deepseek/deepseek-v4-pro
api_base: https://llm.internal.company.com/v1
api_key: null
api_key_env: INTERNAL_API_KEY
extra_headers:
  X-Auth-Token: "my-service-token"
```

### 场景 3：离线测试

```yaml
provider: stub
# 以下参数对 stub 无实际效果，但保留以通过配置加载
model: ""
max_tokens: 100
temperature: 0.0
early_stopping: false
```

Supervisor 将写入 audit 日志但不会实际发 LLM 请求，适合 CI/CD 中的集成测试。

---

## 代码定位参考

| 配置加载处 | 文件 |
|-----------|------|
| `SupervisorConfig` 数据类定义 | `config.py:13-78` |
| YAML 加载 + 已知/未知字段分离 | `config.py:67-78` |
| CLI 传入 Supervisor 构造链 | `train.py:266-267`, `train_fpo.py:308-309` |
| `sup_cfg` 在 Runner 中的使用 | `on_policy_runner.py:23-33`, `on_policy_fpo_runner.py:32-34` |
| LLM Client 根据 provider 构建 | `llm_client.py:46-56` |
| 早停触发逻辑 | `supervisor.py:_check_completion` |
| 护栏校验逻辑 | `guardrails.py:33-88` |
| 回滚规则解析 | `rollback.py:17-64` |
