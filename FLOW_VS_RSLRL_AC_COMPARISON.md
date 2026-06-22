# Flow Matching ActorCritic vs. rsl_rl ActorCritic 对比分析

> 对比对象
> - **Flow-AC**：`ref_lib/fpo-control/isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/modules/actor_critic.py`
>   （基于 Flow Matching / CFM 的策略，配合 FPO 算法使用）
> - **rsl_rl-AC**：`rsl_rl/modules/actor_critic.py`
>   （rsl_rl 标准的高斯策略 MLP，与原版 PPO 配套）

两者继承自相同的 ETH/NVIDIA `nn.Module` 模板（`is_recurrent=False`、`reset`、`forward=NotImplementedError`、`evaluate(critic_obs)`、`load_state_dict` 等签名一致），但 **Actor 的概率模型从“高斯分布”被替换为“Flow Matching 的连续向量场”**，由此衍生出多处接口与张量形状上的不一致。下面按维度逐项对比。

---

## 1. 策略表示与采样机制

| 维度 | rsl_rl-AC（高斯策略） | Flow-AC（Flow Matching 策略） |
|---|---|---|
| 概率模型 | `Normal(mean, std)`，显式高斯分布 | 隐式分布：用 ODE 从 `x_1 ~ N(0, I)` 积分到 `x_0` |
| 采样过程 | 一次前向：`mean = actor(obs)`，再 `Normal.sample()` | 多步积分：`for i in range(sampling_steps): x_t += u(obs, t, x_t) * dt` |
| 网络输入 | `obs`（维度 `num_actor_obs`） | `[obs, timestep_embedding(t), x_t]`（维度 `num_actor_obs + timestep_embed_dim + num_actions`） |
| 噪声 / 探索 | 可学习参数 `std` / `log_std`（`nn.Parameter`） | 训练态从 `randn` 起步 + 推理态 `actor_scale * x_t` + 可选 `action_perturb_std * randn` 加性扰动；**无可学习的 std** |
| 评估时确定性 | `act_inference(obs) = actor(obs)`（直接取均值） | `act_inference` 提供 `eval_mode ∈ {"zero", "fixed_seed", "random"}` 三种初始噪声策略，再做 ODE 积分 |

> 推论：rsl_rl 的 actor 是“**一次前向得到动作的均值**”，而 Flow-AC 的 actor 是“**速度场 u**”，必须配合内部的积分循环才能输出动作。两者输出语义完全不同。

---

## 2. 对外 API 与 PPO 关键接口的差异

rsl_rl 原版 PPO 依赖以下 ActorCritic API：
- `act(obs)` → 采样动作
- `get_actions_log_prob(actions)` → 计算 log π(a|s)
- `update_distribution(obs)` / `action_mean` / `action_std` / `entropy`
- `act_inference(obs)`、`evaluate(critic_obs)`

| API | rsl_rl-AC | Flow-AC | 影响 |
|---|---|---|---|
| `update_distribution` | ✅ 维护 `self.distribution = Normal(...)` | ❌ 不存在 | Flow-AC 没有显式分布，所有“分布相关”API 都不能使用 |
| `get_actions_log_prob` | ✅ `Normal.log_prob(a).sum(-1)` | ❌ 不存在 | PPO 计算 ratio = `exp(logp_new - logp_old)` 无法直接套用，FPO 用 **CFM loss** 作为替代 |
| `action_mean` / `action_std` / `entropy` | ✅ property | ❌ 全部缺失 | 标准 PPO 中“熵正则”“KL 早停”都失去对应字段；Flow-AC 用 `action_perturb_std` 显式加噪近似熵 |
| `act(obs)` 行为 | 从 `Normal` 采样 | 从 ODE 积分得到 `actor_scale * x_t`，训练态再叠 `action_perturb_std` | 同名函数但语义不同 |
| `act_inference` 参数 | 仅 `obs` | `obs, eval_mode, eval_fixed_seed` | 推理可复现性可控 |
| `get_cfm_loss(obs, actions, eps, t, actor=None)` | ❌ | ✅ Flow-AC 独有 | FPO 训练通过该接口计算速度场 MSE，**替代** log-prob ratio |
| 外部依赖 | 直接使用 init kwargs | 接收一个 `FpoRslRlPpoActorCriticCfg` 配置对象 | 构造接口签名不兼容，rsl_rl 的 runner 无法直接传参 |

---

## 3. 网络结构与 Actor 输入维度

```text
rsl_rl-AC actor input dim  = num_actor_obs
Flow-AC   actor input dim  = num_actor_obs + timestep_embed_dim + num_actions
```

差异来自 Flow Matching 需要把 **当前时间步 t** 和 **当前噪声状态 x_t** 一起作为条件喂给速度网络：

- 时间编码：`_embed_timestep` 用 `2**k` 的频率基（不是常见 Transformer 的 `10000**k`），并 `[cos, sin]` 拼接成 `timestep_embed_dim` 维。
- Critic 部分完全相同：`mlp_input_dim_c = num_critic_obs`，结构是同一份 MLP（输出 1 维 value）。

此外 Flow-AC 额外提供：
- `actor_final_layer_weight_scale`：对最终层权重/偏置统一缩放（rsl_rl 没有）。
- `mlp_output_scale`：每次前向后对 velocity 做整体缩放。
- `actor_scale`：把 ODE 终点 `x_0` 缩放到真实动作空间。

这些缩放参数是 Flow Matching 训练稳定性需要的“**输出量级控制旋钮**”，rsl_rl-AC 仅靠 `init_noise_std` 一项就涵盖了。

---

## 4. 训练损失与训练态行为

| 维度 | rsl_rl-AC | Flow-AC |
|---|---|---|
| 主损失来源 | 由 PPO 在外部计算 `surrogate + value + entropy`，AC 不参与损失 | AC 内部 `get_cfm_loss` 计算 **conditional flow matching MSE**：`‖u_pred − (eps − scaled_a)‖²` |
| 训练态噪声 | 行为由 `Normal` 自动决定 | `act()` 中 `x_t = randn`（vs. 推理 `x_t = zeros`）；额外可叠 `action_perturb_std` 噪声 |
| `actor_scale` 反归一化 | 不涉及 | 训练时把外部 `actions` 除以 `actor_scale`，保证速度场在归一化空间学习 |
| 损失 reduction | 不适用 | `cfm_loss_reduction ∈ {"mean","sum","sqrt"}`，对最后一维做不同归约 |
| t 采样分布 | 不适用 | 配置 `cfm_loss_t_inverse_cdf_beta` 控制训练态 t 的逆 CDF 采样形状（外部生成 t 传入） |

---

## 5. 推理与性能优化

- **rsl_rl-AC**：推理 = 一次 `actor(obs)` 前向，无额外结构。
- **Flow-AC**：
  - 在 `__init__` 中对 `_integrate_flow` 调用 `torch.compile(mode="reduce-overhead")`，开启 CUDA Graph 重放，代码注释中说明可带来 3–9× 加速。
  - `_integrate_flow` 内部把时间嵌入手动内联、用静态 shape 写法，避免 `assert` 与 graph break。
  - `sampling_steps` 与 `training_sampling_steps` 可分别设置，允许“训练用少步 / 推理用多步”等不同配置。

---

## 6. 上下游兼容性影响（与 rsl_rl runner / PPO 配套）

`OnPolicyRunner` + 标准 PPO 直接使用 Flow-AC 会在以下位置失败：

1. **`PPO.act()` 流程**：rsl_rl 在 rollout 中需要存储 `actions_log_prob`、`action_mean`、`action_std`。Flow-AC 没有 `get_actions_log_prob` / `action_mean` / `action_std` / `entropy`，会 `AttributeError`。
2. **KL 早停 / `adaptive` 学习率调度**：依赖前后两次分布的 KL，Flow 策略没有解析 KL，需要替代方案。
3. **熵正则项**：`entropy_coef * entropy` 无值可用。
4. **policy ratio**：PPO 的 `surrogate_loss = ratio * advantages` 中 ratio 需要新旧 log-prob，Flow 策略需要换成 FPO 的“**CFM 损失加权 ratio**”或类似形式。

这也是为什么仓库里另起一份 `FlowPPO`（`class_free_guide/pineline/rl/mjlab/tasks/velocity/rsl_rl/alg/flow_ppo.py`）以及 `OnPolicyFlowRunner` 的根本原因——必须替换掉 rsl_rl 的标准 PPO 才能跑通 Flow-AC。

---

## 7. 总结：差异本质与对接建议

**本质差异**：rsl_rl-AC 是“**显式高斯策略 + 一次前向**”，Flow-AC 是“**隐式 ODE 策略 + 多步积分 + CFM 训练**”。两者只共享 Critic 结构和顶层模块骨架，Actor 在采样/训练/接口层面几乎是两套独立体系。

**对接 rsl_rl 的关键改造点**（按优先级）：

1. **替换 PPO 算法层**：直接复用 rsl_rl 自带 `PPO` 不可行，需要参考 `FlowPPO` 完成 `act` / `update` 中所有 log-prob、ratio、entropy、KL 相关替换。
2. **补齐 rollout 缓存字段**：若仍想使用 rsl_rl 的 `RolloutStorage`，需要将 `actions_log_prob` 等条目替换/旁路为 Flow 训练所需的 `(eps, t)` 等量。
3. **runner 适配**：`on_policy_flow_runner.py` 需要绕开 `OnPolicyRunner` 中调用 `action_mean / action_std / entropy` 的统计逻辑。
4. **入口签名对齐**：Flow-AC 的 `__init__` 强依赖 `FpoRslRlPpoActorCriticCfg`，若要被 rsl_rl 的 `class_to_dict + **kwargs` 风格构造，需要写一个适配层把 dict 拆成 cfg。
5. **熵 / KL 的替代量**：用 `action_perturb_std` 控制探索，KL 用 *动作空间* MSE 近似（FPO 论文常用做法）。

---

## 8. 关键代码定位

- Flow-AC actor 输入构造：`actor_critic.py:54`
- Flow ODE 积分：`actor_critic.py:224-269`（`_integrate_flow`）
- CFM 损失：`actor_critic.py:157-213`（`get_cfm_loss`）
- 推理多模式：`actor_critic.py:285-331`（`act_inference`）
- rsl_rl 高斯分布维护：`actor_critic.py:107-122`（`update_distribution` / `act`）
- rsl_rl log-prob & entropy：`actor_critic.py:103-105`，`actor_critic.py:124-125`
