# Flow Matching Behavior Cloning (BC) 预训练流程分析

## 概述

`pretrain_flow_bc.py` 是一个离线强化学习训练脚本，用于在 LeRobot 数据集上预训练 Flow Matching 策略模型。该脚本支持分布式训练（DDP）、检查点管理、Weights & Biases 日志记录，以及在 DexMimicGen 环境中的在线评估。

---

## 核心配置 (`TrainFlowBCConfig`)

### 数据集配置
- **dataset**: HuggingFace Hub 数据集 repo-id（必需）
- **max_num_episodes**: 最大加载的 episode 数（可选，用于内存优化或快速实验）

### 策略选择
- **policy**: 策略架构类型，目前仅支持 `"flowmatching"`

### 训练超参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `steps` | 3,000 | 总优化步数 |
| `batch_size` | 256 | 批次大小 |
| `learning_rate` | 1e-4 | Flow 模型学习率 |
| `lr_backbone` | 1e-5 | 视觉骨干网络学习率 |
| `weight_decay` | 1e-6 | 权重衰减 |
| `grad_clip_norm` | 10.0 | 梯度裁剪范数 |
| `gradient_accumulation_steps` | 1 | 梯度累积步数 |
| `num_workers` | 4 | 数据加载工作线程数 |

### 模型架构参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vision_backbone` | "resnet18" | 视觉骨干网络（支持 resnet18, clip 等） |
| `horizon` | 16 | 预测时间步长 |
| `n_action_steps` | 8 | 执行的动作步数 |
| `sampling_steps` | 10 | Flow 模型采样步数 |
| `ema_power` | 0.995 | EMA 衰减系数（0.0 表示禁用） |
| `network_architecture` | "mlp" | 网络架构：`"unet"` 或 `"mlp"` 或 `"residual_mlp"` |
| `mlp_dims` | [512, 512, 512] | MLP 隐层维度 |

### Flow Matching 特定参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `flow_network_output_param` | "u" | 输出参数：`"u"`（速度）或 `"x0"`（数据） |
| `cfm_loss_mode` | "u" | 损失模式：`"u"`、`"x0"` 或 `"eps"` |
| `transported_clip_value` | None | 预测值裁剪范围（None 表示无裁剪） |
| `cfm_loss_use_huber` | False | 是否使用 Huber 损失 |
| `cfm_loss_huber_delta` | 0.5 | Huber 损失 delta 参数 |

### 日志与检查点
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | None | 输出目录（None 则自动生成） |
| `log_freq` | 100 | 日志打印频率（步数） |
| `save_freq` | 1,000 | 检查点保存频率（步数） |

### Weights & Biases 配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `wandb_enable` | True | 启用 W&B 日志 |
| `wandb_project` | None | W&B 项目名称（必需） |
| `wandb_entity` | "far-wandb" | W&B 实体名称 |
| `experiment` | "train_flow_bc" | 实验名称 |

### 检查点恢复
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `resume_ckpt` | None | 本地检查点路径 |
| `resume_run_id` | None | W&B run ID（用于下载检查点） |
| `checkpoint_step` | "latest" | 加载的检查点步数：`"latest"`、`"best"` 或具体步数 |
| `resume_wandb_run` | False | 是否恢复 W&B run（推荐 False） |
| `load_ema` | False | 是否加载 EMA 权重 |

### 评估配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rollout_freq` | 100 | 评估频率（步数，None 表示禁用） |
| `eval_env` | None | 评估环境名称（DexMimicGen 环境） |
| `eval_num_envs` | 2 | 并行评估环境数 |
| `eval_num_episodes` | 10 | 每次评估的 episode 数 |
| `eval_camera_size` | 84 | 评估图像大小 |
| `debug` | False | 调试模式（使用同步环境） |
| `image_observation_keys` | None | 自定义图像观察键 |

---

## 训练流程详解

### 1. 初始化阶段

#### 1.1 DDP 设置
```
if is_ddp:
    - 初始化分布式进程组（NCCL 后端）
    - 设置本地 rank 和 device
    - 非主进程抑制日志输出
```

#### 1.2 随机种子与设备
- 设置随机种子（如果指定）
- 确定计算设备（CUDA 或 CPU）
- 创建运行目录结构

#### 1.3 数据集加载
```
流程：
1. 获取数据集元数据（LeRobotDatasetMetadata）
2. 解析 delta timestamps（基于策略配置和数据集 FPS）
3. 构建图像变换管道（ImageTransforms）
4. 创建 LeRobotDataset 实例
5. 构建 DataLoader（支持 DDP 采样）
```

**关键参数**：
- `delta_timestamps`: 观察和动作之间的时间差
- `image_transforms`: 包含几何增强（可选）和标准化
- `episodes_to_load`: 可选的 episode 过滤

---

### 2. 模型初始化

#### 2.1 策略配置构建
```python
FlowMatchingConfig(
    horizon=16,
    n_action_steps=8,
    sampling_steps=10,
    vision_backbone="resnet18",
    ema_power=0.995,
    network_architecture="mlp",
    mlp_dims=[512, 512, 512],
    flow_network_output_param="u",
    cfm_loss_mode="u",
    ...
)
```

#### 2.2 检查点恢复逻辑
```
if resume_run_id is not None:
    ├─ 从 W&B 下载检查点
    └─ 加载策略配置和权重

elif resume_ckpt is not None:
    ├─ 从本地路径加载检查点
    └─ 加载策略配置和权重

else:
    └─ 从零开始创建新模型
```

#### 2.3 模型创建
```python
policy = FlowMatchingPolicy(
    policy_cfg,
    dataset_stats=ds_meta.stats  # 用于输入归一化
)
```

#### 2.4 优化器初始化
```python
optimizer = torch.optim.AdamW(
    policy.get_optim_params(),
    lr=learning_rate,
    weight_decay=weight_decay
)
```

---

### 3. 主训练循环

#### 3.1 循环结构
```
for step in range(start_step, cfg.steps):
    ├─ 梯度累积内循环
    ├─ 优化器步进
    ├─ EMA 更新
    ├─ 日志记录
    ├─ 检查点保存
    └─ 评估（可选）
```

#### 3.2 梯度累积内循环
```python
for i in range(gradient_accumulation_steps):
    1. 获取批次数据
    2. 数据移到设备
    3. 处理数据兼容性（state → observation.state）
    4. 计算 CFM 损失：loss, loss_dict = policy.get_cfm_loss(batch)
    5. 反向传播：loss.backward()
    6. 累积损失用于日志
```

**DDP 优化**：
- 在最后一个累积步骤之前禁用梯度同步（`policy.no_sync()`）
- 在最后一步进行梯度同步

#### 3.3 优化器步进
```python
1. 梯度裁剪：clip_grad_norm_(policy.parameters(), grad_clip_norm)
2. 优化器步进：optimizer.step()
3. 梯度清零：optimizer.zero_grad(set_to_none=True)
4. EMA 更新：policy.step_ema()
```

#### 3.4 日志记录
```
记录内容：
├─ 训练损失（loss_dict）
├─ 学习率（learning_rate）
├─ 梯度范数（grad_norm_before_clip）
├─ 当前 epoch（fractional_epoch）
├─ 数据加载时间（data_load_ms）
└─ 迭代时间（iter_ms）

频率：每 log_freq 步记录一次
目标：
├─ 控制台输出
└─ Weights & Biases
```

---

### 4. 检查点管理

#### 4.1 检查点保存
```
保存频率：每 save_freq 步或训练结束时

保存内容：
├─ 策略权重（policy.save_pretrained()）
├─ 优化器状态（optimizer.state_dict()）
├─ EMA 权重（如果启用）
└─ 训练步数

检查点组织：
run_dir/
├─ checkpoints/
│  ├─ step_0/
│  ├─ step_1000/
│  ├─ step_2000/
│  ├─ latest/      # 最新检查点
│  └─ best/        # 最佳检查点（基于评估成功率）
└─ wandb/          # W&B 日志
```

#### 4.2 W&B 工件上传
```
每次保存检查点时：
1. 创建 wandb.Artifact（type="model"）
2. 添加检查点目录
3. 上传工件
4. 为 "latest" 检查点添加别名
```

---

### 5. 评估循环（可选）

#### 5.1 评估触发条件
```
if (
    rank == 0 and
    rollout_freq is not None and
    eval_env is not None and
    (step % rollout_freq == 0 or step == cfg.steps or step == 1)
):
    执行评估
```

#### 5.2 环境创建
```
支持的环境：
├─ DexMimicGen 环境（9 种）
│  ├─ TwoArmCoffee
│  ├─ TwoArmThreading
│  ├─ TwoArmThreePieceAssembly
│  ├─ TwoArmTransport
│  ├─ TwoArmLiftTray
│  ├─ TwoArmBoxCleanup
│  ├─ TwoArmDrawerCleanup
│  ├─ TwoArmPouring
│  └─ TwoArmCanSortRandom
└─ Robomimic 环境（3 种）
   ├─ Lift
   ├─ Can
   └─ Square

环境配置：
├─ num_envs: 并行环境数
├─ camera_size: 图像大小
├─ debug: 同步 vs 异步模式
└─ expected_image_keys: 图像观察键
```

#### 5.3 评估执行
```python
def _run_rollouts(...):
    1. 设置策略为评估模式
    2. 重置环境和策略
    3. 运行 num_episodes 个 episode：
       ├─ 获取观察
       ├─ 策略推理：policy.select_action(obs)
       ├─ 环境步进：env.step(action)
       ├─ 渲染帧并注释
       └─ 写入视频
    4. 计算成功率
    5. 返回 (success_rate, video_path, fps)
```

#### 5.4 评估指标
```
记录指标：
├─ eval/success_rate: 成功率（0-1）
├─ time/rollout_ms: 评估耗时
├─ eval/rollout_video: 评估视频（W&B UI）
└─ 视频工件（版本化存储）

帧注释内容：
├─ 环境索引
├─ Episode 编号
├─ 当前步数
└─ 成功/失败状态
```

#### 5.5 最佳检查点保存
```
if success_rate > best_success_rate:
    ├─ 更新 best_success_rate
    ├─ 保存本地最佳检查点
    ├─ 上传到 W&B（别名 "best"）
    └─ 记录日志
```

---

### 6. 训练完成

#### 6.1 清理操作
```python
if rank == 0:
    ├─ 记录 "Training finished!" 日志
    ├─ 关闭 W&B 运行（wandb.finish()）
    └─ 关闭评估环境（eval_env.close()）

if is_ddp:
    └─ 销毁分布式进程组
```

---

## 关键特性

### 1. 分布式训练（DDP）
- **多 GPU 支持**: 使用 PyTorch DDP 进行数据并行
- **梯度同步优化**: 梯度累积时禁用不必要的同步
- **屏障同步**: 确保所有进程在评估前同步

### 2. 检查点管理
- **本地保存**: 定期保存到磁盘
- **W&B 工件**: 版本化存储和下载
- **EMA 权重**: 支持加载和使用 EMA 权重
- **恢复机制**: 支持从本地或 W&B 恢复

### 3. 数据处理
- **LeRobot 数据集**: 自动下载和处理
- **图像变换**: 支持几何增强（旋转、平移、缩放）
- **数据兼容性**: 自动处理不同数据集的字段名称
- **梯度累积**: 支持有效的内存管理

### 4. 日志与监控
- **实时日志**: 控制台输出关键指标
- **W&B 集成**: 完整的实验跟踪
- **视频记录**: 评估过程的可视化
- **性能指标**: FPS、epoch、数据加载时间等

### 5. 评估与验证
- **在线评估**: 在 DexMimicGen 环境中评估
- **成功率追踪**: 自动保存最佳检查点
- **视频生成**: 带注释的评估视频
- **并行评估**: 多环境并行执行

---

## 数据流

```
输入数据
    ↓
LeRobotDataset（带变换）
    ↓
DataLoader（批处理 + DDP 采样）
    ↓
批次处理（数据移到设备）
    ↓
FlowMatchingPolicy.get_cfm_loss()
    ↓
反向传播 + 梯度累积
    ↓
梯度裁剪 + 优化器步进
    ↓
EMA 更新
    ↓
日志记录（W&B + 控制台）
    ↓
检查点保存（本地 + W&B 工件）
    ↓
评估（可选）
    ↓
最佳检查点保存
```

---

## 常见使用场景

### 场景 1: 基础训练
```bash
python pretrain_flow_bc.py \
    --dataset ankile/franka-lift-dataset \
    --wandb_project my-project \
    --steps 3000 \
    --batch_size 256
```

### 场景 2: 从检查点恢复
```bash
python pretrain_flow_bc.py \
    --dataset ankile/franka-lift-dataset \
    --resume_ckpt ./runs/train_flow_bc_2024-01-01_12-00-00/checkpoints/latest \
    --wandb_project my-project
```

### 场景 3: 从 W&B 恢复
```bash
python pretrain_flow_bc.py \
    --dataset ankile/franka-lift-dataset \
    --resume_run_id abc123xyz \
    --wandb_project my-project \
    --checkpoint_step best
```

### 场景 4: 带评估的训练
```bash
python pretrain_flow_bc.py \
    --dataset ankile/franka-lift-dataset \
    --eval_env TwoArmLiftTray \
    --rollout_freq 100 \
    --eval_num_episodes 10 \
    --wandb_project my-project
```

### 场景 5: 多 GPU 分布式训练
```bash
torchrun --nproc_per_node=4 pretrain_flow_bc.py \
    --dataset ankile/franka-lift-dataset \
    --is_ddp true \
    --wandb_project my-project
```

---

## 性能优化建议

1. **梯度累积**: 增加 `gradient_accumulation_steps` 以模拟更大的批次
2. **数据加载**: 调整 `num_workers` 以平衡 CPU 和 I/O
3. **EMA**: 启用 EMA（`ema_power > 0`）以稳定训练
4. **学习率**: 为骨干网络使用较小的学习率（`lr_backbone`）
5. **梯度裁剪**: 调整 `grad_clip_norm` 以防止梯度爆炸
6. **检查点频率**: 平衡磁盘空间和恢复灵活性

---

## 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| CUDA 内存不足 | 批次太大 | 减小 `batch_size` 或增加 `gradient_accumulation_steps` |
| 数据加载缓慢 | 工作线程不足 | 增加 `num_workers` |
| 梯度爆炸 | 学习率过高 | 减小 `learning_rate` 或增加 `grad_clip_norm` |
| W&B 连接失败 | 网络问题 | 设置 `wandb_enable=false` 或检查网络 |
| 评估环境创建失败 | 环境名称错误 | 检查 `eval_env` 是否在支持的列表中 |

---

## 总结

`pretrain_flow_bc.py` 是一个功能完整的离线强化学习训练框架，具有以下特点：

- ✅ **完整的训练管道**: 从数据加载到模型评估
- ✅ **分布式支持**: 多 GPU 训练和同步
- ✅ **灵活的检查点**: 本地和云端存储
- ✅ **详细的日志**: W&B 集成和实时监控
- ✅ **在线评估**: 环境中的性能验证
- ✅ **生产就绪**: 错误处理和恢复机制

适用于预训练 Flow Matching 策略模型，用于机器人操作任务。
