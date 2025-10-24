# Calligrapher RL 训练完整指南

> 本指南包含 EMA 平滑和 DiffusionNFT 训练方式的完整说明

---

## 📋 目录

1. [快速开始](#快速开始)
2. [训练方法选择](#训练方法选择)
3. [远程服务器部署](#远程服务器部署)
4. [核心功能说明](#核心功能说明)
5. [参数详解](#参数详解)
6. [常见问题](#常见问题)
7. [实现细节](#实现细节)

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 进入项目根目录
cd /path/to/Calligrapher

# 2. 安装依赖
pip install -r requirements.txt

# 3. 查看帮助
python -m train.train --help
```

### GRPO 训练（推荐新手）

```bash
# 使用示例脚本
bash examples/train_grpo.sh

# 或手动运行
python -m train.train \
    --model_type flux \
    --pretrained_model_name_or_path "/path/to/flux-fill" \
    --siglip_path "/path/to/siglip" \
    --train_data_json "/path/to/data.json" \
    --output_dir "output/grpo" \
    --use_rl \
    --rl_method grpo \
    --use_ema
```

### DiffusionNFT 训练（追求稳定性）

```bash
# 使用示例脚本
bash examples/train_nft.sh

# 或手动运行
python -m train.train \
    --model_type flux \
    --pretrained_model_name_or_path "/path/to/flux-fill" \
    --siglip_path "/path/to/siglip" \
    --train_data_json "/path/to/data.json" \
    --output_dir "output/nft" \
    --use_rl \
    --rl_method nft \
    --nft_beta 0.5 \
    --use_ema
```

---

## 🎯 训练方法选择

### 方法对比

| 特性 | GRPO | DiffusionNFT |
|------|------|--------------|
| **训练稳定性** | 中等 | ⭐⭐⭐ 高 |
| **收敛速度** | ⭐⭐⭐ 快 | 中等 |
| **内存使用** | ⭐⭐⭐ 低 | 高（需 old 模型） |
| **计算开销** | ⭐⭐⭐ 低 | 高（3 次前向） |
| **适用场景** | 快速实验、资源受限 | 追求最佳效果 |

### GRPO (Group Relative Policy Optimization)

**原理：** PPO-style 策略梯度，使用 log probabilities 和 advantages

**优势：**
- ✅ 快速收敛
- ✅ 内存友好
- ✅ 计算高效

**推荐配置：**
```bash
--rl_method grpo \
--rl_grpo_clip_range 0.2 \
--rl_kl_beta 0.1 \
--use_ema \
--ema_decay 0.9999
```

### DiffusionNFT (Negative Feedback Training)

**原理：** 隐式负反馈 + 自适应权重 + 三模型预测

**优势：**
- ✅ 训练稳定
- ✅ 高质量结果
- ✅ 多种 advantage 模式

**推荐配置：**
```bash
--rl_method nft \
--nft_beta 0.5 \
--nft_decay_type 1 \
--rl_kl_beta 0.01 \
--use_ema \
--ema_decay 0.999
```

---

## 🖥️ 远程服务器部署

### 重要变更说明

**本项目已优化为相对导入，适合远程服务器部署。**

#### 之前（不推荐）：
```bash
python train/train.py --model_type flux ...
```

#### 现在（必须）：
```bash
python -m train.train --model_type flux ...
```

### 完整部署流程

#### 1. 上传代码
```bash
# 本地压缩
cd /path/to/
tar -czf calligrapher.tar.gz Calligrapher/

# 上传
scp calligrapher.tar.gz user@server:/path/to/

# 服务器解压
ssh user@server
cd /path/to/
tar -xzf calligrapher.tar.gz
```

#### 2. 安装依赖
```bash
cd Calligrapher
pip install -r requirements.txt
```

#### 3. 验证部署
```bash
python verify_deployment.py
```

#### 4. 运行训练
```bash
# 前台运行
python -m train.train --rl_method grpo ...

# 后台运行
nohup python -m train.train --rl_method grpo ... > training.log 2>&1 &

# 查看日志
tail -f training.log
```

#### 5. 监控训练
```bash
# 本地端口转发
ssh -L 6006:localhost:6006 user@server

# 服务器启动 TensorBoard
tensorboard --logdir output/logs --port 6006

# 浏览器访问 http://localhost:6006
```

---

## ⚙️ 核心功能说明

### 1. EMA (Exponential Moving Average)

**作用：** 平滑模型参数，减少训练波动

**使用方法：**
```bash
--use_ema \
--ema_decay 0.9999 \      # GRPO 推荐
--ema_decay 0.999 \       # NFT 推荐
--ema_update_interval 1
```

**效果：**
- ✅ 更稳定的训练过程
- ✅ 推理时使用 EMA 参数获得更好效果
- ✅ 自动保存在 checkpoint 中

### 2. GRPO 训练器

**核心算法：**
```python
ratio = exp(log_prob - old_log_prob)
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * clamp(ratio, 1-ε, 1+ε)
loss = max(unclipped_loss, clipped_loss)
```

**关键参数：**
- `--rl_grpo_clip_range`: PPO 裁剪范围 (默认: 0.2)
- `--rl_kl_beta`: KL 散度系数 (默认: 0.1)
- `--rl_adv_clip_max`: Advantage 裁剪 (默认: 5.0)

### 3. DiffusionNFT 训练器

**核心算法：**
```python
# 正预测
positive_pred = β * forward + (1-β) * old

# 负预测（隐式）
negative_pred = (1+β) * old - β * forward

# 自适应权重
weight = |x0_pred - x0|

# 损失
loss = r * positive_loss/β + (1-r) * negative_loss/β + α * kl_loss
```

**关键参数：**
- `--nft_beta`: 正/负混合系数 (默认: 0.5)
  - 0.3-0.5: 更依赖旧模型（稳定）
  - 0.7-0.9: 更依赖当前模型（快速）
- `--nft_adv_mode`: Advantage 模式
  - `normal`: 标准模式
  - `positive_only`: 仅正 advantage
  - `negative_only`: 仅负 advantage
  - `one_only`: 二值化
  - `binary`: 符号函数
- `--nft_decay_type`: Old 模型更新策略
  - `0`: 不更新（固定参考）
  - `1`: 缓慢更新（推荐）
  - `2`: 快速更新

---

## 📖 参数详解

### 通用 RL 参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--use_rl` | 启用 RL 训练 | false | true |
| `--rl_method` | 训练方法 | grpo | grpo / nft |
| `--rl_warmup_steps` | 预热步数 | 1000 | 500-1000 |
| `--rl_num_images_per_prompt` | 每 prompt 图像数 | 1 | 4-8 |
| `--rl_num_batches_per_epoch` | 每 epoch 批次数 | 1 | 1-2 |
| `--rl_num_inference_steps` | 采样步数 | 50 | 30-50 |
| `--rl_guidance_scale` | 引导强度 | 3.5 | 3.0-4.0 |
| `--rl_timestep_fraction` | 训练时间步比例 | 1.0 | 0.8-1.0 |
| `--rl_num_inner_epochs` | 内部训练轮数 | 1 | 1-3 |
| `--rl_adv_clip_max` | Advantage 裁剪 | 5.0 | 3.0-10.0 |
| `--rl_per_prompt_stat_tracking` | Per-prompt 统计 | false | true |

### GRPO 专用参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--rl_grpo_clip_range` | PPO 裁剪范围 | 0.2 | 0.1-0.3 |
| `--rl_kl_beta` | KL 散度系数 | 0.1 | 0.05-0.2 |

### NFT 专用参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--nft_beta` | 正/负混合系数 | 0.5 | 0.3-0.7 |
| `--nft_adv_mode` | Advantage 模式 | normal | normal |
| `--nft_decay_type` | Old 模型更新 | 0 | 1 |
| `--rl_kl_beta` | KL 散度系数 | 0.01 | 0.005-0.02 |

### EMA 参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--use_ema` | 启用 EMA | false | true |
| `--ema_decay` | 衰减率 | 0.9999 | GRPO: 0.9999, NFT: 0.999 |
| `--ema_update_interval` | 更新间隔 | 1 | 1 |

---

## ❓ 常见问题

### Q1: 两种方法可以切换吗？
**A:** 可以，只需改变 `--rl_method` 参数。但建议从头开始训练。

### Q2: ImportError: attempted relative import
**A:** 使用 `python -m train.train` 而非 `python train/train.py`

### Q3: ModuleNotFoundError: No module named 'train'
**A:** 确保在项目根目录运行
```bash
cd /path/to/Calligrapher
python -m train.train
```

### Q4: 内存不足怎么办？
**A:**
```bash
# NFT 切换到 GRPO
--rl_method grpo

# 减少图像数
--rl_num_images_per_prompt 2

# 使用 offloading
--offload_text_encoder_two

# 降低分辨率
--resolution 512
```

### Q5: 训练不稳定？
**A:**
```bash
# 使用 NFT 方法
--rl_method nft

# 启用 EMA
--use_ema --ema_decay 0.999

# 增加预热
--rl_warmup_steps 1000
```

### Q6: 训练太慢？
**A:**
```bash
# 使用 GRPO
--rl_method grpo

# 减少采样步数
--rl_num_inference_steps 30

# 减少图像数
--rl_num_images_per_prompt 2
```

---

## ✅ Flux-Fill + IP-Adapter 兼容性

### 完全兼容！

经过全面检查和修复，代码**完全支持** flux-fill + IP-Adapter 的所有使用场景：

#### 1. 训练兼容性 ✅
- ✅ IP-Adapter 正确集成到 Flux-Fill
- ✅ 支持 inpainting 输入格式
- ✅ 图像特征正确注入到 transformer

#### 2. 保存兼容性 ✅
```python
# 保存格式（save_model_hook）
state_dict = {
    'image_proj': image_proj_model.state_dict(),
    'ip_adapter': attn_processors.state_dict()
}
# 保存为: checkpoint-{step}/ip-adapter.bin
```

#### 3. 加载兼容性 ✅
```python
# 支持两种格式（向后兼容）
if "image_proj" in state_dict:
    # 标准格式（推荐）
    load_with_standard_keys()
elif "image_proj_mlp" in state_dict:
    # 遗留格式（兼容旧版本）
    load_with_legacy_keys()
```

#### 4. 推理兼容性 ✅
- ✅ 训练保存的权重可直接用于推理
- ✅ 权重 key 与推理代码统一
- ✅ 可从任何 checkpoint 加载

### 权重格式说明

**标准格式（推荐）：**
```python
{
    'image_proj': {...},      # 图像投影模型
    'ip_adapter': {...}       # Attention processors
}
```

**支持的遗留格式：**
```python
{
    'image_proj_mlp': {...},  # 旧版本的图像投影模型
    'attn_adapter': {...}     # 旧版本的 attention processors
}
```

### 完整工作流程

```
训练 (python -m train.train)
  ↓
保存 (checkpoint-{step}/ip-adapter.bin)
  ↓
继续训练 (--initial_ip_adapter_path) ✅
  ↓
推理 (使用保存的权重) ✅
```

---

## 🔍 实现细节

### 项目结构

```
Calligrapher/
├── train/
│   ├── train.py              # 主训练脚本
│   ├── rl_ip/
│   │   ├── ema.py           # ✨ EMA 实现
│   │   ├── reward.py        # Reward 客户端
│   │   ├── stat_tracking.py # 统计跟踪
│   │   └── grpo_utils.py    # GRPO 工具
│   ├── rl_logic/             # ✨ RL 训练逻辑
│   │   ├── grpo_trainer.py  # GRPO 训练器
│   │   ├── nft_trainer.py   # NFT 训练器
│   │   └── nft_forward.py   # NFT 辅助
│   └── tests/
│       └── test_rl_trainers.py
├── examples/                 # ✨ 训练脚本
│   ├── train_grpo.sh
│   └── train_nft.sh
└── RL_TRAINING_GUIDE.md      # 本文档
```

### 设计特点

1. **模块化设计** - 训练逻辑与主流程解耦
2. **相对导入** - 适合远程服务器部署
3. **灵活配置** - 命令行参数轻松切换
4. **完善文档** - 从入门到精通

### 训练流程

#### GRPO 流程
1. Warmup（监督学习）
2. Rollout（生成图像）
3. Compute Rewards
4. Compute Advantages
5. **GRPO Loss**: PPO-style policy gradient
6. Update Model + EMA

#### NFT 流程
1. Warmup（监督学习）
2. Initialize Old Model
3. Rollout（生成图像）
4. Compute Rewards
5. Compute Advantages
6. **NFT Loss**: 
   - Forward pred (current)
   - Old pred (negative feedback)
   - Ref pred (KL penalty)
7. Update Model + EMA + Old Model

### 性能考虑

**内存：**
- GRPO: 基准 + 梯度 + EMA ≈ 2x 参数
- NFT: 基准 + 梯度 + Old + EMA ≈ 3x 参数

**计算：**
- GRPO: 1 次前向 + 1 次反向
- NFT: 3 次前向 + 1 次反向

---

## 📝 完整配置示例

### 最小配置（快速测试）

```bash
python -m train.train \
    --model_type flux \
    --pretrained_model_name_or_path "/path/to/model" \
    --siglip_path "/path/to/siglip" \
    --train_data_json "/path/to/data.json" \
    --output_dir "output/test" \
    --resolution 512 \
    --train_batch_size 1 \
    --max_train_steps 1000 \
    --use_rl \
    --rl_method grpo
```

### 生产配置（GRPO）

```bash
python -m train.train \
    --model_type flux \
    --pretrained_model_name_or_path "/path/to/model" \
    --siglip_path "/path/to/siglip" \
    --train_data_json "/path/to/data.json" \
    --output_dir "output/grpo_prod" \
    --resolution 512 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_train_steps 10000 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --offload_text_encoder_two \
    --use_rl \
    --rl_method grpo \
    --rl_warmup_steps 500 \
    --rl_num_images_per_prompt 4 \
    --rl_num_inference_steps 50 \
    --rl_grpo_clip_range 0.2 \
    --rl_kl_beta 0.1 \
    --rl_per_prompt_stat_tracking \
    --use_ema \
    --ema_decay 0.9999 \
    --checkpointing_steps 500 \
    --report_to tensorboard
```

### 生产配置（NFT）

```bash
python -m train.train \
    --model_type flux \
    --pretrained_model_name_or_path "/path/to/model" \
    --siglip_path "/path/to/siglip" \
    --train_data_json "/path/to/data.json" \
    --output_dir "output/nft_prod" \
    --resolution 512 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_train_steps 10000 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --offload_text_encoder_two \
    --use_rl \
    --rl_method nft \
    --rl_warmup_steps 1000 \
    --rl_num_images_per_prompt 4 \
    --rl_num_inference_steps 50 \
    --nft_beta 0.5 \
    --nft_adv_mode normal \
    --nft_decay_type 1 \
    --rl_kl_beta 0.01 \
    --rl_per_prompt_stat_tracking \
    --use_ema \
    --ema_decay 0.999 \
    --checkpointing_steps 500 \
    --report_to tensorboard
```

---

## 🎓 参考资料

- **GRPO 实现**: 参考 flow_grpo 项目
- **DiffusionNFT 实现**: 参考 Edit-R1 项目
- **EMA 模块**: 参考 flow_grpo/ema.py
- **测试脚本**: `train/tests/test_rl_trainers.py`

---

## ✅ 检查清单

部署前检查：
- [ ] 代码已上传到服务器
- [ ] 依赖已安装 (`pip install -r requirements.txt`)
- [ ] 在项目根目录 (`cd /path/to/Calligrapher`)
- [ ] 数据和模型路径正确
- [ ] 使用 `python -m train.train` 运行

训练监控：
- [ ] TensorBoard 正常
- [ ] 日志文件正常写入
- [ ] Reward 服务器响应正常
- [ ] 内存使用在合理范围
- [ ] Checkpoint 正常保存

---

**祝训练顺利！** 🎉

有问题请查看日志或使用 `python -m train.train --help`

