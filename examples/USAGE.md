# 训练脚本使用指南

## ✅ 完全兼容

`train_grpo.sh` 和 `train_nft.sh` 已经完全兼容 `run_server.sh` 和 `run_training.sh` 的逻辑！

---

## 🚀 快速启动（完整流程）

### 步骤 1: 启动 Reward Server

在**第一个终端**运行：

```bash
cd /path/to/Calligrapher

# 启动 4 个 reward servers（对应 4 个训练 GPU）
bash run_server.sh \
    --vlm-model-path "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B" \
    --port 8000 \
    --gpu-ids "4,5,6,7"
```

**输出：**
```
Launching server on GPU 4 at 0.0.0.0:8000 ✓
Launching server on GPU 5 at 0.0.0.0:8001 ✓
Launching server on GPU 6 at 0.0.0.0:8002 ✓
Launching server on GPU 7 at 0.0.0.0:8003 ✓
Press Ctrl+C to stop all servers.
```

### 步骤 2: 启动训练

在**第二个终端**运行：

```bash
cd /path/to/Calligrapher

# 方式 1: GRPO 训练
bash examples/train_grpo.sh

# 方式 2: NFT 训练
bash examples/train_nft.sh

# 方式 3: 使用原始脚本（仅支持 GRPO）
bash run_training.sh
```

---

## 🔌 接口兼容性说明

### Reward Server 接口

**API Endpoint:**
```
POST http://127.0.0.1:{port}/score
```

**Request:**
```json
{
    "image": "base64_encoded_png",
    "prompt": "text description"
}
```

**Response:**
```json
{
    "vlm_score": 0.85,        // VLM 评分 (0-1)
    "ocr_confidence": 0.92    // OCR 置信度 (0-1)
}
```

### GPU 分配策略

```
训练 GPU: 0, 1, 2, 3  → 连接到 → Reward Servers: 8000, 8001, 8002, 8003
                                    ↓
训练进程 0 (GPU 0) ──────────→ Server 8000 (GPU 4)
训练进程 1 (GPU 1) ──────────→ Server 8001 (GPU 5)
训练进程 2 (GPU 2) ──────────→ Server 8002 (GPU 6)
训练进程 3 (GPU 3) ──────────→ Server 8003 (GPU 7)
```

**自动负载均衡：** 每个训练进程独享一个 reward server，避免竞争。

---

## 📊 脚本对比

| 特性 | run_training.sh | train_grpo.sh | train_nft.sh |
|------|----------------|---------------|--------------|
| **RL 方法** | GRPO only | GRPO | NFT |
| **多GPU** | ✅ accelerate | ✅ accelerate | ✅ accelerate |
| **单GPU** | ❌ 不支持 | ✅ 自动切换 | ✅ 自动切换 |
| **Server URLs** | ✅ 自动构建 | ✅ 自动构建 | ✅ 自动构建 |
| **EMA** | ❌ 无 | ✅ 支持 | ✅ 支持 |
| **NFT 参数** | ❌ 无 | ❌ 无 | ✅ 支持 |

---

## 🎯 使用场景

### 场景 1: 多 GPU 训练（推荐）

```bash
# 1. 启动 reward servers（GPU 4,5,6,7）
bash run_server.sh --gpu-ids "4,5,6,7"

# 2. 启动训练（GPU 0,1,2,3）
# 编辑脚本，设置：
# TRAINING_GPU_IDS="0,1,2,3"

bash examples/train_grpo.sh
```

**自动行为：**
- 使用 `accelerate launch` 启动 4 个训练进程
- 自动连接到 4 个 reward servers
- 每个训练进程独享一个 server

### 场景 2: 单 GPU 训练

```bash
# 1. 启动单个 reward server
bash run_server.sh --gpu-ids "1"

# 2. 编辑训练脚本
# TRAINING_GPU_IDS=""  # 留空

bash examples/train_grpo.sh
```

**自动行为：**
- 使用 `python -m train.train` 单进程启动
- 连接到单个 server (http://127.0.0.1:8000/score)

### 场景 3: 调试模式（不使用 reward server）

```bash
# 编辑脚本，添加：
# DISABLE_RL_REWARD_MODEL=true

# 然后运行（不需要启动 run_server.sh）
bash examples/train_grpo.sh
```

---

## 📝 配置对照表

### GPU 配置

| 配置文件 | GPU 用途 | 默认值 |
|---------|---------|--------|
| `run_server.sh` | Reward 评分 | GPU 4,5,6,7 |
| `run_training.sh` | 训练 | GPU 0,1,2,3 |
| `train_grpo.sh` | 训练 | GPU 0,1,2,3 |
| `train_nft.sh` | 训练 | GPU 0,1,2,3 |

### 端口配置

| Server | 端口 | GPU |
|--------|------|-----|
| Server 0 | 8000 | GPU 4 |
| Server 1 | 8001 | GPU 5 |
| Server 2 | 8002 | GPU 6 |
| Server 3 | 8003 | GPU 7 |

---

## 🔧 如何修改配置

### 修改训练 GPU

编辑 `examples/train_grpo.sh`:
```bash
# 单 GPU
TRAINING_GPU_IDS=""

# 2 GPUs
TRAINING_GPU_IDS="0,1"

# 4 GPUs（默认）
TRAINING_GPU_IDS="0,1,2,3"
```

### 修改 Reward Server

编辑 `examples/train_grpo.sh`:
```bash
REWARD_SERVER_HOST="127.0.0.1"  # 本地
# 或
REWARD_SERVER_HOST="10.0.0.5"   # 远程服务器

REWARD_SERVER_PORT=8000  # 起始端口
```

---

## 📋 完整启动示例

### 示例 1: 完整多 GPU 训练（GRPO）

```bash
# Terminal 1: 启动 Reward Servers
cd /path/to/Calligrapher
bash run_server.sh

# Terminal 2: 启动 GRPO 训练
cd /path/to/Calligrapher
bash examples/train_grpo.sh

# Terminal 3: 监控训练
tensorboard --logdir output_grpo/logs

# Terminal 4: 监控 Reward Server 日志
tail -f logs/reward_server_gpu_4.log
```

### 示例 2: NFT 训练

```bash
# Terminal 1: Reward Servers（同上）
bash run_server.sh

# Terminal 2: NFT 训练
bash examples/train_nft.sh
```

### 示例 3: 单 GPU 快速测试

```bash
# Terminal 1: 单个 Reward Server
bash run_server.sh --gpu-ids "4"

# Terminal 2: 编辑 train_grpo.sh
# 设置 TRAINING_GPU_IDS=""

# Terminal 2: 运行训练
bash examples/train_grpo.sh
```

---

## ⚠️ 注意事项

### 1. GPU 不冲突
- ✅ 训练 GPU: 0,1,2,3
- ✅ Reward GPU: 4,5,6,7
- ❌ 避免重叠（会导致 OOM）

### 2. 端口匹配
```bash
# run_server.sh 启动在端口 8000-8003
# 训练脚本必须连接到相同端口
REWARD_SERVER_PORT=8000  # 必须一致！
```

### 3. 进程数量匹配
```bash
# 如果训练用 4 个 GPU
TRAINING_GPU_IDS="0,1,2,3"

# 则需要 4 个 reward servers
bash run_server.sh --gpu-ids "4,5,6,7"
```

---

## 🎉 总结

### ✅ 现在完全兼容！

| 功能 | 状态 |
|------|------|
| **接口统一** | ✅ 所有脚本使用相同的 reward API |
| **多GPU支持** | ✅ 自动检测并使用 accelerate |
| **单GPU支持** | ✅ 自动降级为 python -m |
| **Server 负载均衡** | ✅ 每个训练进程独享 server |
| **GRPO 训练** | ✅ train_grpo.sh |
| **NFT 训练** | ✅ train_nft.sh |

### 启动命令

```bash
# 1. 启动 Reward Servers
bash run_server.sh

# 2. 启动训练（选择一个）
bash examples/train_grpo.sh    # GRPO 方法
bash examples/train_nft.sh     # NFT 方法
bash run_training.sh           # 原始方法（仅 GRPO）
```

**所有脚本现在完全兼容，可以互换使用！** 🎉

