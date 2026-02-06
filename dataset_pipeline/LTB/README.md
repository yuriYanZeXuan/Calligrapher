# LTB 数据集构建工具

用于构建 LTB_iid 和 Web_rendered 两个数据集的工具集。

## 🚀 快速开始

### 1. 安装依赖
```bash
cd dataset_pipeline/LTB
./setup.sh
```

### 2. 配置 API
在项目根目录创建 `.env` 文件：
```bash
QST_API_KEY=your_api_key
QST_BASE_URL=https://api.example.com/v1
```

### 3. 测试运行
```bash
# Debug模式测试（10条样本）
./run_build.sh all --debug
```

## 📖 使用方法

### 串行构建
```bash
./run_build.sh all              # 构建所有数据集
./run_build.sh ltb              # 仅构建 LTB_iid
./run_build.sh web              # 仅构建 Web_rendered
./run_build.sh all --debug      # Debug模式
```

### 并行构建（推荐）
```bash
./run_build_parallel.sh                           # 默认配置
./run_build_parallel.sh --debug                   # Debug模式
./run_build_parallel.sh --ltb-samples 500 --web-samples 300  # 自定义
```

### 直接运行
```bash
python3 build_ltb_iid.py --samples 100
python3 build_web_rendered.py --samples 50
```

## 📊 样本数量

| 模式 | LTB_iid | Web_rendered |
|------|---------|--------------|
| 默认 | 2000条/语种 | 1000条/语种 |
| Debug | 10条/语种 | 10条/语种 |
| 自定义 | --samples N | --samples N |

支持 6 种语言：英语、中文、韩语、日语、阿拉伯语、法语

## 📂 输出路径

```
eval/
├── LTB_iid/
│   ├── ltb_iid_en.jsonl
│   ├── ltb_iid_zh.jsonl
│   └── ...
└── Web_rendered/
    ├── images/
    │   ├── en/
    │   └── ...
    ├── web_rendered_en.jsonl
    └── ...
```

## 🔄 Resume 机制

中断后可继续，进度保存在：
- `eval/LTB_iid/progress.json`
- `eval/Web_rendered/progress.json`

## ⚡ 并行构建

```bash
# 启动并行构建
./run_build_parallel.sh --ltb-samples 100 --web-samples 50

# 监控进度（另一个终端）
tail -f logs/ltb_*.log
tail -f logs/web_*.log
```

## 🔧 本地 MathJax

工具使用本地 MathJax（位于 `assets/mathjax/`），避免网络问题：
- ✅ 无需网络连接
- ✅ 渲染速度更快
- ✅ 避免 CDN 超时

如果 MathJax 缺失，运行 `./setup.sh` 会自动下载。

## 📝 数据格式

### LTB_iid
```json
{
  "id": "LTB_en_0",
  "prompt": "场景描述...",
  "category": "dialogue",
  "text": ["TEXT1", "TEXT2"],
  "text_length": 25,
  "length": "short"
}
```

### Web_rendered
```json
{
  "category": "academic",
  "prompt": "图像prompt...",
  "text": ["关键文本1"],
  "text_length": 150,
  "image_path": "eval/Web_rendered/images/en/WR_en_0.png"
}
```

## 🛠️ 故障排除

### Playwright 错误
```bash
playwright install chromium
```

### MathJax 缺失
```bash
./setup.sh
```

### API 连接失败
检查 `.env` 文件配置

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `setup.sh` | 一键安装脚本 |
| `run_build.sh` | 串行构建脚本 |
| `run_build_parallel.sh` | 并行构建脚本 |
| `build_ltb_iid.py` | LTB_iid 构建器 |
| `build_web_rendered.py` | Web_rendered 构建器 |
| `extract_content.py` | MD内容提取 |
| `llm_processor.py` | LLM API调用 |
| `render_richtext.py` | 富文本渲染 |
| `config.yaml` | 配置文件 |
| `assets/mathjax/` | 本地MathJax |

## 💡 最佳实践

1. **首次使用**: 运行 `./setup.sh` 安装依赖
2. **测试**: 先用 `--debug` 模式验证环境
3. **生产**: 使用并行构建提高效率
4. **监控**: 使用 `tail -f logs/*.log` 查看进度
5. **恢复**: 中断后直接重新运行相同命令
