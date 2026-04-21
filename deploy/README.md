# Modal 部署指南 - Manga Image Translator

将 Manga Image Translator 部署到 Modal 的 serverless GPU 平台的完整指南。

## 📦 目录结构

```
deploy/
├── README.md              # 本文件 - 完整部署指南
├── modal_app.py          # Modal 应用定义（核心文件）
├── modal_config.py       # 配置常量
├── deploy.sh             # 自动化部署脚本
├── prepare_models.py     # 模型下载和验证工具
└── smoke_test.py         # 健康检查和测试脚本
```

## 🚀 快速开始（5分钟）

### 1. 安装 Modal CLI
```bash
pip install modal
modal token new  # 会打开浏览器进行登录
```

### 2. 配置环境变量
```bash
# 复制环境变量模板
cp .env.modal.example .env

# 生成认证 nonce 并添加到 .env
echo "MT_WEB_NONCE=$(openssl rand -hex 32)" >> .env

# 编辑并添加你的翻译 API 密钥（可选）
vim .env
```

### 3. 一键部署
```bash
# 使用自动化脚本
./deploy/deploy.sh setup    # 创建 Modal secrets
./deploy/deploy.sh deploy   # 部署应用 (~15-20分钟)
./deploy/deploy.sh models   # 下载模型 (~30-60分钟)
./deploy/deploy.sh test     # 运行测试
```

### 4. 获取 URL 并测试
```bash
# URL 格式: https://YOUR-USERNAME--manga-translator-web.modal.run
curl https://YOUR-URL/health
```

**完成！** 🎉 你的漫画翻译 API 已经运行在 Modal 上了。

---

## 📋 详细部署步骤

### 步骤 1: 配置 Secrets

#### 方式 A: 从 .env 文件创建（推荐）
```bash
# 1. 准备 .env 文件
cp .env.modal.example .env

# 2. 生成 nonce 并添加到 .env
echo "MT_WEB_NONCE=$(openssl rand -hex 32)" >> .env

# 3. 编辑 .env 添加翻译 API 密钥（可选）
vim .env

# 4. 一次性创建 Modal secret（包含所有变量）
modal secret create manga-translator-env --from-dotenv-file .env
```

#### 方式 B: 命令行直接创建
```bash
modal secret create manga-translator-env \
  MT_WEB_NONCE=$(openssl rand -hex 32) \
  OPENAI_API_KEY=sk-... \
  DEEPL_AUTH_KEY=...
```

#### 验证 Secret
```bash
modal secret list
# 应该看到: manga-translator-env

# 查看 Secret 详情（值会被隐藏）
modal secret view manga-translator-env
```

### 步骤 2: 部署应用

```bash
# 自动化部署
./deploy/deploy.sh deploy

# 或手动部署
modal deploy deploy/modal_app.py
```

**首次部署时间**: 15-20分钟（构建镜像）

**部署完成后会显示**:
```
✓ App deployed!
Web endpoint: https://YOUR-USERNAME--manga-translator-web.modal.run
```

### 步骤 3: 下载模型

```bash
# 自动化下载
./deploy/deploy.sh models

# 或手动下载
modal run deploy/modal_app.py::download_models
```

**下载时间**: 30-60分钟（约5.2GB模型）

### 步骤 4: 测试部署

```bash
# 使用自动化测试脚本
./deploy/deploy.sh test

# 或手动测试
export MODAL_URL="https://YOUR-USERNAME--manga-translator-web.modal.run"
python deploy/smoke_test.py --url $MODAL_URL --verbose
```

**预期结果**: 所有测试通过 ✅

---

## 🏗️ 架构说明

### Master/Worker 架构

项目使用 Master/Worker 模式：

```
┌─────────────────────────────────────────────┐
│ Modal Container (GPU: T4/A10G/A100)        │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │ Master Process (Port 5003)           │  │
│  │ - FastAPI HTTP 服务器                │  │
│  │ - 接收客户端请求                      │  │
│  │ - 管理任务队列                        │  │
│  └─────────┬────────────────────────────┘  │
│            │ localhost HTTP + pickle       │
│            ▼                                │
│  ┌──────────────────────────────────────┐  │
│  │ Worker Process (Port 5004)           │  │
│  │ - 加载 ML 模型 (GPU)                 │  │
│  │ - 执行 Detection/OCR/Inpainting      │  │
│  │ - 返回翻译结果                        │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  Persistent Volumes:                       │
│  • /app/models  (~5.2GB) - 模型缓存       │
│  • /app/result  - 翻译结果存储             │
└─────────────────────────────────────────────┘
```

**为什么需要 Worker 子进程？**
1. **资源隔离**: 模型加载和推理在独立进程
2. **并发控制**: Worker 使用锁机制防止 OOM
3. **错误隔离**: Worker 崩溃不影响 Master
4. **生命周期管理**: 独立的模型 TTL 和缓存策略

### 关键组件

| 组件 | 说明 | 大小/配置 |
|------|------|-----------|
| **Container Image** | PyTorch + CUDA + 依赖 | ~15-18GB |
| **Model Volume** | ML 模型缓存 | ~5.2GB |
| **Result Volume** | 翻译结果存储 | 动态增长 |
| **GPU** | T4 (默认) / A10G / A100 | 可配置 |
| **Memory** | RAM | 16GB (默认) |

---

## ⚙️ 配置选项

### GPU 配置

编辑 `deploy/modal_config.py`:

```python
# 成本优化（默认）- 按需启动
GPU_CONFIG = {
    "gpu": "T4",
    "cpu": 4.0,
    "memory": 16384,  # 16GB
    "keep_warm": 0,   # 无常驻实例
}

# 平衡性能 - 保持1个实例温暖
GPU_CONFIG = {
    "gpu": "A10G",
    "cpu": 8.0,
    "memory": 32768,  # 32GB
    "keep_warm": 1,
}

# 高性能 - 低延迟
GPU_CONFIG = {
    "gpu": "A100",
    "cpu": 16.0,
    "memory": 65536,  # 64GB
    "keep_warm": 2,
}
```

修改后重新部署:
```bash
modal deploy deploy/modal_app.py
```

### 翻译服务配置

在 `.env` 中配置（所有服务都是可选的）:

| 服务 | 环境变量 | 免费额度 |
|------|----------|----------|
| **OpenAI/ChatGPT** | `OPENAI_API_KEY` | ❌ 按量付费 |
| **DeepL** | `DEEPL_AUTH_KEY` | ⚠️ 50万字符/月 |
| **Google Gemini** | `GEMINI_API_KEY` | ✅ 有免费额度 |
| **Groq** | `GROQ_API_KEY` | ✅ 有免费额度 |
| **DeepSeek** | `DEEPSEEK_API_KEY` | ✅ 价格低廉 |
| **Baidu** | `BAIDU_APP_ID`, `BAIDU_SECRET_KEY` | ⚠️ 有限额度 |
| **Youdao** | `YOUDAO_APP_KEY`, `YOUDAO_SECRET_KEY` | ⚠️ 有限额度 |
| **离线模式** | 无需配置 | ✅ 完全免费 |

**更新 API 密钥**:
```bash
modal secret update manga-translator-env \
  OPENAI_API_KEY=sk-new-key
```

---

## 🧪 API 使用示例

### 健康检查
```bash
curl https://YOUR-URL/health

# 响应示例
{
  "status": "healthy",
  "service": "manga-translator",
  "gpu_available": true,
  "worker": {
    "pid": 42,
    "alive": true,
    "healthy": true,
    "host": "127.0.0.1",
    "port": 5004
  }
}
```

### 翻译请求（Python）
```python
import requests
import base64

# 加载图片
with open("manga_page.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 发送翻译请求
response = requests.post(
    "https://YOUR-URL/translate/json",
    json={
        "image": image_b64,
        "config": {
            "translator": {
                "translator": "gpt3.5",  # 或 "none" 用于离线OCR
                "target_lang": "ENG",    # 目标语言
            },
            "detector": {
                "detector": "default",   # 文本检测器
            },
            "ocr": {
                "ocr": "48px",          # OCR 模型
            },
            "inpainter": {
                "inpainter": "aot",     # 修复器
            }
        }
    },
    timeout=120
)

result = response.json()
print(f"找到 {len(result['blocks'])} 个文本块")

# 保存翻译后的图片
translated_img = base64.b64decode(result['image_base64'])
with open("translated.png", "wb") as f:
    f.write(translated_img)
```

### 翻译请求（cURL）
```bash
curl -X POST https://YOUR-URL/translate/json \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -i manga_page.png)'",
    "config": {
      "translator": {
        "translator": "none",
        "target_lang": "ENG"
      }
    }
  }'
```

### 批量翻译
```bash
curl -X POST https://YOUR-URL/translate/batch/json \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["'$(base64 -i page1.png)'", "'$(base64 -i page2.png)'"],
    "config": {
      "translator": {
        "translator": "gpt3.5",
        "target_lang": "ENG"
      }
    },
    "batch_size": 2
  }'
```

### 主要 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/queue-size` | POST | 当前队列长度 |
| `/translate/json` | POST | 同步翻译（JSON） |
| `/translate/image` | POST | 同步翻译（图片） |
| `/translate/json/stream` | POST | 流式翻译（JSON） |
| `/translate/with-form/image/stream/web` | POST | Web优化流式翻译 |
| `/translate/batch/json` | POST | 批量翻译 |
| `/results/list` | GET | 列出所有结果 |
| `/result/{folder}/final.png` | GET | 获取特定结果 |

---

## 📊 监控和维护

### 查看日志
```bash
# 实时日志
modal logs manga-translator --follow

# 查看最近日志
modal logs manga-translator

# 应该看到:
# "Starting worker subprocess..."
# "Worker subprocess started with PID: XX"
# "Registered worker at 127.0.0.1:5004"
```

### 清理旧结果
```bash
# 删除7天前的结果
./deploy/deploy.sh cleanup

# 或手动指定
modal run deploy/modal_app.py::cleanup_old_results \
  --max-age-days 7 \
  --max-count 100
```

### 查看 Volume 内容
```bash
modal run deploy/modal_app.py::list_volumes
```

### 更新部署
```bash
# 代码修改后重新部署
./deploy/deploy.sh deploy

# Modal 会:
# - 重新构建镜像（如果有变化）
# - 零停机部署
# - 保留 Volumes 和 Secrets
```

### 监控面板
```bash
# 打开 Modal 控制面板
open https://modal.com/apps

# 查看应用状态
modal app list

# 查看使用统计
open https://modal.com/usage
```

---

## 💰 成本估算

### 计费方式
Modal 按**实际使用时间**计费（秒级）：

| 配置 | GPU | 每小时成本 | 适用场景 |
|------|-----|------------|----------|
| **成本优化** | T4 | ~$0.60/小时 | 低流量，偶尔使用 |
| **平衡配置** | A10G | ~$1.10/小时 | 中等流量，生产环境 |
| **高性能** | A100 | ~$4.00/小时 | 高流量，低延迟需求 |

### 成本示例

#### 按需使用（keep_warm=0，默认）
```
场景: 每天处理 100 张图片，每张耗时 20 秒
计算: 100 × 20秒 = 2000秒 = 0.56小时/天
成本: 0.56 × $0.60 = $0.34/天 ≈ $10/月
```

#### 常驻实例（keep_warm=1）
```
场景: 保持 1 个实例始终运行（T4）
计算: 24小时/天 × 30天 = 720小时/月
成本: 720 × $0.60 = $432/月
```

### 成本优化建议

1. **使用按需模式** (keep_warm=0)
   - 适合低到中等流量
   - 接受 30-45 秒冷启动

2. **定期清理结果**
   ```bash
   ./deploy/deploy.sh cleanup  # 减少存储成本
   ```

3. **只下载需要的模型**
   ```python
   # 在 modal_config.py 中
   DEFAULT_MODELS_TO_DOWNLOAD = ["default", "48px", "aot"]
   ```

4. **批量处理**
   - 使用 `/translate/batch/*` 端点
   - 一次处理多张图片

5. **监控使用情况**
   ```bash
   open https://modal.com/usage
   ```

---

## 🔧 故障排查

### 问题: 部署失败

**检查**:
```bash
# 1. 验证 Modal 认证
modal token show

# 2. 检查 Secrets 是否存在
modal secret list

# 3. 查看部署日志
modal logs manga-translator
```

**解决**:
```bash
# 重新认证
modal token new

# 重新创建 Secrets
./deploy/deploy.sh setup
```

### 问题: Worker 未启动

**症状**: 所有翻译请求超时

**检查**:
```bash
# 查看日志
modal logs manga-translator --follow

# 应该看到:
# "Worker subprocess started with PID: XX"
# 如果看不到，说明 Worker 启动失败
```

**可能原因**:
1. GPU 不可用
2. 模型未下载
3. 内存不足

**解决**:
```bash
# 重新下载模型
modal run deploy/modal_app.py::download_models

# 增加内存（修改 modal_config.py）
GPU_CONFIG["memory"] = 32768  # 增加到 32GB
modal deploy deploy/modal_app.py
```

### 问题: 内存不足 (OOM)

**症状**: 容器崩溃，日志显示 "Out of memory"

**解决**:
```python
# 方案 1: 增加内存
# 编辑 modal_config.py
GPU_CONFIG["memory"] = 32768  # 从 16GB 增加到 32GB

# 方案 2: 减少并发
GPU_CONFIG["allow_concurrent_inputs"] = 1  # 从 2 减少到 1

# 重新部署
modal deploy deploy/modal_app.py
```

### 问题: 模型加载慢

**症状**: 首次请求耗时很长（>2分钟）

**这是正常的**:
- 首次请求需要加载模型到 GPU (~30-60秒)
- 后续请求会快得多 (~10-30秒)

**优化**:
```python
# 保持实例温暖（增加成本）
GPU_CONFIG["keep_warm"] = 1
modal deploy deploy/modal_app.py
```

### 问题: 翻译服务 API 错误

**症状**: "Invalid API key" 或 "Unauthorized"

**解决**:
```bash
# 更新 API 密钥
modal secret update manga-translator-env \
  OPENAI_API_KEY=sk-new-key

# 重新部署
modal deploy deploy/modal_app.py
```

### 问题: 测试失败

**检查健康状态**:
```bash
curl https://YOUR-URL/health

# 确认响应:
# - status: "healthy"
# - worker.alive: true
# - worker.healthy: true
```

**详细测试**:
```bash
python deploy/smoke_test.py \
  --url https://YOUR-URL \
  --verbose \
  --test health  # 只测试健康检查
```

---

## 🔄 回滚部署

如果新部署出现问题：

```bash
# 1. 查看部署历史
modal app list-deployments manga-translator

# 2. 找到之前正常的部署 ID (例如: dp_abc123)

# 3. 回滚到该版本
modal app set-default manga-translator --deployment-id=dp_abc123

# 4. 验证
python deploy/smoke_test.py --url https://YOUR-URL
```

---

## 🛠️ 高级用法

### 本地测试模型下载
```bash
# 在本地测试模型下载（不使用 Modal）
python deploy/prepare_models.py --test-local

# 验证模型
python deploy/prepare_models.py --verify --list
```

### 自定义配置
```python
# 编辑 deploy/modal_config.py

# 修改 GPU 类型
GPU_CONFIG["gpu"] = "A100"

# 修改容器空闲超时
GPU_CONFIG["container_idle_timeout"] = 600  # 10分钟

# 修改并发限制
GPU_CONFIG["allow_concurrent_inputs"] = 4

# 重新部署
modal deploy deploy/modal_app.py
```

### 开发调试
```bash
# 运行特定功能测试
modal run deploy/modal_app.py::download_models
modal run deploy/modal_app.py::list_volumes
modal run deploy/modal_app.py::cleanup_old_results

# 本地运行 smoke test
python deploy/smoke_test.py \
  --url http://localhost:5003 \
  --verbose
```

---

## 📚 可用工具

### 自动化脚本 (`deploy.sh`)
```bash
./deploy/deploy.sh setup      # 初始化 Secrets
./deploy/deploy.sh deploy     # 部署应用
./deploy/deploy.sh models     # 下载模型
./deploy/deploy.sh test       # 运行测试
./deploy/deploy.sh logs       # 查看日志
./deploy/deploy.sh cleanup    # 清理结果
./deploy/deploy.sh help       # 显示帮助
```

### 模型工具 (`prepare_models.py`)
```bash
python deploy/prepare_models.py --test-local   # 本地测试
python deploy/prepare_models.py --verify       # 验证模型
python deploy/prepare_models.py --list         # 列出模型
```

### 测试工具 (`smoke_test.py`)
```bash
python deploy/smoke_test.py --url URL          # 完整测试
python deploy/smoke_test.py --url URL --test health     # 特定测试
python deploy/smoke_test.py --url URL --verbose         # 详细输出
```

---

## 📖 环境变量参考

完整的环境变量列表见 `.env.modal.example`。

**必需**:
```bash
MT_WEB_NONCE=...  # 内部认证（自动生成）
```

**可选翻译服务**（按需配置）:
```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# DeepL
DEEPL_AUTH_KEY=...

# Gemini
GEMINI_API_KEY=...

# 更多服务见 .env.modal.example
```

---

## ❓ 常见问题

### Q: 我需要 GPU 吗？
A: 是的，推理需要 GPU。Modal 自动提供 GPU，你只需选择型号（T4/A10G/A100）。

### Q: 可以不用任何翻译 API 吗？
A: 可以！使用 `"translator": "none"` 仅进行 OCR 和文本检测，或使用离线翻译模型。

### Q: 冷启动有多慢？
A: 30-45秒（首次请求时加载模型）。后续请求快得多（10-30秒）。

### Q: 如何减少成本？
A: 使用 T4 GPU + keep_warm=0（按需启动）+ 定期清理结果。

### Q: 支持批量翻译吗？
A: 支持！使用 `/translate/batch/*` 端点一次处理多张图片。

### Q: 如何更新代码？
A: 修改后运行 `modal deploy deploy/modal_app.py` 即可，零停机部署。

### Q: 数据会保留吗？
A: 模型和结果存储在 Volume 中，部署更新不会丢失。

### Q: 可以自定义模型吗？
A: 可以，修改 `modal_config.py` 中的 `DEFAULT_MODELS_TO_DOWNLOAD`。

---

## 🆘 获取帮助

1. **查看日志**: `modal logs manga-translator --follow`
2. **运行测试**: `python deploy/smoke_test.py --url YOUR-URL --verbose`
3. **检查健康**: `curl https://YOUR-URL/health`
4. **Modal 文档**: https://modal.com/docs
5. **Modal 社区**: https://discord.gg/modal
6. **项目 Issues**: https://github.com/zyddnys/manga-image-translator/issues

---

## 📄 许可证

遵循主项目的许可证。

---

**部署完成！** 享受你的 serverless 漫画翻译服务！ 🎉📚✨

**版本**: 2.0
**更新**: 2024-04
**状态**: ✅ 生产就绪
