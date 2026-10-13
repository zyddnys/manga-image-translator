# Manga Image Translator

<div align="center">

**把生肉漫画放进口袋，把翻译引擎放在自己的服务器上。**

Android 漫画阅读器 · 自建翻译后端 · 云端书库 · 多设备进度同步 · 局域网遥控阅读

[![CI](https://github.com/Dovahlore/manga-raw-reader-app/actions/workflows/ci.yml/badge.svg)](https://github.com/Dovahlore/manga-raw-reader-app/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Dovahlore/manga-raw-reader-app?label=release)](https://github.com/Dovahlore/manga-raw-reader-app/releases)
[![Platform](https://img.shields.io/badge/platform-Android%208.0%2B-3DDC84?logo=android&logoColor=white)](android/README.md)
[![License](https://img.shields.io/github/license/Dovahlore/manga-raw-reader-app)](LICENSE)
[![Discord](https://img.shields.io/discord/739305951085199490?logo=discord&label=discord&logoColor=white)](https://discord.gg/Ak8APNy4vb)

[功能总览](#功能总览) · [快速开始](#快速开始) · [架构](#架构) · [Android 文档](android/README.md) · [API 文档](app_api/API.md)

</div>

---

## 这是一个什么项目？

Manga Image Translator 是一套可以完全自托管的漫画翻译和阅读系统：

1. 手机 App 导入 EPUB、MOBI、CBZ、CBR、ZIP、RAR 或图片文件夹；
2. 后端把每页图片交给检测、OCR、抹字、嵌字和翻译模型；
3. App 在阅读器里直接显示译文页，也能随时切回原图；
4. 书籍、译文、阅读进度、收藏夹和任务状态同步到你自己的服务器；
5. 多台设备共享同一个书架，平板阅读、手机遥控、断点续传。

项目 Fork 自 [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)，保留并复用其成熟翻译管线，在此之上新增了 Android 阅读器、任务型后端、账号隔离、云同步和设备协作能力。

> **翻译在服务器跑，手机只负责展示和翻页。** 老手机也能阅读复杂页面的大图，不会把电量耗在模型推理上。

## 核心亮点

- 📖 **真漫画阅读器**：RTL 日漫模式、LTR 普通模式、缩放平移、边缘翻页、译文/原图一键切换、音量键翻页。
- 🧠 **完整翻译管线**：文字检测、OCR、气泡分割、背景修复、译文排版和多语言翻译模型都在服务端完成。
- ☁️ **自托管云书库**：一个 API Key 就是一个账号，不同 Key 的书架完全隔离。
- 🔄 **多设备接续阅读**：按内容 hash 识别同一本书，阅读进度最后写入者胜，换机后可继续读。
- 📶 **局域网设备对传**：NSD 自动发现设备，分片断点续传、SHA-256 校验、原子导入。
- 🎮 **手机遥控平板**：6 位配对码、token 鉴权、心跳保活、单遥控器绑定，躺床上用音量键翻页。
- 🧱 **App 级任务系统**：扫描、导入、同步、全书翻译、下载和搬运不会因为离开页面而中断。
- 💾 **内部存储 / SD 卡**：新导入可选择写入位置，旧位置可手动搬运，书库继续聚合显示。

## 功能总览

| 能力 | 说明 |
| --- | --- |
| 导入格式 | EPUB、MOBI、AZW/AZW3、CBZ、CBR、ZIP、RAR、单层图片文件夹 |
| 合集处理 | 自动穿透包装目录，把“单层图片文件夹”拆成多本书 |
| 去重 | 源文件 SHA-256、页内容 hash、页面指纹，防止重复入库和重复翻译 |
| 书库视图 | 网格/列表、全部/本地/云端筛选、搜索、排序、收藏夹、继续阅读 |
| 状态角标 | 仅本地、仅云端、已同步、翻译中、排队中、失败、已完成 |
| 阅读方向 | 日漫 RTL、普通 LTR，逻辑页码与视觉方向解耦 |
| 阅读手势 | 双击放大、双指缩放、单指平移、惯性滑动、边缘外滑翻页、底部菜单热区 |
| 音量键 | 阅读器内直接上/下一页；遥控器页内远控另一台设备 |
| 单页翻译 | 当前页手动翻译，已完成页可强制重翻 |
| 全书翻译 | App 级后台队列，可停止、可恢复、可查看失败页 |
| 自动预翻 | 自动预翻译当前页之后 3 页，减少等待 |
| 云同步 | 原书包、页面索引、译文状态、收藏夹、阅读进度 |
| 离线体验 | 本地书正常阅读；云端书和翻译进度保留最后一次在线快照 |
| 仅云端书 | 点击卡片或列表自动下载还原；离线时明确拦截 |
| 设备对传 | 同局域网免流量传书，支持断点续传和校验 |
| 遥控器 | Wi-Fi 局域网配对，一台被控端只允许一台遥控器，角色互斥 |
| 自动更新 | 后端维护 APK 发布表，App 内检查、下载、校验并安装 |
| 存储位置 | App 专用内部存储、机身共享存储、SD 卡 App 专用目录 |
| 任务恢复 | 进程重启后恢复中断导入、扫描、翻译和存储搬运状态 |

## 阅读器细节

- 翻到已翻译页时直接显示译文，不先闪原图。
- 菜单隐藏时，单击图片切换译文/原文；双击放大或复位。
- 菜单显示时，点击正文或菜单空白处直接隐藏菜单。
- 菜单隐藏时，点击底部热区呼出菜单。
- 放大后图片 1:1 跟手平移，滑到边缘继续外滑可翻页。
- 阅读进度本地立即保存，并防抖同步到云端。
- 全书翻译进行中，阅读器会周期性补拉新完成的译文页。

## 云同步与离线策略

- **云端保存什么**：原书包、页面索引、任务状态、翻译结果、收藏夹、阅读进度。
- **怎么识别同一本书**：优先内容 hash，兼容云端书 ID，避免同一本书因导入来源不同而重复。
- **离线时能看到什么**：本地书完整可读；云端书列表和翻译进度使用最后一次在线同步快照。
- **重新联网后**：后台同步成功会用服务器最新状态覆盖本地快照。
- **仅云端书离线时**：可以看见元数据和进度，但点击下载会被拦截并提示需要联网。

## 架构

```text
┌────────────────────────────┐
│ Android App (Kotlin/Compose) │
│ 阅读器 · 书库 · 导入 · 遥控器 │
└────────────┬───────────────┘
             │ HTTP / JSON / Multipart
             ▼
┌────────────────────────────┐
│ app-api (FastAPI, :8020)    │
│ 账号 · 书籍 · 任务 · 云同步 │
└──┬──────────┬──────────┬───┘
   │          │          │
   ▼          ▼          ▼
 MySQL       Redis      Disk
 元数据/进度  结果缓存   原图/译文/云端书
                          │
                          ▼
              ┌────────────────────┐
              │ translation engine │
              │ 检测 · OCR · 修复 · 嵌字 │
              └────────────────────┘
```

## 快速开始

### 1. 准备服务器

推荐配置：

- NVIDIA GPU 与可用的 CUDA Runtime；
- Docker Desktop 或 Docker Engine；
- 足够存放模型与书页的磁盘空间；
- Android 设备与服务器在同一局域网；公网访问建议使用 VPN 或 HTTPS 反代。

### 2. 启动后端

```bash
git clone https://github.com/Dovahlore/manga-raw-reader-app.git
cd manga-image-translator

cp app.env.example app.env
# 编辑 app.env：设置数据库密码、Redis 密码和 MIT_API_TOKEN

cat > secret.env <<'EOF'
DEEPSEEK_API_KEY=your-key
EOF

docker compose --env-file app.env -f docker-compose.full.yml up -d
```

首次启动会下载模型和镜像，耗时取决于网络与 GPU 环境。检查服务：

```bash
docker compose --env-file app.env -f docker-compose.full.yml ps
curl http://127.0.0.1:8020/v1/health
```

### 3. 构建 Android App

```bash
cd android
gradle assembleRelease --no-daemon
```

Windows 环境也可以按 [android/README.md](android/README.md) 使用 JDK 17、Android SDK 和 Gradle 8.7 构建。

### 4. 连接 App 与服务器

1. 安装 APK；
2. 打开 App 的「设置」；
3. 服务器地址填写 `http://<服务器局域网IP>:8020`；
4. API Key 填写 `app.env` 里的 `MIT_API_TOKEN`；
5. 点击「测试连接」。

> 不要把真实 API Key 写进 README、截图或 Issue。

## 配置速查

| 配置 | 说明 |
| --- | --- |
| `MIT_API_TOKEN` | App API Key；支持逗号分隔多账号；留空则不鉴权，仅建议隔离内网使用 |
| `MIT_TRANSLATOR` | 默认翻译器，例如 `deepseek` |
| `MIT_TARGET_LANG` | 默认目标语言，例如 `CHS` |
| `MIT_CONTEXT_PAGES` | 跨页上下文页数 |
| `MIT_CACHE_TTL` | 结构化结果缓存 TTL，`0` 表示永不过期 |
| `MIT_IMAGE_COMPRESS` | 译文图 WebP 压缩开关 |
| `MIT_RUNTIME` | 运行期数据目录 |
| `MIT_APP_PORT` | app-api 暴露端口 |
| `MIT_WEB_PORT` | 引擎 Web 端口 |

更多变量见 [`app.env.example`](app.env.example)。

## 安全建议

- 公网部署必须开启 `MIT_API_TOKEN`，并使用 HTTPS 反代或 VPN。
- 为 MySQL 和 Redis 设置强密码，不要把数据库端口暴露到公网。
- `app.env`、`secret.env`、签名密钥和真实截图不要提交到 Git。
- 多用户场景使用不同 API Key，数据会按 Key 隔离。
- 云端书和翻译结果默认存放在服务器运行期目录，请自行备份。

## 项目结构

```text
android/                    Android App
app_api/                    FastAPI 服务层：账号、任务、云同步、自动更新
manga_translator/           翻译管线：检测、OCR、修复、嵌字、上色、超分
server/                     引擎 HTTP 服务
docker-compose.full.yml     engine + app-api + MySQL + Redis
docker-compose.tunnel.yml   可选 frp 隧道
dict/                       术语表与替换字典
```

## 文档

- [Android App 说明](android/README.md)
- [后端 API 速查](app_api/API.md)
- [Docker 部署](DOCKER_SETUP.md)
- [更新日志](CHANGELOG.md)

## 开发路线

- [ ] 蓝牙遥控传输层
- [ ] 阅读器页面预取策略调优
- [ ] 更完整的译文校对工作台
- [ ] 服务端任务指标与告警
- [ ] Web 端书库管理
- [ ] 自动备份与恢复向导

## 贡献

欢迎 Issue 和 PR。提交前请：

1. 保持改动聚焦；
2. 补充必要的最小复现信息；
3. 不要提交密钥、模型、运行期数据和私人书籍；
4. Kotlin / Python 代码尽量遵循现有风格。

## 许可证

[GPL-3.0](LICENSE)

## 致谢

本项目基于 [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) 继续开发，感谢上游作者和所有模型、依赖项目的贡献者。

<div align="center">

如果这个项目让你的漫画阅读更轻松，欢迎点一个 Star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Dovahlore/manga-raw-reader-app&type=Date)](https://star-history.com/#Dovahlore/manga-raw-reader-app&Date)

</div>
