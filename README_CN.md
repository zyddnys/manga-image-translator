# Manga Image Translator · 中文说明

> 中文说明与 [项目首页](README.md) 保持同步。根 README 已默认使用中文，并包含完整功能、架构与部署指南。

MIT Reader 是一套自托管漫画翻译与阅读系统，包含 Android App、FastAPI 服务层、GPU 翻译引擎、MySQL / Redis 与云同步。

## 核心能力

- EPUB、MOBI、AZW/AZW3、CBZ、CBR、ZIP、RAR 与图片文件夹导入；
- 合集自动拆分、源文件与页内容去重；
- 书库、收藏夹、搜索、排序、网格/列表、任务进度与状态角标；
- 日漫 RTL 与普通 LTR 阅读器；
- 双击放大、双指缩放、单指平移、惯性滑动、边缘翻页；
- 音量键翻页与底部菜单热区；
- 单页翻译、自动预翻、全书后台队列、失败重试与停止；
- 按内容 hash 的多设备云同步与阅读进度接续；
- 云端书点击下载还原，离线保留最后一次在线快照；
- 局域网设备对传，支持断点续传与 SHA-256 校验；
- 手机遥控平板阅读，6 位配对码、token 鉴权、单设备绑定；
- 内部存储 / 机身存储 / SD 卡 App 专用目录；
- 自托管 APK 自动更新。

## 推荐阅读

- [项目首页与快速开始](README.md)
- [Android App 构建与功能说明](android/README.md)
- [后端 API](app_api/API.md)
- [Docker 部署](DOCKER_SETUP.md)
- [更新日志](CHANGELOG_CN.md)

## 许可证

[GPL-3.0](LICENSE)
