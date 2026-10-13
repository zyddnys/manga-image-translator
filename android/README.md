# MIT Reader · Android App

MIT Reader 是 Manga Image Translator 的 Android 客户端，使用 Kotlin 与 Jetpack Compose 开发，负责导入、书库、阅读、后台任务状态和跨设备协作。

## 系统要求

- Android 8.0（API 26）及以上；
- Android Studio Ladybug 或更新版本；
- JDK 17；
- Android SDK 34；
- Gradle 8.7；
- 如需访问自建后端，设备与服务器应在同一局域网，或通过 VPN / HTTPS 反代访问。

仓库当前未提交 Gradle Wrapper。命令行构建需要本机安装 Gradle 8.7，或直接使用 Android Studio 管理 Gradle。

## 功能

### 导入与扫描

- 支持 EPUB、MOBI、AZW/AZW3、CBZ、CBR、ZIP、RAR；
- 支持系统文件选择器选择单层图片文件夹；
- 自动识别合集包中的多本书；
- 按源文件 hash、页内容 hash 和页面指纹去重；
- 支持指定书库文件夹并自动扫描；
- 扫描、导入、翻译和存储操作使用 App 级队列，退出页面不中断；
- 进程重启后恢复未完成的导入和扫描。

### 书库

- 本地书、云端书、已同步书统一聚合显示；
- 网格 / 列表视图；
- 全部 / 本地 / 云端筛选；
- 名称、最近阅读、创建时间排序；
- 搜索书名与收藏夹；
- 收藏夹展开、收起、移动、重命名；
- 显示仅本地、仅云端、已同步、翻译中、排队中、失败等状态；
- 点击仅云端书自动下载还原，离线时拦截并提示。

### 阅读器

- 日漫 RTL 与普通 LTR；
- 跳页滑块与页码显示；
- 菜单隐藏时，单击图片切换译文 / 原图；
- 双击放大 / 复位；
- 双指缩放，最大 5 倍；
- 放大后单指 1:1 平移，支持惯性滑动；
- 放大到边缘继续外滑翻页；
- 音量上 = 上一页，音量下 = 下一页；
- 菜单隐藏时点击底部热区呼出菜单；
- 菜单显示时点击正文或菜单空白处隐藏；
- 自动预翻当前页之后 3 页；
- 阅读进度本地即时记录并同步云端。

### 云同步

- 原书包、页面索引、翻译状态、收藏夹和阅读进度同步；
- 以内容 hash 识别同一本书，减少换机或重复导入造成的重复项；
- 云端书可下载还原，译文页按需补拉；
- 离线时保留最后一次在线成功的云端书列表和翻译进度快照；
- 重新联网后由最新同步结果覆盖快照。

### 设备对传

- 同一局域网自动发现设备；
- 传输前生成清单并检查空间；
- 支持断点续传；
- 每个文件 SHA-256 校验；
- 校验通过后原子导入，失败不会污染书库。

### 遥控器

一台设备作为阅读端，另一台设备作为遥控器：

- 阅读端在阅读器菜单中开启被控模式；
- 遥控端从书库左下角进入遥控器模式；
- 使用 6 位配对码和 token 鉴权；
- 心跳保活，异常断开自动清理；
- 同一阅读端只允许一台遥控器；
- 遥控器与被控端角色互斥；
- 仅阅读端停留在阅读器前台时接受翻页命令；
- 遥控器页内音量上 / 下分别控制上一页 / 下一页；
- 支持屏幕按钮翻页和连接状态显示。

### 存储与更新

- 可选择 App 专用内部存储、机身共享存储或 SD 卡 App 专用目录；
- 新导入按当前设置写入；
- 旧位置数据可通过设置手动搬运；
- 搬运期间书库保持可见，任务可恢复；
- App 支持从自建后端检查更新、下载 APK、校验并安装。

## 构建

### Android Studio

1. 用 Android Studio 打开 `android/` 目录；
2. 等待 Gradle Sync；
3. 选择设备或模拟器；
4. 点击 Run。

### 命令行

Debug 构建：

```bash
cd android
gradle assembleDebug --no-daemon
```

Release 构建：

```bash
cd android
gradle assembleRelease --no-daemon
```

Release 签名配置读取 `android/app/keystore.properties`：

```properties
storeFile=../your-release-key.jks
storePassword=your-store-password
keyAlias=your-alias
keyPassword=your-key-password
```

该文件与密钥不要提交到 Git。

## 连接后端

1. 启动项目根目录的后端；
2. 打开 App「设置」；
3. 填写服务器地址，例如 `http://192.168.x.x:8020`；
4. 填写与后端一致的 API Key；
5. 点击「测试连接」。

Android 模拟器访问本机后端时可使用 `http://10.0.2.2:8020`。

## 调试

查看连接设备：

```bash
adb devices
```

安装 Debug 包：

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

查看日志：

```bash
adb logcat --pid=$(adb shell pidof -s com.mit.reader)
```

## 主要目录

```text
android/app/src/main/java/com/mit/reader/
  MainActivity.kt     入口与音量键分发
  ReaderApp.kt        App 级任务、同步、缓存、更新、遥控状态
  ReaderViewModel.kt  阅读状态与页面翻译
  data/               导入解析、书库仓库、网络 API、传输与遥控
  ui/                 书库、阅读器、设置、进度、遥控器界面
```

## 安全提醒

- 不要把真实服务器地址、API Key、书名或截图提交到公开仓库；
- 公网访问请使用 HTTPS 或 VPN；
- Release 密钥只保存在本机；
- App 权限仅覆盖网络、存储访问和系统打开文件等实际需求。
