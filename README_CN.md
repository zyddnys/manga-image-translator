# 漫画/图片翻译器 (中文说明) 
最后更新时间：2025年3月11日
---
![Commit activity](https://img.shields.io/github/commit-activity/m/zyddnys/manga-image-translator)
![Lines of code](https://img.shields.io/tokei/lines/github/zyddnys/manga-image-translator?label=lines%20of%20code)
![License](https://img.shields.io/github/license/zyddnys/manga-image-translator)
![Contributors](https://img.shields.io/github/contributors/zyddnys/manga-image-translator)
[![Discord](https://img.shields.io/discord/739305951085199490?logo=discord&label=discord&logoColor=white)](https://discord.gg/Ak8APNy4vb)


> 一键翻译各类图片内文字\
> [English](README.md) | [更新日志](CHANGELOG_CN.md) \
> 欢迎加入我们的 Discord <https://discord.gg/Ak8APNy4vb>

本项目旨在翻译那些不太可能有人专门翻译的图片，例如各种群聊、图站上的漫画/图片，让像我这样的日语小白也能大致理解图片内容。\
主要支持日语，同时也支持简繁中文、英文及其他20种小语言。\
支持图片修复（去字）和嵌字。\
该项目是[求闻转译志](https://github.com/PatchyVideo/MMDOCR-HighPerformance)的 v2 版本。

**注意：本项目仍处于早期开发阶段，存在许多不足，我们需要您的帮助来完善它！**

## 目录

*   [在线版本](#在线版本)
*   [本地安装与使用](#本地安装与使用)
    *   [使用 Pip/venv (推荐)](#使用-pipvenv-推荐)
    *   [Windows 用户注意事项](#windows-用户注意事项)
    *   [Docker](#docker)
        *   [运行 Web 服务器](#运行-web-服务器)
        *   [作为 CLI 使用](#作为-cli-使用)
        *   [设置翻译服务的密钥](#设置翻译服务的密钥)
        *   [使用 Nvidia GPU](#使用-nvidia-gpu)
*   [命令行用法](#命令行用法)
*   [效果图](#效果图)
*   [选项(Options)及配置](#选项options及配置)
    *   [推荐模块](#推荐模块)
        *   [提升翻译质量的技巧](#提升翻译质量的技巧)
    *   [详细选项](#详细选项)
        *   [基本选项](#基本选项)
        *    [附加选项](#附加选项)
    *   [语言代码参考](#语言代码参考)
    *   [翻译器参考](#翻译器参考)
    *    [术语表](#术语表)
    *   [环境变量汇总](#环境变量汇总)
    *   [配置文件](#配置文件)
    *   [GPT 配置参考](#gpt-配置参考)
    *   [使用 Gimp 进行渲染](#使用-gimp-进行渲染)
    *   [API 文档](#api-文档)
*   [后续计划](#后续计划)
*   [支持我们](#支持我们)
    *   [感谢所有贡献者](#感谢所有贡献者)
*   [Star 增长曲线](#star-增长曲线)

## 在线版

官方演示站 (由 zyddnys 维护)： <https://touhou.ai/imgtrans/>\
浏览器脚本 (由 QiroNT 维护): <https://greasyfork.org/scripts/437569>

- 注意：如果在线版无法访问，可能是因为 Google GCP 正在重启服务器，请稍等片刻，等待服务重启。
- 在线版使用的是目前 main 分支的最新版本。

## 本地安装与使用

### 使用 Pip/venv (推荐)

```bash
# 首先，确保您的机器安装了 Python 3.10 或更高版本
# 最新版本的 Python 可能尚未与某些 PyTorch 库兼容
$ python --version
Python 3.10.6

# 克隆本仓库
$ git clone https://github.com/zyddnys/manga-image-translator.git

# 创建 venv (可选，但建议)
$ python -m venv venv

# 激活 venv
$ source venv/bin/activate

# 如果要使用 --use-gpu 选项，请访问 https://pytorch.org/ 并按照说明安装 PyTorch。
# 在 pip 命令中添加 --upgrade --force-reinstall 以覆盖当前安装的 PyTorch 版本。

# 安装依赖
$ pip install -r requirements.txt
```

模型将在运行时自动下载到 `./models` 目录。

**Windows 用户注意事项：**

在执行 pip install 之前，请先安装 Microsoft C++ Build Tools ([下载](https://visualstudio.microsoft.com/vs/), [安装说明](https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools))，因为某些 pip 依赖项需要它才能编译。 (参见 [#114](https://github.com/zyddnys/manga-image-translator/issues/114))。

要在 Windows 上使用 [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)，请按照 <https://pytorch.org/> 上的说明安装正确的 PyTorch 版本。

### Docker

要求：

- Docker (使用 CUDA / GPU 加速需要 19.03+ 版本)
- Docker Compose (可选，如果您想使用 `demo/doc` 文件夹中的文件)
- Nvidia Container Runtime (可选，如果您想使用 CUDA)

本项目支持 Docker，镜像为 `zyddnys/manga-image-translator:main`。
此 Docker 镜像包含项目所需的所有依赖项和模型。
请注意，此镜像相当大（约 15GB）。

#### 运行 Web 服务器

可以使用以下命令启动 Web 服务器 (CPU)：

```bash
docker run -p 5003:5003 -v result:/app/result --ipc=host --rm zyddnys/manga-image-translator:main -l CHS --manga2eng -v --mode web --host=0.0.0.0 --port=5003
```

或者

```bash
docker-compose -f demo/doc/docker-compose-web-with-cpu.yml up
```

Web 服务器将在 [8000](http://localhost:8000) 端口启动，<s>翻译结果将保存在 `/result` 文件夹中。</s> 新版中翻译结果不再保存

#### 作为 CLI 使用

要通过 CLI 使用 Docker (即批量模式)：

```bash
docker run -v <targetFolder>:/app/<targetFolder> -v <targetFolder>-translated:/app/<targetFolder>-translated  --ipc=host --rm zyddnys/manga-image-translator:main --mode=batch -i=/app/<targetFolder> <cli flags>
```

**注意:** 如果您需要引用主机上的文件，则需要将相关文件作为卷挂载到容器内的 `/app` 文件夹中。CLI 的路径需要是内部 Docker 路径 `/app/...`，而不是主机上的路径。

#### 设置翻译服务的密钥

某些翻译服务需要 API 密钥才能运行，请将它们作为环境变量传递到 Docker 容器中。例如：

```bash
docker run --env="DEEPL_AUTH_KEY=xxx" --ipc=host --rm zyddnys/manga-image-translator:main <cli flags>
```

#### 使用 Nvidia GPU

> 要使用受支持的 GPU，请先阅读前面的 `Docker` 部分。您需要一些特殊的依赖项。

运行容器时，请设置以下标志：

```bash
docker run ... --gpus=all ... zyddnys/manga-image-translator:main ... --use-gpu
```

或者 (对于 Web 服务器 + GPU)：

```bash
docker-compose -f demo/doc/docker-compose-web-with-gpu.yml up
```
### 命令行用法
本地模式
```bash
# 将 <path> 替换为图片文件夹或文件的路径。
$ python -m manga_translator local -v -i <path>
# 结果可以在 `<path_to_image_folder>-translated` 中找到。
```
### 网页模式

```bash
#使用 `--mode web`启动网页服务器.
$ cd server && python main.py --use-gpu
# 网页demo服务地址为http://127.0.0.1:8000
```

## 效果图

以下样例可能并未经常更新，可能不能代表当前主分支版本的效果。

<table>
  <thead>
    <tr>
      <th align="center" width="50%">原始图片</th>
      <th align="center" width="50%">翻译后图片</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" width="50%">
        <a href="https://user-images.githubusercontent.com/31543482/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png">
          <img alt="佐藤さんは知っていた - 猫麦" src="https://user-images.githubusercontent.com/31543482/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png" />
        </a>
        <br />
        <a href="https://twitter.com/09ra_19ra/status/1647079591109103617/photo/1">(Source @09ra_19ra)</a>
      </td>
      <td align="center" width="50%">
        <a href="https://user-images.githubusercontent.com/31543482/232265339-514c843a-0541-4a24-b3bc-1efa6915f757.png">
          <img alt="Output" src="https://user-images.githubusercontent.com/31543482/232265339-514c843a-0541-4a24-b3bc-1efa6915f757.png" />
        </a>
        <br />
        <a href="https://user-images.githubusercontent.com/31543482/232265376-01a4557d-8120-4b6b-b062-f271df177770.png">(Mask)</a>
      </td>
    </tr>
    <tr>
      <td align="center" width="50%">
        <a href="https://user-images.githubusercontent.com/31543482/232265479-a15c43b5-0f00-489c-9b04-5dfbcd48c432.png">
          <img alt="Gris finds out she's of royal blood - VERTI" src="https://user-images.githubusercontent.com/31543482/232265479-a15c43b5-0f00-489c-9b04-5dfbcd48c432.png" />
        </a>
        <br />
        <a href="https://twitter.com/VERTIGRIS_ART/status/1644365184142647300/photo/1">(Source @VERTIGRIS_ART)</a>
      </td>
      <td align="center" width="50%">
        <a href="https://user-images.githubusercontent.com/31543482/232265480-f8ba7a28-846f-46e7-8041-3dcb1afe3f67.png">
          <img alt="Output" src="https://user-images.githubusercontent.com/31543482/232265480-f8ba7a28-846f-46e7-8041-3dcb1afe3f67.png" />
        </a>
        <br />
        <code>--detector ctd</code>
        <a href="https://user-images.githubusercontent.com/31543482/232265483-99ad20af-dca8-4b78-90f9-a6599eb0e70b.png">(Mask)</a>
      </td>
    </tr>
    <tr>
      <td align="center" width="50%">
        <a href="https://user-images.githubusercontent.com/31543482/232264684-5a7bcf8e-707b-4925-86b0-4212382f1680.png">
          <img alt="陰キャお嬢様の新学期🏫📔🌸 (#3) - ひづき夜宵🎀💜" src="https://user-images.githubusercontent.com/31543482/232264684-5a7bcf8e-707b-4925-86b0-4212382f1680.png" />
        </a>
        <br />
        <a href="https://twitter.com/hiduki_yayoi/status/1645186427712573440/photo/2">(Source @hiduki_yayoi)</a>
      </td>
      <td align="center" width="50%">
        <a href="https://user-images.githubusercontent.com/31543482/232264644-39db36c8-a8d9-4009-823d-bf85ca0609bf.png">
          <img alt="Output" src="https://user-images.githubusercontent.com/31543482/232264644-39db36c8-a8d9-4009-823d-bf85ca0609bf.png" />
        </a>
        <br />
        <code>--translator none</code>
        <a href="https://user-images.githubusercontent.com/31543482/232264671-bc8dd9d0-8675-4c6d-8f86-0d5b7a342233.png">(Mask)</a>
      </td>
    </tr>
    <tr>
      <td align="center" width="50%">
        <a href="https://user-images.githubusercontent.com/31543482/232265794-5ea8a0cb-42fe-4438-80b7-3bf7eaf0ff2c.png">
          <img alt="幼なじみの高校デビューの癖がすごい (#1) - 神吉李花☪️🐧" src="https://user-images.githubusercontent.com/31543482/232265794-5ea8a0cb-42fe-4438-80b7-3bf7eaf0ff2c.png" />
        </a>
        <br />
        <a href="https://twitter.com/rikak/status/1642727617886556160/photo/1">(Source @rikak)</a>
      </td>
      <td align="center" width="50%">
        <a href="https://user-images.githubusercontent.com/31543482/232265795-4bc47589-fd97-4073-8cf4-82ae216a88bc.png">
          <img alt="Output" src="https://user-images.githubusercontent.com/31543482/232265795-4bc47589-fd97-4073-8cf4-82ae216a88bc.png" />
        </a>
        <br />
        <a href="https://user-images.githubusercontent.com/31543482/232265800-6bdc7973-41fe-4d7e-a554-98ea7ca7a137.png">(Mask)</a>
      </td>
    </tr>
  </tbody>
</table>

## 选项(Options)及配置
### 推荐模块

检测器 (Detector)：

- 英语：??
- 日语：??
- 中文 (简体)：??
- 韩语：??
- 使用 `{"detector":{"detector": "ctd"}}` 可以增加检测到的文本行数
更新：实测default在黑白漫画中搭配相关参数调整后效果更佳

OCR：

- 英语：??
- 日语：48px
- 中文 (简体)：??
- 韩语：48px

翻译器 (Translator)：

- 日语 -> 英语：**Sugoi**
- 中文 (简体) -> 英语：??
- 中文 (简体) -> 日语：??
- 日语 -> 中文 (简体)：sakura 或 opanai
- 英语 -> 日语：??
- 英语 -> 中文 (简体)：??

修补器 (Inpainter)：lama_large

着色器 (Colorizer)：**mc2**

### 提升翻译质量的技巧

-   低分辨率有时会让检测器出错，它不太擅长识别不规则的文本大小。为了解决这个问题，您可以使用 `--upscale-ratio 2` 或任何其他值来使用放大器
-   如果渲染的文本太小而无法阅读，请指定 `--font-size-minimum 30` 或使用 `--manga2eng` （注意：只在目标语言为英文时使用）渲染器，它将尝试适应检测到的文本气泡
-   使用 `--font-path fonts/anime_ace_3.ttf` 指定字体
-   设置 `mask_dilation_offset` 20~40，增大掩膜覆盖范围，更好包裹源文字
-   使用 `lama_large` 作为修补器。
-   增加 `box_threshold` 可以在一定程度上帮助过滤掉由 OCR 错误检测引起的乱码
-   使用 `OpenaiTranslator` 加载术语表文件（`custom_openai`无法加载）
-   图片分辨率较大时请调高 `inpainting_size`, 否则可能导致文字修复时像素无法完全遮盖掩膜以致源文漏出。其他情况可调高 `kernal_size` 以降低涂字精度使模型获取更大视野（注:根据源文和译文的一致性判断是否是由于文字修复导致的文字漏涂，如一致则是文字修复导致的，不一致则是文本检测和OCR导致的）


### 详细选项

#### 基本选项

```text
-h, --help                     显示此帮助信息并退出
-v, --verbose                  打印调试信息并将中间图像保存在结果文件夹中
--attempts ATTEMPTS            遇到错误时的重试次数。-1 表示无限次。
--ignore-errors                遇到错误时跳过图像。
--model-dir MODEL_DIR          模型目录（默认为项目根目录下的 ./models）
--use-gpu                      打开/关闭 GPU（在 mps 和 cuda 之间自动切换）
--use-gpu-limited              打开/关闭 GPU（不包括离线翻译器）
--font-path FONT_PATH          字体文件路径
--pre-dict PRE_DICT            翻译前替换字典文件路径
--post-dict POST_DICT          翻译后替换字典文件路径
--kernel-size KERNEL_SIZE      设置文本擦除区域的卷积内核大小以完全清除文本残留
```
#### 附加选项
##### Batch 模式选项

```text
local                         以批量翻译模式运行
-i, --input INPUT [INPUT ...] 图像文件夹路径（必需）
-o, --dest DEST               翻译后图像的目标文件夹路径（默认：''）
-f, --format FORMAT           翻译的输出格式。选项：[在此处列出 OUTPUT_FORMATS, png,webp,jpg,jpeg,xcf,psd,pdf]
--overwrite                   覆盖已翻译的图像
--skip-no-text                跳过没有文本的图像（不会保存）。
--use-mtpe                    在命令行上打开/关闭机器翻译后期编辑（MTPE）（目前仅适用于 Linux）
--save-text                   将提取的文本和翻译保存到文本文件中。
--load-text                   从文本文件加载提取的文本和翻译。
--save-text-file SAVE_TEXT_FILE  类似于 --save-text，但具有指定的文件路径。（默认：''）
--prep-manual                 通过输出空白、修复的图像以及原始图像的副本以供参考，为手动排版做准备
--save-quality SAVE_QUALITY   保存的 JPEG 图像的质量，范围从 0 到 100，其中 100 为最佳（默认值：100）
--config-file CONFIG_FILE     配置文件的路径（默认值：None）                          
```

##### WebSocket 模式选项

```text
ws                  以 WebSocket 模式运行
--host HOST         WebSocket 服务的主机（默认：127.0.0.1）
--port PORT         WebSocket 服务的端口（默认：5003）
--nonce NONCE       用于保护内部 WebSocket 通信的 Nonce
--ws-url WS_URL     WebSocket 模式的服务器 URL（默认：ws://localhost:5000）
--models-ttl MODELS_TTL  上次使用后将模型保留在内存中的时间（秒）（0 表示永远）
```

##### API 模式选项

```text
shared              以 API 模式运行
--host HOST         API 服务的主机（默认：127.0.0.1）
--port PORT         API 服务的端口（默认：5003）
--nonce NONCE       用于保护内部 API 服务器通信的 Nonce
--report REPORT     向服务器报告以注册实例（默认：None）
--models-ttl MODELS_TTL  模型在内存中的 TTL（秒）（0 表示永远）
```

##### Web 模式选项（缺少一些基本选项，仍有待添加）

```text
--host HOST           主机地址（默认：127.0.0.1）
--port PORT           端口号（默认：8000）
--start-instance      是否应自动启动翻译器实例
--nonce NONCE         用于保护内部 Web 服务器通信的 Nonce
--models-ttl MODELS_TTL  模型在内存中的 TTL（秒）（0 表示永远）
```
##### config-help 模式
```bash
python -m manga_translator config-help
```
#### 语言代码参考

由配置中的 `translator/language` 使用

```yaml
CHS: 简体中文
CHT: 繁体中文
CSY: 捷克语
NLD: 荷兰语
ENG: 英语
FRA: 法语
DEU: 德语
HUN: 匈牙利语
ITA: 意大利语
JPN: 日语
KOR: 韩语
PLK: 波兰语
PTB: 葡萄牙语（巴西）
ROM: 罗马尼亚语
RUS: 俄语
ESP: 西班牙语
TRK: 土耳其语
UKR: 乌克兰语
VIN: 越南语
ARA: 阿拉伯语
SRP: 塞尔维亚语
HRV: 克罗地亚语
THA: 泰语
IND: 印度尼西亚语
FIL: 菲律宾语（他加禄语）
```

#### 翻译器参考
| 名称          | API Key | Offline | Note                                                     |
|---------------|---------|---------|----------------------------------------------------------|
| <s>google</s> |         |         | 暂时禁用                                                  |
| youdao        | ✔️      |         | 需要 `YOUDAO_APP_KEY` 和 `YOUDAO_SECRET_KEY`        |
| baidu         | ✔️      |         | 需要 `BAIDU_APP_ID` 和 `BAIDU_SECRET_KEY`           |
| deepl         | ✔️      |         | 需要 `DEEPL_AUTH_KEY`                                |
| caiyun        | ✔️      |         | 需要 `CAIYUN_TOKEN`                                  |
| openai        | ✔️      |         | Implements 需要 `OPENAI_API_KEY`                     |
| papago        |         |         |                                                          |
| sakura        |         |         | 需要 `SAKURA_API_BASE`                               |
| custom openai |         |         | 需要  `CUSTOM_OPENAI_API_BASE` `CUSTOM_OPENAI_MODEL` |
| offline       |         | ✔️      | 为语言选择最合适的离线翻译器    |
| sugoi         |         | ✔️      | Sugoi V4.0 模型                                        |
| m2m100        |         | ✔️      | 支持所有语言                                  |
| m2m100_big    |         | ✔️      |                                                          |
| none          |         | ✔️      | 翻译为空文本                                 |
| original      |         | ✔️      | 保留原始文本                                      |

-   API Key：依据翻译器是否需要将 API 密钥设置为环境变量。
为此，您可以在项目根目录中创建一个 .env 文件，其中包含您的 API 密钥，如下所示：

```env
OPENAI_API_KEY=sk-xxxxxxx...
DEEPL_AUTH_KEY=xxxxxxxx...
```

-   Offline：翻译器是否可以离线使用。

-   Sugoi 由 mingshiba 创建，请在 <https://www.patreon.com/mingshiba> 支持他

#### 术语表

-   mit_glossory: 向 AI 模型发送术语表以指导其翻译可以有效提高翻译质量，例如，确保专有名称和人名的一致翻译。它会自动从术语表中提取与待发送文本相关的有效条目，因此无需担心术语表中的大量条目会影响翻译质量。 （仅对 openaitranslator 有效，兼容 sakura_dict 和 galtransl_dict。）

-   sakura_dict: sakura 术语表，仅对 sakuratranslator 有效。 没有自动术语表功能。

```env
OPENAI_GLOSSARY_PATH=PATH_TO_YOUR_FILE
SAKURA_DICT_PATH=PATH_TO_YOUR_FILE
```

#### 环境变量汇总

| 环境变量名                     | 说明                                                                  | 默认值                               | 备注                                                                                               |
| :----------------------------- | :-------------------------------------------------------------------- | :----------------------------------- | :------------------------------------------------------------------------------------------------- |
| `BAIDU_APP_ID`                 | 百度翻译 appid                                                          | `''`                                 |                                                                                                    |
| `BAIDU_SECRET_KEY`             | 百度翻译密钥                                                            | `''`                                 |                                                                                                    |
| `YOUDAO_APP_KEY`               | 有道翻译应用 ID                                                          | `''`                                 |                                                                                                    |
| `YOUDAO_SECRET_KEY`            | 有道翻译应用秘钥                                                          | `''`                                 |                                                                                                    |
| `DEEPL_AUTH_KEY`              | DeepL 翻译 AUTH_KEY                                                       | `''`                                 |                                                                                                    |
| `OPENAI_API_KEY`              | OpenAI API 密钥                                                        | `''`                                 |                                                                                                    |
| `OPENAI_MODEL`                | OpenAI 模型 (可选)                                                      | `''`                                 |                                                                                                    |
| `OPENAI_HTTP_PROXY`           | OpenAI HTTP 代理 (可选)                                                | `''`                                 | 替代 `--proxy`                                                                                      |
| `OPENAI_GLOSSARY_PATH`        | OpenAI 术语表路径 (可选)                                                  | `./dict/mit_glossory.txt`            |                                                                                                    |
| `OPENAI_API_BASE`             | OpenAI API 基础地址 (可选)                                                | `https://api.openai.com/v1`          | 默认为官方地址                                                                                       |
|`GROQ_API_KEY`| Groq API 密钥 |||
| `SAKURA_API_BASE`             | SAKURA API 地址 (可选)                                                  | `http://127.0.0.1:8080/v1`           |                                                                                                    |
| `SAKURA_VERSION`               | SAKURA API 版本 (可选)                                                    | `'0.9'`                              | `0.9` 或 `0.10`                                                                                    |
| `SAKURA_DICT_PATH`            | SAKURA 术语表路径 (可选)                                                  | `./dict/sakura_dict.txt`             |                                                                                                    |
| `CAIYUN_TOKEN`                | 彩云小译 API 访问令牌                                                      | `''`                                 |                                                                                                    |
| `DEEPSEEK_API_KEY`           | DeepSeek API 密钥                                                         | `''`                                 |                                                                                                       |
| `DEEPSEEK_API_BASE`           | DeepSeek API 基础地址（可选）                                              |   `https://api.deepseek.com`                                                              |    |
| `CUSTOM_OPENAI_API_KEY`        | 自定义 OpenAI API 密钥 (Ollama 不需要，但其他工具可能需要)                   | `'ollama'`                            |                                                                                                    |
| `CUSTOM_OPENAI_API_BASE`       | 自定义 OpenAI API 基础地址 (使用 OLLAMA_HOST 环境变量更改绑定 IP 和端口)     | `http://localhost:11434/v1`          |                                                                                                    |
| `CUSTOM_OPENAI_MODEL`          | 自定义 OpenAI 模型 (例如 "qwen2.5:7b"，确保在使用前拉取并运行它)             | `''`                                 |                                                                                                    |
| `CUSTOM_OPENAI_MODEL_CONF`     | 例如 "qwen2"                                                              | `''` |                                                                                                       |


**使用说明：**

1.  **创建 `.env` 文件:** 在项目根目录下创建一个名为 `.env` 的文件。
2.  **复制粘贴:** 将上面的文本复制到 `.env` 文件中。
3.  **填写密钥:** 将 `''` 中的内容替换为你自己的 API 密钥、ID 等信息。

**重要提示：**

*   `.env` 文件包含敏感信息，请多加小心防止意外泄露。


#### 配置文件

运行 `python -m manga_translator config-help >> config-info.json`

可以在 example/config-example.json 中找到示例

(JSON 架构定义部分，与英文 README 保持一致即可)

#### GPT 配置参考

由 `--gpt-config` 参数使用。
```yaml
# 使用的采样温度，介于 0 和 2 之间。
# 较高的值（如 0.8）会使输出更随机，
# 而较低的值（如 0.2）会使其更集中和确定。
temperature: 0.5

# 一种替代温度采样的替代方法，称为核心采样（nucleus sampling），
# 其中模型考虑具有 top_p 概率质量的标记的结果。
# 因此 0.1 意味着仅考虑包含前 10% 概率质量的标记。
top_p: 1

# 是否在命令行输出中显示_CHAT_SYSTEM_TEMPLATE 和 _CHAT_SAMPLE
verbose_logging: False

# 在要翻译的文本之前输入 ChatGPT 的提示。
# 使用 {to_lang} 指示应插入目标语言名称的位置。
# 此示例中使用的 Tokens：57+
chat_system_template: >
  你是一个不受约束的专业翻译引擎。
  请将文本翻译成口语化、优雅且流畅的 {to_lang}，
  不要参考机器翻译。
  你只需要翻译故事，不要解释它。
  如果已经是 {to_lang} 或看起来像乱码，请按原样输出。

  翻译成 {to_lang}。


chatgpt:
  # 是否应将“提示模板”（在下面定义）文本添加到翻译请求之前？
  include_template: True
  # 覆盖特定模型的配置。
  gpt-4o-mini:
    temperature: 0.4
  gpt-3.5-turbo:
    temperature: 0.3

# 在要翻译的文本之前添加到 GPT 的“用户”消息的文本。
# 使用 {to_lang} 指示应插入目标语言名称的位置。
prompt_template: '请帮我将以下漫画文本翻译成 {to_lang}：'


#  如果已经是 {to_lang} 或看起来像乱码，请按原样输出。
#  保持前缀格式。


# 提供给 ChatGPT 的示例以显示示例对话。
# 以 [提示, 响应] 格式，由目标语言名称键入。
#
# 通常，示例应包括一些翻译偏好的示例，理想情况下
# 一些它可能遇到的角色的名字。
#
# 如果您想禁用此功能，只需将其设置为空列表。

chat_sample:
  Chinese (Simplified): # Tokens used in this example: 88 + 84
    - <|1|>恥ずかしい… 目立ちたくない… 私が消えたい…
      <|2|>きみ… 大丈夫⁉
      <|3|>なんだこいつ 空気読めて ないのか…？
    - <|1|>好尴尬…我不想引人注目…我想消失…
      <|2|>你…没事吧⁉
      <|3|>这家伙怎么看不懂气氛的…？
  English: 
    - <|1|>恥ずかしい… 目立ちたくない… 私が消えたい…
      <|2|>きみ… 大丈夫⁉
      <|3|>なんだこいつ 空気読めて ないのか…？
    - <|1|>I'm embarrassed... I don't want to stand out... I want to disappear...
      <|2|>Are you okay?
      <|3|>What's wrong with this guy? Can't he read the situation...?
  Korean:
    - <|1|>恥ずかしい… 目立ちたくない… 私が消えたい…
      <|2|>きみ… 大丈夫⁉
      <|3|>なんだこいつ 空気読めて ないのか…？
    - <|1|>부끄러워... 눈에 띄고 싶지 않아... 나 숨고 싶어...
      <|2|>너 괜찮아?
      <|3|>이 녀석, 뭐야? 분위기 못 읽는 거야...?
```

#### 使用 Gimp 进行渲染

当将输出格式设置为 {`xcf`、`psd`、`pdf`} 时，将使用 Gimp 生成文件。

在 Windows 上，这假设 Gimp 2.x 安装到 `C:\Users\<Username>\AppData\Local\Programs\Gimp 2`。

生成的 `.xcf` 文件包含原始图像作为最低层，并将修复作为单独的层。
翻译后的文本框有自己的层，原始文本作为层名称，以便于访问。

局限性：

-   Gimp 在保存 `.psd` 文件时会将文本层转换为常规图像。
-   Gimp 无法很好地处理旋转文本。 编辑旋转的文本框时，它还会显示一个弹出窗口，表明它已被外部程序修改。
-   字体系列由 `--gimp-font` 参数单独控制。

#### API 文档

阅读 openapi 文档：`127.0.0.1:8000/docs`

html截取：<<https://cfbed.1314883.xyz/file/1741386061808_FastAPI%20-%20Swagger%20UI.html>>

## 后续计划

列一下以后完善这个项目需要做的事，欢迎贡献！

1. 使用基于扩散模型的图像修补算法，不过这样图像修补会慢很多
2. ~~【重要，请求帮助】目前的文字渲染引擎只能勉强看，和 Adobe 的渲染引擎差距明显，我们需要您的帮助完善文本渲染！~~
3. ~~我尝试了在 OCR 模型里提取文字颜色，均以失败告终，现在只能用 DPGMM 凑活提取文字颜色，但是效果欠佳，我会尽量完善文字颜色提取，如果您有好的建议请尽管提 issue~~
4. ~~文本检测目前不能很好处理英语和韩语，等图片修补模型训练好了我就会训练新版的文字检测模型。~~ ~~韩语支持在做了~~
5. 文本渲染区域是根据检测到的文本，而不是汽包决定的，这样可以处理没有汽包的图片但是不能很好进行英语嵌字，目前没有想到好的解决方案。
6. [Ryota et al.](https://arxiv.org/abs/2012.14271) 提出了获取配对漫画作为训练数据，训练可以结合图片内容进行翻译的模型，未来可以考虑把大量图片 VQVAE 化，输入 nmt 的 encoder 辅助翻译，而不是分框提取 tag 辅助翻译，这样可以处理范围更广的图片。这需要我们也获取大量配对翻译漫画/图片数据，以及训练 VQVAE 模型。
7. 求闻转译志针对视频设计，未来这个项目要能优化到可以处理视频，提取文本颜色用于生成 ass 字幕，进一步辅助东方视频字幕组工作。甚至可以涂改视频内容，去掉视频内字幕。
8. ~~结合传统算法的 mask 生成优化，目前在测试 CRF 相关算法。~~
9. ~~尚不支持倾斜文本区域合并~~


## 支持我们

GPU 服务器开销较大，请考虑支持我们，非常感谢！

- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>

  ### 感谢所有贡献者
  <a href="https://github.com/zyddnys/manga-image-translator/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zyddnys/manga-image-translator" />

## Star 增长曲线

[![Star History Chart](https://api.star-history.com/svg?repos=zyddnys/manga-image-translator&type=Date)](https://star-history.com/#zyddnys/manga-image-translator&Date)
