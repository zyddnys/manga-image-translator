# 漫画图片翻译器 (中文说明)

> 一键翻译各类图片内文字\
> [English](README.md) | [更新日志](CHANGELOG_CN.md) \
> 欢迎加入我们的 Discord <https://discord.gg/Ak8APNy4vb>

针对群内、各个图站上大量不太可能会有人去翻译的图片设计，让我这种日语小白能够勉强看懂图片\
主要支持日语，汉语、英文和韩语\
支持图片修补和嵌字\
该项目是[求闻转译志](https://github.com/PatchyVideo/MMDOCR-HighPerformance)的 v2 版本

**只是初步版本，我们需要您的帮助完善**\
这个项目目前只完成了简单的 demo，依旧存在大量不完善的地方，我们需要您的帮助完善这个项目！

## 支持我们

请支持我们使用 GPU 服务器，谢谢！

- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>

## 在线版

官方演示站 (由 zyddnys 维护)： <https://cotrans.touhou.ai/>\
镜像站 (由 Eidenz 维护): <https://manga.eidenz.com/>\
浏览器脚本 (由 QiroNT 维护): <https://greasyfork.org/scripts/437569>

- 注意如果在线版无法访问说明 Google GCP 又在重启我的服务器，此时请等待我重新开启服务。
- 在线版使用的是目前 main 分支最新版本。

## 使用说明

```bash
# 首先，确信你的机器安装了 Python 3.8 及以上版本，和相应的编译工具
$ python --version
Python 3.8.13

# 拉取仓库
$ git clone https://github.com/zyddnys/manga-image-translator.git

# 安装依赖
$ pip install -r requirements.txt
```

注意：`pydensecrf` 和其他pip包可能需要操作系统的相应编译工具（如Debian的build-essential）。

[使用谷歌翻译时可选]\
申请有道翻译或者 DeepL 的 API，把你的 `APP_KEY` 和 `APP_SECRET` 或 `AUTH_KEY` 写入 `translators/key.py` 中。

### 翻译器列表

| 名称              | 是否需要 API Key | 是否离线可用 | 其他说明                                                   |
|-----------------| ------- |--------|--------------------------------------------------------|
| google          |         |        |                                                        |
| youdao          | ✔️      |        | 需要 `YOUDAO_APP_KEY` 和 `YOUDAO_SECRET_KEY`              |
| baidu           | ✔️      |        | 需要 `BAIDU_APP_ID` 和 `BAIDU_SECRET_KEY`                 |
| deepl           | ✔️      |        | 需要 `DEEPL_AUTH_KEY`                                    |
| caiyun          | ✔️      |        | 需要 `CAIYUN_TOKEN`                                      |
| gpt3            | ✔️      |        | Implements text-davinci-003. Requires `OPENAI_API_KEY` |
| gpt3.5          | ✔️      |        | Implements gpt-3.5-turbo. Requires `OPENAI_API_KEY`    |
| gpt4            | ✔️      |        | Implements gpt-4. Requires `OPENAI_API_KEY`            |
| deepseek        | ✔️      |        | 需要 DEEPSEEK_API_KEY                                    |
| papago          |         |        |                                                        |
| sakura          |         |        | 需要`SAKURA_API_BASE`                                    |
| ollama          |         |        |需要 `OLLAMA_API_BASE` `OLLAMA_MODEL`                     |
| offline         |         | ✔️     | 自动选择可用的离线模型，只是选择器                                      |
| sugoi           |         | ✔️     | 只能翻译英文                                                 |
| m2m100          |         | ✔️     | 可以翻译所有语言                                               |
| m2m100_big      |         | ✔️     | 带big的是完整尺寸，不带是精简版                                      |
| none            |         | ✔️     | 翻译成空白文本                                                |
| mbart50         |         | ✔️     |                                                        |
| original        |         | ✔️     | 翻译成源文本                                                 |

### 语言代码列表

可以填入 `--target-lang` 参数

```yaml
CHS: Chinese (Simplified)
CHT: Chinese (Traditional)
CSY: Czech
NLD: Dutch
ENG: English
FRA: French
DEU: German
HUN: Hungarian
ITA: Italian
JPN: Japanese
KOR: Korean
PLK: Polish
PTB: Portuguese (Brazil)
ROM: Romanian
RUS: Russian
ESP: Spanish
TRK: Turkish
VIN: Vietnames
ARA: Arabic
SRP: Serbian
HRV: Croatian
THA: Thai
IND: Indonesian
FIL: Filipino (Tagalog)
```

<!-- Auto generated start (See devscripts/make_readme.py) -->
## 选项

```text
-h, --help                     show this help message and exit
-v, --verbose                  Print debug info and save intermediate images in result folder
--attempts ATTEMPTS            Retry attempts on encountered error. -1 means infinite times.
--ignore-errors                Skip image on encountered error.
--model-dir MODEL_DIR          Model directory (by default ./models in project root)
--use-gpu                      Turn on/off gpu (auto switch between mps and cuda)
--use-gpu-limited              Turn on/off gpu (excluding offline translator)
--font-path FONT_PATH          Path to font file
--pre-dict PRE_DICT            Path to the pre-translation dictionary file
--post-dict POST_DICT          Path to the post-translation dictionary file
--kernel-size KERNEL_SIZE      Set the convolution kernel size of the text erasure area to
                               completely clean up text residues
--config-file CONFIG_FILE      path to the config file
--models-ttl MODELS_TTL        How long to keep models in memory in seconds after last use (0 means
                               forever)
--concurrency CONCURRENCY      (Only works in local mode)
                               The concurrency of the overall process ,including OCR, translation, and image rendering.
                               If the concurrency is 3, there will be at most 3 files be processed at the same time.
```

<!-- Auto generated end -->

### 使用命令行执行

```bash
# 如果机器有支持 CUDA 的 NVIDIA GPU，可以添加 `--use-gpu` 参数
# 使用 `--use-gpu-limited` 将需要使用大量显存的翻译交由CPU执行，这样可以减少显存占用
# 使用 `--translator=<翻译器名称>` 来指定翻译器
# 使用 `--target-lang=<语言代码>` 来指定目标语言
# 将 <图片文件路径> 替换为图片的路径
# 如果你要翻译的图片比较小或者模糊，可以使用upscaler提升图像大小与质量，从而提升检测翻译效果
$ python -m manga_translator --verbose --use-gpu --translator=google --target-lang=CHS -i <path_to_image_file>
# 结果会存放到 result 文件夹里
```

#### 使用命令行批量翻译

```bash
# 其它参数如上
# 使用 `--mode batch` 开启批量翻译模式
# 将 <图片文件夹路径> 替换为图片文件夹的路径
$ python -m manga_translator --verbose --mode batch --use-gpu --translator=google --target-lang=CHS -i <图片文件夹路径>
# 结果会存放到 `<图片文件夹路径>-translated` 文件夹里
```

### 使用浏览器 (Web 服务器)

```bash
# 其它参数如上
# 使用 `--mode web` 开启 Web 服务器模式
$ python -m manga_translator --verbose --mode web --use-gpu
# 程序服务会开启在 http://127.0.0.1:5003
```

程序提供两个请求模式：同步模式和异步模式。\
同步模式下你的 HTTP POST 请求会一直等待直到翻译完成。\
异步模式下你的 HTTP POST 会立刻返回一个 `task_id`，你可以使用这个 `task_id` 去定期轮询得到翻译的状态。

#### 同步模式

1. POST 提交一个带图片，名字是 file 的 form 到 <http://127.0.0.1:5003/run>
2. 等待返回
3. 从得到的 `task_id` 去 result 文件夹里取结果，例如通过 Nginx 暴露 result 下的内容

#### 异步模式

1. POST 提交一个带图片，名字是 file 的 form 到<http://127.0.0.1:5003/submit>
2. 你会得到一个 `task_id`
3. 通过这个 `task_id` 你可以定期发送 POST 轮询请求 JSON `{"taskid": <task_id>}` 到 <http://127.0.0.1:5003/task-state>
4. 当返回的状态是 `finished`、`error` 或 `error-lang` 时代表翻译完成
5. 去 result 文件夹里取结果，例如通过 Nginx 暴露 result 下的内容

#### 人工翻译

人工翻译允许代替机翻手动填入翻译后文本

POST 提交一个带图片，名字是 file 的 form 到 <http://127.0.0.1:5003/manual-translate>，并等待返回

你会得到一个 JSON 数组，例如：

```json
{
  "task_id": "12c779c9431f954971cae720eb104499",
  "status": "pending",
  "trans_result": [
    {
      "s": "☆上司来ちゃった……",
      "t": ""
    }
  ]
}
```

将翻译后内容填入 t 字符串：

```json
{
  "task_id": "12c779c9431f954971cae720eb104499",
  "status": "pending",
  "trans_result": [
    {
      "s": "☆上司来ちゃった……",
      "t": "☆上司来了..."
    }
  ]
}
```

将该 JSON 发送到 <http://127.0.0.1:5003/post-manual-result>，并等待返回\
之后就可以从得到的 `task_id` 去 result 文件夹里取结果，例如通过 Nginx 暴露 result 下的内容

## 下一步

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
