# Image/Manga Translator

![Commit activity](https://img.shields.io/github/commit-activity/m/zyddnys/manga-image-translator)
![Lines of code](https://img.shields.io/tokei/lines/github/zyddnys/manga-image-translator?label=lines%20of%20code)
![License](https://img.shields.io/github/license/zyddnys/manga-image-translator)
![Contributors](https://img.shields.io/github/contributors/zyddnys/manga-image-translator)
[![Discord](https://img.shields.io/discord/739305951085199490?logo=discord&label=discord&logoColor=white)](https://discord.gg/Ak8APNy4vb)


> Translate texts in manga/images.\
> [中文说明](README_CN.md) | [Change Log](CHANGELOG.md) \
> Join us on discord <https://discord.gg/Ak8APNy4vb>

Some manga/images will never be translated, therefore this project is born.

- [Image/Manga Translator](#imagemanga-translator)
    - [Samples](#samples)
    - [Online Demo](#online-demo)
    - [Disclaimer](#disclaimer)
    - [Installation](#installation)
        - [Local setup](#local-setup)
            - [Pip/venv](#pipvenv)
            - [Additional instructions for **Windows**](#additional-instructions-for-windows)
        - [Docker](#docker)
            - [Hosting the web server](#hosting-the-web-server)
            - [Using as CLI](#using-as-cli)
            - [Setting Translation Secrets](#setting-translation-secrets)
            - [Using with Nvidia GPU](#using-with-nvidia-gpu)
            - [Building locally](#building-locally)
    - [Usage](#usage)
        - [Batch mode (default)](#local-mode)
        - [Web / API Mode](#web-mode)
    - [Related Projects](#related-projects)
    - [Docs](#docs)
        - [Recommended Modules](#recommended-modules)
            - [Tips to improve translation quality](#tips-to-improve-translation-quality)
        - [Options](#options)
        - [Language Code Reference](#language-code-reference)
        - [Translators Reference](#translators-reference)
        - [Config Documentation](#config-file)
        - [GPT Config Reference](#gpt-config-reference)
        - [Using Gimp for rendering](#using-gimp-for-rendering)
        - [Api Documentation](#api-documentation)
    - [Next steps](#next-steps)
    - [Support Us](#support-us)
        - [Thanks To All Our Contributors :](#thanks-to-all-our-contributors-)
    - [Star history chart](#star-history-chart)

## Samples

Please note that the samples may not always be updated, they may not represent the current main branch version.

<table>
  <thead>
    <tr>
      <th align="center" width="50%">Original</th>
      <th align="center" width="50%">Translated</th>
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

## Online Demo

Official Demo (by zyddnys): <https://touhou.ai/imgtrans/>\
Browser Userscript (by QiroNT): <https://greasyfork.org/scripts/437569>

- Note this may not work sometimes due to stupid google gcp kept restarting my instance.
  In that case you can wait for me to restart the service, which may take up to 24 hrs.
- Note this online demo is using the current main branch version.

## Disclaimer

Successor to [MMDOCR-HighPerformance](https://github.com/PatchyVideo/MMDOCR-HighPerformance).\
**This is a hobby project, you are welcome to contribute!**\
Currently this only a simple demo, many imperfections exist, we need your support to make this project better!\
Primarily designed for translating Japanese text, but also supports Chinese, English and Korean.\
Supports inpainting, text rendering and colorization.

## Installation

### Local setup

#### Pip/venv

```bash
# First, you need to have Python(>=3.10) installed on your system
# The latest version often does not work with some pytorch libraries yet
$ python --version
Python 3.10.6

# Clone this repo
$ git clone https://github.com/zyddnys/manga-image-translator.git

# Create venv
$ python -m venv venv

# Activate venv
$ source venv/bin/activate

# For --use-gpu option go to https://pytorch.org/ and follow
# pytorch installation instructions. Add `--upgrade --force-reinstall`
# to the pip command to overwrite the currently installed pytorch version.

# Install the dependencies
$ pip install -r requirements.txt
```

The models will be downloaded into `./models` at runtime.

#### Additional instructions for **Windows**

Before you start the pip install, first install Microsoft C++ Build
Tools ([Download](https://visualstudio.microsoft.com/vs/),
[Instructions](https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools))
as some pip dependencies will not compile without it.
(See [#114](https://github.com/zyddnys/manga-image-translator/issues/114)).

To use [cuda](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
on windows install the correct pytorch version as instructed on <https://pytorch.org/>.

### Docker

Requirements:

- Docker (version 19.03+ required for CUDA / GPU acceleration)
- Docker Compose (Optional if you want to use files in the `demo/doc` folder)
- Nvidia Container Runtime (Optional if you want to use CUDA)

This project has docker support under `zyddnys/manga-image-translator:main` image.
This docker image contains all required dependencies / models for the project.
It should be noted that this image is fairly large (~ 15GB).

#### Hosting the web server

The web server can be hosted using (For CPU)

```bash
docker run -p 5003:5003 -v result:/app/result --ipc=host --rm zyddnys/manga-image-translator:main -l ENG --manga2eng -v --mode web --host=0.0.0.0 --port=5003
```

or

```bash
docker-compose -f demo/doc/docker-compose-web-with-cpu.yml up
```

depending on which you prefer. The web server should start on port [8000](http://localhost:8000)
<s>and images should become in the `/result` folder.</s>
images will not be saved at this point.

#### Using as CLI

To use docker with the CLI (I.e in batch mode)

```bash
docker run -v <targetFolder>:/app/<targetFolder> -v <targetFolder>-translated:/app/<targetFolder>-translated  --ipc=host --rm zyddnys/manga-image-translator:main --mode=batch -i=/app/<targetFolder> <cli flags>
```

**Note:** In the event you need to reference files on your host machine
you will need to mount the associated files as volumes into the `/app` folder inside the container.
Paths for the CLI will need to be the internal docker path `/app/...` instead of the paths on your host machine

#### Setting Translation Secrets

Some translation services require API keys to function to set these pass them as env vars into the docker container. For
example:

```bash
docker run --env="DEEPL_AUTH_KEY=xxx" --ipc=host --rm zyddnys/manga-image-translator:main <cli flags>
```

#### Using with Nvidia GPU

> To use with a supported GPU please first read the initial `Docker` section. There are some special dependencies you
> will need to use

To run the container with the following flags set:

```bash
docker run ... --gpus=all ... zyddnys/manga-image-translator:main ... --use-gpu
```

Or (For the web server + GPU)

```bash
docker-compose -f demo/doc/docker-compose-web-with-gpu.yml up
```

#### Building locally

To build the docker image locally you can run (You will require make on your machine)

```bash
make build-image
```

Then to test the built image run

```bash
make run-web-server
```

## Usage

### Local mode

```bash
# replace <path> with the path to the image folder or file.
$ python -m manga_translator local -v -i <path>
# results can be found under `<path_to_image_folder>-translated`.
```

### Web Mode

```bash
# use `--mode web` to start a web server.
$ cd server && python main.py --use-gpu
# the demo will be serving on http://127.0.0.1:8000
```

## Related Projects

GUI implementation: [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)

## Docs

### Recommended Modules

Detector:

- ENG: ??
- JPN: ??
- CHS: ??
- KOR: ??
- Using `{"detector":{"detector": "ctd"}}` can increase the amount of text lines detected

OCR:

- ENG: ??
- JPN: 48px
- CHS: ??
- KOR: 48px

Translator:

- JPN -> ENG: **Sugoi**
- CHS -> ENG: ??
- CHS -> JPN: ??
- JPN -> CHS: ??
- ENG -> JPN: ??
- ENG -> CHS: ??

Inpainter: lama_large

Colorizer: **mc2**

<!-- Auto generated start (See devscripts/make_readme.py) -->

### Tips to improve translation quality

- Small resolutions can sometimes trip up the detector, which is not so good at picking up irregular text sizes. To
  circumvent this you can use an upscaler by specifying `--upscale-ratio 2` or any other value
- If the text being rendered is too small to read specify `--font-size-minimum 30` for instance or use the `--manga2eng`
  renderer that will try to adapt to detected textbubbles
- Specify a font with `--font-path fonts/anime_ace_3.ttf` for example
- Set `mask_dilation_offset` 20~40.
- Using `lama_large` as impaiter. 
- Increasing the `box_threshold` can help filter out gibberish from OCR error detection to some extent.
- Using glossary file.

### Options

#### Basic Options

```text
-h, --help                     show this help message and exit
-v, --verbose                  Print debug info and save intermediate images in result folder
--attempts ATTEMPTS            Retry attempts on encountered error. -1 means infinite times.
--ignore-errors                Skip image on encountered error.
--model-dir MODEL_DIR          Model directory (by default ./models in project root)
--use-gpu                      Turn on/off gpu (auto switch between mps and cuda)
--use-gpu-limited              Turn on/off gpu (excluding offline translator)
--font-path FONT_PATH          Path to font file
--pre-dict PRE_DICT            Path to the pre-translation replacement dictionary file
--post-dict POST_DICT          Path to the post-translation replacement dictionary file
--kernel-size KERNEL_SIZE      Set the convolution kernel size of the text erasure area to
                               completely clean up text residues
```

#### Additional Options:

##### Batch Mode Options

```text
local                         Run in batch translation mode
-i, --input INPUT [INPUT ...] Path to an image folder (required)
-o, --dest DEST               Path to the destination folder for translated images (default: '')
-f, --format FORMAT           Output format of the translation.  Choices: [list the OUTPUT_FORMATS here, png,webp,jpg,jpeg,xcf,psd,pdf]
--overwrite                   Overwrite already translated images
--skip-no-text                Skip image without text (Will not be saved).
--use-mtpe                    Turn on/off machine translation post editing (MTPE) on the command line (works only on linux right now)
--save-text                   Save extracted text and translations into a text file.
--load-text                   Load extracted text and translations from a text file.
--save-text-file SAVE_TEXT_FILE  Like --save-text but with a specified file path. (default: '')
--prep-manual                 Prepare for manual typesetting by outputting blank, inpainted images, plus copies of the original for reference
--save-quality SAVE_QUALITY   Quality of saved JPEG image, range from 0 to 100 with 100 being best (default: 100)
--config-file CONFIG_FILE     path to the config file (default: None)                          
```

##### WebSocket Mode Options

```text
ws                  Run in WebSocket mode
--host HOST         Host for WebSocket service (default: 127.0.0.1)
--port PORT         Port for WebSocket service (default: 5003)
--nonce NONCE       Nonce for securing internal WebSocket communication
--ws-url WS_URL     Server URL for WebSocket mode (default: ws://localhost:5000)
--models-ttl MODELS_TTL  How long to keep models in memory in seconds after last use (0 means forever)
```

##### API Mode Options

```text
shared              Run in API mode
--host HOST         Host for API service (default: 127.0.0.1)
--port PORT         Port for API service (default: 5003)
--nonce NONCE       Nonce for securing internal API server communication
--report REPORT     reports to server to register instance (default: None)
--models-ttl MODELS_TTL  models TTL in memory in seconds (0 means forever)
```

##### Web Mode Options (Missing some basic options, need readded)

```text
--host HOST           The host address (default: 127.0.0.1)
--port PORT           The port number (default: 8000)
--start-instance      If a translator should be launched automatically
--nonce NONCE         Nonce for securing internal web server communication
--models-ttl MODELS_TTL  models TTL in memory in seconds (0 means forever)
```

##### config-help mode

```bash
python -m manga_translator config-help
```

#### Language Code Reference

Used by the `translator/language` in the config

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
UKR: Ukrainian
VIN: Vietnames
ARA: Arabic
SRP: Serbian
HRV: Croatian
THA: Thai
IND: Indonesian
FIL: Filipino (Tagalog)
```

#### Translators Reference

| Name          | API Key | Offline | Note                                                     |
|---------------|---------|---------|----------------------------------------------------------|
| <s>google</s> |         |         | Disabled temporarily                                     |
| youdao        | ✔️      |         | Requires `YOUDAO_APP_KEY` and `YOUDAO_SECRET_KEY`        |
| baidu         | ✔️      |         | Requires `BAIDU_APP_ID` and `BAIDU_SECRET_KEY`           |
| deepl         | ✔️      |         | Requires `DEEPL_AUTH_KEY`                                |
| caiyun        | ✔️      |         | Requires `CAIYUN_TOKEN`                                  |
| openai        | ✔️      |         | Implements Requires `OPENAI_API_KEY`                     |
| papago        |         |         |                                                          |
| sakura        |         |         | Requires `SAKURA_API_BASE`                               |
| custom openai |         |         | Requires  `CUSTOM_OPENAI_API_BASE` `CUSTOM_OPENAI_MODEL` |
| offline       |         | ✔️      | Chooses most suitable offline translator for language    |
| sugoi         |         | ✔️      | Sugoi V4.0 Models                                        |
| m2m100        |         | ✔️      | Supports every language                                  |
| m2m100_big    |         | ✔️      |                                                          |
| none          |         | ✔️      | Translate to empty texts                                 |
| original      |         | ✔️      | Keep original texts                                      |

- API Key: Whether the translator requires an API key to be set as environment variable.
  For this you can create a .env file in the project root directory containing your api keys like so:

```env
OPENAI_API_KEY=sk-xxxxxxx...
DEEPL_AUTH_KEY=xxxxxxxx...
```

- Offline: Whether the translator can be used offline.

- Sugoi is created by mingshiba, please support him in https://www.patreon.com/mingshiba


#### glossary
- mit_glossary: Sending a glossary to the AI model to guide its translation can effectively improve translation quality, for example, to ensure consistent translations of proprietary names and personal names. It will automatically extract the effective entries from the glossary for the current translation, so there is no need to worry that a large number of entries in the glossary will affect the translation quality. (Only valid for the openaitranslator, Compatible with sakura_dict and galtransl_dict.)

- sakura_dict: sakura glossary, only valid for sakuratranslator. No automated glossary feature.

```env
OPENAI_GLOSSARY_PATH=PATH_TO_YOUR_FILE
SAKURA_DICT_PATH=PATH_TO_YOUR_FILE
```  

#### Environment Variables Summary

| Environment Variable Name      | Description                                                                    | Default Value                     | Notes                                                                                                    |
| :----------------------------- | :----------------------------------------------------------------------------- | :-------------------------------- | :------------------------------------------------------------------------------------------------------- |
| `BAIDU_APP_ID`                 | Baidu Translate App ID                                                         | `''`                              |                                                                                                          |
| `BAIDU_SECRET_KEY`             | Baidu Translate Secret Key                                                     | `''`                              |                                                                                                          |
| `YOUDAO_APP_KEY`               | Youdao Translate App ID                                                        | `''`                              |                                                                                                          |
| `YOUDAO_SECRET_KEY`            | Youdao Translate App Secret Key                                                | `''`                              |                                                                                                          |
| `DEEPL_AUTH_KEY`              | DeepL Translate AUTH_KEY                                                      | `''`                              |                                                                                                          |
| `OPENAI_API_KEY`              | OpenAI API Key                                                                 | `''`                              |                                                                                                          |
| `OPENAI_MODEL`                | OpenAI Model (Optional)                                                        | `''`                              |                                                                                                          |
| `OPENAI_HTTP_PROXY`           | OpenAI HTTP Proxy (Optional)                                                   | `''`                              | Alternative to `--proxy`                                                                                |
| `OPENAI_GLOSSARY_PATH`        | OpenAI Glossary Path (Optional)                                                 | `./dict/mit_glossary.txt`         |                                                                                                          |
| `OPENAI_API_BASE`             | OpenAI API Base URL (Optional)                                                 | `https://api.openai.com/v1`       | Defaults to the official URL.                                                                            |
|`GROQ_API_KEY`| Groq API Key |||
| `SAKURA_API_BASE`             | SAKURA API URL (Optional)                                                      | `http://127.0.0.1:8080/v1`        |                                                                                                          |
| `SAKURA_VERSION`               | SAKURA API Version (Optional)                                                  | `'0.9'`                           | `'0.9'` or `'0.10'`                                                                                      |
| `SAKURA_DICT_PATH`            | SAKURA Glossary Path (Optional)                                                 | `./dict/sakura_dict.txt`          |                                                                                                          |
| `CAIYUN_TOKEN`                | Caiyun Xiaoyi (Colorful Clouds) API Access Token                               | `''`                              |                                                                                                          |
| `DEEPSEEK_API_KEY`           | DeepSeek API Key                                                        | `''`                              |                                                                          |
| `DEEPSEEK_API_BASE`           | DeepSeek API Base URL (Optional)                                              |   `https://api.deepseek.com`                                                              |    |
| `CUSTOM_OPENAI_API_KEY`        | Custom OpenAI API Key (Not needed for Ollama, but other tools might require it) | `'ollama'`                         |                                                                                                          |
| `CUSTOM_OPENAI_API_BASE`       | Custom OpenAI API Base URL (Use OLLAMA_HOST environment variable to change bind IP and port) | `http://localhost:11434/v1` |                                                                                                          |
| `CUSTOM_OPENAI_MODEL`          | Custom OpenAI Model (e.g., "qwen2.5:7b", make sure to pull and run it before using)  | `''`                              |                                                                                                          |
| `CUSTOM_OPENAI_MODEL_CONF`     | For example "qwen2"          | `''` |                                                                                                       |

**Instructions:**

1.  **Create a `.env` file:**  Create a file named `.env` in the root directory of your project.
2.  **Copy and Paste:** Copy the text above into the `.env` file.
3.  **Fill in your keys:** Replace the contents within the `''` (empty strings) with your own API keys, IDs, and other information.

**Important Notes:**

*   The `.env` file contains sensitive information.  Take extra care to prevent accidental leaks.


#### Config file

run `python -m manga_translator config-help >> config-info.json`

an example can be found in example/config-example.json

```json
{
  "$defs": {
    "Alignment": {
      "enum": [
        "auto",
        "left",
        "center",
        "right"
      ],
      "title": "Alignment",
      "type": "string"
    },
    "Colorizer": {
      "enum": [
        "none",
        "mc2"
      ],
      "title": "Colorizer",
      "type": "string"
    },
    "ColorizerConfig": {
      "properties": {
        "colorization_size": {
          "default": 576,
          "title": "Colorization Size",
          "type": "integer"
        },
        "denoise_sigma": {
          "default": 30,
          "title": "Denoise Sigma",
          "type": "integer"
        },
        "colorizer": {
          "$ref": "#/$defs/Colorizer",
          "default": "none"
        }
      },
      "title": "ColorizerConfig",
      "type": "object"
    },
    "Detector": {
      "enum": [
        "default",
        "dbconvnext",
        "ctd",
        "craft",
        "none"
      ],
      "title": "Detector",
      "type": "string"
    },
    "DetectorConfig": {
      "properties": {
        "detector": {
          "$ref": "#/$defs/Detector",
          "default": "default"
        },
        "detection_size": {
          "default": 1536,
          "title": "Detection Size",
          "type": "integer"
        },
        "text_threshold": {
          "default": 0.5,
          "title": "Text Threshold",
          "type": "number"
        },
        "det_rotate": {
          "default": false,
          "title": "Det Rotate",
          "type": "boolean"
        },
        "det_auto_rotate": {
          "default": false,
          "title": "Det Auto Rotate",
          "type": "boolean"
        },
        "det_invert": {
          "default": false,
          "title": "Det Invert",
          "type": "boolean"
        },
        "det_gamma_correct": {
          "default": false,
          "title": "Det Gamma Correct",
          "type": "boolean"
        },
        "box_threshold": {
          "default": 0.7,
          "title": "Box Threshold",
          "type": "number"
        },
        "unclip_ratio": {
          "default": 2.3,
          "title": "Unclip Ratio",
          "type": "number"
        }
      },
      "title": "DetectorConfig",
      "type": "object"
    },
    "Direction": {
      "enum": [
        "auto",
        "horizontal",
        "vertical"
      ],
      "title": "Direction",
      "type": "string"
    },
    "InpaintPrecision": {
      "enum": [
        "fp32",
        "fp16",
        "bf16"
      ],
      "title": "InpaintPrecision",
      "type": "string"
    },
    "Inpainter": {
      "enum": [
        "default",
        "lama_large",
        "lama_mpe",
        "sd",
        "none",
        "original"
      ],
      "title": "Inpainter",
      "type": "string"
    },
    "InpainterConfig": {
      "properties": {
        "inpainter": {
          "$ref": "#/$defs/Inpainter",
          "default": "none"
        },
        "inpainting_size": {
          "default": 2048,
          "title": "Inpainting Size",
          "type": "integer"
        },
        "inpainting_precision": {
          "$ref": "#/$defs/InpaintPrecision",
          "default": "fp32"
        }
      },
      "title": "InpainterConfig",
      "type": "object"
    },
    "Ocr": {
      "enum": [
        "32px",
        "48px",
        "48px_ctc",
        "mocr"
      ],
      "title": "Ocr",
      "type": "string"
    },
    "OcrConfig": {
      "properties": {
        "use_mocr_merge": {
          "default": false,
          "title": "Use Mocr Merge",
          "type": "boolean"
        },
        "ocr": {
          "$ref": "#/$defs/Ocr",
          "default": "48px"
        },
        "min_text_length": {
          "default": 0,
          "title": "Min Text Length",
          "type": "integer"
        },
        "ignore_bubble": {
          "default": 0,
          "title": "Ignore Bubble",
          "type": "integer"
        }
      },
      "title": "OcrConfig",
      "type": "object"
    },
    "RenderConfig": {
      "properties": {
        "renderer": {
          "$ref": "#/$defs/Renderer",
          "default": "default"
        },
        "alignment": {
          "$ref": "#/$defs/Alignment",
          "default": "auto"
        },
        "disable_font_border": {
          "default": false,
          "title": "Disable Font Border",
          "type": "boolean"
        },
        "font_size_offset": {
          "default": 0,
          "title": "Font Size Offset",
          "type": "integer"
        },
        "font_size_minimum": {
          "default": -1,
          "title": "Font Size Minimum",
          "type": "integer"
        },
        "direction": {
          "$ref": "#/$defs/Direction",
          "default": "auto"
        },
        "uppercase": {
          "default": false,
          "title": "Uppercase",
          "type": "boolean"
        },
        "lowercase": {
          "default": false,
          "title": "Lowercase",
          "type": "boolean"
        },
        "gimp_font": {
          "default": "Sans-serif",
          "title": "Gimp Font",
          "type": "string"
        },
        "no_hyphenation": {
          "default": false,
          "title": "No Hyphenation",
          "type": "boolean"
        },
        "font_color": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Font Color"
        },
        "line_spacing": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Line Spacing"
        },
        "font_size": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Font Size"
        }
      },
      "title": "RenderConfig",
      "type": "object"
    },
    "Renderer": {
      "enum": [
        "default",
        "manga2eng",
        "none"
      ],
      "title": "Renderer",
      "type": "string"
    },
    "Translator": {
      "enum": [
        "youdao",
        "baidu",
        "deepl",
        "papago",
        "caiyun",
        "gpt3",
        "gpt3.5",
        "gpt4",
        "none",
        "original",
        "sakura",
        "deepseek",
        "groq",
        "offline",
        "nllb",
        "nllb_big",
        "sugoi",
        "jparacrawl",
        "jparacrawl_big",
        "m2m100",
        "m2m100_big",
        "mbart50",
        "qwen2",
        "qwen2_big"
      ],
      "title": "Translator",
      "type": "string"
    },
    "TranslatorConfig": {
      "properties": {
        "translator": {
          "$ref": "#/$defs/Translator",
          "default": "sugoi"
        },
        "target_lang": {
          "default": "ENG",
          "title": "Target Lang",
          "type": "string"
        },
        "no_text_lang_skip": {
          "default": false,
          "title": "No Text Lang Skip",
          "type": "boolean"
        },
        "skip_lang": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Skip Lang"
        },
        "gpt_config": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Gpt Config"
        },
        "translator_chain": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Translator Chain"
        },
        "selective_translation": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Selective Translation"
        }
      },
      "title": "TranslatorConfig",
      "type": "object"
    },
    "UpscaleConfig": {
      "properties": {
        "upscaler": {
          "$ref": "#/$defs/Upscaler",
          "default": "esrgan"
        },
        "revert_upscaling": {
          "default": false,
          "title": "Revert Upscaling",
          "type": "boolean"
        },
        "upscale_ratio": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Upscale Ratio"
        }
      },
      "title": "UpscaleConfig",
      "type": "object"
    },
    "Upscaler": {
      "enum": [
        "waifu2x",
        "esrgan",
        "4xultrasharp"
      ],
      "title": "Upscaler",
      "type": "string"
    }
  },
  "properties": {
    "filter_text": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Filter Text"
    },
    "render": {
      "$ref": "#/$defs/RenderConfig",
      "default": {
        "renderer": "default",
        "alignment": "auto",
        "disable_font_border": false,
        "font_size_offset": 0,
        "font_size_minimum": -1,
        "direction": "auto",
        "uppercase": false,
        "lowercase": false,
        "gimp_font": "Sans-serif",
        "no_hyphenation": false,
        "font_color": null,
        "line_spacing": null,
        "font_size": null
      }
    },
    "upscale": {
      "$ref": "#/$defs/UpscaleConfig",
      "default": {
        "upscaler": "esrgan",
        "revert_upscaling": false,
        "upscale_ratio": null
      }
    },
    "translator": {
      "$ref": "#/$defs/TranslatorConfig",
      "default": {
        "translator": "sugoi",
        "target_lang": "ENG",
        "no_text_lang_skip": false,
        "skip_lang": null,
        "gpt_config": null,
        "translator_chain": null,
        "selective_translation": null
      }
    },
    "detector": {
      "$ref": "#/$defs/DetectorConfig",
      "default": {
        "detector": "default",
        "detection_size": 1536,
        "text_threshold": 0.5,
        "det_rotate": false,
        "det_auto_rotate": false,
        "det_invert": false,
        "det_gamma_correct": false,
        "box_threshold": 0.7,
        "unclip_ratio": 2.3
      }
    },
    "colorizer": {
      "$ref": "#/$defs/ColorizerConfig",
      "default": {
        "colorization_size": 576,
        "denoise_sigma": 30,
        "colorizer": "none"
      }
    },
    "inpainter": {
      "$ref": "#/$defs/InpainterConfig",
      "default": {
        "inpainter": "none",
        "inpainting_size": 2048,
        "inpainting_precision": "fp32"
      }
    },
    "ocr": {
      "$ref": "#/$defs/OcrConfig",
      "default": {
        "use_mocr_merge": false,
        "ocr": "48px",
        "min_text_length": 0,
        "ignore_bubble": 0
      }
    },
    "kernel_size": {
      "default": 3,
      "title": "Kernel Size",
      "type": "integer"
    },
    "mask_dilation_offset": {
      "default": 0,
      "title": "Mask Dilation Offset",
      "type": "integer"
    }
  },
  "title": "Config",
  "type": "object"
}

```

#### GPT Config Reference

Used by the `--gpt-config` argument.

```yaml
# The prompt being feed into GPT before the text to translate.
# Use {to_lang} to indicate where the target language name should be inserted.
# Note: ChatGPT models don't use this prompt.
prompt_template: >
  Please help me to translate the following text from a manga to {to_lang}
  (if it's already in {to_lang} or looks like gibberish you have to output it as it is instead):\n

# What sampling temperature to use, between 0 and 2.
# Higher values like 0.8 will make the output more random,
# while lower values like 0.2 will make it more focused and deterministic.
temperature: 0.5

# An alternative to sampling with temperature, called nucleus sampling,
# where the model considers the results of the tokens with top_p probability mass.
# So 0.1 means only the tokens comprising the top 10% probability mass are considered.
top_p: 1

#Whether to show _CHAT_SYSTEM_TEMPLATE and _CHAT_SAMPLE in the command line output
verbose_logging: False

# The prompt being feed into ChatGPT before the text to translate.
# Use {to_lang} to indicate where the target language name should be inserted.
# Tokens used in this example: 57+
chat_system_template: >
  You are a professional translation engine, 
  please translate the story into a colloquial, 
  elegant and fluent content, 
  without referencing machine translations. 
  You must only translate the story, never interpret it.
  If there is any issue in the text, output it as is.

  Translate to {to_lang}.

# Samples being feed into ChatGPT to show an example conversation.
# In a [prompt, response] format, keyed by the target language name.
#
# Generally, samples should include some examples of translation preferences, and ideally
# some names of characters it's likely to encounter.
#
# If you'd like to disable this feature, just set this to an empty list.
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

#### Using Gimp for rendering

When setting output format to {`xcf`, `psd`, `pdf`} Gimp will be used to generate the file.

On Windows this assumes Gimp 2.x to be installed to `C:\Users\<Username>\AppData\Local\Programs\Gimp 2`.

The resulting `.xcf` file contains the original image as the lowest layer and it has the inpainting as a separate layer.
The translated textboxes have their own layers with the original text as the layer name for easy access.

Limitations:

- Gimp will turn text layers to regular images when saving `.psd` files.
- Rotated text isn't handled well in Gimp. When editing a rotated textbox it'll also show a popup that it was modified
  by an outside program.
- Font family is controlled separately, with the `--gimp-font` argument.

#### Api Documentation

- Read openapi docs: `127.0.0.1:8000/docs`
- HTML scraping <https://cfbed.1314883.xyz/file/1741386061808_FastAPI%20-%20Swagger%20UI.html>
## Next steps

A list of what needs to be done next, you're welcome to contribute.

1. Use diffusion model based inpainting to achieve near perfect result, but this could be much slower.
2. ~~**IMPORTANT!!!HELP NEEDED!!!** The current text rendering engine is barely usable, we need your help to improve
   text rendering!~~
3. Text rendering area is determined by detected text lines, not speech bubbles.\
   This works for images without speech bubbles, but making it impossible to decide where to put translated English
   text. I have no idea how to solve this.
4. [Ryota et al.](https://arxiv.org/abs/2012.14271) proposed using multimodal machine translation, maybe we can add ViT
   features for building custom NMT models.
5. Make this project works for video(rewrite code in C++ and use GPU/other hardware NN accelerator).\
   Used for detecting hard subtitles in videos, generating ass file and remove them completely.
6. ~~Mask refinement based using non deep learning algorithms, I am currently testing out CRF based algorithm.~~
7. ~~Angled text region merge is not currently supported~~
8. Create pip repository

## Support Us

GPU server is not cheap, please consider to donate to us.

- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>

  ### Thanks To All Our Contributors :
  <a href="https://github.com/zyddnys/manga-image-translator/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zyddnys/manga-image-translator" />

## Star History Chart
[![Star History Chart](https://api.star-history.com/svg?repos=zyddnys/manga-image-translator&type=Date)](https://star-history.com/#zyddnys/manga-image-translator&Date)
</a>
