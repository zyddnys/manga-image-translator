# Image/Manga Translator

![Commit activity](https://img.shields.io/github/commit-activity/m/zyddnys/manga-image-translator)
![Lines of code](https://img.shields.io/tokei/lines/github/zyddnys/manga-image-translator?label=lines%20of%20code)
![License](https://img.shields.io/github/license/zyddnys/manga-image-translator)
![Contributors](https://img.shields.io/github/contributors/zyddnys/manga-image-translator)
[![Discord](https://img.shields.io/discord/739305951085199490?logo=discord&label=discord&logoColor=white)](https://discord.gg/Ak8APNy4vb)

> Translate texts in manga/images.\
> [中文说明](README_CN.md) | [Change Log](CHANGELOG.md) \
> Join us on discord <https://discord.gg/Ak8APNy4vb>

Some manga/images will never be translated, therefore this project is born.\
Primarily designed for translating Japanese text, but also supports Chinese, English and Korean.\
Supports inpainting and text rendering.\
Successor to <https://github.com/PatchyVideo/MMDOCR-HighPerformance>

**This is a hobby project, you are welcome to contribute!**\
Currently this only a simple demo, many imperfections exist, we need your support to make this project better!

## Support Us

GPU server is not cheap, please consider to donate to us.

- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>

## Online Demo

Official Demo (by zyddnys): <https://touhou.ai/imgtrans/>\
Browser Userscript (by QiroNT): <https://greasyfork.org/scripts/437569>

- Note this may not work sometimes due to stupid google gcp kept restarting my instance.
  In that case you can wait for me to restart the service, which may take up to 24 hrs.
- Note this online demo is using the current main branch version.

Sample images can be found [here](#samples)

## Installation

```bash
# First, you need to have Python(>=3.8) installed on your system.
$ python --version
Python 3.10.6

# Clone this repo
$ git clone https://github.com/zyddnys/manga-image-translator.git

# Install the dependencies
$ pip install -r requirements.txt

$ pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

The models will be downloaded into _./models_ at runtime.

#### If you are on windows:
Some pip dependencies will not compile without _Microsoft C++ Build Tools_
(See ![#114](https://github.com/zyddnys/manga-image-translator/issues/114)).

If you have trouble installing pydensecrf with the command above you can download the pre-compiled wheels
from <https://www.lfd.uci.edu/~gohlke/pythonlibs/#_pydensecrf> according to your python version and install it with pip.

## Usage

#### Demo mode (singular image)

```bash
# `--use-cuda` is optional, if you have a compatible NVIDIA GPU, you can use it.
# use `--use-cuda-limited` to defer vram expensive language translations to the cpu
# use `--inpainter=none` to disable inpainting.
# use `--translator=<translator>` to specify a translator.
# use `--translator=none` if you only want to use inpainting (blank bubbles)
# use `--target-lang <language_code>` to specify a target language.
# replace <path_to_image_file> with the path to the image file.
$ python translate_demo.py --verbose --use-cuda --translator=google -l ENG -i <path_to_image_file>
# result can be found in `result/`.
```

#### Batch mode

```bash
# same options as above.
# use `--mode batch` to enable batch translation.
# replace <path_to_image_folder> with the path to the image folder.
$ python translate_demo.py --verbose --mode batch --use-cuda --translator=google -l ENG -i <path_to_image_folder>
# results can be found in `<path_to_image_folder>-translated/`.
```

#### Web Mode

```bash
# same options as above.
# use `--mode web` to start a web server.
$ python translate_demo.py --verbose --mode web --use-cuda
# the demo will be serving on http://127.0.0.1:5003
```

#### Manual translation

Manual translation replaces machine translation with human translators.
Basic manual translation demo can be found at <http://127.0.0.1:5003/manual> when using web mode.

<details closed>
<summary>API</summary>
<br>

Two modes of translation service are provided by the demo: synchronous mode and asynchronous mode.\
In synchronous mode your HTTP POST request will finish once the translation task is finished.\
In asynchronous mode your HTTP POST request will respond with a `task_id` immediately, you can use this `task_id` to poll for translation task state.

#### Synchronous mode

1. POST a form request with form data `file:<content-of-image>` to <http://127.0.0.1:5003/run>
2. Wait for response
3. Use the resultant `task_id` to find translation result in `result/` directory, e.g. using Nginx to expose `result/`

#### Asynchronous mode

1. POST a form request with form data `file:<content-of-image>` to <http://127.0.0.1:5003/submit>
2. Acquire translation `task_id`
3. Poll for translation task state by posting JSON `{"taskid": <task-id>}` to <http://127.0.0.1:5003/task-state>
4. Translation is finished when the resultant state is either `finished`, `error` or `error-lang`
5. Find translation result in `result/` directory, e.g. using Nginx to expose `result/`

#### Manual translation

POST a form request with form data `file:<content-of-image>` to <http://127.0.0.1:5003/manual-translate>
and wait for response.

You will obtain a JSON response like this:

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

Fill in translated texts:

```json
{
  "task_id": "12c779c9431f954971cae720eb104499",
  "status": "pending",
  "trans_result": [
    {
      "s": "☆上司来ちゃった……",
      "t": "☆Boss is here..."
    }
  ]
}
```

Post translated JSON to <http://127.0.0.1:5003/post-translation-result> and wait for response.\
Then you can find the translation result in `result/` directory, e.g. using Nginx to expose `result/`.

</details>

### Translators Reference

| Name        | API Key | Offline | Docker | Note                                                  |
| ----------- | ------- | ------- | ------ | ----------------------------------------------------- |
| google      |         |         | ✔️      |                                                       |
| youdao      | ✔️       |         | ✔️      | Requires `YOUDAO_APP_KEY` and `YOUDAO_SECRET_KEY`     |
| baidu       | ✔️       |         | ✔️      | Requires `BAIDU_APP_ID` and `BAIDU_SECRET_KEY`        |
| deepl       | ✔️       |         | ✔️      | Requires `DEEPL_AUTH_KEY`                             |
| papago      |         |         | ✔️      |                                                       |
| offline     |         | ✔️       | ✔️      | Chooses most suitable offline translator for language |
| offline_big |         | ✔️       |        |                                                       |
| nllb        |         | ✔️       | ✔️      |                                                       |
| nllb_big    |         | ✔️       |        |                                                       |
| sugoi       |         | ✔️       | ✔️      |                                                       |
| sugoi_big   |         | ✔️       |        |                                                       |
| none        |         | ✔️       | ✔️      | Translate to empty texts                              |
| original    |         | ✔️       | ✔️      | Keep original texts                                   |

- API Key: Whether the translator requires an API key to be set as environment variable.
- Offline: Whether the translator can be used offline.
- Docker: Whether the translator is available in the docker image.

### Language Code Reference

Used by the `--target-lang` argument.

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
```

## Docker

Requirements:

- Docker (version 19.03+ required for CUDA / GPU accelaration)
- Docker Compose (Optional if you want to use files in the `demo/doc` folder)
- Nvidia Container Runtime (Optional if you want to use CUDA)

This project has docker support under `zyddnys/manga-image-translator:main` image.
This docker image contains all required dependencies / models for the project.
It should be noted that this image is fairly large (~ 15GB).

### Hosting the web server

The web server can be hosted using (For CPU)

```bash
docker run -p 5003:5003 -v result:/app/result --ipc=host --rm zyddnys/manga-image-translator:main -l ENG --manga2eng --verbose --log-web --mode web --host=0.0.0.0 --port=5003
```

or

```bash
docker-compose -f demo/doc/docker-compose-web-with-cpu.yml up
```

depending on which you prefer. The web server should start on port [5003](http://localhost:5003) and images should become in the `/result` folder.

### Using as CLI

To use docker with the CLI (I.e in batch mode)

```bash
docker run -v <targetFolder>:/app/<targetFolder> -v <targetFolder>-translated:/app/<targetFolder>-translated  --ipc=host --rm zyddnys/manga-image-translator:main --mode=batch -i=/app/<targetFolder> <cli flags>
```

**Note:** In the event you need to reference files on your host machine
you will need to mount the associated files as volumes into the `/app` folder inside the container.
Paths for the CLI will need to be the internal docker path `/app/...` instead of the paths on your host machine

### Setting Translation Secrets

Some translation services require API keys to function to set these pass them as env vars into the docker container. For example:

```bash
docker run --env="DEEPL_AUTH_KEY=xxx" --ipc=host --rm zyddnys/manga-image-translator:main <cli flags>
```

### Using with Nvida GPU

> To use with a supported GPU please first read the initial `Docker` section. There are some special dependencies you will need to use

To run the container with the following flags set:

```bash
docker run ... --gpus=all ... zyddnys/manga-image-translator:main ... --use-cuda
```

Or  (For the web server + GPU)

```bash
docker-compose -f demo/doc/docker-compose-web-with-gpu.yml up
```

### Building locally

To build the docker image locally you can run (You will require make on your machine)

```bash
make build-image
```

Then to test the built image run

```bash
make run-web-server
```

## Next steps

A list of what needs to be done next, you're welcome to contribute.

1. Use diffusion model based inpainting to achieve near perfect result, but this could be much slower.
2. ~~**IMPORTANT!!!HELP NEEDED!!!** The current text rendering engine is barely usable, we need your help to improve text rendering!~~
3. Text rendering area is determined by detected text lines, not speech bubbles.\
   This works for images without speech bubbles, but making it impossible to decide where to put translated English text. I have no idea how to solve this.
4. [Ryota et al.](https://arxiv.org/abs/2012.14271) proposed using multimodal machine translation, maybe we can add ViT features for building custom NMT models.
5. Make this project works for video(rewrite code in C++ and use GPU/other hardware NN accelerator).\
   Used for detecting hard subtitles in videos, generting ass file and remove them completetly.
6. ~~Mask refinement based using non deep learning algorithms, I am currently testing out CRF based algorithm.~~
7. ~~Angled text region merge is not currently supported~~

## Samples

The following samples are from the original version, they do not represent the current main branch version.

|                                             Original                                              |            Translated             |
| :-----------------------------------------------------------------------------------------------: | :-------------------------------: |
|        ![Original](demo/image/original1.jpg 'https://www.pixiv.net/en/artworks/85200179')         | ![Output](demo/image/result1.png) |
| ![Original](demo/image/original2.jpg 'https://twitter.com/mmd_96yuki/status/1320122899005460481') | ![Output](demo/image/result2.png) |
| ![Original](demo/image/original3.jpg 'https://twitter.com/_taroshin_/status/1231099378779082754') | ![Output](demo/image/result3.png) |
|           ![Original](demo/image/original4.jpg 'https://amagi.fanbox.cc/posts/1904941')           | ![Output](demo/image/result4.png) |
