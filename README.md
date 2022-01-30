# 中文说明
[README_CN.md](README_CN.md)
# Online Demo
https://touhou.ai/imgtrans/
* Note this may not work sometimes due to stupid google gcp kept restarting my instance. In that case you can wait for me to restart the service, which may take up to 24 hrs.
* Note this online demo is using the current main branch version.
# Changelogs
### 2022-01-24
1. Added text detection model by [dmMaze](https://github.com/dmMaze)
### 2021-08-21
1. New MST based text region merge algorithm, huge text region merge improvement
2. Add baidu translator in demo mode
3. Add google translator in demo mode
4. Various bugfixes
### 2021-07-29
1. Web demo adds translator, detection resolution and target language option
2. Slight text color extraction improvement
### 2021-07-26
Major upgrades for all components, now we are on beta! \
Note in this version all English texts are detected as capital letters, \
You need Python >= 3.8 for `cached_property` to work
1. Detection model upgrade
2. OCR model upgrade, better at text color extraction
3. Inpainting model upgrade
4. Major text rendering improvement, faster rendering and higher quality text with shadow
5. Slight mask generation improvement
6. Various bugfixes
7. Default detection resolution has been dialed back to 1536 from 2048
### 2021-07-09
1. Fix erroneous image rendering when inpainting is not used
### 2021-06-18
1. Support manual translation
2. Support detection and rendering of angled texts
### 2021-06-13
1. Text mask completion is now based on CRF, mask quality is drastically improved
### 2021-06-10
1. Improve text rendering
### 2021-06-09
1. New text region based text direction detection method
2. Support running demo as web service
### 2021-05-20
1. Text detection model is now based on DBNet with ResNet34 backbone
2. OCR model is now trained with more English sentences
3. Inpaint model is now based on [AOT](https://arxiv.org/abs/2104.01431) which requires far less memory
4. Default inpainting resolution is now increased to 2048, thanks to the new inpainting model
5. Support merging hyphenated English words
### 2021-05-11
1. Add youdao translate and set as default translator
### 2021-05-06
1. Text detection model is now based on DBNet with ResNet101 backbone
2. OCR model is now deeper
3. Default detection resolution has been increased to 2048 from 1536

Note this version is slightly better at handling English texts, other than that it is worse in every other ways
### 2021-03-04
1. Added inpainting model
### 2021-02-17
1. First version launched
# Translate texts in manga/images
Some manga/images will never be translated, therefore this project is born, \
Primarily designed for translating Japanese text, but also support Chinese and English \
Support inpainting and text rendering \
Successor to https://github.com/PatchyVideo/MMDOCR-HighPerformance

# How to use
1. Python>=3.8
2. Clone this repo
3. [Download](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1) `ocr.ckpt`, `detect.ckpt`, `comictextdetector.pt`, `comictextdetector.pt.onnx` and `inpainting.ckpt`, put them in the root directory of this repo
4. [Optional if using Google translate] Apply for youdao or deepl translate API, put your APP_KEY and APP_SECRET or AUTH_KEY in `translators/key.py`
5. Run `python translate_demo.py --image <path_to_image_file> [--use-inpainting] [--verbose] [--use-cuda] [--translator=google] [--target-lang=CHS]`, result can be found in `result/`. Add `--use-inpainting` to enable inpainting, Add `--use-cuda` to use CUDA.

# Language codes
Used by `--target-lang` argument
```
	"CHS": "Chinese (Simplified)",
	"CHT": "Chinese (Traditional)",
	"CSY": "Czech",
	"NLD": "Dutch",
	"ENG": "English",
	"FRA": "French",
	"DEU": "German",
	"HUN": "Hungarian",
	"ITA": "Italian",
	"JPN": "Japanese",
	"KOR": "Korean",
	"PLK": "Polish",
	"PTB": "Portuguese (Brazil)",
	"ROM": "Romanian",
	"RUS": "Russian",
	"ESP": "Spanish",
	"TRK": "Turkish",
	"VIN": "Vietnamese"
```

# How to use (batch translation)
1. Python>=3.8
2. Clone this repo
3. [Download](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1) `ocr.ckpt`, `detect.ckpt`, `comictextdetector.pt`, `comictextdetector.pt.onnx` and `inpainting.ckpt`, put them in the root directory of this repo
4. [Optional if using Google translate] Apply for youdao or deepl translate API, put your APP_KEY and APP_SECRET or AUTH_KEY in `translators/key.py`
5. Run `python translate_demo.py --mode batch --image <path_to_image_folder> [--use-inpainting] [--verbose] [--use-cuda] [--translator=google] [--target-lang=CHS]`, result can be found in `<path_to_image_folder>-translated/`. Add `--use-inpainting` to enable inpainting, Add `--use-cuda` to use CUDA.

# How to use
1. Python>=3.8
2. Clone this repo
3. [Download](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1) `ocr.ckpt`, `detect.ckpt`, `comictextdetector.pt`, `comictextdetector.pt.onnx` and `inpainting.ckpt`, put them in the root directory of this repo
4. [Optional if using Google translate] Apply for youdao or deepl translate API, put your APP_KEY and APP_SECRET or AUTH_KEY in `translators/key.py`
5. Run `python translate_demo.py --mode web [--use-inpainting] [--verbose] [--use-cuda] [--translator=google] [--target-lang=CHS]`, the demo will be serving on http://127.0.0.1:5003

Two modes of translation service are provided by the demo: synchronous mode and asynchronous mode \
In synchronous mode your HTTP POST request will finish once the translation task is finished. \
In asynchronous mode your HTTP POST request will respond with a task_id immediately, you can use this task_id to poll for translation task state.
### Synchronous mode
1. POST a form request with form data `file:<content-of-image>` to http://127.0.0.1:5003/run
2. Wait for response
3. Use the resultant task_id to find translation result in `result/` directory, e.g. using Nginx to expose `result/`
### Asynchronous mode
1. POST a form request with form data `file:<content-of-image>` to http://127.0.0.1:5003/submit
2. Acquire translation task_id
3. Poll for translation task state by posting JSON `{"taskid": <task-id>}`  to http://127.0.0.1:5003/task-state
4. Translation is finished when the resultant state is either `finished`, `error` or `error-lang`
5. Find translation result in `result/` directory, e.g. using Nginx to expose `result/`

### Manual translation
Manual translation replace machine translation with human translators
1. POST a form request with form data `file:<content-of-image>` to http://127.0.0.1:5003/manual-translate
2. Wait for response
3. You will obtain a JSON response like this:
```JSON
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
4. Fill in translated texts
```JSON
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
5. Post translated JSON to http://127.0.0.1:5003/post-translation-result
6. Wait for response
7. Find translation result in `result/` directory, e.g. using Nginx to expose `result/`

# This is a hobby project, you are welcome to contribute
Currently this only a simple demo, many imperfections exist, we need your support to make this project better!

# Next steps
What need to be done
1. Inpainting is based on[Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://arxiv.org/abs/2104.01431)
2. <b>IMPORTANT!!!HELP NEEDED!!!</b> The current text rendering engine is barely usable, we need your help to improve text rendering!
5. Text rendering area is determined by detected text lines, not speech bubbles. This works for images without speech bubbles, but making it impossible to decide where to put translated English text. I have no idea how to solve this.
6. [Ryota et al.](https://arxiv.org/abs/2012.14271) proposed using multimodal machine translation, maybe we can add ViT features for building custom NMT models.
7. Make this project works for video(rewrite code in C++ and use GPU/other hardware NN accelerator). Used for detecting hard subtitles in videos, generting ass file and remove them completetly.
8. <s>Mask refinement based using non deep learning algorithms, I am currently testing out CRF based algorithm.</s>
9. <s>Angled text region merge is not currently supported</s>

# Samples
The following samples are from the original version, they do not represent the current main branch version.
Original             |  Translated
:-------------------------:|:-------------------------:
![Original](demo/original1.jpg "https://www.pixiv.net/en/artworks/85200179")|![Output](demo/result1.png)
![Original](demo/original2.jpg "https://twitter.com/mmd_96yuki/status/1320122899005460481")|![Output](demo/result2.png)
![Original](demo/original3.jpg "https://twitter.com/_taroshin_/status/1231099378779082754")|![Output](demo/result3.png)
![Original](demo/original4.jpg "https://amagi.fanbox.cc/posts/1904941")|![Output](demo/result4.png)
