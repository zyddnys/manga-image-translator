# Online Demo
https://touhou.ai/imgtrans/
* Note this may not work sometimes due to stupid google gcp kept restarting my instance. In that case you can wait for me to restart the service, which may take up to 24 hrs.
* Note this online demo maybe using an earlier version and not representing the current main branch version.
# Changelogs
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
Primarily designed for translating Japanese text, but also support Chinese and sometimes English \
Support simple inpainting and text rendering \
Successor to https://github.com/PatchyVideo/MMDOCR-HighPerformance

# How to use
1. Clone this repo
2. [Download](https://github.com/zyddnys/manga-image-translator/releases/tag/alpha-v3.0.0)ocr.ckpt、detect.ckpt and inpainting.ckpt，put them in the root directory of this repo
3. Apply for youdao translate API, put ypur APP_KEY and APP_SECRET in `key.py`
4. Run `python translate_demo.py --image <path_to_image_file> [--use-inpainting] [--use-cuda]`，result can be found in `result/`. Add `--use-inpainting` to enable inpainting, Add `--use-cuda` to use CUDA.

# How to use
1. Clone this repo
2. [Download](https://github.com/zyddnys/manga-image-translator/releases/tag/alpha-v3.0.0)ocr.ckpt、detect.ckpt and inpainting.ckpt，put them in the root directory of this repo
3. Apply for youdao translate API, put ypur APP_KEY and APP_SECRET in `key.py`
4. Run `python translate_demo.py --mode web [--use-inpainting] [--use-cuda]`, the demo will be serving on http://127.0.0.1:5003

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
4. Translation is finished when the resultant state is either `finished` or `error`
5. Find translation result in `result/` directory, e.g. using Nginx to expose `result/`


# This is a hobby project, you are welcome to contribute
Currently this only a simple demo, many imperfections exist, we need your support to make this project better!

# Next steps
What need to be done
1. Inpainting is based on[Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://arxiv.org/abs/2104.01431)
2. <b>IMPORTANT!!!HELP NEEDED!!!</b> The current text rendering engine is barely usable, we need your help to improve text rendering!
5. Text rendering area is determined by detected text lines, not speech bubbles. This works for images without speech bubbles, but making it impossible to decide where to put translated English text. I have no idea how to solve this.
6. [Ryota et al.](https://arxiv.org/abs/2012.14271) proposed using multimodal machine translation, maybe we can add ViT features for building custom NMT models.
7. Make this project works for video(rewrite code in C++ and use GPU/other hardware NN accelerator). Used for detecting hard subtitles in videos, generting ass file and remove them completetly.
8. Mask refinement based using non deep learning algorithms, I am currently testing out CRF based algorithm.

# Samples
The following samples are from the original version, they do not represent the current main branch version.
Original             |  Translated
:-------------------------:|:-------------------------:
![Original](original1.jpg "https://www.pixiv.net/en/artworks/85200179")|![Output](result1.png)
![Original](original2.jpg "https://twitter.com/mmd_96yuki/status/1320122899005460481")|![Output](result2.png)
![Original](original3.jpg "https://twitter.com/_taroshin_/status/1231099378779082754")|![Output](result3.png)
![Original](original4.jpg "https://amagi.fanbox.cc/posts/1904941")|![Output](result4.png)
