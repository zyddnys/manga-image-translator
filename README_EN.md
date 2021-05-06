# Online Demo
https://touhou.ai/imgtrans/
Note this may not work sometimes due to stupid google gcp kept restarting my instance. In that case you can wait for me to restart the service, which may take up to 24 hrs.
# Changelogs
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
2. [Download](https://github.com/zyddnys/manga-image-translator/releases/tag/alpha-v2.2.1)ocr.ckpt、detect.ckpt and inpainting.ckpt，put them in the root directory of this repo
3. Apply for baidu translate API, put ypur appid and key in `key.py`
4. Run`python translate_demo.py --image <path_to_image_file> [--use-inpainting] [--use-cuda]`，result can be found in `result/`. Add `--use-inpainting` to enable inpainting, Add `--use-cuda` to use CUDA.
# This is a hobby project, you are welcome to contribute
Currently this only a simple demo, many imperfections exist, we need your support to make this project better!

# Next steps
What need to be done
1. Inpainting is based on[Global and Local Attention-Based Free-Form Image Inpainting](https://www.mdpi.com/1424-8220/20/11/3204)，But with norm layers removed[Brock, A. et al.](https://arxiv.org/abs/2101.08692), only Coarse stage is used.
2. <b>IMPORTANT!!!HELP NEEDED!!!</b> The current text rendering engine is barely usable, we need your help to improve text rendering!
5. Text rendering area is determined by detected text lines, not speech bubbles. This works for images without speech bubbles, but making it impossible to decide where to put translated English text. I have no idea how to solve this.
6. [Ryota et al.](https://arxiv.org/abs/2012.14271) proposed using multimodal machine translation, maybe we can add ViT features for building custom NMT models.
7. Make this project works for video(rewrite code in C++ and use GPU/other hardware NN accelerator). Used for detecting hard subtitles in videos, generting ass file and remove them completetly.

# Samples
Original             |  Translated
:-------------------------:|:-------------------------:
![Original](original1.jpg "https://www.pixiv.net/en/artworks/85200179")|![Output](result1.png)
![Original](original2.jpg "https://twitter.com/mmd_96yuki/status/1320122899005460481")|![Output](result2.png)
![Original](original3.jpg "https://twitter.com/_taroshin_/status/1231099378779082754")|![Output](result3.png)
![Original](original4.jpg "https://amagi.fanbox.cc/posts/1904941")|![Output](result4.png)
