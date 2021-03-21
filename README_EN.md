# Online Demo
https://touhou.ai/imgtrans/
# Translate texts in manga/images
Some manga/images will never be translated, therefore this project is born, \
Primarily designed for translating Japanese text, but also support Chinese and sometimes English \
Support simple inpainting and text rendering \
Successor to https://github.com/PatchyVideo/MMDOCR-HighPerformance

# How to use
1. Clone this repo
2. [Download](https://github.com/zyddnys/manga-image-translator/releases/tag/alpha-v2.2)ocr.ckpt、detect.ckpt and inpainting.ckpt，put them in the root directory of this repo
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
# Citation
```
@inproceedings{baek2019character,
  title={Character region awareness for text detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}
@article{hinami2020towards,
  title={Towards Fully Automated Manga Translation},
  author={Hinami, Ryota and Ishiwatari, Shonosuke and Yasuda, Kazuhiko and Matsui, Yusuke},
  journal={arXiv preprint arXiv:2012.14271},
  year={2020}
}
@article{oord2017neural,
  title={Neural discrete representation learning},
  author={Oord, Aaron van den and Vinyals, Oriol and Kavukcuoglu, Koray},
  journal={arXiv preprint arXiv:1711.00937},
  year={2017}
}
@article{uddin2020global,
  title={Global and Local Attention-Based Free-Form Image Inpainting},
  author={Uddin, SM and Jung, Yong Ju},
  journal={Sensors},
  volume={20},
  number={11},
  pages={3204},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
@article{brock2021characterizing,
  title={Characterizing signal propagation to close the performance gap in unnormalized ResNets},
  author={Brock, Andrew and De, Soham and Smith, Samuel L},
  journal={arXiv preprint arXiv:2101.08692},
  year={2021}
}
@inproceedings{fujimoto2016manga109,
  title={Manga109 dataset and creation of metadata},
  author={Fujimoto, Azuma and Ogawa, Toru and Yamamoto, Kazuyoshi and Matsui, Yusuke and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
  booktitle={Proceedings of the 1st international workshop on comics analysis, processing and understanding},
  pages={1--5},
  year={2016}
}
```