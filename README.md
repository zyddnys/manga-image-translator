# 在线版
https://touhou.ai/imgtrans/
Note this may not work sometimes due to stupid google gcp kept restarting my instance. In that case you can wait for me to restart the service, which may take up to 24 hrs.
# English README
[README_EN.md](README_EN.md)
# Changelogs
### 2021-05-20
1. 检测模型更新为基于ResNet34的DBNet
2. OCR模型更新增加更多英语预料训练
3. 图像修补模型升级到基于[AOT](https://arxiv.org/abs/2104.01431)的模型，占用更少小显存
4. 图像修补默认分辨率增加到2048
5. 支持多行英语单词合并
### 2021-05-11
1. 增加并默认使用有道翻译
### 2021-05-06
1. 检测模型更新为基于ResNet101的DBNet
2. OCR模型更新更深
3. 默认检测分辨率增加到2048

注意这个版本除了英文检测稍微好一些，其他方面都不如之前版本
### 2021-03-04
1. 添加图片修补模型
### 2021-02-17
1. 初步版本发布
# 一键翻译各类图片内文字
针对群内、各个图站上大量不太可能会有人去翻译的图片设计，让我这种日语小白能够勉强看懂图片\
主要支持日语，不过也能识别汉语和小写英文 \
支持简单的涂白和嵌字 \
该项目是[求闻转译志](https://github.com/PatchyVideo/MMDOCR-HighPerformance)的v2版本

# 使用说明
1. clone这个repo
2. [下载](https://github.com/zyddnys/manga-image-translator/releases/tag/alpha-v3.0.0)ocr.ckpt、detect.ckpt和inpainting.ckpt，放到这个repo的根目录下
3. 申请有道翻译API，把你的APP_KEY和APP_SECRET存到key.py里
4. 运行`python translate_demo.py --image <图片文件路径> [--use-inpainting] [--use-cuda]`，结果会存放到result文件夹里。请加上`--use-inpainting`使用图像修补，请加上`--use-cuda`使用GPU。
# 只是初步版本，我们需要您的帮助完善
这个项目目前只完成了简单的demo，依旧存在大量不完善的地方，我们需要您的帮助完善这个项目！

# 下一步
完善这个项目
1. <s>图片涂改目前只是简单的涂白，图片修补的模型正在训练中！</s>图片修补基于[Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://arxiv.org/abs/2104.01431)，但是根据[Brock, A. et al.](https://arxiv.org/abs/2101.08692)
2. 【重要，请求帮助】目前的文字渲染引擎只能勉强看，和Adobe的渲染引擎差距明显，我们需要您的帮助完善文本渲染！
3. <s>我尝试了在OCR模型里提取文字颜色，均以失败告终，现在只能用DPGMM凑活提取文字颜色，但是效果欠佳，我会尽量完善文字颜色提取，如果您有好的建议请尽管提issue</s>
4. <s>文本检测目前不能很好处理英语和韩语，等图片修补模型训练好了我就会训练新版的文字检测模型。</s>韩语支持在做了
5. 文本渲染区域是根据检测到的文本，而不是汽包决定的，这样可以处理没有汽包的图片但是不能很好进行英语嵌字，目前没有想到好的解决方案。
6. [Ryota et al.](https://arxiv.org/abs/2012.14271)提出了获取配对漫画作为训练数据，训练可以结合图片内容进行翻译的模型，未来可以考虑把大量图片VQVAE化，输入nmt的encoder辅助翻译，而不是分框提取tag辅助翻译，这样可以处理范围更广的图片。这需要我们也获取大量配对翻译漫画/图片数据，以及训练VQVAE模型。
7. 求闻转译志针对视频设计，未来这个项目要能优化到可以处理视频，提取文本颜色用于生成ass字幕，进一步辅助东方视频字幕组工作。甚至可以涂改视频内容，去掉视频内字幕。
8. 结合传统算法的mask生成优化，目前在测试CRF相关算法。

# 效果图
以下图片为最初版效果，并不代表目前最新版本的效果。
原始图片             |  翻译后图片
:-------------------------:|:-------------------------:
![Original](original1.jpg "https://www.pixiv.net/en/artworks/85200179")|![Output](result1.png)
![Original](original2.jpg "https://twitter.com/mmd_96yuki/status/1320122899005460481")|![Output](result2.png)
![Original](original3.jpg "https://twitter.com/_taroshin_/status/1231099378779082754")|![Output](result3.png)
![Original](original4.jpg "https://amagi.fanbox.cc/posts/1904941")|![Output](result4.png)
