# 在线版
https://touhou.ai/imgtrans/
* Note this may not work sometimes due to stupid google gcp kept restarting my instance. In that case you can wait for me to restart the service, which may take up to 24 hrs.
* Note this online demo maybe using an earlier version and not representing the current main branch version.
* 在线版使用的是老版本，不代表目前main分支的版本
# English README
[README_EN.md](README_EN.md)
# Changelogs
### 2021-07-26
程序所有组件都大幅升级，本程序现已进入beta版本！ \
注意：该版本所有英文检测只会输出大写字母。\
你需要Python>=3.8版本才能运行
1. 检测模型升级
2. OCR模型升级，文本颜色抽取质量大幅提升
3. 图像修补模型升级
4. 文本渲染升级，渲染更快，并支持更高质量的文本和文本阴影渲染
5. 文字掩膜补全算法小幅提升
6. 各类BUG修复
7. 默认检测分辨率为1536
### 2021-07-09
1. 修复不使用inpainting时图片错误
### 2021-06-18
1. 增加手动翻译选项
2. 支持倾斜文本的识别和渲染
### 2021-06-13
1. 文字掩膜补全算法更新为基于CRF算法，补全质量大幅提升
### 2021-06-10
1. 完善文本渲染
### 2021-06-09
1. 使用基于区域的文本方向检测，文本方向检测效果大幅提升
2. 增加web服务功能
### 2021-05-20
1. 检测模型更新为基于ResNet34的DBNet
2. OCR模型更新增加更多英语预料训练
3. 图像修补模型升级到基于[AOT](https://arxiv.org/abs/2104.01431)的模型，占用更少显存
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
1. Python>=3.8
2. clone这个repo
3. [下载](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.0)ocr.ckpt、detect.ckpt和inpainting.ckpt，放到这个repo的根目录下
4. 申请有道翻译API，把你的APP_KEY和APP_SECRET存到key.py里
5. 运行`python translate_demo.py --image <图片文件路径> [--use-inpainting] [--use-cuda]`，结果会存放到result文件夹里。请加上`--use-inpainting`使用图像修补，请加上`--use-cuda`使用GPU。

# Web服务使用说明
1. Python>=3.8
2. clone这个repo
3. [下载](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.0)ocr.ckpt、detect.ckpt和inpainting.ckpt，放到这个repo的根目录下
4. 申请有道翻译API，把你的APP_KEY和APP_SECRET存到key.py里
5. 运行`python translate_demo.py --mode web [--use-inpainting] [--use-cuda]`，程序服务会开启在http://127.0.0.1:5003

程序提供两个请求模式：同步模式和异步模式。 \
同步模式下你的HTTP POST请求会一直等待直到翻译完成。 \
异步模式下你的HTTP POST会立刻返回一个task_id，你可以使用这个task_id去定期轮询得到翻译的状态。 \
### 同步模式
1. POST提交一个带图片，名字是file的form到http://127.0.0.1:5003/run
2. 等待返回
3. 从得到的task_id去result文件夹里取结果，例如通过Nginx暴露result下的内容
### 异步模式
1. POST提交一个带图片，名字是file的form到http://127.0.0.1:5003/submit
2. 你会得到一个task_id
3. 通过这个task_id你可以定期发送POST轮询请求JSON {"taskid": <你的task_id>}到http://127.0.0.1:5003/task-state
4. 当返回的状态是"finished"或"error"时代表翻译完成
5. 去result文件夹里取结果，例如通过Nginx暴露result下的内容

# 只是初步版本，我们需要您的帮助完善
这个项目目前只完成了简单的demo，依旧存在大量不完善的地方，我们需要您的帮助完善这个项目！

# 下一步
完善这个项目
1. <s>图片涂改目前只是简单的涂白，图片修补的模型正在训练中！</s>图片修补基于[Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://arxiv.org/abs/2104.01431)
2. 【重要，请求帮助】目前的文字渲染引擎只能勉强看，和Adobe的渲染引擎差距明显，我们需要您的帮助完善文本渲染！
3. <s>我尝试了在OCR模型里提取文字颜色，均以失败告终，现在只能用DPGMM凑活提取文字颜色，但是效果欠佳，我会尽量完善文字颜色提取，如果您有好的建议请尽管提issue</s>
4. <s>文本检测目前不能很好处理英语和韩语，等图片修补模型训练好了我就会训练新版的文字检测模型。</s>韩语支持在做了
5. 文本渲染区域是根据检测到的文本，而不是汽包决定的，这样可以处理没有汽包的图片但是不能很好进行英语嵌字，目前没有想到好的解决方案。
6. [Ryota et al.](https://arxiv.org/abs/2012.14271)提出了获取配对漫画作为训练数据，训练可以结合图片内容进行翻译的模型，未来可以考虑把大量图片VQVAE化，输入nmt的encoder辅助翻译，而不是分框提取tag辅助翻译，这样可以处理范围更广的图片。这需要我们也获取大量配对翻译漫画/图片数据，以及训练VQVAE模型。
7. 求闻转译志针对视频设计，未来这个项目要能优化到可以处理视频，提取文本颜色用于生成ass字幕，进一步辅助东方视频字幕组工作。甚至可以涂改视频内容，去掉视频内字幕。
8. <s>结合传统算法的mask生成优化，目前在测试CRF相关算法。</s>
9. <s>尚不支持倾斜文本区域合并</s>

# 效果图
以下图片为最初版效果，并不代表目前最新版本的效果。
原始图片             |  翻译后图片
:-------------------------:|:-------------------------:
![Original](original1.jpg "https://www.pixiv.net/en/artworks/85200179")|![Output](result1.png)
![Original](original2.jpg "https://twitter.com/mmd_96yuki/status/1320122899005460481")|![Output](result2.png)
![Original](original3.jpg "https://twitter.com/_taroshin_/status/1231099378779082754")|![Output](result3.png)
![Original](original4.jpg "https://amagi.fanbox.cc/posts/1904941")|![Output](result4.png)
