# Changelogs

### 2022-04-23

Project version is now at beta-0.3

1. Added English text renderer by [dmMaze](https://github.com/dmMaze)
2. Added new CTC based OCR engine, significant speed improvement
3. The new OCR model now support Korean

### 2022-03-19

1. Use new font rendering method by [pokedexter](https://github.com/pokedexter)
2. Added manual translation UI by [rspreet92](https://github.com/rspreet92)

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
