# RAN

This is an reimplement of the work [Region Attention Networks for Pose and Occlusion Robust Facial Expression Recognition](https://arxiv.org/pdf/1905.04075.pdf).
This code is based on [the original author's code](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/) with the following changes. Thanks, Kai Wang.


## Changes

- Pytorch code

Since part of the original pytorch code does not work in the new version of pytorch code, I made some minor changes to accommodate the versioning issue.

- Code to generate intermediate files

Some of the files in the original repository did not have uniform code to generate intermediate files (e.g. cropping, txt of the corresponding folder corresponding to each image)

- The path tree is rather confusing

The original files and script paths were rather messy, so I made some changes here to better organize these scripts.

- Backbone

Now it support resnet50, resnet101, iresnet18, iresnet34, iresnet50 and iresnet100, instead of just resnet18 and resnet34.

- Frozen

You can froze the layers as you want from the comment code.

## Results

| EX ID | Network  | pretrain               | model_path                                                     | acc（ferplus)     | acc（ferplus_occlusion) | acc（ferplus_pose30) | acc（ferplus_pose45) | acc（ferplus_minor） | acc(ferplus_minor_occlusion) |
| ---- | --------- | -------------------- | ------------------------------------------------------------ | ----------------- | ----------------------- | -------------------- | -------------------- | -------------------- | ---------------------------- |
| EX1  | resnet18  | ijba                 | ./model/mytrain/2021-10-20-11:43:56/model_best.pth.tar       | 0.8690(2726/3137) | 0.8231(498/605)         | 0.8701(1018/1170)    | 0.8547(541/633)      | 0.8702(1019/1171)    | 0.8049(165/205)              |

## Usage
After all the pre-generated files is done.

You need to adjust the batchsize on your own
```
python train.py

python test.py
```
## To Do

- [x]  Change variable names that are inconsistent and obscure

- [x]  Removal of unnecessary intermediate files
 
- [x]  Removal of unnecessary script files and functions

- [ ]  Implemetation on Affectnet and RAF-DB

- [x]  Swich the Backbone. [VGG16 is the caffe model from vggface officially published by vgg to pytorch (with soft label)]
