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
| EX1  | resnet18  | ijba                 | ./model/mytrain/2021-10-20-11:43:56/checkpoint_060.pth.tar   |                   |                         |                      |                      | 0.8625(1018/1171)    |                              |
| EX2  | resnet18  | ijba                 | ./model/mytrain/2021-10-26-15:09:35/model_best.pth.tar       | 0.8645(2712/3137) | 0.8083(489/605)         | 0.8615(1008/1170)    | 0.8468(536/633)      | 0.8617(1009/1171)    |                              |
| EX2  | resnet18  | ijba                 | ./model/mytrain/2021-10-26-15:09:35/checkpoint_150.pth.tar   |                   |                         |                      |                      | 0.8625(1010/1171)    |                              |
| EX4  | resnet18  | x                    | ./model/mytrain/2021-10-28-16:36:16/model_best.pth.tar       | 0.8444(2649/3137) | 0.8017(485/605)         | 0.8358(978/1170)     | 0.8167(517/633)      | 0.8292(971/1171)     | 0.7854(161/205)              |
| EX5  | resnet34  | x                    | ./model/mytrain/2021-10-30-16:11:42/model_best.pth.tar       | 0.8482(2661/3137) | 0.8132(492/605)         | 0.8564(1002/1170)    | 0.8278(524/663)      |                      |                              |
| EX6  | resnet50  | x                    | ./model/mytrain/2021-11-06-16:36:56/model_best.pth.tar       | 0.8470(2657/3137) | 0.8066(488/605)         | 0.8479(992 /1170)    | 0.8325(527663)       |                      |                              |
| EX7  | resnet101 | x                    | ./model/mytrain/2021-11-07-21:19:03/训练到30epoch暂时先停时间太长，周末接着训练 |                   |                         |                      |                      |                      |                              |
| EX8  | iresnet50 | faces_emore(arcface) |                                                              |                   |                         |                      |                      |                      |                              |
| EX9  | iresnet50 | x                    |                                                              |                   |                         |                      |                      |                      |                              |
| EX10 | resnet18  | MSCeleb-1m           | ./model/mytrain/2021-11-09-21:31:31/model_best.pth.tar       |                   |                         |                      |                      |                      |                              |


## Usage
After all the pre-generated files is done.

```
python train_attention_rank_loss.py

python test_rank_loss_attention.py
```
## To Do

- [x]  Change variable names that are inconsistent and obscure

- [x]  Removal of unnecessary intermediate files
 
- [x]  Removal of unnecessary script files and functions

- [ ]  Implemetation on Affectnet and RAF-DB

- [x]  Swich the Backbone. [VGG16 is the caffe model from vggface officially published by vgg to pytorch (with soft label)]
