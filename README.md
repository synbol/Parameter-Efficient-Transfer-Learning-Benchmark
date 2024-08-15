<div align="center">
  <a href="https://github.com/synbol/Parameter-Efficient-Transfer-Learning-Benchmark">
    <img src="assets/v-petl.png" alt="Logo" width="400">
  </a>
</div>

<!-- 𝓥𝓲𝓼𝓾𝓪𝓵 𝓟𝓪𝓻𝓪𝓶𝓮𝓽𝓮𝓻-𝓔𝓯𝓯𝓲𝓬𝓲𝓮𝓷𝓽 𝓣𝓻𝓪𝓷𝓼𝓯𝓮𝓻 𝓛𝓮𝓪𝓻𝓷𝓲𝓷𝓰 𝓑𝓮𝓷𝓬𝓱𝓶𝓪𝓻𝓴 -->
<div align=center>
<h2 style="font-size: 50pt;" align=center><strong>A Unified Visual Parameter-Efficient Transfer Learning Benchmark</strong></h2>
<p>
 
 ![GitHub stars](https://img.shields.io/github/stars/synbol/Parameter-Efficient-Transfer-Learning-Benchmark.svg?color=red&style=for-the-badge) 
 ![GitHub forks](https://img.shields.io/github/forks/synbol/Parameter-Efficient-Transfer-Learning-Benchmark.svg?style=for-the-badge) 
 ![GitHub contributors](https://img.shields.io/github/contributors/synbol/Parameter-Efficient-Transfer-Learning-Benchmark.svg?style=for-the-badge) 
 ![GitHub activity](https://img.shields.io/github/last-commit/synbol/Parameter-Efficient-Transfer-Learning-Benchmark?style=for-the-badge) 
</p>
</div>

<div align="center">
  <p align="center">
    ·
    <a href="[111](111)">Paper</a>
    ·
    <a href="https://github.com/synbol/Parameter-Efficient-Transfer-Learning-Benchmark">Benchmark</a>
    ·
    <a href="https://v-petl-bench.github.io/">Homepage</a>
    ·
    <a href="">Document</a>
    ·
<!--     <a href="[111](111)">Video</a> -->
<!--     <a href="[111](111)">Video (Chinese)</a> -->
  </p>
</div>




## 🔥 <span id="head1"> *News and Updates* </span>

* ✅ [2024/07/25] Visual PEFT Benchmark starts releasing dataset, code, etc.

* ✅ [2024/06/20] Visual PEFT Benchmark homepage is created.

* ✅ [2024/06/01] Visual PEFT Benchmark repo is created.


## 📚 <span id="head1"> *Table of Contents* </span>
- [Introduction](#introduction)

- [Getting Started](#getting-started)

- [Quick Start](#quick-start)
  
- [Results and Checkpoints](#results)

- [Community and Contact](#contact)

- [Citation](#citation)


## ⭐ <span id="introduction"> *Introduction* </span>

Parameter-efficient transfer learning (PETL) methods show promise in adapting a pre-trained model to various downstream tasks while training only a few parameters. In the computer vision (CV) domain, numerous PETL algorithms have been proposed, but their direct employment or comparison remains inconvenient. To address this challenge, we construct a Unified Visual PETL Benchmark (V-PETL Bench) for the CV domain by selecting **30 diverse, challenging, and comprehensive datasets from image recognition, video action recognition, and dense prediction tasks**. On these datasets, we systematically evaluate **25 dominant PETL algorithms** and open-source a modular and extensible codebase for a fair evaluation of these algorithms. 

<!--V-PETL Bench runs on NVIDIA A800 GPUs and requires approximately 310 GPU days. We release all the checkpoints and training logs, making them more efficient and friendly to researchers. Additionally, V-PETL Bench
13 will be continuously updated for new PETL algorithms and CV tasks. -->


<div align="center">
<img src="/assets/introduction.png" width="100%" height="100%">
</div>


## ⚙️ <span id="getting-started"> *Getting Started* </span>

This is an example of how to set up V-PETL Bench locally. 

To get a local copy up, running follow these simple example steps.

### 👉 Environment Setup

V-PETL Bench is built on pytorch, with torchvision, torchaudio, and timm, etc.

- To install the required packages, you can create a conda environment.

```sh
conda create --name v-petl-bench python=3.8
```

- Activate conda environment.

```sh
conda activate v-petl-bench
```

- Use pip to install required packages.

```sh
pip install -r requirements.txt
```

### 👉 Data Preparation

#### 1. Image Classification Dataset

- **Fine-Grained Visual Classification tasks (FGVC)**

    FGVC comprises 5 fine-grained visual classification dataset. The datasets can be downloaded following the official links. We split the training data if the public validation set is not available. The splitted dataset can be found here: [Download Link](https://huggingface.co/datasets/XiN0919/FGVC/resolve/main/json.zip?download=true).
  
     - [CUB200 2011](https://data.caltech.edu/records/65de6-vp158)
   
     - [NABirds](http://info.allaboutbirds.org/nabirds/)
   
     - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)
   
     - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)
   
     - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

- **Visual Task Adaptation Benchmark (VTAB)**

    VTAB comprises 19 diverse visual classification datasets. We have processed all the dataset and the data can be downloaded here: [Download Link](https://huggingface.co/datasets/XiN0919/VTAB-1k/tree/main). For specific processing procedures and tips, please see [VTAB_SETUP]().

#### 2. Video Action Recognition Dataset

- **Kinetics-400**

  1. Download the dataset from [Download Link](https://deepmind.com/research/open-source/kinetics) or [Download Link](https://opendatalab.com/OpenMMLab/Kinetics-400).

  2. Preprocess the dataset by resizing the short edge of video to **320px**. You can refer to [MMAction2 Data Benchmark](https://github.com/open-mmlab/mmaction2).

  3. Generate annotations needed for dataloader ("<video_id> <video_class>" in annotations). The annotation usually includes `train.csv`, `val.csv` and `test.csv`. The format of `*.csv` file is like:

     ```
     video_1.mp4  label_1
     video_2.mp4  label_2
     video_3.mp4  label_3
     ...
     video_N.mp4  label_N
     ```
     <br>
    

- **Something-Something V2 (SSv2)**

  1. Download the dataset from [Download Link](https://developer.qualcomm.com/software/ai-datasets/something-something).

  2. Preprocess the dataset by changing the video extension from `webm` to `.mp4` with the **original** height of **240px**. You can refer to [MMAction2 Data Benchmark](https://github.com/open-mmlab/mmaction2).

  3. Generate annotations needed for dataloader ("<video_id> <video_class>" in annotations). The annotation usually includes `train.csv`, `val.csv` and `test.csv`. The format of `*.csv` file is like:

     ```
     video_1.mp4  label_1
     video_2.mp4  label_2
     video_3.mp4  label_3
     ...
     video_N.mp4  label_N
     ```

#### 3. Dense Prediction Dataset

- **MS-COCO**

    MS-COCO are available from this [Download Link](https://cocodataset.org/#download).

- **ADE20K**

    The training and validation set of ADE20K could be download from this [Download Link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
We may also download test set from [Download Link](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

- **PASCAL VOC**

    Pascal VOC 2012 could be downloaded from [Download Link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
Beside, most recent works on Pascal VOC dataset usually exploit extra augmentation data, which could be found [Download Link](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

### 👉 Pre-trained Model Preperation

- Download and place the ViT-B/16 pretrained model to `/path/to/pretrained_models`. 

```sh
mkdir pretrained_models

wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```
- or you can download Swin-B pretrained model. Note that you also need to rename the downloaded Swin-B ckpt from `swin_base_patch4_window7_224_22k.pth` to `Swin-B_16.pth`.

```sh
mkdir pretrained_models

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth

mv swin_base_patch4_window7_224_22k.pth Swin_B_16.pth
```

- Another way is to download pretrained models from below link and put it in `/path/to/pretrained_models`.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained Backbone</th>
<th valign="bottom">Pre-trained Objective</th>
 <th valign="bottom">Pre-trained Dataset</th>
<th valign="bottom">Checkpoint</th>
 
<!-- TABLE BODY -->
<tr><td align="center">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center">ImageNet-21K</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz">Download Link</a></td>
</tr>

<tr><td align="center">ViT-L/16</td>
<td align="center">Supervised</td>
 <td align="center">ImageNet-21K</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz">Download Link</a></td>
</tr>

<tr><td align="center">ViT-H/16</td>
<td align="center">Supervised</td>
<td align="center">ImageNet-21K</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz">Download Link</a></td>
</tr>
 
<tr><td align="center">Swin-B</td>
<td align="center">Supervised</td>
<td align="center">ImageNet-22K</td>
<td align="center"><a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth">Download Link</a></td>
</tr>

<tr><td align="center">Swin-L</td>
<td align="center">Supervised</td>
 <td align="center">ImageNet-22K</td>
<td align="center"><a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth">Download Link</a></td>
</tr>

<tr><td align="center">ViT-B (VideoMAE)</td>
<td align="center">Self-Supervised</td>
 <td align="center">Kinetics-400</td>
<td align="center"><a href="https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1">Download Link</a></td>
</tr>

<tr><td align="center">Video Swin-B</td>
<td align="center">Supervised</td>
 <td align="center">Kinetics-400</td>
<td align="center"><a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth">Download Link</a></td>
</tr>
</tbody></table>

## 🐌 <span id="quick-start"> *Quick Start* </span>

## 🎯 <span id="results"> *Results and Checkpoints* </span>

### Benchmark results of image classification on FGVC
- We evaluate 13 PETL algorithms on five datasets with ViT-B/16 models pre-trained on ImageNet-21K.

- To obtain the checkpoint, please download it at [Download Link]().

| **Method**  | CUB-200-2011 | NABirds | Oxford Flowers | Stanford Dogs | Stanford Cars | Mean | Params.| PPT |
|-----------------------|--------------|---------|----------------|---------------|---------------|------|---------------|-----|
| Full fine-tuning      | 87.3         | 82.7    | 98.8           | 89.4          | 84.5          | 88.54    | 85.8M         | -   |
| Linear probing        | 85.3         | 75.9    | 97.9           | 86.2          | 51.3          | 79.32    | 0 M           | 0.79|
| Adapter               | 87.1         | 84.3    | 98.5           | 89.8          | 68.6          | 85.66    | 0.41M         | 0.84|
| AdaptFormer           | 88.4         | 84.7    | 99.2           | 88.2          | 81.9          | 88.48    | 0.46M         | 0.87|
| Prefix Tuning         | 87.5         | 82.0    | 98.0           | 74.2          | 90.2          | 86.38    | 0.36M         | 0.85|
| U-Tuning              | 89.2         | 85.4    | 99.2           | 84.1          | 92.1          | 90.00    | 0.36M         | 0.89|
| BitFit                | 87.7         | 85.2    | 99.2           | 86.5          | 81.5          | 88.02    | 0.10M         | 0.88|
| VPT-Shallow           | 86.7         | 78.8    | 98.4           | 90.7          | 68.7          | 84.66    | 0.25M         | 0.84|
| VPT-Deep              | 88.5         | 84.2    | 99.0           | 90.2          | 83.6          | 89.10    | 0.85M         | 0.86|
| SSF                   | 89.5         | 85.7    | 99.6           | 89.6          | 89.2          | 90.72    | 0.39M         | 0.89|
| LoRA                  | 85.6         | 79.8    | 98.9           | 87.6          | 72.0          | 84.78    | 0.77M         | 0.82|
| GPS                   | 89.9         | 86.7    | 99.7           | 92.2          | 90.4          | 91.78    | 0.66M         | 0.90|
| HST                   | 89.2         | 85.8    | 99.6           | 89.5          | 88.2          | 90.46    | 0.78M         | 0.88|
| LAST                  | 88.5         | 84.4    | 99.7           | 86.0          | 88.9          | 89.50    | 0.66M         | 0.87|
| SNF                   | 90.2         | 87.4    | 99.7           | 89.5          | 86.9          | 90.74    | 0.25M         | 0.90|



### Benchmark results of image classification on VTAB

- Benchmark results on VTAB. We evaluate 18 PETL algorithms on 19 datasets with ViT-B/16 models pre-trained on ImageNet-21K.

- To obtain the checkpoint, please download it at [Download Link]().
  
| Method | CIFAR-100 | Caltech101 | DTD | Flowers102 | Pets | SVHN | Sun397 | Patch Camelyon | EuroSAT | Resisc45 | Retinopathy | Clevr/count | Clevr/distance | DMLab | KITTI/distance | dSprites/loc | dSprites/ori | SmallNORB/azi | SmallNORB/ele | Mean | Params. | PPT |
| --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full fine-tuning | 68.9 | 87.7 | 64.3 | 97.2 | 86.9 | 87.4 | 38.8 | 79.7 | 95.7 | 84.2 | 73.9 | 56.3 | 58.6 | 41.7 | 65.5 | 57.5 | 46.7 | 25.7 | 29.1 | 65.57 | 85.8M | - |
| Linear probing   | 63.4 | 85.0 | 63.2 | 97.0 | 86.3 | 36.6 | 51.0 | 78.5 | 87.5 | 68.6 | 74.0 | 34.3 | 30.6 | 33.2 | 55.4 | 12.5 | 20.0 | 9.6 | 19.2 | 52.94  | 0M    | 0.53 |
| Adapter          | 69.2 | 90.1 | 68.0 | 98.8 | 89.9 | 82.8 | 54.3 | 84.0 | 94.9 | 81.9 | 75.5 | 80.9 | 65.3 | 48.6 | 78.3 | 74.8 | 48.5 | 29.9 | 41.6 | 71.44 | 0.16M | 0.71 |
| VPT-Shallow      | 77.7 | 86.9 | 62.6 | 97.5 | 87.3 | 74.5 | 51.2 | 78.2 | 92.0 | 75.6 | 72.9 | 50.5 | 58.6 | 40.5 | 67.1 | 68.7 | 36.1 | 20.2 | 34.1 | 64.85 | 0.08M | 0.65 |
| VPT-Deep         | 78.8 | 90.8 | 65.8 | 98.0 | 88.3 | 78.1 | 49.6 | 81.8 | 96.1 | 83.4 | 68.4 | 68.5 | 60.0 | 46.5 | 72.8 | 73.6 | 47.9 | 32.9 | 37.8 | 69.43 | 0.56M | 0.68 |
| BitFit           | 72.8 | 87.0 | 59.2 | 97.5 | 85.3 | 59.9 | 51.4 | 78.7 | 91.6 | 72.9 | 69.8 | 61.5 | 55.6 | 32.4 | 55.9 | 66.6 | 40.0 | 15.7 | 25.1 | 62.05 | 0.10M | 0.61 |
| LoRA             | 67.1 | 91.4 | 69.4 | 98.8 | 90.4 | 85.3 | 54.0 | 84.9 | 95.3 | 84.4 | 73.6 | 82.9 | 69.2 | 49.8 | 78.5 | 75.7 | 47.1 | 31.0 | 44.0 | 72.25 | 0.29M | 0.71 |
| AdaptFormer      | 70.8 | 91.2 | 70.5 | 99.1 | 90.9 | 86.6 | 54.8 | 83.0 | 95.8 | 84.4 | 76.3 | 81.9 | 64.3 | 49.3 | 80.3 | 76.3 | 45.7 | 31.7 | 41.1 | 72.32 | 0.16M | 0.72 |
| SSF              | 69.0 | 92.6 | 75.1 | 99.4 | 91.8 | 90.2 | 52.9 | 87.4 | 95.9 | 87.4 | 75.5 | 75.9 | 62.3 | 53.3 | 80.6 | 77.3 | 54.9 | 29.5 | 37.9 | 73.10 | 0.21M | 0.72 |
| NOAH             | 69.6 | 92.7 | 70.2 | 99.1 | 90.4 | 86.1 | 53.7 | 84.4 | 95.4 | 83.9 | 75.8 | 82.8 | 68.9 | 49.9 | 81.7 | 81.8 | 48.3 | 32.8 | 44.2 | 73.25 | 0.43M | 0.72 |
| SCT              | 75.3 | 91.6 | 72.2 | 99.2 | 91.1 | 91.2 | 55.0 | 85.0 | 96.1 | 86.3 | 76.2 | 81.5 | 65.1 | 51.7 | 80.2 | 75.4 | 46.2 | 33.2 | 45.7 | 73.59 | 0.11M | 0.73 |
| FacT             | 70.6 | 90.6 | 70.8 | 99.1 | 90.7 | 88.6 | 54.1 | 84.8 | 96.2 | 84.5 | 75.7 | 82.6 | 68.2 | 49.8 | 80.7 | 80.8 | 47.4 | 33.2 | 43.0 | 73.23 | 0.07M | 0.73 |
| RepAdapter       | 72.4 | 91.6 | 71.0 | 99.2 | 91.4 | 90.7 | 55.1 | 85.3 | 95.9 | 84.6 | 75.9 | 82.3 | 68.0 | 50.4 | 79.9 | 80.4 | 49.2 | 38.6 | 41.0 | 73.84 | 0.22M | 0.72 |
| Hydra            | 72.7 | 91.3 | 72.0 | 99.2 | 91.4 | 90.7 | 55.5 | 85.8 | 96.0 | 86.1 | 75.9 | 83.2 | 68.2 | 50.9 | 82.3 | 80.3 | 50.8 | 34.5 | 43.1 | 74.21 | 0.28M | 0.73 |
| LST              | 59.5 | 91.5 | 69.0 | 99.2 | 89.9 | 79.5 | 54.6 | 86.9 | 95.9 | 85.3 | 74.1 | 81.8 | 61.8 | 52.2 | 81.0 | 71.7 | 49.5 | 33.7 | 45.2 | 71.70 | 2.38M | 0.65 |
| DTL              | 69.6 | 94.8 | 71.3 | 99.3 | 91.3 | 83.3 | 56.2 | 87.1 | 96.2 | 86.1 | 75.0 | 82.8 | 64.2 | 48.8 | 81.9 | 93.9 | 53.9 | 34.2 | 47.1 | 74.58 | 0.04M | 0.75 |
| HST              | 76.7 | 94.1 | 74.8 | 99.6 | 91.1 | 91.2 | 52.3 | 87.1 | 96.3 | 88.6 | 76.5 | 85.4 | 63.7 | 52.9 | 81.7 | 87.2 | 56.8 | 35.8 | 52.1 | 75.99 | 0.78M | 0.74 |
| GPS              | 81.1 | 94.2 | 75.8 | 99.4 | 91.7 | 91.6 | 52.4 | 87.9 | 96.2 | 86.5 | 76.5 | 79.9 | 62.6 | 55.0 | 82.4 | 84.0 | 55.4 | 29.7 | 46.1 | 75.18 | 0.22M | 0.74 |
| LAST             | 66.7 | 93.4 | 76.1 | 99.6 | 89.8 | 86.1 | 54.3 | 86.2 | 96.3 | 86.8 | 75.4 | 81.9 | 65.9 | 49.4 | 82.6 | 87.9 | 46.7 | 32.3 | 51.5 | 74.15 | 0.66M | 0.72 |
| SNF              | 84.0 | 94.0 | 72.7 | 99.3 | 91.3 | 90.3 | 54.9 | 87.2 | 97.3 | 85.5 | 74.5 | 82.3 | 63.8 | 49.8 | 82.5 | 75.8 | 49.2 | 31.4 | 42.1 | 74.10 | 0.25M | 0.73 |



### Benchmark results of video action recognition  on SSv2 and HMDB51.
- Benchmark results on SSv2 and HMDB51. We evaluate 5 PETL algorithms with ViT-B from VideoMAE and Video Swin Transformer.

- To obtain the checkpoint, please download it at [Download Link]().

| Method                     | Model         | Pre-training  | Params. | SSv2 (Top1)  | SSv2 (PPT)| HMDB51 (Top1) | HMDB51 (PPT) |
|----------------------------|---------------|---------------|------------|-----------|----------|-------------|------------|                      
| Full fine-tuning           | ViT-B         | Kinetics 400  | 85.97 M    | 53.97%    | -        | 46.41%      | -          |
| Frozen                     | ViT-B         | Kinetics 400  | 0 M        | 29.23%    | 0.29     | 49.84%      | 0.50       |
| AdaptFormer                | ViT-B         | Kinetics 400  | 1.19 M     | 59.02%    | 0.56     | 55.69%      | 0.53       |
| BAPAT                      | ViT-B         | Kinetics 400  | 2.06 M     | 57.78%    | 0.53     | 57.18%      | 0.53       |                              
| Full fine-tuning           | Video Swin-B  | Kinetics 400  | 87.64 M    | 50.99%    | -        | 68.07%      | -          |
| Frozen                     | Video Swin-B  | Kinetics 400  | 0 M        | 24.13%    | 0.24     | 71.28%      | 0.71       |
| LoRA                       | Video Swin-B  | Kinetics 400  | 0.75 M     | 38.34%    | 0.37     | 62.12%      | 0.60       |
| BitFit                     | Video Swin-B  | Kinetics 400  | 1.09 M     | 45.94%    | 0.44     | 68.26%      | 0.65       |
| AdaptFormer                | Video Swin-B  | Kinetics 400  | 1.56 M     | 40.80%    | 0.38     | 68.66%      | 0.64       |
| Prefix-tuning              | Video Swin-B  | Kinetics 400  | 6.37 M     | 39.46%    | 0.32     | 56.13%      | 0.45       |
| BAPAT                      | Video Swin-B  | Kinetics 400  | 6.18 M     | 53.36%    | 0.43     | 71.93%      | 0.58       |

### Benchmark results of dense prediction on COCO
- Benchmark results on COCO. We evaluate 9 PETL algorithms with Swin-B models pre-trained on ImageNet-22K. 

- To obtain the checkpoint, please download it at [Download Link]().

| **Swin-B**         | **Params.** | **Memory** | **COCO  ($\mathrm{AP_{Box}}$)** | **COCO (PPT)** | **COCO  ($\mathrm{AP_{Mask}}$)** | **COCO (PPT)** |
|----------------|----------------|------------|-------------------|--------|-------------|--------|
| Full fine-tuning   | 86.75 M    | 17061 MB   | 51.9%             | -      | 45.0%       | -      |
| Frozen             | 0.00 M     | 7137 MB    | 43.5%             | 0.44   | 38.6%       | 0.39   |
| Bitfit             | 0.20 M     | 13657 MB   | 47.9%             | 0.47   | 41.9%       | 0.42   |
| LN TUNE            | 0.06 M     | 12831 MB   | 48.0%             | 0.48   | 41.4%       | 0.41   |
| Partial-1          | 12.60 M    | 7301 MB    | 49.2%             | 0.35   | 42.8%       | 0.30   |
| Adapter            | 3.11 M     | 12557 MB   | 50.9%             | 0.45   | 43.8%       | 0.39   |
| LoRA               | 3.03 M     | 11975 MB   | 51.2%             | 0.46   | 44.3%       | 0.40   |
| AdaptFormer        | 3.11 M     | 13186 MB   | 51.4%             | 0.46   | 44.5%       | 0.40   |
| LoRand             | 1.20 M     | 13598 MB   | 51.0%             | 0.49   | 43.9%       | 0.42   |
| E$^3$VA            | 1.20 M     | 7639 MB    | 50.5%             | 0.48   | 43.8%       | 0.42   |
| Mona               | 4.16 M     | 13996 MB   | 53.4%             | 0.46   | 46.0%       | 0.40   |


### Benchmark results of dense prediction on PASCAL VOC and ADE20K.
- Benchmark results on PASCAL VOC and ADE20K. We evaluate 9 PETL algorithms with Swin-L models pre-trained on ImageNet-22K.

- To obtain the checkpoint, please download it at [Download Link]().


| **Swin-L**             | **Params.** | **Memory (VOC)** | **Pascal VOC  ($\mathrm{AP_{Box}}$)** |Pascal VOC (PPT)| **ADE20K  ($\mathrm{mIoU}$)** | **ADE20K (PPT)** |
|------------------------|----------------|-------------|---------------------------|--------|--------------------------------------|--------|       
| Full fine-tuning       | 198.58 M       | 15679 MB         | 83.5%                | -      | 52.10%                              | -      |
| Frozen                 | 0.00 M         | 3967 MB          | 83.6%                | 0.84   | 46.84%                              | 0.47   |
| Bitfit                 | 0.30 M         | 10861 MB         | 85.7%                | 0.85   | 48.37%                              | 0.48   |
| LN TUNE                | 0.09 M         | 10123 MB         | 85.8%                | 0.86   | 47.98%                              | 0.48   |
| Partial-1              | 28.34 M        | 3943 MB          | 85.4%                | 0.48   | 47.44%                              | 0.27   |
| Adapter                | 4.66 M         | 10793 MB         | 87.1%                | 0.74   | 50.78%                              | 0.43   |
| LoRA                   | 4.57 M         | 10127 MB         | 87.5%                | 0.74   | 50.34%                              | 0.43   |
| AdaptFormer            | 4.66 M         | 11036 MB         | 87.3%                | 0.74   | 50.83%                              | 0.43   |
| LoRand                 | 1.31 M         | 11572 MB         | 86.8%                | 0.82   | 50.76%                              | 0.48   |
| E$^3$VA                | 1.79 M         | 4819 MB          | 86.5%                | 0.81   | 49.64%                              | 0.46   |
| Mona                   | 5.08 M         | 11958 MB         | 87.3%                | 0.73   | 51.36%                              | 0.43   |

## 💬 <span id="contact"> *Community and Contact* </span>

- The V-PETL Bench community is maintained by:

  - Yi Xin (xinyi@smail.nju.edu.cn), Nanjing University.
 
  - Siqi Luo (siqiluo@smail.nju.edu.cn), Shanghai Jiao Tong University.

## 📝 <span id="citation"> *Citation* </span>

- If you find our survey and repository useful for your research, please cite it below:

```bibtex
@article{xin2024bench,
  title={V-PETL Bench: A Unified Visual Parameter-Efficient Transfer Learning Benchmark},
  author={Yi Xin, Siqi Luo, Xuyang Liu, Haodi Zhou, Xinyu Cheng, Christina Luoluo Lee, Junlong Du, Yuntao Du., Haozhe Wang, MingCai Chen, Ting Liu, Guimin Hu, Zhongwei Wan, Rongchao Zhang, Aoxue Li, Mingyang Yi, Xiaohong Liu},
  year={2024}
}

```
