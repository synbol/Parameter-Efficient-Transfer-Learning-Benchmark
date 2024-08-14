## <p style="font-size: 20px;" align=center>𝓥𝓲𝓼𝓾𝓪𝓵 𝓟𝓪𝓻𝓪𝓶𝓮𝓽𝓮𝓻-𝓔𝓯𝓯𝓲𝓬𝓲𝓮𝓷𝓽 𝓣𝓻𝓪𝓷𝓼𝓯𝓮𝓻 𝓛𝓮𝓪𝓻𝓷𝓲𝓷𝓰 𝓑𝓮𝓷𝓬𝓱𝓶𝓪𝓻𝓴</p>
<div align=center>
<p>
 
 ![GitHub stars](https://img.shields.io/github/stars/synbol/Parameter-Efficient-Transfer-Learning-Benchmark.svg?color=red&style=for-the-badge) 
 ![GitHub forks](https://img.shields.io/github/forks/synbol/Parameter-Efficient-Transfer-Learning-Benchmark.svg?style=for-the-badge) 
 ![GitHub contributors](https://img.shields.io/github/contributors/synbol/Parameter-Efficient-Transfer-Learning-Benchmark.svg?style=for-the-badge) 
 ![GitHub activity](https://img.shields.io/github/last-commit/synbol/Parameter-Efficient-Transfer-Learning-Benchmark?style=for-the-badge) 
</p>
</div>

<div align="center">
  <p align="center">
<!--     <a href="[111](111)">Paper</a> -->
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

* ✅ [2024/07/01] Visual PEFT Benchmark starts releasing code, etc.

* ✅ [2024/06/20] Visual PEFT Benchmark homepage is created.

* ✅ [2024/06/01] Visual PEFT Benchmark repo is created.


## 🔥 <span id="head1"> *Introduction* </span>

Parameter-efficient transfer learning (PETL) methods show promise in adapting a pre-trained model to various downstream tasks while training only a few parameters. In the computer vision (CV) domain, numerous PETL algorithms have been proposed, but their direct employment or comparison remains inconvenient. To address this challenge, we construct a Unified Visual PETL Benchmark (V-PETL Bench) for the CV domain by selecting 30 diverse, challenging, and comprehensive datasets from image recognition, video action recognition, and dense prediction tasks. On these datasets, we systematically evaluate 25 dominant PETL algorithms and open-source a modular and extensible codebase for a fair evaluation of these algorithms. V-PETL Bench runs on NVIDIA A800 GPUs and requires approximately 310 GPU days. We release all the checkpoints and training logs, making them more efficient and friendly to researchers. Additionally, V-PETL Bench
13 will be continuously updated for new PETL algorithms and CV tasks.

## 🔥 <span id="head1"> *Getting Started* </span>

This is an example of how to set up V-PETL Bench locally. 

To get a local copy up, running follow these simple example steps.

### Environment Setup

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

### Data Preparation

#### Image Classification Dataset

- **1. Visual Task Adaptation Benchmark (VTAB)**

    VTAB comprises 19 diverse visual classification datasets. We have processed all the dataset and the data can be downloaded here: [Download Link](https://huggingface.co/datasets/XiN0919/VTAB-1k/tree/main). For specific processing procedures and tips, please see [VTAB_SETUP]().

- **2. Fine-Grained Visual Classification tasks (FGVC)**

    FGVC comprises 5 fine-grained visual classification dataset. The datasets can be downloaded following the official links. We split the training data if the public validation set is not available. The splitted dataset can be found here: [Download Link](https://huggingface.co/datasets/XiN0919/FGVC/resolve/main/json.zip?download=true).
  
     - [CUB200 2011](https://data.caltech.edu/records/65de6-vp158)
   
     - [NABirds](http://info.allaboutbirds.org/nabirds/)
   
     - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)
   
     - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)
   
     - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) 

#### Video Action Recognition Dataset

- **1. Kinetics-400**

- **2. Something-Something V2(SSv2)**

- **3. HMDB51**

#### Dense Prediction Dataset

- **1. MS-COCO**

- **2. ADE20K**

- **3. PASCAL VOC**

### Pre-trained Model Preperation


### Quick Start

### Results




## ⭐ <span id="head1"> *Citation* </span>

If you find our survey and repository useful for your research, please cite it below:

```bibtex
@article{xin2024bench,
  title={V-PETL Bench: A Unified Visual Parameter-Efficient Transfer Learning Benchmark},
  author={Yi Xin, Siqi Luo, Xuyang Liu, Haodi Zhou, Xinyu Cheng, Christina Luoluo Lee, Junlong Du, Yuntao Du., Haozhe Wang, MingCai Chen, Ting Liu, Guimin Hu, Zhongwei Wan, Rongchao Zhang, Aoxue Li, Mingyang Yi, Xiaohong Liu},
  year={2024}
}

```


