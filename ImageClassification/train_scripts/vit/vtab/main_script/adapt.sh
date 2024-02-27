#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/caltech101  --dataset caltech101 --num-classes 102  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/caltech101/adapter \
	--amp --tuning-mode adapter --adapt-bottleneck 192 --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/cifar  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/cifar_100/adapter \
	--amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dtd --dataset dtd \
    --num-class 47 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-4 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dtd/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/oxford_flowers102 --dataset flowers102 \
    --num-class 102 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/oxford_flowers102/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/oxford_iiit_pet --dataset pets \
    --num-class 37 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-4 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/pets/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/svhn --dataset svhn \
    --num-class 10 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/svhn/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/sun397 --dataset sun397 \
    --num-class 397 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-4 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/sun397/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/patch_camelyon --dataset patch_camelyon \
    --num-class 2 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/patch_camelyon/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/eurosat --dataset eurosat \
    --num-class 10 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 3e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/eurosat/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/resisc45 --dataset resisc45 \
    --num-class 45 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/resisc45/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained 
    
wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/diabetic_retinopathy --dataset diabetic_retinopathy \
    --num-class 5 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-4 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/diabetic_retinopathy/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/clevr_count  --dataset clevr_count --num-classes 8  --no-aug  --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-4 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/clevr_count/adapter \
	--amp --tuning-mode adapter --adapt-bottleneck 192 --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/clevr_dist  --dataset clevr_dist --num-classes 6  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/clevr_dist/adapter \
	--amp --tuning-mode adapter --adapt-bottleneck 192 --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dmlab \
    --dataset dmlab --num-classes 6  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-4 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dmlab/adapter \
	--amp --tuning-mode adapter  --adapt-bottleneck 192 --pretrained  

wait

CUDA_VISIBLE_DEVICES=0  python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/kitti  --dataset kitti --num-classes 4  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/kitti/adapter \
	--amp --tuning-mode adapter --adapt-bottleneck 192 --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dsprites_loc  --dataset dsprites_loc --num-classes 16  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dsprites_loc/adapter \
	--amp --tuning-mode adapter --adapt-bottleneck 192 --pretrained \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dsprites_ori  --dataset dsprites_ori --num-classes 16  --no-aug   --direct-resize   --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dsprites_ori/adapter \
	--amp --tuning-mode adapter --adapt-bottleneck 192 --pretrained \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/smallnorb_azi  --dataset smallnorb_azi --num-classes 18  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/smallnorb_azi/adapter \
	--amp --tuning-mode adapter --adapt-bottleneck 192 --pretrained  

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/smallnorb_ele  --dataset smallnorb_ele --num-classes 9  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/smallnorb_ele/adapter \
	--amp --tuning-mode adapter --adapt-bottleneck 192 --pretrained  