CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/cifar  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/cifar_100/vpt \
	--amp  --tuning-mode vpt --vpt-num 5 --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dtd --dataset dtd \
    --num-class 47 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dtd/vpt \
    --amp  --tuning-mode vpt --vpt-num 5 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/oxford_flowers102 --dataset flowers102 \
    --num-class 102 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/oxford_flowers102/vpt \
    --amp  --tuning-mode vpt --vpt-num 5 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/oxford_iiit_pet --dataset pets \
    --num-class 37 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/pets/vpt \
    --amp  --tuning-mode vpt --vpt-num 5 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/svhn --dataset svhn \
    --num-class 10 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/svhn/vpt \
    --amp  --tuning-mode vpt --vpt-num 5 --pretrained 

wait

CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/sun397 --dataset sun397 \
    --num-class 397 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-4 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/sun397/vpt \
    --amp  --tuning-mode vpt --vpt-num 5 --pretrained 
