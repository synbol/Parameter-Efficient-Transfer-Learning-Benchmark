CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/caltech101  --dataset caltech101 --num-classes 102  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/caltech101/bitfit \
	--amp --tuning-mode bitfit --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/cifar  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/cifar_100/bitfit \
	--amp  --tuning-mode bitfit --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dtd  --dataset dtd --num-classes 47  --no-aug --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  output/vit_base_patch16_224_in21k/vtab/dtd/bitfit \
	--amp --tuning-mode bitfit --pretrained  \
	--mixup 0 --cutmix 0 --smoothing 0

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/oxford_flowers102 --dataset flowers102 --num-classes 102  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/flowers102/bitfit \
	--amp --tuning-mode bitfit --pretrained  

wait


CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/oxford_iiit_pet  --dataset pets --num-classes 37  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/pets/bitfit \
	--amp --tuning-mode bitfit --pretrained  

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/svhn  --dataset svhn --num-classes 10  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/svhn/bitfit \
	--amp --tuning-mode bitfit --pretrained  


wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/sun397  --dataset sun397 --num-classes 397  --no-aug --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/sun397/bitfit \
	--amp --tuning-mode bitfit --pretrained  

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/patch_camelyon  --dataset patch_camelyon --num-classes 2  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/patch_camelyon/bitfit \
	--amp --tuning-mode bitfit --pretrained  

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/eurosat  --dataset eurosat --num-classes 10  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 3e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/eurosat/bitfit \
	--amp --tuning-mode bitfit --pretrained  

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/resisc45  --dataset resisc45 --num-classes 45  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/resisc45/bitfit \
	--amp --tuning-mode bitfit --pretrained  

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/diabetic_retinopathy  --dataset diabetic_retinopathy --num-classes 5  --no-aug --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/diabetic_retinopathy/bitfit \
	--amp --tuning-mode bitfit --pretrained  

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/clevr_count  --dataset clevr_count --num-classes 8  --no-aug  --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/clevr_count/bitfit \
	--amp --tuning-mode bitfit --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/clevr_dist  --dataset clevr_dist --num-classes 6  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-2 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/clevr_dist/bitfit \
	--amp --tuning-mode bitfit --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dmlab  --dataset dmlab --num-classes 6  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dmlab/bitfit \
	--amp --tuning-mode bitfit --pretrained  

wait

CUDA_VISIBLE_DEVICES=0  python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/kitti  --dataset kitti --num-classes 4  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/kitti/bitfit \
	--amp --tuning-mode bitfit --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dsprites_loc  --dataset dsprites_loc --num-classes 16  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dsprites_loc/bitfit \
	--amp --tuning-mode bitfit --pretrained \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dsprites_ori  --dataset dsprites_ori --num-classes 16  --no-aug   --direct-resize   --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dsprites_ori/bitfit \
	--amp --tuning-mode bitfit --pretrained \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/smallnorb_azi  --dataset smallnorb_azi --num-classes 18  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-2 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/smallnorb_azi/bitfit \
	--amp --tuning-mode bitfit --pretrained  

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/smallnorb_ele  --dataset smallnorb_ele --num-classes 9  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 2 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/smallnorb_ele/bitfit \
	--amp --tuning-mode bitfit --pretrained  