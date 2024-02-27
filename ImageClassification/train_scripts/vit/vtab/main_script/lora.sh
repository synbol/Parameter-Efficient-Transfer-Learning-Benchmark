CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/clevr_count  --dataset clevr_count --num-classes 8  --no-aug  --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-3 \
    --warmup-lr 1e-6 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-5 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/clevr_count/lora \
	--amp --tuning-mode lora --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/clevr_dist  --dataset clevr_dist --num-classes 6  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-3 \
    --warmup-lr 1e-6 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-5 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/clevr_dist/lora \
	--amp --tuning-mode lora --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dmlab  --dataset dmlab --num-classes 6  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 1e-4 \
    --warmup-lr 1e-6 --warmup-epochs 10  \
    --lr 1e-4 --min-lr 1e-5 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dmlab/lora \
	--amp --tuning-mode lora --pretrained  

wait

CUDA_VISIBLE_DEVICES=0  python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/kitti  --dataset kitti --num-classes 4  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-3 \
    --warmup-lr 1e-6 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-5 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/kitti/lora \
	--amp --tuning-mode lora --pretrained  \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dsprites_loc  --dataset dsprites_loc --num-classes 16  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-6 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-5 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dsprites_loc/lora \
	--amp --tuning-mode lora --pretrained \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/dsprites_ori  --dataset dsprites_ori --num-classes 16  --no-aug   --direct-resize   --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-6 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-5 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dsprites_ori/lora \
	--amp --tuning-mode lora --pretrained \

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/smallnorb_azi  --dataset smallnorb_azi --num-classes 18  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-6 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-5 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/smallnorb_azi/lora \
	--amp --tuning-mode lora --pretrained  

wait

CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/smallnorb_ele  --dataset smallnorb_ele --num-classes 9  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-6 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-5 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/smallnorb_ele/lora \
	--amp --tuning-mode lora --pretrained  