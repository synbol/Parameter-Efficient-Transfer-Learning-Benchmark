CUDA_VISIBLE_DEVICES=0 python train.py  ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/sun397 --dataset sun397 \
    --num-class 397 --no-aug --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
    --opt adamw --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 3e-3 --min-lr 1e-8 \
    --drop-path 0. --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/sun397/adapter \
    --amp  --tuning-mode adapter --adapt-bottleneck 192 --pretrained