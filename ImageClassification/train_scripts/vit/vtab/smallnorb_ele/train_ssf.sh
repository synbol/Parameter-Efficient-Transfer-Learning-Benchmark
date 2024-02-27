CUDA_VISIBLE_DEVICES=0 python train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/smallnorb_ele  --dataset smallnorb_ele --num-classes 9  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 --seed 0 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/smallnorb_ele/adapter \
	--amp --tuning-mode adapter --adapt-bottleneck 192 --pretrained  