CUDA_VISIBLE_DEVICES=0,1,  python  -m torch.distributed.launch --nproc_per_node=2  --master_port=12319 \
    train.py /path/to/stanford_dogs  --dataset stanford_dogs --num-classes 120 --simple-aug    --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2.5e-4 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
    --model-ema --model-ema-decay 0.9998  \
	--output  output/vit_base_patch16_224_in21k/stanford_dogs/ssf \
	--amp --tuning-mode ssf --pretrained  \
