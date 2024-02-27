CUDA_VISIBLE_DEVICES=0,1,  python -m torch.distributed.launch --nproc_per_node=2  --master_port=12341 \
    train.py /path/to/oxford_flowers  --dataset oxford_flowers --num-classes 102 --val-split val --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--model-ema --model-ema-decay 0.999  \
	--output  output/vit_base_patch16_224_in21k/oxford_flowers/ssf \
	--amp --tuning-mode ssf --pretrained  