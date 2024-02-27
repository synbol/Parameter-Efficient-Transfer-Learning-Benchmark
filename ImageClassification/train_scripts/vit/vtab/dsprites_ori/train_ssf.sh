CUDA_VISIBLE_DEVICES=0,1,  python  -m torch.distributed.launch --nproc_per_node=2  --master_port=12002  \
	train.py /path/to/vtab-1k/dsprites_ori  --dataset dsprites_ori --num-classes 16  --no-aug   --direct-resize   --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/dsprites_ori/ssf \
	--amp --tuning-mode ssf --pretrained \
	