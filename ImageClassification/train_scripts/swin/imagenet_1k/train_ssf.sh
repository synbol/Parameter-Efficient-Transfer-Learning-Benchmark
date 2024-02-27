CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,  python  -m torch.distributed.launch --nproc_per_node=8  --master_port=33518 \
	train.py /path/to/imagenet_1k --dataset imagenet --num-classes 1000 --model swin_base_patch4_window7_224_in22k \
    --batch-size 32 --epochs 30 \
	--opt adamw --weight-decay 0.05 \
    --warmup-lr 5e-7 --warmup-epochs 5  \
    --lr 5e-3 --min-lr 5e-8 \
    --drop-path 0.1 --img-size 224 \
	--output  output/swin_base_patch4_window7_224_in22k/imagenet_1k/ssf \
	--amp --tuning-mode ssf --pretrained  