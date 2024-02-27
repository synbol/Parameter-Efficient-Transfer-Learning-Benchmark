import subprocess

dataset = []

for data in dataset:
    
    sh_str = f"CUDA_VISIBLE_DEVICES=0  --dataset {data} python /home/p1/zhd/Parameter-Efficient-Benchmark/train.py ~/Parameter-Efficient-Benchmark/path/to/vtab-1k/caltech101  --dataset caltech101 --num-classes 102  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  --batch-size 32 --epochs 100 --seed 0 --opt adamw  --weight-decay 5e-2  --warmup-lr 1e-7 --warmup-epochs 10  --lr 1e-3 --min-lr 1e-8 --drop-path 0.1 --img-size 224 --mixup 0 --cutmix 0 --smoothing 0 --output  output/vit_base_patch16_224_in21k/vtab/caltech101/test --amp  --tuning-mode adapter --pretrained"

    out = subprocess.getoutput(sh_str)
    out.split('/n')

print(out[-1])

