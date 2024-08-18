model=vit_base_patch16_224_in21k_fulltuning
model_type=vit_fulltuning
model_checkpoint=./released_models/ViT-B_16.npz

CUDA_VISIBLE_DEVICES=0 python train_model_fulltuning.py --dataset cifar100 --task vtab --lr 5e-4 --wd 1e-1 --eval True --dpr 0.1 --tuning_mode fulltuning --model_type $model_type --model $model --model_checkpoint $model_checkpoint

CUDA_VISIBLE_DEVICES=0 python train_model_fulltuning.py --dataset caltech101 --task vtab --lr 5e-4 --wd 5e-3 --eval True --dpr 0.1 --tuning_mode fulltuning --model_type $model_type --model $model --model_checkpoint $model_checkpoint

CUDA_VISIBLE_DEVICES=0 python train_model_fulltuning.py --dataset dtd --task vtab --lr 1e-4 --wd 5e-1 --eval True --dpr 0.1 --tuning_mode fulltuning --model_type $model_type --model $model --model_checkpoint $model_checkpoint

CUDA_VISIBLE_DEVICES=0 python train_model_fulltuning.py --dataset dtd --task vtab --lr 1e-4 --wd 5e-2 --eval True --dpr 0.1 --tuning_mode fulltuning --model_type $model_type --model $model --model_checkpoint $model_checkpoint

CUDA_VISIBLE_DEVICES=0 python train_model_fulltuning.py --dataset dtd --task vtab --lr 1e-4 --wd 5e-3 --eval True --dpr 0.1 --tuning_mode fulltuning --model_type $model_type --model $model --model_checkpoint $model_checkpoint

CUDA_VISIBLE_DEVICES=0 python train_model_fulltuning.py --dataset dtd --task vtab --lr 1e-4 --wd 5e-4 --eval True --dpr 0.1 --tuning_mode fulltuning --model_type $model_type --model $model --model_checkpoint $model_checkpoint

CUDA_VISIBLE_DEVICES=0 python train_model_fulltuning.py --dataset dtd --task vtab --lr 5e-5 --wd 5e-2 --eval True --dpr 0.1 --tuning_mode fulltuning --model_type $model_type --model $model --model_checkpoint $model_checkpoint

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset dtd --task vtab --lr 5e-3 --wd 4e-2 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset oxford_flowers102 --task vtab --lr 5e-3 --wd 5e-5 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset oxford_iiit_pet --task vtab --lr 5e-3 --wd 5e-5 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset svhn --task vtab --lr 1e-3 --wd 6e-3 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset sun397 --task vtab --lr 2e-3 --wd 1e-2 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset patch_camelyon --task vtab --lr 7e-3 --wd 2e-3 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset eurosat --task vtab --lr 3e-3 --wd 5e-2 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset resisc45 --task vtab --lr 1e-2 --wd 1e-2 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset diabetic_retinopathy --task vtab --lr 4e-3 --wd 3e-4 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset clevr_count --task vtab --lr 3e-3 --wd 1e-2 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset clevr_dist --task vtab --lr 2e-3 --wd 5e-3 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset dmlab --task vtab --lr 9e-3 --wd 1e-1 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset kitti --task vtab --lr 1e-2 --wd 1e-4 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset dsprites_loc --task vtab --lr 3e-3 --wd 1e-3 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset dsprites_ori --task vtab --lr 1e-2 --wd 5e-2 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset smallnorb_azi --task vtab --lr 6e-3 --wd 5e-3 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset smallnorb_ele --task vtab --lr 1e-3 --wd 5e-4 --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

