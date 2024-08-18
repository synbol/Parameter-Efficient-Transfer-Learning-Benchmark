
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from torch.optim import AdamW, SGD
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy
from argparse import ArgumentParser
from utils import *
import numpy as np
import psutil


from models import vision_transformer_lora

# from logger import create_logger


def train(config, model, criterion, dl, opt, scheduler, logger, epoch, task):

    model.train()
    model = model.cuda()

    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            # torch.cuda.empty_cache()
            if task == 'vtab':
                x, y = batch[0].cuda(), batch[1].cuda()
            elif task == 'fgvc':
                if not isinstance(batch["image"], torch.Tensor):
                    for k, v in batch.items():
                        data[k] = torch.from_numpy(v)
                x = batch["image"].float().cuda()
                y = batch["label"].cuda()
            else:
                print("Error Task Name")
                break
            out = model(x)

            # loss = F.cross_entropy(out, y)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step(ep)
        
        ram_used = psutil.virtual_memory().used / (1024.0 * 1024.0)
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        # logger.info('RAM used: '+str(ram_used)+' memory: '+str(memory_used)+'MB')

        if ep % 10 == 9:
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            acc = test(model, test_dl, task)
            if acc > config['best_acc']:
                config['best_acc'] = acc
                print(acc)
                # save('vit_sct', config['task'], config['name'], model, acc, ep)
            logger.info(str(ep)+' '+str(acc)+' memory: '+str(memory_used)+'MB')
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl, task):
    model.eval()
    acc = Accuracy()
    #pbar = tqdm(dl)
    model = model.cuda()
    for batch in dl:  # pbar:
        torch.cuda.empty_cache()
        if task == 'vtab':
            x, y = batch[0].cuda(), batch[1].cuda()
        elif task == 'fgvc':
            if not isinstance(batch["image"], torch.Tensor):
                for k, v in batch.items():
                    data[k] = torch.from_numpy(v)
            x = batch["image"].float().cuda()
            y = batch["label"].cuda()
        # x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 0)

    return acc.result()[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--eval', type=str, default='True')
    parser.add_argument('--dpr', type=float, default=0.1)
    parser.add_argument('--topN', type=int, default=None)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k_lora')
    parser.add_argument('--model_checkpoint', type=str, default='./released_models/ViT-B_16.npz')
    parser.add_argument('--model_type', type=str, default='vit_lora')
    parser.add_argument('--task', type=str, default='vtab')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--tuning_mode', type=str, default='lora')

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    config = get_config('model_lora', args.task, args.dataset)

    if args.topN is not None:
        topN = args.topN
    else:
        topN = config['topN']

    exp_base_path = './output/%s/%s/%s'%(args.model_type, args.task, config['name']+'_dim_%d'%(topN))
    mkdirss(exp_base_path)
    logger = create_logger(log_path=exp_base_path, log_name='training')

    logger.info(args)
    logger.info(config)

    ## prepare training data
    if args.eval == 'True':
        evalflag = True
    else:
        evalflag = False

    if 'train_aug' in config.keys():
        train_aug = config['train_aug']
    else:
        train_aug = False
    
    if args.task == 'vtab':
        from vtab import *
        basedir = '../vtab-1k'
        train_dl, test_dl = get_data(basedir, args.dataset, logger, evaluate=evalflag, train_aug=train_aug, batch_size=config['batch_size'])

    elif args.task == 'fgvc':
        from dataload.loader import construct_train_loader, construct_test_loader
        train_dl = construct_train_loader(args.dataset, batch_size=config['batch_size'])
        test_dl = construct_test_loader(args.dataset, batch_size=config['batch_size'])
        print(len(train_dl), len(test_dl))


    if 'swin' in args.model:
        model = create_model(args.model, pretrained=False, drop_path_rate=args.dpr, tuning_mode=args.tuning_mode, topN=topN)
        model.load_state_dict(torch.load(args.model_checkpoint)['model'], False) ## not include adapt module
    else:
        model = create_model(args.model, checkpoint_path=args.model_checkpoint, drop_path_rate=args.dpr, tuning_mode=args.tuning_mode, topN=topN)

    model.reset_classifier(config['class_num'])    
    
    logger.info(str(model))

    config['best_acc'] = 0
    config['task'] = args.task

    trainable = []
    for n, p in model.named_parameters():
        if 'linear_a' in n or 'linear_b' in n or 'head' in n:
            trainable.append(p)
            logger.info(str(n))
        else:
            p.requires_grad = False

    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    if 'cycle_decay' in config.keys():
        cycle_decay = config['cycle_decay']
    else:
        # default 0.1
        cycle_decay = 0.1

    scheduler = CosineLRScheduler(opt, t_initial=config['epochs'],
                                  warmup_t=config['warmup_epochs'], lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=cycle_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of extra params:{}M".format(n_parameters/1000000))

    logger.info(f"number of extra params: {n_parameters}")

    if config['labelsmoothing'] > 0.:
        ## label smoothing
        criterion = LabelSmoothingCrossEntropy(smoothing=config['labelsmoothing'])
        logger.info('label smoothing')
    else:
        criterion = torch.nn.CrossEntropyLoss()
        logger.info('CrossEntropyLoss')
    
    model = train(config, model, criterion, train_dl, opt, scheduler, logger, config['epochs'], args.task)
    print(config['best_acc'])

    logger.info('end')