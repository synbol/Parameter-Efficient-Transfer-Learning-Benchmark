import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy
from argparse import ArgumentParser
from utils import *

import numpy as np
import psutil


from models import vision_transformer_adapter, vision_transformer_adaptformer, vision_transformer_fulltuning

# from logger import create_logger

def train(config, model, criterion, dl, opt, scheduler, epoch, tuning_mode):

    model.train()
    model = model.cuda()

    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            # torch.cuda.empty_cache()
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)

            # loss = F.cross_entropy(out, y)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step(ep)
        
        # ram_used = psutil.virtual_memory().used / (1024.0 * 1024.0)
        # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        # logger.info('*'*40)
        # logger.info('RAM used: '+str(ram_used)+' memory: '+str(memory_used)+'MB')

        if ep % 20 == 9 or ep > 90:
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            acc = test(model, test_dl)
            if acc > config['best_acc']:
                config['best_acc'] = acc
                save_path = './output/models_save'
                mkdirss(save_path)
                torch.save(model.state_dict(), save_path + '/' + tuning_mode +  '.pt')
                # save('vit_sct', config['task'], config['name'], model, acc, ep)
            # logger.info(str(ep)+' '+str(acc)+' memory: '+str(memory_used)+'MB')
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    #pbar = tqdm(dl)
    model = model.cuda()
    for batch in dl:  # pbar:
        torch.cuda.empty_cache()
        x, y = batch[0].cuda(), batch[1].cuda()
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
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k_fulltuning')
    parser.add_argument('--model_checkpoint', type=str, default='./released_models/ViT-B_16.npz')
    parser.add_argument('--model_type', type=str, default='vit_fulltuning')
    parser.add_argument('--task', type=str, default='vtab')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--tuning_mode', type=str, default='fulltuning')

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    config = get_config('model_fulltuning', args.task, args.dataset)   # get yaml config 


    # output file
    exp_base_path = './output/%s/%s/%s'%(args.model_type, args.task, config['name'].replace('fulltuning', args.tuning_mode))
    mkdirss(exp_base_path)
    logger = create_logger(log_path=exp_base_path, log_name='training')
    logger.info('*'*40)
    logger.info(args)
    logger.info(config)

    ## prepare training data
    if args.eval == 'True':
        evalflag = True
    else:
        evalflag = False

    if args.task == 'vtab':
        from vtab import *
        basedir = '/home/ma-user/work/haozhe/synbol/vtab-1k'
    elif args.task == 'fgvc':
        from fgvc import *

    if 'train_aug' in config.keys():
        train_aug = config['train_aug']
    else:
        train_aug = False

    train_dl, test_dl = get_data(basedir, args.dataset, evaluate=evalflag, train_aug=train_aug, batch_size=config['batch_size'])

    if 'swin' in args.model:
        model = create_model(args.model, pretrained=False, drop_path_rate=args.dpr, tuning_mode=args.tuning_mode)
        model.load_state_dict(torch.load(args.model_checkpoint)['model'], False) ## not include adapt module
    else:
        model = create_model(args.model, checkpoint_path=args.model_checkpoint, drop_path_rate=args.dpr, tuning_mode=args.tuning_mode)
    model.reset_classifier(config['class_num'])    
    # logger.info('*'*40)
    # logger.info(str(model))

    config['best_acc'] = 0
    config['task'] = args.task

    trainable = []
    logger.info('*'*40)
    for n, p in model.named_parameters():
        trainable.append(p)
        logger.info(str(n))

    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    
    if 'cycle_decay' in config.keys():
        cycle_decay = config['cycle_decay']
    else:
        cycle_decay = 0.1
    scheduler = CosineLRScheduler(opt, t_initial=config['epochs'],
                                  warmup_t=config['warmup_epochs'], lr_min=1e-6, warmup_lr_init=1e-4, cycle_decay=cycle_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info('*'*40)
    logger.info(f"number of extra params: {n_parameters/1000000}M")

    if config['labelsmoothing'] > 0.:
        ## label smoothing
        criterion = LabelSmoothingCrossEntropy(smoothing=config['labelsmoothing'])
        logger.info('*'*40)
        logger.info('label smoothing')
    else:
        criterion = torch.nn.CrossEntropyLoss()
        logger.info('*'*40)
        logger.info('CrossEntropyLoss')
    
    model = train(config, model, criterion, train_dl, opt, scheduler, config['epochs'], tuning_mode=args.tuning_mode)
    print(config['best_acc'])

    logger.info('end')