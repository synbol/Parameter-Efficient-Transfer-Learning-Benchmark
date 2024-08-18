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


from models import vision_transformer_sct

# from logger import create_logger

def getSalientChannels(model_name, checkpoint_path, train_dl, topN, emb_dim, layer_depth, tuning_mode):

    cali_model = create_model(model_name, checkpoint_path=checkpoint_path, drop_path_rate=0)
    
    with torch.no_grad():
        cali_model = cali_model.cuda()
        cali_model.eval()

        features_in_hook = []
        features_out_hook = []

        def hook(module, fea_in, fea_out):
            features_in_hook.append(fea_in[0].detach().cpu()[:, 1:, :])
            features_out_hook.append(fea_out.clone().detach().cpu()[:, 1:, :])
            return None

        
        if tuning_mode == 'sct_attn':
            layer_names = ['blocks.%d.norm2'%(i) for i in range(layer_depth)]

        elif tuning_mode == 'sct_mlp':
            layer_names = ['blocks.%d'%(i) for i in range(layer_depth)]

        elif tuning_mode == 'sct_qkv':
            layer_names = ['blocks.%d.attn.qkv'%(i) for i in range(layer_depth)]


        for i in range(len(layer_names)):
            layer_name = layer_names[i]

            for (name, module) in cali_model.named_modules():
                if name == layer_name:
                    module.register_forward_hook(hook=hook)

        labels = []
        for idx, (img, target) in enumerate(train_dl):

            # print(img.shape)
            # print(target.shape)

            labels.append(target.detach().cpu().numpy())
            y = cali_model(img.cuda())

            torch.cuda.empty_cache()


        if tuning_mode == 'sct_qkv':
            features = [torch.cat(features_out_hook[i::layer_depth], 0) for i in range(layer_depth)]
        elif tuning_mode == 'sct_attn':
            features = [torch.cat(features_in_hook[i::layer_depth], 0) for i in range(layer_depth)]
        elif tuning_mode == 'sct_mlp':
            features = [torch.cat(features_out_hook[i::layer_depth], 0) for i in range(layer_depth)]

    
    # print(labels, len(labels))
    labels_ = np.concatenate(labels)
    idx_l = [np.argwhere(labels_ == i).squeeze() for i in np.unique(labels_)]

    l2_value_all = []
    for ln in range(layer_depth):
        l2_value_onelayer = [torch.norm(features[ln][idx_l[i]].unsqueeze(0), 2, dim=(0, 1)) if idx_l[i].shape == () else torch.norm(features[ln][idx_l[i]], 2, dim=(0, 1)) for i in range(len(idx_l))]
        l2_value_onelayer = torch.stack(l2_value_onelayer, 0)
        l2_value_all.append(l2_value_onelayer)
    
    idx_selection_all = []

    for i in range(layer_depth):
        fea = torch.mean(l2_value_all[i], 0) # for CA

        if tuning_mode == 'sct_qkv':
            q, v = fea[:emb_dim], fea[-emb_dim:]

            qthrea = np.sort(q)[emb_dim-1-topN]
            qidx_selection = np.argwhere(q > qthrea)

            vthrea = np.sort(v)[emb_dim-1-topN]
            vidx_selection = np.argwhere(v > vthrea)

            idx_selection_all.append({
                'q': qidx_selection.squeeze(),
                'v': vidx_selection.squeeze(),
            })
        else:

            fea_s = np.sort(fea)
            threa = fea_s[emb_dim-1-topN]
            idx_selection = np.argwhere(fea > threa)
            idx_selection_all.append(idx_selection.squeeze())

    return {'index': idx_selection_all}


def train(config, model, criterion, dl, opt, scheduler, logger, epoch):

    model.train()
    model = model.cuda()

    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step(ep)
        
        # ram_used = psutil.virtual_memory().used / (1024.0 * 1024.0)
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        # logger.info('RAM used: '+str(ram_used)+' memory: '+str(memory_used)+'MB')

        if ep % 10 == 9 or ep > 50:
            acc = test(model, test_dl)
            if acc > config['best_acc']:
                config['best_acc'] = acc
                # save('vit_sct', config['task'], config['name'], model)
            logger.info(str(ep)+' '+str(acc)+' memory: '+str(memory_used)+'MB')
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    model = model.cuda()
    for batch in dl:
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
    parser.add_argument('--scale', type=float, default=None)
    parser.add_argument('--eval', type=str, default='True')
    parser.add_argument('--dpr', type=float, default=0.1)
    parser.add_argument('--topN', type=int, default=None)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k_sct')
    parser.add_argument('--base_model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--model_checkpoint', type=str, default='./released_models/ViT-B_16.npz')
    parser.add_argument('--model_type', type=str, default='vit_sct')
    parser.add_argument('--task', type=str, default='vtab')
    parser.add_argument('--basedir', type=str, default='../vtab-1k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--tuning_mode', type=str, default='sct_attn')

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    config = get_config('model_sct', args.task, args.dataset)
    
    if args.topN is not None:
        topN = args.topN
    else:
        topN = config['topN']

    exp_base_path = './output/%s/%s/%s'%(args.model_type, args.task, config['name'].replace('sct_attn', args.tuning_mode)+'_dim_%d'%(topN))
    
    mkdirss(exp_base_path)

    logger = create_logger(log_path=exp_base_path, log_name='training')
    logger.info(args)
    logger.info(config)

    ## prepare training data
    if args.eval == 'True':
        evalflag = True
    else:
        evalflag = False

    if args.task == 'vtab':
        from vtab import *
    
    basedir = args.basedir

    if 'train_aug' in config.keys():
        train_aug = config['train_aug']
    else:
        train_aug = False

    train_dl, test_dl = get_data(basedir, args.dataset, logger, evaluate=evalflag, train_aug=train_aug, batch_size=config['batch_size'])

    train_dl_noaug, test_dl_noaug = get_data(basedir, args.dataset, logger, evaluate=evalflag, train_aug=False, batch_size=50)

    ## calculate the salient channels
    selection_dict = getSalientChannels(args.base_model, args.model_checkpoint, train_dl_noaug, topN, 768, 12, args.tuning_mode)

    # print(selection_dict)
    
    if args.scale != None:
        scale = args.scale
    else:
        scale = config['scale']
    
    if 'swin' in args.model:
        model = create_model(args.model, pretrained=False, drop_path_rate=args.dpr, tuning_mode=args.tuning_mode, channel_index_dict=selection_dict, topN=topN, scale=scale, scaleonx=config['scaleonx'])
        model.load_state_dict(torch.load(args.model_checkpoint)['model'], False) ## not include adapt module
    else:
        model = create_model(args.model, checkpoint_path=args.model_checkpoint, drop_path_rate=args.dpr, tuning_mode=args.tuning_mode, channel_index_dict=selection_dict, topN=topN, scale=scale, scaleonx=config['scaleonx'])
    
    model.reset_classifier(config['class_num'])    
    
    logger.info(str(model))

    config['best_acc'] = 0
    config['task'] = args.task

    trainable = []
    for n, p in model.named_parameters():
        if 'q_l' in n or 'k_l' in n or 'v_l' in n or 'sct_attn' in n or 'sct_mlp' in n or 'head' in n:
            trainable.append(p)
            logger.info(str(n))
        else:
            p.requires_grad = False


    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    if 'cycle_decay' in config.keys():
        cycle_decay = config['cycle_decay']
    else:
        cycle_decay = 0.1
    
    scheduler = CosineLRScheduler(opt, t_initial=config['epochs'],
                                  warmup_t=config['warmup_epochs'], lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=cycle_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"number of extra params: {n_parameters}")
    print("number of extra params:{}M".format(n_parameters/1000000))
    if config['labelsmoothing'] > 0.:
        ## label smoothing
        criterion = LabelSmoothingCrossEntropy(smoothing=config['labelsmoothing'])
        logger.info('label smoothing')
    else:
        criterion = torch.nn.CrossEntropyLoss()
        logger.info('CrossEntropyLoss')
    
    
    model = train(config, model, criterion, train_dl, opt, scheduler, logger, config['epochs'])
    print(config['best_acc'])

    logger.info('end')