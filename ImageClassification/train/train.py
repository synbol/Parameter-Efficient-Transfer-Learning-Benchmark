
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
# from utils import *
import numpy as np
import psutil

def train(config, model, criterion, dl, opt, scheduler, logger, epoch, task, dataset):

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

    #     if ep % 10 == 9 and ep > 80:
    #         # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    #         acc = test(model, test_dl, task)
    #         if acc > config['best_acc']:
    #             config['best_acc'] = acc
    #             print(acc)
    #             save_path = './output/models_save/' + 'adaptformer_' + dataset + '.pth'
    #             torch.save(model, save_path)
    #             # save('vit_bitfit', config['task'], config['name'], model, acc, ep)
    #         logger.info(str(ep)+' '+str(acc)+' memory: '+str(memory_used)+'MB')
    # model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl, task):
    model.eval()
    acc = Accuracy()
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

