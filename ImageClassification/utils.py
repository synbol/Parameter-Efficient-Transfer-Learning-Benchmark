import os
import torch
import random
import numpy as np
import yaml
import logging
from pathlib import Path

def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def save(model_type, task, name, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if 'sct_mlp' in n or 'sct_mlp' in n or 'head' in n or 'q_l' in n or 'k_l' in n or 'v_l' in n:
            trainable[n] = p.data

    torch.save(trainable, '../output')
    

def load(model_type, task, name, model):
    model = model.cpu()
    st = torch.load('../output/%s/%s/%s/ckpt_epoch_best.pt'%(model_type, task, name))
    model.load_state_dict(st, False)
    return model

def get_config(model_type, task, dataset_name):
    with open('Parameter-Efficient-Transfer-Learning-Benchmark/ImageClassification/configs/%s/%s/%s.yaml'%(model_type, task, dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config



def create_logger(log_path, log_name):
    """
    Creates a logger to log messages to a file.

    :param log_path: The path where the log file should be saved.
    :param log_name: The name of the log file.
    :return: A logger instance.
    """
    # Create the directory if it doesn't exist
    Path(log_path).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)  # Set the logging level to INFO
    
    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(f"{log_path}/{log_name}.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    return logger
