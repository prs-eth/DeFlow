# portion of code adapted from 
# https://github.com/prs-eth/PCAccumulation 
# https://github.com/visinf/multi-mono-sf
# author: Liyuan Zhu

import numpy as np
import torch
import yaml, os
import torch.optim as optim

def init_dirs(options):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists(f"checkpoints/{options.exp_name}"):
        os.makedirs(f"checkpoints/{options.exp_name}")
    if not os.path.exists(f"checkpoints/{options.exp_name}/models"):
        os.makedirs(f"checkpoints/{options.exp_name}/models")
    os.system(f"cp main.py checkpoints/{options.exp_name}/main.py.backup")
    os.system(f"cp model.py checkpoints/{options.exp_name}/model.py.backup")
    os.system(f"cp data.py checkpoints/{options.exp_name}/data.py.backup")

def set_deterministic_seeds(options):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    np.random.seed(options.seed)

def add_config(parser):
    parser.add_argument('--exp_name', type=str, default='MMF', metavar='N', help='Name of the experiment.')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='Random seed (default: 1234).')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N', help='Number of workers for data loading.')
    # parser.add_argument('--is_training', type=bool, default=True, metavar='T', help='Set mode to train or validation.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/MSF/pretrained_model.ckpt',metavar='C', help='Checkpoint path')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_size_val", type=int, default=1)
    # parser.add_argument("--checkpoint", type=tools.str2str_or_none, default=None)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--evaluation", type=bool, default=False)
    # parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save", "-s", default="temp_exp/", type=str)
    # parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=4)
    
    parser.add_argument("--finetuning", type=bool, default=False)
    parser.add_argument("--calculate_disparity_scale", type=bool, default=False)
    parser.add_argument("--correlation_cuda_enabled", type=bool, default=False)
    parser.add_argument("--conv_padding_mode", type=str, default="zeros", choices=["zeros", "replicate", "reflect"])

    parser.add_argument("--save_out", type=bool, default=False)
    parser.add_argument("--save_vis", type=bool, default=False)


def get_config(path, default_path='configs/base.yaml'):
    ''' 
    Loads config file.
    
    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # load default setting
    with open(default_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' 
    Update two config dictionaries recursively.
    
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
            
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def get_optimizer(cfg, model):
    ''' 
    Returns an optimizer instance.
    Args:
        cfg (dict): config dictionary
        model (nn.Module): the model used for training
    Returns:
        optimizer (optimizer instance): optimizer used to train the network
    '''
    
    method = cfg.optimizer.name
    cfg['optimizer'] = cfg[method]

    if method == "SGD":
        optimizer = getattr(optim, method)(model.parameters(), 
                                           lr=cfg.optimizer.learning_rate,
                                           momentum=cfg.optimizer.momentum,
                                           weight_decay=cfg.optimizer.weight_decay,
                                           nesterov = cfg.optimizer.nesterov)

    elif method == "Adam":
        optimizer = getattr(optim, method)(model.parameters(), 
                                           lr=cfg.optimizer.learning_rate,
                                           weight_decay=cfg.optimizer.weight_decay)
    else: 
        print("{} optimizer is not implemented, must be one of the [SGD, Adam]".format(method))

    return optimizer


def get_scheduler(cfg, optimizer):
    ''' 
    Returns a learning rate scheduler
    Args:
        cfg (dict): config dictionary
        optimizer (torch.optim): optimizer used for training the network
    Returns:
        scheduler (optimizer instance): learning rate scheduler
    '''
    
    method = cfg['scheduler']['name']

    if method == "ExponentialLR":
        scheduler = getattr(optim.lr_scheduler, method)(optimizer, 
                                                        gamma=cfg.scheduler.exp_gamma)
    elif method == 'MultiStepLR':
        scheduler = getattr(optim.lr_scheduler, method)(optimizer, 
                                                        gamma=cfg.scheduler.gamma, 
                                                        milestones = cfg.scheduler.milestones)
    # elif method == 'OneCycleLR':
    #     steps_per_epoch = len(cfg['instances']['dataloader'][0]) // (cfg['train']['iter_size'] * cfg['train']['batch_size'])
        # scheduler = getattr(optim.lr_scheduler, method)(optimizer, 
        #                                                 max_lr = cfg['scheduler']['max_lr'],
        #                                                 epochs = cfg['train']['max_epoch'],
        #                                                 steps_per_epoch = steps_per_epoch)
    else: 
        print("{} scheduler is not implemented, must be one of the [ExponentialLR]".format(method))

    return scheduler

def dump_config(cfg, path):
    ''' 
    Save the config to the given path
    
    Args:
        cfg (dict): configuration parameters
        path (str): save path
    '''

    with open(path, 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)
        

if __name__ == "__main__":
    cfgs = get_config('configs/mmsf.yaml')
    