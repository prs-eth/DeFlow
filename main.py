import logging, sys
import torch
from configs import config

from models.deflow import DeFlowNet
from models.raft import RAFT
from libs.dataloader import get_dataloaders
from libs.trainer import Trainer
from libs.losses import SceneFlow_Loss, optical_flow_ft_loss
from toolbox.utils import setup_seed
from omegaconf import DictConfig
import argparse

def instantiate_config(cfg: dict):
    """instantiate all obejct from configurations
    
    Args:
        cfg (dict): dict from yaml

    Returns:
        args (dict): dict with instances
    """
    logging.info('Instantiating configurations')
    instances = dict()
    ## device
    if cfg.misc.use_gpu == True:
        instances['device'] = torch.device('cuda')
    else:
        instances['device'] = torch.device('cpu')
        
    ## dataloader
    instances['dataloader'] = get_dataloaders(cfg)
        
    ## model, optimizer, scheduler
    if cfg.network.model == 'deflow':
        instances['model'] = DeFlowNet(cfg)
        instances['loss'] = SceneFlow_Loss(cfg)
    elif cfg.network.model == 'raft':
        instances['model'] = RAFT(cfg.network)
        instances['loss'] = optical_flow_ft_loss(cfg)
    else: 
        raise NotImplementedError('Unknown model! Instantiation fails!')
    
    instances['optimizer'] = config.get_optimizer(cfg, instances['model'])
    instances['scheduler'] = config.get_scheduler(cfg, instances['optimizer'])
    
    
    return instances

def main(cfg_path:str):
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)])
    
    cfg = config.get_config(cfg_path)
    cfg = DictConfig(cfg)
    instances = instantiate_config(cfg)

    setup_seed(cfg.misc.seed)
    
    trainer = Trainer(cfg, instances)
    trainer.run()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default = 'configs/deflow_default.yaml', help='Path to pretrained weights')
    config_path = parser.parse_args().config_path
    main(config_path)

