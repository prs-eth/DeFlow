import os, torch
import numpy as np
from .dataset import DeFlow_Dataset, Full_DeFlow_Dataset, MultiFrame_DeFlow_Dataset
from .dataset import get_train_val_test_split

def get_dataloaders(config):

    dataset = DeFlow_Dataset(config)
    
    train_ids, val_ids, test_ids = get_train_val_test_split(len(dataset), config.dataset.val_ids, config.dataset.test_ids)
    
    train_set = torch.utils.data.Subset(dataset, train_ids)
    val_set = torch.utils.data.Subset(dataset, val_ids)
    test_set = torch.utils.data.Subset(dataset, test_ids)
    
    val_set.augmentation = False
    test_set.augmentation = False
    
    train_loader = torch.utils.data.DataLoader(train_set, 
                                        batch_size=config.train.batch_size, 
                                        num_workers=config.train.num_workers,
                                        shuffle=config.train.shuffle,
                                        pin_memory=False)
    
    val_loader = torch.utils.data.DataLoader(val_set, 
                                        batch_size=config.val.batch_size, 
                                        num_workers=config.val.num_workers)
    
    test_loader = torch.utils.data.DataLoader(test_set, 
                                        batch_size=config.test.batch_size, 
                                        num_workers=config.test.num_workers)
    return train_loader, val_loader


def get_full_dataloader(config):
    dataset = Full_DeFlow_Dataset(config)
    dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=config.val.batch_size, 
                                        num_workers=config.val.num_workers)
    
    return dataloader


def get_multiframe_dataloader(config):
    dataset = MultiFrame_DeFlow_Dataset(config)
    dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=config.val.batch_size, 
                                        num_workers=config.val.num_workers)
    
    return dataloader
    