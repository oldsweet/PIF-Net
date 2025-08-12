
from ast import arg
from turtle import up
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR
from models import * 


def build_trainer(args):
    pass

def build_model(args):
    """
    build model
    args:
        args: arguments from command line and config file
    
    return:
        model: HISISR model
        loss_func: loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    import importlib
    mod = importlib.import_module('models')
    model_class = getattr(mod, args.model)
    model = model_class(args)
    loss_func = build_loss_func(args)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    return model,loss_func,optimizer,scheduler


def build_loss_func(args):
    """
    build loss function
    
    args:
        args: arguments from command line and config file
        
    return:
        loss_func: loss function
    """
    if args.loss_func == 'L1Loss':
        loss_func = nn.L1Loss()
    elif args.loss_func == 'MSELoss':
        loss_func = nn.MSELoss()
    elif args.loss_func == 'HuberLoss':
        loss_func = nn.HuberLoss()
    elif args.loss_func == 'SmoothL1Loss':
        loss_func = nn.SmoothL1Loss()
    else:
        raise NotImplementedError('Loss function [{:s}] is not found.'.format(args.loss_func))
    return loss_func


def build_optimizer(args, model):
    """
    build optimizer
    
    args:
        args: arguments from command line and config file
        model: HISISR model
        
    return:
        optimizer: optimizer
    """
    if args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(lr=args.lr,params=model.parameters())
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(lr=args.lr,params=model.parameters(),momentum=0.9,weight_decay=1e-4)
    else:
        raise NotImplementedError('Optimizer [{:s}] is not found.'.format(args.optimizer))  
    return optimizer


def build_scheduler(args, optimizer):
    """
    build scheduler
    
    args:
        args: arguments from command line and config file
        optimizer: optimizer
        
    return:
        scheduler: scheduler
    """
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.decay)
    return scheduler