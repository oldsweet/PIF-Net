import torch
import logging
import os
import json
from utils import *
from model import *
from data import *
from train import *
from test import *
from options import args


if __name__ == '__main__':
    # Enable cudnn benchmark for speedup
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    set_seed(args.seed)

    print(f"Using device: {device}")
    # ===== Log directory setup =====
    if args.check_point is None:
        log_dir = f'./results/{args.dataset}x{args.upscale}/{args.model},{beijing_time()},{os.getpid()}'
        if not os.path.exists(log_dir) and args.log == 1:
            os.makedirs(log_dir)
        args.log_dir = log_dir
        args.start_epoch = 0
    else:
        if args.train == 1:
            args.log_dir = os.path.dirname(args.check_point)
        else:
            log_dir = f'./outputs/{args.dataset}x{args.upscale}/{args.model},{beijing_time()},{os.getpid()}'
            os.makedirs(log_dir, exist_ok=True)
            args.log_dir = log_dir

    # ===== Initialize logger =====
    logger = set_logger(args)
    
    # ===== Load dataset =====
    train_dataloader, test_dataloader = build_dataset(args, logger)

    # ===== Build model =====
    model, loss_func, optimizer, scheduler = build_model(args)
    model.to(device)
        
    # ===== Model statistics =====
    inference_time, flops, params, params_table = test_speed_thop(args, model)
    
    # ===== Print parameter summary table =====
    if args.param_table == 1:
        logger.info(params_table)
        
    # ===== Log startup info =====
    args.model_size = f"{params}M"
    args.inference_time = f"{(inference_time * 1000):.6f}ms"
    args.FLOPs = f"{flops}G"
    args.pid = os.getpid()

    logger.info("=" * 80)
    logger.info("[Startup Configuration]")
    log_args_summary(args, logger)

    logger.info("[System Resource Summary]")
    log_items = {
        "Model Size": args.model_size,
        "Inference Time": args.inference_time,
        "FLOPs": args.FLOPs,
        "Log Directory": args.log_dir,
    }
    for key, val in log_items.items():
        logger.info(f"{key}: {val}")
    logger.info("=" * 80)

    # ===== Start training =====
    logger.info("[Starting Training]")
    if args.train == 1:
        trainer = BaseTrainer(args, model, train_dataloader, test_dataloader,
                              optimizer, loss_func, scheduler, logger)
        trainer.fit()
