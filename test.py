
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib
import argparse
import os
import numpy as np
from utils import restrurct_hsi,down_sample,Metrics
# from model import * 

def test_regression_model(args, model, device, logger):
    if args.upscale == 4 or args.upscale == 2:
        args.patch_size = 64
        
    if args.upscale == 8:
        args.patch_size = 128

    if args.dataset == "PaviaC":
        args.dataset_path = './datasets/form_data/PaviaC_dataset.npz'
        args.hsi_bands = 103
        args.msi_bands = 4
        
    if args.dataset == "PaviaU":
        args.dataset_path = './datasets/form_data/PaviaU_dataset.npz'
        args.hsi_bands = 103
        args.msi_bands = 4
        
    if args.dataset == "Chikusei":
        args.dataset_path = './datasets/form_data/Chikusei_dataset.npz'
        args.hsi_bands = 128
        args.msi_bands = 4
        
    if args.dataset == "Houston":
        args.dataset_path = './datasets/form_data/Houston_dataset.npz'
        args.hsi_bands = 144
        args.msi_bands = 4
        
    if args.dataset == "CAVE":
        args.dataset_path = './datasets/form_data/CAVE_dataset.npz'
        args.hsi_bands = 31
        args.msi_bands = 3
        
    if args.dataset == "BayArea":
        args.dataset_path = './datasets/form_data/BayArea_dataset.npz'
        args.hsi_bands = 224
        args.msi_bands = 3
    
    if args.dataset == "Santa":
        args.dataset_path = './datasets/form_data/BayArea_dataset.npz'
        args.hsi_bands = 224
        args.msi_bands = 3
    
    np_dataset = np.load(args.dataset_path)
    test_gt_list = np_dataset['test_gt_list']
    test_hrmsi_list = np_dataset['test_hrmsi_list']
    model_state_dict = torch.load(args.check_point,map_location='cpu')['net']
    model.load_state_dict(model_state_dict)
    model.to(device)
    mean_psnr,mean_ssim,mean_ergas,mean_sam = 0,0,0,0
    sr_hsi_list = []
    
    
    logger.info('*' * 80)
    for idx in range(len(test_gt_list)):
        gt_hsi = torch.from_numpy(test_gt_list[idx],).float().to(device)
        msi = torch.from_numpy(test_hrmsi_list[idx]).float().to(device)
        lr_hsi = down_sample(gt_hsi)
        sr_hsi = restrurct_hsi(model,hsi=gt_hsi,msi=msi,lr_hsi=lr_hsi,upscale=args.upscale).to(device)
        sr_hsi_list.append(sr_hsi.detach().cpu().numpy())
        psnr = Metrics.cal_psnr(sr_hsi,gt_hsi)
        ssim = Metrics.cal_ssim(sr_hsi,gt_hsi)
        ergas = Metrics.cal_ergas(sr_hsi,gt_hsi)
        sam = Metrics.cal_sam(sr_hsi,gt_hsi)
        mean_ergas += ergas
        mean_psnr += psnr
        mean_sam += sam
        mean_ssim += ssim
        logger.info(f"Pic:{idx + 1}, psnr:{psnr:.4f}, ssim:{ssim:.4f}, sam:{sam:.4f}, ergas:{ergas:.4f}")
    logger.info('*' * 80)
    logger.info(f"Dataset:{args.dataset}x{args.upscale}, Model:{args.model}, Size:{args.model_size}, inference_time: {args.inference_time}, FLOPs: {args.FLOPs}")
    logger.info(f"mean_psnr:{mean_psnr/len(test_gt_list):.4f}, mean_ssim:{mean_ssim/len(test_gt_list):.4f}, mean_sam:{mean_sam/len(test_gt_list):.4f}, mean_ergas:{mean_ergas/len(test_gt_list):.4f}")
    logger.info('*' * 80)
    logger.info('\n')
    data_dict = {
        'sr_hsi_list':sr_hsi_list
    }
    np.savez(args.log_dir + '/' + f'{args.dataset}x{args.upscale},{args.model},sr_hsi.npz',**data_dict)