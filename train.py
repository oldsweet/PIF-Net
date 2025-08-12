import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from PIL import Image

class BaseTrainer:
    def __init__(self, args, model, train_loader, val_loader, optimizer, loss_fn, scheduler, logger):
        self.args = args
        self.model = model.to(args.device)
        self.device = torch.device(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.logger = logger

    def get_results(self, args, model):
        device = torch.device(args.device)
        loaded_dataset = np.load(args.dataset_path)
        test_gt_list = loaded_dataset['test_gt_list']
        test_hrmsi_list = loaded_dataset['test_hrmsi_list']
        mean_psnr, mean_ssim, mean_sam, mean_ergas = 0, 0, 0, 0

        if args.upscale in [2, 4]:
            patch_size = stride = 64
        elif args.upscale == 8:
            patch_size = stride = 32

        pre_hsi_list = []
        metric_list = []
        for idx in range(len(test_gt_list)):
            gt, hrmsi = test_gt_list[idx], test_hrmsi_list[idx]
            gt, hrmsi = torch.from_numpy(gt).float().to(device), torch.from_numpy(hrmsi).float().to(device)
            lrhsi = down_sample(gt, args.upscale)
            pre_hsi = restrurct_hsi(model, gt, lrhsi, hrmsi, args.upscale, patch_size, stride)
            pre_hsi.clip_(0, 1)

            psnr, ssim, sam, ergas = (
                Metrics.cal_psnr(pre_hsi, gt),
                Metrics.cal_ssim(pre_hsi, gt),
                Metrics.cal_sam(pre_hsi, gt),
                Metrics.cal_ergas(pre_hsi, gt),
            )
            pre_hsi_list.append(pre_hsi.detach().cpu().numpy())
            metric_list.append({
                'psnr': psnr.item(),
                'ssim': ssim.item(),
                'sam': sam.item(),
                'ergas': ergas,
            })
            mean_psnr += psnr
            mean_ssim += ssim
            mean_sam += sam
            mean_ergas += ergas

        mean_psnr /= len(test_gt_list)
        mean_ssim /= len(test_gt_list)
        mean_sam /= len(test_gt_list)
        mean_ergas /= len(test_gt_list)
        mean_metrics = {
            'mean_psnr': mean_psnr.item(),
            'mean_ssim': mean_ssim.item(),
            'mean_sam': mean_sam.item(),
            'mean_ergas': mean_ergas,
        }
        return pre_hsi_list, metric_list, mean_metrics

    def val(self, epoch):
        # 用原始模型做验证
        self.model.eval()

        val_loss = 0
        mean_psnr = 0
        with torch.no_grad():
            for _, (gt_batch, lrhsi_batch, hrmsi_batch) in enumerate(self.val_loader):
                gt_batch = gt_batch.to(self.device)
                hrmsi_batch = hrmsi_batch.to(self.device)
                lrhsi_batch = lrhsi_batch.to(self.device)
                sr_batch = self.model(lrhsi_batch, hrmsi_batch).clip(0, 1)
                mean_psnr += Metrics.cal_psnr(sr_batch, gt_batch).item()
                val_loss += self.loss_fn(sr_batch, gt_batch).item()

        mean_psnr /= len(self.val_loader)
        val_loss /= len(self.val_loader)
        self.logger.info(f"{beijing_time()}, model:{self.args.model}, dataset:{self.args.dataset}x{self.args.upscale}, epoch:{epoch}/{self.args.epochs}, val_loss:{val_loss:.4f}, PSNR:{mean_psnr:.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        self.model.train()

    def test(self, epoch):
        # 用原始模型做最终测试
        with torch.no_grad():
            self.logger.info('')
            self.logger.info(f' model_name:{self.args.model}, dataset:{self.args.dataset}x{self.args.upscale} '.center(100, '='))
            pre_hsi_list, metric_list, mean_metrics = self.get_results(self.args, self.model)
            mean_psnr, mean_ssim, mean_sam, mean_ergas = (
                mean_metrics['mean_psnr'],
                mean_metrics['mean_ssim'],
                mean_metrics['mean_sam'],
                mean_metrics['mean_ergas'],
            )
            self.logger.info(f'model_size:{self.args.model_size}, inference_time:{self.args.inference_time}, FLOPs:{self.args.FLOPs}')
            self.logger.info(f'mean_psnr:{mean_psnr:.4f}, mean_ssim:{mean_ssim:.4f}, mean_sam:{mean_sam:.4f},mean_ergas:{mean_ergas:.4f}')
            self.logger.info(''.center(100, '-'))
            for idx, metric in enumerate(metric_list):
                psnr, ssim, sam, ergas = metric['psnr'], metric['ssim'], metric['sam'], metric['ergas']
                self.logger.info(f'PIC-{idx}: psnr:{psnr:.4f}, ssim:{ssim:.4f}, sam:{sam:.4f}, ergas:{ergas:.4f}')

            if self.args.log == 1:
                # np.savez(os.path.join(self.args.log_dir, f'{self.args.model}_{self.args.dataset}x{self.args.upscale}_{epoch}.npz'),
                #          pre_hsi_list=pre_hsi_list, metric_list=metric_list, mean_metrics=mean_metrics)
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.log_dir, f'{self.args.model}_{self.args.dataset}x{self.args.upscale}_{epoch}_{mean_psnr:.4f}.pth'))
            self.logger.info(''.center(100, '='))
            self.logger.info('')

    def fit(self):
        self.model.train()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.is_train == 0:
                break
            for _, (gt_batch, lrhsi_batch, hrmsi_batch) in enumerate(self.train_loader):
                gt_batch = gt_batch.to(self.device)
                lrhsi_batch = lrhsi_batch.to(self.device)
                hrmsi_batch = hrmsi_batch.to(self.device)
                self.optimizer.zero_grad()
                sr_batch = self.model(lrhsi_batch, hrmsi_batch)
                loss = self.loss_fn(sr_batch, gt_batch)
                loss.backward(retain_graph=True)
                self.optimizer.step()

            self.scheduler.step()

            # 验证
            if (epoch + 1) % self.args.val_step == 0:
                self.val(epoch)

            # 测试 & 保存
            if (epoch + 1) % self.args.save_step == 0:
                self.test(epoch)

        self.test(epoch)  # 最后测试一次
