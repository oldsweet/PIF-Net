import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional
from utils import * 
import logging

import numpy as np
import matplotlib.pyplot as plt

import random

def random_augment(gt, hrmsi, lrhsi):
    # 以相同方式增强三个patch，保证对齐

    # 随机水平翻转
    if random.random() > 0.5:
        gt = torch.flip(gt, dims=[1])       # 翻转高维度H或W，根据shape调整
        hrmsi = torch.flip(hrmsi, dims=[1])
        lrhsi = torch.flip(lrhsi, dims=[1])

    # 随机垂直翻转
    if random.random() > 0.5:
        gt = torch.flip(gt, dims=[2])
        hrmsi = torch.flip(hrmsi, dims=[2])
        lrhsi = torch.flip(lrhsi, dims=[2])

    # 随机90度旋转 (0/90/180/270度)
    k = random.randint(0, 3)
    gt = torch.rot90(gt, k, dims=[1, 2])
    hrmsi = torch.rot90(hrmsi, k, dims=[1, 2])
    lrhsi = torch.rot90(lrhsi, k, dims=[1, 2])

    return gt, hrmsi, lrhsi


def tensor_to_normalized_rgb(img_tensor):
    if not isinstance(img_tensor, np.ndarray):
        img_tensor = img_tensor.cpu().numpy()

    # 确认形状为3维 (C,H,W)
    if img_tensor.ndim != 3:
        raise ValueError(f"Expect img_tensor with 3 dims (C,H,W), got shape {img_tensor.shape}")

    c = img_tensor.shape[0]
    if c >= 3:
        img = img_tensor[:3]
    else:
        img = np.concatenate([img_tensor] + [img_tensor[-1:]]*(3 - c), axis=0)

    img = np.transpose(img, (1, 2, 0))  # 转HWC
    return img


def show_three_patches(gt_patch, hrmsi_patch, lrhsi_patch, save_path=None):
    """
    展示第 index 张patch的三个tensor：
    gt_patch, hrmsi_patch, lrhsi_patch
    它们形状都为 (B, C, H, W)，取第 index 张图，
    只用前三通道显示。
    """
    gt_img = tensor_to_normalized_rgb(gt_patch)
    hrmsi_img = tensor_to_normalized_rgb(hrmsi_patch)
    lrhsi_img = tensor_to_normalized_rgb(lrhsi_patch)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs[0].imshow(gt_img)
    axs[0].set_title('GT Patch')
    axs[1].imshow(hrmsi_img)
    axs[1].set_title('HRMSI Patch')
    axs[2].imshow(lrhsi_img)
    axs[2].set_title('LRHSI Patch')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ========= NPZ Dataset 类 =========
class NPZDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 is_train: bool = True,
                 upscale: int = 4,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None,
                 data_size=-1):
        super(NPZDataset, self).__init__()

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        loaded_data = np.load(data_path)
        if upscale == 4 or upscale == 2:
            patch_size, stride = 128, 32
        elif upscale == 8:
            patch_size, stride = 128, 32
        else:
            raise ValueError(f"Unsupported upscale factor: {upscale}")

        if is_train:
            gt_list = loaded_data['train_gt_list']
            hrmsi_list = loaded_data['train_hrmsi_list']
        else:
            gt_list = loaded_data['test_gt_list']
            hrmsi_list = loaded_data['test_hrmsi_list']
        
        gt_list = np.array(gt_list)
        hrmsi_list = np.array(hrmsi_list)
   
        self.gt_list = torch.from_numpy(gt_list).float()
        self.hrmsi_list = torch.from_numpy(hrmsi_list).float()
        
        self.gt_patch_list = []
        self.hrmsi_patch_list = []
        self.lrhsi_patch_list = []
        for i in range(len(self.gt_list)):
            self.gt_patch_list.extend(crop_patches(self.gt_list[i], stride, patch_size))
            self.hrmsi_patch_list.extend(crop_patches(self.hrmsi_list[i], stride, patch_size))
            lrhsi = down_sample(self.gt_list[i], upscale)
            self.lrhsi_patch_list.extend(crop_patches(lrhsi, stride // upscale, patch_size // upscale))
        logger.info(f"Patch extraction completed: GT={len(self.gt_patch_list):5d}, LR-HSI={len(self.lrhsi_patch_list):5d}, HR-MSI={len(self.hrmsi_patch_list):5d}")

    def __getitem__(self, index):
        
        gt_patch = self.gt_patch_list[index]
        hrmsi_patch = self.hrmsi_patch_list[index]
        lrhsi_patch = self.lrhsi_patch_list[index]
        # 展示训练数据
        # print(f"gt_patch.shape:{gt_patch.shape}, lrhsi_patch.shape:{lrhsi_patch.shape}, hrmsi_patch.shape:{hrmsi_patch.shape}")
        # show_three_patches(gt_patch, hrmsi_patch, lrhsi_patch, save_path=f"./test.png")
        gt_patch, hrmsi_patch, lrhsi_patch = random_augment(gt_patch, hrmsi_patch, lrhsi_patch) # same augmentation for all three patches
        return gt_patch, lrhsi_patch, hrmsi_patch

    def __len__(self):
        return len(self.gt_patch_list)


# ========= 构建 Dataset 和 DataLoader =========
def build_dataset(args, logger):
    logger.info("=" * 80)
    logger.info("Starting dataset creation ...")
    start_time = time.time()
    dataset_config = {
        "PaviaU":  ("./datasets/PaviaU_dataset.npz", 103, 4),
        "PaviaC":  ("./datasets/form_data/PaviaC_dataset.npz", 103, 4),
        "Chikusei":("./datasets/Chikusei_dataset.npz", 128, 3),
        "Houston": ("./datasets/form_data/Houston_dataset.npz", 144, 4),
        "CAVE":    ("./datasets/CAVE_dataset.npz", 31, 3),
        "BayArea": ("./datasets/form_data/BayArea_dataset.npz", 224, 3),
        "Santa":   ("./datasets/form_data/BayArea_dataset.npz", 224, 3),
    }

    if args.dataset not in dataset_config:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    args.dataset_path, args.hsi_bands, args.msi_bands = dataset_config[args.dataset]

    if args.upscale in [2, 4]:
        args.patch_size = 128
    elif args.upscale == 8:
        args.patch_size = 128
    else:
        raise ValueError(f"Unsupported upscale factor: {args.upscale}")

    train_dataset = NPZDataset(args.dataset_path, is_train=True, upscale=args.upscale, logger=logger)
    test_dataset  = NPZDataset(args.dataset_path, is_train=False, upscale=args.upscale, logger=logger)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size * 6)

    logger.info(f"[Dataset loading completed]")
    logger.info(f"Training samples: {len(train_dataset):5d}")
    logger.info(f"Testing samples: {len(test_dataset):5d}")
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")
    return train_dataloader, test_dataloader
