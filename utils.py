import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.init as init
import numpy as np
import logging
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import sys
import os
import shutil
import time
import shutil
import copy

from PIL import ImageDraw
from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count_table
sys.path.append('../datasets')
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError 


class Colors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def log_step(msg): 
    print(f"{Colors.OKBLUE}==> {msg}{Colors.RESET}")

def log_args_summary(args, logger):
    printable_keys = [
        "dataset", "upscale", "model", "batch_size", "epochs", "lr", 
        "loss", "FLOPs", "model_size", "inference_time", "log_dir", "pid","device"
    ]
    args_dict = vars(args)
    for key in printable_keys:
        if key in args_dict:
            val = args_dict[key]
            logger.info(f"{key:<15}: {val}")
class Metrics:
    """
    metrics
    """
    def __init__(self, logger):
        pass
    @staticmethod
    @torch.no_grad()
    def cal_psnr(pre_hsi, gt):
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        mse = torch.mean((pre_hsi - gt) ** 2, dim=[1, 2, 3])  # 计
        max_values, _ = torch.max(gt, dim=2, keepdim=True)  
        max_values, _ = torch.max(max_values, dim=3, keepdim=True) 
        max_values, _ = torch.max(max_values, dim=1, keepdim=True) 
        max_pixel = max_values.squeeze()  
        if torch.any(mse == 0): 
            return torch.ones(pre_hsi.shape[0], device=pre_hsi.device) * float('inf')  
        psnr = 20 * torch.log10(max_pixel) - 10 * torch.log10(mse)
        return psnr.mean()
            
    @staticmethod
    @torch.no_grad()
    def cal_ssim(pre_hsi, gt):
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(gt.device)
        return ssim(pre_hsi, gt)
    
    @staticmethod
    @torch.no_grad()
    def cal_sam(pre_hsi: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        计算 Spectral Angle Mapper (SAM)，单位为度。
        支持输入为 3D 或 4D (B, C, H, W) 张量。

        Returns:
            平均 SAM (float tensor)
        """
        assert pre_hsi.shape == gt.shape, "Shape mismatch between prediction and ground truth"

        if pre_hsi.ndim == 3:  # (C, H, W)
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        elif pre_hsi.ndim != 4:
            raise ValueError("Input must be 3D or 4D tensor")

        # (B, C, H, W) -> (B, H, W, C)
        pre_hsi = pre_hsi.permute(0, 2, 3, 1)
        gt = gt.permute(0, 2, 3, 1)

        dot_product = torch.sum(pre_hsi * gt, dim=-1)
        norm_pred = torch.norm(pre_hsi, dim=-1)
        norm_gt = torch.norm(gt, dim=-1)

        denominator = norm_pred * norm_gt + 1e-10
        cos_theta = dot_product / denominator
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        sam_map = torch.acos(cos_theta)  # in radians
        sam_mean = torch.mean(sam_map) * 180 / torch.pi
        return sam_mean


    @staticmethod
    @torch.no_grad()
    def cal_ergas(pre_hsi, gt, up_scale=4):
        assert gt.shape == pre_hsi.shape
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        rmse_bands = torch.sqrt(torch.mean((gt - pre_hsi) ** 2, dim=(2, 3)))  # (B, C)
        mean_bands = torch.mean(gt, dim=(2, 3))  # (B, C)
        ergas = 100 / up_scale * torch.sqrt(torch.mean((rmse_bands / mean_bands) ** 2, dim=1))  # (B,)
        return ergas.mean().item()
    

def beijing_time():
    """获取北京时间
    Returns:
        str: 北京时间字符串
    """
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    beijing_now = utc_now.astimezone(SHA_TZ)
    fmt = '%Y-%m-%d,%H:%M:%S'
    now_fmt=beijing_now.strftime(fmt)
    return  now_fmt

def set_seed(seed=9999):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_logger(args):
    """获取日志的打印对象
    """
    logger_dir = args.log_dir
    model_name = args.model.split('_')[0]
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    if args.log == 1:
        log_file = f"{logger_dir}/out.log" 
        if not os.path.exists(log_file):
            os.mknod(log_file)  
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        source_file = f'./models/{model_name}'
        target_file = f"{logger_dir}/{model_name}"
        shutil.copytree(source_file, target_file)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO) 
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger

def test_speed_thop(args, model):
    """测试模型的推理速度，FLOPs，参数量，参数表
    """
    test_model = copy.deepcopy(model)
    test_model.eval()
    device = torch.device(args.device)
    lr_hsi = torch.rand(1, args.hsi_bands, args.patch_size // args.upscale,  args.patch_size // args.upscale).to(device)
    hr_msi = torch.rand(1, args.msi_bands, args.patch_size,  args.patch_size).to(device)
    flops, params = profile(test_model.to(device), inputs=(lr_hsi,hr_msi,),verbose=False)
    start_time = time.time()
    with torch.no_grad():
        test_model.eval()
        test_model(lr_hsi,hr_msi)
    inference_time = time.time() - start_time
    param_table = parameter_count_table(test_model)
    del test_model
    return inference_time,flops / 1000000000. ,params / 1000000.0, param_table


# ======= 用 torchvision 替换 PIL 读取 =======
from torchvision.io import read_image

def get_cave_hsi_image(image_path):
    """
    使用 torchvision 读取 CAVE 数据集的 HSI 图片，按通道堆叠
    输出: (C, H, W) 的 numpy 数组，归一化到 [0,1]
    """
    hsi = []
    for file in sorted(os.listdir(image_path)):
        if file.endswith('.png'):
            img_path = os.path.join(image_path, file)
            img_tensor = read_image(img_path).float() / 65535.0  # 读取为 float32 并归一化
            hsi.append(img_tensor)
    hsi = torch.stack(hsi, dim=0)  # (num_bands, C, H, W) - 但 CAVE 通道是单通道，通常是灰度
    hsi = hsi.squeeze(1).numpy()  # 去掉通道维度 -> (num_bands, H, W)
    return hsi


def plot_tensor_image(image_data):
    if len(image_data.shape)==2:
        image_data = image_data.unsqueeze(0)
    return image_data.permute(1,2,0).detach().cpu().numpy()


def down_sample(hsi, down_scale=4):
    hsi = hsi.unsqueeze(0)
    down_hsi = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=0.5)(hsi)
    down_hsi = F.interpolate(down_hsi,scale_factor=1/down_scale)
    down_hsi = down_hsi.squeeze(0)
    return down_hsi

def sam_errormap(reference_image, reconstructed_image):
    height, width, num_bands = reference_image.shape
    errormap = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            ref_spectrum = reference_image[i, j, :]
            rec_spectrum = reconstructed_image[i, j, :]
            numerator = np.dot(ref_spectrum, rec_spectrum)
            denominator = np.linalg.norm(ref_spectrum) * np.linalg.norm(rec_spectrum)
            if denominator == 0:
                errormap[i, j] = 0
            else:
                cos_theta = numerator / denominator
                cos_theta = np.clip(cos_theta, -1, 1)
                errormap[i, j] = np.arccos(cos_theta)
    errormap = errormap * (180 / np.pi)
    return errormap


def restrurct_hsi(model, hsi, lr_hsi, msi=None, upscale=4, patch_size=64, stride=64, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    c,h,w = lr_hsi.shape
    h,w = h*upscale,w*upscale
    scale = upscale
    p = patch_size
    stride = stride
    hsi = hsi.to(device)
    
    if msi is not None:
        msi = msi.to(device)
    lr_hsi = lr_hsi.to(device)
    
    pre_hsi = torch.zeros_like(hsi).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        i,j = 0,0
        while True:
            if i+p > h:
                i = h-p
            while True:
                if j+p > w:
                    j = w-p
                if msi == None:
                    lr_hsi_patch = lr_hsi[:,i//scale:i//scale+p//scale,j//scale:j//scale+p//scale]
                    pre_hsi[:,i:i+p,j:j+p] = model(lr_hsi_patch.unsqueeze(0)).squeeze(0)
                else:
                    lr_hsi_patch = lr_hsi[:,i//scale:i//scale+p//scale,j//scale:j//scale+p//scale]
                    msi_patch = msi[:,i:i+p,j:j+p]
                    pre_hsi[:,i:i+p,j:j+p] = model(lr_hsi_patch.unsqueeze(0),msi_patch.unsqueeze(0)).squeeze(0)
                if j+p == w:
                    j = 0
                    is_last_row = True
                    break
                else:
                    j += stride
            if i + p == h and is_last_row:
                break
            i += stride
            is_last_row = False
    return pre_hsi

def img_resize(img, size):
    img = img.transpose(1, 2, 0)
    img = mmcv.imresize(img, size)
    img = img.transpose(2, 0, 1)
    return img

def crop_patches(image, stride, patch_size):
    """
    从3维图像 (C, H, W) 中按照步长stride切patch，patch大小为patch_size。
    边缘会补充最后一个patch确保覆盖全图。
    返回一个numpy数组，形状为 (num_patches, C, patch_size, patch_size)
    """
    C, H, W = image.shape
    patches = []
    
    # 计算切片起点（包含边缘patch）
    h_steps = list(range(0, H - patch_size + 1, stride))
    w_steps = list(range(0, W - patch_size + 1, stride))
    
    if h_steps[-1] != H - patch_size:
        h_steps.append(H - patch_size)
    if w_steps[-1] != W - patch_size:
        w_steps.append(W - patch_size)
    
    for i in h_steps:
        for j in w_steps:
            patch = image[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    
    return patches



def draw_rec(image,x,y,h,w,scale=2,loc='left',line_width=4):
    image = torch.from_numpy(image).permute(2,0,1).float()
    tar_region = image[:,x:x+h,y:y+w]
    _,H,W = image.shape
    scale_tar_region = F.interpolate(tar_region.unsqueeze(0),scale_factor=scale).squeeze(0)
    c,scale_h,scale_w = scale_tar_region.shape
    if loc == 'left':
        image[:,-scale_h:,:scale_w] = scale_tar_region
    elif loc == 'right':
        image[:,-scale_h:,-scale_w:] = scale_tar_region
    image = torchvision.transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle([y, x, y+w, x+h], outline=(255, 0, 0), width=line_width)
    if loc == 'left':
        draw.rectangle([0, H-scale_h-1, scale_w, H-1],
               outline=(255, 0, 0),
               width=line_width)
    elif loc == 'right':
        draw.rectangle([W-scale_w-1, H-scale_h-1, W-1, H-1],
               outline=(255, 0, 0),
               width=line_width)
    image = torchvision.transforms.ToTensor()(image)
    image = image.permute(1,2,0).numpy()
    return image
