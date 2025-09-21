## PIF-Net: Ill-Posed Prior Guided Multispectral and Hyperspectral Image Fusion via Invertible Mamba and Fusion-Aware LoRA
## Abstract  
Multispectral and hyperspectral image fusion (MHIF) aims to generate high-quality images that combine rich spectral information with fine spatial details. However, this task is fundamentally ill-posed due to the inherent trade-off between spectral and spatial resolution and the limited availability of aligned observations. Existing methods often struggle with challenges arising from data misalignment.
To address this, we propose **PIF-Net**, a novel fusion framework that explicitly incorporates ill-posed priors to effectively merge multispectral and hyperspectral images. Our approach leverages an **invertible Mamba architecture** to ensure information consistency during feature transformation and fusion, maintaining stable gradient flow and enabling process reversibility while balancing global spectral modeling with computational efficiency. Additionally, we introduce a **Fusion-Aware Low-Rank Adaptation (LoRA) module** that dynamically calibrates spectral and spatial features, keeping the model lightweight and adaptable.
Extensive experiments on benchmark datasets demonstrate that **PIF-Net** achieves significantly superior image restoration performance compared to current state-of-the-art methods, while maintaining efficiency.

---

## 🔧 Installation

This codebase was tested with the following environment configuration. Other versions may also work but are not guaranteed.

- **OS:** Ubuntu 20.04  
- **CUDA:** 11.7  
- **Python:** 3.9  
- **PyTorch:** 2.0.1 + cu117  

> **Note:** If you use a newer CUDA version (e.g., 12.x), please refer to the official GitHub repositories of [`causal_conv_1d`](https://github.com/...) and [`mamba_ssm`](https://github.com/...) to find compatible versions.

---

### Installing Mamba-related libraries

To utilize the selective scan with efficient hardware design, please install the `mamba_ssm` library as follows:

```bash
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```
## 📂 Dataset

We will release pretrained model weights and benchmark datasets soon for reproducibility and further research.

Currently, the following datasets are supported:

- Pavia University  
- Chikusei  
- Houston  

You can download the datasets here: [**Dataset Download Link (Coming Soon)**]()

---

## 🛠️ Usage

1. Place the dataset in the `dataset` directory.  
2. Run the following command to train or test the model:

```bash
python main.py --model PIFNet --dataset Pavia
```
Replace `Pavia` with the desired dataset name (`Chikusei`, `Houston`, etc.).
