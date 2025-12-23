import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

# Thư viện từ Hugging Face (vẫn cần dùng cho các thành phần của Stable Diffusion)
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
