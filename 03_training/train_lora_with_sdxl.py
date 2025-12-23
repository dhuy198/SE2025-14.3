import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

print("***** Bắt đầu huấn luyện SDXL LoRA *****")
