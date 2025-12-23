import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel  # <--- BẮT BUỘC PHẢI CÓ DÒNG NÀY
from tqdm.auto import tqdm