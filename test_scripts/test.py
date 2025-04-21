from typing import List, Tuple, Optional
import os
import math
from argparse import ArgumentParser, Namespace
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from tqdm import tqdm

from utils.image import auto_resize, pad

from utils.image import (
    wavelet_reconstruction, adaptive_instance_normalization
)

from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts

from scripts.DMD.transformer_train.generate import generate_sample_1step, forward_model

from utils.image import center_crop_arr

from diffusers import Transformer2DModel, DDPMScheduler, StableDiffusionPipeline

from diffusers.models import AutoencoderKL


model = Transformer2DModel.from_pretrained('PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512', subfolder='transformer')