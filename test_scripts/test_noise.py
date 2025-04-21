import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from scripts.DMD.transformer_train.utils import save_image
from diffusers.models import AutoencoderKL

import numpy as np

from dataset.codeformer import CodeformerDataset

from diffusers import Transformer2DModel, DDPMScheduler, StableDiffusionPipeline

from torchvision.utils import save_image as save_image_util

import einops

from utils.common import instantiate_from_config,load_state_dict,frozen_module

from omegaconf import OmegaConf

from scripts.DMD.transformer_train.generate import generate_sample_1step, forward_model
from PIL import Image

def get_input(batch, vae ,preprocess_model, bs=None,return_xc = False,device = 'cpu', *args, **kwargs):
    #batch = batch[0]
    x = batch['jpg'].to(device)
    x = einops.rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()

    control = batch["hint"].to(device)
    if bs is not None:
        control = control[:bs]
    control = control
    control = einops.rearrange(control, 'b h w c -> b c h w')
    control = control.to(memory_format=torch.contiguous_format).float()
    lq = control *2 -1
    # apply preprocess model

    # control = preprocess_model(control)
    
    
    control_norm = control * 2 - 1
    #save_image(control,'./control1.png')
    #pdb.set_trace()
    # apply condition encoder
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            posterior = vae.encode(x).latent_dist
            z = posterior.mode()
            
            posterior_c = vae.encode(control_norm).latent_dist
            c_latent = posterior_c.mode()
    if return_xc:
        return z, dict(c_latent=[c_latent], lq=[lq], c_concat=[control_norm],x=[x])
    else:
        return z, dict(c_latent=[c_latent], lq=[lq], c_concat=[control_norm])
    
kernel_list = ['iso', 'aniso']
kernel_prob = [0.5, 0.5]
blur_sigma = [0.1, 10]
downsample_range = [4, 8]
noise_range = [0, 20]
jpeg_range = [60, 100]
    


device = 'cuda:4'
vae = AutoencoderKL.from_pretrained('/data3/zyx/pixart/vae_for_dmd').to(device)

vae_half = vae.to(torch.float16)
model_real = Transformer2DModel.from_pretrained('/data3/zyx/pixart/PixART-XL-512')
# model_real = model_real.unet
#Transformer2DModel.from_pretrained('/data3/zyx/pixart/PixART-XL-512')

model_real.requires_grad_(False)
 
model_real.to(device)

load_from = "/data3/zyx/pixart/models--PixArt-alpha--PixArt-Alpha-DMD-XL-2-512x512/snapshots/57df7bb32daad69bc1d6c825a275935113802169"





negative_prompt_embeds_dict = torch.load(
        f'/home/zyx/PixArt-sigma/output/pretrained_models/null_embed_diffusers_300token.pth', map_location='cpu')
negative_prompt_embeds = negative_prompt_embeds_dict['uncond_prompt_embeds']
negative_prompt_attention_masks = negative_prompt_embeds_dict['uncond_prompt_embeds_mask']

noise_scheduler = DDPMScheduler.from_pretrained(load_from, subfolder="scheduler")

preprocess_config = '/home/zyx/PixArt-sigma/configs/swinir.yaml'

preprocess_model = instantiate_from_config(OmegaConf.load(preprocess_config))
load_state_dict(preprocess_model, torch.load('/data3/zyx/visionhub/face_swinir_v1.ckpt', map_location="cpu"), strict=True)
frozen_module(preprocess_model)
preprocess_model.to(device)

y_null_all = torch.load("/home/zyx/PixArt-sigma/output/tmp/face_300token.pth", map_location="cpu")
   

y_null = y_null_all['caption_embeds'].to(device)
y_null_mask = y_null_all['emb_mask'].to(device)

y_null = negative_prompt_embeds.to(device)
y_null_mask = negative_prompt_attention_masks.to(device)

gt_path = '/home/zyx/PixArt-sigma/paper_input/x0_bar/butterfly.png'
pil_img = Image.open(gt_path).convert("RGB")
if min(*pil_img.size)!= 512:
    pil_img = pil_img.resize((512,512),resample=Image.BOX)
pil_img_gt = np.array(pil_img)
img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
source = img_gt[..., ::-1].astype(np.float32)

batch = {}
batch['hint'] = torch.Tensor(source).unsqueeze(0).to(device)
batch['jpg'] = torch.Tensor(target).unsqueeze(0).to(device)

z, cond = get_input(batch,vae,preprocess_model,device=device,return_xc=True)
        
bs = z.shape[0]

y = y_null.unsqueeze(0).repeat((bs,1,1,1)).to(device).to(torch.float32)  # 4 x 1 x 120 x 4096 # T5 extracted feature of caption, 120 token, 4096
y_mask = y_null_mask.unsqueeze(0).unsqueeze(0).repeat((bs,1,1,1)).to(device).to(torch.float32)  # 4 x 1 x 1 x 120 # caption indicate whether valid

latents = z*vae.config.scaling_factor

noise = torch.randn_like(latents).to(device)

bsz = latents.shape[0]

timesteps = torch.ones((bsz,), device=latents.device)*240

timesteps = timesteps.long()
                # add noise to the one-step result
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

with torch.no_grad():
    noisy_latents_cat = torch.cat([noisy_latents, noisy_latents], 0).to(torch.float32)
    timesteps_cat = torch.cat([timesteps, timesteps], 0)

    uncond_encoder_hidden_states = negative_prompt_embeds.repeat(
        bsz, 1, 1, 1).to(device).to(torch.float32)
    uncond_attention_mask = negative_prompt_attention_masks.repeat(
        bsz, 1, 1, 1).to(device).to(torch.float32)

    encoder_cat = torch.cat([uncond_encoder_hidden_states, y], dim=0)
    mask_cat = torch.cat([uncond_attention_mask, y_mask], dim=0)

    # Real model forward
    model_real_output = forward_model(model_real,
                                        noisy_latents_cat,
                                        timesteps_cat,
                                        encoder_cat,
                                        mask_cat)
    # print(noisy_latents)
    # print(timesteps)
    score_real_uncond, score_real_cond = (-model_real_output).chunk(2)
    score_real = score_real_uncond + 7.5 * (score_real_cond - score_real_uncond)
    
    alpha_prod_t = noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)[timesteps]
    beta_prod_t = 1.0 - alpha_prod_t
    
    pred_latents = (
                            (
                                    noisy_latents + beta_prod_t.view(-1, 1, 1, 1) ** 0.5 * score_real
                            ) / alpha_prod_t.view(-1, 1, 1, 1) ** 0.5
                    )
    
    pred_real = vae_half.decode(pred_latents.to(torch.float16)/vae.config.scaling_factor).sample.detach()
    noisy_real = vae_half.decode(noisy_latents.to(torch.float16)/vae.config.scaling_factor).sample.detach()

    
    save_image(noisy_real, '/home/zyx/PixArt-sigma/paper_input/x0_bar/noisy.png')
    save_image(pred_real, '/home/zyx/PixArt-sigma/paper_input/x0_bar/pred-butterfly.png')
    save_image(cond['x'][0],'/home/zyx/PixArt-sigma/paper_input/x0_bar/gt.png')
    assert False

