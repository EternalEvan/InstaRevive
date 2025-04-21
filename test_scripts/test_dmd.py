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
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.data.datasets import get_chunks
from diffusion.data.datasets.utils import *

from dataset.codeformer import CodeformerDataset,CodeformerDatasetLQ

from diffusers import Transformer2DModel, DDPMScheduler, StableDiffusionPipeline

from torchvision.utils import save_image 

import einops

from utils.common import instantiate_from_config,load_state_dict,frozen_module

from omegaconf import OmegaConf

from scripts.DMD.transformer_train.generate import generate_sample_1step, forward_model

def save_batch(images,imgname_batch, save_path, watch_step=False):
        if watch_step:
            for list_idx, img_list in enumerate(images):
                for img_idx, img in enumerate(img_list):
                    imgname = str(list_idx)+"_"+imgname_batch[img_idx]
                
                    save_img = os.path.join(save_path,imgname)
                    save_image(img,save_img)
        else:   
            for img_idx, img in enumerate(images):
                imgname = imgname_batch[img_idx]
                if imgname[-3:] == 'jpg':
                    imgname = imgname[:-3] + 'png'
                save_img = os.path.join(save_path,imgname)
                save_image(img,save_img)
                
def get_input(batch, vae ,preprocess_model, bs=None,return_xc = False,device = 'cpu', *args, **kwargs):
    #batch = batch[0]
    x = batch['jpg'].to(device)
    x = einops.rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).to(torch.float32)

    control = batch["hint"].to(device)
 
    if bs is not None:
        control = control[:bs]

    control = einops.rearrange(control, 'b h w c -> b c h w')

    control = control.to(memory_format=torch.contiguous_format).to(torch.float32)
    lq = control *2 -1
    # apply preprocess model

    control = preprocess_model(control)
    
    
    control_norm = control * 2 - 1
    #save_image(control,'./control1.png')
    #pdb.set_trace()
    # apply condition encoder
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            posterior = vae.encode(x).latent_dist
            z = posterior.mode().to(torch.float32)
            
            posterior_c = vae.encode(control_norm).latent_dist
            c_latent = posterior_c.mode().to(torch.float32)
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
    
train_dataset = CodeformerDatasetLQ(
        lq_list='/data1/zyx/WebPhoto-Test/web-test.list', #/data3/zyx/FFHQ512/celeba-test.list',
        out_size=512,
        crop_type="center",
        use_hflip=True,
        blur_kernel_size=41,
        kernel_list=kernel_list,
        kernel_prob=kernel_prob,
        blur_sigma=blur_sigma,
        downsample_range=downsample_range,
        noise_range=noise_range,
        jpeg_range=jpeg_range
        )

device = 'cuda'
vae = AutoencoderKL.from_pretrained('/data3/zyx/pixart/vae_for_dmd').to(device)

# vae_half = vae.to(torch.float16)
model = Transformer2DModel.from_pretrained('/data1/whl/pixart/PixArt-Alpha-DMD-XL-2-512x512', subfolder='transformer')
#Transformer2DModel.from_pretrained('/data3/zyx/pixart/PixART-XL-512')
noise_scheduler = DDPMScheduler.from_pretrained('/data1/whl/pixart/PixArt-Alpha-DMD-XL-2-512x512', subfolder="scheduler")

state_dict = torch.load('/data3/zyx/pixart/dmd-unet-more_dm_regression_1distep_constant1e-06sgmul1.0warmup0_cfg3.0_999ts_acc2_maxgrad10.0_mixedprecisionfp16_bs4_one_step_maxt400/checkpoint-5000/pytorch_model.bin')
    #'/data1/zyx/pixart/dmd-unet-more_dm_regression_1distep_constant1e-06sgmul1.0warmup0_cfg3.0_999ts_acc2_maxgrad10.0_mixedprecisionfp16_bs4_one_step_maxt400/checkpoint-15000/pytorch_model.bin')
                       # '/data3/zyx/pixart/dmd-unet-more_dm_regression_1distep_constant1e-06sgmul1.0warmup0_cfg3.0_999ts_acc2_maxgrad10.0_mixedprecisionfp16_bs4_one_step_maxt400/checkpoint-5000/pytorch_model.bin')
        #'/data3/zyx/pixart/dmd-unet_dm_regression_1distep_constant1e-06sgmul1.0warmup0_cfg3.0_999ts_acc2_maxgrad10.0_mixedprecisionfp16_bs2_one_step_maxt400/checkpoint-25000/pytorch_model.bin',map_location='cpu')
model.load_state_dict(state_dict)
 
model.to(device)


train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=1,
    )


y_null_all = torch.load("/home/zyx/PixArt-sigma/output/tmp/portrait photo of human face, photograph, film, professional, 4k, highly detailed_300token.pth", map_location="cpu")
   
y_null = y_null_all['caption_embeds'].to(device)
y_null_mask = y_null_all['emb_mask'].to(device)
# negative_prompt_embeds = negative_prompt_embeds_dict['uncond_prompt_embeds']
# negative_prompt_attention_masks = negative_prompt_embeds_dict['uncond_prompt_embeds_mask']

output_dir = '/data1/zyx/pixart/output/web-dmd'
os.makedirs(output_dir,exist_ok=True)

preprocess_config = '/home/zyx/PixArt-sigma/configs/swinir.yaml'

preprocess_model = instantiate_from_config(OmegaConf.load(preprocess_config))
load_state_dict(preprocess_model, torch.load('/data3/zyx/visionhub/face_swinir_v1.ckpt', map_location="cpu"), strict=True)
frozen_module(preprocess_model)
preprocess_model.to(device)


for step, batch in enumerate(train_dataloader):
    print(step)
    z, cond = get_input(batch,vae,preprocess_model,device=device,return_xc=True)
           
    bs = z.shape[0]
    
    y = y_null.unsqueeze(0).repeat((bs,1,1,1)).to(device).to(torch.float32)  # 4 x 1 x 120 x 4096 # T5 extracted feature of caption, 120 token, 4096
    y_mask = y_null_mask.unsqueeze(0).unsqueeze(0).repeat((bs,1,1,1)).to(device).to(torch.float32)  # 4 x 1 x 1 x 120 # caption indicate whether valid
    


                    # add noise to the one-step result

    with torch.no_grad():
                
        bs = z.shape[0]

        y = y_null.unsqueeze(0).repeat((bs,1,1,1)).to(torch.float32)  # 4 x 1 x 120 x 4096 # T5 extracted feature of caption, 120 token, 4096
        y_mask = y_null_mask.unsqueeze(0).unsqueeze(0).repeat((bs,1,1,1)).to(torch.float32)  # 4 x 1 x 1 x 120 # caption indicate whether valid

        print(y.shape)
        print(y_mask.shape)
        init_noise = cond['c_latent'][0]*vae.config.scaling_factor
        
        latents = generate_sample_1step(model, noise_scheduler, init_noise, 400, y, y_mask)
        _image = latents.detach() / vae.config.scaling_factor
        
        _image = _image
        
        images = vae.decode(_image).sample/2+0.5
        
        imgname = batch['imgname']
        save_batch(images,imgname,output_dir)
        # print(noisy_latents)
        # print(timesteps)
        

