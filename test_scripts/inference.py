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

noise_scheduler = DDPMScheduler.from_pretrained('PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512', subfolder="scheduler")


@torch.no_grad()
def _sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> Tuple[int, int, int, int]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)
    
    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)
    
    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords

@torch.no_grad()
def process(
    model,
    control_imgs: List[np.ndarray],
    strength: float,
    color_fix_type: str,
    disable_preprocess_model: bool,
    tiled: bool,
    tile_size: int,
    tile_stride: int,
    preprocess_model = None,
    vae = None,
    y = None,
    y_mask = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply DiffBIR model on a list of low-quality images.
    
    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
        cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        tile_size (int): Size of patch.
        tile_stride (int): Stride of sliding patch.
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]). 
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same 
            as low-quality inputs.
    """
    n_samples = len(control_imgs)
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    
    
    if not disable_preprocess_model:
        control = preprocess_model(control)
    
    img_buffer = torch.zeros_like(control).to(control)
    
    height, width = control.size(-2), control.size(-1)
    h,w = height//8, width//8
    
    control_norm = control * 2 - 1
    
    posterior_c = vae.encode(control_norm).latent_dist
    c_latent = posterior_c.mode().to(torch.float32)
    
    init_noise = c_latent*vae.config.scaling_factor
    
    if not tiled:
        
        # noise_buffer = torch.zeros_like(init_noise).to(init_noise)
        latents = generate_sample_1step(model, noise_scheduler, init_noise, 400, y, y_mask)

        latents = latents.detach() / vae.config.scaling_factor
        img_buffer = vae.decode(latents).sample/2+0.5
        
    else:
        tiles_iterator = tqdm(_sliding_windows(h, w, tile_size // 8, tile_stride // 8))
        
        shape = (n_samples, 4, height // 8, width // 8)
        
        count = torch.zeros(shape, dtype=torch.long).to(init_noise)
        
        noise_buffer = torch.zeros_like(init_noise).to(init_noise)
        
        for hi, hi_end, wi, wi_end in tiles_iterator:
            tiles_iterator.set_description(f"Process tile with location ({hi} {hi_end}) ({wi} {wi_end})")
            tile_cond = init_noise[:, :, hi:hi_end, wi:wi_end]
            tile_latents = generate_sample_1step(model, noise_scheduler, tile_cond, 400, y, y_mask)
            
            noise_buffer[:, :, hi:hi_end, wi:wi_end] += tile_latents
            count[:, :, hi:hi_end, wi:wi_end] += 1
        
        noise_buffer.div_(count)
        count = torch.zeros_like(control, dtype=torch.long)
        
        for hi, hi_end, wi, wi_end in _sliding_windows(h, w, tile_size // 8, tile_stride // 8):
            tile_latents = noise_buffer[:, :, hi:hi_end, wi:wi_end]
            tile_latents = tile_latents.detach() / vae.config.scaling_factor
            tile_img_pixel = vae.decode(tile_latents).sample/2+0.5
            
            tile_cond_img = control[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8]
                # apply color correction (borrowed from StableSR)
            if color_fix_type == "adain":
                tile_img_pixel = adaptive_instance_normalization(tile_img_pixel, tile_cond_img)
            elif color_fix_type == "wavelet":
                tile_img_pixel = wavelet_reconstruction(tile_img_pixel, tile_cond_img)
                    
            img_buffer[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8] += tile_img_pixel
            count[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8] += 1
        img_buffer.div_(count)
        
        
    
    samples = img_buffer
    
    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    preds = [x_samples[i] for i in range(n_samples)]
    stage1_preds = [control[i] for i in range(n_samples)]
    
    return preds, stage1_preds


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt", required=True, type=str, help="full checkpoint path",default='./weights/InstaRevive_v1.ckpt')
    
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sr_scale", type=float, default=1)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--disable_preprocess_model", action="store_true")
    
    # patch-based sampling
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=448)
    
    # latent image guidance
    parser.add_argument("--use_guidance", action="store_true")
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_t_start", type=int, default=1001)
    parser.add_argument("--g_t_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=5)
    
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    
    parser.add_argument("--use_prompt", action="store_true")
    
    parser.add_argument("--use_center_crop", action="store_true")
    
    return parser.parse_args()

def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device

def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    
    args.device = check_device(args.device)
    
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
    vae = vae.to(torch.float32).to(args.device)
    model = Transformer2DModel.from_pretrained('PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512', subfolder='transformer')
    state_dict = torch.load(
        './weights/InstaRevive_v1.ckpt',map_location='cpu')
        
    model.load_state_dict(state_dict)
    # reload preprocess model if specified
    
    preprocess_config = './configs/swinir.yaml'

    preprocess_model = instantiate_from_config(OmegaConf.load(preprocess_config))
    load_state_dict(preprocess_model, torch.load('./weights/general_swinir_v1.ckpt', map_location="cpu"), strict=True)

    model.to(args.device)
    
    preprocess_model.to(args.device)
    
    assert os.path.isdir(args.input)
    
    y_null_all = torch.load("./output/tmp/real-world image, realistic, high quality, photograph, film, professional, 4k, highly detailed_300token.pth", map_location="cpu")

    y_null = y_null_all['caption_embeds'].to(args.device)
    y_null_mask = y_null_all['emb_mask'].to(args.device)
    # assert False
    for file_path in list_image_files(args.input, follow_links=True):
        
        lq = Image.open(file_path).convert("RGB")
        
        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size),
                Image.BICUBIC
            )
            
        
   
        y = y_null.unsqueeze(0).repeat((1,1,1,1)).to(torch.float32)  # 4 x 1 x 120 x 4096 # T5 extracted feature of caption, 120 token, 4096
        y_mask = y_null_mask.unsqueeze(0).unsqueeze(0).repeat((1,1,1,1)).to(torch.float32)  # 4 x 1 x 1 x 120 # caption indicate whether valid

        y = y.squeeze(0)
        y_mask = y_mask.squeeze(0)
        # assert False
        
        
        if not args.tiled:
            if args.use_center_crop:
                lq_resized = center_crop_arr(lq,512) #auto_resize(lq, 512)
                x = np.array(lq_resized)
            else:
                lq_resized = auto_resize(lq, 512)
                x = pad(np.array(lq_resized), scale=64)
        else:
            lq_resized = auto_resize(lq, args.tile_size)
            
            x = pad(np.array(lq_resized), scale=64)
        
       
        for i in range(args.repeat_times):
            save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
            parent_path, stem, _ = get_file_name_parts(save_path)
            save_path = os.path.join(parent_path, f"{stem}_{i}.png")
            
            # if os.path.exists(save_path):
            #     if args.skip_if_exist:
            #         print(f"skip {save_path}")
            #         continue
            #     else:
            #         raise RuntimeError(f"{save_path} already exist")
            os.makedirs(parent_path, exist_ok=True)
            
            # initialize latent image guidance
            
            
            cond_fn = None
            
            preds, stage1_preds = process(
                model, [x], 
                strength=1,
                color_fix_type=args.color_fix_type,
                disable_preprocess_model=args.disable_preprocess_model,
                tiled=args.tiled, tile_size=args.tile_size, tile_stride=args.tile_stride,
                vae=vae,
                preprocess_model=preprocess_model,
                y = y,
                y_mask=y_mask,
            )
            pred, stage1_pred = preds[0], stage1_preds[0]
            
            # remove padding
            if not args.use_center_crop:
                pred = pred[:lq_resized.height, :lq_resized.width, :]
                stage1_pred = stage1_pred[:lq_resized.height, :lq_resized.width, :]
            
            if args.show_lq:
                if not args.use_center_crop:
                    pred = np.array(Image.fromarray(pred).resize(lq.size, Image.LANCZOS))
                    stage1_pred = np.array(Image.fromarray(stage1_pred).resize(lq.size, Image.LANCZOS))
                    lq = np.array(lq)
                else:
                    lq = x
                images = [lq, pred] if args.disable_preprocess_model else [lq, stage1_pred, pred]
                #Image.fromarray(lq).save(save_path)
                #assert False

                Image.fromarray(np.concatenate(images, axis=1)).save(save_path)
            else:
                if not args.use_center_crop:
                    Image.fromarray(pred).resize(lq.size, Image.LANCZOS).save(save_path)
                else: 
                    Image.fromarray(pred).save(save_path)
            print(f"save to {save_path}")

if __name__ == "__main__":
    main()
