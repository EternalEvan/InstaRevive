import argparse
import datetime
import os
import sys
import time
import types
import warnings
from pathlib import Path
import einops

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from mmcv.runner import LogBuffer
from torch.utils.data import RandomSampler
from PIL import Image
import numpy as np
from os.path import join

from transformers import T5EncoderModel, T5Tokenizer

from diffusion import IDDPM,DPMS
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.model.nets import PixArtMS
from diffusion.model.nets.pixart_controlnet import ControlPixArtHalf, ControlPixArtMSHalf

from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

from diffusers.models import AutoencoderKL

from utils.common import frozen_module
from utils.common import instantiate_from_config,load_state_dict
from dataset.codeformer import CodeformerDataset, CodeformerDatasetLQ
from omegaconf import OmegaConf

from tools.download import find_model
from torchvision.utils import save_image
warnings.filterwarnings("ignore")  # ignore warning

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

def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


def get_input(batch, vae ,preprocess_model, bs=None,return_xc = False, *args, **kwargs):
    #batch = batch[0]
    x = batch['jpg']
    x = einops.rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()

    control = batch["hint"]
    if bs is not None:
        control = control[:bs]
    control = control
    control = einops.rearrange(control, 'b h w c -> b c h w')
    control = control.to(memory_format=torch.contiguous_format).float()
    lq = control
    # apply preprocess model

    control = preprocess_model(control)
    
    
    control_norm = control * 2 - 1
    #save_image(control,'./control1.png')
    #pdb.set_trace()
    # apply condition encoder
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
            posterior = vae.encode(x).latent_dist
            z = posterior.mode()
            
            posterior_c = vae.encode(control_norm).latent_dist
            c_latent = posterior_c.mode()
    if return_xc:
        return z, dict(c_latent=[c_latent], lq=[lq], c_concat=[control],x=[x])
    else:
        return z, dict(c_latent=[c_latent], lq=[lq], c_concat=[control])
 
@torch.inference_mode()
def log_validation(model, epoch, step, batch, device, vae=None, preprocess_model = None):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()
    hw = torch.tensor([[config.image_size, config.image_size]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.]], device=device).repeat(1, 1)
    null_y = torch.load(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')
    null_y_mask = null_y['uncond_prompt_embeds_mask'].to(device)
    null_y = null_y['uncond_prompt_embeds'].to(device)
    
    bs = batch['jpg'].shape[0]
    imgname = batch['imgname']
    
    null_y = null_y.unsqueeze(0).repeat((bs,1,1,1))  # 4 x 1 x 120 x 4096 # T5 extracted feature of caption, 120 token, 4096
    null_y_mask = null_y_mask.unsqueeze(0).unsqueeze(0).repeat((bs,1,1,1))
    # print(null_y.shape)
    # print(null_y_mask.shape)
    # assert False
    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []
    latents = []

    
    
    z = torch.randn(bs, 4, latent_size, latent_size, device=device)
    #embed = torch.load(f'output/tmp/{prompt}_{max_length}token.pth', map_location='cpu')
    caption_embs, emb_masks = null_y,null_y_mask #embed['caption_embeds'].to(device), embed['emb_mask'].to(device)
    # caption_embs = caption_embs[:, None]
    # emb_masks = emb_masks[:, None]
    clean_image, cond = get_input(batch,vae,preprocess_model,return_xc=True)
    c = cond['c_latent'][0] * config.scale_factor
    model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks,c = c)

    dpm_solver = DPMS(model.forward_with_dpmsolver,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=1.,
                        model_kwargs=model_kwargs)
    denoised = dpm_solver.sample(
        z,
        steps=20,
        order=2,
        skip_type="time_uniform",
        method="multistep",
    )
    latents = denoised

    torch.cuda.empty_cache()
    if vae is None:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(accelerator.device).to(torch.float16)
    
    latents = latents.to(torch.float16)
    samples = vae.decode(latents.detach() / vae.config.scaling_factor).sample/2+0.5
    #samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    #image = Image.fromarray(samples)
    
    # control_samples = vae.decode(c.detach() / vae.config.scaling_factor).sample/2+0.5
    
    # lq_name = join(config.work_dir,'visual','lq_epoch'+str(epoch)+'_step_'+str(step)+'.png')
    # hq_name = join(config.work_dir,'visual','hq_epoch'+str(epoch)+'_step_'+str(step)+'.png')
    # cond_name = join(config.work_dir,'visual','cond_'+str(epoch)+'_step_'+str(step)+'.png')
    # sample_name = join(config.work_dir,'visual','sample_'+str(epoch)+'_step_'+str(step)+'.png')
    #control_samples = torch.clamp(127.5 * control_samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    #control_images = Image.fromarray(control_samples)
    
    save_batch(samples,imgname,config.work_dir)
    
    # save_image(control_samples,'./control0.png')
    # assert False
    #image_logs.append({"control_images": control_images, "images": [image]})

    # for tracker in accelerator.trackers:
    #     if tracker.name == "tensorboard":
    #         for log in image_logs:
    #             images = log["images"]
    #             validation_prompt = log["validation_prompt"]
    #             formatted_images = []
    #             for image in images:
    #                 formatted_images.append(np.asarray(image))

    #             formatted_images = np.stack(formatted_images)

    #             tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
    #     elif tracker.name == "wandb":
    #         import wandb
    #         formatted_images = []

    #         for log in image_logs:
    #             images = log["images"]
    #             validation_prompt = log["validation_prompt"]
    #             for image in images:
    #                 image = wandb.Image(image, caption=validation_prompt)
    #                 formatted_images.append(image)

    #         tracker.log({"validation": formatted_images})
    #     else:
    #         logger.warn(f"image logging not implemented for {tracker.name}")

    del vae
    flush()
    return image_logs
   
def test():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    
    
    preprocess_config = '/home/zyx/PixArt-sigma/configs/swinir.yaml'

    preprocess_model = instantiate_from_config(OmegaConf.load(preprocess_config))
    load_state_dict(preprocess_model, torch.load('/data3/zyx/visionhub/general_swinir_v1.ckpt', map_location="cpu"), strict=True)
    frozen_module(preprocess_model)
    preprocess_model.to(accelerator.device)

    vae = AutoencoderKL.from_pretrained(f'{args.vae_models_dir}', torch_dtype=torch.float16).to(accelerator.device)

    #if not load_vae_feat:
    #    raise ValueError("Only support load vae features for now.")
    # Now you train the model
    y_null_all = torch.load("/home/zyx/PixArt-sigma/output/pretrained_models/null_embed_diffusers_300token.pth")
    y_null = y_null_all['uncond_prompt_embeds'].to(accelerator.device)
    y_null_mask = y_null_all['uncond_prompt_embeds_mask'].to(accelerator.device)

    
    for step, batch in enumerate(train_dataloader):      
        log_validation(model, 0, step, batch, device=accelerator.device, vae=vae, preprocess_model=preprocess_model)

      


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the dir to save logs and models')
    parser.add_argument('--load_from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--resume_optimizer', action='store_true')
    parser.add_argument('--resume_lr_scheduler', action='store_true')
    
    parser.add_argument(
        '--vae_models_dir', default='/data3/zyx/pixart/pixart_sigma_sdxlvae_T5_diffusers/vae', type=str
    )
    parser.add_argument(
        "--pipeline_load_from", default='/data3/zyx/pixart/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.data_root:
        config.data_root = args.data_root
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=args.resume_optimizer,
            resume_lr_scheduler=args.resume_lr_scheduler)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 6
        config.optimizer.update({'lr': args.lr})

    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(join(config.work_dir,'visual'), exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=9600)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [512, 1024]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    
    tokenizer = text_encoder = None
    
    if config.visualize:
        # preparing embeddings for visualization. We put it here for saving GPU memory
        validation_prompts = [
            "dog",
            "portrait photo of a girl, photograph, highly detailed face, depth of field",
            "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        ]
        skip = True
        for prompt in validation_prompts:
            if not (os.path.exists(f'output/tmp/{prompt}_{max_length}token.pth')
                    and os.path.exists(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break
        if accelerator.is_main_process and not skip:
            if config.data.load_t5_feat and (tokenizer is None or text_encoder is None):
                logger.info(f"Loading text encoder and tokenizer from {args.pipeline_load_from} ...")
                tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
                text_encoder = T5EncoderModel.from_pretrained(
                    args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)
            for prompt in validation_prompts:
                txt_tokens = tokenizer(
                    prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(accelerator.device)
                caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
                torch.save(
                    {'caption_embeds': caption_emb, 'emb_mask': txt_tokens.attention_mask},
                    f'output/tmp/{prompt}_{max_length}token.pth')
            null_tokens = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
            torch.save(
                {'uncond_prompt_embeds': null_token_emb, 'uncond_prompt_embeds_mask': null_tokens.attention_mask},
                f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')
            if config.data.load_t5_feat:
                del tokenizer
                del txt_tokens
            flush()
            
    model_kwargs={"pe_interpolation": config.pe_interpolation, "config":config,
                  "model_max_length": max_length, "qk_norm": config.qk_norm,
                  "kv_compress_config": kv_compress_config, "micro_condition": config.micro_condition}

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps))
    model = build_model(config.model,
                                  config.grad_checkpointing,
                                  config.get('fp32_attention', False),
                                  input_size=latent_size,
                                  learn_sigma=learn_sigma,
                                  pred_sigma=pred_sigma,
                                  **model_kwargs)

    
 
    if image_size == 1024 or 512:
        model: ControlPixArtMSHalf = ControlPixArtMSHalf(model, copy_blocks_num=config.copy_blocks_num).train()
    else:
        model: ControlPixArtHalf = ControlPixArtHalf(model, copy_blocks_num=config.copy_blocks_num).train()
    
    if config.load_from is not None and args.resume_from is None:
        # load from PixArt model
        print(config.load_from)
        state_dict = find_model(config.load_from)['state_dict']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # missing, unexpected = load_checkpoint(config.load_from, model,load_ema=config.get('load_ema', False), max_length=max_length)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
        
    
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"T5 max token length: {config.model_max_length}")

    # if args.local_rank == 0:
    #     for name, params in model.named_parameters():
    #         if params.requires_grad == False: logger.info(f"freeze param: {name}")
    #
    #     for name, params in model.named_parameters():
    #         if params.requires_grad == True: logger.info(f"trainable param: {name}")

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    set_data_root(config.data_root)
    #dataset = build_dataset(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type, train_ratio=config.train_ratio)
    
    kernel_list = ['iso', 'aniso']
    kernel_prob = [0.5, 0.5]
    blur_sigma = [0.1, 10]
    downsample_range = [4, 8]
    noise_range = [0, 20]
    jpeg_range = [60, 100]
    #vae = AutoencoderKL.from_pretrained(f'{args.vae_models_dir}', torch_dtype=torch.float16).to(accelerator.device)

    
    dataset = CodeformerDatasetLQ(
        lq_list='/data3/zyx/FFHQ512/celeba-test-real.list',
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
    val_dataset = CodeformerDatasetLQ(
        lq_list='/data3/zyx/FFHQ512/val.list',
        out_size=512,
        crop_type='center',
        use_hflip=True,
        blur_kernel_size=41,
        kernel_list=kernel_list,
        kernel_prob=kernel_prob,
        blur_sigma=blur_sigma,
        downsample_range=downsample_range,
        noise_range=noise_range,
        jpeg_range=jpeg_range
        )
    
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                ratio_nums=dataset.ratio_nums, config=config, valid_num=1)
        # batch_sampler = BalancedAspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
        #                                                 batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio,
        #                                                 ratio_nums=dataset.ratio_nums)
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)
    else:
        train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)
        val_dataloader = build_dataloader(val_dataset, num_workers=config.num_workers, batch_size=config.val_batch_size, shuffle=True)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    optimizer = build_optimizer(model.controlnet, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        if args.resume_optimizer == False or args.resume_lr_scheduler == False:
            missing, unexpected = load_checkpoint(args.resume_from, model)
        else:
            start_epoch, missing, unexpected = load_checkpoint(**config.resume_from,
                                                               model=model,
                                                               optimizer=optimizer,
                                                               lr_scheduler=lr_scheduler,
                                                               )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model,)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    test()