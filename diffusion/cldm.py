from typing import Mapping, Any
import copy
from collections import OrderedDict
import itertools
import pdb
import os 
from os.path import join
import cv2
import einops
import torch
import torch as th
import torch.nn as nn

from torchvision.utils import save_image
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.common import frozen_module

import torch.nn.functional as F
from diffusers import AutoencoderTiny


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = torch.cat((x, hint), dim=1)
        outs = []

        h = x.type(self.dtype)
        
        
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        preprocess_config,
        output='./',
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        self.output = output
        os.makedirs(self.output,exist_ok=True)
        # instantiate preprocess module (SwinIR)
        self.preprocess_model = instantiate_from_config(preprocess_config)
        frozen_module(self.preprocess_model)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))
        frozen_module(self.cond_encoder)

    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = c_cat
        log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        log["lq"] = c_lq
        log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]},
            steps=sample_steps
        )
        #x_samples = self.decode_first_stage(samples)
        log["samples"] = samples

        return log

    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h // 8, w // 8)
        samples = sampler.sample(
            steps, shape, cond["c_concat"][0],positive_prompt="", negative_prompt="",
        cfg_scale=1.0, color_fix_type="wavelet"
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def validation_step(self, batch, batch_idx):
        # TODO: 
        pass

    def test_step(self, batch, batch_idx):
        final_path = self.output
        imgname_batch = batch['imgname']
        # cv2.imwrite("watchlq.jpg",(batch['hint'][0]*255.0).cpu().numpy())
        # assert False
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]},
            steps=50
        )
        save_batch(images = samples,
            imgname_batch = imgname_batch,
            save_path = final_path,
            watch_step=False)
    
class Reflow_ControlLDM(LatentDiffusion):
    
    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        preprocess_config,
        lora_rank,
        output='./',
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        self.criterion = nn.MSELoss(reduction='mean')
        self.output_path = output
    
        # instantiate preprocess module (SwinIR)
        self.preprocess_model = instantiate_from_config(preprocess_config)
        frozen_module(self.preprocess_model)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))
        frozen_module(self.cond_encoder)
        #self.model.diffusion_model.requires_grad_(False)
        #self.control_model.requires_grad_(False)
        #self.unet_lora_params, self.train_names = inject_trainable_lora(self.model.diffusion_model,r=lora_rank)
        # print(self.train_names)
        # self.vae_for_regression = AutoencoderTiny.from_pretrained("/data3/zyx/pixart/tinyvae")
        # self.vae_for_regression.requires_grad_(False)
        
        self.values = torch.tensor([999])
        
        

    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        #batch = batch[0]
        x, c, hq  = super().get_input(batch, self.first_stage_key,*args, **kwargs, return_x=True)

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control]), hq

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c, hq = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        #z = z[:3,...]
        # log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["hq"] = hq/2+0.5
        
        log["control"] = c_cat
        
        log["lq"] = c_lq
        #log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]},
            steps=sample_steps
        )
        # samples = samples[:4,...]
        # c_latent = c_latent[:4,...]
        # log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        x_samples = self.decode_first_stage(samples)
        log["samples"] = (x_samples + 1) / 2

        return log

    @torch.no_grad()
    def sample_log(self, cond, steps):
       
        b, c, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h // 8, w // 8)
        zT = torch.randn(shape,device=self.device)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        ts = torch.ones(zT.shape[0],device=self.device)*(self.num_timesteps-1)
        if cond['c_latent'] is None:
            v = diffusion_model(x=zT, timesteps=ts, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=zT, hint=torch.cat(cond['c_latent'], 1),
                timesteps=ts, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            v = diffusion_model(x=zT, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        
        return zT+v

    def configure_optimizers(self):
        
        
        lr = self.learning_rate
        params = list(self.control_model.parameters())

        # if not self.sd_locked:
        #len_output_blocks = len(self.model.diffusion_model.output_blocks)
        #params += list(self.model.diffusion_model.output_blocks[len_output_blocks//2:].parameters())
        params += list(self.model.diffusion_model.parameters())
        #params += list(self.unet_lora_params)  
        
        #opt = torch.optim.AdamW(itertools.chain(self.unet_lora_params,params), lr=lr)
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    @torch.no_grad()
    def validation_step(self, batch,batch_idx):
        return
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]}
        

        b, _, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h // 8, w // 8)
        zT = torch.randn(shape,device=self.device)
        zt = zT
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        eular_steps = [999,801,601,401,201]
        #eular_steps = [999,901,801,701,601,501,401,301,201,101]
        for i,step in enumerate(eular_steps):
            ts = torch.ones(zT.shape[0],device=self.device)*step

            t_norm = ts.float()/(self.num_timesteps-1)
            t_norm = t_norm.view(zT.shape[0],1,1,1)

            #zt = t_norm * zT + (1 - t_norm) * zt
            if cond['c_latent'] is None:
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
            else:
                control = self.control_model(
                    x=zt, hint=torch.cat(cond['c_latent'], 1),
                    timesteps=ts, context=cond_txt
                )
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            #ztt = zt+v
            zt = zt+v/len(eular_steps)

            if i==3:
                x_samples = self.decode_first_stage(zT+v)
                log["samples_3"] = (x_samples + 1) / 2
        x_samples = self.decode_first_stage(zt)
        log["samples"] = (x_samples + 1) / 2
        return log

        # TODO: 
        # z0,cond = self.get_input(batch,self.first_stage_key)
        # zT = torch.rand_like(z0,device=self.device)
        # B = z0.shape[0]
        
        # t = torch.randint(0, self.num_timesteps, (z0.shape[0],), device=self.device).long()
        # t_norm = t.float()/(self.num_timesteps-1)
        # t_norm = t_norm.view(B,1,1,1)

        # zt = t_norm * zT + (1 - t_norm) * z0
        
        # diffusion_model = self.model.diffusion_model

        # cond_txt = torch.cat(cond['c_crossattn'], 1)

        # if cond['c_latent'] is None:
        #      v = diffusion_model(x=zt, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        # else:
        #     control = self.control_model(
        #         x=zt, hint=torch.cat(cond['c_latent'], 1),
        #         timesteps=t.float(), context=cond_txt
        #     )
        #     control = [c * scale for c, scale in zip(control, self.control_scales)]
        #     v = diffusion_model(x=zt, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        
        # loss_mse = self.criterion(z0-zT,v)
        # log_prefix = 'train' if self.training else 'val'
        # self.log(f'{log_prefix}/loss',loss_mse)

        # return
        pass

    def training_step(self, batch, batch_idx):
        #self.log_images(batch)
        z0,cond,hq = self.get_input(batch,self.first_stage_key)
        zT = torch.randn_like(z0,device=self.device)
        B = z0.shape[0]
        
        t = torch.randint(1, self.num_timesteps, (z0.shape[0],), device=self.device).long()
        #t = torch.ones(zT.shape[0],device=self.device)*(self.num_timesteps-1)
        
        #random_indices = torch.randint(0, len(self.values), (B,))

        #t = self.values[random_indices].to(zT)
        
        t_norm = t.float()/(self.num_timesteps-1)
        t_norm = t_norm.view(B,1,1,1)

        zt = t_norm * zT + (1 - t_norm) * z0
        #zt = zT
        diffusion_model = self.model.diffusion_model
        #print(diffusion_model.middle_block[1].transformer_blocks[0].attn1.to_q.lora_down.weight[0])
        #print(diffusion_model.middle_block[1].transformer_blocks[0].attn1.to_q.lora_up.weight[0])
        #print(diffusion_model.middle_block[1].transformer_blocks[0].attn1.to_q.lora_up.weight[0].requires_grad)
        # for k in range(len(self.unet_lora_params)):
        #      print(self.unet_lora_params[k][0][0])
        #      print(self.unet_lora_params[k][0][0].grad)
         
        #pdb.set_trace()
        # for k in itertools.chain(*self.unet_lora_params):
        #     print(k)
        #     print(k.grad)

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            v = diffusion_model(x=zt, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=zt, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t.float(), context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            v = diffusion_model(x=zt, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        losses = {}

        # loss_mse = self.criterion(z0-zT,v)
        
      
      
        x_refine = self.vae_for_regression.decode(zT+v).sample
        #x_hq = self.vae_for_regression.decode(z0).sample
        x_refine = torch.clamp(x_refine, min=-1.0, max=1.0)
        #x_hq = torch.clamp(x_hq, min=-1.0, max=1.0)

        if False:
            save_img = (hq / 2 + 0.5).clamp(0, 1)
            save_image(save_img,"watch_hq.jpg")
            refine_img = (x_refine / 2 + 0.5).clamp(0, 1)
            save_image(refine_img,"watch_refine.jpg")
            assert False
        #print(v.shape)
        loss_perc = self.criterion_lpips(hq,x_refine)*0.1
        log_prefix = 'train' if self.training else 'val'
        #losses.update({f'{log_prefix}/loss':loss_mse,f'{log_prefix}/loss_MSE':loss_mse})
        # self.log(f'{log_prefix}/loss_mse',loss_mse,logger=True)
        self.log(f'{log_prefix}/loss_lpips',loss_perc,logger=True)
        return loss_perc

    @torch.no_grad()
    def test_step(self, batch,batch_idx):
        #if batch_idx <=392:
        #    return 
        final_path = join(self.output_path,'final')
        mid_path = join(self.output_path,'mid')
        
        #final_path = '/home/zyx/DiffBIR-main/outputs/custom-test1/final'
        #mid_path = '/home/zyx/DiffBIR-main/outputs/custom-test1/midd'
        cond_path = join(self.output_path,'cond')
        
        z0_path = join(self.output_path,'z0')
        z1_path = join(self.output_path,'z1')
        z2_path = join(self.output_path,'z2')
        '''z1_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z1'
        z2_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z2'
        z4_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z4' '''
        hq_path = join(self.output_path,'hq')
        lq_path = join(self.output_path,'lq')
  
        os.makedirs(mid_path,exist_ok=True)
        os.makedirs(lq_path,exist_ok=True)
        os.makedirs(cond_path,exist_ok=True)
    
 
        '''os.makedirs(z1_path,exist_ok=True)
        os.makedirs(z2_path,exist_ok=True)
        os.makedirs(z4_path,exist_ok=True)'''
        
        images_hq = (batch['jpg'] + 1) / 2
        images_hq = images_hq.permute((0,3,1,2))
        imgname_batch = batch['imgname']

        save_batch(images=images_hq,
                   imgname_batch=imgname_batch,
                   save_path=lq_path,
                   watch_step=False)
        
        
        #os.makedirs(lq_path,exist_ok=True)
        os.makedirs(z0_path,exist_ok=True)
        os.makedirs(z1_path,exist_ok=True)
        os.makedirs(z2_path,exist_ok=True)
        os.makedirs(final_path,exist_ok=True)
        
        log = dict()
        #if batch_idx <=398:
        #    return
        imgname_batch = batch['imgname']
        z, c,_ = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        images_lq = c_lq
        B,C,H,W = z.shape
       
      
        #pdb.set_trace()
        # images_lq = images_lq.permute((0,3,1,2))
        # save_batch(images=images_lq,
        #            imgname_batch=imgname_batch,
        #            save_path=lq_path,
        #            watch_step=False)
        
        images_cond = c_cat
        log["control"] = c_cat
        #log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        
        #log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        
        cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]}
        

        b, _, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h // 8, w // 8)
        zT = torch.randn(shape,device=self.device)
        zt = zT
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        #eular_steps = [999,499]
        eular_steps = [999,799,599,349,199]
        #eular_steps = [999,801,601,401,201]
        #eular_steps = [999,899,799,699,599,499,399,299,199,99]
        for i,step in enumerate(eular_steps):
            ts = torch.ones(zT.shape[0],device=self.device)*step

            t_norm = ts.float()/(self.num_timesteps-1)
            t_norm = t_norm.view(zT.shape[0],1,1,1)

            #zt = t_norm * zT + (1 - t_norm) * zt
            if cond['c_latent'] is None:
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
            else:
                control = self.control_model(
                    x=zt, hint=torch.cat(cond['c_latent'], 1),
                    timesteps=ts, context=cond_txt
                )
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            #print(i)
            #ztt = zt+v
            zt = zt+v/len(eular_steps)
            
            ''' z_noise = torch.randn(shape,device=self.device)
            z_gt = z_noise*t_norm+ z*(1-t_norm)
            
            zt = zt*mask+z_gt*(1-mask)'''
            #zt = zt*mask + z*(1-mask)
            #x_samples = self.decode_first_stage(zT+v)
            #images_aa = (x_samples + 1) / 2
            #save_image(images_aa,'./aa_'+str(i)+'.png')
            #pdb.set_trace()
            if i == len(eular_steps)-1:
                x_samples = self.decode_first_stage(zt)
                images_final = (x_samples + 1) / 2
            #save_image(log['samples'],'aa{}_step.png'.format(str(i)))
            
            
            if i == 3:
                x_samples = self.decode_first_stage(zT+v )
                images_mid = (x_samples + 1) / 2
                images_midd = images_mid
                
                save_batch(images=images_midd,
                   imgname_batch=imgname_batch,
                   save_path=mid_path,
                   watch_step=False)
                
            
            if i == 0:
                x_samples = self.decode_first_stage(zT+v)
                images_z0 = (x_samples + 1) / 2
                images_z0 = images_z0
                
                save_batch(images=images_z0,
                   imgname_batch=imgname_batch,
                   save_path=z0_path,
                   watch_step=False)
                
            if i == 1:
                x_samples = self.decode_first_stage(zT+v)
                images_z1 = (x_samples + 1) / 2
                images_z1 = images_z1
                
                save_batch(images=images_z1,
                   imgname_batch=imgname_batch,
                   save_path=z1_path,
                   watch_step=False)
                
            if i == 2:
                x_samples = self.decode_first_stage(zT+v)
                images_z2 = (x_samples + 1) / 2
                images_z2 = images_z2
                
                save_batch(images=images_z2,
                   imgname_batch=imgname_batch,
                   save_path=z2_path,
                   watch_step=False)
            #loss_mse = self.criterion(z,zT+v)
            #loss_lpips = self.criterion_lpips(images_mid,images_hq)
            #print('lpips:',loss_lpips)
            #print('mse:',loss_mse)
        #loss_lpips = self.criterion_lpips(images,images_hq)
        #print('lpips_final:',loss_lpips)
        save_batch(images = images_final,
                   imgname_batch = imgname_batch,
                   save_path = final_path,
                   watch_step=False)
        

      
        
        save_batch(images=images_cond,
                   imgname_batch=imgname_batch,
                   save_path=cond_path,
                   watch_step=False)
        
        
        '''save_batch(images=images_z4,
                   imgname_batch=imgname_batch,
                   save_path=z4_path,
                   watch_step=False)'''
        
        
        
        
        
        return log
    
    def test_step_mask(self, batch,batch_idx):
        #if batch_idx <= 400 or batch_idx>=500:
        #    return 
        final_path = join(self.output_path,'final')
        mid_path = join(self.output_path,'mid')
        
        #final_path = '/home/zyx/DiffBIR-main/outputs/custom-test1/final'
        #mid_path = '/home/zyx/DiffBIR-main/outputs/custom-test1/midd'
        cond_path = join(self.output_path,'cond')
        
        z0_path = join(self.output_path,'z0')
        z1_path = join(self.output_path,'z1')
        z2_path = join(self.output_path,'z2')
        '''z1_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z1'
        z2_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z2'
        z4_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z4' '''
        hq_path = join(self.output_path,'hq')
        lq_path = join(self.output_path,'lq')
        os.makedirs(final_path,exist_ok=True)
        os.makedirs(mid_path,exist_ok=True)
        os.makedirs(cond_path,exist_ok=True)
        os.makedirs(hq_path,exist_ok=True)
        os.makedirs(lq_path,exist_ok=True)
        
        
        os.makedirs(z0_path,exist_ok=True)
        os.makedirs(z1_path,exist_ok=True)
        os.makedirs(z2_path,exist_ok=True)
        '''os.makedirs(z1_path,exist_ok=True)
        os.makedirs(z2_path,exist_ok=True)
        os.makedirs(z4_path,exist_ok=True)'''
        
        images_hq = (batch['jpg'] + 1) / 2
        images_hq = images_hq.permute((0,3,1,2))
        imgname_batch = batch['imgname']

        save_batch(images=images_hq,
                   imgname_batch=imgname_batch,
                   save_path=hq_path,
                   watch_step=False)
        
        
        #os.makedirs(lq_path,exist_ok=True)
        '''os.makedirs(z1_path,exist_ok=True)
        os.makedirs(z2_path,exist_ok=True)'''
        
        log = dict()
        #if batch_idx <=398:
        #    return
        imgname_batch = batch['imgname']
        for idx,name in enumerate(imgname_batch):
            imgname_batch[idx] = name[:-4]+'_'+str(batch_idx)+'_'+str(idx)+'.png'
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        images_lq = c_lq
        B,C,H,W = z.shape
        mask = torch.zeros(B,1,512, 512)
        print(c_lq.shape)
        m_ind = torch.sum(c_lq, dim=1).view(B,1,512, 512)
        mask[m_ind==3] = 1.0
        #pdb.set_trace()
        mask = mask.view(B,1,512, 512).to(c_lq)
        mask = F.interpolate(mask, size=(H,W), mode='nearest')
        #pdb.set_trace()
        #images_lq = images_lq.permute((0,3,1,2))
        save_batch(images=images_lq,
                   imgname_batch=imgname_batch,
                   save_path=lq_path,
                   watch_step=False)
        
        images_cond = c_cat
        log["control"] = c_cat
        #log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        
        #log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        
        cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]}
        

        b, _, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h // 8, w // 8)
        torch.seed()
        zT = torch.randn(shape,device=self.device)
        zt = zT
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        eular_steps = [999,749,499,249]
        #eular_steps = [999,801,601,401,201]
        #eular_steps = [999,899,799,699,599,499,399,299,199,99]
        for i,step in enumerate(eular_steps):
            ts = torch.ones(zT.shape[0],device=self.device)*step

            t_norm = ts.float()/(self.num_timesteps-1)
            t_norm = t_norm.view(zT.shape[0],1,1,1)

            #zt = t_norm * zT + (1 - t_norm) * zt
            if cond['c_latent'] is None:
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
            else:
                control = self.control_model(
                    x=zt, hint=torch.cat(cond['c_latent'], 1),
                    timesteps=ts, context=cond_txt
                )
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            #print(i)
            #ztt = zt+v
            zt = zt+v/len(eular_steps)
            
            ''' z_noise = torch.randn(shape,device=self.device)
            z_gt = z_noise*t_norm+ z*(1-t_norm)
            
            zt = zt*mask+z_gt*(1-mask)'''
            #zt = zt*mask + z*(1-mask)
            #x_samples = self.decode_first_stage(zT+v)
            #images_aa = (x_samples + 1) / 2
            #save_image(images_aa,'./aa_'+str(i)+'.png')
            #pdb.set_trace()
            if i == len(eular_steps)-1:
                x_samples = self.decode_first_stage(zt*mask + z*(1-mask))
                images_final = (x_samples + 1) / 2
            #save_image(log['samples'],'aa{}_step.png'.format(str(i)))
            
            
            '''if i == 3:
                x_samples = self.decode_first_stage((zT+v)*mask + z*(1-mask))
                images_mid = (x_samples + 1) / 2
                images_midd = images_mid
                
                save_batch(images=images_midd,
                   imgname_batch=imgname_batch,
                   save_path=mid_path,
                   watch_step=False)'''
            
            '''if i == 0:
                x_samples = self.decode_first_stage(zT+v)
                images_z0 = (x_samples + 1) / 2
                images_z0 = images_z0
                
                save_batch(images=images_z0,
                   imgname_batch=imgname_batch,
                   save_path=z0_path,
                   watch_step=False)'''
                
            '''if i == 1:
                x_samples = self.decode_first_stage(zT+v)
                images_z1 = (x_samples + 1) / 2
                images_z1 = images_z1
                
                save_batch(images=images_z1,
                   imgname_batch=imgname_batch,
                   save_path=z1_path,
                   watch_step=False)'''
                
            if i == 2:
                x_samples = self.decode_first_stage((zT+v)*mask + z*(1-mask))
                images_z2 = (x_samples + 1) / 2
                images_z2 = images_z2
                
                save_batch(images=images_z2,
                   imgname_batch=imgname_batch,
                   save_path=z2_path,
                   watch_step=False)
            #loss_mse = self.criterion(z,zT+v)
            #loss_lpips = self.criterion_lpips(images_mid,images_hq)
            #print('lpips:',loss_lpips)
            #print('mse:',loss_mse)
        #loss_lpips = self.criterion_lpips(images,images_hq)
        #print('lpips_final:',loss_lpips)
        save_batch(images = images_final,
                   imgname_batch = imgname_batch,
                   save_path = final_path,
                   watch_step=False)
        

      
        
        save_batch(images=images_cond,
                   imgname_batch=imgname_batch,
                   save_path=cond_path,
                   watch_step=False)
        
        
        '''save_batch(images=images_z4,
                   imgname_batch=imgname_batch,
                   save_path=z4_path,
                   watch_step=False)'''
        
        
        
        
        
        return log    

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
                
   
                
                
class Dist_ControlLDM(LatentDiffusion):
    
    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        preprocess_config,
        lora_rank,
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        self.criterion = nn.MSELoss(reduction='mean')
        
        #self.criterion_lpips = LPIPS(
        #    net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
        #    version='0.1'  # Currently, v0.1 is supported
        # )
        # instantiate preprocess module (SwinIR)
        self.preprocess_model = instantiate_from_config(preprocess_config)
        frozen_module(self.preprocess_model)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))
        
        frozen_module(self.cond_encoder)
        self.model.diffusion_model.requires_grad_(False)
        self.unet_lora_params, self.train_names = inject_trainable_lora(self.model.diffusion_model,r=lora_rank)
        print(self.train_names)
        

    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        #batch = batch[0]
        x, c  = super().get_input(batch, self.first_stage_key,*args, **kwargs)

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        #control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        z = z[:3,...]
        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        
        log["control"] = c_cat
        
        log["lq"] = c_lq
        #log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]},
            steps=sample_steps
        )
        samples = samples[:3,...]
        c_latent = c_latent[:3,...]
        log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        x_samples = self.decode_first_stage(samples)
        log["samples"] = (x_samples + 1) / 2

        return log

    @torch.no_grad()
    def sample_log(self, cond, steps):
       
        b, c, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h // 8, w // 8)
        zT = torch.randn(shape,device=self.device)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        ts = torch.ones(zT.shape[0],device=self.device)*(self.num_timesteps-1)
        if cond['c_latent'] is None:
            v = diffusion_model(x=zT, timesteps=ts, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=zT, hint=torch.cat(cond['c_latent'], 1),
                timesteps=ts, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            v = diffusion_model(x=zT, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        
        return zT+v

    def configure_optimizers(self):
        
        
        lr = self.learning_rate
        params =   list(self.control_model.parameters())
        # if not self.sd_locked:
        #len_output_blocks = len(self.model.diffusion_model.output_blocks)
        #params += list(self.model.diffusion_model.output_blocks[len_output_blocks//2:].parameters())
        #params += list(self.model.diffusion_model.out.parameters())
        #   params += list(self.unet_lora_params)  
        
        opt = torch.optim.AdamW(itertools.chain(self.unet_lora_params,params), lr=lr)
        # opt = torch.optim.AdamW(params, lr=lr)
        return opt

    @torch.no_grad()
    def validation_step(self, batch,batch_idx):
        return
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]}
        

        b, _, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h // 8, w // 8)
        zT = torch.randn(shape,device=self.device)
        zt = zT
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        eular_steps = [999,801,601,401,201]
        #eular_steps = [999,901,801,701,601,501,401,301,201,101]
        for i,step in enumerate(eular_steps):
            ts = torch.ones(zT.shape[0],device=self.device)*step

            t_norm = ts.float()/(self.num_timesteps-1)
            t_norm = t_norm.view(zT.shape[0],1,1,1)

            #zt = t_norm * zT + (1 - t_norm) * zt
            if cond['c_latent'] is None:
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
            else:
                control = self.control_model(
                    x=zt, hint=torch.cat(cond['c_latent'], 1),
                    timesteps=ts, context=cond_txt
                )
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            #ztt = zt+v
            zt = zt+v/len(eular_steps)

            if i==3:
                x_samples = self.decode_first_stage(zT+v)
                log["samples_3"] = (x_samples + 1) / 2
        x_samples = self.decode_first_stage(zt)
        log["samples"] = (x_samples + 1) / 2
        return log

        # TODO: 
        # z0,cond = self.get_input(batch,self.first_stage_key)
        # zT = torch.rand_like(z0,device=self.device)
        # B = z0.shape[0]
        
        # t = torch.randint(0, self.num_timesteps, (z0.shape[0],), device=self.device).long()
        # t_norm = t.float()/(self.num_timesteps-1)
        # t_norm = t_norm.view(B,1,1,1)

        # zt = t_norm * zT + (1 - t_norm) * z0
        
        # diffusion_model = self.model.diffusion_model

        # cond_txt = torch.cat(cond['c_crossattn'], 1)

        # if cond['c_latent'] is None:
        #      v = diffusion_model(x=zt, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        # else:
        #     control = self.control_model(
        #         x=zt, hint=torch.cat(cond['c_latent'], 1),
        #         timesteps=t.float(), context=cond_txt
        #     )
        #     control = [c * scale for c, scale in zip(control, self.control_scales)]
        #     v = diffusion_model(x=zt, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        
        # loss_mse = self.criterion(z0-zT,v)
        # log_prefix = 'train' if self.training else 'val'
        # self.log(f'{log_prefix}/loss',loss_mse)

        # return
        pass

    def training_step(self, batch, batch_idx):
        #self.log_images(batch)
        z0,cond = self.get_input(batch,self.first_stage_key)
        zT = torch.randn_like(z0,device=self.device)
        B = z0.shape[0]
        
        ts = torch.ones(zT.shape[0],device=self.device)*(self.num_timesteps-1)
        '''t = torch.randint(1, self.num_timesteps, (z0.shape[0],), device=self.device).long()
        t_norm = t.float()/(self.num_timesteps-1)
        t_norm = t_norm.view(B,1,1,1)'''

        zt =  zT 
        
        diffusion_model = self.model.diffusion_model
        #print(diffusion_model.middle_block[1].transformer_blocks[0].attn1.to_q.lora_down.weight[0])
        #print(diffusion_model.middle_block[1].transformer_blocks[0].attn1.to_q.lora_up.weight[0])
        #print(diffusion_model.middle_block[1].transformer_blocks[0].attn1.to_q.lora_up.weight[0].requires_grad)
        # for k in range(len(self.unet_lora_params)):
        #      print(self.unet_lora_params[k][0][0])
        #      print(self.unet_lora_params[k][0][0].grad)
         
        #pdb.set_trace()
        # for k in itertools.chain(*self.unet_lora_params):
        #     print(k)
        #     print(k.grad)

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=zt, hint=torch.cat(cond['c_latent'], 1),
                timesteps=ts.float(), context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        losses = {}
        loss_mse = self.criterion(z0-zT,v)
      
      
        #x_refine = self.decode_first_stage(zT+v)
        #x_hq = self.decode_first_stage(z0)
        #print(v.shape)
        #loss_perc = self.criterion_lpips(x_hq,x_refine)
        log_prefix = 'train' if self.training else 'val'
        #losses.update({f'{log_prefix}/loss':loss_mse,f'{log_prefix}/loss_MSE':loss_mse})
        self.log(f'{log_prefix}/loss_mse',loss_mse,logger=True)
        #self.log(f'{log_prefix}/loss_lpips',loss_perc,logger=True)
        return loss_mse#+loss_perc

    @torch.no_grad()
    def test_step(self, batch,batch_idx):
        if batch_idx <= 475 or batch_idx>=550:
            return 
        final_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-5/final'
        mid_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-5/midd'
        
        #final_path = '/home/zyx/DiffBIR-main/outputs/custom-test1/final'
        #mid_path = '/home/zyx/DiffBIR-main/outputs/custom-test1/midd'
        cond_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-5/cond'
        
        '''z0_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z0'
        z1_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z1'
        z2_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z2'
        z4_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-4/z4' '''
        hq_path = '/home/user001/zwl/zyx/Diffbir/outputs/inpainting-5/hq'
        os.makedirs(final_path,exist_ok=True)
        os.makedirs(mid_path,exist_ok=True)
        os.makedirs(cond_path,exist_ok=True)
        os.makedirs(hq_path,exist_ok=True)
        
        '''os.makedirs(z0_path,exist_ok=True)
        os.makedirs(z1_path,exist_ok=True)
        os.makedirs(z2_path,exist_ok=True)
        os.makedirs(z4_path,exist_ok=True)'''
        
        images_hq = (batch['jpg'] + 1) / 2
        images_hq = images_hq.permute((0,3,1,2))
        imgname_batch = batch['imgname']

        save_batch(images=images_hq,
                   imgname_batch=imgname_batch,
                   save_path=hq_path,
                   watch_step=False)
        
        #os.makedirs(lq_path,exist_ok=True)
        '''os.makedirs(z1_path,exist_ok=True)
        os.makedirs(z2_path,exist_ok=True)'''
        
        log = dict()
        #if batch_idx <=398:
        #    return
        imgname_batch = batch['imgname']
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        
        images_cond = c_cat
        log["control"] = c_cat
        #log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        
        #log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        
        cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]}
        

        b, _, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h // 8, w // 8)
        zT = torch.randn(shape,device=self.device)
        zt = zT
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        #eular_steps = [999,666,333]
        eular_steps = [999,801,601,401,201]
        #eular_steps = [999,899,799,699,599,499,399,299,199,99]
        for i,step in enumerate(eular_steps):
            ts = torch.ones(zT.shape[0],device=self.device)*step

            t_norm = ts.float()/(self.num_timesteps-1)
            t_norm = t_norm.view(zT.shape[0],1,1,1)

            #zt = t_norm * zT + (1 - t_norm) * zt
            if cond['c_latent'] is None:
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
            else:
                control = self.control_model(
                    x=zt, hint=torch.cat(cond['c_latent'], 1),
                    timesteps=ts, context=cond_txt
                )
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                v = diffusion_model(x=zt, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            #print(i)
            #ztt = zt+v
            zt = zt+v/len(eular_steps)
            #x_samples = self.decode_first_stage(zT+v)
            #images_aa = (x_samples + 1) / 2
            #save_image(images_aa,'./aa_'+str(i)+'.png')
            #pdb.set_trace()
            if i == len(eular_steps)-1:
                x_samples = self.decode_first_stage(zt)
                images_final = (x_samples + 1) / 2
                x_samples = self.decode_first_stage(zT+v)
                images_z4 = (x_samples + 1) / 2
            #save_image(log['samples'],'aa{}_step.png'.format(str(i)))
            
            
            if i == 3:
                x_samples = self.decode_first_stage(zT+v)
                images_mid = (x_samples + 1) / 2
                images_midd = images_mid
            
            if i == 0:
                x_samples = self.decode_first_stage(zT+v)
                images_z0 = (x_samples + 1) / 2
                images_z0 = images_z0
                
            if i == 1:
                x_samples = self.decode_first_stage(zT+v)
                images_z1 = (x_samples + 1) / 2
                images_z1 = images_z1
                
            if i == 2:
                x_samples = self.decode_first_stage(zT+v)
                images_z2 = (x_samples + 1) / 2
                images_z2 = images_z2
            #loss_mse = self.criterion(z,zT+v)
            #loss_lpips = self.criterion_lpips(images_mid,images_hq)
            #print('lpips:',loss_lpips)
            #print('mse:',loss_mse)
        #loss_lpips = self.criterion_lpips(images,images_hq)
        #print('lpips_final:',loss_lpips)
        save_batch(images = images_final,
                   imgname_batch = imgname_batch,
                   save_path = final_path,
                   watch_step=False)
        save_batch(images=images_midd,
                   imgname_batch=imgname_batch,
                   save_path=mid_path,
                   watch_step=False)

      
        
        save_batch(images=images_cond,
                   imgname_batch=imgname_batch,
                   save_path=cond_path,
                   watch_step=False)
        
        
        '''save_batch(images=images_z4,
                   imgname_batch=imgname_batch,
                   save_path=z4_path,
                   watch_step=False)'''
        '''save_batch(images=images_z0,
                   imgname_batch=imgname_batch,
                   save_path=z0_path,
                   watch_step=False)'''
        
        '''save_batch(images=images_z1,
                   imgname_batch=imgname_batch,
                   save_path=z1_path,
                   watch_step=False)'''
        '''save_batch(images=images_z2,
                   imgname_batch=imgname_batch,
                   save_path=z2_path,
                   watch_step=False)'''
        
        return log

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