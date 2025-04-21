import re
import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn import Module, Linear, init
from typing import Any, Mapping, Optional, Dict

from diffusion.model.nets import PixArtMSBlock, PixArtMS, PixArt
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed
from diffusion.model.utils import auto_grad_checkpoint

from diffusers import Transformer2DModel
from diffusers.models.attention import  BasicTransformerBlock
import pdb
# The implementation of ControlNet-Half architrecture
# https://github.com/lllyasviel/ControlNet/discussions/188
class ControlT2IDitBlockHalf(Module):
    def __init__(self, base_block: BasicTransformerBlock, block_index: 0) -> None:
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index

        for p in self.copied_block.parameters():
            p.requires_grad_(True)

        self.copied_block.load_state_dict(base_block.state_dict())
        self.copied_block.train()
        
        # pdb.set_trace()        
        self.hidden_size = hidden_size = base_block.hidden_size
        if self.block_index == 0:
            self.before_proj = Linear(hidden_size, hidden_size)
            init.zeros_(self.before_proj.weight)
            init.zeros_(self.before_proj.bias)
        self.after_proj = Linear(hidden_size, hidden_size) 
        init.zeros_(self.after_proj.weight)
        init.zeros_(self.after_proj.bias)

    def forward(self, x, y, t, mask=None, c=None):
        
        if self.block_index == 0:
            # the first block
            c = self.before_proj(c)
            c = self.copied_block(x + c, encoder_hidden_states = y, timestep = t, encoder_attention_mask = mask)
            # c = self.copied_block(x + c, y, t, mask)
            c_skip = self.after_proj(c)
        else:
            # load from previous c and produce the c for skip connection
            c = self.copied_block(c, encoder_hidden_states = y, timestep = t, encoder_attention_mask = mask)
            c_skip = self.after_proj(c)
        
        return c, c_skip
        

# The implementation of ControlPixArtHalf net
class ControlTransformerHalf(Module):
    # only support single res model
    def __init__(self, base_model: Transformer2DModel, copy_blocks_num: int = 13) -> None:
        super().__init__()
        self.base_model = base_model.eval()
        self.controlnet = []
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(base_model.transformer_blocks)
        # for p in self.base_model.parameters():
        #     print('block_lock!')
        #     p.requires_grad_(False)

        # Copy first copy_blocks_num block
        for i in range(copy_blocks_num):
            self.controlnet.append(ControlT2IDitBlockHalf(base_model.transformer_blocks[i], i))
        self.controlnet = nn.ModuleList(self.controlnet)
    
    def __getattr__(self, name: str):
        if name in ['forward', 'forward_with_dpmsolver', 'forward_with_cfg', 'forward_c', 'load_state_dict']:
            return self.__dict__[name]
        elif name in ['base_model', 'controlnet']:
            return super().__getattr__(name)
        else:
            return getattr(self.base_model, name)

    def forward_c(self, c):
        self.h, self.w = c.shape[-2]//self.patch_size, c.shape[-1]//self.patch_size
        # pos_embed = torch.from_numpy(
        #     get_2d_sincos_pos_embed(
        #         self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=self.pe_interpolation,
        #         base_size=self.base_size
        #     )
        # ).unsqueeze(0).to(c.device).to(self.dtype)
        #torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(c.device).to(self.dtype)
        return self.pos_embed(c) if c is not None else c

    # def forward(self, x, t, c, **kwargs):
    #     return self.base_model(x, t, c=self.forward_c(c), **kwargs)
    def forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        return_dict: bool = True,):
        # modify the original PixArtMS forward function
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c)
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
                hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
            )

        hidden_states = self.base_model.transformer_blocks[0](
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )
        for index in range(1, self.copy_blocks_num + 1):
            
            c, c_skip = self.controlnet[index-1](x=hidden_states,
                    y=encoder_hidden_states,
                    mask=encoder_attention_mask,
                    t=timestep,
                    c=c,
            )
            
            hidden_states = self.base_model.transformer_blocks[index](
                hidden_states+c_skip,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
        
        for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
            hidden_states = self.base_model.transformer_blocks[index](
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        output = self._get_output_for_patched_inputs(
                hidden_states=hidden_states,
                timestep=timestep,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep,
                height=height,
                width=width,
            )

        if not return_dict:
            return (output,)

        return output
    
        

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if all((k.startswith('base_model') or k.startswith('controlnet')) for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict)
        else:
            new_key = {}
            for k in state_dict.keys():
                new_key[k] = re.sub(r"(blocks\.\d+)(.*)", r"\1.base_block\2", k)
            for k, v in new_key.items():
                if k != v:
                    print(f"replace {k} to {v}")
                    state_dict[v] = state_dict.pop(k)

            return self.base_model.load_state_dict(state_dict, strict)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


