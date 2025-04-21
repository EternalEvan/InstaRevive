# Config for PixArt-DMD
_base_ = ['../PixArt_xl2_internal.py']
data_root = 'pixart-sigma-toy-dataset'

image_list_json = ['data_info.json']

data = dict(
    type='DMD', root='InternData', image_list_json=image_list_json, transform='default_train',
    load_vae_feat=False, load_t5_feat=False
)
image_size = 512

# model setting
model = 'PixArtMS_XL_2'     # model for multi-scale training
fp32_attention = True
load_from = "/data3/zyx/pixart/models--PixArt-alpha--PixArt-Alpha-DMD-XL-2-512x512/snapshots/57df7bb32daad69bc1d6c825a275935113802169"
vae_pretrained = "/data3/zyx/pixart/vae_for_dmd"
teacher_model_load_from = '/data3/zyx/pixart/PixART-XL-512'
tiny_vae_pretrained = "output/pretrained_models/tinyvae"
aspect_ratio_type = 'ASPECT_RATIO_512'
multi_scale = True     # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 10
train_batch_size = 1   # max 40 for PixArt-xL/2 when grad_checkpoint
num_epochs = 10  # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=1000)

log_interval = 20
save_model_epochs=1
save_model_steps=2000
work_dir = '/data3/zyx/pixart/dmd-bfr'

save_unet_only = True