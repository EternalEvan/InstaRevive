from typing import Dict, Sequence
import math
import random
import time

import numpy as np
import torch
from torch.utils import data
from PIL import Image

from utils.degradation import circular_lowpass_kernel, random_mixed_kernels
from utils.image import augment, random_crop_arr, center_crop_arr
from utils.file import load_file_list

from os.path import join

class RealESRGANDataset(data.Dataset):
    """
    # TODO: add comment
    """

    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        use_rot: bool,
        # blur kernel settings of the first degradation stage
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        betag_range: Sequence[float],
        betap_range: Sequence[float],
        sinc_prob: float,
        # blur kernel settings of the second degradation stage
        blur_kernel_size2: int,
        kernel_list2: Sequence[str],
        kernel_prob2: Sequence[float],
        blur_sigma2: Sequence[float],
        betag_range2: Sequence[float],
        betap_range2: Sequence[float],
        sinc_prob2: float,
        final_sinc_prob: float
    ) -> "RealESRGANDataset":
        super(RealESRGANDataset, self).__init__()
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["center", "random", "none"], f"invalid crop type: {self.crop_type}"

        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        # a list for each kernel probability
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        # betag used in generalized Gaussian blur kernels
        self.betag_range = betag_range
        # betap used in plateau blur kernels
        self.betap_range = betap_range
        # the probability for sinc filters
        self.sinc_prob = sinc_prob

        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2
        
        # a final sinc filter
        self.final_sinc_prob = final_sinc_prob
        
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        
        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1
        
        self.t5_file_dir = '/data1/zyx/pixart/imagenet_prompts'

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # -------------------------------- Load hq images -------------------------------- #
        hq_path = self.paths[index]
        success = False
        

        imgname = hq_path.split('/')[-1]
        
        class_id = hq_path.split('/')[-2]
        
        t5_json_path = join(self.t5_file_dir,class_id+'.npz')
        
        txt_info = np.load(t5_json_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
            
        for _ in range(3):
            try:
                pil_img = Image.open(hq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {hq_path}"
        
        if self.crop_type == "random":
            pil_img = random_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "center":
            pil_img = center_crop_arr(pil_img, self.out_size)
        # self.crop_type is "none"
        else:
            pil_img = np.array(pil_img)
            assert pil_img.shape[:2] == (self.out_size, self.out_size)
        # hwc, rgb to bgr, [0, 255] to [0, 1], float32
        img_hq = (pil_img[..., ::-1] / 255.0).astype(np.float32)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_hq = augment(img_hq, self.use_hflip, self.use_rot)
        
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = torch.from_numpy(
            img_hq[..., ::-1].transpose(2, 0, 1).copy()
        ).float()
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return {
            "hq": img_hq, "kernel1": kernel, "kernel2": kernel2,
            "sinc_kernel": sinc_kernel, "txt_fea": txt_fea, "attention_mask": attention_mask
        }

    def __len__(self) -> int:
        return len(self.paths)
    
    # def degrade_fun(self, conf_degradation, im_gt, kernel1, kernel2, sinc_kernel):
    #     if not hasattr(self, 'jpeger'):
    #         self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts

    #     ori_h, ori_w = im_gt.size()[2:4]
    #     sf = conf_degradation.sf

    #     # ----------------------- The first degradation process ----------------------- #
    #     # blur
    #     out = filter2D(im_gt, kernel1)
    #     # random resize
    #     updown_type = random.choices(
    #             ['up', 'down', 'keep'],
    #             conf_degradation['resize_prob'],
    #             )[0]
    #     if updown_type == 'up':
    #         scale = random.uniform(1, conf_degradation['resize_range'][1])
    #     elif updown_type == 'down':
    #         scale = random.uniform(conf_degradation['resize_range'][0], 1)
    #     else:
    #         scale = 1
    #     mode = random.choice(['area', 'bilinear', 'bicubic'])
    #     out = F.interpolate(out, scale_factor=scale, mode=mode)
    #     # add noise
    #     gray_noise_prob = conf_degradation['gray_noise_prob']
    #     if random.random() < conf_degradation['gaussian_noise_prob']:
    #         out = random_add_gaussian_noise_pt(
    #             out,
    #             sigma_range=conf_degradation['noise_range'],
    #             clip=True,
    #             rounds=False,
    #             gray_prob=gray_noise_prob,
    #             )
    #     else:
    #         out = random_add_poisson_noise_pt(
    #             out,
    #             scale_range=conf_degradation['poisson_scale_range'],
    #             gray_prob=gray_noise_prob,
    #             clip=True,
    #             rounds=False)
    #     # JPEG compression
    #     jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range'])
    #     out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    #     out = self.jpeger(out, quality=jpeg_p)

    #     # ----------------------- The second degradation process ----------------------- #
    #     # blur
    #     if random.random() < conf_degradation['second_order_prob']:
    #         if random.random() < conf_degradation['second_blur_prob']:
    #             out = filter2D(out, kernel2)
    #         # random resize
    #         updown_type = random.choices(
    #                 ['up', 'down', 'keep'],
    #                 conf_degradation['resize_prob2'],
    #                 )[0]
    #         if updown_type == 'up':
    #             scale = random.uniform(1, conf_degradation['resize_range2'][1])
    #         elif updown_type == 'down':
    #             scale = random.uniform(conf_degradation['resize_range2'][0], 1)
    #         else:
    #             scale = 1
    #         mode = random.choice(['area', 'bilinear', 'bicubic'])
    #         out = F.interpolate(
    #                 out,
    #                 size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
    #                 mode=mode,
    #                 )
    #         # add noise
    #         gray_noise_prob = conf_degradation['gray_noise_prob2']
    #         if random.random() < conf_degradation['gaussian_noise_prob2']:
    #             out = random_add_gaussian_noise_pt(
    #                 out,
    #                 sigma_range=conf_degradation['noise_range2'],
    #                 clip=True,
    #                 rounds=False,
    #                 gray_prob=gray_noise_prob,
    #                 )
    #         else:
    #             out = random_add_poisson_noise_pt(
    #                 out,
    #                 scale_range=conf_degradation['poisson_scale_range2'],
    #                 gray_prob=gray_noise_prob,
    #                 clip=True,
    #                 rounds=False,
    #                 )

    #     # JPEG compression + the final sinc filter
    #     # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    #     # as one operation.
    #     # We consider two orders:
    #     #   1. [resize back + sinc filter] + JPEG compression
    #     #   2. JPEG compression + [resize back + sinc filter]
    #     # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    #     if random.random() < 0.5:
    #         # resize back + the final sinc filter
    #         mode = random.choice(['area', 'bilinear', 'bicubic'])
    #         out = F.interpolate(
    #                 out,
    #                 size=(ori_h // sf, ori_w // sf),
    #                 mode=mode,
    #                 )
    #         out = filter2D(out, sinc_kernel)
    #         # JPEG compression
    #         jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
    #         out = torch.clamp(out, 0, 1)
    #         out = self.jpeger(out, quality=jpeg_p)
    #     else:
    #         # JPEG compression
    #         jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
    #         out = torch.clamp(out, 0, 1)
    #         out = self.jpeger(out, quality=jpeg_p)
    #         # resize back + the final sinc filter
    #         mode = random.choice(['area', 'bilinear', 'bicubic'])
    #         out = F.interpolate(
    #                 out,
    #                 size=(ori_h // sf, ori_w // sf),
    #                 mode=mode,
    #                 )
    #         out = filter2D(out, sinc_kernel)

    #     # clamp and round
    #     im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    #     return {'lq':im_lq.contiguous(), 'gt':im_gt}
