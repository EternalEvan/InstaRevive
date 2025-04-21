from typing import Sequence, Dict, Union
import math
import time
import torch
import numpy as np
import cv2
from PIL import Image,ImageDraw
import torch.utils.data as data
from os.path import join
from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr
from utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)

import os
from os.path import join
import json

def brush_stroke_mask(img, color=(255,255,255)):
    min_num_vertex = 8
    max_num_vertex = 28
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('RGB', (W, H), 0)
        #pdb.set_trace()
     
        if img is not None: mask = img
        np.random.seed()
        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))
            #print(mask.szie)
            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=color, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=color)

        return mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask

class CodeformerDataset(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        if min(*pil_img.size)!= self.out_size:
            pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.paths)


class CodeformerDataset_prompts(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset_prompts":
        super(CodeformerDataset_prompts, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        
        self.t5_file_dir = '/data1/zyx/pixart/imagenet_prompts'

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]
        
        class_id = gt_path.split('/')[-2]
        
        t5_json_path = join(self.t5_file_dir,class_id+'.npz')
        
        txt_info = np.load(t5_json_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        # if min(*pil_img.size)!= self.out_size:
        #     pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        # if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname,txt_fea=txt_fea,attention_mask= attention_mask)

    def __len__(self) -> int:
        return len(self.paths)

class CodeformerDataset_prompts_face(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset_prompts_face":
        super(CodeformerDataset_prompts_face, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        
        self.t5_file_dir = '/data1/zyx/'

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]
        
        
        
        dataset_id = gt_path.split('/')[3]
        t5_json_path = join(self.t5_file_dir,dataset_id,imgname[:-4]+'.npz')
        
        txt_info = np.load(t5_json_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        if min(*pil_img.size)!= self.out_size:
            pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname,txt_fea=txt_fea,attention_mask= attention_mask)

    def __len__(self) -> int:
        return len(self.paths)
    

class CodeformerDataset_style_face(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset_style_face":
        super(CodeformerDataset_style_face, self).__init__()
        
        self.file_list = json.load(open(file_list, 'r'))
        # self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        
        self.t5_file_dir = '/data1/zyx/FFHQ_style/prompt_embedding'
        
        self.input_dir = '/data3/zyx/FFHQ512/FFHQ_512'

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        file_dict = self.file_list[index]
        
        gt_path = file_dict["filepath"]
        
        style_type = file_dict["type"]
        
        success = False
        imgname = gt_path.split('/')[-1]
        
        input_path = join(self.input_dir,imgname)
        
        
        
        dataset_id = 'FFHQ512'
        t5_json_path = join(self.t5_file_dir,dataset_id,imgname[:-4]+'.npz')
        
        txt_info = np.load(t5_json_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                img_lq = Image.open(input_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        # if min(*pil_img.size)!= self.out_size:
        #     pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        # elif self.crop_type == "random":
        #     pil_img_gt = random_crop_arr(pil_img, self.out_size)
        # else:
            
        #     pil_img_gt = np.array(pil_img)
        #     assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        pil_img = np.array(pil_img)
        img_gt = (pil_img[..., ::-1] / 255.0).astype(np.float32)
        img_lq = np.array(img_lq)/255.0
        img_lq = img_lq[..., ::-1].astype(np.float32)
        # random horizontal flip
        imgs = augment([img_gt,img_lq], hflip=self.use_hflip, rotation=False, return_status=False)
        img_gt = imgs[0]
        img_lq = imgs[1]
            
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        # kernel = random_mixed_kernels(
        #     self.kernel_list,
        #     self.kernel_prob,
        #     self.blur_kernel_size,
        #     self.blur_sigma,
        #     self.blur_sigma,
        #     [-math.pi, math.pi],
        #     noise_range=None
        # )
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        # # downsample
        # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # # noise
        # if self.noise_range is not None:
        #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # # jpeg compression
        # if self.jpeg_range is not None:
        #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname,txt_fea=txt_fea,attention_mask= attention_mask)

    def __len__(self) -> int:
        return len(self.file_list)
    
class CodeformerDataset_Mask_prompt(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset_Mask_prompt":
        super(CodeformerDataset_Mask_prompt, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.t5_file_dir = '/data1/zyx/'

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        dataset_id = gt_path.split('/')[3]
        t5_json_path = join(self.t5_file_dir,dataset_id,imgname[:-4]+'.npz')
        
        txt_info = np.load(t5_json_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
            
        for _ in range(3):
            try:
                pil_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        img_gt = pil_img
        if pil_img.shape[0]!= self.out_size:
            img_gt = cv2.resize(pil_img,(self.out_size,self.out_size))
            #pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
       
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        #img_gt = img_gt[..., ::-1].astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        #print(img_gt.shape)

        img_lq = img_gt
        img_lq = np.asarray(brush_stroke_mask(Image.fromarray(img_gt,mode='RGB')))/255.0
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] /255.0 * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname,txt_fea=txt_fea,attention_mask= attention_mask)

    def __len__(self) -> int:
        return len(self.paths)
    
class CodeformerDatasetLQ(data.Dataset):
    
    def __init__(
        self,
        lq_list:str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDatasetLQ, self).__init__()
        self.lq_list = lq_list
        self.lq_paths = load_file_list(lq_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
       
        lq_path = self.lq_paths[index]
        success = False

        
        for _ in range(3):
            try:
                lq_img = Image.open(lq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {lq_path}"
        # if min(*pil_img.size)!= self.out_size:
        #     pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if self.crop_type == "center":
        #     pil_img_gt = center_crop_arr(pil_img, self.out_size)
        # elif self.crop_type == "random":
        #     pil_img_gt = random_crop_arr(pil_img, self.out_size)
        # else:
        #     pil_img_gt = np.array(pil_img)
        #     assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)

        if min(*lq_img.size)!= self.out_size:
            lq_img = lq_img.resize((self.out_size,self.out_size),resample=Image.BOX)        

        img_lq = np.array(lq_img)
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        
        imgname = lq_path.split('/')[-1]
        # random horizontal flip
        # img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_lq.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        # kernel = random_mixed_kernels(
        #     self.kernel_list,
        #     self.kernel_prob,
        #     self.blur_kernel_size,
        #     self.blur_sigma,
        #     self.blur_sigma,
        #     [-math.pi, math.pi],
        #     noise_range=None
        # )
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        # # downsample
        # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # # noise
        # if self.noise_range is not None:
        #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # # jpeg compression
        # if self.jpeg_range is not None:
        #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # # BGR to RGB, [-1, 1]
       
        dummyhq = img_lq[..., ::-1].astype(np.float32)
        # # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        return dict(jpg=dummyhq, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.lq_paths)
    
class CodeformerDatasetLQ_prompts(data.Dataset):
    
    def __init__(
        self,
        lq_list:str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        file_dir=None,
    ) -> "CodeformerDataset":
        super(CodeformerDatasetLQ_prompts, self).__init__()
        if file_dir is not None:
            file_list = os.listdir(file_dir)
            self.lq_paths = [join(file_dir,file) for file in file_list]
        
        else:    
            self.lq_list = lq_list
            self.lq_paths = load_file_list(lq_list)
            
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.t5_file_dir = '/data1/zyx/testsets/captions'

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
       
        lq_path = self.lq_paths[index]
        success = False

        imgname = lq_path.split('/')[-1]
        
        
        
        dataset_dir = lq_path.split('/')[3]
        
        t5_json_path = join(self.t5_file_dir,imgname[:-4]+'.npz')
        
        txt_fea = 0
        attention_mask = 0
        if os.path.exists(t5_json_path):
            txt_info = np.load(t5_json_path)
            txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
            if 'attention_mask' in txt_info.keys():
                attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
        
        for _ in range(3):
            try:
                lq_img = Image.open(lq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {lq_path}"
        # if min(*pil_img.size)!= self.out_size:
        #     pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if self.crop_type == "center":
        #     pil_img_gt = center_crop_arr(pil_img, self.out_size)
        # elif self.crop_type == "random":
        #     pil_img_gt = random_crop_arr(pil_img, self.out_size)
        # else:
        #     pil_img_gt = np.array(pil_img)
        #     assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)

        img_size = lq_img.size
        if min(*lq_img.size)!= self.out_size:
            lq_img = lq_img.resize((self.out_size,self.out_size),resample=Image.BOX)        

        img_lq = np.array(lq_img)
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        
        imgname = lq_path.split('/')[-1]
        # random horizontal flip
        # img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_lq.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        # kernel = random_mixed_kernels(
        #     self.kernel_list,
        #     self.kernel_prob,
        #     self.blur_kernel_size,
        #     self.blur_sigma,
        #     self.blur_sigma,
        #     [-math.pi, math.pi],
        #     noise_range=None
        # )
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        # # downsample
        # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # # noise
        # if self.noise_range is not None:
        #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # # jpeg compression
        # if self.jpeg_range is not None:
        #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # # BGR to RGB, [-1, 1]
       
        dummyhq = img_lq[..., ::-1].astype(np.float32)
        # # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        return dict(jpg=dummyhq, txt="", hint=source,imgname=imgname,txt_fea=txt_fea,attention_mask= attention_mask,img_size=img_size)

    def __len__(self) -> int:
        return len(self.lq_paths)



class CodeformerDataset_lora(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset_lora, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        '''kernel = random_mixed_kernels(
             self.kernel_list,
             self.kernel_prob,
             self.blur_kernel_size,
             self.blur_sigma,
             self.blur_sigma,
             [-math.pi, math.pi],
             noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)'''
        # downsample
        #scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        #img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_gt, self.noise_range)
            #img_lq = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            #img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
        # jpeg compression
        #if self.jpeg_range is not None:
        #    img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # resize to original size
        #img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = (img_lq[..., ::-1] * 2 - 1).astype(np.float32)
        #print(img_gt.shape)
        #print(img_lq.shape)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.paths)
    
    
class CodeformerDataset_lora_color(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset_lora_color, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        '''kernel = random_mixed_kernels(
             self.kernel_list,
             self.kernel_prob,
             self.blur_kernel_size,
             self.blur_sigma,
             self.blur_sigma,
             [-math.pi, math.pi],
             noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)'''
        # downsample
        #scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        #img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            #img_lq = random_add_gaussian_noise(img_gt, self.noise_range)
            img_lq = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
        # jpeg compression
        #if self.jpeg_range is not None:
        #    img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # resize to original size
        #img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = (img_lq[..., ::-1] * 2 - 1).astype(np.float32)
        #print(img_gt.shape)
        #print(img_lq.shape)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.paths)
    
    
class CodeformerDataset_lora_sr(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset_lora_sr, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        '''kernel = random_mixed_kernels(
             self.kernel_list,
             self.kernel_prob,
             self.blur_kernel_size,
             self.blur_sigma,
             self.blur_sigma,
             [-math.pi, math.pi],
             noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)'''
        # downsample
        #scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_gt, (int(w // 4), int(h // 4)), interpolation=cv2.INTER_LINEAR)
        # noise
        #if self.noise_range is not None:
            #img_lq = random_add_gaussian_noise(img_gt, self.noise_range)
            #img_lq = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            #img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
        # jpeg compression
        #if self.jpeg_range is not None:
        #    img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = (img_lq[..., ::-1] * 2 - 1).astype(np.float32)
        #print(img_gt.shape)
        #print(img_lq.shape)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.paths)
    

class CodeformerDataset_lora_lol(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset_lora_lol, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        lq_path = gt_path.replace('high','low')
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                img_lq = Image.open(lq_path).convert("RGB")
                pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
                img_lq = img_lq.resize((self.out_size,self.out_size),resample=Image.BOX)
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt,img_lq = center_crop_arr(pil_img, self.out_size,lq_image=img_lq)
        elif self.crop_type == "random":
            pil_img_gt,img_lq = random_crop_arr(pil_img, self.out_size,lq_image=img_lq)
        else:
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        #img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        #img_lq = augment(img_lq, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        '''kernel = random_mixed_kernels(
             self.kernel_list,
             self.kernel_prob,
             self.blur_kernel_size,
             self.blur_sigma,
             self.blur_sigma,
             [-math.pi, math.pi],
             noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)'''
        # downsample
        #scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        #img_lq = cv2.resize(img_gt, (int(w // 4), int(h // 4)), interpolation=cv2.INTER_LINEAR)
        # noise
        #if self.noise_range is not None:
            #img_lq = random_add_gaussian_noise(img_gt, self.noise_range)
            #img_lq = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            #img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
        # jpeg compression
        #if self.jpeg_range is not None:
        #    img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # resize to original size
        #img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = (img_lq[..., ::-1] * 2 - 1).astype(np.float32)
        #print(img_gt.shape)
        #print(img_lq.shape)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.paths)