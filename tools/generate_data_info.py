import os
from PIL import Image
import json

from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import sys
from tqdm import tqdm


device = sys.argv[1]

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", force_download=False, resume_download=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(f"cuda:{device}")



# 图片文件夹路径
images_folder = ["/data3/zyx/FFHQ512",'/data3/zyx/CelebA-HQ-img']
images_folder = ['/data3/zyx/celeba_512_validation']
# 存储结果的列表
results = []

# 遍历每个子文件夹（clip）
for clip_folder in images_folder:
    print(f'clip_folder = {clip_folder}')
    clip_path = clip_folder 
    if os.path.isdir(clip_path):
        # 遍历每张图片
        for image_file in os.listdir(clip_path):
            print(f'image_file = {image_file}', end=' ')
            image_path = os.path.join(clip_path, image_file)
            if image_file.endswith(".png") or image_file.endswith(".jpg"):
                # 使用 PIL 库获取图片的高度和宽度
                with Image.open(image_path).convert('RGB') as img:
                    width, height = img.size
                    # 计算图片的横纵比
                    ratio = width / height
                    # 使用文本处理库生成图片内容的文本提示
                    # 这里用简单的示例代替，你可以根据实际情况使用更复杂的文本处理方法
                    # prompt = "Description of the image content."
                    inputs = processor(img, return_tensors="pt").to(f"cuda:{device}")
                    out = model.generate(**inputs)
                    prompt = processor.decode(out[0], skip_special_tokens=True)
                    print(f'prompt = {prompt}')

                    # 将结果组织成字典
                    result = {
                        "height": height,
                        "width": width,
                        "ratio": ratio,
                        "path": os.path.join(clip_folder, image_file),
                        "prompt": prompt
                    }
                    results.append(result)

# 将结果输出为 JSON 格式
with open("/data3/zyx/FFHQ512/data_info_celeba_test.json", "w") as f:
    json.dump(results, f)


# import json

# # 读取 JSON 文件
# with open("/home/whl/workspace/srvideo/data/REDS/train_sharp_SimpleVRT/data_info.json", "r") as f:
#     data = json.load(f)

# # 对数据按照 path 排序
# sorted_data = sorted(data, key=lambda x: x["path"])

# # 将排序后的结果保存到新的 JSON 文件中
# with open("/home/whl/workspace/srvideo/data/REDS/train_sharp_SimpleVRT/sorted_data_info.json", "w") as f:
#     json.dump(sorted_data, f, indent=4)
