import re
import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageSequence, ImageOps
import torchvision.transforms.functional as F
from collections import namedtuple
import folder_paths
from io import BytesIO
import requests
import time
import copy


from hashlib import sha1, md5, new as hashlib_new
from base64 import urlsafe_b64encode, urlsafe_b64decode


def _file_iter(input_stream, size, offset=0):

    input_stream.seek(offset)
    d = input_stream.read(size)
    while d:
        yield d
        d = input_stream.read(size)
    input_stream.seek(0)


def _sha1(data):
    h = sha1()
    h.update(data)
    return h.digest()

def urlsafe_base64_encode(data):
    ret = urlsafe_b64encode(b(data))
    return s(ret)

def etag_stream(input_stream):
    def b(data):
        if isinstance(data, str):
            return data.encode('utf-8')
        return data
    array = [_sha1(block) for block in _file_iter(input_stream, _BLOCK_SIZE)]
    if len(array) == 0:
        array = [_sha1(b'')]
    if len(array) == 1:
        data = array[0]
        prefix = b'\x16'
    else:
        sha1_str = b('').join(array)
        data = _sha1(sha1_str)
        prefix = b'\x96'
    return urlsafe_base64_encode(prefix + data)

def etag(filePath):
    """计算文件的etag:

    Args:
        filePath: 待计算etag的文件路径

    Returns:
        输入文件的etag值
    """
    with open(filePath, 'rb') as f:
        return etag_stream(f)
    



def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


TEXT_TYPE = "STRING"

class CleanPromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (TEXT_TYPE, {"forceInput":  (True if TEXT_TYPE == 'STRING' else False)}),
                "prompt_filter": ("STRING", {
                    "multiline": True,
                    "default": ""
                })
            }
        }
    
    RETURN_TYPES = (TEXT_TYPE,)
    RETURN_NAMES = ("textc")

    FUNCTION = "run_it"

    CATEGORY = "biubiubiu/Prompt"

    def run_it(self, text: str, prompt_filter: str):
        prompt_filter_list = prompt_filter.split(',')

        text = ',' + text
        for p in prompt_filter_list:
            p = p.strip().replace('\n', '')
            if p == '':
                continue
            text = re.sub(f'(,\s*{p})(\s*,)', ', ', text)
            
        text = text.strip(',').strip()
        text = re.sub(f',\s*,', ', ', text)
        return (text, )


class Image_Filters:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),

            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_it"

    CATEGORY = "biubiubiu/Image"

    def run_it(self, image, contrast, saturation, sharpness):
        tensors = []
        for img in image:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            if contrast > 1.0 or contrast < 1.0:
                img = F.adjust_contrast(img, contrast_factor=contrast)
            if saturation > 1.0 or saturation < 1.0:
                img = F.adjust_saturation(img, saturation_factor=saturation)

            if sharpness > 1.0 or sharpness < 1.0:
                img = F.adjust_sharpness(img, sharpness_factor=sharpness)
            
            img = img.permute(1,2,0)
            tensors.append(img)

        return (tensors, )
    

class LoadUrlImage:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ('STRING', {"default": ""}),
                "cache_name": ("STRING", {"default": ""}),
                "hash": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "run_it"

    CATEGORY = "biubiubiu/Image"

    def _download(self, url, saved_fn):
        download_OK = False
        attempts = 0 
        try:
            while not download_OK:
                assert attempts <= 5
                try:
                    response = requests.get(url)
                except Exception as e:
                    print(f"Download {url} network error" + e)
                if response.status_code == 200:
                    image_file = BytesIO(response.content)
                    image = Image.open(image_file)
                    image.save(saved_fn)
                    download_OK = True
                time.sleep(0.1)
                attempts += 1
        except:
            raise RuntimeError(f"Download image {url} failed after {attempts} attempts.")
        
        return download_OK


    def run_it(self, url, cache_name, hash):
        image_cached_path = os.path.join(folder_paths.get_input_directory(), cache_name)
        cached = False
        if os.path.exists(image_cached_path):
            if hash == etag(image_cached_path):
                cached = True
            else:
                print(f"{cache_name} uncorrect hash checking from cache")
        if not cached:
            self._download(url, saved_fn=image_cached_path)
        if hash != etag(image_cached_path):
            raise RuntimeError(f"Uncorrect hash checking from url:{url}")
        
        img = Image.open(image_cached_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)
        

class SaveNamedImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                        "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                        "filename_suffix":  ("STRING", {"default": ".png"}),
                        "callback_url": ("STRING", {"default": ""}),
                        "timeout": ("INT", {"default": 5}),
                        },
                }

    RETURN_TYPES = ()
    FUNCTION = "run_it"

    OUTPUT_NODE = True

    CATEGORY = "biubiubiu/Image"

    def callback(self, url, timeout):
        try:
            requests.post(url=url, timeout=timeout)
        except Exception as e:
            print(e)

    def run_it(self, images, filename_prefix="ComfyUI", filename_suffix='.png', callback_url="", timeout=5):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))            

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

            file = f"{filename_with_batch_num}_{counter:05}_{filename_suffix}"
            file_path = os.path.join(full_output_folder, file)
            img.save(file_path)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        if callback_url != "":
            self.callback(callback_url,timeout)

        return { "ui": { "images": results } }

        
class RepeatMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("MASK",),
                              "amount": ("INT", {"default": 1, "min": 1, "max": 64}),
                              }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "repeat"

    CATEGORY = "biubiubiu/Tools"

    def repeat(self, samples, amount):
        s = samples.clone()
        s = torch.concat([s]*amount, dim=0)
        return (s,)

class RepeatBBoxDetailBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("BBOXDETAIL",),
                              "amount": ("INT", {"default": 1, "min": 1, "max": 64}),
                              }}
    RETURN_TYPES = ("BBOXDETAIL",)
    FUNCTION = "repeat"

    CATEGORY = "biubiubiu/Tools"

    def repeat(self, samples, amount):
        s = copy.deepcopy(samples)
        out = []
        for _ in range(amount):
            out.extend(s)
        return (out,)


class ColorFix:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                        "color": ("IMAGE", ),
                        "is_work": ("BOOLEAN", {"default": False})
                    },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_it"

    CATEGORY = "biubiubiu/Image"

    @staticmethod
    def color_transfer(sc, dc):
        """
        Transfer color distribution from of sc, referred to dc.
        
        Args:
            sc (numpy.ndarray): input image to be transfered.
            dc (numpy.ndarray): reference image 

        Returns:
            numpy.ndarray: Transferred color distribution on the sc.
        """

        def get_mean_and_std(img):
            x_mean, x_std = cv2.meanStdDev(img)
            x_mean = np.hstack(np.around(x_mean, 2))
            x_std = np.hstack(np.around(x_std, 2))
            return x_mean, x_std
        sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
        s_mean, s_std = get_mean_and_std(sc)
        dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
        t_mean, t_std = get_mean_and_std(dc)
        img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
        np.putmask(img_n, img_n > 255, 255)
        np.putmask(img_n, img_n < 0, 0)
        dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
        return dst

    def run_it(self, images, color, is_work):
        if not is_work:
            return (images, )
        color_image = np.uint8(color[0].cpu().numpy()*255)
        tensors = []
        for img in images:

            shift_image = self.color_transfer(np.uint8(img.cpu().numpy()*255), color_image)
            tensors.append(torch.from_numpy(shift_image)[None,...])
        tensors = torch.concat(tensors, dim=0).float()/255
        return (tensors,)
