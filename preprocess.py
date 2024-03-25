import comfy.model_management as model_management
import os
import json
import torch
import numpy as np
from scipy.io import loadmat
from .faceskin import BBRegression, tensorToNP, BBREGRESSOR_PARAM, tensor2pil, pil2tensor
from comfyui_controlnet_aux.utils import common_annotator_call, create_node_input_types
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as TT
import folder_paths


# BBREGRESSOR_PARAM = os.path.join(folder_paths.models_dir, "biubiubiu/BBRegressorParam_r.mat")

class OpenPose_Preprocessor:
    def __init__(self) -> None:
        from controlnet_aux.open_pose import OpenposeDetector
        self.model = OpenposeDetector.from_pretrained().to(model_management.get_torch_device())        


    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            detect_hand = ("BOOLEAN", {"default": False}),
            detect_body = ("BOOLEAN", {"default": True}),
            detect_face = ("BOOLEAN", {"default": False}),
        )

        
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "biubiubiu/Image"

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, resolution=512, **kwargs):

        # detect_hand = detect_hand == "enable"
        # detect_body = detect_body == "enable"
        # detect_face = detect_face == "enable"

        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = self.model(image, **kwargs)
            self.openpose_dicts.append(openpose_dict)
            return pose_img
        
        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)
        # del model
        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts)
        }
    
def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)
    
def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)

class PrepImageForFace:
    def __init__(self) -> None:
        self.bbreg_param = loadmat(BBREGRESSOR_PARAM)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "faceanalysis": ("FACEANALYSIS", ),
            "type_string": ("STRING", {"default": ""}),

            "interpolation": (["LANCZOS", "BICUBIC", "HAMMING", "BILINEAR", "BOX", "NEAREST"],),
            "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            "type_": (["face_center", "fusion", "resize"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_image"

    CATEGORY = "biubiubiu/Image"

    @staticmethod
    def contrast_adaptive_sharpening(image, amount):
        img = F.pad(image, pad=(1, 1, 1, 1)).cpu()

        a = img[..., :-2, :-2]
        b = img[..., :-2, 1:-1]
        c = img[..., :-2, 2:]
        d = img[..., 1:-1, :-2]
        e = img[..., 1:-1, 1:-1]
        f = img[..., 1:-1, 2:]
        g = img[..., 2:, :-2]
        h = img[..., 2:, 1:-1]
        i = img[..., 2:, 2:]
        
        # Computing contrast
        cross = (b, d, e, f, h)
        mn = min_(cross)
        mx = max_(cross)
        
        diag = (a, c, g, i)
        mn2 = min_(diag)
        mx2 = max_(diag)
        mx = mx + mx2
        mn = mn + mn2
        
        # Computing local weight
        inv_mx = torch.reciprocal(mx)
        amp = inv_mx * torch.minimum(mn, (2 - mx))

        # scaling
        amp = torch.sqrt(amp)
        w = - amp * (amount * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4*w)

        output = ((b + d + f + h)*w + e) * div
        output = torch.nan_to_num(output)
        output = output.clamp(0, 1)

        return (output)

    def get_bbox(self,image, insightface, expand_ratio=1.0):
        image = tensorToNP(image)
        face = insightface.get(image[0])
        h, w = image.shape[0], image.shape[1]
        if face.__len__() == 0:
            print('Detected no face, use all image')
            return np.array([0,0,w,h])

        face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        
        kps = face['kps']
        bbox = BBRegression(np.array(kps).reshape([1, 10]), self.bbreg_param)
        size = min(bbox[2], bbox[3])*expand_ratio
        center0 = bbox[0] + bbox[2]/2
        center1 = bbox[1] + bbox[3]/2
        if center0 < 0 or center1 < 0:
            print('Detected half face, use all image')
            return np.array([0,0,w,h])

        if center0 + size/2 > w or center0 - size/2 < 0:
            size = min(center0, w-center0)*2
            
        if center1 + size/2 > h or center1 - size/2 < 0:
            size = min(center1, h-center1)*2

        bbox = np.array([center0 - size/2, center1 - size/2, size, size]).astype(np.int32)
        

        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        bbox = np.array(bbox, dtype=np.int32)
        return bbox

    def crop_image(self, image, crop_position, interpolation="LANCZOS", size=(224,224), sharpening=0.0):
        _, oh, ow, _ = image.shape
        output = image.permute([0,3,1,2])

        crop_size = min(oh, ow)

        c1, c0 = crop_position
        if c0 - crop_size/2 < 0:
            y = 0
        else:
            if oh - c0 < crop_size/2:
                y = oh-crop_size
            else:
                y = int(c0 - crop_size/2)

        if c1 - crop_size/2 < 0:
            x = 0
        else:
            if ow - c1 < crop_size/2:
                x = ow-crop_size
            else:
                x = int(c1 - crop_size/2)
        
        x2 = x+crop_size
        y2 = y+crop_size

        # crop
        output = output[:, :, y:y2, x:x2]

        imgs = []
        for i in range(output.shape[0]):
            img = TT.ToPILImage()(output[i])
            img = img.resize(size, resample=Image.Resampling[interpolation])
            imgs.append(TT.ToTensor()(img))
        output = torch.stack(imgs, dim=0)
        imgs = None # zelous GC
        
        if sharpening > 0:
            output = self.contrast_adaptive_sharpening(output, sharpening)
        

        output = output.permute([0,2,3,1])

        return output



    def prep_image(self, image, faceanalysis, type_string="", interpolation="LANCZOS",  sharpening=0.0, type_="center"):
        if type_string == "":
            this_type = type_
        else:
            this_type = type_string
        size = (224, 224)
        imgs = []
        if this_type == "face_center":
            for i in range(image.shape[0]):
                img = image[i:i+1]
                face_bbox = self.get_bbox(img, faceanalysis)
                center_x = (face_bbox[0] + face_bbox[2])/2
                center_y = (face_bbox[1] + face_bbox[3])/2
                output_f = self.crop_image(image, (center_x, center_y), size=size, interpolation=interpolation, sharpening=sharpening)
                imgs.append(output_f)
        elif this_type == 'fusion':
            for i in range(image.shape[0]):
                img = image[i:i+1]
                face_bbox = self.get_bbox(img, faceanalysis)
                center_x = (face_bbox[0] + face_bbox[2])/2
                center_y = (face_bbox[1] + face_bbox[3])/2
                output_f = self.crop_image(image, (center_x, center_y), size=size, interpolation=interpolation, sharpening=sharpening)
                imgs.append(output_f)

                _, oh, ow, _ = img.shape
                output0 = self.crop_image(image, (0, 0), size=size, interpolation=interpolation, sharpening=sharpening)
                output1 = self.crop_image(image, (oh-1, ow-1), size=size, interpolation=interpolation, sharpening=sharpening)
                imgs.append(output0)
                imgs.append(output1)

        elif this_type == "resize":
            for i in range(image.shape[0]):
                img = image[i].permute(2, 0, 1)
                img = TT.ToPILImage()(img)
                img = img.resize(size, resample=Image.Resampling[interpolation])
                img = TT.ToTensor()(img)[None, ...]
                imgs.append(img.permute(0, 2,3,1))
        else:
            return image
        
        output = torch.concat(imgs, dim=0)

        return (output, )


class FaceCrop:
    def __init__(self):
        self.bbreg_param = loadmat(BBREGRESSOR_PARAM)


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "insightface": ("FACEANALYSIS", ),
                "expand_ratio": ("FLOAT", {"default": 1.2}),

                "resized": ("INT", {"default": 768})
            },
        }

    RETURN_TYPES = ("IMAGE", "BBOXDETAIL")
    RETURN_NAMES = ("image_croped", "bbox detail")

    FUNCTION = "run_it"

    CATEGORY = "biubiubiu/Image"

    def get_bbox(self,image, insightface, expand_ratio):
        image = tensorToNP(image)
        face = insightface.get(image)
        h, w = image.shape[0], image.shape[1]
        if face.__len__() == 0:
            print('Detected no face, use all image')
            return np.array([0,0,w,h])

        face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        
        kps = face['kps']
        bbox = BBRegression(np.array(kps).reshape([1, 10]), self.bbreg_param)
        size = min(bbox[2], bbox[3])*expand_ratio
        center0 = bbox[0] + bbox[2]/2
        center1 = bbox[1] + bbox[3]/2
        if center0 < 0 or center1 < 0:
            print('Detected half face, use all image')
            return np.array([0,0,w,h])

        if center0 + size/2 > w or center0 - size/2 < 0:
            size = min(center0, w-center0)*2
            
        if center1 + size/2 > h or center1 - size/2 < 0:
            size = min(center1, h-center1)*2

        bbox = np.array([center0 - size/2, center1 - size/2, size, size]).astype(np.int32)
        

        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        bbox = np.array(bbox, dtype=np.int32)
        return bbox

    def run_it(self, image, insightface, resized, expand_ratio):
        # tensors = []
        image_crop_list = []
        bbox_list = []
        for img in image:
            bbox = self.get_bbox(img, insightface, expand_ratio)
            img = tensor2pil(img)
            img_crop = img.crop(bbox)
            img_crop = pil2tensor(img_crop.resize((resized, resized)) if resized != -1 else img_crop)

            image_crop_list.append(img_crop)
            bbox_list.append(bbox)
        croped = torch.concat(image_crop_list, dim=0)

        return (croped, bbox_list)