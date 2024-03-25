import torch
import numpy as np
import math
import cv2
import copy
import PIL
from skimage import transform

import torchvision.transforms.v2 as T


def tensorToNP(image):
    out = torch.clamp(255. * image.detach().cpu(), 0, 255).to(torch.uint8)
    out = out[..., [2, 1, 0]]
    out = out.numpy()
    return out


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def extractFeatures(insightface, image, template_image, keep_nose=False):
    face_img = tensorToNP(image)
    template = tensorToNP(template_image)
    out = []

    insightface.det_model.input_size = (320,320) # reset the detection size

    template_face = insightface.get(template[0])
    if not template_face:
        raise AssertionError("Not detect face on template image")
    template_face = sorted(template_face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
    
    template_kps = template_face['kps']


    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size # TODO: hacky but seems to be working
            face = insightface.get(face_img[i])
            if face:
                face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
                kps = face['kps']
                tform = transform.SimilarityTransform()
                if keep_nose:
                    tform.estimate(kps[[0,1,3,4]], template_kps[[0,1,3,4]])
                    e_kps_4 = np.concatenate([kps[[0,1,3,4]], np.ones(shape=(4,1))], axis=1)@tform.params.T
                    e_kps_4 = e_kps_4[:, :2]
                    e_kps = copy.deepcopy(template_kps)
                    e_kps[0:2] = e_kps_4[0:2]
                    e_kps[3:5] = e_kps_4[2:4]
                else:
                    tform.estimate(kps, template_kps)
                    e_kps = np.concatenate([kps, np.ones(shape=(5,1))], axis=1)@tform.params.T
                    e_kps = e_kps[:, :2]

                # if extract_kps:
                out.append(draw_kps(template[0], e_kps))
                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
    else:
        out = torch.stack(T.ToTensor()([draw_kps(template[0], template_kps)]), dim=0).permute([0,2,3,1])
    return out


class FaceKeypointsSwapper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "faceanalysis": ("FACEANALYSIS", ),
                "image_src": ("IMAGE", ),
                "image_template": ("IMAGE", ),
                "is_swapper": ("BOOLEAN", {"default": False})
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_it"
    CATEGORY = "biubiubiu/KPS"

    def run_it(self, faceanalysis, image_src, image_template, is_swapper=False):
        if not is_swapper:
            return (image_template,)
        face_kps = extractFeatures(faceanalysis, image_src[0].unsqueeze(0), template_image=image_template, keep_nose=False)

        if face_kps is None:
            face_kps = torch.zeros_like(image_src)
            print(f"\033[33mWARNING: no face detected, unable to extract the keypoints!\033[0m")

        return (face_kps,)