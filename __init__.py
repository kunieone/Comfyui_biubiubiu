import importlib
import sys, os

from .tools import CleanPromptNode, Image_Filters, LoadUrlImage, SaveNamedImage, RepeatBBoxDetailBatch, RepeatMaskBatch, ColorFix,EmptyLatentImageLongside, FacePaste
from .segmentation import FaceSegment
from .faceskin import FaceSkinSegmentation, FaceSkinPaste
from .face_kps import FaceKeypointsSwapper
from .sample import KSamplerHires
from .input_node import PipeInputN1
from .preprocess import OpenPose_Preprocessor, PrepImageForFace, FaceCrop


dir = os.path.dirname(__file__)
files = [os.path.join(dir, n) for n in ['qiniuio.py']]
for file in files:
    name = os.path.splitext(file)[0]
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


NODE_CLASS_MAPPINGS= {
    "PROMPT_CLEAN": CleanPromptNode,
    "BImageFilter": Image_Filters,
    "FaceSegment": FaceSegment,
    "SkinSegment": FaceSkinSegmentation,
    "SkinPaste": FaceSkinPaste,
    "FaceKpsSwapper": FaceKeypointsSwapper,
    "LoadUrlImage": LoadUrlImage,
    "SaveNamedImage": SaveNamedImage,
    "RepeatBBoxDetailBatch": RepeatBBoxDetailBatch,
    "RepeatMaskBatch": RepeatMaskBatch,
    "ColorFix": ColorFix,
    "KSamplerHires": KSamplerHires, 
    "PipeInputN1": PipeInputN1,
    "PosePreprocessor": OpenPose_Preprocessor,
    "PrepImageForFace": PrepImageForFace,
    "FaceCrop": FaceCrop,
    "EmptyLatentImageLongside": EmptyLatentImageLongside,
    "FacePaste": FacePaste
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "PROMPT_CLEAN": "Clean Prompt",
    "BImageFilter": "ImageFilter",
    "FaceSegment": "Face Segment",
    "SkinSegment": "Face Skin Segment",
    "SkinPaste": "Face Skin Paste",
    "FaceKpsSwapper": "Face Keypoints Swapper",
    "LoadUrlImage": "Load Network Image ",
    "SaveNamedImage": "Save Named Image",
    "RepeatBBoxDetailBatch": "Repeat BBoxDetail Batch",
    "RepeatMaskBatch": "Repeat Mask Batch",
    "ColorFix": "Color Fix",
    "KSamplerHires": "KSampler Hires",
    "PipeInputN1": "PipeInputN1",
    "PosePreprocessor": "Pose Preprocessor",
    "PrepImageForFace": "PrepImageForFace",
    "FaceCrop": "Face Crop",
    "EmptyLatentImageLongside": "Empty Latent Longside",
    "FacePaste": "Face Paste"

}

