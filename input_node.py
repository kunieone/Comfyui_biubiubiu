import random

class PipeInputN1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {   
                        "batch_size": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                        "base_step": ("INT", {"default": 25, "min": 1, "max": 35, "step": 1}),
                        "base_cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 4.5, "step": 0.1}),
                        "hires_step": ("INT", {"default": 20, "min": 1, "max": 30, "step": 1}),
                        "hires_cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.5, "step": 0.1}),
                        "hires_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3, "step": 0.1}),
                        "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                        "is_pose_control": ("BOOLEAN", {"default": True}),
                        "is_color_fix": ("BOOLEAN", {"default": False}),
                        "is_hires": ("BOOLEAN", {"default": False}),
                        "is_kps_swapper": ("BOOLEAN", {"default": False}),
                        "fusion_style_type": (["face_center", "fusion", "resize"],),
                        "is_pose_hands": ("BOOLEAN", {"default": False}),

                    },
        }
    
    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT", "FLOAT", "FLOAT", "INT", 
                    "BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING", "BOOLEAN")
    RETURN_NAMES = ("batch size", "base step", "base cfg", "hires step", "hires cfg", "hires scale", "seed", 
                    "is pose control", "is color fix", "is hires", "is kps swapper", "fusion_style_type", "is_pose_hands")
    FUNCTION = "run_it"

    CATEGORY = "biubiubiu/Input"

    def run_it(self, batch_size, base_step, base_cfg, hires_step, hires_cfg, hires_scale, seed, 
               is_pose_control, is_color_fix, is_hires, is_kps_swapper, fusion_style_type, is_pose_hands):
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        return (batch_size, base_step, base_cfg, hires_step, hires_cfg, hires_scale, seed, 
                is_pose_control, is_color_fix, is_hires, is_kps_swapper, fusion_style_type, is_pose_hands)