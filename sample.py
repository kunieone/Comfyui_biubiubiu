import comfy.samplers
import comfy.utils
from nodes import common_ksampler



class KSamplerHires:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "is_hires": ("BOOLEAN", {"default": False}),
                     "upscale_method": (s.upscale_methods,),
                    "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),

                    "steps_hires": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg_hires": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name_hires": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler_hires": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise_hires": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "biubiubiu/sampling"


    def upscale(self, samples, upscale_method, scale_by):
        s = samples.copy()
        width = round(samples["samples"].shape[3] * scale_by)
        height = round(samples["samples"].shape[2] * scale_by)
        s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, "disabled")
        return (s,)

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise,
               is_hires, upscale_method, scale_by, steps_hires, cfg_hires, sampler_name_hires, scheduler_hires, denoise_hires, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        latent_image, =  common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

        if is_hires:
            latent_image,  =  self.upscale(latent_image, upscale_method=upscale_method, scale_by=scale_by)
            latent_image, = common_ksampler(model, noise_seed, steps_hires, cfg_hires, sampler_name_hires, scheduler_hires, positive, negative, latent_image, denoise=denoise_hires)
        return (latent_image, )
