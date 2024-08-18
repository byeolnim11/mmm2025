from share import *
import config

import cv2
import einops
from PIL import Image
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    #return [255 - detected_map] + results
    return results

def base_canny(file_list, main_object, prompt, num_samples):
    torch.cuda.set_device(0)
    global apply_canny, model, ddim_sampler
    apply_canny = CannyDetector()
    model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./ControlNet/models/control_sd15_canny.pth', location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    
    prompt = 'a photo of ' + main_object + ', ' + prompt
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'lowres, bad anatomy, extra digit, fewer digits, worst quality, low quality'
    num_samples = num_samples
    #image_resolution = 256
    ddim_steps = 20
    guess_mode = False
    strength = 1.0
    scale = 9.0
    #seed = random_integer = random.randint(-1, 2147483647)
    seed = 42
    eta = 0.0
    low_threshold = 100
    high_threshold = 200

    for j in range(len(file_list)):
        input_image = Image.open(file_list[j])
        w,h = input_image.size
        image_resolution = int(min(w, h) / max(w, h) * 512)
        print(image_resolution)
        input_image = np.array(input_image)
        result = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)

        for i in range(len(result)):
            image = Image.fromarray(result[i], mode='RGB')
            image.save('images/result/image_' + str(j*10+i) + '.png')
