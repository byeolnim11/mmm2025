from diffusers import StableDiffusionPipeline
import torch
from pytorch_lightning import seed_everything

seed_everything(42)

model_id = "model_path"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
device = 'cuda:1'
pipe = pipe.to(device)

num_images_per_prompt = 5


prompt_path = 'prompt_path'


import os
import glob

max_length = pipe.tokenizer.model_max_length

file_list = glob.glob(os.path.join(prompt_path, '*.txt'))

folders = os.listdir(os.path.join(prompt_path, 'result'))

for filename in file_list:
    index = int(filename.split('/')[-1].split('.')[0])
    # if index!=2051: continue
    # if index==1: break
    if str(index) in folders: continue
    print(index)
    save_path = os.path.join(prompt_path, os.path.join('result', str(index)))
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(prompt_path, filename)
    with open(filename, 'r') as f:
        prompt = f.readline()
        prompt = prompt[:min(800,len(prompt))]
        print(prompt)
        input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        
        negative_ids = pipe.tokenizer("", truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
        negative_ids = negative_ids.to(device)

        concat_embeds = []
        neg_embeds = []

        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)


        # image = pipe(prompt = prompt, num_images_per_prompt = num_images_per_prompt).images

        image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_images_per_prompt = num_images_per_prompt).images

        i = 0
        for img in image:
            img.save(os.path.join(save_path, str(i)+".png"))
            i+=1
    
