import os
import torch
import clip
import glob
from PIL import Image
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything

def t2t_cmp(data_path, prompt):
    device = "cuda:0"
    model, preprocess = clip.load("ViT-L/14", device=device)
    cos = nn.CosineSimilarity(dim=0)
    text_path = data_path + "/text"
    image_path = data_path +"/image"
    trn_text=[]
    for p in os.listdir(text_path):
        with open(os.path.join(text_path, p), "r") as f:
            l = f.readlines()
        text = ""
        for j in l:
            text = text + j.rstrip()+" "
        trn_text.append(text)
    query = clip.tokenize(prompt, truncate = True).to(device)
    trn = clip.tokenize(trn_text, truncate = True).to(device)

    with torch.no_grad():
        tst_features = model.encode_text(query)
        trn_features = model.encode_text(trn)
    best = 0
    best_j = 0
    for j in range(len(trn)):
        sim = cos(tst_features[0],trn_features[j]).item() 
        sim = (sim+1)/2
        if sim > best:
            best = sim
            best_j = j
    print(best_j)
    return os.path.join(image_path, os.listdir(image_path)[j])

def prompt2img(prompt, output_path):
    
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
    device = 'cuda:0'
    pipe = pipe.to(device)
    print(prompt)
    seed_everything(42)
    max_length = pipe.tokenizer.model_max_length
    save_path = os.path.join(output_path, 'syn_sd')
    os.makedirs(save_path, exist_ok=True)

    num_images_per_prompt = 5
    prompt = prompt

    image = pipe(prompt, num_images_per_prompt = num_images_per_prompt).images 
    i = 0
    for img in image:
        img.save(os.path.join(save_path, str(i)+".png"))
        i+=1
    return save_path

def template_gen(main_object, prompt):
    keys = prompt.split(', ')
    re_prompt = "a photo of " + main_object + ", "
    for j in range(0, len(keys)):
        if j != len(keys)-1:
            re_prompt = re_prompt + keys[j] +", "
        else:
            re_prompt = re_prompt + keys[j] +"."
    return re_prompt

def llama2_gen(main_object, prompt):
    l_prompt = "Please use less than 120 words to describe " + main_object + " with the following words: " + prompt
    endpoint = "your endpoint"
    response = requests.post(endpoint, json=data)
    re_prompt = response.json()['choices'][0]['text'][2:-1]
    return re_prompt

def t2i_cmp(prompt, file_list):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    cos = nn.CosineSimilarity(dim=0)
    tst1 = clip.tokenize(prompt, truncate = True).to(device)
    with torch.no_grad():
        tst_features = model.encode_text(tst1).float()
        best = 0
        best_j = ''
        for j in range(len(file_list)):
            image = Image.open(os.path.join(file_list[j])).convert("RGB")
            trn_img = [preprocess(image)]
            trn1 = torch.tensor(np.stack(trn_img)).to(device)
            with torch.no_grad():
                trn_features = model.encode_image(trn1).float()
            sim = cos(tst_features[0],trn_features[0]).item() 
            sim = (sim+1)/2
            #print(sim)
            if sim > best:
                best = sim
                best_j = file_list[j]
    return best_j

def t2i_eval(prompt, file_list):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    cos = nn.CosineSimilarity(dim=0)
    tst1 = clip.tokenize(prompt, truncate = True).to(device)
    with torch.no_grad():
        tst_features = model.encode_text(tst1).float()
        best = 0
        best_j = ''
        for j in range(len(file_list)):
            image = Image.open(os.path.join(file_list[j])).convert("RGB")
            trn_img = [preprocess(image)]
            trn1 = torch.tensor(np.stack(trn_img)).to(device)
            with torch.no_grad():
                trn_features = model.encode_image(trn1).float()
            sim = cos(tst_features[0],trn_features[0]).item() 
            sim = (sim+1)/2
            #print(sim)
            if sim > best:
                best = sim
                best_j = file_list[j]
    return best_j, sim*100

def i2i_eval(tgt_img, file_list):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    cos = nn.CosineSimilarity(dim=0)
    
    with torch.no_grad():
        image = Image.open(tgt_img).convert("RGB")
        tst1 = [preprocess(image)]
        tst1 = torch.tensor(np.stack(tst1)).to(device)
        tst_features = model.encode_image(tst1).float()
        best = 0
        best_j = ''
        for j in range(len(file_list)):
            image = Image.open(os.path.join(file_list[j])).convert("RGB")
            trn_img = [preprocess(image)]
            trn1 = torch.tensor(np.stack(trn_img)).to(device)
            with torch.no_grad():
                trn_features = model.encode_image(trn1).float()
            sim = cos(tst_features[0],trn_features[0]).item() 
            sim = (sim+1)/2
            #print(sim)
            if sim > best:
                best = sim
                best_j = file_list[j]
    return best_j, sim*100