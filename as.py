import torch
import clip
from PIL import Image
import torch.nn as nn
import os
import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
cos = nn.CosineSimilarity(dim=0)

text_path = "path to input text"
store_path = "path to store the best baseline image"
img_path = "path to baseline image set"

with open("/dataset/cub/test.txt", "r") as f:
    tst = f.readlines()
tst = [i.rstrip() for i in tst]
with open("/dataset/cub/train.txt", "r") as f:
    trn = f.readlines()
trn = [i.rstrip() for i in trn]
with open("/dataset/cub/images.txt", "r") as f:
    path2 = f.readlines()
path2 = [i.split()[1] for i in path2]

trn_img = []
tst_text = []
for i in tst:
    with open(os.path.join(text_path, i+'.txt'), "r") as f:
        l = f.readlines()
    text = ""
    for j in l:
        text = text + j.rstrip()+", "
    tst_text.append(text)

tst1 = clip.tokenize(tst_text, truncate = True).to(device)

with torch.no_grad():
    tst_features = model.encode_text(tst1).float()

for i in range(len(tst)):
    ids = tst[i]
    best = 0
    best_j = 0
    for j in range(len(trn)):
        image = Image.open(os.path.join(img_path, path2[int(trn[j])-1])).convert("RGB")
        trn_img = [preprocess(image)]
        trn1 = torch.tensor(np.stack(trn_img)).to(device)
        with torch.no_grad():
            trn_features = model.encode_image(trn1).float()
        sim = cos(tst_features[i],trn_features[0]).item() 
        sim = (sim+1)/2
        #print(sim)
        if sim > best:
            best = sim
            best_j = j
    print(best)
    print(path2[int(trn[best_j])-1])
    with open(os.path.join(store_path, ids+'.txt'), "w") as f:
        f.write(trn[best_j])