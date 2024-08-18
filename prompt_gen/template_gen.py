import os

key_path = "path to input text"
store_path = "path to store synthetic prompt"
with open("/dataset/cub/train_test_split.txt", "r") as f:
    l = f.readlines()

img_id = []
flag = []
for i in l:
    img_id.append(i.split()[0])
    flag.append(i.split()[1])

for i in range(len(img_id)):
    if flag[i] == "0":
        with open(os.path.join(key_path, img_id[i])+".txt", "r") as f:
            keys = f.readlines()
        keys = [k.rstrip().replace("_", " ") for k in keys]
        prompt = "a photo of " + keys[0] + ", a bird of this species with "
        for j in range(1, len(keys)):
            if keys[j].find("lack") == -1:
                if j != len(keys)-1:
                    prompt = prompt + keys[j] +", "
                else:
                    prompt = prompt + keys[j] +"."
        with open(os.path.join(store_path, img_id[i])+".txt", "w") as f:
            f.write(prompt)
