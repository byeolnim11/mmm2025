# Basic text completion
import os
import requests

key_path = "path to input text"
store_path = "path to store synthetic prompt"
with open("/dataset/cub/train_test_split.txt", "r") as f:
    l = f.readlines()

img_id = []
flag = []
for i in l:
    img_id.append(i.split()[0])
    flag.append(i.split()[1])
test = []
for i in range(len(img_id)):
    if flag[i] == "0":
        with open(os.path.join(key_path, img_id[i])+".txt", "r") as f:
            keys = f.readlines()
        keys = [k.rstrip().replace("_", " ") for k in keys]
        print(keys)
        prompt = "Please use less than 120 words to describe the bird(" + keys[0] + ") with the following words: "
        for j in range(1, len(keys)):
            if keys[j].find("lack") == -1:
                if j != len(keys)-1:
                    prompt = prompt + keys[j] +", "
                else:
                    prompt = prompt + keys[j] +"."         
        data = {
          "prompt": prompt
        }
        print(prompt)
        endpoint = ""
        response = requests.post(endpoint, json=data)
        with open(os.path.join(store_path, img_id[i])+".txt", "w") as f:
            f.write(response.json()['choices'][0]['text'][2:-1])
