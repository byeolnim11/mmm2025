# **Enhancing Visual Details in Diffusion-based Image Generation**

This project contains implementation for our mmm2025 (under review) paper.

![fig1](image/overview.jpg)
  
## ABSTRACT

The advent of diffusion models marks significant progress in high-quality image generation, enabling diverse and realistic images from simple text descriptions. While these models accurately capture overall semantic information, they often fall short in textures and even generate details inconsistent with the input prompts. Addressing this challenge, we set a new image generation scenario that obtaining a base image via related resources or text-to-image models by given a text description and generating high-quality image that aligns with adding detailed descriptions. Under this scenario, we propose a baseline approach including three stages: text descriptions optimization, base image acquisition and targeted image regeneration. Optimizing initial textual descriptions through Language Models(LMs) can generate prompts more conducive to text-to-image generation models. Images obtained via retrieval or generation models are matched with text descriptions to select an appropriate base image. By employing ControlNet, the baseline method can regenerate images that conform to detailed descriptions while preserving the outlines of the base image. We simulate this image generation scenario with two datasets, CUB and Flower, and evaluate the efficiency of our baseline method. Furthermore, we compare the impact of different base images on the generation quality.

## Prerequisites

### Dataset

- CUB-200-2011: [Perona Lab - CUB-200-2011 (caltech.edu)](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- Flower-102: [Oxford 102 Flower Dataset (kaggle.com)](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset)

### Pretrained Models

- ControlNet: https://github.com/lllyasviel/ControlNet
- Stable Diffusion: https://github.com/runwayml/stable-diffusion.git
- Llama2: https: https://github.com/liltom-eth/llama2-webui.git

## Environment

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.

```
git clone https://github.com/byeolnim11/mmm2025.git
cd mmm2025
conda env create -f environment.yaml
conda activate enhanced-diffusion
```

## Implemention

### pipeline

We have established the pipeline for experiments. If llama2 need to be used for prompt generation, it is necessary to proceed with the following step first.

```
bash llama2-webui/backend_start.sh
```

Run

```
python main.py
```

### prompt generation

- generate template prompt:

  ```
  python prompt_gen/template_gen.py
  ```

- generate llama2 prompt:

  ```
  # start the llama2 server
  bash llama2-webui/backend_start.sh
  # generate prompts
  python prompt_gen/llama2_gen.py
  ```

### baseline image acquisition

- retrieval of images from the real world

  ```
  python t2t_retrieval.py
  ```

- generate from stable diffusion

  ```
  python stable-diffusion/prompt2img.py
  ```

### Regenerate with Controlnet

- select best baseline image

  ```
  python as.py
  ```

- generate target image using reference image and prompt

  ```
  python ControlNet/base_canny.py
  ```

## Results

<div align=center><img src=image/gen.jpg width=90%>

<div align=center><img src=image/table.jpg width=60%></div>

<div align=center><img src=image/table2.jpg width=60%></div>
