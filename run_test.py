import os
import torch
from PIL import Image
from DAI.pipeline_onestep import OneStepPipeline
from DAI.controlnetvae import ControlNetVAEModel
import numpy as np
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
    DDIMScheduler,
)
from transformers import CLIPTextModel, AutoTokenizer
from glob import glob
import json
import random
from diffusers.utils import make_image_grid, load_image
from peft import PeftModel
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from safetensors.torch import load_file

from DAI.pipeline_all import DAIPipeline
from DAI.decoder import CustomAutoencoderKL
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weight_dtype = torch.float32
model_dir = "/data_hdd2/users/yangchen/projects/SIRR/StableDelight/models/model_dual2/checkpoint-20000"
vae_2_dir = "/data_hdd2/users/yangchen/projects/SIRR/StableDelight/models/model_decoder_0.2/checkpoint-42000"
pretrained_model_name_or_path = "sjtu-deepvision/dereflection-any-image-v0"
revision = None
variant = None
# Load the model
# normal
controlnet = ControlNetVAEModel.from_pretrained(model_dir + "/controlnet", torch_dtype=weight_dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(model_dir + "/unet", torch_dtype=weight_dtype).to(device)
vae_2 = CustomAutoencoderKL.from_pretrained(vae_2_dir + "/vae_2", torch_dtype=weight_dtype).to(device)

# Load other components of the pipeline
vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
    ).to(device)

text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )
pipeline = DAIPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        scheduler=None,
        feature_extractor=None,
        t_start=0,
    ).to(device)

# Create a directory to save the results
output_dir = "/data_hdd2/users/yangchen/projects/SIRR/Dereflection-Any-Image/outputs/output"
os.makedirs(output_dir, exist_ok=True)

dataset_names = ['real', 'nature', 'sir_object', 'sir_wild', 'sir_postcard', 'drr_c', 'drr_s']
for dataset_name in dataset_names:
    print(f"Testing on {dataset_name} dataset")
    jsonl_path = f"/data_hdd2/users/yangchen/projects/SIRR/DRR_jsonl/metadata_test_{dataset_name}.jsonl"
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    input_images = [load_image(entry['conditioning_image']) for entry in data]
    gt_images = [load_image(entry['image']) for entry in data]

    for idx, (input_image, gt_image) in enumerate(tqdm(zip(input_images, gt_images), total=len(input_images), desc=f"Processing {dataset_name}")):
        # Apply the model to the image
        resolution = 0 #original resolution
        if max(input_image.size) < 768:
            resolution = None #768
        # resolution = None
        result_image = pipeline(
            image=torch.tensor(np.array(input_image)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device),
            prompt="remove glass reflection",
            vae_2=vae_2,
            processing_resolution=resolution,
        ).prediction[0]

        result_image = (result_image + 1) / 2
        result_image = result_image.clip(0., 1.)
        result_image = result_image * 255
        result_image = result_image.astype(np.uint8)
        result_image = Image.fromarray(result_image)

        concat_image = make_image_grid([input_image, result_image, gt_image], rows=1, cols=3)

        # Extract the base name of the image file
        base_name = os.path.basename(data[idx]['conditioning_image']).split('.')[0]

        # Save the concatenated image
        os.makedirs(f"{output_dir}/test/{dataset_name}/result", exist_ok=True)
        os.makedirs(f"{output_dir}/test/{dataset_name}/gt", exist_ok=True)
        os.makedirs(f"{output_dir}/test/{dataset_name}/concat", exist_ok=True)

        result_image.save(f"{output_dir}/test/{dataset_name}/result/{base_name}.png")
        gt_image.save(f"{output_dir}/test/{dataset_name}/gt/{base_name}.png")
        concat_image.save(f"{output_dir}/test/{dataset_name}/concat/{base_name}.png")
