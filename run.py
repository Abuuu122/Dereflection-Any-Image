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
    StableDiffusionPipeline
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
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight_dtype = torch.float32
model_dir = "./weights"
pretrained_model_name_or_path = "JichenHu/dereflection-any-image-v0"
pretrained_model_name_or_path2 = "stabilityai/stable-diffusion-2-1"
revision = None
variant = None
# Load the model
# normal
controlnet = ControlNetVAEModel.from_pretrained(pretrained_model_name_or_path, subfolder="controlnet", torch_dtype=weight_dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", torch_dtype=weight_dtype).to(device)
vae_2 = CustomAutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae_2", torch_dtype=weight_dtype).to(device)


# Load other components of the pipeline
vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path2, subfolder="vae", revision=revision, variant=variant
    ).to(device)

# import pdb; pdb.set_trace()
text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path2, subfolder="text_encoder", revision=revision, variant=variant
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path2,
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
        t_start=0
    ).to(device)


# Create a directory to save the results
# Parse command line arguments
parser = argparse.ArgumentParser(description="Run reflection removal on images.")
parser.add_argument("--input_dir", type=str, required=True, help="Directory for evaluation inputs.")
parser.add_argument("--result_dir", type=str, required=True, help="Directory for evaluation results.")
parser.add_argument("--concat_dir", type=str, required=True, help="Directory for concat evaluation results.")

args = parser.parse_args()

input_dir = args.input_dir
result_dir = args.result_dir
concat_dir = args.concat_dir

os.makedirs(result_dir, exist_ok=True)
os.makedirs(concat_dir, exist_ok=True)

input_files = sorted(glob(os.path.join(input_dir, "*")))

for input_file in tqdm(input_files, desc="Processing images"):
    input_image = load_image(input_file)
    
    resolution = 0
    if max(input_image.size) < 768:
        resolution = None
    result_image = pipeline(
        image=torch.tensor(np.array(input_image)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device),
        prompt="remove glass reflection",
        vae_2=vae_2,
        processing_resolution=resolution
    ).prediction[0]

    result_image = (result_image + 1) / 2
    result_image = result_image.clip(0., 1.)
    result_image = result_image * 255
    result_image = result_image.astype(np.uint8)
    result_image = Image.fromarray(result_image)

    concat_image = make_image_grid([input_image, result_image], rows=1, cols=2)

    # Save the concatenated image
    input_filename = os.path.basename(input_file)
    concat_image.save(os.path.join(concat_dir, f"{input_filename}"))
    result_image.save(os.path.join(result_dir, f"{input_filename}"))

