# Copyright 2024 Anton Obukhov, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
from __future__ import annotations

import functools
import os
import tempfile

import gradio as gr
import imageio as imageio
import numpy as np
import spaces
import torch as torch
torch.backends.cuda.matmul.allow_tf32 = True
from PIL import Image
from gradio_imageslider import ImageSlider

from pathlib import Path
import gradio
from gradio.utils import get_cache_folder

from DAI.pipeline_all import DAIPipeline

from DAI.controlnetvae import ControlNetVAEModel

from DAI.decoder import CustomAutoencoderKL

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)

from transformers import CLIPTextModel, AutoTokenizer


class Examples(gradio.helpers.Examples):
    def __init__(self, *args, directory_name=None, **kwargs):
        super().__init__(*args, **kwargs, _initiated_directly=False)
        if directory_name is not None:
            self.cached_folder = get_cache_folder() / directory_name
            self.cached_file = Path(self.cached_folder) / "log.csv"
        self.create()


def process_image_check(path_input):
    if path_input is None:
        raise gr.Error(
            "Missing image in the first pane: upload a file or use one from the gallery below."
        )

def process_image(
    pipe,
    vae_2,
    path_input,
):
    name_base, name_ext = os.path.splitext(os.path.basename(path_input))
    print(f"Processing image {name_base}{name_ext}")

    path_output_dir = tempfile.mkdtemp()
    path_out_png = os.path.join(path_output_dir, f"{name_base}_delight.png")
    input_image = Image.open(path_input)
    resolution = None

    pipe_out = pipe(
        image=input_image,
        prompt="remove glass reflection",
        vae_2=vae_2,
        processing_resolution=resolution,
    )

    processed_frame = (pipe_out.prediction.clip(-1, 1) + 1) / 2
    processed_frame = (processed_frame[0] * 255).astype(np.uint8)
    processed_frame = Image.fromarray(processed_frame)
    processed_frame.save(path_out_png)
    yield [input_image, path_out_png]


def run_demo_server(pipe, vae_2):
    process_pipe_image = spaces.GPU(functools.partial(process_image, pipe, vae_2))

    gradio_theme = gr.themes.Default()

    with gr.Blocks(
        theme=gradio_theme,
        title="DAI",
        css="""
            #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
            .tabs button.selected {
                font-size: 20px !important;
                color: crimson !important;
            }
            h1 {
                text-align: center;
                display: block;
            }
            h2 {
                text-align: center;
                display: block;
            }
            h3 {
                text-align: center;
                display: block;
            }
            .md_feedback li {
                margin-bottom: 0px !important;
            }
        """,
        head="""
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
        """,
    ) as demo:
        gr.Markdown(
            """
            # Dereflection Any Image
            <p align="center">
        """
        )

        with gr.Tabs(elem_classes=["tabs"]):
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Input Image",
                            type="filepath",
                        )
                        with gr.Row():
                            image_submit_btn = gr.Button(
                                value="Dereflection", variant="primary"
                            )
                            image_reset_btn = gr.Button(value="Reset")
                    with gr.Column():
                        image_output_slider = ImageSlider(
                            label="outputs",
                            type="filepath",
                            show_download_button=True,
                            show_share_button=True,
                            interactive=False,
                            elem_classes="slider",
                            # position=0.25,
                        )

                Examples(
                    fn=process_pipe_image,
                    examples=sorted([
                        os.path.join("files", "image", name)
                        for name in os.listdir(os.path.join("files", "image"))
                    ]),
                    inputs=[image_input],
                    outputs=[image_output_slider],
                    cache_examples=False,
                    directory_name="examples_image",
                )

        ### Image tab
        image_submit_btn.click(
            fn=process_image_check,
            inputs=image_input,
            outputs=None,
            preprocess=False,
            queue=False,
        ).success(
            fn=process_pipe_image,
            inputs=[
                image_input,
            ],
            outputs=[image_output_slider],
            concurrency_limit=1,
        )

        image_reset_btn.click(
            fn=lambda: (
                None,
                None,
                None,
            ),
            inputs=[],
            outputs=[
                image_input,
                image_output_slider,
            ],
            queue=False,
        )

        ### Server launch

        demo.queue(
            api_open=False,
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
        )


def main():
    os.system("pip freeze")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_dtype = torch.float32
    pretrained_model_name_or_path = "sjtu-deepvision/dereflection-any-image-v0"
    pretrained_model_name_or_path2 = "stabilityai/stable-diffusion-2-1"
    revision = None
    variant = None

    # Load the model
    controlnet = ControlNetVAEModel.from_pretrained(pretrained_model_name_or_path, subfolder="controlnet", torch_dtype=weight_dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", torch_dtype=weight_dtype).to(device)
    vae_2 = CustomAutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae_2", torch_dtype=weight_dtype).to(device)

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path2, subfolder="vae", revision=revision, variant=variant
    ).to(device)

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path2, subfolder="text_encoder", revision=revision, variant=variant
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path2,
        subfolder="tokenizer",
        revision=revision,
        use_fast=False,
    )
    pipe = DAIPipeline(
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

    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    run_demo_server(pipe, vae_2)


if __name__ == "__main__":
    main()
