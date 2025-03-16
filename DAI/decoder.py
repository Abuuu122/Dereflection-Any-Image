
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution, Encoder, Decoder
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.unets.unet_2d_blocks import (
    AutoencoderTinyBlock,
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)
from diffusers.utils.accelerate_utils import apply_forward_hook

class ZeroConv2d(nn.Module):
    """
    Zero Convolution layer, similar to the one used in ControlNet.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)

class CustomAutoencoderKL(AutoencoderKL):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
            mid_block_add_attention=mid_block_add_attention,
        )

        # Add Zero Convolution layers to the encoder
        # self.zero_convs = nn.ModuleList()
        # for i, out_channels_ in enumerate(block_out_channels):
        #     self.zero_convs.append(ZeroConv2d(out_channels_, out_channels_))

        # Modify the decoder to accept skip connections
        self.decoder = CustomDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.encoder = CustomEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        # Get the encoder outputs
        _, skip_connections = self.encoder(x)

        return skip_connections

    def decode(self, z: torch.Tensor, skip_connections: list, return_dict: bool = True):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        # Decode the latent representation with skip connections
        dec = self.decoder(z, skip_connections)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
    
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        # Encode the input and get the skip connections
        posterior, skip_connections = self.encode(sample, return_dict=True)

        # Sample from the posterior
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        # Decode the latent representation with skip connections
        dec = self.decode(z, skip_connections, return_dict=return_dict)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


class CustomDecoder(Decoder):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_block_types: Tuple[str, ...],
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        norm_num_groups: int,
        act_fn: str,
        mid_block_add_attention: bool,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

    def forward(
        self,
        sample: torch.Tensor,
        skip_connections: list,
        latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            # for up_block in self.up_blocks:
            #     sample = up_block(sample, latent_embeds)
            for i, up_block in enumerate(self.up_blocks):
                # Add skip connections directly
                if i < len(skip_connections):
                    skip_connection = skip_connections[-(i + 1)]
                    # import pdb; pdb.set_trace()
                    sample = sample + skip_connection
                # import pdb; pdb.set_trace() #torch.Size([1, 512, 96, 96]
                sample = up_block(sample)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

class CustomEncoder(Encoder):
    r"""
    Custom Encoder that adds Zero Convolution layers to each block's output
    to generate skip connections.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=double_z,
            mid_block_add_attention=mid_block_add_attention,
        )

        # Add Zero Convolution layers to each block's output
        self.zero_convs = nn.ModuleList()
        for i, out_channels in enumerate(block_out_channels):
            if i < 2:
                self.zero_convs.append(ZeroConv2d(out_channels, out_channels * 2))
            else:
                self.zero_convs.append(ZeroConv2d(out_channels, out_channels))

    def forward(self, sample: torch.Tensor) -> list[torch.Tensor]:
        r"""
        Forward pass of the CustomEncoder.

        Args:
            sample (`torch.Tensor`): Input tensor.

        Returns:
            `Tuple[torch.Tensor, List[torch.Tensor]]`:
                - The final latent representation.
                - A list of skip connections from each block.
        """
        skip_connections = []

        # Initial convolution
        sample = self.conv_in(sample)

        # Down blocks
        for i, (down_block, zero_conv) in enumerate(zip(self.down_blocks, self.zero_convs)):
            # import pdb; pdb.set_trace()
            sample = down_block(sample)
            if i != len(self.down_blocks) - 1:
                sample_out = nn.functional.interpolate(zero_conv(sample), scale_factor=2, mode='bilinear', align_corners=False)
            else:
                sample_out = zero_conv(sample)
            skip_connections.append(sample_out)


        # import pdb; pdb.set_trace()
        # torch.Size([1, 128, 768, 768])
        # torch.Size([1, 128, 384, 384])
        # torch.Size([1, 256, 192, 192])
        # torch.Size([1, 512, 96, 96])
        # torch.Size([1, 512, 96, 96])

        # # Middle block
        # sample = self.mid_block(sample)

        # # Post-process
        # sample = self.conv_norm_out(sample)
        # sample = self.conv_act(sample)
        # sample = self.conv_out(sample)

        return sample, skip_connections