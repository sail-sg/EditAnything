# Based on https://raw.githubusercontent.com/okotaku/diffusers/feature/reference_only_control/examples/community/stable_diffusion_reference.py
# Inspired by: https://github.com/Mikubill/sd-webui-controlnet/discussions/1236
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import PIL.Image
import torch

from diffusers import StableDiffusionPipeline
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import PIL_INTERPOLATION, logging
import torch.nn.functional as F


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import UniPCMultistepScheduler
        >>> from diffusers.utils import load_image

        >>> input_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")

        >>> pipe = StableDiffusionReferencePipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                safety_checker=None,
                torch_dtype=torch.float16
                ).to('cuda:0')

        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe_controlnet.scheduler.config)

        >>> result_img = pipe(ref_image=input_image,
                        prompt="1girl",
                        num_inference_steps=20,
                        reference_attn=True,
                        reference_adain=True).images[0]

        >>> result_img.show()
        ```
"""


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class StableDiffusionReferencePipeline:
    # def _default_height_width(self, height, width, image):
    #     # NOTE: It is possible that a list of images have different
    #     # dimensions for each image, so just checking the first image
    #     # is not _exactly_ correct, but it is simple.
    #     while isinstance(image, list):
    #         image = image[0]

    #     if height is None:
    #         if isinstance(image, PIL.Image.Image):
    #             height = image.height
    #         elif isinstance(image, torch.Tensor):
    #             height = image.shape[2]

    #         height = (height // 8) * 8  # round down to nearest multiple of 8

    #     if width is None:
    #         if isinstance(image, PIL.Image.Image):
    #             width = image.width
    #         elif isinstance(image, torch.Tensor):
    #             width = image.shape[3]

    #         width = (width // 8) * 8  # round down to nearest multiple of 8

    #     return height, width

    def prepare_ref_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize(
                        (width, height), resample=PIL_INTERPOLATION["lanczos"]
                    )
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_ref_latents(
        self,
        refimage,
        batch_size,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        refimage = refimage.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            ref_image_latents = [
                self.vae.encode(refimage[i : i + 1]).latent_dist.sample(
                    generator=generator[i]
                )
                for i in range(batch_size)
            ]
            ref_image_latents = torch.cat(ref_image_latents, dim=0)
        else:
            ref_image_latents = self.vae.encode(refimage).latent_dist.sample(
                generator=generator
            )
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        # duplicate mask and ref_image_latents for each generation per prompt, using mps friendly method
        if ref_image_latents.shape[0] < batch_size:
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            ref_image_latents = ref_image_latents.repeat(
                batch_size // ref_image_latents.shape[0], 1, 1, 1
            )

        ref_image_latents = (
            torch.cat([ref_image_latents] * 2)
            if do_classifier_free_guidance
            else ref_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents

    def check_ref_input(self, reference_attn, reference_adain):
        assert (
            reference_attn or reference_adain
        ), "`reference_attn` or `reference_adain` must be True."

    def redefine_ref_model(
        self, model, reference_attn, reference_adain, model_type="unet"
    ):
        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if self.only_cross_attention
                    else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if self.MODE == "write":
                    if self.attention_auto_machine_weight > self.attn_weight:
                        # print("hacked_basic_transformer_inner_forward")
                        scale_ratio = (
                            (self.ref_mask.shape[2] * self.ref_mask.shape[3])
                            / norm_hidden_states.shape[1]
                        ) ** 0.5
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(norm_hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        resize_norm_hidden_states = norm_hidden_states.view(
                            norm_hidden_states.shape[0],
                            this_ref_mask.shape[2],
                            this_ref_mask.shape[3],
                            -1,
                        ).permute(0, 3, 1, 2)

                        ref_scale = 1.0
                        resize_norm_hidden_states = F.interpolate(
                            resize_norm_hidden_states,
                            scale_factor=ref_scale,
                            mode="bilinear",
                        )
                        this_ref_mask = F.interpolate(
                            this_ref_mask, scale_factor=ref_scale
                        )
                        # print("this_ref_mask",this_ref_mask.shape)

                        # this_ref_mask = this_ref_mask.view(1,-1,1)
                        this_ref_mask = this_ref_mask.repeat(
                            resize_norm_hidden_states.shape[0],
                            resize_norm_hidden_states.shape[1],
                            1,
                            1,
                        ).bool()
                        masked_norm_hidden_states = (
                            resize_norm_hidden_states[this_ref_mask]
                            .detach()
                            .clone()
                            .view(
                                resize_norm_hidden_states.shape[0],
                                resize_norm_hidden_states.shape[1],
                                -1,
                            )
                        )
                        masked_norm_hidden_states = masked_norm_hidden_states.permute(
                            0, 2, 1
                        )
                        self.bank.append(masked_norm_hidden_states)
                        # self.bank.append(norm_hidden_states.detach().clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states
                        if self.only_cross_attention
                        else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if self.MODE == "read":
                    if self.attention_auto_machine_weight > self.attn_weight:
                        # scale_ratio = ((self.ref_mask.shape[2] * self.ref_mask.shape[3])/norm_hidden_states.shape[1])**0.5
                        # print(scale_ratio)
                        # this_ref_mask = F.interpolate(self.ref_mask.to(norm_hidden_states.device), scale_factor=1/scale_ratio).view(1,1,-1)
                        # print("resized mask", this_ref_mask.shape, this_ref_mask.max(), this_ref_mask.min(), this_ref_mask.sum())
                        # ref_hidden_states = torch.cat([norm_hidden_states] + self.bank, dim=1)
                        # if attention_mask is None:
                        #     attention_mask = torch.ones(
                        #             norm_hidden_states.shape[0], norm_hidden_states.shape[1], ref_hidden_states.shape[1], dtype=norm_hidden_states.dtype, device=norm_hidden_states.device
                        #         )
                        # this_ref_mask = this_ref_mask.repeat(norm_hidden_states.shape[0], norm_hidden_states.shape[1], 1)
                        # this_ref_mask = torch.zeros(
                        #             norm_hidden_states.shape[0], norm_hidden_states.shape[1], this_ref_mask.shape[1], dtype=norm_hidden_states.dtype, device=norm_hidden_states.device
                        #         )
                        # print(attention_mask.shape, this_ref_mask.shape)
                        # attention_mask = torch.cat((attention_mask, this_ref_mask), dim=-1)
                        # print("merge", attention_mask.shape)
                        ref_hidden_states = torch.cat(
                            [norm_hidden_states] + self.bank, dim=1
                        )
                        attn_output_uc = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=ref_hidden_states,
                            # attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
                        attn_output_c = attn_output_uc.clone()
                        if self.do_classifier_free_guidance and self.style_fidelity > 0:
                            attn_output_c[self.uc_mask] = self.attn1(
                                norm_hidden_states[self.uc_mask],
                                encoder_hidden_states=norm_hidden_states[self.uc_mask],
                                **cross_attention_kwargs,
                            )
                        attn_output = (
                            self.style_fidelity * attn_output_c
                            + (1.0 - self.style_fidelity) * attn_output_uc
                        )
                        self.bank.clear()
                    else:
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states
                            if self.only_cross_attention
                            else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        def hacked_mid_forward(self, *args, **kwargs):
            eps = 1e-6
            x = self.original_forward(*args, **kwargs)
            if self.MODE == "write":
                if self.gn_auto_machine_weight >= self.gn_weight:
                    # mask var mean
                    scale_ratio = self.ref_mask.shape[2] / x.shape[2]
                    this_ref_mask = F.interpolate(
                        self.ref_mask.to(x.device), scale_factor=1 / scale_ratio
                    )
                    this_ref_mask = this_ref_mask.repeat(
                        x.shape[0], x.shape[1], 1, 1
                    ).bool()
                    masked_x = (
                        x[this_ref_mask]
                        .detach()
                        .clone()
                        .view(x.shape[0], x.shape[1], -1, 1)
                    )
                    var, mean = torch.var_mean(
                        masked_x, dim=(2, 3), keepdim=True, correction=0
                    )

                    self.mean_bank.append(mean)
                    self.var_bank.append(var)
            if self.MODE == "read":
                if (
                    self.gn_auto_machine_weight >= self.gn_weight
                    and len(self.mean_bank) > 0
                    and len(self.var_bank) > 0
                ):
                    # print("hacked_mid_forward")
                    var, mean = torch.var_mean(
                        x, dim=(2, 3), keepdim=True, correction=0
                    )
                    std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                    mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                    var_acc = sum(self.var_bank) / float(len(self.var_bank))
                    std_acc = (
                        torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                    )
                    x_uc = (((x - mean) / std) * std_acc) + mean_acc
                    x_c = x_uc.clone()
                    if self.do_classifier_free_guidance and self.style_fidelity > 0:
                        x_c[self.uc_mask] = x[self.uc_mask]
                    x = self.style_fidelity * x_c + (1.0 - self.style_fidelity) * x_uc
                self.mean_bank = []
                self.var_bank = []
            return x

        def hack_CrossAttnDownBlock2D_forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            eps = 1e-6
            # TODO(Patrick, William) - attention mask is not used
            output_states = ()

            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                hidden_states = resnet(hidden_states, temb)

                if self.MODE == "write":
                    if self.gn_auto_machine_weight >= self.gn_weight:
                        # var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        # mask var mean
                        scale_ratio = self.ref_mask.shape[2] / hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        this_ref_mask = this_ref_mask.repeat(
                            hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        ).bool()
                        masked_hidden_states = (
                            hidden_states[this_ref_mask]
                            .detach()
                            .clone()
                            .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        )
                        var, mean = torch.var_mean(
                            masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        self.mean_bank0.append(mean)
                        self.var_bank0.append(var)
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank0) > 0
                        and len(self.var_bank0) > 0
                    ):
                        # print("hacked_CrossAttnDownBlock2D_forward0")
                        var, mean = torch.var_mean(
                            hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank0[i]) / float(
                            len(self.mean_bank0[i])
                        )
                        var_acc = sum(self.var_bank0[i]) / float(len(self.var_bank0[i]))
                        std_acc = (
                            torch.maximum(var_acc, torch.zeros_like(var_acc) + eps)
                            ** 0.5
                        )
                        hidden_states_uc = (
                            ((hidden_states - mean) / std) * std_acc
                        ) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if self.do_classifier_free_guidance and self.style_fidelity > 0:
                            hidden_states_c[self.uc_mask] = hidden_states[self.uc_mask]
                        hidden_states = (
                            self.style_fidelity * hidden_states_c
                            + (1.0 - self.style_fidelity) * hidden_states_uc
                        )

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # attention_mask=attention_mask,
                    # encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if self.MODE == "write":
                    if self.gn_auto_machine_weight >= self.gn_weight:
                        # mask var mean
                        scale_ratio = self.ref_mask.shape[2] / hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        this_ref_mask = this_ref_mask.repeat(
                            hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        ).bool()
                        masked_hidden_states = (
                            hidden_states[this_ref_mask]
                            .detach()
                            .clone()
                            .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        )
                        var, mean = torch.var_mean(
                            masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        self.mean_bank.append(mean)
                        self.var_bank.append(var)
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank) > 0
                        and len(self.var_bank) > 0
                    ):
                        # print("hack_CrossAttnDownBlock2D_forward")
                        var, mean = torch.var_mean(
                            hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(
                            len(self.mean_bank[i])
                        )
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = (
                            torch.maximum(var_acc, torch.zeros_like(var_acc) + eps)
                            ** 0.5
                        )
                        hidden_states_uc = (
                            ((hidden_states - mean) / std) * std_acc
                        ) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if self.do_classifier_free_guidance and self.style_fidelity > 0:
                            hidden_states_c[self.uc_mask] = hidden_states[self.uc_mask]
                        hidden_states = (
                            self.style_fidelity * hidden_states_c
                            + (1.0 - self.style_fidelity) * hidden_states_uc
                        )

                output_states = output_states + (hidden_states,)

            if self.MODE == "read":
                self.mean_bank0 = []
                self.var_bank0 = []
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_DownBlock2D_forward(self, hidden_states, temb=None):
            eps = 1e-6

            output_states = ()

            for i, resnet in enumerate(self.resnets):
                hidden_states = resnet(hidden_states, temb)

                if self.MODE == "write":
                    if self.gn_auto_machine_weight >= self.gn_weight:
                        # var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        # mask var mean
                        scale_ratio = self.ref_mask.shape[2] / hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        this_ref_mask = this_ref_mask.repeat(
                            hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        ).bool()
                        masked_hidden_states = (
                            hidden_states[this_ref_mask]
                            .detach()
                            .clone()
                            .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        )
                        var, mean = torch.var_mean(
                            masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        self.mean_bank.append(mean)
                        self.var_bank.append(var)
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank) > 0
                        and len(self.var_bank) > 0
                    ):
                        # print("hacked_DownBlock2D_forward")
                        var, mean = torch.var_mean(
                            hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(
                            len(self.mean_bank[i])
                        )
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = (
                            torch.maximum(var_acc, torch.zeros_like(var_acc) + eps)
                            ** 0.5
                        )
                        hidden_states_uc = (
                            ((hidden_states - mean) / std) * std_acc
                        ) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if self.do_classifier_free_guidance and self.style_fidelity > 0:
                            hidden_states_c[self.uc_mask] = hidden_states[self.uc_mask]
                        hidden_states = (
                            self.style_fidelity * hidden_states_c
                            + (1.0 - self.style_fidelity) * hidden_states_uc
                        )

                output_states = output_states + (hidden_states,)

            if self.MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_CrossAttnUpBlock2D_forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            eps = 1e-6
            # TODO(Patrick, William) - attention mask is not used
            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

                if self.MODE == "write":
                    if self.gn_auto_machine_weight >= self.gn_weight:
                        # var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        # mask var mean
                        scale_ratio = self.ref_mask.shape[2] / hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        this_ref_mask = this_ref_mask.repeat(
                            hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        ).bool()
                        masked_hidden_states = (
                            hidden_states[this_ref_mask]
                            .detach()
                            .clone()
                            .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        )
                        var, mean = torch.var_mean(
                            masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        self.mean_bank0.append(mean)
                        self.var_bank0.append(var)
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank0) > 0
                        and len(self.var_bank0) > 0
                    ):
                        # print("hacked_CrossAttnUpBlock2D_forward1")
                        var, mean = torch.var_mean(
                            hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank0[i]) / float(
                            len(self.mean_bank0[i])
                        )
                        var_acc = sum(self.var_bank0[i]) / float(len(self.var_bank0[i]))
                        std_acc = (
                            torch.maximum(var_acc, torch.zeros_like(var_acc) + eps)
                            ** 0.5
                        )
                        hidden_states_uc = (
                            ((hidden_states - mean) / std) * std_acc
                        ) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if self.do_classifier_free_guidance and self.style_fidelity > 0:
                            hidden_states_c[self.uc_mask] = hidden_states[self.uc_mask]
                        hidden_states = (
                            self.style_fidelity * hidden_states_c
                            + (1.0 - self.style_fidelity) * hidden_states_uc
                        )

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # attention_mask=attention_mask,
                    # encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                if self.MODE == "write":
                    if self.gn_auto_machine_weight >= self.gn_weight:
                        # var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        # mask var mean
                        scale_ratio = self.ref_mask.shape[2] / hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        this_ref_mask = this_ref_mask.repeat(
                            hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        ).bool()
                        masked_hidden_states = (
                            hidden_states[this_ref_mask]
                            .detach()
                            .clone()
                            .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        )
                        var, mean = torch.var_mean(
                            masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        self.mean_bank.append(mean)
                        self.var_bank.append(var)
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank) > 0
                        and len(self.var_bank) > 0
                    ):
                        # print("hacked_CrossAttnUpBlock2D_forward")
                        var, mean = torch.var_mean(
                            hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(
                            len(self.mean_bank[i])
                        )
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = (
                            torch.maximum(var_acc, torch.zeros_like(var_acc) + eps)
                            ** 0.5
                        )
                        hidden_states_uc = (
                            ((hidden_states - mean) / std) * std_acc
                        ) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if self.do_classifier_free_guidance and self.style_fidelity > 0:
                            hidden_states_c[self.uc_mask] = hidden_states[self.uc_mask]
                        hidden_states = (
                            self.style_fidelity * hidden_states_c
                            + (1.0 - self.style_fidelity) * hidden_states_uc
                        )

            if self.MODE == "read":
                self.mean_bank0 = []
                self.var_bank0 = []
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        def hacked_UpBlock2D_forward(
            self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None
        ):
            eps = 1e-6
            for i, resnet in enumerate(self.resnets):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

                if self.MODE == "write":
                    if self.gn_auto_machine_weight >= self.gn_weight:
                        # var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        # mask var mean
                        scale_ratio = self.ref_mask.shape[2] / hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        this_ref_mask = this_ref_mask.repeat(
                            hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        ).bool()
                        masked_hidden_states = (
                            hidden_states[this_ref_mask]
                            .detach()
                            .clone()
                            .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        )
                        var, mean = torch.var_mean(
                            masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        self.mean_bank.append(mean)
                        self.var_bank.append(var)
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank) > 0
                        and len(self.var_bank) > 0
                    ):
                        # print("hacked_UpBlock2D_forward")
                        var, mean = torch.var_mean(
                            hidden_states, dim=(2, 3), keepdim=True, correction=0
                        )
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(
                            len(self.mean_bank[i])
                        )
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = (
                            torch.maximum(var_acc, torch.zeros_like(var_acc) + eps)
                            ** 0.5
                        )
                        hidden_states_uc = (
                            ((hidden_states - mean) / std) * std_acc
                        ) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if self.do_classifier_free_guidance and self.style_fidelity > 0:
                            hidden_states_c[self.uc_mask] = hidden_states[self.uc_mask]
                        hidden_states = (
                            self.style_fidelity * hidden_states_c
                            + (1.0 - self.style_fidelity) * hidden_states_uc
                        )

            if self.MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        if model_type == "unet":

            if reference_attn:
                attn_modules = [
                    module
                    for module in torch_dfs(model)
                    if isinstance(module, BasicTransformerBlock)
                ]
                attn_modules = sorted(
                    attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
                )

                for i, module in enumerate(attn_modules):
                    module._original_inner_forward = module.forward
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                    module.bank = []
                    module.attn_weight = float(i) / float(len(attn_modules))
                    module.attention_auto_machine_weight = (
                        self.attention_auto_machine_weight
                    )
                    module.gn_auto_machine_weight = self.gn_auto_machine_weight
                    module.do_classifier_free_guidance = (
                        self.do_classifier_free_guidance
                    )
                    module.do_classifier_free_guidance = (
                        self.do_classifier_free_guidance
                    )
                    module.uc_mask = self.uc_mask
                    module.style_fidelity = self.style_fidelity
                    module.ref_mask = self.ref_mask
            else:
                attn_modules = None
            if reference_adain:
                gn_modules = [model.mid_block]
                model.mid_block.gn_weight = 0

                down_blocks = model.down_blocks
                for w, module in enumerate(down_blocks):
                    module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                    gn_modules.append(module)
                    # print(module.__class__.__name__,module.gn_weight)

                up_blocks = model.up_blocks
                for w, module in enumerate(up_blocks):
                    module.gn_weight = float(w) / float(len(up_blocks))
                    gn_modules.append(module)
                    # print(module.__class__.__name__,module.gn_weight)

                for i, module in enumerate(gn_modules):
                    if getattr(module, "original_forward", None) is None:
                        module.original_forward = module.forward
                    if i == 0:
                        # mid_block
                        module.forward = hacked_mid_forward.__get__(
                            module, torch.nn.Module
                        )
                    elif isinstance(module, CrossAttnDownBlock2D):
                        module.forward = hack_CrossAttnDownBlock2D_forward.__get__(
                            module, CrossAttnDownBlock2D
                        )
                        module.mean_bank0 = []
                        module.var_bank0 = []
                    elif isinstance(module, DownBlock2D):
                        module.forward = hacked_DownBlock2D_forward.__get__(
                            module, DownBlock2D
                        )
                    # elif isinstance(module, CrossAttnUpBlock2D):
                    #     module.forward = hacked_CrossAttnUpBlock2D_forward.__get__(module, CrossAttnUpBlock2D)
                    #     module.mean_bank0 = []
                    #     module.var_bank0 = []
                    elif isinstance(module, UpBlock2D):
                        module.forward = hacked_UpBlock2D_forward.__get__(
                            module, UpBlock2D
                        )
                        module.mean_bank0 = []
                        module.var_bank0 = []
                    module.mean_bank = []
                    module.var_bank = []
                    module.attention_auto_machine_weight = (
                        self.attention_auto_machine_weight
                    )
                    module.gn_auto_machine_weight = self.gn_auto_machine_weight
                    module.do_classifier_free_guidance = (
                        self.do_classifier_free_guidance
                    )
                    module.do_classifier_free_guidance = (
                        self.do_classifier_free_guidance
                    )
                    module.uc_mask = self.uc_mask
                    module.style_fidelity = self.style_fidelity
                    module.ref_mask = self.ref_mask
            else:
                gn_modules = None
        elif model_type == "controlnet":
            model = model.nets[-1]  # only hack the inpainting controlnet
            if reference_attn:
                attn_modules = [
                    module
                    for module in torch_dfs(model)
                    if isinstance(module, BasicTransformerBlock)
                ]
                attn_modules = sorted(
                    attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
                )
                for i, module in enumerate(attn_modules):
                    module._original_inner_forward = module.forward
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                    module.bank = []
                    module.attn_weight = 0.0  # float(i) / float(len(attn_modules))
                    module.attention_auto_machine_weight = (
                        self.attention_auto_machine_weight
                    )
                    module.gn_auto_machine_weight = self.gn_auto_machine_weight
                    module.do_classifier_free_guidance = (
                        self.do_classifier_free_guidance
                    )
                    module.do_classifier_free_guidance = (
                        self.do_classifier_free_guidance
                    )
                    module.uc_mask = self.uc_mask
                    module.style_fidelity = self.style_fidelity
                    module.ref_mask = self.ref_mask
            else:
                attn_modules = None
            gn_modules = None

        return attn_modules, gn_modules

    def change_module_mode(self, mode, attn_modules, gn_modules):
        if attn_modules is not None:
            for i, module in enumerate(attn_modules):
                module.MODE = mode
        if gn_modules is not None:
            for i, module in enumerate(gn_modules):
                module.MODE = mode