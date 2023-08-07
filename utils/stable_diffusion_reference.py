# Based on https://raw.githubusercontent.com/okotaku/diffusers/feature/reference_only_control/examples/community/stable_diffusion_reference.py
# Inspired by: https://github.com/Mikubill/sd-webui-controlnet/discussions/1236
import torch.fft as fft
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


def memory_usage(func):
    def wrapper(*args, **kwargs):
        before_mem = torch.cuda.memory_allocated()
        result = func(*args, **kwargs)
        after_mem = torch.cuda.memory_allocated()
        mem_diff = after_mem - before_mem
        print(
            f"Memory used during function '{func.__name__}': {mem_diff} bytes")
        return result

    return wrapper


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

# def add_freq_feature(feature1, feature2, num_freq_components):
#     """
#     将一个特征的高频成分添加到另一个特征上。

#     参数:
#         - feature1: 第一个特征 (torch.Tensor)
#         - feature2: 第二个特征 (torch.Tensor)
#         - num_freq_components: 保留的频率分量数量 (int)

#     返回值:
#         - mixed_feature: 混合后的特征 (torch.Tensor)
#     """
#     # 对两个特征进行二维傅里叶变换
#     dtype = feature2.dtype
#     # print("feature1", feature1.shape)
#     # print("feature2", feature2.shape)
#     # 转换特征的数据类型为指定类型
#     feature1 = feature1.to(torch.float32)
#     feature2 = feature2.to(torch.float32)

#     # 对两个特征进行二维傅里叶变换
#     spectrum1 = fft.fftn(feature1, dim=(-2, -1))
#     spectrum2 = fft.fftn(feature2, dim=(-2, -1))
#     # print("spectrum1", spectrum1.shape)
#     # print("spectrum2", spectrum2.shape)

#     # 提取feature1的高频成分的幅值
#     # high_freq_magnitude1 = torch.abs(spectrum1[..., :num_freq_components, :])
#     high_freq_magnitude1 = torch.abs(spectrum1)
#     phase1 = torch.angle(spectrum1)

#     # 提取feature2的幅值和相位信息
#     magnitude2 = torch.abs(spectrum2)
#     phase2 = torch.angle(spectrum2)

#     # 将feature1的高频幅值成分添加到feature2的幅值上
#     mixed_magnitude = magnitude2.clone()
#     # print("mixed_magnitude", mixed_magnitude.shape)
#     # mixed_magnitude[..., :num_freq_components, :] = mixed_magnitude[..., :num_freq_components, :]*0.0 + high_freq_magnitude1 * 1.0
#     mixed_magnitude = mixed_magnitude*0.75 + high_freq_magnitude1 * 0.25

#     mixed_phase = phase2.clone()
#     mixed_phase = mixed_phase*0.75 + phase1 * 0.25

#     # 合并幅值和相位信息
#     mixed_spectrum = torch.polar(mixed_magnitude, mixed_phase)

#     # 对混合后的频域进行逆傅里叶变换得到混合后的特征
#     mixed_feature = fft.ifftn(mixed_spectrum, dim=(-2, -1))

#     return mixed_feature.to(dtype)
# @memory_usage


@torch.no_grad()
def add_freq_feature(feature1, feature2, ref_ratio):
    # Convert features to float32 (if not already) for compatibility with fft operations
    data_type = feature2.dtype
    feature1 = feature1.to(torch.float32)
    feature2 = feature2.to(torch.float32)

    # Compute the Fourier transforms of both features
    spectrum1 = fft.fftn(feature1, dim=(-2, -1))
    spectrum2 = fft.fftn(feature2, dim=(-2, -1))

    # Extract high-frequency magnitude and phase from feature1
    high_freq_magnitude1 = torch.abs(spectrum1)
    phase1 = torch.angle(spectrum1)

    # Extract magnitude and phase from feature2
    magnitude2 = torch.abs(spectrum2)
    phase2 = torch.angle(spectrum2)

    # Add high-frequency magnitude and phase to feature2
    # magnitude2.mul_(1-ref_ratio).add_(high_freq_magnitude1 * ref_ratio)
    # phase2.mul_(1-ref_ratio).add_(phase1 * ref_ratio)
    # magnitude2.mul_(1.0).add_(high_freq_magnitude1 * 0.0)
    # phase2.mul_(1-ref_ratio).add_(phase1 * ref_ratio)
    magnitude2.mul_(0.5).add_(high_freq_magnitude1 * 0.5)
    phase2.mul_(1.0).add_(phase1 * 0.0)

    # Combine magnitude and phase information
    mixed_spectrum = torch.polar(magnitude2, phase2)

    # Compute the inverse Fourier transform to get the mixed feature
    mixed_feature = fft.ifftn(mixed_spectrum, dim=(-2, -1))

    del feature1, feature2, spectrum1, spectrum2, high_freq_magnitude1, phase1, magnitude2, phase2, mixed_spectrum

    # Convert back to the original data type and return the result
    return mixed_feature.to(data_type)


@torch.no_grad()
def save_ref_feature(feature, mask):
    """
    feature: n,c,h,w
    mask: n,1,h,w

    return n,c,h,w
    """
    return feature * mask


@torch.no_grad()
def mix_ref_feature(feature, ref_fea_bank, cfg=True, dim3=False):
    """
    feature: n,l,c or n,c,h,w
    ref_fea_bank: [(n,c,h,w)]
    cfg: True/False

    return n,l,c or n,c,h,w
    """
    if cfg:
        ref_fea = torch.cat(
            (ref_fea_bank+ref_fea_bank), dim=0)
    else:
        ref_fea = ref_fea_bank
    
    if dim3:
        feature = feature.permute(0, 2, 1).view(ref_fea.shape)

    mixed_feature = add_freq_feature(ref_fea, feature, 1.0)
    
    if dim3:
        mixed_feature = mixed_feature.view(
        ref_fea.shape[0], ref_fea.shape[1], -1).permute(0, 2, 1)

    del ref_fea
    del feature
    return mixed_feature


class StableDiffusionReferencePipeline:
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
                self.vae.encode(refimage[i: i + 1]).latent_dist.sample(
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
        # @memory_usage
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
                        self.fea_bank.append(save_ref_feature(
                            resize_norm_hidden_states, this_ref_mask))
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
                        # print("write", masked_norm_hidden_states.shape)
                        self.bank.append(masked_norm_hidden_states)
                        del masked_norm_hidden_states
                        del this_ref_mask
                        del resize_norm_hidden_states
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
                        freq_norm_hidden_states = mix_ref_feature(
                            norm_hidden_states, self.fea_bank, cfg=self.do_classifier_free_guidance, dim3=True)
                        self.fea_bank.clear()

                        this_bank = torch.cat(self.bank+self.bank, dim=0)
                        ref_hidden_states = torch.cat(
                            (freq_norm_hidden_states, this_bank), dim=1
                        )
                        # ref_hidden_states = this_bank
                        # ref_hidden_states = freq_norm_hidden_states
                        del this_bank
                        self.bank.clear()

                        attn_output_uc = self.attn1(
                            # norm_hidden_states,
                            freq_norm_hidden_states,
                            encoder_hidden_states=ref_hidden_states,
                            # attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
                        del ref_hidden_states
                        attn_output_c = attn_output_uc.clone()

                        if self.style_fidelity <= 0 and self.ref_scale <= 0:
                            # if self.style_fidelity <= 0 and tmp_ref_scale <= 0:
                            attn_output_ori = attn_output_c
                        else:
                            attn_output_ori = self.attn1(
                                norm_hidden_states,
                                encoder_hidden_states=norm_hidden_states,
                                **cross_attention_kwargs,
                            )
                        if self.do_classifier_free_guidance and self.style_fidelity > 0:
                            # attn_output_c[self.uc_mask] = self.attn1(
                            #     norm_hidden_states[self.uc_mask],
                            #     encoder_hidden_states=norm_hidden_states[self.uc_mask],
                            #     **cross_attention_kwargs,
                            # )
                            attn_output_c[self.uc_mask] = attn_output_ori[self.uc_mask]

                        attn_output = (
                            self.style_fidelity * attn_output_c
                            + (1.0 - self.style_fidelity) * attn_output_uc
                        )

                        attn_output = attn_output + self.ref_scale * \
                            (attn_output-attn_output_ori)
                        self.bank.clear()
                        self.fea_bank.clear()
                        del attn_output_c
                        del attn_output_uc
                        del attn_output_ori
                    else:
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states
                            if self.only_cross_attention
                            else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
                    self.bank.clear()
                    self.fea_bank.clear()

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
                    norm_hidden_states *
                    (1 + scale_mlp[:, None]) + shift_mlp[:, None]
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

                    self.fea_bank.append(save_ref_feature(
                        x, this_ref_mask))

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

                    self.mean_bank.append(torch.cat([mean]*2, dim=0))
                    self.var_bank.append(torch.cat([var]*2, dim=0))
            if self.MODE == "read":
                if (
                    self.gn_auto_machine_weight >= self.gn_weight
                    and len(self.mean_bank) > 0
                    and len(self.var_bank) > 0
                ):
                    # print("hacked_mid_forward")
                    x = mix_ref_feature(
                        x, self.fea_bank, cfg=self.do_classifier_free_guidance)
                    # scale_ratio = self.inpaint_mask.shape[2] / x.shape[2]
                    # this_inpaint_mask = F.interpolate(
                    #     self.inpaint_mask.to(x.device), scale_factor=1 / scale_ratio
                    # )
                    # this_inpaint_mask = this_inpaint_mask.repeat(
                    #     x.shape[0], x.shape[1], 1, 1
                    # ).bool()
                    # masked_x = (
                    #     x[this_inpaint_mask]
                    #     .detach()
                    #     .clone()
                    #     .view(x.shape[0], x.shape[1], -1, 1)
                    # )
                    # var, mean = torch.var_mean(
                    #     masked_x, dim=(2, 3), keepdim=True, correction=0
                    # )
                    # std = torch.maximum(
                    #     var, torch.zeros_like(var) + eps) ** 0.5
                    # mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                    # var_acc = sum(self.var_bank) / float(len(self.var_bank))
                    # std_acc = (
                    #     torch.maximum(var_acc, torch.zeros_like(
                    #         var_acc) + eps) ** 0.5
                    # )
                    # x_uc = (((masked_x - mean) / std) * std_acc) + mean_acc

                    # x_c = x_uc.clone()
                    # if self.do_classifier_free_guidance and self.style_fidelity > 0:
                    #     x_c[self.uc_mask] = masked_x[self.uc_mask]
                    # x_c = self.style_fidelity * x_c + \
                    #     (1.0 - self.style_fidelity) * x_uc

                    # x_c = x_c + self.ref_scale * (x_c-masked_x)
                    # x[this_inpaint_mask] = x_c.view(-1)
                self.mean_bank = []
                self.var_bank = []
                self.fea_bank = []
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
                        scale_ratio = self.ref_mask.shape[2] / \
                            hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        self.fea_bank0.append(save_ref_feature(
                            hidden_states, this_ref_mask))
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
                        self.mean_bank0.append(torch.cat([mean]*2, dim=0))
                        self.var_bank0.append(torch.cat([var]*2, dim=0))
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank0) > 0
                        and len(self.var_bank0) > 0
                    ):
                        # print("hacked_CrossAttnDownBlock2D_forward0")
                        hidden_states = mix_ref_feature(
                            hidden_states, [self.fea_bank0[i]], cfg=self.do_classifier_free_guidance)
                        # scale_ratio = self.inpaint_mask.shape[2] / \
                        #     hidden_states.shape[2]
                        # this_inpaint_mask = F.interpolate(
                        #     self.inpaint_mask.to(hidden_states.device), scale_factor=1 / scale_ratio
                        # )
                        # this_inpaint_mask = this_inpaint_mask.repeat(
                        #     hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        # ).bool()
                        # masked_hidden_states = (
                        #     hidden_states[this_inpaint_mask]
                        #     .detach()
                        #     .clone()
                        #     .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        # )
                        # var, mean = torch.var_mean(
                        #     masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        # )
                        # std = torch.maximum(
                        #     var, torch.zeros_like(var) + eps) ** 0.5
                        # mean_acc = sum(self.mean_bank0[i]) / float(
                        #     len(self.mean_bank0[i])
                        # )
                        # var_acc = sum(
                        #     self.var_bank0[i]) / float(len(self.var_bank0[i]))
                        # std_acc = (
                        #     torch.maximum(
                        #         var_acc, torch.zeros_like(var_acc) + eps)
                        #     ** 0.5
                        # )
                        # hidden_states_uc = (
                        #     ((masked_hidden_states - mean) / std) * std_acc
                        # ) + mean_acc
                        # hidden_states_c = hidden_states_uc.clone()
                        # if self.do_classifier_free_guidance and self.style_fidelity > 0:
                        #     hidden_states_c[self.uc_mask] = masked_hidden_states[self.uc_mask]
                        # hidden_states_c = (
                        #     self.style_fidelity * hidden_states_c
                        #     + (1.0 - self.style_fidelity) * hidden_states_uc
                        # )

                        # hidden_states_c = hidden_states_c + self.ref_scale * \
                        #     (hidden_states_c-masked_hidden_states)
                        # hidden_states[this_inpaint_mask] = hidden_states_c.view(
                        #     -1)

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
                        scale_ratio = self.ref_mask.shape[2] / \
                            hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        self.fea_bank.append(save_ref_feature(
                            hidden_states, this_ref_mask))
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
                        self.mean_bank.append(torch.cat([mean]*2, dim=0))
                        self.var_bank.append(torch.cat([var]*2, dim=0))
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank) > 0
                        and len(self.var_bank) > 0
                    ):
                        # print("hack_CrossAttnDownBlock2D_forward")
                        hidden_states = mix_ref_feature(
                            hidden_states, [self.fea_bank[i]], cfg=self.do_classifier_free_guidance)
                        # scale_ratio = self.inpaint_mask.shape[2] / \
                        #     hidden_states.shape[2]
                        # this_inpaint_mask = F.interpolate(
                        #     self.inpaint_mask.to(hidden_states.device), scale_factor=1 / scale_ratio
                        # )
                        # this_inpaint_mask = this_inpaint_mask.repeat(
                        #     hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        # ).bool()
                        # masked_hidden_states = (
                        #     hidden_states[this_inpaint_mask]
                        #     .detach()
                        #     .clone()
                        #     .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        # )
                        # var, mean = torch.var_mean(
                        #     masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        # )
                        # std = torch.maximum(
                        #     var, torch.zeros_like(var) + eps) ** 0.5
                        # mean_acc = sum(self.mean_bank[i]) / float(
                        #     len(self.mean_bank[i])
                        # )
                        # var_acc = sum(
                        #     self.var_bank[i]) / float(len(self.var_bank[i]))
                        # std_acc = (
                        #     torch.maximum(
                        #         var_acc, torch.zeros_like(var_acc) + eps)
                        #     ** 0.5
                        # )
                        # hidden_states_uc = (
                        #     ((masked_hidden_states - mean) / std) * std_acc
                        # ) + mean_acc
                        # hidden_states_c = hidden_states_uc.clone()
                        # if self.do_classifier_free_guidance and self.style_fidelity > 0:
                        #     hidden_states_c[self.uc_mask] = masked_hidden_states[self.uc_mask]
                        # hidden_states_c = (
                        #     self.style_fidelity * hidden_states_c
                        #     + (1.0 - self.style_fidelity) * hidden_states_uc
                        # )

                        # hidden_states_c = hidden_states_c + self.ref_scale * \
                        #     (hidden_states_c-masked_hidden_states)
                        # hidden_states[this_inpaint_mask] = hidden_states_c.view(
                        #     -1)

                output_states = output_states + (hidden_states,)

            if self.MODE == "read":
                self.mean_bank0 = []
                self.var_bank0 = []
                self.mean_bank = []
                self.var_bank = []
                self.fea_bank0 = []
                self.fea_bank = []

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
                        scale_ratio = self.ref_mask.shape[2] / \
                            hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        self.fea_bank.append(save_ref_feature(
                            hidden_states, this_ref_mask))
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
                        self.mean_bank.append(torch.cat([mean]*2, dim=0))
                        self.var_bank.append(torch.cat([var]*2, dim=0))
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank) > 0
                        and len(self.var_bank) > 0
                    ):
                        # print("hacked_DownBlock2D_forward")
                        hidden_states = mix_ref_feature(
                            hidden_states, [self.fea_bank[i]], cfg=self.do_classifier_free_guidance)
                        # scale_ratio = self.inpaint_mask.shape[2] / \
                        #     hidden_states.shape[2]
                        # this_inpaint_mask = F.interpolate(
                        #     self.inpaint_mask.to(hidden_states.device), scale_factor=1 / scale_ratio
                        # )
                        # this_inpaint_mask = this_inpaint_mask.repeat(
                        #     hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        # ).bool()
                        # masked_hidden_states = (
                        #     hidden_states[this_inpaint_mask]
                        #     .detach()
                        #     .clone()
                        #     .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        # )
                        # var, mean = torch.var_mean(
                        #     masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        # )
                        # std = torch.maximum(
                        #     var, torch.zeros_like(var) + eps) ** 0.5
                        # mean_acc = sum(self.mean_bank[i]) / float(
                        #     len(self.mean_bank[i])
                        # )
                        # var_acc = sum(
                        #     self.var_bank[i]) / float(len(self.var_bank[i]))
                        # std_acc = (
                        #     torch.maximum(
                        #         var_acc, torch.zeros_like(var_acc) + eps)
                        #     ** 0.5
                        # )
                        # hidden_states_uc = (
                        #     ((masked_hidden_states - mean) / std) * std_acc
                        # ) + mean_acc
                        # hidden_states_c = hidden_states_uc.clone()
                        # if self.do_classifier_free_guidance and self.style_fidelity > 0:
                        #     hidden_states_c[self.uc_mask] = masked_hidden_states[self.uc_mask]
                        # hidden_states_c = (
                        #     self.style_fidelity * hidden_states_c
                        #     + (1.0 - self.style_fidelity) * hidden_states_uc
                        # )

                        # hidden_states_c = hidden_states_c + self.ref_scale * \
                        #     (hidden_states_c-masked_hidden_states)
                        # hidden_states[this_inpaint_mask] = hidden_states_c.view(
                        #     -1)

                output_states = output_states + (hidden_states,)

            if self.MODE == "read":
                self.mean_bank = []
                self.var_bank = []
                self.fea_bank = []

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
                hidden_states = torch.cat(
                    [hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

                if self.MODE == "write":
                    if self.gn_auto_machine_weight >= self.gn_weight:
                        # var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        # mask var mean
                        scale_ratio = self.ref_mask.shape[2] / \
                            hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        self.fea_bank0.append(save_ref_feature(
                            hidden_states, this_ref_mask))
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
                        self.mean_bank0.append(torch.cat([mean]*2, dim=0))
                        self.var_bank0.append(torch.cat([var]*2, dim=0))
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank0) > 0
                        and len(self.var_bank0) > 0
                    ):
                        # print("hacked_CrossAttnUpBlock2D_forward1")
                        hidden_states = mix_ref_feature(
                            hidden_states, [self.fea_bank0[i]], cfg=self.do_classifier_free_guidance)
                        # scale_ratio = self.inpaint_mask.shape[2] / \
                        #     hidden_states.shape[2]
                        # this_inpaint_mask = F.interpolate(
                        #     self.inpaint_mask.to(hidden_states.device), scale_factor=1 / scale_ratio
                        # )
                        # self.fea_bank.append(save_ref_feature(
                        #     hidden_states, this_ref_mask))
                        # this_inpaint_mask = this_inpaint_mask.repeat(
                        #     hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        # ).bool()
                        # masked_hidden_states = (
                        #     hidden_states[this_inpaint_mask]
                        #     .detach()
                        #     .clone()
                        #     .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        # )
                        # var, mean = torch.var_mean(
                        #     masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        # )
                        # std = torch.maximum(
                        #     var, torch.zeros_like(var) + eps) ** 0.5
                        # mean_acc = sum(self.mean_bank0[i]) / float(
                        #     len(self.mean_bank0[i])
                        # )
                        # var_acc = sum(
                        #     self.var_bank0[i]) / float(len(self.var_bank0[i]))
                        # std_acc = (
                        #     torch.maximum(
                        #         var_acc, torch.zeros_like(var_acc) + eps)
                        #     ** 0.5
                        # )
                        # hidden_states_uc = (
                        #     ((masked_hidden_states - mean) / std) * std_acc
                        # ) + mean_acc
                        # hidden_states_c = hidden_states_uc.clone()
                        # if self.do_classifier_free_guidance and self.style_fidelity > 0:
                        #     hidden_states_c[self.uc_mask] = masked_hidden_states[self.uc_mask]
                        # hidden_states_c = (
                        #     self.style_fidelity * hidden_states_c
                        #     + (1.0 - self.style_fidelity) * hidden_states_uc
                        # )

                        # hidden_states_c = hidden_states_c + self.ref_scale * \
                        #     (hidden_states_c-masked_hidden_states)
                        # hidden_states[this_inpaint_mask] = hidden_states_c.view(
                        #     -1)

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
                        scale_ratio = self.ref_mask.shape[2] / \
                            hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        self.fea_bank.append(save_ref_feature(
                            hidden_states, this_ref_mask))
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
                        self.mean_bank.append(torch.cat([mean]*2, dim=0))
                        self.var_bank.append(torch.cat([var]*2, dim=0))
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank) > 0
                        and len(self.var_bank) > 0
                    ):
                        # print("hacked_CrossAttnUpBlock2D_forward")
                        hidden_states = mix_ref_feature(
                            hidden_states, [self.fea_bank[i]], cfg=self.do_classifier_free_guidance)
                        # scale_ratio = self.inpaint_mask.shape[2] / \
                        #     hidden_states.shape[2]
                        # this_inpaint_mask = F.interpolate(
                        #     self.inpaint_mask.to(hidden_states.device), scale_factor=1 / scale_ratio
                        # )
                        # this_inpaint_mask = this_inpaint_mask.repeat(
                        #     hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        # ).bool()
                        # masked_hidden_states = (
                        #     hidden_states[this_inpaint_mask]
                        #     .detach()
                        #     .clone()
                        #     .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        # )
                        # var, mean = torch.var_mean(
                        #     masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        # )
                        # std = torch.maximum(
                        #     var, torch.zeros_like(var) + eps) ** 0.5
                        # mean_acc = sum(self.mean_bank[i]) / float(
                        #     len(self.mean_bank[i])
                        # )
                        # var_acc = sum(
                        #     self.var_bank[i]) / float(len(self.var_bank[i]))
                        # std_acc = (
                        #     torch.maximum(
                        #         var_acc, torch.zeros_like(var_acc) + eps)
                        #     ** 0.5
                        # )
                        # hidden_states_uc = (
                        #     ((masked_hidden_states - mean) / std) * std_acc
                        # ) + mean_acc
                        # hidden_states_c = hidden_states_uc.clone()
                        # if self.do_classifier_free_guidance and self.style_fidelity > 0:
                        #     hidden_states_c[self.uc_mask] = masked_hidden_states[self.uc_mask]
                        # hidden_states_c = (
                        #     self.style_fidelity * hidden_states_c
                        #     + (1.0 - self.style_fidelity) * hidden_states_uc
                        # )

                        # hidden_states_c = hidden_states_c + self.ref_scale * \
                        #     (hidden_states_c-masked_hidden_states)
                        # hidden_states[this_inpaint_mask] = hidden_states_c.view(
                        #     -1)

            if self.MODE == "read":
                self.mean_bank0 = []
                self.var_bank0 = []
                self.mean_bank = []
                self.var_bank = []
                self.fea_bank = []
                self.fea_bank0 = []

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
                hidden_states = torch.cat(
                    [hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

                if self.MODE == "write":
                    if self.gn_auto_machine_weight >= self.gn_weight:
                        # var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        # mask var mean
                        scale_ratio = self.ref_mask.shape[2] / \
                            hidden_states.shape[2]
                        this_ref_mask = F.interpolate(
                            self.ref_mask.to(hidden_states.device),
                            scale_factor=1 / scale_ratio,
                        )
                        self.fea_bank.append(save_ref_feature(
                            hidden_states, this_ref_mask))
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
                        self.mean_bank.append(torch.cat([mean]*2, dim=0))
                        self.var_bank.append(torch.cat([var]*2, dim=0))
                if self.MODE == "read":
                    if (
                        self.gn_auto_machine_weight >= self.gn_weight
                        and len(self.mean_bank) > 0
                        and len(self.var_bank) > 0
                    ):
                        # print("hacked_UpBlock2D_forward")
                        hidden_states = mix_ref_feature(
                            hidden_states, [self.fea_bank[i]], cfg=self.do_classifier_free_guidance)
                        # scale_ratio = self.inpaint_mask.shape[2] / \
                        #     hidden_states.shape[2]
                        # this_inpaint_mask = F.interpolate(
                        #     self.inpaint_mask.to(hidden_states.device), scale_factor=1 / scale_ratio
                        # )
                        # this_inpaint_mask = this_inpaint_mask.repeat(
                        #     hidden_states.shape[0], hidden_states.shape[1], 1, 1
                        # ).bool()
                        # masked_hidden_states = (
                        #     hidden_states[this_inpaint_mask]
                        #     .detach()
                        #     .clone()
                        #     .view(hidden_states.shape[0], hidden_states.shape[1], -1, 1)
                        # )
                        # var, mean = torch.var_mean(
                        #     masked_hidden_states, dim=(2, 3), keepdim=True, correction=0
                        # )
                        # std = torch.maximum(
                        #     var, torch.zeros_like(var) + eps) ** 0.5
                        # mean_acc = sum(self.mean_bank[i]) / float(
                        #     len(self.mean_bank[i])
                        # )
                        # var_acc = sum(
                        #     self.var_bank[i]) / float(len(self.var_bank[i]))
                        # std_acc = (
                        #     torch.maximum(
                        #         var_acc, torch.zeros_like(var_acc) + eps)
                        #     ** 0.5
                        # )
                        # hidden_states_uc = (
                        #     ((masked_hidden_states - mean) / std) * std_acc
                        # ) + mean_acc
                        # hidden_states_c = hidden_states_uc.clone()
                        # if self.do_classifier_free_guidance and self.style_fidelity > 0:
                        #     hidden_states_c[self.uc_mask] = masked_hidden_states[self.uc_mask]
                        # hidden_states_c = (
                        #     self.style_fidelity * hidden_states_c
                        #     + (1.0 - self.style_fidelity) * hidden_states_uc
                        # )

                        # hidden_states_c = hidden_states_c + self.ref_scale * \
                        #     (hidden_states_c-masked_hidden_states)
                        # hidden_states[this_inpaint_mask] = hidden_states_c.view(
                        #     -1)

            if self.MODE == "read":
                self.mean_bank = []
                self.var_bank = []
                self.fea_bank = []

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
                    module.fea_bank = []
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
                    module.ref_scale = self.ref_scale
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
                        module.fea_bank0 = []
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
                        module.fea_bank0 = []
                    module.mean_bank = []
                    module.var_bank = []
                    module.fea_bank = []
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
                    module.inpaint_mask = self.inpaint_mask
                    module.ref_scale = self.ref_scale
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
                    module.fea_bank = []
                    # float(i) / float(len(attn_modules))
                    module.attn_weight = 0.0
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
                    module.ref_scale = self.ref_scale
            else:
                attn_modules = None
            gn_modules = None

        return attn_modules, gn_modules

    def change_module_mode(self, mode, attn_modules, gn_modules, step=None):
        if attn_modules is not None:
            for i, module in enumerate(attn_modules):
                module.MODE = mode
                # module.step = step
        if gn_modules is not None:
            for i, module in enumerate(gn_modules):
                module.MODE = mode
                # module.step = step
                

def align_feature(target, ref, target_mask, ref_mask, iter=100):
    target = target.detach().to(torch.float32).clone().requires_grad_(True)
    ref.detach()
    target_mask.detach()
    ref_mask.detach()

    optimizer = torch.optim.Adam([target], lr=1e-2)
    scaler = torch.cuda.amp.GradScaler()

    for i in range(iter):
        with torch.autocast(device_type='cuda'):
            loss = find_closest_features(target, ref, target_mask, ref_mask, k=20)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return target.detach().requires_grad_(False).half()


def find_closest_features(target, ref, target_mask, ref_mask, k):
    """
    Find the k closest features in the regions where target_mask and ref_mask are 1,
    and calculate the alignment loss for each target feature and its k closest reference features.

    Args:
    target (torch.Tensor): Target feature tensor with shape (N, C, H, W).
    ref (torch.Tensor): Reference feature tensor with shape (N, C, H, W).
    target_mask (torch.Tensor): Target mask tensor with shape (N, 1, H, W), where value 1 indicates the valid region.
    ref_mask (torch.Tensor): Reference mask tensor with shape (N, C, H, W), where value 1 indicates the valid region.
    k (int): Number of closest features to consider.

    Returns:
    loss (torch.Tensor): Alignment loss tensor.
    """
    # Expand target_mask and ref_mask to the same shape as feature tensors for element-wise multiplication
    n = target.shape[0]
    ref_mask = ref_mask.to(target.device)
    target_mask_expand = target_mask.expand_as(target)
    ref_mask_expand = ref_mask.expand_as(ref)

    # Extract valid feature regions based on the masks
    target_features = target[target_mask_expand == 1]
    ref_features = ref[ref_mask_expand == 1]
    # Reshape to (N, k, C) where k is the number of closest features
    target_features = target_features.view(n, -1, target.shape[1])
    ref_features = ref_features.view(n, -1, ref.shape[1])
    # print(target_features.shape, ref_features.shape)

    # Calculate the distance matrix between target features and reference features
    distance_matrix = torch.norm(ref_features[:, :, None] - target_features[:, None], dim=3)
    min_values, min_indices = torch.min(distance_matrix, dim=2)
    top_k_values, ref_k_indices = torch.topk(min_values, k, dim=1, largest=False)
    
    target_k_indices = torch.gather(min_indices, dim=1, index=ref_k_indices)

    
    target_k_feature = torch.gather(target_features, dim=1, index = target_k_indices.unsqueeze(dim=-1).repeat(1,1,target_features.shape[-1]))
    ref_k_feature = torch.gather(ref_features, dim=1, index = ref_k_indices.unsqueeze(dim=-1).repeat(1,1,ref_features.shape[-1]))

    
    # Calculate the loss
    loss = torch.nn.functional.l1_loss(target_k_feature, ref_k_feature.detach())
    
    # Reshape the indices to (N, k, 1) to gather the coordinates
    target_k_indices_expanded = target_k_indices[:, :, None]
    ref_k_indices_expanded = ref_k_indices[:, :, None]

    # Gather coordinates of the closest features
    target_mask = target_mask.squeeze().squeeze() # h,w
    ref_mask = ref_mask.squeeze().squeeze() # h,w
    target_coordinates = torch.nonzero(target_mask).view(1, -1, 2).repeat(n,1,1)
    ref_coordinates = torch.nonzero(ref_mask).view(1, -1, 2).repeat(n,1,1)
    

    # Gather the coordinates of the k closest features using the indices
    target_k_coordinates = torch.gather(target_coordinates, dim=1, index=target_k_indices_expanded.repeat(1, 1, 2))
    ref_k_coordinates = torch.gather(ref_coordinates, dim=1, index=ref_k_indices_expanded.repeat(1, 1, 2).to(ref_coordinates.device))
    loss_patch = calculate_patch_loss(target, ref, target_k_coordinates, ref_k_coordinates, r=5)
    print("loss", loss.item(), "loss_patch", loss_patch.item())

    return loss + loss_patch

def calculate_patch_loss(target, ref, target_k_coordinates, ref_k_coordinates, r):
    device = target.device
    n, c, h, w = target.size()

    total_loss = 0.0

    for i in range(n):
        for j in range(target_k_coordinates.shape[1]):
            # Get the coordinates of the patch centers
            target_x, target_y = target_k_coordinates[i, j]
            ref_x, ref_y = ref_k_coordinates[i, j]

            # Calculate the coordinates of the patch in the original image
            target_patch_x = torch.arange(target_x - r, target_x + r + 1).long().to(device)
            target_patch_y = torch.arange(target_y - r, target_y + r + 1).long().to(device)
            ref_patch_x = torch.arange(ref_x - r, ref_x + r + 1).long().to(device)
            ref_patch_y = torch.arange(ref_y - r, ref_y + r + 1).long().to(device)

            # Clip the coordinates to handle out-of-bounds values
            target_patch_x = torch.clamp(target_patch_x, 0, h - 1)
            target_patch_y = torch.clamp(target_patch_y, 0, w - 1)
            ref_patch_x = torch.clamp(ref_patch_x, 0, h - 1)
            ref_patch_y = torch.clamp(ref_patch_y, 0, w - 1)

            # Crop the patches from the target and ref images using indexing
            target_patch = target[i, :, target_patch_x[:, None], target_patch_y]
            ref_patch = ref[i, :, ref_patch_x[:, None], ref_patch_y]

            # Calculate the loss between the patches
            loss = F.mse_loss(target_patch, ref_patch)

            # Accumulate the loss
            total_loss += loss

    # Average the total loss
    total_loss /= (n * target_k_coordinates.shape[1])

    return total_loss
# def calculate_patch_loss(target, ref, target_k_coordinates, ref_k_coordinates, r):
#     device = target.device
#     n, c, h, w = target.size()
#     k = target_k_coordinates.shape[1]

#     # Generate grid coordinates for cropping patches
#     grid_y, grid_x = torch.meshgrid(torch.arange(-r, r+1), torch.arange(-r, r+1))
#     grid = torch.stack((grid_x, grid_y), dim=0).float().to(device)
#     # Reshape grid to match target_k_coordinates
#     grid = grid.unsqueeze(0).unsqueeze(0).repeat(n, k, 1, 1, 1)
#     # n,k,2,7,7

#     # Calculate target and ref grid coordinates
#     target_grid = target_k_coordinates.unsqueeze(-1).unsqueeze(-1).repeat(1,1, 1, 2*r+1, 2*r+1) + grid

#     # n,k,2  -> n,k,2,1,1
#     ref_grid = ref_k_coordinates.unsqueeze(-1).unsqueeze(-1).repeat(1,1, 1, 2*r+1, 2*r+1)  + grid

#     # Clip the grid coordinates to handle out-of-bounds values
#     # target_grid = torch.clamp(target_grid[:,:,0], 0, h-1)
#     # ref_grid = torch.clamp(ref_grid[:,:,0], 0, h-1)
#     # target_grid = torch.clamp(target_grid[:,:,1], 0, w-1)
#     # ref_grid = torch.clamp(ref_grid[:,:,1], 0, w-1)
#     # n k 2 7 7 
#     target_grid = target_grid.view(n,k,2,-1).permute(0,1,3,2).reshape(n,-1,2).long() #
#     ref_grid = ref_grid.view(n,k,2,-1).permute(0,1,3,2).reshape(n,-1,2).long()
    

#     # Sample the patches using grid_sample
#     target_patches = target[:, :, target_grid[:, :, 0], target_grid[:, :, 1]]
#     ref_patches = ref[:, :, ref_grid[:, :, 0], ref_grid[:, :, 1]]

#     # target_patches = F.grid_sample(target, target_grid, align_corners=False)
#     # ref_patches = F.grid_sample(ref, ref_grid, align_corners=False)


#     # Step 3: Calculate the loss between the patches
#     patch_loss = torch.nn.functional.l1_loss(target_patches, ref_patches.detach())

#     return patch_loss