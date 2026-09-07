# import os
# import re
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # import torchvision.transforms as T  # 注释掉避免版本兼容问题
# import argparse
# import json
# from PIL import Image
# from tqdm import tqdm
# from transformers import logging
# import math
# import inspect
# from diffusers import DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel,AutoencoderKL
# from diffusers.image_processor import VaeImageProcessor
# try:
#     from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
#     from diffusers.models.attention_processor import FusedAttnProcessor2_0
# except ImportError:
#     # 旧版本diffusers兼容
#     AttnProcessor2_0 = None
#     XFormersAttnProcessor = None
#     FusedAttnProcessor2_0 = None
# import gc
# import numpy as np
# from attention_mask_utils import *
# from sentence_transformers.util import (semantic_search, 
#                                         dot_score, 
#                                         normalize_embeddings)
# from contextlib import nullcontext

# class SimpleAttentionStore:
#     """ToMe风格的注意力存储器，用于收集cross-attention maps"""
#     def __init__(self):
#         self.attention_maps = []
#         self.step_count = 0
    
#     def __call__(self, attn_probs, is_cross, place_in_unet):
#         """存储cross-attention概率分布"""
#         if is_cross and attn_probs.shape[-1] == 77:  # 只存储text cross-attention
#             # 存储detached版本避免梯度图增长
#             self.attention_maps.append({
#                 'probs': attn_probs.detach(),
#                 'place': place_in_unet
#             })
    
#     def reset(self):
#         """清空存储"""
#         self.attention_maps = []
#         self.step_count += 1
    
#     def get_average_attention(self):
#         """聚合所有层的attention maps"""
#         if not self.attention_maps:
#             return None
#         # 简单平均所有层的attention
#         all_probs = torch.stack([m['probs'] for m in self.attention_maps])
#         return all_probs.mean(dim=0)  # [batch, query_len, 77]

# def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
#     """
#     根据 guidance_rescale 对 noise_cfg 进行重标定。
#     该方法基于论文《Common Diffusion Noise Schedules and Sample Steps are Flawed》（https://arxiv.org/pdf/2305.08891.pdf）第3.4节的发现。
#     主要作用：修正扩散模型在使用 classifier-free guidance 时可能出现的过曝或图像过于平淡的问题。
#     """
#     # 计算文本条件噪声预测的标准差
#     std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
#     # 计算 guidance 条件噪声预测的标准差
#     std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
#     # 对 guidance 结果进行重标定（修正过曝）
#     noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
#     # 按 guidance_rescale 权重混合原始和重标定结果，避免图像过于平淡
#     noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
#     return noise_cfg

# logging.set_verbosity_error()
# def tokenize_prompt(tokenizer, prompt):
#     """
#     使用指定的分词器将文本提示词转换为模型可理解的数字序列
    
#     Args:
#         tokenizer: 分词器对象（CLIP或OpenCLIP的分词器）
#         prompt: 输入的文本提示词（字符串或字符串列表）
    
#     Returns:
#         text_input_ids: token ID序列，形状为[1, max_length]
    
#     注意：
#         - 使用padding="max_length"确保所有序列长度一致
#         - truncation=True处理超长文本
#         - 返回的是PyTorch tensor格式，便于GPU计算
#     """
#     # 使用指定的分词器（tokenizer）对输入的 prompt 进行分词编码
#     text_inputs = tokenizer(
#         prompt,                         # 输入的文本 prompt
#         padding="max_length",           # 填充到最大长度
#         max_length=tokenizer.model_max_length,  # 最大长度由分词器模型决定
#         truncation=True,                # 超过最大长度则截断
#         return_tensors="pt",            # 返回 PyTorch tensor 格式
#     )
#     text_input_ids = text_inputs.input_ids  # 获取编码后的 token id
#     # 示例输出（SDXL CLIP分词器）：
#     # 输入: "photo of a <panda1> panda playing with a ball, castle background",
#     # 输出: [49406, 1125, 539, 320, 6442, 304, 271, 2675, 49407, 49407, ...]
#     # 对应: [<start>, "photo", "of", "a", "<panda1>", "playing", "with", "a", "ball", ",", "castle", "background", <end>, <pad>, ...]
#     # 长度: 77（SDXL标准长度）
#     return text_input_ids                   # 返回 token id
   

# def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
#     """
#     将文本提示词编码为高维语义向量表示，支持双编码器架构
    
#     这是MultiCompose系统的核心文本理解组件，使用CLIP和OpenCLIP双编码器
#     提供更强的文本理解能力，支持多概念融合的复杂语义处理。
    
#     Args:
#         text_encoders: 文本编码器列表 [CLIP编码器, OpenCLIP编码器]
#         tokenizers: 分词器列表 [CLIP分词器, OpenCLIP分词器]  
#         prompt: 输入的文本提示词（字符串或字符串列表）
#         text_input_ids_list: 可选的预分词token ID列表（用于优化性能）
    
#     Returns:
#         prompt_embeds: 序列嵌入 tensor [batch_size, seq_len, 2048]
#             - batch_size: 提示词数量
#             - seq_len: 序列长度（SDXL标准为77）
#             - 2048: 双编码器拼接维度（1024+1024）
#         pooled_prompt_embeds: 全局嵌入 tensor [batch_size, 2048]
#             - 用于SDXL的额外条件输入
#             - 提供全局语义信息
    
#     Technical Details:
#         - 使用倒数第二层隐藏状态而非最后一层（更稳定的表示）
#         - 双编码器结果在最后一维拼接，增强语义理解能力
#         - 支持批处理，提高计算效率
#     """
#     prompt_embeds_list = []  # 用于存储每个编码器的嵌入

#     for i, text_encoder in enumerate(text_encoders):
#         # 遍历每个文本编码器（CLIP和OpenCLIP）
#         if tokenizers is not None:
#             # 如果有分词器列表，则用对应分词器对 prompt 进行分词
#             tokenizer = tokenizers[i]
#             text_input_ids = tokenize_prompt(tokenizer, prompt)
#         else:
#             # 否则使用传入的 token id 列表（节省重复分词时间）
#             assert text_input_ids_list is not None
#             text_input_ids = text_input_ids_list[i]

#         # 用编码器对 token id 进行编码，获取输出和隐藏状态
#         prompt_embeds = text_encoder(
#             text_input_ids.to(text_encoder.device),  # 将 token id 移动到编码器所在设备
#             output_hidden_states=True,               # 返回隐藏层状态（用于提取中间层特征）
#         )

#         # 提取pooled输出（用于SDXL的额外条件输入）
#         pooled_prompt_embeds = prompt_embeds[0]     # 获取 pooled 输出 [batch_size, hidden_dim]
#         # 使用倒数第二层隐藏状态（更稳定的文本表示，避免过拟合最后一层）
#         prompt_embeds = prompt_embeds.hidden_states[-2]  # 获取倒数第二层的隐藏状态
#         bs_embed, seq_len, _ = prompt_embeds.shape  # 获取 batch size 和序列长度
#         prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)  # 重新调整形状
#         prompt_embeds_list.append(prompt_embeds)    # 添加到列表

#     # 将所有编码器的嵌入在最后一个维度拼接（CLIP + OpenCLIP）
#     prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)  # [batch, seq_len, 2048]
#     # 对 pooled 输出也做形状调整
#     pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)  # [batch, 2048]
#     return prompt_embeds, pooled_prompt_embeds       # 返回嵌入和 pooled 嵌入

# def compute_time_ids():
#     # 生成 SDXL 所需的时间编码（time_ids），用于条件控制
#     # 参考 StableDiffusionXLPipeline._get_add_time_ids 的实现
#     original_size = (opt.resolution_h, opt.resolution_w)  # 原始图片尺寸
#     target_size = (opt.resolution_h, opt.resolution_w)    # 目标图片尺寸
#     crops_coords_top_left = (opt.crops_coords_top_left_h, opt.crops_coords_top_left_w)  # 裁剪左上角坐标
#     add_time_ids = list(original_size + crops_coords_top_left + target_size)  # 拼成一个列表
#     add_time_ids = torch.tensor([add_time_ids])  # 转为 tensor，增加 batch 维
#     # add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)  # 可选：转到指定设备和类型
#     return add_time_ids  # 返回时间编码

# def preprocess_mask(mask_path, h, w, device):
#     # 读取掩码图片并预处理，返回指定尺寸和设备上的二值掩码
#     mask = np.array(Image.open(mask_path).convert("L"))  # 读取图片并转为灰度
#     mask = mask.astype(np.float32) / 255.0               # 归一化到 0-1
#     mask = mask[None, None]                              # 增加 batch 和 channel 维度
#     mask[mask < 0.5] = 0                                 # 小于 0.5 设为 0
#     mask[mask >= 0.5] = 1                                # 大于等于 0.5 设为 1
#     mask = torch.from_numpy(mask).to(device)              # 转为 tensor 并移动到指定设备
#     mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')  # 插值到目标尺寸
#     return mask                                          # 返回处理后的掩码

# def preprocess_mask_raw(mask_path, h, w, device):
#     # 预处理掩码（原始版，未读取图片，假设 mask 已经是 tensor）
#     mask[mask < 0.5] = 0                                 # 小于 0.5 设为 0
#     mask[mask >= 0.5] = 1                                # 大于等于 0.5 设为 1
#     mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')  # 插值到目标尺寸
#     return mask                                          # 返回处理后的掩码

# def parse_external_boxes(boxes_str):
#     # 解析形如 "x1,y1,x2,y2+x1,y1,x2,y2" 的字符串为列表
#     boxes_str = boxes_str.strip()
#     if boxes_str == "":
#         return []
#     parts = boxes_str.split('+')
#     boxes = []
#     for p in parts:
#         nums = p.split(',')
#         if len(nums) != 4:
#             continue
#         try:
#             x1, y1, x2, y2 = [float(v) for v in nums]
#             boxes.append([x1, y1, x2, y2])
#         except Exception:
#             continue
#     return boxes

# def build_masks_from_boxes(boxes, img_h, img_w, latent_h, latent_w, device):
#     # 将像素坐标的矩形框转为 latent 分辨率下的二值掩码（每个框一个通道）
#     if len(boxes) == 0:
#         return None
#     masks = []
#     for (x1, y1, x2, y2) in boxes:
#         lx1 = int(max(0, min(latent_w, round(x1 / img_w * latent_w))))
#         ly1 = int(max(0, min(latent_h, round(y1 / img_h * latent_h))))
#         lx2 = int(max(0, min(latent_w, round(x2 / img_w * latent_w))))
#         ly2 = int(max(0, min(latent_h, round(y2 / img_h * latent_h))))
#         if lx2 <= lx1 or ly2 <= ly1:
#             m = torch.zeros(1, 1, latent_h, latent_w, device=device)
#         else:
#             m = torch.zeros(1, 1, latent_h, latent_w, device=device)
#             m[:, :, ly1:ly2, lx1:lx2] = 1.0
#         masks.append(m)
#     masks = torch.cat(masks, dim=0)
#     return masks


# class MultiCompose(nn.Module):
#     def __init__(self, config):
#         super().__init__()  # 调用父类 nn.Module 的初始化方法
#         self.config = config  # 保存配置参数
#         sd_version = config.sd_version  # 获取 stable diffusion 版本
#         model_override = str(getattr(config, "pretrained_model_name_or_path", "")).strip()
#         env_model_name = str(os.environ.get("MODEL_NAME", "")).strip()

#         # 根据配置选择不同的 stable diffusion 预训练模型
#         if model_override:
#             model_key = model_override
#         elif sd_version == '2.1':
#             model_key = "stabilityai/stable-diffusion-2-1-base"
#         elif sd_version == '2.0':
#             model_key = "stabilityai/stable-diffusion-2-base"
#         elif sd_version == '1.5':
#             model_key = "runwayml/stable-diffusion-v1-5"
#         elif sd_version =='1.4':
#             model_key = "CompVis/stable-diffusion-v1-4"
#         elif sd_version =='xl':
#             model_key = env_model_name if env_model_name else "stabilityai/stable-diffusion-xl-base-1.0"
#         else:
#             raise ValueError(f'Stable-diffusion version {sd_version} not supported.')  # 不支持的版本报错
        
#         # 创建 SD 模型
#         print(f'Loading SD model from: {model_key}')

#         # 加载 SDXL 管道：优先 fp16 variant；本地目录缺失时自动回退
#         try:
#             pipe = StableDiffusionXLPipeline.from_pretrained(
#                 model_key, torch_dtype=torch.float16, variant="fp16", use_safetensors=False
#             ).to("cuda")
#         except OSError as exc:
#             msg = str(exc)
#             if "fp16" not in msg:
#                 raise
#             print(f"[WARN] fp16 variant not found for base pipeline ({model_key}), fallback to default weights.")
#             pipe = StableDiffusionXLPipeline.from_pretrained(
#                 model_key, torch_dtype=torch.float16, use_safetensors=False
#             ).to("cuda")

#         self.use_xformers = bool(not getattr(config, "disable_xformers", 0))
#         if self.use_xformers:
#             try:
#                 pipe.enable_xformers_memory_efficient_attention()  # 启用 xformers 高效注意力机制
#             except Exception as exc:
#                 print(f"[WARN] failed to enable xformers on pipeline: {exc}; continue without xformers.")
#                 self.use_xformers = False

#         pipe.enable_vae_slicing()  # 启用 VAE 切片，节省显存
#         # 默认直接使用基础模型自带 VAE（离线环境更稳）。
#         # 仅在显式提供路径/ID时才尝试额外加载 VAE，避免无意联网请求。
#         self.vae = pipe.vae
#         self.vae_source = "base_pipeline"
#         vae_override = str(getattr(config, "vae_model_name_or_path", "")).strip()
#         if not vae_override:
#             vae_override = str(os.environ.get("VAE_MODEL_NAME", "")).strip()
#         if vae_override:
#             try:
#                 vae_kwargs = {"torch_dtype": torch.float16}
#                 if os.path.isdir(vae_override):
#                     vae_kwargs["local_files_only"] = True
#                 self.vae = AutoencoderKL.from_pretrained(vae_override, **vae_kwargs).to("cuda")
#                 self.vae_source = "external_override"
#                 print(f"Loaded external VAE from: {vae_override}")
#             except Exception as exc:
#                 print(f"[WARN] failed to load external VAE ({vae_override}): {exc}; fallback to pipeline VAE.")
#                 self.vae = pipe.vae
#                 self.vae_source = "base_pipeline"
#         else:
#             print("Using VAE from base pipeline (no external VAE override).")
        
#         # 计算 VAE 的缩放因子
#         self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
#         # 创建 VAE 图像处理器
#         self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
#         # 保存分词器、文本编码器、UNet等
#         self.tokenizer = pipe.tokenizer
#         self.tokenizer_2 = pipe.tokenizer_2
#         self.text_encoder = pipe.text_encoder
#         self.text_encoder_2 = pipe.text_encoder_2
#         self.unet = pipe.unet
#         if self.use_xformers:
#             try:
#                 self.unet.enable_xformers_memory_efficient_attention()  # 启用高效注意力
#             except Exception as exc:
#                 print(f"[WARN] failed to enable xformers on base UNet: {exc}; continue without xformers.")
#                 self.use_xformers = False
#         self.device = self.unet.device  # 记录设备
        
#         self.sts = []  # 用于存储自定义权重
#         self.masks = None  # 初始化掩码

#         # 解析配置中的模型路径、原始提示词、分割提示词、概念、修饰 token
#         model_paths = config.personal_checkpoint.split('+')

#         #photo of a cat wearing tie and a dog with sunglasses running,<castle1> castle background
#         prompt_orig = config.prompt_orig.split('+')[0]
        
#         #photo of a cat wearing tie,castle background
#         #photo of a dog with sunglasses,castle background
#         #castle background
#         prompt_sep = config.prompt.split('+')


#         #CONCEPTS="cat+dog+castle"
#         concepts = config.concepts.split('+')
#         #MODIFIER="<cat5>+<dog2>+<castle1>"
#         modifier_token_user = config.modifier_token.split('+')
#         prompts = []
#         prompts.append(prompt_orig)  # 添加原始 prompt

#         prompt_orig_clean = getattr(config, 'prompt_orig_clean', prompt_orig)
#         self.prompt_orig_clean = prompt_orig_clean
#         #photo of a cat wearing tie,castle background+photo of a dog with sunglasses,castle background+castle background
#         concept_num = len(concepts)  # 概念数量


#         #photo of a cat wearing tie,castle background
#         #photo of a dog with sunglasses,castle background
#         prompts_single = prompt_sep[:concept_num-1]
#         self.prompts_single = prompts_single  # 单独的 prompt

#         weights_cfg = getattr(config, "concept_weights", "")
#         if weights_cfg:
#             parsed_weights = [float(w) for w in weights_cfg.split(",") if w.strip() != ""]
#             if len(parsed_weights) < concept_num:
#                 parsed_weights.extend([1.0] * (concept_num - len(parsed_weights)))
#             self.concept_weights = parsed_weights[:concept_num]
#         else:
#             self.concept_weights = [1.0] * concept_num

#         prompts_single_clean = []
#         self.prompts_single_clean = prompts_single_clean
#          #"cat wearing tie+dog with sunglasses"
#         prompt_clean_cfg = getattr(config, 'prompt_clean', '')
#         if prompt_clean_cfg:
#             prompt_clean_sep = prompt_clean_cfg.split('+')
#             #[cat wearing tie,dog with sunglasses]
#             if len(prompt_clean_sep) >= concept_num - 1:
#                 prompts_single_clean = prompt_clean_sep[:concept_num-1]
#                 #[cat wearing tie,dog with sunglasses]
#         # 保存最终的字符串列表到实例，供调试打印使用
#         self.prompts_single_clean = prompts_single_clean
#         # 仅用于“重采样单概念”的干净句（与 MSE 锚点的 PROMPT_CLEAN 解耦）
#         prompts_single_resample = []
#         self.prompts_single_resample = prompts_single_resample
#         prompt_clean_resample_cfg = getattr(config, 'prompt_clean_resample', '')
#         if prompt_clean_resample_cfg:
#             prompt_clean_resample_sep = prompt_clean_resample_cfg.split('+')
#             if len(prompt_clean_resample_sep) >= concept_num - 1:
#                 prompts_single_resample = prompt_clean_resample_sep[:concept_num-1]
#         #[photo of a cat,photo of a dog]
#         # 同步保存实例属性（供调试打印与后续使用）
#         self.prompts_single_resample = prompts_single_resample
#         self.concept_num = concept_num  # 保存概念数量
#         # 构造每个概念的 prompt，插入修饰 token


#         ##photo of a cat wearing tie,castle background
#         #photo of a dog with sunglasses,castle background
#         #castle background
#         #"cat+dog+castle"
#         #modifier_token_user="<cat5>+<dog2>+<castle1>"
#         for i, wd in enumerate(concepts):
#             base_prompt = prompt_sep[i] if i < len(prompt_sep) else ""
#             modifier_token = modifier_token_user[i] if i < len(modifier_token_user) else ""
#             index = base_prompt.find(wd)  # 找到概念在 prompt 中的位置
#             if modifier_token and index >= 0:
#                 result = base_prompt[:index] + modifier_token + " " + base_prompt[index:]
#             else:
#                 result = base_prompt

#             prompts.append(result)
        
#         ##photo of a cat wearing tie and a dog with sunglasses running,<castle1> castle background
#         #1. photo of a <cat5> cat wearing tie,castle background
#         # 2. photo of a <dog2> dog with sunglasses,castle background
#         # 3. <castle1> castle background

#         # 加载每个自定义权重文件
#         #PERSONAL_CHECKPOINT="/path/to/checkpoints/subject_a.bin+/path/to/checkpoints/subject_b.bin+/path/to/checkpoints/background.bin"
#         for sp in model_paths:
#             self.sts.append(torch.load(sp))
#         #sts=[..pet_cat5/delta-200.bin,..pet_dog2/delta-200.bin,..scene_castle/delta-200.bin]
#         modifier_token_id = []
#         modifier_token_id_2 = []

#         # 如果权重中包含 modifier_token，则进行 token 嵌入的替换
#         if 'modifier_token' in self.sts[0]:
#             modifier_tokens = []
#             modifier_tokens_2 = []

#             # 收集所有 modifier_token 和 modifier_token_2
#             for single_st in self.sts:
#                 modifier_tokens += list(single_st['modifier_token'].keys())
#                 modifier_tokens_2 += list(single_st['modifier_token_2'].keys())
                
#             # 遍历用户指定的修饰 token，添加到分词器，并获取其 id
#             for i, modifier_token in enumerate(modifier_token_user):
#                 # self.find_disc(self.sts[i]['modifier_token'][modifier_token],self.sts[i]['modifier_token_2'][modifier_token])
#                 num_added_tokens = self.tokenizer.add_tokens(modifier_token)## 添加新token到分词器
#                 modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token))
#             # 同理，添加到第二个分词器
#                 num_added_tokens = self.tokenizer_2.add_tokens(modifier_token)
#                 modifier_token_id_2.append(self.tokenizer_2.convert_tokens_to_ids(modifier_token))
                
#             # 调整编码器的 token 嵌入表大小
#             # 调整嵌入表大小
#             # 原始大小：49407个token
#             # 新增3个token后：49410个token
#             #self.text_encoder.resize_token_embeddings(49410)  # 49407 + 3
#             self.text_encoder.resize_token_embeddings(len(self.tokenizer))
#             self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))
#             token_embeds = self.text_encoder.get_input_embeddings().weight.data
#             token_embeds_2 = self.text_encoder_2.get_input_embeddings().weight.data

#             # 用自定义权重替换 token 嵌入
#             # token_embeds[49408] = self.sts[0]['modifier_token']['<panda1>']
#             # token_embeds[49409] = self.sts[1]['modifier_token']['<teddybear1>']
#             # token_embeds[49410] = self.sts[2]['modifier_token']['<castle1>']
#             for i, id_ in enumerate(modifier_token_id):
#                 single_st = self.sts[i]
#                 token_embeds[id_] = single_st['modifier_token'][modifier_tokens[i]]
#             for i, id_ in enumerate(modifier_token_id_2):
#                 single_st = self.sts[i]
#                 token_embeds_2[id_] = single_st['modifier_token_2'][modifier_tokens_2[i]]
                
#         null_prompt = [config.negative_prompt]  # 负面提示词

#         # 获取所有 prompt 的文本嵌入
#         # torch.tensor([5, 77, 2048]),  # 4个prompt + 1个negative
#         # torch.tensor([5, 2048])       # 对应的pooled嵌入

#          ##photo of a cat wearing tie and a dog with sunglasses running,<castle1> castle background
#         #1. photo of a <cat5> cat wearing tie,castle background
#         # 2. photo of a <dog2> dog with sunglasses,castle background
#         # 3. <castle1> castle background
#         embeds_all = self.get_text_embeds(prompts, null_prompt, device=self.unet.device)

#         self.text_embeds_raw = (
#             embeds_all[0].clone(),
#             embeds_all[1].clone(),
#         )
#         self.text_embeds = (
#             self.text_embeds_raw[0].clone(),
#             self.text_embeds_raw[1].clone(),
#         )
#         self.text_embeds_multi_clean = None
#         if self.prompt_orig_clean:
#             embeds_multi_clean = self.get_text_embeds([self.prompt_orig_clean], null_prompt, device=self.unet.device)
#             seq_clean = embeds_multi_clean[0][1:2].to(device=self.unet.device, dtype=self.unet.dtype)
#             pool_clean = embeds_multi_clean[1][1:2].to(device=self.unet.device, dtype=self.unet.dtype)
#             self.text_embeds_multi_clean = (seq_clean, pool_clean)

#         # Token 合并与 ETS
#         self.binding_entries = self._load_binding_entries(getattr(config, "binding_json", ""))
#         self._binding_prompt_list = list(prompts)
#         self.binding_prompt_text = prompt_orig
#         self.binding_subject_indices = []
#         self.binding_subject_indices_merged = []
#         self.binding_clean_cache = {}
#         self.binding_clean_cache_merged = {}
#         self.binding_subject_groups = [] #保存 subject 分组（例如 cat tokens 在第 0 组，dog 在第 1 组）
#         self.binding_subject_groups_merged = {}
#         self.text_embeds_merged = (
#             self.text_embeds_raw[0].clone(),
#             self.text_embeds_raw[1].clone(),
#         )#先把未处理的嵌入复制一份，作为“合并后嵌入”的初始值——如果没有 binding，后续就直接用它。
#         binding_subject_map = {}
#         binding_subject_group_map = {}#临时字典，用于接收 _apply_token_merging_and_ets 返回的“每个 prompt 的 subject 索引 / 分组信息”。
#         if self.binding_entries:
#             (
#                 merged_embeds,
#                 binding_subject_map,
#                 binding_clean_cache,
#                 binding_subject_group_map,
#             ) = self._apply_token_merging_and_ets(
#                 (
#                     self.text_embeds_raw[0].clone(),
#                     self.text_embeds_raw[1].clone(),
#                 ),
#                 prompts,
#                 self.binding_entries,
#             )
#             self.text_embeds_merged = (
#                 merged_embeds[0].clone(),
#                 merged_embeds[1].clone(),
#             )
#             self.binding_subject_indices_merged = sorted(binding_subject_map.get(prompt_orig, []))
#             self.binding_clean_cache_merged = binding_clean_cache
#             self.binding_subject_groups_merged = binding_subject_group_map
#             self.binding_subject_groups = binding_subject_group_map.get(prompt_orig, [])
#         self._binding_entries = []
#         self._binding_clean_cache = {}
#         self._binding_prompt_to_index = {}
#         self._binding_active = False

#         concept_labels = self.config.concepts.split('+')
#         seg_focus = set()
#         seg_concepts_raw = getattr(self.config, 'seg_concepts', '')
#         if seg_concepts_raw:
#             seg_entries = [entry.strip().lower() for entry in seg_concepts_raw.split('+') if entry.strip()]
#             for label in concept_labels:
#                 label_lower = label.lower()
#                 for entry in seg_entries:
#                     if label_lower in entry or entry in label_lower:
#                         seg_focus.add(label)
#                         break
#         if not seg_focus:
#             seg_focus = {"panda", "cat"}
#         self._entropy_focus_labels = seg_focus
#         self.subject_token_ids = sorted(binding_subject_map.get(prompt_orig, []))
#         self.subject_token_labels = {}
#         subject_groups = self.binding_subject_groups or []
#         if subject_groups:
#             for group_idx, token_list in enumerate(subject_groups):
#                 label = concept_labels[group_idx] if group_idx < len(concept_labels) else f"group{group_idx}"
#                 for token_idx in token_list:
#                     self.subject_token_labels[int(token_idx)] = label
#         if not self.subject_token_labels:
#             for i, idx in enumerate(self.subject_token_ids):
#                 label = concept_labels[i] if i < len(concept_labels) else f"token{idx}"
#                 self.subject_token_labels[idx] = label
#         else:
#             for idx in self.subject_token_ids:
#                 if idx not in self.subject_token_labels:
#                     self.subject_token_labels[idx] = f"token{idx}"
#         # Disable legacy attention-entropy guidance by default; use Cones-style
#         # mask attention bias during fusion instead.
#         self.enable_attention_entropy = False
#         self.attn_entropy_weight = float(getattr(self.config, "attn_entropy_weight", 0.0))
#         self.attn_entropy_outside_weight = float(getattr(self.config, "attn_entropy_outside_weight", 1.0))
#         self.attn_entropy_inside_weight = float(getattr(self.config, "attn_entropy_inside_weight", 0.0))
#         self.attn_entropy_steps = int(getattr(self.config, "attn_entropy_steps", 1))
#         self.attn_entropy_lr = float(getattr(self.config, "attn_entropy_lr", 1e-3))
#         self.attn_entropy_min_step = int(getattr(self.config, "attn_entropy_min_step", 0))
#         self.attn_entropy_max_step = int(getattr(self.config, "attn_entropy_max_step", -1))
#         layers_cfg = str(getattr(self.config, "attn_entropy_layers", "up")).lower()
#         if layers_cfg in ("", "all", "none"):
#             self._entropy_layer_filter = None if layers_cfg != "none" else set()
#         else:
#             layer_tokens = []
#             for token in layers_cfg.replace(",", "+").split("+"):
#                 token = token.strip()
#                 if token:
#                     layer_tokens.append(token)
#             mapped_prefix = set()
#             for token in layer_tokens:
#                 if token == "up":
#                     mapped_prefix.add("up_res")
#                 elif token == "mid":
#                     mapped_prefix.add("mid_block")
#                 elif token == "down":
#                     mapped_prefix.add("down_res")
#                 else:
#                     mapped_prefix.add(token)
#             self._entropy_layer_filter = mapped_prefix or None
#         self.attn_entropy_mask_downscale = max(1, int(getattr(self.config, "attn_entropy_mask_downscale", 1)))
#         self.attn_entropy_low_mem = bool(getattr(self.config, "attn_entropy_low_mem", 0))
#         self.attn_entropy_enable_checkpointing = bool(
#             getattr(self.config, "attn_entropy_enable_checkpointing", 0)
#         )
#         self._entropy_token_map = {}
#         focus_labels = self._entropy_focus_labels or {"panda", "cat"}
#         for idx, label in self.subject_token_labels.items():
#             if label in focus_labels:
#                 self._entropy_token_map.setdefault(label, []).append(idx)
#         self._entropy_masks_by_grid = {}
#         self._entropy_loss_terms = []
#         self._entropy_active = False
#         self._entropy_masks_ready = False

#         # Cones-style mask attention guidance config.
#         self.enable_cones_mask_attention = bool(getattr(self.config, "enable_cones_mask_attention", 1))
#         self.cones_guidance_weight = float(getattr(self.config, "cones_guidance_weight", 0.08))
#         self.cones_guidance_steps = int(getattr(self.config, "cones_guidance_steps", -1))
#         self.cones_positive_value = float(getattr(self.config, "cones_positive_value", 2.5))
#         self.cones_negative_value = float(getattr(self.config, "cones_negative_value", -0.02))
#         self.cones_use_sim_std = bool(getattr(self.config, "cones_use_sim_std", 1))
#         self.cones_use_modifier_only = bool(getattr(self.config, "cones_use_modifier_only", 1))
#         self.cones_focus_fg_only = bool(getattr(self.config, "cones_focus_fg_only", 1))
#         self.cones_use_plain_prompt_before_fusion = bool(
#             getattr(self.config, "cones_use_plain_prompt_before_fusion", 1)
#         )
#         self.cones_debug_tokens = bool(getattr(self.config, "cones_debug_tokens", 0))
#         # Extra gate for noisy per-step token-map prints; default off.
#         self.cones_debug_token_maps = bool(getattr(self.config, "cones_debug_token_maps", 0))
#         self._cones_masks_ready = False
#         self._cones_mask_cache = {}
#         self._cones_guidance_timestep_set = None
#         self._cones_token_map = {}
#         self._cones_token_map_plain = {}
#         self._cones_debug_seen = set()
#         self.use_sts_kv_hook = bool(getattr(self.config, "use_sts_kv_hook", 1))
#         self._sts_have_unet_kv = all(
#             isinstance(st, dict) and isinstance(st.get("unet", None), dict) for st in self.sts
#         )
#         fg_concept_limit = self.concept_num - 1 if (self.cones_focus_fg_only and self.concept_num > 1) else self.concept_num
#         for concept_idx, concept_label in enumerate(concept_labels):
#             if concept_idx >= fg_concept_limit:
#                 self._cones_token_map[concept_idx] = []
#                 self._cones_token_map_plain[concept_idx] = []
#                 continue

#             prompt_idx = min(concept_idx + 1, len(prompts) - 1)
#             prompt_text = prompts[prompt_idx]
#             modifier_id = modifier_token_id[concept_idx] if concept_idx < len(modifier_token_id) else None
#             concept_tokens = sorted(
#                 [idx for idx, label in self.subject_token_labels.items() if label == concept_label]
#             )
#             if self.cones_use_modifier_only and modifier_id is not None:
#                 # NOTE: Cones needs token POSITIONS in the text sequence, not vocab ids.
#                 concept_tokens = self._auto_cones_token_indices(
#                     prompt_text=prompt_text,
#                     concept_label=concept_label,
#                     modifier_token_id=modifier_id,
#                 )
#             if not concept_tokens and concept_idx < len(self.subject_token_ids):
#                 concept_tokens = [self.subject_token_ids[concept_idx]]
#             if not concept_tokens:
#                 concept_tokens = self._auto_cones_token_indices(
#                     prompt_text=prompt_text,
#                     concept_label=concept_label,
#                     modifier_token_id=modifier_id,
#                 )
#             self._cones_token_map[concept_idx] = concept_tokens
#             self._cones_token_map_plain[concept_idx] = self._auto_cones_token_indices(
#                 prompt_text=prompt_orig,
#                 concept_label=concept_label,
#                 modifier_token_id=None,
#             )
#         print(
#             f"[ConesMask2] token map sizes personalized="
#             f"{[len(v) for _, v in sorted(self._cones_token_map.items())]}, "
#             f"plain={ [len(v) for _, v in sorted(self._cones_token_map_plain.items())] }"
#         )

#         # 获取单独 prompt 的文本嵌入
#         # torch.tensor([3, 77, 2048]),  # 2个prompt + 1个negative
#         # torch.tensor([3, 2048])       # 对应的pooled嵌入
#         self.text_embeds_single = self.get_text_embeds(prompts_single, null_prompt, device=self.unet.device)
#         # Prepare clean single-concept embeds BEFORE building anchors so that --sem_binding_use_clean can take effect
#         self.text_embeds_single_clean = None
#         if prompts_single_clean:
#             embeds_clean = self.get_text_embeds(prompts_single_clean, null_prompt, device=self.unet.device)
#             self.text_embeds_single_clean = embeds_clean
#         # 专供重采样使用的“干净单概念”嵌入（如 PROMPT_CLEAN_resample）
#         self.text_embeds_single_resample = None
#         if len(self.prompts_single_resample) > 0:
#             embeds_resample = self.get_text_embeds(self.prompts_single_resample, null_prompt, device=self.unet.device)
#             self.text_embeds_single_resample = embeds_resample
#         # Now build semantic anchors (will choose clean or default based on --sem_binding_use_clean)
#         self.semantic_anchor_pairs = self._prepare_semantic_anchors()
#         # 释放分词器、编码器等资源，节省显存
#         del self.tokenizer, self.tokenizer_2, self.text_encoder, self.text_encoder_2
#         del pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2, pipe.unet, pipe.vae
#         gc.collect()
#         torch.cuda.empty_cache()
        
#         # 低显存优先：直接从 checkpoint 的 sts['unet'] 挂载 K/V（不再额外加载 unet_i）。
#         # 当 checkpoint 不含 unet 权重或显式关闭该模式时，回退到旧路径。
#         if self.use_sts_kv_hook and self._sts_have_unet_kv:
#             print("[ConesMask2] Using sts K/V hook path (no extra per-concept UNet load).")
#         else:
#             # 为每个自定义权重加载一份 UNet，并替换 attn2 层参数
#             for i, single_st in enumerate(self.sts):
#                 try:
#                     concept_unet = UNet2DConditionModel.from_pretrained(
#                         model_key, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
#                     ).to(self.device)
#                 except OSError as exc:
#                     msg = str(exc)
#                     if "fp16" not in msg:
#                         raise
#                     print(f"[WARN] fp16 variant not found for UNet ({model_key}/unet), fallback to default weights.")
#                     concept_unet = UNet2DConditionModel.from_pretrained(
#                         model_key, subfolder="unet", torch_dtype=torch.float16
#                     ).to(self.device)
#                 setattr(
#                     self,
#                     f"unet_{i}",
#                     concept_unet
#                 )
#                 model_name = f"unet_{i}"
#                 for name, params in getattr(self, model_name).named_parameters():
#                     if 'attn2' in name:
#                         if name in single_st['unet']:
#                             params.data.copy_(single_st['unet'][f'{name}'])
#                 if self.use_xformers:
#                     try:
#                         getattr(self, model_name).enable_xformers_memory_efficient_attention()
#                     except Exception as exc:
#                         print(f"[WARN] failed to enable xformers on {model_name}: {exc}; continue without xformers.")
#                         self.use_xformers = False
#         # 'vae': AutoencoderKL(...),
#         # 'unet': UNet2DConditionModel(...),  # 基础UNet
#         # 'unet_0': UNet2DConditionModel(...), # 熊猫概念UNet
#         # 'unet_1': UNet2DConditionModel(...), # 泰迪熊概念UNet
#         # 'unet_2': UNet2DConditionModel(...), # 城堡概念UNet
#         # 'scheduler': DDIMScheduler(...),
#         # 'text_embeds': (torch.tensor([5, 77, 2048]), torch.tensor([5, 2048])),
#         # 'text_embeds_single': (torch.tensor([3, 77, 2048]), torch.tensor([3, 2048])),
#         # 'concept_num': 3,
#         # 'sts': [weight_dict_0, weight_dict_1, weight_dict_2],



#         # 加载调度器（采样器）
#         self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
#         N_ts = len(self.scheduler.timesteps)  # 总步数
#         self.scheduler.set_timesteps(config.n_timesteps, device=self.unet.device)  # 设置采样步数
        
#         self.skip = N_ts // config.n_timesteps  # 步长
#         self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(self.unet.device)  # 最终 alpha
#         # 拼接 alphas_cumprod，首位加 1.0
#         self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

#         print('custom checkpoint loaded')  # 打印加载完成

#         # 计算并保存时间编码
#         self.add_time_ids = compute_time_ids()
#         self.add_time_ids = self.add_time_ids.to(self.unet.device)

#         # 语义绑定参数
#         self.sem_binding_steps = int(getattr(self.config, "sem_binding_steps", 0))
#         self.sem_binding_lr = float(getattr(self.config, "sem_binding_lr", 1e-3))
#         self.sem_binding_applied = False
#         # 在融合阶段允许执行语义绑定的时间步次数（默认仅执行一次）
#         self.sem_binding_max_steps = int(getattr(self.config, "sem_binding_max_steps", 1))
#         self.sem_binding_applied_steps = 0

#         # 低显存融合（串行概念推理），避免一次性堆叠所有概念分支
#         self.fusion_low_mem = bool(getattr(self.config, "fusion_low_mem", 0))
#         self._entropy_disabled = False

#     def _load_binding_entries(self, binding_json):
#         entries = []
#         if not binding_json:
#             binding_json = os.getenv("TOME_TOKEN_BINDINGS", "")
#         print("[Binding] raw:", binding_json) 
#         if not binding_json:
#             return entries
#         try:
#             if os.path.isfile(binding_json):
#                 with open(binding_json, "r", encoding="utf-8") as f:
#                     binding_json = f.read()
#             print("[Binding] after file load:", binding_json)
#         except OSError:
#             pass
#         try:
#             entries = json.loads(binding_json)
#             print("[Binding] parsed entries:", entries)
#         except Exception as exc:
#             print(f"[Binding] failed to parse binding_json: {exc}")
#             entries = []
#         return entries

#     def _apply_token_merging_and_ets(self, embeds_tuple, prompt_list, binding_entries):
#         if embeds_tuple is None or not binding_entries:
#             return embeds_tuple, {}, {}, {}
#         text_embeddings, pooled_embeddings = embeds_tuple
#         if text_embeddings is None:
#             return embeds_tuple, {}, {}, {}
#         subject_idx_map = {}
#         subject_group_map = {}
#         clean_embed_cache = {}
#         eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
#         max_length = getattr(self.tokenizer, "model_max_length", 77)
#         debug_binding = bool(getattr(self.config, "debug_binding", 0))

#         def _find_prompt_index(target_text):
#             for idx, text in enumerate(prompt_list):
#                 if text == target_text:
#                     return idx
#             return None

#         for entry in binding_entries:
#             prompt_text = entry.get("prompt_text")
#             if not prompt_text:
#                 continue
#             prompt_idx = _find_prompt_index(prompt_text)
#             if prompt_idx is None:
#                 continue
#             embed_idx = prompt_idx + 1  # index 0 is negative prompt
#             if embed_idx >= text_embeddings.shape[0]:
#                 continue
#             embed_slice = text_embeddings[embed_idx].clone()
#             prompt_ids = tokenize_prompt(self.tokenizer, prompt_text).to(embed_slice.device)[0]
#             if debug_binding:
#                 try:
#                     toks = self.tokenizer.convert_ids_to_tokens(prompt_ids.tolist())
#                     printable = []
#                     for i, tk in enumerate(toks):
#                         if i >= max_length:
#                             break
#                         printable.append(f"{i}:{tk}")
#                     print("[Debug][Binding] prompt_text tokens:")
#                     print("  " + " | ".join(printable))
#                 except Exception as _:
#                     pass

#             for pair in entry.get("pairs", []):
#                 subject_indices = pair.get("subject") or []
#                 if not subject_indices:
#                     continue
#                 subject_indices = [int(si) for si in subject_indices]
#                 # Track exact grouping order for downstream features (e.g., entropy masks)
#                 subject_group_map.setdefault(prompt_text, []).append(list(subject_indices))
#                 subject_tensor = torch.tensor(subject_indices, device=embed_slice.device, dtype=torch.long)
#                 merged_vec = embed_slice[subject_tensor].sum(dim=0)
#                 attr_groups = []
#                 for key in ("attributes", "modifiers", "extras", "additional"):
#                     attr_groups.extend(pair.get(key, []))
#                 if debug_binding:
#                     print(f"[Debug][Binding] subject={subject_indices} attrs={attr_groups}")
#                 for group in attr_groups:
#                     if not group:
#                         continue
#                     attr_tensor = torch.tensor(group, device=embed_slice.device, dtype=torch.long)
#                     merged_vec = merged_vec + embed_slice[attr_tensor].sum(dim=0)
#                     embed_slice[attr_tensor] = 0.0
#                 if subject_tensor.numel() > 1:
#                     embed_slice[subject_tensor[1:]] = 0.0
#                 embed_slice[subject_tensor[0]] = merged_vec
#                 subject_idx_map.setdefault(prompt_text, set()).update(subject_indices)

#             clean_prompt = entry.get("eot_clean_prompt")
#             if clean_prompt and eos_token_id is not None:
#                 if clean_prompt not in clean_embed_cache:
#                     clean_embeds = self.get_text_embeds([clean_prompt], [self.config.negative_prompt], device=self.unet.device)
#                     clean_seq = clean_embeds[0][1].to(embed_slice.device, dtype=embed_slice.dtype)
#                     clean_embed_cache[clean_prompt] = clean_seq
#                 clean_slice = clean_embed_cache[clean_prompt]
#                 mask = prompt_ids == eos_token_id
#                 if mask.any():
#                     embed_slice[mask] = clean_slice[mask]
#                     if debug_binding:
#                         print(f"[Debug][Binding] ETS applied from clean prompt at EOT positions.")

#             text_embeddings[embed_idx] = embed_slice

#         subject_idx_map = {k: sorted(v) for k, v in subject_idx_map.items()}
#         subject_group_map = {
#             key: [list(group) for group in groups] for key, groups in subject_group_map.items()
#         }
#         if debug_binding:
#             print(f"[Debug][Binding] merged subjects per prompt: {subject_idx_map}")
#         return (text_embeddings, pooled_embeddings), subject_idx_map, clean_embed_cache, subject_group_map

#     def _activate_token_binding(self):
#         if getattr(self, "_binding_active", False):
#             print("[Binding] already active")
#             return
#         if not self.binding_entries:
#             print("[Binding] activate skipped: no entries")
#             return
#         print("[Binding] activating with entries:", self.binding_entries)
#         seq_merged, pool_merged = self.text_embeds_merged
#         self.text_embeds = (
#             seq_merged.clone().to(device=self.unet.device, dtype=self.unet.dtype),
#             pool_merged.clone().to(device=self.unet.device, dtype=self.unet.dtype),
#         )
#         self.binding_subject_indices = list(self.binding_subject_indices_merged)
#         self.binding_subject_groups = self.binding_subject_groups_merged.get(
#             self.binding_prompt_text, []
#         )
#         print("[Binding] subject indices:", self.binding_subject_indices)
#         cache_converted = {}
#         for key, value in (self.binding_clean_cache_merged or {}).items():
#             if isinstance(value, torch.Tensor):
#                 cache_converted[key] = value.to(device=self.unet.device, dtype=self.unet.dtype)
#             elif isinstance(value, tuple) and len(value) == 2:
#                 cache_converted[key] = (
#                     value[0].to(device=self.unet.device, dtype=self.unet.dtype),
#                     value[1].to(device=self.unet.device),
#                 )
#             else:
#                 cache_converted[key] = value
#         self.binding_clean_cache = cache_converted
#         self._binding_entries = self.binding_entries
#         prompt_to_index = {}
#         for entry in self.binding_entries:
#             prompt_text = entry.get("prompt_text")
#             if prompt_text and prompt_text in self._binding_prompt_list:
#                 prompt_to_index[prompt_text] = self._binding_prompt_list.index(prompt_text) + 1
#         self._binding_prompt_to_index = prompt_to_index
#         self._binding_clean_cache = self.binding_clean_cache
#         self._binding_active = True

#     def _prepare_semantic_anchors(self):
#         """
#         Build teacher anchors for semantic binding.

#         Default (backwards compatible): use prompts_single as anchors.
#         If --sem_binding_use_clean is set and PROMPT_CLEAN was provided (text_embeds_single_clean not None),
#         prefer the clean single-concept anchors so that subjects (e.g., <panda1> panda wearing tie)
#         include personalized tokens during MSE alignment.
#         """
#         anchors = []
#         # Whether to prefer clean single-concept prompts as anchors
#         use_clean = bool(getattr(self.config, "sem_binding_use_clean", 0))
#         debug_binding = bool(getattr(self.config, "debug_binding", 0))

#         embeds_tuple = None
#         if use_clean and getattr(self, "text_embeds_single_clean", None) is not None:
#             embeds_tuple = self.text_embeds_single_clean
#         else:
#             embeds_tuple = self.text_embeds_single

#         if embeds_tuple is None:
#             if debug_binding:
#                 print("[Debug][Binding] no anchors available (embeds_tuple is None)")
#             return anchors

#         seq_single, pool_single = embeds_tuple
#         if seq_single is None or seq_single.shape[0] <= 1:
#             if debug_binding:
#                 print("[Debug][Binding] anchors tensor empty or only negative prompt present")
#             return anchors

#         # Optional: decode each anchor's prompt into wordpieces for verification
#         if debug_binding:
#             try:
#                 src_name = "PROMPT_CLEAN" if (use_clean and getattr(self, "text_embeds_single_clean", None) is not None) else "PROMPT"
#                 src_prompts = getattr(self, "prompts_single_clean", None) if src_name == "PROMPT_CLEAN" else getattr(self, "prompts_single", None)
#                 if src_prompts is not None and hasattr(self, "tokenizer"):
#                     print(f"[Debug][Binding] anchors source={src_name}, count={seq_single.shape[0]-1}")
#                     for aidx in range(1, seq_single.shape[0]):
#                         text = src_prompts[aidx-1] if (aidx-1) < len(src_prompts) else "<unknown>"
#                         ids = tokenize_prompt(self.tokenizer, text)[0]
#                         toks = self.tokenizer.convert_ids_to_tokens(ids.tolist())
#                         max_len = getattr(self.tokenizer, "model_max_length", len(toks))
#                         printable = []
#                         for i, tk in enumerate(toks):
#                             if i >= max_len:
#                                 break
#                             printable.append(f"{i}:{tk}")
#                         print(f"  [Anchor {aidx-1}] {text}")
#                         print("    " + " | ".join(printable))
#             except Exception:
#                 pass

#         for idx in range(1, seq_single.shape[0]):
#             anchor_seq = seq_single[idx:idx + 1]
#             anchor_pool = pool_single[idx:idx + 1]
#             anchors.append({
#                 "seq": anchor_seq.to(device=self.unet.device, dtype=self.unet.dtype),
#                 "pool": anchor_pool.to(device=self.unet.device, dtype=self.unet.dtype),
#             })
#         if debug_binding:
#             src = "PROMPT_CLEAN" if (use_clean and getattr(self, "text_embeds_single_clean", None) is not None) else "PROMPT"
#             print(f"[Debug][Binding] anchors built: {len(anchors)} from {src}")
#         return anchors

#     def _semantic_binding_step(self, latents, t):
#         print(f"[Binding] 第{self.sem_binding_applied_steps + 1}次语义绑定，t={t.item() if torch.is_tensor(t) else t}")

#         if self.sem_binding_steps <= 0 or not self.binding_subject_indices:
#             return
#         if not getattr(self, "_binding_active", False):
#             return
#         if not getattr(self, "semantic_anchor_pairs", None):
#             return
#         if not self.semantic_anchor_pairs:
#             return
#         seq_embeds, pooled_embeds = self.text_embeds
#         seq_local = seq_embeds.clone()
#         subject_tensor = torch.tensor(self.binding_subject_indices, device=seq_local.device, dtype=torch.long)
#         if subject_tensor.numel() == 0:
#             return
        
#         latents_single = latents.detach().to(device=self.unet.device, dtype=self.unet.dtype)
#         time_ids_single = self.add_time_ids
#         device_type = self.unet.device.type
        
#         for _ in range(self.sem_binding_steps):
#             stokens = seq_local[1, subject_tensor].detach().clone().requires_grad_(True)
#             seq_step = seq_local.clone()
#             seq_step[1, subject_tensor] = stokens
#             cond_seq = seq_step[1:2]
#             cond_pool = pooled_embeds[1:2]
#             cond_kwargs = {"time_ids": time_ids_single, "text_embeds": cond_pool}
#             autocast_ctx = torch.autocast(device_type=device_type, dtype=self.unet.dtype) if device_type == "cuda" else nullcontext()
#             with autocast_ctx:
#                 noise_token = self.unet(
#                     latents_single,
#                     t,
#                     encoder_hidden_states=cond_seq,
#                     added_cond_kwargs=cond_kwargs,
#                 )["sample"]

#             loss = 0.0
#             for anchor in self.semantic_anchor_pairs:
#                 anchor_kwargs = {"time_ids": time_ids_single, "text_embeds": anchor["pool"]}
#                 with torch.no_grad():
#                     anchor_ctx = torch.autocast(device_type=device_type, dtype=self.unet.dtype) if device_type == "cuda" else nullcontext()
#                     with anchor_ctx:
#                         noise_anchor = self.unet(
#                             latents_single,
#                             t,
#                             encoder_hidden_states=anchor["seq"],
#                             added_cond_kwargs=anchor_kwargs,
#                         )["sample"]
#                 loss = loss + F.mse_loss(noise_token, noise_anchor)

#             loss = loss / max(len(self.semantic_anchor_pairs), 1)
#             print(f"[Binding] step loss: {loss.item():.6f}")
#             grad = torch.autograd.grad(loss, stokens, retain_graph=False, allow_unused=False)[0]
#             if grad is None:
#                 break
#             stokens = (stokens - self.sem_binding_lr * grad).detach()
#             seq_local[1, subject_tensor] = stokens

#         self.text_embeds = (seq_local, pooled_embeds)

#     def _prepare_entropy_targets(self):
#         if not self.enable_attention_entropy or self.masks is None:
#             self._entropy_masks_ready = False
#             self._entropy_masks_by_grid.clear()
#             return
#         valid_labels = set(self._entropy_token_map.keys())
#         if self._entropy_focus_labels:
#             valid_labels &= self._entropy_focus_labels
#         if not valid_labels:
#             self._entropy_masks_ready = False
#             self._entropy_masks_by_grid.clear()
#             return
#         self._entropy_masks_by_grid.clear()
#         self._entropy_masks_ready = True

#     def _prepare_cones_targets(self):
#         self._cones_mask_cache.clear()
#         self._cones_masks_ready = bool(self.enable_cones_mask_attention and self.masks is not None)

#     def _cones_subject_limit(self):
#         if self.masks is None:
#             return 0
#         concept_limit = min(int(getattr(self, "concept_num", 0)), int(self.masks.shape[0]))
#         if self.cones_focus_fg_only and concept_limit > 1:
#             return concept_limit - 1
#         return concept_limit

#     def _auto_cones_token_indices(self, prompt_text, concept_label, modifier_token_id=None):
#         token_ids = tokenize_prompt(self.tokenizer, prompt_text)[0].tolist()
#         positions = []
#         if modifier_token_id is not None:
#             positions = [idx for idx, tid in enumerate(token_ids) if tid == int(modifier_token_id)]

#         if not positions:
#             concept_ids = self.tokenizer(
#                 concept_label,
#                 add_special_tokens=False,
#                 truncation=True,
#             )["input_ids"]
#             if concept_ids:
#                 width = len(concept_ids)
#                 for start in range(0, len(token_ids) - width + 1):
#                     if token_ids[start:start + width] == concept_ids:
#                         positions.extend(range(start, start + width))

#         if not positions:
#             bos_id = getattr(self.tokenizer, "bos_token_id", None)
#             eos_id = getattr(self.tokenizer, "eos_token_id", None)
#             pad_id = getattr(self.tokenizer, "pad_token_id", None)
#             for idx, tid in enumerate(token_ids):
#                 if bos_id is not None and tid == bos_id:
#                     continue
#                 if eos_id is not None and tid == eos_id:
#                     continue
#                 if pad_id is not None and tid == pad_id:
#                     continue
#                 positions.append(idx)

#         return sorted(set(int(v) for v in positions))

#     def _infer_cones_query_hw(self, query_len, source_h, source_w):
#         if query_len <= 0:
#             return None
#         square = int(math.sqrt(query_len))
#         if square * square == query_len:
#             return square, square
#         if source_h <= 0 or source_w <= 0:
#             return None

#         target_aspect = float(source_h) / float(max(source_w, 1))
#         best_hw = None
#         best_err = float("inf")
#         for h in range(1, int(math.sqrt(query_len)) + 1):
#             if query_len % h != 0:
#                 continue
#             w = query_len // h
#             for cand_h, cand_w in ((h, w), (w, h)):
#                 cand_aspect = float(cand_h) / float(max(cand_w, 1))
#                 err = abs(cand_aspect - target_aspect)
#                 if err < best_err:
#                     best_err = err
#                     best_hw = (cand_h, cand_w)
#         return best_hw

#     def _resize_cones_mask_flat(self, source, cache_key, query_len, device, dtype):
#         if cache_key not in self._cones_mask_cache:
#             source_h, source_w = int(source.shape[-2]), int(source.shape[-1])
#             target_hw = self._infer_cones_query_hw(query_len, source_h, source_w)
#             if target_hw is None:
#                 return None
#             target_h, target_w = target_hw
#             resized = F.interpolate(
#                 source.to(dtype=torch.float32),
#                 size=(target_h, target_w),
#                 mode="bilinear",
#                 align_corners=False,
#             ).clamp_(0.0, 1.0)
#             self._cones_mask_cache[cache_key] = resized.reshape(query_len)
#         return self._cones_mask_cache[cache_key].to(device=device, dtype=dtype)

#     def _get_cones_mask_flat(self, concept_idx, query_len, device, dtype):
#         if not self._cones_masks_ready or self.masks is None:
#             return None
#         concept_limit = self._cones_subject_limit()
#         if concept_idx < 0 or concept_idx >= concept_limit:
#             return None
#         source = self.masks[concept_idx:concept_idx + 1]
#         return self._resize_cones_mask_flat(
#             source,
#             ("target", concept_idx, query_len),
#             query_len,
#             device,
#             dtype,
#         )

#     def _get_cones_irrelevant_mask_flat(self, concept_idx, query_len, device, dtype):
#         if not self._cones_masks_ready or self.masks is None:
#             return None
#         concept_limit = self._cones_subject_limit()
#         if concept_limit <= 1:
#             return torch.zeros((query_len,), device=device, dtype=dtype)

#         source_masks = []
#         for idx in range(concept_limit):
#             if idx == concept_idx:
#                 continue
#             source_masks.append(self.masks[idx:idx + 1].to(dtype=torch.float32))
#         if not source_masks:
#             return torch.zeros((query_len,), device=device, dtype=dtype)

#         union_source = torch.stack(source_masks, dim=0).amax(dim=0)
#         flat = self._resize_cones_mask_flat(
#             union_source,
#             ("irrelevant", concept_idx, query_len),
#             query_len,
#             device,
#             dtype,
#         )
#         if flat is None:
#             return None
#         return flat

#     def _cones_eta(self, timestep, sim):
#         t_val = 0.0
#         if timestep is not None:
#             t_val = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)
#         max_t = 1000.0
#         if hasattr(self, "scheduler") and hasattr(self.scheduler, "timesteps") and len(self.scheduler.timesteps) > 0:
#             first_t = self.scheduler.timesteps[0]
#             max_t = float(first_t.item()) if isinstance(first_t, torch.Tensor) else float(first_t)
#             max_t = max(max_t, 1.0)
#         t_norm = max(0.0, min(1.0, t_val / max_t))
#         eta = self.cones_guidance_weight * math.log1p(t_norm * t_norm)
#         if self.cones_use_sim_std:
#             eta *= float(sim.detach().std().item())
#         return eta

#     def apply_cones_attention_bias(self, sim, heads, num_batch, key_len, timestep, active_concept_idx=None):
#         if not self._cones_masks_ready or not self.enable_cones_mask_attention:
#             return sim
#         if key_len <= 0 or sim.ndim != 3:
#             return sim

#         t_int = None
#         if timestep is not None:
#             t_int = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
#         in_fusion = bool(
#             t_int is not None
#             and hasattr(self, "_t_cond_timestep_set")
#             and self._t_cond_timestep_set is not None
#             and t_int in self._t_cond_timestep_set
#         )

#         # Step gating is defined on fusion timesteps; keep pre-fusion plain-prompt
#         # guidance available when enabled.
#         if in_fusion and self._cones_guidance_timestep_set is not None:
#             if t_int not in self._cones_guidance_timestep_set:
#                 return sim

#         query_len = sim.shape[1]
#         eta = self._cones_eta(timestep, sim)
#         if eta == 0.0:
#             return sim

#         batch_concept_map = {}
#         token_map = self._cones_token_map
#         if in_fusion:
#             if num_batch == (1 + self.concept_num):
#                 for batch_idx in range(1, min(num_batch, self.concept_num + 1)):
#                     batch_concept_map[batch_idx] = [batch_idx - 1]
#             elif num_batch == 2 and active_concept_idx is not None:
#                 batch_concept_map[1] = [int(active_concept_idx)]
#             else:
#                 return sim
#         else:
#             if not self.cones_use_plain_prompt_before_fusion:
#                 return sim
#             subject_limit = self._cones_subject_limit()
#             if subject_limit <= 0 or num_batch < 2:
#                 return sim
#             token_map = self._cones_token_map_plain if self._cones_token_map_plain else self._cones_token_map
#             # For non-fusion phase, branch 1 is the multi-concept prompt branch.
#             batch_concept_map[1] = list(range(subject_limit))

#         if self.cones_debug_tokens and self.cones_debug_token_maps and t_int is not None:
#             phase = "fusion" if in_fusion else "pre_fusion"
#             debug_key = (phase, t_int)
#             if debug_key not in self._cones_debug_seen:
#                 entries = []
#                 for batch_idx, concept_indices in batch_concept_map.items():
#                     for concept_idx in concept_indices:
#                         token_indices = [
#                             idx for idx in token_map.get(concept_idx, []) if 0 <= idx < key_len
#                         ]
#                         if not token_indices:
#                             continue
#                         preview = token_indices[:8]
#                         tail = "..." if len(token_indices) > 8 else ""
#                         entries.append(f"b{batch_idx}:c{concept_idx}->{preview}{tail}")
#                 source = "personalized" if in_fusion else "plain"
#                 entry_text = "; ".join(entries) if entries else "none"
#                 print(
#                     f"[ConesMask2][Debug] phase={phase} t={t_int} source={source} "
#                     f"key_len={key_len} token_maps={entry_text}"
#                 )
#                 self._cones_debug_seen.add(debug_key)

#         for batch_idx, concept_indices in batch_concept_map.items():
#             bias = torch.zeros((query_len, key_len), device=sim.device, dtype=sim.dtype)
#             has_bias = False
#             for concept_idx in concept_indices:
#                 token_indices = [
#                     idx for idx in token_map.get(concept_idx, []) if 0 <= idx < key_len
#                 ]
#                 if not token_indices:
#                     continue

#                 mask_flat = self._get_cones_mask_flat(concept_idx, query_len, sim.device, sim.dtype)
#                 if mask_flat is None:
#                     continue
#                 irrelevant_flat = self._get_cones_irrelevant_mask_flat(
#                     concept_idx, query_len, sim.device, sim.dtype
#                 )
#                 if irrelevant_flat is None:
#                     continue

#                 spatial_bias = torch.zeros_like(mask_flat)
#                 spatial_bias = torch.where(
#                     mask_flat > 0.5,
#                     torch.full_like(mask_flat, self.cones_positive_value),
#                     spatial_bias,
#                 )
#                 spatial_bias = torch.where(
#                     (mask_flat <= 0.5) & (irrelevant_flat > 0.5),
#                     torch.full_like(mask_flat, self.cones_negative_value),
#                     spatial_bias,
#                 )
#                 bias[:, token_indices] = bias[:, token_indices] + spatial_bias.unsqueeze(-1).expand(
#                     -1, len(token_indices)
#                 )
#                 has_bias = True

#             if not has_bias:
#                 continue
#             start = batch_idx * heads
#             end = min((batch_idx + 1) * heads, sim.shape[0])
#             if start >= end:
#                 continue
#             sim[start:end] = sim[start:end] + eta * bias

#         return sim

#     def _get_entropy_mask(self, label, grid):
#         if not self._entropy_masks_ready or self.masks is None:
#             return None
#         cache = self._entropy_masks_by_grid.setdefault(grid, {})
#         if label in cache:
#             return cache[label]
#         concept_labels = self.config.concepts.split('+')
#         try:
#             mask_idx = concept_labels.index(label)
#         except ValueError:
#             return None
#         source = self.masks[mask_idx:mask_idx + 1].to(dtype=torch.float32)
#         resized = F.interpolate(
#             source,
#             size=(grid, grid),
#             mode="bilinear",
#             align_corners=False,
#         ).clamp_(0.0, 1.0)
#         flattened = resized.view(1, 1, grid * grid)
#         cache[label] = flattened
#         return cache[label]

#     def _reset_entropy_accumulator(self):
#         self._entropy_loss_terms = []
#         if self._entropy_layer_filter == set():
#             self._entropy_active = False
#         else:
#             self._entropy_active = self.enable_attention_entropy and self._entropy_masks_ready
#         # ✅ 不在这里重置_stored_attention_maps和_collecting_attention
#         # 让optimization loop自己管理这两个属性
    
#     def _store_attention_for_entropy(self, attn, heads, query_len, key_len, place):
#         """存储detached attention maps（不计算loss）"""
#         # ✅ 添加调试
#         collecting = getattr(self, '_collecting_attention', False)
#         if not hasattr(self, '_store_debug_once'):
#             self._store_debug_once = True
#             print(f"[Store] _collecting_attention={collecting}, has_list={hasattr(self, '_stored_attention_maps')}")
        
#         if not collecting:
#             return
        
#         # ✅ 确保列表存在
#         if not hasattr(self, '_stored_attention_maps'):
#             self._stored_attention_maps = []
        
#         self._stored_attention_maps.append({
#             'attn': attn,  # 已经detached
#             'heads': heads,
#             'query_len': query_len,
#             'key_len': key_len,
#             'place': place
#         })
        
#         # ✅ 打印首次成功存储
#         if len(self._stored_attention_maps) == 1:
#             print(f"[Store] 首次存储attention: place={place}, shape={attn.shape}")

#     def _accumulate_attention_entropy(self, attn, heads, query_len, key_len, place):
#         if not self._entropy_active or self.attn_entropy_weight <= 0.0:
#             return
#         if query_len < 1:
#             return
#         batch = attn.shape[0] // heads
#         if batch == 0:
#             return
#         attn = attn.view(batch, heads, query_len, key_len)
#         grid = int(math.sqrt(query_len))
#         if grid * grid != query_len:
#             return
#         device = attn.device
        
#         # ✅ 打印注意力收集信息（仅首次）
#         if not hasattr(self, '_attn_accum_logged'):
#             self._attn_accum_logged = True
#             print(f"[AttnAccum] place={place}, grid={grid}x{grid}, batch={batch}, heads={heads}")
#             print(f"[AttnAccum] attn.shape={attn.shape}, key_len={key_len}")
#             print(f"[AttnAccum] token_map={list(self._entropy_token_map.keys())}")
        
#         for label, token_indices in self._entropy_token_map.items():
#             if self._entropy_layer_filter is not None:
#                 if not any(str(place).startswith(prefix) for prefix in self._entropy_layer_filter):
#                     continue
#             mask_flat = self._get_entropy_mask(label, grid)
#             if mask_flat is None:
#                 continue
#             mask_flat = mask_flat.to(device=device, dtype=attn.dtype)
#             ds = self.attn_entropy_mask_downscale
#             attn_tokens = []
#             for idx in token_indices:
#                 if idx >= key_len:
#                     continue
#                 token_attn = attn[..., idx]
#                 if ds > 1 and grid % ds == 0:
#                     token_attn = token_attn.view(batch, heads, grid // ds, ds, grid // ds, ds).sum(dim=(3, 5))
#                     token_attn = token_attn.view(batch, heads, (grid // ds) * (grid // ds))
#                     mask_use = mask_flat[:, :, ::ds]
#                 else:
#                     token_attn = token_attn.view(batch, heads, grid * grid)
#                     mask_use = mask_flat
#                 token_attn = token_attn / token_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
#                 outside_mass = (token_attn * (1 - mask_use)).sum(dim=-1)
#                 inside_mass = (token_attn * mask_use).sum(dim=-1)
#                 loss = 0.0
#                 if self.attn_entropy_outside_weight > 0.0:
#                     loss = loss + self.attn_entropy_outside_weight * outside_mass.mean()
#                 if self.attn_entropy_inside_weight > 0.0:
#                     loss = loss - self.attn_entropy_inside_weight * torch.log(inside_mass.clamp(min=1e-6)).mean()
#                 if isinstance(loss, torch.Tensor):
#                     self._entropy_loss_terms.append(loss)

#     def _attention_entropy_guidance(self, latent, t, mode="fusion"):
#         if not self.enable_attention_entropy or self.attn_entropy_weight <= 0.0:
#             return
#         if not self._entropy_masks_ready or not self._entropy_token_map:
#             return
#         if getattr(self, "_entropy_disabled", False):
#             return
#         t_value = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
#         if t_value < self.attn_entropy_min_step:
#             return
#         if self.attn_entropy_max_step >= 0 and t_value > self.attn_entropy_max_step:
#             return
#         if isinstance(t, torch.Tensor):
#             timestep = t
#         else:
#             dtype = self.scheduler.timesteps.dtype if hasattr(self.scheduler, "timesteps") else torch.float32
#             timestep = torch.tensor([t], device=self.unet.device, dtype=dtype)[0]
#         steps = max(1, self.attn_entropy_steps)
#         lr = self.attn_entropy_lr
#         text_cond, text_cond_pool = self.text_embeds
#         text_cond_local = text_cond.clone()
#         if text_cond.shape[0] <= 1:
#             return
#         branch_indices = [1]  # Only guide multi-condition branch
#         print(f"[AttnLoss] 仅约束多概念分支 {branch_indices}")
        
#         # ✅ 打印详细调试信息
#         print(f"[AttnLoss] mode={mode}, text_cond.shape={text_cond.shape}")
#         print(f"[AttnLoss] masks_ready={self._entropy_masks_ready}")
#         print(f"[AttnLoss] token_map={list(self._entropy_token_map.keys())}")
#         print(f"[AttnLoss] subject_token_ids={self.subject_token_ids}")
        
#         latent_model_input = latent.detach()
#         target_device = getattr(self.unet, "device", latent_model_input.device)
#         latent_model_input = latent_model_input.to(device=target_device, dtype=self.unet.dtype).contiguous()
#         device_type = target_device.type if hasattr(target_device, "type") else "cuda"

#         use_checkpoint = self.attn_entropy_enable_checkpointing
#         checkpoint_was_enabled = getattr(self.unet, "is_gradient_checkpointing", False)
#         checkpoint_enabled = False
#         if use_checkpoint and not checkpoint_was_enabled and hasattr(self.unet, "enable_gradient_checkpointing"):
#             self.unet.enable_gradient_checkpointing()
#             checkpoint_enabled = True

#         subject_tensor = torch.tensor(
#             self.subject_token_ids, device=text_cond_local.device, dtype=torch.long
#         ) if self.subject_token_ids else None

#         def _inject_subject_tokens(base_tensor, tokens):
#             if subject_tensor is None or subject_tensor.numel() == 0:
#                 return base_tensor
#             row = base_tensor[1].clone()
#             row.index_copy_(0, subject_tensor, tokens)
#             updated = base_tensor.clone()
#             updated[1] = row
#             return updated

#         try:
#             for step_idx in range(steps):
#                 # ✅ 使用ToMe风格的梯度优化：对比两次UNet forward
#                 with torch.enable_grad():
#                     # === 步骤1：用当前stokens执行UNet（带梯度） ===
#                     stokens = torch.stack(
#                         [text_cond_local[1, idx].clone() for idx in self.subject_token_ids]
#                     ).to(text_cond_local.device, dtype=text_cond_local.dtype).requires_grad_(True)
                    
#                     text_cond_step = _inject_subject_tokens(text_cond_local, stokens)
#                     text_embed = text_cond_step[branch_indices]
#                     text_embed_pool = text_cond_pool[branch_indices]
#                     cond_kwargs = {
#                         "time_ids": self.add_time_ids.repeat(text_embed.shape[0], 1),
#                         "text_embeds": text_embed_pool,
#                     }
                    
#                     # ✅ 启用attention收集（用于监控，不用于loss）
#                     self._stored_attention_maps = []
#                     self._collecting_attention = True
#                     self._reset_entropy_accumulator()
                    
#                     print(f"[AttnLoss] step {step_idx + 1}: _collecting_attention=True")
                    
#                     # ✅ 执行UNet（保持梯度）
#                     noise_pred_current = self.unet(
#                         latent_model_input,
#                         timestep,
#                         encoder_hidden_states=text_embed,
#                         added_cond_kwargs=cond_kwargs,
#                     )["sample"]
                    
#                     self._collecting_attention = False
                    
#                     print(f"[AttnLoss] step {step_idx + 1}/{steps}: 收集了 {len(self._stored_attention_maps)} 个attention maps")
                    
#                     # === 步骤2：计算spatial loss（用于决定优化方向） ===
#                     loss_spatial_value = 0.0
#                     num_violations = 0
                    
#                     if len(self._stored_attention_maps) > 0:
#                         for map_info in self._stored_attention_maps[:10]:
#                             attn_detached = map_info['attn']
#                             query_len = map_info['query_len']
#                             key_len = map_info['key_len']
#                             heads = map_info['heads']
                            
#                             grid = int(math.sqrt(query_len))
#                             if grid * grid != query_len:
#                                 continue
                            
#                             batch = attn_detached.shape[0] // heads
#                             attn_spatial = attn_detached.view(batch, heads, query_len, key_len)
#                             attn_avg = attn_spatial.mean(dim=1)[0]
#                             attn_2d = attn_avg.view(grid, grid, 77)
                            
#                             for label, token_indices in self._entropy_token_map.items():
#                                 concept_attn_list = []
#                                 for token_idx in token_indices:
#                                     if token_idx >= key_len:
#                                         continue
#                                     token_attn = attn_2d[:, :, token_idx]
#                                     concept_attn_list.append(token_attn)
                                
#                                 if not concept_attn_list:
#                                     continue
                                
#                                 concept_attn = torch.stack(concept_attn_list).mean(dim=0)
#                                 concept_attn_norm = concept_attn / (concept_attn.sum() + 1e-8)
                                
#                                 mask_flat = self._get_entropy_mask(label, grid)
#                                 if mask_flat is not None:
#                                     mask_2d = mask_flat.view(grid, grid)
#                                     outside_mass = (concept_attn_norm * (1 - mask_2d)).sum()
#                                     loss_spatial_value += outside_mass.item()
#                                     num_violations += 1
                    
#                     # === 步骤3：构建per-concept的spatial-aware loss ===
#                     # ✅ 最终方案：为每个概念单独计算loss，使用其spatial violation
                    
#                     # 收集每个概念的spatial loss
#                     concept_losses = {}  # {label: loss_value}
#                     for label in self._entropy_token_map.keys():
#                         concept_losses[label] = 0.0
                    
#                     # 重新遍历maps，分别统计每个概念的violation
#                     for map_info in self._stored_attention_maps[:10]:
#                         attn_detached = map_info['attn']
#                         query_len = map_info['query_len']
#                         key_len = map_info['key_len']
#                         heads = map_info['heads']
                        
#                         grid = int(math.sqrt(query_len))
#                         if grid * grid != query_len:
#                             continue
                        
#                         batch = attn_detached.shape[0] // heads
#                         attn_spatial = attn_detached.view(batch, heads, query_len, key_len)
#                         attn_avg = attn_spatial.mean(dim=1)[0]
#                         attn_2d = attn_avg.view(grid, grid, 77)
                        
#                         for label, token_indices in self._entropy_token_map.items():
#                             concept_attn_list = []
#                             for token_idx in token_indices:
#                                 if token_idx >= key_len:
#                                     continue
#                                 token_attn = attn_2d[:, :, token_idx]
#                                 concept_attn_list.append(token_attn)
                            
#                             if not concept_attn_list:
#                                 continue
                            
#                             concept_attn = torch.stack(concept_attn_list).mean(dim=0)
#                             concept_attn_norm = concept_attn / (concept_attn.sum() + 1e-8)
                            
#                             mask_flat = self._get_entropy_mask(label, grid)
#                             if mask_flat is not None:
#                                 mask_2d = mask_flat.view(grid, grid)
#                                 outside_mass = (concept_attn_norm * (1 - mask_2d)).sum()
#                                 concept_losses[label] += outside_mass.item()
                    
#                     # 构建per-token的loss
#                     loss_total = torch.tensor(0.0, device=stokens.device, dtype=stokens.dtype)
#                     token_idx_to_label = {}  # 映射token index到concept label
#                     for label, indices in self._entropy_token_map.items():
#                         for local_idx, global_idx in enumerate(indices):
#                             # 找到这个global_idx在subject_token_ids中的位置
#                             if global_idx in self.subject_token_ids:
#                                 stoken_idx = self.subject_token_ids.index(global_idx)
#                                 token_idx_to_label[stoken_idx] = label
                    
#                     # 为每个stoken添加对应concept的spatial loss
#                     for stoken_idx, label in token_idx_to_label.items():
#                         if stoken_idx < len(stokens):
#                             # 该stoken的能量，加权于其concept的spatial violation
#                             token_energy = (stokens[stoken_idx] ** 2).sum()
#                             loss_total = loss_total + self.attn_entropy_weight * concept_losses[label] * token_energy
                    
#                     print(f"[AttnLoss] concept_losses={concept_losses}")
#                     print(f"[AttnLoss] total_spatial={sum(concept_losses.values()):.6f}, loss_total={loss_total.item():.6f}")
#                     print(f"[AttnLoss] weight={self.attn_entropy_weight}, lr={lr}")
                    
#                     if not loss_total.requires_grad or loss_total.item() == 0:
#                         print(f"[AttnLoss] WARNING: loss不可优化，跳过")
#                         break
                    
#                     # 计算梯度
#                     grad = torch.autograd.grad(loss_total, stokens, retain_graph=False)[0]
#                     grad_norm_val = grad.norm().item()
#                     print(f"[AttnLoss] grad_norm={grad_norm_val:.6f}")
                    
#                     # ✅ 如果梯度太小，警告
#                     if grad_norm_val < 1e-5:
#                         print(f"[AttnLoss] WARNING: 梯度过小，可能需要增大学习率或weight")
                
#                 # 更新stokens并记录变化
#                 stokens_old_norm = stokens.norm().item()
#                 stokens = (stokens - lr * grad).detach()
#                 stokens_new_norm = stokens.norm().item()
#                 delta_norm = abs(stokens_new_norm - stokens_old_norm)
                
#                 print(f"[AttnLoss] stoken update: old_norm={stokens_old_norm:.6f}, new_norm={stokens_new_norm:.6f}, delta={delta_norm:.6f}")
                
#                 text_cond_local = _inject_subject_tokens(text_cond_local, stokens.to(text_cond_local.dtype))
                
#                 del text_embed, text_embed_pool, cond_kwargs, noise_pred_current, loss_total, token_energy, grad
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
                    
#         except torch.cuda.OutOfMemoryError:
#             print("[AttnLoss] WARNING: OOM during attention-entropy guidance, skipping this step.")
#             torch.cuda.empty_cache()
#             self._entropy_disabled = True
#             del latent_model_input
#             return
#         finally:
#             if checkpoint_enabled and hasattr(self.unet, "disable_gradient_checkpointing"):
#                 self.unet.disable_gradient_checkpointing()

#         self.text_embeds = (text_cond_local, text_cond_pool)
#         del latent_model_input
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def _apply_attention_guidance_at_fusion_start(self, latent, t):
#         """
#         在融合开始前(t=31)应用一次注意力引导
#         确保panda*和cat*等概念在正确的空间位置
#         """
#         if not self.enable_attention_entropy or not self._entropy_masks_ready:
#             return
#         # 使用attention entropy指引来直接更新subject tokens（仅需uncond+multi分支）
#         self._attention_entropy_guidance(latent, t, mode="after_binding")

#     def _apply_attention_guidance_after_binding(self, latent, t):
#         """
#         在语义绑定完成后应用注意力引导
#         此时cat*和panda*等super tokens已经形成，需要约束它们的空间位置
#         """
#         if not self.enable_attention_entropy or not self._entropy_masks_ready:
#             return
#         self._attention_entropy_guidance(latent, t, mode="after_binding")

#     def upcast_vae(self):
#         dtype = self.vae.dtype  # 记录当前 VAE 的数据类型（通常为 float16）
#         self.vae.to(dtype=torch.float32)  # 将 VAE 整体转换为 float32 精度，提升数值稳定性
#         use_torch_2_0_or_xformers = isinstance(
#             self.vae.decoder.mid_block.attentions[0].processor,
#             (
#                 AttnProcessor2_0,           # 判断 attention processor 是否为 torch2.0 或 xformers 相关类型
#                 XFormersAttnProcessor,
#                 FusedAttnProcessor2_0,
#             ),
#         )
#         # 如果使用 xformers 或 torch2.0 的注意力机制，则部分模块可以继续用原始精度（如 float16），节省显存
#         if use_torch_2_0_or_xformers:
#             self.vae.post_quant_conv.to(dtype)         # post_quant_conv 层恢复为原始精度
#             self.vae.decoder.conv_in.to(dtype)         # decoder 的输入卷积层恢复为原始精度
#             self.vae.decoder.mid_block.to(dtype)       # decoder 的中间块恢复为原始精度
        
#     def find_disc(self,embed,embed2):
#         '''
#         该方法的主要功能是：
#         给定两个嵌入向量 embed 和 embed2，分别在两个文本编码器的 token embedding 空间中，
#         查找与它们最相近（点积最大）的 token embedding。
#         这种查找可以用于分析自定义 token embedding 与原始词表 embedding 的相似性，或者调试 embedding 的分布情况。

#         '''
#         with torch.no_grad():  # 关闭梯度计算，加速推理且节省显存
#             token_embedding = self.text_encoder.get_input_embeddings()  # 获取第一个文本编码器的 token embedding 层
#             token_embedding2 = self.text_encoder_2.get_input_embeddings()  # 获取第二个文本编码器的 token embedding 层

#             embedding_matrix = token_embedding.weight  # 获取第一个编码器的所有 token embedding 权重（形状：[vocab_size, hidden_dim]）
#             embedding_matrix2 = token_embedding2.weight  # 获取第二个编码器的所有 token embedding 权重

#             embed = embed.unsqueeze(0)  # 将输入的 embed 扩展 batch 维度，变成 shape [1, hidden_dim]
#             embed2 = embed2.unsqueeze(0)  # 同理，扩展 embed2

#             # 在 embedding_matrix 中查找与 embed 最相近的 token（余弦相似度/点积最大）
#             hits = semantic_search(
#                 embed,                      # 查询向量
#                 embedding_matrix.float(),   # 语料库向量（所有 token embedding）
#                 query_chunk_size=1,         # 查询分块大小
#                 top_k=1,                    # 只取最相近的一个
#                 score_function=dot_score    # 使用点积作为相似度分数
#             )
#             # 在 embedding_matrix2 中查找与 embed2 最相近的 token
#             hits2 = semantic_search(
#                 embed2,
#                 embedding_matrix2.float(),
#                 query_chunk_size=1,
#                 top_k=1,
#                 score_function=dot_score
#             )

#             # 提取最相近 token 的索引（corpus_id），并转为 tensor
#             nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=embed.device)
#             nn_indices2 = torch.tensor([hit[0]["corpus_id"] for hit in hits2], device=embed.device)
            
#     @torch.no_grad()
#     def get_text_embeds(self, prompt, negative_prompt, device="cuda"):     
#         """
#         生成正向和负向提示词的文本嵌入，用于扩散模型的条件控制
        
#         该方法会分别对正向 prompt 和负向 prompt 进行编码，得到它们在两个文本编码器下的 embedding 表示。
#         最终返回的 text_embeddings 和 pooled_text_embeddings，
#         会被用作扩散模型（如 Stable Diffusion）在生成图像时的条件输入。
        
#         Args:
#             prompt: 正向提示词列表，包含原始prompt和修饰后的概念prompt
#             negative_prompt: 负向提示词列表，用于classifier-free guidance
#             device: 指定计算设备，默认 "cuda"
        
#         Returns:
#             text_embeddings: 序列嵌入 [batch_size, 77, 2048]
#             pooled_text_embeddings: 全局嵌入 [batch_size, 2048]
            
#             其中batch_size = len(prompt) + len(negative_prompt)
#             索引0: 负向prompt嵌入
#             索引1+: 正向prompt嵌入（按prompt列表顺序）
        
#         注意：
#             - 使用@torch.no_grad()禁用梯度计算，节省显存和计算时间
#             - 双编码器架构提供更强的文本理解能力
#             - 负向嵌入在前，正向嵌入在后，用于classifier-free guidance
#         """
#         # 该方法用于获取正向和负向 prompt 的文本嵌入（embedding），用于扩散模型的条件控制
#         # prompt: 正向提示词（可以是字符串或字符串列表）
#         # negative_prompt: 负向提示词（通常用于 classifier-free guidance）
#         # device: 指定计算设备，默认 "cuda"

#         # 对正向 prompt 进行编码，得到其 embedding 和 pooled embedding
#         prompt_embeds, pooled_prompt_embeds = encode_prompt(
#             text_encoders=[self.text_encoder, self.text_encoder_2],   # 使用两个文本编码器
#             tokenizers=[self.tokenizer, self.tokenizer_2],            # 对应的两个分词器
#             prompt=prompt,                                            # 输入正向 prompt
#             text_input_ids_list=None                                  # 不直接传 token id
#         )
#         # 对负向 prompt 进行编码，得到其 embedding 和 pooled embedding
#         uncond_embeds, pooled_uncond_embeds = encode_prompt(
#             text_encoders=[self.text_encoder, self.text_encoder_2],   # 同样用两个编码器
#             tokenizers=[self.tokenizer, self.tokenizer_2],            # 两个分词器
#             prompt=negative_prompt,                                   # 输入负向 prompt
#             text_input_ids_list=None
#         )
#         # 将负向和正向的 embedding 拼接，形成最终的文本条件输入
#         # 负向嵌入在前，用于classifier-free guidance的无条件分支
#         text_embeddings = torch.cat([uncond_embeds, prompt_embeds])
#         # 将 pooled embedding 也拼接
#         pooled_text_embeddings = torch.cat([pooled_uncond_embeds, pooled_prompt_embeds])
#         # 返回拼接后的 embedding 和 pooled embedding
#         return text_embeddings, pooled_text_embeddings
    
#     def prepare_extra_step_kwargs(self, generator, eta):
#         '''
#         该方法根据当前 scheduler（采样器）的 step 方法签名，
#         动态判断是否需要传递 eta 和 generator 参数，并将它们组织成一个字典返回。
#       这样做可以兼容不同版本或不同类型的 scheduler，避免因参数不兼容导致报错
#         '''
#         # 该方法用于为扩散采样器 scheduler 的 step 函数准备额外的参数字典
#         # generator: 随机数生成器（用于可复现性）
#         # eta: 采样噪声参数（影响采样的随机性）

#         # 判断 scheduler 的 step 方法是否接受 eta 参数
#         accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
#         extra_step_kwargs = {}  # 初始化参数字典
#         if accepts_eta:
#             extra_step_kwargs["eta"] = eta  # 如果支持，则添加 eta 参数

#         # 判断 scheduler 的 step 方法是否接受 generator 参数
#         accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
#         if accepts_generator:
#             extra_step_kwargs["generator"] = generator  # 如果支持，则添加 generator 参数
#         return extra_step_kwargs  # 返回最终的参数字典

#     def _build_base_filename(self):
#         """Build a stable filename stem from key sampling params."""
#         resample_steps = int(getattr(self.config, "resampling_steps", 0))
#         t_cond_ratio = float(getattr(self.config, "t_cond", 0.0))
#         modifier = getattr(self.config, "modifier_token", "").replace("+", "_")
#         guidance = float(getattr(self.config, "guidance_scale", 7.5))
#         seed = int(getattr(self.config, "seed", 0))

#         safe_modifier = re.sub(r'[<>:"/\\\\|?*\\x00-\\x1F]', "", modifier)
#         safe_modifier = safe_modifier.strip().strip(".").replace(" ", "_")
#         if not safe_modifier:
#             safe_modifier = "output"

#         runner = os.path.splitext(os.path.basename(sys.argv[0]))[0]
#         runner = runner.strip().strip(".").replace(" ", "_")
#         prefix = f"{runner}__" if runner else ""

#         filename = f"{prefix}{safe_modifier}_g{guidance:.2f}_t{t_cond_ratio:.2f}_res{resample_steps}_seed{seed}.png"
#         stem, ext = os.path.splitext(filename)
#         return stem, ext if ext else ".png"

#     @torch.no_grad()
#     def _decode_latent_to_pil(self, latent):
#         """
#         Decode latent with the same SDXL-safe path used for final output.
#         Returns: (pil_images, decoded_tensor)
#         """
#         x = latent
#         needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

#         if needs_upcasting:
#             self.upcast_vae()
#         vae_in_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
#         vae_device = next(iter(self.vae.post_quant_conv.parameters())).device
#         x = x.to(device=vae_device, dtype=vae_in_dtype)

#         has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
#         has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
#         if has_latents_mean and has_latents_std:
#             latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(x.device, x.dtype)
#             latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(x.device, x.dtype)
#             x = x * latents_std / self.vae.config.scaling_factor + latents_mean
#         else:
#             x = x / self.vae.config.scaling_factor

#         decoded_latent = self.vae.decode(x, return_dict=False)[0]
#         decoded_latent = torch.nan_to_num(decoded_latent, nan=0.0, posinf=1.0, neginf=-1.0)
#         decoded_latent = decoded_latent.clamp_(-1.0, 1.0)

#         if needs_upcasting:
#             self.vae.to(dtype=torch.float16)

#         images = self.image_processor.postprocess(decoded_latent, output_type='pil')
#         return images, decoded_latent
    
#     @torch.no_grad()
#     def decode_latent(self, latent):
#         '''
#         该方法用于将潜在表示（latent）解码为图像，
#         它首先将潜在表示缩放回原始尺度，然后使用 VAE 解码器生成图像。
#         '''
#         with torch.autocast(device_type='cuda', dtype=torch.float32):
#             latent = 1 / 0.18215 * latent
#             img = self.vae.decode(latent).sample
#             img = (img / 2 + 0.5).clamp(0, 1)
#         return img
    
#     def alpha(self, t):
#         '''
#         该方法用于计算扩散模型的 alpha 值，
#         它根据当前时间步 t 从 scheduler 中获取 alpha 值，
#         如果 t 小于 0，则使用 final_alpha_cumprod 作为默认值。
#         '''
#         at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
#         return at
    
#     @torch.no_grad()
#     def denoise_step(self, x, t):
#         """
#         MultiCompose多概念融合的核心去噪步骤
        
#         这个函数实现了两阶段的扩散去噪过程：
#         1. 融合阶段：使用多个概念特定的UNet进行并行处理
#         2. 融合后阶段：使用标准UNet进行最终生成
        
#         Args:
#             x: 当前的潜变量（latent）[1, 4, 128, 128] - 当前步的图像表示
#             t: 当前的采样步数 (如 50, 49, 48, ... 1)
        
#         Returns:
#             denoised_latent: 去噪后的潜变量，用于下一步计算
        
#         工作流程：
#             1. 根据时间步判断进入融合阶段还是融合后阶段
#             2. 在融合阶段：使用多概念UNet进行并行去噪
#             3. 在融合后阶段：使用标准UNet进行最终生成
#             4. 生成掩码用于概念分离
#          Prompt结构说明：
#             text_embeds包含5个文本嵌入：
#             [0]: 负向prompt "blurry, ugly, black, low res, unrealistic, blurry face"
#             [1]: 原始prompt "photo of a panda and a teddybear playing with a ball, castle background"
#             [2]: 熊猫概念 "photo of a <panda1> panda playing with a ball, castle background"
#             [3]: 泰迪熊概念 "photo of a <teddybear1> teddybear playing with a ball, castle background"
#             [4]: 城堡概念 "photo of a panda and a teddybear playing with a ball, <castle1> waterfall background"
#         """
#         text_embed_cond, text_embed_cond_pool = self.text_embeds  # 获取正负 prompt 的嵌入和 pooled 嵌入

#         next_t = t - self.skip  # 计算下一个采样步
#         at = self.alpha(t)      # 当前步的 alpha
#         at_next = self.alpha(next_t)  # 下一个步的 alpha

#         sizes = x.shape  # 记录当前潜变量的形状
#         # if self.masks is not None:
#         #     register_time(self, t.item(),self.masks)
#         # else:
#         register_time(self, t.item())  # 记录当前步（如保存中间状态等） # [1, 4, 128, 128]
       
#         # "photo of a panda and a teddybear playing with a ball, castle background",
#         # "photo of a <panda1> panda playing with a ball, castle background",
#         # "photo of a <teddybear1> teddybear playing with a ball, castle background",
#         # "photo of a panda and a teddybear playing with a ball, <castle1> castle background"

#         if t <= self.t_cond_cur:  # 如果当前步在融合阶段
#             """
#                 融合阶段：使用多个概念特定的UNet进行并行处理
                
#                 在这个阶段，系统会：
#                 1. 使用多个概念UNet分别处理每个概念
#                 2. 通过掩码将不同概念的去噪结果融合
#                 3. 实现多概念的协调生成
#                 使用索引[0, 2, 3, 4]的文本嵌入：
#                 - [0]: 负向prompt用于classifier-free guidance
#                 - [2]: 熊猫概念prompt，使用<panda1>修饰符
#                 - [3]: 泰迪熊概念prompt，使用<teddybear1>修饰符
#                 - [4]: 城堡概念prompt，使用<castle1>修饰符
#             """
#             text_embed_uncond = text_embed_cond[0].unsqueeze(0)  # ȡ������Ƕ��
#             # ���ݣ�����prompt "blurry, ugly, black, low res, unrealistic, blurry face"

#             text_embed_concept = text_embed_cond[2:]             # ȡ���������Ƕ��
#             # ���ݣ���è����prompt "photo of a <panda1> panda playing with a ball, castle background"
#             # ���ݣ�̩���ܸ���prompt "photo of a <teddybear1> teddybear playing with a ball, castle background"
#             # ���ݣ��Ǳ�����prompt "photo of a panda and a teddybear playing with a ball, <castle1> castle background"

#             text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)  # pooled ������Ƕ��
#             text_embed_concept_pool = text_embed_cond_pool[2:]             # pooled ����Ƕ��

#             # 4����֧��1������ + 3������
#             text_embed = torch.cat([text_embed_uncond, text_embed_concept], dim=0)  # ƴ������Ƕ�� # [4, 77, 2048]
#             text_embed_pool = torch.cat([text_embed_uncond_pool, text_embed_concept_pool], dim=0)  # ƴ�� pooled Ƕ�� # [4, 2048]

#             if self.fusion_low_mem:
#                 # �ʹ�ʵ��ģʽ�������ȼ��㲻��������֧��Ȼ��˳��ͨ����������
#                 cond_kwargs_uncond = {
#                     "time_ids": self.add_time_ids[:1],
#                     "text_embeds": text_embed_uncond_pool,
#                 }
#                 noise_pred_uncond_only = self.unet(
#                     x, t, encoder_hidden_states=text_embed_uncond, added_cond_kwargs=cond_kwargs_uncond
#                 )["sample"]
#                 concept_preds = []
#                 repeat_time_ids = self.add_time_ids.repeat(2, 1)
#                 for idx in range(text_embed_concept.shape[0]):
#                     concept_embed = text_embed_concept[idx:idx + 1]
#                     concept_pool = text_embed_concept_pool[idx:idx + 1]
#                     pair_embed = torch.cat([text_embed_uncond, concept_embed], dim=0)
#                     pair_pool = torch.cat([text_embed_uncond_pool, concept_pool], dim=0)
#                     cond_kwargs_pair = {
#                         "time_ids": repeat_time_ids,
#                         "text_embeds": pair_pool,
#                     }
#                     latent_pair = torch.cat([x, x])
#                     pair_noise = self.unet(
#                         latent_pair, t, encoder_hidden_states=pair_embed, added_cond_kwargs=cond_kwargs_pair
#                     )["sample"]
#                     concept_preds.append(pair_noise[1:2])
#                 noise_pred = torch.cat([noise_pred_uncond_only] + concept_preds, dim=0)
#             else:
#                 # ��չǱ���������������
#                 latent_model_input = torch.cat([x] * (self.concept_num + 1))  # ��չ batch����������# [4, 4, 128, 128]

#                 # ����UNet�Ķ���������ʱ���������ı�Ƕ��������
#                 unet_added_conditions = {"time_ids": self.add_time_ids.repeat(text_embed.shape[0], 1)}  # ����ʱ������ # [4, 1]
#                 unet_added_conditions.update({"text_embeds": text_embed_pool})  # �����ı�Ƕ������ # [4, 2048]

#                 # ���� UNet���õ�����Ԥ��
#                 noise_pred = self.unet(
#                     latent_model_input, t, encoder_hidden_states=text_embed, added_cond_kwargs=unet_added_conditions
#                 )["sample"]  # ���� UNet���õ�����Ԥ�� # [4, 4, 128, 128] - 4����֧������Ԥ��

#             # 噪声预测结果：
#             # noise_pred[0]: 负向prompt的噪声预测
#             # noise_pred[1]: 熊猫概念的噪声预测
#             # noise_pred[2]: 泰迪熊概念的噪声预测
#             # noise_pred[3]: 城堡概念的噪声预测
#         else:  # 否则，进入融合后阶段
#             """
#            内容感知采样
            
#             在这个阶段，系统会：
#             1. 使用融合后的UNet进行标准扩散生成
#             2. 可选择性进行重采样和跳步采样
#             3. 生成最终的高质量图像
            
#             Prompt处理逻辑：
#             使用索引[0, 1]的文本嵌入：
#             - [0]: 负向prompt用于classifier-free guidance
#             - [1]: 原始prompt "photo of a panda and a teddybear playing with a ball, castle background"
#             """
        
#             text_embed_uncond = text_embed_cond[0].unsqueeze(0)  # 负向文本嵌入 [1, 77, 2048]
#             # 内容：负向prompt "blurry, ugly, black, low res, unrealistic, blurry face"
            
#             text_embed_multi = text_embed_cond[1].unsqueeze(0)   # 多概念融合文本嵌入 [1, 77, 2048]
#             # 内容：原始prompt "photo of a panda and a teddybear playing with a ball, castle background"
            
#             text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)  # 负向pooled嵌入 [1, 2048]
#             text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)   # 多概念pooled嵌入 [1, 2048]

#             if t == self.start_t:  # 如果是采样起始步
#                 """
#                 采样起始步：需要额外的单独概念处理
                
#                 在第一步，系统会：
#                 1. 使用多概念融合UNet
#                 2. 使用单独的每个概念UNet
#                 3. 通过重采样优化生成质量
                
#                 Prompt处理逻辑：
#                 使用索引[0, 1, 2, 3]的文本嵌入：
#                 - [0]: 负向prompt
#                 - [1]: 原始prompt
#                 - [2]: 熊猫概念prompt（单独）
#                 - [3]: 泰迪熊概念prompt（单独）
#                 """
            
#                 text_embed_cond_single, text_embed_cond_pool_single = self.text_embeds_single  # [3, 77, 2048] 和 [3, 2048]
#                 # 单独概念prompt：
#                 # [0]: 负向prompt "blurry, ugly, black, low res, unrealistic, blurry face"
#                 # [1]:"photo of a panda playing with a ball, castle background",
#                 # [2]:"photo of a teddybear playing with a ball, castle background"
                
#                 text_embed_cond_single = text_embed_cond_single[1:]  # 跳过第一个（无条件）[2, 77, 2048]
#                 text_embed_cond_pool_single = text_embed_cond_pool_single[1:]  # 同上 [2, 2048]
                
#                 latent_model_input = torch.cat([x] * (self.concept_num + 1))  # [4, 4, 128, 128]
                
#                 text_embed = torch.cat([
#                     text_embed_uncond,    # 负向文本
#                     text_embed_multi,     # 多概念融合文本
#                     text_embed_cond_single  # 单独概念文本
#                 ], dim=0)  # [4, 77, 2048]
                
#                 text_embed_pool = torch.cat([
#                     text_embed_uncond_pool,
#                     text_embed_multi_pool,
#                     text_embed_cond_pool_single
#                 ], dim=0)  # [4, 2048]

#                 if (
#                     self.sem_binding_steps > 0
#                     and getattr(self, "_binding_active", False)
#                     and self.binding_subject_indices
#                     and self.sem_binding_applied_steps < self.sem_binding_max_steps
#                 ):
#                     with torch.enable_grad():
#                         self._semantic_binding_step(x, t)
#                     self.sem_binding_applied_steps += 1
#                     self.sem_binding_applied = True
#                     text_embed_cond, text_embed_cond_pool = self.text_embeds
#                     text_embed_uncond = text_embed_cond[0].unsqueeze(0)
#                     text_embed_multi = text_embed_cond[1].unsqueeze(0)
#                     text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)
#                     text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)
#                     text_embed = torch.cat([
#                         text_embed_uncond,
#                         text_embed_multi,
#                         text_embed_cond_single
#                     ], dim=0)
#                     text_embed_pool = torch.cat([
#                         text_embed_uncond_pool,
#                         text_embed_multi_pool,
#                         text_embed_cond_pool_single
#                     ], dim=0)
#             else:  # 其它步只用无条件和多概念嵌入
#                 """
#                 其他步骤：使用简化的两分支处理
                
#                 在后续步骤中，系统只使用：
#                 1. 负向文本分支
#                 2. 多概念融合文本分支
                
#                 Prompt处理逻辑：
#                 使用索引[0, 1]的文本嵌入：
#                 - [0]: 负向prompt
#                 - [1]: 原始prompt
#                 """
#                 latent_model_input = torch.cat([x] + [x])  # [2, 4, 128, 128] - 两分支
#                 text_embed = torch.cat([text_embed_uncond, text_embed_multi], dim=0)  # [2, 77, 2048]
#                 text_embed_pool = torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0)  # [2, 2048]

#                 if (
#                     self.sem_binding_steps > 0
#                     and getattr(self, "_binding_active", False)
#                     and self.binding_subject_indices
#                     and self.sem_binding_applied_steps < self.sem_binding_max_steps
#                 ):
#                     with torch.enable_grad():
#                         self._semantic_binding_step(x, t)
#                     self.sem_binding_applied_steps += 1
#                     self.sem_binding_applied = True
#                     text_embed_cond, text_embed_cond_pool = self.text_embeds
#                     text_embed_uncond = text_embed_cond[0].unsqueeze(0)
#                     text_embed_multi = text_embed_cond[1].unsqueeze(0)
#                     text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)
#                     text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)
#                     text_embed = torch.cat([text_embed_uncond, text_embed_multi], dim=0)
#                     text_embed_pool = torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0)
        

#             unet_added_conditions = {"time_ids": self.add_time_ids.repeat(text_embed.shape[0], 1)}  # 时间条件
#             unet_added_conditions.update({"text_embeds": text_embed_pool})  # 文本嵌入条件

#             unet_added_conditions_single = {"time_ids": self.add_time_ids.repeat(text_embed[:2].shape[0], 1)}  # 前两个分支的时间条件
#             unet_added_conditions_single.update({"text_embeds": text_embed_pool[:2]})  # 前两个分支的文本嵌入

#             noise_pred = self.unet(
#                 latent_model_input, t, encoder_hidden_states=text_embed, added_cond_kwargs=unet_added_conditions
#             )['sample']  # 输入 UNet，得到噪声预测  # [2或4, 4, 128, 128] - 噪声预测

#         noise_pred_uncond = noise_pred[:1]  # 取出无条件分支的噪声预测 # [1, 4, 128, 128] - 无条件噪声预测

#         # ❌ 移除t=31的注意力引导（已在语义绑定后执行）
#         # 注意力引导现在在重采样+语义绑定完成后执行，时机更合适
        
#         if t <= self.t_cond_cur:  # 如果在融合阶段
#             """
#             融合阶段的噪声处理：使用掩码融合多个概念
            
#             在这个阶段，系统会：
#             1. 为每个概念计算去噪结果
#             2. 使用掩码将不同概念的结果融合
#             3. 实现多概念的协调生成
            
#             Prompt处理逻辑：
#             使用4个分支的噪声预测：
#             - noise_pred[0]: 负向prompt的噪声预测
#             - noise_pred[1]: 熊猫概念的噪声预测
#             - noise_pred[2]: 泰迪熊概念的噪声预测
#             - noise_pred[3]: 城堡概念的噪声预测
#             """
#             # 简单掩码累加版本：直接累加各概念的掩码加权去噪结果，不进行归一化
#             # 这种方式假设掩码之间没有重叠，适用于外部boxes定义的场景
#             denoised_tweedie = 0

#             for cc in range(self.concept_num):
#                 noise_pred_cond = noise_pred[(1 + cc):(2 + cc)]
#                 noise_pred_concept = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
#                 mask = self.masks[cc].unsqueeze(0)
#                 # 直接累加掩码加权的去噪结果（与fusion_sampling.py保持一致）
#                 denoised_tweedie += mask * ((x - (1 - at).sqrt() * noise_pred_concept) / at.sqrt())

#             # 掩码融合过程：
#             # self.masks[0]: 熊猫概念的空间掩码
#             # self.masks[1]: 泰迪熊概念的空间掩码
#             # self.masks[2]: 城堡概念的空间掩码
#             # 通过空间掩码确保每个概念在正确的位置出现
#         else:  # 融合后阶段
#             """
#                 融合后阶段的噪声处理：使用标准扩散生成
                
#                 在这个阶段，系统会：
#                 1. 使用标准UNet进行去噪
#                 2. 可选择性进行重采样优化
#                 3. 生成最终的高质量图像
#                 使用2个分支的噪声预测：
#             - noise_pred[0]: 负向prompt的噪声预测
#             - noise_pred[1]: 原始prompt的噪声预测
#             """
#             if t == self.start_t:  # 如果是采样起始步
#                 if self.config.resampling_steps > 0:  # 如果需要重采样
#                     """
#                     重采样过程：通过多次迭代优化生成质量
                    
#                     重采样的目的是：
#                     1. 提高生成质量
#                     2. 减少概念间的冲突
#                     3. 优化多概念融合效果
                    
#                     Prompt处理逻辑：
#                     在重采样过程中，使用4个分支的噪声预测：
#                     - noise_pred[0]: 负向prompt的噪声预测
#                     - noise_pred[1]: 原始prompt的噪声预测
#                     - noise_pred[2]: 熊猫概念prompt的噪声预测
#                     - noise_pred[3]: 泰迪熊概念prompt的噪声预测
#                     """
#                     clean_single = None
#                     clean_single_pool = None
#                     raw_seq, raw_pool = getattr(self, "text_embeds_raw", (None, None))
#                     if raw_seq is not None:
#                         raw_seq = raw_seq.to(device=self.unet.device, dtype=self.unet.dtype)
#                     if raw_pool is not None:
#                         raw_pool = raw_pool.to(device=self.unet.device, dtype=self.unet.dtype)
#                     if raw_seq is not None and raw_pool is not None:
#                         text_embed_uncond_raw = raw_seq[0:1]
#                         text_embed_multi_raw = raw_seq[1:2]
#                         text_embed_uncond_pool_raw = raw_pool[0:1]
#                         text_embed_multi_pool_raw = raw_pool[1:2]
#                     else:
#                         text_embed_uncond_raw = text_embed_uncond
#                         text_embed_multi_raw = text_embed_multi
#                         text_embed_uncond_pool_raw = text_embed_uncond_pool
#                         text_embed_multi_pool_raw = text_embed_multi_pool
#                     # 优先使用“重采样专用”的干净单概念（PROMPT_CLEAN_resample）；否则退回 PROMPT_CLEAN
#                     if getattr(self, "text_embeds_single_resample", None) is not None:
#                         clean_single, clean_single_pool = self.text_embeds_single_resample
#                         # debug
#                         if bool(getattr(self.config, "debug_binding", 0)):
#                             print("[Debug][Resample] using PROMPT_CLEAN_resample for single-concept guidance")
#                     elif self.text_embeds_single_clean is not None:
#                         clean_single, clean_single_pool = self.text_embeds_single_clean
#                         if bool(getattr(self.config, "debug_binding", 0)):
#                             print("[Debug][Resample] using PROMPT_CLEAN for single-concept guidance")
#                     clean_single = clean_single.to(device=self.unet.device, dtype=self.unet.dtype)
#                     clean_single_pool = clean_single_pool.to(device=self.unet.device, dtype=self.unet.dtype)
#                     if clean_single.shape[0] > 1:
#                         clean_single = clean_single[1:]
#                         clean_single_pool = clean_single_pool[1:]
#                     else:
#                         clean_single = None
#                         clean_single_pool = None
#                     multi_clean = getattr(self, "text_embeds_multi_clean", None)
#                     for _ in range(self.config.resampling_steps):
#                         print('resampling')
#                         noise_pred_uncond = noise_pred[:1]
#                         if multi_clean is not None:
#                             clean_seq, clean_pool = multi_clean
#                             latent_model_input_multi = torch.cat([x] + [x]).to(device=self.unet.device, dtype=self.unet.dtype)
#                             text_embed_multi_clean = torch.cat([text_embed_uncond_raw, clean_seq], dim=0)
#                             text_embed_multi_clean_pool = torch.cat([text_embed_uncond_pool_raw, clean_pool], dim=0)
#                             cond_kwargs_multi_clean = {
#                                 "time_ids": self.add_time_ids.repeat(text_embed_multi_clean.shape[0], 1),
#                                 "text_embeds": text_embed_multi_clean_pool,
#                             }
#                             noise_pred_clean_multi = self.unet(
#                                 latent_model_input_multi,
#                                 t,
#                                 encoder_hidden_states=text_embed_multi_clean,
#                                 added_cond_kwargs=cond_kwargs_multi_clean,
#                             )["sample"]
#                             noise_pred_mult = noise_pred_uncond + self.config.guidance_scale * (noise_pred_clean_multi[1:2] - noise_pred_uncond)
#                         else:
#                             noise_pred_mult = noise_pred[1:2]
#                             noise_pred_mult = noise_pred_uncond + self.config.guidance_scale * (noise_pred_mult - noise_pred_uncond)
#                         denoised_tweedie_mult = (x - (1 - at).sqrt() * noise_pred_mult) / at.sqrt()
#                         denoised_tweedie = (self.concept_num - 1) * denoised_tweedie_mult
#                         for cc in range(self.concept_num - 1):
#                             if clean_single is not None and cc < clean_single.shape[0]:
#                                 clean_embed = clean_single[cc:cc + 1]
#                                 clean_pool_embed = clean_single_pool[cc:cc + 1]
#                                 latent_model_input_clean = torch.cat([x] + [x])
#                                 clean_text_embed = torch.cat([text_embed_uncond_raw, clean_embed], dim=0)
#                                 clean_text_embed_pool = torch.cat([text_embed_uncond_pool_raw, clean_pool_embed], dim=0)
#                                 cond_kwargs_clean = {
#                                     'time_ids': self.add_time_ids.repeat(clean_text_embed.shape[0], 1),
#                                     'text_embeds': clean_text_embed_pool,
#                                 }
#                                 noise_pred_clean_batch = self.unet(
#                                     latent_model_input_clean,
#                                     t,
#                                     encoder_hidden_states=clean_text_embed,
#                                     added_cond_kwargs=cond_kwargs_clean,
#                                 )['sample']
#                                 noise_pred_single = noise_pred_uncond + self.config.guidance_scale * (
#                                     noise_pred_clean_batch[1:2] - noise_pred_uncond
#                                 )
#                             else:
#                                 noise_pred_single = noise_pred_uncond + self.config.guidance_scale * (
#                                     noise_pred[2 + cc:3 + cc] - noise_pred_uncond
#                                 )
#                             denoised_tweedie_single = (x - (1 - at).sqrt() * noise_pred_single) / at.sqrt()
#                             denoised_tweedie -= denoised_tweedie_single
#                         denoised_latent = at_next.sqrt() * denoised_tweedie + (1 - at_next).sqrt() * noise_pred_uncond
#                         latent_model_next = torch.cat([denoised_latent] + [denoised_latent])  # batch=2 # [2, 4, 128, 128]

#                         noise_pred_next = self.unet(
#                             latent_model_next, next_t, encoder_hidden_states=text_embed[:2], added_cond_kwargs=unet_added_conditions_single
#                         )['sample']  # 下一个步的噪声预测

#                         #多概念预测的噪声
#                         noise_pred_cond_next = noise_pred_next[1:2]
#                         noise_pred_uncond_next = noise_pred_next[:1]
#                         #预测的噪声    
#                         noise_pred_next = noise_pred_uncond_next + self.config.guidance_scale * (noise_pred_cond_next - noise_pred_uncond_next)
#                         # 计算去噪结果
#                         #xˆ[εθ(xt-1, t, c)] := (xt-1 − √1 − α ̄tεθ(xt-1, t, c))/√α ̄t-1,   x=z     zT −1
#                         # 反向采样：从z_{T-1}回到当前步z_T（文档“DDIM forward sampling”）
#                         denoised_tweedie_next = (denoised_latent - (1 - at_next).sqrt() * noise_pred_next) / at_next.sqrt()
#                         # 回到当前步
#                         #
#                         return_x = at.sqrt() * denoised_tweedie_next + (1 - at).sqrt() * noise_pred_uncond_next  # 回到当前步
#                         # 循环迭代：更新输入，准备下一次重采样（共self.config.resampling_steps次）
#                         latent_model_input = torch.cat([return_x] * (self.concept_num + 1))  # batch
#                         noise_pred = self.unet(
#                             latent_model_input, t, encoder_hidden_states=text_embed, added_cond_kwargs=unet_added_conditions
#                         )['sample']  # 再次采样
#                     x = return_x  # 更新 x
#                 if not getattr(self, "_binding_active", False):
#                     self._activate_token_binding()
#                     text_embed_cond, text_embed_cond_pool = self.text_embeds
#                 # 在融合阶段的若干时间步执行语义绑定，持续约束属性语义
#                 if (
#                     self.sem_binding_steps > 0
#                     and getattr(self, "_binding_active", False)
#                     and self.binding_subject_indices
#                     and self.sem_binding_applied_steps < self.sem_binding_max_steps
#                 ):
#                     with torch.enable_grad():
#                         self._semantic_binding_step(x, t)
#                     self.sem_binding_applied_steps += 1
#                     self.sem_binding_applied = True
#                     text_embed_cond, text_embed_cond_pool = self.text_embeds
                
#                 # ✅ 在语义绑定完成后，应用注意力引导
#                 # 此时cat*和panda*已经形成，需要约束它们的空间位置
#                 if (
#                     self.enable_attention_entropy 
#                     and self._entropy_masks_ready 
#                     and self.sem_binding_applied
#                 ):
#                     print(f"[AttnLoss] 重采样+语义绑定完成，t={t}, 应用注意力引导约束空间位置")
#                     self._apply_attention_guidance_after_binding(x, t)
#                     text_embed_cond, text_embed_cond_pool = self.text_embeds
                
#                 text_embed_uncond = text_embed_cond[0].unsqueeze(0)
#                 text_embed_multi = text_embed_cond[1].unsqueeze(0)
#                 text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)
#                 text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)
#                 text_embed = torch.cat([
#                     text_embed_uncond,
#                     text_embed_multi,
#                     text_embed_cond_single
#                 ], dim=0)
#                 text_embed_pool = torch.cat([
#                     text_embed_uncond_pool,
#                     text_embed_multi_pool,
#                     text_embed_cond_pool_single
#                 ], dim=0)

#                 del noise_pred_next, noise_pred_cond_next, noise_pred_uncond_next, denoised_tweedie_next, latent_model_next  # 释放变量
#                 gc.collect()
#                 torch.cuda.empty_cache()
#                 # 标准处理
#                 noise_pred_cond = noise_pred[1:2]
#                 noise_pred_uncond = noise_pred[:1]
#                 noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
#             else:  # 其它步
#                 noise_pred_cond = noise_pred[1:2]
#                 noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
#             #xˆ[εθ(xt, t, c)] := (xt − √1 − α ̄tεθ(xt, t, c))/√α ̄t,
#             denoised_tweedie = (x - (1 - at).sqrt() * noise_pred) / at.sqrt()  # 计算去噪结果
        
#         #公式9 zt−1 = √α ̄t−1{  N  X  i=1  Mi · zˆ[ε ̃i]} + p1 − α ̄t−1εθ0 (zt, t, ∅),
#         denoised_latent = at_next.sqrt() * denoised_tweedie + (1 - at_next).sqrt() * noise_pred_uncond  # 计算下一个步的潜变量
#         if t == self.t_cond_prev:  # 如果是关键步，生成掩码
#             """
#                 掩码生成：在关键时间步生成概念分离掩码
                
#                 掩码的作用：
#                 1. 分离不同概念的空间位置
#                 2. 指导多概念融合过程
#                 3. 确保概念间的协调生成

#                 触发条件：
#                 - t == self.t_cond_prev：当前时间步是融合阶段的前一步
#                 - 例如：如果融合阶段是[30, 29, 28, 27, 26]，那么t_cond_prev = 31
#                 - 在时间步31时生成掩码，为后续的融合阶段做准备
                
#                 掩码生成的重要性：
#                 - 掩码决定了每个概念在图像中的空间位置
#                 - 没有掩码，不同概念可能会相互干扰
#                 - 掩码确保熊猫、泰迪熊、城堡各自在正确的位置出现
#             """
#             denoised_latent_temp = denoised_latent
#             t_temp = next_t
#             if self.config.jumping_steps > 0:  # 跳步采样
#                 """
#                 跳步采样过程：通过多次迭代优化生成质量
                
#                 跳步采样的目的是：
#                 1. 提高生成质量
#                 2. 减少概念间的冲突
#                 3. 优化多概念融合效果

#                 跳步采样原理：
#                 - 不是按顺序执行每个时间步
#                 - 而是跳过某些时间步，直接跳到更早的时间步
#                 - 通过这种方式加速生成过程，同时保持质量
                
#                 具体例子：
#                 - 如果jumping_steps=3，当前t_temp=25
#                 - 第1次跳步：t_temp = 25 - 150 = -125 (实际上会跳到更早的时间步)
#                 - 第2次跳步：t_temp = -125 - 150 = -275
#                 - 第3次跳步：t_temp = -275 - 150 = -425
#                 """
#                 for _ in range(self.config.jumping_steps):
#                     # 计算当前临时时间步的alpha值
#                     at_temp = self.alpha(t_temp)
                    
#                      # 准备输入：使用当前去噪结果作为输入
#                     latent_model_next = torch.cat([denoised_latent_temp] + [denoised_latent_temp])
#                     # [2, 4, 128, 128]
#                     # 两个分支：1个负向 + 1个正向

#                     # 计算下一个时间步的噪声预测
#                     noise_pred_next = self.unet(
#                         latent_model_next, t_temp, encoder_hidden_states=text_embed[:2], added_cond_kwargs=unet_added_conditions_single
#                     )['sample']# [2, 4, 128, 128]
                    
#                     # 分离噪声预测
#                     noise_pred_cond_next = noise_pred_next[1:2] # 正向噪声预测
#                     noise_pred_uncond_next = noise_pred_next[:1] # 负向噪声预测
#                     noise_pred_next = noise_pred_uncond_next + self.config.guidance_scale * (noise_pred_cond_next - noise_pred_uncond_next)
                    
#                     # 跳步：直接跳到更早的时间步
#                     t_temp = t_temp - 150  # 跳步幅度
#                     at_temp_next = self.alpha(t_temp)  # 计算跳步后的alpha值
                    
#                     # 计算跳步后的去噪结果
#                     denoised_tweedie = (denoised_latent_temp - (1 - at_temp).sqrt() * noise_pred_next) / at_temp.sqrt()
                    
#                     # 更新临时潜变量
#                     denoised_latent_temp = at_temp_next.sqrt() * denoised_tweedie + (1 - at_temp_next).sqrt() * noise_pred_uncond_next
            
#                 del noise_pred_next, noise_pred_cond_next, noise_pred_uncond_next, denoised_latent_temp, latent_model_next
#                 gc.collect()
#                 torch.cuda.empty_cache()

#                 decoded_tweedie = self.decode_latent(denoised_tweedie)  # 解码为图片
#             else:
#                 # 如果不使用跳步采样，直接解码
#                 decoded_tweedie = self.decode_latent(denoised_tweedie)  # 解码为图片

            
#             os.makedirs(self.config.output_path, exist_ok=True)
#             path_tweedie = os.path.join(self.config.output_path, 'tweedie.jpg')
#             # 使用PIL直接保存，避免torchvision依赖
#             img_array = decoded_tweedie[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#             Image.fromarray(img_array).save(path_tweedie)
#             # 保存中间图像
#             use_ext = getattr(self.config, 'use_external_boxes', 0) == 1
#             boxes = parse_external_boxes(getattr(self.config, 'external_boxes', ''))
#             boxes_only = getattr(self.config, 'boxes_only', 0) == 1
            
#             # ✅ 方案B：如果已经提前生成了掩码（run_fusion中），跳过这里的生成
#             if self.masks is None:
#                 if use_ext and len(boxes) > 0:
#                     latent_h = self.config.resolution_h // 8
#                     latent_w = self.config.resolution_w // 8
#                     expected = self.concept_num - 1
#                     if len(boxes) != expected:
#                         raise ValueError(f'Number of boxes ({len(boxes)}) must equal num foreground concepts ({expected}). '
#                                         f'Concepts={self.config.concepts}, seg_concepts={self.config.seg_concepts}')
#                     fg_masks = build_masks_from_boxes(
#                         boxes,
#                         self.config.resolution_h, self.config.resolution_w,
#                         latent_h, latent_w,
#                         self.unet.device,
#                     )
#                     bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
#                     bg_mask[bg_mask < 0] = 0
#                     self.masks = torch.cat([fg_masks, bg_mask])
#                     self._prepare_cones_targets()
#                     if self.enable_attention_entropy:
#                         self._prepare_entropy_targets()
#                         print(f"[AttnLoss] t=31时生成掩码（原有逻辑）")
#                 elif boxes_only:
#                     raise ValueError('boxes_only=1 but no valid external_boxes provided.')
#                 else:
#                     """
#                     使用文本引导分割生成掩码
                    
#                     这个过程会：
#                     1. 调用外部分割脚本
#                     2. 根据文本条件生成分割掩码
#                     3. 将掩码转换为tensor格式

#                     分割脚本的工作原理：
#                     - 输入：中间生成的图像 + 文本条件
#                     - 输出：每个概念的分割掩码
#                     - 使用深度学习模型进行语义分割
#                     """
#                     # 构建分割命令
#                     test_cmd = f'CUDA_VISIBLE_DEVICES={self.config.seg_gpu} python text_segment/run_expand.py --input_path={path_tweedie} --text_condition="{self.config.seg_concepts}" --output_path={self.config.output_path}'
                    
#                     # 执行分割脚本
#                     os.system(test_cmd)
                    
#                     # 加载生成的掩码
#                     mask_paths = []
#                     concept_list = self.config.seg_concepts.split('+') if self.config.seg_concepts != '' else []
#                     # concept_list = ["a panda", "a teddybear"]
                
#                     for sp in concept_list:
#                         mask_paths.append(os.path.join(self.config.output_path, sp + '.jpg'))
#                     if len(mask_paths) > 0:
#                         # 预处理掩码
#                         fg_masks = torch.cat([
#                             preprocess_mask(mask_path, self.config.resolution_h // 8, self.config.resolution_w // 8, self.unet.device)
#                             for mask_path in mask_paths
#                         ])  # [2, 1, 128, 128] - 2个概念的前景掩码
                        
#                         # 构建背景掩码
#                         bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)  # [1, 1, 128, 128]
#                         bg_mask[bg_mask < 0] = 0  # 确保背景掩码非负
                        
#                         # 合并所有掩码
#                         self.masks = torch.cat([fg_masks, bg_mask])  # [3, 1, 128, 128]
#                         self._prepare_cones_targets()
#                         # 掩码结构：
#                         # self.masks[0]: 熊猫概念掩码
#                         # self.masks[1]: 泰迪熊概念掩码
#                         # self.masks[2]: 背景掩码
#             else:
#                 # 掩码已经提前生成，直接使用
#                 print(f"[AttnLoss] 使用提前生成的掩码（方案B）")
#                 self._prepare_cones_targets()
            
#             # ✅ 掩码生成后立即准备注意力熵目标
#             if self.enable_attention_entropy:
#                 self._prepare_entropy_targets()
#                 print(f"[AttnLoss] 掩码已准备完成")
#                 print(f"[AttnLoss] masks.shape={self.masks.shape if self.masks is not None else None}")
#                 if self.masks is not None:
#                     for i in range(self.masks.shape[0]):
#                         mask_sum = self.masks[i].sum().item()
#                         print(f"[AttnLoss] masks[{i}] sum={mask_sum:.1f} (应该>0)")
#                 print(f"[AttnLoss] entropy_masks_ready={self._entropy_masks_ready}")
#                 print(f"[AttnLoss] entropy_token_map={self._entropy_token_map}")

#         if t == 1:  # 如果是最后一步
#             denoised_latent = denoised_tweedie  # 直接返回去噪结果

#         return denoised_latent  # 返回当前步的去噪潜变量

#     def init_fusion(self, t_cond):
#         # 初始化融合采样的相关参数和状态
#         self.t_cond = self.scheduler.timesteps[t_cond:] if t_cond >= 0 else []  # 记录融合阶段的时间步（从 t_cond 开始到最后）
#         self.t_cond_prev = self.scheduler.timesteps[t_cond-1]  # 融合阶段前一个时间步
#         self.t_cond_cur = self.scheduler.timesteps[t_cond]      # 当前融合阶段的起始时间步
#         self.start_t = self.scheduler.timesteps[0]              # 采样的起始时间步
#         self._t_cond_timestep_set = set(
#             int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in self.t_cond
#         )
#         if self.cones_guidance_steps is not None and self.cones_guidance_steps > 0:
#             active = self.t_cond[: min(len(self.t_cond), self.cones_guidance_steps)]
#             self._cones_guidance_timestep_set = set(
#                 int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in active
#             )
#         else:
#             self._cones_guidance_timestep_set = None
#         os.makedirs(self.config.output_path_all, exist_ok=True) # 创建输出目录（如果不存在则创建）
#         if self.use_sts_kv_hook and self._sts_have_unet_kv:
#             register_attention_control_efficient_from_sts(self, self.t_cond, self.sts)
#         else:
#             register_attention_control_efficient(self, self.t_cond, self.concept_num) # 注册高效注意力控制（多概念融合相关）
#             for i in range(self.concept_num):
#                 model_name = f'unet_{i}'
#                 if hasattr(self, model_name):
#                     delattr(self, model_name)  # 删除每个概念对应的 UNet 属性，释放显存
            
#     def run_fusion(self):
#         '''
#         该方法用于启动融合采样过程，
#         它首先根据配置参数计算出融合阶段的时间步，
#         然后调用 init_fusion 方法初始化相关参数和状态，
#         最后调用 sample_loop 方法开始采样。
#         '''
#         t_cond = int(self.config.n_timesteps * self.config.t_cond)  # 根据配置计算融合阶段的起始步数（如 0.4*50=20）
#         self.init_fusion(t_cond=t_cond)  # 初始化融合采样相关参数和状态
        
#         # ✅ 方案B：如果提供了外部boxes，在采样开始前就生成掩码
#         use_ext = getattr(self.config, 'use_external_boxes', 0) == 1
#         boxes = parse_external_boxes(getattr(self.config, 'external_boxes', ''))
#         if use_ext and len(boxes) > 0:
#             latent_h = self.config.resolution_h // 8
#             latent_w = self.config.resolution_w // 8
#             expected = self.concept_num - 1
#             if len(boxes) != expected:
#                 raise ValueError(f'Number of boxes ({len(boxes)}) must equal num foreground concepts ({expected})')
            
#             fg_masks = build_masks_from_boxes(
#                 boxes,
#                 self.config.resolution_h, self.config.resolution_w,
#                 latent_h, latent_w,
#                 self.unet.device,
#             )
#             bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
#             bg_mask[bg_mask < 0] = 0
#             self.masks = torch.cat([fg_masks, bg_mask])
#             self._prepare_cones_targets()
            
#             # 准备注意力熵目标
#             if self.enable_attention_entropy:
#                 self._prepare_entropy_targets()
#                 print(f"[AttnLoss] 掩码提前准备完成（方案B：早期约束）")
#                 print(f"[AttnLoss] masks.shape={self.masks.shape}")
#                 for i in range(self.masks.shape[0]):
#                     mask_sum = self.masks[i].sum().item()
#                     print(f"[AttnLoss] masks[{i}] sum={mask_sum:.1f}")
        
#         # 生成初始噪声（标准正态分布），形状为 [1, 4, H/8, W/8]，并乘以初始噪声系数
#         normal = torch.randn(1, 4, self.config.resolution_h // 8, self.config.resolution_w // 8).to(self.unet.device) * self.scheduler.init_noise_sigma
#         _ = self.sample_loop(normal)  # 启动采样主循环，生成最终图像

#     @torch.no_grad()
#     def sample_loop(self, x):   
#         '''
#         该方法实现了扩散模型的采样过程，
#         它首先将初始噪声 x 输入到去噪模型中，
#         然后根据时间步 t 逐步进行去噪，
#         最终得到生成图像。
#         '''
#         stem, ext = self._build_base_filename()
#         out_dir = self.config.output_path_all
#         os.makedirs(out_dir, exist_ok=True)

#         with torch.autocast(device_type='cuda', dtype=torch.float16):  # 自动混合精度，节省显存
#             for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):  # 遍历所有采样步
#                 x = self.denoise_step(x, t)  # 对当前潜变量进行一步去噪

#             image, decoded_latent = self._decode_latent_to_pil(x)

#             filename = f"{stem}{ext}"
#             out_path = os.path.join(out_dir, filename)
#             try:
#                 # 保存图片，包含关键参数
#                 image[0].save(out_path)
#             except OSError as e:
#                 # Extremely defensive fallback (e.g., weird filesystem).
#                 guidance = float(getattr(self.config, "guidance_scale", 7.5))
#                 t_cond_ratio = float(getattr(self.config, "t_cond", 0.0))
#                 resample_steps = int(getattr(self.config, "resampling_steps", 0))
#                 seed = int(getattr(self.config, "seed", 0))
#                 fallback_path = os.path.join(out_dir, f"output_g{guidance:.2f}_t{t_cond_ratio:.2f}_res{resample_steps}_seed{seed}.png")
#                 image[0].save(fallback_path)
#                 print(f"[WARN] Failed to save '{out_path}' ({e}); saved as '{fallback_path}' instead.")
                
#         return decoded_latent  # 返回解码后的 latent（图像张量）

# if __name__ == '__main__':    
#     parser = argparse.ArgumentParser()  # 创建命令行参数解析器
#     parser.add_argument('--seed', type=int, default=182)  # 随机种子
#     parser.add_argument('--device', type=str, default='cuda:0')  # 计算设备
#     parser.add_argument('--output_path', type=str)  # 中间结果输出目录
#     parser.add_argument('--output_path_all', type=str)  # 最终图片输出目录
#     parser.add_argument('--negative_prompt', type=str, default='blurry, ugly, black, low res, unrealistic, blurry face')
#     # 负面提示词，默认用于去除模糊、低质量等
#     parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.4','1.5', '2.0','2.1','xl'],
#                         help="stable diffusion version")  # Stable Diffusion 版本
#     parser.add_argument('--pretrained_model_name_or_path', type=str, default='',
#                         help='可选：显式指定基础模型路径（本地目录或HF id）')
#     parser.add_argument('--vae_model_name_or_path', type=str, default='',
#                         help='可选：显式指定VAE路径（本地目录优先）；留空则使用基础模型自带VAE')
#     parser.add_argument('--t_cond', type=float, default=0.4)  # 融合阶段的比例（如0.4表示前40%步数用于融合）
#     parser.add_argument('--guidance_scale', type=float, default=9.0)  # classifier-free guidance 强度
#     parser.add_argument('--n_timesteps', type=int, default=50)  # 采样步数
#     parser.add_argument('--prompt', type=str, default='')  # 多概念融合的完整 prompt
#     parser.add_argument('--concept_weights', type=str, default='')  # 概念权重列表，逗号分隔
#     parser.add_argument('--prompt_clean', type=str, default='')
#     parser.add_argument('--prompt_clean_resample', type=str, default='', help='Clean single-concept prompts used only in resampling (decoupled from MSE anchors)')
#     parser.add_argument('--prompt_orig_clean', type=str, default='')  # 多概念无属性 prompt
#     parser.add_argument('--prompt_orig', type=str, default='')  # 原始 prompt
#     parser.add_argument('--seg_concepts', type=str, default='')  # 分割掩码的概念列表
#     parser.add_argument('--personal_checkpoint', type=str, default='')  # 自定义权重路径
#     parser.add_argument('--concepts', type=str)  # 概念词列表
#     parser.add_argument('--modifier_token', type=str)  # 修饰 token 列表
#     parser.add_argument('--resampling_steps',  type=int, default=10)  # 重采样步数
#     parser.add_argument('--jumping_steps',  type=int, default=5)  # 跳步采样步数
#     parser.add_argument('--seg_gpu',  type=int, default=1)  # 分割脚本使用的 GPU
#     parser.add_argument('--disable_xformers', type=int, default=0, help='1=禁用xformers（稳定优先）')
#     parser.add_argument('--use_external_boxes', type=int, default=0)  # 是否使用外部矩形框生成掩码
#     parser.add_argument('--external_boxes', type=str, default='')  # 外部矩形框列表：x1,y1,x2,y2+x1,y1,x2,y2
#     parser.add_argument('--boxes_only', type=int, default=0)  # 1=只用外部框，缺框就报错
#     parser.add_argument(
#         "--crops_coords_top_left_h",
#         type=int,
#         default=0,
#         help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
#     )  # SDXL 裁剪左上角高度
#     parser.add_argument(
#         "--crops_coords_top_left_w",
#         type=int,
#         default=0,
#         help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
#     )  # SDXL 裁剪左上角宽度
#     parser.add_argument(
#         "--resolution_h",
#         type=int,
#         default=1024,
#         help=(
#             "The resolution for input images, all the images in the train/validation dataset will be resized to this"
#             " resolution"
#         ),
#     )  # 输入图片高度
#     parser.add_argument(
#         "--resolution_w",
#         type=int,
#         default=1024,
#         help=(
#             "The resolution for input images, all the images in the train/validation dataset will be resized to this"
#             " resolution"
#         ),
#     )  # 输入图片宽度
#     parser.add_argument('--binding_json', type=str, default='', help='JSON string or file path for manual token bindings')
#     parser.add_argument('--sem_binding_steps', type=int, default=0, help='Semantic binding optimization steps at content fusion start')
#     parser.add_argument('--sem_binding_lr', type=float, default=1e-3, help='Semantic binding learning rate')
#     parser.add_argument('--sem_binding_use_clean', type=int, default=0, help='Use PROMPT_CLEAN single-concept anchors for semantic binding (1=yes, 0=no)')
#     parser.add_argument('--debug_binding', type=int, default=0, help='Print tokenization and binding debug info')
#     parser.add_argument('--sem_binding_max_steps', type=int, default=1, help='语义绑定在融合阶段执行的最大时间步数')
#     parser.add_argument('--enable_attention_entropy', type=int, default=0, help='是否开启注意力熵约束')
#     parser.add_argument('--attn_entropy_weight', type=float, default=0.0, help='注意力熵损失的权重')
#     parser.add_argument('--attn_entropy_outside_weight', type=float, default=1.0, help='注意力落在掩码外部时的惩罚系数')
#     parser.add_argument('--attn_entropy_inside_weight', type=float, default=0.0, help='注意力落在掩码内部时的奖励系数')
#     parser.add_argument('--attn_entropy_steps', type=int, default=1, help='每个时间步执行注意力熵优化的迭代次数')
#     parser.add_argument('--attn_entropy_lr', type=float, default=5e-4, help='注意力熵优化的学习率')
#     parser.add_argument('--attn_entropy_min_step', type=int, default=0, help='注意力熵开始生效的最小时间步')
#     parser.add_argument('--attn_entropy_max_step', type=int, default=-1, help='注意力熵停止生效的最大时间步（-1 表示不限制）')
#     parser.add_argument('--attn_entropy_layers', type=str, default='up', help='参与熵计算的 UNet 模块前缀（up/down/mid）')
#     parser.add_argument('--attn_entropy_mask_downscale', type=int, default=1, help='注意力掩码下采样因子')
#     parser.add_argument('--attn_entropy_low_mem', type=int, default=0, help='是否开启低显存模式（仅使用部分分支参与熵优化）')
#     parser.add_argument('--attn_entropy_enable_checkpointing', type=int, default=0, help='是否在熵优化时开启 UNet 梯度检查点')
#     parser.add_argument('--enable_cones_mask_attention', type=int, default=1, help='是否启用Cones风格的logits掩码注意力引导')
#     parser.add_argument('--cones_guidance_weight', type=float, default=0.08, help='Cones布局引导强度')
#     parser.add_argument('--cones_guidance_steps', type=int, default=-1, help='Cones布局引导生效步数（在融合阶段内，-1表示融合阶段全程）')
#     parser.add_argument('--cones_positive_value', type=float, default=2.5, help='Cones掩码在目标区域的正偏置值')
#     parser.add_argument('--cones_negative_value', type=float, default=-0.02, help='Cones掩码在非目标区域的负偏置值')
#     parser.add_argument('--cones_use_sim_std', type=int, default=1, help='Cones偏置是否乘以当前注意力logits标准差')
#     parser.add_argument('--cones_use_modifier_only', type=int, default=1, help='1=Cones仅对modifier token列注入（推荐个性化概念）')
#     parser.add_argument('--cones_focus_fg_only', type=int, default=1, help='1=Cones仅作用前景概念（不对背景概念注入）')
#     parser.add_argument('--cones_use_plain_prompt_before_fusion', type=int, default=1, help='1=在多概念融合前对原始multi prompt的cat/dog位置做Cones框注入')
#     parser.add_argument('--cones_debug_tokens', type=int, default=0, help='1=启用Cones token调试逻辑')
#     parser.add_argument('--cones_debug_token_maps', type=int, default=0, help='1=打印[ConesMask2][Debug] token_maps明细（很冗长，默认0关闭）')
#     parser.add_argument('--use_sts_kv_hook', type=int, default=1, help='1=使用sts里的unet K/V直接挂载attention（低显存），0=回退旧版加载unet_i')
#     parser.add_argument('--fusion_low_mem', type=int, default=0, help='融合阶段使用串行概念推理，降低一次性batch显存占用')
#     opt = parser.parse_args()  # 解析命令行参数，保存到 opt

#     seed_everything(opt.seed)  # 设置随机种子，保证实验可复现
#     tweedie = MultiCompose(opt)  # 创建 MultiCompose 融合采样对象
    # tweedie.run_fusion()       # 启动融合采样流程
import os
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as T  # 注释掉避免版本兼容问题
import argparse
import json
from PIL import Image
from tqdm import tqdm
from transformers import logging
import math
import inspect
from diffusers import DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel,AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
try:
    from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
    from diffusers.models.attention_processor import FusedAttnProcessor2_0
except ImportError:
    # 旧版本diffusers兼容
    AttnProcessor2_0 = None
    XFormersAttnProcessor = None
    FusedAttnProcessor2_0 = None
import gc
import numpy as np
from attention_mask_utils import *
from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
from contextlib import nullcontext

class SimpleAttentionStore:
    """ToMe风格的注意力存储器，用于收集cross-attention maps"""
    def __init__(self):
        self.attention_maps = []
        self.step_count = 0
    
    def __call__(self, attn_probs, is_cross, place_in_unet):
        """存储cross-attention概率分布"""
        if is_cross and attn_probs.shape[-1] == 77:  # 只存储text cross-attention
            # 存储detached版本避免梯度图增长
            self.attention_maps.append({
                'probs': attn_probs.detach(),
                'place': place_in_unet
            })
    
    def reset(self):
        """清空存储"""
        self.attention_maps = []
        self.step_count += 1
    
    def get_average_attention(self):
        """聚合所有层的attention maps"""
        if not self.attention_maps:
            return None
        # 简单平均所有层的attention
        all_probs = torch.stack([m['probs'] for m in self.attention_maps])
        return all_probs.mean(dim=0)  # [batch, query_len, 77]

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 guidance_rescale 对 noise_cfg 进行重标定。
    该方法基于论文《Common Diffusion Noise Schedules and Sample Steps are Flawed》（https://arxiv.org/pdf/2305.08891.pdf）第3.4节的发现。
    主要作用：修正扩散模型在使用 classifier-free guidance 时可能出现的过曝或图像过于平淡的问题。
    """
    # 计算文本条件噪声预测的标准差
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算 guidance 条件噪声预测的标准差
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 对 guidance 结果进行重标定（修正过曝）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 按 guidance_rescale 权重混合原始和重标定结果，避免图像过于平淡
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

logging.set_verbosity_error()
def tokenize_prompt(tokenizer, prompt):
    """
    使用指定的分词器将文本提示词转换为模型可理解的数字序列
    
    Args:
        tokenizer: 分词器对象（CLIP或OpenCLIP的分词器）
        prompt: 输入的文本提示词（字符串或字符串列表）
    
    Returns:
        text_input_ids: token ID序列，形状为[1, max_length]
    
    注意：
        - 使用padding="max_length"确保所有序列长度一致
        - truncation=True处理超长文本
        - 返回的是PyTorch tensor格式，便于GPU计算
    """
    # 使用指定的分词器（tokenizer）对输入的 prompt 进行分词编码
    text_inputs = tokenizer(
        prompt,                         # 输入的文本 prompt
        padding="max_length",           # 填充到最大长度
        max_length=tokenizer.model_max_length,  # 最大长度由分词器模型决定
        truncation=True,                # 超过最大长度则截断
        return_tensors="pt",            # 返回 PyTorch tensor 格式
    )
    text_input_ids = text_inputs.input_ids  # 获取编码后的 token id
    # 示例输出（SDXL CLIP分词器）：
    # 输入: "photo of a <panda1> panda playing with a ball, castle background",
    # 输出: [49406, 1125, 539, 320, 6442, 304, 271, 2675, 49407, 49407, ...]
    # 对应: [<start>, "photo", "of", "a", "<panda1>", "playing", "with", "a", "ball", ",", "castle", "background", <end>, <pad>, ...]
    # 长度: 77（SDXL标准长度）
    return text_input_ids                   # 返回 token id
   

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    """
    将文本提示词编码为高维语义向量表示，支持双编码器架构
    
    这是MultiCompose系统的核心文本理解组件，使用CLIP和OpenCLIP双编码器
    提供更强的文本理解能力，支持多概念融合的复杂语义处理。
    
    Args:
        text_encoders: 文本编码器列表 [CLIP编码器, OpenCLIP编码器]
        tokenizers: 分词器列表 [CLIP分词器, OpenCLIP分词器]  
        prompt: 输入的文本提示词（字符串或字符串列表）
        text_input_ids_list: 可选的预分词token ID列表（用于优化性能）
    
    Returns:
        prompt_embeds: 序列嵌入 tensor [batch_size, seq_len, 2048]
            - batch_size: 提示词数量
            - seq_len: 序列长度（SDXL标准为77）
            - 2048: 双编码器拼接维度（1024+1024）
        pooled_prompt_embeds: 全局嵌入 tensor [batch_size, 2048]
            - 用于SDXL的额外条件输入
            - 提供全局语义信息
    
    Technical Details:
        - 使用倒数第二层隐藏状态而非最后一层（更稳定的表示）
        - 双编码器结果在最后一维拼接，增强语义理解能力
        - 支持批处理，提高计算效率
    """
    prompt_embeds_list = []  # 用于存储每个编码器的嵌入

    for i, text_encoder in enumerate(text_encoders):
        # 遍历每个文本编码器（CLIP和OpenCLIP）
        if tokenizers is not None:
            # 如果有分词器列表，则用对应分词器对 prompt 进行分词
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            # 否则使用传入的 token id 列表（节省重复分词时间）
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        # 用编码器对 token id 进行编码，获取输出和隐藏状态
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),  # 将 token id 移动到编码器所在设备
            output_hidden_states=True,               # 返回隐藏层状态（用于提取中间层特征）
        )

        # 提取pooled输出（用于SDXL的额外条件输入）
        pooled_prompt_embeds = prompt_embeds[0]     # 获取 pooled 输出 [batch_size, hidden_dim]
        # 使用倒数第二层隐藏状态（更稳定的文本表示，避免过拟合最后一层）
        prompt_embeds = prompt_embeds.hidden_states[-2]  # 获取倒数第二层的隐藏状态
        bs_embed, seq_len, _ = prompt_embeds.shape  # 获取 batch size 和序列长度
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)  # 重新调整形状
        prompt_embeds_list.append(prompt_embeds)    # 添加到列表

    # 将所有编码器的嵌入在最后一个维度拼接（CLIP + OpenCLIP）
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)  # [batch, seq_len, 2048]
    # 对 pooled 输出也做形状调整
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)  # [batch, 2048]
    return prompt_embeds, pooled_prompt_embeds       # 返回嵌入和 pooled 嵌入

def compute_time_ids():
    # 生成 SDXL 所需的时间编码（time_ids），用于条件控制
    # 参考 StableDiffusionXLPipeline._get_add_time_ids 的实现
    original_size = (opt.resolution_h, opt.resolution_w)  # 原始图片尺寸
    target_size = (opt.resolution_h, opt.resolution_w)    # 目标图片尺寸
    crops_coords_top_left = (opt.crops_coords_top_left_h, opt.crops_coords_top_left_w)  # 裁剪左上角坐标
    add_time_ids = list(original_size + crops_coords_top_left + target_size)  # 拼成一个列表
    add_time_ids = torch.tensor([add_time_ids])  # 转为 tensor，增加 batch 维
    # add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)  # 可选：转到指定设备和类型
    return add_time_ids  # 返回时间编码

def preprocess_mask(mask_path, h, w, device):
    # 读取掩码图片并预处理，返回指定尺寸和设备上的二值掩码
    mask = np.array(Image.open(mask_path).convert("L"))  # 读取图片并转为灰度
    mask = mask.astype(np.float32) / 255.0               # 归一化到 0-1
    mask = mask[None, None]                              # 增加 batch 和 channel 维度
    mask[mask < 0.5] = 0                                 # 小于 0.5 设为 0
    mask[mask >= 0.5] = 1                                # 大于等于 0.5 设为 1
    mask = torch.from_numpy(mask).to(device)              # 转为 tensor 并移动到指定设备
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')  # 插值到目标尺寸
    return mask                                          # 返回处理后的掩码

def preprocess_mask_raw(mask_path, h, w, device):
    # 预处理掩码（原始版，未读取图片，假设 mask 已经是 tensor）
    mask[mask < 0.5] = 0                                 # 小于 0.5 设为 0
    mask[mask >= 0.5] = 1                                # 大于等于 0.5 设为 1
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')  # 插值到目标尺寸
    return mask                                          # 返回处理后的掩码

def parse_external_boxes(boxes_str):
    # 解析形如 "x1,y1,x2,y2+x1,y1,x2,y2" 的字符串为列表
    boxes_str = boxes_str.strip()
    if boxes_str == "":
        return []
    parts = boxes_str.split('+')
    boxes = []
    for p in parts:
        nums = p.split(',')
        if len(nums) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in nums]
            boxes.append([x1, y1, x2, y2])
        except Exception:
            continue
    return boxes

def build_masks_from_boxes(boxes, img_h, img_w, latent_h, latent_w, device):
    # 将像素坐标的矩形框转为 latent 分辨率下的二值掩码（每个框一个通道）
    if len(boxes) == 0:
        return None
    masks = []
    for (x1, y1, x2, y2) in boxes:
        lx1 = int(max(0, min(latent_w, round(x1 / img_w * latent_w))))
        ly1 = int(max(0, min(latent_h, round(y1 / img_h * latent_h))))
        lx2 = int(max(0, min(latent_w, round(x2 / img_w * latent_w))))
        ly2 = int(max(0, min(latent_h, round(y2 / img_h * latent_h))))
        if lx2 <= lx1 or ly2 <= ly1:
            m = torch.zeros(1, 1, latent_h, latent_w, device=device)
        else:
            m = torch.zeros(1, 1, latent_h, latent_w, device=device)
            m[:, :, ly1:ly2, lx1:lx2] = 1.0
        masks.append(m)
    masks = torch.cat(masks, dim=0)
    return masks


class MultiCompose(nn.Module):
    def __init__(self, config):
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.config = config  # 保存配置参数
        sd_version = config.sd_version  # 获取 stable diffusion 版本
        model_override = str(getattr(config, "pretrained_model_name_or_path", "")).strip()
        env_model_name = str(os.environ.get("MODEL_NAME", "")).strip()

        # 根据配置选择不同的 stable diffusion 预训练模型
        if model_override:
            model_key = model_override
        elif sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version =='1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
        elif sd_version =='xl':
            model_key = env_model_name if env_model_name else "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')  # 不支持的版本报错
        
        # 创建 SD 模型
        print(f'Loading SD model from: {model_key}')

        # 加载 SDXL 管道：优先 fp16 variant；本地目录缺失时自动回退
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_key, torch_dtype=torch.float16, variant="fp16", use_safetensors=False
            ).to("cuda")
        except OSError as exc:
            msg = str(exc)
            if "fp16" not in msg:
                raise
            print(f"[WARN] fp16 variant not found for base pipeline ({model_key}), fallback to default weights.")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_key, torch_dtype=torch.float16, use_safetensors=False
            ).to("cuda")

        self.use_xformers = bool(not getattr(config, "disable_xformers", 0))
        if self.use_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()  # 启用 xformers 高效注意力机制
            except Exception as exc:
                print(f"[WARN] failed to enable xformers on pipeline: {exc}; continue without xformers.")
                self.use_xformers = False

        pipe.enable_vae_slicing()  # 启用 VAE 切片，节省显存
        # 默认直接使用基础模型自带 VAE（离线环境更稳）。
        # 仅在显式提供路径/ID时才尝试额外加载 VAE，避免无意联网请求。
        self.vae = pipe.vae
        self.vae_source = "base_pipeline"
        vae_override = str(getattr(config, "vae_model_name_or_path", "")).strip()
        if not vae_override:
            vae_override = str(os.environ.get("VAE_MODEL_NAME", "")).strip()
        if not vae_override:
            vae_override = str(os.environ.get("VAE_MODEL", "")).strip()
        if not vae_override and sd_version == 'xl':
            auto_fix = str(os.environ.get("AUTO_SDXL_VAE_FP16_FIX", "1")).strip().lower()
            if auto_fix not in {"0", "false", "no"}:
                vae_override = "madebyollin/sdxl-vae-fp16-fix"
        if vae_override:
            try:
                vae_kwargs = {"torch_dtype": torch.float16}
                if os.path.isdir(vae_override):
                    vae_kwargs["local_files_only"] = True
                self.vae = AutoencoderKL.from_pretrained(vae_override, **vae_kwargs).to("cuda")
                self.vae_source = "external_override"
                print(f"Loaded external VAE from: {vae_override}")
            except Exception as exc:
                print(f"[WARN] failed to load external VAE ({vae_override}): {exc}; fallback to pipeline VAE.")
                self.vae = pipe.vae
                self.vae_source = "base_pipeline"
        else:
            print("Using VAE from base pipeline (no external VAE override).")
        
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建 VAE 图像处理器
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        # 保存分词器、文本编码器、UNet等
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.unet = pipe.unet
        if self.use_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()  # 启用高效注意力
            except Exception as exc:
                print(f"[WARN] failed to enable xformers on base UNet: {exc}; continue without xformers.")
                self.use_xformers = False
        self.device = self.unet.device  # 记录设备
        
        self.sts = []  # 用于存储自定义权重
        self.masks = None  # 初始化掩码

        # 解析配置中的模型路径、原始提示词、分割提示词、概念、修饰 token
        model_paths = config.personal_checkpoint.split('+')

        #photo of a cat wearing tie and a dog with sunglasses running,<castle1> castle background
        prompt_orig = config.prompt_orig.split('+')[0]
        
        #photo of a cat wearing tie,castle background
        #photo of a dog with sunglasses,castle background
        #castle background
        prompt_sep = config.prompt.split('+')


        #CONCEPTS="cat+dog+castle"
        concepts = config.concepts.split('+')
        #MODIFIER="<cat5>+<dog2>+<castle1>"
        modifier_token_user = config.modifier_token.split('+')
        prompts = []
        prompts.append(prompt_orig)  # 添加原始 prompt

        prompt_orig_clean = getattr(config, 'prompt_orig_clean', prompt_orig)
        self.prompt_orig_clean = prompt_orig_clean
        #photo of a cat wearing tie,castle background+photo of a dog with sunglasses,castle background+castle background
        concept_num = len(concepts)  # 概念数量


        #photo of a cat wearing tie,castle background
        #photo of a dog with sunglasses,castle background
        prompts_single = prompt_sep[:concept_num-1]
        self.prompts_single = prompts_single  # 单独的 prompt

        weights_cfg = getattr(config, "concept_weights", "")
        if weights_cfg:
            parsed_weights = [float(w) for w in weights_cfg.split(",") if w.strip() != ""]
            if len(parsed_weights) < concept_num:
                parsed_weights.extend([1.0] * (concept_num - len(parsed_weights)))
            self.concept_weights = parsed_weights[:concept_num]
        else:
            self.concept_weights = [1.0] * concept_num

        prompts_single_clean = []
        self.prompts_single_clean = prompts_single_clean
         #"cat wearing tie+dog with sunglasses"
        prompt_clean_cfg = getattr(config, 'prompt_clean', '')
        if prompt_clean_cfg:
            prompt_clean_sep = prompt_clean_cfg.split('+')
            #[cat wearing tie,dog with sunglasses]
            if len(prompt_clean_sep) >= concept_num - 1:
                prompts_single_clean = prompt_clean_sep[:concept_num-1]
                #[cat wearing tie,dog with sunglasses]
        # 保存最终的字符串列表到实例，供调试打印使用
        self.prompts_single_clean = prompts_single_clean
        # 仅用于“重采样单概念”的干净句（与 MSE 锚点的 PROMPT_CLEAN 解耦）
        prompts_single_resample = []
        self.prompts_single_resample = prompts_single_resample
        prompt_clean_resample_cfg = getattr(config, 'prompt_clean_resample', '')
        if prompt_clean_resample_cfg:
            prompt_clean_resample_sep = prompt_clean_resample_cfg.split('+')
            if len(prompt_clean_resample_sep) >= concept_num - 1:
                prompts_single_resample = prompt_clean_resample_sep[:concept_num-1]
        #[photo of a cat,photo of a dog]
        # 同步保存实例属性（供调试打印与后续使用）
        self.prompts_single_resample = prompts_single_resample
        self.concept_num = concept_num  # 保存概念数量
        # 构造每个概念的 prompt，插入修饰 token


        ##photo of a cat wearing tie,castle background
        #photo of a dog with sunglasses,castle background
        #castle background
        #"cat+dog+castle"
        #modifier_token_user="<cat5>+<dog2>+<castle1>"
        for i, wd in enumerate(concepts):
            base_prompt = prompt_sep[i] if i < len(prompt_sep) else ""
            modifier_token = modifier_token_user[i] if i < len(modifier_token_user) else ""
            index = base_prompt.find(wd)  # 找到概念在 prompt 中的位置
            if modifier_token and index >= 0:
                result = base_prompt[:index] + modifier_token + " " + base_prompt[index:]
            else:
                result = base_prompt

            prompts.append(result)
        
        ##photo of a cat wearing tie and a dog with sunglasses running,<castle1> castle background
        #1. photo of a <cat5> cat wearing tie,castle background
        # 2. photo of a <dog2> dog with sunglasses,castle background
        # 3. <castle1> castle background

        # 加载每个自定义权重文件
        #PERSONAL_CHECKPOINT="/path/to/checkpoints/subject_a.bin+/path/to/checkpoints/subject_b.bin+/path/to/checkpoints/background.bin"
        for sp in model_paths:
            self.sts.append(torch.load(sp))
        #sts=[..pet_cat5/delta-200.bin,..pet_dog2/delta-200.bin,..scene_castle/delta-200.bin]
        modifier_token_id = []
        modifier_token_id_2 = []

        # 如果权重中包含 modifier_token，则进行 token 嵌入的替换
        if 'modifier_token' in self.sts[0]:
            modifier_tokens = []
            modifier_tokens_2 = []

            # 收集所有 modifier_token 和 modifier_token_2
            for single_st in self.sts:
                modifier_tokens += list(single_st['modifier_token'].keys())
                modifier_tokens_2 += list(single_st['modifier_token_2'].keys())
                
            # 遍历用户指定的修饰 token，添加到分词器，并获取其 id
            for i, modifier_token in enumerate(modifier_token_user):
                # self.find_disc(self.sts[i]['modifier_token'][modifier_token],self.sts[i]['modifier_token_2'][modifier_token])
                num_added_tokens = self.tokenizer.add_tokens(modifier_token)## 添加新token到分词器
                modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token))
            # 同理，添加到第二个分词器
                num_added_tokens = self.tokenizer_2.add_tokens(modifier_token)
                modifier_token_id_2.append(self.tokenizer_2.convert_tokens_to_ids(modifier_token))
                
            # 调整编码器的 token 嵌入表大小
            # 调整嵌入表大小
            # 原始大小：49407个token
            # 新增3个token后：49410个token
            #self.text_encoder.resize_token_embeddings(49410)  # 49407 + 3
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            token_embeds_2 = self.text_encoder_2.get_input_embeddings().weight.data

            # 用自定义权重替换 token 嵌入
            # token_embeds[49408] = self.sts[0]['modifier_token']['<panda1>']
            # token_embeds[49409] = self.sts[1]['modifier_token']['<teddybear1>']
            # token_embeds[49410] = self.sts[2]['modifier_token']['<castle1>']
            for i, id_ in enumerate(modifier_token_id):
                single_st = self.sts[i]
                token_embeds[id_] = single_st['modifier_token'][modifier_tokens[i]]
            for i, id_ in enumerate(modifier_token_id_2):
                single_st = self.sts[i]
                token_embeds_2[id_] = single_st['modifier_token_2'][modifier_tokens_2[i]]
                
        null_prompt = [config.negative_prompt]  # 负面提示词

        # 获取所有 prompt 的文本嵌入
        # torch.tensor([5, 77, 2048]),  # 4个prompt + 1个negative
        # torch.tensor([5, 2048])       # 对应的pooled嵌入

         ##photo of a cat wearing tie and a dog with sunglasses running,<castle1> castle background
        #1. photo of a <cat5> cat wearing tie,castle background
        # 2. photo of a <dog2> dog with sunglasses,castle background
        # 3. <castle1> castle background
        embeds_all = self.get_text_embeds(prompts, null_prompt, device=self.unet.device)

        self.text_embeds_raw = (
            embeds_all[0].clone(),
            embeds_all[1].clone(),
        )
        self.text_embeds = (
            self.text_embeds_raw[0].clone(),
            self.text_embeds_raw[1].clone(),
        )
        self.text_embeds_multi_clean = None
        if self.prompt_orig_clean:
            embeds_multi_clean = self.get_text_embeds([self.prompt_orig_clean], null_prompt, device=self.unet.device)
            seq_clean = embeds_multi_clean[0][1:2].to(device=self.unet.device, dtype=self.unet.dtype)
            pool_clean = embeds_multi_clean[1][1:2].to(device=self.unet.device, dtype=self.unet.dtype)
            self.text_embeds_multi_clean = (seq_clean, pool_clean)

        # Token 合并与 ETS
        self.binding_entries = self._load_binding_entries(getattr(config, "binding_json", ""))
        self._binding_prompt_list = list(prompts)
        self.binding_prompt_text = prompt_orig
        self.binding_subject_indices = []
        self.binding_subject_indices_merged = []
        self.binding_clean_cache = {}
        self.binding_clean_cache_merged = {}
        self.binding_subject_groups = [] #保存 subject 分组（例如 cat tokens 在第 0 组，dog 在第 1 组）
        self.binding_subject_groups_merged = {}
        self.text_embeds_merged = (
            self.text_embeds_raw[0].clone(),
            self.text_embeds_raw[1].clone(),
        )#先把未处理的嵌入复制一份，作为“合并后嵌入”的初始值——如果没有 binding，后续就直接用它。
        binding_subject_map = {}
        binding_subject_group_map = {}#临时字典，用于接收 _apply_token_merging_and_ets 返回的“每个 prompt 的 subject 索引 / 分组信息”。
        if self.binding_entries:
            (
                merged_embeds,
                binding_subject_map,
                binding_clean_cache,
                binding_subject_group_map,
            ) = self._apply_token_merging_and_ets(
                (
                    self.text_embeds_raw[0].clone(),
                    self.text_embeds_raw[1].clone(),
                ),
                prompts,
                self.binding_entries,
            )
            self.text_embeds_merged = (
                merged_embeds[0].clone(),
                merged_embeds[1].clone(),
            )
            self.binding_subject_indices_merged = sorted(binding_subject_map.get(prompt_orig, []))
            self.binding_clean_cache_merged = binding_clean_cache
            self.binding_subject_groups_merged = binding_subject_group_map
            self.binding_subject_groups = binding_subject_group_map.get(prompt_orig, [])
        self._binding_entries = []
        self._binding_clean_cache = {}
        self._binding_prompt_to_index = {}
        self._binding_active = False

        concept_labels = self.config.concepts.split('+')
        seg_focus = set()
        seg_concepts_raw = getattr(self.config, 'seg_concepts', '')
        if seg_concepts_raw:
            seg_entries = [entry.strip().lower() for entry in seg_concepts_raw.split('+') if entry.strip()]
            for label in concept_labels:
                label_lower = label.lower()
                for entry in seg_entries:
                    if label_lower in entry or entry in label_lower:
                        seg_focus.add(label)
                        break
        if not seg_focus:
            seg_focus = {"panda", "cat"}
        self._entropy_focus_labels = seg_focus
        self.subject_token_ids = sorted(binding_subject_map.get(prompt_orig, []))
        self.subject_token_labels = {}
        subject_groups = self.binding_subject_groups or []
        if subject_groups:
            for group_idx, token_list in enumerate(subject_groups):
                label = concept_labels[group_idx] if group_idx < len(concept_labels) else f"group{group_idx}"
                for token_idx in token_list:
                    self.subject_token_labels[int(token_idx)] = label
        if not self.subject_token_labels:
            for i, idx in enumerate(self.subject_token_ids):
                label = concept_labels[i] if i < len(concept_labels) else f"token{idx}"
                self.subject_token_labels[idx] = label
        else:
            for idx in self.subject_token_ids:
                if idx not in self.subject_token_labels:
                    self.subject_token_labels[idx] = f"token{idx}"
        # Disable legacy attention-entropy guidance by default; use Cones-style
        # mask attention bias during fusion instead.
        self.enable_attention_entropy = False
        self.attn_entropy_weight = float(getattr(self.config, "attn_entropy_weight", 0.0))
        self.attn_entropy_outside_weight = float(getattr(self.config, "attn_entropy_outside_weight", 1.0))
        self.attn_entropy_inside_weight = float(getattr(self.config, "attn_entropy_inside_weight", 0.0))
        self.attn_entropy_steps = int(getattr(self.config, "attn_entropy_steps", 1))
        self.attn_entropy_lr = float(getattr(self.config, "attn_entropy_lr", 1e-3))
        self.attn_entropy_min_step = int(getattr(self.config, "attn_entropy_min_step", 0))
        self.attn_entropy_max_step = int(getattr(self.config, "attn_entropy_max_step", -1))
        layers_cfg = str(getattr(self.config, "attn_entropy_layers", "up")).lower()
        if layers_cfg in ("", "all", "none"):
            self._entropy_layer_filter = None if layers_cfg != "none" else set()
        else:
            layer_tokens = []
            for token in layers_cfg.replace(",", "+").split("+"):
                token = token.strip()
                if token:
                    layer_tokens.append(token)
            mapped_prefix = set()
            for token in layer_tokens:
                if token == "up":
                    mapped_prefix.add("up_res")
                elif token == "mid":
                    mapped_prefix.add("mid_block")
                elif token == "down":
                    mapped_prefix.add("down_res")
                else:
                    mapped_prefix.add(token)
            self._entropy_layer_filter = mapped_prefix or None
        self.attn_entropy_mask_downscale = max(1, int(getattr(self.config, "attn_entropy_mask_downscale", 1)))
        self.attn_entropy_low_mem = bool(getattr(self.config, "attn_entropy_low_mem", 0))
        self.attn_entropy_enable_checkpointing = bool(
            getattr(self.config, "attn_entropy_enable_checkpointing", 0)
        )
        self._entropy_token_map = {}
        focus_labels = self._entropy_focus_labels or {"panda", "cat"}
        for idx, label in self.subject_token_labels.items():
            if label in focus_labels:
                self._entropy_token_map.setdefault(label, []).append(idx)
        self._entropy_masks_by_grid = {}
        self._entropy_loss_terms = []
        self._entropy_active = False
        self._entropy_masks_ready = False

        # Cones-style mask attention guidance config.
        self.enable_cones_mask_attention = bool(getattr(self.config, "enable_cones_mask_attention", 1))
        self.cones_guidance_weight = float(getattr(self.config, "cones_guidance_weight", 0.08))
        self.cones_guidance_steps = int(getattr(self.config, "cones_guidance_steps", -1))
        self.cones_positive_value = float(getattr(self.config, "cones_positive_value", 2.5))
        self.cones_negative_value = float(getattr(self.config, "cones_negative_value", -0.02))
        self.cones_use_sim_std = bool(getattr(self.config, "cones_use_sim_std", 1))
        self.cones_use_modifier_only = bool(getattr(self.config, "cones_use_modifier_only", 1))
        self.cones_focus_fg_only = bool(getattr(self.config, "cones_focus_fg_only", 1))
        self.cones_use_plain_prompt_before_fusion = bool(
            getattr(self.config, "cones_use_plain_prompt_before_fusion", 1)
        )
        self.cones_debug_tokens = bool(getattr(self.config, "cones_debug_tokens", 0))
        # Extra gate for noisy per-step token-map prints; default off.
        self.cones_debug_token_maps = bool(getattr(self.config, "cones_debug_token_maps", 0))
        self._cones_masks_ready = False
        self._cones_mask_cache = {}
        self._cones_guidance_timestep_set = None
        self._cones_token_map = {}
        self._cones_token_map_plain = {}
        self._cones_debug_seen = set()
        self.use_sts_kv_hook = bool(getattr(self.config, "use_sts_kv_hook", 1))
        self._sts_have_unet_kv = all(
            isinstance(st, dict) and isinstance(st.get("unet", None), dict) for st in self.sts
        )
        fg_concept_limit = self.concept_num - 1 if (self.cones_focus_fg_only and self.concept_num > 1) else self.concept_num
        for concept_idx, concept_label in enumerate(concept_labels):
            if concept_idx >= fg_concept_limit:
                self._cones_token_map[concept_idx] = []
                self._cones_token_map_plain[concept_idx] = []
                continue

            prompt_idx = min(concept_idx + 1, len(prompts) - 1)
            prompt_text = prompts[prompt_idx]
            modifier_id = modifier_token_id[concept_idx] if concept_idx < len(modifier_token_id) else None
            concept_tokens = sorted(
                [idx for idx, label in self.subject_token_labels.items() if label == concept_label]
            )
            if self.cones_use_modifier_only and modifier_id is not None:
                # NOTE: Cones needs token POSITIONS in the text sequence, not vocab ids.
                concept_tokens = self._auto_cones_token_indices(
                    prompt_text=prompt_text,
                    concept_label=concept_label,
                    modifier_token_id=modifier_id,
                )
            if not concept_tokens and concept_idx < len(self.subject_token_ids):
                concept_tokens = [self.subject_token_ids[concept_idx]]
            if not concept_tokens:
                concept_tokens = self._auto_cones_token_indices(
                    prompt_text=prompt_text,
                    concept_label=concept_label,
                    modifier_token_id=modifier_id,
                )
            self._cones_token_map[concept_idx] = concept_tokens
            self._cones_token_map_plain[concept_idx] = self._auto_cones_token_indices(
                prompt_text=prompt_orig,
                concept_label=concept_label,
                modifier_token_id=None,
            )
        print(
            f"[ConesMask2] token map sizes personalized="
            f"{[len(v) for _, v in sorted(self._cones_token_map.items())]}, "
            f"plain={ [len(v) for _, v in sorted(self._cones_token_map_plain.items())] }"
        )

        # 获取单独 prompt 的文本嵌入
        # torch.tensor([3, 77, 2048]),  # 2个prompt + 1个negative
        # torch.tensor([3, 2048])       # 对应的pooled嵌入
        self.text_embeds_single = self.get_text_embeds(prompts_single, null_prompt, device=self.unet.device)
        # Prepare clean single-concept embeds BEFORE building anchors so that --sem_binding_use_clean can take effect
        self.text_embeds_single_clean = None
        if prompts_single_clean:
            embeds_clean = self.get_text_embeds(prompts_single_clean, null_prompt, device=self.unet.device)
            self.text_embeds_single_clean = embeds_clean
        # 专供重采样使用的“干净单概念”嵌入（如 PROMPT_CLEAN_resample）
        self.text_embeds_single_resample = None
        if len(self.prompts_single_resample) > 0:
            embeds_resample = self.get_text_embeds(self.prompts_single_resample, null_prompt, device=self.unet.device)
            self.text_embeds_single_resample = embeds_resample
        # Now build semantic anchors (will choose clean or default based on --sem_binding_use_clean)
        self.semantic_anchor_pairs = self._prepare_semantic_anchors()
        # 释放分词器、编码器等资源，节省显存
        del self.tokenizer, self.tokenizer_2, self.text_encoder, self.text_encoder_2
        del pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2, pipe.unet, pipe.vae
        gc.collect()
        torch.cuda.empty_cache()
        
        # 低显存优先：直接从 checkpoint 的 sts['unet'] 挂载 K/V（不再额外加载 unet_i）。
        # 当 checkpoint 不含 unet 权重或显式关闭该模式时，回退到旧路径。
        if self.use_sts_kv_hook and self._sts_have_unet_kv:
            print("[ConesMask2] Using sts K/V hook path (no extra per-concept UNet load).")
        else:
            # 为每个自定义权重加载一份 UNet，并替换 attn2 层参数
            for i, single_st in enumerate(self.sts):
                try:
                    concept_unet = UNet2DConditionModel.from_pretrained(
                        model_key, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
                    ).to(self.device)
                except OSError as exc:
                    msg = str(exc)
                    if "fp16" not in msg:
                        raise
                    print(f"[WARN] fp16 variant not found for UNet ({model_key}/unet), fallback to default weights.")
                    concept_unet = UNet2DConditionModel.from_pretrained(
                        model_key, subfolder="unet", torch_dtype=torch.float16
                    ).to(self.device)
                setattr(
                    self,
                    f"unet_{i}",
                    concept_unet
                )
                model_name = f"unet_{i}"
                for name, params in getattr(self, model_name).named_parameters():
                    if 'attn2' in name:
                        if name in single_st['unet']:
                            params.data.copy_(single_st['unet'][f'{name}'])
                if self.use_xformers:
                    try:
                        getattr(self, model_name).enable_xformers_memory_efficient_attention()
                    except Exception as exc:
                        print(f"[WARN] failed to enable xformers on {model_name}: {exc}; continue without xformers.")
                        self.use_xformers = False
        # 'vae': AutoencoderKL(...),
        # 'unet': UNet2DConditionModel(...),  # 基础UNet
        # 'unet_0': UNet2DConditionModel(...), # 熊猫概念UNet
        # 'unet_1': UNet2DConditionModel(...), # 泰迪熊概念UNet
        # 'unet_2': UNet2DConditionModel(...), # 城堡概念UNet
        # 'scheduler': DDIMScheduler(...),
        # 'text_embeds': (torch.tensor([5, 77, 2048]), torch.tensor([5, 2048])),
        # 'text_embeds_single': (torch.tensor([3, 77, 2048]), torch.tensor([3, 2048])),
        # 'concept_num': 3,
        # 'sts': [weight_dict_0, weight_dict_1, weight_dict_2],



        # 加载调度器（采样器）
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        N_ts = len(self.scheduler.timesteps)  # 总步数
        self.scheduler.set_timesteps(config.n_timesteps, device=self.unet.device)  # 设置采样步数
        
        self.skip = N_ts // config.n_timesteps  # 步长
        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(self.unet.device)  # 最终 alpha
        # 拼接 alphas_cumprod，首位加 1.0
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

        print('custom checkpoint loaded')  # 打印加载完成

        # 计算并保存时间编码
        self.add_time_ids = compute_time_ids()
        self.add_time_ids = self.add_time_ids.to(self.unet.device)

        # 语义绑定参数
        self.sem_binding_steps = int(getattr(self.config, "sem_binding_steps", 0))
        self.sem_binding_lr = float(getattr(self.config, "sem_binding_lr", 1e-3))
        self.sem_binding_applied = False
        # 在融合阶段允许执行语义绑定的时间步次数（默认仅执行一次）
        self.sem_binding_max_steps = int(getattr(self.config, "sem_binding_max_steps", 1))
        self.sem_binding_applied_steps = 0

        # 低显存融合（串行概念推理），避免一次性堆叠所有概念分支
        self.fusion_low_mem = bool(getattr(self.config, "fusion_low_mem", 0))
        self._entropy_disabled = False

    def _load_binding_entries(self, binding_json):
        entries = []
        if not binding_json:
            binding_json = os.getenv("TOME_TOKEN_BINDINGS", "")
        print("[Binding] raw:", binding_json) 
        if not binding_json:
            return entries
        try:
            if os.path.isfile(binding_json):
                with open(binding_json, "r", encoding="utf-8") as f:
                    binding_json = f.read()
            print("[Binding] after file load:", binding_json)
        except OSError:
            pass
        try:
            entries = json.loads(binding_json)
            print("[Binding] parsed entries:", entries)
        except Exception as exc:
            print(f"[Binding] failed to parse binding_json: {exc}")
            entries = []
        return entries

    def _apply_token_merging_and_ets(self, embeds_tuple, prompt_list, binding_entries):
        if embeds_tuple is None or not binding_entries:
            return embeds_tuple, {}, {}, {}
        text_embeddings, pooled_embeddings = embeds_tuple
        if text_embeddings is None:
            return embeds_tuple, {}, {}, {}
        subject_idx_map = {}
        subject_group_map = {}
        clean_embed_cache = {}
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        max_length = getattr(self.tokenizer, "model_max_length", 77)
        debug_binding = bool(getattr(self.config, "debug_binding", 0))

        def _find_prompt_index(target_text):
            for idx, text in enumerate(prompt_list):
                if text == target_text:
                    return idx
            return None

        for entry in binding_entries:
            prompt_text = entry.get("prompt_text")
            if not prompt_text:
                continue
            prompt_idx = _find_prompt_index(prompt_text)
            if prompt_idx is None:
                continue
            embed_idx = prompt_idx + 1  # index 0 is negative prompt
            if embed_idx >= text_embeddings.shape[0]:
                continue
            embed_slice = text_embeddings[embed_idx].clone()
            prompt_ids = tokenize_prompt(self.tokenizer, prompt_text).to(embed_slice.device)[0]
            if debug_binding:
                try:
                    toks = self.tokenizer.convert_ids_to_tokens(prompt_ids.tolist())
                    printable = []
                    for i, tk in enumerate(toks):
                        if i >= max_length:
                            break
                        printable.append(f"{i}:{tk}")
                    print("[Debug][Binding] prompt_text tokens:")
                    print("  " + " | ".join(printable))
                except Exception as _:
                    pass

            for pair in entry.get("pairs", []):
                subject_indices = pair.get("subject") or []
                if not subject_indices:
                    continue
                subject_indices = [int(si) for si in subject_indices]
                # Track exact grouping order for downstream features (e.g., entropy masks)
                subject_group_map.setdefault(prompt_text, []).append(list(subject_indices))
                subject_tensor = torch.tensor(subject_indices, device=embed_slice.device, dtype=torch.long)
                merged_vec = embed_slice[subject_tensor].sum(dim=0)
                attr_groups = []
                for key in ("attributes", "modifiers", "extras", "additional"):
                    attr_groups.extend(pair.get(key, []))
                if debug_binding:
                    print(f"[Debug][Binding] subject={subject_indices} attrs={attr_groups}")
                for group in attr_groups:
                    if not group:
                        continue
                    attr_tensor = torch.tensor(group, device=embed_slice.device, dtype=torch.long)
                    merged_vec = merged_vec + embed_slice[attr_tensor].sum(dim=0)
                    embed_slice[attr_tensor] = 0.0
                if subject_tensor.numel() > 1:
                    embed_slice[subject_tensor[1:]] = 0.0
                embed_slice[subject_tensor[0]] = merged_vec
                subject_idx_map.setdefault(prompt_text, set()).update(subject_indices)

            clean_prompt = entry.get("eot_clean_prompt")
            if clean_prompt and eos_token_id is not None:
                if clean_prompt not in clean_embed_cache:
                    clean_embeds = self.get_text_embeds([clean_prompt], [self.config.negative_prompt], device=self.unet.device)
                    clean_seq = clean_embeds[0][1].to(embed_slice.device, dtype=embed_slice.dtype)
                    clean_embed_cache[clean_prompt] = clean_seq
                clean_slice = clean_embed_cache[clean_prompt]
                mask = prompt_ids == eos_token_id
                if mask.any():
                    embed_slice[mask] = clean_slice[mask]
                    if debug_binding:
                        print(f"[Debug][Binding] ETS applied from clean prompt at EOT positions.")

            text_embeddings[embed_idx] = embed_slice

        subject_idx_map = {k: sorted(v) for k, v in subject_idx_map.items()}
        subject_group_map = {
            key: [list(group) for group in groups] for key, groups in subject_group_map.items()
        }
        if debug_binding:
            print(f"[Debug][Binding] merged subjects per prompt: {subject_idx_map}")
        return (text_embeddings, pooled_embeddings), subject_idx_map, clean_embed_cache, subject_group_map

    def _activate_token_binding(self):
        if getattr(self, "_binding_active", False):
            print("[Binding] already active")
            return
        if not self.binding_entries:
            print("[Binding] activate skipped: no entries")
            return
        print("[Binding] activating with entries:", self.binding_entries)
        seq_merged, pool_merged = self.text_embeds_merged
        self.text_embeds = (
            seq_merged.clone().to(device=self.unet.device, dtype=self.unet.dtype),
            pool_merged.clone().to(device=self.unet.device, dtype=self.unet.dtype),
        )
        self.binding_subject_indices = list(self.binding_subject_indices_merged)
        self.binding_subject_groups = self.binding_subject_groups_merged.get(
            self.binding_prompt_text, []
        )
        print("[Binding] subject indices:", self.binding_subject_indices)
        cache_converted = {}
        for key, value in (self.binding_clean_cache_merged or {}).items():
            if isinstance(value, torch.Tensor):
                cache_converted[key] = value.to(device=self.unet.device, dtype=self.unet.dtype)
            elif isinstance(value, tuple) and len(value) == 2:
                cache_converted[key] = (
                    value[0].to(device=self.unet.device, dtype=self.unet.dtype),
                    value[1].to(device=self.unet.device),
                )
            else:
                cache_converted[key] = value
        self.binding_clean_cache = cache_converted
        self._binding_entries = self.binding_entries
        prompt_to_index = {}
        for entry in self.binding_entries:
            prompt_text = entry.get("prompt_text")
            if prompt_text and prompt_text in self._binding_prompt_list:
                prompt_to_index[prompt_text] = self._binding_prompt_list.index(prompt_text) + 1
        self._binding_prompt_to_index = prompt_to_index
        self._binding_clean_cache = self.binding_clean_cache
        self._binding_active = True

    def _prepare_semantic_anchors(self):
        """
        Build teacher anchors for semantic binding.

        Default (backwards compatible): use prompts_single as anchors.
        If --sem_binding_use_clean is set and PROMPT_CLEAN was provided (text_embeds_single_clean not None),
        prefer the clean single-concept anchors so that subjects (e.g., <panda1> panda wearing tie)
        include personalized tokens during MSE alignment.
        """
        anchors = []
        # Whether to prefer clean single-concept prompts as anchors
        use_clean = bool(getattr(self.config, "sem_binding_use_clean", 0))
        debug_binding = bool(getattr(self.config, "debug_binding", 0))

        embeds_tuple = None
        if use_clean and getattr(self, "text_embeds_single_clean", None) is not None:
            embeds_tuple = self.text_embeds_single_clean
        else:
            embeds_tuple = self.text_embeds_single

        if embeds_tuple is None:
            if debug_binding:
                print("[Debug][Binding] no anchors available (embeds_tuple is None)")
            return anchors

        seq_single, pool_single = embeds_tuple
        if seq_single is None or seq_single.shape[0] <= 1:
            if debug_binding:
                print("[Debug][Binding] anchors tensor empty or only negative prompt present")
            return anchors

        # Optional: decode each anchor's prompt into wordpieces for verification
        if debug_binding:
            try:
                src_name = "PROMPT_CLEAN" if (use_clean and getattr(self, "text_embeds_single_clean", None) is not None) else "PROMPT"
                src_prompts = getattr(self, "prompts_single_clean", None) if src_name == "PROMPT_CLEAN" else getattr(self, "prompts_single", None)
                if src_prompts is not None and hasattr(self, "tokenizer"):
                    print(f"[Debug][Binding] anchors source={src_name}, count={seq_single.shape[0]-1}")
                    for aidx in range(1, seq_single.shape[0]):
                        text = src_prompts[aidx-1] if (aidx-1) < len(src_prompts) else "<unknown>"
                        ids = tokenize_prompt(self.tokenizer, text)[0]
                        toks = self.tokenizer.convert_ids_to_tokens(ids.tolist())
                        max_len = getattr(self.tokenizer, "model_max_length", len(toks))
                        printable = []
                        for i, tk in enumerate(toks):
                            if i >= max_len:
                                break
                            printable.append(f"{i}:{tk}")
                        print(f"  [Anchor {aidx-1}] {text}")
                        print("    " + " | ".join(printable))
            except Exception:
                pass

        for idx in range(1, seq_single.shape[0]):
            anchor_seq = seq_single[idx:idx + 1]
            anchor_pool = pool_single[idx:idx + 1]
            anchors.append({
                "seq": anchor_seq.to(device=self.unet.device, dtype=self.unet.dtype),
                "pool": anchor_pool.to(device=self.unet.device, dtype=self.unet.dtype),
            })
        if debug_binding:
            src = "PROMPT_CLEAN" if (use_clean and getattr(self, "text_embeds_single_clean", None) is not None) else "PROMPT"
            print(f"[Debug][Binding] anchors built: {len(anchors)} from {src}")
        return anchors

    def _semantic_binding_step(self, latents, t):
        print(f"[Binding] 第{self.sem_binding_applied_steps + 1}次语义绑定，t={t.item() if torch.is_tensor(t) else t}")

        if self.sem_binding_steps <= 0 or not self.binding_subject_indices:
            return
        if not getattr(self, "_binding_active", False):
            return
        if not getattr(self, "semantic_anchor_pairs", None):
            return
        if not self.semantic_anchor_pairs:
            return
        seq_embeds, pooled_embeds = self.text_embeds
        seq_local = seq_embeds.clone()
        subject_tensor = torch.tensor(self.binding_subject_indices, device=seq_local.device, dtype=torch.long)
        if subject_tensor.numel() == 0:
            return
        
        latents_single = latents.detach().to(device=self.unet.device, dtype=self.unet.dtype)
        time_ids_single = self.add_time_ids
        device_type = self.unet.device.type
        
        for _ in range(self.sem_binding_steps):
            stokens = seq_local[1, subject_tensor].detach().clone().requires_grad_(True)
            seq_step = seq_local.clone()
            seq_step[1, subject_tensor] = stokens
            cond_seq = seq_step[1:2]
            cond_pool = pooled_embeds[1:2]
            cond_kwargs = {"time_ids": time_ids_single, "text_embeds": cond_pool}
            autocast_ctx = torch.autocast(device_type=device_type, dtype=self.unet.dtype) if device_type == "cuda" else nullcontext()
            with autocast_ctx:
                noise_token = self.unet(
                    latents_single,
                    t,
                    encoder_hidden_states=cond_seq,
                    added_cond_kwargs=cond_kwargs,
                )["sample"]

            loss = 0.0
            for anchor in self.semantic_anchor_pairs:
                anchor_kwargs = {"time_ids": time_ids_single, "text_embeds": anchor["pool"]}
                with torch.no_grad():
                    anchor_ctx = torch.autocast(device_type=device_type, dtype=self.unet.dtype) if device_type == "cuda" else nullcontext()
                    with anchor_ctx:
                        noise_anchor = self.unet(
                            latents_single,
                            t,
                            encoder_hidden_states=anchor["seq"],
                            added_cond_kwargs=anchor_kwargs,
                        )["sample"]
                loss = loss + F.mse_loss(noise_token, noise_anchor)

            loss = loss / max(len(self.semantic_anchor_pairs), 1)
            print(f"[Binding] step loss: {loss.item():.6f}")
            grad = torch.autograd.grad(loss, stokens, retain_graph=False, allow_unused=False)[0]
            if grad is None:
                break
            stokens = (stokens - self.sem_binding_lr * grad).detach()
            seq_local[1, subject_tensor] = stokens

        self.text_embeds = (seq_local, pooled_embeds)

    def _prepare_entropy_targets(self):
        if not self.enable_attention_entropy or self.masks is None:
            self._entropy_masks_ready = False
            self._entropy_masks_by_grid.clear()
            return
        valid_labels = set(self._entropy_token_map.keys())
        if self._entropy_focus_labels:
            valid_labels &= self._entropy_focus_labels
        if not valid_labels:
            self._entropy_masks_ready = False
            self._entropy_masks_by_grid.clear()
            return
        self._entropy_masks_by_grid.clear()
        self._entropy_masks_ready = True

    def _prepare_cones_targets(self):
        self._cones_mask_cache.clear()
        self._cones_masks_ready = bool(self.enable_cones_mask_attention and self.masks is not None)

    def _cones_subject_limit(self):
        if self.masks is None:
            return 0
        concept_limit = min(int(getattr(self, "concept_num", 0)), int(self.masks.shape[0]))
        if self.cones_focus_fg_only and concept_limit > 1:
            return concept_limit - 1
        return concept_limit

    def _auto_cones_token_indices(self, prompt_text, concept_label, modifier_token_id=None):
        token_ids = tokenize_prompt(self.tokenizer, prompt_text)[0].tolist()
        positions = []
        if modifier_token_id is not None:
            positions = [idx for idx, tid in enumerate(token_ids) if tid == int(modifier_token_id)]

        if not positions:
            concept_ids = self.tokenizer(
                concept_label,
                add_special_tokens=False,
                truncation=True,
            )["input_ids"]
            if concept_ids:
                width = len(concept_ids)
                for start in range(0, len(token_ids) - width + 1):
                    if token_ids[start:start + width] == concept_ids:
                        positions.extend(range(start, start + width))

        if not positions:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            for idx, tid in enumerate(token_ids):
                if bos_id is not None and tid == bos_id:
                    continue
                if eos_id is not None and tid == eos_id:
                    continue
                if pad_id is not None and tid == pad_id:
                    continue
                positions.append(idx)

        return sorted(set(int(v) for v in positions))

    def _infer_cones_query_hw(self, query_len, source_h, source_w):
        if query_len <= 0:
            return None
        square = int(math.sqrt(query_len))
        if square * square == query_len:
            return square, square
        if source_h <= 0 or source_w <= 0:
            return None

        target_aspect = float(source_h) / float(max(source_w, 1))
        best_hw = None
        best_err = float("inf")
        for h in range(1, int(math.sqrt(query_len)) + 1):
            if query_len % h != 0:
                continue
            w = query_len // h
            for cand_h, cand_w in ((h, w), (w, h)):
                cand_aspect = float(cand_h) / float(max(cand_w, 1))
                err = abs(cand_aspect - target_aspect)
                if err < best_err:
                    best_err = err
                    best_hw = (cand_h, cand_w)
        return best_hw

    def _resize_cones_mask_flat(self, source, cache_key, query_len, device, dtype):
        if cache_key not in self._cones_mask_cache:
            source_h, source_w = int(source.shape[-2]), int(source.shape[-1])
            target_hw = self._infer_cones_query_hw(query_len, source_h, source_w)
            if target_hw is None:
                return None
            target_h, target_w = target_hw
            resized = F.interpolate(
                source.to(dtype=torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).clamp_(0.0, 1.0)
            self._cones_mask_cache[cache_key] = resized.reshape(query_len)
        return self._cones_mask_cache[cache_key].to(device=device, dtype=dtype)

    def _get_cones_mask_flat(self, concept_idx, query_len, device, dtype):
        if not self._cones_masks_ready or self.masks is None:
            return None
        concept_limit = self._cones_subject_limit()
        if concept_idx < 0 or concept_idx >= concept_limit:
            return None
        source = self.masks[concept_idx:concept_idx + 1]
        return self._resize_cones_mask_flat(
            source,
            ("target", concept_idx, query_len),
            query_len,
            device,
            dtype,
        )

    def _get_cones_irrelevant_mask_flat(self, concept_idx, query_len, device, dtype):
        if not self._cones_masks_ready or self.masks is None:
            return None
        concept_limit = self._cones_subject_limit()
        if concept_limit <= 1:
            return torch.zeros((query_len,), device=device, dtype=dtype)

        source_masks = []
        for idx in range(concept_limit):
            if idx == concept_idx:
                continue
            source_masks.append(self.masks[idx:idx + 1].to(dtype=torch.float32))
        if not source_masks:
            return torch.zeros((query_len,), device=device, dtype=dtype)

        union_source = torch.stack(source_masks, dim=0).amax(dim=0)
        flat = self._resize_cones_mask_flat(
            union_source,
            ("irrelevant", concept_idx, query_len),
            query_len,
            device,
            dtype,
        )
        if flat is None:
            return None
        return flat

    def _cones_eta(self, timestep, sim):
        t_val = 0.0
        if timestep is not None:
            t_val = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)
        max_t = 1000.0
        if hasattr(self, "scheduler") and hasattr(self.scheduler, "timesteps") and len(self.scheduler.timesteps) > 0:
            first_t = self.scheduler.timesteps[0]
            max_t = float(first_t.item()) if isinstance(first_t, torch.Tensor) else float(first_t)
            max_t = max(max_t, 1.0)
        t_norm = max(0.0, min(1.0, t_val / max_t))
        eta = self.cones_guidance_weight * math.log1p(t_norm * t_norm)
        if self.cones_use_sim_std:
            eta *= float(sim.detach().std().item())
        return eta

    def apply_cones_attention_bias(self, sim, heads, num_batch, key_len, timestep, active_concept_idx=None):
        if not self._cones_masks_ready or not self.enable_cones_mask_attention:
            return sim
        if key_len <= 0 or sim.ndim != 3:
            return sim

        t_int = None
        if timestep is not None:
            t_int = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        in_fusion = bool(
            t_int is not None
            and hasattr(self, "_t_cond_timestep_set")
            and self._t_cond_timestep_set is not None
            and t_int in self._t_cond_timestep_set
        )

        # Step gating is defined on fusion timesteps; keep pre-fusion plain-prompt
        # guidance available when enabled.
        if in_fusion and self._cones_guidance_timestep_set is not None:
            if t_int not in self._cones_guidance_timestep_set:
                return sim

        query_len = sim.shape[1]
        eta = self._cones_eta(timestep, sim)
        if eta == 0.0:
            return sim

        batch_concept_map = {}
        token_map = self._cones_token_map
        if in_fusion:
            if num_batch == (1 + self.concept_num):
                for batch_idx in range(1, min(num_batch, self.concept_num + 1)):
                    batch_concept_map[batch_idx] = [batch_idx - 1]
            elif num_batch == 2 and active_concept_idx is not None:
                batch_concept_map[1] = [int(active_concept_idx)]
            else:
                return sim
        else:
            if not self.cones_use_plain_prompt_before_fusion:
                return sim
            subject_limit = self._cones_subject_limit()
            if subject_limit <= 0 or num_batch < 2:
                return sim
            token_map = self._cones_token_map_plain if self._cones_token_map_plain else self._cones_token_map
            # For non-fusion phase, branch 1 is the multi-concept prompt branch.
            batch_concept_map[1] = list(range(subject_limit))

        if self.cones_debug_tokens and self.cones_debug_token_maps and t_int is not None:
            phase = "fusion" if in_fusion else "pre_fusion"
            debug_key = (phase, t_int)
            if debug_key not in self._cones_debug_seen:
                entries = []
                for batch_idx, concept_indices in batch_concept_map.items():
                    for concept_idx in concept_indices:
                        token_indices = [
                            idx for idx in token_map.get(concept_idx, []) if 0 <= idx < key_len
                        ]
                        if not token_indices:
                            continue
                        preview = token_indices[:8]
                        tail = "..." if len(token_indices) > 8 else ""
                        entries.append(f"b{batch_idx}:c{concept_idx}->{preview}{tail}")
                source = "personalized" if in_fusion else "plain"
                entry_text = "; ".join(entries) if entries else "none"
                print(
                    f"[ConesMask2][Debug] phase={phase} t={t_int} source={source} "
                    f"key_len={key_len} token_maps={entry_text}"
                )
                self._cones_debug_seen.add(debug_key)

        for batch_idx, concept_indices in batch_concept_map.items():
            bias = torch.zeros((query_len, key_len), device=sim.device, dtype=sim.dtype)
            has_bias = False
            for concept_idx in concept_indices:
                token_indices = [
                    idx for idx in token_map.get(concept_idx, []) if 0 <= idx < key_len
                ]
                if not token_indices:
                    continue

                mask_flat = self._get_cones_mask_flat(concept_idx, query_len, sim.device, sim.dtype)
                if mask_flat is None:
                    continue
                irrelevant_flat = self._get_cones_irrelevant_mask_flat(
                    concept_idx, query_len, sim.device, sim.dtype
                )
                if irrelevant_flat is None:
                    continue

                spatial_bias = torch.zeros_like(mask_flat)
                spatial_bias = torch.where(
                    mask_flat > 0.5,
                    torch.full_like(mask_flat, self.cones_positive_value),
                    spatial_bias,
                )
                spatial_bias = torch.where(
                    (mask_flat <= 0.5) & (irrelevant_flat > 0.5),
                    torch.full_like(mask_flat, self.cones_negative_value),
                    spatial_bias,
                )
                bias[:, token_indices] = bias[:, token_indices] + spatial_bias.unsqueeze(-1).expand(
                    -1, len(token_indices)
                )
                has_bias = True

            if not has_bias:
                continue
            start = batch_idx * heads
            end = min((batch_idx + 1) * heads, sim.shape[0])
            if start >= end:
                continue
            sim[start:end] = sim[start:end] + eta * bias

        return sim

    def _get_entropy_mask(self, label, grid):
        if not self._entropy_masks_ready or self.masks is None:
            return None
        cache = self._entropy_masks_by_grid.setdefault(grid, {})
        if label in cache:
            return cache[label]
        concept_labels = self.config.concepts.split('+')
        try:
            mask_idx = concept_labels.index(label)
        except ValueError:
            return None
        source = self.masks[mask_idx:mask_idx + 1].to(dtype=torch.float32)
        resized = F.interpolate(
            source,
            size=(grid, grid),
            mode="bilinear",
            align_corners=False,
        ).clamp_(0.0, 1.0)
        flattened = resized.view(1, 1, grid * grid)
        cache[label] = flattened
        return cache[label]

    def _reset_entropy_accumulator(self):
        self._entropy_loss_terms = []
        if self._entropy_layer_filter == set():
            self._entropy_active = False
        else:
            self._entropy_active = self.enable_attention_entropy and self._entropy_masks_ready
        # ✅ 不在这里重置_stored_attention_maps和_collecting_attention
        # 让optimization loop自己管理这两个属性
    
    def _store_attention_for_entropy(self, attn, heads, query_len, key_len, place):
        """存储detached attention maps（不计算loss）"""
        # ✅ 添加调试
        collecting = getattr(self, '_collecting_attention', False)
        if not hasattr(self, '_store_debug_once'):
            self._store_debug_once = True
            print(f"[Store] _collecting_attention={collecting}, has_list={hasattr(self, '_stored_attention_maps')}")
        
        if not collecting:
            return
        
        # ✅ 确保列表存在
        if not hasattr(self, '_stored_attention_maps'):
            self._stored_attention_maps = []
        
        self._stored_attention_maps.append({
            'attn': attn,  # 已经detached
            'heads': heads,
            'query_len': query_len,
            'key_len': key_len,
            'place': place
        })
        
        # ✅ 打印首次成功存储
        if len(self._stored_attention_maps) == 1:
            print(f"[Store] 首次存储attention: place={place}, shape={attn.shape}")

    def _accumulate_attention_entropy(self, attn, heads, query_len, key_len, place):
        if not self._entropy_active or self.attn_entropy_weight <= 0.0:
            return
        if query_len < 1:
            return
        batch = attn.shape[0] // heads
        if batch == 0:
            return
        attn = attn.view(batch, heads, query_len, key_len)
        grid = int(math.sqrt(query_len))
        if grid * grid != query_len:
            return
        device = attn.device
        
        # ✅ 打印注意力收集信息（仅首次）
        if not hasattr(self, '_attn_accum_logged'):
            self._attn_accum_logged = True
            print(f"[AttnAccum] place={place}, grid={grid}x{grid}, batch={batch}, heads={heads}")
            print(f"[AttnAccum] attn.shape={attn.shape}, key_len={key_len}")
            print(f"[AttnAccum] token_map={list(self._entropy_token_map.keys())}")
        
        for label, token_indices in self._entropy_token_map.items():
            if self._entropy_layer_filter is not None:
                if not any(str(place).startswith(prefix) for prefix in self._entropy_layer_filter):
                    continue
            mask_flat = self._get_entropy_mask(label, grid)
            if mask_flat is None:
                continue
            mask_flat = mask_flat.to(device=device, dtype=attn.dtype)
            ds = self.attn_entropy_mask_downscale
            attn_tokens = []
            for idx in token_indices:
                if idx >= key_len:
                    continue
                token_attn = attn[..., idx]
                if ds > 1 and grid % ds == 0:
                    token_attn = token_attn.view(batch, heads, grid // ds, ds, grid // ds, ds).sum(dim=(3, 5))
                    token_attn = token_attn.view(batch, heads, (grid // ds) * (grid // ds))
                    mask_use = mask_flat[:, :, ::ds]
                else:
                    token_attn = token_attn.view(batch, heads, grid * grid)
                    mask_use = mask_flat
                token_attn = token_attn / token_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                outside_mass = (token_attn * (1 - mask_use)).sum(dim=-1)
                inside_mass = (token_attn * mask_use).sum(dim=-1)
                loss = 0.0
                if self.attn_entropy_outside_weight > 0.0:
                    loss = loss + self.attn_entropy_outside_weight * outside_mass.mean()
                if self.attn_entropy_inside_weight > 0.0:
                    loss = loss - self.attn_entropy_inside_weight * torch.log(inside_mass.clamp(min=1e-6)).mean()
                if isinstance(loss, torch.Tensor):
                    self._entropy_loss_terms.append(loss)

    def _attention_entropy_guidance(self, latent, t, mode="fusion"):
        if not self.enable_attention_entropy or self.attn_entropy_weight <= 0.0:
            return
        if not self._entropy_masks_ready or not self._entropy_token_map:
            return
        if getattr(self, "_entropy_disabled", False):
            return
        t_value = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        if t_value < self.attn_entropy_min_step:
            return
        if self.attn_entropy_max_step >= 0 and t_value > self.attn_entropy_max_step:
            return
        if isinstance(t, torch.Tensor):
            timestep = t
        else:
            dtype = self.scheduler.timesteps.dtype if hasattr(self.scheduler, "timesteps") else torch.float32
            timestep = torch.tensor([t], device=self.unet.device, dtype=dtype)[0]
        steps = max(1, self.attn_entropy_steps)
        lr = self.attn_entropy_lr
        text_cond, text_cond_pool = self.text_embeds
        text_cond_local = text_cond.clone()
        if text_cond.shape[0] <= 1:
            return
        branch_indices = [1]  # Only guide multi-condition branch
        print(f"[AttnLoss] 仅约束多概念分支 {branch_indices}")
        
        # ✅ 打印详细调试信息
        print(f"[AttnLoss] mode={mode}, text_cond.shape={text_cond.shape}")
        print(f"[AttnLoss] masks_ready={self._entropy_masks_ready}")
        print(f"[AttnLoss] token_map={list(self._entropy_token_map.keys())}")
        print(f"[AttnLoss] subject_token_ids={self.subject_token_ids}")
        
        latent_model_input = latent.detach()
        target_device = getattr(self.unet, "device", latent_model_input.device)
        latent_model_input = latent_model_input.to(device=target_device, dtype=self.unet.dtype).contiguous()
        device_type = target_device.type if hasattr(target_device, "type") else "cuda"

        use_checkpoint = self.attn_entropy_enable_checkpointing
        checkpoint_was_enabled = getattr(self.unet, "is_gradient_checkpointing", False)
        checkpoint_enabled = False
        if use_checkpoint and not checkpoint_was_enabled and hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()
            checkpoint_enabled = True

        subject_tensor = torch.tensor(
            self.subject_token_ids, device=text_cond_local.device, dtype=torch.long
        ) if self.subject_token_ids else None

        def _inject_subject_tokens(base_tensor, tokens):
            if subject_tensor is None or subject_tensor.numel() == 0:
                return base_tensor
            row = base_tensor[1].clone()
            row.index_copy_(0, subject_tensor, tokens)
            updated = base_tensor.clone()
            updated[1] = row
            return updated

        try:
            for step_idx in range(steps):
                # ✅ 使用ToMe风格的梯度优化：对比两次UNet forward
                with torch.enable_grad():
                    # === 步骤1：用当前stokens执行UNet（带梯度） ===
                    stokens = torch.stack(
                        [text_cond_local[1, idx].clone() for idx in self.subject_token_ids]
                    ).to(text_cond_local.device, dtype=text_cond_local.dtype).requires_grad_(True)
                    
                    text_cond_step = _inject_subject_tokens(text_cond_local, stokens)
                    text_embed = text_cond_step[branch_indices]
                    text_embed_pool = text_cond_pool[branch_indices]
                    cond_kwargs = {
                        "time_ids": self.add_time_ids.repeat(text_embed.shape[0], 1),
                        "text_embeds": text_embed_pool,
                    }
                    
                    # ✅ 启用attention收集（用于监控，不用于loss）
                    self._stored_attention_maps = []
                    self._collecting_attention = True
                    self._reset_entropy_accumulator()
                    
                    print(f"[AttnLoss] step {step_idx + 1}: _collecting_attention=True")
                    
                    # ✅ 执行UNet（保持梯度）
                    noise_pred_current = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=text_embed,
                        added_cond_kwargs=cond_kwargs,
                    )["sample"]
                    
                    self._collecting_attention = False
                    
                    print(f"[AttnLoss] step {step_idx + 1}/{steps}: 收集了 {len(self._stored_attention_maps)} 个attention maps")
                    
                    # === 步骤2：计算spatial loss（用于决定优化方向） ===
                    loss_spatial_value = 0.0
                    num_violations = 0
                    
                    if len(self._stored_attention_maps) > 0:
                        for map_info in self._stored_attention_maps[:10]:
                            attn_detached = map_info['attn']
                            query_len = map_info['query_len']
                            key_len = map_info['key_len']
                            heads = map_info['heads']
                            
                            grid = int(math.sqrt(query_len))
                            if grid * grid != query_len:
                                continue
                            
                            batch = attn_detached.shape[0] // heads
                            attn_spatial = attn_detached.view(batch, heads, query_len, key_len)
                            attn_avg = attn_spatial.mean(dim=1)[0]
                            attn_2d = attn_avg.view(grid, grid, 77)
                            
                            for label, token_indices in self._entropy_token_map.items():
                                concept_attn_list = []
                                for token_idx in token_indices:
                                    if token_idx >= key_len:
                                        continue
                                    token_attn = attn_2d[:, :, token_idx]
                                    concept_attn_list.append(token_attn)
                                
                                if not concept_attn_list:
                                    continue
                                
                                concept_attn = torch.stack(concept_attn_list).mean(dim=0)
                                concept_attn_norm = concept_attn / (concept_attn.sum() + 1e-8)
                                
                                mask_flat = self._get_entropy_mask(label, grid)
                                if mask_flat is not None:
                                    mask_2d = mask_flat.view(grid, grid)
                                    outside_mass = (concept_attn_norm * (1 - mask_2d)).sum()
                                    loss_spatial_value += outside_mass.item()
                                    num_violations += 1
                    
                    # === 步骤3：构建per-concept的spatial-aware loss ===
                    # ✅ 最终方案：为每个概念单独计算loss，使用其spatial violation
                    
                    # 收集每个概念的spatial loss
                    concept_losses = {}  # {label: loss_value}
                    for label in self._entropy_token_map.keys():
                        concept_losses[label] = 0.0
                    
                    # 重新遍历maps，分别统计每个概念的violation
                    for map_info in self._stored_attention_maps[:10]:
                        attn_detached = map_info['attn']
                        query_len = map_info['query_len']
                        key_len = map_info['key_len']
                        heads = map_info['heads']
                        
                        grid = int(math.sqrt(query_len))
                        if grid * grid != query_len:
                            continue
                        
                        batch = attn_detached.shape[0] // heads
                        attn_spatial = attn_detached.view(batch, heads, query_len, key_len)
                        attn_avg = attn_spatial.mean(dim=1)[0]
                        attn_2d = attn_avg.view(grid, grid, 77)
                        
                        for label, token_indices in self._entropy_token_map.items():
                            concept_attn_list = []
                            for token_idx in token_indices:
                                if token_idx >= key_len:
                                    continue
                                token_attn = attn_2d[:, :, token_idx]
                                concept_attn_list.append(token_attn)
                            
                            if not concept_attn_list:
                                continue
                            
                            concept_attn = torch.stack(concept_attn_list).mean(dim=0)
                            concept_attn_norm = concept_attn / (concept_attn.sum() + 1e-8)
                            
                            mask_flat = self._get_entropy_mask(label, grid)
                            if mask_flat is not None:
                                mask_2d = mask_flat.view(grid, grid)
                                outside_mass = (concept_attn_norm * (1 - mask_2d)).sum()
                                concept_losses[label] += outside_mass.item()
                    
                    # 构建per-token的loss
                    loss_total = torch.tensor(0.0, device=stokens.device, dtype=stokens.dtype)
                    token_idx_to_label = {}  # 映射token index到concept label
                    for label, indices in self._entropy_token_map.items():
                        for local_idx, global_idx in enumerate(indices):
                            # 找到这个global_idx在subject_token_ids中的位置
                            if global_idx in self.subject_token_ids:
                                stoken_idx = self.subject_token_ids.index(global_idx)
                                token_idx_to_label[stoken_idx] = label
                    
                    # 为每个stoken添加对应concept的spatial loss
                    for stoken_idx, label in token_idx_to_label.items():
                        if stoken_idx < len(stokens):
                            # 该stoken的能量，加权于其concept的spatial violation
                            token_energy = (stokens[stoken_idx] ** 2).sum()
                            loss_total = loss_total + self.attn_entropy_weight * concept_losses[label] * token_energy
                    
                    print(f"[AttnLoss] concept_losses={concept_losses}")
                    print(f"[AttnLoss] total_spatial={sum(concept_losses.values()):.6f}, loss_total={loss_total.item():.6f}")
                    print(f"[AttnLoss] weight={self.attn_entropy_weight}, lr={lr}")
                    
                    if not loss_total.requires_grad or loss_total.item() == 0:
                        print(f"[AttnLoss] WARNING: loss不可优化，跳过")
                        break
                    
                    # 计算梯度
                    grad = torch.autograd.grad(loss_total, stokens, retain_graph=False)[0]
                    grad_norm_val = grad.norm().item()
                    print(f"[AttnLoss] grad_norm={grad_norm_val:.6f}")
                    
                    # ✅ 如果梯度太小，警告
                    if grad_norm_val < 1e-5:
                        print(f"[AttnLoss] WARNING: 梯度过小，可能需要增大学习率或weight")
                
                # 更新stokens并记录变化
                stokens_old_norm = stokens.norm().item()
                stokens = (stokens - lr * grad).detach()
                stokens_new_norm = stokens.norm().item()
                delta_norm = abs(stokens_new_norm - stokens_old_norm)
                
                print(f"[AttnLoss] stoken update: old_norm={stokens_old_norm:.6f}, new_norm={stokens_new_norm:.6f}, delta={delta_norm:.6f}")
                
                text_cond_local = _inject_subject_tokens(text_cond_local, stokens.to(text_cond_local.dtype))
                
                del text_embed, text_embed_pool, cond_kwargs, noise_pred_current, loss_total, token_energy, grad
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except torch.cuda.OutOfMemoryError:
            print("[AttnLoss] WARNING: OOM during attention-entropy guidance, skipping this step.")
            torch.cuda.empty_cache()
            self._entropy_disabled = True
            del latent_model_input
            return
        finally:
            if checkpoint_enabled and hasattr(self.unet, "disable_gradient_checkpointing"):
                self.unet.disable_gradient_checkpointing()

        self.text_embeds = (text_cond_local, text_cond_pool)
        del latent_model_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_attention_guidance_at_fusion_start(self, latent, t):
        """
        在融合开始前(t=31)应用一次注意力引导
        确保panda*和cat*等概念在正确的空间位置
        """
        if not self.enable_attention_entropy or not self._entropy_masks_ready:
            return
        # 使用attention entropy指引来直接更新subject tokens（仅需uncond+multi分支）
        self._attention_entropy_guidance(latent, t, mode="after_binding")

    def _apply_attention_guidance_after_binding(self, latent, t):
        """
        在语义绑定完成后应用注意力引导
        此时cat*和panda*等super tokens已经形成，需要约束它们的空间位置
        """
        if not self.enable_attention_entropy or not self._entropy_masks_ready:
            return
        self._attention_entropy_guidance(latent, t, mode="after_binding")

    def upcast_vae(self):
        dtype = self.vae.dtype  # 记录当前 VAE 的数据类型（通常为 float16）
        self.vae.to(dtype=torch.float32)  # 将 VAE 整体转换为 float32 精度，提升数值稳定性
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,           # 判断 attention processor 是否为 torch2.0 或 xformers 相关类型
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # 如果使用 xformers 或 torch2.0 的注意力机制，则部分模块可以继续用原始精度（如 float16），节省显存
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)         # post_quant_conv 层恢复为原始精度
            self.vae.decoder.conv_in.to(dtype)         # decoder 的输入卷积层恢复为原始精度
            self.vae.decoder.mid_block.to(dtype)       # decoder 的中间块恢复为原始精度
        
    def find_disc(self,embed,embed2):
        '''
        该方法的主要功能是：
        给定两个嵌入向量 embed 和 embed2，分别在两个文本编码器的 token embedding 空间中，
        查找与它们最相近（点积最大）的 token embedding。
        这种查找可以用于分析自定义 token embedding 与原始词表 embedding 的相似性，或者调试 embedding 的分布情况。

        '''
        with torch.no_grad():  # 关闭梯度计算，加速推理且节省显存
            token_embedding = self.text_encoder.get_input_embeddings()  # 获取第一个文本编码器的 token embedding 层
            token_embedding2 = self.text_encoder_2.get_input_embeddings()  # 获取第二个文本编码器的 token embedding 层

            embedding_matrix = token_embedding.weight  # 获取第一个编码器的所有 token embedding 权重（形状：[vocab_size, hidden_dim]）
            embedding_matrix2 = token_embedding2.weight  # 获取第二个编码器的所有 token embedding 权重

            embed = embed.unsqueeze(0)  # 将输入的 embed 扩展 batch 维度，变成 shape [1, hidden_dim]
            embed2 = embed2.unsqueeze(0)  # 同理，扩展 embed2

            # 在 embedding_matrix 中查找与 embed 最相近的 token（余弦相似度/点积最大）
            hits = semantic_search(
                embed,                      # 查询向量
                embedding_matrix.float(),   # 语料库向量（所有 token embedding）
                query_chunk_size=1,         # 查询分块大小
                top_k=1,                    # 只取最相近的一个
                score_function=dot_score    # 使用点积作为相似度分数
            )
            # 在 embedding_matrix2 中查找与 embed2 最相近的 token
            hits2 = semantic_search(
                embed2,
                embedding_matrix2.float(),
                query_chunk_size=1,
                top_k=1,
                score_function=dot_score
            )

            # 提取最相近 token 的索引（corpus_id），并转为 tensor
            nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=embed.device)
            nn_indices2 = torch.tensor([hit[0]["corpus_id"] for hit in hits2], device=embed.device)
            
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):     
        """
        生成正向和负向提示词的文本嵌入，用于扩散模型的条件控制
        
        该方法会分别对正向 prompt 和负向 prompt 进行编码，得到它们在两个文本编码器下的 embedding 表示。
        最终返回的 text_embeddings 和 pooled_text_embeddings，
        会被用作扩散模型（如 Stable Diffusion）在生成图像时的条件输入。
        
        Args:
            prompt: 正向提示词列表，包含原始prompt和修饰后的概念prompt
            negative_prompt: 负向提示词列表，用于classifier-free guidance
            device: 指定计算设备，默认 "cuda"
        
        Returns:
            text_embeddings: 序列嵌入 [batch_size, 77, 2048]
            pooled_text_embeddings: 全局嵌入 [batch_size, 2048]
            
            其中batch_size = len(prompt) + len(negative_prompt)
            索引0: 负向prompt嵌入
            索引1+: 正向prompt嵌入（按prompt列表顺序）
        
        注意：
            - 使用@torch.no_grad()禁用梯度计算，节省显存和计算时间
            - 双编码器架构提供更强的文本理解能力
            - 负向嵌入在前，正向嵌入在后，用于classifier-free guidance
        """
        # 该方法用于获取正向和负向 prompt 的文本嵌入（embedding），用于扩散模型的条件控制
        # prompt: 正向提示词（可以是字符串或字符串列表）
        # negative_prompt: 负向提示词（通常用于 classifier-free guidance）
        # device: 指定计算设备，默认 "cuda"

        # 对正向 prompt 进行编码，得到其 embedding 和 pooled embedding
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=[self.text_encoder, self.text_encoder_2],   # 使用两个文本编码器
            tokenizers=[self.tokenizer, self.tokenizer_2],            # 对应的两个分词器
            prompt=prompt,                                            # 输入正向 prompt
            text_input_ids_list=None                                  # 不直接传 token id
        )
        # 对负向 prompt 进行编码，得到其 embedding 和 pooled embedding
        uncond_embeds, pooled_uncond_embeds = encode_prompt(
            text_encoders=[self.text_encoder, self.text_encoder_2],   # 同样用两个编码器
            tokenizers=[self.tokenizer, self.tokenizer_2],            # 两个分词器
            prompt=negative_prompt,                                   # 输入负向 prompt
            text_input_ids_list=None
        )
        # 将负向和正向的 embedding 拼接，形成最终的文本条件输入
        # 负向嵌入在前，用于classifier-free guidance的无条件分支
        text_embeddings = torch.cat([uncond_embeds, prompt_embeds])
        # 将 pooled embedding 也拼接
        pooled_text_embeddings = torch.cat([pooled_uncond_embeds, pooled_prompt_embeds])
        # 返回拼接后的 embedding 和 pooled embedding
        return text_embeddings, pooled_text_embeddings
    
    def prepare_extra_step_kwargs(self, generator, eta):
        '''
        该方法根据当前 scheduler（采样器）的 step 方法签名，
        动态判断是否需要传递 eta 和 generator 参数，并将它们组织成一个字典返回。
      这样做可以兼容不同版本或不同类型的 scheduler，避免因参数不兼容导致报错
        '''
        # 该方法用于为扩散采样器 scheduler 的 step 函数准备额外的参数字典
        # generator: 随机数生成器（用于可复现性）
        # eta: 采样噪声参数（影响采样的随机性）

        # 判断 scheduler 的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}  # 初始化参数字典
        if accepts_eta:
            extra_step_kwargs["eta"] = eta  # 如果支持，则添加 eta 参数

        # 判断 scheduler 的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator  # 如果支持，则添加 generator 参数
        return extra_step_kwargs  # 返回最终的参数字典

    @torch.no_grad()
    def _decode_latent_to_pil(self, latent, fallback_latent=None):
        """
        Decode latent with SDXL-safe dtype/device alignment.
        Returns: (pil_images, decoded_tensor)
        """
        x = latent
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()

        # Always align latent dtype/device with VAE input conv to avoid decode instability.
        vae_in_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
        vae_device = next(iter(self.vae.post_quant_conv.parameters())).device
        x = x.to(device=vae_device, dtype=vae_in_dtype)

        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(x.device, x.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(x.device, x.dtype)
            x = x * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            x = x / self.vae.config.scaling_factor

        decoded_latent = self.vae.decode(x, return_dict=False)[0]
        if not torch.isfinite(decoded_latent).all() and fallback_latent is not None:
            print("[WARN] decoded latent non-finite; fallback decode from last healthy latent")
            y = fallback_latent.to(device=vae_device, dtype=vae_in_dtype)
            if has_latents_mean and has_latents_std:
                y = y * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                y = y / self.vae.config.scaling_factor
            decoded_latent = self.vae.decode(y, return_dict=False)[0]

        decoded_latent = torch.nan_to_num(decoded_latent, nan=0.0, posinf=1.0, neginf=-1.0)
        decoded_latent = decoded_latent.clamp_(-1.0, 1.0)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        images = self.image_processor.postprocess(decoded_latent, output_type='pil')
        return images, decoded_latent
    
    @torch.no_grad()
    def decode_latent(self, latent):
        '''
        该方法用于将潜在表示（latent）解码为图像，
        它首先将潜在表示缩放回原始尺度，然后使用 VAE 解码器生成图像。
        '''
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img
    
    def alpha(self, t):
        '''
        该方法用于计算扩散模型的 alpha 值，
        它根据当前时间步 t 从 scheduler 中获取 alpha 值，
        如果 t 小于 0，则使用 final_alpha_cumprod 作为默认值。
        '''
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at
    
    @torch.no_grad()
    def denoise_step(self, x, t):
        """
        MultiCompose多概念融合的核心去噪步骤
        
        这个函数实现了两阶段的扩散去噪过程：
        1. 融合阶段：使用多个概念特定的UNet进行并行处理
        2. 融合后阶段：使用标准UNet进行最终生成
        
        Args:
            x: 当前的潜变量（latent）[1, 4, 128, 128] - 当前步的图像表示
            t: 当前的采样步数 (如 50, 49, 48, ... 1)
        
        Returns:
            denoised_latent: 去噪后的潜变量，用于下一步计算
        
        工作流程：
            1. 根据时间步判断进入融合阶段还是融合后阶段
            2. 在融合阶段：使用多概念UNet进行并行去噪
            3. 在融合后阶段：使用标准UNet进行最终生成
            4. 生成掩码用于概念分离
         Prompt结构说明：
            text_embeds包含5个文本嵌入：
            [0]: 负向prompt "blurry, ugly, black, low res, unrealistic, blurry face"
            [1]: 原始prompt "photo of a panda and a teddybear playing with a ball, castle background"
            [2]: 熊猫概念 "photo of a <panda1> panda playing with a ball, castle background"
            [3]: 泰迪熊概念 "photo of a <teddybear1> teddybear playing with a ball, castle background"
            [4]: 城堡概念 "photo of a panda and a teddybear playing with a ball, <castle1> waterfall background"
        """
        text_embed_cond, text_embed_cond_pool = self.text_embeds  # 获取正负 prompt 的嵌入和 pooled 嵌入

        next_t = t - self.skip  # 计算下一个采样步
        at = self.alpha(t)      # 当前步的 alpha
        at_next = self.alpha(next_t)  # 下一个步的 alpha

        sizes = x.shape  # 记录当前潜变量的形状
        # if self.masks is not None:
        #     register_time(self, t.item(),self.masks)
        # else:
        register_time(self, t.item())  # 记录当前步（如保存中间状态等） # [1, 4, 128, 128]
       
        # "photo of a panda and a teddybear playing with a ball, castle background",
        # "photo of a <panda1> panda playing with a ball, castle background",
        # "photo of a <teddybear1> teddybear playing with a ball, castle background",
        # "photo of a panda and a teddybear playing with a ball, <castle1> castle background"

        if t <= self.t_cond_cur:  # 如果当前步在融合阶段
            """
                融合阶段：使用多个概念特定的UNet进行并行处理
                
                在这个阶段，系统会：
                1. 使用多个概念UNet分别处理每个概念
                2. 通过掩码将不同概念的去噪结果融合
                3. 实现多概念的协调生成
                使用索引[0, 2, 3, 4]的文本嵌入：
                - [0]: 负向prompt用于classifier-free guidance
                - [2]: 熊猫概念prompt，使用<panda1>修饰符
                - [3]: 泰迪熊概念prompt，使用<teddybear1>修饰符
                - [4]: 城堡概念prompt，使用<castle1>修饰符
            """
            text_embed_uncond = text_embed_cond[0].unsqueeze(0)  # ȡ������Ƕ��
            # ���ݣ�����prompt "blurry, ugly, black, low res, unrealistic, blurry face"

            text_embed_concept = text_embed_cond[2:]             # ȡ���������Ƕ��
            # ���ݣ���è����prompt "photo of a <panda1> panda playing with a ball, castle background"
            # ���ݣ�̩���ܸ���prompt "photo of a <teddybear1> teddybear playing with a ball, castle background"
            # ���ݣ��Ǳ�����prompt "photo of a panda and a teddybear playing with a ball, <castle1> castle background"

            text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)  # pooled ������Ƕ��
            text_embed_concept_pool = text_embed_cond_pool[2:]             # pooled ����Ƕ��

            # 4����֧��1������ + 3������
            text_embed = torch.cat([text_embed_uncond, text_embed_concept], dim=0)  # ƴ������Ƕ�� # [4, 77, 2048]
            text_embed_pool = torch.cat([text_embed_uncond_pool, text_embed_concept_pool], dim=0)  # ƴ�� pooled Ƕ�� # [4, 2048]

            if self.fusion_low_mem:
                # �ʹ�ʵ��ģʽ�������ȼ��㲻��������֧��Ȼ��˳��ͨ����������
                cond_kwargs_uncond = {
                    "time_ids": self.add_time_ids[:1],
                    "text_embeds": text_embed_uncond_pool,
                }
                prev_active_concept_idx = getattr(self, "_active_concept_idx", None)
                self._active_concept_idx = None
                noise_pred_uncond_only = self.unet(
                    x, t, encoder_hidden_states=text_embed_uncond, added_cond_kwargs=cond_kwargs_uncond
                )["sample"]
                concept_preds = []
                repeat_time_ids = self.add_time_ids.repeat(2, 1)
                try:
                    for idx in range(text_embed_concept.shape[0]):
                        self._active_concept_idx = int(idx)
                        concept_embed = text_embed_concept[idx:idx + 1]
                        concept_pool = text_embed_concept_pool[idx:idx + 1]
                        pair_embed = torch.cat([text_embed_uncond, concept_embed], dim=0)
                        pair_pool = torch.cat([text_embed_uncond_pool, concept_pool], dim=0)
                        cond_kwargs_pair = {
                            "time_ids": repeat_time_ids,
                            "text_embeds": pair_pool,
                        }
                        latent_pair = torch.cat([x, x])
                        pair_noise = self.unet(
                            latent_pair, t, encoder_hidden_states=pair_embed, added_cond_kwargs=cond_kwargs_pair
                        )["sample"]
                        concept_preds.append(pair_noise[1:2])
                finally:
                    if prev_active_concept_idx is None:
                        self._active_concept_idx = None
                    else:
                        self._active_concept_idx = prev_active_concept_idx
                noise_pred = torch.cat([noise_pred_uncond_only] + concept_preds, dim=0)
            else:
                # ��չǱ���������������
                latent_model_input = torch.cat([x] * (self.concept_num + 1))  # ��չ batch����������# [4, 4, 128, 128]

                # ����UNet�Ķ���������ʱ���������ı�Ƕ��������
                unet_added_conditions = {"time_ids": self.add_time_ids.repeat(text_embed.shape[0], 1)}  # ����ʱ������ # [4, 1]
                unet_added_conditions.update({"text_embeds": text_embed_pool})  # �����ı�Ƕ������ # [4, 2048]

                # ���� UNet���õ�����Ԥ��
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embed, added_cond_kwargs=unet_added_conditions
                )["sample"]  # ���� UNet���õ�����Ԥ�� # [4, 4, 128, 128] - 4����֧������Ԥ��

            # 噪声预测结果：
            # noise_pred[0]: 负向prompt的噪声预测
            # noise_pred[1]: 熊猫概念的噪声预测
            # noise_pred[2]: 泰迪熊概念的噪声预测
            # noise_pred[3]: 城堡概念的噪声预测
        else:  # 否则，进入融合后阶段
            """
           内容感知采样
            
            在这个阶段，系统会：
            1. 使用融合后的UNet进行标准扩散生成
            2. 可选择性进行重采样和跳步采样
            3. 生成最终的高质量图像
            
            Prompt处理逻辑：
            使用索引[0, 1]的文本嵌入：
            - [0]: 负向prompt用于classifier-free guidance
            - [1]: 原始prompt "photo of a panda and a teddybear playing with a ball, castle background"
            """
        
            text_embed_uncond = text_embed_cond[0].unsqueeze(0)  # 负向文本嵌入 [1, 77, 2048]
            # 内容：负向prompt "blurry, ugly, black, low res, unrealistic, blurry face"
            
            text_embed_multi = text_embed_cond[1].unsqueeze(0)   # 多概念融合文本嵌入 [1, 77, 2048]
            # 内容：原始prompt "photo of a panda and a teddybear playing with a ball, castle background"
            
            text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)  # 负向pooled嵌入 [1, 2048]
            text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)   # 多概念pooled嵌入 [1, 2048]

            if t == self.start_t:  # 如果是采样起始步
                """
                采样起始步：需要额外的单独概念处理
                
                在第一步，系统会：
                1. 使用多概念融合UNet
                2. 使用单独的每个概念UNet
                3. 通过重采样优化生成质量
                
                Prompt处理逻辑：
                使用索引[0, 1, 2, 3]的文本嵌入：
                - [0]: 负向prompt
                - [1]: 原始prompt
                - [2]: 熊猫概念prompt（单独）
                - [3]: 泰迪熊概念prompt（单独）
                """
            
                text_embed_cond_single, text_embed_cond_pool_single = self.text_embeds_single  # [3, 77, 2048] 和 [3, 2048]
                # 单独概念prompt：
                # [0]: 负向prompt "blurry, ugly, black, low res, unrealistic, blurry face"
                # [1]:"photo of a panda playing with a ball, castle background",
                # [2]:"photo of a teddybear playing with a ball, castle background"
                
                text_embed_cond_single = text_embed_cond_single[1:]  # 跳过第一个（无条件）[2, 77, 2048]
                text_embed_cond_pool_single = text_embed_cond_pool_single[1:]  # 同上 [2, 2048]
                
                latent_model_input = torch.cat([x] * (self.concept_num + 1))  # [4, 4, 128, 128]
                
                text_embed = torch.cat([
                    text_embed_uncond,    # 负向文本
                    text_embed_multi,     # 多概念融合文本
                    text_embed_cond_single  # 单独概念文本
                ], dim=0)  # [4, 77, 2048]
                
                text_embed_pool = torch.cat([
                    text_embed_uncond_pool,
                    text_embed_multi_pool,
                    text_embed_cond_pool_single
                ], dim=0)  # [4, 2048]

                if (
                    self.sem_binding_steps > 0
                    and getattr(self, "_binding_active", False)
                    and self.binding_subject_indices
                    and self.sem_binding_applied_steps < self.sem_binding_max_steps
                ):
                    with torch.enable_grad():
                        self._semantic_binding_step(x, t)
                    self.sem_binding_applied_steps += 1
                    self.sem_binding_applied = True
                    text_embed_cond, text_embed_cond_pool = self.text_embeds
                    text_embed_uncond = text_embed_cond[0].unsqueeze(0)
                    text_embed_multi = text_embed_cond[1].unsqueeze(0)
                    text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)
                    text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)
                    text_embed = torch.cat([
                        text_embed_uncond,
                        text_embed_multi,
                        text_embed_cond_single
                    ], dim=0)
                    text_embed_pool = torch.cat([
                        text_embed_uncond_pool,
                        text_embed_multi_pool,
                        text_embed_cond_pool_single
                    ], dim=0)
            else:  # 其它步只用无条件和多概念嵌入
                """
                其他步骤：使用简化的两分支处理
                
                在后续步骤中，系统只使用：
                1. 负向文本分支
                2. 多概念融合文本分支
                
                Prompt处理逻辑：
                使用索引[0, 1]的文本嵌入：
                - [0]: 负向prompt
                - [1]: 原始prompt
                """
                latent_model_input = torch.cat([x] + [x])  # [2, 4, 128, 128] - 两分支
                text_embed = torch.cat([text_embed_uncond, text_embed_multi], dim=0)  # [2, 77, 2048]
                text_embed_pool = torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0)  # [2, 2048]

                if (
                    self.sem_binding_steps > 0
                    and getattr(self, "_binding_active", False)
                    and self.binding_subject_indices
                    and self.sem_binding_applied_steps < self.sem_binding_max_steps
                ):
                    with torch.enable_grad():
                        self._semantic_binding_step(x, t)
                    self.sem_binding_applied_steps += 1
                    self.sem_binding_applied = True
                    text_embed_cond, text_embed_cond_pool = self.text_embeds
                    text_embed_uncond = text_embed_cond[0].unsqueeze(0)
                    text_embed_multi = text_embed_cond[1].unsqueeze(0)
                    text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)
                    text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)
                    text_embed = torch.cat([text_embed_uncond, text_embed_multi], dim=0)
                    text_embed_pool = torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0)
        

            unet_added_conditions = {"time_ids": self.add_time_ids.repeat(text_embed.shape[0], 1)}  # 时间条件
            unet_added_conditions.update({"text_embeds": text_embed_pool})  # 文本嵌入条件

            unet_added_conditions_single = {"time_ids": self.add_time_ids.repeat(text_embed[:2].shape[0], 1)}  # 前两个分支的时间条件
            unet_added_conditions_single.update({"text_embeds": text_embed_pool[:2]})  # 前两个分支的文本嵌入

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embed, added_cond_kwargs=unet_added_conditions
            )['sample']  # 输入 UNet，得到噪声预测  # [2或4, 4, 128, 128] - 噪声预测

        noise_pred_uncond = noise_pred[:1]  # 取出无条件分支的噪声预测 # [1, 4, 128, 128] - 无条件噪声预测

        # ❌ 移除t=31的注意力引导（已在语义绑定后执行）
        # 注意力引导现在在重采样+语义绑定完成后执行，时机更合适
        
        if t <= self.t_cond_cur:  # 如果在融合阶段
            """
            融合阶段的噪声处理：使用掩码融合多个概念
            
            在这个阶段，系统会：
            1. 为每个概念计算去噪结果
            2. 使用掩码将不同概念的结果融合
            3. 实现多概念的协调生成
            
            Prompt处理逻辑：
            使用4个分支的噪声预测：
            - noise_pred[0]: 负向prompt的噪声预测
            - noise_pred[1]: 熊猫概念的噪声预测
            - noise_pred[2]: 泰迪熊概念的噪声预测
            - noise_pred[3]: 城堡概念的噪声预测
            """
            # 简单掩码累加版本：直接累加各概念的掩码加权去噪结果，不进行归一化
            # 这种方式假设掩码之间没有重叠，适用于外部boxes定义的场景
            denoised_tweedie = 0

            for cc in range(self.concept_num):
                noise_pred_cond = noise_pred[(1 + cc):(2 + cc)]
                noise_pred_concept = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                mask = self.masks[cc].unsqueeze(0)
                # 直接累加掩码加权的去噪结果（与fusion_sampling.py保持一致）
                denoised_tweedie += mask * ((x - (1 - at).sqrt() * noise_pred_concept) / at.sqrt())

            # 掩码融合过程：
            # self.masks[0]: 熊猫概念的空间掩码
            # self.masks[1]: 泰迪熊概念的空间掩码
            # self.masks[2]: 城堡概念的空间掩码
            # 通过空间掩码确保每个概念在正确的位置出现
        else:  # 融合后阶段
            """
                融合后阶段的噪声处理：使用标准扩散生成
                
                在这个阶段，系统会：
                1. 使用标准UNet进行去噪
                2. 可选择性进行重采样优化
                3. 生成最终的高质量图像
                使用2个分支的噪声预测：
            - noise_pred[0]: 负向prompt的噪声预测
            - noise_pred[1]: 原始prompt的噪声预测
            """
            if t == self.start_t:  # 如果是采样起始步
                if self.config.resampling_steps > 0:  # 如果需要重采样
                    """
                    重采样过程：通过多次迭代优化生成质量
                    
                    重采样的目的是：
                    1. 提高生成质量
                    2. 减少概念间的冲突
                    3. 优化多概念融合效果
                    
                    Prompt处理逻辑：
                    在重采样过程中，使用4个分支的噪声预测：
                    - noise_pred[0]: 负向prompt的噪声预测
                    - noise_pred[1]: 原始prompt的噪声预测
                    - noise_pred[2]: 熊猫概念prompt的噪声预测
                    - noise_pred[3]: 泰迪熊概念prompt的噪声预测
                    """
                    clean_single = None
                    clean_single_pool = None
                    raw_seq, raw_pool = getattr(self, "text_embeds_raw", (None, None))
                    if raw_seq is not None:
                        raw_seq = raw_seq.to(device=self.unet.device, dtype=self.unet.dtype)
                    if raw_pool is not None:
                        raw_pool = raw_pool.to(device=self.unet.device, dtype=self.unet.dtype)
                    if raw_seq is not None and raw_pool is not None:
                        text_embed_uncond_raw = raw_seq[0:1]
                        text_embed_multi_raw = raw_seq[1:2]
                        text_embed_uncond_pool_raw = raw_pool[0:1]
                        text_embed_multi_pool_raw = raw_pool[1:2]
                    else:
                        text_embed_uncond_raw = text_embed_uncond
                        text_embed_multi_raw = text_embed_multi
                        text_embed_uncond_pool_raw = text_embed_uncond_pool
                        text_embed_multi_pool_raw = text_embed_multi_pool
                    # 优先使用“重采样专用”的干净单概念（PROMPT_CLEAN_resample）；否则退回 PROMPT_CLEAN
                    if getattr(self, "text_embeds_single_resample", None) is not None:
                        clean_single, clean_single_pool = self.text_embeds_single_resample
                        # debug
                        if bool(getattr(self.config, "debug_binding", 0)):
                            print("[Debug][Resample] using PROMPT_CLEAN_resample for single-concept guidance")
                    elif self.text_embeds_single_clean is not None:
                        clean_single, clean_single_pool = self.text_embeds_single_clean
                        if bool(getattr(self.config, "debug_binding", 0)):
                            print("[Debug][Resample] using PROMPT_CLEAN for single-concept guidance")
                    clean_single = clean_single.to(device=self.unet.device, dtype=self.unet.dtype)
                    clean_single_pool = clean_single_pool.to(device=self.unet.device, dtype=self.unet.dtype)
                    if clean_single.shape[0] > 1:
                        clean_single = clean_single[1:]
                        clean_single_pool = clean_single_pool[1:]
                    else:
                        clean_single = None
                        clean_single_pool = None
                    multi_clean = getattr(self, "text_embeds_multi_clean", None)
                    for _ in range(self.config.resampling_steps):
                        print('resampling')
                        noise_pred_uncond = noise_pred[:1]
                        if multi_clean is not None:
                            clean_seq, clean_pool = multi_clean
                            latent_model_input_multi = torch.cat([x] + [x]).to(device=self.unet.device, dtype=self.unet.dtype)
                            text_embed_multi_clean = torch.cat([text_embed_uncond_raw, clean_seq], dim=0)
                            text_embed_multi_clean_pool = torch.cat([text_embed_uncond_pool_raw, clean_pool], dim=0)
                            cond_kwargs_multi_clean = {
                                "time_ids": self.add_time_ids.repeat(text_embed_multi_clean.shape[0], 1),
                                "text_embeds": text_embed_multi_clean_pool,
                            }
                            noise_pred_clean_multi = self.unet(
                                latent_model_input_multi,
                                t,
                                encoder_hidden_states=text_embed_multi_clean,
                                added_cond_kwargs=cond_kwargs_multi_clean,
                            )["sample"]
                            noise_pred_mult = noise_pred_uncond + self.config.guidance_scale * (noise_pred_clean_multi[1:2] - noise_pred_uncond)
                        else:
                            noise_pred_mult = noise_pred[1:2]
                            noise_pred_mult = noise_pred_uncond + self.config.guidance_scale * (noise_pred_mult - noise_pred_uncond)
                        denoised_tweedie_mult = (x - (1 - at).sqrt() * noise_pred_mult) / at.sqrt()
                        denoised_tweedie = (self.concept_num - 1) * denoised_tweedie_mult
                        for cc in range(self.concept_num - 1):
                            if clean_single is not None and cc < clean_single.shape[0]:
                                clean_embed = clean_single[cc:cc + 1]
                                clean_pool_embed = clean_single_pool[cc:cc + 1]
                                latent_model_input_clean = torch.cat([x] + [x])
                                clean_text_embed = torch.cat([text_embed_uncond_raw, clean_embed], dim=0)
                                clean_text_embed_pool = torch.cat([text_embed_uncond_pool_raw, clean_pool_embed], dim=0)
                                cond_kwargs_clean = {
                                    'time_ids': self.add_time_ids.repeat(clean_text_embed.shape[0], 1),
                                    'text_embeds': clean_text_embed_pool,
                                }
                                noise_pred_clean_batch = self.unet(
                                    latent_model_input_clean,
                                    t,
                                    encoder_hidden_states=clean_text_embed,
                                    added_cond_kwargs=cond_kwargs_clean,
                                )['sample']
                                noise_pred_single = noise_pred_uncond + self.config.guidance_scale * (
                                    noise_pred_clean_batch[1:2] - noise_pred_uncond
                                )
                            else:
                                noise_pred_single = noise_pred_uncond + self.config.guidance_scale * (
                                    noise_pred[2 + cc:3 + cc] - noise_pred_uncond
                                )
                            denoised_tweedie_single = (x - (1 - at).sqrt() * noise_pred_single) / at.sqrt()
                            denoised_tweedie -= denoised_tweedie_single
                        denoised_latent = at_next.sqrt() * denoised_tweedie + (1 - at_next).sqrt() * noise_pred_uncond
                        latent_model_next = torch.cat([denoised_latent] + [denoised_latent])  # batch=2 # [2, 4, 128, 128]

                        noise_pred_next = self.unet(
                            latent_model_next, next_t, encoder_hidden_states=text_embed[:2], added_cond_kwargs=unet_added_conditions_single
                        )['sample']  # 下一个步的噪声预测

                        #多概念预测的噪声
                        noise_pred_cond_next = noise_pred_next[1:2]
                        noise_pred_uncond_next = noise_pred_next[:1]
                        #预测的噪声    
                        noise_pred_next = noise_pred_uncond_next + self.config.guidance_scale * (noise_pred_cond_next - noise_pred_uncond_next)
                        # 计算去噪结果
                        #xˆ[εθ(xt-1, t, c)] := (xt-1 − √1 − α ̄tεθ(xt-1, t, c))/√α ̄t-1,   x=z     zT −1
                        # 反向采样：从z_{T-1}回到当前步z_T（文档“DDIM forward sampling”）
                        denoised_tweedie_next = (denoised_latent - (1 - at_next).sqrt() * noise_pred_next) / at_next.sqrt()
                        # 回到当前步
                        #
                        return_x = at.sqrt() * denoised_tweedie_next + (1 - at).sqrt() * noise_pred_uncond_next  # 回到当前步
                        # 循环迭代：更新输入，准备下一次重采样（共self.config.resampling_steps次）
                        latent_model_input = torch.cat([return_x] * (self.concept_num + 1))  # batch
                        noise_pred = self.unet(
                            latent_model_input, t, encoder_hidden_states=text_embed, added_cond_kwargs=unet_added_conditions
                        )['sample']  # 再次采样
                    x = return_x  # 更新 x
                if not getattr(self, "_binding_active", False):
                    self._activate_token_binding()
                    text_embed_cond, text_embed_cond_pool = self.text_embeds
                # 在融合阶段的若干时间步执行语义绑定，持续约束属性语义
                if (
                    self.sem_binding_steps > 0
                    and getattr(self, "_binding_active", False)
                    and self.binding_subject_indices
                    and self.sem_binding_applied_steps < self.sem_binding_max_steps
                ):
                    with torch.enable_grad():
                        self._semantic_binding_step(x, t)
                    self.sem_binding_applied_steps += 1
                    self.sem_binding_applied = True
                    text_embed_cond, text_embed_cond_pool = self.text_embeds
                
                # ✅ 在语义绑定完成后，应用注意力引导
                # 此时cat*和panda*已经形成，需要约束它们的空间位置
                if (
                    self.enable_attention_entropy 
                    and self._entropy_masks_ready 
                    and self.sem_binding_applied
                ):
                    print(f"[AttnLoss] 重采样+语义绑定完成，t={t}, 应用注意力引导约束空间位置")
                    self._apply_attention_guidance_after_binding(x, t)
                    text_embed_cond, text_embed_cond_pool = self.text_embeds
                
                text_embed_uncond = text_embed_cond[0].unsqueeze(0)
                text_embed_multi = text_embed_cond[1].unsqueeze(0)
                text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)
                text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)
                text_embed = torch.cat([
                    text_embed_uncond,
                    text_embed_multi,
                    text_embed_cond_single
                ], dim=0)
                text_embed_pool = torch.cat([
                    text_embed_uncond_pool,
                    text_embed_multi_pool,
                    text_embed_cond_pool_single
                ], dim=0)

                del noise_pred_next, noise_pred_cond_next, noise_pred_uncond_next, denoised_tweedie_next, latent_model_next  # 释放变量
                gc.collect()
                torch.cuda.empty_cache()
                # 标准处理
                noise_pred_cond = noise_pred[1:2]
                noise_pred_uncond = noise_pred[:1]
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:  # 其它步
                noise_pred_cond = noise_pred[1:2]
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            #xˆ[εθ(xt, t, c)] := (xt − √1 − α ̄tεθ(xt, t, c))/√α ̄t,
            denoised_tweedie = (x - (1 - at).sqrt() * noise_pred) / at.sqrt()  # 计算去噪结果
        
        #公式9 zt−1 = √α ̄t−1{  N  X  i=1  Mi · zˆ[ε ̃i]} + p1 − α ̄t−1εθ0 (zt, t, ∅),
        denoised_latent = at_next.sqrt() * denoised_tweedie + (1 - at_next).sqrt() * noise_pred_uncond  # 计算下一个步的潜变量
        if t == self.t_cond_prev:  # 如果是关键步，生成掩码
            """
                掩码生成：在关键时间步生成概念分离掩码
                
                掩码的作用：
                1. 分离不同概念的空间位置
                2. 指导多概念融合过程
                3. 确保概念间的协调生成

                触发条件：
                - t == self.t_cond_prev：当前时间步是融合阶段的前一步
                - 例如：如果融合阶段是[30, 29, 28, 27, 26]，那么t_cond_prev = 31
                - 在时间步31时生成掩码，为后续的融合阶段做准备
                
                掩码生成的重要性：
                - 掩码决定了每个概念在图像中的空间位置
                - 没有掩码，不同概念可能会相互干扰
                - 掩码确保熊猫、泰迪熊、城堡各自在正确的位置出现
            """
            use_ext = getattr(self.config, 'use_external_boxes', 0) == 1
            boxes = parse_external_boxes(getattr(self.config, 'external_boxes', ''))
            boxes_only = getattr(self.config, 'boxes_only', 0) == 1
            if use_ext and len(boxes) > 0 and self.masks is not None:
                print(f"[AttnLoss] 使用提前生成的掩码（方案B），跳过tweedie中间解码")
                self._prepare_cones_targets()
                if self.enable_attention_entropy:
                    self._prepare_entropy_targets()
                    print(f"[AttnLoss] 掩码已准备完成")
                    print(f"[AttnLoss] masks.shape={self.masks.shape if self.masks is not None else None}")
                    if self.masks is not None:
                        for i in range(self.masks.shape[0]):
                            mask_sum = self.masks[i].sum().item()
                            print(f"[AttnLoss] masks[{i}] sum={mask_sum:.1f} (应该>0)")
                    print(f"[AttnLoss] entropy_masks_ready={self._entropy_masks_ready}")
                    print(f"[AttnLoss] entropy_token_map={self._entropy_token_map}")
                if t == 1:
                    denoised_latent = denoised_tweedie
                return denoised_latent

            denoised_latent_temp = denoised_latent
            t_temp = next_t
            if self.config.jumping_steps > 0:  # 跳步采样
                """
                跳步采样过程：通过多次迭代优化生成质量
                
                跳步采样的目的是：
                1. 提高生成质量
                2. 减少概念间的冲突
                3. 优化多概念融合效果

                跳步采样原理：
                - 不是按顺序执行每个时间步
                - 而是跳过某些时间步，直接跳到更早的时间步
                - 通过这种方式加速生成过程，同时保持质量
                
                具体例子：
                - 如果jumping_steps=3，当前t_temp=25
                - 第1次跳步：t_temp = 25 - 150 = -125 (实际上会跳到更早的时间步)
                - 第2次跳步：t_temp = -125 - 150 = -275
                - 第3次跳步：t_temp = -275 - 150 = -425
                """
                for _ in range(self.config.jumping_steps):
                    # 计算当前临时时间步的alpha值
                    at_temp = self.alpha(t_temp)
                    
                     # 准备输入：使用当前去噪结果作为输入
                    latent_model_next = torch.cat([denoised_latent_temp] + [denoised_latent_temp])
                    # [2, 4, 128, 128]
                    # 两个分支：1个负向 + 1个正向

                    # 计算下一个时间步的噪声预测
                    noise_pred_next = self.unet(
                        latent_model_next, t_temp, encoder_hidden_states=text_embed[:2], added_cond_kwargs=unet_added_conditions_single
                    )['sample']# [2, 4, 128, 128]
                    
                    # 分离噪声预测
                    noise_pred_cond_next = noise_pred_next[1:2] # 正向噪声预测
                    noise_pred_uncond_next = noise_pred_next[:1] # 负向噪声预测
                    noise_pred_next = noise_pred_uncond_next + self.config.guidance_scale * (noise_pred_cond_next - noise_pred_uncond_next)
                    
                    # 跳步：直接跳到更早的时间步
                    t_temp = t_temp - 150  # 跳步幅度
                    at_temp_next = self.alpha(t_temp)  # 计算跳步后的alpha值
                    
                    # 计算跳步后的去噪结果
                    denoised_tweedie = (denoised_latent_temp - (1 - at_temp).sqrt() * noise_pred_next) / at_temp.sqrt()
                    
                    # 更新临时潜变量
                    denoised_latent_temp = at_temp_next.sqrt() * denoised_tweedie + (1 - at_temp_next).sqrt() * noise_pred_uncond_next
            
                del noise_pred_next, noise_pred_cond_next, noise_pred_uncond_next, denoised_latent_temp, latent_model_next
                gc.collect()
                torch.cuda.empty_cache()

                decoded_tweedie = self.decode_latent(denoised_tweedie)  # 解码为图片
            else:
                # 如果不使用跳步采样，直接解码
                decoded_tweedie = self.decode_latent(denoised_tweedie)  # 解码为图片

            
            os.makedirs(self.config.output_path, exist_ok=True)
            path_tweedie = os.path.join(self.config.output_path, 'tweedie.jpg')
            # 使用PIL直接保存，避免torchvision依赖
            img_array = decoded_tweedie[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            Image.fromarray(img_array).save(path_tweedie)
            # 保存中间图像
            # ✅ 方案B：如果已经提前生成了掩码（run_fusion中），跳过这里的生成
            if self.masks is None:
                if use_ext and len(boxes) > 0:
                    latent_h = self.config.resolution_h // 8
                    latent_w = self.config.resolution_w // 8
                    expected = self.concept_num - 1
                    if len(boxes) != expected:
                        raise ValueError(f'Number of boxes ({len(boxes)}) must equal num foreground concepts ({expected}). '
                                        f'Concepts={self.config.concepts}, seg_concepts={self.config.seg_concepts}')
                    fg_masks = build_masks_from_boxes(
                        boxes,
                        self.config.resolution_h, self.config.resolution_w,
                        latent_h, latent_w,
                        self.unet.device,
                    )
                    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
                    bg_mask[bg_mask < 0] = 0
                    self.masks = torch.cat([fg_masks, bg_mask])
                    self._prepare_cones_targets()
                    if self.enable_attention_entropy:
                        self._prepare_entropy_targets()
                        print(f"[AttnLoss] t=31时生成掩码（原有逻辑）")
                elif boxes_only:
                    raise ValueError('boxes_only=1 but no valid external_boxes provided.')
                else:
                    """
                    使用文本引导分割生成掩码
                    
                    这个过程会：
                    1. 调用外部分割脚本
                    2. 根据文本条件生成分割掩码
                    3. 将掩码转换为tensor格式

                    分割脚本的工作原理：
                    - 输入：中间生成的图像 + 文本条件
                    - 输出：每个概念的分割掩码
                    - 使用深度学习模型进行语义分割
                    """
                    # 构建分割命令
                    test_cmd = f'CUDA_VISIBLE_DEVICES={self.config.seg_gpu} python text_segment/run_expand.py --input_path={path_tweedie} --text_condition="{self.config.seg_concepts}" --output_path={self.config.output_path}'
                    
                    # 执行分割脚本
                    os.system(test_cmd)
                    
                    # 加载生成的掩码
                    mask_paths = []
                    concept_list = self.config.seg_concepts.split('+') if self.config.seg_concepts != '' else []
                    # concept_list = ["a panda", "a teddybear"]
                
                    for sp in concept_list:
                        mask_paths.append(os.path.join(self.config.output_path, sp + '.jpg'))
                    if len(mask_paths) > 0:
                        # 预处理掩码
                        fg_masks = torch.cat([
                            preprocess_mask(mask_path, self.config.resolution_h // 8, self.config.resolution_w // 8, self.unet.device)
                            for mask_path in mask_paths
                        ])  # [2, 1, 128, 128] - 2个概念的前景掩码
                        
                        # 构建背景掩码
                        bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)  # [1, 1, 128, 128]
                        bg_mask[bg_mask < 0] = 0  # 确保背景掩码非负
                        
                        # 合并所有掩码
                        self.masks = torch.cat([fg_masks, bg_mask])  # [3, 1, 128, 128]
                        self._prepare_cones_targets()
                        # 掩码结构：
                        # self.masks[0]: 熊猫概念掩码
                        # self.masks[1]: 泰迪熊概念掩码
                        # self.masks[2]: 背景掩码
            else:
                # 掩码已经提前生成，直接使用
                print(f"[AttnLoss] 使用提前生成的掩码（方案B）")
                self._prepare_cones_targets()
            
            # ✅ 掩码生成后立即准备注意力熵目标
            if self.enable_attention_entropy:
                self._prepare_entropy_targets()
                print(f"[AttnLoss] 掩码已准备完成")
                print(f"[AttnLoss] masks.shape={self.masks.shape if self.masks is not None else None}")
                if self.masks is not None:
                    for i in range(self.masks.shape[0]):
                        mask_sum = self.masks[i].sum().item()
                        print(f"[AttnLoss] masks[{i}] sum={mask_sum:.1f} (应该>0)")
                print(f"[AttnLoss] entropy_masks_ready={self._entropy_masks_ready}")
                print(f"[AttnLoss] entropy_token_map={self._entropy_token_map}")

        if t == 1:  # 如果是最后一步
            denoised_latent = denoised_tweedie  # 直接返回去噪结果

        return denoised_latent  # 返回当前步的去噪潜变量

    def init_fusion(self, t_cond):
        # 初始化融合采样的相关参数和状态
        self.t_cond = self.scheduler.timesteps[t_cond:] if t_cond >= 0 else []  # 记录融合阶段的时间步（从 t_cond 开始到最后）
        self.t_cond_prev = self.scheduler.timesteps[t_cond-1]  # 融合阶段前一个时间步
        self.t_cond_cur = self.scheduler.timesteps[t_cond]      # 当前融合阶段的起始时间步
        self.start_t = self.scheduler.timesteps[0]              # 采样的起始时间步
        self._t_cond_timestep_set = set(
            int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in self.t_cond
        )
        if self.cones_guidance_steps is not None and self.cones_guidance_steps > 0:
            active = self.t_cond[: min(len(self.t_cond), self.cones_guidance_steps)]
            self._cones_guidance_timestep_set = set(
                int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in active
            )
        else:
            self._cones_guidance_timestep_set = None
        os.makedirs(self.config.output_path_all, exist_ok=True) # 创建输出目录（如果不存在则创建）
        if self.use_sts_kv_hook and self._sts_have_unet_kv:
            register_attention_control_efficient_from_sts(self, self.t_cond, self.sts)
        else:
            register_attention_control_efficient(self, self.t_cond, self.concept_num) # 注册高效注意力控制（多概念融合相关）
            for i in range(self.concept_num):
                model_name = f'unet_{i}'
                if hasattr(self, model_name):
                    delattr(self, model_name)  # 删除每个概念对应的 UNet 属性，释放显存
            
    def run_fusion(self):
        '''
        该方法用于启动融合采样过程，
        它首先根据配置参数计算出融合阶段的时间步，
        然后调用 init_fusion 方法初始化相关参数和状态，
        最后调用 sample_loop 方法开始采样。
        '''
        t_cond = int(self.config.n_timesteps * self.config.t_cond)  # 根据配置计算融合阶段的起始步数（如 0.4*50=20）
        self.init_fusion(t_cond=t_cond)  # 初始化融合采样相关参数和状态
        
        # ✅ 方案B：如果提供了外部boxes，在采样开始前就生成掩码
        use_ext = getattr(self.config, 'use_external_boxes', 0) == 1
        boxes = parse_external_boxes(getattr(self.config, 'external_boxes', ''))
        if use_ext and len(boxes) > 0:
            latent_h = self.config.resolution_h // 8
            latent_w = self.config.resolution_w // 8
            expected = self.concept_num - 1
            if len(boxes) != expected:
                raise ValueError(f'Number of boxes ({len(boxes)}) must equal num foreground concepts ({expected})')
            
            fg_masks = build_masks_from_boxes(
                boxes,
                self.config.resolution_h, self.config.resolution_w,
                latent_h, latent_w,
                self.unet.device,
            )
            bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
            bg_mask[bg_mask < 0] = 0
            self.masks = torch.cat([fg_masks, bg_mask])
            self._prepare_cones_targets()
            
            # 准备注意力熵目标
            if self.enable_attention_entropy:
                self._prepare_entropy_targets()
                print(f"[AttnLoss] 掩码提前准备完成（方案B：早期约束）")
                print(f"[AttnLoss] masks.shape={self.masks.shape}")
                for i in range(self.masks.shape[0]):
                    mask_sum = self.masks[i].sum().item()
                    print(f"[AttnLoss] masks[{i}] sum={mask_sum:.1f}")
        
        # 生成初始噪声（标准正态分布），形状为 [1, 4, H/8, W/8]，并乘以初始噪声系数
        normal = torch.randn(1, 4, self.config.resolution_h // 8, self.config.resolution_w // 8).to(self.unet.device) * self.scheduler.init_noise_sigma
        _ = self.sample_loop(normal)  # 启动采样主循环，生成最终图像

    @torch.no_grad()
    def sample_loop(self, x):   
        '''
        该方法实现了扩散模型的采样过程，
        它首先将初始噪声 x 输入到去噪模型中，
        然后根据时间步 t 逐步进行去噪，
        最终得到生成图像。
        '''
        with torch.autocast(device_type='cuda', dtype=torch.float16):  # 自动混合精度，节省显存
            last_finite_x = x.detach().clone()
            last_healthy_x = x.detach().clone()
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):  # 遍历所有采样步
                x = self.denoise_step(x, t)  # 对当前潜变量进行一步去噪
                if not torch.isfinite(x).all():
                    t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                    print(f"[WARN] non-finite latent at step={i}, t={t_int}; fallback to last finite latent")
                    x = last_finite_x.clone()
                    continue
                last_finite_x = x.detach().clone()
                try:
                    # Track the latest non-collapsed latent to avoid flat-gray final decode.
                    latent_std = float(x.float().std().item())
                    if latent_std > 1e-6:
                        last_healthy_x = x.detach().clone()
                except Exception:
                    pass

            latent_for_decode = x
            if not torch.isfinite(latent_for_decode).all():
                print("[WARN] final latent is non-finite; fallback to last finite latent")
                latent_for_decode = last_finite_x.clone()
            try:
                final_std = float(latent_for_decode.float().std().item())
                if final_std < 1e-6:
                    print("[WARN] final latent nearly constant; fallback to last healthy latent")
                    latent_for_decode = last_healthy_x.clone()
            except Exception:
                pass

            # 关闭 autocast 做最终解码，避免半精度解码出现发灰/塌缩。
            decode_ctx = torch.autocast(device_type='cuda', enabled=False) if torch.cuda.is_available() else nullcontext()
            with decode_ctx:
                image, decoded_latent = self._decode_latent_to_pil(
                    latent_for_decode,
                    fallback_latent=last_healthy_x,
                )

            os.makedirs(self.config.output_path_all, exist_ok=True)  # 创建输出目录
            resample_steps = int(getattr(self.config, "resampling_steps", 0))
            t_cond_ratio = float(getattr(self.config, "t_cond", 0.0))

            # Always use Windows-safe filenames so results can be copied to Windows easily.
            modifier = getattr(self.config, "modifier_token", "").replace("+", "_")
            guidance = float(getattr(self.config, "guidance_scale", 7.5))
            seed = int(getattr(self.config, "seed", 0))
            safe_modifier = re.sub(r'[<>:"/\\\\|?*\\x00-\\x1F]', "", modifier)
            safe_modifier = safe_modifier.strip().strip(".").replace(" ", "_")
            if not safe_modifier:
                safe_modifier = "output"

            # Prefix with the runner script name so different inference entrypoints won't overwrite each other
            # when writing to the same output directory.
            runner = os.path.splitext(os.path.basename(sys.argv[0]))[0]
            runner = runner.strip().strip(".").replace(" ", "_")
            prefix = f"{runner}__" if runner else ""

            filename = f"{prefix}{safe_modifier}_g{guidance:.2f}_t{t_cond_ratio:.2f}_res{resample_steps}_seed{seed}.png"

            out_dir = self.config.output_path_all
            out_path = os.path.join(out_dir, filename)
            try:
                # 保存图片，包含关键参数
                image[0].save(out_path)
            except OSError as e:
                # Extremely defensive fallback (e.g., weird filesystem).
                fallback_path = os.path.join(out_dir, f"output_g{guidance:.2f}_t{t_cond_ratio:.2f}_res{resample_steps}_seed{seed}.png")
                image[0].save(fallback_path)
                print(f"[WARN] Failed to save '{out_path}' ({e}); saved as '{fallback_path}' instead.")
                
        return decoded_latent  # 返回解码后的 latent（图像张量）

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--seed', type=int, default=182)  # 随机种子
    parser.add_argument('--device', type=str, default='cuda:0')  # 计算设备
    parser.add_argument('--output_path', type=str)  # 中间结果输出目录
    parser.add_argument('--output_path_all', type=str)  # 最终图片输出目录
    parser.add_argument('--negative_prompt', type=str, default='blurry, ugly, black, low res, unrealistic, blurry face')
    # 负面提示词，默认用于去除模糊、低质量等
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.4','1.5', '2.0','2.1','xl'],
                        help="stable diffusion version")  # Stable Diffusion 版本
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='',
                        help='可选：显式指定基础模型路径（本地目录或HF id）')
    parser.add_argument('--vae_model_name_or_path', type=str, default='',
                        help='可选：显式指定VAE路径（本地目录或HF id）；不传则使用base pipeline自带VAE')
    parser.add_argument('--t_cond', type=float, default=0.4)  # 融合阶段的比例（如0.4表示前40%步数用于融合）
    parser.add_argument('--guidance_scale', type=float, default=9.0)  # classifier-free guidance 强度
    parser.add_argument('--n_timesteps', type=int, default=50)  # 采样步数
    parser.add_argument('--prompt', type=str, default='')  # 多概念融合的完整 prompt
    parser.add_argument('--concept_weights', type=str, default='')  # 概念权重列表，逗号分隔
    parser.add_argument('--prompt_clean', type=str, default='')
    parser.add_argument('--prompt_clean_resample', type=str, default='', help='Clean single-concept prompts used only in resampling (decoupled from MSE anchors)')
    parser.add_argument('--prompt_orig_clean', type=str, default='')  # 多概念无属性 prompt
    parser.add_argument('--prompt_orig', type=str, default='')  # 原始 prompt
    parser.add_argument('--seg_concepts', type=str, default='')  # 分割掩码的概念列表
    parser.add_argument('--personal_checkpoint', type=str, default='')  # 自定义权重路径
    parser.add_argument('--concepts', type=str)  # 概念词列表
    parser.add_argument('--modifier_token', type=str)  # 修饰 token 列表
    parser.add_argument('--resampling_steps',  type=int, default=10)  # 重采样步数
    parser.add_argument('--jumping_steps',  type=int, default=5)  # 跳步采样步数
    parser.add_argument('--seg_gpu',  type=int, default=1)  # 分割脚本使用的 GPU
    parser.add_argument('--disable_xformers', type=int, default=0, help='1=禁用xformers（稳定优先）')
    parser.add_argument('--use_external_boxes', type=int, default=0)  # 是否使用外部矩形框生成掩码
    parser.add_argument('--external_boxes', type=str, default='')  # 外部矩形框列表：x1,y1,x2,y2+x1,y1,x2,y2
    parser.add_argument('--boxes_only', type=int, default=0)  # 1=只用外部框，缺框就报错
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )  # SDXL 裁剪左上角高度
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )  # SDXL 裁剪左上角宽度
    parser.add_argument(
        "--resolution_h",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )  # 输入图片高度
    parser.add_argument(
        "--resolution_w",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )  # 输入图片宽度
    parser.add_argument('--binding_json', type=str, default='', help='JSON string or file path for manual token bindings')
    parser.add_argument('--sem_binding_steps', type=int, default=0, help='Semantic binding optimization steps at content fusion start')
    parser.add_argument('--sem_binding_lr', type=float, default=1e-3, help='Semantic binding learning rate')
    parser.add_argument('--sem_binding_use_clean', type=int, default=0, help='Use PROMPT_CLEAN single-concept anchors for semantic binding (1=yes, 0=no)')
    parser.add_argument('--debug_binding', type=int, default=0, help='Print tokenization and binding debug info')
    parser.add_argument('--sem_binding_max_steps', type=int, default=1, help='语义绑定在融合阶段执行的最大时间步数')
    parser.add_argument('--enable_attention_entropy', type=int, default=0, help='是否开启注意力熵约束')
    parser.add_argument('--attn_entropy_weight', type=float, default=0.0, help='注意力熵损失的权重')
    parser.add_argument('--attn_entropy_outside_weight', type=float, default=1.0, help='注意力落在掩码外部时的惩罚系数')
    parser.add_argument('--attn_entropy_inside_weight', type=float, default=0.0, help='注意力落在掩码内部时的奖励系数')
    parser.add_argument('--attn_entropy_steps', type=int, default=1, help='每个时间步执行注意力熵优化的迭代次数')
    parser.add_argument('--attn_entropy_lr', type=float, default=5e-4, help='注意力熵优化的学习率')
    parser.add_argument('--attn_entropy_min_step', type=int, default=0, help='注意力熵开始生效的最小时间步')
    parser.add_argument('--attn_entropy_max_step', type=int, default=-1, help='注意力熵停止生效的最大时间步（-1 表示不限制）')
    parser.add_argument('--attn_entropy_layers', type=str, default='up', help='参与熵计算的 UNet 模块前缀（up/down/mid）')
    parser.add_argument('--attn_entropy_mask_downscale', type=int, default=1, help='注意力掩码下采样因子')
    parser.add_argument('--attn_entropy_low_mem', type=int, default=0, help='是否开启低显存模式（仅使用部分分支参与熵优化）')
    parser.add_argument('--attn_entropy_enable_checkpointing', type=int, default=0, help='是否在熵优化时开启 UNet 梯度检查点')
    parser.add_argument('--enable_cones_mask_attention', type=int, default=1, help='是否启用Cones风格的logits掩码注意力引导')
    parser.add_argument('--cones_guidance_weight', type=float, default=0.08, help='Cones布局引导强度')
    parser.add_argument('--cones_guidance_steps', type=int, default=-1, help='Cones布局引导生效步数（在融合阶段内，-1表示融合阶段全程）')
    parser.add_argument('--cones_positive_value', type=float, default=2.5, help='Cones掩码在目标区域的正偏置值')
    parser.add_argument('--cones_negative_value', type=float, default=-0.02, help='Cones掩码在非目标区域的负偏置值')
    parser.add_argument('--cones_use_sim_std', type=int, default=1, help='Cones偏置是否乘以当前注意力logits标准差')
    parser.add_argument('--cones_use_modifier_only', type=int, default=1, help='1=Cones仅对modifier token列注入（推荐个性化概念）')
    parser.add_argument('--cones_focus_fg_only', type=int, default=1, help='1=Cones仅作用前景概念（不对背景概念注入）')
    parser.add_argument('--cones_use_plain_prompt_before_fusion', type=int, default=1, help='1=在多概念融合前对原始multi prompt的cat/dog位置做Cones框注入')
    parser.add_argument('--cones_debug_tokens', type=int, default=0, help='1=启用Cones token调试逻辑')
    parser.add_argument('--cones_debug_token_maps', type=int, default=0, help='1=打印[ConesMask2][Debug] token_maps明细（很冗长，默认0关闭）')
    parser.add_argument('--use_sts_kv_hook', type=int, default=1, help='1=使用sts里的unet K/V直接挂载attention（低显存），0=回退旧版加载unet_i')
    parser.add_argument('--fusion_low_mem', type=int, default=0, help='融合阶段使用串行概念推理，降低一次性batch显存占用')
    opt = parser.parse_args()  # 解析命令行参数，保存到 opt

    seed_everything(opt.seed)  # 设置随机种子，保证实验可复现
    tweedie = MultiCompose(opt)  # 创建 MultiCompose 融合采样对象
    tweedie.run_fusion()       # 启动融合采样流程
