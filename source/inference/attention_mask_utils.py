"""
MultiCompose系统的核心工具函数库

这个模块包含了MultiCompose多概念融合系统所需的核心工具函数，
主要用于注意力机制控制、时间步注册和随机种子设置等功能。

主要功能：
1. seed_everything: 设置随机种子，确保实验可复现性
2. register_time: 为UNet注意力模块注册当前时间步
3. register_attention_control_efficient: 实现多概念融合的注意力控制机制

技术原理：
- MultiCompose通过动态修改UNet的交叉注意力机制实现多概念融合
- 在融合阶段使用多个概念特定的UNet进行并行处理
- 在融合后阶段使用标准UNet进行最终生成
"""

import torch
import os
import random
import numpy as np
import math
import copy
from einops import rearrange  # 用于张量重排列操作
import torch.nn.functional as F  # PyTorch函数库
import torchvision  # 计算机视觉工具库

def seed_everything(seed):
    """
    设置所有随机数生成器的种子，确保实验可复现性
    
    这是深度学习实验中的重要步骤，确保每次运行都能得到相同的结果，
    便于调试和结果对比。
    
    Args:
        seed (int): 随机种子值
        
    注意：
        - 设置PyTorch CPU随机种子
        - 设置PyTorch CUDA随机种子
        - 设置Python标准库random模块种子
        - 设置NumPy随机种子
    """
    torch.manual_seed(seed)        # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)   # 设置PyTorch CUDA随机种子
    random.seed(seed)              # 设置Python random模块种子
    np.random.seed(seed)           # 设置NumPy随机种子

def register_time(model, t):
    """
    为UNet的所有交叉注意力模块注册当前时间步
    
    这个函数是MultiCompose系统的关键组件，用于在扩散采样过程中
    将当前时间步信息传递给UNet的注意力机制，使得注意力计算能够
    根据时间步进行相应的调整。
    
    Args:
        model: MultiCompose模型实例，包含UNet结构
        t: 当前扩散采样时间步（timestep）
        
    技术细节：
        - UNet结构包含down_blocks、mid_block、up_blocks三部分
        - 每个block包含多个attention层和transformer_blocks
        - 只对attn2（交叉注意力）层进行时间步注册
        - attn2层负责处理文本条件信息
    """
    
    # SDXL UNet的结构配置
    # down_blocks: 下采样块，分辨率逐渐降低
    down_res_dict = {1: [0, 1], 2: [0, 1]}  # 分辨率1和2对应的attention块索引
    down_transformers_len = {1:[2,2],2:[10,10]}  # 每个attention块的transformer数量
    
    # up_blocks: 上采样块，分辨率逐渐恢复
    up_transformers_len = {0:[10,10,10],1:[2,2,2]}  # 上采样块中每个attention的transformer数量
    up_res_dict = {0:[ 1, 2],1: [0, 1, 2]}  # 上采样块的attention索引
    up_res_dict_cross = {0:[ 0, 1, 2],1: [0, 1, 2]}  # 交叉注意力的索引
    
    # mid_block: 中间块，处理最低分辨率的特征
    mid_transformers_len = 10  # 中间块的transformer数量
    
    # 为上采样块的交叉注意力模块注册时间步
    for res in up_res_dict_cross:  # 遍历分辨率级别
        for block in up_res_dict_cross[res]:  # 遍历attention块
            for idx in range(up_transformers_len[res][block]):  # 遍历transformer块
                # 获取交叉注意力模块
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[idx].attn2
                # 注册当前时间步
                setattr(module, 't', t)
            
    # 为下采样块的交叉注意力模块注册时间步
    for res in down_res_dict:  # 遍历分辨率级别
        for block in down_res_dict[res]:  # 遍历attention块
            for idx in range(down_transformers_len[res][block]):  # 遍历transformer块
                # 获取交叉注意力模块
                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[idx].attn2
                # 注册当前时间步
                setattr(module, 't', t)

    # 为中间块的交叉注意力模块注册时间步
    for idx in range(mid_transformers_len):  # 遍历中间块的所有transformer
        # 获取中间块的交叉注意力模块
        module = model.unet.mid_block.attentions[0].transformer_blocks[idx].attn2
        # 注册当前时间步
        setattr(module, 't', t)

def register_attention_control_efficient(model, t_cond, num_concepts):
    """
    注册高效的多概念注意力控制机制
    
    这是MultiCompose系统的核心创新，通过动态修改UNet的交叉注意力机制，
    实现在融合阶段使用多个概念特定的UNet进行并行处理，在融合后阶段
    使用标准UNet进行生成。
    
    Args:
        model: MultiCompose模型实例
        t_cond: 融合阶段的时间步列表
        num_concepts: 概念数量
    
    技术原理：
        1. 在融合阶段(t in t_cond)：使用多个概念UNet的K、V矩阵
        2. 在融合后阶段：使用标准UNet的K、V矩阵
        3. 通过动态替换forward函数实现注意力机制的切换
    
    工作流程：
        1. 为每个概念加载对应的UNet（unet_0, unet_1, unet_2...）
        2. 将概念UNet的K、V矩阵复制到主UNet的对应模块
        3. 替换注意力模块的forward函数，实现动态切换
        4. 在融合阶段使用多概念K、V矩阵，在标准阶段使用单一K、V矩阵
    """
    
    def sa_forward(self):
        """
        动态生成的交叉注意力前向传播函数
        
        这个函数会根据当前时间步和条件动态选择使用多概念融合
        还是标准注意力机制。
        
        Returns:
            forward: 修改后的前向传播函数
        """
        # 获取输出投影层
        # 调试信息：如需查看绑定状态，取消下面注释
        # print("merge hook called", getattr(model, "_binding_entries", None))
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            """
            交叉注意力的前向传播计算
            
            Args:
                x: 输入特征 [batch_size, sequence_length, dim]
                encoder_hidden_states: 文本编码状态 [batch_size, seq_len, hidden_dim]
                attention_mask: 注意力掩码（可选）
            
            Returns:
                out: 注意力输出 [batch_size, sequence_length, dim]
            """
            place_in_unet = getattr(self, "place_in_unet", "unknown")
            batch_size, sequence_length, dim = x.shape
            h = self.heads  # 注意力头数
            
            # 判断是否为交叉注意力（有文本条件）
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            
            if is_cross:
             binding_entries = getattr(model, "_binding_entries", [])
             clean_cache = getattr(model, "_binding_clean_cache", {})
             if binding_entries:
                 delta = torch.zeros_like(encoder_hidden_states)

                 for entry in binding_entries:
                     prompt_text = entry.get("prompt_text")
                     prompt_idx = model._binding_prompt_to_index.get(prompt_text)
                     if prompt_idx is None or prompt_idx >= encoder_hidden_states.shape[0]:
                         continue

                     for pair in entry.get("pairs", []):
                         subj = pair.get("subject", [])
                         if not subj:
                             continue
                         subj_tensor = torch.tensor(subj, device=encoder_hidden_states.device)

                         # 主体向量的总和（先取主体 token，再加上属性/修饰/额外 token）
                         merged = encoder_hidden_states[prompt_idx, subj_tensor, :].sum(dim=0)
                         attr_idx = []
                         for key in ("attributes", "modifiers", "extras", "additional"):
                             for group in pair.get(key, []):
                                 attr_idx.extend(group)
                         if attr_idx:
                             attr_tensor = torch.tensor(attr_idx, device=encoder_hidden_states.device)
                             merged = merged + encoder_hidden_states[prompt_idx, attr_tensor, :].sum(dim=0)
                             delta[prompt_idx, attr_tensor, :] -= encoder_hidden_states[prompt_idx, attr_tensor, :]

                         delta[prompt_idx, subj_tensor[0], :] += merged - encoder_hidden_states[prompt_idx, subj_tensor[0], :]
                         if subj_tensor.numel() > 1:
                             delta[prompt_idx, subj_tensor[1:], :] -= encoder_hidden_states[prompt_idx, subj_tensor[1:], :]

                     clean_prompt = entry.get("eot_clean_prompt")
                     if clean_prompt and clean_prompt in clean_cache:
                        entry_clean = clean_cache.get(clean_prompt)
                        if isinstance(entry_clean, tuple):
                            clean_embed, _ = entry_clean
                        else:
                            clean_embed = entry_clean
                        clean_embed = clean_embed.to(encoder_hidden_states.device, encoder_hidden_states.dtype)
                        eos_id = getattr(model, "eos_token_id", None)

                        if eos_id is not None and eos_id < encoder_hidden_states.shape[1]:
                             delta[prompt_idx, eos_id, :] += clean_embed[eos_id] - encoder_hidden_states[prompt_idx, eos_id, :]

                 encoder_hidden_states = encoder_hidden_states + delta


            # 多概念融合阶段的条件判断
            # - 标准融合：encoder_hidden_states = [uncond] + [concept_0..concept_{N-1}]，shape = 1 + num_concepts
            # - 低显存融合：一次只跑一个概念，encoder_hidden_states = [uncond, concept_k]，shape = 2
            active_concept_idx = getattr(model, "_active_concept_idx", None)
            if (
                is_cross
                and (self.t in self.t_cond)
                and encoder_hidden_states.shape[0] == (1 + self.num_concepts)
            ):
                """
                融合阶段：使用多概念UNet的K、V矩阵
                条件：
                1. 是交叉注意力
                2. 当前时间步在融合阶段内
                3. 文本嵌入包含 1+num_concepts 个部分（1个负向 + num_concepts个概念）
                """

                # 计算查询矩阵Q（使用当前UNet）
                q = self.to_q(x)

                # 计算键矩阵K（多概念融合）
                # 使用负向文本的K矩阵
                k = self.to_k(encoder_hidden_states[0].unsqueeze(0))
                ks = [k]
                
                # 为每个概念添加对应的K矩阵
                for i in range(self.num_concepts):
                    # 从概念UNet_i获取K矩阵
                    ks.append(getattr(self, f"to_k_{i}")(encoder_hidden_states[i+1].unsqueeze(0)))

                # 拼接所有K矩阵
                k = torch.cat(ks, dim=0)

                # 调整维度用于多头注意力计算
                q = self.head_to_batch_dim(q)
                num_batch = k.shape[0]
                k = self.head_to_batch_dim(k)
                
                # 计算值矩阵V（多概念融合）
                # 使用负向文本的V矩阵
                v = self.to_v(encoder_hidden_states[0].unsqueeze(0))
                vs = [v]
                
                # 为每个概念添加对应的V矩阵
                for i in range(num_concepts):
                    # 从概念UNet_i获取V矩阵
                    vs.append(getattr(self, f"to_v_{i}")(encoder_hidden_states[i+1].unsqueeze(0)))

                # 拼接所有V矩阵
                v = torch.cat(vs, dim=0)
            elif (
                is_cross
                and (self.t in self.t_cond)
                and encoder_hidden_states.shape[0] == 2
                and active_concept_idx is not None
            ):
                """
                低显存融合阶段：一次只跑一个概念分支（uncond + concept_k）
                这里需要显式指定当前 concept_k 对应的 to_k_k / to_v_k。
                """

                q = self.to_q(x)

                # uncond 使用当前UNet的K/V，concept_k 使用概念UNet_k的K/V
                k_uncond = self.to_k(encoder_hidden_states[0].unsqueeze(0))
                k_concept = getattr(self, f"to_k_{int(active_concept_idx)}")(
                    encoder_hidden_states[1].unsqueeze(0)
                )
                k = torch.cat([k_uncond, k_concept], dim=0)

                q = self.head_to_batch_dim(q)
                num_batch = k.shape[0]
                k = self.head_to_batch_dim(k)

                v_uncond = self.to_v(encoder_hidden_states[0].unsqueeze(0))
                v_concept = getattr(self, f"to_v_{int(active_concept_idx)}")(
                    encoder_hidden_states[1].unsqueeze(0)
                )
                v = torch.cat([v_uncond, v_concept], dim=0)
            else:
                """
                标准阶段：使用单一UNet的K、V矩阵
                条件：非交叉注意力或在融合后阶段
                """
                # 标准注意力计算
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                num_batch = k.shape[0]
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.to_v(encoder_hidden_states)
                
            # 调整V矩阵维度
            v = self.head_to_batch_dim(v)

            # 计算注意力分数：Q与K的点积
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            # 应用注意力掩码（如果提供）
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # Cones-style layout guidance: inject mask bias in logits before softmax.
            if (
                is_cross
                and getattr(model, "enable_cones_mask_attention", False)
                and hasattr(model, "apply_cones_attention_bias")
            ):
                try:
                    sim = model.apply_cones_attention_bias(
                        sim=sim,
                        heads=h,
                        num_batch=num_batch,
                        key_len=sim.shape[2],
                        timestep=getattr(self, "t", None),
                        active_concept_idx=active_concept_idx,
                    )
                except Exception as e:
                    if not hasattr(model, "_cones_hook_error_logged"):
                        model._cones_hook_error_logged = True
                        print(f"[ConesMask] WARNING: failed to apply attention bias: {e}")

            # 计算注意力权重（softmax）
            attn = sim.softmax(dim=-1)

            if getattr(model, 'save_attention_maps', False) and is_cross and hasattr(model, '_record_attention'):
                timestep = getattr(self, 't', None)
                if timestep is not None:
                    try:
                        model._record_attention(attn.detach(), h, attn.shape[1], attn.shape[2], int(timestep), place_in_unet)
                    except Exception:
                        pass

            # Ӧ��ע����Ȩ�ص�ֵ����V
            out = torch.einsum("b i j, b j d -> b i d", attn, v)

            out = self.batch_to_head_dim(out)
            out = to_out(out)

            return out

        return forward

    # UNet结构配置（与register_time函数相同）
    down_res_dict = {1: [0, 1], 2: [0, 1]}
    down_transformers_len = {1:[2,2],2:[10,10]}
    up_transformers_len = {0:[10,10,10],1:[2,2,2]}
    up_res_dict_cross = {0:[ 0, 1, 2],1: [0, 1, 2]}
    mid_transformers_len = 10
    
    # 为上采样块注册多概念注意力控制
    for res in up_res_dict_cross:  # 遍历分辨率级别
        for block in up_res_dict_cross[res]:  # 遍历attention块
            for idx in range(up_transformers_len[res][block]):  # 遍历transformer块
                # 获取主UNet的交叉注意力模块
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[idx].attn2

                # 为每个概念添加对应的K、V矩阵
                for i in range(num_concepts):
                    # 获取概念UNet_i对应的注意力模块
                    module_temp = getattr(model, f"unet_{i}").up_blocks[res].attentions[block].transformer_blocks[idx].attn2
                    # 将概念UNet的V矩阵添加到主UNet
                    setattr(module, f'to_v_{i}', module_temp.to_v)
                    # 将概念UNet的K矩阵添加到主UNet
                    setattr(module, f'to_k_{i}', module_temp.to_k)

                # 替换前向传播函数
                module.forward = sa_forward(module)
                setattr(module, "place_in_unet", f"up_res{res}_block{block}_attn{idx}")

                # 设置概念数量和时间步条件
                setattr(module, 'num_concepts', num_concepts)
                setattr(module, 't_cond', t_cond)
    
    # 为下采样块注册多概念注意力控制
    for res in down_res_dict:  # 遍历分辨率级别
        for block in down_res_dict[res]:  # 遍历attention块
            for idx in range(down_transformers_len[res][block]):  # 遍历transformer块

                # 获取主UNet的交叉注意力模块
                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[idx].attn2
                
                # 为每个概念添加对应的K、V矩阵
                for i in range(num_concepts):
                    # 获取概念UNet_i对应的注意力模块
                    module_temp = getattr(model, f"unet_{i}").down_blocks[res].attentions[block].transformer_blocks[idx].attn2
                    # 将概念UNet的V矩阵添加到主UNet
                    setattr(module, f'to_v_{i}', module_temp.to_v)
                    # 将概念UNet的K矩阵添加到主UNet
                    setattr(module, f'to_k_{i}', module_temp.to_k)

                # 替换前向传播函数
                module.forward = sa_forward(module)
                setattr(module, "place_in_unet", f"down_res{res}_block{block}_attn{idx}")

                # 设置时间步条件和概念数量
                setattr(module, 't_cond', t_cond)
                setattr(module, 'num_concepts', num_concepts)
    
    # 为中间块注册多概念注意力控制
    for idx in range(mid_transformers_len):  # 遍历中间块的所有transformer
        # 获取主UNet的中间块交叉注意力模块
        module = model.unet.mid_block.attentions[0].transformer_blocks[idx].attn2
        
        # 为每个概念添加对应的K、V矩阵
        for i in range(num_concepts):
            # 获取概念UNet_i对应的中间块注意力模块
            module_temp = getattr(model, f"unet_{i}").mid_block.attentions[0].transformer_blocks[idx].attn2
            # 将概念UNet的V矩阵添加到主UNet
            setattr(module, f'to_v_{i}', module_temp.to_v)
            # 将概念UNet的K矩阵添加到主UNet
            setattr(module, f'to_k_{i}', module_temp.to_k)

        # 替换前向传播函数
        module.forward = sa_forward(module)
        setattr(module, "place_in_unet", f"mid_block_attn{idx}")

        # 设置时间步条件和概念数量
        setattr(module, 't_cond', t_cond)
        setattr(module, 'num_concepts', num_concepts)


def register_attention_control_efficient_from_sts(model, t_cond, sts):
    """
    Register multi-concept attention control using per-concept UNet K/V projections stored in `sts`.

    This is a memory-friendly alternative to `register_attention_control_efficient()`:
    - It does NOT require creating extra UNet instances (unet_0, unet_1, ...).
    - It attaches `to_k_i` / `to_v_i` Linear layers to each cross-attn module from `sts[i]['unet']`
      (keys like `...attn2.to_k.weight` / `...attn2.to_v.weight`).

    Args:
        model: MultiCompose instance holding `model.unet` (SDXL UNet2DConditionModel).
        t_cond: timesteps tensor/list that defines the fusion stage.
        sts: list of checkpoint dicts, one per concept. Each must contain 'unet' with attn2 to_k/to_v weights.
    """

    num_concepts = len(sts)
    if num_concepts <= 0:
        raise ValueError("sts must be a non-empty list")

    # Validate format early to fail fast with a clear error.
    for i, st in enumerate(sts):
        if not isinstance(st, dict) or "unet" not in st:
            raise ValueError(f"sts[{i}] must be a dict containing key 'unet'")
        if not isinstance(st["unet"], dict):
            raise ValueError(f"sts[{i}]['unet'] must be a dict of parameter_name -> tensor")

    def sa_forward(self):
        # Reuse the same cross-attn logic as `register_attention_control_efficient`, but rely on
        # already-attached `to_k_i` / `to_v_i` modules instead of reading from unet_{i}.
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            place_in_unet = getattr(self, "place_in_unet", "unknown")
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x

            if is_cross:
                binding_entries = getattr(model, "_binding_entries", [])
                clean_cache = getattr(model, "_binding_clean_cache", {})
                if binding_entries:
                    delta = torch.zeros_like(encoder_hidden_states)

                    for entry in binding_entries:
                        prompt_text = entry.get("prompt_text")
                        prompt_idx = model._binding_prompt_to_index.get(prompt_text)
                        if prompt_idx is None or prompt_idx >= encoder_hidden_states.shape[0]:
                            continue

                        for pair in entry.get("pairs", []):
                            subj = pair.get("subject", [])
                            if not subj:
                                continue
                            subj_tensor = torch.tensor(subj, device=encoder_hidden_states.device)

                            merged = encoder_hidden_states[prompt_idx, subj_tensor, :].sum(dim=0)
                            attr_idx = []
                            for key in ("attributes", "modifiers", "extras", "additional"):
                                for group in pair.get(key, []):
                                    attr_idx.extend(group)
                            if attr_idx:
                                attr_tensor = torch.tensor(attr_idx, device=encoder_hidden_states.device)
                                merged = merged + encoder_hidden_states[prompt_idx, attr_tensor, :].sum(dim=0)
                                delta[prompt_idx, attr_tensor, :] -= encoder_hidden_states[prompt_idx, attr_tensor, :]

                            delta[prompt_idx, subj_tensor[0], :] += merged - encoder_hidden_states[prompt_idx, subj_tensor[0], :]
                            if subj_tensor.numel() > 1:
                                delta[prompt_idx, subj_tensor[1:], :] -= encoder_hidden_states[prompt_idx, subj_tensor[1:], :]

                        clean_prompt = entry.get("eot_clean_prompt")
                        if clean_prompt and clean_prompt in clean_cache:
                            entry_clean = clean_cache.get(clean_prompt)
                            if isinstance(entry_clean, tuple):
                                clean_embed, _ = entry_clean
                            else:
                                clean_embed = entry_clean
                            clean_embed = clean_embed.to(encoder_hidden_states.device, encoder_hidden_states.dtype)
                            eos_id = getattr(model, "eos_token_id", None)

                            if eos_id is not None and eos_id < encoder_hidden_states.shape[1]:
                                delta[prompt_idx, eos_id, :] += clean_embed[eos_id] - encoder_hidden_states[prompt_idx, eos_id, :]

                    encoder_hidden_states = encoder_hidden_states + delta

            active_concept_idx = getattr(model, "_active_concept_idx", None)
            if (
                is_cross
                and (self.t in self.t_cond)
                and encoder_hidden_states.shape[0] == (1 + self.num_concepts)
            ):
                q = self.to_q(x)

                k = self.to_k(encoder_hidden_states[0].unsqueeze(0))
                ks = [k]
                for i in range(self.num_concepts):
                    ks.append(getattr(self, f"to_k_{i}")(encoder_hidden_states[i + 1].unsqueeze(0)))
                k = torch.cat(ks, dim=0)

                q = self.head_to_batch_dim(q)
                num_batch = k.shape[0]
                k = self.head_to_batch_dim(k)

                v = self.to_v(encoder_hidden_states[0].unsqueeze(0))
                vs = [v]
                for i in range(self.num_concepts):
                    vs.append(getattr(self, f"to_v_{i}")(encoder_hidden_states[i + 1].unsqueeze(0)))
                v = torch.cat(vs, dim=0)
            elif (
                is_cross
                and (self.t in self.t_cond)
                and encoder_hidden_states.shape[0] == 2
                and active_concept_idx is not None
            ):
                q = self.to_q(x)

                k_uncond = self.to_k(encoder_hidden_states[0].unsqueeze(0))
                k_concept = getattr(self, f"to_k_{int(active_concept_idx)}")(
                    encoder_hidden_states[1].unsqueeze(0)
                )
                k = torch.cat([k_uncond, k_concept], dim=0)

                q = self.head_to_batch_dim(q)
                num_batch = k.shape[0]
                k = self.head_to_batch_dim(k)

                v_uncond = self.to_v(encoder_hidden_states[0].unsqueeze(0))
                v_concept = getattr(self, f"to_v_{int(active_concept_idx)}")(
                    encoder_hidden_states[1].unsqueeze(0)
                )
                v = torch.cat([v_uncond, v_concept], dim=0)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                num_batch = k.shape[0]
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.to_v(encoder_hidden_states)

            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                attention_mask = attention_mask.repeat(num_batch, 1, 1)
                sim.masked_fill_(~attention_mask.bool(), max_neg_value)

            # Cones-style layout guidance: inject mask bias in logits before softmax.
            if (
                is_cross
                and getattr(model, "enable_cones_mask_attention", False)
                and hasattr(model, "apply_cones_attention_bias")
            ):
                try:
                    sim = model.apply_cones_attention_bias(
                        sim=sim,
                        heads=h,
                        num_batch=num_batch,
                        key_len=sim.shape[2],
                        timestep=getattr(self, "t", None),
                        active_concept_idx=active_concept_idx,
                    )
                except Exception as e:
                    if not hasattr(model, "_cones_hook_error_logged"):
                        model._cones_hook_error_logged = True
                        print(f"[ConesMask] WARNING: failed to apply attention bias: {e}")

            attn = sim.softmax(dim=-1)

            if getattr(model, "store_attention_for_entropy", False) and is_cross and hasattr(model, "_store_attention_for_entropy"):
                timestep = getattr(self, "t", None)
                if timestep is not None:
                    try:
                        attn_to_store = attn.detach()
                        model._store_attention_for_entropy(attn_to_store, h, attn.shape[1], attn.shape[2], place_in_unet)
                    except Exception as e:
                        if not hasattr(model, "_hook_error_logged"):
                            model._hook_error_logged = True
                            print(f"[Hook] ERROR: {e}")

            if getattr(model, "save_attention_maps", False) and is_cross and hasattr(model, "_record_attention"):
                timestep = getattr(self, "t", None)
                if timestep is not None:
                    try:
                        model._record_attention(attn.detach(), h, attn.shape[1], attn.shape[2], int(timestep), place_in_unet)
                    except Exception:
                        pass

            out = torch.einsum("b i j, b j d -> b i d", attn, v)

            out = self.batch_to_head_dim(out)
            out = to_out(out)

            return out

        return forward

    # UNet structure configuration (SDXL). Must match the checkpoints' key naming.
    down_res_dict = {1: [0, 1], 2: [0, 1]}
    down_transformers_len = {1: [2, 2], 2: [10, 10]}
    up_transformers_len = {0: [10, 10, 10], 1: [2, 2, 2]}
    up_res_dict_cross = {0: [0, 1, 2], 1: [0, 1, 2]}
    mid_transformers_len = 10

    def _attach_kv_from_sts(attn_module, key_prefix):
        # Create and attach per-concept K/V projection layers for this cross-attn module.
        # Fallback to base weights if a key is missing (shouldn't happen for properly trained checkpoints).
        for i in range(num_concepts):
            unet_weights = sts[i]["unet"]
            k_key = f"{key_prefix}.to_k.weight"
            v_key = f"{key_prefix}.to_v.weight"
            k_w = unet_weights.get(k_key)
            v_w = unet_weights.get(v_key)

            to_k_i = copy.deepcopy(attn_module.to_k)
            to_v_i = copy.deepcopy(attn_module.to_v)

            if k_w is not None:
                to_k_i.weight.data.copy_(k_w.to(device=to_k_i.weight.device, dtype=to_k_i.weight.dtype))
            if v_w is not None:
                to_v_i.weight.data.copy_(v_w.to(device=to_v_i.weight.device, dtype=to_v_i.weight.dtype))

            setattr(attn_module, f"to_k_{i}", to_k_i)
            setattr(attn_module, f"to_v_{i}", to_v_i)

    # up_blocks cross-attn
    for res in up_res_dict_cross:
        for block in up_res_dict_cross[res]:
            for idx in range(up_transformers_len[res][block]):
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[idx].attn2
                key_prefix = f"up_blocks.{res}.attentions.{block}.transformer_blocks.{idx}.attn2"
                _attach_kv_from_sts(module, key_prefix)
                module.forward = sa_forward(module)
                setattr(module, "place_in_unet", f"up_res{res}_block{block}_attn{idx}")
                setattr(module, "num_concepts", num_concepts)
                setattr(module, "t_cond", t_cond)

    # down_blocks cross-attn
    for res in down_res_dict:
        for block in down_res_dict[res]:
            for idx in range(down_transformers_len[res][block]):
                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[idx].attn2
                key_prefix = f"down_blocks.{res}.attentions.{block}.transformer_blocks.{idx}.attn2"
                _attach_kv_from_sts(module, key_prefix)
                module.forward = sa_forward(module)
                setattr(module, "place_in_unet", f"down_res{res}_block{block}_attn{idx}")
                setattr(module, "num_concepts", num_concepts)
                setattr(module, "t_cond", t_cond)

    # mid_block cross-attn
    for idx in range(mid_transformers_len):
        module = model.unet.mid_block.attentions[0].transformer_blocks[idx].attn2
        key_prefix = f"mid_block.attentions.0.transformer_blocks.{idx}.attn2"
        _attach_kv_from_sts(module, key_prefix)
        module.forward = sa_forward(module)
        setattr(module, "place_in_unet", f"mid_block_attn{idx}")
        setattr(module, "num_concepts", num_concepts)
        setattr(module, "t_cond", t_cond)
