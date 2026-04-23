# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
import torch.distributed as dist
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from PIL import Image
from data.data_utils import (
    create_sparse_mask,
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    patchify,
    pil_img2rgb
)
from data.dataset_base import PackedDataset
from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.qwen2 import Qwen2Tokenizer
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding

from tqdm import tqdm

def _broadcast_sequence_status(sequence_status, device, src: int = 0):
    """
     rank0 sequence_status rank
    - CPU CPU GPU->CPU
    - rank device
    """
    if dist.is_available() and dist.is_initialized():
        obj_list = [sequence_status] if dist.get_rank() == src else [None]
        dist.broadcast_object_list(obj_list, src=src)
        sequence_status = obj_list[0]
    # rank device
    for k, v in sequence_status.items():
        if torch.is_tensor(v):
            sequence_status[k] = v.to(device, non_blocking=True)
    return sequence_status

global_vae_model=None
global_tokenizer=None
class BagelConfigGenerationREPA(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class BagelGenerationREPA(PreTrainedModel):
    config_class = BagelConfigGenerationREPA
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfigGenerationREPA, vae_model=None):
        super().__init__(config)
        global global_vae_model, global_tokenizer
        from data.transforms import ImageTransform,ImageTransformTensor
        self.vit_transform = ImageTransform(
            max_image_size=980,
            min_image_size=224,
            image_stride=14,
        )
        self.vae_transform = ImageTransform(
            max_image_size=1024,
            min_image_size=512,
            image_stride=16,
        )
        self.vit_tensor_transform = ImageTransformTensor(
            max_image_size=980,
            min_image_size=224,
            image_stride=14,
        )
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads
        self.fake_cls_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size//4),
            nn.ReLU(),
            nn.Linear(self.hidden_size//4, 2)
        )
        global_vae_model = vae_model
        global_tokenizer = Qwen2Tokenizer.from_pretrained("/path/to/project/pretrained/")
        global_tokenizer, new_token_ids, _ = add_special_tokens(global_tokenizer)

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        self.vit_model = vit_model
        self.vit_patch_size = config.vit_config.patch_size
        self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
        self.vit_hidden_size = config.vit_config.hidden_size
        self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
        self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)
        self.repa_mlp = nn.Sequential(
            Qwen2RMSNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        # New module to map reconstructed VAE latent to ViT embedding space
        if config.visual_gen:
            self.latent_to_vit_connector = nn.Sequential(
                nn.Linear(self.patch_latent_dim, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

        # self.repa_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,

        # # uid fake
        generation_auth_explanation_list = None,
        gt_vit_list = None,
        gt_prompt_list = None,
        dataset = None

        # vae_group_uids: Optional[torch.LongTensor] = None,
        # vae_group_is_fake: Optional[torch.BoolTensor] = None,
        # vit_group_uids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor, where 0.0 means attention and
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens,
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )
        # import bisect

        # PyTorch CPU GPU
        def find_included_samples_torch(sample_lens, packed_vae_token_indexes,packed_timesteps):
            """
             PyTorch
            """
            # sample_lens
            if not torch.is_tensor(sample_lens):
                sample_lens = torch.tensor(sample_lens, device=packed_vae_token_indexes.device)

            cumulative = torch.cat([torch.tensor([0], device=sample_lens.device),
                                torch.cumsum(sample_lens, dim=0)])

            included_indices = []
            for i in range(len(sample_lens)):
                start = cumulative[i]
                end = cumulative[i+1] - 1

                # PyTorch
                # [start, end]
                mask = ((packed_vae_token_indexes >= start) & (packed_vae_token_indexes <= end) & (packed_timesteps > 1e-7))
                if torch.any(mask):
                    included_indices.append(i)

            return included_indices

        generation_repa_hidden_states, last_hidden_state = self.language_model.forward_repa(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        # --- Flow Matching MSE ---
        mse = None
        has_mse = None
        if self.config.visual_gen and mse_loss_indexes is not None and packed_latent_clean is not None:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes]) # v_t
            target = noise - packed_latent_clean # v_t = noise - x0
            has_mse = (packed_timesteps > 0) # t>0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        # --- x0_hat → → ViT → det_packed_sequence ---
        det_ce = None
        if (
            self.config.visual_gen
        ):
            estimated_v_t = packed_mse_preds
            full_est_clean = packed_latent_clean
            # has_mse = (packed_timesteps > 0)
            # full_est_clean[has_mse] = noise[has_mse] - estimated_v_t
            # safer out-of-place
            mask = has_mse.unsqueeze(-1) # align dims as needed
            new_full = torch.where(mask, noise - estimated_v_t, packed_latent_clean)
            full_est_clean = new_full

            # 2) rank0 sequence_status rank
            device = packed_text_ids.device
            sequence_status = None
            do_build_here = (not (dist.is_available() and dist.is_initialized())) or (dist.get_rank() == 0)
            included_samples = find_included_samples_torch(sample_lens, packed_vae_token_indexes, packed_timesteps)
            # if do_build_here:
            # with torch.no_grad():
            if vit_token_seqlens is None:
                num_gen_imgs = len(patchified_vae_latent_shapes)
            else:
                num_gen_imgs = len(patchified_vae_latent_shapes) - len(vit_token_seqlens)
            gen_tok_ptr = 0
            seq_stat = dataset.set_sequence_status()
            vit_image_shape_list = []
            for i in range(num_gen_imgs):
                text_ids_list = []
                sequence_plan = []
                h, w = patchified_vae_latent_shapes[included_samples[i]]
                n_gen_tok = h * w
                gen_slice = slice(gen_tok_ptr, gen_tok_ptr + n_gen_tok)
                gen_tok_ptr += n_gen_tok

                vit_image_tensor_list, vae_image_tensor_list = [], []
                # with torch.no_grad():
                    # # latent -> ( rank0 )
                    # p = self.config.latent_patch_size
                    # C = self.latent_channel
                num_tokens = 0

                    # latent_chHW = torch.einsum(
                    # "hwpqc->chpwq",
                    # full_est_clean[gen_slice].reshape(h, w, p, p, C)
                    # ).reshape(C, h * p, w * p)
                    # img_tensor = global_vae_model.decode(latent_chHW.unsqueeze(0))[0] # [-1,1]
                    # img_tensor = (img_tensor * 0.5 + 0.5).clamp(0, 1) # [0,1]

                    # # vit_image = self.vit_transform(pil_img2rgb(
                    # # Image.fromarray((img_tensor.float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    # # ))
                    # vae_image_tensor_list.append(img_tensor)

                vae_image = self.vae_transform(gt_vit_list[i])
                vae_image_tensor_list.append(vae_image)

                vit_image = self.vit_transform(gt_vit_list[i])
                # vit_image = vit_image # CPU
                vit_image_tensor_list.append(vit_image)

                # sequence_plan.append({'type': 'vit_image', 'loss': 0})
                # sequence_plan.append({'type': 'vae_image', 'loss': 0})

                sequence_plan.append({'type': 'vae_image', 'loss': 0})
                # rank0 tokenize
                head_txt = global_tokenizer.encode(f"""Act as a forensic image analyst. The image was generated for the prompt: "{gt_prompt_list[i]}".
                        Briefly and professionally state the specific visual evidence that supports the image's authenticity.
                        For each point, concisely link it to a prompt element (e.g., “<element>: <visual evidence>”).
                        Use very brief, factual language and limit the output to only the most critical 2-3 points.""")
                # head_txt = global_tokenizer.encode("Act as a forensic image analyst. Briefly state your professional basis for considering this image authentic and real, focusing on the key evidence.")
                text_ids_list.append(head_txt)
                sequence_plan.append({'type': 'text', 'loss': 0})
                num_tokens += len(head_txt)

                sequence_plan.append({'type': 'vit_image', 'loss': 0})


                # text_data = generation_auth_explanation_list[i]
                # text_ids = global_tokenizer.encode(text_data)
                # text_ids_list.append(text_ids)
                # num_tokens += len(text_ids)
                # sequence_plan.append({'type': 'text', 'loss': 1})
                # print(text_data)
                height, width = vit_image.shape[1:]
                num_tokens += width * height // 14 ** 2

                vae_height, vae_width = vae_image.shape[1:]
                num_tokens += vae_width * vae_height // 16 ** 2
                vit_image_shape_list.append((height//14, width//14))
                # Finalize sequence plan details
                for plan in sequence_plan:
                    plan.update({
                        'enable_cfg': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                sequence = dict(
                    vit_image_tensor_list=vit_image_tensor_list,
                    gen_image_tensor_list=vae_image_tensor_list,
                    text_ids_list=text_ids_list,
                    num_tokens=num_tokens,
                    sequence_plan=sequence_plan,
                    # dataset.pack_sequence
                    data_indexes = {
                        "data_indexes": 0,
                        "worker_id": 0,
                        "dataset_name": "t2i_pretrain",
                    },
                    label=0,
                    generation_auth_explanation_list=[generation_auth_explanation_list[i]],
                    gt_vit_list=[gt_vit_list[i]],
                    gt_prompt_list = [gt_prompt_list[i]],
                )
                seq_stat = dataset.pack_sequence(sequence, seq_stat)

            # CPU
            sequence_status = dataset.to_tensor(seq_stat)
            for k, v in sequence_status.items():
                if torch.is_tensor(v):
                    # sequence_status[k] = v.cpu()
                    sequence_status[k] = v.to(device)

            # # rank device
            # sequence_status = _broadcast_sequence_status(sequence_status, device, src=0)

            # ===== rank sequence_status =====
            with torch.no_grad():
                packed_text_embedding = self.language_model.model.embed_tokens(sequence_status['packed_text_ids'])
                packed_sequence = packed_text_embedding.new_zeros(size=(sequence_status['sequence_length'], self.hidden_size))
                packed_sequence[sequence_status['packed_text_indexes']] = packed_text_embedding

                if sequence_status['nested_attention_masks'] is None:
                    sparse_mask = create_sparse_mask(sequence_status['sample_lens'], sequence_status['split_lens'], sequence_status['attn_modes'], packed_text_embedding.device)
                    seqlen = sum(sequence_status['sample_lens'])
                    block_mask = create_block_mask(
                        sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                        device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
                    )
                    attention_mask = block_mask
                else:
                    attention_mask = [mask.to(device) for mask in sequence_status['nested_attention_masks']]

                cu_seqlens = torch.nn.functional.pad(torch.cumsum(sequence_status['vit_token_seqlens'], dim=0), (1, 0))
                cu_seqlens = cu_seqlens.to(torch.int32)
                max_seqlen = torch.max(sequence_status['vit_token_seqlens']).item()
                packed_vit_token_embed = self.vit_model(
                    packed_pixel_values=sequence_status['packed_vit_tokens'],
                    packed_flattened_position_ids=sequence_status['packed_vit_position_ids'],
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                packed_vit_token_embed = self.connector(packed_vit_token_embed)
                vit_token_pos_emb = self.vit_pos_embed(sequence_status['packed_vit_position_ids'])
                packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
                packed_sequence[sequence_status['packed_vit_token_indexes']] = packed_vit_token_embed
                extra_inputs = {}
                if self.use_moe:
                    packed_und_token_indexes = sequence_status['packed_text_indexes']
                    if sequence_status['packed_vit_token_indexes'] is not None:
                        packed_und_token_indexes = torch.cat([sequence_status['packed_text_indexes'], sequence_status['packed_vit_token_indexes']], dim=0)
                    extra_inputs.update(
                        packed_und_token_indexes=packed_und_token_indexes,
                        packed_gen_token_indexes=sequence_status['packed_vae_token_indexes'],
                    )

                # padded_latent = global_vae_model.encode(sequence_status['padded_images'].to(device))
                packed_latent = []
                p = self.config.latent_patch_size
                C = self.latent_channel
                num_tokens = 0

                # latent_chHW = torch.einsum(
                # "hwpqc->chpwq",
                # full_est_clean[gen_slice].reshape(h, w, p, p, C)
                # ).reshape(C, h * p, w * p)
                # gen_tok_ptr = 0
                # for i, (h, w) in zip(range(len(included_samples)), sequence_status['patchified_vae_latent_shapes']):
                # h, w = patchified_vae_latent_shapes[included_samples[i]]
                # n_gen_tok = h * w
                # gen_slice = slice(gen_tok_ptr, gen_tok_ptr + n_gen_tok)
                # gen_tok_ptr += n_gen_tok
                # latent = torch.einsum(
                # "hwpqc->chpwq",
                # full_est_clean[gen_slice].reshape(h, w, p, p, C)
                # ).reshape(C, h * p, w * p)
                # latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                # latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                # packed_latent.append(latent)
                # packed_latent_clean = torch.cat(packed_latent, dim=0)
                padded_latent = global_vae_model.encode(sequence_status.pop('padded_images'))
                p = self.latent_patch_size
                packed_latent = []
                for latent, (h, w) in zip(padded_latent, sequence_status['patchified_vae_latent_shapes']):
                    latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                    latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                    packed_latent.append(latent)
                packed_latent_clean = torch.cat(packed_latent, dim=0)

                noise = torch.randn_like(packed_latent_clean)
                packed_timesteps = torch.sigmoid(sequence_status['packed_timesteps'].to(device))
                packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
                packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
                packed_timestep_embeds = self.time_embedder(packed_timesteps)
                latent_token_pos_emb = self.latent_pos_embed(sequence_status['packed_latent_position_ids'])
                packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
                packed_sequence[sequence_status['packed_vae_token_indexes']] = packed_latent
                idx_exp = sequence_status['packed_vae_token_indexes'].unsqueeze(1).expand(-1, self.hidden_size) # long tensor
                packed_sequence = packed_sequence.scatter(0, idx_exp, packed_latent)

                # det_repa_hidden_states = self.language_model(
                # packed_sequence=packed_sequence,
                # sample_lens=sequence_status['sample_lens'],
                # attention_mask=attention_mask,
                # packed_position_ids=sequence_status['packed_position_ids'],
                # **extra_inputs,

            # print(packed_sequence.shape,det_repa_hidden_states[sequence_status['packed_vit_token_indexes']].shape,generation_repa_hidden_states.shape)

            # --- Start of requested modification ---

            # 1. Extract ViT tokens from the second pass (det)
            vit_tokens_det = packed_sequence[sequence_status['packed_vit_token_indexes']]

            # 2. Split, reshape, and resize ViT tokens
            split_vit_tokens_det = torch.split(vit_tokens_det, sequence_status['vit_token_seqlens'].tolist())

            resized_vit_tokens_list = []
            for i, tokens in enumerate(split_vit_tokens_det):

                h_vit, w_vit = vit_image_shape_list[i]
                h_vae, w_vae = sequence_status['patchified_vae_latent_shapes'][i]
                if i==0 and dist.get_rank() == 0:
                    print("(h_vit, w_vit):", (h_vit, w_vit), "(h_vae, w_vae):", (h_vae, w_vae))
                # Reshape to (H, W, D), then permute to (D, H, W) for interpolation
                tokens_grid = tokens.view(h_vit, w_vit, -1).permute(2, 0, 1).unsqueeze(0) # Add batch dim

                # Interpolate
                resized_tokens_grid = F.interpolate(
                    tokens_grid,
                    size=(h_vae, w_vae),
                    mode='bilinear',
                    align_corners=False
                )

                # Permute back to (H', W', D) and flatten
                resized_tokens_flat = resized_tokens_grid.squeeze(0).permute(1, 2, 0).reshape(-1, self.hidden_size)
                resized_vit_tokens_list.append(resized_tokens_flat)

            resized_vit_tokens = torch.cat(resized_vit_tokens_list, dim=0)

            # 3. Extract and process latent tokens from the first pass (generation)
            latent_tokens_gen = generation_repa_hidden_states[packed_vae_token_indexes]
            processed_latent_tokens = self.repa_mlp(latent_tokens_gen)

            # 4. Calculate cosine similarity loss
            # Ensure both tensors are normalized before dot product, or use F.cosine_similarity
            cos_sim_loss = 1 - F.cosine_similarity(resized_vit_tokens, processed_latent_tokens, dim=1).mean()

            # You can assign this to det_ce or a new variable to be returned
            repa_loss = cos_sim_loss

            # --- End of requested modification ---

        ce = None
        fake_det_token_numbers = None
        # if ce_loss_indexes is not None:
        # def find_mutation_points_v2(ce_loss_indexes):
        # # return [ce_loss_indexes[i-1] for i in range(1, len(ce_loss_indexes))
        # # if ce_loss_indexes[i] - ce_loss_indexes[i-1] != 1] + [ce_loss_indexes[-1]]
        # return [ce_loss_indexes[0]]+[ce_loss_indexes[i] for i in range(1, len(ce_loss_indexes))
        # if ce_loss_indexes[i] - ce_loss_indexes[i-1] != 1]
        # # get first_token
        # packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
        # # print(find_mutation_points_v2(ce_loss_indexes))
        # fake_det_token_numbers = torch.tensor(find_mutation_points_v2(ce_loss_indexes))
        # # print(fake_det_token_numbers)
        # ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        fake_predicted = self.fake_cls_head(last_hidden_state[fake_det_token_numbers])

        return dict(mse=mse, ce=ce, repa_loss=repa_loss), fake_predicted
        # return dict(mse=mse, ce=ce), None


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = output.packed_query_sequence

        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = cfg_text_output.packed_query_sequence
            v_t = cfg_text_v_t * cfg_text_scale + v_t * (1 - cfg_text_scale)

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = cfg_img_output.packed_query_sequence
            v_t = cfg_img_v_t * cfg_img_scale + v_t * (1 - cfg_img_scale)

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens),
                device=key_values_lens.device,
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    @torch.no_grad
    def generate_text_det(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens),
                device=key_values_lens.device,
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            # print(packed_query_sequence)
            fake_predicted = self.fake_cls_head(packed_query_sequence)
            # pred_logits = self.language_model.lm_head(packed_query_sequence)

            # if do_sample:
            # probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
            # curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # else:
            # curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break
            break

        output_device = generated_sequence[0].device
        return fake_predicted


    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[image],
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[prompt],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output