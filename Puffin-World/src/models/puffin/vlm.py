import math
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence
from mmengine.logging import print_log
from mmengine.model import BaseModel
from xtuner.utils import IGNORE_INDEX
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from xtuner.dataset.map_fns.template_map_fn import template_map_fn

from src.datasets.utils import (
    encode_fn,
    DEFAULT_IMAGE_TOKEN,
    INPUT_IMAGE_TOKEN_INDEX,
)


# transformers 2.x removed `is_safetensors_available` which Qwen3.5 imports.
import transformers.utils as _tu
if not hasattr(_tu, 'is_safetensors_available'):
    _tu.is_safetensors_available = lambda: True


original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_load(*args, **kwargs)
torch.load = patched_load


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def pad_an_image_tensor(image, pad_value=0):
    h, w = image.shape[-2:]
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        p2d = (pad_left, pad_right, 0, 0)
    else:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        p2d = (0, 0, pad_top, pad_bottom)
    return F.pad(image, p2d, "constant", pad_value)


class _RoPEIndexHelper:
    """Lightweight proxy for Qwen3.5's `get_rope_index` (3D MRoPE).

    The real `Qwen3_5Model.get_rope_index` only needs
    `self.config.vision_config.spatial_merge_size` and
    `self.get_vision_position_ids`, so we provide exactly that — avoiding the
    cost of instantiating a full Qwen3.5-VL model just to compute MRoPE indices.
    For RADIO features (already spatially merged), `spatial_merge_size=1`.
    """

    def __init__(self, spatial_merge_size=1):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        self._impl = Qwen3_5Model
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size)
        )

    def get_vision_position_ids(self, *args, **kwargs):
        return self._impl.get_vision_position_ids(self, *args, **kwargs)

    def get_rope_index(self, *args, **kwargs):
        return self._impl.get_rope_index(self, *args, **kwargs)


_LEGACY_FAMILIES = {'qwen2_5', 'qwen3'}
_MROPE_FAMILIES = {'qwen3_5'}
_SUPPORTED_FAMILIES = _LEGACY_FAMILIES | _MROPE_FAMILIES


class RadioVLM(BaseModel):
    """Vision-Language expert model — RADIO visual encoder + projector + LLM.

    Supports three LLM families via `llm_family`:
      - 'qwen2_5' / 'qwen3' : 1D RoPE positions, xtuner-style prompt_template
        (uses `template_map_fn` + `encode_fn`, with `INPUT_IMAGE_TOKEN_INDEX`
        sentinel for image-token placeholders).
      - 'qwen3_5'           : 3D MRoPE positions via `Qwen3_5Model.get_rope_index`,
        HF native `tokenizer.apply_chat_template` for prompt construction (real
        `image_token_id` written into input_ids).

    The visual encoder + projector + extract_visual_features path is shared.
    """

    def __init__(self,
                 llm,
                 tokenizer,
                 prompt_template,
                 visual_encoder,
                 llm_family='qwen2_5',
                 max_length=256,
                 freeze_visual_encoder=True,
                 freeze_projector=False,
                 freeze_llm=True,
                 visual_encoder_grad_scale=0.1,
                 fold_size=2,
                 pretrained_pth=None,
                 use_activation_checkpointing=False,
                 *args, **kwargs):
        super().__init__()

        assert llm_family in _SUPPORTED_FAMILIES, (
            f"llm_family={llm_family!r} not in {_SUPPORTED_FAMILIES}"
        )
        self.llm_family = llm_family
        self.use_mrope = llm_family in _MROPE_FAMILIES

        self.max_length = max_length
        self.fold_size = fold_size
        self.prompt_template = prompt_template

        # understanding branch
        self.visual_encoder = BUILDER.build(visual_encoder)
        self.llm = BUILDER.build(llm)
        self.tokenizer = BUILDER.build(tokenizer)
        self.projector = build_mlp(
            hidden_size=self.visual_encoder.model.embed_dim * fold_size ** 2,
            projector_dim=self.llm.config.hidden_size,
            z_dim=self.llm.config.hidden_size,
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(
            prompt_template['IMG_CONTEXT_TOKEN']
        )

        # MRoPE helper (only constructed for Qwen3.5 family to avoid the
        # transformers.models.qwen3_5 import path on legacy installs).
        self._rope_helper = _RoPEIndexHelper(spatial_merge_size=1) if self.use_mrope else None

        # freeze switches
        if freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
        self.freeze_visual_encoder = freeze_visual_encoder
        if freeze_projector:
            self.projector.requires_grad_(False)
        self.freeze_projector = freeze_projector
        if freeze_llm:
            self.llm.requires_grad_(False)
        self.freeze_llm = freeze_llm

        self.visual_encoder_grad_scale = visual_encoder_grad_scale

        self.use_activation_checkpointing = use_activation_checkpointing
        if use_activation_checkpointing:
            self.llm.enable_input_require_grads()
            self.gradient_checkpointing_enable()

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

    # ------------------------------------------------------------------
    # core hooks
    # ------------------------------------------------------------------
    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def train(self, mode=True):
        super().train(mode=mode)
        if self.freeze_visual_encoder:
            self.visual_encoder.train(mode=False)
        if self.freeze_projector:
            self.projector.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()
        return self

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # visual feature extraction (shared)
    # ------------------------------------------------------------------
    def extract_visual_features(self, pixel_values):
        pixel_values = (pixel_values + 1.0) / 2     # [-1, 1] -> [0, 1]
        height, width = pixel_values.shape[-2:]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            summary, features = self.visual_encoder(pixel_values)
        patch_size = int((height * width // features.shape[1]) ** 0.5)
        height, width = height // (patch_size * self.fold_size), width // (patch_size * self.fold_size)
        features = rearrange(
            features, 'b (h p w q) d -> b (h w) (p q d)',
            h=height, w=width, p=self.fold_size, q=self.fold_size,
        )
        return features

    # ------------------------------------------------------------------
    # prompt prep — branches on llm_family
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prepare_und_prompts(self, conversations, data_type='image2text',
                            image_lengths=None, input_ids_with_output=True):
        if self.use_mrope:
            return self._prepare_und_prompts_mrope(
                conversations, data_type, image_lengths, input_ids_with_output,
            )
        return self._prepare_und_prompts_legacy(
            conversations, data_type, image_lengths, input_ids_with_output,
        )

    # ---- legacy path (Qwen2.5 / Qwen3) ----
    def _prepare_und_prompts_legacy(self, conversations, data_type,
                                    image_lengths, input_ids_with_output):
        input_ids, labels, input_lengths = [], [], []

        if data_type == 'image2text':
            assert image_lengths is not None, "`image_lengths` must be provided for image2text"
            if isinstance(image_lengths, int):
                image_lengths = [image_lengths] * len(conversations)
        elif data_type == 'text2text':
            image_lengths = [None] * len(conversations)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        for conv, image_len in zip(conversations, image_lengths):
            data_dict = template_map_fn(example=dict(conversation=deepcopy(conv)),
                                        template=self.prompt_template)
            data_dict.update(encode_fn(
                data_dict,
                tokenizer=self.tokenizer,
                max_length=None,
                input_ids_with_output=input_ids_with_output,
                with_image_token=(data_type == 'image2text'),
                image_length=image_len,
                prompt_template=self.prompt_template,
            ))

            input_ids.append(torch.tensor(data_dict['input_ids'], dtype=torch.long, device=self.device))
            labels.append(torch.tensor(data_dict['labels'], dtype=torch.long, device=self.device))
            input_lengths.append(len(data_dict['input_ids']))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0, padding_side='left')
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side='left')

        attention_mask = torch.zeros_like(input_ids).bool()
        for i in range(len(input_ids)):
            attention_mask[i, -input_lengths[i]:] = True

        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        return dict(input_ids=input_ids, attention_mask=attention_mask,
                    labels=labels, position_ids=position_ids)

    # ---- MRoPE path (Qwen3.5) ----
    def _prepare_und_prompts_mrope(self, conversations, data_type,
                                   image_lengths, input_ids_with_output):
        input_ids_list, labels_list, input_lengths = [], [], []

        if data_type == 'image2text':
            assert image_lengths is not None, "`image_lengths` must be provided for image2text"
            if isinstance(image_lengths, int):
                image_lengths = [image_lengths] * len(conversations)
        elif data_type == 'text2text':
            image_lengths = [None] * len(conversations)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        for conv, image_len in zip(conversations, image_lengths):
            messages = []
            for turn in conv:
                user_input = turn.get('input', '')
                if image_len is not None:
                    image_tokens = (
                        self.prompt_template['IMG_START_TOKEN']
                        + self.prompt_template['IMG_CONTEXT_TOKEN'] * image_len
                        + self.prompt_template['IMG_END_TOKEN']
                    )
                    user_input = user_input.replace(DEFAULT_IMAGE_TOKEN, image_tokens)
                messages.append({'role': 'user', 'content': user_input})
                if 'output' in turn and input_ids_with_output:
                    messages.append({'role': 'assistant', 'content': turn['output']})

            if input_ids_with_output and messages and messages[-1]['role'] == 'assistant':
                input_messages = messages[:-1]
                input_text = self.tokenizer.apply_chat_template(
                    input_messages, tokenize=False, add_generation_prompt=True)
                input_len = len(self.tokenizer(input_text, add_special_tokens=False).input_ids)

                full_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
                full_token_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
                lbl = [IGNORE_INDEX] * input_len + full_token_ids[input_len:]
            else:
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                full_token_ids = self.tokenizer(input_text, add_special_tokens=False).input_ids
                lbl = [IGNORE_INDEX] * len(full_token_ids)
                input_len = len(full_token_ids)

            input_ids_list.append(torch.tensor(full_token_ids, dtype=torch.long, device=self.device))
            labels_list.append(torch.tensor(lbl, dtype=torch.long, device=self.device))
            input_lengths.append(input_len)

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0, padding_side='left')
        labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX, padding_side='left')

        attention_mask = torch.zeros_like(input_ids).bool()
        for i in range(len(input_ids_list)):
            attention_mask[i, -len(input_ids_list[i]):] = True

        _img_lens = image_lengths if data_type == 'image2text' else None
        mm_token_type_ids, image_grid_thw = self._build_rope_inputs(
            input_ids, attention_mask, image_lengths=_img_lens,
        )
        position_ids, _ = self._rope_helper.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        return dict(input_ids=input_ids, attention_mask=attention_mask,
                    labels=labels, position_ids=position_ids)

    def _build_rope_inputs(self, input_ids, attention_mask, image_embeds=None,
                            image_lengths=None):
        """Build `mm_token_type_ids` and `image_grid_thw` for MRoPE
        (Qwen3.5 family only). Grid inferred from image_embeds shape when
        present, otherwise from `image_lengths` (assumes square images)."""
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        mm_token_type_ids[input_ids == self.image_token_id] = 1

        image_grid_thw = None
        if image_embeds is not None:
            num_patches = image_embeds.shape[1]
            h = w = int(math.sqrt(num_patches))
            num_images = image_embeds.shape[0]
            image_grid_thw = torch.tensor(
                [[1, h, w]] * num_images, dtype=torch.long, device=input_ids.device)
        elif image_lengths is not None:
            img_len = image_lengths[0] if isinstance(image_lengths, list) else image_lengths
            h = w = int(math.sqrt(img_len))
            num_images = (input_ids == self.image_token_id).sum().item() // img_len
            if num_images > 0:
                image_grid_thw = torch.tensor(
                    [[1, h, w]] * num_images, dtype=torch.long, device=input_ids.device)
        return mm_token_type_ids, image_grid_thw

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------
    def _slice_pos(self, position_ids, max_length):
        """Slice the last `max_length` tokens. MRoPE has an extra leading axis."""
        if self.use_mrope:
            return position_ids[:, :, -max_length:]
        return position_ids[:, -max_length:]

    def _image_slot_mask(self, input_ids):
        """Where to inject `image_embeds` into `inputs_embeds`.

        Legacy path uses the negative sentinel `INPUT_IMAGE_TOKEN_INDEX` (from
        xtuner's `encode_fn`); MRoPE path uses the real tokenizer image_token_id.
        """
        if self.use_mrope:
            return input_ids == self.image_token_id
        return input_ids == INPUT_IMAGE_TOKEN_INDEX

    def image2text_loss(self, data_dict):
        pixel_values = [pad_an_image_tensor(img) for img in data_dict['pixel_values']]
        pixel_values = torch.stack(pixel_values).to(dtype=self.dtype, device=self.device)
        image_embeds = self.extract_visual_features(pixel_values)

        if not self.freeze_visual_encoder:
            image_embeds = _ScaleGradient.apply(image_embeds, self.visual_encoder_grad_scale)

        image_embeds = self.projector(image_embeds)
        text_inputs = self.prepare_und_prompts(
            conversations=data_dict['conversations'],
            data_type='image2text',
            image_lengths=image_embeds.shape[1],
        )
        labels = text_inputs['labels']
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
        position_ids = text_inputs['position_ids']

        img_mask = self._image_slot_mask(input_ids)
        inputs_embeds = torch.zeros(*input_ids.shape, self.llm.config.hidden_size,
                                    device=self.device, dtype=self.dtype)
        inputs_embeds[img_mask] = image_embeds.flatten(0, 1)
        inputs_embeds[~img_mask] = self.llm.get_input_embeddings()(input_ids[~img_mask])

        max_length = self.max_length + image_embeds.shape[1]
        inputs_embeds = inputs_embeds[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
        position_ids = self._slice_pos(position_ids, max_length)
        labels = labels[:, -max_length:]

        output = self.llm.model(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                return_dict=True)

        hidden_states = output.last_hidden_state[:, :-1]
        labels = labels[:, 1:]
        hidden_states = hidden_states[labels >= 0]
        labels = labels[labels >= 0]

        logits = self.llm.get_output_embeddings()(hidden_states)
        return F.cross_entropy(input=logits, target=labels)

    def text2text_loss(self, data_dict):
        text_inputs = self.prepare_und_prompts(
            conversations=data_dict['conversations'], data_type='text2text',
        )
        labels = text_inputs['labels']
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
        position_ids = text_inputs['position_ids']

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        max_length = self.max_length
        inputs_embeds = inputs_embeds[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
        position_ids = self._slice_pos(position_ids, max_length)
        labels = labels[:, -max_length:]

        output = self.llm.model(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                return_dict=True)

        hidden_states = output.last_hidden_state[:, :-1]
        labels = labels[:, 1:]
        hidden_states = hidden_states[labels >= 0]
        labels = labels[labels >= 0]

        logits = self.llm.get_output_embeddings()(hidden_states)
        return F.cross_entropy(input=logits, target=labels)

    def compute_loss(self, data=None, data_dict=None):
        if data_dict is None:
            data_dict = data
        if data_dict is None:
            raise ValueError("compute_loss expects `data` or `data_dict`, but got None.")

        if isinstance(data_dict, dict) and "data" in data_dict:
            batch_root = data_dict["data"]
        else:
            batch_root = data_dict

        loss_fn_map = {
            'image2text': self.image2text_loss,
            'text2text': self.text2text_loss,
            'image2text_cross_view': self.image2text_loss,
        }
        known_types = set(loss_fn_map.keys())

        multi_task_keys = []
        if isinstance(batch_root, dict):
            multi_task_keys = [k for k in batch_root.keys() if k in known_types]
        if len(multi_task_keys) > 0:
            per_type_batches = {k: batch_root[k] for k in multi_task_keys}
        else:
            per_type_batches = {"image2text": batch_root}

        losses = {}
        for data_type, batch_data in per_type_batches.items():
            if data_type not in loss_fn_map:
                raise ValueError(f"Unsupported data_type: {data_type}")
            losses[f"loss_{data_type}"] = loss_fn_map[data_type](batch_data)
        return losses

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def understand(self, prompt, pixel_values, max_new_tokens=512, progress_bar=True):
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = [pixel_values]

        bsz = len(prompt)
        assert len(pixel_values) == bsz

        pixel_values = [pad_an_image_tensor(img) for img in pixel_values]
        pixel_values = torch.stack(pixel_values).to(dtype=self.dtype, device=self.device)
        image_embeds = self.extract_visual_features(pixel_values)
        image_embeds = self.projector(image_embeds)

        conversations = [[{'input': f"{DEFAULT_IMAGE_TOKEN}\n{p}"}] for p in prompt]
        text_inputs = self.prepare_und_prompts(
            conversations=conversations,
            image_lengths=image_embeds.shape[1],
            input_ids_with_output=False,
        )

        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
        position_ids = text_inputs['position_ids']

        img_mask = self._image_slot_mask(input_ids)
        inputs_embeds = torch.zeros(*input_ids.shape, self.llm.config.hidden_size,
                                    device=self.device, dtype=self.dtype)
        inputs_embeds[img_mask] = image_embeds.flatten(0, 1)
        inputs_embeds[~img_mask] = self.llm.get_input_embeddings()(input_ids[~img_mask])

        # Start with no cache and let the HF forward create the right class on
        # the first step: hybrid models (qwen3_5 gated-deltanet layers) need
        # their model-specific cache — a plain DynamicCache lacks
        # `has_previous_state` and crashes inside the linear-attention layers.
        past_key_values = None
        output_ids = []

        for _ in tqdm(range(max_new_tokens), disable=not progress_bar):
            output = self.llm.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = self.llm.get_output_embeddings()(output.last_hidden_state[:, -1:])
            new_ids = torch.argmax(logits, dim=-1)
            if len(output_ids) > 0:
                new_ids = torch.where(
                    output_ids[-1] == self.tokenizer.eos_token_id,
                    output_ids[-1], new_ids,
                )
            output_ids.append(new_ids)

            if (new_ids == self.tokenizer.eos_token_id).all():
                break

            inputs_embeds = self.llm.get_input_embeddings()(new_ids)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(bsz, 1)], dim=1)
            if self.use_mrope:
                # MRoPE: [3, B, L] -> increment along token axis (dim=2)
                position_ids = position_ids.max(dim=2, keepdim=True).values + 1
            else:
                # 1D: [B, L] -> increment along token axis (dim=1)
                position_ids = torch.max(position_ids, dim=1, keepdim=True).values + 1
            past_key_values = output.past_key_values

        output_ids = torch.cat(output_ids, dim=1)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


# ---- Backwards-compat alias ----
# Pre-merge configs referenced `Qwen2p5RadioVLM`. New code should use `RadioVLM`
# directly with `llm_family=...`; the alias keeps existing configs loading.
Qwen2p5RadioVLM = RadioVLM
