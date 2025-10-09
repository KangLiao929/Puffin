import random
import torch
import math
from tqdm import tqdm
from einops import rearrange
from copy import deepcopy
from six.moves import zip
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence
from mmengine.logging import print_log
from mmengine.model import BaseModel
from xtuner.utils import IGNORE_INDEX
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from xtuner.dataset.map_fns.template_map_fn import template_map_fn
from transformers.cache_utils import DynamicCache
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from src.models.connector import ConnectorConfig, ConnectorEncoder
from src.models.stable_diffusion3.pipeline_stable_diffusion_3_dynamic import StableDiffusion3Pipeline
from src.datasets.utils import encode_fn, QUERY_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, INPUT_IMAGE_TOKEN_INDEX

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
        nn.Linear(projector_dim, z_dim),)

def pad_an_image_tensor(image, pad_value=0):
    h, w = image.shape[-2:]
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        p2d = (pad_left, pad_right, 0, 0)
    else:
        pad_top = (h - w) // 2
        pad_bottom = h - w - pad_top
        p2d = (0, 0, pad_top, pad_bottom)

    image = F.pad(image, p2d, "constant", pad_value)

    return image

class Qwen2p5RadioStableDiffusion3HFDynamic(BaseModel):
    def __init__(self,
                 llm,
                 tokenizer,
                 prompt_template,
                 visual_encoder,
                 vae,
                 transformer,
                 train_scheduler,
                 test_scheduler,
                 connector_1,
                 connector_2,
                 num_queries=64,
                 freeze_transformer=True,
                 max_length=256,
                 freeze_visual_encoder=True,
                 freeze_llm=True,
                 visual_encoder_grad_scale=0.1,
                 fold_size=2,
                 unconditional=0.1,
                 unconditional_cross_view=0.1,
                 pretrained_pth=None,
                 use_activation_checkpointing=False,
                 *args, **kwargs):
        super().__init__()
        
        # basic settings
        self.max_length = max_length
        self.fold_size = fold_size
        self.prompt_template = prompt_template
        self.unconditional = unconditional
        self.unconditional_cross_view = unconditional_cross_view
        
        # networks building
        # understanding branch
        self.visual_encoder = BUILDER.build(visual_encoder)
        self.llm = BUILDER.build(llm)
        self.tokenizer = BUILDER.build(tokenizer)
        self.projector = build_mlp(hidden_size=self.visual_encoder.model.embed_dim*fold_size**2,
                                   projector_dim=self.llm.config.hidden_size,
                                   z_dim=self.llm.config.hidden_size)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
        
        # generation branch
        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)
        self.transformer = BUILDER.build(transformer)
        self.num_queries = num_queries
        self.connector_1 = ConnectorEncoder(ConnectorConfig(**connector_1))
        self.connector_2 = ConnectorEncoder(ConnectorConfig(**connector_2))

        self.llm2connector_1 = nn.Linear(self.llm.config.hidden_size, self.connector_1.config.hidden_size)
        self.llm2connector_2 = nn.Linear(self.llm.config.hidden_size, self.connector_2.config.hidden_size)
        self.projector_1 = nn.Linear(self.connector_1.config.hidden_size, self.transformer.config.pooled_projection_dim)
        self.projector_2 = nn.Linear(self.connector_2.config.hidden_size, self.transformer.config.joint_attention_dim)
        nn.init.zeros_(self.projector_1.weight)
        nn.init.zeros_(self.projector_2.weight)
        nn.init.zeros_(self.projector_1.bias)
        nn.init.zeros_(self.projector_2.bias)

        self.meta_queries = nn.Parameter(
            torch.zeros(num_queries, self.llm.config.hidden_size))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size))
        
        # networks and training initialization
        if freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
        self.freeze_visual_encoder = freeze_visual_encoder
        if freeze_llm:
            self.llm.requires_grad_(False)
        self.freeze_llm = freeze_llm
        if freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = freeze_transformer
        
        self.visual_encoder_grad_scale = visual_encoder_grad_scale
        self.train_scheduler = BUILDER.build(train_scheduler)
        self.test_scheduler = BUILDER.build(test_scheduler)

        self.use_activation_checkpointing = use_activation_checkpointing
        if use_activation_checkpointing:
            self.llm.enable_input_require_grads()
            self.gradient_checkpointing_enable()

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')
            
    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.transformer.enable_gradient_checkpointing()
        self.connector_1.gradient_checkpointing = True
        self.connector_2.gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.transformer.disable_gradient_checkpointing()
        self.connector_1.gradient_checkpointing = False
        self.connector_2.gradient_checkpointing = False
        
    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def extract_visual_features(self, pixel_values):
        pixel_values = (pixel_values + 1.0) / 2     # [0, 1]
        height, width = pixel_values.shape[-2:]
        summary, features = self.visual_encoder(pixel_values)
        patch_size = int((height * width // features.shape[1]) ** 0.5)
        height, width = height // (patch_size * self.fold_size), width // (patch_size * self.fold_size)
        features = rearrange(features, 'b (h p w q) d -> b (h w) (p q d)',
                             h=height, w=width, p=self.fold_size, q=self.fold_size)
        
        return features

    def llm2dit(self, x):
        x_1 = self.connector_1(self.llm2connector_1(x))
        x_1 = self.projector_1(x_1.mean(1))
        x_2 = self.connector_2(self.llm2connector_2(x))
        x_2 = self.projector_2(x_2)
        
        return x_1, x_2
    
    
    @torch.no_grad()
    def prepare_gen_prompts(self, texts, data_type='text2image', num_refs=None, ref_lens=None, gen_type='GENERATION_CROSS'):
        if data_type == 'text2image':
            prompts = [self.prompt_template['GENERATION'].format(input=text) for text in texts]
            prompts = [self.prompt_template['INSTRUCTION'].format(input=text) for text in prompts]

        elif data_type == 'image2image':
            assert num_refs is not None and ref_lens is not None, "num_refs and ref_lens are required for image2image"
            prompts = []
            cnt = 0
            for text, num_ref in zip(texts, num_refs):
                image_tokens = ''
                for _ in range(num_ref):
                    image_tokens += (
                        self.prompt_template['IMG_START_TOKEN'] +
                        self.prompt_template['IMG_CONTEXT_TOKEN'] * ref_lens[cnt] +
                        self.prompt_template['IMG_END_TOKEN']
                    )
                    cnt += 1

                text = self.prompt_template[gen_type].format(input=text)
                prompt = self.prompt_template['INSTRUCTION'].format(input=f'{image_tokens}\n{text}')
                prompts.append(prompt)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        return self.tokenizer(
            prompts, add_special_tokens=True, return_tensors='pt', padding=True, padding_side='left').to(self.device)


    @torch.no_grad()
    def prepare_und_prompts(self, conversations, data_type='image2text', image_lengths=None, input_ids_with_output=True):
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
            data_dict = template_map_fn(example=dict(conversation=deepcopy(conv)), template=self.prompt_template)
            data_dict.update(encode_fn(data_dict,
                                      tokenizer=self.tokenizer,
                                      max_length=None,
                                      input_ids_with_output=input_ids_with_output,
                                      with_image_token=(data_type == 'image2text'),
                                      image_length=image_len,
                                      prompt_template=self.prompt_template))

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

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels, position_ids=position_ids)

    def train(self, mode=True):
        super().train(mode=mode)
        self.vae.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()

        return self

    @torch.no_grad()
    def pixels_to_latents(self, x):
        z = self.vae.encode(x).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z):
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        x_rec = self.vae.decode(z).sample
        return x_rec

    def prepare_forward_input(self,
                              query_embeds,
                              input_ids=None,
                              image_embeds=None,
                              attention_mask=None,
                              past_key_values=None,
                              append_queries=True):
        b, l, _ = query_embeds.shape
        assert l > 0
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        assert l == self.num_queries

        if append_queries:
            input_ids = torch.cat([
                input_ids, input_ids.new_full(size=(b, l), fill_value=QUERY_TOKEN_INDEX)], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)

        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        # prepare context
        if past_key_values is not None:
            inputs_embeds = query_embeds
            position_ids = position_ids[..., -l:]
        else:
            inputs_embeds = torch.zeros(*input_ids.shape, self.llm.config.hidden_size,
                                        device=self.device, dtype=self.dtype)
            if image_embeds is not None:
                inputs_embeds[input_ids == self.image_token_id] = \
                    image_embeds.contiguous().view(-1, self.llm.config.hidden_size)

            inputs_embeds[input_ids == QUERY_TOKEN_INDEX] = \
                query_embeds.contiguous().view(-1, self.llm.config.hidden_size)

            text_places = torch.logical_and(input_ids != self.image_token_id, input_ids != QUERY_TOKEN_INDEX)

            inputs_embeds[text_places] = self.llm.get_input_embeddings()(input_ids[text_places])

        inputs = dict(inputs_embeds=inputs_embeds,
                      attention_mask=attention_mask,
                      position_ids=position_ids,
                      past_key_values=past_key_values)

        return inputs

    def get_sigmas(self, timesteps, n_dim=4):
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def diff_loss(self, model_input, pooled_prompt_embeds, prompt_embeds, cond_input=None):
        noise = [torch.randn_like(x) for x in model_input]
        bsz = len(model_input)

        u = compute_density_for_timestep_sampling(
            weighting_scheme='none',
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
        )
        indices = (u * self.train_scheduler.config.num_train_timesteps).long()
        timesteps = self.train_scheduler.timesteps[indices].to(device=self.device)

        # Add noise according to flow matching
        sigmas = self.get_sigmas(timesteps, n_dim=model_input[0].ndim + 1)
        noisy_model_input = [(1.0 - x) * y + x * z  for x, y, z in zip(sigmas, model_input, noise)]

        # Predict the noise residual
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            cond_hidden_states=cond_input,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            return_dict=False,
        )[0]

        weighting = compute_loss_weighting_for_sd3(weighting_scheme='none', sigmas=sigmas)

        # flow matching loss
        target = [x - y for x, y in zip(noise, model_input)]

        loss = [(x.float() * (y.float() - z.float()) ** 2).mean() for x, y, z in zip(weighting, model_pred, target)]
        loss = sum(loss) / len(loss)

        return loss

    '''text-to-image generation (single-view)'''
    def text2image_loss(self, data_dict):
        pixel_values = [p.to(dtype=self.dtype, device=self.device) for p in data_dict['pixel_values']]
        image_latents = [self.pixels_to_latents(p[None])[0] for p in pixel_values]

        b = len(image_latents)

        texts = ['' if random.uniform(0, 1) < self.unconditional else text
                 for text in data_dict['texts']]

        text_inputs = self.prepare_gen_prompts(texts)
        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

        max_length = self.max_length + self.num_queries
        inputs_embeds = inputs['inputs_embeds'][:, -max_length:]
        attention_mask = inputs['attention_mask'][:, -max_length:]
        position_ids = inputs['position_ids'][:, -max_length:]

        output = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True)

        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        pooled_prompt_embeds, prompt_embeds = self.llm2dit(hidden_states)

        loss_diff = self.diff_loss(model_input=image_latents,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   prompt_embeds=prompt_embeds)

        return loss_diff
    
    '''text-to-image generation (single-view) with camera map'''
    def cam2image_loss(self, data_dict):
        pixel_values = [p.to(dtype=self.dtype, device=self.device) for p in data_dict['pixel_values']]
        image_latents = [self.pixels_to_latents(p[None])[0] for p in pixel_values]
        b = len(image_latents)
        # camera map as condition for the diffusion model
        cam_values = [[img.to(dtype=self.dtype, device=self.device) for img in ref_images]
                            for ref_images in data_dict['cam_values']]
        cam_latents = [[self.pixels_to_latents(img[None])[0] for img in ref_images]
                            for ref_images in cam_values]

        texts = ['' if random.uniform(0, 1) < self.unconditional else text
                for text in data_dict['texts']]

        text_inputs = self.prepare_gen_prompts(texts)
        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

        max_length = self.max_length + self.num_queries
        inputs_embeds = inputs['inputs_embeds'][:, -max_length:]
        attention_mask = inputs['attention_mask'][:, -max_length:]
        position_ids = inputs['position_ids'][:, -max_length:]

        output = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True)

        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        pooled_prompt_embeds, prompt_embeds = self.llm2dit(hidden_states)

        loss_diff = self.diff_loss(model_input=image_latents,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   prompt_embeds=prompt_embeds,
                                   cond_input=cam_latents)
        
        return loss_diff
    
    '''image-to-image (cross-view) generation'''
    def image2image_loss(self, data_dict):
        # condition for the diffusion model (concat the camera map and the initial view)
        cam_values = [[img.to(dtype=self.dtype, device=self.device) for img in ref_images]
                            for ref_images in data_dict['cam_values']]
        cam_latents = [[self.pixels_to_latents(img[None])[0] for img in ref_images]
                            for ref_images in cam_values]
        pixel_values_init = [[img.to(dtype=self.dtype, device=self.device) for img in ref_images]
                            for ref_images in data_dict['pixel_values_init']]
        image_latents_init = [[self.pixels_to_latents(img[None])[0] for img in ref_images]
                            for ref_images in pixel_values_init]
        mix_latents = [cam + img for cam, img in zip(cam_latents, image_latents_init)]
        
        # condition embedding for querying the LLM (only initial view)
        num_refs = [len(ref_images) for ref_images in pixel_values_init]
        image_embeds = self.extract_visual_features(
            torch.stack([pad_an_image_tensor(img) for ref_images in pixel_values_init for img in ref_images]))

        image_embeds = self.projector(image_embeds)
        ref_lens = [len(x) for x in image_embeds]
        text_inputs = self.prepare_gen_prompts(data_dict['texts'], data_type='image2image', 
                                                    num_refs=num_refs, ref_lens=ref_lens)
        
        # input for the diffusion model
        pixel_values = [p.to(dtype=self.dtype, device=self.device) for p in data_dict['pixel_values']]
        image_latents = [self.pixels_to_latents(p[None])[0] for p in pixel_values]

        # querying the LLM
        b = len(image_latents)
        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)
        inputs = self.prepare_forward_input(query_embeds=hidden_states, image_embeds=image_embeds, **text_inputs)

        max_length = self.max_length + max(num_refs) * max(ref_lens) + self.num_queries
        inputs_embeds = inputs['inputs_embeds'][:, -max_length:]
        attention_mask = inputs['attention_mask'][:, -max_length:]
        position_ids = inputs['position_ids'][:, -max_length:]

        output = self.llm.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          position_ids=position_ids,
                          return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        pooled_prompt_embeds, prompt_embeds = self.llm2dit(hidden_states)
        loss_diff = self.diff_loss(model_input=image_latents,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   prompt_embeds=prompt_embeds,
                                   cond_input=mix_latents)
        
        return loss_diff
    
    '''image-to-text(camera) understanding, mixed base, thinking, and instruction tuning'''
    def image2text_loss(self, data_dict):
        pixel_values = [pad_an_image_tensor(img) for img in data_dict['pixel_values']]
        pixel_values = torch.stack(pixel_values).to(dtype=self.dtype, device=self.device)
        image_embeds = self.extract_visual_features(pixel_values)

        if not self.freeze_visual_encoder:
            image_embeds = _ScaleGradient.apply(image_embeds, self.visual_encoder_grad_scale)

        image_embeds = self.projector(image_embeds)
        text_inputs = self.prepare_und_prompts(conversations=data_dict['conversations'],
                                               data_type='image2text',
                                               image_lengths=image_embeds.shape[1])

        labels, input_ids, attention_mask, position_ids = \
            text_inputs['labels'], text_inputs['input_ids'], text_inputs['attention_mask'], text_inputs['position_ids']


        inputs_embeds = torch.zeros(*input_ids.shape, self.llm.config.hidden_size,
                                    device=self.device, dtype=self.dtype)
        inputs_embeds[input_ids == INPUT_IMAGE_TOKEN_INDEX] = image_embeds.flatten(0, 1)
        inputs_embeds[input_ids != INPUT_IMAGE_TOKEN_INDEX] = \
            self.llm.get_input_embeddings()(input_ids[input_ids != INPUT_IMAGE_TOKEN_INDEX])

        max_length = self.max_length + image_embeds.shape[1]
        inputs_embeds = inputs_embeds[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
        position_ids = position_ids[:, -max_length:]
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
        loss = F.cross_entropy(input=logits, target=labels)

        return loss
    
    '''text-to-text understanding, offering the enhanced caption for the generation'''
    def text2text_loss(self, data_dict):
        text_inputs = self.prepare_und_prompts(conversations=data_dict['conversations'], data_type='text2text')
        labels, input_ids, attention_mask, position_ids = \
            text_inputs['labels'], text_inputs['input_ids'], text_inputs['attention_mask'], text_inputs['position_ids']

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        max_length = self.max_length
        inputs_embeds = inputs_embeds[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
        position_ids = position_ids[:, -max_length:]
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
        loss = F.cross_entropy(input=logits, target=labels)

        return loss
    
    '''distribute different losses for each task'''
    def compute_loss(self, data_dict):
        loss_fn_map = {
            'text2image': self.text2image_loss,
            'cam2image': self.cam2image_loss,
            'image2text': self.image2text_loss,
            'text2text': self.text2text_loss,
            'image2image': self.image2image_loss,
            'image2text_cross_view': self.image2text_loss,
        }

        losses = {}
        for data_type, batch_data in data_dict.items():
            if data_type not in loss_fn_map:
                raise ValueError(f"Unsupported data_type: {data_type}")
            loss_fn = loss_fn_map[data_type]
            loss = loss_fn(batch_data)
            losses[f'loss_{data_type}'] = loss
        return losses

    @torch.no_grad()
    def generate(self,
                 prompt,
                 cfg_prompt,
                 cam_values=None,
                 pixel_values_init=None,
                 cfg_scale=4.5,
                 num_steps=50,
                 generator=None,
                 height=512,
                 width=512,
                 max_new_tokens=512,
                 reasoning=False,
                 prompt_reasoning=None,
                 progress_bar=True):
        assert len(prompt) == len(cfg_prompt)
        b = len(prompt)
        output_reasoning = [''] * b
        
        if reasoning:
            # enrich the prompt if required reasoning generation
            assert prompt_reasoning is not None, \
                "prompt_reasoning must be provided for reasoning generation"
            if isinstance(prompt_reasoning, str):
                prompt_reasoning = [prompt_reasoning]
            if isinstance(prompt, str):
                prompt = [prompt]

            conversations = [[{'input': f"{p1} {p2}",}] 
                                    for p1, p2 in zip(prompt_reasoning, prompt)]

            text_inputs = self.prepare_und_prompts(
                conversations=conversations, data_type="text2text", input_ids_with_output=False)
            input_ids, attention_mask, position_ids = \
                text_inputs['input_ids'], text_inputs['attention_mask'], text_inputs['position_ids']

            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            past_key_values = DynamicCache.from_legacy_cache()

            output_ids = []
            for _ in tqdm(range(max_new_tokens), disable=not progress_bar):
                output = self.llm.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True)
                logits = self.llm.get_output_embeddings()(output.last_hidden_state[:, -1:])
                input_ids = torch.argmax(logits, dim=-1)   # b 1
                if len(output_ids) > 0:
                    input_ids = torch.where(output_ids[-1] == self.tokenizer.eos_token_id,
                                            output_ids[-1], input_ids)
                output_ids.append(input_ids)

                if (input_ids == self.tokenizer.eos_token_id).all():
                    break

                inputs_embeds =  self.llm.get_input_embeddings()(input_ids)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, 1)], dim=1)
                position_ids = torch.max(position_ids, dim=1, keepdim=True).values + 1
                past_key_values = output.past_key_values

            output_ids = torch.cat(output_ids, dim=1)
            output_reasoning = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            prompt = [f"{p} {o}" for p, o in zip(prompt, output_reasoning)]
        
        if cam_values is not None:
            # for the generation with the camera map
            cam_values = [[img.to(dtype=self.dtype, device=self.device) for img in ref_images]
                                for ref_images in cam_values]
            cond_latents = [[self.pixels_to_latents(img[None])[0] for img in ref_images]
                                for ref_images in cam_values]
            text_inputs = self.prepare_gen_prompts(prompt + cfg_prompt)
            if pixel_values_init is not None:
                # for the generation with the camera map and initial view (cross-view generation)
                num_refs = [len(ref_images) for ref_images in pixel_values_init]
                pixel_values_init = [[img.to(dtype=self.dtype, device=self.device) for img in ref_images]
                                    for ref_images in pixel_values_init]
                image_embeds = self.extract_visual_features(
                    torch.stack([pad_an_image_tensor(img) for ref_images in pixel_values_init for img in ref_images]))
                image_embeds = self.projector(image_embeds)

                ref_lens = [len(x) for x in image_embeds]
                text_inputs = self.prepare_gen_prompts(prompt + cfg_prompt, data_type='image2image', num_refs=num_refs*2, ref_lens=ref_lens*2)
                text_inputs.update(image_embeds=torch.cat([image_embeds]*2))
                
                cond_latents_init = [[self.pixels_to_latents(img[None])[0] for img in ref_imgs]
                                for ref_imgs in pixel_values_init]
                cond_latents = [cam + img for cam, img in zip(cond_latents, cond_latents_init)]
            
            cond_latents = cond_latents * 2
        else:
            # for the text2image generation
            text_inputs = self.prepare_gen_prompts(prompt + cfg_prompt)
            cond_latents = None

        hidden_states = self.meta_queries[None].expand(2*b, self.num_queries, -1)
        inputs = self.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

        output = self.llm.model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        pooled_prompt_embeds, prompt_embeds = self.llm2dit(hidden_states)

        pipeline = StableDiffusion3Pipeline(
            transformer=self.transformer,
            scheduler=self.test_scheduler,
            vae=self.vae,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            text_encoder_3=None,
            tokenizer_3=None,
        )

        pipeline.set_progress_bar_config(disable=not progress_bar)

        samples = pipeline(
            height=height,
            width=width,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            prompt_embeds=prompt_embeds[:b],
            pooled_prompt_embeds=pooled_prompt_embeds[:b],
            negative_prompt_embeds=prompt_embeds[b:],
            negative_pooled_prompt_embeds=pooled_prompt_embeds[b:],
            generator=generator,
            output_type='latent',
            cond_latents=cond_latents
        ).images.to(self.dtype)

        return self.latents_to_pixels(samples), output_reasoning
    
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

        conversations = [[{'input': f"{DEFAULT_IMAGE_TOKEN}\n{p}",}] for p in prompt]

        text_inputs = self.prepare_und_prompts(conversations=conversations, image_lengths=image_embeds.shape[1], 
                                                input_ids_with_output=False)

        input_ids, attention_mask, position_ids = \
            text_inputs['input_ids'], text_inputs['attention_mask'], text_inputs['position_ids']

        inputs_embeds = torch.zeros(*input_ids.shape, self.llm.config.hidden_size,
                                    device=self.device, dtype=self.dtype)
        inputs_embeds[input_ids == INPUT_IMAGE_TOKEN_INDEX] = image_embeds.flatten(0, 1)
        inputs_embeds[input_ids != INPUT_IMAGE_TOKEN_INDEX] = \
            self.llm.get_input_embeddings()(input_ids[input_ids != INPUT_IMAGE_TOKEN_INDEX])

        past_key_values = DynamicCache.from_legacy_cache()

        output_ids = []

        for _ in tqdm(range(max_new_tokens), disable=not progress_bar):
            output = self.llm.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True)
            logits = self.llm.get_output_embeddings()(output.last_hidden_state[:, -1:])
            input_ids = torch.argmax(logits, dim=-1)   # b 1
            if len(output_ids) > 0:
                input_ids = torch.where(output_ids[-1] == self.tokenizer.eos_token_id,
                                        output_ids[-1], input_ids)
            output_ids.append(input_ids)

            if (input_ids == self.tokenizer.eos_token_id).all():
                break

            inputs_embeds =  self.llm.get_input_embeddings()(input_ids)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(bsz, 1)], dim=1)
            position_ids = torch.max(position_ids, dim=1, keepdim=True).values + 1
            past_key_values = output.past_key_values

        output_ids = torch.cat(output_ids, dim=1)
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return output_text