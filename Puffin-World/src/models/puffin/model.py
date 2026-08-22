import random
import torch
import math
import numpy as np
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
from diffusers.utils.torch_utils import randn_tensor

from src.models.connector import ConnectorConfig, ConnectorEncoder
from src.models.stable_diffusion3.pipeline_stable_diffusion_3_dynamic import StableDiffusion3Pipeline, calculate_shift
from src.datasets.utils import encode_fn, QUERY_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, INPUT_IMAGE_TOKEN_INDEX
from src.models.puffin.modules import CameraConditionEncoder as CameraConditionEncoder

from scripts.camera.utils.text import parse_camera_params
from scripts.camera.geometry.camera import SimpleRadial
from scripts.camera.geometry.gravity import Gravity
from scripts.camera.geometry.perspective_fields import get_perspective_field
from scripts.camera.utils.conversions import fov2focal
from scripts.camera.visualization.visualize_batch import make_perspective_figures
from src.dust3r.datasets.base.base_multiview_dataset import visionbanana_to_depth
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        nn.Linear(projector_dim, z_dim),)


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
                 freeze_projector=False,
                 freeze_llm=True,
                 visual_encoder_grad_scale=0.1,
                 fold_size=2,
                 unconditional=0.1,
                 unconditional_cross_view=0.1,
                 uncond_pf=0.0,
                 pretrained_pth=None,
                 use_activation_checkpointing=False,
                 initial_view_num=1,
                 max_view_num=8,
                 logit_mean=0.0,
                 logit_std=1.0,
                 inner_dim=1536,
                 ray_downsampled=False,
                 physical_propagation=False,
                 load_gt_camera_params=False,
                 geometry_state=False,
                 max_shift_override=None,
                 debug_save_dir=None,
                 cam_inject_multi=False,
                 cam_inject_layers=None,
                 depth_loss_split=False,
                 depth_loss_warmup_iters=0,
                 depth_loss_weight_max=1.0,
                 rgb_depth_attn_isolation=False,
                 depth_attn_open_iters=0,
                 depth_modality_embed=False,
                 depth_attn_closed_infer=False,
                 rgb_blind_to_depth=False,
                 *args, **kwargs):
        super().__init__()

        # basic settings
        self.max_length = max_length
        self.fold_size = fold_size
        self.prompt_template = prompt_template
        self.unconditional = unconditional
        self.unconditional_cross_view = unconditional_cross_view
        self.uncond_pf = uncond_pf
        self.initial_view_num = initial_view_num
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.max_view_num = max_view_num
        self.inner_dim = inner_dim
        # === Physical propagation (perspective-field source), unified knob ===
        #   'offline': per-view PF from PRECOMPUTED VLM annotations (the
        #              dataloader's 'gt_cam_params', annotated offline by the
        #              Puffin understanding branch). Conceptually still physical
        #              propagation -- just run offline to save training time.
        #   'online' : the VLM estimates frame-0 (roll/pitch/vfov/k1) at
        #              runtime and propagates it to all views via the relative
        #              cam_pose.
        #   'off'    : constant default PF (the channel carries no per-view info).
        if isinstance(physical_propagation, bool):
            physical_propagation = 'online' if physical_propagation else 'off'
        #   'offline_prop': eval-oriented hybrid — the GLOBAL first frame's GT
        #              annotation (gt_cam_params[b][0], or the relayed
        #              pp_anchor_params for chunks > 0) is the ONLY annotation
        #              consumed; every other view's PF comes from
        #              _physical_propagation via the relative poses, exactly
        #              like 'online' but without running the VLM.
        if physical_propagation not in ('off', 'offline', 'online',
                                        'offline_prop'):
            raise ValueError(
                f"physical_propagation must be 'off', 'offline', 'online' or "
                f"'offline_prop' (or a legacy bool), got {physical_propagation!r}")
        if load_gt_camera_params:
            if physical_propagation != 'off':
                raise ValueError(
                    "load_gt_camera_params (deprecated) cannot be combined "
                    f"with physical_propagation={physical_propagation!r}; set "
                    "physical_propagation='offline' alone.")
            print_log("[deprecated] load_gt_camera_params=True -> use "
                      "physical_propagation='offline' instead.")
            physical_propagation = 'offline'
        self.physical_propagation = physical_propagation
        self.max_shift_override = max_shift_override
        self.debug_save_dir = debug_save_dir
        
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
        
        # Condition per view: ray map (6) + perspective field (3) +
        # role mask (4: [is_target, is_init, is_i2i, is_depth]). The fused map
        # is added to the VAE latent.
        self.cond_fuser = CameraConditionEncoder(
            in_channels=6 + 3 + 4,
            out_channels=self.transformer.config.in_channels,
            ray_downsampled=ray_downsampled,
        )
        self.ray_downsampled = ray_downsampled

        # networks and training initialization
        if freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
        self.freeze_visual_encoder = freeze_visual_encoder
        if freeze_projector:
            self.projector.requires_grad_(False)
        self.freeze_projector = freeze_projector
        if freeze_llm:
            self.llm.requires_grad_(False)
        self.freeze_llm = freeze_llm
        if freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = freeze_transformer

        self.cam_inject_multi = cam_inject_multi
        self.transformer.cam_inject_multi = cam_inject_multi
        if cam_inject_multi:
            n_blocks = len(self.transformer.transformer_blocks)
            if cam_inject_layers is None:
                cam_inject_layers = [n_blocks // 4, n_blocks // 2, (3 * n_blocks) // 4]
            cam_inject_layers = sorted({int(l) for l in cam_inject_layers if 0 <= int(l) < n_blocks})
            hidden = self.transformer.pos_embed.proj.out_channels
            projs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in cam_inject_layers])
            for lin in projs:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.transformer.cam_inject_projs = projs
            self.transformer.cam_inject_layers = cam_inject_layers
            print_log(f'[cam_inject_multi] re-injecting cam at blocks '
                      f'{cam_inject_layers} (hidden={hidden}) of {n_blocks}')

        self.visual_encoder_grad_scale = visual_encoder_grad_scale
        self.train_scheduler = BUILDER.build(train_scheduler)
        self.test_scheduler = BUILDER.build(test_scheduler)

        self.use_activation_checkpointing = use_activation_checkpointing
        if use_activation_checkpointing:
            self.llm.enable_input_require_grads()
            self.gradient_checkpointing_enable()

        # RGB-preserving depth transition
        self.depth_loss_split = depth_loss_split
        self.depth_loss_warmup_iters = depth_loss_warmup_iters
        self.depth_loss_weight_max = depth_loss_weight_max
        self.rgb_blind_to_depth = rgb_blind_to_depth
        if rgb_blind_to_depth:
            rgb_depth_attn_isolation = True
            depth_attn_open_iters = -1
            depth_attn_closed_infer = True
        self.rgb_depth_attn_isolation = rgb_depth_attn_isolation
        self.depth_attn_open_iters = depth_attn_open_iters
        self.depth_attn_closed_infer = depth_attn_closed_infer
        self.use_depth_modality_embed = depth_modality_embed
        self.depth_loss_weight = 1.0
        self.transformer.rgb_depth_attn_isolation = rgb_depth_attn_isolation
        self.transformer.depth_attn_closed = False
        if depth_modality_embed:
            hidden = self.transformer.pos_embed.proj.out_channels
            self.transformer.depth_modality_embed = nn.Parameter(torch.zeros(hidden))
            print_log(f'[depth_modality_embed] added zero-init depth modality '
                      f'embedding (hidden={hidden})')

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')
        self.geometry_state = geometry_state
            
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
    
    def init_weights(self):
            pass
        
    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def extract_visual_features(self, pixel_values):
        pixel_values = (pixel_values + 1.0) / 2     # [0, 1]
        height, width = pixel_values.shape[-2:]
        with torch.autocast('cuda', dtype=torch.bfloat16):
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
    def prepare_gen_prompts(self, texts, data_type='cam2image', num_refs=None, ref_lens=None, gen_type='GENERATION_CROSS'):
        if data_type == 'cam2image':
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
        if self.freeze_visual_encoder:
            self.visual_encoder.train(mode=False)
        if self.freeze_projector:
            self.projector.train(mode=False)
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

    
    '''diffusion loss for text/cam-to-image generation (single-view)'''
    def diff_loss(
        self, 
        model_input, 
        cam_input,
        pooled_prompt_embeds, 
        prompt_embeds,
        mask,
    ):
        
        # view ids for view-axis RoPE (single view in T2I).
        B = len(model_input)
        T = 1
        _, H_lat, W_lat = model_input[0][0].shape
        view_ids = torch.arange(T, device=self.device)

        # sample noise and timesteps for flow matching
        noise = [[torch.randn_like(x) for x in imgs] for imgs in model_input]
        u = torch.normal(mean=self.logit_mean, 
                         std=self.logit_std, 
                         size=(B,),
                         device=self.device, 
                         dtype=self.dtype)
        sigmas = torch.nn.functional.sigmoid(u).bfloat16()
        
        image_seq_len = T * (H_lat // self.transformer.config.patch_size) * (
            W_lat // self.transformer.config.patch_size
        )
        
        mu = calculate_shift(
            image_seq_len,
            self.train_scheduler.config.get("base_image_seq_len", 256),
            self.train_scheduler.config.get("max_image_seq_len", 8192),
            self.train_scheduler.config.get("base_shift", 0.5),
            self.train_scheduler.config.get("max_shift", 0.9),
        )

        if self.train_scheduler.config.time_shift_type == "exponential":
            shift = math.exp(mu)
        elif self.train_scheduler.config.time_shift_type == "linear":
            shift = mu
        else:
            shift = 1.0

        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * self.train_scheduler.num_train_timesteps
        noisy_latents_tgt = [
            [(1.0 - sigma) * x_0 + sigma * eps for x_0, eps in zip(seq_imgs, seq_noise)]
                for sigma, seq_imgs, seq_noise in zip(sigmas, model_input, noise)
            ]

        # fuse camera + mask per batch item
        input_tgt_cond = []
        for b in range(B):
            cam_b = torch.stack(cam_input[b], dim=0)            # [T, C_cam, H_b, W_b]
            mask_b = mask[b]                                     # [T, 4, H_b, W_b]
            fused_b = self.cond_fuser(cam_b, mask_b)            # [T, C_lat, H_lat_b, W_lat_b]
            input_tgt_cond.append([fused_b[t] for t in range(T)])
        
        model_pred = self.transformer(
            hidden_states=noisy_latents_tgt,
            cond_hidden_states_cam_tgt=input_tgt_cond,
            cond_view_ids_tgt=view_ids.tolist(),
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            return_dict=False,
        )[0]

        weighting = compute_loss_weighting_for_sd3(weighting_scheme='none', sigmas=sigmas)
        target = [[x - y for x, y in zip(xs, ys)] for xs, ys in zip(noise, model_input)]
        loss = [(x.float() * (y.float() - z.float()) ** 2).mean()
                for x, ys, zs in zip(weighting, model_pred, target)
                for y, z in zip(ys, zs)]
        loss = sum(loss) / len(loss)

        return loss
    
    
    '''diffusion loss for image-to-image generation (multi-view)'''
    def multi_view_diff_loss_scheduler(
        self,
        model_input,           # list[B][T] of [C_lat,H_lat,W_lat] image latent
        cam_input,             # list[B][T] of [C_cam,H_lat,W_lat] camera latents
        pooled_prompt_embeds,  # [B,D_pooled]
        prompt_embeds,         # [B,S,D_txt]
        mask,                  # [B,T,4,H_lat,W_lat]
        initial_view_num,
        geometry_flag=False,
    ):
        # distribute the initial views and target views
        B = len(model_input)
        T = len(model_input[0])

        T_tgt = T - initial_view_num
        C_lat, H_lat, W_lat = model_input[0][0].shape
        C_cam, H_cam, W_cam = cam_input[0][0].shape
        image_latents_init = [views[: initial_view_num] for views in model_input]
        image_latents_tgt  = [views[initial_view_num:] for views in model_input]
        cam_latents_init   = [views[: initial_view_num] for views in cam_input]
        cam_latents_tgt    = [views[initial_view_num:] for views in cam_input]
        mask_init = mask[:, :initial_view_num]
        mask_tgt = mask[:, initial_view_num:]

        # view ids for view-axis RoPE
        # When geometry_flag, T = 2 * T_img: image at view k and depth at
        # view k share the same view_id k.
        if geometry_flag:
            base_T = T // 2
            img_ids = torch.arange(base_T, device=self.device)
            view_ids = torch.cat([img_ids, img_ids])
        else:
            view_ids = torch.arange(T, device=self.device)

        # sample noises for the target views with logit_norm scheduler
        noise = [[torch.randn_like(x) for x in tgt_imgs] for tgt_imgs in image_latents_tgt]
        u = torch.normal(mean=self.logit_mean, 
                         std=self.logit_std, 
                         size=(B,),
                         device=self.device, 
                         dtype=self.dtype)
        sigmas = torch.nn.functional.sigmoid(u).bfloat16()
        
        image_seq_len = T_tgt * (H_lat // self.transformer.config.patch_size) * (
            W_lat // self.transformer.config.patch_size
        )
        
        # multi-view path requests a stronger shift than the single-view path
        # to bias training towards higher-noise timesteps
        mu = calculate_shift(
            image_seq_len,
            self.train_scheduler.config.get("base_image_seq_len", 256),
            self.train_scheduler.config.get("max_image_seq_len", 8192),
            self.train_scheduler.config.get("base_shift", 0.5),
            self.max_shift_override
            if self.max_shift_override is not None
            else self.train_scheduler.config.get("max_shift", 0.9),
        )

        if self.train_scheduler.config.time_shift_type == "exponential":
            shift = math.exp(mu)
        elif self.train_scheduler.config.time_shift_type == "linear":
            shift = mu
        else:
            shift = 1.0

        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * self.train_scheduler.num_train_timesteps
        noisy_latents_tgt = [
            [(1.0 - sigma) * x_0 + sigma * eps for x_0, eps in zip(seq_imgs, seq_noise)]
                for sigma, seq_imgs, seq_noise in zip(sigmas, image_latents_tgt, noise)
            ]

        # convert list-of-list structure into 5D tensors [B,T,C,H,W]
        cam_latents_init_tensor = torch.stack(
            [torch.stack(cam_latents_init[b], dim=0) for b in range(B)],
            dim=0
        )
        cam_latents_tgt_tensor = torch.stack(
            [torch.stack(cam_latents_tgt[b], dim=0) for b in range(B)],
            dim=0
        )
        
        # fuse camera + mask for both initial views and target views.
        cam_init_flat  = cam_latents_init_tensor.reshape(B*initial_view_num, -1, H_cam, W_cam)
        mask_init_flat = mask_init.reshape(B*initial_view_num, -1, H_cam, W_cam)
        fused_init_flat = self.cond_fuser(cam_init_flat, mask_init_flat)
        fused_init_flat = fused_init_flat.view(B, initial_view_num, C_lat, H_lat, W_lat)
        input_init_cond = [[fused_init_flat[b, t] for t in range(initial_view_num)]
                           for b in range(B)]

        T_tgt = T - initial_view_num
        cam_tgt_flat  = cam_latents_tgt_tensor.reshape(B*T_tgt, -1, H_cam, W_cam)
        mask_tgt_flat = mask_tgt.reshape(B*T_tgt, -1, H_cam, W_cam)
        fused_tgt_flat = self.cond_fuser(cam_tgt_flat, mask_tgt_flat)
        fused_tgt_flat = fused_tgt_flat.view(B, T_tgt, C_lat, H_lat, W_lat)
        input_tgt_cond = [[fused_tgt_flat[b, t] for t in range(T_tgt)]
                          for b in range(B)]
        
        # per-view is_depth (for the zero-init depth modality embedding and the
        # annealed RGB<-depth attention isolation).
        if getattr(self, 'use_depth_modality_embed', False) or \
                getattr(self, 'rgb_depth_attn_isolation', False):
            if geometry_flag:
                base_T = T // 2
                is_depth_full = [0] * base_T + [1] * base_T
            else:
                is_depth_full = [0] * T
            cond_is_depth_init = is_depth_full[:initial_view_num]
            cond_is_depth_tgt = is_depth_full[initial_view_num:]
        else:
            cond_is_depth_init = cond_is_depth_tgt = None

        # prepare model input by combining initial views and noisy target views
        model_pred = self.transformer(
            hidden_states=noisy_latents_tgt,
            cond_hidden_states=image_latents_init,
            cond_hidden_states_cam_init=input_init_cond,
            cond_hidden_states_cam_tgt=input_tgt_cond,
            cond_view_ids_init=view_ids[:initial_view_num].tolist(),
            cond_view_ids_tgt=view_ids[initial_view_num:].tolist(),
            cond_is_depth_init=cond_is_depth_init,
            cond_is_depth_tgt=cond_is_depth_tgt,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            return_dict=False,
        )[0]

        # flow matching loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme='none', sigmas=sigmas)
        target = [[x - y for x, y in zip(xs, ys)] for xs, ys in zip(noise, image_latents_tgt)]

        if geometry_flag and getattr(self, 'depth_loss_split', False):
            # Split RGB vs depth target views and weight depth by the ramped
            # self.depth_loss_weight (0 at step 0 -> protects the shared head /
            # RGB color from depth gradients early).
            base_T = T // 2
            n_rgb_tgt = base_T - initial_view_num
            w_depth = float(getattr(self, 'depth_loss_weight', 1.0))
            rgb_terms, depth_terms = [], []
            for x, ys, zs in zip(weighting, model_pred, target):
                for j, (y, z) in enumerate(zip(ys, zs)):
                    term = (x.float() * (y.float() - z.float()) ** 2).mean()
                    (depth_terms if j >= n_rgb_tgt else rgb_terms).append(term)
            loss_rgb = sum(rgb_terms) / len(rgb_terms) if rgb_terms \
                else sum(depth_terms) * 0.0
            loss_depth = sum(depth_terms) / len(depth_terms) if depth_terms \
                else loss_rgb * 0.0
            loss = loss_rgb + w_depth * loss_depth
        else:
            loss = [(x.float() * (y.float() - z.float()) ** 2).mean()
                    for x, ys, zs in zip(weighting, model_pred, target)
                    for y, z in zip(ys, zs)]
            loss = sum(loss) / len(loss)

        return loss
    
    
    '''text-to-image generation (single-view) with camera map'''
    def cam2image_loss(self, data_dict):
        
        # 1. load images and camera maps to list
        pixel_values = [p.to(dtype=self.dtype, device=self.device) for p in data_dict['pixel_values']]
        image_latents = [self.pixels_to_latents(p[None])[0] for p in pixel_values]
        b = len(image_latents)
        cam_latents = [[img.to(dtype=self.dtype, device=self.device) for img in ref_images]
                            for ref_images in data_dict['cam_values']]

        # 2. build per-item 4-ch mask: [is_target, is_init, is_i2i, is_depth].
        # T2I has only target view(s) so channels are [1, 0, 0, 0]:
        # is_target=1; is_init=0; is_i2i=0 (distinguishes T2I from I2I targets);
        # is_depth=0 (T2I doesn't produces depth views currently).
        mask = []
        for i in range(b):
            _, H_i, W_i = pixel_values[i].shape
            _, H_lat_i, W_lat_i = image_latents[i].shape
            H_mask_i, W_mask_i = (H_lat_i, W_lat_i) if self.ray_downsampled else (H_i, W_i)
            mask_i = torch.zeros(
                1, 4, H_mask_i, W_mask_i,
                dtype=self.dtype,
                device=self.device,
            )
            mask_i[:, 0] = 1.0   # is_target (is_init=0, is_i2i=0, is_depth=0)
            mask.append(mask_i)
        
        # 3. text prompts for generation and meta-queries for DiT conditioning embeddings
        uncond = random.uniform(0, 1) < self.unconditional
        texts = ['' if uncond else text
                for text in data_dict['texts']]
        cam_latents = [
            ([torch.zeros_like(lat) for lat in views] if uncond else views)
                for views in cam_latents
        ]        

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

        # 4. single-view diffusion loss with fused (camera + mask) conditioning
        loss_diff = self.diff_loss(model_input=[[lat] for lat in image_latents],
                                   cam_input=cam_latents,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   prompt_embeds=prompt_embeds,
                                   mask=mask)
        
        return loss_diff

    
    '''image-to-image generation (multi-view) with camera map'''
    def image2image_loss(self, data_dict):
            
        # 1. load images, camera maps, depth (optional) to list
        pixel_values = [
            [img.to(dtype=self.dtype, device=self.device) for img in ref_images]
            for ref_images in data_dict['pixel_values_init']
        ]
        image_latents = [
            [self.pixels_to_latents(img[None])[0] for img in ref_images]
            for ref_images in pixel_values
        ]

        B = len(image_latents)
        T = len(image_latents[0])
        if self.initial_view_num == -1: 
            initial_view_num = np.random.randint(1, 4)
        else:
            initial_view_num = self.initial_view_num
        _, H, W = pixel_values[0][0].shape
        _, H_lat, W_lat = image_latents[0][0].shape
        cam_latents = [
            [img.to(dtype=self.dtype, device=self.device) for img in ref_images]
            for ref_images in data_dict['cam_values']
        ]
        
        # for depth in data_dict['depth_values'] is optional
        has_depth = self.geometry_state and data_dict.get('depth_values') and len(data_dict['depth_values']) > 0
        if has_depth:
            depth_latents = []
            for ref_depths in data_dict['depth_values']:
                depth_latents.append([
                    self.pixels_to_latents(depth.to(dtype=self.dtype, device=self.device)[None])[0]
                    for depth in ref_depths
                ])
        
        # 2. physical propagation: estimate absolute camera pose (to world) from VLM, then propagate to all views
        if self.physical_propagation == 'online':
            cam_latents, _, _ = self._physical_propagation(
                pixel_values, cam_latents, data_dict, B, T, H, W,
            )
            if self.debug_save_dir is not None:
                import os as _os
                rank = int(_os.environ.get('RANK', _os.environ.get('LOCAL_RANK', 0)))
                if rank == 0:
                    self._debug_visualize_camera_fields(
                        pixel_values, cam_latents, self.debug_save_dir, B, T,
                    )
        elif self.physical_propagation == 'offline':
            gt_cam_params = data_dict.get('gt_cam_params', None)
            assert gt_cam_params is not None, (
                "physical_propagation='offline' but data_dict has no "
                "'gt_cam_params'."
            )
            cam_latents = self._apply_gt_camera_params(
                cam_latents, gt_cam_params, B, T, H, W,
                cam_intrinsics=data_dict.get('cam_intrinsics', None),
            )
            if self.debug_save_dir is not None:
                import os as _os
                rank = int(_os.environ.get('RANK', _os.environ.get('LOCAL_RANK', 0)))
                if rank == 0:
                    self._debug_visualize_camera_fields(
                        pixel_values, cam_latents, self.debug_save_dir, B, T,
                    )
        else:
            cam_latents = self._append_default_perspective_field(
                cam_latents, B, T, H, W,
            )

        # 3. build 4-ch mask: [is_target, is_init, is_i2i, is_depth]
        H_mask, W_mask = (H_lat, W_lat) if self.ray_downsampled else (H, W)
        mask = torch.zeros(
            B, T, 4, H_mask, W_mask,
            dtype=self.dtype,
            device=self.device,
        )
        mask[:, :initial_view_num, 1] = 1.0   # init views: is_init
        mask[:, initial_view_num:, 0] = 1.0   # target views: is_target
        mask[:, :, 2] = 1.0                    # all views: is_i2i

        # if 3D clue is available, append 3D latents to image latents and expand cam latents and mask by 2x
        geometry_flag = False
        if has_depth:
            for b in range(B):
                image_latents[b].extend(depth_latents[b])
                cam_latents[b] = cam_latents[b] + list(cam_latents[b])

            del depth_latents
            depth_mask = mask.clone()
            depth_mask[:, :, 3] = 1.0          # depth views: is_depth
            mask = torch.cat([mask, depth_mask], dim=1)
            geometry_flag = True

        # 4. visual features of initial images from the VLM
        num_refs = [
            min(initial_view_num, len(ref_images))
            for ref_images in pixel_values
        ]

        vis_inputs = torch.stack([
            pad_an_image_tensor(img)
            for ref_images in pixel_values
            for img in ref_images[: initial_view_num]
        ])

        image_embeds = self.extract_visual_features(vis_inputs)
        del vis_inputs, pixel_values
        image_embeds = self.projector(image_embeds)
        ref_lens = [len(x) for x in image_embeds]

        assert sum(num_refs) == len(ref_lens), \
            f"sum(num_refs)={sum(num_refs)} != len(ref_lens)={len(ref_lens)}"

        # 5. text prompts for generation and meta-queries for DiT conditioning embeddings
        # Conditioning dropout (global, all image2image datasets), two modes:
        #   joint_uncond -> zero ray+PF+text (the CFG-null sample; matches the
        #     inference uncond branch).
        #   pf_drop       -> zero ONLY the PF channels (6:9), KEEP the ray map, so
        #     the model must read camera geometry/motion from ray (anti PF-
        #     dominance / fixes translation collapse)
        joint_uncond = random.uniform(0, 1) < self.unconditional
        pf_drop = (not joint_uncond) and self.physical_propagation != 'off' \
            and (random.uniform(0, 1) < self.uncond_pf)

        texts = ['' if joint_uncond else text for text in data_dict['texts']]
        if joint_uncond:
            cam_latents = [[torch.zeros_like(lat) for lat in views] for views in cam_latents]
        elif pf_drop:
            dropped = []
            for views in cam_latents:
                dropped_views = []
                for lat in views:
                    lat = lat.clone()
                    lat[6:9] = 0.0    # zero perspective field (PF), keep ray map
                    dropped_views.append(lat)
                dropped.append(dropped_views)
            cam_latents = dropped
        
        text_inputs = self.prepare_gen_prompts(
            texts,
            data_type='image2image',
            num_refs=num_refs,
            ref_lens=ref_lens,
        )
        hidden_states = self.meta_queries[None].expand(B, self.num_queries, -1)
        inputs = self.prepare_forward_input(
            query_embeds=hidden_states,
            image_embeds=image_embeds,
            **text_inputs
        )

        max_length = self.max_length + max(num_refs) * max(ref_lens) + self.num_queries
        inputs_embeds = inputs['inputs_embeds'][:, -max_length:]
        attention_mask = inputs['attention_mask'][:, -max_length:]
        position_ids = inputs['position_ids'][:, -max_length:]

        output = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        del output, inputs_embeds, attention_mask, position_ids, inputs
        pooled_prompt_embeds, prompt_embeds = self.llm2dit(hidden_states)
        del hidden_states
        torch.cuda.empty_cache()

        # 6. multi-view diffusion loss with fused (initial view + camera + mask) conditioning
        loss_diff = self.multi_view_diff_loss_scheduler(
            model_input=image_latents,
            cam_input=cam_latents,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_embeds=prompt_embeds,
            mask=mask,
            initial_view_num=initial_view_num,
            geometry_flag=geometry_flag,
        )

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

        loss_diff = self.diff_loss(model_input=[[lat] for lat in image_latents],
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   prompt_embeds=prompt_embeds)

        return loss_diff
 
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
            'text2image': self.text2image_loss,
            'cam2image': self.cam2image_loss,
            'image2text': self.image2text_loss,
            'text2text': self.text2text_loss,
            'image2image': self.image2image_loss,
            'image2text_cross_view': self.image2text_loss,
        }
        known_types = set(loss_fn_map.keys())

        multi_task_keys = []
        if isinstance(batch_root, dict):
            multi_task_keys = [k for k in batch_root.keys() if k in known_types]

        if len(multi_task_keys) > 0:
            per_type_batches = {k: batch_root[k] for k in multi_task_keys}
        else:
            per_type_batches = {"image2image": batch_root}

        losses = {}
        for data_type, batch_data in per_type_batches.items():
            if data_type not in loss_fn_map:
                raise ValueError(f"Unsupported data_type: {data_type}")

            loss_fn = loss_fn_map[data_type]
            loss = loss_fn(batch_data)
            losses[f"loss_{data_type}"] = loss

        return losses

    @staticmethod
    def _parse_cam_pose_str(cam_pose_str):
        """Parse a numpy 4x4 matrix string back to numpy array."""
        import re as _re
        nums = [float(x) for x in _re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', cam_pose_str)]
        return np.array(nums, dtype=np.float32).reshape(4, 4)

    @staticmethod
    def _parse_cam_intrinsics_str(cam_intrinsics_str):
        """Parse a numpy 3x3 intrinsics matrix string back to numpy array."""
        import re as _re
        nums = [float(x) for x in _re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', cam_intrinsics_str)]
        return np.array(nums, dtype=np.float32).reshape(3, 3)

    @torch.no_grad()
    def _apply_gt_camera_params(self, cam_latents, gt_cam_params, B, T, H, W,
                                cam_intrinsics=None):
        """Build per-view perspective fields directly from GT camera parameters.

        gt_cam_params: list[B] of list[T] of strings, each string encoding
        "roll pitch vfov k1" in radians (k1 dimensionless).
        """
        for b in range(B):
            assert len(gt_cam_params[b]) >= T, (
                f"gt_cam_params[{b}] has {len(gt_cam_params[b])} entries, "
                f"expected at least {T}."
            )
            for t in range(T):
                tokens = gt_cam_params[b][t].strip().split()
                if len(tokens) == 4:
                    roll_t, pitch_t, vfov_t, k1_t = (float(x) for x in tokens)
                else:
                    # missing / unparsable caption for this view -> default PF
                    roll_t, pitch_t, vfov_t, k1_t = 0.0, 0.0, math.radians(90.0), 0.0

                if cam_intrinsics is not None:
                    # crop-consistent focal + principal point from the dataloader
                    K_t = self._parse_cam_intrinsics_str(cam_intrinsics[b][t])
                    f = float(K_t[1, 1])
                    px, py = float(K_t[0, 2]), float(K_t[1, 2])
                else:
                    f = fov2focal(torch.tensor(vfov_t), H)
                    px, py = W / 2.0, H / 2.0
                params = torch.tensor([W, H, f, f, px, py, k1_t, 0.0]).float()
                camera = SimpleRadial(params).float()
                camera = camera.scale(torch.Tensor([1, 1]))
                gravity_obj = Gravity.from_rp(
                    torch.tensor(roll_t).float(),
                    torch.tensor(pitch_t).float(),
                )

                up_field, lat_field = get_perspective_field(
                    camera, gravity_obj, use_up=True, use_latitude=True
                )
                pf = torch.cat([up_field[0], lat_field[0]], dim=0)
                pf = pf / (math.pi / 2)
                pf = pf.to(dtype=cam_latents[b][t].dtype,
                           device=cam_latents[b][t].device)
                cam_latents[b][t] = torch.cat([cam_latents[b][t], pf], dim=0)

                del camera, gravity_obj
        return cam_latents

    def _append_default_perspective_field(self, cam_latents, B, T, H, W):
        """Append a default perspective field (roll=0, pitch=0, vfov=90deg) to 6-ch ray maps."""
        f = fov2focal(torch.tensor(90.0), H)
        px, py = W / 2.0, H / 2.0
        params = torch.tensor([W, H, f, f, px, py, 0.0, 0.0]).float()
        camera = SimpleRadial(params).float()
        camera = camera.scale(torch.Tensor([1, 1]))
        gravity_obj = Gravity.from_rp(torch.tensor(0.0), torch.tensor(0.0))
        up_field, lat_field = get_perspective_field(
            camera, gravity_obj, use_up=True, use_latitude=True
        )
        pf = torch.cat([up_field[0], lat_field[0]], dim=0)  # [3, H, W]
        pf = pf / (math.pi / 2)
        for b in range(B):
            for t in range(T):
                pf_bt = pf.to(dtype=cam_latents[b][t].dtype, device=cam_latents[b][t].device)
                cam_latents[b][t] = torch.cat([cam_latents[b][t], pf_bt], dim=0)
        del camera, gravity_obj
        return cam_latents

    @torch.no_grad()
    def _physical_propagation(self, pixel_values, cam_latents, data_dict, B, T, H, W,
                              anchor_params=None):
        """Estimate absolute camera params for the first view via VLM,
        then propagate to all views using relative cam_pose, and concat
        the resulting perspective field to the existing ray_map.

        anchor_params: optional list[B] of "roll pitch vfov k1" strings for
        view 0. When given, the VLM is SKIPPED and these are propagated
        instead (chunked AR carries the chunk-0 estimate forward this way).

        Returns (cam_latents, output_texts, pp_cam_params):
        output_texts is None when the VLM was skipped; pp_cam_params is
        list[B][T] of "roll pitch vfov k1" strings — the propagated per-view
        absolute params (vfov / k1 constant from view 0), formatted like
        gt_cam_params so callers can feed the LAST view's entry back in as
        the next window's anchor.
        """

        # --- Step 1: absolute params of view 0 (caller-provided or VLM) ---
        output_texts = None
        abs_params = []  # list of (roll, pitch, fov, k1) per batch
        if anchor_params is not None:
            for b, s in enumerate(anchor_params):
                tokens = str(s).strip().split()
                if len(tokens) == 4:
                    roll, pitch, vfov, k1 = (float(x) for x in tokens)
                else:
                    print(f"[PhysicalProp] WARNING batch {b}: bad anchor_params "
                          f"{s!r}, using defaults.")
                    roll, pitch, vfov, k1 = 0.0, 0.0, 1.0, 0.0
                abs_params.append((roll, pitch, vfov, k1))
        else:
            prompt = (
                "Describe the image in detail. Then reason its spatial distribution "
                "and estimate its camera parameters (roll, pitch, field-of-view, and radial distortion)."
            )
            first_views = [pixel_values[b][0] for b in range(B)]
            was_training = self.training
            self.eval()
            output_texts = self.understand(
                prompt=[prompt] * B, pixel_values=first_views,
                max_new_tokens=180, progress_bar=False,
            )
            if was_training:
                self.train()
            torch.cuda.empty_cache()

            # Parse absolute camera params for the first view
            for b, text in enumerate(output_texts):
                try:
                    roll, pitch, vfov, k1 = parse_camera_params(text, mode='radial')
                except ValueError:
                    print(f"[PhysicalProp] WARNING batch {b}: failed to parse camera params, "
                              f"using defaults. VLM output: {text}")
                    roll, pitch, vfov, k1 = 0.0, 0.0, 1.0, 0.0
                abs_params.append((roll, pitch, vfov, k1))

        # --- Step 2: Propagate absolute (roll, pitch) to all views ---
        cam_pose_strs = data_dict.get('cam_pose', None)
        all_view_abs_params = []  # list[B] of list[T] of (roll, pitch)

        for b in range(B):
            roll_0, pitch_0, vfov_0, k1_0 = abs_params[b]

            if cam_pose_strs is not None:
                # Parse absolute c2w matrices (same arbitrary world frame)
                c2w_mats = [self._parse_cam_pose_str(s) for s in cam_pose_strs[b]]
                R_0 = c2w_mats[0][:3, :3]

                # Compute relative rotation from view 0 to each view t:
                #   R_rel_t = R_t^{-1} @ R_0 = R_t.T @ R_0
                R_rels = [c2w_mats[t][:3, :3].T @ R_0 for t in range(T)]

                # VLM-estimated gravity in camera 0 frame
                g_cam0 = Gravity.from_rp(
                    torch.tensor(roll_0), torch.tensor(pitch_0)
                ).vec3d.numpy()

                # Propagate: g_cam_t = R_rel_t @ g_cam0
                view_params = []
                for t in range(T):
                    g_cam_t = R_rels[t] @ g_cam0
                    grav_t = Gravity(torch.tensor(g_cam_t).float())
                    roll_t = grav_t.roll.item()
                    pitch_t = grav_t.pitch.item()
                    view_params.append((roll_t, pitch_t))
            else:
                # No cam_pose available, use the same params for all views
                view_params = [(roll_0, pitch_0)] * T

            all_view_abs_params.append(view_params)

        # --- Step 3: Build perspective field and normalize, then concat ---
        # Focal: PREFER the dataloader intrinsics (data_dict['cam_intrinsics'])
        # over the VLM-estimated vfov. The dataloader crops/rescales images and
        # returns intrinsics consistent with the PROCESSED image, whereas the
        # VLM vfov refers to whatever crop the VLM saw — using the dataloader
        # focal keeps the PF geometrically consistent with the training crop.
        cam_intr_strs = data_dict.get('cam_intrinsics', None)
        for b in range(B):
            _, vfov_0, k1_0 = abs_params[b][1], abs_params[b][2], abs_params[b][3]
            k2 = 0.0

            for t in range(T):
                roll_t, pitch_t = all_view_abs_params[b][t]

                if cam_intr_strs is not None:
                    # per-view post-crop intrinsics (fy + principal point)
                    K_t = self._parse_cam_intrinsics_str(cam_intr_strs[b][t])
                    f = float(K_t[1, 1])
                    px, py = float(K_t[0, 2]), float(K_t[1, 2])
                else:
                    # fallback: VLM-estimated vfov (legacy behavior)
                    f = fov2focal(torch.tensor(vfov_0), H)
                    px, py = W / 2.0, H / 2.0

                # Build camera and gravity objects
                params = torch.tensor([W, H, f, f, px, py, k1_0, k2]).float()
                camera = SimpleRadial(params).float()
                camera = camera.scale(torch.Tensor([1, 1]))
                gravity_obj = Gravity.from_rp(
                    torch.tensor(roll_t).float(),
                    torch.tensor(pitch_t).float(),
                )

                # Generate perspective field: up_field [1, 2, H, W], lat_field [1, 1, H, W]
                up_field, lat_field = get_perspective_field(
                    camera, gravity_obj, use_up=True, use_latitude=True
                )
                pf = torch.cat([up_field[0], lat_field[0]], dim=0)
                pf = pf / (math.pi / 2)

                # Concat with existing ray_map on channel dim
                pf = pf.to(dtype=cam_latents[b][t].dtype, device=cam_latents[b][t].device)
                cam_latents[b][t] = torch.cat([cam_latents[b][t], pf], dim=0)

                del camera, gravity_obj

        # Propagated per-view absolute params in gt_cam_params format, so a
        # chunked caller can carry the last view's entry into the next
        # window's anchor (one VLM estimate propagated across all chunks).
        pp_cam_params = [
            [f"{all_view_abs_params[b][t][0]} {all_view_abs_params[b][t][1]} "
             f"{abs_params[b][2]} {abs_params[b][3]}" for t in range(T)]
            for b in range(B)
        ]
        return cam_latents, output_texts, pp_cam_params

    @torch.no_grad()
    def _debug_visualize_camera_fields(self, pixel_values, cam_latents, save_dir, B, T):
        """Visualize propagated perspective fields for all views and save to disk.

        For each batch b and view t, builds a dict with:
          - image: the RGB image tensor [1, 3, H, W] in [0, 1]
          - up_field: [1, 2, H, W]  (un-normalized)
          - latitude_field: [1, 1, H, W]  (un-normalized)
        Then calls make_perspective_figures and saves the resulting figures.
        """
        import os as _os
        import hashlib as _hl
        _os.makedirs(save_dir, exist_ok=True)
        iter_hash = _hl.sha256(torch.randn(1).numpy().tobytes()).hexdigest()[:8]

        for b in range(B):
            for t in range(T):
                # cam_latents[b][t] is [C_ray + 3, H, W] where last 3 channels are the PF
                pf = cam_latents[b][t][-3:]  # [3, H, W], normalized by pi/2
                # un-normalize to radians for visualization
                pf = pf.float().cpu() * (math.pi / 2)
                up_field = pf[:2].unsqueeze(0)       # [1, 2, H, W]
                lat_field = pf[2:].unsqueeze(0)      # [1, 1, H, W]

                # image: [-1, 1] RGB -> [0, 1] RGB
                img = pixel_values[b][t].float().cpu()
                img = ((img + 1) / 2).clamp(0, 1).unsqueeze(0)   # [1, 3, H, W]

                single_batch = {
                    "image": img,
                    "up_field": up_field,
                    "latitude_field": lat_field,
                }
                figs = make_perspective_figures(single_batch, single_batch, n_pairs=1)
                for k, fig in figs.items():
                    suffix = "_up" if "up" in k else "_lat" if "lat" in k else f"_{k}"
                    out_path = _os.path.join(save_dir, f"b{b}_v{t}_camera_field{suffix}_{iter_hash}.png")
                    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

        print(f"[PhysicalProp] Camera field visualizations saved to: {save_dir}")

    @torch.no_grad()
    def generate(self,
                 prompt,
                 cfg_prompt,
                 cam_values=None,
                 cfg_scale=4.5,
                 num_steps=50,
                 generator=None,
                 height=512,
                 width=512,
                 progress_bar=True):
        
        assert len(prompt) == len(cfg_prompt)
        B = len(prompt)
        device = self.device
        dtype = self.dtype
        cam_values = [
            [img.to(dtype=dtype, device=device) for img in ref_images]
            for ref_images in cam_values
        ]

        # when ray_downsampled, cam latent is at latent resolution; otherwise at pixel resolution
        H_lat = height // self.vae_scale_factor if hasattr(self, 'vae_scale_factor') else height // 8
        W_lat = width // self.vae_scale_factor if hasattr(self, 'vae_scale_factor') else width // 8
        H_mask, W_mask = (H_lat, W_lat) if self.ray_downsampled else (height, width)

        latents_cam = []
        for b in range(B):
            cam_b = torch.stack(cam_values[b], dim=0)
            # 4-ch mask: T2I single target view -> [1, 0, 0, 0]
            # (is_target, is_init, is_i2i, is_depth)
            mask_b = torch.zeros(1, 4, H_mask, W_mask, device=device, dtype=dtype)
            mask_b[:, 0] = 1.0
            fused_b = self.cond_fuser(cam_b, mask_b)
            latents_cam.append([fused_b[0]])

        # view_ids for view-axis RoPE (single view -> view_id=0)
        view_ids = torch.arange(1, device=device)

        # initial Gaussian noise for the target view
        noise = []
        for b in range(B):
            noise.append([
                randn_tensor(
                    (latents_cam[b][0].shape[0], H_lat, W_lat),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            ])

        # text prompts (no init image, pure text+camera generation)
        text_inputs = self.prepare_gen_prompts(prompt+cfg_prompt)
        hidden_states = self.meta_queries[None].expand(2*B, self.num_queries, -1)
        inputs = self.prepare_forward_input(query_embeds=hidden_states, **text_inputs)
        output = self.llm.model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        pooled_prompt_embeds, prompt_embeds = self.llm2dit(hidden_states)

        # dynamic SD3 pipeline
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
            prompt_embeds=prompt_embeds[:B],
            pooled_prompt_embeds=pooled_prompt_embeds[:B],
            negative_prompt_embeds=prompt_embeds[B:],
            negative_pooled_prompt_embeds=pooled_prompt_embeds[B:],
            generator=generator,
            output_type='latent',
            cond_latents=None,
            cond_latents_cam=None,
            latents=noise,
            latents_cam=latents_cam,
            cond_view_ids_tgt=view_ids.tolist(),
        ).images

        sample_t = torch.stack([s[0] for s in samples]).to(dtype)
        return self.latents_to_pixels(sample_t)
    
    @torch.no_grad()
    def generate_multi_view(
        self,
        prompt,
        cfg_prompt,
        cam_values=None,          # list[B][T] of [C_cam, H_img, W_img]
        pixel_values_init=None,   # list[B][T] of [3, H_img, W_img]
        cam_pose=None,            # list[B][T] of cam_pose strings (for physical_propagation)
        cam_intrinsics=None,      # list[B][T] of 3x3 intrinsics strings; when given,
                                  # physical_propagation derives the PF focal /
                                  # principal point from these (crop-consistent)
                                  # instead of the VLM-estimated vfov.
        gt_cam_params=None,       # list[B][T] of "roll pitch vfov k1" strings (for physical_propagation='offline')
        pp_anchor_params=None,    # list[B] of "roll pitch vfov k1" strings: view-0
                                  # absolute params for physical_propagation='online',
                                  # SKIPPING the VLM (chunked AR carries the chunk-0
                                  # estimate forward through this).
        image_latents_init=None,  # list[B][T] of pre-encoded [C_lat, H_lat, W_lat] tensors;
                                  # when provided, VAE.encode is skipped (used by
                                  # chunked autoregressive inference to avoid the
                                  # VAE round-trip degradation at chunk boundaries).
        cfg_scale=4.5,
        num_steps=50,
        generator=None,
        height=512,
        width=512,
        K=None,
        progress_bar=True,
        return_latents=False,
        view_ids_override=None,  # list[T_img] of geometric view-RoPE ids for
                                 # non-temporally-ordered windows (anchor_k2)
    ):

        B = len(prompt)
        device = self.device
        dtype = self.dtype

        cam_latents_all = [
            [img.to(dtype=dtype, device=device) for img in ref_images]
            for ref_images in cam_values
        ]
        pixel_values_init = [
            [img.to(dtype=dtype, device=device) for img in ref_images]
            for ref_images in pixel_values_init
        ]

        # If the caller provides pre-encoded latents, skip the VAE.encode step
        # entirely (essential for chunked autoregressive inference — VAE
        # encode/decode round-trip introduces compounding artifacts otherwise)
        if image_latents_init is not None:
            assert len(image_latents_init) == B, (
                f"image_latents_init must be list[B={B}], got {len(image_latents_init)}"
            )
            image_latents_all = [
                [lat.to(dtype=dtype, device=device) for lat in ref_latents]
                for ref_latents in image_latents_init
            ]
        else:
            image_latents_all = [
                [self.pixels_to_latents(img[None])[0] for img in ref_images]
                for ref_images in pixel_values_init
            ]
        _, H, W = pixel_values_init[0][0].shape
        _, H_lat, W_lat = image_latents_all[0][0].shape

        # append a 3-channel perspective field to each cam_latent
        # - physical_propagation='online' : VLM-estimated PF (frame 0), propagated to all views.
        # - physical_propagation='offline': PF from precomputed per-view VLM annotations (gt_cam_params).
        # - physical_propagation='off'    : constant default PF placeholder.
        vlm_texts = None
        pp_cam_params = None
        if self.physical_propagation == 'online':
            T_all = len(pixel_values_init[0])
            cam_latents_all, vlm_texts, pp_cam_params = self._physical_propagation(
                pixel_values_init, cam_latents_all,
                {'cam_pose': cam_pose, 'cam_intrinsics': cam_intrinsics},
                B, T_all, H, W,
                anchor_params=pp_anchor_params,
            )
        elif self.physical_propagation == 'offline_prop':
            # GT-anchored propagation: consume ONLY the global first frame's
            # GT params (chunks > 0 receive the relayed pp_anchor_params
            # instead, so the anchor stays the trajectory's true frame 0);
            # all other views get their PF via the relative poses. No VLM
            # call, so generation-only checkpoints work.
            T_all = len(pixel_values_init[0])
            anchor = pp_anchor_params
            if anchor is None:
                anchor = [
                    (gt_cam_params[b][0] if gt_cam_params is not None else '')
                    for b in range(B)
                ]
            cam_latents_all, vlm_texts, pp_cam_params = self._physical_propagation(
                pixel_values_init, cam_latents_all,
                {'cam_pose': cam_pose, 'cam_intrinsics': cam_intrinsics},
                B, T_all, H, W,
                anchor_params=anchor,
            )
        elif self.physical_propagation == 'offline':
            T_all = len(pixel_values_init[0])
            assert gt_cam_params is not None, (
                "physical_propagation='offline' but generate_multi_view was "
                "called without gt_cam_params."
            )
            cam_latents_all = self._apply_gt_camera_params(
                cam_latents_all, gt_cam_params, B, T_all, H, W,
                cam_intrinsics=cam_intrinsics,
            )
        else:
            T_all = len(pixel_values_init[0])
            cam_latents_all = self._append_default_perspective_field(
                cam_latents_all, B, T_all, H, W,
            )

        T_img = len(pixel_values_init[0])
        assert T_img > K, "Total number of views T must be larger than initial_view_num."

        # when geometry_state is enabled, mirror the training doubling:
        # append a second block of T_img "depth" views after the image views
        geometry_flag = getattr(self, 'geometry_state', False)
        cam_latents_all_orig = cam_latents_all
        if geometry_flag:
            image_latents_all = [views + views for views in image_latents_all]
            cam_latents_all   = [views + views for views in cam_latents_all]
            T = 2 * T_img
        else:
            T = T_img

        # initial views (conditions): only the first K of the image block
        image_latents_init = [views[:K] for views in image_latents_all]        # list[B][K]
        cam_latents_init   = [views[:K] for views in cam_latents_all]          # list[B][K]
        
        # target views (to be generated): everything else (image tail + optional depth block)
        image_latents_tgt_clean = [views[K:] for views in image_latents_all]   # list[B][T-K]
        cam_latents_tgt         = [views[K:] for views in cam_latents_all]     # list[B][T-K]
        T_tgt = T - K
        T_tgt_img = T_img - K   # number of novel image views (no depth)

        # image and depth at the same view share view_id. view_ids_override
        # lets a NON-temporally-ordered window (e.g. --anchor_k2's
        # [anchor_k, anchor_k+1, intermediates]) keep GEOMETRIC view-RoPE
        # ids so the RoPE order matches the trajectory, as in training.
        if view_ids_override is not None:
            img_ids = torch.as_tensor(
                view_ids_override, device=self.device, dtype=torch.long)
            assert img_ids.numel() == T_img, (
                f"view_ids_override needs {T_img} ids, got {img_ids.numel()}")
            view_ids = (torch.cat([img_ids, img_ids])
                        if geometry_flag else img_ids)
        elif geometry_flag:
            img_ids = torch.arange(T_img, device=self.device)
            view_ids = torch.cat([img_ids, img_ids])
        else:
            view_ids = torch.arange(T, device=self.device)

        # 4-ch mask: [is_target, is_init, is_i2i, is_depth]
        H_mask, W_mask = (H_lat, W_lat) if self.ray_downsampled else (H, W)
        mask = torch.zeros(B, T, 4, H_mask, W_mask, device=device, dtype=dtype)
        mask[:, :K, 1] = 1.0    # init views: is_init
        mask[:, K:, 0] = 1.0    # target views: is_target
        mask[:, :, 2] = 1.0     # all views: is_i2i
        if geometry_flag:
            mask[:, T_img:, 3] = 1.0    # depth half: is_depth
        mask_init = mask[:, :K]   # [B, K, 4, H_lat, W_lat]
        mask_tgt  = mask[:, K:]   # [B, T-K, 4, H_lat, W_lat]

        # fuse [cam_latent, mask] per init view through cond_fuser
        cond_latents_cam = [
            [self.cond_fuser(cam_latents_init[b][t], mask_init[b, t]) for t in range(K)]
            for b in range(B)
        ]

        # each target view: sample diffusion noise + fuse cam condition
        latents, latents_cam = [], []
        for b in range(B):
            noise_views, cam_views = [], []
            for t in range(T_tgt):
                noise_views.append(randn_tensor(
                    image_latents_tgt_clean[b][t].shape,
                    generator=generator, device=device, dtype=dtype,
                ))
                cam_views.append(self.cond_fuser(cam_latents_tgt[b][t], mask_tgt[b, t]))
            latents.append(noise_views)
            latents_cam.append(cam_views)

        num_refs = [
            min(K, len(ref_images))
            for ref_images in pixel_values_init
        ]

        vis_inputs = torch.stack([
            pad_an_image_tensor(img)
            for ref_images in pixel_values_init
            for img in ref_images[: K]
        ])

        image_embeds = self.extract_visual_features(vis_inputs)
        image_embeds = self.projector(image_embeds)
        ref_lens = [len(x) for x in image_embeds]  
        text_inputs = self.prepare_gen_prompts(prompt + cfg_prompt, 
                                               data_type='image2image', 
                                               num_refs=num_refs*2, 
                                               ref_lens=ref_lens*2)
        text_inputs.update(image_embeds=torch.cat([image_embeds]*2))


        hidden_states = self.meta_queries[None].expand(2*B, self.num_queries, -1)
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

        # RGB<-depth attention isolation at INFERENCE
        self.transformer.depth_attn_closed = bool(
            getattr(self, 'depth_attn_closed_infer', False))

        if getattr(self, 'use_depth_modality_embed', False) or \
                getattr(self, 'rgb_depth_attn_isolation', False):
            is_depth_full = ([0] * T_img + [1] * T_img) if geometry_flag else [0] * T
            cond_is_depth_init = is_depth_full[:K]
            cond_is_depth_tgt = is_depth_full[K:]
        else:
            cond_is_depth_init = cond_is_depth_tgt = None

        samples = pipeline(
            height=height,
            width=width,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            prompt_embeds=prompt_embeds[:B],
            pooled_prompt_embeds=pooled_prompt_embeds[:B],
            negative_prompt_embeds=prompt_embeds[B:],
            negative_pooled_prompt_embeds=pooled_prompt_embeds[B:],
            generator=generator,
            output_type="latent",
            cond_latents=image_latents_init,
            latents=latents,
            cond_latents_cam=cond_latents_cam,
            latents_cam=latents_cam,
            cond_view_ids_init=view_ids[:K].tolist(),
            cond_view_ids_tgt=view_ids[K:].tolist(),
            cond_is_depth_init=cond_is_depth_init,
            cond_is_depth_tgt=cond_is_depth_tgt,
            max_shift_override=self.max_shift_override,
            cam_cfg=getattr(self, "cam_cfg_mode", "fix"),
        ).images

        # cam_latents for the image views only (before geometry doubling)
        cam_latents_img = [views[:T_img] for views in cam_latents_all_orig]

        if not geometry_flag:
            out = {
                "images": [
                    list(self.latents_to_pixels(torch.stack(sample).to(dtype)))
                    for sample in samples
                ],
                "cam_latents": cam_latents_img,
                "vlm_texts": vlm_texts,
                "pp_cam_params": pp_cam_params,
            }
            if return_latents:
                # Per-sample list of generated target latents in their pre-decode
                # form. The chunked AR caller takes the last one as the next
                # chunk's anchor latent, skipping VAE encode/decode.
                out["latents"] = [
                    [t.detach() for t in sample] for sample in samples
                ]
            return out

        # geometry state mode: the target block is [image_targets (T_tgt_img) | depth_targets (T_img)]
        img_list, depth_list, depth_vis_list, latents_list = [], [], [], []
        for sample in samples:
            sample_t = torch.stack(sample).to(dtype)
            img_latents = sample_t[:T_tgt_img]
            depth_latents = sample_t[T_tgt_img:]

            img_views = list(self.latents_to_pixels(img_latents))

            # The depth targets are trained as vision banana RGB encodings in
            # [-1, 1] (the wrappers feed depth_vision banana straight into the
            # VAE), so decode scalar depth through the Hilbert-colormap
            # inverse and keep the raw RGB for dataloader-style visualization.
            depth_rgb = self.latents_to_pixels(depth_latents).float().cpu()  # [T_img, 3, H, W]
            depth_views_t = visionbanana_to_depth(
                depth_rgb.permute(0, 2, 3, 1))                  # [T_img, H, W]
            depth_views = [depth_views_t[i] for i in range(depth_views_t.shape[0])]
            img_list.append(img_views)
            depth_list.append(depth_views)
            depth_vis_list.append([depth_rgb[i] for i in range(depth_rgb.shape[0])])
            # Only the IMAGE target latents are useful for AR linkage; depth
            # latents stay confined to the depth-decoding path.
            latents_list.append([img_latents[i].detach() for i in range(img_latents.shape[0])])

        out = {
            "images": img_list,
            "depths": depth_list,
            "depths_vis": depth_vis_list,   # list[B][T_img] of [3,H,W] vision banana RGB in [-1,1]
            "cam_latents": cam_latents_img,
            "vlm_texts": vlm_texts,
            "pp_cam_params": pp_cam_params,
        }
        if return_latents:
            out["latents"] = latents_list
        return out
    
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

        past_key_values = DynamicCache()
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
            input_ids = torch.argmax(logits, dim=-1)
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