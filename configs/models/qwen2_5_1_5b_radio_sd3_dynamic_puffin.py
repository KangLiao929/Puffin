import torch
from src.models.puffin.model import Qwen2p5RadioStableDiffusion3HFDynamic
from src.models.stable_diffusion3.transformer_sd3_dynamic import SD3Transformer2DModel
from src.models.radiov3.hf_model import RADIOModel
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_name_or_path = 'Qwen/Qwen2.5-1.5B-Instruct'
sd3_model_name_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"

prompt_template = dict(
    SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
    INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n'),
    SUFFIX='<|im_end|>',
    IMG_START_TOKEN='<|vision_start|>',
    IMG_END_TOKEN='<|vision_end|>',
    IMG_CONTEXT_TOKEN='<|image_pad|>',
    GENERATION='Generate an image: {input}',
    GENERATION_CROSS='Generate a target image given an initial view: {input}',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>']
)

model = dict(type=Qwen2p5RadioStableDiffusion3HFDynamic,
             num_queries=64,
             connector_1=dict(
                 hidden_size=1024,
                 intermediate_size=4096,
                 num_hidden_layers=6,
                 _attn_implementation='flash_attention_2',
                 num_attention_heads=16, ),
             connector_2=dict(
                 hidden_size=1024,
                 intermediate_size=4096,
                 num_hidden_layers=6,
                 _attn_implementation='flash_attention_2',
                 num_attention_heads=16, ),
             transformer=dict(
                 type=SD3Transformer2DModel.from_pretrained,
                 pretrained_model_name_or_path=sd3_model_name_or_path,
                 subfolder="transformer",
                 torch_dtype=torch.bfloat16),
             test_scheduler=dict(
                 type=FlowMatchEulerDiscreteScheduler.from_pretrained,
                 pretrained_model_name_or_path=sd3_model_name_or_path,
                 subfolder="scheduler"),
             train_scheduler=dict(
                 type=FlowMatchEulerDiscreteScheduler.from_pretrained,
                 pretrained_model_name_or_path=sd3_model_name_or_path,
                 subfolder="scheduler"),
             vae=dict(
                 type=AutoencoderKL.from_pretrained,
                 pretrained_model_name_or_path=sd3_model_name_or_path,
                 subfolder="vae",
                 torch_dtype=torch.bfloat16),
             freeze_visual_encoder=True,
             freeze_llm=True,
             llm=dict(
                 type=AutoModelForCausalLM.from_pretrained,
                 pretrained_model_name_or_path=llm_name_or_path,
                 torch_dtype=torch.bfloat16,
                 attn_implementation='flash_attention_2',
             ),
             tokenizer=dict(
                 type=AutoTokenizer.from_pretrained,
                 pretrained_model_name_or_path=llm_name_or_path),
             prompt_template=prompt_template,
             pretrained_pth=None,
             use_activation_checkpointing=False,
             visual_encoder=dict(
                 type=RADIOModel.from_pretrained,
                 pretrained_model_name_or_path="nvidia/C-RADIOv3-H",
                 torch_dtype=torch.bfloat16,),
             )
