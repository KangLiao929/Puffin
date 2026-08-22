import torch
from src.models.puffin.vlm import RadioVLM
from src.models.radiov3.hf_model import RADIOModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.datasets.processors import prompt_template

# LLM family selector — controls prompt construction + position-id layout:
#   'qwen2_5' / 'qwen3'  : 1D RoPE, xtuner-style prompt_template
#   'qwen3_5'            : 3D MRoPE (multi-axis), HF apply_chat_template
llm_family = 'qwen3_5'

LLM_PATHS = {
    'qwen2_5': '/mnt/afs_100t/NTU_slab/kliao/models/Qwen2.5-7B-Instruct',
    'qwen3':   '/mnt/afs_100t/NTU_slab/kliao/models/Qwen3-7B',
    'qwen3_5': '/mnt/afs_100t/NTU_slab/kliao/models/Qwen3.5-0.8B',
}
llm_name_or_path = LLM_PATHS[llm_family]
vision_encoder_name_or_path = "/mnt/afs_100t/NTU_slab/kliao/models/C-RADIOv3-H"

prompt_template = dict(
    SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
    INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n'),
    SUFFIX='<|im_end|>',
    IMG_START_TOKEN='<|vision_start|>',
    IMG_END_TOKEN='<|vision_end|>',
    IMG_CONTEXT_TOKEN='<|image_pad|>',
    GENERATION='Generate an image: {input}',
    GENERATION_CROSS='Generate the target images given an initial view: {input}',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>']
)

model = dict(type=RadioVLM,
             llm_family=llm_family,
             freeze_visual_encoder=True,
             freeze_llm=True,
             freeze_projector=False,
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
                 pretrained_model_name_or_path=vision_encoder_name_or_path,
                 torch_dtype=torch.bfloat16,),
             )
