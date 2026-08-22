# Elements
image_size = 640
image_process = 'identity'
camera_model = 'radial'

# Prompt Template
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