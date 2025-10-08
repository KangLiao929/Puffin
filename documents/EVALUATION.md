## 🖼️ Evaluation

### Camera Understanding

To evaluate the camera understanding performance, please customize the config path, model weights, and test set. We recommend the ```accelerate``` command with multiple GPUs for fast evaluations:

```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/evaluation/understanding.py configs/pipelines/stage_2_base.py \
          --checkpoint checkpoints/Puffin-Base.pth --image_dir ./Puffin-Und/images/ \
          --output Puffin-Und_test.json
```
The understanding results can be found at ```--output```.

The thinking mode can be enabled by changing the settings and append ```--thinking``` flag:

```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/evaluation/understanding.py configs/pipelines/stage_3_thinking.py \
          --checkpoint checkpoints/Puffin-Thinking.pth --image_dir ./Puffin-Und/images/ \
          --output Puffin-Und_test_thinking.json \
          --thinking
```
The results of understanding with thinking can be found at ```--output```.


### Camera-controllable Image Generation

To evaluate the camera-controllable generation performance, please customize the config path, model weights, and test set. We recommend the ```accelerate``` command with multiple GPUs for fast evaluations:

```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/evaluation/generation.py configs/pipelines/stage_2_base.py \
          --checkpoint checkpoints/Puffin-Base.pth --output gen_evaluation/ \
          --prompt_path ./Puffin-Gen/caption/caption_src --camera_path  ./Puffin-Gen/camera  \
```
The generation results can be found at ```--output```.

To enable the thinking mode, please simply change the settings and append ```--thinking``` flag:

```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/evaluation/generation.py configs/pipelines/stage_3_thinking.py \
          --checkpoint checkpoints/Puffin-Thinking.pth --output gen_evaluation_thinking/ \
          --prompt_path ./Puffin-Gen/caption/caption_src --camera_path  ./Puffin-Gen/camera  \
          --thinking --output_thinking gen_evaluation_thinking/gen_thinking.json

```
The generation results and textural reasoning results can be found at ```--output``` and ```--output_thinking```, respectively.
