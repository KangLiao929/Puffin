## 🖼️ Evaluation

### Camera Understanding

To evaluate the camera understanding performance on 🤗 [Puffin-Und benchmark](https://huggingface.co/datasets/KangLiao/Puffin-4M/tree/main/benchmark/Puffin-Und), please customize the config path, model weights, and test set. We recommend the ```accelerate``` command with multiple GPUs for fast evaluations:

```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/evaluation/understanding.py configs/pipelines/stage_2_base.py \
          --checkpoint checkpoints/Puffin-Base.pth --image_dir Puffin-Und/images/ \
          --output outputs/Puffin-Und_test.json
```
The understanding results can be found at ```--output```.

The thinking mode can be enabled by changing the settings and append ```--thinking``` flag:

```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/evaluation/understanding.py configs/pipelines/stage_3_thinking.py \
          --checkpoint checkpoints/Puffin-Thinking.pth --image_dir Puffin-Und/images/ \
          --output outputs/Puffin-Und_test_thinking.json \
          --thinking
```
The results of understanding with thinking can be found at ```--output```.

To compute the quantitative metrics like median error and AUC of the camera understanding results, we provide a handy script based on [GeoCalib/siclib](https://github.com/cvg/GeoCalib/tree/main/siclib). First, please install the required library by:
```shell
cd scripts/
python -m pip install -e siclib
```
Then, run the script with the corresponding camera understanding results (```.json```), output path, and the GT camera parameters (```.csv```):
```shell
python -m siclib.eval.eval_understanding --input_json outputs/Puffin-Und_test.json --output_dir outputs/Puffin-Und_eval/ --gt_csv Puffin-Und/cameras.csv
```
For evaluating on the classical camera calibration datasets ([Stanford2D3D](https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/stanford2d3d.zip), [Lamar2k](https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/lamar2k.zip), [Megadepth2k](https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/megadepth2k.zip), [Tartanair](https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/tartanair.zip)), please download them and conduct the above scripts with corresponding paths.


### Camera-controllable Image Generation

To evaluate the camera-controllable generation performance on 🤗 [Puffin-Gen benchmark](https://huggingface.co/datasets/KangLiao/Puffin-4M/tree/main/benchmark/Puffin-Gen), please customize the config path, model weights, and test set. We recommend the ```accelerate``` command with multiple GPUs for fast evaluations:

```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/evaluation/generation.py configs/pipelines/stage_2_base.py \
          --checkpoint checkpoints/Puffin-Base.pth --output outputs/Puffin-Gen_eval/ \
          --prompt_path Puffin-Gen/caption/caption_src --camera_path  Puffin-Gen/camera  \
```
The generation results can be found at ```--output```.

To enable the thinking mode, please simply change the settings and append ```--thinking``` flag:

```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/evaluation/generation.py configs/pipelines/stage_3_thinking.py \
          --checkpoint checkpoints/Puffin-Thinking.pth --output outputs/Puffin-Gen_eval_thinking/ \
          --prompt_path Puffin-Gen/caption/caption_src --camera_path  Puffin-Gen/camera  \
          --thinking --output_thinking outputs/Puffin-Gen_eval_thinking/gen_thinking.json

```
The generation results and textural reasoning results can be found at ```--output``` and ```--output_thinking```, respectively.

Similar to the camera understanding evaluation, please install the required library using the command below (skip this step if already installed):
```shell
cd scripts/
python -m pip install -e siclib
```
Then, run the script with the corresponding paths of the test dataset and camera-controllable generation results:
```shell
python -m siclib.eval.eval_generation --dataset_dir Puffin-Gen/ --test_img_dir outputs/Puffin-Gen_eval/ --conf geocalib-pinhole --tag puffin --overwrite
```