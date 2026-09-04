<h1>
  <img src="Puffin/assets/website/Puffin_logo.png" alt="logo" width="65" style="vertical-align: middle; margin-right: 8px;">
  Puffin Series: Towards Unified Multimodal 3D World Models
</h1>

**Puffin** is a series of unified multimodal models advancing toward 3D world
modeling. It starts from camera-centric spatial intelligence — understanding
and generating the world from arbitrary viewpoints and orientations — and
scales to a world model that represents worlds through three complementary
native 3D world states (**physics**, **geometry**, and **appearance**),
supporting camera-to-world understanding, camera-controllable generation, and
image-/text-to-3D world generation without external perception or
reconstruction modules. Each release lives in its own subdirectory of this
repository:

- [**`Puffin/`**](Puffin/) — *Thinking with Camera: A Unified Multimodal Model
  for Camera-Centric Understanding and Generation* (ICLR 2026)<br>
  [![arXiv](https://img.shields.io/badge/arXiv-2510.08673-b31b1b.svg)](https://arxiv.org/abs/2510.08673)
  [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://kangliao929.github.io/projects/puffin/)
  [![Puffin Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange)](https://huggingface.co/KangLiao/Puffin)
  [![Puffin-4M Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/KangLiao/Puffin-4M)
  [![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/KangLiao/Puffin)
- [**`Puffin-World/`**](Puffin-World/) — *Puffin-World: Scaling a Unified
  Multimodal Model with Native 3D World States* (2026)<br>
  [![arXiv](https://img.shields.io/badge/arXiv-2609.04196-b31b1b.svg)](https://arxiv.org/abs/2609.04196)
  [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://kangliao929.github.io/projects/puffin-world/)
  [![Puffin-World Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange)](https://huggingface.co/KangLiao/Puffin-World)
  [![Puffin-16M Dataset](https://img.shields.io/badge/Dataset-Puffin--16M-orange)](https://kangliao929.github.io/projects/puffin-16m/)
  [![Hugging Face Blog](https://img.shields.io/badge/🤗%20Hugging%20Face-Blog-blue)](https://huggingface.co/blog/KangLiao/puffin-world)

## 📝 Changelog & News

- [x] 2026.09.04: The paper of **Puffin-World** is released on [arXiv](https://arxiv.org/abs/2609.04196).
- [x] 2026.08.23: The model weights of **Puffin-World** (Base / Pro / Caption) are released on [Hugging Face](https://huggingface.co/KangLiao/Puffin-World).
- [x] 2026.08.23: The scripts of the dataset construction pipeline have been released.
- [x] 2026.08.23: The camera captions (by our method) of the commonly used large-scale text-to-image datasets, such as megalith-10m, have been released.
- [x] 2026.08.22: The training and evaluation code of **Puffin-World** is released.
- [x] 2026.01.26: Puffin has been accepted at ICLR 2026.
- [x] 2026.01.15: Puffin-4M dataset reached 20,000 downloads on Hugging Face within three months of release.
- [x] 2026.01.10: The scripts of the camera-centric evaluation has been released.
- [x] 2025.10.10: The paper, project page, code, model, dataset, and demo of Puffin are online.

## 📖 Overview

| Project | Paper | Project Page | Model | Dataset | Code |
|---|---|---|---|---|---|
| **Puffin**<br>*Thinking with Camera* | [arXiv:2510.08673](https://arxiv.org/abs/2510.08673)<br>(ICLR 2026) | [Page](https://kangliao929.github.io/projects/puffin/) | [🤗 KangLiao/Puffin](https://huggingface.co/KangLiao/Puffin) | [🤗 Puffin-4M](https://huggingface.co/datasets/KangLiao/Puffin-4M) | [`Puffin/`](Puffin/) |
| **Puffin-World**<br>*Native 3D World States* | [arXiv:2609.04196](https://arxiv.org/abs/2609.04196) | [Page](https://kangliao929.github.io/projects/puffin-world/) | [🤗 KangLiao/Puffin-World](https://huggingface.co/KangLiao/Puffin-World) | [🤗 Puffin-16M](https://huggingface.co/datasets/KangLiao/Puffin-16M) | [`Puffin-World/`](Puffin-World/) |

## 🗞️ License

This project is licensed under [NTU S-Lab License 1.0](LICENSE).

## 📚 Citation

If you find Puffin useful for your research or applications, please cite our
papers using the following BibTeX:

```bibtex
@article{liao2025puffin,
  title={Thinking with Camera: A Unified Multimodal Model for Camera-Centric Understanding and Generation},
  author={Liao, Kang and Wu, Size and Wu, Zhonghua and Jin, Linyi and Wang, Chao and Wang, Yikai and Wang, Fei and Li, Wei and Loy, Chen Change},
  journal={arXiv preprint arXiv:2510.08673},
  year={2025}
}

@article{liao2026puffinworld,
  title={Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States},
  author={Liao, Kang and Luo, Yihang and Wu, Xiao-Ming and Jin, Linyi and Wu, Size and Lin, Chunyu and Zhao, Yao and Wang, Fei and Li, Wei and Loy, Chen Change},
  journal={arXiv preprint arXiv:2609.04196},
  year={2026}
}
```
