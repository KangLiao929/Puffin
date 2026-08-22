<h1>
  <img src="Puffin/assets/website/Puffin_logo.png" alt="logo" width="65" style="vertical-align: middle; margin-right: 8px;">
  Puffin: Camera-Centric Unified Multimodal Models
</h1>

**Puffin** is a series of camera-centric unified multimodal models for spatial
intelligence, unifying understanding and generation of the world across
viewpoints, orientations, and — with Puffin-World — native 3D world states.
Each release lives in its own subdirectory of this repository:

- [**`Puffin/`**](Puffin/) — *Thinking with Camera: A Unified Multimodal Model
  for Camera-Centric Understanding and Generation* (ICLR 2026)
- [**`Puffin-World/`**](Puffin-World/) — *Puffin-World: Scaling a Unified
  Multimodal Model with Native 3D World States* (2026)

## 📝 Changelog & News

- [x] 2026.08.22: The training and evaluation code of **Puffin-World** is released.
- [x] 2026.01.26: Puffin has been accepted at ICLR 2026.
- [x] 2026.01.15: Puffin-4M dataset reached 20,000 downloads on Hugging Face within three months of release.
- [x] 2026.01.10: The scripts of the camera-centric evaluation has been released.
- [x] 2025.10.10: The paper, project page, code, model, dataset, and demo of Puffin are online.
- [ ] Release the scripts of the dataset construction pipeline.
- [ ] Release the camera caption (by our method) of the commonly used large-scale text-to-image datasets, such as megalith-10m.

## 📖 Overview

| Project | Paper | Project Page | Model | Dataset | Code |
|---|---|---|---|---|---|
| **Puffin**<br>*Thinking with Camera* | [arXiv:2510.08673](https://arxiv.org/abs/2510.08673)<br>(ICLR 2026) | [Page](https://kangliao929.github.io/projects/puffin/) | [🤗 KangLiao/Puffin](https://huggingface.co/KangLiao/Puffin) | [🤗 Puffin-4M](https://huggingface.co/datasets/KangLiao/Puffin-4M) | [`Puffin/`](Puffin/) |
| **Puffin-World**<br>*Native 3D World States* | Coming soon | [Page](https://kangliao929.github.io/projects/puffin-world/) | Coming soon | [🤗 Puffin-16M](https://huggingface.co/datasets/KangLiao/Puffin-16M) | [`Puffin-World/`](Puffin-World/) |

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
  title   = {Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States},
  author  = {Liao, Kang and Luo, Yihang and Wu, Xiao-Ming and Jin, Linyi and Wu, Size and Lin, Chunyu and Zhao, Yao and Wang, Fei and Li, Wei and Loy, Chen Change},
  journal = {Preprint},
  year    = {2026}
}
```
