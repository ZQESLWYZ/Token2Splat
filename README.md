# ✦ Token2Splat ✦

### Learning Generative 3D Gaussian Distributions via Canonical Patch Tokens

[![Status: Submitted to IJCV](https://img.shields.io/badge/status-Submitted%20to%20IJCV-6f42c1?style=for-the-badge&logo=bookstack&logoColor=white)](https://github.com/)
[![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](#-installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#-installation)


## ⚡ Overview

Raw 3D Gaussian Splatting (3DGS) sets are unordered, variable-cardinality point clouds. **Token2Splat** solves this via a deterministic, permutation-invariant **canonical tokenization**, modeled through a hierarchical VAE and a class-conditional rectified flow with differentiable render supervision.

* **Stage A**: Canonical patch tokenizer (anchor bank, right-handed canonical frames, Morton ordering).
* **Stage B**: Hierarchical VAE with global/patch latents and occupancy/count heads.
* **Stage C**: Class-conditional global–local rectified flow with endpoint photometric supervision.
* **Stage D**: Pure-PyTorch differentiable splat renderer (no `gsplat`, no CUDA required).

---

## 🛠 Installation

```powershell
conda create -n Token2Splat python=3.10 -y
conda activate Token2Splat
pip install -r requirements.txt
pip install -e .

```


## 🚀 Quick Start

Run the end-to-end CPU smoke demo in ~1 minute:

```powershell
conda run -n Token2Splat python scripts/demo.py --config configs/data_smoke.yaml --output-dir runs/demo --samples 2 --steps 24

```

---

## 📁 Core Pipeline

1. **Data Preparation**: Use ObjectGS-34 scene extraction.
```powershell
python scripts/make_data_data.py --config configs/data_smoke.yaml --out data/data

```


2. **Stage 1 Training (VAE)**:
```powershell
python scripts/train_stage1.py --config configs/data_smoke.yaml --data data/data --work-dir runs/data/stage1 --device cpu --cache-tokens

```


3. **Stage 2 Training (Flow)**:
```powershell
python scripts/train_stage2.py --config configs/data_smoke.yaml --data data/data --work-dir runs/data/stage2 --vae-checkpoint runs/data/stage1/vae_best.pt --device cpu --cache-tokens

```


4. **Inference & Evaluation**:
* Reconstruct: `python scripts/reconstruct.py ...`
* Generate: `python scripts/generate.py ...`
* Evaluate: `python scripts/evaluate.py ...`


```

## 🐍 Python API Example

```python
from token2splat import CanonicalPatchTokenizer, load_config
from token2splat.inference import generate_objects, load_pipeline
from token2splat.render import build_cameras, render_views

cfg = load_config("configs/data_smoke.yaml")
cfg, ckpt, tokenizer, vae, flow = load_pipeline("runs/data/stage2/flow_best.pt", device="cpu")

# Generate samples
samples = generate_objects(vae, flow, cfg, classes=[0, 1], samples=4, steps=32, seed=42, device="cpu")

```

## 📜 Citation

```bibtex
@article{token2splat2027,
  title  = {Token2Splat: Learning Generative 3D Gaussian Distributions via Canonical Patch Tokens},
  year   = {2027},
  note   = {Submitted to IJCV}
}

