<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">EEGRFusion</h2>

</div>

EEGRFusion is a lightweight codebase for EEG-based **retrieval** and **reconstruction/generation** with a montage-aware encoder (**MAMD**).

---

## Repository Structure (Key Files)

- `setup.sh` — environment setup (dependencies / initialization)
- `MAMD_retrieval.py` — train & evaluate EEG→image **retrieval**
- `MAMD_reconstruction.py` — train EEG→image **reconstruction** model
- `generation.ipynb` — run **image generation** / visualization
- `reconstruction_newmetrics.ipynb` — compute **reconstruction metrics**

---

## Quick Start

### 1) Setup

```bash
bash setup.sh
