# MEDIT — Manifold-Embedded Diffusion Preprocessing (Weinreb)

MEDIT is the **single-cell preprocessing and infrastructure layer** for diffusion
and manifold-learning experiments on the Weinreb in-vitro hematopoiesis dataset.
It takes raw single-cell RNA-seq data, runs a shared QC + HVG pipeline, and
produces a clean, reusable `.h5ad` file in `data/interim/` that downstream code
can consume.

**Pipeline:**

> Raw `.h5ad` → QC filters → HVGs → normalize + log → `data/interim/weinreb_qc.h5ad`

---

## 1. Environment Setup

Create and activate the Conda environment, then install the package in editable mode:

```bash
# From repo root
conda env create -f env.yml
conda activate medit

# Install as an editable Python package
pip install -e .
```

Quick sanity check:

```bash
python -c "import medit, medit.qc.pipeline as qp; print('OK', qp)"
```

---

## 2. Data Layout

All data lives under `data/`:

```text
data/
  raw/
    stateFate_inVitro_normed_counts.h5ad   # Weinreb input
    GSM4185642_stateFate_inVitro_*.mtx.gz  # original component files (optional)
    RAW_SOURCES.md                         # provenance notes
  interim/
    weinreb_qc.h5ad                        # QC’d + HVG’d AnnData (output)
    weinreb_eggfm_diffmap.h5ad             # QC + EGGFM-based diffusion embedding
    paul15/                                # (optional) other datasets / scratch
```

- `data/raw/` should contain the original Weinreb `.h5ad` (or the GSM components).
- `data/interim/weinreb_qc.h5ad` is the **preprocessed artifact** for
  downstream models.
- `data/interim/weinreb_eggfm_diffmap.h5ad` adds an EGGFM-based diffusion embedding.

---

## 3. Repository Structure

```text
README.md
env.yml
pyproject.toml

data/
  raw/
  interim/
out/
  qc/
  models/
    eggfm/

scripts/
  make_h5ad.py          # Build Weinreb AnnData from GSM component files
  qc.py                 # QC + HVG preprocessing for Weinreb
  train_energy.py       # Train EGGFM energy model on Weinreb
  diffusion_embed.py    # Build EGGFM-based diffusion embedding

src/
  medit/
    __init__.py
    config.py           # Dataset-level configuration (Weinreb paths / schema)
    qc/
      __init__.py
      pipeline.py       # Core QC + HVG + normalization logic
    eggfm/
      __init__.py
      models.py         # EnergyMLP definition
      dataset.py        # AnnDataExpressionDataset wrapper
      trainer.py        # DSM training loop (train_energy_model)
    diffusion/
      __init__.py
      config.py         # DiffusionConfig dataclass
      engine.py         # EGGFMDiffusionEngine
      core.py           # DiffusionMapBuilder and kernels
      metrics.py        # Geometry / kernel metric helpers
      embed.py          # High-level embedding helpers
```

- Core logic lives under `src/medit/...`.
- `scripts/` provides thin CLIs that parse arguments and call into `medit`.
- `data/` is data-only; no code lives there.
- `out/` holds model artifacts (e.g. EGGFM checkpoints) and QC outputs.

---

## 4. Configuration

QC, HVG, energy, and diffusion settings are driven by `configs/params.yml`.
Minimal required fields:

```yaml
dataset:
  name: weinreb

qc:
  min_genes: 200        # minimum detected genes per cell
  max_pct_mt: 0.2       # optional; set to 1.0 to disable mito filter
  # min_cells: 3        # optional; default is 0.1% of cells

hvg_n_top_genes: 2000   # number of HVGs to flag via Seurat v3 flavor

eggfm_model:
  hidden_dims: [256, 256]

eggfm_train:
  batch_size: 256
  num_epochs: 100
  lr: 1.0e-3
  weight_decay: 0.0
  sigma: 0.15             # DSM noise scale
  max_grad_norm: 5.0
  early_stop_patience: 0  # 0 = no early stopping

eggfm_diffusion:
  n_neighbors: 30         # k for kNN graph
  t: 10                   # diffusion time
  metric_mode: "scm"      # or "euclidean", etc.
  device: "cuda"          # or "cpu"
```

`medit.qc.pipeline.prep` uses the `qc` and `hvg_n_top_genes` entries and will:

- filter genes by `min_cells` (or 0.1% of cells if omitted),
- filter cells by `min_genes`,
- optionally filter by mitochondrial fraction if `max_pct_mt < 1.0`,
- compute HVGs and subset to them,
- normalize + log1p on `ad.X`, preserving raw counts in `ad.layers["counts"]`.

`eggfm_model`, `eggfm_train`, and `eggfm_diffusion` blocks are consumed by the
EGGFM training and diffusion embedding steps described below.

---

## 5. QC + Preprocessing (Weinreb)

### 5.1 Build the Weinreb `.h5ad`

Starting from the raw GSM files, `scripts/make_h5ad.py` builds a single normalized,
cell × gene AnnData object that serves as the Weinreb input.

From repo root:

```bash
python scripts/make_h5ad.py \
  --data-dir data/raw \
  --out data/raw/stateFate_inVitro_normed_counts.h5ad
```

This will:

1. Load the Weinreb in-vitro hematopoiesis matrices from `data/raw/`.
2. Harmonize gene and cell indices.
3. Store normalized counts in `ad.X`.
4. Write `data/raw/stateFate_inVitro_normed_counts.h5ad`.

This step only needs to be rerun if the underlying GSM inputs change.

---

### 5.2 QC + HVG selection

After the environment is set up and the package is installed (`pip install -e .`),
run:

```bash
python scripts/qc.py \
  --params configs/params.yml \
  --ad data/raw/stateFate_inVitro_normed_counts.h5ad \
  --out data/interim/weinreb_qc.h5ad
```

This will:

1. Load the raw `.h5ad`.
2. Filter genes by `min_cells` (or 0.1% of cells by default).
3. Filter cells by `min_genes`.
4. Optionally apply a mitochondrial filter (`max_pct_mt`).
5. Flag HVGs (`hvg_n_top_genes`) and subset to them.
6. Normalize and log1p expression, preserving raw counts in a `counts` layer.
7. Write the QC’d HVG-restricted AnnData to `data/interim/weinreb_qc.h5ad`.

Quick inspection:

```python
import scanpy as sc

ad = sc.read_h5ad("data/interim/weinreb_qc.h5ad")
print(ad)
print("layers:", ad.layers.keys())
print("obs:", list(ad.obs.columns)[:10])
print("var:", list(ad.var.columns)[:10])
```

`data/interim/weinreb_qc.h5ad` is the input for the energy and diffusion steps.

---

## 6. EGGFM + Diffusion

This section describes how to train an EGGFM energy model on Weinreb and then
build an EGGFM-aware diffusion embedding.

All commands are run from the repo root.

---

### 6.1 Train EGGFM on Weinreb (HVG space)

```bash
python scripts/train_energy.py \
  --params configs/params.yml \
  --ad data/interim/weinreb_qc.h5ad \
  --out-dir out/models/eggfm
```

This will:

1. Load `configs/params.yml` and read `eggfm_model` and `eggfm_train`.
2. Load `data/interim/weinreb_qc.h5ad`.
3. Wrap the expression matrix in an `AnnDataExpressionDataset`.
4. Run DSM training of `EnergyMLP` (`medit.eggfm.train_energy_model`).
5. Compute mean and standard deviation for the HVG space.
6. Write a checkpoint bundle to:

   ```text
   out/models/eggfm/eggfm_energy_weinreb.pt
   ```

The checkpoint contains:

- `state_dict` — EnergyMLP weights  
- `model_cfg` — copy of the `eggfm_model` block  
- `train_cfg` — copy of the `eggfm_train` block  
- `n_genes` — dimensionality of the HVG space  
- `var_names` — HVG gene names (aligned to `weinreb_qc.h5ad`)  
- `mean`, `std` — standardization stats in the HVG feature space  

---

### 6.2 Build an EGGFM-based diffusion embedding

Once the EGGFM checkpoint exists, construct a diffusion embedding that uses the
learned energy as a geometry prior:

```bash
python scripts/diffusion_embed.py \
  --params configs/params.yml \
  --ad data/interim/weinreb_qc.h5ad \
  --energy-ckpt out/models/eggfm/eggfm_energy_weinreb.pt \
  --out data/interim/weinreb_eggfm_diffmap.h5ad \
  --obsm-key X_eggfm_diffmap
```

This will:

1. Load `eggfm_diffusion` from `configs/params.yml` (e.g. `n_neighbors`, `t`,
   `metric_mode`, `device`).
2. Load the EGGFM checkpoint and rebuild `EnergyMLP`.
3. Construct an `EGGFMDiffusionEngine` from `medit.diffusion`.
4. Compute a diffusion map (or similar embedding) over `weinreb_qc` using the
   chosen metric (e.g. SCM or Euclidean) and the energy model.
5. Attach the embedding as `ad.obsm["X_eggfm_diffmap"]`.
6. Write the resulting AnnData to:

   ```text
   data/interim/weinreb_eggfm_diffmap.h5ad
   ```

To verify the embedding:

```python
import scanpy as sc

ad = sc.read_h5ad("data/interim/weinreb_eggfm_diffmap.h5ad")
print("embedding shape:", ad.obsm["X_eggfm_diffmap"].shape)
```
