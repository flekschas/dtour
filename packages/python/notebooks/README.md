# dtour example notebooks

Interactive [marimo](https://marimo.io) notebooks that demonstrate dtour on real
text, image, and single-cell datasets. Each notebook is **fully self-contained**:
it declares its own dependencies via inline
[PEP 723](https://peps.python.org/pep-0723/) script metadata and downloads (and
caches) its data on first run into `__cache__/`, so there is nothing to set up
beforehand other than making sure [`uv` is installed](https://docs.astral.sh/uv/getting-started/installation/).

## Running

The notebooks run straight from a fresh checkout with [uv](https://docs.astral.sh/uv/).

Open one interactively in the marimo editor:

```sh
uvx marimo edit --sandbox demo_spectral.py
```

Use `uvx marimo run` to run the notebooks in read-only mode.

## Notebooks

### [`demo_attraction_repulsion.py`](demo_attraction_repulsion.py)

Attraction–repulsion spectrum tour of **Fashion-MNIST** (70K images, PCA to 50D).

- **Data:** Fashion-MNIST, fetched from OpenML and cached (the one notebook that
  regenerates rather than downloading a precomputed tour).
- **Tour:** `attraction_repulsion_tour` — a sequence of openTSNE embeddings at
  decreasing exaggeration (ρ), Procrustes-aligned into a smooth morph from pure
  attraction (LE-like, ρ=100) through UMAP-like (ρ=4) to t-SNE (ρ=1).
- Select points to preview the corresponding Fashion-MNIST images in the sidebar.

### [`demo_brain_atlas.py`](demo_brain_atlas.py)

293K cells from the **La Manno et al. 2021** developing mouse brain scRNA-seq atlas.

- **Data:** precomputed PC1–PC8 coordinates + cell-type annotations, downloaded
  from `data.dtour.dev`.
- **Tour:** dtour `little_tour` over PC1–PC8 (a dimensions tour), shown
  side-by-side with a 2D UMAP rendered by
  [jupyter-scatter](https://jupyter-scatter.dev).
- Lasso-select in the dtour view to highlight the same cells in the UMAP.

### [`demo_image_embedding.py`](demo_image_embedding.py)

Joint pixel + caption embedding of **ShareGPT4V × COCO** (α=0.5).

- **Data:** precomputed 2D/4D DensMAP-UMAP and Fisher-LE tours, downloaded from
  `data.dtour.dev`.
- **Tours:** a label-aware Fisher Laplacian Eigenmaps tour and a 4D UMAP
  dimensions tour (dtour), alongside a 2D UMAP (jupyter-scatter).
- Select points to preview the corresponding COCO images.

### [`demo_immune_cell_markers.py`](demo_immune_cell_markers.py)

**Mair et al. 2022** tumor CyTOF dataset, colored by FAUST-derived phenotypes.

- **Data:** the Ozette tumor sample + phenotype map, downloaded from `data.dtour.dev`.
- **Tour:** an 8-D UMAP of the 18 winsorized marker columns, explored as a little tour.
- Radial bar charts show per-keyframe phenotype **confusion** (`cev` metric); click
  a phenotype label to see its confusion.

### [`demo_spectral.py`](demo_spectral.py)

**Mair et al. 2022** tumor dataset via **Laplacian Eigenmaps**.

- **Data:** the same Ozette tumor sample + phenotype map as above.
- **Tour:** an 8-D LE embedding of the 18 markers, toured through consecutive
  eigenvector pairs ([LE1, LE2] → [LE2, LE3] → …). Includes vanilla, sign-flipped,
  and Fisher (label-aware) variants, plus a heatmap of eigenvector ↔ marker
  correlations.
