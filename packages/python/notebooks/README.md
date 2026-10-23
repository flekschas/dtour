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

### [`demo_digits.py`](demo_digits.py)

PCA tour of scikit-learn's **digits** dataset (1,797 8×8 images).

- **Data:** bundled with scikit-learn, so nothing is downloaded.
- **Tour:** `little_tour` over the top 6 principal components of the 64 pixels.
- Select points to show their images. The selection is read reactively from
  `mo.ui.anywidget`, so no observers are needed.
- Also runs in the browser via WebAssembly:
  `uvx marimo export html-wasm demo_digits.py -o out --mode run`, then serve `out/`.

### [`demo_molab.py`](demo_molab.py)

A quick tour of using dtour in a marimo analysis, linked from dtour.dev and hosted on
[molab](https://molab.marimo.io/github/flekschas/dtour/blob/main/packages/python/notebooks/demo_molab.py).

- **Data:** the brain atlas and tumor data with their UMAP embeddings precomputed by
  [`scripts/make_molab_demo_data.py`](../../../scripts/make_molab_demo_data.py),
  downloaded from `data.dtour.dev`, so it starts in seconds.
- **Brain atlas:** a little tour linked with a 2D UMAP; selections list the subclasses
  they are enriched for.
- **Tumor immune cells:** a little tour of the 8D UMAP; selections show their marker
  profile.
- **Handwritten digits:** a PCA tour of the pixels linked with a 2D UMAP via
  `dtour.link`; selections show their images and which digits they look like.

### [`demo_molab_cev.py`](demo_molab_cev.py)

`demo_molab.py` plus live confusion bars in the immune cell tour, computed with
[`cev-metrics`](https://pypi.org/project/cev-metrics/); clicking phenotypes shows their
confusion. Needs Python 3.12 or older, as `cev-metrics` has no wheels for newer versions
yet, so it does not run on molab.

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

### [`demo_sherlock_umap.py`](demo_sherlock_umap.py)

**SHERLOCK** latent space vs. expression space on the **Replogle et al. 2022**
(CRISPRi) and **Norman et al. 2019** (CRISPRa) Perturb-seq screens.

- **Data:** precomputed SHERLOCK latent dimensions, principal components, 2D/4D
  UMAPs, and 2D UMAPs of 32 training snapshots, downloaded from `data.dtour.dev`.
- **Tours:** a sequential tour through the training snapshots with one preview
  per epoch, a morph from the expression UMAP to the SHERLOCK UMAP, and
  side-by-side little tours of both spaces and of their 4D UMAPs.
- All views are linked with `dtour.link`, so a selection in one shows in all.

### [`demo_spectral.py`](demo_spectral.py)

**Mair et al. 2022** tumor dataset via **Laplacian Eigenmaps**.

- **Data:** the same Ozette tumor sample + phenotype map as above.
- **Tour:** an 8-D LE embedding of the 18 markers, toured through consecutive
  eigenvector pairs ([LE1, LE2] → [LE2, LE3] → …). Includes vanilla, sign-flipped,
  and Fisher (label-aware) variants, plus a heatmap of eigenvector ↔ marker
  correlations.
