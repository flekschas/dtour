# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "dtour>=0.5.1",
#     "marimo",
#     "pandas>=2.0",
#     "pyarrow>=14",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SHERLOCK — Latent Space vs. Expression Space

    [SHERLOCK](https://github.com/azizilab/SHERLOCK) ([Zhang, Myers, Shi et al. 2026](https://doi.org/10.64898/2026.09.25.754573))
    learns a 16-dimensional latent cell state from Perturb-seq screens. This notebook compares that latent
    space with the standard expression space on the same perturbed cells:

    - **Expression (PCA):** normalize, log-transform, PCA
    - **SHERLOCK:** the latent space of a model trained with the manuscript's settings

    Two public datasets are available: [Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013)
    (genome-scale CRISPRi, cells whose target gene is in one of eight annotated pathways) and
    [Norman et al. 2019](https://doi.org/10.1126/science.aax4438) (combinatorial CRISPRa, cells whose
    perturbation includes a gene from the Table S6 clusters). The embeddings are precomputed, since
    training SHERLOCK takes one to two hours on a CPU.

    All views show the same cells and share their selection: lasso-select points or click a legend
    entry in one view to select the same cells everywhere.
    """)
    return


@app.cell
def _(mo):
    dataset = mo.ui.dropdown(["replogle", "norman"], value="replogle", label="Dataset")
    dataset
    return (dataset,)


@app.cell
def _():
    import re
    from pathlib import Path

    import dtour
    import pandas as pd

    cache_dir = Path(__file__).parent / "__cache__"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir, dtour, pd, re


@app.cell
def _(cache_dir, dataset, pd):
    _name = f"sherlock_{dataset.value}.pq"
    _local_pq = cache_dir / _name

    if _local_pq.exists():
        df = pd.read_parquet(_local_pq)
    else:
        df = pd.read_parquet(
            f"https://data.dtour.dev/notebooks/{_name}", storage_options={"User-Agent": "dtour"}
        )
        df.to_parquet(_local_pq)

    label_cols = list(df.select_dtypes("category").columns)
    df
    return df, label_cols


@app.cell
def _(label_cols, mo):
    color_by = mo.ui.dropdown(label_cols, value=label_cols[1], label="Color by")
    color_by
    return (color_by,)


@app.cell
def _(df, dtour, label_cols, re):
    color_maps = {col: dtour.build_color_map(sorted(df[col].cat.categories)) for col in label_cols}

    def columns(prefix):
        return [c for c in df.columns if re.fullmatch(rf"{prefix}\d+", c)]

    def make_widget(data, tour, **settings):
        return dtour.Widget(
            data,
            tour,
            **{
                "point_color_by": label_cols[1],
                "color_map": color_maps[label_cols[1]],
                "point_opacity": 0.6,
                "preview_size": "small",
                "theme_mode": "light",
                "height": 560,
                **settings,
            },
        )

    def dimension_tour(prefix):
        """Little tour through consecutive pairs of a space's principal axes."""
        cols = columns(prefix)
        return make_widget(df[cols + label_cols], dtour.little_tour(df[cols]))

    def morph_tour(frames, keyframe_descriptions, description, **settings):
        """Morph between 2D embeddings, given as (x, y) column-name pairs."""
        tour = dtour.sequential_tour(
            [df[list(frame)] for frame in frames],
            # The embeddings are precomputed, so each frame is its own embedding
            method=lambda embedding, _previous: embedding,
            keyframe_descriptions=keyframe_descriptions,
            description=description,
        )
        return make_widget(df[label_cols], tour, **settings)

    return color_maps, dimension_tour, morph_tour


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Watching SHERLOCK learn

    The latent states of these cells were saved every 5 epochs during training. The tour shows
    32 of them: every snapshot up to epoch 95, where the latent space changes fastest, then evenly
    spaced ones through the last epoch. Each keyframe is a 2D UMAP of one snapshot, initialized
    from the previous one so that points only move as much as the latent space changes. Each
    preview shows one snapshot, labeled by its epoch. Drag the slider to watch the perturbations
    separate as the model trains.
    """)
    return


@app.cell(expand_output=True)
def _(df, morph_tour, re):
    _epochs = sorted(
        int(m[1]) for c in df.columns if (m := re.fullmatch(r"training_epoch(\d+)_1", c))
    )
    learning = morph_tour(
        [(f"training_epoch{e}_1", f"training_epoch{e}_2") for e in _epochs],
        [f"Epoch {e}" for e in _epochs],
        "SHERLOCK latent UMAP during training",
        height=800,
        # Fits all 32 previews, which "small" doesn't
        preview_size="large",
    )
    learning
    return (learning,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. From the expression UMAP to the SHERLOCK UMAP

    Drag the slider to morph between the 2D UMAP of the expression PCA and the 2D UMAP of the
    trained SHERLOCK latent space. The second embedding is rotated or mirrored (Procrustes) to best
    match the first, so that only the change in structure moves the points.
    """)
    return


@app.cell
def _(morph_tour):
    morph = morph_tour(
        [("pca_umap2d_1", "pca_umap2d_2"), ("sherlock_umap2d_1", "sherlock_umap2d_2")],
        ["Expression UMAP", "SHERLOCK UMAP"],
        "Expression UMAP ↔ SHERLOCK latent UMAP",
    )
    morph
    return (morph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Touring the spaces themselves

    Instead of a UMAP of each space, tour the spaces directly: the first 16 principal components
    of expression (left) and SHERLOCK's 16 latent dimensions (right). Each keyframe projects a
    space onto a pair of its principal axes, from (1, 2) and (2, 3) through (10, 1), covering the
    10 axes with the most spread.
    """)
    return


@app.cell
def _(dimension_tour, mo):
    pca_tour = dimension_tour("pca_")
    sherlock_tour = dimension_tour("sherlock_z_")
    mo.hstack([pca_tour, sherlock_tour], widths="equal")
    return pca_tour, sherlock_tour


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Touring 4D UMAPs

    A middle ground: UMAP each space to four dimensions instead of two (from the same neighbor
    graph as the 2D UMAPs), then tour those. The four keyframes pair the principal axes of each
    4D UMAP: (1, 2), (2, 3), (3, 4), and (4, 1). Left: expression; right: SHERLOCK.
    """)
    return


@app.cell
def _(dimension_tour, mo):
    pca_umap_tour = dimension_tour("pca_umap4d_")
    sherlock_umap_tour = dimension_tour("sherlock_umap4d_")
    mo.hstack([pca_umap_tour, sherlock_umap_tour], widths="equal")
    return pca_umap_tour, sherlock_umap_tour


@app.cell
def _(
    dtour,
    learning,
    morph,
    pca_tour,
    pca_umap_tour,
    sherlock_tour,
    sherlock_umap_tour,
):
    widgets = [learning, morph, pca_tour, sherlock_tour, pca_umap_tour, sherlock_umap_tour]
    _unlink = dtour.link(*widgets)
    return (widgets,)


@app.cell
def _(color_by, color_maps, widgets):
    for _w in widgets:
        _w.color_map = color_maps[color_by.value]
        _w.point_color_by = color_by.value
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
