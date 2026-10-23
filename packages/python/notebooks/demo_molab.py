# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anywidget==0.11.0",
#     "dtour[cev]",
#     "jupyter-scatter>=0.22.3",
#     "marimo>=0.25.2",
#     "matplotlib>=3.10.8",
#     "polars>=1.38.1",
#     "pyarrow>=14",
#     "traitlets==5.16.1",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # dtour in Marimo

    [dtour](https://dtour.dev) is a tool for exploring high-dimensional data through interactive tours.
    A single 2D projection or embedding can only capture a fraction of high-dimensional
    structure. dtour smoothly animates through a sequence of 2D projections instead, so you can build a
    sense for the full space: play and rewind a tour, drag the axes to project by hand, or
    watch an endless grand tour.

    dtour comes as widget such that you can integrate it in your analysis: what you select in
    the widget is exposed as a traitlet, so your code can act on it. The three examples
    below show how selections can:

    1. **Drive other views**: highlight the selected brain cells in a 2D UMAP and list the
       cell types they are enriched for
    2. **Update the tour itself**: show how intermixed the selected immune cell phenotypes
       are in each projection, and which markers set selected cells apart
    3. **Feed custom widgets**: show the selected handwritten digits as images and which
       digits they look like

    /// note | Precomputed embeddings
    To start in seconds, this notebook loads precomputed UMAP embeddings. Everything
    else, including the tours, runs live. The [full notebooks](https://github.com/flekschas/dtour/tree/main/packages/python/notebooks)
    compute everything from the raw data.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Developing mouse brain

    276K cells from the [La Manno et al. 2021](https://pubmed.ncbi.nlm.nih.gov/34321664/)
    developing mouse brain atlas: a tour through PC1-PC8 (left) and a 2D UMAP of the
    same PCs (right), with linked selections.
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def _(brain_umap, brain_widget, mo):
    mo.hstack([brain_widget, brain_umap.widget], widths="equal", gap=0)
    return


@app.cell(hide_code=True)
def _(brain_enrichment):
    # Not part of the cell above, as selections rerun this cell, and jupyter-scatter
    # stops responding when the cell that shows it reruns
    brain_enrichment
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Try this:**

    - Click a class in the legend, e.g., _Neuron_, to see where its cells are in the UMAP
      and which subclasses make it up.
    - Press and hold in either view, then drag to lasso a group of cells. The table under
      the views lists the subclasses the selection is enriched for.
    - Press play to tour through the PCs and see which projection separates your
      selection best.
    """)
    return


@app.cell
def _(brain, brain_selection, mo, pl):
    # The subclasses the selected cells are enriched for
    _rows = brain_selection()
    if len(_rows):
        _selection = brain[_rows]
        _enrichment = (
            _selection.group_by("Class", "Subclass")
            .len("Cells")
            .join(brain.group_by("Subclass").len("All"), on="Subclass")
            .select(
                "Class",
                "Subclass",
                "Cells",
                Share=pl.col("Cells") / len(_selection),
                Enrichment=(pl.col("Cells") / len(_selection)) / (pl.col("All") / len(brain)),
            )
            .sort("Cells", descending=True)
            .head(10)
        )
        brain_enrichment = mo.vstack(
            [
                mo.md(
                    f"**{len(_selection):,} selected cells** "
                    f"(median age E{_selection['Age'].median():.1f}), "
                    "by their most frequent subclasses:"
                ),
                mo.ui.table(
                    _enrichment,
                    selection=None,
                    format_mapping={"Share": "{:.1%}".format, "Enrichment": "{:.1f}x".format},
                ),
            ]
        )
    else:
        brain_enrichment = mo.md("_Select cells to see which subclasses they are enriched for._")
    return (brain_enrichment,)


@app.cell
def _(DATA_URL, dtour, pl):
    brain = pl.read_parquet(f"{DATA_URL}/notebooks/lamanno2021-pca-umap.pq")
    pcs = [f"PC{i}" for i in range(1, 9)]
    class_colors = dtour.build_color_map(sorted(brain["Class"].unique()), theme="light")
    return brain, class_colors, pcs


@app.cell
def _(brain, class_colors, dtour, jscatter, pcs, selection):
    brain_widget = dtour.Widget(
        data=brain.select(*pcs, "Age", "Class", "Subclass"),
        tour=dtour.little_tour(brain.select(pcs)),
        tour_dimensions=pcs,
        point_color_by="Class",
        point_opacity=0.5,
        color_map=class_colors,
        preview_size="small",
        preview_count=8,
        theme_mode="light",
        height=600,
    )
    brain_selection = selection(brain_widget, brain, "Class")
    brain_umap = jscatter.Scatter(
        data=brain.select("UMAP1", "UMAP2", "Class", "Subclass").to_pandas(),
        x="UMAP1",
        y="UMAP2",
        color_by="Class",
        color_map=class_colors,
        height=600,
        axes=False,
        tooltip=True,
    )
    return brain_selection, brain_umap, brain_widget


@app.cell
def _(brain, brain_umap, brain_widget, selected_rows):
    # Keep the selections of both views in sync
    def _to_umap(_):
        rows = selected_rows(brain_widget, brain, "Class")
        if set(rows) != set(brain_umap.selection()):
            brain_umap.selection(rows if len(rows) else None)

    def _to_tour(change):
        if set(change.new) != set(selected_rows(brain_widget, brain, "Class")):
            brain_widget.select(change.new)

    brain_widget.observe(_to_umap, names=["selected_indices", "selected_labels"])
    brain_umap.widget.observe(_to_tour, names="selection")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tumor immune cells

    346K cells from a tumor sample of the [Mair et al. 2022](https://pubmed.ncbi.nlm.nih.gov/35545675/)
    CyTOF dataset, colored by [FAUST](https://pubmed.ncbi.nlm.nih.gov/34950900/)-derived
    phenotypes: a tour through an 8D UMAP of 18 markers. The bars around the circle show
    how intermixed the phenotypes are in each keyframe
    ([confusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC11875997/), computed live).
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def _(cells_profile, cells_widget, mo):
    mo.vstack([cells_widget, cells_profile])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Try this:**

    - Click a phenotype in the legend, e.g., _Treg_. The bars now show how intermixed
      just that phenotype is in each keyframe. Click a keyframe with a short bar to jump
      to the projection that separates it best.
    - Click more phenotypes to see their mean confusion.
    - Press and hold, then drag to lasso a group of cells. The chart under the tour shows
      which markers set them apart from all other cells.
    """)
    return


@app.cell
def _(cells, cells_selection, markers, mo, plt):
    # How far the mean marker expression of the selected cells is from all cells
    _rows = cells_selection()
    if len(_rows):
        _all = cells.select(markers)
        _shift = ((_all[_rows].mean() - _all.mean()) / _all.std()).row(0)
        _order = sorted(range(len(markers)), key=lambda i: _shift[i])
        _fig, _ax = plt.subplots(figsize=(6, 4.5))
        _ax.barh(
            [markers[i] for i in _order],
            [_shift[i] for i in _order],
            color=["#0072b2" if _shift[i] < 0 else "#d55e00" for i in _order],
        )
        _ax.axvline(0, color="black", linewidth=0.5)
        _ax.set_xlabel("Mean marker expression vs. all cells (standard deviations)")
        _ax.spines[["top", "right"]].set_visible(False)
        _fig.tight_layout()
        _top = cells[_rows]["phenotypes"].value_counts(sort=True).head(3)
        cells_profile = mo.vstack(
            [
                mo.md(
                    f"**{len(_rows):,} selected cells**, mostly "
                    + ", ".join(f"{p} ({n:,})" for p, n in _top.iter_rows())
                ),
                _fig,
            ]
        )
    else:
        cells_profile = mo.md("_Select cells to see their marker profile._")
    return (cells_profile,)


@app.cell
def _(
    cells_selection,
    cells_widget,
    dtour,
    per_phenotype_confusion,
    phenotypes,
):
    # The bars show the mean confusion of the clicked phenotypes, or of all of them.
    # Reading the selection reruns this cell when it changes.
    cells_selection()
    _clicked = [phenotypes.index(p) for p in cells_widget.selected_labels if p in phenotypes]
    _confusion = per_phenotype_confusion[:, _clicked or slice(None)].mean(axis=1)
    cells_widget.set_metrics(
        dtour.MetricResult(values={"confusion": _confusion.tolist()}, metric_names=["confusion"])
    )
    return


@app.cell
def _(DATA_URL, dtour, pl):
    cells = pl.read_parquet(f"{DATA_URL}/notebooks/mair-2022-tumor-umap-8d.pq")
    umaps = [c for c in cells.columns if c.startswith("UMAP")]
    markers = [c for c in cells.columns if c not in umaps and c != "phenotypes"]
    phenotype_colors = dtour.build_color_map(
        sorted(cells["phenotypes"].unique()),
        theme="light",
        overrides={"Unassigned": "#b0b0b0"},
    )
    return cells, markers, phenotype_colors, umaps


@app.cell
def _(cells, dtour, phenotype_colors, selection, umaps):
    cells_tour = dtour.little_tour(cells.select(umaps))
    cells_widget = dtour.Widget(
        data=cells.select(*umaps, "phenotypes"),
        tour=cells_tour,
        tour_dimensions=umaps,
        point_color_by="phenotypes",
        color_map=phenotype_colors,
        preview_size="small",
        preview_count=8,
        metric_bar_width=24,
        metric_tracks=[{"metric": "confusion", "height": 64, "domain": [0, 1]}],
        theme_mode="light",
        height=720,
    )
    cells_selection = selection(cells_widget, cells, "phenotypes")
    return cells_selection, cells_tour, cells_widget


@app.cell
def _(cells, cells_tour, np, pd, pl, umaps):
    import cev_metrics

    # Label confusion matrix of every keyframe, ignoring unassigned cells. Projects
    # like the viewer, which scales each dimension to [-0.5, 0.5].
    _assigned = cells.filter(pl.col("phenotypes") != "Unassigned")
    _X = _assigned.select(umaps).to_numpy()
    _X = (_X - _X.min(axis=0)) / np.ptp(_X, axis=0) - 0.5
    _labels = pd.Categorical(_assigned["phenotypes"].cast(pl.String).to_numpy())
    phenotypes = list(_labels.categories)
    per_phenotype_confusion = []
    for _basis in cells_tour.keyframes:
        _xy = _X @ _basis
        _cm = np.asarray(
            cev_metrics.confusion(pd.DataFrame({"x": _xy[:, 0], "y": _xy[:, 1], "label": _labels}))
        )
        per_phenotype_confusion.append(1 - np.diag(_cm) / _cm.sum(axis=1))
    per_phenotype_confusion = np.array(per_phenotype_confusion)
    return per_phenotype_confusion, phenotypes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handwritten digits

    The 1,797 8x8 images of scikit-learn's
    [digits dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset):
    a tour through the top principal components of the 64 pixels (left), a 2D UMAP of
    the pixels (middle), and the selected images (right). The two dtour views are linked
    with `dtour.link`.
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def _(digits_gallery, digits_similarity, digits_tour, digits_umap, mo):
    mo.vstack(
        [
            mo.hstack([digits_tour, digits_umap, digits_gallery], widths=[2, 2, 1], gap=0),
            digits_similarity,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Try this:**

    - Hover over a point to see its image.
    - Press and hold, then drag to lasso a cluster in either view. Its images appear on
      the right, and the matrix under the views shows which digits the selected images
      look like.
    - Lasso where two clusters touch, e.g., between 1s and 8s, to find the digits that
      are easy to mix up.
    """)
    return


@app.cell
def _(digit_colors, digits, digits_gallery, digits_selection):
    # Show the selected images next to the views
    digit_rows = digits_selection()
    digits_gallery.images = [[int(i), digit_colors[digits["digit"][int(i)]]] for i in digit_rows]
    return (digit_rows,)


@app.cell
def _(digit_rows, digits, mo, neighbors, np, plt):
    # For each selected image, the digits of its 10 most similar images by pixels
    if len(digit_rows):
        _labels = digits["digit"].cast(int).to_numpy()
        _counts = np.zeros((10, 10))
        np.add.at(
            _counts,
            (
                np.repeat(_labels[digit_rows], neighbors.shape[1]),
                _labels[neighbors[digit_rows]].ravel(),
            ),
            1,
        )
        _shown = np.flatnonzero(_counts.sum(axis=1))
        _shares = _counts[_shown] / _counts[_shown].sum(axis=1, keepdims=True)
        _fig, _ax = plt.subplots(figsize=(5, 0.5 * len(_shown) + 1))
        _ax.imshow(_shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (_r, _c), _share in np.ndenumerate(_shares):
            if _share >= 0.01:
                _ax.text(
                    _c,
                    _r,
                    f"{_share:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if _share > 0.5 else "black",
                )
        _ax.set_xticks(range(10))
        _ax.set_yticks(range(len(_shown)), _shown)
        _ax.set_xlabel("Digit of the 10 most similar images")
        _ax.set_ylabel("Selected digit")
        _fig.tight_layout()
        digits_similarity = mo.vstack([mo.md(f"**{len(digit_rows):,} selected images**"), _fig])
    else:
        digits_similarity = mo.md("_Select images to see which digits they look like._")
    return (digits_similarity,)


@app.cell
def _(DATA_URL, dtour, pl):
    from sklearn.datasets import load_digits

    # The UMAP rows are in the same order as scikit-learn's images
    _umap = pl.read_parquet(f"{DATA_URL}/digits-umap.pq")
    _pixels = load_digits().data
    pixels = [f"px{i}" for i in range(_pixels.shape[1])]
    digits = pl.concat([pl.DataFrame(_pixels, schema=pixels), _umap], how="horizontal")
    digit_colors = dtour.build_color_map(sorted(digits["digit"].unique()), theme="light")
    return digit_colors, digits, pixels


@app.cell
def _(digit_colors, digits, dtour, pixels, selection):
    _settings = dict(
        point_color_by="digit",
        color_map=digit_colors,
        tooltip_columns=["image", "digit"],
        theme_mode="light",
        height=480,
    )
    digits_tour = dtour.Widget(
        data=digits.select(*pixels, "digit", "image"),
        tour=dtour.little_tour(digits.select(pixels), n_components=6),
        tour_dimensions=pixels,
        preview_count=6,
        preview_size="small",
        **_settings,
    )
    digits_umap = dtour.Widget(
        data=digits.select("umap_x", "umap_y", "digit", "image"),
        tour_dimensions=["umap_x", "umap_y"],
        show_legend=False,
        **_settings,
    )
    dtour.link(digits_tour, digits_umap)
    digits_selection = selection(digits_tour, digits, "digit")
    return digits_selection, digits_tour, digits_umap


@app.cell
def _(digits, pixels):
    from sklearn.neighbors import NearestNeighbors

    # The 10 most similar images of every image, by pixels
    neighbors = (
        NearestNeighbors(n_neighbors=10)
        .fit(digits.select(pixels))
        .kneighbors(return_distance=False)
    )
    return (neighbors,)


@app.cell
def _():
    from anywidget import AnyWidget
    from traitlets import List

    class Gallery(AnyWidget):
        """The digit images of `images`, a list of [row, color] pairs."""

        _esm = """
        export default {
          render({ model, el }) {
            el.classList.add("gallery");
            const show = () => {
              const images = model.get("images");
              el.replaceChildren(...images.slice(0, 200).map(([row, color]) => {
                const frame = document.createElement("div");
                frame.style.borderColor = color;
                const img = frame.appendChild(document.createElement("img"));
                img.src = `https://data.dtour.dev/digits/${String(row).padStart(4, "0")}.png`;
                return frame;
              }));
              if (!images.length) el.textContent = "Select images to see them here";
            };
            model.on("change:images", show);
            show();
          },
        };
        """
        _css = """
        .gallery {
          display: flex;
          flex-wrap: wrap;
          align-content: flex-start;
          gap: 4px;
          height: 480px;
          overflow: auto;
          padding-left: 8px;
          color: #888;
          font-size: 0.85rem;
        }
        .gallery div {
          border: 2px solid;
          line-height: 0;
        }
        .gallery img {
          width: 32px;
          height: 32px;
          image-rendering: pixelated;
          filter: invert(1);
        }
        """
        images = List().tag(sync=True)

    digits_gallery = Gallery()
    return (digits_gallery,)


@app.cell
def _():
    import dtour
    import jscatter
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl

    DATA_URL = "https://data.dtour.dev"

    def selected_rows(widget, df, column):
        """Rows selected in *widget*, by lasso or by clicking labels of *column*."""
        if widget.selected_labels:
            return np.flatnonzero(
                df[column].cast(pl.String).is_in(widget.selected_labels).to_numpy()
            )
        return np.asarray(widget.selected_indices, dtype=int)

    def selection(widget, df, column):
        """A state of the rows selected in *widget*, by lasso or by clicking labels of *column*.

        Cells that read it rerun only when the selection changes, while cells that
        reference `mo.ui.anywidget(widget)` rerun on any change, e.g., of the camera.
        """
        get_rows, set_rows = mo.state(selected_rows(widget, df, column))
        widget.observe(
            lambda _: set_rows(selected_rows(widget, df, column)),
            names=["selected_indices", "selected_labels"],
        )
        return get_rows

    return (
        DATA_URL,
        dtour,
        jscatter,
        mo,
        np,
        pd,
        pl,
        plt,
        selected_rows,
        selection,
    )


if __name__ == "__main__":
    app.run()
