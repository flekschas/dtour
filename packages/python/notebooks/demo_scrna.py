import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # La Manno 2021 — scRNA-seq PCA Tour

    Interactive exploration of **293k cells** from the La Manno et al. 2021 developing mouse brain dataset:

    - **Left**: Dimensions tour of PC1-PC8 (dtour)
    - **Right**: 2D UMAP computed from PC1-PC8 (jupyter-scatter)

    Lasso-select points in the dtour widget to highlight them in the scatter view.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import dtour
    import jscatter
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    cache_dir = Path(__file__).parent / "__cache__"
    return cache_dir, dtour, jscatter, np, pa, pd


@app.cell
def _(cache_dir, pd):
    pc_cols = [f"PC{i}" for i in range(1, 9)]
    df = pd.read_parquet(cache_dir / "lamanno2021.pq").dropna(subset=pc_cols).reset_index(drop=True)
    df
    return df, pc_cols


@app.cell
def _(df, dtour):
    class_cmap = dtour.build_color_map(sorted(df["Class"].unique()), theme="dark")
    return (class_cmap,)


@app.cell
def _(cache_dir, df, mo, np, pc_cols):
    import umap

    umap_cache = cache_dir / "lamanno2021-umap-2d.npy"

    if umap_cache.exists():
        embedding_2d = np.load(umap_cache)
        mo.md(f"Loaded cached 2D UMAP embedding ({len(embedding_2d):,} points)")
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(df[pc_cols].values)
        np.save(umap_cache, embedding_2d)
        mo.md(f"Computed and cached 2D UMAP embedding ({len(embedding_2d):,} points)")
    return (embedding_2d,)


@app.cell
def _(class_cmap, df, dtour, pa, pc_cols):
    _df_tour = df[*pc_cols, "Age", "Class", "Subclass"]

    tour_widget = dtour.Widget(
        data=pa.Table.from_pandas(_df_tour),
        point_color="Class",
        point_opacity=0.5,
        color_map=class_cmap,
        tour_by="dimensions",
        tour_dimensions=pc_cols,
        preview_size="small",
        preview_count=8,
        theme="dark",
        height=1080,
    )
    return (tour_widget,)


@app.cell
def _(class_cmap, df, embedding_2d, jscatter, pd):
    scatter_df = pd.DataFrame(
        {
            "x": embedding_2d[:, 0],
            "y": embedding_2d[:, 1],
            "Age": df["Age"],
            "Class": df["Class"],
            "Subclass": df["Subclass"],
        }
    )

    scatter_2d = jscatter.Scatter(
        data=scatter_df,
        x="x",
        y="y",
        color_by="Class",
        color_map=class_cmap,
        background_color="black",
        width=960,
        height=1080,
        tooltip=True,
        axes=False,
    )
    return (scatter_2d,)


@app.cell
def _(df, scatter_2d, tour_widget):
    def _dtour_indices():
        """Effective dtour selection as a set of row indices."""
        labels = tour_widget.selected_labels
        if labels:
            return set(df.index[df["Class"].isin(labels)])
        return set(tour_widget.selected_indices)

    def handle_dtour_selection(change):
        new_sel = change.new or []
        if set(new_sel) == set(scatter_2d.selection()):
            return
        scatter_2d.selection(new_sel if new_sel else None)

    tour_widget.observe(handle_dtour_selection, names="selected_indices")

    def handle_dtour_label_selection(change):
        labels = change.new or []
        if labels:
            idx = list(df.index[df["Class"].isin(labels)])
        else:
            idx = None
        if set(idx or []) == set(scatter_2d.selection()):
            return
        scatter_2d.selection(idx)

    tour_widget.observe(handle_dtour_label_selection, names="selected_labels")

    def handle_jscatter_selection(change):
        new_sel = list(change.new) if len(change.new) else []
        if set(new_sel) == _dtour_indices():
            return
        tour_widget.select(new_sel)

    scatter_2d.widget.observe(handle_jscatter_selection, names="selection")
    return


@app.cell
def _(mo, scatter_2d, tour_widget):
    # mo.hstack(
    #     [tour_widget, mo.ui.anywidget(scatter_2d.widget)],
    #     widths=[960,960],
    #     gap=0,
    # )
    mo.Html(
        '<div style="display:flex;gap:0">'
        f'<div style="width:960px;flex-shrink:0">{mo.ui.anywidget(tour_widget)}</div>'
        f'<div style="width:960px;flex-shrink:0">{mo.ui.anywidget(scatter_2d.widget)}</div>'
        "</div>"
    )

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
