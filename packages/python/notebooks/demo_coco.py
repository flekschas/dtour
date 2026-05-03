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
    # ShareGPT4V x COCO — dtour + jupyter-scatter viewer

    Interactive exploration of the **alpha=0.5** joint pixel + caption embedding:

    - **Left**: Signed Laplacian Eigenmaps tour (label-aware structure via dtour)
    - **Middle**: 2D DensMAP UMAP at alpha=0.5 (jupyter-scatter)
    - **Right**: 4D DensMAP UMAP tour at alpha=0.5 (dtour, dimensions tour)

    Lasso-select points in either dtour widget to highlight them across all views.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache_dir = Path(__file__).parent / "__cache__"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir, pa, pd, pq


@app.cell
def _(cache_dir, mo, pd, pq):
    # Labels (from the combined parquet with all alphas)
    df_labels = pd.read_parquet(
        cache_dir / "joint-embeddings-umap-dense-2d-all-alphas.pq",
        columns=["coco_label_idf"],
    )

    # 2D UMAP at alpha=0.5
    df_2d = pd.read_parquet(cache_dir / "joint-embeddings-umap-dense-2d-alpha0.50.pq")

    # 4D UMAP at alpha=0.5
    df_4d = pd.read_parquet(cache_dir / "joint-embeddings-umap-dense-4d-alpha0.50.pq")

    # Signed LE tour (precomputed with dtour spec in metadata)
    le_signed_table = pq.read_table(cache_dir / "joint-embeddings-le-signed-tour-alpha0.50.pq")

    mo.md(f"""
    Loaded **{len(df_2d):,}** points (2D UMAP),
    **{len(df_4d):,}** (4D UMAP),
    **{le_signed_table.num_rows:,}** (signed LE)
    """)
    return df_2d, df_4d, df_labels, le_signed_table


@app.cell
def _(le_signed_table):
    import dtour

    le_tour = dtour.TourResult.from_parquet(le_signed_table)

    le_widget = dtour.Widget(
        data=le_signed_table,
        tour=le_tour,
        point_color_by="coco_label_idf",
        point_opacity=0.5,
        preview_size="small",
        theme="light",
        height=720,
    )
    return dtour, le_tour, le_widget


@app.cell
def _(df_4d, df_labels, dtour, pa, pd):
    _df_tour = pd.DataFrame(
        {
            "d0": df_4d["umap_0"].values,
            "d1": df_4d["umap_1"].values,
            "d2": df_4d["umap_2"].values,
            "d3": df_4d["umap_3"].values,
            "coco_label_idf": df_labels["coco_label_idf"].values,
        }
    )

    umap_widget = dtour.Widget(
        data=pa.Table.from_pandas(_df_tour),
        point_color_by="coco_label_idf",
        point_opacity=0.5,
        tour_by="dimensions",
        preview_size="small",
        theme="light",
        height=720,
    )
    return (umap_widget,)


@app.cell
def _(df_2d, df_labels, pd):
    import jscatter

    scatter_df = pd.DataFrame(
        {
            "x": df_2d["umap_0"].values,
            "y": df_2d["umap_1"].values,
            "coco_label_idf": df_labels["coco_label_idf"].values,
        }
    )

    scatter_2d = jscatter.Scatter(
        data=scatter_df,
        x="x",
        y="y",
        color_by="coco_label_idf",
        height=720,
    )
    return (scatter_2d,)


@app.cell
def _(le_widget, scatter_2d, umap_widget):
    _sel = le_widget.selected_indices
    umap_widget.select(_sel)
    scatter_2d.selection(_sel if _sel else None)
    return


@app.cell
def _(scatter_2d, umap_widget):
    _sel = umap_widget.selected_indices
    if _sel:
        scatter_2d.selection(_sel)
    return


@app.cell
def _(le_widget, mo, scatter_2d, umap_widget):
    mo.hstack(
        [le_widget, mo.ui.anywidget(scatter_2d.widget), umap_widget],
        widths=[1, 1, 1],
        gap=0,
    )
    return


if __name__ == "__main__":
    app.run()
