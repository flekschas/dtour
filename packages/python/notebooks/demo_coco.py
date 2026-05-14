import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ShareGPT4V x COCO — dtour + jupyter-scatter viewer

    Interactive exploration of the **alpha=0.5** joint pixel + caption embedding:

    - **Left**: Fisher Laplacian Eigenmaps tour (label-aware structure via dtour)
    - **Middle**: 2D DensMAP UMAP at alpha=0.5 (jupyter-scatter)
    - **Right**: 4D DensMAP UMAP tour at alpha=0.5 (dtour, dimensions tour)

    Lasso-select points in either dtour widget to highlight them across all views.
    """)
    return


@app.cell(hide_code=True)
def _(le_widget, mo, scatter_2d, umap_widget):
    mo.hstack([le_widget, mo.ui.anywidget(scatter_2d.widget), umap_widget], widths=[1, 1, 1], gap=0)
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
    _base_url = "https://data.dtour.dev/notebooks"

    def _load_pq(name, columns=None):
        local = cache_dir / name
        if local.exists():
            return pd.read_parquet(local, columns=columns)
        df = pd.read_parquet(f"{_base_url}/{name}", columns=columns)
        df.to_parquet(local)
        return df

    def _load_pq_table(name):
        local = cache_dir / name
        if local.exists():
            return pq.read_table(local)
        table = pq.read_table(f"{_base_url}/{name}")
        pq.write_table(table, local)
        return table

    # Labels (from the combined parquet with all alphas)
    df_labels = _load_pq("joint-embeddings-umap-dense-2d-all-alphas.pq", columns=["coco_label_idf"])

    # 2D UMAP at alpha=0.5
    df_2d = _load_pq("joint-embeddings-umap-dense-2d-alpha0.50.pq")

    # 4D UMAP at alpha=0.5
    df_4d = _load_pq("joint-embeddings-umap-dense-4d-alpha0.50.pq")

    # Signed LE tour (precomputed with dtour spec in metadata)
    le_fisher_table = _load_pq_table("joint-embeddings-le-fisher-tour-alpha0.50.pq")

    mo.md(f"""
    Loaded **{len(df_2d):,}** points (2D UMAP),
    **{len(df_4d):,}** (4D UMAP),
    **{le_fisher_table.num_rows:,}** (fisher LE)
    """)
    return df_2d, df_4d, df_labels, le_fisher_table


@app.cell
def _(le_fisher_table):
    import dtour

    le_tour = dtour.TourResult.from_parquet(le_fisher_table)

    le_widget = dtour.Widget(
        data=le_fisher_table,
        tour=le_tour,
        point_color_by="coco_label_idf",
        point_opacity=0.5,
        preview_size="small",
        theme="light",
        height=720,
    )
    return dtour, le_widget


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
        tooltip=True,
        axes=False
    )
    return (scatter_2d,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
