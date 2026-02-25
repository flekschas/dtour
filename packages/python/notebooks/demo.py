import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # dtour Demo: UMAP Tour of Immune Cell Markers

        This notebook loads the [Mair 2022 tumor dataset](https://pubmed.ncbi.nlm.nih.gov/35545675/), embeds the 18 winsorized
        marker columns into 8D with [UMAP](https://umap-learn.readthedocs.io/), then runs a **little tour** through
        the embedding with points colored by [FAUST cell-type label](https://pubmed.ncbi.nlm.nih.gov/34950900/).
        """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl
    from sklearn.preprocessing import StandardScaler

    return StandardScaler, np, pl


@app.cell
def _(pl):
    df = pl.read_parquet("https://storage.googleapis.com/flekschas/jupyter-scatter-tutorial/mair-2022-tumor-006-ozette.pq")

    win_cols = [c for c in df.columns if c.endswith("Windsorized")]
    df.select(win_cols).head()
    return df, win_cols


@app.cell
def _(StandardScaler, df, np, win_cols):
    X = df.select(win_cols).to_numpy()
    X_scaled = StandardScaler().fit_transform(X).astype(np.float32)
    return (X_scaled,)


@app.cell
def _(X_scaled):
    import dtour

    tour = dtour.little_umap_tour(X_scaled, n_components=8)
    return


app._unparsable_cell(
    r"""
        w = dtour.Widget(
            data=tour.embedding,
            tour=tour,
            preview_count=8,
            point_color="faustLabels",
        )
        w
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
