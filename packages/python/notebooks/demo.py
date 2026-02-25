import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # dtour Demo: UMAP Tour of Immune Cell Markers

        This notebook loads the Mair 2022 tumor dataset, embeds the 18 winsorized
        marker columns into 8D with UMAP, then runs a **little tour** through
        the embedding with points colored by FAUST cell-type label.
        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl
    from sklearn.preprocessing import StandardScaler

    return Path, StandardScaler, np, pl


@app.cell
def _(Path, pl):
    data_path = Path(__file__).resolve().parent / "../../../data/mair-2022-tumor-006-ozette.pq"
    df = pl.read_parquet(data_path)

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
    from umap import UMAP

    embedding = UMAP(n_components=8, random_state=42).fit_transform(X_scaled)
    return (embedding,)


@app.cell
def _(df, embedding, np, pl):
    umap_df = pl.DataFrame(
        {f"umap_{i}": embedding[:, i].astype(np.float32) for i in range(8)}
    ).with_columns(df.get_column("faustLabels"))
    umap_df.head()
    return (umap_df,)


@app.cell
def _(embedding):
    import dtour

    tour = dtour.little_tour(embedding, n_components=8)
    return dtour, tour


@app.cell
def _(dtour, tour, umap_df):
    w = dtour.DtourWidget(
        data=umap_df,
        tour=tour,
        preview_count=8,
        point_color="faustLabels",
    )
    w
    return (w,)


if __name__ == "__main__":
    app.run()
