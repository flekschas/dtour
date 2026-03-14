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


@app.cell(hide_code=True)
def _(df, dtour, pl, tour):
    # Combine embedding columns with the categorical label for coloring.
    # The scatter engine uses numeric columns for projection and categorical
    # columns for per-point color encoding.
    widget_df = pl.DataFrame(
        {f"dim_{i}": tour.embedding[:, i] for i in range(tour.embedding.shape[1])}
    ).with_columns(df["faustLabels"])

    w = dtour.Widget(
        data=widget_df,
        tour=tour,
        preview_count=8,
        point_color="faustLabels",
        metric_bar_width=24,
        height=960,
    )
    w
    return (w,)


@app.cell
def _(Path, df, dtour, np, tour, w):
    _metric_names = ["neighborhood_hit", "confusion", "hdbscan_score"]
    metrics_path = Path("metrics_cache.npz")
    if metrics_path.exists():
        _cached = np.load(metrics_path)
        metrics = dtour.MetricResult(
            values={name: _cached[name].tolist() for name in _metric_names},
            metric_names=_metric_names,
        )
    else:
        metrics = dtour.compute_metrics(
            tour.embedding,
            tour.views,
            labels=df["faustLabels"].to_numpy(),
            metrics=_metric_names,
            exclude_labels=["0_0_0_0_0"],
        )
        np.savez_compressed(metrics_path, **{name: vals for name, vals in metrics.values.items()})
    w.set_metrics(metrics)
    return (metrics,)


@app.cell
def _(metrics):
    metrics
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl
    from sklearn.preprocessing import StandardScaler

    return Path, StandardScaler, np, pl


@app.cell
def _(pl):
    df = pl.read_parquet(
        "https://storage.googleapis.com/flekschas/jupyter-scatter-tutorial/mair-2022-tumor-006-ozette.pq"
    )

    win_cols = [c for c in df.columns if c.endswith("Windsorized")]
    df.select(win_cols).head()
    return df, win_cols


@app.cell
def _(StandardScaler, df, np, win_cols):
    X = df.select(win_cols).to_numpy()
    X_scaled = StandardScaler().fit_transform(X).astype(np.float32)
    return (X_scaled,)


@app.cell
def _(Path, X_scaled):
    import dtour

    tour_path = Path("tour_cache.npz")
    if tour_path.exists():
        tour = dtour.TourResult.load(tour_path)
    else:
        tour = dtour.little_umap_tour(X_scaled, n_components=8)
        tour.save(tour_path)
    return dtour, tour


if __name__ == "__main__":
    app.run()
