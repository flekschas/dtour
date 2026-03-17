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
        metric_tracks=[{"metric": "confusion", "height": 64, "domain": [0, 1]}],
        height=960,
        theme="light",
    )
    w
    return (w,)


@app.cell
def _(cache_dir, df, dtour, np, tour, w):
    _metric_names = ["confusion"]
    metrics_path = cache_dir / "metrics.npz"
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
def _(cache_dir, df, dtour, metrics, np, tour, w):
    import cev_metrics
    import pandas as pd

    _original_confusion = metrics.values["confusion"]
    _labels = df["faustLabels"].to_numpy()
    _exclude = {"0_0_0_0_0"}
    _mask = np.array([lbl not in _exclude for lbl in _labels])
    _X = tour.embedding[_mask]
    _labels_clean = _labels[_mask]

    # Precompute confusion matrices for all views (expensive, done once)
    _cm_cache_path = cache_dir / "confusion_matrices.npz"
    if _cm_cache_path.exists():
        _cm_data = np.load(_cm_cache_path)
        _confusion_matrices = [_cm_data[f"cm_{i}"] for i in range(len(tour.views))]
    else:
        _cat = pd.Categorical(_labels_clean)
        _confusion_matrices = []
        for basis in tour.views:
            proj = _X @ basis
            cm_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "label": _cat})
            _confusion_matrices.append(np.asarray(cev_metrics.confusion(cm_df), dtype=np.float64))
        np.savez_compressed(
            _cm_cache_path, **{f"cm_{i}": cm for i, cm in enumerate(_confusion_matrices)}
        )

    _cat_labels = sorted(set(_labels_clean))
    _confusion_cache: dict[frozenset, list[float]] = {}

    def _on_selection(change):
        selected = change["new"]
        if not selected:
            w.set_metrics(
                dtour.MetricResult(
                    values={**metrics.values, "confusion": _original_confusion},
                    metric_names=metrics.metric_names,
                )
            )
            return

        key = frozenset(selected)
        if key not in _confusion_cache:
            sel_idx = [_cat_labels.index(s) for s in selected if s in _cat_labels]
            if not sel_idx:
                return
            vals = []
            for cm in _confusion_matrices:
                rows = cm[sel_idx, :]
                total = rows.sum()
                diag = sum(cm[i, i] for i in sel_idx)
                vals.append(0.0 if total == 0 else float(1.0 - diag / total))
            _confusion_cache[key] = vals

        w.set_metrics(
            dtour.MetricResult(
                values={**metrics.values, "confusion": _confusion_cache[key]},
                metric_names=metrics.metric_names,
            )
        )

    w.observe(_on_selection, names=["selected_labels"])
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl
    from sklearn.preprocessing import StandardScaler

    cache_dir = Path(__file__).parent / "__cache__"
    cache_dir.mkdir(exist_ok=True)

    return Path, StandardScaler, cache_dir, np, pl


@app.cell
def _(cache_dir, pl):
    _data_url = "https://storage.googleapis.com/flekschas/jupyter-scatter-tutorial/mair-2022-tumor-006-ozette.pq"
    _local_pq = cache_dir / "mair-2022-tumor-006-ozette.pq"

    if _local_pq.exists():
        df = pl.read_parquet(_local_pq)
    else:
        df = pl.read_parquet(_data_url)
        df.write_parquet(_local_pq)

    win_cols = [c for c in df.columns if c.endswith("Windsorized")]
    df.select(win_cols).head()
    return df, win_cols


@app.cell
def _(StandardScaler, df, np, win_cols):
    X = df.select(win_cols).to_numpy()
    X_scaled = StandardScaler().fit_transform(X).astype(np.float32)
    return (X_scaled,)


@app.cell
def _(X_scaled, cache_dir):
    import dtour

    tour_path = cache_dir / "tour.npz"
    if tour_path.exists():
        tour = dtour.TourResult.load(tour_path)
    else:
        tour = dtour.little_umap_tour(X_scaled, n_components=8)
        tour.save(tour_path)
    return dtour, tour


if __name__ == "__main__":
    app.run()
