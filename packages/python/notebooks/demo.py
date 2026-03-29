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
def _(df, dtour, phenotype_colors, phenotypes, pl, tour):
    # Combine embedding columns with the categorical label for coloring.
    # The scatter engine uses numeric columns for projection and categorical
    # columns for per-point color encoding.
    widget_df = pl.DataFrame(
        {f"dim_{i}": tour.embedding[:, i] for i in range(tour.embedding.shape[1])}
    ).with_columns(df["faustLabels"], phenotypes)

    w = dtour.Widget(
        data=widget_df,
        tour=tour,
        preview_count=8,
        preview_size="small",
        point_color="phenotypes",
        color_map=phenotype_colors,
        metric_bar_width=24,
        metric_tracks=[{"metric": "confusion", "height": 64, "domain": [0, 1]}],
        camera_zoom=0.5,
        height=960,
        theme="light",
    )
    w
    return w, widget_df


@app.cell
def _(X_scaled, cache_dir, np, phenotype_colors, phenotypes):
    # Classic UMAP Plot
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from umap import UMAP

    umap2d_path = cache_dir / "umap_2d.npy"
    if umap2d_path.exists():
        umap_2d = np.load(umap2d_path)
    else:
        pca_init = PCA(n_components=2).fit_transform(X_scaled)
        umap_2d = UMAP(n_components=2, init=pca_init, random_state=42).fit_transform(X_scaled)
        np.save(umap2d_path, umap_2d)

    labels = phenotypes.to_numpy()
    unique_labels = sorted(set(labels), key=lambda lb: (lb != "Unassigned", str.lower(lb)))

    # Convert the shared phenotype_colors (hex strings) to matplotlib RGBA
    _mpl_cmap = {label: mcolors.to_rgba(phenotype_colors[label]) for label in unique_labels}

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            umap_2d[mask, 0],
            umap_2d[mask, 1],
            c=[_mpl_cmap[label]],
            label=label,
            s=1,
            alpha=0.5,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("2D UMAP — Cell Phenotypes")
    ax.legend(markerscale=5, fontsize=8, loc="best")
    fig.tight_layout()
    fig
    return


@app.cell
def _(cache_dir, dtour, np, tour, w, widget_df):
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
            labels=widget_df["phenotypes"].to_numpy(),
            metrics=_metric_names,
            exclude_labels=["Unassigned"],
        )
        np.savez_compressed(metrics_path, **{name: vals for name, vals in metrics.values.items()})
    w.set_metrics(metrics)
    return (metrics,)


@app.cell
def _(cache_dir, dtour, metrics, np, tour, w, widget_df):
    import cev_metrics
    import pandas as pd

    _original_confusion = metrics.values["confusion"]
    _labels = widget_df["phenotypes"].to_numpy()
    _exclude = {"Unassigned"}
    _mask = np.array([lbl not in _exclude for lbl in _labels])
    _X = tour.embedding[_mask]
    _labels_clean = _labels[_mask]

    # Normalize to [-0.5, 0.5] per dimension to match GPU shader projection
    _mins = _X.min(axis=0)
    _ranges = _X.max(axis=0) - _mins
    _ranges[_ranges == 0] = 1e-6
    _X_norm = (_X - _mins) / _ranges - 0.5

    # Precompute confusion matrices for all views (expensive, done once)
    _cm_cache_path = cache_dir / "confusion_matrices.npz"
    if _cm_cache_path.exists():
        _cm_data = np.load(_cm_cache_path)
        _confusion_matrices = [_cm_data[f"cm_{i}"] for i in range(len(tour.views))]
    else:
        _cat = pd.Categorical(_labels_clean)
        _confusion_matrices = []
        for basis in tour.views:
            proj = _X_norm @ basis
            cm_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "label": _cat})
            _confusion_matrices.append(np.asarray(cev_metrics.confusion(cm_df), dtype=np.float64))
        np.savez_compressed(
            _cm_cache_path, **{f"cm_{i}": cm for i, cm in enumerate(_confusion_matrices)}
        )

    # _cat_labels are phenotype names — directly match what the viewer sends
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
                # Row-normalize then average per-label confusion for selected labels
                row_sums = cm[sel_idx, :].sum(axis=1)
                row_sums[row_sums == 0] = 1.0
                per_label = 1.0 - np.array([cm[i, i] for i in sel_idx]) / row_sums
                vals.append(float(np.mean(per_label)))
            _confusion_cache[key] = vals

        w.set_metrics(
            dtour.MetricResult(
                values={**metrics.values, "confusion": _confusion_cache[key]},
                metric_names=metrics.metric_names,
            )
        )

    w.observe(_on_selection, names=["selected_labels"])
    cat_labels = _cat_labels
    confusion_matrices = _confusion_matrices
    return


@app.cell
def _(X_scaled, cache_dir, dtour, phenotype_colors, phenotypes, pl):
    import arro3.io

    # Compute a 4D UMAP tour (cached)
    tour_4d_path = cache_dir / "tour_4d.npz"
    if tour_4d_path.exists():
        tour_4d = dtour.TourResult.load(tour_4d_path)
    else:
        tour_4d = dtour.little_umap_tour(X_scaled, n_components=4)
        tour_4d.save(tour_4d_path)

    # Build a DataFrame with the 4D embedding + categorical labels
    df_4d = (
        pl.DataFrame(
            {f"umap_{i}": tour_4d.embedding[:, i] for i in range(tour_4d.embedding.shape[1])}
        )
        .with_columns(phenotypes)
        .sort(pl.col("phenotypes") != "Unassigned")
    )

    # Embed dtour metadata (spec + tour + colormap) and write parquet
    table = dtour.add_spec_to_parquet(
        df_4d,
        point_color="phenotypes",
        tour_by="dimensions",
        preview_count=4,
        camera_zoom=0.75,
        color_map=phenotype_colors,
        tour=tour_4d,
    )

    out_path = cache_dir / "mair-2022-tumor-4d.pq"
    arro3.io.write_parquet(table, str(out_path))
    out_path
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl
    from sklearn.preprocessing import StandardScaler

    cache_dir = Path(__file__).parent / "__cache__"
    cache_dir.mkdir(exist_ok=True)
    return StandardScaler, cache_dir, np, pl


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
def _(df, dtour, faust_to_celltype):
    phenotypes = (
        df["faustLabels"]
        .replace_strict(faust_to_celltype, default="Unassigned")
        .alias("phenotypes")
    )

    # Build a shared color map from dtour's light palette (Okabe-Ito + Glasbey Light)
    # with "Unassigned" overridden to gray. Used by both the widget and matplotlib.
    _labels = sorted(phenotypes.unique().to_list())
    phenotype_colors = dtour.build_color_map(
        _labels,
        theme="light",
        overrides={"Unassigned": "#808080"},
    )
    return phenotype_colors, phenotypes


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


@app.cell(hide_code=True)
def _():
    faust_to_celltype = {
        "CD4+CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4+CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4+CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4+CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "DN T",
        "CD4+CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4+CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4+CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4+CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4+CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4+CD8-CD3+CD45RA+CD27+CD19+CD103-CD28+CD69-PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Naive",
        "CD4+CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Naive",
        "CD4+CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25+ICOS-TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Naive",
        "CD4+CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Naive",
        "CD4+CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Naive",
        "CD4+CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Naive",
        "CD4+CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Naive",
        "CD4+CD8-CD3+CD45RA-CD27+CD19+CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25+ICOS-TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1-HLADR-GranzymeB-CD25+ICOS-TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25+ICOS+TCRgd-CD38+CD127-Tim3+": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25+ICOS+TCRgd-CD38+CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3+": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3+": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25+ICOS-TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR+GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25+ICOS-TCRgd-CD38-CD127+Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25+ICOS-TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR+GranzymeB-CD25+ICOS+TCRgd-CD38+CD127-Tim3+": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR+GranzymeB-CD25+ICOS+TCRgd-CD38+CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38+CD127-Tim3+": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3+": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25+ICOS-TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27+CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3+": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR+GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25+ICOS-TCRgd-CD38-CD127-Tim3-": "Treg",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD4 Memory",
        "CD4+CD8-CD3-CD45RA+CD27-CD19+CD103-CD28-CD69+PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4+CD8-CD3-CD45RA+CD27-CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4+CD8-CD3-CD45RA-CD27+CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "Plasmablast",
        "CD4+CD8-CD3-CD45RA-CD27-CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4+CD8-CD3-CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "ILC",
        "CD4+CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "Myeloid",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27+CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Naive",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR+GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR+GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38+CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR+GranzymeB+CD25+ICOS+TCRgd-CD38+CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR+GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS+TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103+CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR+GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR+GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27+CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR+GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Trm",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103+CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1+HLADR+GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "CD8 Memory",
        "CD4-CD8+CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "NK",
        "CD4-CD8+CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "NK",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA+CD27+CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27-CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103+CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25+ICOS+TCRgd-CD38-CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS+TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127+Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127+Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27+CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103+CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127+Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd+CD38-CD127-Tim3-": "gdT",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3+CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "DN T",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28+CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "B Memory",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "B Memory",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28-CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Memory",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28-CD69+PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Memory",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Memory",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28-CD69-PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Memory",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "Plasmablast",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Memory",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "Plasmablast",
        "CD4-CD8-CD3-CD45RA+CD27+CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Memory",
        "CD4-CD8-CD3-CD45RA+CD27-CD19+CD103-CD28+CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "B Naive",
        "CD4-CD8-CD3-CD45RA+CD27-CD19+CD103-CD28+CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA+CD27-CD19+CD103-CD28-CD69+PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA+CD27-CD19+CD103-CD28-CD69+PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA+CD27-CD19+CD103-CD28-CD69-PD1+HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA+CD27-CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA+CD27-CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA+CD27-CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "NK",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3+": "ILC",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "ILC",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "Myeloid",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "Myeloid",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "NK",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "NK",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25+ICOS-TCRgd-CD38-CD127+Tim3-": "ILC",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127+Tim3-": "ILC",
        "CD4-CD8-CD3-CD45RA+CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "ILC",
        "CD4-CD8-CD3-CD45RA-CD27+CD19+CD103+CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "Plasmablast",
        "CD4-CD8-CD3-CD45RA-CD27+CD19+CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "Plasmablast",
        "CD4-CD8-CD3-CD45RA-CD27+CD19+CD103-CD28-CD69-PD1+HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "Plasmablast",
        "CD4-CD8-CD3-CD45RA-CD27+CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "Plasmablast",
        "CD4-CD8-CD3-CD45RA-CD27+CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Memory",
        "CD4-CD8-CD3-CD45RA-CD27+CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127+Tim3-": "Plasmablast",
        "CD4-CD8-CD3-CD45RA-CD27+CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "Plasmablast",
        "CD4-CD8-CD3-CD45RA-CD27+CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "Plasmablast",
        "CD4-CD8-CD3-CD45RA-CD27+CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127+Tim3-": "ILC",
        "CD4-CD8-CD3-CD45RA-CD27-CD19+CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "B Naive",
        "CD4-CD8-CD3-CD45RA-CD27-CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA-CD27-CD19+CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA-CD27-CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "B Naive",
        "CD4-CD8-CD3-CD45RA-CD27-CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA-CD27-CD19+CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "B Naive",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103+CD28-CD69+PD1+HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103+CD28-CD69+PD1-HLADR+GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "Myeloid",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103+CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103+CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28+CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "ILC",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "NK",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69+PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38-CD127+Tim3-": "ILC",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3+": "Myeloid",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "Myeloid",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3+": "Myeloid",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR+GranzymeB-CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "Myeloid",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38+CD127-Tim3-": "NK",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB+CD25-ICOS-TCRgd-CD38-CD127-Tim3-": "NK",
        "CD4-CD8-CD3-CD45RA-CD27-CD19-CD103-CD28-CD69-PD1-HLADR-GranzymeB-CD25-ICOS-TCRgd-CD38+CD127+Tim3-": "ILC",
    }
    return (faust_to_celltype,)


if __name__ == "__main__":
    app.run()
