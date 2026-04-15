import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # dtour Demo: Attraction-Repulsion Spectrum Tour

        This notebook loads **Fashion MNIST** (70K points, PCA to 50D) and computes
        embeddings at different points on the **attraction-repulsion spectrum**
        ([Bohm, Berens & Kobak, JMLR 2022](https://jmlr.org/papers/v23/21-0055.html)):
        from pure attraction (LE-like, rho=100) through UMAP-like (rho=4) to standard t-SNE (rho=1).

        The tour smoothly morphs between these 2D embeddings, showing how cluster
        structure emerges as repulsion increases.
        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import numpy as np

    cache_dir = Path(__file__).parent / "__cache__"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir, np


@app.cell
def _(cache_dir, np):
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA

    _cache_path = cache_dir / "fashion_mnist_pca50.npz"

    if _cache_path.exists():
        _data = np.load(_cache_path)
        X_pca = _data["X_pca"]
        label_ints = _data["label_ints"]
    else:
        fmnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
        # PCA to 50D (following Bohm et al.)
        X_pca = (
            PCA(n_components=50).fit_transform(fmnist.data.astype(np.float32)).astype(np.float32)
        )
        label_ints = fmnist.target.astype(np.int8)
        np.savez_compressed(_cache_path, X_pca=X_pca, label_ints=label_ints)

    _FASHION_NAMES = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Boot",
    ]
    label_names = np.array([_FASHION_NAMES[i] for i in label_ints])
    feature_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    return X_pca, feature_names, label_names


@app.cell
def _(X_pca, cache_dir, feature_names):
    import dtour

    tour_path = cache_dir / "spectrum_tour_fmnist.npz"
    if tour_path.exists():
        tour = dtour.TourResult.load(tour_path)
    else:
        tour = dtour.spectrum_tour(
            X_pca,
            n_frames=4,
            n_neighbors=15,
            init="le",
            feature_names=feature_names,
            random_state=42,
        )
        tour.save(tour_path)
    return dtour, tour


@app.cell
def _(dtour, label_names, tour):
    import polars as pl

    color_map = dtour.build_color_map(
        sorted(set(label_names)),
        theme="light",
    )

    sp_df = pl.DataFrame(
        {f"sp_{i}": tour.embedding[:, i] for i in range(tour.embedding.shape[1])}
    ).with_columns(pl.Series("label", label_names))

    w = dtour.Widget(
        data=sp_df,
        tour=tour,
        preview_count=4,
        preview_size="large",
        point_color="label",
        color_map=color_map,
        camera_zoom=0.5,
        height=900,
        theme="light",
    )
    w
    return color_map, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## PyMDE with concave anchor regularization

    Same spectrum tour but using **PyMDE** with concave (log) regularization
    (``regularization=0.25``). Small jitter is suppressed while large,
    structurally meaningful movements are allowed through.
    """)
    return


@app.cell
def _(X_pca, cache_dir, dtour, feature_names):
    tour_pymde_path = cache_dir / "spectrum_tour_fmnist_pymde.npz"
    if tour_pymde_path.exists():
        tour_pymde = dtour.TourResult.load(tour_pymde_path)
    else:
        tour_pymde = dtour.spectrum_tour(
            X_pca,
            rhos=[100, 20, 4, 1],
            n_neighbors=15,
            init="le",
            method="pymde",
            regularization=0.25,
            feature_names=feature_names,
            random_state=42,
        )
        tour_pymde.save(tour_pymde_path)
    return (tour_pymde,)


@app.cell
def _(color_map, dtour, label_names, pl, tour_pymde):
    sp_pymde_df = pl.DataFrame(
        {f"sp_{i}": tour_pymde.embedding[:, i] for i in range(tour_pymde.embedding.shape[1])}
    ).with_columns(pl.Series("label", label_names))

    w_pymde = dtour.Widget(
        data=sp_pymde_df,
        tour=tour_pymde,
        preview_count=4,
        preview_size="large",
        point_color="label",
        color_map=color_map,
        camera_zoom=0.5,
        height=900,
        theme="light",
    )
    w_pymde
    return


if __name__ == "__main__":
    app.run()
