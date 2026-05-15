import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # dtour Demo: Attraction-Repulsion Tour

    This notebook loads **[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)** (70K points, PCA to 50D) and computes
    embeddings at different points on the **attraction-repulsion spectrum**
    ([Bohm, Berens & Kobak, JMLR 2022](https://jmlr.org/papers/v23/21-0055.html)):
    from pure attraction (LE-like, rho=100) through UMAP-like (rho=4) to standard t-SNE (rho=1).

    The tour smoothly morphs between these 2D embeddings, showing how cluster
    structure emerges as repulsion increases.

    **Select points** in the scatter plot to see corresponding Fashion MNIST images
    in the sidebar.
    """)
    return


@app.cell(hide_code=True)
def _(mo, w, w_images):
    mo.hstack([w, w_images], widths=[4, 1])
    return


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


@app.cell(hide_code=True)
def _():
    from anywidget import AnyWidget
    from traitlets import Int, List

    class ImagesWidget(AnyWidget):
        _esm = """
        const baseUrl = 'https://data.dtour.dev/fashion-mnist/';
        function render({ model, el }) {
          const container = document.createElement('div');
          container.classList.add('images-container');

          const title = document.createElement('div');
          title.classList.add('images-title');
          container.appendChild(title);

          const grid = document.createElement('div');
          grid.classList.add('images-grid');
          container.appendChild(grid);

          function choose(x, k) {
            const idxs = Array.from({ length: x.length }, (_, i) => i);
            return Array.from({ length: Math.min(k, x.length) }, () => {
              const i = Math.round(Math.random() * (idxs.length - 1));
              const idx = idxs[i];
              idxs.splice(i, 1);
              return x[idx];
            });
          }

          function renderImages() {
            const images = model.get("images");
            title.textContent = images.length > 0
              ? `Selected Images (${images.length})`
              : 'Select points to see images';
            grid.textContent = '';
            choose(images, model.get("max")).forEach(([image, color]) => {
              const imgId = String(image).padStart(5, '0');
              const img = document.createElement('div');
              img.classList.add('images-fashion-mnist');
              img.style.backgroundColor = color;
              img.style.backgroundImage = `url(${baseUrl}${imgId}.png)`;
              grid.appendChild(img);
            });
          }

          model.on("change:images", renderImages);
          model.on("change:max", renderImages);
          renderImages();

          el.appendChild(container);
        }
        export default { render };
        """

        _css = """
        .images-container {
          width: 100%;
          height: 900px;
          padding: 0 0 0 0.25rem;
          overflow: auto;
        }
        .images-title {
          font-size: 0.85rem;
          font-weight: bold;
          text-align: center;
          line-height: 28px;
          color: #555;
        }
        .images-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(32px, 1fr));
          align-content: flex-start;
          gap: 4px;
          width: 100%;
          height: calc(100% - 32px);
          margin-top: 4px;
          overflow: auto;
        }
        .images-fashion-mnist {
          display: flex;
          width: 32px;
          height: 32px;
          background-repeat: no-repeat;
          background-position: center;
        }
        """

        images = List().tag(sync=True)
        max = Int(100).tag(sync=True)

    return (ImagesWidget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Tour Computation
    Compute 2D embeddings at several attraction-repulsion levels and align them into a smooth tour.
    """)
    return


@app.cell
def _(X_pca, cache_dir, feature_names):
    import dtour

    tour_path = cache_dir / "attraction_repulsion_fmnist.npz"
    if tour_path.exists():
        tour = dtour.TourResult.load(tour_path)
    else:
        tour = dtour.attraction_repulsion_tour(
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
def _(dtour, label_names):
    color_map = dtour.build_color_map(
        sorted(set(label_names)),
        theme="light",
    )
    return (color_map,)


@app.cell
def _(ImagesWidget, color_map, dtour, label_names, tour):
    import polars as pl

    sp_df = pl.DataFrame(
        {f"sp_{i}": tour.embedding[:, i] for i in range(tour.embedding.shape[1])}
    ).with_columns(pl.Series("label", label_names))

    w = dtour.Widget(
        data=sp_df,
        tour=tour,
        preview_count=4,
        preview_size="medium",
        point_color_by="label",
        color_map=color_map,
        camera_zoom=0.5,
        height=900,
        theme="light",
    )

    w_images = ImagesWidget()

    def _on_selection(change):
        w_images.images = [[int(i), color_map.get(label_names[i], "#888")] for i in change["new"]]

    w.observe(_on_selection, names=["selected_indices"])
    return sp_df, w, w_images


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Parquet Export
    Export the tour as a self-contained Parquet file with embedded dtour metadata.
    """)
    return


@app.cell
def _(cache_dir, color_map, dtour, sp_df, tour):
    import pyarrow.parquet as pq

    _pa_table = sp_df.to_arrow()
    _meta_json = dtour.build_dtour_metadata(
        tour=tour,
        point_color_by="label",
        point_color_map=color_map,
        camera_zoom=0.5,
        preview_count=4,
        theme_mode="light",
    )
    _existing = _pa_table.schema.metadata or {}
    _existing[b"dtour"] = _meta_json.encode("utf-8")

    pq.write_table(
        _pa_table.replace_schema_metadata(_existing),
        cache_dir / "fashion_mnist_attraction_repulsion_tour.pq",
        compression="zstd",
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
