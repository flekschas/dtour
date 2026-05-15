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
    **Select points** to see corresponding COCO images below.
    """)
    return


@app.cell(hide_code=True)
def _(le_widget, mo, scatter_2d, umap_widget, w_images):
    mo.vstack([
        mo.hstack([le_widget, mo.ui.anywidget(scatter_2d.widget), umap_widget], widths=[1, 1, 1], gap=0),
        w_images,
    ])
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Data Loading
    Load precomputed embeddings (2D UMAP, 4D UMAP, Fisher LE tour) and COCO category labels from Parquet files.
    """)
    return


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


@app.cell(hide_code=True)
def _():
    from anywidget import AnyWidget
    from traitlets import Int, List

    class ImagesWidget(AnyWidget):
        _esm = """
        const baseUrl = 'https://data.dtour.dev/sharegpt4v-coco/';
        function render({ model, el }) {
          const container = document.createElement('div');
          container.classList.add('coco-images-container');

          const title = document.createElement('div');
          title.classList.add('coco-images-title');
          container.appendChild(title);

          const grid = document.createElement('div');
          grid.classList.add('coco-images-grid');
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
              ? `Selected (${images.length})`
              : 'Select points to see images';
            grid.textContent = '';
            choose(images, model.get("max")).forEach((imageId) => {
              const imgId = String(imageId).padStart(12, '0');
              const img = document.createElement('div');
              img.classList.add('coco-image');
              img.style.backgroundImage = `url(${baseUrl}${imgId}.webp)`;
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
        .coco-images-container {
          width: 100%;
          max-height: 360px;
          overflow: auto;
        }
        .coco-images-title {
          font-size: 0.85rem;
          font-weight: bold;
          text-align: center;
          line-height: 28px;
          color: #555;
        }
        .coco-images-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, 100px);
          align-content: flex-start;
          gap: 4px;
          width: 100%;
          margin-top: 4px;
        }
        .coco-image {
          width: 100px;
          height: 100px;
          background-size: contain;
          background-repeat: no-repeat;
          background-position: center;
        }
        """

        images = List().tag(sync=True)
        max = Int(36).tag(sync=True)

    return (ImagesWidget,)


@app.cell
def _(le_fisher_table):
    import dtour

    # Build a shared color map for all three views
    _labels = sorted(set(le_fisher_table.column("coco_label_idf").to_pylist()))
    color_map = dtour.build_color_map(_labels, theme="light")
    return color_map, dtour


@app.cell
def _(ImagesWidget, color_map, dtour, le_fisher_table):
    le_tour = dtour.TourResult.from_parquet(le_fisher_table)

    le_widget = dtour.Widget(
        data=le_fisher_table,
        tour=le_tour,
        point_color_by="coco_label_idf",
        color_map=color_map,
        point_opacity=0.5,
        preview_size="small",
        theme="light",
        height=720,
    )

    # image_id lives in the LE fisher table itself
    image_ids = le_fisher_table.column("image_id").to_pylist()
    w_images = ImagesWidget()
    return image_ids, le_widget, w_images


@app.cell
def _(color_map, df_4d, df_labels, dtour, pa, pd):
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
        color_map=color_map,
        point_opacity=0.5,
        tour_by="dimensions",
        preview_size="small",
        theme="light",
        height=720,
    )
    return (umap_widget,)


@app.cell
def _(color_map, df_2d, df_labels, pd):
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
        color_map=color_map,
        height=720,
        tooltip=True,
        axes=False,
    )
    return (scatter_2d,)


@app.cell
def _(df_labels, image_ids, le_widget, scatter_2d, umap_widget, w_images):
    _labels = df_labels["coco_label_idf"].values

    def _label_to_indices(labels):
        if not labels:
            return []
        label_set = set(labels)
        return [i for i, lb in enumerate(_labels) if lb in label_set]

    def _handle_le_selection(change):
        new_sel = change.new or []
        if set(new_sel) != set(umap_widget.selected_indices):
            umap_widget.select(new_sel)
        if set(new_sel) != set(scatter_2d.selection()):
            scatter_2d.selection(new_sel if new_sel else None)
        w_images.images = [image_ids[i] for i in new_sel]

    def _handle_le_label_selection(change):
        labels = change.new or []
        idx = _label_to_indices(labels)
        if set(labels) != set(umap_widget.selected_labels):
            umap_widget.select_by_labels(labels)
        if set(idx) != set(scatter_2d.selection()):
            scatter_2d.selection(idx if idx else None)
        w_images.images = [image_ids[i] for i in idx]

    def _handle_umap_selection(change):
        new_sel = change.new or []
        if set(new_sel) != set(le_widget.selected_indices):
            le_widget.select(new_sel)
        if set(new_sel) != set(scatter_2d.selection()):
            scatter_2d.selection(new_sel if new_sel else None)
        w_images.images = [image_ids[i] for i in new_sel]

    def _handle_umap_label_selection(change):
        labels = change.new or []
        idx = _label_to_indices(labels)
        if set(labels) != set(le_widget.selected_labels):
            le_widget.select_by_labels(labels)
        if set(idx) != set(scatter_2d.selection()):
            scatter_2d.selection(idx if idx else None)
        w_images.images = [image_ids[i] for i in idx]

    def _effective_selection(widget):
        """Return the effective index set, accounting for label selections."""
        if widget.selected_labels:
            return set(_label_to_indices(widget.selected_labels))
        return set(widget.selected_indices)

    def _handle_jscatter_selection(change):
        new_sel = list(change.new) if len(change.new) else []
        s = set(new_sel)
        if s != _effective_selection(le_widget):
            le_widget.select(new_sel)
        if s != _effective_selection(umap_widget):
            umap_widget.select(new_sel)
        w_images.images = [image_ids[i] for i in new_sel]

    le_widget.observe(_handle_le_selection, names="selected_indices")
    le_widget.observe(_handle_le_label_selection, names="selected_labels")
    umap_widget.observe(_handle_umap_selection, names="selected_indices")
    umap_widget.observe(_handle_umap_label_selection, names="selected_labels")
    scatter_2d.widget.observe(_handle_jscatter_selection, names="selection")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
