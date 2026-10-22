"""Write the precomputed data of the molab demo notebook (`demo_molab.py`).

The demo skips the slow UMAP fits of the full notebooks by loading their results
from two Parquet files, one row per cell in the order of the source data:

- `lamanno2021-pca-umap.pq`: `PC1`-`PC8`, `Age`, `Class`, `Subclass`, and `UMAP1`,
  `UMAP2` (the 2D UMAP of `demo_brain_atlas.py`).
- `mair-2022-tumor-umap-8d.pq`: `UMAP1`-`UMAP8` (the 8D UMAP of
  `demo_immune_cell_markers.py`), `phenotypes`, and the 18 winsorized markers.

Run both notebooks first to fill their cache, then upload the files to
`https://data.dtour.dev/notebooks/`.

Usage (from the repository root):

    uv run --project packages/python python scripts/make_molab_demo_data.py \
        packages/python/notebooks/__cache__ data/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtour
import numpy as np
import polars as pl


def brain_atlas(cache: Path) -> pl.DataFrame:
    pcs = [f"PC{i}" for i in range(1, 9)]
    df = pl.read_parquet(cache / "lamanno2021.pq").drop_nulls(pcs)
    umap = np.load(cache / "lamanno2021-umap-2d.npy").astype(np.float32)
    return df.select(
        pl.col(pcs).cast(pl.Float32),
        "Age",
        pl.col("Class", "Subclass").cast(pl.Categorical),
    ).with_columns(UMAP1=umap[:, 0], UMAP2=umap[:, 1])


def immune_cells(cache: Path) -> pl.DataFrame:
    df = pl.read_parquet(cache / "mair-2022-tumor-006-ozette.pq")
    to_phenotype = json.loads((cache / "mair-ozette-simple-phenotypes.json").read_text())
    embedding = dtour.TourResult.load(cache / "tour.npz").embedding.astype(np.float32)
    markers = [c for c in df.columns if c.endswith("_Windsorized")]
    umap = pl.DataFrame({f"UMAP{i + 1}": embedding[:, i] for i in range(embedding.shape[1])})
    return pl.concat(
        [
            umap,
            df.select(
                pl.col("faustLabels")
                .replace_strict(to_phenotype, default="Unassigned")
                .cast(pl.Categorical)
                .alias("phenotypes"),
                *[
                    pl.col(c).cast(pl.Float32).alias(c.removesuffix("_Windsorized"))
                    for c in markers
                ],
            ),
        ],
        how="horizontal",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cache", type=Path, help="the notebooks' __cache__ directory")
    parser.add_argument("out", type=Path, help="output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in [
        ("lamanno2021-pca-umap.pq", brain_atlas(args.cache)),
        ("mair-2022-tumor-umap-8d.pq", immune_cells(args.cache)),
    ]:
        df.write_parquet(args.out / name, compression="zstd", compression_level=9)
        print(f"{name}: {df.height:,} rows, {(args.out / name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
