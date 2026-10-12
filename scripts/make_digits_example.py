"""Write the handwritten digits example for dtour.dev: a PCA tour and a 2D UMAP.

scikit-learn's 1,797 8x8 digit images become two Parquet files with the same
rows in the same order, so their views can link selections:

- `digits-pca-tour.pq`: the 64 pixel columns (`pixel_<row>_<col>`, 0-16), `digit`
  ("0"-"9"), `image` (the URL of the row's PNG), and a precomputed PCA tour over the
  pixels. The viewer's in-browser PCA handles at most 32 columns.
- `digits-umap.pq`: `umap_x`, `umap_y` (a 2D UMAP of the pixels), `digit`, and `image`.
- `digits/<row>.png`: an 8-bit grayscale 8x8 PNG per row, e.g. `digits/0000.png`, to
  upload to `--image-url`.

Both color by `digit` and show the image and digit on hover.

Usage (from the repository root):

    uv run --project packages/python --extra umap python scripts/make_digits_example.py data/
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import arro3.core as ac
import arro3.io
import dtour
import numpy as np
from sklearn.datasets import load_digits


def png(pixels: np.ndarray) -> bytes:
    """An 8-bit grayscale PNG of *pixels* (2D uint8)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    height, width = pixels.shape
    # Bit depth 8, color type 0 (grayscale); each row starts with filter type 0
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


def column(values: np.ndarray) -> ac.Array:
    return ac.Array.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random state")
    parser.add_argument(
        "--image-url",
        default="https://data.dtour.dev/digits",
        help="URL the digits/ folder of PNGs is uploaded to",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    X = digits.data.astype(np.float32)
    pixel_columns = [f"pixel_{i // 8}_{i % 8}" for i in range(X.shape[1])]
    image_dir = args.out / "digits"
    image_dir.mkdir(exist_ok=True)
    for i, image in enumerate(digits.images):
        (image_dir / f"{i:04d}.png").write_bytes(png(np.round(image * 255 / 16).astype(np.uint8)))
    labels = {
        "digit": ac.Array([str(d) for d in digits.target], ac.DataType.string()),
        "image": ac.Array(
            [f"{args.image_url}/{i:04d}.png" for i in range(len(digits.images))],
            ac.DataType.string(),
        ),
    }
    settings = {"point_color_by": "digit", "tooltip_columns": ["image", "digit"]}

    # The viewer divides each column by its range, so the tour must too
    span = X.max(axis=0) - X.min(axis=0)
    tour = dtour.little_tour(X / np.where(span > 0, span, 1))
    pca = ac.Table.from_pydict(
        {name: column(X[:, i]) for i, name in enumerate(pixel_columns)} | labels
    )
    pca = dtour.add_spec_to_parquet(pca, tour=tour, tour_dimensions=pixel_columns, **settings)

    from umap import UMAP

    embedding = UMAP(n_components=2, random_state=args.seed).fit_transform(X)
    umap = ac.Table.from_pydict(
        {"umap_x": column(embedding[:, 0]), "umap_y": column(embedding[:, 1])} | labels
    )
    umap = dtour.add_spec_to_parquet(umap, tour_dimensions=["umap_x", "umap_y"], **settings)

    for name, table in (("digits-pca-tour.pq", pca), ("digits-umap.pq", umap)):
        path = args.out / name
        arro3.io.write_parquet(table, str(path), compression="zstd(19)")
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
