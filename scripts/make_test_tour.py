"""Write a small sequential-tour test dataset with an embedded dtour spec.

Two 2D Gaussians orbit each other while they stretch and rotate, over
`--keyframes` steps that loop back to the start. Each point keeps its own noise
sample, so points move smoothly from keyframe to keyframe.

Usage (from the repository root):

    uv run --project packages/python python scripts/make_test_tour.py
    uv run --project packages/python python scripts/make_test_tour.py --keyframes 33

Then open http://localhost:5173/?url=/data/two-gaussians-32-keyframes.pq with
`pnpm dev`, or pass the file to `dtour.Widget(data=...)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arro3.core as ac
import arro3.io
import dtour
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def blob_frames(n_points: int, n_keyframes: int, seed: int) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((2, n_points, 2))
    frames = []
    for k in range(n_keyframes):
        phase = 2 * np.pi * k / n_keyframes
        # The blobs orbit their common center. sequential_tour aligns
        # consecutive frames, so only motion relative to each other remains.
        offset = 3 * np.array([np.cos(phase), np.sin(phase)])
        a = noise[0] @ np.diag([1 + 0.7 * np.sin(phase), 0.5]) @ rotation(phase / 2).T
        b = noise[1] @ np.diag([0.6, 1 + 0.5 * np.cos(2 * phase)]) @ rotation(-phase).T
        frames.append(np.vstack([a + offset, b - offset]).astype(np.float32))
    labels = ["A"] * n_points + ["B"] * n_points
    return frames, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keyframes", type=int, default=32)
    parser.add_argument("--points", type=int, default=2000, help="points per Gaussian")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="defaults to data/two-gaussians-<k>-keyframes.pq")
    args = parser.parse_args()

    frames, labels = blob_frames(args.points, args.keyframes, args.seed)
    tour = dtour.sequential_tour(
        frames,
        method=lambda embedding, _previous: embedding,
        description=(
            f"Two Gaussians orbit each other and change shape over {args.keyframes} keyframes."
        ),
        keyframe_descriptions=[f"Step {k + 1}" for k in range(args.keyframes)],
    )

    columns = [f"sp_{i}" for i in range(tour.embedding.shape[1])]
    table = ac.Table.from_pydict(
        {
            name: ac.Array.from_numpy(np.ascontiguousarray(tour.embedding[:, i]))
            for i, name in enumerate(columns)
        }
        | {"blob": ac.Array(labels, ac.DataType.string())}
    )
    table = dtour.add_spec_to_parquet(
        table,
        tour=tour,
        tour_dimensions=columns,
        point_color_by="blob",
        point_color_map={"A": "#4e79a7", "B": "#f28e2b"},
    )

    out = args.out or ROOT / "data" / f"two-gaussians-{args.keyframes}-keyframes.pq"
    out.parent.mkdir(parents=True, exist_ok=True)
    arro3.io.write_parquet(table, str(out))
    print(f"Wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
