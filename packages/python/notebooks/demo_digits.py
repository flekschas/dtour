# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "dtour",
#     "marimo",
#     "pandas>=2.0",
#     "scikit-learn>=1.7",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Handwritten Digits

    A PCA tour of the 1,797 8x8 images in scikit-learn's
    [digits dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset).

    Lasso-select points or click a digit in the legend: the cells below
    rerun with your selection and show the selected images.
    """)
    return


@app.cell
async def _():
    import sys

    # In the browser (WebAssembly), install dtour without its dependencies, as
    # pyamg, which only le_tour() needs, has no WebAssembly build.
    if sys.platform == "emscripten":
        import micropip

        await micropip.install(["anywidget", "arro3-core", "arro3-io"])
        await micropip.install("dtour", deps=False)
    return


@app.cell
def _():
    import dtour
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_digits

    return dtour, load_digits, mo, np, pd


@app.cell
def _(load_digits, pd):
    digits = load_digits()
    pixels = [f"px{i}" for i in range(digits.data.shape[1])]
    df = pd.DataFrame(digits.data, columns=pixels)
    df["digit"] = digits.target.astype(str)
    return df, digits, pixels


@app.cell
def _(df, dtour, mo, pixels):
    view = mo.ui.anywidget(
        dtour.Widget(
            data=df,
            tour=dtour.little_tour(df[pixels], n_components=6),
            point_color_by="digit",
            point_size=4,
            preview_count=6,
            height=560,
        )
    )
    view
    return (view,)


@app.cell
def _(df, view):
    # Rerun whenever the selection in the view changes
    if view.selected_labels:
        selected = df.index[df["digit"].isin(view.selected_labels)].to_numpy()
    else:
        selected = view.selected_indices
    return (selected,)


@app.cell(hide_code=True)
def _(df, digits, mo, np, selected):
    def digit_svg(image):
        cells = "".join(
            f'<rect x="{x}" y="{y}" width="1" height="1" fill-opacity="{v / 16:.2f}"/>'
            for (y, x), v in np.ndenumerate(image)
            if v
        )
        return f'<svg viewBox="0 0 8 8" width="40" height="40">{cells}</svg>'

    if len(selected):
        counts = df["digit"].iloc[selected].value_counts().sort_index()
        summary = ", ".join(f"**{d}**: {n}" for d, n in counts.items())
        gallery = "".join(digit_svg(digits.images[i]) for i in selected[:120])
        output = mo.vstack(
            [
                mo.md(f"{len(selected):,} selected digits ({summary})"),
                mo.Html(f'<div style="display:flex;flex-wrap:wrap;gap:4px">{gallery}</div>'),
            ]
        )
    else:
        output = mo.md("_Select points to see their images._")
    output
    return


if __name__ == "__main__":
    app.run()
