# dtour: Web app

This is the thin web frontend behind [dtour.dev](https://dtour.dev). It's a lightweight single-page [Vite](https://vitejs.dev) + React app that wraps the [`@dtour/viewer`](../viewer) `<Dtour>` component and adds only the shell around it: drag-and-drop / file-picker data loading, a set of curated demo datasets, and per-file spec persistence in `localStorage` (so a dataset reopens with the tour state, theme, and coloring you left it in). Settings also live in the URL, named like the `DtourSpec` fields (e.g. `?url=…&pointColorBy=label`), so a link opens the data with the same settings. `?views=[…]` shows several views side by side: a JSON array of settings per view, where `data` names a view's own file and settings outside `views` apply to all views. `?layout=[[0,1],[2,3]]` arranges the views in rows and lists each view once (default: one row). Views with the same `link` share their selections, which link by row, so their files must hold the same rows in the same order, e.g., `?url=<pca.pq>&views=[{"link":"a"},{"data":"<umap.pq>","link":"a"}]`. The views share the page's theme, so a view's own `themeMode` is ignored, and their settings live only in the URL, not in `localStorage`. Projections dragged in manual mode or reached in grand mode are not part of the settings.

It is **not published** (`private: true`) and exposes no importable API — all the reusable logic lives in [`@dtour/scatter`](../scatter) and [`@dtour/viewer`](../viewer). This package is just the deployable app that composes them.

## How it connects

```
@dtour/scatter  →  @dtour/viewer  →  webapp (this package)  →  dtour.dev
   renderer          React UI           app shell
```

The app stays deliberately thin: it renders a `<Dtour>` for each view, feeds in its `ArrayBuffer` and `DtourSpec`, and reflects settings changes back out via `onSpecChange`. A single file also uses the settings saved for it. Anything data-generic belongs in `@dtour/viewer`, not here.

## Run it

From the repo root (starts the viewer in watch mode alongside the app):

```sh
pnpm dev
```

Or drive this package directly:

```sh
pnpm --filter webapp dev       # start the Vite dev server
pnpm --filter webapp build     # production build to dist/
pnpm --filter webapp preview    # preview the production build
```

Then open the printed local URL and drop a Parquet or Arrow file into the window — or pick one of the built-in demo datasets.
