# dtour: Web app

This is the thin web frontend behind [dtour.dev](https://dtour.dev). It's a lightweight single-page [Vite](https://vitejs.dev) + React app that wraps the [`@dtour/viewer`](../viewer) `<Dtour>` component and adds only the shell around it: drag-and-drop / file-picker data loading, a set of curated demo datasets, and per-file spec persistence in `localStorage` (so a dataset reopens with the tour state, theme, and coloring you left it in).

It is **not published** (`private: true`) and exposes no importable API — all the reusable logic lives in [`@dtour/scatter`](../scatter) and [`@dtour/viewer`](../viewer). This package is just the deployable app that composes them.

## How it connects

```
@dtour/scatter  →  @dtour/viewer  →  webapp (this package)  →  dtour.dev
   renderer          React UI           app shell
```

The app renders a single `<Dtour>` and stays deliberately thin: it feeds in the loaded `ArrayBuffer` and a persisted `DtourSpec`, and reflects state changes back out via `onSpecChange`. Anything data-generic belongs in `@dtour/viewer`, not here.

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
