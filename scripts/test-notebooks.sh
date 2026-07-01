#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the demo notebooks locally against the LOCAL dtour working copy.
#
# marimo notebooks are plain Python: `uv run <nb>.py` executes every cell in
# dataflow order and exits non-zero if any cell raises. Each notebook declares
# its own deps via inline PEP 723 metadata. Data is downloaded/cached into
# notebooks/__cache__ on first run.
#
# We want `dtour` to ALWAYS come from your working tree — never PyPI — so that
# uncommitted changes (including a bumped version) are what's tested. The robust
# way to pin a script's dependency to a local path is a `[tool.uv.sources]` entry
# in the PEP 723 metadata. We can't commit that into the notebooks (it would break
# the self-contained `uvx marimo --sandbox` path, which must pull dtour from PyPI),
# so we inject it into a throwaway copy of each notebook at run time. The copy
# lives beside the original (same dir) so `Path(__file__).parent / "__cache__"`
# still resolves to notebooks/__cache__.
#
# Everything else (marimo, umap-learn, matplotlib, …) resolves normally, honoring
# any global `exclude-newer` you have set. `--python 3.12` is the interpreter
# where every native dep (pyamg, numba, openTSNE, cev-metrics, umap-learn) ships
# wheels.
#
# This mirrors the `test-notebooks` job in .github/workflows/publish.yml, except
# CI tests the published PyPI artifact while this tests your local source.
#
# Usage:
#   pnpm test:notebooks                        # run all notebooks
#   pnpm test:notebooks demo_spectral          # run one (name, with or without .py)
#   pnpm test:notebooks demo_spectral demo_brain_atlas

cd "$(dirname "$0")/.."

nb_dir="packages/python/notebooks"

if [[ $# -gt 0 ]]; then
  notebooks=()
  for arg in "$@"; do
    notebooks+=("${arg%.py}")
  done
else
  notebooks=(
    demo_attraction_repulsion
    demo_brain_atlas
    demo_image_embedding
    demo_immune_cell_markers
    demo_spectral
  )
fi

# Clean up any injected throwaway copies on exit (incl. Ctrl-C / failure).
tmp_files=()
cleanup() { rm -f "${tmp_files[@]}"; }
trap cleanup EXIT

failed=()
for nb in "${notebooks[@]}"; do
  file="${nb_dir}/${nb}.py"
  if [[ ! -f "$file" ]]; then
    echo "✗ ${nb}: no such notebook ($file)" >&2
    failed+=("$nb")
    continue
  fi

  # Inject a `[tool.uv.sources]` override before the closing `# ///` delimiter so
  # `dtour` resolves to the local editable package (../ from the notebooks dir).
  tmp="${nb_dir}/.dtour-test.${nb}.py"
  tmp_files+=("$tmp")
  awk '
    /^# \/\/\/$/ && !done {
      print "#"
      print "# [tool.uv.sources]"
      print "# dtour = { path = \"..\", editable = true }"
      done = 1
    }
    { print }
  ' "$file" > "$tmp"

  # MPLBACKEND=Agg: force matplotlib's non-interactive backend. Headless runs
  # have no display, and the notebooks call plt.subplots(); without this,
  # matplotlib tries to load an interactive GUI backend (e.g. tkagg) and fails.
  echo "▶ Running ${nb} …"
  if MPLBACKEND=Agg uv run --python 3.12 "$tmp"; then
    echo "✓ ${nb}"
  else
    echo "✗ ${nb} failed" >&2
    failed+=("$nb")
  fi
  rm -f "$tmp"
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo >&2
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
echo "All notebooks ran successfully."
