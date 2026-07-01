#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the demo notebooks locally against the LOCAL dtour working copy.
#
# marimo notebooks are plain Python: `uv run <nb>.py` executes every cell in
# dataflow order and exits non-zero if any cell raises. Each notebook declares
# its own deps via inline PEP 723 metadata; `--with-editable packages/python`
# swaps the published dtour for your working tree, so you test uncommitted
# changes. Data is downloaded/cached into notebooks/__cache__ on first run.
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

failed=()
for nb in "${notebooks[@]}"; do
  file="${nb_dir}/${nb}.py"
  if [[ ! -f "$file" ]]; then
    echo "✗ ${nb}: no such notebook ($file)" >&2
    failed+=("$nb")
    continue
  fi
  echo "▶ Running ${nb} …"
  if uv run --python 3.12 --with-editable packages/python "$file"; then
    echo "✓ ${nb}"
  else
    echo "✗ ${nb} failed" >&2
    failed+=("$nb")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo >&2
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
echo "All notebooks ran successfully."
