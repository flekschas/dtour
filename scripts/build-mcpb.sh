#!/usr/bin/env bash
# Build the Claude Desktop extension for the dtour MCP server:
# packages/python/mcpb/dist/dtour.mcpb. It installs dtour from PyPI, or with
# --local, from a wheel of this checkout that the bundle includes.
set -euo pipefail
cd "$(dirname "$0")/.."

src=packages/python/mcpb
out="$PWD/$src/dist/dtour.mcpb"
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT

cp -R "$src/manifest.json" "$src/icon.png" "$src/pyproject.toml" "$src/src" "$build/"
if [[ "${1:-}" == "--local" ]]; then
  (cd packages/python && uv build --wheel --out-dir "$build/wheels")
  wheel=$(basename "$build"/wheels/*.whl)
  printf '\n[tool.uv.sources]\ndtour = { path = "wheels/%s" }\n' "$wheel" >> "$build/pyproject.toml"
fi

npx -y @anthropic-ai/mcpb@2.1.2 validate "$build/manifest.json"
npx -y @anthropic-ai/mcpb@2.1.2 pack "$build" "$out"
