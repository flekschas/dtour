#!/usr/bin/env bash
set -euo pipefail

level="${1:-}"
if [[ "$level" != "major" && "$level" != "minor" && "$level" != "patch" ]]; then
  echo "Usage: pnpm bump <major|minor|patch>" >&2
  exit 1
fi

# Read current version from scatter (single source of truth)
current=$(node -p "require('./packages/scatter/package.json').version")
IFS='.' read -r major minor patch <<< "$current"

case "$level" in
  major) major=$((major + 1)); minor=0; patch=0 ;;
  minor) minor=$((minor + 1)); patch=0 ;;
  patch) patch=$((patch + 1)) ;;
esac

next="${major}.${minor}.${patch}"
echo "Bumping $current → $next"

# Bump JS packages (inline sed to avoid npm version reformatting JSON)
for pkg in packages/scatter/package.json packages/viewer/package.json; do
  sed -i '' "s/\"version\": \".*\"/\"version\": \"$next\"/" "$pkg"
done

# Bump Python package (sed the version line in pyproject.toml, then refresh lock)
sed -i '' "s/^version = \".*\"/version = \"$next\"/" packages/python/pyproject.toml
uv lock --project packages/python

# Bump the Claude Desktop extension, which installs the same dtour release
sed -i '' "s/\"version\": \".*\"/\"version\": \"$next\"/" packages/python/mcpb/manifest.json
sed -i '' -e "s/^version = \".*\"/version = \"$next\"/" -e "s/dtour\[mcp,umap\]==[^\"]*/dtour[mcp,umap]==$next/" packages/python/mcpb/pyproject.toml

# Commit and tag
git add packages/scatter/package.json packages/viewer/package.json packages/python/pyproject.toml packages/python/uv.lock packages/python/mcpb/manifest.json packages/python/mcpb/pyproject.toml
git commit -m "chore: release v${next}"
git tag -m "v${next}" "v${next}"

echo "Done. Run 'git push origin main --tags' to publish."
