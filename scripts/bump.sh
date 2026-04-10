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

# Bump JS packages (npm handles package.json)
cd packages/scatter && npm version "$next" --no-git-tag-version && cd ../..
cd packages/viewer  && npm version "$next" --no-git-tag-version && cd ../..

# Bump Python package (sed the version line in pyproject.toml)
sed -i '' "s/^version = \".*\"/version = \"$next\"/" packages/python/pyproject.toml

# Commit and tag
git add packages/scatter/package.json packages/viewer/package.json packages/python/pyproject.toml
git commit -m "chore: release v${next}"
git tag "v${next}"

echo "Done. Run 'git push origin main --tags' to publish."
