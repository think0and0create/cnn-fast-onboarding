#!/usr/bin/env bash
# Sync article.md → docs/ for VuePress build.
# Run before vuepress build (manually or via CI).
set -euo pipefail

cd "$(dirname "$0")/.."

# 4 chapter article.md files (symlinks don't survive VuePress 2.x page discovery)
# Use subdirectory structure so URLs are /ch1/, /ch2/, etc. (not /ch1.html)
mkdir -p docs/ch1 docs/ch2 docs/ch3 docs/ch4
cp articles/01-python-crash-course/article.md    docs/ch1/README.md
cp articles/02-linear-algebra-numpy/article.md    docs/ch2/README.md
cp articles/03-optimization-torch/article.md      docs/ch3/README.md
cp articles/04-optimization-applications/article.md docs/ch4/index.md

# Ch 4 has 14 matplotlib figures
rm -rf docs/ch4/output
cp -r articles/04-optimization-applications/output docs/ch4/output

echo "synced 4 chapters + 14 figures → docs/"
