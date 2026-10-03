#!/usr/bin/env bash
# sync.sh — mirror the Quarto presentation into this repo.
set -euo pipefail

SRC="$HOME/HARI_UBUNTU_HOME/Shared Files/Quarto_reveal/Presentation"
DST="$HOME/HARI_UBUNTU_HOME/Shared Files/Hierarchical_RL_GH"

echo "→ syncing from $SRC"

# ---- slides sources ----
mkdir -p "$DST/slides/data" "$DST/slides/derivations"
cp "$SRC/Presentation.qmd"        "$DST/slides/"
cp "$SRC/custom.scss"             "$DST/slides/"
cp "$SRC/data/papers.csv"         "$DST/slides/data/"

# ---- all derivation PDFs ----
rsync -a --delete "$SRC/derivations/" "$DST/slides/derivations/"

# ---- figures (both places get the same set) ----
rsync -a --delete "$SRC/figures/" "$DST/figures/"
rsync -a --delete "$SRC/figures/" "$DST/gallery/"

# ---- rendered presentation (optional; comment out if too big) ----
rsync -a --delete "$SRC/Presentation.html"      "$DST/slides/" 2>/dev/null || true
rsync -a --delete "$SRC/Presentation_files/"    "$DST/slides/" 2>/dev/null || true

echo "→ done. Now:"
echo "    cd $DST"
echo "    git status"
echo "    git add . && git commit -m 'sync slides and derivations' && git push"
