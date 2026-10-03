#!/usr/bin/env bash
set -euo pipefail

SRC="$HOME/HARI_UBUNTU_HOME/Shared Files/Quarto_reveal/Presentation"
DST="$HOME/HARI_UBUNTU_HOME/Shared Files/Hierarchical_RL_GH"
echo "→ rendering Presentation.qmd"
(cd "$SRC" && quarto render Presentation.qmd)
echo "→ source: $SRC"
echo "→ dest:   $DST"

# ---- slides sources ----
mkdir -p "$DST/slides/data" "$DST/slides/derivations"
mkdir -p "$DST/code" "$DST/slides"
cp "$SRC/derivations/index.html" "$DST/slides/derivations/" 2>/dev/null || true

cp -f "$SRC/Presentation.qmd"  "$DST/slides/"
cp -f "$SRC/Presentation.html" "$DST/slides/"
cp -f "$SRC/custom.scss"       "$DST/slides/"
cp -f "$SRC/data/papers.csv"   "$DST/slides/data/"

# ---- Quarto render assets (so Presentation.html works on Pages) ----
rsync -a --delete "$SRC/Presentation_files/" "$DST/slides/Presentation_files/"

# ---- all derivation PDFs ----
rsync -a "$SRC/derivations/" "$DST/slides/derivations/"

# ---- figures (both places get the same set) ----
rsync -a --delete "$SRC/figures/" "$DST/figures/"
rsync -a --delete "$SRC/figures/" "$DST/gallery/"
rsync -a --delete "$SRC/figures/" "$DST/slides/figures/"

echo "→ sync complete. Files now in $DST:"
echo "    slides/       $(ls -1 "$DST/slides"      | wc -l) entries"
echo "    slides/deriv/ $(ls -1 "$DST/slides/derivations" | wc -l) PDFs"
echo "    figures/      $(ls -1 "$DST/figures"      | wc -l) files"
