#!/usr/bin/env bash
# Compile the manuscript and the Supporting Information with the local TeX Live
# installation. Run from this directory:
#
#   ./build.sh          # both documents
#   ./build.sh main     # manuscript only
#   ./build.sh SI       # Supporting Information only
#
# main.tex uses bibtex (elsarticle-num); SI.tex uses biblatex/biber.
set -euo pipefail

TEXBIN="${TEXBIN:-/usr/local/texlive/2024/bin/x86_64-linux}"
PATH="$TEXBIN:$PATH"
export PATH

cd "$(dirname "$0")"

for tool in xelatex bibtex biber; do
    command -v "$tool" >/dev/null || { echo "Missing $tool; set TEXBIN to a TeX Live bin directory" >&2; exit 1; }
done

run() { echo "+ $*"; "$@" >/dev/null; }

build_main() {
    run xelatex -interaction=nonstopmode -halt-on-error main.tex
    run bibtex main
    run xelatex -interaction=nonstopmode -halt-on-error main.tex
    run xelatex -interaction=nonstopmode -halt-on-error main.tex
}

build_si() {
    run xelatex -interaction=nonstopmode -halt-on-error SI.tex
    run biber SI
    run xelatex -interaction=nonstopmode -halt-on-error SI.tex
    run xelatex -interaction=nonstopmode -halt-on-error SI.tex
}

case "${1:-all}" in
    main) build_main ;;
    SI|si) build_si ;;
    all) build_main; build_si ;;
    *) echo "Usage: $0 [main|SI|all]" >&2; exit 2 ;;
esac

for document in main SI; do
    [ -f "$document.pdf" ] || continue
    pages=$(pdfinfo "$document.pdf" 2>/dev/null | awk '/^Pages/ {print $2}')
    undefined=$(grep -c 'undefined' "$document.log" 2>/dev/null || true)
    echo "$document.pdf: ${pages:-?} pages, $undefined undefined-reference warnings"
done
