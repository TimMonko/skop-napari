#!/usr/bin/env bash
# napari with the Ops panel listing one or more op collections.
#
#   ./napari-ops.sh       every installed one: scikit-ops and tnia-ops
#   ./napari-ops.sh 1     scikit-ops only
#   ./napari-ops.sh 2     tnia-ops only
#
# Anything after the choice goes to napari, e.g. an image to open. tnia-ops
# comes from the sibling checkout (../tnia-ops), added for this run only;
# skop-napari's .venv is left as it is. The op worker's stdout and stderr
# print in this terminal (SKOP_NAPARI_DEBUG).
set -euo pipefail
cd "$(dirname "$0")"

choice="${1:-0}"
case "$choice" in
    0) collections="" ;;
    1) collections="scikit-ops" ;;
    2) collections="tnia-ops" ;;
    *) echo "usage: $0 [0|1|2] [napari args...]  (0 both, 1 scikit-ops, 2 tnia-ops)" >&2
       exit 2 ;;
esac
[ $# -gt 0 ] && shift

SKOP_NAPARI_COLLECTION="$collections" SKOP_NAPARI_DEBUG=1 \
    uv run --offline --with-editable ../tnia-ops napari "$@"
