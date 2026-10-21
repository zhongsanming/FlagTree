#!/bin/bash
# Convenience wrapper for commonir_bench.py -- run this on the NPU box.
#
#   ./run_example.sh                # selfcheck + quick smoke run with IR dumps
#   ./run_example.sh full           # default case set, IR dumps, pack a tarball
#   ./run_example.sh perf           # default case set, no artifacts (perf only)
#   ./run_example.sh compare DIR    # CPU-only re-compare + re-report of DIR
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PYTHON:-python3}"
STAMP="$(date +%Y%m%d-%H%M%S)"

case "${1:-smoke}" in
  smoke)
    "$PY" commonir_bench.py selfcheck
    "$PY" commonir_bench.py list
    "$PY" commonir_bench.py run --outdir "out/smoke-$STAMP" \
          --cases quick --dump-ir
    ;;
  full)
    "$PY" commonir_bench.py selfcheck
    OUT="out/full-$STAMP"
    "$PY" commonir_bench.py run --outdir "$OUT" --cases default \
          --dump-ir --iters 20 --rounds 5 --warmup 5
    tar Jcf "$OUT.tar.xz" "$OUT"
    echo "packed: $OUT.tar.xz  ($(du -h "$OUT.tar.xz" | cut -f1))"
    ;;
  perf)
    "$PY" commonir_bench.py run --outdir "out/perf-$STAMP" --cases default \
          --no-save-artifacts --iters 50 --rounds 7
    ;;
  compare)
    DIR="${2:?usage: $0 compare <outdir>}"
    "$PY" commonir_bench.py compare --outdir "$DIR"
    "$PY" commonir_bench.py report  --outdir "$DIR"
    ;;
  *)
    echo "usage: $0 [smoke|full|perf|compare <outdir>]" >&2
    exit 2
    ;;
esac
