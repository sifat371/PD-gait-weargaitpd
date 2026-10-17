#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$(realpath "${1:?Usage: bash scripts/run_analysis.sh /path/to/authorized-data}")"
[[ -d "$DATA_DIR" ]] || { echo "Input directory does not exist" >&2; exit 2; }
[[ "$DATA_DIR" != "$ROOT"* ]] || { echo "Keep source data outside Git" >&2; exit 2; }
(cd "$DATA_DIR" && sha256sum -c "$ROOT/data_sha256.txt")
OUT_ROOT="$(realpath -m "${WEARGAIT_OUTPUT_DIR:-$HOME/weargait-analysis-runs}")"
[[ "$OUT_ROOT" != "$ROOT"* ]] || { echo "Keep runtime results outside Git" >&2; exit 2; }
mkdir -p "$OUT_ROOT"
RUN="$OUT_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -m 700 "$RUN"
cp -a "$ROOT/analysis/." "$RUN/"
mkdir -p "$RUN/private-inputs" "$RUN/outputs" "$RUN/logs" "$RUN/metadata" "$RUN/.matplotlib"
(cd "$RUN" && sha256sum -c "$ROOT/analysis/reference/code_sha256.txt")
ln -s "$DATA_DIR/PD - Demographic+Clinical - datasetV1.csv" "$RUN/private-inputs/PD - Demographic+Clinical - datasetV1(8).csv"
ln -s "$DATA_DIR/CONTROLS - Demographic+Clinical - datasetV1.csv" "$RUN/private-inputs/CONTROLS - Demographic+Clinical - datasetV1(8).csv"
ln -s "$DATA_DIR/PKMAS Walkway Gait Metrics - HP+SP.csv" "$RUN/private-inputs/PKMAS Walkway Gait Metrics - HP+SP(10).csv"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.13 "$RUN/.venv" --seed
  uv pip install --python "$RUN/.venv/bin/python" -r "$ROOT/requirements.lock.txt"
elif command -v python3.13 >/dev/null 2>&1; then
  python3.13 -m venv "$RUN/.venv"
  "$RUN/.venv/bin/python" -m pip install -r "$ROOT/requirements.lock.txt"
else
  echo "Python 3.13 or uv is required" >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR="$RUN/.matplotlib"
"$RUN/.venv/bin/python" "$RUN/verify_models.py" 2>&1 | tee "$RUN/logs/models.log"
"$RUN/.venv/bin/python" "$RUN/run-code/weargait_site_audit_primary.py" 2>&1 | tee "$RUN/logs/primary.log"
"$RUN/.venv/bin/python" "$RUN/run-code/weargait_site_audit_additional.py" 2>&1 | tee "$RUN/logs/additional.log"
"$RUN/.venv/bin/python" "$ROOT/scripts/check_results.py" --workspace "$RUN" --out "$RUN/metadata/check.json"
echo "Analysis complete: $RUN"
echo "Do not publish the complete working directory; it may contain participant-level records."
