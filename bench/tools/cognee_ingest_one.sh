#!/bin/bash
# Ingest one preprocessed BEAM conversation folder into its own Cognee root.
# Usage: cognee_ingest_one.sh <folder name under beam-cognee/prepared/100k>
cd "$(dirname "$0")/.." || exit 1
folder="$1"; id="${folder##*_id_}"; root="beam-cognee/cognee-store/beam-$id"
[ -f "$root/ingested" ] && { echo "already $id"; exit 0; }
source tools/cognee.env.sh
# Wait for at least 3 GB of available memory before starting (the machine has 16 GB).
until .venv/bin/python -c "import psutil,sys; sys.exit(psutil.virtual_memory().available < 3 * 2**30)"; do sleep 20; done
if external/cognee-venv/bin/python tools/cognee_beam.py ingest --folder "${PREPARED:-beam-cognee/prepared}/100k/$folder" --root "$root" > "beam-cognee/logs/ingest-$id.log" 2>&1; then
  echo "ingested $id"
else
  echo "FAILED $id"
fi
