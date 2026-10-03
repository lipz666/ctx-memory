#!/bin/bash
# BEAM 100K test split (conversations 11-20) for ctx-m, Mem0, Cognee and Hindsight, one
# system at a time so a 16 GB machine is not overcommitted. Results: results/beam/100K-test.
# Needs CTX_GW_KEY in the environment, a release build of ctx (local embeddings), and for
# Cognee / Hindsight their venvs (bench/external/README.md).
#   nohup tools/run_beam_test_split.sh [ctxm] [mem0] [cognee] [hindsight] &   (default: all)
set -u
cd "$(dirname "$0")/.." || exit 1
export BENCH_MIN_AVAILABLE_GB=3
OUT=results/beam/100K-test
CONVS=11,12,13,14,15,16,17,18,19,20
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
beam() { .venv/bin/python -m tracks.beam --split 100K --conversations "$CONVS" --out "$OUT" "$@"; }

run_ctxm() {
log "ctx-m"
beam --systems ctxm-v1-render,ctxm-v1-render.r2 --budget 1000 --workers 10 --workdir beam-ctxm > "$OUT/ctxm.log" 2>&1
log "ctx-m exit=$?"
}

run_mem0() {
log "mem0"
beam --systems mem0-1k --budget 1000 --workers 5 --workdir beam-mem0 > "$OUT/mem0.log" 2>&1 &&
beam --systems mem0-8k --budget 8000 --workers 5 --workdir beam-mem0 >> "$OUT/mem0.log" 2>&1
log "mem0 exit=$?"
}

run_cognee() {
log "cognee"
rm -f beam-cognee/embedder.json
.venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from adapters.ctx import CtxService
s = CtxService('beam-cognee/embedder-home', embed_workers=2).start()
open('beam-cognee/embedder.json','w').write(json.dumps({'port': s.port, 'token': s.token}))
s.process.wait()
" > beam-cognee/embedder.log 2>&1 &
EMBED=$!
until [ -f beam-cognee/embedder.json ]; do sleep 2; done
(source tools/cognee.env.sh && cd beam-cognee &&
 ../external/cognee-venv/bin/python ../tools/cognee_beam.py prepare --parquet ../data/beam/100K.parquet --start 10 --conversations 10 --out prepared-test) > "$OUT/cognee-prepare.log" 2>&1
ls beam-cognee/prepared-test/100k | PREPARED=beam-cognee/prepared-test xargs -P 2 -n 1 tools/cognee_ingest_one.sh > "$OUT/cognee-ingest.log" 2>&1
beam --systems cognee-1k --budget 1000 --workers 2 --workdir beam-cognee > "$OUT/cognee.log" 2>&1 &&
beam --systems cognee-native --budget 1000 --workers 2 --workdir beam-cognee >> "$OUT/cognee.log" 2>&1
log "cognee exit=$?"
kill $EMBED; pkill -f "beam-cognee/embedder-home"
}

run_hindsight() {
log "hindsight"
(cd external && HINDSIGHT_API_HOST=127.0.0.1 HINDSIGHT_API_LLM_PROVIDER=openai HINDSIGHT_API_LLM_BASE_URL=https://vps.lpzproxy.xyz/v1 \
  HINDSIGHT_API_LLM_MODEL=gemini-3.8-flash-high HINDSIGHT_API_LLM_API_KEY="$CTX_GW_KEY" HINDSIGHT_API_PORT=8890 \
  HINDSIGHT_API_DATABASE_URL=pg0://hindsight-bench HINDSIGHT_API_LLM_TIMEOUT=300 hindsight-venv/bin/hindsight-api >> hindsight-run/server.log 2>&1) &
HS=$!
until curl -s -m 3 127.0.0.1:8890/health >/dev/null; do sleep 3; done
beam --systems hindsight-1k,hindsight-amb --budget 1000 --workers 5 --workdir beam-hindsight > "$OUT/hindsight.log" 2>&1
log "hindsight exit=$?"
pkill -f "hindsight-venv/bin/hindsight-api"; kill $HS 2>/dev/null
}

for stage in "${@:-ctxm mem0 cognee hindsight}"; do
  for name in $stage; do "run_$name"; done
done
log "ALL DONE"
