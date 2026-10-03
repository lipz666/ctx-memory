# Environment for bench/tools/cognee_beam.py: the benchmark gateway model and the local
# EmbeddingGemma service (beam-cognee/embedder.json). The key comes from CTX_GW_KEY.
_emb="$(dirname "${BASH_SOURCE[0]:-$0}")/../beam-cognee/embedder.json"
export LLM_PROVIDER=custom
export LLM_MODEL="openai/${BENCH_MODEL:-gemini-3.8-flash-high}"
export LLM_ENDPOINT="${BENCH_BASE_URL:-https://vps.lpzproxy.xyz/v1}"
export LLM_API_KEY="$CTX_GW_KEY"
export EMBEDDING_PROVIDER=openai_compatible
export EMBEDDING_MODEL=embeddinggemma
export EMBEDDING_DIMENSIONS=768
export EMBEDDING_BATCH_SIZE=128
export EMBEDDING_ENDPOINT="http://127.0.0.1:$(python3 -c "import json;print(json.load(open('$_emb'))['port'])")/api/v1"
export EMBEDDING_API_KEY="$(python3 -c "import json;print(json.load(open('$_emb'))['token'])")"
export TELEMETRY_DISABLED=1
