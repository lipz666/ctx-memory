"""Minimal MCP stdio server over Mem0 OSS, exposing the two core tools of Mem0's official
MCP server (add_memory, search_memories). Namespace: MEM0_NS; storage: MEM0_DIR.
LLM: the benchmark gateway model; embedder: ctx's OpenAI-compatible endpoint (EMBED_URL,
EMBED_TOKEN)."""
import json
import os
import sys

os.environ.setdefault("MEM0_TELEMETRY", "False")
from mem0 import Memory  # noqa: E402

NS = os.environ["MEM0_NS"]
memory = Memory.from_config({
    "llm": {"provider": "openai", "config": {"model": os.environ.get("BENCH_MODEL", "gemini-3.8-flash-high"), "temperature": 0,
                                             "api_key": os.environ["GW_KEY"], "openai_base_url": os.environ["GW_URL"], "max_tokens": 2000}},
    "embedder": {"provider": "openai", "config": {"model": "embeddinggemma", "api_key": os.environ["EMBED_TOKEN"],
                                                  "openai_base_url": os.environ["EMBED_URL"], "embedding_dims": 768}},
    "vector_store": {"provider": "qdrant", "config": {"collection_name": "mem", "path": os.path.join(os.environ["MEM0_DIR"], "qdrant"),
                                                      "embedding_model_dims": 768, "on_disk": True}},
    "history_db_path": os.path.join(os.environ["MEM0_DIR"], "history.db"),
})
TOOLS = [
    {"name": "add_memory", "description": "Store information in long-term memory: preferences, facts, conventions, decisions and anything worth remembering for future sessions.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string", "description": "What to remember"}}, "required": ["text"]}},
    {"name": "search_memories", "description": "Search long-term memory for relevant information from past sessions using a natural-language query.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}},
]


def call(name, args):
    if name == "add_memory":
        result = memory.add([{"role": "user", "content": args["text"]}], user_id=NS)
        return {"stored": [r.get("memory") for r in result.get("results", [])]}
    if name == "search_memories":
        result = memory.search(args["query"], filters={"user_id": NS}, limit=int(args.get("limit") or 5))
        return [{"memory": r["memory"], "score": round(r.get("score") or 0, 3)} for r in result.get("results", [])]
    raise ValueError("unknown tool")


for line in sys.stdin:
    if not line.strip():
        continue
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request.get("method")
    try:
        if method == "initialize":
            result = {"protocolVersion": request.get("params", {}).get("protocolVersion", "2025-06-18"),
                      "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "mem0", "version": "bench"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            value = call(request["params"]["name"], request["params"].get("arguments") or {})
            result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}
        elif method == "ping":
            result = {}
        else:
            raise ValueError(f"unknown method {method}")
        response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
    except Exception as error:  # noqa: BLE001
        response = {"jsonrpc": "2.0", "id": request["id"], "result": {"content": [{"type": "text", "text": str(error)}], "isError": True}}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
