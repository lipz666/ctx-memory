"""Cognee 1.6.2 on BEAM, run with bench/external/cognee-venv (see bench/adapters/cognee.py and
bench/tools/README-external-systems.md).

Follows Cognee's own BEAM setup (cognee/eval_framework/beam/REPORT.md): preprocessing into
one JSON-list file per batch (one item per turn, LLM compression of the few over-limit
turns), `local_ingest` per conversation (add + cognify with JsonListChunker, session
distillation, global context index), and the reported 100K retrieval: HybridRetriever with
chunks_top_k=20, entities_top_k=20. Each conversation gets its own Cognee root directory,
as Cognee's evaluation uses one pruned instance per conversation.

  python cognee_beam.py prepare --parquet ../data/beam/100K.parquet --conversations 10 --out DIR
  python cognee_beam.py ingest --folder DIR/100k/conversation_... --root ROOT
  python cognee_beam.py serve --root ROOT      # JSON lines: {"query": ...} -> {"context": ...}

Model and embedding settings come from the environment (LLM_*, EMBEDDING_*), set by the caller.
"""
import argparse
import asyncio
import json
import os
import runpy
import sys
from pathlib import Path


def set_root(root):
    root = Path(root).resolve()
    os.environ["SYSTEM_ROOT_DIRECTORY"] = str(root / "system")
    os.environ["DATA_ROOT_DIRECTORY"] = str(root / "data")
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    return root


def prepare(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(args.parquet).to_pylist()[args.start:]
    from cognee.eval_framework.beam.preprocessing import preprocess
    preprocess.load_beam_dataset = lambda split: rows  # the local copy of Mohammadta/BEAM
    sys.argv = ["preprocess", "--dataset", "beam", "--splits", "100K", "--max-conversations", str(args.conversations),
                "--output-dir", str(args.out), "--execute-compressions"]
    asyncio.run(preprocess.main())


def ingest(args):
    root = set_root(args.root)
    sys.argv = ["local_ingest", str(args.folder), "--dataset-name", "beam_100k_local", "--run-dir", str(root / "run"),
                "--prune-first"] + (["--max-sessions", str(args.max_sessions)] if args.max_sessions else [])
    runpy.run_module("cognee.eval_framework.beam.local_ingest", run_name="__main__")
    (root / "ingested").touch()


def serve(args):
    set_root(args.root)
    os.environ.setdefault("CACHING", "true")
    os.environ.setdefault("CACHE_BACKEND", "fs")
    os.environ["LOG_LEVEL"] = "ERROR"
    import logging
    logging.disable(logging.WARNING)
    from cognee.modules.retrieval.hybrid_retriever import HybridRetriever

    async def answer(query):
        retriever = HybridRetriever(chunks_top_k=20, entities_top_k=20)
        objects = await retriever.get_retrieved_objects(query=query)
        context = await retriever.get_context_from_objects(query=query, retrieved_objects=objects)
        return context if isinstance(context, str) else json.dumps(context, ensure_ascii=False, default=str)

    loop = asyncio.new_event_loop()
    out = sys.__stdout__
    print(json.dumps({"ready": True}), file=out, flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            reply = {"context": loop.run_until_complete(answer(request["query"]))}
        except Exception as error:  # noqa: BLE001 - reported to the caller
            reply = {"error": f"{type(error).__name__}: {error}"[:500]}
        print(json.dumps(reply, ensure_ascii=False), file=out, flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--conversations", type=int, default=10)
    p.add_argument("--start", type=int, default=0, help="skip the first N conversations (test split: 10)")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--max-sessions", type=int, default=0, help="smoke tests: first N batches only")
    p = sub.add_parser("serve")
    p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "ingest": ingest, "serve": serve}[args.command](args)


if __name__ == "__main__":
    main()
