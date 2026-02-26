#!/usr/bin/env python3
"""
DWARF RAG — Query CLI
Semantic search over indexed DWARF documents.

Usage:
  python3 rag/query.py "coverage density structural regularization"
  python3 rag/query.py "condP ep1 result" --n 5
  python3 rag/query.py "ELU normalizer" --type holocron_note
  python3 rag/query.py "passkey retrieval" --date 2026-02-25
  python3 rag/query.py "copy attractor standard transformer" --full

Options:
  --n N          Number of results (default: 5)
  --type TYPE    Filter by doc_type: holocron_note | journal | writeup_draft |
                                     training_log | feedback
  --date DATE    Filter by date (YYYY-MM-DD)
  --full         Print full chunk text (default: truncated to 400 chars)
  --json         Output raw JSON
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
import requests

OLLAMA_URL  = "http://192.168.64.126:11434"
EMBED_MODEL = "qwen3-embedding:0.6b"
DB_PATH     = Path(__file__).parent / "chroma_db"
COLLECTION  = "dwarf"


def embed_one(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def query(
    q: str,
    n: int = 5,
    doc_type: str | None = None,
    date: str | None = None,
    full: bool = False,
    as_json: bool = False,
):
    if not DB_PATH.exists():
        print("No ChromaDB found. Run `python3 rag/ingest.py` first.", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    col    = client.get_collection(COLLECTION)

    # Build optional where filter
    where = None
    conditions = []
    if doc_type:
        conditions.append({"doc_type": {"$eq": doc_type}})
    if date:
        conditions.append({"date": {"$eq": date}})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    query_emb = embed_one(q)
    results = col.query(
        query_embeddings=[query_emb],
        n_results=min(n, col.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    if as_json:
        out = [
            {"rank": i+1, "score": round(1-d, 4), "metadata": m, "text": doc}
            for i, (doc, m, d) in enumerate(zip(docs, metas, distances))
        ]
        print(json.dumps(out, indent=2))
        return

    print(f"\n{'━'*70}")
    print(f"  Query: {q!r}")
    if doc_type or date:
        filters = []
        if doc_type: filters.append(f"type={doc_type}")
        if date:     filters.append(f"date={date}")
        print(f"  Filter: {', '.join(filters)}")
    print(f"  Results: {len(docs)}")
    print(f"{'━'*70}\n")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        score = 1 - dist  # cosine similarity → higher is better
        filename = meta.get("filename", "?")
        doc_type_str = meta.get("doc_type", "?")
        date_str = meta.get("date", "")
        chunk_idx = meta.get("chunk_index", 0)

        header = f"[{i+1}] {filename}"
        if date_str:
            header += f"  ({date_str})"
        header += f"  [{doc_type_str}]  score={score:.3f}  chunk={chunk_idx}"
        print(header)
        print(f"     {meta.get('filepath', '')}")
        print()

        text = doc if full else (doc[:400] + "…" if len(doc) > 400 else doc)
        # Indent the text block
        for line in text.split("\n"):
            print(f"     {line}")
        print(f"\n{'─'*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the DWARF RAG index")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--n",    type=int, default=5,   help="Number of results")
    parser.add_argument("--type", default=None,          help="Filter by doc_type")
    parser.add_argument("--date", default=None,          help="Filter by date (YYYY-MM-DD)")
    parser.add_argument("--full", action="store_true",   help="Print full chunk text")
    parser.add_argument("--json", action="store_true",   help="Output JSON")
    args = parser.parse_args()

    query(
        q=args.query,
        n=args.n,
        doc_type=args.type,
        date=args.date,
        full=args.full,
        as_json=args.json,
    )
