#!/usr/bin/env python3
"""
DWARF RAG — Ingestion Script
Collects all DWARF-related documents, chunks them, embeds via Ollama,
and stores in a persistent ChromaDB collection.

Sources indexed:
  - Holocron DWARF notes       (~/.../Holocron/Claude/DWARF/)
  - All Claude journals        (~/.../Holocron/Journal/2026-*-claude*.md)
  - Writeup draft              (~/.openclaw/workspace/wave-field-writeup-draft.md)
  - Training logs              (./benchmarks/logs/)
  - feedback.md                (~/.openclaw/workspace/feedback.md)

Usage:
  python3 rag/ingest.py          # full ingest (clears + rebuilds)
  python3 rag/ingest.py --update # add/update changed docs only
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Generator

import chromadb
import requests

# ── Config ───────────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://192.168.64.126:11434"
EMBED_MODEL   = "qwen3-embedding:0.6b"
DB_PATH       = Path(__file__).parent / "chroma_db"
COLLECTION    = "dwarf"

CHUNK_SIZE    = 800    # characters
CHUNK_OVERLAP = 150    # characters

HOLOCRON      = Path.home() / "Holocron/Holocron"
WORKSPACE     = Path.home() / ".openclaw/workspace"
DWARF_REPO    = Path(__file__).parent.parent

SOURCES = {
    "dwarf_notes": {
        "path":    HOLOCRON / "Claude/DWARF",
        "pattern": "*.md",
        "type":    "holocron_note",
        "recurse": False,
    },
    "journals": {
        "path":    HOLOCRON / "Journal",
        "pattern": "2026-*-claude*.md",
        "type":    "journal",
        "recurse": False,
    },
    "writeup_draft": {
        "path":    WORKSPACE,
        "pattern": "wave-field-writeup-draft.md",
        "type":    "writeup_draft",
        "recurse": False,
    },
    "feedback": {
        "path":    WORKSPACE,
        "pattern": "feedback*.md",
        "type":    "feedback",
        "recurse": False,
    },
    "training_logs": {
        "path":    DWARF_REPO / "benchmarks/logs",
        "pattern": "*.log",
        "type":    "training_log",
        "recurse": False,
    },
}

# ── Ollama embedding ──────────────────────────────────────────────────────────

def embed(texts: list[str]) -> list[list[float]]:
    """Batch embed texts via Ollama."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Chunk on markdown headers first, then fall back to sliding window.
    Preserves header context by prepending the last seen h1/h2 to each chunk.
    """
    # Split on markdown headers (##, ###) while keeping the header line
    header_re = re.compile(r'^(#{1,3} .+)$', re.MULTILINE)
    parts = header_re.split(text)

    chunks = []
    current_header = ""
    buffer = ""

    def flush(buf: str, header: str):
        buf = buf.strip()
        if not buf:
            return
        # Prefix with header context if not already present
        full = (header + "\n\n" + buf) if header and not buf.startswith(header) else buf
        # Sliding window if still too long
        if len(full) <= size:
            chunks.append(full)
        else:
            for i in range(0, len(full), size - overlap):
                piece = full[i:i + size].strip()
                if piece:
                    chunks.append(piece)

    for part in parts:
        if header_re.match(part):
            flush(buffer, current_header)
            current_header = part.strip()
            buffer = ""
        else:
            buffer += part

    flush(buffer, current_header)
    return [c for c in chunks if len(c) > 60]  # skip tiny fragments


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ── Document loading ──────────────────────────────────────────────────────────

def iter_files() -> Generator[tuple[str, Path, dict], None, None]:
    """Yield (source_name, path, metadata) for all configured sources."""
    for source_name, cfg in SOURCES.items():
        base = cfg["path"]
        if not base.exists():
            print(f"  [skip] {base} — not found")
            continue
        glob = "**/" + cfg["pattern"] if cfg["recurse"] else cfg["pattern"]
        for path in sorted(base.glob(glob)):
            if path.is_file() and path.stat().st_size > 0:
                meta = {
                    "source":    source_name,
                    "doc_type":  cfg["type"],
                    "filename":  path.name,
                    "filepath":  str(path),
                }
                # Extract date from filename if present (YYYY-MM-DD)
                m = re.search(r'(\d{4}-\d{2}-\d{2})', path.name)
                if m:
                    meta["date"] = m.group(1)
                yield source_name, path, meta


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest(update_mode: bool = False):
    client = chromadb.PersistentClient(path=str(DB_PATH))

    if not update_mode:
        # Full rebuild: delete existing collection
        try:
            client.delete_collection(COLLECTION)
            print(f"Cleared existing '{COLLECTION}' collection.")
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Load existing doc_ids for update mode
    existing_ids: set[str] = set()
    if update_mode:
        existing = col.get(include=[])
        existing_ids = set(existing["ids"])

    total_chunks = 0
    total_docs   = 0
    skipped      = 0

    print(f"\nIngesting DWARF documents → ChromaDB at {DB_PATH}")
    print(f"Mode: {'update' if update_mode else 'full rebuild'}\n")

    for source_name, path, base_meta in iter_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [error] {path.name}: {e}")
            continue

        doc_hash = file_hash(path)
        doc_id_prefix = f"{source_name}::{path.name}::{doc_hash}"

        # In update mode, skip if already indexed (same hash = same content)
        if update_mode:
            # Check if any chunk from this file+hash exists
            sample_id = f"{doc_id_prefix}::0"
            if sample_id in existing_ids:
                skipped += 1
                continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        print(f"  {path.name:<55} {len(chunks):>3} chunks", end="", flush=True)

        # Embed in small batches to avoid timeouts
        BATCH = 16
        ids, embeddings, documents, metadatas = [], [], [], []

        for i in range(0, len(chunks), BATCH):
            batch_chunks = chunks[i:i + BATCH]
            try:
                batch_embs = embed(batch_chunks)
            except Exception as e:
                print(f"\n    [embed error] {e}")
                continue

            for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embs)):
                chunk_id = f"{doc_id_prefix}::{i + j}"
                meta = dict(base_meta)
                meta["chunk_index"] = i + j
                meta["doc_hash"]    = doc_hash
                ids.append(chunk_id)
                embeddings.append(emb)
                documents.append(chunk)
                metadatas.append(meta)

        if ids:
            col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            total_chunks += len(ids)

        total_docs += 1
        print(f"  ✓")

    print(f"\n{'─'*60}")
    print(f"  Documents processed : {total_docs}")
    if update_mode:
        print(f"  Documents skipped   : {skipped} (unchanged)")
    print(f"  Total chunks stored : {total_chunks}")
    print(f"  Collection size     : {col.count()} chunks")
    print(f"  DB path             : {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DWARF docs into ChromaDB")
    parser.add_argument("--update", action="store_true",
                        help="Add/update changed documents only (skip unchanged)")
    args = parser.parse_args()
    ingest(update_mode=args.update)
