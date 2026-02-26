# DWARF RAG

Semantic search over all DWARF-related documents using Ollama embeddings + ChromaDB.

## Setup

ChromaDB installed in the DWARF venv: `/home/dlewis3/Desktop/AI/DWARF/.venv`
Embedding server: `http://192.168.64.126:11434` (Ollama, separate machine — no 4090/3090 usage)
Embedding model: `qwen3-embedding:0.6b`
Vector DB: `rag/chroma_db/` (persistent, ~1538 chunks across 85 documents)

## Sources Indexed

| Source | Type | Path |
|---|---|---|
| Holocron DWARF notes | `holocron_note` | `~/Holocron/Holocron/Claude/DWARF/` |
| All Claude journals | `journal` | `~/Holocron/Holocron/Journal/2026-*-claude*.md` |
| Writeup draft | `writeup_draft` | `~/.openclaw/workspace/wave-field-writeup-draft.md` |
| feedback.md | `feedback` | `~/.openclaw/workspace/feedback*.md` |
| Training logs | `training_log` | `./benchmarks/logs/*.log` |

## Usage

```bash
cd /home/dlewis3/Desktop/AI/DWARF

# Basic query
.venv/bin/python3 rag/query.py "coverage density collapse attractor"

# More results
.venv/bin/python3 rag/query.py "ELU normalizer variance" --n 8

# Filter by source type
.venv/bin/python3 rag/query.py "condN architecture" --type holocron_note
.venv/bin/python3 rag/query.py "passkey result" --type training_log
.venv/bin/python3 rag/query.py "two memory systems" --type journal

# Filter by date
.venv/bin/python3 rag/query.py "interference pooling" --date 2026-02-25

# Full text (not truncated)
.venv/bin/python3 rag/query.py "condP gap 33-47" --full

# JSON output (for scripting)
.venv/bin/python3 rag/query.py "copy attractor standard transformer" --json
```

## Maintenance

```bash
# Full rebuild (after major doc additions)
.venv/bin/python3 rag/ingest.py

# Incremental update (add new/changed docs only)
.venv/bin/python3 rag/ingest.py --update
```

Run `--update` after:
- New training logs arrive (condP, condQ, etc.)
- New journal entries written
- Writeup draft updated
- feedback.md updated

## Bidirectional Comms Connection

This RAG index is the memory layer for future inter-session communication.
Sessions can query it to get oriented on recent work without loading full files.
The notification layer (push-based realtime) is a separate project — same DB,
different access pattern.
