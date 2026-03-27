# Local LLM Toolkit

Local-first Python toolkit for running GGUF models through `llama-server`, with:
- interactive chat (`chat.py`)
- hierarchical PDF RAG (`rag.py`)
- per-agent working memory
- optional two-agent critique loop
- model download helper (`models.py`)

All behavior is driven by `settings.json` plus optional CLI overrides.

## What This Project Optimizes For

- Keep model calls simple and controllable.
- Avoid sending full rolling chat history by default.
- Retrieve only relevant context from memory/docs.
- Keep retrieval and runtime settings easy to tune.

## Architecture (High Level)

Core modules in `local_llm/`:

- `settings.py`: loads and validates `settings.json` into typed config.
- `chat_runtime.py`: runtime option models used by chat execution.
- `cli_chat.py`: CLI entrypoint for chat, option resolution, store setup.
- `client.py`: interactive loop + HTTP calls to `llama-server`.
- `context_pipeline.py`: context assembly (working memory + RAG + loop context).
- `working_memory.py`: event store, chunking, embedding/BM25 index, retrieval.
- `rag.py`: PDF ingest, chunking, embedding/BM25 index, retrieval.
- `llama_server.py`: builds server command with compatibility checks.
- `model_overrides.py`: applies per-model chat/server overrides.
- `persistent_storage.py`: cache path defaults for persistent volumes (RunPod).

Thin wrappers at repo root:

- `chat.py` -> `local_llm.cli_chat.main`
- `server.py` -> `local_llm.cli_server.main`
- `rag.py` -> `local_llm.cli_rag.main`
- `models.py` -> `local_llm.cli_models.main`

## Request Flow

### Standard chat turn

1. Read user input.
2. Build context via `collect_context_for_query(...)`:
   - explicit last-`N` rounds context from recent completed turns
   - working-memory retrieval (recent lane + older lane)
   - PDF RAG retrieval
   - concise JSON context envelope assembly
3. Send one request to `llama-server` with:
   - system prompt
   - current user prompt
   - optional injected supplemental context JSON
4. Stream response to terminal.
5. Persist user/assistant turn into working memory (if enabled).

### Agent-loop mode

Per user question:

1. Agent 1 answers directly.
2. Wait for control input:
   - `Enter`: run critique cycle
   - new text: start a new question
   - `/clear`, `/clear-memory`, `exit`: control commands
3. Critique cycle on each `Enter`:
   - Agent 2 critiques Agent 1
   - Agent 1 revises/defends
   - saved as one completed round
4. Last-`N` rounds context is always injected as explicit context (`runtime.last_n_rounds`, default `1`).

## Context Pipeline Behavior

`context_pipeline.py` controls what gets fed to the model:

- RAG gate: `runtime.rag.mode` with thresholds/margins in `auto` mode.
- Working-memory gate: `runtime.working_memory.mode` with rerank thresholds/margins in `auto` mode.
- Two memory lanes:
  - recent lane (`recent_window_rounds`, `recent_top_k_rounds`)
  - older lane (`older_top_k_rounds`)
- Recency exclusion:
  - `exclude_latest_turns` excludes newest turns from both recent and older retrieval.
- Parent-only context contract:
  - child chunks are used internally for retrieval/reranking only.
  - model context contains parent/round context plus metadata, not child snippets.
- Final envelope:
  - model input is JSON with `working_memory`, `knowledge`, and `user_query`.
  - `working_memory` is a single chronological list (oldest-to-newest) of entries with `round_index`, `role`, and `content`.
  - `knowledge` is a list of supplemental entries with `source` and `content`.
  - retrieval uses the current turn text directly as the lookup query (no separate retrieval agent).

## RAG System (PDF)

Ingest (`rag.py ingest`):

1. Extract and normalize PDF text.
2. Detect section-like headings.
3. Build parent chunks (large sections).
4. Build child chunks (smaller retrieval units).
5. Embed children with sentence-transformers.
6. Build BM25 over child text.
7. Overwrite the target store directory and persist fresh artifacts.

Retrieve (`HierarchicalRagStore.retrieve`):

1. Build embedding query + BM25 tokens.
2. Run vector similarity + BM25.
3. Union candidates and compute hybrid score.
4. Optional cross-encoder rerank.
5. Keep top reranked child candidates, then promote/rank related parent sections.
6. Expose parent-only retrieval output plus child-score metadata for gating decisions.

## Working Memory System

Working memory is session/agent scoped and stores only user/assistant text events.

Write path:

1. `append_turn(user, assistant)` normalizes text.
2. Appends two events (user + assistant) at same `turn_index`.
3. Rebuilds parent/child chunks from events.
4. Rebuilds child embeddings + BM25.
5. Persists manifest/events/chunks/embeddings/index.

Read path (`retrieve`):

1. Filter eligible turns by lane bounds and excluded latest turns.
2. Run vector + BM25 hybrid retrieval over child chunks internally.
3. Optional rerank and aggregate scores to round-level parent hits.
4. Return parent/round hits with metadata (parent-only output).

## Data Layout

### RAG store (`data/rag_store/<name>`)

- `manifest.json`
- `parents.json`
- `children.json`
- `child_embeddings.npy`
- `bm25_index.json`

### Working memory (`data/working_memory/<agent_id>/<session_id>`)

- `manifest.json`
- `events.json`
- `parents.json`
- `children.json`
- `child_embeddings.npy`
- `bm25_index.json`

## Setup

Requirements:

- Python 3.10+
- `llama-server` installed
- GPU recommended for larger models

Install dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

```bash
python3 -m pip install -r requirements.txt
```

## Quick Start

1. Configure `settings.json`.
2. Download a model.
3. Start server.
4. Start chat in another terminal.

Windows:

```powershell
py -3 models.py --model qwen35_35b
py -3 server.py --model qwen35_35b
py -3 chat.py --model qwen35_35b
```

Linux:

```bash
python3 models.py --model qwen35_35b
python3 server.py --model qwen35_35b
python3 chat.py --model qwen35_35b
```

## Useful Commands

Chat:

```powershell
py -3 chat.py --list-models
py -3 chat.py --model qwen35_35b
py -3 chat.py --model qwen35_35b --rag-store .\data\rag_store\main
py -3 chat.py --model qwen35_35b --working-memory-mode auto
py -3 chat.py --model qwen35_35b --last-n-rounds 1 --rag-top-k 8 --working-memory-recent-window-rounds 5 --working-memory-recent-top-k-rounds 8 --working-memory-older-top-k-rounds 5
py -3 chat.py --model qwen35_35b --agent-loop
py -3 chat.py --model qwen35_35b --md-log .\data\chat_logs\session.md
```

Server:

```powershell
py -3 server.py --list-models
py -3 server.py --model qwen35_35b
py -3 server.py --model qwen35_35b --dry-run
```

Model download:

```powershell
py -3 models.py --list-models
py -3 models.py --model qwen35_35b
py -3 models.py --all
```

RAG ingest/query:

```powershell
py -3 rag.py --config settings.json ingest --pdf-dir .\data\pdfs --store .\data\rag_store\main
py -3 rag.py --config settings.json query --store .\data\rag_store\main --question "What are the warranty limits?" --top-k 8
```

## Key Settings (`settings.json`)

- `default_model`: alias selected when no `--model` is passed.
- `models.<alias>`: GGUF path, HF download metadata, and per-model overrides.
- `chat`: endpoint + generation settings + retries + system prompt.
- `server`: `llama-server` launch settings.
- `runtime.last_n_rounds`: explicit short-term round context size (default `1`).
- `runtime.debug_output`: show/hide retrieval debug prints (`retrieval_query`, context payload, and retrieval status lines). Default `false`.
- `runtime.rag`: RAG gates, top-k values, context size, chunk settings.
- `runtime.working_memory`: memory gates, recent/older top-k, and exclusion controls.
- `runtime.agent_loop`: critique loop enablement, naming, prompts.

Important behavior notes:

- Standard chat does not rely on full rolling history as prompt input.
- Context is assembled from retrieval lanes and gated before inclusion.
- Context injection is JSON-first (`working_memory`, `knowledge`, `user_query`) and last-round scope is controlled by `runtime.last_n_rounds`.
- Public retrieval/context output is parent-only; child chunks are internal retrieval artifacts.

## RunPod / Persistent Volumes

If `/workspace` exists (or `LOCAL_LLM_VOLUME_ROOT` is set), cache env defaults are pinned to persistent storage:

- `HF_HOME`
- `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE`
- `TRANSFORMERS_CACHE`
- `SENTENCE_TRANSFORMERS_HOME`
- `PIP_CACHE_DIR`

Helper scripts:

- `scripts/setup-runpod-git.sh`
- `scripts/start-runpod-git.sh`

### RunPod Quickstart (Scripts + Chat/Agent + RAG)

1. Deploy a Linux GPU Pod and attach a persistent volume (typically mounted at `/workspace`).
2. SSH into the Pod, then clone and install:

```bash
cd /workspace
git clone <YOUR_REPO_URL> heretic-models
cd heretic-models
bash scripts/setup-runpod-git.sh
source /workspace/.local_llm_runpod_env.sh
```

3. (Optional, one-time per PDF set) build a RAG store:

```bash
python3 rag.py --config settings.json ingest --pdf-dir data/pdfs --store data/rag_store/main
```

4. Start `llama-server` with scripts:

- First run on a fresh Pod (download model now):

```bash
MODEL_ALIAS=qwen35_35b DOWNLOAD_MODEL=1 bash scripts/start-runpod-git.sh
```

- Model already downloaded:

```bash
MODEL_ALIAS=qwen35_35b DOWNLOAD_MODEL=0 bash scripts/start-runpod-git.sh
```

- Validate startup command without launching:

```bash
SERVER_DRY_RUN=1 bash scripts/start-runpod-git.sh
```

Notes:
- `DOWNLOAD_MODEL` defaults to `0`.
- If `MODEL_ALIAS` is omitted, `default_model` from `settings.json` is used.
- `CONFIG_PATH` can point to a non-default config file (default: `settings.json`).
- `setup-runpod-git.sh` writes a reusable env file: `/workspace/.local_llm_runpod_env.sh`.

5. In a second terminal/session, launch chat:

- Standard chat:

```bash
python3 chat.py --model qwen35_35b
```

- Standard chat with RAG injected:

```bash
python3 chat.py --model qwen35_35b --rag-store data/rag_store/main
```

- Agent chat (two-agent critique loop) with RAG:

```bash
python3 chat.py --model qwen35_35b --agent-loop --rag-store data/rag_store/main
```

6. No-thinking mode:

```bash
python3 chat.py --model qwen35_35b --disable-thinking
python3 chat.py --model qwen35_35b --agent-loop --rag-store data/rag_store/main --disable-thinking
```

Notes:
- This repo already defaults to no-thinking in `settings.json` (`chat.disable_thinking: true`).
- `--disable-thinking` only affects models with `supports_thinking_toggle: true`.

## Troubleshooting

- `llama-server binary not found`:
  - fix `server.llama_server_path` or `server.llama_server_paths`
- `GGUF model file not found`:
  - verify `models.<alias>.gguf_path`
- empty/weak RAG results:
  - ensure PDFs are text-extractable
  - tune `runtime.rag` thresholds/top-k
- weak memory retrieval:
  - tune `runtime.working_memory` thresholds, `recent_window_rounds`, and top-k settings
- slow responses/timeouts:
  - reduce `chat.max_tokens`
  - reduce context chars and top-k settings
  - tune server `ctx_size`/batch settings
