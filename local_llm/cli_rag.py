from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .defaults import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RAG_BATCH_SIZE,
    DEFAULT_RAG_CHILD_CHUNK_OVERLAP_WORDS,
    DEFAULT_RAG_CHILD_CHUNK_WORDS,
    DEFAULT_RAG_PARENT_CHUNK_OVERLAP_WORDS,
    DEFAULT_RAG_PARENT_CHUNK_WORDS,
    DEFAULT_RAG_PARENT_CONTEXT_CHARS,
    DEFAULT_RAG_TOP_K,
    DEFAULT_RERANKER_MODEL,
)
from .rag import HierarchicalRagStore, ingest_pdfs_to_store
from .settings import RagDefaultsConfig, load_settings


DEFAULT_STORE_PATH = "data/rag_store/main"
LEGACY_QUERY_FLAGS: dict[str, str] = {
    "--source-k": "--top-k",
    "--parent-k": "--top-k",
    "--child-k": "--top-k",
}


def _coalesce(value: object, fallback: object) -> object:
    return fallback if value is None else value


def _fail_on_legacy_cli_flags(parser: argparse.ArgumentParser) -> None:
    for raw_arg in sys.argv[1:]:
        for legacy_flag, replacement in LEGACY_QUERY_FLAGS.items():
            if raw_arg == legacy_flag or raw_arg.startswith(f"{legacy_flag}="):
                parser.error(
                    f"Flag '{legacy_flag}' is removed in hard-cutover mode. "
                    f"Use {replacement} instead."
                )


def _runtime_rag_defaults(config_path: str | None) -> RagDefaultsConfig:
    if config_path is None:
        return RagDefaultsConfig()
    config = load_settings(config_path)
    return config.runtime.rag


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hierarchical PDF RAG tooling (ingest + query)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to settings JSON for default runtime.rag values",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Extract/clean PDFs, build parent+child chunks, then store vector+BM25 indexes",
    )
    ingest_parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="Path to source PDF (repeat for multiple files)",
    )
    ingest_parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Directory containing source PDFs (*.pdf)",
    )
    ingest_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --pdf-dir for PDFs",
    )
    ingest_parser.add_argument(
        "--store",
        default=DEFAULT_STORE_PATH,
        help=f"Output directory for RAG artifacts (default: {DEFAULT_STORE_PATH})",
    )
    ingest_parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Sentence-transformers embedding model "
            "(defaults to settings.runtime.rag.embedding_model; "
            f"fallback: {DEFAULT_EMBEDDING_MODEL})"
        ),
    )
    ingest_parser.add_argument(
        "--reranker-model",
        default=None,
        help=(
            "Cross-encoder reranker model stored in manifest "
            "(defaults to settings.runtime.rag.reranker_model; "
            f"fallback: {DEFAULT_RERANKER_MODEL}; use 'off' to disable)"
        ),
    )
    ingest_parser.add_argument(
        "--parent-words",
        type=int,
        default=None,
        help="Parent chunk size in words",
    )
    ingest_parser.add_argument(
        "--parent-overlap",
        type=int,
        default=None,
        help="Parent chunk overlap in words",
    )
    ingest_parser.add_argument(
        "--child-words",
        type=int,
        default=None,
        help="Child chunk size in words",
    )
    ingest_parser.add_argument(
        "--child-overlap",
        type=int,
        default=None,
        help="Child chunk overlap in words",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size",
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Query an existing hierarchical RAG store",
    )
    query_parser.add_argument(
        "--store",
        required=True,
        help="Path to an existing RAG store directory",
    )
    query_parser.add_argument(
        "--reranker-model",
        default=None,
        help=(
            "Override reranker model for query-time reranking "
            "(example: BAAI/bge-reranker-v2-m3, or 'off')"
        ),
    )
    query_parser.add_argument("--question", required=True, help="Query text")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Top parent sections to return (default: settings.runtime.rag.top_k or {DEFAULT_RAG_TOP_K})",
    )
    query_parser.add_argument(
        "--parent-context-chars",
        type=int,
        default=None,
        help="Parent context characters included per returned parent section",
    )
    query_parser.add_argument(
        "--json",
        action="store_true",
        help="Print query results as JSON",
    )
    return parser


def _run_ingest(args: argparse.Namespace, *, defaults: RagDefaultsConfig) -> int:
    pdf_inputs: list[Path] = [Path(item) for item in args.pdf]
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir.resolve()}")
        if not pdf_dir.is_dir():
            raise ValueError(f"PDF directory path is not a folder: {pdf_dir.resolve()}")
        glob_pattern = "**/*.[pP][dD][fF]" if args.recursive else "*.[pP][dD][fF]"
        pdf_inputs.extend(sorted(pdf_dir.glob(glob_pattern)))

    if not pdf_inputs:
        raise ValueError("Provide at least one --pdf or a --pdf-dir containing PDF files.")

    resolved_inputs: list[Path] = []
    seen: set[Path] = set()
    for item in pdf_inputs:
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_inputs.append(resolved)

    embedding_model = str(_coalesce(args.embedding_model, defaults.embedding_model))
    reranker_model_value = _coalesce(args.reranker_model, defaults.reranker_model)
    if reranker_model_value is None:
        reranker_model_value = "off"

    parent_words = max(
        120,
        int(_coalesce(args.parent_words, defaults.parent_chunk_words)),
    )
    parent_overlap = max(
        0,
        int(_coalesce(args.parent_overlap, defaults.parent_chunk_overlap_words)),
    )
    child_words = max(
        60,
        int(_coalesce(args.child_words, defaults.child_chunk_words)),
    )
    child_overlap = max(
        0,
        int(_coalesce(args.child_overlap, defaults.child_chunk_overlap_words)),
    )
    if parent_overlap >= parent_words:
        parent_overlap = parent_words - 1
    if child_overlap >= child_words:
        child_overlap = child_words - 1

    batch_size = max(1, int(_coalesce(args.batch_size, defaults.batch_size)))

    manifest = ingest_pdfs_to_store(
        pdf_paths=resolved_inputs,
        store_dir=Path(args.store),
        embedding_model=embedding_model,
        reranker_model=str(reranker_model_value),
        parent_chunk_words=parent_words,
        parent_chunk_overlap_words=parent_overlap,
        child_chunk_words=child_words,
        child_chunk_overlap_words=child_overlap,
        batch_size=batch_size,
    )
    print("RAG ingest complete.")
    print(f"Store: {Path(args.store).resolve()}")
    print(f"Sources: {manifest.get('source_count', 1)} PDF(s)")
    print(
        "Parents: {parent_count}, Children: {child_count}, Sections: {section_count}".format(
            parent_count=manifest["parent_count"],
            child_count=manifest["child_count"],
            section_count=manifest["section_count"],
        )
    )
    print(f"Embedding model: {manifest['embedding_model']}")
    print(
        "Reranker model: {model}".format(
            model=str(manifest.get("reranker_model") or "off")
        )
    )
    return 0


def _run_query(args: argparse.Namespace, *, defaults: RagDefaultsConfig) -> int:
    reranker_model_value = _coalesce(args.reranker_model, defaults.reranker_model)
    if reranker_model_value is None:
        reranker_model_value = "off"

    store = HierarchicalRagStore.load(
        args.store,
        reranker_model_name=str(reranker_model_value),
    )
    result = store.retrieve(
        args.question,
        top_k=max(1, int(_coalesce(args.top_k, defaults.top_k))),
    )
    parent_context_chars = max(
        120,
        int(_coalesce(args.parent_context_chars, defaults.parent_context_chars)),
    )

    if args.json:
        payload = {
            "sources": [vars(item) for item in result.source_hits],
            "parents": [vars(item) for item in result.parent_hits],
            "context": store.format_context(
                result,
                parent_context_chars=parent_context_chars,
            ),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if not result.parent_hits:
        print("No relevant parent sections found.")
        return 0

    print(
        "Matched {source_count} source docs and {parent_count} parent sections.".format(
            source_count=len(result.source_hits),
            parent_count=len(result.parent_hits),
        )
    )
    print()
    print(
        store.format_context(
            result,
            parent_context_chars=parent_context_chars,
        )
    )
    return 0


def main() -> int:
    parser = _build_parser()
    _fail_on_legacy_cli_flags(parser)
    args = parser.parse_args()
    try:
        defaults = _runtime_rag_defaults(args.config)
        if args.command == "ingest":
            return _run_ingest(args, defaults=defaults)
        if args.command == "query":
            return _run_query(args, defaults=defaults)
        parser.error("Unknown command")
        return 2
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
