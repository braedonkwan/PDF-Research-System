from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL
from .persistent_storage import apply_persistent_cache_env_defaults

STORE_VERSION = 3
BM25_K1 = 1.5
BM25_B = 0.75
HYBRID_VECTOR_WEIGHT = 0.62
HYBRID_BM25_WEIGHT = 0.38
HYBRID_BOTH_RETRIEVERS_BONUS = 0.05
HYBRID_CANDIDATE_MULTIPLIER = 8
HYBRID_MIN_CANDIDATES = 24
HYBRID_RERANKER_WEIGHT = 0.50
HYBRID_PRE_RERANK_WEIGHT = 0.50
_BM25_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_MODEL_CACHE_LOCK = threading.RLock()
_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}
_RERANKER_MODEL_CACHE: dict[str, Any] = {}
_MODEL_INFERENCE_LOCKS: dict[int, threading.RLock] = {}


@dataclass(frozen=True)
class ParentChunk:
    id: str
    heading: str
    text: str
    page_start: int
    page_end: int
    word_count: int
    source_id: str = ""
    source_path: str = ""


@dataclass(frozen=True)
class ChildChunk:
    id: str
    parent_id: str
    text: str
    word_start: int
    word_end: int
    word_count: int


@dataclass(frozen=True)
class ParentHit:
    parent_id: str
    heading: str
    page_start: int
    page_end: int
    score: float
    source_id: str = ""
    source_path: str = ""
    text: str = ""


@dataclass(frozen=True)
class SourceHit:
    source_id: str
    source_path: str
    score: float
    parent_count: int


@dataclass(frozen=True)
class RetrievalResult:
    source_hits: list[SourceHit] = field(default_factory=list)
    parent_hits: list[ParentHit] = field(default_factory=list)


def _load_sentence_transformer(model_name: str) -> Any:
    apply_persistent_cache_env_defaults()
    cache_key = str(model_name).strip()
    if not cache_key:
        raise ValueError("Embedding model name cannot be empty.")
    with _MODEL_CACHE_LOCK:
        cached = _EMBEDDING_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            detail = str(exc).strip() or repr(exc)
            raise RuntimeError(
                "Missing dependency 'sentence-transformers'. Install with: "
                "python3 -m pip install -r requirements.txt. "
                "If you see torchvision/torch operator errors, install a matched "
                "torch/torchvision pair. "
                f"Import error detail: {detail}"
            ) from exc
        model = SentenceTransformer(cache_key)
        _EMBEDDING_MODEL_CACHE[cache_key] = model
        _MODEL_INFERENCE_LOCKS.setdefault(id(model), threading.RLock())
        return model


def _normalize_optional_model_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip()
    if not normalized:
        return None
    if normalized.lower() in {"off", "none", "null", "false", "disable", "disabled"}:
        return None
    return normalized


def _load_cross_encoder(model_name: str) -> Any:
    apply_persistent_cache_env_defaults()
    cache_key = str(model_name).strip()
    if not cache_key:
        raise ValueError("Reranker model name cannot be empty.")
    with _MODEL_CACHE_LOCK:
        cached = _RERANKER_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            detail = str(exc).strip() or repr(exc)
            raise RuntimeError(
                "Missing dependency 'sentence-transformers'. Install with: "
                "python3 -m pip install -r requirements.txt. "
                "If you see torchvision/torch operator errors, install a matched "
                "torch/torchvision pair. "
                f"Import error detail: {detail}"
            ) from exc
        model = CrossEncoder(cache_key)
        _RERANKER_MODEL_CACHE[cache_key] = model
        _MODEL_INFERENCE_LOCKS.setdefault(id(model), threading.RLock())
        return model


def _model_inference_lock(model: Any) -> threading.RLock:
    model_id = id(model)
    with _MODEL_CACHE_LOCK:
        lock = _MODEL_INFERENCE_LOCKS.get(model_id)
        if lock is None:
            lock = threading.RLock()
            _MODEL_INFERENCE_LOCKS[model_id] = lock
        return lock


def _score_reranker(
    model: Any,
    query: str,
    texts: list[str],
    *,
    batch_size: int = 16,
) -> np.ndarray:
    if not texts:
        return np.zeros((0,), dtype=np.float32)
    pairs = [[query, text] for text in texts]
    with _model_inference_lock(model):
        scores = model.predict(
            pairs,
            batch_size=max(1, int(batch_size)),
            show_progress_bar=False,
        )
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    return arr


def _encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    with _model_inference_lock(model):
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _clean_line(line: str) -> str:
    cleaned = line.replace("\u00a0", " ")
    cleaned = cleaned.replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join line-break hyphenation artifacts: "exam-\nple" -> "example"
    normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)
    lines = [_clean_line(line) for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line)


def _tokenize_for_bm25(text: str) -> list[str]:
    return _BM25_TOKEN_PATTERN.findall(text.lower())


def _build_bm25_index(
    texts: list[str],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> dict[str, Any]:
    doc_count = len(texts)
    doc_lengths: list[int] = []
    postings: dict[str, list[list[int]]] = {}

    for doc_idx, text in enumerate(texts):
        tokens = _tokenize_for_bm25(text)
        doc_lengths.append(len(tokens))
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            postings.setdefault(token, []).append([int(doc_idx), int(count)])

    avg_doc_len = (
        float(sum(doc_lengths) / doc_count)
        if doc_count > 0
        else 0.0
    )

    idf: dict[str, float] = {}
    for token, token_postings in postings.items():
        df = len(token_postings)
        # Smoothed BM25 IDF variant used in many production implementations.
        idf[token] = float(math.log(1.0 + ((doc_count - df + 0.5) / (df + 0.5))))

    return {
        "version": 1,
        "k1": float(k1),
        "b": float(b),
        "doc_count": doc_count,
        "avg_doc_len": avg_doc_len,
        "doc_lengths": doc_lengths,
        "idf": idf,
        "postings": postings,
    }


def _score_bm25(index: dict[str, Any], query_tokens: list[str]) -> np.ndarray:
    doc_count = int(index.get("doc_count", 0))
    if doc_count <= 0:
        return np.zeros((0,), dtype=np.float32)

    doc_lengths = np.asarray(index.get("doc_lengths", []), dtype=np.float32)
    if doc_lengths.size != doc_count:
        raise ValueError("BM25 index doc length count does not match doc_count.")
    avg_doc_len = float(index.get("avg_doc_len", 0.0))
    if avg_doc_len <= 0.0:
        avg_doc_len = 1.0

    idf = index.get("idf", {})
    postings = index.get("postings", {})
    if not isinstance(idf, dict) or not isinstance(postings, dict):
        raise ValueError("BM25 index is malformed.")

    k1 = float(index.get("k1", BM25_K1))
    b = float(index.get("b", BM25_B))
    scores = np.zeros((doc_count,), dtype=np.float32)
    query_counts = Counter(query_tokens)

    for token, qtf in query_counts.items():
        token_idf = float(idf.get(token, 0.0))
        if token_idf <= 0.0:
            continue
        token_postings = postings.get(token) or []
        query_weight = 1.0 + math.log1p(float(qtf))
        for item in token_postings:
            if not isinstance(item, list) or len(item) != 2:
                continue
            doc_idx = int(item[0])
            if doc_idx < 0 or doc_idx >= doc_count:
                continue
            tf = float(item[1])
            if tf <= 0.0:
                continue
            doc_norm = 1.0 - b + b * (float(doc_lengths[doc_idx]) / avg_doc_len)
            denom = tf + k1 * doc_norm
            if denom <= 0.0:
                continue
            bm25 = token_idf * ((tf * (k1 + 1.0)) / denom)
            scores[doc_idx] += float(bm25 * query_weight)

    return scores


def _top_k_indices(
    scores: np.ndarray,
    k: int,
    *,
    positive_only: bool = False,
) -> list[int]:
    if k <= 0 or scores.size == 0:
        return []

    max_k = min(k, int(scores.size))
    ranked = np.argsort(scores)[::-1][:max_k]
    output: list[int] = []
    for idx in ranked:
        value = float(scores[int(idx)])
        if positive_only and value <= 0.0:
            continue
        output.append(int(idx))
    return output


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    if score_max - score_min < 1e-9:
        if score_max <= 0.0:
            return np.zeros_like(scores, dtype=np.float32)
        return np.ones_like(scores, dtype=np.float32)
    normalized = (scores - score_min) / (score_max - score_min)
    return normalized.astype(np.float32)


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _select_unique_child_indices(
    ranked_child_indices: list[int],
    children: list[ChildChunk],
    k: int,
) -> list[int]:
    if k <= 0:
        return []
    selected: list[int] = []
    seen_text: set[str] = set()
    seen_pair: set[tuple[str, str]] = set()
    for child_idx in ranked_child_indices:
        if child_idx < 0 or child_idx >= len(children):
            continue
        child = children[child_idx]
        normalized_text = _normalize_for_dedupe(child.text)
        pair_key = (child.parent_id, normalized_text)
        if not normalized_text or normalized_text in seen_text or pair_key in seen_pair:
            continue
        seen_text.add(normalized_text)
        seen_pair.add(pair_key)
        selected.append(child_idx)
        if len(selected) >= k:
            break
    return selected


def _prepare_query_for_lookup(query: str) -> tuple[str, list[str]]:
    cleaned = _clean_line(_normalize_text(query).replace("\n", " "))
    if not cleaned:
        return "", []
    tokens = _tokenize_for_bm25(cleaned)
    if not tokens:
        return cleaned, []

    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        if len(token) <= 1:
            continue
        if token in _QUERY_STOPWORDS:
            continue
        keywords.append(token)

    bm25_tokens = keywords if keywords else tokens
    if keywords:
        keyword_line = " ".join(keywords)
        embedding_query = f"{cleaned}\n\nsearch keywords: {keyword_line}"
    else:
        embedding_query = cleaned

    return embedding_query, bm25_tokens


def _is_heading_candidate(line: str) -> bool:
    if len(line) < 4 or len(line) > 120:
        return False
    words = line.split()
    if len(words) > 14:
        return False
    if re.match(r"^\d+(\.\d+)*\s+\S+", line):
        return True
    if line.isupper() and any(ch.isalpha() for ch in line):
        return True
    if line.endswith((".", "?", "!", ";", ",")):
        return False
    title_word_count = sum(1 for word in words if word[:1].isupper())
    return title_word_count >= max(2, len(words) // 2)


def _extract_pdf_lines(pdf_path: Path) -> list[tuple[int, list[str]]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'pypdf'. Install with: py -3 -m pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, list[str]]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        normalized = _normalize_text(raw_text)
        lines = [line for line in normalized.split("\n") if line]
        if lines:
            pages.append((page_num, lines))
    return pages


def _build_sections(
    pages: list[tuple[int, list[str]]],
    *,
    source_id: str = "",
    source_path: str = "",
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_heading = "Document"
    current_lines: list[str] = []
    section_start_page = pages[0][0] if pages else 1
    section_end_page = section_start_page

    def flush_section() -> None:
        nonlocal current_lines, section_start_page, section_end_page
        text = " ".join(current_lines).strip()
        if text:
            sections.append(
                {
                    "heading": current_heading or f"Page {section_start_page}",
                    "text": text,
                    "page_start": section_start_page,
                    "page_end": section_end_page,
                    "source_id": source_id,
                    "source_path": source_path,
                }
            )
        current_lines = []

    for page_num, lines in pages:
        for line in lines:
            if _is_heading_candidate(line):
                if current_lines:
                    flush_section()
                current_heading = line
                section_start_page = page_num
                section_end_page = page_num
                continue
            current_lines.append(line)
            section_end_page = page_num

    if current_lines:
        flush_section()
    return sections


def _chunk_words(words: list[str], chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    spans: list[tuple[int, int]] = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        spans.append((start, end))
        if end >= len(words):
            break
        start += step
    return spans


def _build_parent_chunks(
    sections: list[dict[str, Any]],
    parent_chunk_words: int,
    parent_chunk_overlap_words: int,
) -> list[ParentChunk]:
    parents: list[ParentChunk] = []
    for section in sections:
        words = section["text"].split()
        if not words:
            continue
        spans = _chunk_words(words, parent_chunk_words, parent_chunk_overlap_words)
        for idx, (start, end) in enumerate(spans, start=1):
            chunk_words = words[start:end]
            heading = str(section["heading"])
            if len(spans) > 1:
                heading = f"{heading} (part {idx})"
            parent_id = f"p{len(parents) + 1:05d}"
            parents.append(
                ParentChunk(
                    id=parent_id,
                    heading=heading,
                    text=" ".join(chunk_words),
                    page_start=int(section["page_start"]),
                    page_end=int(section["page_end"]),
                    word_count=len(chunk_words),
                    source_id=str(section.get("source_id") or ""),
                    source_path=str(section.get("source_path") or ""),
                )
            )
    return parents


def _build_child_chunks(
    parents: list[ParentChunk],
    child_chunk_words: int,
    child_chunk_overlap_words: int,
) -> list[ChildChunk]:
    children: list[ChildChunk] = []
    for parent in parents:
        words = parent.text.split()
        if not words:
            continue
        spans = _chunk_words(words, child_chunk_words, child_chunk_overlap_words)
        for start, end in spans:
            child_id = f"c{len(children) + 1:06d}"
            chunk_words = words[start:end]
            children.append(
                ChildChunk(
                    id=child_id,
                    parent_id=parent.id,
                    text=" ".join(chunk_words),
                    word_start=start,
                    word_end=end,
                    word_count=len(chunk_words),
                )
            )
    return children


def ingest_pdfs_to_store(
    pdf_paths: list[str | Path],
    store_dir: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model: str | None = DEFAULT_RERANKER_MODEL,
    parent_chunk_words: int = 1200,
    parent_chunk_overlap_words: int = 120,
    child_chunk_words: int = 260,
    child_chunk_overlap_words: int = 40,
    batch_size: int = 32,
) -> dict[str, Any]:
    if not pdf_paths:
        raise ValueError("No PDF paths provided.")

    resolved_files: list[Path] = []
    seen: set[Path] = set()
    for item in pdf_paths:
        pdf_file = Path(item).resolve()
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_file}")
        if pdf_file in seen:
            continue
        seen.add(pdf_file)
        resolved_files.append(pdf_file)

    if not resolved_files:
        raise ValueError("No unique PDF files were resolved for ingestion.")

    sections: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    total_pages = 0
    for index, pdf_file in enumerate(resolved_files, start=1):
        pages = _extract_pdf_lines(pdf_file)
        if not pages:
            raise ValueError(
                f"No text extracted from PDF: {pdf_file}. The file may be image-only/scanned."
            )

        source_id = f"pdf_{index:03d}"
        source_sections = _build_sections(
            pages,
            source_id=source_id,
            source_path=str(pdf_file),
        )
        if not source_sections:
            raise ValueError(f"No sections were built from extracted PDF text: {pdf_file}")

        total_pages += len(pages)
        sections.extend(source_sections)
        source_entries.append(
            {
                "source_id": source_id,
                "source_pdf": str(pdf_file),
                "page_count": len(pages),
                "section_count": len(source_sections),
            }
        )

    if not sections:
        raise ValueError("No sections were built from extracted PDF text.")

    parents = _build_parent_chunks(
        sections,
        parent_chunk_words=parent_chunk_words,
        parent_chunk_overlap_words=parent_chunk_overlap_words,
    )
    children = _build_child_chunks(
        parents,
        child_chunk_words=child_chunk_words,
        child_chunk_overlap_words=child_chunk_overlap_words,
    )
    if not parents or not children:
        raise ValueError("Chunking produced no parent/child chunks. Adjust chunk sizing.")

    model = _load_sentence_transformer(embedding_model)
    parent_embeddings = _encode_texts(
        model, [chunk.text for chunk in parents], batch_size=batch_size
    )
    child_embeddings = _encode_texts(
        model, [chunk.text for chunk in children], batch_size=batch_size
    )
    bm25_index = _build_bm25_index([chunk.text for chunk in children])

    out_dir = Path(store_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": STORE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pdf": str(resolved_files[0]),
        "source_pdfs": [str(item) for item in resolved_files],
        "source_count": len(resolved_files),
        "sources": source_entries,
        "embedding_model": embedding_model,
        "reranker_model": _normalize_optional_model_name(reranker_model),
        "parent_chunk_words": parent_chunk_words,
        "parent_chunk_overlap_words": parent_chunk_overlap_words,
        "child_chunk_words": child_chunk_words,
        "child_chunk_overlap_words": child_chunk_overlap_words,
        "batch_size": batch_size,
        "page_count": total_pages,
        "section_count": len(sections),
        "parent_count": len(parents),
        "child_count": len(children),
        "stores": {
            "vector_db": {
                "parent_embeddings": "parent_embeddings.npy",
                "child_embeddings": "child_embeddings.npy",
                "metric": "cosine",
            },
            "document_store": {
                "parents": "parents.json",
                "children": "children.json",
                "manifest": "manifest.json",
            },
            "bm25_index": "bm25_index.json",
        },
        "bm25": {
            "k1": float(bm25_index.get("k1", BM25_K1)),
            "b": float(bm25_index.get("b", BM25_B)),
            "doc_count": int(bm25_index.get("doc_count", len(children))),
            "avg_doc_len": float(bm25_index.get("avg_doc_len", 0.0)),
            "term_count": len(bm25_index.get("idf", {})),
        },
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "parents.json").write_text(
        json.dumps([asdict(item) for item in parents], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "children.json").write_text(
        json.dumps([asdict(item) for item in children], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "bm25_index.json").write_text(
        json.dumps(bm25_index, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(out_dir / "parent_embeddings.npy", parent_embeddings)
    np.save(out_dir / "child_embeddings.npy", child_embeddings)

    return manifest


class HierarchicalRagStore:
    def __init__(
        self,
        *,
        store_dir: Path,
        manifest: dict[str, Any],
        parents: list[ParentChunk],
        children: list[ChildChunk],
        parent_embeddings: np.ndarray,
        child_embeddings: np.ndarray,
        bm25_index: dict[str, Any],
        embedding_model: Any,
        reranker_model_name: str | None = None,
        reranker_model: Any | None = None,
    ) -> None:
        self.store_dir = store_dir
        self.manifest = manifest
        self.parents = parents
        self.children = children
        self.parent_embeddings = parent_embeddings
        self.child_embeddings = child_embeddings
        self.bm25_index = bm25_index
        self.embedding_model = embedding_model
        self.reranker_model_name = _normalize_optional_model_name(reranker_model_name)
        self.reranker_model = reranker_model
        self.default_source_path = str(manifest.get("source_pdf") or "")
        self.parent_by_id = {item.id: item for item in parents}
        self.child_indices_by_parent: dict[str, list[int]] = {}
        self.parent_indices_by_source: dict[str, list[int]] = {}
        self.source_id_by_key: dict[str, str] = {}
        self.source_path_by_key: dict[str, str] = {}
        for parent_idx, parent in enumerate(parents):
            source_key = self._source_key_for_parent(parent)
            source_id, source_path = self._source_fields_for_parent(parent)
            self.parent_indices_by_source.setdefault(source_key, []).append(parent_idx)
            self.source_id_by_key.setdefault(source_key, source_id)
            self.source_path_by_key.setdefault(source_key, source_path)
        for idx, child in enumerate(children):
            self.child_indices_by_parent.setdefault(child.parent_id, []).append(idx)

    def _source_key_for_parent(self, parent: ParentChunk) -> str:
        if parent.source_id:
            return f"id:{parent.source_id}"
        if parent.source_path:
            return f"path:{parent.source_path}"
        if self.default_source_path:
            return f"path:{self.default_source_path}"
        return "source:unknown"

    def _source_fields_for_parent(self, parent: ParentChunk) -> tuple[str, str]:
        source_id = parent.source_id
        source_path = parent.source_path
        if not source_path and self.default_source_path:
            source_path = self.default_source_path
        if not source_id:
            if source_path:
                source_id = f"path:{source_path}"
            else:
                source_id = "unknown-source"
        return source_id, source_path

    def _get_reranker(self) -> Any | None:
        if not self.reranker_model_name:
            return None
        if self.reranker_model is not None:
            return self.reranker_model
        try:
            self.reranker_model = _load_cross_encoder(self.reranker_model_name)
        except Exception:
            # Fail open to keep retrieval available even if reranker download/load fails.
            self.reranker_model_name = None
            self.reranker_model = None
            return None
        return self.reranker_model

    @classmethod
    def load(
        cls,
        store_dir: str | Path,
        *,
        embedding_model_name: str | None = None,
        reranker_model_name: str | None = None,
    ) -> "HierarchicalRagStore":
        base = Path(store_dir).resolve()
        if not base.exists():
            raise FileNotFoundError(f"RAG store not found: {base}")

        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        model_name = embedding_model_name or str(manifest.get("embedding_model") or "")
        if not model_name:
            raise ValueError("RAG manifest is missing 'embedding_model'")

        parents_raw = json.loads((base / "parents.json").read_text(encoding="utf-8"))
        children_raw = json.loads((base / "children.json").read_text(encoding="utf-8"))
        parent_embeddings = np.load(base / "parent_embeddings.npy").astype(np.float32)
        child_embeddings = np.load(base / "child_embeddings.npy").astype(np.float32)
        bm25_index_path = base / "bm25_index.json"

        parents = [ParentChunk(**item) for item in parents_raw]
        children = [ChildChunk(**item) for item in children_raw]
        if len(parents) != len(parent_embeddings):
            raise ValueError("Parent metadata count does not match parent embedding count.")
        if len(children) != len(child_embeddings):
            raise ValueError("Child metadata count does not match child embedding count.")

        if bm25_index_path.exists():
            bm25_index = json.loads(bm25_index_path.read_text(encoding="utf-8"))
        else:
            # Backward compatibility for stores created before BM25 indexing.
            bm25_index = _build_bm25_index([child.text for child in children])
            bm25_index_path.write_text(
                json.dumps(bm25_index, ensure_ascii=False),
                encoding="utf-8",
            )

        bm25_doc_count = int(bm25_index.get("doc_count", 0))
        if bm25_doc_count != len(children):
            raise ValueError(
                "BM25 index document count does not match child chunk count."
            )

        model = _load_sentence_transformer(model_name)
        resolved_reranker_model_name = _normalize_optional_model_name(
            reranker_model_name
            or str(manifest.get("reranker_model") or "")
            or DEFAULT_RERANKER_MODEL
        )
        return cls(
            store_dir=base,
            manifest=manifest,
            parents=parents,
            children=children,
            parent_embeddings=parent_embeddings,
            child_embeddings=child_embeddings,
            bm25_index=bm25_index,
            embedding_model=model,
            reranker_model_name=resolved_reranker_model_name,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
    ) -> RetrievalResult:
        if not query.strip():
            return RetrievalResult()
        if top_k <= 0:
            return RetrievalResult()

        query_for_embedding, bm25_tokens = _prepare_query_for_lookup(query)
        if not query_for_embedding:
            return RetrievalResult()

        query_vec = _encode_texts(self.embedding_model, [query_for_embedding], batch_size=1)[0]
        vector_scores = self.child_embeddings @ query_vec
        bm25_scores = _score_bm25(self.bm25_index, bm25_tokens)
        if bm25_scores.size != vector_scores.size:
            raise ValueError("BM25 score size does not match child embedding count.")

        candidate_k = max(
            HYBRID_MIN_CANDIDATES,
            top_k * HYBRID_CANDIDATE_MULTIPLIER,
        )
        vector_top_indices = _top_k_indices(vector_scores, candidate_k, positive_only=False)
        bm25_top_indices = _top_k_indices(bm25_scores, candidate_k, positive_only=True)
        vector_top_set = set(vector_top_indices)
        bm25_top_set = set(bm25_top_indices)

        candidate_indices = sorted(vector_top_set | bm25_top_set)
        if not candidate_indices:
            candidate_indices = _top_k_indices(vector_scores, top_k, positive_only=False)
        if not candidate_indices:
            return RetrievalResult()

        candidate_vector_scores = np.asarray(
            [float(vector_scores[idx]) for idx in candidate_indices],
            dtype=np.float32,
        )
        candidate_bm25_scores = np.asarray(
            [float(bm25_scores[idx]) for idx in candidate_indices],
            dtype=np.float32,
        )
        vector_norm = _min_max_normalize(candidate_vector_scores)
        bm25_norm = _min_max_normalize(candidate_bm25_scores)

        pre_rerank_score_by_child_idx: dict[int, float] = {}
        for local_idx, child_idx in enumerate(candidate_indices):
            score = (
                HYBRID_VECTOR_WEIGHT * float(vector_norm[local_idx])
                + HYBRID_BM25_WEIGHT * float(bm25_norm[local_idx])
            )
            if child_idx in vector_top_set and child_idx in bm25_top_set:
                score += HYBRID_BOTH_RETRIEVERS_BONUS
            score = float(score)
            pre_rerank_score_by_child_idx[child_idx] = score

        combined_score_by_child_idx: dict[int, float] = dict(pre_rerank_score_by_child_idx)
        reranker = self._get_reranker()
        if reranker is not None:
            reranker_input_indices = list(candidate_indices)
            reranker_scores_raw = _score_reranker(
                reranker,
                query,
                [self.children[idx].text for idx in reranker_input_indices],
            )
            reranker_scores = _min_max_normalize(reranker_scores_raw)
            for local_idx, child_idx in enumerate(reranker_input_indices):
                pre_score = pre_rerank_score_by_child_idx.get(child_idx, 0.0)
                cross_score = float(reranker_scores[local_idx])
                combined_score_by_child_idx[child_idx] = float(
                    HYBRID_PRE_RERANK_WEIGHT * pre_score
                    + HYBRID_RERANKER_WEIGHT * cross_score
                )

        combined_scores = [combined_score_by_child_idx[idx] for idx in candidate_indices]

        ranked_child_pairs = sorted(
            zip(candidate_indices, combined_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked_child_indices = [item[0] for item in ranked_child_pairs]
        top_child_indices = _select_unique_child_indices(
            ranked_child_indices,
            self.children,
            max(top_k * 2, top_k),
        )

        parent_score_by_id: dict[str, float] = {}
        for child_idx, combined_score in ranked_child_pairs:
            parent_id = self.children[child_idx].parent_id
            previous = parent_score_by_id.get(parent_id)
            if previous is None or combined_score > previous:
                parent_score_by_id[parent_id] = combined_score

        selected_parent_ids: list[str] = []
        selected_parent_set: set[str] = set()
        for child_idx in top_child_indices:
            parent_id = self.children[child_idx].parent_id
            if parent_id in selected_parent_set:
                continue
            selected_parent_set.add(parent_id)
            selected_parent_ids.append(parent_id)
            if len(selected_parent_ids) >= top_k:
                break

        if len(selected_parent_ids) < top_k:
            for child_idx, _ in ranked_child_pairs:
                parent_id = self.children[child_idx].parent_id
                if parent_id in selected_parent_set:
                    continue
                selected_parent_set.add(parent_id)
                selected_parent_ids.append(parent_id)
                if len(selected_parent_ids) >= top_k:
                    break

        selected_parent_ids.sort(
            key=lambda parent_id: parent_score_by_id.get(parent_id, 0.0),
            reverse=True,
        )
        selected_parent_ids = selected_parent_ids[:top_k]

        parent_hits: list[ParentHit] = []
        source_score_by_key: dict[str, float] = {}
        for parent_id in selected_parent_ids:
            parent = self.parent_by_id[parent_id]
            parent_score = float(parent_score_by_id.get(parent_id, 0.0))
            source_key = self._source_key_for_parent(parent)
            source_id, source_path = self._source_fields_for_parent(parent)
            source_score_by_key[source_key] = max(
                source_score_by_key.get(source_key, 0.0),
                parent_score,
            )
            parent_hits.append(
                ParentHit(
                    parent_id=parent.id,
                    heading=parent.heading,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    score=parent_score,
                    source_id=source_id,
                    source_path=source_path,
                    text=parent.text,
                )
            )

        source_entries = sorted(
            source_score_by_key.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        source_hits: list[SourceHit] = []
        for source_key, source_score in source_entries:
            source_hits.append(
                SourceHit(
                    source_id=self.source_id_by_key.get(source_key, "unknown-source"),
                    source_path=self.source_path_by_key.get(source_key, ""),
                    score=float(source_score),
                    parent_count=len(self.parent_indices_by_source.get(source_key, [])),
                )
            )
            if len(source_hits) >= max(1, top_k):
                break

        return RetrievalResult(
            source_hits=source_hits,
            parent_hits=parent_hits,
        )

    def format_context(
        self,
        result: RetrievalResult,
        *,
        parent_context_chars: int = 500,
    ) -> str:
        if not result.parent_hits:
            return "No relevant document context was retrieved."

        lines: list[str] = []
        lines.append("Retrieved document context:")
        if result.source_hits:
            top_sources = ", ".join(
                Path(hit.source_path).name if hit.source_path else hit.source_id
                for hit in result.source_hits
            )
            lines.append(f"Top Sources: {top_sources}")
            lines.append("")
        for idx, hit in enumerate(result.parent_hits, start=1):
            parent_preview = hit.text[: max(120, int(parent_context_chars))].strip()
            source_id, source_path = hit.source_id, hit.source_path
            source_label = (
                Path(source_path).name
                if source_path
                else (source_id or "unknown-source")
            )
            lines.append(
                f"[Parent {idx}] Source: {source_label} | Parent: {hit.heading} "
                f"(pages {hit.page_start}-{hit.page_end})"
            )
            lines.append(f"Score: {hit.score:.3f}")
            lines.append(f"Parent Context: {parent_preview}")
            lines.append("")
        return "\n".join(lines).strip()
