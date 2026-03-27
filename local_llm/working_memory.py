from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .defaults import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_WORKING_MEMORY_BATCH_SIZE,
    DEFAULT_WORKING_MEMORY_CHILD_CHUNK_OVERLAP_WORDS,
    DEFAULT_WORKING_MEMORY_CHILD_CHUNK_WORDS,
    DEFAULT_WORKING_MEMORY_PARENT_CHUNK_OVERLAP_WORDS,
    DEFAULT_WORKING_MEMORY_PARENT_CHUNK_WORDS,
    DEFAULT_WORKING_MEMORY_PARENT_CONTEXT_CHARS,
)
from .rag import (
    BM25_B,
    BM25_K1,
    HYBRID_BM25_WEIGHT,
    HYBRID_BOTH_RETRIEVERS_BONUS,
    HYBRID_CANDIDATE_MULTIPLIER,
    HYBRID_MIN_CANDIDATES,
    HYBRID_PRE_RERANK_WEIGHT,
    HYBRID_RERANKER_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    _build_bm25_index,
    _chunk_words,
    _encode_texts,
    _load_cross_encoder,
    _load_sentence_transformer,
    _min_max_normalize,
    _normalize_optional_model_name,
    _prepare_query_for_lookup,
    _score_reranker,
    _score_bm25,
    _select_unique_child_indices,
    _tokenize_for_bm25,
    _top_k_indices,
)

WORKING_MEMORY_VERSION = 1
DEFAULT_PARENT_CHUNK_WORDS = DEFAULT_WORKING_MEMORY_PARENT_CHUNK_WORDS
DEFAULT_PARENT_CHUNK_OVERLAP_WORDS = DEFAULT_WORKING_MEMORY_PARENT_CHUNK_OVERLAP_WORDS
DEFAULT_CHILD_CHUNK_WORDS = DEFAULT_WORKING_MEMORY_CHILD_CHUNK_WORDS
DEFAULT_CHILD_CHUNK_OVERLAP_WORDS = DEFAULT_WORKING_MEMORY_CHILD_CHUNK_OVERLAP_WORDS
DEFAULT_PARENT_CONTEXT_CHARS = DEFAULT_WORKING_MEMORY_PARENT_CONTEXT_CHARS

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class MemoryEvent:
    id: str
    session_id: str
    agent_id: str
    turn_index: int
    role: str
    text: str
    created_at_utc: str
    speaker: str | None = None


@dataclass(frozen=True)
class MemoryParentChunk:
    id: str
    text: str
    word_start: int
    word_end: int
    word_count: int
    event_start: int
    event_end: int
    turn_start: int
    turn_end: int


@dataclass(frozen=True)
class MemoryChildChunk:
    id: str
    parent_id: str
    text: str
    word_start: int
    word_end: int
    word_count: int


@dataclass(frozen=True)
class MemoryParentHit:
    parent_id: str
    score: float
    turn_start: int
    turn_end: int
    text: str


@dataclass(frozen=True)
class MemoryRetrievalResult:
    parent_hits: list[MemoryParentHit] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_agent_id(agent_id: str) -> str:
    stripped = agent_id.strip()
    if not stripped:
        return "default-agent"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", stripped)


def _normalize_event_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _THINK_BLOCK_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _normalize_speaker_label(speaker: str | None, *, fallback: str) -> str:
    if speaker is None:
        return fallback
    cleaned = _WHITESPACE_RE.sub(" ", str(speaker)).strip()
    return cleaned or fallback


def _event_speaker_label(event: MemoryEvent) -> str:
    fallback = "User" if event.role == "user" else "Assistant"
    return _normalize_speaker_label(event.speaker, fallback=fallback)


def _event_surface(event: MemoryEvent) -> str:
    return f"{_event_speaker_label(event)}: {event.text}".strip()


def _empty_bm25_index(
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> dict[str, Any]:
    return {
        "version": 1,
        "k1": float(k1),
        "b": float(b),
        "doc_count": 0,
        "avg_doc_len": 0.0,
        "doc_lengths": [],
        "idf": {},
        "postings": {},
    }


def _finalize_bm25_index(index: dict[str, Any]) -> dict[str, Any]:
    doc_lengths_raw = index.get("doc_lengths", [])
    postings_raw = index.get("postings", {})
    if not isinstance(doc_lengths_raw, list) or not isinstance(postings_raw, dict):
        raise ValueError("BM25 index is malformed.")

    doc_lengths = [max(0, int(item)) for item in doc_lengths_raw]
    doc_count = len(doc_lengths)
    cleaned_postings: dict[str, list[list[int]]] = {}
    idf: dict[str, float] = {}
    for token, token_postings in postings_raw.items():
        if not isinstance(token_postings, list):
            continue
        filtered: list[list[int]] = []
        for item in token_postings:
            if not isinstance(item, list) or len(item) != 2:
                continue
            doc_idx = int(item[0])
            tf = int(item[1])
            if doc_idx < 0 or doc_idx >= doc_count or tf <= 0:
                continue
            filtered.append([doc_idx, tf])
        if not filtered:
            continue
        token_key = str(token)
        cleaned_postings[token_key] = filtered
        df = len(filtered)
        idf[token_key] = float(math.log(1.0 + ((doc_count - df + 0.5) / (df + 0.5))))

    index["version"] = 1
    index["k1"] = float(index.get("k1", BM25_K1))
    index["b"] = float(index.get("b", BM25_B))
    index["doc_count"] = doc_count
    index["avg_doc_len"] = (
        float(sum(doc_lengths) / doc_count)
        if doc_count > 0
        else 0.0
    )
    index["doc_lengths"] = doc_lengths
    index["idf"] = idf
    index["postings"] = cleaned_postings
    return index


def _truncate_bm25_index(index: dict[str, Any], *, keep_docs: int) -> dict[str, Any]:
    resolved_keep_docs = max(0, int(keep_docs))
    truncated = _empty_bm25_index(
        k1=float(index.get("k1", BM25_K1)),
        b=float(index.get("b", BM25_B)),
    )
    if resolved_keep_docs <= 0:
        return truncated

    doc_lengths = index.get("doc_lengths", [])
    postings = index.get("postings", {})
    if not isinstance(doc_lengths, list) or not isinstance(postings, dict):
        raise ValueError("BM25 index is malformed.")
    if len(doc_lengths) < resolved_keep_docs:
        raise ValueError("BM25 index doc_lengths is shorter than requested keep_docs.")

    truncated["doc_lengths"] = [max(0, int(item)) for item in doc_lengths[:resolved_keep_docs]]
    truncated_postings: dict[str, list[list[int]]] = {}
    for token, token_postings in postings.items():
        if not isinstance(token_postings, list):
            continue
        filtered: list[list[int]] = []
        for item in token_postings:
            if not isinstance(item, list) or len(item) != 2:
                continue
            doc_idx = int(item[0])
            tf = int(item[1])
            if doc_idx < 0 or doc_idx >= resolved_keep_docs or tf <= 0:
                continue
            filtered.append([doc_idx, tf])
        if filtered:
            truncated_postings[str(token)] = filtered
    truncated["postings"] = truncated_postings
    return _finalize_bm25_index(truncated)


def _append_to_bm25_index(index: dict[str, Any], texts: list[str]) -> dict[str, Any]:
    if not texts:
        return _finalize_bm25_index(index)

    postings = index.setdefault("postings", {})
    doc_lengths = index.setdefault("doc_lengths", [])
    if not isinstance(postings, dict) or not isinstance(doc_lengths, list):
        raise ValueError("BM25 index is malformed.")

    doc_count = len(doc_lengths)
    for text in texts:
        tokens = _tokenize_for_bm25(text)
        doc_lengths.append(len(tokens))
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            postings.setdefault(token, []).append([doc_count, int(count)])
        doc_count += 1
    return _finalize_bm25_index(index)


class WorkingMemoryStore:
    def __init__(
        self,
        *,
        root_dir: Path,
        agent_id: str,
        session_id: str,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_name: str | None = DEFAULT_RERANKER_MODEL,
        parent_chunk_words: int = DEFAULT_PARENT_CHUNK_WORDS,
        parent_chunk_overlap_words: int = DEFAULT_PARENT_CHUNK_OVERLAP_WORDS,
        child_chunk_words: int = DEFAULT_CHILD_CHUNK_WORDS,
        child_chunk_overlap_words: int = DEFAULT_CHILD_CHUNK_OVERLAP_WORDS,
        batch_size: int = DEFAULT_WORKING_MEMORY_BATCH_SIZE,
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.agent_id = _safe_agent_id(agent_id)
        self.session_id = session_id.strip() or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = _normalize_optional_model_name(reranker_model_name)
        self.parent_chunk_words = max(120, int(parent_chunk_words))
        self.parent_chunk_overlap_words = min(
            max(0, int(parent_chunk_overlap_words)),
            self.parent_chunk_words - 1,
        )
        self.child_chunk_words = max(40, int(child_chunk_words))
        self.child_chunk_overlap_words = min(
            max(0, int(child_chunk_overlap_words)),
            self.child_chunk_words - 1,
        )
        self.batch_size = max(1, int(batch_size))

        self.agent_dir = (self.root_dir / self.agent_id).resolve()
        self.session_dir = (self.agent_dir / self.session_id).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.session_dir / "manifest.json"
        self.events_path = self.session_dir / "events.json"
        self.parents_path = self.session_dir / "parents.json"
        self.children_path = self.session_dir / "children.json"
        self.embeddings_path = self.session_dir / "child_embeddings.npy"
        self.bm25_path = self.session_dir / "bm25_index.json"

        self.embedding_model = _load_sentence_transformer(self.embedding_model_name)
        self.reranker_model: Any | None = None
        self.events: list[MemoryEvent] = []
        self.parents: list[MemoryParentChunk] = []
        self.children: list[MemoryChildChunk] = []
        self.parent_by_id: dict[str, MemoryParentChunk] = {}
        self.child_embeddings = np.zeros((0, 0), dtype=np.float32)
        self.bm25_index = _build_bm25_index([])
        self._flat_words: list[str] = []
        self._flat_event_indices: list[int] = []
        self._load_or_initialize()

    @classmethod
    def create(
        cls,
        root_dir: str | Path,
        *,
        agent_id: str,
        session_id: str | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_name: str | None = DEFAULT_RERANKER_MODEL,
        parent_chunk_words: int = DEFAULT_PARENT_CHUNK_WORDS,
        parent_chunk_overlap_words: int = DEFAULT_PARENT_CHUNK_OVERLAP_WORDS,
        child_chunk_words: int = DEFAULT_CHILD_CHUNK_WORDS,
        child_chunk_overlap_words: int = DEFAULT_CHILD_CHUNK_OVERLAP_WORDS,
        batch_size: int = DEFAULT_WORKING_MEMORY_BATCH_SIZE,
        clear_old_sessions_on_start: bool = False,
    ) -> "WorkingMemoryStore":
        resolved_root_dir = Path(root_dir).resolve()
        resolved_agent_id = _safe_agent_id(agent_id)
        resolved_session_id = session_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        if clear_old_sessions_on_start:
            cls._clear_agent_sessions(
                resolved_root_dir,
                resolved_agent_id,
                keep_session_id=resolved_session_id,
            )
        return cls(
            root_dir=resolved_root_dir,
            agent_id=resolved_agent_id,
            session_id=resolved_session_id,
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name,
            parent_chunk_words=parent_chunk_words,
            parent_chunk_overlap_words=parent_chunk_overlap_words,
            child_chunk_words=child_chunk_words,
            child_chunk_overlap_words=child_chunk_overlap_words,
            batch_size=batch_size,
        )

    @classmethod
    def _clear_agent_sessions(
        cls,
        root_dir: Path,
        agent_id: str,
        *,
        keep_session_id: str | None = None,
    ) -> None:
        agent_dir = (root_dir / _safe_agent_id(agent_id)).resolve()
        if not agent_dir.exists() or not agent_dir.is_dir():
            return
        for session_dir in agent_dir.iterdir():
            if not session_dir.is_dir():
                continue
            if keep_session_id is not None and session_dir.name == keep_session_id:
                continue
            shutil.rmtree(session_dir, ignore_errors=True)

    def drop_session(self) -> None:
        if self.session_dir.exists() and self.session_dir.is_dir():
            shutil.rmtree(self.session_dir, ignore_errors=True)
        if self.agent_dir.exists() and self.agent_dir.is_dir():
            try:
                if not any(self.agent_dir.iterdir()):
                    self.agent_dir.rmdir()
            except OSError:
                pass

    def fork_for_agent(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        clear_old_sessions_on_start: bool = False,
    ) -> "WorkingMemoryStore":
        return WorkingMemoryStore.create(
            root_dir=self.root_dir,
            agent_id=agent_id,
            session_id=session_id or self.session_id,
            embedding_model_name=self.embedding_model_name,
            reranker_model_name=self.reranker_model_name,
            parent_chunk_words=self.parent_chunk_words,
            parent_chunk_overlap_words=self.parent_chunk_overlap_words,
            child_chunk_words=self.child_chunk_words,
            child_chunk_overlap_words=self.child_chunk_overlap_words,
            batch_size=self.batch_size,
            clear_old_sessions_on_start=clear_old_sessions_on_start,
        )

    def _load_or_initialize(self) -> None:
        if not self.events_path.exists():
            self._persist()
            return

        if self.manifest_path.exists() and self.reranker_model_name is not None:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.reranker_model_name = _normalize_optional_model_name(
                self.reranker_model_name
                or str(manifest.get("reranker_model") or "")
                or DEFAULT_RERANKER_MODEL
            )

        raw_events = json.loads(self.events_path.read_text(encoding="utf-8"))
        self.events = [MemoryEvent(**item) for item in raw_events]
        if self.parents_path.exists():
            raw_parents = json.loads(self.parents_path.read_text(encoding="utf-8"))
            self.parents = [MemoryParentChunk(**item) for item in raw_parents]
        if self.children_path.exists():
            raw_children = json.loads(self.children_path.read_text(encoding="utf-8"))
            self.children = [MemoryChildChunk(**item) for item in raw_children]
        if self.embeddings_path.exists():
            loaded = np.load(self.embeddings_path).astype(np.float32)
            if loaded.ndim == 1:
                loaded = loaded.reshape(1, -1)
            self.child_embeddings = loaded
        if self.bm25_path.exists():
            self.bm25_index = json.loads(self.bm25_path.read_text(encoding="utf-8"))

        bm25_doc_count = (
            int(self.bm25_index.get("doc_count", -1))
            if isinstance(self.bm25_index, dict)
            else -1
        )
        if len(self.children) != len(self.child_embeddings) or bm25_doc_count != len(
            self.children
        ):
            # Rebuild if metadata/artifacts drifted out of sync.
            self._rebuild_chunks_and_indexes()
        else:
            self.parent_by_id = {item.id: item for item in self.parents}
            self._rebuild_flat_word_cache()

    def _get_reranker(self) -> Any | None:
        if not self.reranker_model_name:
            return None
        if self.reranker_model is not None:
            return self.reranker_model
        try:
            self.reranker_model = _load_cross_encoder(self.reranker_model_name)
        except Exception:
            self.reranker_model_name = None
            self.reranker_model = None
            return None
        return self.reranker_model

    @property
    def turn_count(self) -> int:
        if not self.events:
            return 0
        return max(event.turn_index for event in self.events)

    def clear(self) -> None:
        self.events = []
        self._flat_words = []
        self._flat_event_indices = []
        self._rebuild_chunks_and_indexes()

    def append_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        user_speaker: str = "User",
        assistant_speaker: str = "Assistant",
    ) -> None:
        normalized_user = _normalize_event_text(user_text)
        normalized_assistant = _normalize_event_text(assistant_text)
        if not normalized_user and not normalized_assistant:
            return

        normalized_user_speaker = _normalize_speaker_label(user_speaker, fallback="User")
        normalized_assistant_speaker = _normalize_speaker_label(
            assistant_speaker,
            fallback="Assistant",
        )
        previous_event_count = len(self.events)
        previous_word_count = len(self._flat_words)
        next_turn_index = self.turn_count + 1
        timestamp = _utc_now()
        event_base = len(self.events)
        if normalized_user:
            self.events.append(
                MemoryEvent(
                    id=f"e{event_base + 1:08d}",
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    turn_index=next_turn_index,
                    role="user",
                    text=normalized_user,
                    created_at_utc=timestamp,
                    speaker=normalized_user_speaker,
                )
            )
        if normalized_assistant:
            self.events.append(
                MemoryEvent(
                    id=f"e{len(self.events) + 1:08d}",
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    turn_index=next_turn_index,
                    role="assistant",
                    text=normalized_assistant,
                    created_at_utc=timestamp,
                    speaker=normalized_assistant_speaker,
                )
            )
        self._append_flat_word_cache(start_event_idx=previous_event_count)
        self._append_turn_incremental_indexes(previous_word_count=previous_word_count)

    def _rebuild_flat_word_cache(self) -> None:
        self._flat_words = []
        self._flat_event_indices = []
        for event_idx, event in enumerate(self.events):
            words = _event_surface(event).split()
            if not words:
                continue
            self._flat_words.extend(words)
            self._flat_event_indices.extend([event_idx] * len(words))

    def _append_flat_word_cache(self, *, start_event_idx: int) -> None:
        for event_idx in range(max(0, int(start_event_idx)), len(self.events)):
            words = _event_surface(self.events[event_idx]).split()
            if not words:
                continue
            self._flat_words.extend(words)
            self._flat_event_indices.extend([event_idx] * len(words))

    def _build_parent_chunks_from_flat_words(
        self,
        *,
        start_word: int,
        starting_parent_index: int,
    ) -> list[MemoryParentChunk]:
        if not self._flat_words:
            return []
        resolved_start_word = max(0, min(int(start_word), len(self._flat_words)))
        if resolved_start_word >= len(self._flat_words):
            return []

        spans = _chunk_words(
            self._flat_words[resolved_start_word:],
            self.parent_chunk_words,
            self.parent_chunk_overlap_words,
        )
        parents: list[MemoryParentChunk] = []
        next_parent_index = max(1, int(starting_parent_index))
        for start, end in spans:
            span_words = self._flat_words[resolved_start_word + start : resolved_start_word + end]
            if not span_words:
                continue
            span_events = self._flat_event_indices[
                resolved_start_word + start : resolved_start_word + end
            ]
            if not span_events:
                continue
            event_start = int(min(span_events))
            event_end = int(max(span_events))
            parents.append(
                MemoryParentChunk(
                    id=f"mp{next_parent_index:06d}",
                    text=" ".join(span_words),
                    word_start=resolved_start_word + start,
                    word_end=resolved_start_word + end,
                    word_count=len(span_words),
                    event_start=event_start,
                    event_end=event_end,
                    turn_start=self.events[event_start].turn_index,
                    turn_end=self.events[event_end].turn_index,
                )
            )
            next_parent_index += 1
        return parents

    def _build_child_chunks_for_parents(
        self,
        parents: list[MemoryParentChunk],
        *,
        starting_child_index: int,
    ) -> list[MemoryChildChunk]:
        children: list[MemoryChildChunk] = []
        next_child_index = max(1, int(starting_child_index))
        for parent in parents:
            words = parent.text.split()
            if not words:
                continue
            spans = _chunk_words(words, self.child_chunk_words, self.child_chunk_overlap_words)
            for start, end in spans:
                snippet_words = words[start:end]
                if not snippet_words:
                    continue
                children.append(
                    MemoryChildChunk(
                        id=f"mc{next_child_index:07d}",
                        parent_id=parent.id,
                        text=" ".join(snippet_words),
                        word_start=start,
                        word_end=end,
                        word_count=len(snippet_words),
                    )
                )
                next_child_index += 1
        return children

    def _append_turn_incremental_indexes(self, *, previous_word_count: int) -> None:
        current_word_count = len(self._flat_words)
        if current_word_count <= 0:
            self._rebuild_chunks_and_indexes()
            return
        if previous_word_count < 0 or previous_word_count > current_word_count:
            self._rebuild_chunks_and_indexes()
            return
        if previous_word_count == current_word_count:
            self._persist()
            return

        old_parents = self.parents
        old_children = self.children
        old_child_embeddings = self.child_embeddings

        if previous_word_count <= 0 or not old_parents:
            rebuild_start_word = 0
            parent_prefix_count = 0
        else:
            rebuild_starts = [
                parent.word_start
                for parent in old_parents
                if parent.word_end == previous_word_count
            ]
            if not rebuild_starts:
                self._rebuild_chunks_and_indexes()
                return
            rebuild_start_word = min(rebuild_starts)
            parent_prefix_count = len(old_parents)
            for idx, parent in enumerate(old_parents):
                if parent.word_start >= rebuild_start_word:
                    parent_prefix_count = idx
                    break

        retained_parents = old_parents[:parent_prefix_count]
        rebuilt_parent_tail = self._build_parent_chunks_from_flat_words(
            start_word=rebuild_start_word,
            starting_parent_index=parent_prefix_count + 1,
        )
        self.parents = retained_parents + rebuilt_parent_tail
        self.parent_by_id = {item.id: item for item in self.parents}

        replaced_parent_ids = {item.id for item in old_parents[parent_prefix_count:]}
        retained_child_count = len(old_children)
        for idx, child in enumerate(old_children):
            if child.parent_id in replaced_parent_ids:
                retained_child_count = idx
                break

        retained_children = old_children[:retained_child_count]
        rebuilt_child_tail = self._build_child_chunks_for_parents(
            rebuilt_parent_tail,
            starting_child_index=retained_child_count + 1,
        )
        self.children = retained_children + rebuilt_child_tail

        if old_child_embeddings.ndim != 2 or old_child_embeddings.shape[0] != len(old_children):
            self._rebuild_chunks_and_indexes()
            return
        retained_embeddings = old_child_embeddings[:retained_child_count]
        rebuilt_tail_embeddings = _encode_texts(
            self.embedding_model,
            [item.text for item in rebuilt_child_tail],
            batch_size=self.batch_size,
        )
        if rebuilt_tail_embeddings.size == 0:
            self.child_embeddings = retained_embeddings.astype(np.float32, copy=False)
        elif retained_embeddings.size == 0:
            self.child_embeddings = rebuilt_tail_embeddings.astype(np.float32, copy=False)
        else:
            if retained_embeddings.shape[1] != rebuilt_tail_embeddings.shape[1]:
                self._rebuild_chunks_and_indexes()
                return
            self.child_embeddings = np.vstack([retained_embeddings, rebuilt_tail_embeddings]).astype(
                np.float32,
                copy=False,
            )

        if len(self.children) != self.child_embeddings.shape[0]:
            self._rebuild_chunks_and_indexes()
            return

        try:
            bm25_prefix = _truncate_bm25_index(self.bm25_index, keep_docs=retained_child_count)
            self.bm25_index = _append_to_bm25_index(
                bm25_prefix,
                [item.text for item in rebuilt_child_tail],
            )
        except Exception:
            self.bm25_index = _build_bm25_index(
                [item.text for item in self.children],
                k1=BM25_K1,
                b=BM25_B,
            )

        bm25_doc_count = (
            int(self.bm25_index.get("doc_count", -1))
            if isinstance(self.bm25_index, dict)
            else -1
        )
        if bm25_doc_count != len(self.children):
            self.bm25_index = _build_bm25_index(
                [item.text for item in self.children],
                k1=BM25_K1,
                b=BM25_B,
            )
        self._persist()

    def _rebuild_chunks_and_indexes(self) -> None:
        self._rebuild_flat_word_cache()
        self.parents = self._build_parent_chunks_from_flat_words(
            start_word=0,
            starting_parent_index=1,
        )
        self.children = self._build_child_chunks_for_parents(
            self.parents,
            starting_child_index=1,
        )
        self.parent_by_id = {item.id: item for item in self.parents}
        child_texts = [item.text for item in self.children]
        self.child_embeddings = _encode_texts(
            self.embedding_model,
            child_texts,
            batch_size=self.batch_size,
        )
        self.bm25_index = _build_bm25_index(
            child_texts,
            k1=BM25_K1,
            b=BM25_B,
        )
        self._persist()

    def _persist(self) -> None:
        manifest = {
            "version": WORKING_MEMORY_VERSION,
            "updated_at_utc": _utc_now(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "embedding_model": self.embedding_model_name,
            "reranker_model": self.reranker_model_name,
            "event_count": len(self.events),
            "turn_count": self.turn_count,
            "parent_count": len(self.parents),
            "child_count": len(self.children),
            "parent_chunk_words": self.parent_chunk_words,
            "parent_chunk_overlap_words": self.parent_chunk_overlap_words,
            "child_chunk_words": self.child_chunk_words,
            "child_chunk_overlap_words": self.child_chunk_overlap_words,
            "stores": {
                "document_store": {
                    "events": self.events_path.name,
                    "parents": self.parents_path.name,
                    "children": self.children_path.name,
                },
                "vector_db": {
                    "child_embeddings": self.embeddings_path.name,
                    "metric": "cosine",
                },
                "bm25_index": self.bm25_path.name,
            },
        }

        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.events_path.write_text(
            json.dumps([asdict(item) for item in self.events], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.parents_path.write_text(
            json.dumps([asdict(item) for item in self.parents], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.children_path.write_text(
            json.dumps([asdict(item) for item in self.children], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.bm25_path.write_text(
            json.dumps(self.bm25_index, ensure_ascii=False),
            encoding="utf-8",
        )

        if self.child_embeddings.size == 0:
            np.save(self.embeddings_path, np.zeros((0, 0), dtype=np.float32))
        else:
            np.save(self.embeddings_path, self.child_embeddings.astype(np.float32))

    def retrieve(
        self,
        query: str,
        *,
        top_k_rounds: int = 5,
        turn_floor: int = 1,
        turn_ceiling: int | None = None,
    ) -> MemoryRetrievalResult:
        if not query.strip():
            return MemoryRetrievalResult()
        if top_k_rounds <= 0:
            return MemoryRetrievalResult()
        if not self.children or self.child_embeddings.size == 0:
            return MemoryRetrievalResult()

        resolved_turn_floor = max(1, int(turn_floor))
        resolved_turn_ceiling = (
            int(turn_ceiling)
            if turn_ceiling is not None
            else int(self.turn_count)
        )
        if resolved_turn_ceiling <= 0 or resolved_turn_floor > resolved_turn_ceiling:
            return MemoryRetrievalResult()

        eligible_indices: list[int] = []
        for idx, child in enumerate(self.children):
            parent = self.parent_by_id.get(child.parent_id)
            if parent is None:
                continue
            if parent.turn_end < resolved_turn_floor:
                continue
            if parent.turn_start > resolved_turn_ceiling:
                continue
            eligible_indices.append(idx)
        if not eligible_indices:
            return MemoryRetrievalResult()

        query_for_embedding, bm25_tokens = _prepare_query_for_lookup(query)
        if not query_for_embedding:
            return MemoryRetrievalResult()

        query_vec = _encode_texts(self.embedding_model, [query_for_embedding], batch_size=1)[0]
        vector_scores = self.child_embeddings @ query_vec
        bm25_scores = _score_bm25(self.bm25_index, bm25_tokens)
        if bm25_scores.size != vector_scores.size:
            return MemoryRetrievalResult()

        eligible_vector_scores = np.asarray(
            [float(vector_scores[idx]) for idx in eligible_indices],
            dtype=np.float32,
        )
        eligible_bm25_scores = np.asarray(
            [float(bm25_scores[idx]) for idx in eligible_indices],
            dtype=np.float32,
        )

        candidate_k = max(
            HYBRID_MIN_CANDIDATES,
            top_k_rounds * HYBRID_CANDIDATE_MULTIPLIER,
        )
        vector_top_local = _top_k_indices(
            eligible_vector_scores,
            candidate_k,
            positive_only=False,
        )
        bm25_top_local = _top_k_indices(
            eligible_bm25_scores,
            candidate_k,
            positive_only=True,
        )
        vector_top_indices = [eligible_indices[idx] for idx in vector_top_local]
        bm25_top_indices = [eligible_indices[idx] for idx in bm25_top_local]
        vector_top_set = set(vector_top_indices)
        bm25_top_set = set(bm25_top_indices)

        candidate_indices = sorted(vector_top_set | bm25_top_set)
        if not candidate_indices:
            fallback_local = _top_k_indices(
                eligible_vector_scores,
                top_k_rounds,
                positive_only=False,
            )
            candidate_indices = [eligible_indices[idx] for idx in fallback_local]
        if not candidate_indices:
            return MemoryRetrievalResult()

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
            pre_rerank_score_by_child_idx[child_idx] = float(score)

        rerank_score_by_child_idx: dict[int, float] = dict(pre_rerank_score_by_child_idx)
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
                rerank_score_by_child_idx[child_idx] = float(
                    HYBRID_PRE_RERANK_WEIGHT * pre_score
                    + HYBRID_RERANKER_WEIGHT * cross_score
                )

        ranked_child_indices: list[int] = sorted(
            candidate_indices,
            key=lambda idx: rerank_score_by_child_idx[idx],
            reverse=True,
        )
        turn_score_by_index: dict[int, float] = {}
        for child_idx in ranked_child_indices:
            child = self.children[child_idx]
            parent = self.parent_by_id.get(child.parent_id)
            if parent is None:
                continue
            score = float(rerank_score_by_child_idx[child_idx])
            floor = max(parent.turn_start, resolved_turn_floor)
            ceiling = min(parent.turn_end, resolved_turn_ceiling)
            if floor > ceiling:
                continue
            for turn_index in range(floor, ceiling + 1):
                previous = turn_score_by_index.get(turn_index)
                if previous is None or score > previous:
                    turn_score_by_index[turn_index] = score
        if not turn_score_by_index:
            return MemoryRetrievalResult()

        turn_lines: dict[int, list[str]] = {}
        for event in self.events:
            turn_index = int(event.turn_index)
            if turn_index < resolved_turn_floor or turn_index > resolved_turn_ceiling:
                continue
            speaker = _event_speaker_label(event)
            turn_lines.setdefault(turn_index, []).append(f"{speaker}: {event.text}")

        parent_hits: list[MemoryParentHit] = []
        ranked_turn_indices = sorted(
            turn_score_by_index.keys(),
            key=lambda idx: turn_score_by_index[idx],
            reverse=True,
        )[:top_k_rounds]
        for turn_index in ranked_turn_indices:
            text_lines = turn_lines.get(turn_index, [])
            round_text = "\n".join(text_lines).strip() or "(empty round)"
            parent_hits.append(
                MemoryParentHit(
                    parent_id=f"turn_{turn_index:06d}",
                    score=float(turn_score_by_index[turn_index]),
                    turn_start=turn_index,
                    turn_end=turn_index,
                    text=round_text,
                )
            )

        return MemoryRetrievalResult(parent_hits=parent_hits)

    def format_context(
        self,
        result: MemoryRetrievalResult,
        *,
        parent_context_chars: int = DEFAULT_PARENT_CONTEXT_CHARS,
    ) -> str:
        if not result.parent_hits:
            return "No relevant working-memory context was retrieved."

        lines: list[str] = []
        lines.append("Retrieved working-memory context:")
        for idx, hit in enumerate(result.parent_hits, start=1):
            parent_preview = hit.text[: max(120, int(parent_context_chars))].strip()
            lines.append(
                f"[Memory {idx}] turns {hit.turn_start}-{hit.turn_end} "
                f"(score={hit.score:.3f})"
            )
            lines.append(f"Parent Context: {parent_preview}")
            lines.append("")
        return "\n".join(lines).strip()
