from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .settings import ModelConfig


def _markdown_fence(text: str, language: str = "") -> str:
    body = text.replace("\r\n", "\n").strip()
    if not body:
        body = "(empty)"
    fence = "```"
    if "```" in body:
        fence = "````"
    lang = language.strip()
    if lang:
        return f"{fence}{lang}\n{body}\n{fence}"
    return f"{fence}\n{body}\n{fence}"


def append_markdown_session_header(
    log_path: Path,
    *,
    model: ModelConfig,
    rag_enabled: bool,
    working_memory_enabled: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# Chat Transcript\n\n", encoding="utf-8")
    started = datetime.now(timezone.utc).isoformat()
    lines = [
        f"## Session {started}",
        "",
        f"- Model alias: `{model.alias}`",
        f"- API model: `{model.api_name}`",
        f"- RAG enabled: `{'yes' if rag_enabled else 'no'}`",
        f"- Working memory enabled: `{'yes' if working_memory_enabled else 'no'}`",
        "",
        "---",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def append_markdown_turn(
    log_path: Path,
    *,
    turn_index: int,
    user_text: str,
    assistant_text: str,
    rag_summary: str | None = None,
    memory_summary: str | None = None,
) -> None:
    assistant_body = assistant_text.replace("\r\n", "\n").strip() or "_(empty)_"
    lines = [
        f"### Turn {turn_index}",
        "",
        "#### User",
        "",
        _markdown_fence(user_text, "text"),
        "",
    ]
    if rag_summary:
        lines.extend(
            [
                "#### RAG",
                "",
                rag_summary,
                "",
            ]
        )
    if memory_summary:
        lines.extend(
            [
                "#### Working Memory",
                "",
                memory_summary,
                "",
            ]
        )
    lines.extend(
        [
            "#### Assistant",
            "",
            assistant_body,
            "",
            "---",
            "",
        ]
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


@dataclass
class MarkdownTranscriptLogger:
    path: Path
    turn_index: int = 0

    @classmethod
    def start(
        cls,
        path: Path,
        *,
        model: ModelConfig,
        rag_enabled: bool,
        working_memory_enabled: bool,
    ) -> "MarkdownTranscriptLogger":
        append_markdown_session_header(
            path,
            model=model,
            rag_enabled=rag_enabled,
            working_memory_enabled=working_memory_enabled,
        )
        return cls(path=path)

    def append_turn(
        self,
        *,
        user_text: str,
        assistant_text: str,
        rag_summary: str | None = None,
        memory_summary: str | None = None,
    ) -> None:
        self.turn_index += 1
        append_markdown_turn(
            self.path,
            turn_index=self.turn_index,
            user_text=user_text,
            assistant_text=assistant_text,
            rag_summary=rag_summary,
            memory_summary=memory_summary,
        )
