from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .working_memory import WorkingMemoryStore


@dataclass(frozen=True)
class ContextRound:
    input_text: str
    output_text: str
    input_speaker: str | None = None
    output_speaker: str | None = None


@dataclass
class LastRoundsBuffer:
    memory_store: "WorkingMemoryStore | None" = None
    max_context_rounds: int = 1
    rounds: list[ContextRound] = field(default_factory=list)

    def reset(self) -> None:
        self.rounds = []

    def build_rounds(
        self,
        *,
        user_name: str,
        assistant_name: str,
    ) -> list[dict[str, object]]:
        if not self.rounds:
            return []
        context_limit = max(1, int(self.max_context_rounds))
        selected_rounds = self.rounds[-context_limit:]
        start_index = len(self.rounds) - len(selected_rounds) + 1
        payload: list[dict[str, object]] = []
        for index, item in enumerate(selected_rounds, start=start_index):
            input_body = item.input_text.strip() or "(empty)"
            output_body = item.output_text.strip() or "(empty)"
            payload.append(
                {
                    "round_index": index,
                    "user_query": {
                        "speaker": (item.input_speaker or user_name),
                        "text": input_body,
                    },
                    "response": {
                        "speaker": (item.output_speaker or assistant_name),
                        "text": output_body,
                    },
                }
            )
        return payload

    def append(
        self,
        input_text: str,
        output_text: str,
        *,
        input_speaker: str | None = None,
        output_speaker: str | None = None,
    ) -> None:
        if not output_text.strip():
            return

        resolved_input_speaker = (
            input_speaker.strip() if input_speaker is not None and input_speaker.strip() else None
        )
        resolved_output_speaker = (
            output_speaker.strip()
            if output_speaker is not None and output_speaker.strip()
            else None
        )
        self.rounds.append(
            ContextRound(
                input_text=input_text,
                output_text=output_text,
                input_speaker=resolved_input_speaker,
                output_speaker=resolved_output_speaker,
            )
        )
        context_limit = max(1, int(self.max_context_rounds))
        while len(self.rounds) > context_limit:
            oldest_round = self.rounds.pop(0)
            if self.memory_store is not None and oldest_round.output_text.strip():
                self.memory_store.append_turn(
                    oldest_round.input_text,
                    oldest_round.output_text,
                    user_speaker=(oldest_round.input_speaker or "User"),
                    assistant_speaker=(oldest_round.output_speaker or "Assistant"),
                )
