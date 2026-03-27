from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .working_memory import WorkingMemoryStore


@dataclass
class LastRoundsBuffer:
    memory_store: "WorkingMemoryStore | None" = None
    max_context_rounds: int = 1
    rounds: list[tuple[str, str]] = field(default_factory=list)

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
        for index, (input_text, output_text) in enumerate(selected_rounds, start=start_index):
            input_body = input_text.strip() or "(empty)"
            output_body = output_text.strip() or "(empty)"
            payload.append(
                {
                    "round_index": index,
                    "user_query": {
                        "speaker": user_name,
                        "text": input_body,
                    },
                    "response": {
                        "speaker": assistant_name,
                        "text": output_body,
                    },
                }
            )
        return payload

    def append(self, input_text: str, output_text: str) -> None:
        if not output_text.strip():
            return

        self.rounds.append((input_text, output_text))
        context_limit = max(1, int(self.max_context_rounds))
        while len(self.rounds) > context_limit:
            oldest_input, oldest_output = self.rounds.pop(0)
            if self.memory_store is not None and oldest_output.strip():
                self.memory_store.append_turn(oldest_input, oldest_output)
