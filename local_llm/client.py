from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, RequestException, Timeout
from urllib3.util.retry import Retry

from .chat_runtime import ChatRuntimeOptions
from .chat_context_service import RetrievedContext, collect_context_with_last_rounds
from .chat_markdown_log import MarkdownTranscriptLogger
from .last_rounds_buffer import LastRoundsBuffer
from .model_overrides import apply_model_chat_overrides
from .settings import AppConfig, ChatConfig, ModelConfig

if TYPE_CHECKING:
    from .rag import HierarchicalRagStore
    from .working_memory import WorkingMemoryStore

RETRIEVAL_HELPER_SYSTEM_PROMPT = (
    "Identity: You are the Retrieval Query Agent. "
    "Objective: Produce one concise retrieval query that best represents the current user/agent intent for memory and PDF-RAG lookup. "
    "System awareness: You only receive structured JSON input with keys latest_input, current_agent_system_prompt, and last_n_rounds, plus this system prompt. "
    "No additional hidden context is provided. "
    "Context usage rules: Treat latest_input as primary intent. Use current_agent_system_prompt and last_n_rounds (oldest-to-newest rounds with user_query/response fields) only for recall/disambiguation of entities, scope, and references. "
    "Do not add unrelated terms or speculative expansions. "
    "Conflict handling: If latest_input conflicts with prior rounds, prioritize latest_input. If intent is ambiguous, choose a safe broad query anchored to latest_input wording. "
    "Output style constraints: Return strict JSON only with exactly one key: "
    "{\"retrieval_query\":\"...\"}. No markdown, no code fences, no extra keys, no commentary."
)


def _create_session(chat: ChatConfig) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=chat.retry_total,
        backoff_factor=chat.retry_backoff_factor,
        status_forcelist=list(chat.retry_status_forcelist),
        allowed_methods=frozenset(["POST"]),
    )
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _add_optional_payload_fields(payload: dict[str, object], chat: ChatConfig) -> None:
    if chat.top_k is not None:
        payload["top_k"] = chat.top_k
    if chat.min_p is not None:
        payload["min_p"] = chat.min_p
    if chat.typical_p is not None:
        payload["typical_p"] = chat.typical_p
    if chat.repeat_penalty is not None:
        payload["repeat_penalty"] = chat.repeat_penalty
    if chat.presence_penalty is not None:
        payload["presence_penalty"] = chat.presence_penalty
    if chat.frequency_penalty is not None:
        payload["frequency_penalty"] = chat.frequency_penalty
    if chat.seed is not None:
        payload["seed"] = chat.seed
    if chat.stop:
        payload["stop"] = list(chat.stop)


def _coerce_context_payload(context_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(context_text)
    except json.JSONDecodeError:
        return {
            "working_memory": {
                "last_n_rounds": [],
                "long_term_memory": None,
            },
            "knowledge": {"raw_context": context_text},
        }

    if isinstance(parsed, dict):
        working_memory = parsed.get("working_memory")
        knowledge = parsed.get("knowledge")
        if not isinstance(working_memory, dict):
            working_memory = {
                "last_n_rounds": [],
                "long_term_memory": None,
            }
        else:
            working_memory = {
                "last_n_rounds": working_memory.get("last_n_rounds") or [],
                "long_term_memory": working_memory.get("long_term_memory"),
            }
        return {
            "working_memory": working_memory,
            "knowledge": knowledge,
        }

    return {
        "working_memory": {
            "last_n_rounds": [],
            "long_term_memory": None,
        },
        "knowledge": {"raw_context": str(parsed)},
    }


def _build_contextual_user_input(user_text: str, context_text: str) -> str:
    context_payload = _coerce_context_payload(context_text)
    payload = {
        "working_memory": context_payload.get("working_memory"),
        "knowledge": context_payload.get("knowledge"),
        "user_query": user_text,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass
class ChatClient:
    config: AppConfig
    model: ModelConfig
    system_prompt_override: str | None = None

    def __post_init__(self) -> None:
        self.chat = apply_model_chat_overrides(self.config.chat, self.model)
        self.session = _create_session(self.chat)
        disable_thinking = _should_disable_thinking(self.chat, self.model)
        base_system_prompt = self.system_prompt_override or self.model.system_prompt or self.chat.system_prompt
        system_prompt = _with_no_think_if_enabled(base_system_prompt, disable_thinking)
        self.system_message: dict[str, str] = {"role": "system", "content": system_prompt}

    def _build_payload(
        self,
        *,
        user_text: str,
    ) -> dict[str, object]:
        chat = self.chat
        disable_thinking = _should_disable_thinking(chat, self.model)
        payload_messages = [
            self.system_message,
            {
                "role": "user",
                "content": _with_no_think_if_enabled(user_text, disable_thinking),
            },
        ]
        payload: dict[str, object] = {
            "model": self.model.api_name,
            "messages": payload_messages,
            "max_tokens": chat.max_tokens,
            "temperature": chat.temperature,
            "top_p": chat.top_p,
            "stream": chat.stream,
        }
        _add_optional_payload_fields(payload, chat)
        chat_template_kwargs = dict(chat.chat_template_kwargs)
        chat_template_kwargs.update(self.model.chat_template_kwargs)
        if disable_thinking:
            chat_template_kwargs["enable_thinking"] = False
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        payload.update(chat.request_overrides)
        payload.update(self.model.request_overrides)
        return payload

    def stream_reply(self, user_text: str, context_text: str | None = None) -> Iterable[str]:
        chat = self.chat
        request_user_text = user_text
        if context_text:
            request_user_text = _build_contextual_user_input(user_text, context_text)

        payload = self._build_payload(user_text=request_user_text)
        full_text = ""
        try:
            with self.session.post(
                chat.url,
                json=payload,
                timeout=(chat.connect_timeout_sec, chat.read_timeout_sec),
                stream=chat.stream,
            ) as response:
                response.raise_for_status()
                if chat.stream:
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/event-stream" in content_type:
                        for line in response.iter_lines(decode_unicode=True):
                            piece = _parse_stream_line(line, include_reasoning=chat.show_reasoning)
                            if piece is None:
                                continue
                            full_text += piece
                            yield piece
                    else:
                        # Some llama-server builds ignore stream=True and return JSON.
                        piece = _parse_non_stream_response(response.json(), include_reasoning=chat.show_reasoning)
                        if piece:
                            full_text += piece
                            yield piece
                else:
                    piece = _parse_non_stream_response(response.json(), include_reasoning=chat.show_reasoning)
                    if piece:
                        full_text += piece
                        yield piece

            if chat.stream and not full_text:
                # Last-resort compatibility path for servers with partial/odd streaming behavior.
                fallback_payload = dict(payload)
                fallback_payload["stream"] = False
                with self.session.post(
                    chat.url,
                    json=fallback_payload,
                    timeout=(chat.connect_timeout_sec, chat.read_timeout_sec),
                    stream=False,
                ) as response:
                    response.raise_for_status()
                    piece = _parse_non_stream_response(response.json(), include_reasoning=chat.show_reasoning)
                    if piece:
                        full_text += piece
                        yield piece

            full_text = full_text.strip()
            if not full_text:
                raise RuntimeError("Empty response received.")
        except (ConnectionError, Timeout, RequestException):
            raise
        except Exception:
            raise


def _parse_retrieval_query_reply(reply_text: str, *, fallback_query: str) -> str:
    cleaned = reply_text.strip()
    if not cleaned:
        return fallback_query
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        first_line = cleaned.splitlines()[0].strip()
        return first_line or fallback_query
    if isinstance(payload, dict):
        value = payload.get("retrieval_query")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback_query


def _build_retrieval_query(
    helper_client: ChatClient,
    *,
    latest_input: str,
    current_agent_system_prompt: str,
    last_n_rounds: list[dict[str, object]] | None,
) -> str:
    fallback_query = latest_input.strip()
    if not fallback_query:
        return latest_input
    helper_input = {
        "latest_input": fallback_query,
        "current_agent_system_prompt": current_agent_system_prompt.strip(),
        "last_n_rounds": last_n_rounds or [],
    }
    helper_prompt = json.dumps(helper_input, ensure_ascii=False, separators=(",", ":"))
    try:
        response_text = ""
        for piece in helper_client.stream_reply(helper_prompt, context_text=None):
            response_text += piece
        return _parse_retrieval_query_reply(response_text, fallback_query=fallback_query)
    except Exception:
        return fallback_query


def _with_no_think_if_enabled(text: str, disable_thinking: bool) -> str:
    if not disable_thinking:
        return text
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            return text
        except json.JSONDecodeError:
            pass
    if "/no_think" in text:
        return text
    return f"{text.rstrip()}\n/no_think"


def _should_disable_thinking(chat: ChatConfig, model: ModelConfig) -> bool:
    return chat.disable_thinking and model.supports_thinking_toggle


def _parse_stream_line(line: str | bytes | None, include_reasoning: bool = False) -> str | None:
    if not line:
        return None
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="ignore")
    if not line.startswith("data: "):
        return None

    data = line[6:].strip()
    if data == "[DONE]":
        return None

    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return None
    return _extract_piece_from_payload(chunk, include_reasoning=include_reasoning)


def _extract_piece_from_payload(payload: dict[str, Any], include_reasoning: bool = False) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    return _extract_piece_from_choice(first, include_reasoning=include_reasoning)


def _extract_piece_from_choice(choice: dict[str, Any], include_reasoning: bool = False) -> str | None:
    # Streaming format
    delta = choice.get("delta")
    if isinstance(delta, dict):
        for key in ("content", "text", "response"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return value
        if include_reasoning:
            value = delta.get("reasoning_content")
            if isinstance(value, str) and value:
                return value

    # Non-stream or mixed format
    message = choice.get("message")
    if isinstance(message, dict):
        for key in ("content", "text"):
            value = message.get(key)
            if isinstance(value, str) and value:
                return value
        if include_reasoning:
            value = message.get("reasoning_content")
            if isinstance(value, str) and value:
                return value

    for key in ("text", "content", "response"):
        value = choice.get(key)
        if isinstance(value, str) and value:
            return value
    if include_reasoning:
        value = choice.get("reasoning_content")
        if isinstance(value, str) and value:
            return value
    return None


def _parse_non_stream_response(payload: dict[str, object], include_reasoning: bool = False) -> str | None:
    return _extract_piece_from_payload(payload, include_reasoning=include_reasoning)


def _stream_labeled_reply(
    client: ChatClient,
    *,
    label: str,
    prompt: str,
    context_text: str | None = None,
) -> tuple[bool, str]:
    print(f"\n{label}: ", end="", flush=True)
    try:
        output = ""
        for piece in client.stream_reply(prompt, context_text=context_text):
            print(piece, end="", flush=True)
            output += piece
        print("\n")
        return True, output
    except ConnectionError:
        print("\nCannot connect to server. Check host/port and start the server.\n")
    except Timeout:
        print("\nRequest timed out. Try lowering max_tokens or context size.\n")
    except RequestException as exc:
        print(f"\nHTTP error: {exc}\n")
    except Exception as exc:
        print(f"\nError: {exc}\n")
    return False, ""


def _is_exit_command(value: str) -> bool:
    return value in {"exit", "quit"}


def _is_clear_command(value: str) -> bool:
    return value in {"/clear", "/reset"}


def _reset_last_round_buffers(*buffers: LastRoundsBuffer, message: str) -> None:
    for buffer in buffers:
        buffer.reset()
    print(message)


def _clear_memory_stores(
    *stores: "WorkingMemoryStore | None",
    success_message: str,
) -> None:
    resolved = [store for store in stores if store is not None]
    if not resolved:
        print("[Memory] Working memory is not enabled.\n")
        return
    for store in resolved:
        store.clear()
    print(success_message)


def _print_context_status_lines(context_result: RetrievedContext) -> None:
    for line in context_result.status_lines:
        print(line)


def _collect_turn_context(
    *,
    query_prompt: str,
    retrieval_input: str,
    current_agent_system_prompt: str,
    retrieval_helper: ChatClient,
    options: ChatRuntimeOptions,
    rag_store: "HierarchicalRagStore | None",
    working_memory_store: "WorkingMemoryStore | None",
    last_rounds_buffer: LastRoundsBuffer | None,
    context_user_name: str,
    context_assistant_name: str,
    recent_memory_exclude_last_turns: int = 0,
) -> RetrievedContext:
    last_n_rounds: list[dict[str, object]] | None = None
    if last_rounds_buffer is not None:
        last_n_rounds = last_rounds_buffer.build_rounds(
            user_name=context_user_name,
            assistant_name=context_assistant_name,
        )
    retrieval_query = _build_retrieval_query(
        retrieval_helper,
        latest_input=retrieval_input,
        current_agent_system_prompt=current_agent_system_prompt,
        last_n_rounds=last_n_rounds,
    )
    return collect_context_with_last_rounds(
        query_prompt,
        options=options,
        rag_store=rag_store,
        working_memory_store=working_memory_store,
        last_rounds_buffer=last_rounds_buffer,
        user_name=context_user_name,
        assistant_name=context_assistant_name,
        last_n_rounds_override=last_n_rounds,
        retrieval_query_text=retrieval_query,
        recent_memory_exclude_last_turns=recent_memory_exclude_last_turns,
    )


def _run_contextual_turn(
    *,
    client: ChatClient,
    label: str,
    prompt: str,
    retrieval_input: str,
    retrieval_helper: ChatClient,
    options: ChatRuntimeOptions,
    rag_store: "HierarchicalRagStore | None",
    working_memory_store: "WorkingMemoryStore | None",
    last_rounds_buffer: LastRoundsBuffer | None,
    context_user_name: str,
    context_assistant_name: str,
    recent_memory_exclude_last_turns: int = 0,
) -> tuple[bool, str, RetrievedContext]:
    context_result = _collect_turn_context(
        query_prompt=prompt,
        retrieval_input=retrieval_input,
        current_agent_system_prompt=str(client.system_message.get("content", "")),
        retrieval_helper=retrieval_helper,
        options=options,
        rag_store=rag_store,
        working_memory_store=working_memory_store,
        last_rounds_buffer=last_rounds_buffer,
        context_user_name=context_user_name,
        context_assistant_name=context_assistant_name,
        recent_memory_exclude_last_turns=recent_memory_exclude_last_turns,
    )
    _print_context_status_lines(context_result)
    ok, reply = _stream_labeled_reply(
        client,
        label=label,
        prompt=prompt,
        context_text=context_result.context_text,
    )
    return ok, reply, context_result


def _build_retrieval_helper(config: AppConfig, model: ModelConfig) -> ChatClient:
    return ChatClient(
        config=config,
        model=model,
        system_prompt_override=RETRIEVAL_HELPER_SYSTEM_PROMPT,
    )


def _run_agent_loop_chat(
    config: AppConfig,
    model: ModelConfig,
    *,
    rag_store: "HierarchicalRagStore | None",
    working_memory_store: "WorkingMemoryStore | None",
    options: ChatRuntimeOptions,
) -> int:
    loop_options = options.agent_loop
    markdown_logger: MarkdownTranscriptLogger | None = None

    agent1 = ChatClient(
        config=config,
        model=model,
        system_prompt_override=loop_options.agent1_system_prompt,
    )
    agent2 = ChatClient(
        config=config,
        model=model,
        system_prompt_override=loop_options.agent2_system_prompt,
    )
    retrieval_helper = _build_retrieval_helper(config, model)

    agent1_memory_store: WorkingMemoryStore | None = None
    agent2_memory_store: WorkingMemoryStore | None = None
    clear_agent_memory_on_exit = False
    if working_memory_store is not None:
        clear_old_sessions_on_start = config.runtime.working_memory.clear_old_sessions_on_start
        agent1_memory_store = working_memory_store.fork_for_agent(
            f"{working_memory_store.agent_id}.agent1",
            session_id=working_memory_store.session_id,
            clear_old_sessions_on_start=clear_old_sessions_on_start,
        )
        agent2_memory_store = working_memory_store.fork_for_agent(
            f"{working_memory_store.agent_id}.agent2",
            session_id=working_memory_store.session_id,
            clear_old_sessions_on_start=clear_old_sessions_on_start,
        )
        clear_agent_memory_on_exit = True

    agent1_last_rounds = LastRoundsBuffer(
        memory_store=agent1_memory_store,
        max_context_rounds=options.last_n_rounds,
    )
    agent2_last_rounds = LastRoundsBuffer(
        memory_store=agent2_memory_store,
        max_context_rounds=options.last_n_rounds,
    )

    def _cleanup_agent_memory() -> None:
        if not clear_agent_memory_on_exit:
            return
        for store in (agent1_memory_store, agent2_memory_store):
            if store is None:
                continue
            try:
                store.drop_session()
            except Exception:
                pass

    if options.markdown_log_path is not None:
        markdown_logger = MarkdownTranscriptLogger.start(
            options.markdown_log_path,
            model=model,
            rag_enabled=rag_store is not None,
            working_memory_enabled=working_memory_store is not None,
        )

    print(f"Using model alias: {model.alias} ({model.api_name})")
    print(
        "Agent loop enabled: "
        f"{loop_options.agent1_name} -> {loop_options.agent2_name} -> {loop_options.agent1_name}"
    )
    if rag_store is not None:
        print(
            "RAG enabled: parent-only retrieval context "
            f"(mode={options.rag.mode}, top_k={options.rag.top_k})"
        )
    if agent1_memory_store is not None and agent2_memory_store is not None:
        print(
            "Working memory enabled: round retrieval "
            f"(mode={options.working_memory.mode}, "
            f"recent_window={options.working_memory.recent_window_rounds}, "
            f"recent_top_k={options.working_memory.recent_top_k_rounds}, "
            f"older_top_k={options.working_memory.older_top_k_rounds}) "
            f"session={agent1_memory_store.session_id} "
            f"agent1={agent1_memory_store.agent_id} agent2={agent2_memory_store.agent_id}"
        )
    if markdown_logger is not None:
        print(f"Markdown logging: {markdown_logger.path}")
    print(
        "Ask a question to start. After the first answer, press Enter to run a critique cycle. "
        "After each critique cycle, press Enter again to continue. "
        "Type a new question to pivot, '/clear' to reset loop state and last-rounds context, "
        "'/clear-memory' to reset memory, or 'exit'/'quit' to stop.\n"
    )

    pending_question: str | None = None
    try:
        while True:
            user_question = pending_question
            pending_question = None
            if user_question is None:
                user_question = input("You: ").strip()

            if not user_question:
                continue
            lower_question = user_question.lower()
            if _is_exit_command(lower_question):
                return 0
            if _is_clear_command(lower_question):
                _reset_last_round_buffers(
                    agent1_last_rounds,
                    agent2_last_rounds,
                    message="[Context] Reset loop state and cleared last-rounds context.\n",
                )
                continue
            if lower_question == "/clear-memory":
                _clear_memory_stores(
                    agent1_memory_store,
                    agent2_memory_store,
                    success_message="[Memory] Cleared working memory for both agent sessions.\n",
                )
                continue

            ok, agent1_reply, agent1_context = _run_contextual_turn(
                client=agent1,
                label=loop_options.agent1_name,
                prompt=user_question,
                retrieval_input=user_question,
                retrieval_helper=retrieval_helper,
                options=options,
                rag_store=rag_store,
                working_memory_store=agent1_memory_store,
                last_rounds_buffer=agent1_last_rounds,
                context_user_name="Input",
                context_assistant_name=loop_options.agent1_name,
                recent_memory_exclude_last_turns=0,
            )
            if not ok:
                continue

            agent1_last_rounds.append(user_question, agent1_reply)

            if markdown_logger is not None:
                markdown_logger.append_turn(
                    user_text=user_question,
                    assistant_text=f"{loop_options.agent1_name}:\n{agent1_reply}",
                    rag_summary=agent1_context.rag_summary,
                    memory_summary=agent1_context.memory_summary,
                )

            round_index = 0
            while True:
                if loop_options.max_rounds > 0 and round_index >= loop_options.max_rounds:
                    print(f"[Loop] Reached max rounds ({loop_options.max_rounds}). Ask a new question to continue.")
                    break

                control = input(
                    "[Loop] Enter=run critique cycle | new question text | /clear | /clear-memory | exit: "
                ).strip()
                if control:
                    lower_control = control.lower()
                    if _is_exit_command(lower_control):
                        return 0
                    if _is_clear_command(lower_control):
                        _reset_last_round_buffers(
                            agent1_last_rounds,
                            agent2_last_rounds,
                            message="[Context] Reset loop state and cleared last-rounds context.\n",
                        )
                        break
                    if lower_control == "/clear-memory":
                        _clear_memory_stores(
                            agent1_memory_store,
                            agent2_memory_store,
                            success_message="[Memory] Cleared working memory for both agent sessions.\n",
                        )
                        continue
                    pending_question = control
                    break

                ok, agent2_reply, _ = _run_contextual_turn(
                    client=agent2,
                    label=loop_options.agent2_name,
                    prompt=agent1_reply,
                    retrieval_input=agent1_reply,
                    retrieval_helper=retrieval_helper,
                    options=options,
                    rag_store=rag_store,
                    working_memory_store=agent2_memory_store,
                    last_rounds_buffer=agent2_last_rounds,
                    context_user_name="Input",
                    context_assistant_name=loop_options.agent2_name,
                    recent_memory_exclude_last_turns=0,
                )
                if not ok:
                    break

                agent2_last_rounds.append(agent1_reply, agent2_reply)

                ok, next_agent1_reply, _ = _run_contextual_turn(
                    client=agent1,
                    label=loop_options.agent1_name,
                    prompt=agent2_reply,
                    retrieval_input=agent2_reply,
                    retrieval_helper=retrieval_helper,
                    options=options,
                    rag_store=rag_store,
                    working_memory_store=agent1_memory_store,
                    last_rounds_buffer=agent1_last_rounds,
                    context_user_name="Input",
                    context_assistant_name=loop_options.agent1_name,
                    recent_memory_exclude_last_turns=0,
                )
                if not ok:
                    break

                agent1_reply = next_agent1_reply
                round_index += 1
                agent1_last_rounds.append(agent2_reply, agent1_reply)

                if markdown_logger is not None:
                    markdown_logger.append_turn(
                        user_text=f"{loop_options.agent2_name}:\n{agent2_reply}",
                        assistant_text=f"{loop_options.agent1_name}:\n{agent1_reply}",
                    )

            if pending_question:
                continue
    finally:
        _cleanup_agent_memory()


def run_interactive_chat(
    config: AppConfig,
    model: ModelConfig,
    *,
    rag_store: HierarchicalRagStore | None = None,
    working_memory_store: WorkingMemoryStore | None = None,
    runtime_options: ChatRuntimeOptions | None = None,
) -> int:
    options = (runtime_options or ChatRuntimeOptions()).normalized()
    if options.agent_loop.enabled:
        return _run_agent_loop_chat(
            config=config,
            model=model,
            rag_store=rag_store,
            working_memory_store=working_memory_store,
            options=options,
        )

    client = ChatClient(config=config, model=model)
    retrieval_helper = _build_retrieval_helper(config, model)
    chat_rounds = LastRoundsBuffer(
        memory_store=working_memory_store,
        max_context_rounds=options.last_n_rounds,
    )
    markdown_logger: MarkdownTranscriptLogger | None = None
    if options.markdown_log_path is not None:
        markdown_logger = MarkdownTranscriptLogger.start(
            options.markdown_log_path,
            model=model,
            rag_enabled=rag_store is not None,
            working_memory_enabled=working_memory_store is not None,
        )
    print(f"Using model alias: {model.alias} ({model.api_name})")
    if rag_store is not None:
        print(
            "RAG enabled: parent-only retrieval context "
            f"(mode={options.rag.mode}, top_k={options.rag.top_k})"
        )
    if working_memory_store is not None:
        print(
            "Working memory enabled: round retrieval "
            f"(mode={options.working_memory.mode}, "
            f"recent_window={options.working_memory.recent_window_rounds}, "
            f"recent_top_k={options.working_memory.recent_top_k_rounds}, "
            f"older_top_k={options.working_memory.older_top_k_rounds}) "
            f"session={working_memory_store.session_id} agent={working_memory_store.agent_id}"
        )
    if markdown_logger is not None:
        print(f"Markdown logging: {markdown_logger.path}")
    print(
        "Type 'exit' or 'quit' to stop. "
        f"Last-{options.last_n_rounds} rounds context is always injected. "
        "Type '/clear' to reset last-rounds context and '/clear-memory' to reset working memory.\n"
    )

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        lower_user = user.lower()
        if _is_exit_command(lower_user):
            return 0
        if _is_clear_command(lower_user):
            _reset_last_round_buffers(
                chat_rounds,
                message="[Context] Cleared last-rounds context.\n",
            )
            continue
        if lower_user == "/clear-memory":
            _clear_memory_stores(
                working_memory_store,
                success_message="[Memory] Cleared working memory for this session.\n",
            )
            continue

        ok, assistant_text, context_result = _run_contextual_turn(
            client=client,
            label="Assistant",
            prompt=user,
            retrieval_input=user,
            retrieval_helper=retrieval_helper,
            options=options,
            rag_store=rag_store,
            working_memory_store=working_memory_store,
            last_rounds_buffer=chat_rounds,
            context_user_name="User",
            context_assistant_name="Assistant",
            recent_memory_exclude_last_turns=0,
        )
        if not ok:
            continue
        chat_rounds.append(user, assistant_text)
        if markdown_logger is not None:
            markdown_logger.append_turn(
                user_text=user,
                assistant_text=assistant_text,
                rag_summary=context_result.rag_summary,
                memory_summary=context_result.memory_summary,
            )
