"""Typed Gemini client: persona/user and assistant turns, with retry and cost caps."""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from bangla_datasets.schema import Message, Role, ToolCall, ToolDef

_log = logging.getLogger("bangla_datasets.gemini")

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_MAX_RETRIES = 3


@dataclass
class AssistantResponse:
    """Parsed output of an assistant turn."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


class GeminiClient:
    """Wraps google-genai with typed methods + retry + cost tracking."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_input_tokens_per_run: int | None = None,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY env var."
            )
        self._api_key = key
        self._model = model
        self._max_retries = max_retries
        self._max_input_tokens = max_input_tokens_per_run
        self._tokens_used = 0
        self._client = genai.Client(api_key=key)

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    def _with_retry(
        self,
        fn: Callable[[], types.GenerateContentResponse],
    ) -> types.GenerateContentResponse:
        """Retry with exponential backoff on transient errors."""
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                last_err = e
                _log.warning("Gemini call failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini call failed after {self._max_retries} retries") from last_err

    def _check_budget(self, est_tokens: int) -> None:
        if (
            self._max_input_tokens is not None
            and self._tokens_used + est_tokens > self._max_input_tokens
        ):
            raise RuntimeError(
                f"Cost cap exceeded: {self._tokens_used + est_tokens} > {self._max_input_tokens}"
            )

    def _track_usage(self, response: types.GenerateContentResponse, fallback_text: str) -> None:
        """Count input tokens from real usage_metadata when present, else estimate.

        Real usage is the authoritative signal returned by the API; the
        len//4 + constant estimate is a conservative fallback only.
        """
        usage = getattr(response, "usage_metadata", None)
        if usage is not None and getattr(usage, "prompt_token_count", None):
            self._tokens_used += usage.prompt_token_count or 0
        else:
            self._tokens_used += len(fallback_text) // 4 + 200

    def persona_turn(
        self, system_prompt: str, history: list[Message], seed: int, script: str = "bengali",
    ) -> str:
        """Ask Gemini-as-user for the next user turn.

        ``script`` selects the opening-seed language so the persona's first
        utterance matches its conversation-script axis (bengali | banglish).
        """
        self._check_budget(est_tokens=800)
        if history:
            contents = "\n".join(f"{m.role.value}: {m.content or ''}" for m in history if m.content)
        else:
            # Opening turn: seed with an explicit instruction to ASK for help,
            # not offer it (prevents the persona from playing assistant).
            if script == "banglish":
                contents = (
                    "notun kothopokkhop shuru hocche. apni shahajjo cacchen — "
                    "ekhon apnar lokkho onujayi shohogarirke apnar prosno ba onurodh likhun. "
                    "mone rakbhen: apni shahajjokari nom, apni shahajjo cacchen. "
                    "banglish (romanized bangla) e likhun."
                )
            else:
                contents = (
                    "নতুন কথোপকথন শুরু হচ্ছে। আপনি সাহায্য চাইছেন — "
                    "এখন আপনার লক্ষ্য অনুযায়ী সহকারীকে আপনার প্রশ্ন বা অনুরোধ লিখুন। "
                    "মনে রাখবেন: আপনি সাহায্যকারী নন, আপনি সাহায্য চাইছেন।"
                )
        resp = self._with_retry(lambda: self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.9,
            ),
        ))
        text = (resp.text or "").strip()
        self._track_usage(resp, text)
        return text

    def assistant_turn(
        self,
        system_prompt: str,
        history: list[Message],
        tools: list[ToolDef],
        seed: int,
    ) -> AssistantResponse:
        """Ask Gemini-as-assistant for the next turn. Returns tool calls OR final."""
        self._check_budget(est_tokens=1500)
        declarations = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters_json_schema=t.parameters_json_schema,
            )
            for t in tools
        ]
        contents = self._history_to_contents(history)

        def call() -> types.GenerateContentResponse:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[types.Tool(function_declarations=declarations)],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    # AUTO lets the model choose between a tool call and a natural-language
                    # answer. ANY forces a tool call every turn, which causes infinite loops
                    # when the model has the result but cannot emit a final answer.
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                    ),
                    temperature=0.7,
                ),
            )
        resp = self._with_retry(call)

        # Gemini 3.x models emit an opaque thought_signature on each function-
        # call Part that MUST be replayed when the call is fed back in history.
        # We capture it here by walking the candidate parts (resp.function_calls
        # drops it), then thread it through ToolCall → history reconstruction.
        sig_by_index: dict[int, bytes] = {}
        try:
            parts = resp.candidates[0].content.parts or []
            fc_idx = 0
            for p in parts:
                if getattr(p, "function_call", None):
                    sig = getattr(p, "thought_signature", None)
                    if sig:
                        sig_by_index[fc_idx] = sig
                    fc_idx += 1
        except (IndexError, AttributeError):
            pass

        tool_calls: list[ToolCall] = []
        for i, fc in enumerate(resp.function_calls or []):
            sig_b64 = base64.b64encode(sig_by_index[i]).decode() if i in sig_by_index else None
            # Preserve the model-assigned id when present (3.x); fall back to a
            # deterministic id for 2.x models that don't emit one.
            tc_id = getattr(fc, "id", None) or f"call_{seed}_{i}"
            tool_calls.append(ToolCall(
                id=tc_id,
                name=fc.name,
                arguments=dict(fc.args or {}),
                thought_signature_b64=sig_b64,
            ))
        text = (resp.text or "").strip()
        self._track_usage(resp, text)
        return AssistantResponse(text=text, tool_calls=tool_calls)

    @staticmethod
    def _history_to_contents(history: list[Message]) -> list[types.Content]:
        """Remap the stored message history into google-genai ``Content`` objects.

        Gemini's API differs from the stored model in two places this handles:
        function-response parts must carry role ``"user"`` (not ``"tool"``), and
        each function-call part may need its captured ``thought_signature``
        replayed for Gemini 3.x models. Tool results are passed as parsed
        objects, not double-encoded JSON strings.
        """
        out: list[types.Content] = []
        for m in history:
            if m.role is Role.SYSTEM:
                continue
            # Gemini requires functionResponse parts to carry role="user"; the
            # client supplies the function output. role="model" is rejected with
            # HTTP 400 on the next assistant turn (FIX 1).
            role = "user" if m.role in (Role.TOOL, Role.USER) else "model"
            parts: list[types.Part] = []
            if m.tool_calls:
                for tc in m.tool_calls:
                    fc_part = types.Part(
                        function_call=types.FunctionCall(
                            name=tc.name, args=tc.arguments, id=tc.id,
                        )
                    )
                    # Replay the thought_signature captured from the original
                    # response — required by Gemini 3.x models, ignored by 2.x.
                    if tc.thought_signature_b64:
                        try:
                            fc_part.thought_signature = base64.b64decode(tc.thought_signature_b64)
                        except Exception:  # noqa: BLE001 - signature is best-effort
                            pass
                    parts.append(fc_part)
            if m.role is Role.TOOL:
                # Use the real tool name (not a "tool" placeholder) and pass the
                # result as a parsed object, not a double-encoded JSON string.
                # Find the matching tool call to get the name.
                tc_name = "tool"
                for msg in history:
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc.id == m.tool_call_id:
                                tc_name = tc.name
                                break
                try:
                    result_obj = json.loads(m.content or "{}")
                except (ValueError, TypeError):
                    result_obj = {"result": m.content or ""}
                parts.append(types.Part(function_response=types.FunctionResponse(
                    name=tc_name, response=result_obj, id=m.tool_call_id,
                )))
            elif m.content:
                parts.append(types.Part(text=m.content))
            out.append(types.Content(role=role, parts=parts))
        return out
