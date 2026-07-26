from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from evidencegap_backend.common import EvidenceGapError, sha256_text

SUPPORTED_PROVIDERS = ("deepseek", "anthropic")
DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "anthropic": "claude-sonnet-4-6",
}
DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "anthropic": "https://api.anthropic.com",
}
DEFAULT_API_KEY_ENVS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_THINKING_DISABLED_MODELS = frozenset({"claude-sonnet-5"})


@dataclass(frozen=True)
class ProviderResponse:
    payload: Mapping[str, Any]
    request_id: str | None
    usage: Mapping[str, int]
    raw_response_sha256: str
    finish_reason: str | None


class _ProviderError(EvidenceGapError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


_RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _post_json(
    url: str,
    *,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[Mapping[str, Any], str]:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(raw_error)
            detail = json.dumps(error_payload, ensure_ascii=False)
        except json.JSONDecodeError:
            detail = raw_error[:2000]
        raise _ProviderError(
            f"HTTP {exc.code} from {url}: {detail}",
            retryable=exc.code in _RETRYABLE_HTTP_STATUS,
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise _ProviderError(
            f"Network error calling {url}: {exc}", retryable=True
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _ProviderError(
            f"Provider returned non-JSON HTTP response: {raw[:1000]!r}",
            retryable=True,
        ) from exc
    if not isinstance(payload, Mapping):
        raise _ProviderError("Provider response must be a JSON object", retryable=True)
    return payload, raw


def _parse_json_content(content: str) -> Mapping[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text:
        raise _ProviderError("Provider returned empty content", retryable=True)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _ProviderError(
            f"Provider content is not valid JSON: {exc}", retryable=True
        ) from exc
    if not isinstance(payload, Mapping):
        raise _ProviderError("Provider JSON content must be an object", retryable=True)
    return payload


def _anthropic_thinking_config(model: str) -> dict[str, str] | None:
    normalized = model.strip().lower()
    if normalized in ANTHROPIC_THINKING_DISABLED_MODELS:
        return {"type": "disabled"}
    return None


def call_structured_llm(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: Mapping[str, Any],
    max_tokens: int,
    timeout_seconds: float,
    thinking: bool = False,
) -> ProviderResponse:
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise EvidenceGapError(
            f"provider must be one of {SUPPORTED_PROVIDERS}, got {provider!r}"
        )
    if provider != "deepseek" and thinking:
        raise EvidenceGapError("thinking is only supported for DeepSeek")

    if provider == "deepseek":
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "stream": False,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        response, raw = _post_json(
            base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            body=body,
            timeout_seconds=timeout_seconds,
        )
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _ProviderError(
                "Unexpected DeepSeek response shape: "
                + json.dumps(response, ensure_ascii=False)[:2000],
                retryable=True,
            ) from exc
        finish_reason = (
            None
            if choice.get("finish_reason") is None
            else str(choice["finish_reason"])
        )
        if finish_reason == "length":
            raise _ProviderError(
                "DeepSeek output was truncated (finish_reason=length); "
                "increase --max-tokens or reduce --request-batch-size",
                retryable=False,
            )
        usage_raw = response.get("usage") or {}
        return ProviderResponse(
            payload=_parse_json_content(str(content or "")),
            request_id=None if response.get("id") is None else str(response["id"]),
            usage={
                "input_tokens": int(usage_raw.get("prompt_tokens") or 0),
                "output_tokens": int(usage_raw.get("completion_tokens") or 0),
                "total_tokens": int(usage_raw.get("total_tokens") or 0),
            },
            raw_response_sha256=sha256_text(raw),
            finish_reason=finish_reason,
        )

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": dict(response_schema),
            }
        },
    }
    thinking_config = _anthropic_thinking_config(model)
    if thinking_config is not None:
        body["thinking"] = thinking_config
    response, raw = _post_json(
        base_url.rstrip("/") + "/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        },
        body=body,
        timeout_seconds=timeout_seconds,
    )
    stop_reason = (
        None if response.get("stop_reason") is None else str(response["stop_reason"])
    )
    if stop_reason in {"max_tokens", "model_context_window_exceeded"}:
        raise _ProviderError(
            f"Claude output was truncated (stop_reason={stop_reason}); "
            "increase --max-tokens or reduce --request-batch-size",
            retryable=False,
        )
    if stop_reason == "refusal":
        raise _ProviderError("Claude refused the structured request", retryable=False)
    content_blocks = response.get("content")
    if not isinstance(content_blocks, list):
        raise _ProviderError("Unexpected Claude content shape", retryable=True)
    text_blocks = [
        str(block.get("text"))
        for block in content_blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    ]
    if not text_blocks:
        raise _ProviderError("Claude returned no text content block", retryable=True)
    usage_raw = response.get("usage") or {}
    input_tokens = int(usage_raw.get("input_tokens") or 0)
    output_tokens = int(usage_raw.get("output_tokens") or 0)
    return ProviderResponse(
        payload=_parse_json_content("\n".join(text_blocks)),
        request_id=None if response.get("id") is None else str(response["id"]),
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        raw_response_sha256=sha256_text(raw),
        finish_reason=stop_reason,
    )
