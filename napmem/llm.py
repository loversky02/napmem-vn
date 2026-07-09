from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def load_dotenv(start: Path) -> None:
    """Best-effort KEY=VALUE loader, matching the lightweight AutoMem pattern."""

    candidates = [directory / ".env" for directory in (start, *start.parents)]
    workspace = start
    for directory in start.parents:
        if directory.name == "Build with Paper":
            workspace = directory
            break
    for sibling in ("automem-vn", "Agent-as-a-Router", "System-III-Router", "jerp-docex", "rtl-gauntlet"):
        candidates.append(workspace / sibling / ".env")

    for path in candidates:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 700
    verify_ssl: bool = True
    timeout_s: float = 120.0

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        verify_ssl: bool | None = None,
        timeout_s: float | None = None,
    ) -> "LLMConfig":
        load_dotenv(Path(__file__).resolve().parent)
        base_url = (
            os.environ.get("NINEROUTER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or ""
        ).rstrip("/")
        api_key = os.environ.get("NINEROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        model_name = (
            model
            or os.environ.get("NAPMEM_MODEL")
            or os.environ.get("AUTOMEM_MODEL")
            or os.environ.get("ACROUTER_ROUTER_MODEL")
            or os.environ.get("SYS3_SOLVER_MODEL")
            or os.environ.get("JERP_LLM_MODEL")
            or os.environ.get("RTLG_MODEL")
            or "gpt-5.5"
        )
        if not base_url or not api_key:
            raise RuntimeError(
                "No 9router/OpenAI-compatible endpoint configured. Set "
                "NINEROUTER_BASE_URL + NINEROUTER_API_KEY, or OPENAI_BASE_URL + OPENAI_API_KEY."
            )
        if verify_ssl is None:
            verify_ssl = os.environ.get("NAPMEM_INSECURE_SSL", "").lower() not in {"1", "true", "yes"}
        if timeout_s is None:
            timeout_s = float(os.environ.get("NAPMEM_HTTP_TIMEOUT", "120"))
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            verify_ssl=verify_ssl,
            timeout_s=timeout_s,
        )


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, messages: list[dict[str, str]], **overrides) -> str:
        body = {
            "model": overrides.get("model", self.config.model),
            "messages": messages,
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            context = None
            if not self.config.verify_ssl:
                context = ssl._create_unverified_context()
            timeout = float(overrides.get("timeout_s", self.config.timeout_s))
            with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
                raw = response.read().decode("utf-8")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    if raw.lstrip().startswith("data:"):
                        return _parse_sse_content(raw)
                    raise RuntimeError(
                        f"LLM returned non-JSON HTTP {response.status}: {raw[:500]}"
                    ) from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8")[:500]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        return payload["choices"][0]["message"]["content"] or ""


def _parse_sse_content(raw: str) -> str:
    parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        chunk = json.loads(data)
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        parts.append(delta.get("content") or message.get("content") or "")
    return "".join(parts)


Backend = Callable[[list[dict[str, str]]], str]


def client_from_env(
    model: str | None = None,
    verify_ssl: bool | None = None,
    timeout_s: float | None = None,
) -> Backend:
    client = LLMClient(LLMConfig.from_env(model, verify_ssl, timeout_s))
    return lambda messages: client.complete(messages)


def is_configured() -> bool:
    try:
        LLMConfig.from_env()
        return True
    except RuntimeError:
        return False
