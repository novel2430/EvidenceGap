from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from typing import Mapping

from evidencegap_backend.common import sha256_text

_BUILTIN_PROMPT_FILES = frozenset(
    {
        "statement_decomposition.txt",
        "article_evidence.txt",
        "inference_gap.txt",
        "localization.txt",
    }
)
_TEMPLATE_PATTERN = re.compile(r"\{\{([A-Za-z][A-Za-z0-9_]*)\}\}")


@dataclass(frozen=True)
class PromptOverride:
    """Optional system-prompt replacement loaded by the configuration layer."""

    system_prompt: str | None = None
    additional_instructions: str | None = None
    version: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "system_prompt",
            "additional_instructions",
            "version",
            "source",
        ):
            value = getattr(self, field_name)
            if value is not None:
                value = str(value).strip()
                object.__setattr__(self, field_name, value or None)


@dataclass(frozen=True)
class ResolvedPrompt:
    system_prompt: str
    version: str
    source: str
    sha256: str

    def manifest_fields(self) -> dict[str, str]:
        return {
            "prompt_version": self.version,
            "prompt_sha256": self.sha256,
            "prompt_source": self.source,
        }


def load_builtin_prompt(
    prompt_name: str,
    *,
    substitutions: Mapping[str, object] | None = None,
) -> str:
    """Load and render one packaged built-in system prompt.

    Only named, package-owned prompt files are accepted. Template replacement is
    deliberately limited to ``{{name}}`` tokens supplied by the caller; user
    prompt payload construction remains code-owned.
    """

    if prompt_name not in _BUILTIN_PROMPT_FILES:
        raise ValueError(f"Unknown built-in prompt: {prompt_name!r}")
    prompt = (
        resources.files("evidencegap_backend.prompts")
        .joinpath(prompt_name)
        .read_text(encoding="utf-8")
        .strip()
    )
    values = {str(key): str(value) for key, value in (substitutions or {}).items()}
    referenced = set(_TEMPLATE_PATTERN.findall(prompt))
    missing = sorted(referenced - values.keys())
    if missing:
        raise ValueError(
            f"Missing built-in prompt substitutions for {prompt_name}: {missing}"
        )
    unused = sorted(values.keys() - referenced)
    if unused:
        raise ValueError(
            f"Unused built-in prompt substitutions for {prompt_name}: {unused}"
        )
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", value)
    unresolved = sorted(set(_TEMPLATE_PATTERN.findall(prompt)))
    if unresolved:
        raise ValueError(
            f"Unresolved built-in prompt substitutions for {prompt_name}: "
            f"{unresolved}"
        )
    if not prompt:
        raise ValueError(f"Built-in prompt cannot be blank: {prompt_name}")
    return prompt


def resolve_prompt(
    *,
    default_system_prompt: str,
    default_version: str,
    override: PromptOverride | None,
    default_source: str = "builtin",
) -> ResolvedPrompt:
    value = override or PromptOverride()
    system_prompt = value.system_prompt or default_system_prompt
    if value.additional_instructions:
        system_prompt = (
            system_prompt.rstrip()
            + "\n\nAdditional configured instructions:\n"
            + value.additional_instructions.strip()
        )
    system_prompt = system_prompt.strip()
    if not system_prompt:
        raise ValueError("Resolved system prompt cannot be blank")
    return ResolvedPrompt(
        system_prompt=system_prompt,
        version=value.version or default_version,
        source=value.source or default_source,
        sha256=sha256_text(system_prompt),
    )


def resolve_builtin_prompt(
    *,
    prompt_name: str,
    default_version: str,
    override: PromptOverride | None,
    substitutions: Mapping[str, object] | None = None,
) -> ResolvedPrompt:
    return resolve_prompt(
        default_system_prompt=load_builtin_prompt(
            prompt_name,
            substitutions=substitutions,
        ),
        default_version=default_version,
        override=override,
        default_source=f"package:evidencegap_backend.prompts/{prompt_name}",
    )
