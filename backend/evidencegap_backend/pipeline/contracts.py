from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from evidencegap_backend.common import EvidenceGapError, sha256_text

RUNTIME_SENTENCE_SCHEMA_VERSION = "1.0.0"
RUNTIME_SENTENCE_CONTRACT_ID = "phase07.runtime-sentence.v1"
ARTICLE_RECORD_TYPE = "RuntimeArticle"
SENTENCE_RECORD_TYPE = "RuntimeSentence"


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or None


def _json_safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise EvidenceGapError("Article metadata must be JSON-serializable") from exc
    if not isinstance(decoded, dict):
        raise EvidenceGapError("Article metadata must be a JSON object")
    return decoded


@dataclass(frozen=True)
class ArticleSection:
    section: str
    text: str

    def validate(self) -> None:
        if not self.section.strip():
            raise EvidenceGapError("Article section name cannot be empty")
        if not self.text.strip():
            raise EvidenceGapError(f"Article section {self.section!r} cannot be empty")


@dataclass(frozen=True)
class RuntimeArticle:
    article_id: str
    pmid: str | None = None
    title: str | None = None
    abstract: str | None = None
    sections: tuple[ArticleSection, ...] = ()
    article_rank: int | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.article_id.strip():
            raise EvidenceGapError("Runtime article_id cannot be empty")
        if self.article_rank is not None and self.article_rank <= 0:
            raise EvidenceGapError(
                f"{self.article_id}: article_rank must be positive when present"
            )
        for section in self.sections:
            section.validate()
        if self.abstract is not None and self.sections:
            raise EvidenceGapError(
                f"{self.article_id}: use abstract or sections, not both"
            )
        if self.title is None and self.abstract is None and not self.sections:
            raise EvidenceGapError(
                f"{self.article_id}: at least one of title, abstract, text, or sections is required"
            )
        _json_safe_metadata(self.source_metadata)

    @property
    def body_sections(self) -> tuple[ArticleSection, ...]:
        if self.sections:
            return self.sections
        if self.abstract is None:
            return ()
        return (ArticleSection(section="abstract", text=self.abstract),)

    @property
    def content_fingerprint(self) -> str:
        payload = {
            "article_id": self.article_id,
            "pmid": self.pmid,
            "title": self.title,
            "sections": [
                {"section": section.section, "text": section.text}
                for section in self.body_sections
            ],
        }
        return sha256_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": RUNTIME_SENTENCE_SCHEMA_VERSION,
            "contract_id": RUNTIME_SENTENCE_CONTRACT_ID,
            "record_type": ARTICLE_RECORD_TYPE,
            "article_id": self.article_id,
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "sections_json": json.dumps(
                [
                    {"section": section.section, "text": section.text}
                    for section in self.sections
                ],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "article_rank": self.article_rank,
            "content_fingerprint": self.content_fingerprint,
            "source_metadata_json": json.dumps(
                _json_safe_metadata(self.source_metadata),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeArticle":
        article_id_value = value.get("article_id")
        if article_id_value is None:
            article_id_value = value.get("paper_id")
        if article_id_value is None:
            article_id_value = value.get("pmid")
        if article_id_value is None:
            raise EvidenceGapError(
                "Runtime article input requires article_id, paper_id, or pmid"
            )

        title = _clean_optional_text(value.get("title"))
        explicit_abstract = _clean_optional_text(value.get("abstract"))
        compatibility_text = _clean_optional_text(value.get("text"))
        sections = _parse_sections(value.get("sections"))
        if sections and (explicit_abstract is not None or compatibility_text is not None):
            raise EvidenceGapError(
                f"{article_id_value}: sections cannot be combined with abstract/text"
            )
        abstract = explicit_abstract if explicit_abstract is not None else compatibility_text

        metadata_excluded = {
            "article_id",
            "paper_id",
            "pmid",
            "title",
            "abstract",
            "text",
            "sections",
            "article_rank",
            "final_article_rank",
        }
        metadata = {
            str(key): item
            for key, item in value.items()
            if str(key) not in metadata_excluded
        }
        article_rank_value = value.get("article_rank", value.get("final_article_rank"))
        article = cls(
            article_id=str(article_id_value),
            pmid=None if value.get("pmid") is None else str(value.get("pmid")),
            title=title,
            abstract=abstract,
            sections=sections,
            article_rank=(
                None if article_rank_value is None else int(article_rank_value)
            ),
            source_metadata=_json_safe_metadata(metadata),
        )
        article.validate()
        return article


def _parse_sections(value: Any) -> tuple[ArticleSection, ...]:
    if value is None:
        return ()
    raw_sections: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        raw_sections = [(str(key), text) for key, text in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise EvidenceGapError(
                    f"sections[{index}] must be an object with section/name and text"
                )
            name = item.get("section", item.get("name"))
            if name is None:
                raise EvidenceGapError(
                    f"sections[{index}] requires section or name"
                )
            raw_sections.append((str(name), item.get("text")))
    else:
        raise EvidenceGapError("sections must be an object or a list of objects")

    result: list[ArticleSection] = []
    for name, raw_text in raw_sections:
        text = _clean_optional_text(raw_text)
        if text is None:
            continue
        section = ArticleSection(section=name.strip().lower(), text=text)
        section.validate()
        result.append(section)
    return tuple(result)


@dataclass(frozen=True)
class RuntimeSentence:
    sentence_id: str
    article_id: str
    pmid: str | None
    article_rank: int | None
    sentence_index: int
    sentence_index_within_section: int
    sentence_type: str
    section: str
    section_index: int
    sentence_text: str
    character_start: int
    character_end: int
    section_character_start: int
    section_character_end: int
    source_text_fingerprint: str
    splitter_fingerprint: str

    def validate(self, *, source_text: str | None = None) -> None:
        if not self.sentence_id.strip():
            raise EvidenceGapError("Runtime sentence_id cannot be empty")
        if not self.article_id.strip():
            raise EvidenceGapError(f"{self.sentence_id}: article_id cannot be empty")
        if self.sentence_index < 0:
            raise EvidenceGapError(f"{self.sentence_id}: sentence_index cannot be negative")
        if self.sentence_index_within_section < 0:
            raise EvidenceGapError(
                f"{self.sentence_id}: sentence_index_within_section cannot be negative"
            )
        if self.section_index < 0:
            raise EvidenceGapError(f"{self.sentence_id}: section_index cannot be negative")
        if not self.sentence_text:
            raise EvidenceGapError(f"{self.sentence_id}: sentence_text cannot be empty")
        if not 0 <= self.character_start < self.character_end:
            raise EvidenceGapError(f"{self.sentence_id}: invalid document character offsets")
        if not 0 <= self.section_character_start < self.section_character_end:
            raise EvidenceGapError(f"{self.sentence_id}: invalid section character offsets")
        if source_text is not None:
            recovered = source_text[self.character_start : self.character_end]
            if recovered != self.sentence_text:
                raise EvidenceGapError(
                    f"{self.sentence_id}: offsets do not reproduce sentence_text"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": RUNTIME_SENTENCE_SCHEMA_VERSION,
            "contract_id": RUNTIME_SENTENCE_CONTRACT_ID,
            "record_type": SENTENCE_RECORD_TYPE,
            "sentence_id": self.sentence_id,
            "article_id": self.article_id,
            "pmid": self.pmid,
            "article_rank": self.article_rank,
            "sentence_index": self.sentence_index,
            "sentence_index_within_section": self.sentence_index_within_section,
            "sentence_type": self.sentence_type,
            "section": self.section,
            "section_index": self.section_index,
            "sentence_text": self.sentence_text,
            "character_start": self.character_start,
            "character_end": self.character_end,
            "section_character_start": self.section_character_start,
            "section_character_end": self.section_character_end,
            "source_text_fingerprint": self.source_text_fingerprint,
            "splitter_fingerprint": self.splitter_fingerprint,
        }
