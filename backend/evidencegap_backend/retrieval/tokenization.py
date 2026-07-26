from __future__ import annotations

import unicodedata

TOKEN_PATTERN = r"(?u)\b[\w]+(?:[-/][\w]+)*\b"


def normalize_for_search(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def build_tokenizer():
    try:
        from bm25s.tokenization import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing bm25s. Install the backend runtime dependencies"
        ) from exc
    return Tokenizer(
        lower=True,
        splitter=TOKEN_PATTERN,
        stopwords=None,
        stemmer=None,
    )
