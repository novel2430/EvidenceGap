from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

TOKEN_PATTERN = r"(?u)\b[\w]+(?:[-/][\w]+)*\b"
TOKENIZER_CONTRACT = {
    "unicode_normalization": "NFKC",
    "lowercase": True,
    "token_pattern": TOKEN_PATTERN,
    "stemming": False,
    "stopword_removal": False,
}


def normalize_for_search(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


class ParquetTextStream:
    """Re-iterable stream over one string column in a Parquet file."""

    def __init__(
        self,
        path: Path,
        *,
        column: str = "text",
        batch_size: int = 8192,
        limit: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.column = column
        self.batch_size = batch_size
        self.limit = limit

    def __iter__(self) -> Iterator[str]:
        import pyarrow.parquet as pq

        emitted = 0
        parquet = pq.ParquetFile(self.path)
        for batch in parquet.iter_batches(
            batch_size=self.batch_size,
            columns=[self.column],
        ):
            for value in batch.column(0).to_pylist():
                if self.limit is not None and emitted >= self.limit:
                    return
                yield normalize_for_search(str(value or ""))
                emitted += 1

    def __len__(self) -> int:
        import pyarrow.parquet as pq

        rows = pq.read_metadata(self.path).num_rows
        return min(rows, self.limit) if self.limit is not None else rows


def build_tokenizer():
    try:
        from bm25s.tokenization import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing bm25s. Install requirements/v1-phase02.txt"
        ) from exc
    return Tokenizer(
        lower=True,
        splitter=TOKEN_PATTERN,
        stopwords=None,
        stemmer=None,
    )
