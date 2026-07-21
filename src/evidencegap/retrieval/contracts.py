from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SearchHit:
    doc_idx: int
    article_id: str
    score: float
    rank: int


class SparseRetriever(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int,
        exclude_doc_indices: Sequence[int] = (),
    ) -> list[SearchHit]:
        raise NotImplementedError

    @abstractmethod
    def score_documents(
        self,
        query: str,
        doc_indices: Sequence[int],
    ) -> list[float]:
        raise NotImplementedError
