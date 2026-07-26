"""Runtime dense query encoding and FAISS search."""

from evidencegap_backend.dense.encoders import DenseEncoder
from evidencegap_backend.dense.faiss_backend import DenseFaissBackend

__all__ = ["DenseEncoder", "DenseFaissBackend"]
