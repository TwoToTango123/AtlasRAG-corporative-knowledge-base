"""Central configuration for AtlasRAG.

All tunables live here so the rest of the codebase never reaches for an
environment variable directly. Model IDs default to the latest Claude models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    val = os.environ.get(key)
    return val if val else default


@dataclass
class Settings:
    """Runtime configuration.

    Every field can be overridden via constructor kwargs or environment
    variables (see ``.env.example``).
    """

    # --- Models -----------------------------------------------------------
    # The primary reasoning model: query planning, the retrieval agent loop,
    # reranking, grading, grounded generation, and evaluation.
    model: str = field(default_factory=lambda: _env("ATLAS_MODEL", "claude-opus-4-8"))
    # A cheaper/faster model used for high-volume, low-stakes calls
    # (e.g. relevance grading of many candidates).
    fast_model: str = field(
        default_factory=lambda: _env("ATLAS_FAST_MODEL", "claude-haiku-4-5")
    )

    # --- Embeddings -------------------------------------------------------
    # "auto" uses sentence-transformers if importable, else the local hashing
    # fallback. Force one with "sentence-transformers" or "hashing".
    embedding_backend: str = field(
        default_factory=lambda: _env("ATLAS_EMBEDDING_BACKEND", "auto")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("ATLAS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    hashing_dim: int = 768  # dimension of the local fallback embedder

    # --- Chunking ---------------------------------------------------------
    chunk_target_tokens: int = 320      # ~ target chunk size (approx tokens)
    chunk_overlap_tokens: int = 60      # overlap between adjacent chunks
    chars_per_token: float = 4.0        # rough heuristic for the splitter

    # --- Retrieval --------------------------------------------------------
    dense_top_k: int = 12               # candidates from vector search
    sparse_top_k: int = 12              # candidates from BM25
    rrf_k: int = 60                     # Reciprocal Rank Fusion constant
    fused_top_k: int = 10               # candidates kept after fusion
    rerank_top_k: int = 5               # passages kept after LLM rerank
    use_hyde: bool = True               # Hypothetical Document Embeddings
    use_reranker: bool = True           # LLM cross-encoder-style rerank

    # --- Agent ------------------------------------------------------------
    max_agent_steps: int = 6            # max search rounds the agent may take
    min_relevance_to_answer: float = 5.0  # avg rerank score (0-10) gate

    # --- Persistence ------------------------------------------------------
    index_dir: str = field(default_factory=lambda: _env("ATLAS_INDEX_DIR", ".atlas_index"))

    # --- Behaviour --------------------------------------------------------
    verbose: bool = True                # print the agent's reasoning trace

    def require_api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or `export ANTHROPIC_API_KEY=...` before running."
            )
        return key
