"""Central configuration for AtlasRAG.

All tunables live here so the rest of the codebase never reaches for an
environment variable directly. Model IDs default per LLM provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from .env (project root or cwd); real env vars win."""
    root = Path(__file__).resolve().parent.parent
    for path in {root / ".env", Path.cwd() / ".env"}:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env(key: str, default: str) -> str:
    val = os.environ.get(key)
    return val if val else default


# (main model, fast model, judge model, env var with the key)
_PROVIDERS: dict[str, tuple[str, str, str, str]] = {
    "groq": ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "GROQ_API_KEY"),
    "anthropic": ("claude-opus-4-8", "claude-haiku-4-5", "claude-opus-4-8", "ANTHROPIC_API_KEY"),
    "routerai": ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "ROUTERAI_API_KEY"),
}


@dataclass
class Settings:
    """Runtime configuration.

    Every field can be overridden via constructor kwargs or environment
    variables (see ``.env.example``).
    """

    # --- Models -----------------------------------------------------------
    # "groq" (default) or "anthropic"; empty model fields get the provider's defaults.
    llm_provider: str = field(default_factory=lambda: _env("ATLAS_LLM_PROVIDER", "groq"))
    # The primary reasoning model: the retrieval agent loop and grounded generation.
    model: str = field(default_factory=lambda: _env("ATLAS_MODEL", ""))
    # A cheaper/faster model for high-volume, low-stakes calls
    # (HyDE, decomposition, relevance grading, faithfulness check).
    fast_model: str = field(default_factory=lambda: _env("ATLAS_FAST_MODEL", ""))
    # LLM judge for evaluation; by default from another model family than
    # the generator, so it is less inclined to favour its own answers.
    judge_model: str = field(default_factory=lambda: _env("ATLAS_JUDGE_MODEL", ""))
    llm_max_retries: int = 8            # free-tier rate limits make 429 routine
    judge_max_tokens: int = 512        # output budget for the original answer judge
    llm_cache_path: str | None = None   # JSONL response cache (used by evaluation)
    routerai_budget_rub: float = 90.0  # shared persistent ledger, leaves a balance reserve
    routerai_min_balance_rub: float = 10.0
    routerai_ledger_path: str = "results/raw/routerai/usage.jsonl"
    routerai_providers: dict[str, str] = field(default_factory=dict)

    # --- Embeddings -------------------------------------------------------
    # "auto" uses sentence-transformers if importable, else the local hashing
    # fallback. Force one with "sentence-transformers" or "hashing".
    embedding_backend: str = field(
        default_factory=lambda: _env("ATLAS_EMBEDDING_BACKEND", "auto")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("ATLAS_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    )
    hashing_dim: int = 768  # dimension of the local fallback embedder

    # --- Chunking ---------------------------------------------------------
    chunk_target_tokens: int = 320      # ~ target chunk size (approx tokens)
    chunk_overlap_tokens: int = 60      # overlap between adjacent chunks
    chars_per_token: float = 4.0        # rough heuristic for the splitter

    # --- Retrieval --------------------------------------------------------
    retrieval_mode: str = "hybrid"      # hybrid | dense | sparse
    dense_top_k: int = 12               # candidates from vector search
    sparse_top_k: int = 12              # candidates from BM25
    rrf_k: int = 60                     # Reciprocal Rank Fusion constant
    fused_top_k: int = 10               # candidates kept after fusion
    rerank_top_k: int = 5               # passages kept after LLM rerank
    use_hyde: bool = True               # Hypothetical Document Embeddings
    use_reranker: bool = True           # LLM cross-encoder-style rerank

    # --- Agent ------------------------------------------------------------
    max_agent_steps: int = 4            # max search rounds; caps cost per question
    agent_preview_chars: int | None = 500  # None exposes complete chunks to the agent
    context_selection: str = "anchored"  # anchored | anchored_score | legacy
    citation_mode: str = "unit_parts"    # exact short spans; units: strict v1; free: baseline
    min_relevance_to_answer: float = 5.0  # avg rerank score (0-10) gate

    # --- Persistence ------------------------------------------------------
    index_dir: str = field(default_factory=lambda: _env("ATLAS_INDEX_DIR", ".atlas_index"))

    # --- Behaviour --------------------------------------------------------
    verbose: bool = True                # print the agent's reasoning trace

    def __post_init__(self) -> None:
        if self.llm_provider not in _PROVIDERS:
            raise ValueError(f"unknown llm_provider: {self.llm_provider!r}")
        main, fast, judge, _ = _PROVIDERS[self.llm_provider]
        self.model = self.model or main
        self.fast_model = self.fast_model or fast
        self.judge_model = self.judge_model or judge

    @property
    def api_key_env(self) -> str:
        return _PROVIDERS[self.llm_provider][3]

    def has_api_key(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def require_api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} is not set (llm_provider={self.llm_provider}). "
                f"Copy .env.example to .env and add your key, or export it before running."
            )
        return key
