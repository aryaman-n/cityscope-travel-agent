from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT if (SOURCE_ROOT / "pyproject.toml").exists() else Path.cwd()


@dataclass(frozen=True)
class Settings:
    data_mode: str = "mock"
    mock_latency_seconds: float = 0.10
    api_timeout_seconds: float = 10.0
    use_llm: bool = False
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    vector_db_path: Path = PROJECT_ROOT / ".travel_data" / "chroma"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env")
        mode = os.getenv("DATA_MODE", "mock").strip().lower()
        if mode not in {"mock", "live"}:
            raise ValueError("DATA_MODE must be either 'mock' or 'live'")
        configured_path = os.getenv("VECTOR_DB_PATH", "").strip()
        return cls(
            data_mode=mode,
            mock_latency_seconds=float(os.getenv("MOCK_LATENCY_SECONDS", "0.10")),
            api_timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", "10")),
            use_llm=os.getenv("USE_LLM", "false").strip().lower() in {"1", "true", "yes"},
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            vector_db_path=Path(configured_path) if configured_path else cls.vector_db_path,
        )
