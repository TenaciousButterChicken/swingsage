"""Environment-driven configuration for SwingSage.

All paths use pathlib.Path. Platform-specific defaults are applied so the
same code runs on Mac (dev, fixtures) and Windows (prod, real VTrack).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_VTRACK_DIR = REPO_ROOT / "tests" / "fixtures" / "vtrack_shots"


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    vtrack_shotdata_path: Path
    use_mock_vtrack: bool
    data_dir: Path
    db_path: Path
    llm_model: str
    ollama_host: str
    log_level: str

    @property
    def is_mac(self) -> bool:
        return sys.platform == "darwin"

    @property
    def is_windows(self) -> bool:
        return sys.platform == "win32"


def load_config(env_file: Path | None = None) -> Config:
    """Load .env (if present) and resolve a Config with platform-aware defaults."""
    if env_file is None:
        env_file = REPO_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    # Mock VTrack is the default on anything that isn't Windows, unless explicitly disabled.
    explicit_mock = os.environ.get("SWINGSAGE_USE_MOCK_VTRACK")
    if explicit_mock is None or explicit_mock == "":
        use_mock = sys.platform != "win32"
    else:
        use_mock = _truthy(explicit_mock)

    raw_vtrack_path = os.environ.get("VTRACK_SHOTDATA_PATH", "").strip()
    if raw_vtrack_path:
        vtrack_path = Path(raw_vtrack_path).expanduser()
    elif use_mock:
        vtrack_path = FIXTURES_VTRACK_DIR
    else:
        # Real VTrack on Windows but no path set — fail loudly later, not here.
        vtrack_path = Path("")

    data_dir = Path(os.environ.get("SWINGSAGE_DATA_DIR", REPO_ROOT / "data" / "runtime")).expanduser()
    db_path = Path(os.environ.get("SWINGSAGE_DB_PATH", data_dir / "swingsage.db")).expanduser()

    return Config(
        vtrack_shotdata_path=vtrack_path,
        use_mock_vtrack=use_mock,
        data_dir=data_dir,
        db_path=db_path,
        llm_model=os.environ.get("SWINGSAGE_LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M"),
        ollama_host=os.environ.get("SWINGSAGE_OLLAMA_HOST", "http://127.0.0.1:11434"),
        log_level=os.environ.get("SWINGSAGE_LOG_LEVEL", "INFO"),
    )
