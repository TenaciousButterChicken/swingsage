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
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
DEFAULT_GOLFDB_DIR = THIRD_PARTY_DIR / "golfdb"
DEFAULT_SWINGNET_WEIGHTS = DEFAULT_GOLFDB_DIR / "models" / "swingnet_1800.pth.tar"
DEFAULT_NLF_WEIGHTS = THIRD_PARTY_DIR / "nlf" / "weights" / "nlf_l_multi_0.3.2.torchscript"


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    vtrack_shotdata_path: Path
    use_mock_vtrack: bool
    data_dir: Path
    db_path: Path
    swingnet_weights: Path
    golfdb_dir: Path
    nlf_weights: Path
    llm_model: str
    llm_api_base: str
    log_level: str
    # VTrack ↔ GSPro bridge (OpenConnect v1 over localhost TCP).
    openconnect_enabled: bool
    openconnect_host: str
    openconnect_port: int
    openconnect_gspro_host: str
    openconnect_gspro_port: int
    # How fresh a shot must be to be joined to an uploaded swing video.
    ball_shot_max_age_sec: int

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

    # OpenConnect bridge defaults to ON on Windows (real hardware), OFF elsewhere
    # unless explicitly enabled. The env var always wins when set.
    explicit_openconnect = os.environ.get("SWINGSAGE_OPENCONNECT_ENABLED")
    if explicit_openconnect is None or explicit_openconnect == "":
        openconnect_enabled = sys.platform == "win32"
    else:
        openconnect_enabled = _truthy(explicit_openconnect)

    return Config(
        vtrack_shotdata_path=vtrack_path,
        use_mock_vtrack=use_mock,
        data_dir=data_dir,
        db_path=db_path,
        swingnet_weights=Path(os.environ.get("SWINGSAGE_SWINGNET_WEIGHTS", DEFAULT_SWINGNET_WEIGHTS)).expanduser(),
        golfdb_dir=Path(os.environ.get("SWINGSAGE_GOLFDB_DIR", DEFAULT_GOLFDB_DIR)).expanduser(),
        nlf_weights=Path(os.environ.get("SWINGSAGE_NLF_WEIGHTS", DEFAULT_NLF_WEIGHTS)).expanduser(),
        llm_model=os.environ.get("SWINGSAGE_LLM_MODEL", "qwen3-14b"),
        llm_api_base=os.environ.get("SWINGSAGE_LLM_API_BASE", "http://127.0.0.1:1234/v1"),
        log_level=os.environ.get("SWINGSAGE_LOG_LEVEL", "INFO"),
        openconnect_enabled=openconnect_enabled,
        openconnect_host=os.environ.get("SWINGSAGE_OPENCONNECT_HOST", "127.0.0.1"),
        openconnect_port=int(os.environ.get("SWINGSAGE_OPENCONNECT_PORT", "921")),
        openconnect_gspro_host=os.environ.get("SWINGSAGE_OPENCONNECT_GSPRO_HOST", "127.0.0.1"),
        openconnect_gspro_port=int(os.environ.get("SWINGSAGE_OPENCONNECT_GSPRO_PORT", "922")),
        ball_shot_max_age_sec=int(os.environ.get("SWINGSAGE_BALL_SHOT_MAX_AGE_SEC", "120")),
    )
