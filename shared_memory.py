"""Configure the one durable Cognee store shared by every model/client."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Capture path configuration before importing Cognee. Cognee initializes its
# own dotenv-backed settings at import time, which must not be allowed to
# replace an explicit per-process test root.
_CONFIGURED_SHARED_ROOT = os.environ.get(
    "COGNEE_SHARED_ROOT", str(PROJECT_ROOT / ".cognee_data")
)

import cognee  # noqa: E402  Provider settings are loaded before this import.


SHARED_ROOT = Path(_CONFIGURED_SHARED_ROOT).expanduser().resolve()
DATA_ROOT = SHARED_ROOT / "data"
SYSTEM_ROOT = SHARED_ROOT / "system"


def configure_shared_memory() -> None:
    """Point every Cognee process at the same model-independent state."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SYSTEM_ROOT.mkdir(parents=True, exist_ok=True)
    cognee.config.data_root_directory(str(DATA_ROOT))
    cognee.config.system_root_directory(str(SYSTEM_ROOT))


configure_shared_memory()
