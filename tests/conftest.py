import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _engine_available() -> bool:
    try:
        from kaggle_environments import make

        make("cabt")
        return True
    except Exception:
        return False


ENGINE_AVAILABLE = _engine_available()

requires_engine = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="cabt engine unavailable (Linux x86-64 only — run inside Docker)",
)
