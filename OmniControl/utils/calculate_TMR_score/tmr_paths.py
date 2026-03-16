import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY_ROOT = REPO_ROOT / "third-party"
TMR_ROOT = THIRD_PARTY_ROOT / "TMR"


def ensure_tmr_on_path():
    for path in (THIRD_PARTY_ROOT, TMR_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def get_tmr_path(*parts):
    return str(TMR_ROOT.joinpath(*parts))


def resolve_tmr_path(path):
    if path in (None, ""):
        return path

    normalized = os.path.expanduser(str(path)).replace("\\", "/")
    legacy_prefixes = (
        "TMR",
        "./TMR",
        "third-party/TMR",
        "./third-party/TMR",
    )

    for prefix in legacy_prefixes:
        if normalized == prefix:
            return str(TMR_ROOT)
        if normalized.startswith(prefix + "/"):
            suffix = normalized[len(prefix) + 1 :]
            return str(TMR_ROOT / suffix)

    return os.path.expanduser(str(path))
