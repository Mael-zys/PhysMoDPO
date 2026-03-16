from pathlib import Path


OMNICONTROL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = OMNICONTROL_ROOT.parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third-party"
PROTO_MOTIONS_ROOT = THIRD_PARTY_ROOT / "ProtoMotions"
PROTO_MOTIONS3_ROOT = THIRD_PARTY_ROOT / "ProtoMotions3"


def resolve_omnicontrol_path(path, *parts):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = OMNICONTROL_ROOT / resolved
    if parts:
        resolved = resolved.joinpath(*map(str, parts))
    return str(resolved.resolve())
