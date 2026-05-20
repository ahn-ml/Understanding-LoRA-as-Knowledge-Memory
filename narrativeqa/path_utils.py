from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "runs"
LOGS_DIR = ROOT_DIR / "logs"
CONFIGS_DIR = ROOT_DIR / "configs"


def resolve_with_root(value, default=None):
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path
