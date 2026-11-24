from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
CONFIGS_DIR = DATA_DIR / "configs"
SPLITS_DIR = DATA_DIR / "splits"