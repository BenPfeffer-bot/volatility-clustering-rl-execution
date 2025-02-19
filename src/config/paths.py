import os
from pathlib import Path


ROOT_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Storage Directories
DATAS_DIR = ROOT_DIR / "db"
LOGS_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "output"
SRC_DIR = ROOT_DIR / "src"


# RAW DATA DIRECTORIES
DATA_DIR = SRC_DIR / "data"
RAW_DIR = DATAS_DIR / "raw"
PROCESSED_DIR = DATAS_DIR / "processed"

# OUTPUT DIRECTORIES
SIGNALS_DIR = OUTPUT_DIR / "signals"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
BACKTESTS_DIR = OUTPUT_DIR / "backtests"
PLOTS_DIR = OUTPUT_DIR / "plots"


# Create all directories
DIRS = [
    DATAS_DIR,
    LOGS_DIR,
    OUTPUT_DIR,
    SRC_DIR,
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SIGNALS_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    BACKTESTS_DIR,
    PLOTS_DIR,
]


def create_directories():
    """Create all required directories if they don't exist."""
    for directory in DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories when module is imported
create_directories()
