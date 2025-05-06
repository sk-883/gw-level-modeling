from pathlib import Path

# Base project directory
this_dir = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = this_dir / "data" / "raw"
PROCESSED_DATA_DIR = this_dir / "data" / "processed"

# Model output directory (same as processed for simplicity)
MODEL_OUTPUT_DIR = PROCESSED_DATA_DIR