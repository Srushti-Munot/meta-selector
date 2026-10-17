from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_ROWS = 5000        # subsample bigger datasets down to this
MAX_FEATURES = 200     # skip datasets wider than this
CV_FOLDS = 5