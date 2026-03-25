from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, DOCS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
