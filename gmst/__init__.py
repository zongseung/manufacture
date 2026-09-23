"""Paths shared by the local KAMP forecasting pipeline."""
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
DATA: Final = ROOT / "5. 자원 최적화 AI 데이터셋"
RESULTS: Final = ROOT / "results"
