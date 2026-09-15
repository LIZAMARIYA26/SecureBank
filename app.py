"""Compatibility launcher for the retained fraud-detection application."""

from pathlib import Path
import os
import runpy
import sys


PROJECT_DIR = Path(__file__).resolve().parent / "fraud-detection-randomforest"
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
runpy.run_path(str(PROJECT_DIR / "app.py"), run_name="__main__")
