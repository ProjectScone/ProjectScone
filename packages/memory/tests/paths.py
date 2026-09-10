"""Shared fixture and repository locations independent of test category depth."""
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = TESTS_ROOT.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
