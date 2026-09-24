"""Load the research-only package from the checkout without installing it."""
import importlib.util
from pathlib import Path
import sys

_package = Path(__file__).resolve().parents[2] / 'benchmarks' / 'research20'
_spec = importlib.util.spec_from_file_location('research20', _package / '__init__.py',
                                             submodule_search_locations=[str(_package)])
if _spec is None or _spec.loader is None:
    raise RuntimeError('research package unavailable')
_module = importlib.util.module_from_spec(_spec)
sys.modules['research20'] = _module
_spec.loader.exec_module(_module)
