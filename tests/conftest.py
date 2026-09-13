"""Keep temporary test files inside this checkout, avoiding shared Windows temp ACLs."""
from pathlib import Path
from uuid import uuid4


def pytest_configure(config):
    if config.option.basetemp is None:
        root = Path(__file__).resolve().parents[1] / '.cache' / 'pytest'
        root.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = root / str(uuid4())
