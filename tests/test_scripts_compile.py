import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_entry_points_compile():
    for path in [ROOT / "cli.py", ROOT / "examples" / "demo.py", *(ROOT / "scripts").glob("*.py")]:
        py_compile.compile(str(path), doraise=True)
