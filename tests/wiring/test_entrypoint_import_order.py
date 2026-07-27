"""The one thing ``AppTest`` cannot catch: how Streamlit launches the script.

``streamlit run ui/app.py`` prepends the **script's directory** to ``sys.path``.
That puts ``ui/`` in front, and from there ``import app`` resolves to
``ui/app.py`` — a module, not the backend package — so the first line that
needs ``app.config`` dies with::

    ModuleNotFoundError: No module named 'app.config'; 'app' is not a package

Under pytest none of this happens: the repository root is already first, so the
entire wiring suite passed while the real application crashed on its login
screen. That gap is the whole reason this file exists. It reproduces the
launcher's path ordering in a subprocess and imports what the entry script
imports — cheap, and it fails for exactly the reason the browser did.

Run in a subprocess deliberately: the fix mutates ``sys.path``, and doing that
inside the test process would leak into every test after it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Put ``ui/`` first, exactly as the launcher does, then run the entry script's
#: import chain the way the login screen reaches it.
PROBE = """
import sys
sys.path.insert(0, {ui!r})
import runpy
runpy.run_path({app!r}, run_name="__main__")
print("ENTRYPOINT-OK")
"""


def test_the_entry_script_survives_streamlits_path_ordering():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE.format(ui=str(ROOT / "ui"), app=str(ROOT / "ui" / "app.py")),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    combined = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in combined, (
        "the entry script shadowed the backend package:\n" + combined
    )
    assert "ENTRYPOINT-OK" in result.stdout, (
        "the entry script did not run to completion:\n" + combined
    )


def test_the_backend_package_still_wins_after_the_fix():
    """The narrow version: with ``ui/`` first, ``app`` must be the package."""
    probe = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'ui')!r});\n"
        f"sys.path.insert(0, {str(ROOT)!r});\n"
        "import app.config; print('PACKAGE-OK', bool(app.config.get_settings()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "PACKAGE-OK" in result.stdout, result.stdout + result.stderr
