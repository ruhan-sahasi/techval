"""Every techval command starts without paying for libraries only some commands use."""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_cli_loads_no_model_or_chart_library():
    # scikit-learn, scipy.stats and matplotlib together took two seconds of
    # every command's start, --help included; each is imported where it is used.
    probe = (
        "import sys, techval.cli; "
        "print(','.join(m for m in ('sklearn', 'scipy.stats', 'matplotlib') if m in sys.modules))"
    )
    loaded = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert loaded.stdout.strip() == ""
