"""Test pip install in a fresh environment."""

import subprocess
import sys
from pathlib import Path


class TestPipInstall:
    def test_build_succeeds(self):
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0, result.stderr

    def test_package_imports(self):
        import ragcheck
        import ragcheck.analyzers.chunkers
        import ragcheck.cli
        import ragcheck.core.config
        import ragcheck.reports.html_report
        import ragcheck.testers.auto_qa

        assert ragcheck.__doc__ is not None
