"""Tests for CLI commands."""


from typer.testing import CliRunner

from ragcheck.cli import app

runner = CliRunner()


class TestInit:
    def test_init_creates_config(self, tmp_path):
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "ragcheck.yaml").exists()


class TestRun:
    def test_run_missing_docs(self):
        result = runner.invoke(app, ["run", "--docs", "./nonexistent"])
        assert result.exit_code == 1

    def test_run_with_docs(self, tmp_path):
        docs_dir = tmp_path / "data"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("RAG is great. It uses embeddings.")
        result = runner.invoke(app, ["run", "--docs", str(docs_dir)])
        assert result.exit_code == 0
        assert "Report saved to" in result.output

    def test_ci_mode_pass(self, tmp_path):
        docs_dir = tmp_path / "data"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("RAG is great. It uses embeddings.")
        result = runner.invoke(app, ["run", "--docs", str(docs_dir), "--ci", "--min-score", "0.0"])
        assert result.exit_code == 0
        assert "CI PASSED" in result.output

    def test_ci_mode_fail(self, tmp_path):
        docs_dir = tmp_path / "data"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("")
        result = runner.invoke(app, ["run", "--docs", str(docs_dir), "--ci", "--min-score", "1.0"])
        assert result.exit_code == 1
