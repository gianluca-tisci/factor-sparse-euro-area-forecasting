"""Smoke test for src/_style.py: each helper writes its file and its manifest row."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from src import _style


def test_helpers_write_outputs_and_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fig, ax = _style.new_fig()
    ax.plot([0, 1], [0, 1])
    _style.save_fig(fig, "fig_01_smoke")

    table = pd.DataFrame({"rel_MSE": [0.98, -1.01]}, index=["lasso", "Total"])
    _style.save_table(table, "tab_01_smoke", notes="Smoke test, D-16.", rule_before="Total")

    _style.record_number("SmokeValue", "0.98")
    _style.record_number("SmokeValue", "0.97")

    assert (tmp_path / "outputs/figures/fig_01_smoke.pdf").exists()
    tex = (tmp_path / "outputs/tables/tab_01_smoke.tex").read_text()
    assert r"\toprule" in tex and r"rel\_MSE" in tex and "Notes:" in tex
    assert "$-$1.010" in tex and "D-16" in tex and "\\midrule\nTotal &" in tex
    numbers = (tmp_path / "outputs/numbers.tex").read_text()
    assert numbers.count(r"\newcommand{\SmokeValue}") == 1 and "{0.97}" in numbers
    manifest = (tmp_path / "outputs/README.md").read_text()
    assert "| fig_01_smoke |" in manifest and "| tab_01_smoke |" in manifest


def test_macro_names_are_letters_only():
    with pytest.raises(ValueError):
        _style.record_number("Rmse1", "0.5")


def test_stars_follow_conventional_levels():
    assert [_style.stars(p) for p in (0.005, 0.03, 0.07, 0.2)] == ["***", "**", "*", ""]


def test_failed_export_preserves_previous_figure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig, _ = _style.new_fig()
    _style.save_fig(fig, "fig_02_preserved")
    path = Path("outputs/figures/fig_02_preserved.pdf")
    previous = path.read_bytes()

    def failed_export(destination, **kwargs):
        Path(destination).write_bytes(b"incomplete export")
        raise OSError("simulated interrupted export")

    monkeypatch.setattr(fig, "savefig", failed_export)
    with pytest.raises(OSError, match="interrupted export"):
        _style.save_fig(fig, "fig_02_preserved")
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]
