"""Project style: palette, figure theme, saving figures, tables and quoted numbers.

Import in every script that makes a figure, table or number for the paper:
    from src._style import PALETTE, record_number, save_fig, save_table, stars
Choices were made at project setup (AGENTS.md § Conventions); change them here only.
Paths are relative to the repo root, so run scripts from there.
"""

import re
import sys
import tempfile
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# ---- Choices (filled in by /project-init) ------------------------------------
PROFILE = "paper"  # paper | course | report | slides
PALETTE = ["#004488", "#BB5566", "#DDAA33"]  # A: Tol High-contrast, 2-3 series
PALETTE_MANY = ["#332288", "#CC6677", "#117733", "#DDCC77", "#88CCEE"]  # Tol Muted, 4-5 series
STARS = True  # significance stars, always next to SEs or p-values
# Serif face matching the paper's Palatino (mathpazo); the first one installed is used:
# Palatino (macOS), Palatino Linotype (Windows), TeX Gyre Pagella / URW Palladio (Linux).
FONT = ["Palatino", "Palatino Linotype", "TeX Gyre Pagella", "URW Palladio L", "DejaVu Serif"]

OUT = Path("outputs")

# ---- Figures -----------------------------------------------------------------
# Figures are drawn at their printed size (SIZES) and included at that width, so the
# text in them prints at 7-8 pt, just below the 9 pt captions. Titles are plain panel
# labels; what the figure shows goes in the LaTeX caption and notes.
_PAPER_TEXT = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
}
mpl.rcParams.update(
    {
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#E3E3E3",
        "grid.linewidth": 0.5,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "font.family": "serif",
        "font.serif": FONT,
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
        "pdf.fonttype": 42,  # embed TrueType, so the PDF text stays text
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 300,
        **({"font.size": 16} if PROFILE == "slides" else _PAPER_TEXT),
    }
)

# Widths of the paper's text block (US letter, 0.45 in margins, two columns, 18 pt gap).
SIZES = {
    "column": (3.6, 2.4),
    "full": (7.5, 3.4),
    "report": (7.0, 4.5),
    "slide": (10, 5.6),
}


def new_fig(preset=None):
    """A figure sized for its destination."""
    preset = preset or ("slide" if PROFILE == "slides" else "full")
    return plt.subplots(figsize=SIZES[preset])


def save_fig(fig, fig_id, used_in="", formats=None):
    """Save under outputs/figures/ and record it in outputs/README.md."""
    formats = formats or (["pdf"] if PROFILE in ("paper", "course") else ["png"])
    out = OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    files = [str(out / f"{fig_id}.{fmt}") for fmt in formats]
    for f, fmt in zip(files, formats, strict=True):
        # Write beside the destination, then replace atomically. On Windows a
        # PDF viewer or virus scanner can briefly hold the existing file; an
        # interrupted export must not truncate the last valid exhibit.
        with tempfile.NamedTemporaryFile(dir=out, suffix=f".{fmt}", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            fig.savefig(temporary, format=fmt)
            for attempt in range(10):
                try:
                    temporary.replace(f)
                    break
                except OSError:
                    if attempt == 9:
                        raise
                    time.sleep(0.1)
        finally:
            temporary.unlink(missing_ok=True)
    record_output(fig_id, files, used_in)
    return files


# ---- Tables ------------------------------------------------------------------
def stars(p):
    """Stars for a p-value per the STARS switch: * 0.10, ** 0.05, *** 0.01."""
    if not STARS:
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


_NUMBER = re.compile(r"[$\\textbf{(]*[-+]?(\d[\d,]*\.?\d*|\.\d+)[)}$]*%?\**|-{1,3}|\u2014|\u2013")


def _numeric(column):
    """True if every cell is a number, a number written as text, or a dash."""
    if pd.api.types.is_numeric_dtype(column):
        return True
    cells = column.dropna().astype(str).str.strip()
    return bool(len(cells)) and all(_NUMBER.fullmatch(c) for c in cells)


def save_table(
    df,
    table_id,
    notes="",
    used_in="",
    float_format="%.3f",
    index=True,
    rule_before=None,
    column_format=None,
    escape=True,
):
    """Booktabs LaTeX tabular from a DataFrame (forecast metrics, test results).

    The caption lives in the paper. Notes, if given, go under the table at the table's own
    width (threeparttable; the paper loads the package). Negative numbers get a math minus.
    rule_before: index label of a row (e.g. "Total"), or a list of them, that gets a \\midrule above it.
    column_format: LaTeX column spec; by default the index is left-aligned and every column
    that holds numbers (also when stored as strings, or "---" for empty) is right-aligned.
    The index name is printed in the header row, not on a row of its own.
    escape: False passes LaTeX in cells (e.g. \\textbf) through; labels must then be LaTeX-safe.
    No regression tables are planned, so this replaces the template's pyfixest version.
    """
    out = OUT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if index and df.index.name and df.columns.name is None and df.columns.nlevels == 1:
        df.columns.name, df.index.name = df.index.name, None
    if column_format is None:
        column_format = "l" * (df.index.nlevels if index else 0) + "".join(
            "r" if _numeric(df.iloc[:, k]) else "l" for k in range(df.shape[1])
        )
    tex = df.to_latex(
        index=index, float_format=float_format, escape=escape, column_format=column_format
    )
    tex = re.sub(r"(?<![\w.\-])-(?=\d)", "$-$", tex)
    labels = [rule_before] if isinstance(rule_before, str) else rule_before or []
    for label in labels:
        tex = tex.replace(f"\n{label} &", f"\n\\midrule\n{label} &")
    if notes:
        tex = (
            "\\begin{threeparttable}\n"
            + tex
            + "\\begin{tablenotes}[flushleft]\\footnotesize\n"
            + f"\\item \\textit{{Notes:}} {notes}\n"
            + "\\end{tablenotes}\n\\end{threeparttable}\n"
        )
    f = out / f"{table_id}.tex"
    f.write_text(tex)
    record_output(table_id, [str(f)], used_in)
    return [str(f)]


# ---- Numbers quoted in the text ----------------------------------------------
def record_number(name, value):
    r"""Write \newcommand{\<name>}{<value>} to outputs/numbers.tex, replacing an old one.

    value is the text the paper prints, already formatted (e.g. f"{rmse:.2f}").
    LaTeX macro names are letters only, in CamelCase.
    """
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"macro name {name!r} must be letters only (no digits or underscores)")
    path = OUT / "numbers.tex"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [ln for ln in lines if not ln.startswith(rf"\newcommand{{\{name}}}")]
    lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---- Manifest ----------------------------------------------------------------
def record_output(out_id, files, used_in=""):
    """One row per output id in outputs/README.md: id · files · script · used in."""
    path = OUT / "README.md"
    header = ["# Outputs", "", "| id | files | script | used in |", "|---|---|---|---|"]
    rows = path.read_text().splitlines() if path.exists() else header
    rows = [r for r in rows if not r.startswith(f"| {out_id} |")]
    rows.append(f"| {out_id} | {', '.join(files)} | {_caller()} | {used_in} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def _caller():
    # sys.argv[0] is the script path for both `python file.py` and `python -m src.x`
    if not sys.argv or not sys.argv[0]:
        return "interactive"
    script = Path(sys.argv[0]).resolve()
    try:
        return str(script.relative_to(Path.cwd()))
    except ValueError:
        return script.name
