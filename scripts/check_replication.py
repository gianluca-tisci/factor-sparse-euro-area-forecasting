"""Compare the rebuilt tables and numbers with the committed ones in outputs/.previous/.

Run from the repository root after replicate.sh:  uv run python -m scripts.check_replication
Tables (.tex) are compared after normalizing line endings; numbers files macro by macro, so a
different order of the same macros is not a difference. Figures, CSV files and audit files are
not compared: images depend on the platform's fonts and renderer, and audit files record the
local paths and line endings of the machine that ran them.
"""

import re
from pathlib import Path

NEW = Path("outputs")
OLD = NEW / ".previous"
MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")


def macros(path):
    return dict(MACRO.findall(path.read_text(encoding="utf-8")))


def lf(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def main():
    if not OLD.exists():
        raise FileNotFoundError("Run replicate.sh first: it moves the committed exhibits there")
    same, differ = 0, []
    for old in sorted(OLD.rglob("*.tex")):
        new = NEW / old.relative_to(OLD)
        if not new.exists():
            differ.append(f"not rebuilt: {new}")
            continue
        if old.name == "numbers.tex":
            a, b = macros(old), macros(new)
            bad = [m for m in a if m in b and a[m] != b[m]]
            only_old = [m for m in a if m not in b]
            if bad:
                differ.append(f"{new}: values differ for {', '.join(bad)}")
            if only_old:
                differ.append(f"{new}: not rewritten by the code: {', '.join(only_old)}")
            if not bad and not only_old:
                same += 1
            continue
        if lf(old) == lf(new):
            same += 1
        else:
            differ.append(f"{new}: text differs")
    print(f"{same} tables and numbers files identical to the committed ones.")
    for line in differ:
        print("  " + line)


if __name__ == "__main__":
    main()
