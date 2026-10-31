"""Print a Pine source with presentation noise stripped, for porting work.

Keeps every line that participates in a calculation (inputs, assignments, control
flow, function definitions) and drops the drawing layer (`plot*`, `fill`, `line.*`,
`label.*`, `box.*`, `table.*`, `barcolor`, `bgcolor`, `alert*`, `hline`). Comments and
blank lines are removed, and each surviving line keeps its original number so a port
can cite the upstream location.

Usage:
    python scripts/pine_digest.py <path-to.pine> [--keep-drawing]
"""
from __future__ import annotations

import argparse
import pathlib
import re

DRAWING = re.compile(
    r"^\s*(plot\w*|fill|bgcolor|barcolor|hline|alert\w*|line\.|label\.|box\.|table\.|"
    r"polyline\.|array\.(?:new_line|new_label|new_box|push\s*\(\s*\w*lines))")
KEEP = re.compile(r"^\s*(var\s|varip\s|if\s|else|for\s|while\s|switch\s|=>|\w+\s*=)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--keep-drawing", action="store_true")
    parser.add_argument("--max-lines", type=int, default=0)
    options = parser.parse_args()

    path = pathlib.Path(options.path)
    source = path.read_text(encoding="utf-8-sig", errors="replace")
    in_block = False
    kept = 0
    for number, raw in enumerate(source.splitlines(), start=1):
        line = raw
        if "/*" in line:
            in_block = True
        if in_block:
            if "*/" in line:
                in_block = False
            continue
        if "//" in line:
            line = line.split("//", 1)[0]
        if not line.strip():
            continue
        if not options.keep_drawing and DRAWING.match(line):
            continue
        print(f"{number:5d} {line.rstrip()}")
        kept += 1
        if options.max_lines and kept >= options.max_lines:
            print("... (truncated)")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
