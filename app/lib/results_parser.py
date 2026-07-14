"""Parse SearchLibrium's sa_runs/ output and metacountregressor's JSON result
files for the GUI's results dashboard.

Formats verified directly against source (SearchLibrium/siman.py's
open_files/log_solution/log_decision/log_archive, and metacountregressor's
ExperimentBuilder.run()/SearchOutputConfig JSON writer) and against real
files produced by real runs this session — not assumed from docs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# SearchLibrium sa_runs/sa_<id>_<timestamp>/ directories
# ─────────────────────────────────────────────────────────────────────────────

SA_RUN_DIR_RE = re.compile(r"^sa_\d+_\d{8}_\d{6}$")


def discover_sa_runs(base_dir: str) -> list[Path]:
    """Find sa_<id>_<timestamp> run directories under base_dir.

    If base_dir itself matches the sa_<id>_<timestamp> naming (or simply
    contains the expected files), treat it as a single run directory.
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    if (base / "progress.csv").exists() or (base / "best.txt").exists():
        return [base]
    runs = [p for p in base.iterdir() if p.is_dir() and SA_RUN_DIR_RE.match(p.name)]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def parse_progress_csv(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "progress.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def parse_best_txt(run_dir: Path) -> dict[str, str]:
    """Parse best.txt's `key   =  value` lines (SA solver's log_decision())."""
    path = run_dir / "best.txt"
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


@dataclass
class SolutionBlock:
    title: str
    objectives: dict[str, float] = field(default_factory=dict)
    specification: dict[str, str] = field(default_factory=dict)
    model_statistics: str = ""


_BANNER_RE = re.compile(r"^═{10,}$")
_OBJ_LINE_RE = re.compile(r"^\s*\[\d+\]\s*\((Maximise|Minimise)\)\s*(\S+)\s*=\s*([\-\d.eE]+)")
_SPEC_LINE_RE = re.compile(r"^(\S[\w ]*?)\s*=\s*(.*)$")


def parse_solution_blocks(text: str) -> list[SolutionBlock]:
    """Parse the repeated banner-delimited blocks written by SA's
    log_solution() (used for both results.txt's Initial/Final Solution
    sections and archive.txt's Non-Dominated-#N sections — same function,
    same format, confirmed directly against siman.py).

    Block shape:
        ════...
          <title>
        ════...
        Objectives:
          [0] (Minimise) bic = 1234.5
        <blank>
        Model Statistics:
        <free text>
        <blank>
        Specification:
        asvars   =  [...]
        ...
    """
    lines = text.splitlines()
    banner_idx = [i for i, ln in enumerate(lines) if _BANNER_RE.match(ln.strip())]
    blocks: list[SolutionBlock] = []
    i = 0
    while i < len(banner_idx) - 1:
        start, mid = banner_idx[i], banner_idx[i + 1]
        if mid != start + 2:
            i += 1
            continue
        title = lines[start + 1].strip()
        # section ends at the next banner pair's opening banner, or EOF
        end = banner_idx[i + 2] if i + 2 < len(banner_idx) else len(lines)
        body = lines[mid + 1:end]

        objectives: dict[str, float] = {}
        spec: dict[str, str] = {}
        stats_lines: list[str] = []
        section = None
        for ln in body:
            stripped = ln.strip()
            if stripped == "Objectives:":
                section = "obj"
                continue
            if stripped == "Model Statistics:":
                section = "stats"
                continue
            if stripped == "Specification:":
                section = "spec"
                continue
            if section == "obj":
                m = _OBJ_LINE_RE.match(ln)
                if m:
                    objectives[m.group(2)] = float(m.group(3))
                elif stripped:
                    section = None
            elif section == "stats":
                stats_lines.append(ln)
            elif section == "spec":
                # finalise() appends "#Converged=.../#Accepted=..." run-level
                # counters straight to results.txt after the last block, with
                # no section header to close Specification: first -- exclude.
                if stripped.startswith("#"):
                    continue
                m = _SPEC_LINE_RE.match(stripped)
                if m:
                    spec[m.group(1).strip()] = m.group(2).strip()

        blocks.append(SolutionBlock(
            title=title, objectives=objectives, specification=spec,
            model_statistics="\n".join(stats_lines).strip("\n"),
        ))
        i += 2
    return blocks


def read_results_txt(run_dir: Path) -> str:
    path = run_dir / "results.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_archive_txt(run_dir: Path) -> str:
    path = run_dir / "archive.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def run_header(run_dir: Path) -> dict[str, str]:
    """First few lines of results.txt (Run ID/Started/Criterions/Models/temp schedule)."""
    text = read_results_txt(run_dir)
    header: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("─" * 5) or _BANNER_RE.match(line.strip()):
            break
        if ":" in line:
            key, _, value = line.partition(":")
            header[key.strip()] = value.strip()
        elif line.strip():
            header.setdefault("_line", line.strip())
    return header


# ─────────────────────────────────────────────────────────────────────────────
# metacountregressor JSON result files (SearchOutputConfig(save_json=True))
# ─────────────────────────────────────────────────────────────────────────────

def discover_metacount_json(base_dir: str) -> list[Path]:
    base = Path(base_dir)
    if not base.exists():
        return []
    if base.is_file() and base.suffix == ".json":
        return [base]
    return sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def parse_metacount_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
