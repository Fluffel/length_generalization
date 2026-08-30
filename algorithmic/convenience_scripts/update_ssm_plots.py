#!/usr/bin/env python3
"""Regenerate the SSM plots and summarise them in two HTML contact sheets.

Runs ``run_formal_lang_ssm_plot.sh`` and ``run_tasks_ssm_plot.sh`` (each writes
one SVG per task into its own folder), then writes one HTML page per group that
lays those SVGs out in a grid of ``--columns`` columns.

A panel gets a green border when the architecture given via ``--arch`` has at
least one run whose accuracy exceeds ``--threshold`` in *every* validation bin;
the qualifying models are listed under the plot.  Task lists, plot paths, the
summary CSV and the bin count are read out of the shell scripts, so the HTML
always describes exactly what was plotted.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from plot_utils import load_summary_dataframe

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

_TASKS_ARRAY_RE = re.compile(r"tasks=\(\s*(.*?)\)", re.DOTALL)
_TOKEN_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'|(\S+)")
_TASK_PLACEHOLDERS = ("${TASK}", "$TASK")


@dataclass(frozen=True)
class PlotScript:
    """A ``run_*_plot.sh`` script and everything it tells us about its plots."""

    name: str          # short id used in the HTML filename
    title: str         # heading shown on the page
    path: Path
    tasks: list[str]
    plot_template: str  # path with ${TASK} placeholder, relative to the repo root
    csv_path: Path
    num_bins: int | None

    def plot_path(self, task: str) -> Path:
        rel = self.plot_template
        for placeholder in _TASK_PLACEHOLDERS:
            rel = rel.replace(placeholder, task)
        return REPO_ROOT / rel


@dataclass(frozen=True)
class Panel:
    """One task's plot plus the architecture's qualifying runs for it."""

    task: str
    plot_path: Path
    models: list[tuple[str, float]]

    @property
    def highlighted(self) -> bool:
        return bool(self.models)


def _flag_value(text: str, flag: str) -> str | None:
    """Value of ``--flag value`` / ``--flag=value`` in a shell script."""
    m = re.search(rf"{re.escape(flag)}[=\s]+(?:\"([^\"]*)\"|'([^']*)'|(\S+))", text)
    if not m:
        return None
    return next(g for g in m.groups() if g is not None)


def _parse_task_array(text: str, script: Path) -> list[str]:
    m = _TASKS_ARRAY_RE.search(text)
    if not m:
        raise SystemExit(f"{script}: could not find a tasks=( ... ) array.")
    tasks: list[str] = []
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0]
        for quoted_d, quoted_s, bare in _TOKEN_RE.findall(line):
            token = quoted_d or quoted_s or bare
            if token:
                tasks.append(token)
    if not tasks:
        raise SystemExit(f"{script}: tasks=( ... ) array is empty.")
    return tasks


def parse_plot_script(path: Path, *, name: str, title: str) -> PlotScript:
    if not path.exists():
        raise SystemExit(f"Plot script not found: {path}")
    text = path.read_text()

    template = _flag_value(text, "--output")
    if not template:
        raise SystemExit(f"{path}: could not find an --output path.")
    if not any(p in template for p in _TASK_PLACEHOLDERS):
        raise SystemExit(
            f"{path}: --output {template!r} has no ${{TASK}} placeholder, so the "
            "script does not write one plot per task."
        )

    csv_value = _flag_value(text, "--csv") or "exports/summary.csv"
    num_bins_value = _flag_value(text, "--num-bins")

    return PlotScript(
        name=name,
        title=title,
        path=path,
        tasks=_parse_task_array(text, path),
        plot_template=template,
        csv_path=(REPO_ROOT / csv_value).resolve(),
        num_bins=int(num_bins_value) if num_bins_value and num_bins_value.isdigit() else None,
    )


def run_plot_script(script: PlotScript) -> int:
    """Run the shell script from the repo root with our own interpreter first on PATH."""
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    print(f"Running {script.path.relative_to(REPO_ROOT)} ...")
    proc = subprocess.run(["bash", str(script.path)], cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        print(f"  {script.path.name} exited with code {proc.returncode}.")
    return proc.returncode


def qualifying_runs(
    df,
    *,
    task: str,
    arch: str,
    threshold: float,
    num_bins: int | None,
) -> list[tuple[str, float]]:
    """Models of *arch* that stay above *threshold* in every bin, worst bin first.

    A run qualifies on its own: the minimum over its bins must exceed the
    threshold, so a model that is strong on short inputs only never counts.
    """
    sub = df[(df["task"].astype(str) == task) & (df["arch"].astype(str) == arch)]
    if num_bins is not None and "num_bins" in sub.columns:
        sub = sub[sub["num_bins"].astype(int) == num_bins]
    if sub.empty:
        return []

    best: dict[str, float] = {}
    for _run_id, run_rows in sub.groupby("run_id", sort=False):
        worst_bin = float(run_rows["accuracy"].min())
        if worst_bin <= threshold:
            continue
        model = str(run_rows["model"].iloc[0])
        best[model] = max(best.get(model, 0.0), worst_bin)
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


def build_panels(
    df,
    script: PlotScript,
    *,
    arch: str,
    threshold: float,
    num_bins: int | None,
) -> list[Panel]:
    return [
        Panel(
            task=task,
            plot_path=script.plot_path(task),
            models=qualifying_runs(
                df, task=task, arch=arch, threshold=threshold, num_bins=num_bins
            ),
        )
        for task in script.tasks
    ]


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; line-height: 1.6; }}
  .legend-dot {{ display: inline-block; width: 0.8rem; height: 0.8rem;
                 border: 3px solid #2e7d32; border-radius: 3px;
                 vertical-align: middle; }}
  .grid {{ display: grid; grid-template-columns: repeat({columns}, minmax(0, 1fr));
           gap: 1.25rem; }}
  .panel {{ border: 3px solid #d8d8d8; border-radius: 8px; padding: 0.75rem;
            background: #fff; }}
  .panel.pass {{ border-color: #2e7d32; background: #f3fbf4; }}
  .panel h2 {{ font-size: 1rem; margin: 0 0 0.5rem; }}
  .panel img {{ width: 100%; height: auto; display: block; }}
  .models {{ font-size: 0.8rem; color: #2e7d32; margin-top: 0.5rem;
             word-break: break-all; line-height: 1.5; }}
  .missing {{ font-size: 0.9rem; color: #b71c1c; padding: 2rem 0; text-align: center; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">
  {meta}
</div>
<div class="grid">
{panels}
</div>
</body>
</html>
"""


MAX_LISTED_MODELS = 5


def _render_panel(panel: Panel, html_dir: Path) -> str:
    task = html.escape(panel.task)
    parts = [f"<h2>{task}</h2>"]
    if panel.plot_path.exists():
        src = html.escape(os.path.relpath(panel.plot_path, html_dir))
        parts.append(f'<img src="{src}" alt="{task}">')
    else:
        parts.append('<div class="missing">no plot generated</div>')
    if panel.models:
        shown = panel.models[:MAX_LISTED_MODELS]
        listed = ", ".join(
            f"{html.escape(model)} ({worst * 100:.1f}%)" for model, worst in shown
        )
        if len(panel.models) > len(shown):
            listed += f", +{len(panel.models) - len(shown)} more"
        parts.append(f'<div class="models">{listed}</div>')
    classes = "panel pass" if panel.highlighted else "panel"
    body = "\n    ".join(parts)
    return f'  <div class="{classes}">\n    {body}\n  </div>'


def render_page(
    script: PlotScript,
    panels: list[Panel],
    *,
    arch: str,
    threshold: float,
    columns: int,
    num_bins: int | None,
    html_dir: Path,
) -> str:
    passed = sum(1 for p in panels if p.highlighted)
    missing = sum(1 for p in panels if not p.plot_path.exists())
    bins_note = f"{num_bins} bins" if num_bins is not None else "all bin counts"
    meta_lines = [
        f"<span class='legend-dot'></span> green border: "
        f"<strong>{html.escape(arch)}</strong> has a run above "
        f"{threshold * 100:.1f}% in every validation bin "
        f"({passed} of {len(panels)} tasks).",
        f"Source: {html.escape(str(script.csv_path.relative_to(REPO_ROOT)))}, "
        f"{bins_note}, plots from {html.escape(script.path.name)}.",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.",
    ]
    if missing:
        meta_lines.append(f"{missing} task(s) produced no plot.")
    return _PAGE_TEMPLATE.format(
        title=html.escape(script.title),
        columns=columns,
        meta="<br>\n  ".join(meta_lines),
        panels="\n".join(_render_panel(p, html_dir) for p in panels),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the formal-language and task SSM plots, then write one "
            "HTML overview per group with the plots side by side. Plots where the "
            "given architecture exceeds the accuracy threshold in every "
            "validation bin get a green border."
        )
    )
    parser.add_argument(
        "--arch",
        required=True,
        help="Architecture to highlight, matched against the CSV arch column "
        "(e.g. ssm, olmossm, hyb, lm).",
    )
    parser.add_argument(
        "--columns",
        "--ncols",
        dest="columns",
        type=int,
        default=3,
        help="Number of plots per row in the HTML pages (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "exports",
        help="Directory for the HTML pages (default: exports/).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Minimum accuracy required in every bin, as a fraction or a "
        "percentage (default: 0.98).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Summary CSV used for the highlight check (default: the --csv value "
        "of each shell script).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=None,
        help="Bin count a run must have to be checked (default: the --num-bins "
        "value of each shell script).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only rebuild the HTML pages from the plots that are already on disk.",
    )
    args = parser.parse_args()

    if args.columns < 1:
        raise SystemExit("--columns must be >= 1.")
    threshold = args.threshold / 100.0 if args.threshold > 1 else args.threshold
    if not 0 <= threshold < 1:
        raise SystemExit("--threshold must be a fraction in [0, 1) or a percentage.")

    scripts = [
        parse_plot_script(
            SCRIPT_DIR / "run_formal_lang_ssm_plot.sh",
            name="formal",
            title="Formal languages — SSM plots",
        ),
        parse_plot_script(
            SCRIPT_DIR / "run_tasks_ssm_plot.sh",
            name="tasks",
            title="Algorithmic tasks — SSM plots",
        ),
    ]

    if not args.skip_plots:
        for script in scripts:
            run_plot_script(script)

    html_dir = args.output
    html_dir.mkdir(parents=True, exist_ok=True)

    for script in scripts:
        csv_path = args.csv or script.csv_path
        num_bins = args.num_bins if args.num_bins is not None else script.num_bins
        df = load_summary_dataframe(csv_path)
        if args.arch not in set(df["arch"].astype(str)):
            available = ", ".join(sorted(set(df["arch"].astype(str))))
            print(
                f"Warning: arch={args.arch!r} does not occur in {csv_path}; "
                f"nothing will be highlighted. Available: {available}"
            )

        panels = build_panels(
            df, script, arch=args.arch, threshold=threshold, num_bins=num_bins
        )
        page = render_page(
            script,
            panels,
            arch=args.arch,
            threshold=threshold,
            columns=args.columns,
            num_bins=num_bins,
            html_dir=html_dir,
        )
        out_path = html_dir / f"{script.name}_{args.arch}_plots.html"
        out_path.write_text(page)

        passed = [p.task for p in panels if p.highlighted]
        print(
            f"Wrote {out_path} ({len(panels)} plots, "
            f"{len(passed)} above {threshold * 100:.1f}% for {args.arch})"
        )
        if passed:
            print(f"  highlighted: {', '.join(passed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
