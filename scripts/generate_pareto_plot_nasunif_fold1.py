#!/usr/bin/env python3
"""BraTS25-pre (= BraTS23), fold1-only Pareto plots: whole-volume GMACs vs. Dice,
for the nasunif FLOPs-budget sweep (b01..b16) against the reproduced SOTA models.

Only two families are drawn:
- "Reproduced": the literature models (family "Baseline" of collect_points).
  The mimosa_[size] sweep and the MiMoSe-P parameter budgets are left out.
- "nasunif": read from /work/phd_mimose/results/nasunif_bXX_23/results_fold1.txt
  (downloaded from the AImageLab-Zip/mimose_runs dataset repo).

nasunif x positions: the per-forward GMACs budget of each run (128^3 window),
times 12 windows per whole volume (SimpleUnet's 50%-overlap sliding window,
see generate_pareto_plot_et_gmacs_fold1.py). These are budget targets, not
measured FLOPs. b08's budget is not recorded anywhere; the grid is geometric
(x1.387 per step), so it is interpolated as sqrt(b07 * b09).

Writes two plots into generate_pareto_table.py's OUTPUT_DIR:
- pareto_nasunif_et_dice_vs_gmacs_fold1.png   (ET Dice)
- pareto_nasunif_mean_dice_vs_gmacs_fold1.png (mean of WT/TC/ET Dice)
"""

from __future__ import annotations

import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_pareto_plot as gpp  # noqa: E402
import generate_summary_tables as gst  # noqa: E402
import pareto_budgets as pbud  # noqa: E402

FOLDS = ["fold1"]
GROUPS = pbud.GROUPS
GMACS = 2e9  # raw FLOPs -> GMACs (flops = 2 * macs)
WINDOWS_PER_VOLUME = 12
RESULTS_DIR = "/work/phd_mimose/results"

# Per-forward (128^3) GMACs budget of each nasunif run.
BUDGET_GMACS = {
    "b01": 40.0, "b02": 55.5, "b03": 76.9, "b04": 106.7, "b05": 148.0,
    "b06": 205.3, "b07": 284.7, "b09": 547.6, "b10": 759.5, "b11": 1053.3,
    "b12": 1460.9, "b13": 2026.2, "b14": 2810.1, "b15": 3897.4, "b16": 5405.4,
}
BUDGET_GMACS["b08"] = math.sqrt(BUDGET_GMACS["b07"] * BUDGET_GMACS["b09"])  # interpolated

FAMILY = "NASUnif"
gpp.FAMILY_STYLE[FAMILY] = {"marker": "D", "color": "#C44E52"}
gpp.LEGEND_LABEL[FAMILY] = "nnUnif (b01-b16)"

AVG_RE = re.compile(r"^Avg scores\s+-->\s+(.*)$")


def read_avg(path: str) -> dict[str, float]:
    with open(path) as f:
        for line in f:
            m = AVG_RE.match(line.strip())
            if m:
                return {k.strip(): float(v) for k, v in (kv.split("=") for kv in m.group(1).split(","))}
    raise ValueError(f"no 'Avg scores' line in {path}")


def nasunif_points(regions: list[str]) -> dict[str, tuple[float, float]]:
    pts = {}
    for b, g in sorted(BUDGET_GMACS.items()):
        path = os.path.join(RESULTS_DIR, f"nasunif_{b}_23", "results_fold1.txt")
        if not os.path.exists(path):
            print(f"missing {path}", file=sys.stderr)
            continue
        avg = read_avg(path)
        pts[b] = (g * WINDOWS_PER_VOLUME, 100 * sum(avg[r] for r in regions) / len(regions))
    return pts


def make_plot(regions: list[str], y_label: str, out_name: str) -> None:
    gst.REGIONS = regions
    points, families = gpp.collect_points("flops", x_scale=GMACS, expected_folds=FOLDS, groups=GROUPS)
    keep = {n for n, fam in families.items() if fam == "Baseline"}
    points = {n: p for n, p in points.items() if n in keep}
    families = {n: families[n] for n in keep}
    for name, p in nasunif_points(regions).items():
        points[name] = p
        families[name] = FAMILY
    out = os.path.join(gpp.OUTPUT_DIR, out_name)
    # The legend lists every family in FAMILY_STYLE; keep only the two drawn here.
    all_styles = gpp.FAMILY_STYLE
    gpp.FAMILY_STYLE = {f: s for f, s in all_styles.items() if f in ("Baseline", FAMILY)}
    try:
        gpp.render_pareto_plot(points, families, x_label="Whole-volume GMACs", y_label=y_label, x_log=True, out_path=out)
    finally:
        gpp.FAMILY_STYLE = all_styles
    print(f"\n{out}")
    for name in sorted(points, key=lambda n: points[n][0]):
        print(f"{families[name]:>10}  {name:<14} {points[name][0]:9.1f} GMACs  {points[name][1]:6.2f}")


def main() -> None:
    os.makedirs(gpp.OUTPUT_DIR, exist_ok=True)
    make_plot(["ET"], "ET Dice (%)", "pareto_nasunif_et_dice_vs_gmacs_fold1.png")
    make_plot(["WT", "TC", "ET"], "Average Dice (%)", "pareto_nasunif_mean_dice_vs_gmacs_fold1.png")


if __name__ == "__main__":
    main()
