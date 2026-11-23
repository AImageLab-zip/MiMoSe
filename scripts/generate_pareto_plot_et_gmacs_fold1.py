#!/usr/bin/env python3
"""BraTS25-pre (= BraTS23), fold1-only 2D Pareto plot: whole-volume GMACs vs.
Enhancing Tumor (ET) Dice.

Same data collection and rendering as generate_pareto_plot_flops_fold1.py, but
with two differences:

- Dice is ET only, not the WT/TC/ET mean (gst.REGIONS is narrowed to ["ET"]
  before collecting, which collect_dice_hd95 reads at call time).
- The x axis is GMACs (FLOPs / 2 / 1e9; src/mimose/flops.py counts
  flops = 2 * macs), so the b01..b16 FLOPs-budget sweep below, whose budgets
  are per-forward GMACs, can sit on the same axis once scaled to a whole volume
  (x WINDOWS_PER_VOLUME).

The b01..b16 points are hard-coded in EXTRA_POINTS (fold1, BraTS25-pre, ET;
b09..b16 reported to one decimal).

The x axis is log-scaled: the points span ~480 to ~65,000 GMACs.

Writes pareto_et_dice_vs_gmacs_fold1.png into generate_pareto_table.py's
OUTPUT_DIR.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_pareto_plot as gpp  # noqa: E402
import generate_summary_tables as gst  # noqa: E402
import pareto_budgets as pbud  # noqa: E402

FOLDS = ["fold1"]
GROUPS = pbud.GROUPS

# Raw FLOP counts -> GMACs.
GMACS = 2e9

EXTRA_FAMILY = "FlopsBudget"
# The b-budgets are MACs of ONE 128^3 forward (pareto_flops_23.yaml:
# budget_metric). ElasticUnet predicts whole volumes with SimpleUnet's 50%-overlap
# 128^3 sliding window, which every 128^3-patch model's flops_gpu.csv measures as
# exactly 12 forwards per volume (whole_volume / patch mean_macs).
WINDOWS_PER_VOLUME = 12
# {name: (per-forward GMACs, ET Dice %)}
EXTRA_POINTS = {
    "b01": (40.0, 70.78),
    "b02": (55.5, 71.62),
    "b03": (76.9, 71.50),
    "b04": (106.7, 71.55),
    "b05": (148.0, 72.38),
    "b06": (205.3, 71.93),
    "b07": (284.7, 72.71),
    # b08 has no result yet.
    "b09": (547.6, 72.9),
    "b10": (759.5, 73.4),
    "b11": (1053.3, 73.7),
    "b12": (1460.9, 74.0),
    "b13": (2026.2, 74.8),
    "b14": (2810.1, 74.5),
    "b15": (3897.4, 74.1),
    "b16": (5405.4, 74.2),
}

gpp.FAMILY_STYLE[EXTRA_FAMILY] = {"marker": "D", "color": "#C44E52"}
gpp.LEGEND_LABEL[EXTRA_FAMILY] = "FLOPs budget (b01-b16)"


def main() -> None:
    os.makedirs(gpp.OUTPUT_DIR, exist_ok=True)
    gst.REGIONS = ["ET"]
    points, families = gpp.collect_points("flops", x_scale=GMACS, expected_folds=FOLDS, groups=GROUPS)
    for name, (gmacs_per_forward, dice) in EXTRA_POINTS.items():
        points[name] = (gmacs_per_forward * WINDOWS_PER_VOLUME, dice)
        families[name] = EXTRA_FAMILY
    gpp.render_pareto_plot(
        points,
        families,
        x_label="Whole-volume GMACs",
        y_label="ET Dice (%)",
        x_log=True,
        out_path=os.path.join(gpp.OUTPUT_DIR, "pareto_et_dice_vs_gmacs_fold1.png"),
    )
    for name in sorted(points, key=lambda n: points[n][0]):
        print(f"{families[name]:>12}  {name:<24} {points[name][0]:9.1f} GMACs  {points[name][1]:6.2f}")


if __name__ == "__main__":
    main()
