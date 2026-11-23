#!/usr/bin/env python3
"""BraTS25-pre (= BraTS23), fold1-only Pareto plots with the three tested
paretoreg23 subnetworks (1.34M, 11.59M, 36.2M realized parameters, i.e. the
1.4M / 11.4M / 91.4M budgets) drawn as real points: x is their measured cost,
y their fold-1 test ET Dice.

Three plots, all with ET Dice on the y axis and a log x axis:
- whole-volume GMACs      -> pareto_paretoreg_et_dice_vs_gmacs_fold1.png
- whole-volume GPU latency -> pareto_paretoreg_et_dice_vs_latency_fold1.png
- whole-volume peak GPU memory -> pareto_paretoreg_et_dice_vs_memory_fold1.png

Points: the reproduced SOTA models (family "Baseline" of collect_points, plus
MambaVit-AKD, MST-KDNet and ShaSpec read straight from their flops csv -- M3AE
has none, `mimose flops` cannot run it on a 128^3 window) and the 16 nasunif
models, on all three plots. nasunif x values are MEASURED (flops_gpu.csv of each
nasunif_bXX_23, from flops_missing_models_l40s.sh), not the budget targets that
generate_pareto_plot_nasunif_fold1.py uses.

paretoreg23 cost comes from `mimose flops` on the FINAL checkpoint (the one the
tests used), timed on L40S like the baselines' csvs: flops_paretoreg_final_l40s/
(sbatch_files/paretoreg/flops/flops_missing_models_l40s.sh). Until that exists it
falls back to the epoch-721 L40S measurements (flops_mfi_vs_paretoreg_ep721_l40s/)
and says so in the legend: at epoch 721 the 1.4M-budget subnet already had its
final widths, the other two were within a few channels of them.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_pareto_plot as gpp  # noqa: E402
import generate_pareto_plot_nasunif_fold1 as nas  # noqa: E402
import generate_summary_tables as gst  # noqa: E402
import pareto_budgets as pbud  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

FOLDS = ["fold1"]
GROUPS = pbud.GROUPS
RESULTS_DIR = "/work/phd_mimose/results"
COST_DIR = os.path.join(RESULTS_DIR, "flops_paretoreg_final_l40s")
# (point label, budget passed to `mimose test` / `mimose flops`, results dir of its fold-1 test)
BUDGETS = [
    ("RDoll-UNet 1.34M", 1431336, "paretoreg_23_b1.4M"),
    ("RDoll-UNet 11.59M", 11435747, "paretoreg_23_b11.4M"),
    ("RDoll-UNet 36.2M", 91366608, "paretoreg_23"),
]
STALE_COST_DIR = os.path.join(RESULTS_DIR, "flops_mfi_vs_paretoreg_ep721_l40s")
STALE_BUDGET = {1431336: 1430000, 11435747: 11400000, 91366608: 91370000}
PR_FAMILY = "ParetoReg"

PR_COLOR = "#55A868"
gpp.FAMILY_STYLE[PR_FAMILY] = {"marker": "^", "color": PR_COLOR}
gpp.LEGEND_LABEL[PR_FAMILY] = "RDoll-UNet (tested subnets)"
GiB = 2**30

# (metric key in collect_all_rows, x scale, x label, file suffix, csv column, plain x ticks or None)
AXES = [
    ("flops", 2e9, "Whole-volume GMACs", "gmacs", "mean_flops", None),
    ("lat_gpu", 1.0, "Whole-volume GPU inference time (s)", "latency", "mean_latency_seconds", [0.5, 1, 2, 3, 4]),
    ("mem", GiB, "Whole-volume peak GPU memory (GiB)", "memory", "mean_peak_memory_bytes", [1, 1.5, 2, 3, 4, 5, 6]),
]


def paretoreg_costs() -> tuple[dict[str, dict[str, float]], bool]:
    """{point label: {csv column: value}} from the whole_volume rows, and whether
    they come from the final checkpoint (True) or the epoch-721 fallback."""
    final = all(os.path.exists(os.path.join(COST_DIR, f"paretoreg_b{b}_gpu.csv")) for _, b, _ in BUDGETS)
    out = {}
    for label, b, _ in BUDGETS:
        path = os.path.join(COST_DIR, f"paretoreg_b{b}_gpu.csv") if final else os.path.join(STALE_COST_DIR, f"paretoreg_b{STALE_BUDGET[b]}_gpu.csv")
        with open(path) as f:
            row = next(r for r in csv.DictReader(f) if r["scope"] == "whole_volume")
        out[label] = {a[4]: float(row[a[4]]) for a in AXES}
    return out, final


# Reproduced models that collect_points does not know about: display -> results dir.
EXTRA_SOTA = {"MambaVit-AKD": "mambavitakd_23", "MST-KDNet": "mstkdnet_23", "ShaSpec": "shaspec_23"}


def measured_cost(results_dir: str, col: str) -> float | None:
    """Whole-volume value of `col` from results_dir/flops_gpu.csv, or None."""
    path = os.path.join(RESULTS_DIR, results_dir, "flops_gpu.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        row = next(r for r in csv.DictReader(f) if r["scope"] == "whole_volume")
    return float(row[col])


def paretoreg_et(results_dir: str) -> float:
    return 100 * nas.read_avg(os.path.join(RESULTS_DIR, results_dir, "results_fold1.txt"))["ET"]


def make_decorator(ticks: list[float] | None):
    def decorate(ax):
        if ticks:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        return []

    return decorate


def main() -> None:
    os.makedirs(gpp.OUTPUT_DIR, exist_ok=True)
    gst.REGIONS = ["ET"]
    pr, final = paretoreg_costs()
    if not final:
        gpp.LEGEND_LABEL[PR_FAMILY] = "RDoll-UNet (tested subnets; cost from epoch-721 snapshot)"
        print("WARNING: final-checkpoint costs missing, using the epoch-721 measurements", file=sys.stderr)
    all_styles = gpp.FAMILY_STYLE
    for key, scale, x_label, suffix, col, ticks in AXES:
        points, families = gpp.collect_points(key, x_scale=scale, expected_folds=FOLDS, groups=GROUPS)
        keep = [n for n, fam in families.items() if fam == "Baseline"]
        points = {n: points[n] for n in keep}
        families = {n: "Baseline" for n in keep}
        for name, results_dir in EXTRA_SOTA.items():
            x = measured_cost(results_dir, col)
            if name not in points and x is not None:
                points[name], families[name] = (x / scale, paretoreg_et(results_dir)), "Baseline"
        for i in range(1, 17):
            tag = f"b{i:02d}"
            x = measured_cost(f"nasunif_{tag}_23", col)
            if x is not None:
                points[tag], families[tag] = (x / scale, paretoreg_et(f"nasunif_{tag}_23")), nas.FAMILY
        for label, _, results_dir in BUDGETS:
            points[label] = (pr[label][col] / scale, paretoreg_et(results_dir))
            families[label] = PR_FAMILY
        drawn = set(families.values())
        gpp.FAMILY_STYLE = {f: s for f, s in all_styles.items() if f in drawn}
        if key == "lat_gpu":
            x_label += " · all on L40S"
        out = os.path.join(gpp.OUTPUT_DIR, f"pareto_paretoreg_et_dice_vs_{suffix}_fold1.png")
        try:
            gpp.render_pareto_plot(
                points, families, x_label=x_label, y_label="ET Dice (%)", x_log=True,
                out_path=out, decorate=make_decorator(ticks),
            )
        finally:
            gpp.FAMILY_STYLE = all_styles
        print(f"  {suffix}:", ", ".join(f"{label} x={points[label][0]:.4g} ET={points[label][1]:.2f}" for label, _, _ in BUDGETS))


if __name__ == "__main__":
    main()
