#!/usr/bin/env python3
"""All-folds version of generate_pareto_plot_paretoreg_fold1.py (GMACs, GPU
latency, peak GPU memory) and generate_pareto_plot_paretoreg_params_fold1.py
(parameters): the y axis is BraTS25-pre Dice averaged over EVERY fold that
has a test result (folds 1-5), not fold 1 only. METRIC picks the Dice: "ET"
(default) or "mean" (mean of WT, TC and ET); `--metric mean` on the command line.

- Baselines: the 5-fold mean; only models with all 5 folds are drawn.
- nasunif (nnUnif): b01-b16; any size without all 5 folds is left out rather
  than mixing a 1-fold point into a 5-fold plot.
- RDoll-UNet: the mean over whichever of its folds exist; while some are
  missing the legend says how many.

x values (cost) are the same measurements as the fold-1 plots.
Writes pareto_paretoreg_{et,mean}_dice_vs_{gmacs,latency,memory,params}_allfolds.png
into outputs/pareto_table/.

RDOLL_Y_OVERRIDE / LEGEND_OVERRIDE / OUT_DIR / OUT_TAG are hooks for callers
that want to draw externally supplied RDoll-UNet scores; unset, the script
uses only real test results.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_pareto_plot as gpp  # noqa: E402
import generate_pareto_plot_nasunif_fold1 as nas  # noqa: E402
import generate_pareto_plot_paretoreg_fold1 as base  # noqa: E402
import generate_pareto_plot_paretoreg_params_fold1 as pbase  # noqa: E402
import generate_summary_tables as gst  # noqa: E402

FOLDS = [f"fold{i}" for i in range(1, 6)]
NASUNIF_ALLFOLDS = [f"b{i:02d}" for i in range(1, 17)]
SOTA = {("MMMViT" if k == "MMMVIT" else k): v for k, v in pbase.SOTA.items()}  # display -> key
PARAMS_AXIS = ("params", 1e6, "Parameters (millions)", "params", None, None)

METRIC = "ET"  # "ET" or "mean" (mean of WT, TC, ET)
METRIC_REGIONS = {"ET": ["ET"], "mean": ["WT", "TC", "ET"]}
METRIC_LABEL = {"ET": ("ET Dice (%)", "et"), "mean": ("Mean Dice (WT, TC, ET) (%)", "mean")}
RDOLL_Y_OVERRIDE: dict[str, float] = {}  # label -> Dice (%) of METRIC to draw instead of the real mean
LEGEND_OVERRIDE: str | None = None
OUT_DIR: str | None = None
OUT_TAG = "allfolds"
FONT_SCALE = 1.0  # passed to generate_pareto_plot.FONT_SCALE
BOLD_TIES: dict[str, set[str]] = {}  # plot suffix (gmacs/latency/memory/params) -> extra bold labels


def folds_present(results_dir: str) -> list[str]:
    return [f for f in FOLDS if os.path.exists(os.path.join(base.RESULTS_DIR, results_dir, f"results_{f}.txt"))]


def metric_of(avg: dict[str, float]) -> float:
    regions = METRIC_REGIONS[METRIC]
    return sum(avg[r] for r in regions) / len(regions)


def y_allfolds(results_dir: str) -> float:
    """Mean over the available folds of each fold's METRIC Dice ('Avg scores' line)."""
    folds = folds_present(results_dir)
    vals = [metric_of(nas.read_avg(os.path.join(base.RESULTS_DIR, results_dir, f"results_{f}.txt"))) for f in folds]
    return 100 * sum(vals) / len(vals)


def x_value(key: str, col: str | None, results_dir: str, run_prefix: str) -> float | None:
    if key == "params":
        try:
            return float(pbase.checkpoint_params(run_prefix))
        except FileNotFoundError:
            return None
    return base.measured_cost(results_dir, col)


def main() -> None:
    out_dir = OUT_DIR or gpp.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    gst.REGIONS = METRIC_REGIONS[METRIC]
    y_label, y_tag = METRIC_LABEL[METRIC]
    pr, _ = base.paretoreg_costs()
    rdoll_n = min(len(folds_present(d)) for _, _, d in base.BUDGETS)
    gpp.LEGEND_LABEL[base.PR_FAMILY] = LEGEND_OVERRIDE or (
        "RDoll-UNet (tested subnets)" if rdoll_n == len(FOLDS) else f"RDoll-UNet (tested subnets; {rdoll_n} of 5 folds)"
    )
    all_styles = gpp.FAMILY_STYLE
    for key, scale, x_label, suffix, col, ticks in base.AXES + [PARAMS_AXIS]:
        points, families = {}, {}
        for name, k in SOTA.items():
            d = f"{k}_23"
            if len(folds_present(d)) != len(FOLDS):
                continue
            x = x_value(key, col, d, f"{k}23")
            if x is not None:
                points[name], families[name] = (x / scale, y_allfolds(d)), "Baseline"
        for tag in NASUNIF_ALLFOLDS:
            d = f"nasunif_{tag}_23"
            if len(folds_present(d)) != len(FOLDS):
                continue
            x = x_value(key, col, d, f"nasunif{tag}23")
            if x is not None:
                points[tag], families[tag] = (x / scale, y_allfolds(d)), nas.FAMILY
        for label, _, results_dir in base.BUDGETS:
            x = pbase.PARETOREG[label][1] if key == "params" else pr[label][col]
            y = RDOLL_Y_OVERRIDE.get(label, None)
            points[label] = (x / scale, y_allfolds(results_dir) if y is None else y)
            families[label] = base.PR_FAMILY
        gpp.FAMILY_STYLE = {f: s for f, s in all_styles.items() if f in set(families.values())}
        lab = x_label + (" · all on L40S" if key == "lat_gpu" else "")
        gpp.FONT_SCALE, gpp.EXTRA_BOLD = FONT_SCALE, BOLD_TIES.get(suffix, set())
        out = os.path.join(out_dir, f"pareto_paretoreg_{y_tag}_dice_vs_{suffix}_{OUT_TAG}.png")
        try:
            gpp.render_pareto_plot(
                points, families, x_label=lab, y_label=y_label, x_log=True,
                out_path=out, decorate=base.make_decorator(ticks),
            )
        finally:
            gpp.FAMILY_STYLE = all_styles
        print(f"{out}: {len(points)} points;", ", ".join(
            f"{label} x={points[label][0]:.4g} y={points[label][1]:.2f}" for label, _, _ in base.BUDGETS))


if __name__ == "__main__":
    if "--metric" in sys.argv:
        METRIC = sys.argv[sys.argv.index("--metric") + 1]
    main()
