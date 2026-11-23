#!/usr/bin/env python3
"""BraTS25-pre (= BraTS23), fold1-only Pareto plot: number of parameters vs. ET Dice.

Companion of generate_pareto_plot_paretoreg_fold1.py (GMACs / latency / memory),
with the parameter count on the (log) x axis. Three families:

- "Reproduced": the 21 SOTA models listed in SOTA below.
- "nasunif": the 16 uniformly-scaled U-Nets b01..b16.
- "ParetoReg23": the three tested subnetworks of the paretoreg23 supernet, drawn
  as real points (they have fold-1 test scores now), at their REALIZED size
  (1.34M / 11.59M / 36.2M), not at the nominal 1.4M / 11.4M / 91.4M budgets.

y: fold-1 "Avg scores" ET Dice of results_fold1.txt (mean over the 15 modality
subsets), the same number the other fold-1 plots use.

x: for the reproduced models and nasunif, the number of values stored in the
fold-1 final_weights_only.safetensors. For the 17 reproduced models that have a
`mimose flops` report this equals its params column exactly; M3AE, MambaVit-AKD,
MST-KDNet and ShaSpec have no such report, so for them it counts everything in
the checkpoint (which may include training-only parts such as a teacher).
ParetoReg23's x is the parameter count of the extracted subnetwork.

Writes pareto_paretoreg_et_dice_vs_params_fold1.png into generate_pareto_table.py's
OUTPUT_DIR.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_pareto_plot as gpp  # noqa: E402
import generate_pareto_plot_nasunif_fold1 as nas  # noqa: E402
from safetensors import safe_open  # noqa: E402

RESULTS_DIR = "/work/phd_mimose/results"
RUNS_DIR = "/work/phd_mimose/runs"

# display name -> (results dir, run dir prefix)
SOTA = {
    "A2FSeg": "a2fseg", "DC-Seg": "dcseg", "IM-Fuse": "imfuse", "IMS2Trans": "ims2trans",
    "InOutFusion": "inoutfusion", "LCKD": "lckd", "M2FTrans": "m2ftrans", "M3AE": "m3ae",
    "M3FeCon": "m3fecon", "MambaVit-AKD": "mambavitakd", "mmFormer": "mmformer", "MMMVIT": "mmmvit",
    "MST-KDNet": "mstkdnet", "RFL": "rfl", "RFNet": "rfnet", "RobustSeg": "robustseg",
    "SFusion": "sfusion", "ShaSpec": "shaspec", "SRMNet": "srmnet", "U-HVED": "uhved",
    "UNET-MFI": "unetmfi",
}
# label -> (results dir, realized parameter count of the extracted subnetwork)
PARETOREG = {
    "RDoll-UNet 1.34M": ("paretoreg_23_b1.4M", 1337476),
    "RDoll-UNet 11.59M": ("paretoreg_23_b11.4M", 11587736),
    "RDoll-UNet 36.2M": ("paretoreg_23", 36216462),
}
PR_FAMILY = "ParetoReg"
gpp.FAMILY_STYLE[PR_FAMILY] = {"marker": "^", "color": "#55A868"}
gpp.LEGEND_LABEL[PR_FAMILY] = "RDoll-UNet (tested subnets)"


def checkpoint_params(run: str) -> int:
    path = os.path.join(RUNS_DIR, f"{run}_fold1", "checkpoints", "final_weights_only.safetensors")
    with safe_open(path, "pt") as f:
        return sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())


def et_dice(results_dir: str) -> float:
    return 100 * nas.read_avg(os.path.join(RESULTS_DIR, results_dir, "results_fold1.txt"))["ET"]


def main() -> None:
    os.makedirs(gpp.OUTPUT_DIR, exist_ok=True)
    points, families = {}, {}
    for name, key in SOTA.items():
        points[name] = (checkpoint_params(f"{key}23") / 1e6, et_dice(f"{key}_23"))
        families[name] = "Baseline"
    for b in range(1, 17):
        tag = f"b{b:02d}"
        points[tag] = (checkpoint_params(f"nasunif{tag}23") / 1e6, et_dice(f"nasunif_{tag}_23"))
        families[tag] = nas.FAMILY
    for name, (results_dir, params) in PARETOREG.items():
        points[name] = (params / 1e6, et_dice(results_dir))
        families[name] = PR_FAMILY

    out = os.path.join(gpp.OUTPUT_DIR, "pareto_paretoreg_et_dice_vs_params_fold1.png")
    all_styles = gpp.FAMILY_STYLE
    gpp.FAMILY_STYLE = {f: s for f, s in all_styles.items() if f in set(families.values())}
    try:
        gpp.render_pareto_plot(
            points, families, x_label="Parameters (millions)", y_label="ET Dice (%)", x_log=True, out_path=out
        )
    finally:
        gpp.FAMILY_STYLE = all_styles
    print(out)
    for name in sorted(points, key=lambda n: points[n][0]):
        print(f"  {families[name]:10s} {name:22s} {points[name][0]:9.3f}M  ET {points[name][1]:.2f}")
    counts = {f: sum(1 for v in families.values() if v == f) for f in set(families.values())}
    print("points per family:", counts)


if __name__ == "__main__":
    main()
