#!/usr/bin/env python3
"""Paired Wilcoxon signed-rank tests between models, from MissingBench per-subject results.

`missingbench test` writes, next to its text report, a `<report stem>_per_subject.csv` file with
one row per (subject, modality combination) and WT/TC/ET Dice and HD95 columns. This script
compares one or more "ours" models against a set of reference models on those files.

Unit of analysis: one subject, scores averaged over the 15 modality combinations.
  - If every subject is tested by exactly one fold (cross-validation test sets), subjects are
    pooled over the folds.
  - If the same subjects are tested by several fold models (an external cohort), each subject's
    scores are averaged over the fold models.
Only subjects present for both models of a pair are compared.

Tests: WT/TC/ET Dice (higher is better) and WT/TC/ET HD95 (lower is better), i.e. 6 tests per
reference model. p-values are Holm-corrected over all tests of one "ours" model. Verdict:
'+' Holm p < alpha and ours better on more subjects, '-' the reverse, '=' otherwise.

Example (results of each model in <results-root>/<model>_23/):

    python scripts/wilcoxon_tests.py \
        --results-root /path/to/results \
        --ours "MatUNet 11.4M=matunet_23_b11.4M" \
        --refs a2fseg_23 imfuse_23 mmformer_23 \
        --folds 1,2,3,4,5 \
        --pattern "results_fold{fold}_per_subject.csv" \
        --out wilcoxon_brats23

Writes <out>.csv and, unless --no-plot, <out>.png/.pdf.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["WT_dice", "TC_dice", "ET_dice", "WT_hd95", "TC_hd95", "ET_hd95"]
HIGHER_IS_BETTER = {m: m.endswith("dice") for m in METRICS}


def parse_model(spec: str) -> tuple[str, str]:
    """'Label=dir' -> (Label, dir); a bare 'dir' is its own label."""
    label, sep, directory = spec.partition("=")
    return (label, directory) if sep else (spec, spec)


def load_fold(model_dir: Path, pattern: str, fold: int) -> pd.DataFrame:
    path = model_dir / pattern.format(fold=fold)
    if not path.is_file():
        raise FileNotFoundError(f"missing per-subject results: {path}")
    df = pd.read_csv(path)
    missing = {"subject", "modalities", *METRICS} - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    df = df[["subject", "modalities", *METRICS]].replace([np.inf, -np.inf], np.nan)
    return df.groupby("subject")[METRICS].mean()  # average over modality combinations


def subject_table(model_dir: Path, pattern: str, folds: list[int]) -> pd.DataFrame:
    per_fold = [load_fold(model_dir, pattern, f) for f in folds]
    stacked = pd.concat(per_fold)
    if stacked.index.has_duplicates:  # same subjects tested by several fold models
        return stacked.groupby(level=0).mean()
    return stacked


def holm(p: np.ndarray) -> np.ndarray:
    order, k = np.argsort(p), len(p)
    out, running = np.empty(k), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[i]))
        out[i] = running
    return out


def compare(ours: dict[str, pd.DataFrame], refs: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    frames = []
    for label, o in ours.items():
        recs = []
        for ref_label, r in refs.items():
            for m in METRICS:
                j = pd.concat([o[m].rename("o"), r[m].rename("r")], axis=1).dropna()
                diff = (j.o - j.r) if HIGHER_IS_BETTER[m] else (j.r - j.o)  # > 0: ours better
                p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
                recs.append(dict(ours=label, reference=ref_label, metric=m, n=len(j),
                                 mean_ours=j.o.mean(), mean_reference=j.r.mean(),
                                 wins=int((diff > 0).sum()), losses=int((diff < 0).sum()), p=p))
        df = pd.DataFrame(recs)
        df["p_holm"] = holm(df.p.to_numpy())
        df["verdict"] = np.where(df.p_holm < alpha, np.where(df.wins > df.losses, "+", "-"), "=")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot(df: pd.DataFrame, title: str, out: Path, alpha: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    labels = list(dict.fromkeys(df.ours))
    ref_order = (df[df.metric.str.endswith("dice")].groupby("reference").mean_reference.mean()
                 .sort_values(ascending=False).index.tolist())
    cmap = ListedColormap(["#d9534f", "#e8e6df", "#3a9a5b"])  # -, =, +
    fig, axes = plt.subplots(1, len(labels), figsize=(3.5 * len(labels), 1.2 + 0.26 * len(ref_order)),
                             sharey=True, squeeze=False)
    for ax, label in zip(axes[0], labels):
        sub = df[df.ours == label].set_index(["reference", "metric"])
        val = np.array([[{"-": 0, "=": 1, "+": 2}[sub.loc[(r, m), "verdict"]] for m in METRICS]
                        for r in ref_order])
        ax.imshow(val, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        for i, r in enumerate(ref_order):
            for j, m in enumerate(METRICS):
                row = sub.loc[(r, m)]
                delta = (row.mean_ours - row.mean_reference) * (100 if m.endswith("dice") else 1)
                ax.text(j, i, f"{delta:+.1f}", ha="center", va="center", fontsize=6.3,
                        color="white" if val[i, j] != 1 else "#333333")
        ax.set_xticks(range(6))
        ax.set_xticklabels(["WT", "TC", "ET", "WT", "TC", "ET"])
        ax.xaxis.tick_top()
        ax.axvline(2.5, color="white", lw=3)
        ax.set_yticks(range(len(ref_order)))
        ax.set_yticklabels(ref_order)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cnt = sub.verdict.value_counts()
        ax.set_title(f"{label}\n+{cnt.get('+', 0)}  ={cnt.get('=', 0)}  −{cnt.get('-', 0)}\n"
                     "Dice (pts)            HD95 (mm)", fontsize=8, pad=18)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.text(0.5, 0.005, "Cell: ours − reference (mean over subjects). Green/red: ours significantly "
             f"better/worse (Wilcoxon signed-rank, Holm, p < {alpha}); grey: no significant difference.",
             ha="center", fontsize=6.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=220, facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Directory containing one results directory per model.")
    ap.add_argument("--ours", nargs="+", required=True,
                    help="Model(s) to test, as 'Label=dir' or 'dir' (relative to --results-root).")
    ap.add_argument("--refs", nargs="+", required=True,
                    help="Reference model(s), as 'Label=dir' or 'dir' (relative to --results-root).")
    ap.add_argument("--folds", default="1,2,3,4,5", help="Comma-separated folds (default: 1,2,3,4,5).")
    ap.add_argument("--pattern", default="results_fold{fold}_per_subject.csv",
                    help="Per-subject file name inside each model directory; '{fold}' is replaced.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level after Holm (default 0.05).")
    ap.add_argument("--title", default="Wilcoxon signed-rank tests")
    ap.add_argument("--out", type=Path, default=Path("wilcoxon"), help="Output path without extension.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    folds = [int(f) for f in args.folds.split(",")]
    load = lambda specs: {lbl: subject_table(args.results_root / d, args.pattern, folds)  # noqa: E731
                          for lbl, d in map(parse_model, specs)}
    df = compare(load(args.ours), load(args.refs), args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    for label, sub in df.groupby("ours", sort=False):
        print(label, {k: int(v) for k, v in sub.verdict.value_counts().items()}, f"n = {sub.n.min()}-{sub.n.max()}")
    if not args.no_plot:
        plot(df, args.title, args.out, args.alpha)
    print("wrote", args.out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
