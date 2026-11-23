#!/usr/bin/env python3
"""Build class-specific (per-region) summary tables: one table per tumor
region (WT, TC, ET), each with one row per model and four column-groups:
  - BraTS18     -- official BraTS2018 test split (results_fold*.xlsx)
  - BraTS25-pre -- official BraTS25-pre test split (the `_23` results dir;
    display label only -- the on-disk dsnum/config suffix stays "23")
  - MB-96 - BraTS18 chp -- internal cohort, scored with the BraTS18-trained
    checkpoint (results_internal_fold*.xlsx under the `_18` results dir)
  - MB-96 - BraTS25-pre chp -- internal cohort, scored with the
    BraTS25-pre-trained checkpoint (both internal evaluations are kept side
    by side rather than picked/merged, since they come from different
    checkpoints)

Each group shows: Rank (by Dice, among models with data for that group) |
Dice mean+-std | HD95 mean+-std, aggregated over ALL modality-presence
combinations and folds.

A model only needs COMPLETE data (all of EXPECTED_FOLDS present, all 15
modality-presence combinations, every WT/TC/ET Dice+HD95 populated) for a
given group to have that group populated; groups it lacks show "--". A model
is included in a table at all if it has at least one complete group.

Per model/region/group, the reported value is:
  mean over the 15 modality combinations of (mean over folds), "+-"
  mean over the 15 modality combinations of (std over folds)
i.e. the column-wise mean of the "Mean" and "Std" columns that
build_reproduction_comparison.py computes per modality row.

Writes one `.tex` file with all three tables plus a standalone `main.tex`
wrapper into OUTPUT_DIR, then compiles it to PDF with `tectonic` (if available).
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict

import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_reproduction_comparison as brc  # noqa: E402  (reuse parse_legacy + constants)

RESULTS_DIR = brc.RESULTS_DIR
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "summary_tables"
)
LEGACY_FILE = os.path.join(RESULTS_DIR, brc.LEGACY_FILE)

REGIONS = brc.REGIONS  # ["WT", "TC", "ET"]
REGION_LABELS = {"WT": "Whole Tumor", "TC": "Tumor Core", "ET": "Enhancing Tumor"}
METRICS = ["Dice", "HD95"]
EXPECTED_FOLDS = brc.EXPECTED_FOLDS  # ["fold1", "fold3", "fold5"]
# This script's official BraTS groups require all five folds; the internal
# MB-96 groups only ever ran folds 1/3/5 and keep EXPECTED_FOLDS. The sibling
# scripts that import EXPECTED_FOLDS are unaffected.
OFFICIAL_EXPECTED_FOLDS = brc.OFFICIAL_EXPECTED_FOLDS
MODALITY_KEY_ORDER = brc.MODALITY_KEY_ORDER  # 15 combinations
DATASET_LABELS = {"18": "BraTS 2018", "23": "BraTS 2025-pre", "25": "BraTS 2025"}
# mimosa_[size] models (mimosa_base, mimosa_large, ...) are excluded here and
# handled instead by generate_mimosa_size_tables.py.
MIMOSA_SIZE_MODELS = {
    brc.normalize(n)
    for n in (
        "mimosa_base",
        "mimosa_gargantuan",
        "mimosa_huge",
        "mimosa_large",
        "mimosa_medium",
        "mimosa_micro",
        "mimosa_small",
        "mimosa_tiny",
    )
}
EXCLUDED_MODELS = {
    brc.normalize(n)
    for n in (
        "tinymimosaweighted",
        "manymimosas",
        "tinymimosa",
        "olduhved",
        # Results on disk are from collapsed runs (MCPL v2/v3 nested-label bug;
        # NAP-SCAF's train/test protocol mismatch). Drop mcpl once v4 is tested.
        "mcpl",
        "napscaf",
    )
} | MIMOSA_SIZE_MODELS
# Fallback display names for models not present in the legacy display-name
# file (kept consistent with generate_flops_table.py's DISPLAY_NAMES).
FALLBACK_DISPLAY_NAMES = {"tinymimosa": "TinyMimosa", "manymimosas": "ManyMimosas"}

# Optional bibkey per model_norm, rendered as "Name~\cite{key}" in the Model
# column when present. Models without an entry just show their display name.
CITATION_KEYS: dict[str, str] = {
    "a2fseg": "Wang2023A",
    "dcseg": "Li2025B",
    "imfuse": "Pipoli2025",
    "ims2trans": "Zhang2024",
    "inoutfusion": "Liu2025",
    "lckd": "Wang2023B",
    "m2ftrans": "Shi2023",
    "m3fecon": "Zeng2024",
    "mifpn": "Diao2025",
    "mmformer": "Zhang2022",
    "mmmvit": "Qiu2024",
    "rfl": "Fan2025",
    "rfnet": "Ding2021",
    "robustseg": "Chen2019",
    "sfusion": "Liu2023",
    "srmnet": "Li2024",
    "uhved": "Dorent2019",
    "unetmfi": "Zhao2022",
}

# (column-group label, source ("official"/"internal"), dsnum). "official"
# reads from `models` (results_fold*.xlsx), "internal" from `internal_models`
# (results_internal_fold*.xlsx, scored with that dsnum's trained checkpoint).
GROUPS = [
    ("BraTS18", "official", "18"),
    ("BraTS25-pre", "official", "23"),
    ("MB-96 - BraTS18 chp", "internal", "18"),
    ("MB-96 - BraTS25-pre chp", "internal", "23"),
]

# Group labels that have a legacy (published) baseline to compare against, and
# hence get the extra "Avg Error" column; the internal MB-96 groups have no
# legacy counterpart, so error is only ever shown for these two.
OFFICIAL_GROUP_LABELS = {lbl for lbl, source, _dsnum in GROUPS if source == "official"}


def dataset_label(dsnum: str) -> str:
    return DATASET_LABELS.get(dsnum, f"Dataset {dsnum}")


# --------------------------------------------------------------------------- #
# Discovery: <model>_<dsnum> directories, official + internal fold files.
# --------------------------------------------------------------------------- #
def discover(results_dir):
    """Return (models, internal_models, dsnums).

    models / internal_models: {model_norm: {dsnum: {fold_label: parsed_dict}}}
    parsed_dict: {mkey: {"WT": v, "WT_hd95": v, "TC": v, ...}} or {} on parse failure.
    """
    models = defaultdict(dict)
    internal_models = defaultdict(dict)
    dsnums = set()

    entries = [
        e for e in sorted(os.listdir(results_dir))
        if os.path.isdir(os.path.join(results_dir, e))
    ]
    for entry in entries:
        m = re.match(r"^(.+)_(\d{2})$", entry)
        if not m:
            continue
        model_raw, dsnum = m.group(1), m.group(2)
        full = os.path.join(results_dir, entry)

        fold_files = sorted(
            glob.glob(os.path.join(full, "results_fold*.xlsx")),
            key=lambda p: int(re.search(r"fold(\d+)", p).group(1)),
        )
        internal_fold_files = sorted(
            glob.glob(os.path.join(full, "results_internal_fold*.xlsx")),
            key=lambda p: int(re.search(r"fold(\d+)", p).group(1)),
        )
        brc.drop_stale_v1(results_dir, model_raw, dsnum, fold_files, internal_fold_files)
        if not fold_files and not internal_fold_files:
            continue

        dsnums.add(dsnum)
        norm = brc.normalize(model_raw)

        if fold_files:
            folds = {}
            for ff in fold_files:
                folds[brc.fold_label(os.path.basename(ff))] = brc.parse_result_file(ff)
            models[norm][dsnum] = folds
        if internal_fold_files:
            internal_folds = {}
            for ff in internal_fold_files:
                internal_folds[brc.fold_label(os.path.basename(ff))] = brc.parse_result_file(ff)
            internal_models[norm][dsnum] = internal_folds

    return models, internal_models, sorted(dsnums)


# --------------------------------------------------------------------------- #
# Completeness + aggregation.
# --------------------------------------------------------------------------- #
def is_complete(folds: dict, expected: list[str] | None = None) -> bool:
    """True iff every fold in `expected` is present and fully populated.
    `expected` defaults to EXPECTED_FOLDS; families that only ever ran a
    subset (e.g. the fold1-only pareto budget sweep, see
    scripts/pareto_budgets.py) pass their own list rather than being
    reported as incomplete."""
    for fold in expected or EXPECTED_FOLDS:
        entry = folds.get(fold)
        if not entry:
            return False
        for mkey in MODALITY_KEY_ORDER:
            row = entry.get(mkey)
            if not row:
                return False
            for region in REGIONS:
                if row.get(region) is None or row.get(f"{region}_hd95") is None:
                    return False
    return True


def aggregate(folds: dict, expected: list[str] | None = None) -> dict:
    """{region: {"Dice": (mean, std), "HD95": (mean, std)}}, averaged over the
    15 modality combinations of the per-combination (mean, std) over folds.
    `expected` selects which folds to average (default EXPECTED_FOLDS); with a
    single fold the std is 0 by construction, which is what a one-fold family
    should report."""
    expected = expected or EXPECTED_FOLDS
    out = {}
    for region in REGIONS:
        out[region] = {}
        for metric, suffix in (("Dice", ""), ("HD95", "_hd95")):
            key = region + suffix
            combo_means, combo_stds = [], []
            for mkey in MODALITY_KEY_ORDER:
                vals = [folds[fold][mkey][key] for fold in expected]
                combo_means.append(statistics.fmean(vals))
                combo_stds.append(statistics.pstdev(vals))
            out[region][metric] = (statistics.fmean(combo_means), statistics.fmean(combo_stds))
    return out


def aggregate_error(
    folds: dict, region: str, legacy_region_block: dict, expected: list[str] | None = None
) -> float | None:
    """Average, over the 15 modality combinations, of (our mean-over-folds Dice
    - legacy Dice) at that combination. legacy_region_block is
    legacy[dataset][region] ({mkey: value}); combos with no legacy value are
    skipped. Dice-only, since the legacy workbook has no HD95 baseline (see
    build_reproduction_comparison.py's module docstring)."""
    if not legacy_region_block:
        return None
    diffs = []
    for mkey in MODALITY_KEY_ORDER:
        legacy_val = legacy_region_block.get(mkey)
        if legacy_val is None:
            continue
        new_vals = [folds[fold][mkey][region] for fold in expected or EXPECTED_FOLDS]
        diffs.append(statistics.fmean(new_vals) - legacy_val)
    return statistics.fmean(diffs) if diffs else None


def build_class_rows(models: dict, internal_models: dict, display_names: dict, legacy: dict):
    """Return [(display_name, {group_label: stats_or_None}), ...], stats shaped
    {region: {"Dice": (mean,std), "HD95": (mean,std), "Error": value_or_None}}
    per GROUPS entry ("Error" only ever set for OFFICIAL_GROUP_LABELS groups).

    A model's entry for a given group is None (rendered "--") unless that
    group's folds are COMPLETE (see is_complete); a model is included in the
    output at all iff at least one of its four groups is complete."""
    all_model_norms = set(models) | set(internal_models)
    rows = []
    for model_norm in sorted(all_model_norms):
        if model_norm in EXCLUDED_MODELS:
            continue

        group_stats = {}
        any_present = False
        for label, source, dsnum in GROUPS:
            source_dict = models if source == "official" else internal_models
            folds = source_dict.get(model_norm, {}).get(dsnum)
            expected = OFFICIAL_EXPECTED_FOLDS if source == "official" else EXPECTED_FOLDS
            if folds and is_complete(folds, expected):
                stats = aggregate(folds, expected)
                if label in OFFICIAL_GROUP_LABELS:
                    dataset = brc.DATASETS.get(dsnum)
                    for region in REGIONS:
                        legacy_region_block = {
                            mkey: vals.get(model_norm)
                            for mkey, vals in legacy.get(dataset, {}).get(region, {}).items()
                        }
                        stats[region]["Error"] = aggregate_error(folds, region, legacy_region_block, expected)
                group_stats[label] = stats
                any_present = True
            else:
                group_stats[label] = None
        if not any_present:
            continue

        display = display_names.get(model_norm) or FALLBACK_DISPLAY_NAMES.get(model_norm, model_norm.upper())
        # Escape the plain name now, then append the \cite{} macro unescaped
        # (build_table renders this string as-is, without a second tex_escape
        # pass, precisely so this macro survives).
        display = tex_escape(display)
        key = CITATION_KEYS.get(model_norm)
        if key:
            display = f"{display}~\\cite{{{key}}}"
        rows.append((display, group_stats))
    rows.sort(key=lambda r: r[0].lower())
    return rows


# --------------------------------------------------------------------------- #
# LaTeX rendering.
# --------------------------------------------------------------------------- #
def tex_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s)


def best_per_group(rows, region, metric, label):
    """Index of the best row for (region, metric, group label): max for Dice,
    min for HD95. Rows lacking that group (None) are excluded."""
    values = [
        (i, r[1][label][region][metric][0]) for i, r in enumerate(rows) if r[1][label] is not None
    ]
    if not values:
        return None
    if metric == "Dice":
        return max(values, key=lambda t: t[1])[0]
    return min(values, key=lambda t: t[1])[0]


def second_best_per_group(rows, region, metric, label, best_idx):
    """Index of the runner-up row for (region, metric, group label), excluding
    the best and any row lacking that group."""
    values = [
        (i, r[1][label][region][metric][0])
        for i, r in enumerate(rows)
        if r[1][label] is not None and i != best_idx
    ]
    if not values:
        return None
    if metric == "Dice":
        return max(values, key=lambda t: t[1])[0]
    return min(values, key=lambda t: t[1])[0]


def dice_ranks_per_group(rows, region, label) -> dict[int, int]:
    """{row_index: rank}, rank 1 = highest Dice for this group/region, among
    rows that have data for this group. Rows lacking it get no entry."""
    present = [i for i, r in enumerate(rows) if r[1][label] is not None]
    order = sorted(present, key=lambda i: -rows[i][1][label][region]["Dice"][0])
    return {row_i: rank for rank, row_i in enumerate(order, start=1)}


def build_table(rows, region, caption, label):
    """One table for a single region: rows are models, columns are, per
    GROUPS entry, a Dice-based rank then Dice/HD95 mean+-std."""
    if not rows:
        return ""

    group_labels = [g[0] for g in GROUPS]
    bests = {
        (lbl, metric): best_per_group(rows, region, metric, lbl)
        for lbl in group_labels
        for metric in METRICS
    }
    runners_up = {
        (lbl, metric): second_best_per_group(rows, region, metric, lbl, bests[(lbl, metric)])
        for lbl in group_labels
        for metric in METRICS
    }
    ranks_by_group = {lbl: dice_ranks_per_group(rows, region, lbl) for lbl in group_labels}

    # Official groups (with a legacy baseline) get a 4th "Avg Error" cell;
    # internal groups (no legacy counterpart) stay at 3.
    n_sub_by_label = {lbl: (4 if lbl in OFFICIAL_GROUP_LABELS else 3) for lbl in group_labels}

    body = []
    for i, (name, group_stats) in enumerate(rows):
        cells = [name]
        for lbl in group_labels:
            stats = group_stats[lbl]
            if stats is None:
                cells.extend(["--"] * n_sub_by_label[lbl])
                continue
            rank = ranks_by_group[lbl].get(i)
            cells.append(str(rank) if rank is not None else "--")
            for metric in METRICS:
                mean, std = stats[region][metric]
                text = f"{mean:.1f}$\\pm${std:.1f}"
                if bests[(lbl, metric)] == i:
                    text = f"\\textbf{{{text}}}"
                elif runners_up[(lbl, metric)] == i:
                    text = f"\\underline{{{text}}}"
                cells.append(text)
            if lbl in OFFICIAL_GROUP_LABELS:
                error = stats[region].get("Error")
                cells.append(f"{error:+.1f}" if error is not None else "--")
        body.append(" & ".join(cells) + r" \\")

    # Level 1: one group per GROUPS entry, each spanning its own sub-columns
    # (4 for official groups incl. Avg Error, 3 for internal). Model is the
    # first (unspanned) column.
    group_header_parts = []
    group_cmidrule_parts = []
    col = 2
    for lbl in group_labels:
        n_sub = n_sub_by_label[lbl]
        group_header_parts.append(f"\\multicolumn{{{n_sub}}}{{c}}{{\\textbf{{{lbl}}}}}")
        group_cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col + n_sub - 1}}}")
        col += n_sub
    group_header = " & ".join(group_header_parts)
    group_cmidrules = " ".join(group_cmidrule_parts)

    # Level 2: Rank / Dice / HD95[/ Avg Error], repeated per group.
    sub_header = " & ".join(
        ("R & Dice & HD95 & Avg Error" if lbl in OFFICIAL_GROUP_LABELS else "R & Dice & HD95")
        for lbl in group_labels
    )

    colspec = "l" + "".join("cccc" if lbl in OFFICIAL_GROUP_LABELS else "ccc" for lbl in group_labels)

    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\fontsize{8}{10}\selectfont",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{3}{gray!8}{white}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        f" & {group_header} \\\\",
        group_cmidrules,
        f"\\textbf{{Model}} & {sub_header} \\\\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    legacy, display_names = {}, {}
    if os.path.exists(LEGACY_FILE):
        legacy, display_names = brc.parse_legacy(LEGACY_FILE)

    models, internal_models, dsnums = discover(RESULTS_DIR)
    if not dsnums:
        print("No result directories found in", RESULTS_DIR)
        return

    rows = build_class_rows(models, internal_models, display_names, legacy)
    if not rows:
        print("No model has at least one complete group (BraTS18/BraTS25-pre/MB-96); nothing to write")
        return

    sections = []
    for region in REGIONS:
        caption = (
            f"{REGION_LABELS[region]} ({region}): mean $\\pm$ std Dice (\\%) and HD95 (mm) "
            "over all modality-presence combinations and folds, per dataset. BraTS18/BraTS25-pre "
            "are the official test splits, averaged over our five folds; MB-96 - BraTS18 chp/"
            "MB-96 - BraTS25-pre chp are the internal cohort, scored with the BraTS18-trained and "
            "BraTS25-pre-trained checkpoints respectively, also averaged over the three fold "
            "checkpoints. BraTS18/BraTS25-pre additionally show Avg Error, the mean over all "
            "modality-presence combinations of (our reproduced Dice mean $-$ the published legacy "
            "Dice), i.e.\\ positive means we reproduce higher than legacy; not shown for the "
            "internal MB-96 groups, which have no legacy baseline, nor for HD95, which legacy "
            "never recorded. "
            "Rank is by Dice, among models with data for that group. Bold marks the best model "
            "per column, underline the runner-up."
        )
        table = build_table(rows, region, caption, f"tab:{region.lower()}")
        sections.append(table)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write("\n".join(sections))

    # NOTE: this wrapper is a throwaway local-preview document only, to eyeball
    # the tables as a standalone PDF. It is NOT AAAI-compliant (geometry and
    # \input are both explicitly banned by the AAAI author instructions) and
    # must never be copied into the paper's actual .tex source -- only the
    # \begin{table*}...\end{table*} blocks from tables.tex should be pasted in.
    main_tex = r"""\documentclass[12pt]{article}
\usepackage[paperwidth=10in, paperheight=13in, margin=0.6in]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage[table]{xcolor}
\usepackage{lmodern}
\pagestyle{plain}
\begin{document}
\input{tables.tex}
\end{document}
"""
    main_path = os.path.join(OUTPUT_DIR, "main.tex")
    with open(main_path, "w") as f:
        f.write(main_tex)

    print(f"Wrote {tables_path}")
    print(f"Wrote {main_path}")

    compile_pdf(main_path)


def compile_pdf(main_path):
    tectonic = shutil.which("tectonic") or os.path.expanduser("~/.local/bin/tectonic")
    if not (tectonic and os.access(tectonic, os.X_OK)):
        print("tectonic not found on PATH or in ~/.local/bin; skipping PDF compile", file=sys.stderr)
        return

    result = subprocess.run(
        [tectonic, os.path.basename(main_path)],
        cwd=os.path.dirname(main_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        print("tectonic compile failed", file=sys.stderr)
        return

    pdf_path = os.path.splitext(main_path)[0] + ".pdf"
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
