"""Reporting for the gradient-interference experiment.

Writes three artifacts next to each other:

``interference_pairs.csv``
    One row per budget pair -- the tidy form for downstream analysis and for
    re-plotting without re-running the measurement (which needs a GPU).
``interference_matrices.csv``
    The raw ``n x n`` matrices, for reproducing the heatmaps.
``interference.png``
    Two heatmaps plus the headline scatter of similarity against log-budget
    distance, with the noise floor drawn in.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from missingbench.nas.interference import InterferenceResult

__all__ = ["write_interference_report", "summarize_interference"]


def _fmt_budget(budget: float) -> str:
    return f"{budget / 1e6:.2f}M"


def summarize_interference(result: InterferenceResult) -> dict[str, Any]:
    """Reduce the pairwise data to the question the interference experiment actually asks.

    The headline test is whether gradient agreement decays with budget distance,
    so this reports the correlation between log-distance and each similarity
    measure, together with the nearest/farthest pair values.
    """
    rows = [r for r in result.pairs() if not math.isnan(r["cosine_full"])]
    if len(rows) < 2:
        return {"n_pairs": len(rows)}

    def _pearson(xs: list[float], ys: list[float]) -> float:
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        if var_x <= 0 or var_y <= 0:
            return float("nan")
        return cov / math.sqrt(var_x * var_y)

    distances = [r["log_distance"] for r in rows]
    nearest = min(rows, key=lambda r: r["log_distance"])
    farthest = max(rows, key=lambda r: r["log_distance"])

    summary: dict[str, Any] = {
        "n_pairs": len(rows),
        "n_batches": result.n_batches,
        "n_shared_parameters": result.n_shared_parameters,
        "corr_distance_vs_full": _pearson(
            distances, [r["cosine_full"] for r in rows]
        ),
        "corr_distance_vs_overlap": _pearson(
            distances, [r["cosine_overlap"] for r in rows]
        ),
        "corr_distance_vs_overlap_fraction": _pearson(
            distances, [r["overlap_fraction"] for r in rows]
        ),
        "nearest_pair_full": nearest["cosine_full"],
        "nearest_pair_overlap": nearest["cosine_overlap"],
        "farthest_pair_full": farthest["cosine_full"],
        "farthest_pair_overlap": farthest["cosine_overlap"],
        "min_noise_floor": min(result.noise_floor),
        "max_noise_floor": max(result.noise_floor),
        "min_relative_overlap": min(r["relative_overlap"] for r in rows),
    }
    return summary


def write_interference_report(
    result: InterferenceResult,
    output_dir: str | Path,
    *,
    console: Any | None = None,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    pairs = result.pairs()

    pairs_path = directory / "interference_pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)

    matrices_path = directory / "interference_matrices.csv"
    with matrices_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["matrix", "row_budget", "col_budget", "value"])
        for name, matrix in (
            ("cosine_full", result.cosine_full),
            ("cosine_overlap", result.cosine_overlap),
            ("overlap_fraction", result.overlap_fraction),
        ):
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    writer.writerow(
                        [name, result.budgets[i], result.budgets[j], value]
                    )
        for index, value in enumerate(result.noise_floor):
            writer.writerow(
                ["noise_floor", result.budgets[index], result.budgets[index], value]
            )

    summary = summarize_interference(result)
    summary_path = directory / "interference_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])

    figure_path = _write_figure(result, directory)

    if console is not None:
        _print_summary(result, summary, console)
        console.print(f"[green]pairs   ->[/green] {pairs_path}")
        console.print(f"[green]matrices->[/green] {matrices_path}")
        console.print(f"[green]summary ->[/green] {summary_path}")
        if figure_path is not None:
            console.print(f"[green]figure  ->[/green] {figure_path}")

    return directory


def _print_summary(
    result: InterferenceResult,
    summary: dict[str, Any],
    console: Any,
) -> None:
    from rich.table import Table

    table = Table(title="Gradient interference across budgets", show_lines=False)
    table.add_column("pair", justify="left")
    table.add_column("log dist", justify="right")
    table.add_column("cos full", justify="right")
    table.add_column("cos overlap", justify="right")
    table.add_column("shared", justify="right")
    table.add_column("noise floor", justify="right")
    table.add_column("rel. overlap", justify="right")

    for row in sorted(result.pairs(), key=lambda r: r["log_distance"]):
        table.add_row(
            f"{_fmt_budget(row['budget_i'])} vs {_fmt_budget(row['budget_j'])}",
            f"{row['log_distance']:.2f}",
            f"{row['cosine_full']:+.3f}",
            f"{row['cosine_overlap']:+.3f}",
            f"{row['overlap_fraction']:.2f}",
            f"{row['noise_floor']:.3f}",
            f"{row['relative_overlap']:.2f}",
        )
    console.print(table)

    verdict = _verdict(summary)
    console.print(verdict)


def _verdict(summary: dict[str, Any]) -> str:
    """State what the numbers imply for local-budget training."""
    corr_overlap = summary.get("corr_distance_vs_overlap", float("nan"))
    min_relative = summary.get("min_relative_overlap", float("nan"))

    if math.isnan(corr_overlap):
        return "[yellow]inconclusive: not enough valid pairs[/yellow]"

    lines = [
        f"corr(log-distance, cos_overlap) = {corr_overlap:+.3f}",
        f"worst relative overlap alignment = {min_relative:.2f}",
    ]
    if corr_overlap < -0.5 and min_relative < 0.8:
        lines.append(
            "[bold green]Interference grows with budget distance[/bold green] -- "
            "local-budget training is empirically motivated."
        )
    elif min_relative >= 0.9:
        lines.append(
            "[bold yellow]No interference beyond minibatch noise[/bold yellow] -- "
            "on shared weights, distant budgets agree about as well as a single "
            "budget agrees with itself across batches. Local-budget training may "
            "not be necessary; check overlap_fraction, which may be the real "
            "constraint."
        )
    else:
        lines.append(
            "[bold]Mixed[/bold] -- see corr(distance, overlap_fraction): distant "
            "budgets may simply share too few weights for interference to be the "
            "binding issue."
        )
    return "\n".join(lines)


def _write_figure(result: InterferenceResult, directory: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a hard dependency
        return None

    labels = [_fmt_budget(b) for b in result.budgets]
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    for ax, matrix, title, vmin in (
        (axes[0], result.cosine_full, "cosine (all shared params)", None),
        (axes[1], result.cosine_overlap, "cosine (co-active params only)", None),
        (axes[2], result.overlap_fraction, "fraction of support shared", 0.0),
    ):
        image = ax.imshow(
            matrix,
            cmap="RdYlGn",
            vmin=0.0 if vmin is None else vmin,
            vmax=1.0,
            origin="upper",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i][j]
                if not math.isnan(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
        figure.colorbar(image, ax=ax, fraction=0.046)

    figure.suptitle(
        f"Gradient interference across parameter budgets "
        f"({result.n_batches} batches, {result.n_shared_parameters:,} shared params)",
        fontsize=11,
    )
    figure.tight_layout()
    heatmap_path = directory / "interference_heatmaps.png"
    figure.savefig(heatmap_path, dpi=170)
    plt.close(figure)

    # ---- the headline plot: does agreement decay with distance? ----------
    rows = [r for r in result.pairs() if not math.isnan(r["cosine_full"])]
    figure, ax = plt.subplots(figsize=(7.2, 5.0))
    distances = [r["log_distance"] for r in rows]

    ax.scatter(
        distances,
        [r["cosine_full"] for r in rows],
        s=34,
        color="#1f77b4",
        label="cosine, all shared params",
    )
    ax.scatter(
        distances,
        [r["cosine_overlap"] for r in rows],
        s=34,
        marker="s",
        color="#2ca02c",
        label="cosine, co-active params only",
    )
    ax.scatter(
        distances,
        [r["noise_floor"] for r in rows],
        s=22,
        marker="_",
        color="#d62728",
        label="noise floor (same budget, other batch)",
    )
    ax.scatter(
        distances,
        [r["overlap_fraction"] for r in rows],
        s=22,
        marker="x",
        color="0.45",
        label="fraction of support shared",
    )

    ax.set_xlabel(r"budget distance  $|\log b_i - \log b_j|$")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Do distant budgets fight over the shared weights?", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="best")
    figure.tight_layout()
    scatter_path = directory / "interference_vs_distance.png"
    figure.savefig(scatter_path, dpi=170)
    plt.close(figure)

    return heatmap_path
