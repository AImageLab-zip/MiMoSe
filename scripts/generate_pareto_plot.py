#!/usr/bin/env python3
"""2D Pareto plot (GPU latency vs. Dice) across every model with complete
accuracy + efficiency data: the 18 literature baselines, the 8
mimosa_[size] variants, and every budget of the pareto_23 supernet sweep
(one point per budget, since each budget is a different extracted
architecture) -- the plotting counterpart to
generate_pareto_table.py's tab:pareto (which computes 5-objective
dominance: Dice, HD95, GPU latency, peak GPU memory, FLOPs).

This plot only looks at 2 of those 5 axes (GPU latency, Dice), so a
point's Pareto status here is its own fresh 2D non-dominated calculation,
not a projection of the 5-objective frontier onto this plane -- a model
that's "Pareto? = Yes" in tab:pareto (because it wins on HD95/memory/
FLOPs) can still look dominated here, and vice versa.

With ~26 models on one plot, labeling every point on the main axes gets
crowded wherever several models land close together. This script instead
auto-detects that densest cluster, leaves only its bare markers on the
main overview, and re-draws just that cluster -- fully labeled -- in a
"magnifying glass" inset, connected back to its region on the overview via
the standard mark_inset zoom lines. Everything outside the cluster keeps
its on-canvas label as before.

Reuses generate_pareto_table.py's collect_all_rows() for the per-model
(Dice, GPU latency) data, so both scripts stay in sync on how those
numbers are computed.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_pareto_table as gpar  # noqa: E402
import pareto_budgets as pbud  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset  # noqa: E402

OUTPUT_DIR = gpar.OUTPUT_DIR

FAMILY_STYLE = {
    "Baseline": {"marker": "s", "color": "#4C72B0"},
    "MimosaSize": {"marker": "o", "color": "#DD8452"},
    pbud.FAMILY: {"marker": "^", "color": "#55A868"},
}

# Legend text differs from the internal family key (which mirrors
# generate_pareto_table.py's "Baseline"/"MimosaSize" labels): on this plot
# the literature models are "Reproduced" and the mimosa_[size] family is
# "Baseline" (this project's own compute-scaling baseline sweep).
LEGEND_LABEL = {
    "Baseline": "Reproduced",
    "MimosaSize": "Baseline",
    # One triangle per budget of the supernet, not one per supernet: the
    # points trace out what a single set of shared weights can be run at.
    pbud.FAMILY: "MiMoSe-P (per budget)",
}

FRONTIER_COLOR = "#2B2B2B"
GRID_COLOR = "#D8D8D8"
INSET_FACE = "#FAFAFA"
INSET_EDGE = "#8A8A8A"

# Candidate corners for the inset (and, mirrored, the legend), tried in
# this preference order -- loc strings understood by mpl's inset_axes, same
# vocabulary as legend `loc`.
INSET_CORNERS = ["lower right", "upper right", "lower left", "upper left"]
OPPOSITE_CORNER = {
    "lower right": "upper left",
    "upper right": "lower left",
    "lower left": "upper right",
    "upper left": "lower right",
}
INSET_FRACTION_CANDIDATES = [0.54, 0.48, 0.42, 0.36, 0.30]  # tried largest first


def mb_display_name(display: str) -> str:
    """"MiMoSe-Base" (generate_model_comparison_table_mimosa_size.py's
    FALLBACK_DISPLAY_NAMES) -> "mb_base", matching the mb_[size] naming used
    in outputs/full_profile_table_mimosa_size/tables.tex."""
    _prefix, _, size = display.partition("-")
    return f"mb_{size.lower()}" if size else display


def repel_labels(fig, ax, entries, iterations=400, pad=1.08, push=2.5, leader_color="gray", extra_boxes=None):
    """Nudge each text label's position (in data coords) until no two label
    bounding boxes overlap anymore, pushing along the line between box
    centers each iteration -- a small hand-rolled stand-in for the
    `adjustText` package (not currently a project dependency, and not worth
    adding just for this one plot). `entries` is [(text_artist,
    anchor_xy_data, marker_halfwidth_px), ...]; anchor_xy_data is the data
    point the label belongs to, used afterwards to draw a leader line for
    any label that ended up displaced from it; marker_halfwidth_px sizes
    the fixed obstacle box around that point's actual marker.
    `extra_boxes` are additional fixed pixel-space Bbox obstacles with no
    associated label (e.g. the zoom inset panel), pushed away from the same
    way as markers."""
    if not entries:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    # Every marker (not just the label's own) is a fixed obstacle a label
    # must clear -- otherwise a label with no *other* label nearby never
    # moves at all, even while sitting right on top of its own square/circle.
    marker_boxes = []
    for _, (ax_, ay_), half_pt in entries:
        half_px = half_pt * fig.dpi / 72.0
        cx, cy = ax.transData.transform((ax_, ay_))
        marker_boxes.append(Bbox.from_extents(cx - half_px, cy - half_px, cx + half_px, cy + half_px))
    marker_boxes.extend(extra_boxes or [])

    for _ in range(iterations):
        boxes = [t.get_window_extent(renderer=renderer) for t, _anchor, _half in entries]
        moved = False
        for i, (text, _anchor, _half) in enumerate(entries):
            fx = fy = 0.0
            bi = boxes[i].expanded(pad, pad)
            for j, box_j in enumerate(boxes):
                if i == j or not bi.overlaps(box_j):
                    continue
                dx = (boxes[i].x0 + boxes[i].x1) / 2 - (box_j.x0 + box_j.x1) / 2
                dy = (boxes[i].y0 + boxes[i].y1) / 2 - (box_j.y0 + box_j.y1) / 2
                dist = max((dx**2 + dy**2) ** 0.5, 1.0)
                fx += dx / dist
                fy += dy / dist
            for mbox in marker_boxes:
                if not bi.overlaps(mbox.expanded(1.5, 1.5)):
                    continue
                dx = (boxes[i].x0 + boxes[i].x1) / 2 - (mbox.x0 + mbox.x1) / 2
                dy = (boxes[i].y0 + boxes[i].y1) / 2 - (mbox.y0 + mbox.y1) / 2
                dist = max((dx**2 + dy**2) ** 0.5, 1.0)
                fx += 1.6 * dx / dist
                fy += 1.6 * dy / dist
            if fx == 0 and fy == 0:
                continue
            moved = True
            x, y = text.get_position()
            px, py = ax.transData.transform((x, y))
            nx, ny = inv.transform((px + fx * push, py + fy * push))
            text.set_position((nx, ny))
        if not moved:
            break

    for text, anchor, _half in entries:
        anchor_px = ax.transData.transform(anchor)
        label_px = ax.transData.transform(text.get_position())
        if ((anchor_px[0] - label_px[0]) ** 2 + (anchor_px[1] - label_px[1]) ** 2) ** 0.5 > 8:
            ax.plot(
                [anchor[0], text.get_position()[0]],
                [anchor[1], text.get_position()[1]],
                color=leader_color,
                linewidth=0.5,
                alpha=0.6,
                zorder=1,
                clip_on=False,
            )


def pareto_frontier(points):
    """Names of the non-dominated points (maximize dice, minimize latency)."""
    frontier = []
    for name, (lat, dice) in points.items():
        dominated = any(
            other_lat <= lat
            and other_dice >= dice
            and (other_lat < lat or other_dice > dice)
            for other_name, (other_lat, other_dice) in points.items()
            if other_name != name
        )
        if not dominated:
            frontier.append(name)
    return frontier


def find_dense_cluster(points, frac=0.3, k=4, pad_frac=0.04):
    """Names of the densest `frac` fraction of points, found by min-max
    normalizing both axes (so latency-seconds and Dice-percent are on a
    comparable scale) and ranking points by distance to their k-th nearest
    neighbor -- small distance means crowded. Also returns a padded
    (xmin, xmax, ymin, ymax) bounding box, in data coordinates, covering
    that cluster. Returns (None, None) if there are too few points to
    bother with a zoom inset."""
    names = list(points.keys())
    if len(names) < 6:
        return None, None

    xs = [points[n][0] for n in names]
    ys = [points[n][1] for n in names]
    xspan = (max(xs) - min(xs)) or 1.0
    yspan = (max(ys) - min(ys)) or 1.0
    nxs = [(x - min(xs)) / xspan for x in xs]
    nys = [(y - min(ys)) / yspan for y in ys]

    k = min(k, len(names) - 1)
    density = []
    for i in range(len(names)):
        dists = sorted(
            ((nxs[i] - nxs[j]) ** 2 + (nys[i] - nys[j]) ** 2) ** 0.5
            for j in range(len(names))
            if j != i
        )
        density.append(dists[k - 1])

    order = sorted(range(len(names)), key=lambda i: density[i])
    n_dense = max(int(len(names) * frac), k + 1)
    dense_idx = order[:n_dense]
    dense_names = [names[i] for i in dense_idx]

    dxs = [xs[i] for i in dense_idx]
    dys = [ys[i] for i in dense_idx]
    x0, x1 = min(dxs), max(dxs)
    y0, y1 = min(dys), max(dys)
    pad_x = max((x1 - x0) * pad_frac, xspan * 0.02)
    pad_y = max((y1 - y0) * pad_frac, yspan * 0.02)
    bbox = (x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y)
    return dense_names, bbox


def choose_corner_and_fraction(ax, outside_points):
    """(loc, fraction) for the inset: the largest fraction (tried in
    descending order) at which some candidate corner covers zero
    `outside_points` -- an inset panel is opaque, so a sparse marker sitting
    underneath it would be genuinely hidden, not just crowded. Shrinks the
    inset rather than accept that. Falls back to whichever (corner,
    fraction) covers the fewest points if no combination reaches zero."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xspan, yspan = xlim[1] - xlim[0], ylim[1] - ylim[0]

    def corner_bounds(loc, frac):
        y0, y1 = (
            (ylim[1] - yspan * frac, ylim[1]) if "upper" in loc else (ylim[0], ylim[0] + yspan * frac)
        )
        x0, x1 = (
            (xlim[1] - xspan * frac, xlim[1]) if "right" in loc else (xlim[0], xlim[0] + xspan * frac)
        )
        return x0, x1, y0, y1

    # A few percent of margin around the tested box: a point sitting right
    # at the boundary still reads, visually, as "under" the inset once its
    # label and leader line are drawn, even if its bare anchor is a hair
    # outside the panel's true edge.
    margin_x, margin_y = xspan * 0.05, yspan * 0.05

    fallback_loc, fallback_frac, fallback_count = INSET_CORNERS[0], INSET_FRACTION_CANDIDATES[0], None
    for frac in INSET_FRACTION_CANDIDATES:
        for loc in INSET_CORNERS:
            x0, x1, y0, y1 = corner_bounds(loc, frac)
            x0, x1, y0, y1 = x0 - margin_x, x1 + margin_x, y0 - margin_y, y1 + margin_y
            count = sum(1 for x, y in outside_points if x0 <= x <= x1 and y0 <= y <= y1)
            if count == 0:
                return loc, frac
            if fallback_count is None or count < fallback_count:
                fallback_loc, fallback_frac, fallback_count = loc, frac, count
    return fallback_loc, fallback_frac


def marker_size_for(name, frontier):
    return 95 if name in frontier else 60


def scatter_markers(ax, points, families, frontier, names):
    """Just the markers -- no labels. Split out from label placement so the
    zoom inset's screen position can be computed (and fed back as an
    obstacle for the main overview's labels) before any label goes down."""
    for name in names:
        lat, dice = points[name]
        style = FAMILY_STYLE[families[name]]
        on_frontier = name in frontier
        ax.scatter(
            lat,
            dice,
            marker=style["marker"],
            color=style["color"],
            s=marker_size_for(name, frontier),
            zorder=3,
            edgecolors=FRONTIER_COLOR if on_frontier else "white",
            linewidths=1.3 if on_frontier else 0.6,
            alpha=1.0 if on_frontier else 0.75,
        )


def add_labels(ax, points, families, frontier, names, *, fontsize, leader_color="gray", repel_kwargs=None, extra_boxes=None):
    """Repelled text label for every point in `names` (markers must already
    be drawn via scatter_markers). Shared between the main overview and the
    zoom inset so both stay visually consistent."""
    label_entries = []
    for label_idx, name in enumerate(names):
        lat, dice = points[name]
        on_frontier = name in frontier
        marker_size = marker_size_for(name, frontier)
        # `s` is marker area in points^2 -- convert to an actual radius (in
        # points) so both the initial offset and the repel obstacle box
        # scale with how big the marker really renders, instead of a guess
        # that's only right for one marker size.
        marker_half_pt = (marker_size / 3.14159) ** 0.5 + 3
        offset_px = marker_half_pt * ax.figure.dpi / 72.0 + 10
        # Spread each label's starting direction around the golden angle
        # instead of always up-right: two points sitting almost on top of
        # each other (common in the dense cluster) would otherwise both
        # start from ~the same spot and the repulsion pass can leave their
        # leader lines crossed. A different starting angle per label sidesteps
        # that near-degenerate case.
        angle = math.radians((label_idx * 137.508) % 360)
        dx_px, dy_px = offset_px * math.cos(angle), offset_px * math.sin(angle)
        px, py = ax.transData.transform((lat, dice))
        start_x, start_y = ax.transData.inverted().transform((px + dx_px, py + dy_px))
        text = ax.text(
            start_x,
            start_y,
            name,
            fontsize=fontsize,
            fontweight="bold" if on_frontier else "normal",
            color=FRONTIER_COLOR if on_frontier else "#333333",
            zorder=4,
        )
        label_entries.append((text, (lat, dice), marker_half_pt))
    repel_labels(
        ax.figure, ax, label_entries, leader_color=leader_color, extra_boxes=extra_boxes, **(repel_kwargs or {})
    )


def style_axes(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")


def collect_points(x_key="lat_gpu", *, x_scale=1.0, expected_folds=None, groups=None):
    """({display: (x, dice)}, {display: family}) over all three families, with
    `x_key` (a metric key of gpar.collect_all_rows' dicts) on the x axis,
    divided by `x_scale` (e.g. 1e12 to turn raw FLOPs into TFLOPs).
    `expected_folds` restricts which folds Dice is averaged over -- None keeps
    each family's own default (3 folds for the baselines and mimosa_[size],
    fold1 for the supernet budgets, which is all that was ever run).
    `groups` restricts which evaluations Dice averages over -- see
    gpar.collect_all_rows."""
    rows = gpar.collect_all_rows(expected_folds=expected_folds, groups=groups)
    points, families = {}, {}
    for display, family, m in rows:
        if m.get("dice") is None or m.get(x_key) is None:
            continue
        if family == "MimosaSize":
            display = mb_display_name(display)
        elif family == pbud.FAMILY:
            display = pbud.short_name(display)
        points[display] = (m[x_key] / x_scale, m["dice"])
        families[display] = family
    return points, families


def render_pareto_plot(points, families, *, x_label, out_path, y_label="Average Dice (%)", x_log=False):
    """Draw the overview + zoom-inset scatter for already-collected points and
    write it to `out_path`. Shared by this script and its fold1-only
    latency/FLOPs counterparts (generate_pareto_plot_latency_fold1.py,
    generate_pareto_plot_flops_fold1.py) so all three stay visually identical
    and only differ in what goes on the x axis."""
    frontier = set(pareto_frontier(points))
    # On a log x axis the points spread out enough to label in place, and
    # find_dense_cluster's linear normalization would pick the wrong cluster.
    dense_names, dense_bbox = (None, None) if x_log else find_dense_cluster(points)
    dense_set = set(dense_names or [])
    sparse_names = [n for n in points if n not in dense_set]

    fig, ax = plt.subplots(figsize=(12.5, 9.5), facecolor="white")
    ax.set_facecolor("white")
    if x_log:
        ax.set_xscale("log")

    # Every point gets its marker on the overview; only the sparse ones get
    # an on-canvas label -- the dense cluster's labels move into the inset.
    scatter_markers(ax, points, families, frontier, list(points.keys()))

    frontier_pts = sorted((points[d] for d in frontier), key=lambda p: p[0])
    if len(frontier_pts) > 1:
        fx, fy = zip(*frontier_pts)
        ax.plot(fx, fy, color=FRONTIER_COLOR, linestyle="--", linewidth=1.1, alpha=0.55, zorder=2)

    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    style_axes(ax)

    inset_corner = None
    sparse_extra_boxes = []
    if dense_names:
        x0, x1, y0, y1 = dense_bbox
        outside_points = [points[n] for n in sparse_names]
        inset_corner, inset_frac = choose_corner_and_fraction(ax, outside_points)
        axins = inset_axes(
            ax,
            width=f"{inset_frac * 100:.0f}%",
            height=f"{inset_frac * 100:.0f}%",
            loc=inset_corner,
            borderpad=1.4,
        )
        axins.set_facecolor(INSET_FACE)
        for spine in axins.spines.values():
            spine.set_edgecolor(INSET_EDGE)
            spine.set_linewidth(1.1)
        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("zoomed", fontsize=11, style="italic", color="#555555", pad=6)

        # The inset panel's own screen footprint becomes an obstacle for the
        # main overview's sparse labels below, so a label never lands on top
        # of the panel it doesn't belong to (e.g. the ones near its title).
        fig.canvas.draw()
        sparse_extra_boxes = [axins.get_window_extent(renderer=fig.canvas.get_renderer())]

        scatter_markers(axins, points, families, frontier, dense_names)
        if len(frontier_pts) > 1:
            axins.plot(fx, fy, color=FRONTIER_COLOR, linestyle="--", linewidth=1.0, alpha=0.5, zorder=2)
        add_labels(
            axins,
            points,
            families,
            frontier,
            dense_names,
            fontsize=12,
            leader_color="#777777",
            repel_kwargs={"pad": 1.6, "push": 5.5, "iterations": 800},
        )

        mark_inset(
            ax, axins, loc1=1, loc2=3, fc="none", ec=INSET_EDGE, linestyle=":", linewidth=0.9, alpha=0.8
        )

    add_labels(ax, points, families, frontier, sparse_names, fontsize=12, extra_boxes=sparse_extra_boxes)

    handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="w",
            markerfacecolor=style["color"],
            markeredgecolor=FRONTIER_COLOR,
            markersize=9,
            label=LEGEND_LABEL[family],
        )
        for family, style in FAMILY_STYLE.items()
    ]
    handles.append(
        Line2D([0], [0], color=FRONTIER_COLOR, linestyle="--", linewidth=1.1, label="Pareto frontier")
    )
    legend_loc = OPPOSITE_CORNER[inset_corner] if inset_corner else "best"
    ax.legend(handles=handles, loc=legend_loc, frameon=True, framealpha=0.9, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    points, families = collect_points("lat_gpu")
    if not points:
        print("No models with both Dice and GPU latency data", file=sys.stderr)
        return
    render_pareto_plot(
        points,
        families,
        x_label="Whole-volume GPU latency (s)",
        out_path=os.path.join(OUTPUT_DIR, "pareto_dice_vs_latency.png"),
    )


if __name__ == "__main__":
    main()
