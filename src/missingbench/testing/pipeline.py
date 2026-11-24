from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
from medpy.metric import binary as medpy_binary
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.amp import autocast
from torch.utils.data import DataLoader

from missingbench.checkpoints import assert_finite_weights, load_weights_only_checkpoint
from missingbench.datasets import DatasetType, IMFuseDataset
from missingbench.enums import Precision
from missingbench.models.abstract_model import AbstractModel
from missingbench.utils.seed import set_seed

MASKS: list[list[bool]] = [
    [False, False, False, True],
    [False, True, False, False],
    [False, False, True, False],
    [True, False, False, False],
    [False, True, False, True],
    [False, True, True, False],
    [True, False, True, False],
    [False, False, True, True],
    [True, False, False, True],
    [True, True, False, False],
    [True, True, True, False],
    [True, False, True, True],
    [True, True, False, True],
    [False, True, True, True],
    [True, True, True, True],
]
MODALITY_NAMES = ("t1c", "t1n", "t2f", "t2w")

# Fixed BraTS-GLI acquisition grid (verified against the raw unpacked NIfTI
# volumes, e.g. /path/to/missingbench/unpacked/BraTS-GLI-*/*.nii.gz -> (182, 218, 182)).
# The HD95 empty-vs-nonempty fallback penalty must be the diagonal of this full,
# uncropped volume -- not of whatever region happens to be loaded, which varies
# per subject after non_empty-bbox preprocessing and would make the penalty
# (and therefore average HD95) inconsistent across subjects/models.
BRATS_FULL_VOLUME_SHAPE = (182, 218, 182)
BRATS_HD95_PENALTY = float(np.sqrt(sum(dim**2 for dim in BRATS_FULL_VOLUME_SHAPE)))


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: np.ndarray | float = 0
        self.avg: np.ndarray | float = 0
        self.sum: np.ndarray | float = 0
        self.count = 0

    def update(self, val: Any, n: int = 1) -> None:
        value = np.asarray(val)
        self.sum = value * n if self.count == 0 else self.sum + (value * n)
        self.count += n
        self.avg = self.sum / self.count
        self.val = value


def softmax_output_dice_class4(
    output: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    if output.ndim != 4 or target.ndim != 4:
        raise ValueError(
            "Expected output and target to have shape [B, H, W, D], "
            f"got {tuple(output.shape)} and {tuple(target.shape)}"
        )

    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    union1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    net_ncr_dice = intersect1 / union1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps
    union2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / union2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps
    union3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect3 / union3

    o4 = o3 * 0.0 if torch.sum(o3) < 500 else o3
    intersect4 = torch.sum(2 * (o4 * t3), dim=(1, 2, 3)) + eps
    union4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect4 / union4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    union_whole = (
        torch.sum(o_whole, dim=(1, 2, 3))
        + torch.sum(t_whole, dim=(1, 2, 3))
        + eps
    )
    dice_whole = intersect_whole / union_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    union_core = (
        torch.sum(o_core, dim=(1, 2, 3))
        + torch.sum(t_core, dim=(1, 2, 3))
        + eps
    )
    dice_core = intersect_core / union_core

    dice_separate = torch.cat(
        (
            torch.unsqueeze(net_ncr_dice, 1),
            torch.unsqueeze(edema_dice, 1),
            torch.unsqueeze(enhancing_dice, 1),
        ),
        dim=1,
    )
    dice_evaluate = torch.cat(
        (
            torch.unsqueeze(dice_whole, 1),
            torch.unsqueeze(dice_core, 1),
            torch.unsqueeze(enhancing_dice, 1),
            torch.unsqueeze(enhancing_dice_postpro, 1),
        ),
        dim=1,
    )
    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def softmax_output_dice_class5(
    output: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Like softmax_output_dice_class4, plus a standalone RC (label 4) dice.

    WT/TC/ET/ETpp are computed exactly as in the class4 variant (RC is not
    folded into any of them); RC is appended as a 4th, independent raw-label
    column alongside NCR/NET, edema, and enhancing.
    """
    eps = 1e-8
    dice_separate, dice_evaluate = softmax_output_dice_class4(output, target)

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps
    union4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    rc_dice = (intersect4 / union4).cpu().numpy()

    dice_separate = np.concatenate((dice_separate, rc_dice[:, None]), axis=1)
    return dice_separate, dice_evaluate


def _hd95_or_penalty(prediction: np.ndarray, target: np.ndarray, penalty: float) -> float:
    prediction_empty = not prediction.any()
    target_empty = not target.any()
    if prediction_empty and target_empty:
        return 0.0
    if prediction_empty or target_empty:
        return penalty
    return float(medpy_binary.hd95(prediction, target, voxelspacing=None))


def softmax_output_hd95_class4(
    output: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    """BraTS-style HD95 for WT/TC/ET/ETpp, mirroring softmax_output_dice_class4.

    Falls back to 0.0 when both masks are empty (no error) and to the
    volume's spatial diagonal as a fixed penalty when only one of the two
    masks is empty (undefined surface distance), matching the BraTS
    challenge convention.
    """
    if output.ndim != 4 or target.ndim != 4:
        raise ValueError(
            "Expected output and target to have shape [B, H, W, D], "
            f"got {tuple(output.shape)} and {tuple(target.shape)}"
        )

    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    batch_size = output_np.shape[0]
    penalty = BRATS_HD95_PENALTY

    results = np.zeros((batch_size, 4), dtype=np.float64)
    for index in range(batch_size):
        o1 = output_np[index] == 1
        t1 = target_np[index] == 1
        o2 = output_np[index] == 2
        t2 = target_np[index] == 2
        o3 = output_np[index] == 3
        t3 = target_np[index] == 3

        o_whole = o1 | o2 | o3
        t_whole = t1 | t2 | t3
        o_core = o1 | o3
        t_core = t1 | t3
        o4 = np.zeros_like(o3) if o3.sum() < 500 else o3

        results[index, 0] = _hd95_or_penalty(o_whole, t_whole, penalty)
        results[index, 1] = _hd95_or_penalty(o_core, t_core, penalty)
        results[index, 2] = _hd95_or_penalty(o3, t3, penalty)
        results[index, 3] = _hd95_or_penalty(o4, t3, penalty)

    return results


def softmax_output_hd95_separate_class4(
    output: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    """Raw per-label HD95 for NCR/NET, edema, enhancing, mirroring dice_separate."""
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    batch_size = output_np.shape[0]
    penalty = BRATS_HD95_PENALTY

    results = np.zeros((batch_size, 3), dtype=np.float64)
    for index in range(batch_size):
        for column, label in enumerate((1, 2, 3)):
            o = output_np[index] == label
            t = target_np[index] == label
            results[index, column] = _hd95_or_penalty(o, t, penalty)

    return results


def softmax_output_hd95_separate_class5(
    output: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    """Like softmax_output_hd95_separate_class4, plus a standalone RC (label 4) HD95."""
    results_class4 = softmax_output_hd95_separate_class4(output, target)

    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    batch_size = output_np.shape[0]
    penalty = BRATS_HD95_PENALTY

    rc_results = np.zeros((batch_size, 1), dtype=np.float64)
    for index in range(batch_size):
        o4 = output_np[index] == 4
        t4 = target_np[index] == 4
        rc_results[index, 0] = _hd95_or_penalty(o4, t4, penalty)

    return np.concatenate((results_class4, rc_results), axis=1)


def _hd95_job(prediction: np.ndarray, target: np.ndarray, include_rc: bool) -> tuple[np.ndarray, np.ndarray]:
    """HD95 for one [1, H, W, D] prediction/target pair: exactly the serial
    calls, run on CPU copies so it can execute in a worker process."""
    prediction_t = torch.from_numpy(prediction)
    target_t = torch.from_numpy(target)
    separate_fn = softmax_output_hd95_separate_class5 if include_rc else softmax_output_hd95_separate_class4
    return softmax_output_hd95_class4(prediction_t, target_t), separate_fn(prediction_t, target_t)


@contextlib.contextmanager
def _hd95_pool():
    """Process pool for HD95 scoring. MISSINGBENCH_HD95_WORKERS overrides the size;
    0 keeps the original serial, in-process scoring. Default: all CPUs in
    this process's affinity mask but one (left to the main process), max 8."""
    env = os.environ.get("MISSINGBENCH_HD95_WORKERS")
    workers = int(env) if env is not None else min(len(os.sched_getaffinity(0)) - 1, 8)
    if workers <= 0:
        yield None
        return
    # spawn, not fork: the parent already holds a CUDA context.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        yield pool


def run_testing(
    *,
    data_dir: Path,
    output_path: Path,
    checkpoint_path: Path,
    dataset_type: DatasetType,
    model_class: type[AbstractModel],
    model_kwargs: dict[str, Any] | None = None,
    split_file: Path,
    fold: int | None = None,
    num_workers: int = 8,
    seed: int = 42,
    precision: str | Precision = Precision.FP32,
    budget: float | None = None,
    budget_ceiling: bool = False,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Testing currently requires at least one CUDA device")
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"{output_path} must be a file path, not a directory")
    if not checkpoint_path.is_file():
        raise click.ClickException(f"Checkpoint not found: {checkpoint_path}")

    set_seed(seed)
    device = torch.device("cuda")

    try:
        split = _load_test_split(split_file=split_file, dataset_type=dataset_type, fold=fold)
        test_set = IMFuseDataset(
            root=data_dir,
            masking_mode=None,
            split=split,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if test_loader.batch_size != 1:
            raise RuntimeError("Unified test expects batch_size=1")

        model = model_class(**(model_kwargs or {}))
        if not isinstance(model, AbstractModel):
            raise RuntimeError(f"{model_class.__name__} must inherit from AbstractModel")
        model = model.to(device)
        load_weights_only_checkpoint(model, checkpoint_path, device=device)
        assert_finite_weights(model, checkpoint_path)
        model.eval()
        if hasattr(model, "is_training"):
            model.is_training = False
        if budget is not None:
            if not hasattr(model, "set_budget"):
                raise click.ClickException(
                    f"{model_class.__name__} has no set_budget -- --budget only "
                    "applies to budget-conditioned models (e.g. MatUnet)."
                )
            if budget_ceiling:
                used = model.set_budget_ceiling(budget)
                print(
                    f"Budget ceiling: budget {budget:.0f} -> selection budget {used:.0f}, "
                    f"widths {model.current_widths()}, {model.actual_parameters()} params",
                    flush=True,
                )
            else:
                model.set_budget(budget)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        include_rc = dataset_type == DatasetType.BRATS25
        evaluate_labels = ("WT", "TC", "ET", "ETpp")
        separate_labels = ("NCR_NET", "Edema", "Enhancing") + (("RC",) if include_rc else ())
        dice_fn = softmax_output_dice_class5 if include_rc else softmax_output_dice_class4
        hd95_separate_fn = (
            softmax_output_hd95_separate_class5 if include_rc else softmax_output_hd95_separate_class4
        )

        total_score = AverageMeter()
        total_hd95 = AverageMeter()
        total_separate_score = AverageMeter()
        total_separate_hd95 = AverageMeter()
        subject_records: list[dict[str, Any]] = []
        total_steps = len(MASKS) * len(test_loader)
        hd95_queue: deque[tuple[str, np.ndarray, np.ndarray, Any]] = deque()
        with torch.no_grad(), _hd95_pool() as hd95_pool:
            hd95_backlog = 2 * hd95_pool._max_workers if hd95_pool is not None else 0
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[metrics]}", style="magenta"),
                transient=True,
            ) as progress:
                task_id = progress.add_task(
                    "Testing masks",
                    total=total_steps,
                    metrics="",
                )
                # Subject-major sweep: each subject volume is loaded (and moved to the
                # GPU) once and all 15 mask combinations run against it, instead of one
                # full DataLoader pass -- 15 disk reads/decompressions per subject --
                # per mask. Every per-mask meter is still updated in DataLoader
                # (subject) order and the per-mask records are emitted in MASKS order
                # afterwards, so the report, the Excel summary and the per-subject CSV
                # are identical in content and row order to the previous mask-major
                # loop.
                mask_tensors = [
                    torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0) for mask in MASKS
                ]
                mask_labels = [_mask_name(mask) for mask in MASKS]
                mask_scores = [AverageMeter() for _ in MASKS]
                mask_hd95s = [AverageMeter() for _ in MASKS]
                mask_separate_scores = [AverageMeter() for _ in MASKS]
                mask_separate_hd95s = [AverageMeter() for _ in MASKS]
                mask_records: list[list[dict[str, Any]]] = [[] for _ in MASKS]

                def score_batch(
                    prediction: torch.Tensor, target: torch.Tensor, sub: str, mask_index: int
                ) -> None:
                    # Runs one forward pass behind the one that produced `prediction`
                    # (see the prefetch loop below), so the next forward is already
                    # running on the GPU while this executes. Dice is computed here on the
                    # GPU; HD95 (medpy distance transforms, the CPU-bound part) is handed to
                    # the process pool and collected, in submission order, by drain_hd95.
                    brats_dice_separate, brats_dice = dice_fn(
                        output=prediction,
                        target=target,
                    )
                    mask_scores[mask_index].update(brats_dice)
                    mask_separate_scores[mask_index].update(brats_dice_separate)
                    prediction_np = prediction.to(torch.uint8).cpu().numpy()
                    target_np = target.to(torch.uint8).cpu().numpy()
                    if hd95_pool is None:
                        hd95_result: Any = _hd95_job(prediction_np, target_np, include_rc)
                    else:
                        hd95_result = hd95_pool.submit(_hd95_job, prediction_np, target_np, include_rc)
                    hd95_queue.append((sub, mask_index, brats_dice, brats_dice_separate, hd95_result))
                    drain_hd95(hd95_backlog)

                def drain_hd95(keep: int) -> None:
                    # Resolve the oldest HD95 results until at most `keep` are outstanding.
                    # Each mask's meters and records are updated strictly in submission
                    # order, i.e. subject order, so averages and the per-subject CSV are
                    # identical to the serial path.
                    while len(hd95_queue) > keep:
                        sub, mask_index, brats_dice, brats_dice_separate, hd95_result = hd95_queue.popleft()
                        if hd95_pool is not None:
                            hd95_result = hd95_result.result()
                        brats_hd95, brats_hd95_separate = hd95_result
                        mask_specific_score = mask_scores[mask_index]
                        mask_specific_hd95 = mask_hd95s[mask_index]
                        mask_specific_separate_score = mask_separate_scores[mask_index]
                        mask_specific_separate_hd95 = mask_separate_hd95s[mask_index]
                        mask_label = mask_labels[mask_index]
                        mask_specific_hd95.update(brats_hd95)
                        mask_specific_separate_hd95.update(brats_hd95_separate)
                        mask_records[mask_index].append(
                                {
                                    "subject": sub,
                                    "modalities": mask_label,
                                    **{
                                        f"{label}_dice": float(brats_dice[0, index])
                                        for index, label in enumerate(evaluate_labels)
                                    },
                                    **{
                                        f"{label}_dice": float(brats_dice_separate[0, index])
                                        for index, label in enumerate(separate_labels)
                                    },
                                    **{
                                        f"{label}_hd95": float(brats_hd95[0, index])
                                        for index, label in enumerate(evaluate_labels)
                                    },
                                    **{
                                        f"{label}_hd95": float(brats_hd95_separate[0, index])
                                        for index, label in enumerate(separate_labels)
                                    },
                            }
                        )
                        current_avg = np.asarray(mask_specific_score.avg)[0]
                        current_hd95_avg = np.asarray(mask_specific_hd95.avg)[0]
                        metrics_text = (
                            f"{mask_label}"
                            f"  |  DS: WT {current_avg[0]:.4f}  "
                            f"TC {current_avg[1]:.4f}  "
                            f"ET {current_avg[2]:.4f}  |  "
                            f"HD95: WT {current_hd95_avg[0]:.4f}  "
                            f"TC {current_hd95_avg[1]:.4f}  "
                            f"ET {current_hd95_avg[2]:.4f}"
                        )
                        if include_rc:
                            current_separate_avg = np.asarray(mask_specific_separate_score.avg)[0]
                            current_separate_hd95_avg = np.asarray(mask_specific_separate_hd95.avg)[0]
                            metrics_text += (
                                f"  |  RC: DS {current_separate_avg[-1]:.4f}  "
                                f"HD95 {current_separate_hd95_avg[-1]:.4f}"
                            )
                        progress.update(task_id, advance=1, metrics=metrics_text)

                # Software-pipelined (prefetch) loop: launch forward i+1 (async CUDA
                # kernel enqueue, returns immediately) before scoring forward i, so the
                # GPU works on i+1 while the CPU scores i. Forwards run subject by
                # subject and, within a subject, mask by mask. `pending` holds the one
                # forward that has been launched but not yet scored; it's flushed after
                # the loop for the final one.
                amp_dtype = torch.bfloat16 if precision == Precision.BF16 else torch.float16
                amp_enabled = Precision(precision) != Precision.FP32 and device.type == "cuda"
                pending: tuple[torch.Tensor, torch.Tensor, str, int] | None = None
                for batch in test_loader:
                    images = batch["images"].to(device, non_blocking=True)
                    target = batch["seg"].to(device, non_blocking=True).squeeze(1).long()
                    sub = str(batch["sub"][0])
                    for mask_index, mask_tensor in enumerate(mask_tensors):
                        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                            output = model.predict(images, mask_tensor)
                        prediction = torch.argmax(output, dim=1)

                        if pending is not None:
                            score_batch(*pending)

                        pending = (prediction, target, sub, mask_index)

                if pending is not None:
                    score_batch(*pending)
                drain_hd95(0)

                for mask_index, mask_label in enumerate(mask_labels):
                    subject_records.extend(mask_records[mask_index])
                    mask_score_avg = np.asarray(mask_scores[mask_index].avg)[0]
                    mask_hd95_avg = np.asarray(mask_hd95s[mask_index].avg)[0]
                    mask_separate_score_avg = np.asarray(mask_separate_scores[mask_index].avg)[0]
                    mask_separate_hd95_avg = np.asarray(mask_separate_hd95s[mask_index].avg)[0]
                    total_score.update(mask_score_avg)
                    total_hd95.update(mask_hd95_avg)
                    total_separate_score.update(mask_separate_score_avg)
                    total_separate_hd95.update(mask_separate_hd95_avg)
                    evaluate_fields = ", ".join(
                        f"{label} = {value:.4f}" for label, value in zip(evaluate_labels, mask_score_avg)
                    )
                    evaluate_hd95_fields = ", ".join(
                        f"{label}_hd95 = {value:.4f}"
                        for label, value in zip(evaluate_labels, mask_hd95_avg)
                    )
                    separate_fields = ", ".join(
                        f"{label}_dice = {value:.4f}"
                        for label, value in zip(separate_labels, mask_separate_score_avg)
                    )
                    separate_hd95_fields = ", ".join(
                        f"{label}_hd95 = {value:.4f}"
                        for label, value in zip(separate_labels, mask_separate_hd95_avg)
                    )
                    _append_report_line(
                        output_path,
                        (
                            f"Available modals = {mask_label:<21}--> "
                            f"{evaluate_fields}, {evaluate_hd95_fields}, "
                            f"{separate_fields}, {separate_hd95_fields}"
                        ),
                    )

        avg_total_score = np.asarray(total_score.avg)
        avg_total_hd95 = np.asarray(total_hd95.avg)
        avg_total_separate_score = np.asarray(total_separate_score.avg)
        avg_total_separate_hd95 = np.asarray(total_separate_hd95.avg)
        avg_evaluate_fields = ", ".join(
            f"{label} = {value:.4f}" for label, value in zip(evaluate_labels, avg_total_score)
        )
        avg_evaluate_hd95_fields = ", ".join(
            f"{label}_hd95 = {value:.4f}" for label, value in zip(evaluate_labels, avg_total_hd95)
        )
        avg_separate_fields = ", ".join(
            f"{label}_dice = {value:.4f}"
            for label, value in zip(separate_labels, avg_total_separate_score)
        )
        avg_separate_hd95_fields = ", ".join(
            f"{label}_hd95 = {value:.4f}"
            for label, value in zip(separate_labels, avg_total_separate_hd95)
        )
        _append_report_line(
            output_path,
            (
                f"Avg scores {'':<29}--> "
                f"{avg_evaluate_fields}, {avg_evaluate_hd95_fields}, "
                f"{avg_separate_fields}, {avg_separate_hd95_fields}"
            ),
        )
        _write_excel_summary(output_path)
        _write_per_subject_report(output_path, subject_records)
        return output_path
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from None


def _pad_to_shape(tensor: torch.Tensor, target_hwd: tuple[int, int, int]) -> torch.Tensor:
    _, h, w, d = tensor.shape
    target_h, target_w, target_d = target_hwd
    return torch.nn.functional.pad(tensor, [0, target_d - d, 0, target_w - w, 0, target_h - h])


def _variable_size_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate for batch_size > 1: subjects have different spatial shapes
    after this pipeline's non-empty-bbox crop preprocessing (see
    BRATS_FULL_VOLUME_SHAPE above), so plain `torch.stack` (default_collate)
    fails across a batch. Zero-pads every sample up to the batch's max H/W/D
    instead; callers must crop predictions/targets back to each sample's
    `orig_shape` before scoring so the padding never affects metrics."""
    max_hwd = (
        max(item["images"].shape[-3] for item in batch),
        max(item["images"].shape[-2] for item in batch),
        max(item["images"].shape[-1] for item in batch),
    )
    return {
        "sub": [item["sub"] for item in batch],
        "images": torch.stack([_pad_to_shape(item["images"], max_hwd) for item in batch], dim=0),
        "seg": torch.stack([_pad_to_shape(item["seg"], max_hwd) for item in batch], dim=0),
        "orig_shape": [tuple(item["images"].shape[-3:]) for item in batch],
    }


def run_testing_fast(
    *,
    data_dir: Path,
    output_path: Path,
    checkpoint_path: Path,
    dataset_type: DatasetType,
    model_class: type[AbstractModel],
    model_kwargs: dict[str, Any] | None = None,
    split_file: Path,
    fold: int | None = None,
    num_workers: int = 8,
    batch_size: int = 1,
    seed: int = 42,
    precision: str | Precision = Precision.FP32,
) -> Path:
    """Same mask-sweep evaluation as `run_testing`, restructured for speed.

    Each subject volume is loaded from disk once and its 15 mask
    combinations are run against that one already-loaded (and already
    GPU-resident) batch, instead of `run_testing`'s one-DataLoader-pass-
    per-mask approach, which reloads/decompresses every subject's `.npz`
    15 times. Subjects are also grouped into `batch_size`-sized batches for
    the forward pass. Writes the same per-subject CSV as `run_testing`.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Testing currently requires at least one CUDA device")
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"{output_path} must be a file path, not a directory")
    if not checkpoint_path.is_file():
        raise click.ClickException(f"Checkpoint not found: {checkpoint_path}")

    set_seed(seed)
    device = torch.device("cuda")

    try:
        split = _load_test_split(split_file=split_file, dataset_type=dataset_type, fold=fold)
        test_set = IMFuseDataset(
            root=data_dir,
            masking_mode=None,
            split=split,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=_variable_size_collate,
        )

        model = model_class(**(model_kwargs or {}))
        if not isinstance(model, AbstractModel):
            raise RuntimeError(f"{model_class.__name__} must inherit from AbstractModel")
        model = model.to(device)
        load_weights_only_checkpoint(model, checkpoint_path, device=device)
        assert_finite_weights(model, checkpoint_path)
        model.eval()
        if hasattr(model, "is_training"):
            model.is_training = False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        include_rc = dataset_type == DatasetType.BRATS25
        evaluate_labels = ("WT", "TC", "ET", "ETpp")
        separate_labels = ("NCR_NET", "Edema", "Enhancing") + (("RC",) if include_rc else ())
        dice_fn = softmax_output_dice_class5 if include_rc else softmax_output_dice_class4
        hd95_separate_fn = (
            softmax_output_hd95_separate_class5 if include_rc else softmax_output_hd95_separate_class4
        )

        total_score = AverageMeter()
        total_hd95 = AverageMeter()
        total_separate_score = AverageMeter()
        total_separate_hd95 = AverageMeter()
        mask_meters: dict[str, tuple[AverageMeter, AverageMeter, AverageMeter, AverageMeter]] = {
            _mask_name(mask): (AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter())
            for mask in MASKS
        }
        subject_records: list[dict[str, Any]] = []

        total_steps = len(MASKS) * len(test_loader)
        # Rich's Progress live-redraws in place using cursor-movement escape
        # codes, which only makes sense on a real terminal. Under sbatch,
        # stdout is redirected to a plain .out file (not a tty), so Rich
        # suppresses the animated redraws entirely -- combined with
        # transient=True (which erases the bar on exit), nothing about
        # progress ever reaches the log file. Fall back to plain periodic
        # lines via click.echo in that case instead.
        log_to_stdout = not sys.stdout.isatty()
        step = 0
        with torch.no_grad():
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[metrics]}", style="magenta"),
                transient=True,
                disable=log_to_stdout,
            ) as progress:
                task_id = progress.add_task(
                    "Testing masks (fast)",
                    total=total_steps,
                    metrics="",
                )
                for batch in test_loader:
                    images = batch["images"].to(device, non_blocking=True)
                    target = batch["seg"].to(device, non_blocking=True).squeeze(1).long()
                    orig_shapes: list[tuple[int, int, int]] = batch["orig_shape"]
                    current_batch_size = images.shape[0]

                    for mask in MASKS:
                        mask_label = _mask_name(mask)
                        mask_score, mask_hd95, mask_separate_score, mask_separate_hd95 = mask_meters[
                            mask_label
                        ]
                        mask_tensor = (
                            torch.tensor(mask, dtype=torch.bool, device=device)
                            .unsqueeze(0)
                            .expand(current_batch_size, -1)
                        )

                        amp_dtype = torch.bfloat16 if precision == Precision.BF16 else torch.float16
                        amp_enabled = Precision(precision) != Precision.FP32 and device.type == "cuda"
                        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                            output = model.predict(images, mask_tensor)
                        prediction = torch.argmax(output, dim=1)

                        # Batch members may have been zero-padded (see
                        # _variable_size_collate) to a common shape for the
                        # stack/forward pass; crop each one back to its own
                        # original volume shape before scoring so the padding
                        # never affects dice/HD95.
                        for i in range(current_batch_size):
                            h, w, d = orig_shapes[i]
                            pred_i = prediction[i : i + 1, :h, :w, :d]
                            target_i = target[i : i + 1, :h, :w, :d]

                            brats_dice_separate, brats_dice = dice_fn(
                                output=pred_i,
                                target=target_i,
                            )
                            brats_hd95 = softmax_output_hd95_class4(
                                output=pred_i,
                                target=target_i,
                            )
                            brats_hd95_separate = hd95_separate_fn(
                                output=pred_i,
                                target=target_i,
                            )
                            mask_score.update(brats_dice)
                            mask_hd95.update(brats_hd95)
                            mask_separate_score.update(brats_dice_separate)
                            mask_separate_hd95.update(brats_hd95_separate)
                            subject_records.append(
                                {
                                    "subject": batch["sub"][i],
                                    "modalities": mask_label,
                                    **{
                                        f"{label}_dice": float(brats_dice[0, index])
                                        for index, label in enumerate(evaluate_labels)
                                    },
                                    **{
                                        f"{label}_dice": float(brats_dice_separate[0, index])
                                        for index, label in enumerate(separate_labels)
                                    },
                                    **{
                                        f"{label}_hd95": float(brats_hd95[0, index])
                                        for index, label in enumerate(evaluate_labels)
                                    },
                                    **{
                                        f"{label}_hd95": float(brats_hd95_separate[0, index])
                                        for index, label in enumerate(separate_labels)
                                    },
                                }
                            )

                        current_avg = np.asarray(mask_score.avg)[0]
                        current_hd95_avg = np.asarray(mask_hd95.avg)[0]
                        metrics_text = (
                            f"{mask_label}"
                            f"  |  DS: WT {current_avg[0]:.4f}  "
                            f"TC {current_avg[1]:.4f}  "
                            f"ET {current_avg[2]:.4f}  |  "
                            f"HD95: WT {current_hd95_avg[0]:.4f}  "
                            f"TC {current_hd95_avg[1]:.4f}  "
                            f"ET {current_hd95_avg[2]:.4f}"
                        )
                        if include_rc:
                            current_separate_avg = np.asarray(mask_separate_score.avg)[0]
                            current_separate_hd95_avg = np.asarray(mask_separate_hd95.avg)[0]
                            metrics_text += (
                                f"  |  RC: DS {current_separate_avg[-1]:.4f}  "
                                f"HD95 {current_separate_hd95_avg[-1]:.4f}"
                            )
                        step += 1
                        if log_to_stdout:
                            click.echo(f"[{step}/{total_steps}] {metrics_text}")
                        progress.update(task_id, advance=1, metrics=metrics_text)

                for mask in MASKS:
                    mask_label = _mask_name(mask)
                    mask_score, mask_hd95, mask_separate_score, mask_separate_hd95 = mask_meters[mask_label]
                    mask_score_avg = np.asarray(mask_score.avg)[0]
                    mask_hd95_avg = np.asarray(mask_hd95.avg)[0]
                    mask_separate_score_avg = np.asarray(mask_separate_score.avg)[0]
                    mask_separate_hd95_avg = np.asarray(mask_separate_hd95.avg)[0]
                    total_score.update(mask_score_avg)
                    total_hd95.update(mask_hd95_avg)
                    total_separate_score.update(mask_separate_score_avg)
                    total_separate_hd95.update(mask_separate_hd95_avg)
                    evaluate_fields = ", ".join(
                        f"{label} = {value:.4f}" for label, value in zip(evaluate_labels, mask_score_avg)
                    )
                    evaluate_hd95_fields = ", ".join(
                        f"{label}_hd95 = {value:.4f}"
                        for label, value in zip(evaluate_labels, mask_hd95_avg)
                    )
                    separate_fields = ", ".join(
                        f"{label}_dice = {value:.4f}"
                        for label, value in zip(separate_labels, mask_separate_score_avg)
                    )
                    separate_hd95_fields = ", ".join(
                        f"{label}_hd95 = {value:.4f}"
                        for label, value in zip(separate_labels, mask_separate_hd95_avg)
                    )
                    _append_report_line(
                        output_path,
                        (
                            f"Available modals = {mask_label:<21}--> "
                            f"{evaluate_fields}, {evaluate_hd95_fields}, "
                            f"{separate_fields}, {separate_hd95_fields}"
                        ),
                    )

        avg_total_score = np.asarray(total_score.avg)
        avg_total_hd95 = np.asarray(total_hd95.avg)
        avg_total_separate_score = np.asarray(total_separate_score.avg)
        avg_total_separate_hd95 = np.asarray(total_separate_hd95.avg)
        avg_evaluate_fields = ", ".join(
            f"{label} = {value:.4f}" for label, value in zip(evaluate_labels, avg_total_score)
        )
        avg_evaluate_hd95_fields = ", ".join(
            f"{label}_hd95 = {value:.4f}" for label, value in zip(evaluate_labels, avg_total_hd95)
        )
        avg_separate_fields = ", ".join(
            f"{label}_dice = {value:.4f}"
            for label, value in zip(separate_labels, avg_total_separate_score)
        )
        avg_separate_hd95_fields = ", ".join(
            f"{label}_hd95 = {value:.4f}"
            for label, value in zip(separate_labels, avg_total_separate_hd95)
        )
        _append_report_line(
            output_path,
            (
                f"Avg scores {'':<29}--> "
                f"{avg_evaluate_fields}, {avg_evaluate_hd95_fields}, "
                f"{avg_separate_fields}, {avg_separate_hd95_fields}"
            ),
        )
        _write_excel_summary(output_path)
        _write_per_subject_report(output_path, subject_records)
        return output_path
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from None


def _load_test_split(
    *, split_file: Path, dataset_type: DatasetType, fold: int | None = None
) -> list[dict[str, Any]]:
    from missingbench.utils.cli_overrides import select_fold

    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    split_payload = select_fold(json.loads(split_file.read_text()), fold, source=split_file)
    dataset_key = str(dataset_type).lower()
    if dataset_key not in split_payload:
        raise RuntimeError(f"Dataset split '{dataset_key}' not found in {split_file}")
    dataset_splits = split_payload[dataset_key]
    return list(dataset_splits.get("test", []))


def _mask_name(mask: list[bool]) -> str:
    present = [name for name, enabled in zip(MODALITY_NAMES, mask) if enabled]
    return "_".join(present)


def _append_report_line(output_path: Path, line: str) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _write_excel_summary(results_path: Path) -> Path:
    excel_path = results_path.with_suffix(".xlsx")
    scores: dict[str, list[float]] = {}
    order: list[int] = []

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            is_avg = "Avg" in line
            parsed = _parse_avg_line(line) if is_avg else _parse_result_line(line)
            order.append(15 if is_avg else _string_to_order(parsed["modals"]))
            for key, value in parsed.items():
                if key == "modals":
                    continue
                multiplier = 1.0 if (is_avg or key.endswith("_hd95")) else 100.0
                scores.setdefault(key, []).append(value * multiplier)

    scores_sorted = {
        key: [value for _, value in sorted(zip(order, values))] for key, values in scores.items()
    }

    for values in scores_sorted.values():
        values[-1] = sum(values[:-1]) / len(values[:-1])

    pd.DataFrame(scores_sorted).to_excel(excel_path, index=False)
    return excel_path


def _write_per_subject_report(results_path: Path, records: list[dict[str, Any]]) -> Path:
    per_subject_path = results_path.with_name(f"{results_path.stem}_per_subject.csv")
    pd.DataFrame(records).to_csv(per_subject_path, index=False)
    return per_subject_path


def _parse_result_line(line: str) -> dict[str, float | str]:
    left, right = line.split("-->")
    modals = left.split("=")[1].strip()

    metrics: dict[str, float | str] = {"modals": modals}
    for item in right.split(","):
        key, value = item.split("=")
        metrics[key.strip()] = float(value.strip())
    return metrics


def _parse_avg_line(line: str) -> dict[str, float]:
    _, right = line.split("-->")
    metrics: dict[str, float] = {}
    for item in right.split(","):
        key, value = item.split("=")
        metrics[key.strip()] = float(value.strip())
    return metrics


def _string_to_code(modals: str) -> int:
    code = 0
    if "t1c" in modals:
        code += 1
    if "t1n" in modals:
        code += 2
    if "t2f" in modals:
        code += 4
    if "t2w" in modals:
        code += 8
    return code


def _string_to_order(modals: str) -> int:
    code = _string_to_code(modals)
    match code:
        case 1:
            return 2
        case 2:
            return 1
        case 3:
            return 7
        case 4:
            return 0
        case 5:
            return 5
        case 6:
            return 4
        case 7:
            return 10
        case 8:
            return 3
        case 9:
            return 9
        case 10:
            return 8
        case 11:
            return 13
        case 12:
            return 6
        case 13:
            return 12
        case 14:
            return 11
        case 15:
            return 14
        case _:
            raise RuntimeError(f"invalid code passed: {modals}, {code}")


__all__ = [
    "AverageMeter",
    "MASKS",
    "MODALITY_NAMES",
    "run_testing",
    "set_seed",
    "softmax_output_dice_class4",
    "softmax_output_dice_class5",
    "softmax_output_hd95_class4",
    "softmax_output_hd95_separate_class4",
    "softmax_output_hd95_separate_class5",
]
