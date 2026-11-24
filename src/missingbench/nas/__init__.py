"""Budget-conditioned architecture search over the nnU-Net macro-architecture.

The goal is to learn ``budget -> architecture`` as a single function, rather than
searching for one architecture per operating point.  Architectures are nested by
construction: ``b1 < b2`` implies ``A(b1)`` is a subnetwork of ``A(b2)``.

Modules:

``budget``
    Log-space budget <-> normalized coordinate conversions.
``param_count``
    Exact (integer) and differentiable (soft) parameter accounting.
``arch_config``
    ``ArchitectureConfig`` + ``build_network`` -- an explicit description of a
    point in the search space.
``gates``
    Monotone, prefix-structured width gates conditioned on the budget.
``calibration``
    Places gate thresholds so requested budget matches realized parameters.
``extract``
    Materializes a standalone ``PlainUnet`` with weights inherited from the
    supernet.
"""

from __future__ import annotations

from missingbench.nas.arch_config import ArchitectureConfig, build_network
from missingbench.nas.budget import budget_to_z, sample_log_uniform_budget, z_to_budget
from missingbench.nas.calibration import (
    alpha_for_budget,
    budget_param_curve,
    calibrate_thresholds_analytic,
    uniform_width_schedule,
)
from missingbench.nas.extract import concat_indices, extract_subnet
from missingbench.nas.gates import WidthGates
from missingbench.nas.param_count import exact_parameters, expected_parameters

__all__ = [
    "ArchitectureConfig",
    "WidthGates",
    "alpha_for_budget",
    "budget_param_curve",
    "budget_to_z",
    "build_network",
    "calibrate_thresholds_analytic",
    "concat_indices",
    "exact_parameters",
    "expected_parameters",
    "extract_subnet",
    "sample_log_uniform_budget",
    "uniform_width_schedule",
    "z_to_budget",
]
