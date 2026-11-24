"""An explicit, instantiable description of an nnU-Net macro-architecture.

The search space is deliberately restricted to nnU-Net's *macro* structure --
number of resolution stages, channels per stage, and encoder/decoder block
counts.  Convolutions, normalization, activations and skip connections stay
exactly as ``PlainUnetBackbone`` configures them, so every architecture this module can
build is a member of the nnU-Net family and remains directly comparable to the
hand-written Plain U-Net baselines (``plainunet_*`` configs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from missingbench.nas.param_count import exact_parameters

__all__ = ["ArchitectureConfig", "build_network"]


@dataclass(frozen=True)
class ArchitectureConfig:
    """A single point in the macro-architecture search space.

    ``n_blocks_encoder``/``n_blocks_decoder`` are per-stage tuples.  They are
    scalars in every configuration used today, but the underlying
    ``PlainConvUNet`` already accepts per-stage lists, so keeping them as tuples
    here means searchable block depth would need no change to this schema.
    The decoder has ``n_stages - 1`` entries: there is no decoder stage for the
    bottleneck itself.
    """

    n_stages: int
    features_per_stage: tuple[int, ...]
    n_blocks_encoder: tuple[int, ...]
    n_blocks_decoder: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_stages < 4:
            raise ValueError(
                "PlainUnetBackbone requires at least four stages for its auxiliary "
                f"heads; got n_stages={self.n_stages}"
            )
        if len(self.features_per_stage) != self.n_stages:
            raise ValueError(
                f"features_per_stage has {len(self.features_per_stage)} entries "
                f"but n_stages={self.n_stages}"
            )
        if len(self.n_blocks_encoder) != self.n_stages:
            raise ValueError(
                f"n_blocks_encoder has {len(self.n_blocks_encoder)} entries "
                f"but n_stages={self.n_stages}"
            )
        if len(self.n_blocks_decoder) != self.n_stages - 1:
            raise ValueError(
                f"n_blocks_decoder has {len(self.n_blocks_decoder)} entries but "
                f"n_stages-1={self.n_stages - 1}"
            )
        if any(w <= 0 for w in self.features_per_stage):
            raise ValueError(f"widths must be positive, got {self.features_per_stage}")
        if any(b <= 0 for b in self.n_blocks_encoder + self.n_blocks_decoder):
            raise ValueError("block counts must be positive")

    @classmethod
    def uniform(
        cls,
        features_per_stage: Sequence[int],
        *,
        n_conv_per_stage: int = 2,
        n_conv_per_stage_decoder: int = 2,
    ) -> ArchitectureConfig:
        """Build a config with the same block count at every stage."""
        widths = tuple(int(w) for w in features_per_stage)
        n_stages = len(widths)
        return cls(
            n_stages=n_stages,
            features_per_stage=widths,
            n_blocks_encoder=(n_conv_per_stage,) * n_stages,
            n_blocks_decoder=(n_conv_per_stage_decoder,) * (n_stages - 1),
        )

    @property
    def is_uniform_depth(self) -> bool:
        """Whether every stage shares one block count (encoder and decoder)."""
        return (
            len(set(self.n_blocks_encoder)) == 1
            and len(set(self.n_blocks_decoder)) == 1
        )

    def parameter_count(self, *, num_modals: int = 4, num_cls: int = 4) -> int:
        """Exact parameter count without instantiating the network."""
        if not self.is_uniform_depth:
            raise NotImplementedError(
                "parameter_count currently assumes a uniform block count per "
                "stage; per-stage depth is not supported"
            )
        return exact_parameters(
            self.features_per_stage,
            num_modals=num_modals,
            num_cls=num_cls,
            n_conv_per_stage=self.n_blocks_encoder[0],
            n_conv_per_stage_decoder=self.n_blocks_decoder[0],
        )


def build_network(
    config: ArchitectureConfig,
    *,
    num_cls: int = 4,
    num_modals: int = 4,
    input_shape: Sequence[int] = (128, 128, 128),
    overlap: float = 0.5,
):
    """Instantiate the ``PlainUnet`` described by ``config``.

    Imported lazily: ``missingbench.models.config._resolve_model`` imports every module
    under ``missingbench/models/`` on each lookup, so a module-level import here would
    add the model package to the import cost of anything touching ``missingbench.nas``.
    """
    from missingbench.models.plain_unet import PlainUnet

    if not config.is_uniform_depth:
        raise NotImplementedError(
            "build_network currently assumes a uniform block count per stage; "
            "per-stage depth is not supported"
        )

    return PlainUnet(
        num_cls=num_cls,
        num_modals=num_modals,
        input_shape=tuple(input_shape),
        features_per_stage=config.features_per_stage,
        n_conv_per_stage=config.n_blocks_encoder[0],
        n_conv_per_stage_decoder=config.n_blocks_decoder[0],
        overlap=overlap,
    )
