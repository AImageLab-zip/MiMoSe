from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from missingbench.checkpoints import load_weights_only_checkpoint, save_weights_only_checkpoint

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    class PyTorchModelHubMixin:  # type: ignore[no-redef]
        pass


class AbstractModel(nn.Module, PyTorchModelHubMixin, ABC):
    def model_for_export(self) -> "AbstractModel":
        """The model whose weights should be persisted as the deployable
        checkpoint (``final_weights_only.safetensors`` / HF export).

        Defaults to ``self``. KD-style wrappers that carry training-only
        submodules alongside the deployable model (e.g. a teacher network,
        hint adapters) override this to return just the deployable part, so
        the resumable ``model_last.pth``/``best.pth`` checkpoints (which use
        ``self.state_dict()`` directly, not this hook) can still keep full
        training state while the exported artifact stays lean.
        """
        return self

    def get_hf_config(self) -> dict[str, Any]:
        target = self.model_for_export()
        return {
            "model_class": target.__class__.__name__,
            "model_kwargs": dict(getattr(target, "_missingbench_model_kwargs", {})),
            "missingbench_model_name": getattr(target, "_missingbench_model_name", target.__class__.__name__),
        }

    def export_hf_pretrained(self, save_directory: str | Path) -> Path:
        export_dir = Path(save_directory)
        export_dir.mkdir(parents=True, exist_ok=True)
        self._save_pretrained(export_dir)
        (export_dir / "config.json").write_text(
            json.dumps(self.get_hf_config(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return export_dir

    def _save_pretrained(self, save_directory: str | Path) -> None:
        target_dir = Path(save_directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        save_weights_only_checkpoint(
            self.model_for_export(),
            target_dir / "final_weights_only.safetensors",
        )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        proxies: dict[str, str] | None,
        resume_download: bool | None,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str | torch.device = "cpu",
        strict: bool = False,
        **model_kwargs: Any,
    ) -> "AbstractModel":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Loading from Hugging Face requires the 'huggingface_hub' package"
            ) from exc

        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )
        config_payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        init_kwargs = dict(config_payload.get("model_kwargs", {}))
        init_kwargs.update(model_kwargs)
        model = cls(**init_kwargs)
        checkpoint_path = hf_hub_download(
            repo_id=model_id,
            filename="final_weights_only.safetensors",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )
        load_weights_only_checkpoint(model, checkpoint_path, device=map_location, strict=strict)
        return model

    @abstractmethod
    def predict(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run test-time inference for a batch of masked input volumes.

        This is the evaluation contract used by the MissingBench testing
        pipeline. Unlike ``forward(...)``, which may return trainer-specific
        tuples for loss computation, ``predict(...)`` must return the final
        segmentation prediction tensor only.

        Expected inputs:
        - ``images``: tensor shaped ``[B, M, H, W, D]`` containing the input
          volumes for a batch, where ``M`` is the modality dimension.
        - ``mask``: boolean tensor describing which modalities are available
          for each sample. Implementations must apply the same modality logic
          used during training, including any internal modality reordering.

        Expected output:
        - a tensor shaped ``[B, C, H, W, D]`` containing segmentation logits
          or probabilities, where ``C`` is the number of output classes.

        Implementations may call ``forward(...)`` internally, run a dedicated
        sliding-window inference path, or disable training-only branches before
        producing the final prediction. The important point is that callers of
        ``predict(...)`` should not need to know any trainer-specific details.
        """
        pass
