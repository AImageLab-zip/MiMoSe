from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from missingbench import __version__
from missingbench.models.abstract_model import AbstractModel
from missingbench.paths import PACKAGE_ROOT

ZIPAPP_SHEBANG = b"#!/usr/bin/env python3\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EXCLUDED_TOP_LEVEL_DIRS = {"testing"}


def build_missingbench_zipapp(output_path: Path) -> Path:
    """Bundle the installed `missingbench` package into a single-file `.pyz`.

    Every `.py` file (plus packaged `data/configs`, `data/splits` assets)
    under `missingbench.paths.PACKAGE_ROOT` is written into the archive as real
    files under a `missingbench/` prefix, so the result is importable via
    `sys.path.insert(0, str(output_path))`. `__pycache__` directories are
    skipped, as is `missingbench/testing/` (`EXCLUDED_TOP_LEVEL_DIRS`) -- the
    mask-sweep/Dice/HD95 scoring loop deliberately never ships inside a
    `.missingbenchpkg`; only what's needed to reconstruct the model and call
    `predict()` does. See `docker/harness/testing_loop.py` for the harness's
    own copy of that loop, which runs outside the package instead.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as fh:
        fh.write(ZIPAPP_SHEBANG)
        with zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as zf:
            written: set[str] = set()
            package_dirs: set[Path] = set()

            for source_path in sorted(PACKAGE_ROOT.rglob("*")):
                if source_path.is_dir():
                    continue
                if "__pycache__" in source_path.parts:
                    continue
                relative_path = source_path.relative_to(PACKAGE_ROOT)
                if relative_path.parts[0] in EXCLUDED_TOP_LEVEL_DIRS:
                    continue
                arcname = (Path("missingbench") / relative_path).as_posix()
                zf.write(source_path, arcname)
                written.add(arcname)
                package_dirs.update(relative_path.parent.parents)
                package_dirs.add(relative_path.parent)

            # zipimport requires an explicit `__init__.py` per package -- unlike
            # the filesystem, it does not resolve PEP 420 implicit namespace
            # packages (e.g. `missingbench/utils/` here has no `__init__.py` on disk
            # but still needs one inside the zip for `import missingbench.utils` to work).
            for package_dir in package_dirs:
                init_arcname = (Path("missingbench") / package_dir / "__init__.py").as_posix()
                if init_arcname not in written:
                    zf.writestr(init_arcname, "")
                    written.add(init_arcname)

    return output_path


def build_export_package(
    model: AbstractModel,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    run_suffix: str | None = None,
    preprocessing_config: dict[str, object] | None = None,
) -> Path:
    """Package a trained model into a single self-contained `.missingbenchpkg` archive.

    The archive contains the model's HF-style config, the weights-only
    checkpoint, and a `.pyz` zipapp of the `missingbench` package (real files, not
    references, minus `missingbench/testing/` -- see `build_missingbench_zipapp`) so it
    can be reconstructed and called with no dependency on the rest of this
    repository being installed.

    `preprocessing_config`, when given, is the exact crop/clamp/norm/dataset_type
    settings this run's training data was preprocessed with (see
    `missingbench.preprocessing.config`). Embedding it lets a downstream harness
    preprocess raw cases the same way the model was trained, instead of
    requiring the operator to separately know and reproduce those settings.
    It's optional and omitted from the manifest entirely when not given, so
    older/manual callers of this function keep working unchanged.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hf_config = model.get_hf_config()
    manifest = {
        "schema_version": "1.0",
        "missingbench_version": __version__,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_suffix": run_suffix,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    if preprocessing_config is not None:
        manifest["preprocessing"] = preprocessing_config

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        zf.writestr("config.json", json.dumps(hf_config, indent=2, sort_keys=True) + "\n")
        zf.write(checkpoint_path, "weights/final_weights_only.safetensors")

        zipapp_path = output_path.with_suffix(".missingbench.pyz.tmp")
        try:
            build_missingbench_zipapp(zipapp_path)
            zf.write(zipapp_path, "code/missingbench.pyz")
        finally:
            zipapp_path.unlink(missing_ok=True)

    return output_path
