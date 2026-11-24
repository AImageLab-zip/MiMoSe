import numpy as np
import typer
import click
from missingbench.preprocessing.config import CropConfig

def template(images:np.ndarray,seg:np.ndarray,config:CropConfig)-> tuple[np.ndarray, np.ndarray]:
    """Template function for cropping multi-modal 3D images and segmentations.

    Args:
        images: Input image array with shape `(C, X, Y, Z)`, where `C` is the
            modality/channel dimension.
        seg: Input segmentation array with shape `(1, X, Y, Z)`.
        config: Crop configuration controlling the cropping behavior.

    Returns:
        A tuple `(cropped_images, cropped_seg)` containing the cropped image
        and segmentation arrays.
    """
    check_size(images,seg)
    # add cropping logic here
    # return cropped_images, cropped_seg
    raise NotImplementedError()

def check_size(images:np.ndarray,seg:np.ndarray):
    if len(images.shape) != 4:
        raise click.ClickException(
            f"`images` must have 4 dimensions, got shape {images.shape}."
        )

    if len(seg.shape) != 4:
        raise click.ClickException(
            f"`segmentations` must have 4 dimensions, got shape {seg.shape}."
        )

    mismatched_dims = [
        f"dim {i}: images={images.shape[i]}, seg={seg.shape[i]}"
        for i in (1, 2, 3)
        if images.shape[i] != seg.shape[i]
    ]

    if mismatched_dims:
        raise click.ClickException(
            "`images` and `seg` must have the same size in dimensions 1, 2, and 3. "
            f"Got images.shape={images.shape}, seg.shape={seg.shape}. "
            f"Mismatched dimensions: {', '.join(mismatched_dims)}."
        )
def center(images: np.ndarray, seg: np.ndarray, config: CropConfig) -> tuple[np.ndarray, np.ndarray]:
    """Crop images and segmentations around the spatial center."""
    check_size(images,seg)

    image_shape = images[0].shape
    crop_size = config.size

    if not isinstance(crop_size, tuple):
        raise click.ClickException(
            "crop_size must be either an int or a tuple matching the image dimensions, "
            f"got {type(crop_size).__name__}: {crop_size!r}."
        )

    if len(crop_size) != len(image_shape):
        raise click.ClickException(
            "crop_size must be either an int or a tuple with the same number of dimensions "
            f"as the image. Got crop_size={crop_size} and image_shape={image_shape}."
        )

    invalid_dims = [
        f"dim {i}: requested {crop_dim}, available {img_dim}"
        for i, (crop_dim, img_dim) in enumerate(zip(crop_size, image_shape))
        if crop_dim > img_dim
    ]

    if invalid_dims:
        raise click.ClickException(
            "crop_size must be either an int or a tuple, and it cannot exceed the image shape "
            f"in any dimension. Got crop_size={crop_size!r}, resolved crop_shape={crop_size}, "
            f"image_shape={image_shape}. Invalid dimensions: {', '.join(invalid_dims)}."
        )

    starts = [(img_dim - crop_dim) // 2 for img_dim, crop_dim in zip(image_shape, crop_size)]
    ends = [start + crop_dim for start, crop_dim in zip(starts, crop_size)]
    slices = tuple(slice(start, end) for start, end in zip(starts, ends))

    cropped_images = images[(slice(None), *slices)]
    cropped_seg = seg[(slice(None), *slices)]
    return cropped_images.astype(np.float32), cropped_seg.astype(np.uint8)

def none(images:np.ndarray,seg: np.ndarray,config:CropConfig)-> tuple[np.ndarray, np.ndarray]:
    """Return images and segmentations unchanged."""
    check_size(images,seg)
    return images.astype(np.float32), seg.astype(np.uint8)

def non_empty(
    images: np.ndarray,
    seg: np.ndarray,
    config:CropConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop images and segmentation to the non-empty bounding box.

    The bounding box is computed from non-zero voxels and expanded so each
    spatial dimension is at least `args.crop_min_size` if provided, otherwise 128.
    """
    check_size(images,seg)
    vol = images
    min_size = config.min_size

    vol_for_bbox = np.amax(vol, axis=0)

    if len(vol_for_bbox.shape) != 3:
        raise click.ClickException(
            f"Expected images with 3D or 4D shape, got {vol.shape}."
        )

    nonzeros = np.where(vol_for_bbox != 0)
    if len(nonzeros[0]) == 0:
        raise click.ClickException("Cannot crop non-empty region: image contains only zeros.")

    x_min, x_max = int(np.amin(nonzeros[0])), int(np.amax(nonzeros[0])) + 1
    y_min, y_max = int(np.amin(nonzeros[1])), int(np.amax(nonzeros[1])) + 1
    z_min, z_max = int(np.amin(nonzeros[2])), int(np.amax(nonzeros[2])) + 1

    bounds = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    shape = vol_for_bbox.shape
    expanded_bounds = []

    for (start, end), dim_size, target_size in zip(bounds, shape, min_size):
        current_size = end - start
        if current_size < target_size:
            pad = target_size - current_size
            start -= pad // 2
            end += pad - (pad // 2)

        if start < 0:
            end = min(dim_size, end - start)
            start = 0
        if end > dim_size:
            start = max(0, start - (end - dim_size))
            end = dim_size

        expanded_bounds.append((start, end))

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = expanded_bounds


    vol_out = vol[:, x_min:x_max, y_min:y_max, z_min:z_max]


    seg_out = seg.astype(np.uint8)[:,x_min:x_max, y_min:y_max, z_min:z_max]

    return vol_out.astype(np.float32), seg_out.astype(np.uint8)
