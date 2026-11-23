import math

import torch
import torch.nn.functional as F

from mimose.training.transforms.base_transforms import TransformManager


class IMFuseTransform:
    def __init__(
        self,
        crop_size: tuple[int, int, int] | None = (128, 128, 128),
        rotation_degrees: float = 10.0,
        intensity_factors: tuple[float, float] = (0.1, 0.1),
        flip_probabilities: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.crop_size = crop_size
        self.rotation_degrees = rotation_degrees
        self.intensity_shift, self.intensity_scale = intensity_factors
        self.flip_probabilities = flip_probabilities

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(dtype=torch.float32)
        labels_dtype = labels.dtype

        if self.crop_size is not None:
            images, labels = self._random_crop(images, labels)
        if self.rotation_degrees > 0:
            images, labels = self._random_rotation(images, labels)
        if self.intensity_shift > 0 or self.intensity_scale > 0:
            images = self._random_intensity(images)
        if any(probability > 0 for probability in self.flip_probabilities):
            images, labels = self._random_flip(images, labels)

        return images.contiguous(), labels.to(dtype=labels_dtype).contiguous()

    def _random_crop(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_shape = images.shape[1:]
        starts = []
        for dim_size, crop_size in zip(spatial_shape, self.crop_size, strict=True):
            if dim_size < crop_size:
                raise ValueError(
                    f"Crop size {self.crop_size} is larger than input "
                    f"spatial shape {spatial_shape}"
                )
            max_start = dim_size - crop_size
            start = 0 if max_start == 0 else torch.randint(0, max_start + 1, ()).item()
            starts.append(start)

        h0, w0, d0 = starts
        h1, w1, d1 = [
            start + size for start, size in zip(starts, self.crop_size, strict=True)
        ]
        return images[:, h0:h1, w0:w1, d0:d1], labels[:, h0:h1, w0:w1, d0:d1]

    def _random_rotation(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        axis = torch.randint(0, 3, ()).item()
        angle = torch.empty((), dtype=images.dtype).uniform_(
            -self.rotation_degrees, self.rotation_degrees
        ).item()
        theta = self._build_affine_matrix(
            math.radians(angle),
            axis,
            images.device,
            images.dtype,
        )

        images_5d = images.unsqueeze(0).permute(0, 1, 4, 2, 3)
        labels_5d = labels.unsqueeze(0).permute(0, 1, 4, 2, 3).to(images.dtype)

        grid = F.affine_grid(theta.unsqueeze(0), images_5d.shape, align_corners=False)
        valid = F.grid_sample(
            torch.ones(
                (1, 1, *images_5d.shape[2:]),
                device=images.device,
                dtype=images.dtype,
            ),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ) > 0.5

        rotated_images = F.grid_sample(
            images_5d,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        rotated_labels = F.grid_sample(
            labels_5d,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

        rotated_images = rotated_images.masked_fill(~valid, -1.0)
        rotated_labels = rotated_labels.masked_fill(~valid, 0)

        images = rotated_images.permute(0, 1, 3, 4, 2).squeeze(0)
        labels = rotated_labels.permute(0, 1, 3, 4, 2).squeeze(0)
        return images, labels

    def _build_affine_matrix(
        self,
        angle_radians: float,
        axis: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sin_a = math.sin(angle_radians)
        cos_a = math.cos(angle_radians)
        theta = torch.eye(3, 4, device=device, dtype=dtype)

        if axis == 0:
            theta[1, 1] = cos_a
            theta[1, 2] = -sin_a
            theta[2, 1] = sin_a
            theta[2, 2] = cos_a
        elif axis == 1:
            theta[0, 0] = cos_a
            theta[0, 2] = sin_a
            theta[2, 0] = -sin_a
            theta[2, 2] = cos_a
        else:
            theta[0, 0] = cos_a
            theta[0, 1] = -sin_a
            theta[1, 0] = sin_a
            theta[1, 1] = cos_a

        return theta

    def _random_intensity(self, images: torch.Tensor) -> torch.Tensor:
        channels = images.shape[0]
        scale = torch.empty(
            (channels, 1, 1, 1),
            device=images.device,
            dtype=images.dtype,
        ).uniform_(
            1.0 - self.intensity_scale,
            1.0 + self.intensity_scale,
        )
        shift = torch.empty(
            (channels, 1, 1, 1),
            device=images.device,
            dtype=images.dtype,
        ).uniform_(
            -self.intensity_shift,
            self.intensity_shift,
        )
        return images * scale + shift

    def _random_flip(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for probability, image_dim, label_dim in zip(
            self.flip_probabilities,
            (1, 2, 3),
            (1, 2, 3),
            strict=True,
        ):
            if probability > 0 and torch.rand(()) < probability:
                images = torch.flip(images, dims=(image_dim,))
                labels = torch.flip(labels, dims=(label_dim,))
        return images, labels


class TinyMimosaTransform(IMFuseTransform):
    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int] = (182, 218, 182),
        features_per_stage: tuple[int, ...] = (8, 16, 32, 64),
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        spatial_multiple = 2 ** (len(features_per_stage) - 1)
        self.tile_shape = tuple(
            self._round_up_to_multiple(int(dim), spatial_multiple)
            for dim in input_shape
        )
        self.spatial_multiple = spatial_multiple

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = super().__call__(images, labels)
        return self._pad_to_compatible_shape(images, labels)

    def _pad_to_compatible_shape(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_shape = tuple(int(dim) for dim in images.shape[1:])
        target_shape = tuple(
            self._round_up_to_multiple(max(current, minimum), self.spatial_multiple)
            for current, minimum in zip(current_shape, self.tile_shape)
        )
        pad_sizes: list[int] = []
        for current, target in zip(reversed(current_shape), reversed(target_shape)):
            total_pad = max(target - current, 0)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_sizes.extend((pad_before, pad_after))
        if any(pad_sizes):
            images = F.pad(images, tuple(pad_sizes))
            labels = F.pad(labels, tuple(pad_sizes))
        return images.contiguous(), labels.contiguous()

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 0:
            raise ValueError("multiple must be positive")
        return ((value + multiple - 1) // multiple) * multiple


class IMFuseTransformManager(TransformManager):
    def __init__(self, crop_size: tuple[int, int, int] = (128, 128, 128)) -> None:
        self.crop_size = tuple(int(dim) for dim in crop_size)
        super().__init__()

    def _setup_transforms(self) -> None:
        self.train_transforms = {
            "paired": IMFuseTransform(
                crop_size=self.crop_size,
                rotation_degrees=10.0,
                intensity_factors=(0.1, 0.1),
                flip_probabilities=(0.5, 0.5, 0.5),
            ),
        }
        self.test_transforms = {
            "paired": IMFuseTransform(
                crop_size=None,
                rotation_degrees=0.0,
                intensity_factors=(0.0, 0.0),
                flip_probabilities=(0.0, 0.0, 0.0),
            ),
        }

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        mode: str = "train",
        *extra_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be 'train' or 'test'")

        transforms = self.train_transforms if mode == "train" else self.test_transforms
        transformed_images, transformed_labels = transforms["paired"](images, labels)
        transformed_extras = list(extra_tensors)
        return (transformed_images, transformed_labels, *transformed_extras)


class TinyMimosaTransformManager(TransformManager):
    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int] = (182, 218, 182),
        features_per_stage: tuple[int, ...] = (8, 16, 32, 64),
    ) -> None:
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.features_per_stage = tuple(int(ch) for ch in features_per_stage)
        super().__init__()

    def _setup_transforms(self) -> None:
        self.train_transforms = {
            "paired": TinyMimosaTransform(
                crop_size=None,
                input_shape=self.input_shape,
                features_per_stage=self.features_per_stage,
                rotation_degrees=10.0,
                intensity_factors=(0.1, 0.1),
                flip_probabilities=(0.5, 0.5, 0.5),
            ),
        }
        self.test_transforms = {
            "paired": TinyMimosaTransform(
                crop_size=None,
                input_shape=self.input_shape,
                features_per_stage=self.features_per_stage,
                rotation_degrees=0.0,
                intensity_factors=(0.0, 0.0),
                flip_probabilities=(0.0, 0.0, 0.0),
            ),
        }


class ALoRAMoETransformManager(IMFuseTransformManager):
    """Legacy ALoRA-MoE augmentation (``lora_moe_hgd/transforms.random_flip``): a random crop plus
    independent p=0.5 flips on each spatial axis; no rotation or intensity jitter."""

    def _setup_transforms(self) -> None:
        super()._setup_transforms()
        self.train_transforms = {
            "paired": IMFuseTransform(
                crop_size=self.crop_size,
                rotation_degrees=0.0,
                intensity_factors=(0.0, 0.0),
                flip_probabilities=(0.5, 0.5, 0.5),
            ),
        }
