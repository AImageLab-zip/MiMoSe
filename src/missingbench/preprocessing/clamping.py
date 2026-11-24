import click
import numpy as np

from missingbench.preprocessing.config import ClampConfig
def check_size(images:np.ndarray):
    if len(images.shape) != 4:
        raise click.ClickException(
            f"`images` must have 4 dimensions, got shape {images.shape}."
        )

def template(images:np.ndarray,config:ClampConfig)-> np.ndarray:
    """Template function for intensity clamping on multi-modal 3D images.

    Args:
        images: Input image array with shape `(C, X, Y, Z)`, where `C` is the
            modality/channel dimension.
        config: Clamping configuration used to control the clamping behavior.

    Returns:
        The clamped image array with the same shape as `images`.
    """
    check_size(images)
    #implement your own clamping logic
    raise NotImplementedError()
    #return out
def none(images:np.ndarray,config:ClampConfig)-> np.ndarray:
    check_size(images)
    return images
def subject(images:np.ndarray,config:ClampConfig)-> np.ndarray:
    check_size(images)
    min_perc = config.percentile[0]
    max_perc = config.percentile[1]
    out = images.copy()

    for i in range(out.shape[0]):
        single_modal = out[i]
        low = np.percentile(single_modal, min_perc)
        high = np.percentile(single_modal, max_perc)
        out[i] = np.clip(single_modal, low, high)
    return out

def dataset(images:np.ndarray,config:ClampConfig)-> np.ndarray\
        :
    lows = config.min
    highs = config.max
    out = images.copy()
    for i in range(out.shape[0]):
        single_modal = out[i]
        low = lows[i]
        high = highs[i]
        out[i] = np.clip(single_modal, low, high)
    return out
