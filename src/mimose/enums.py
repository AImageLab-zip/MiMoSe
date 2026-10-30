from enum import StrEnum

from mimose.utils.cli_overrides import CONFIG_NONE


class DatasetType(StrEnum):
    BRATS18 = "brats18"
    BRATS23 = "brats23"
    BRATS25 = "brats25"
    INTERNAL = "internal"


class CropMode(StrEnum):
    NONE = CONFIG_NONE
    CENTER = "center"
    NON_EMPTY = "non_empty"


class ClampMode(StrEnum):
    NONE = "none"
    SUBJECT = "subject"
    DATASET = "dataset"


class NormMode(StrEnum):
    NONE = "none"
    MIN_MAX = "min_max"
    SUBJECT_ZSCORE = "subject_zscore"
    DATASET_ZSCORE = "dataset_zscore"


class TrainerKind(StrEnum):
    IMFUSE = "imfuse"
    DCSEG = "dcseg"
    UHVED = "uhved"
    ROBUSTSEG = "robustseg"
    SHASPEC = "shaspec"
    M3AE = "m3ae"
    M3FECON = "m3fecon"
    SRMNET = "srmnet"
    IMS2TRANS = "ims2trans"
    MSTKDNET = "mstkdnet"
    MIFPN = "mifpn"
    RFL = "rfl"
    LCKD = "lckd"
    MANYMIMOSAS = "manymimosas"
    MANYMIMOSASKD = "manymimosaskd"
    MCPL = "mcpl"
    MAMBAVITAKD = "mambavitakd"
    PARETO = "pareto"
    ADAMM = "adamm"
    NAPSCAF = "napscaf"
    ALORAMOE = "aloramoe"


class ModelKind(StrEnum):
    IMFUSE = "imfuse"
    MMFORMER = "mmformer"
    DCSEG = "dcseg"
    RFNET = "rfnet"
    A2FSEG = "a2fseg"
    TINYMIMOSA = "tinymimosa"
    SIMPLEUNET = "simpleunet"
    ELASTICUNET = "elasticunet"
    MANYMIMOSAS = "manymimosas"
    MANYMIMOSASKD = "manymimosaskd"
    UHVED = "uhved"
    ROBUSTSEG = "robustseg"
    M2FTRANS = "m2ftrans"
    SFUSION = "sfusion"
    SHASPEC = "shaspec"
    M3AE = "m3ae"
    M3FECON = "m3fecon"
    SRMNET = "srmnet"
    MMMVIT = "mmmvit"
    IMS2TRANS = "ims2trans"
    MSTKDNET = "mstkdnet"
    MIFPN = "mifpn"
    RFL = "rfl"
    UNETMFI = "unetmfi"
    LCKD = "lckd"
    INOUTFUSION = "inoutfusion"
    MAMBAVITAKD = "mambavitakd"
    MCPL = "mcpl"
    ADAMM = "adamm"
    NAPSCAF = "napscaf"
    ALORAMOE = "aloramoe"


class TransformKind(StrEnum):
    IMFUSE = "imfuse"
    TINYMIMOSA = "tinymimosa"
    SHASPEC = "shaspec"
    ALORAMOE = "aloramoe"


class OptimizerKind(StrEnum):
    RADAM = "radam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAM = "adam"


class SchedulerKind(StrEnum):
    POLY = "poly"
    WARMUPPOLY = "warmuppoly"
    COSINE = "cosine"
    STEP = "step"
    MULTISTEP = "multistep"
    PLATEAU = "plateau"


class LossKind(StrEnum):
    IMFUSE = "imfuse"
    TINYMIMOSA = "tinymimosa"
    UHVED = "uhved"
    ROBUSTSEG = "robustseg"
    SHASPEC = "shaspec"
    M3AE = "m3ae"
    M3FECON = "m3fecon"
    SRMNET = "srmnet"
    IMS2TRANS = "ims2trans"
    MSTKDNET = "mstkdnet"
    MIFPN = "mifpn"
    RFL = "rfl"
    LCKD = "lckd"
    INOUTFUSION = "inoutfusion"
    MAMBAVITAKD = "mambavitakd"
    MANYMIMOSASKD = "manymimosaskd"
    MCPL = "mcpl"
    ADAMM = "adamm"
    NAPSCAF = "napscaf"
    ALORAMOE = "aloramoe"


class MaskingMode(StrEnum):
    RANDOM = "random"
    VALIDATION = "validation"
    TEST = "test"
    FULL = "full"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
