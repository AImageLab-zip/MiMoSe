# Third-Party Notices

MissingBench is released under CC BY-NC 4.0 (see [LICENSE](LICENSE)), with the
exceptions listed below.

## Upstream baseline code (`legacy/`)

Each directory under `legacy/` contains the original implementation of a
published method, kept for reference. These files remain under the terms
chosen by their original authors. Where a directory ships a `LICENSE` file
(Apache-2.0, MIT or AGPL-3.0), that file applies to it. The ports of these
methods in `src/missingbench/` are derived from that code, and its copyright
and license notices continue to apply to them.

## M3AE-derived files (AGPL-3.0)

The upstream M3AE code (`legacy/m3ae/`) is licensed under the GNU Affero
General Public License v3.0. The following files are derived from it and are
therefore distributed under AGPL-3.0, not CC BY-NC 4.0:

- `src/missingbench/models/m3ae.py`
- `src/missingbench/losses/m3ae.py`
- `src/missingbench/losses/edice.py`
- `src/missingbench/training/trainers/m3ae.py`
- `src/missingbench/training/masking.py`
- `src/missingbench/training/legacy_augment.py`
- the `irm_min_max` function in `src/missingbench/preprocessing/normalization.py`

The full AGPL-3.0 text is in [legacy/m3ae/LICENSE](legacy/m3ae/LICENSE).

## Sample volume (`src/missingbench/data/dummy_brain.npz`)

The sample volume used by `missingbench flops` is derived from the MNI ICBM152
2009c nonlinear symmetric template (T1, T2 and PD; T1 also stands in for the
contrast-enhanced T1 and PD for FLAIR). The volumes were masked to the brain,
centre-cropped from 193x229x193 to 182x218x182 and z-scored per channel. The
segmentation is a synthetic, nested-ellipsoid tumour (labels 1 NCR, 2 ED,
3 ET) and does not come from any patient.

V.S. Fonov, A.C. Evans, R.C. McKinstry, C.R. Almli, D.L. Collins.
"Unbiased nonlinear average age-appropriate brain templates from birth to
adulthood." NeuroImage 47 (2009) S102.

    Copyright (C) 1993-2004 Louis Collins, McConnell Brain
    Imaging Centre, Montreal Neurological Institute, McGill University.
    Permission to use, copy, modify, and distribute this software and
    its documentation for any purpose and without fee is hereby granted,
    provided that the above copyright notice appear in all copies.  The
    authors and McGill University make no representations about the
    suitability of this software for any purpose.  It is provided "as
    is" without express or implied warranty.  The authors are not
    responsible for any data loss, equipment damage, property loss, or
    injury to subjects or patients resulting from the use or misuse of
    this software package.
