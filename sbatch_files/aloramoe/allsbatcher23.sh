#!/usr/bin/env bash
# Submit every combined (train+test) fold script for BraTS23 ("_23_") in all/,
# each 4x with singleton dependency so the 4 same-named jobs run back-to-back
# (train resumes across the 24h wall limit; the run that finishes training also tests).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
all_dir="${script_dir}/all"

shopt -s nullglob
scripts=("${all_dir}"/*_23_*.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No _23_ all scripts found in ${all_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    for i in 1 2 3 4 5 6 7 8 9; do
        echo "Submitting ${script} (${i}/9)"
        sbatch --nice=10 --dependency=singleton "${script}"
    done
done
