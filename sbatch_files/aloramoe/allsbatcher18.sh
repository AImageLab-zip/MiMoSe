#!/usr/bin/env bash
# Submit every combined (train+test) fold script for BraTS18 ("_18_") in all/,
# each 4x with singleton dependency so the 4 same-named jobs run back-to-back
# (train resumes across the 24h wall limit; the run that finishes training also tests).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
all_dir="${script_dir}/all"

shopt -s nullglob
scripts=("${all_dir}"/*_18_*.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No _18_ all scripts found in ${all_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    for i in 1 2 3 4; do
        echo "Submitting ${script} (${i}/4)"
        sbatch --dependency=singleton "${script}"
    done
done
