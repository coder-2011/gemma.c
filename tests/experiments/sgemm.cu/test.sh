#!/bin/bash
set -euo pipefail

matsize_max=3072
matsize_min=2
matsize_step=1

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
experiment_dir="${repo_root}/experiments/sgemm.cu"
build_dir="${experiment_dir}/build"
save_dir="${experiment_dir}/test_results"
gpucc="${1:-86}"

mkdir -p "${save_dir}"
rm -rf "${build_dir}"
cmake -B "${build_dir}" -S "${experiment_dir}" -DGPUCC="${gpucc}"
cmake --build "${build_dir}" --target test
"${build_dir}/test" --savedir="${save_dir}" --mmax="${matsize_max}" \
  --mmin="${matsize_min}" --mstep="${matsize_step}"
