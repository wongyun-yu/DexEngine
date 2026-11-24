#!/bin/bash
# PyTorch v2.3.1 (py3.8) built from source for RTX 5090 (sm_120), then Isaac Gym Preview 4 on that wheel: needed only to
# run ManipTrans / Isaac Gym on an sm_120 card. Recipe: CUDA_COMMON_GPU_ARCHITECTURES gains "12.0" in
# cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake (line 227), TORCH_CUDA_ARCH_LIST=12.0, nvcc >= 12.8.
# Isolation: a dedicated conda prefix env inside the workspace (build_sm120/env38) with CUDA 12.8 from the conda nvidia
# channel; no system CUDA, driver, sudo or global pip. The build runs at low priority with MAX_JOBS=6 and is paused
# (SIGSTOP) while available RAM is below 40 GB, resumed at 48 GB.
# After the wheel: install it and isaacgym (-e <workspace>/code/isaacgym_pv4/isaacgym/python) into that env and check
# that gymtorch imports on an RTX 5090 (sm_120).
set -euo pipefail
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"
W=$B/build_sm120; E=$W/env38; L=$B/logs/build_sm120.log
mkdir -p $W; exec > >(tee -a $L) 2>&1
echo "[SM120] START $(date '+%F %T') disk_before=$(du -sh $B | cut -f1) free=$(df -h / | awk 'NR==2{print $4}')"
source $HOME/miniconda3/etc/profile.d/conda.sh
[ -d $E ] || conda create -y -p $E --override-channels -c conda-forge python=3.8 cmake ninja "gcc_linux-64=12" "gxx_linux-64=12" pyyaml typing_extensions numpy=1.23.5 astunparse
conda install -y -p $E --override-channels -c nvidia/label/cuda-12.8.0 -c conda-forge cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvtx-dev cuda-cupti-dev libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev libcufft-dev
[ -d $W/pytorch ] || git clone --branch v2.3.1 --depth 1 --recursive --shallow-submodules https://github.com/pytorch/pytorch.git $W/pytorch
F=$W/pytorch/cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
grep -q '"12.0"' $F || sed -i '227s/$/\n  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "12.0")/' $F
grep -n '"12.0"' $F
# RAM gate: pause the build's process group below 40 GB available, resume at 48 GB
( while sleep 20; do pg=$(cat $W/build.pgid 2>/dev/null || true); [ -z "$pg" ] && continue
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$a" -lt 40 ]; then kill -STOP -"$pg" 2>/dev/null && echo "[SM120] PAUSE avail=${a}GB $(date +%T)"; fi
    if [ "$a" -ge 48 ]; then kill -CONT -"$pg" 2>/dev/null || true; fi
    kill -0 -"$pg" 2>/dev/null || break; done ) &
cd $W/pytorch
setsid bash -c 'echo $$ > '"$W"'/build.pgid; exec env CUDA_HOME='"$E"' PATH='"$E"'/bin:$PATH TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=6 USE_CUDA=1 USE_ROCM=0 BUILD_TEST=0 USE_DISTRIBUTED=0 \
  nice -n 19 '"$E"'/bin/python setup.py bdist_wheel'
ls -la dist/
echo "[SM120] WHEEL $(ls dist/*.whl) disk_after=$(du -sh $B | cut -f1) build_dir=$(du -sh $W | cut -f1)"
