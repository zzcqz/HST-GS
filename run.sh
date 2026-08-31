#!/usr/bin/env bash
# Run from HST-GS repo root.
set -euo pipefail

# libcudnn_* needs libnvrtc.so from the same CUDA install; conda libs are often not on the default path.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

DATASETS=(
    # "./data/Deep_Blending/drjohnson"
    # "./data/Deep_Blending/playroom"
    # "./data/Tanks_Temples/Train"
    # "./data/Tanks_Temples/Truck"
    "./data/Mip-NeRF360/bicycle"
    # "./data/Mip-NeRF360/bonsai"
    # "./data/Mip-NeRF360/counter"
    # "./data/Mip-NeRF360/flowers"
    # "./data/Mip-NeRF360/garden"
    # "./data/Mip-NeRF360/kitchen"
    # "./data/Mip-NeRF360/room"
    # "./data/Mip-NeRF360/stump"
    # "./data/Mip-NeRF360/treehill"
)

for dataset in "${DATASETS[@]}"; do
    echo "========== Processing dataset: ${dataset} =========="

    echo ">>> Running training..."
    python ./train.py \
        -s "${dataset}" \
        -m "${dataset}/speedV2" \
        -r 4 \
        --eval \
        --disable_tensorboard \
        --fast_train \
        --sr

    echo ">>> Running rendering..."
    python ./render.py \
        -m "${dataset}/speedV2"

    echo ">>> Running metrics..."
    python ./metrics.py \
        -m "${dataset}/speedV2"

    echo "========== Finished dataset: ${dataset} =========="
    echo
done
