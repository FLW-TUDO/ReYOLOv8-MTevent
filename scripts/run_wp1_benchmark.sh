#!/usr/bin/env bash
# Run the full event-stream benchmark pipeline on all WP1 bags in /home/loki/bags/.
# Must be executed from /home/loki/event/ReYOLOv8/
set -euo pipefail

# System Python3 has rosbag + roslz4 (needed for bag reading/LZ4 decompression)
PYTHON_ROS=/usr/bin/python3
# Conda env has ReYOLOv8 / PyTorch (needed for inference)
PYTHON_ML=/home/loki/anaconda3/envs/reyolov8/bin/python
BAGS_DIR=/home/loki/bags
WEIGHTS=runs/train/mtevent_640x480_fixed_c21/weights/best.pt
TOPIC=/dvxplorer_left/events
CONF=0.25
OUTW=640
OUTH=480

for bag in "$BAGS_DIR"/*.bag; do
    name=$(basename "$bag" .bag)
    out_root="benchmark_results/$name"

    echo "=============================="
    echo "Processing: $name"
    echo "=============================="

    # Step 1: preprocess events → H5 (inference-only, no labels needed)
    $PYTHON_ROS scripts/mtevent_to_reyolo_h5.py \
        --bag_paths "$bag" \
        --out_root  "$out_root" \
        --split     test \
        --topic     "$TOPIC" \
        --outW "$OUTW" --outH "$OUTH"

    # Step 2: run ReYOLOv8 inference + render detections
    $PYTHON_ML scripts/infer_wp1_bags.py \
        --h5      "$out_root/images/test/mtevent_test.h5" \
        --weights "$WEIGHTS" \
        --out_dir "$out_root/detections" \
        --device  cuda:0 \
        --conf    "$CONF"

    echo "Done: $out_root/detections/"
done

echo ""
echo "All bags processed. Results in benchmark_results/"
