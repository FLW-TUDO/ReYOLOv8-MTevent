#!/bin/bash
# Paper Experiments A & B
# A: Test split for best model (GEN1 C11)
# B: Per-class val AP for all 7 ablation models
# Output: runs/paper_experiments/<readable_name>/

PYTHON=/home/loki/anaconda3/envs/reyolov8/bin/python
DATA=vtei_mtevent_50ms.yaml
PROJECT=runs/paper_experiments

echo "====== EXPERIMENT A: Test split — GEN1 C11 (best model) ======"
WANDB_MODE=disabled $PYTHON val.py \
  --model runs/train/mtevent_17cls_v3_adamw_focal2/weights/best.pt \
  --data $DATA \
  --split test \
  --channels 5 --clip_length 11 --clip_stride 5 \
  --device 0 --batch 16 \
  --project $PROJECT --name A_gen1_c11_test_split \
  --exist_ok

echo "====== EXPERIMENT B1: Val per-class — YOLOv8s baseline (Non-recurrent) ======"
WANDB_MODE=disabled $PYTHON val.py \
  --model runs/train/mtevent_baseline_yolov8s_5ch/weights/best.pt \
  --data $DATA \
  --split val \
  --channels 5 --clip_length 1 --clip_stride 1 \
  --device 0 --batch 16 \
  --project $PROJECT --name B1_yolov8s_baseline_val \
  --exist_ok

echo "====== EXPERIMENT B2: Val per-class — ReYOLOv8s C3 scratch ======"
WANDB_MODE=disabled $PYTHON val.py \
  --model runs/train/clip_len_03_stride_01/weights/best.pt \
  --data $DATA \
  --split val \
  --channels 5 --clip_length 3 --clip_stride 1 \
  --device 0 --batch 16 \
  --project $PROJECT --name B2_reyolov8s_c3_scratch_val \
  --exist_ok

echo "====== EXPERIMENT B3: Val per-class — ReYOLOv8s C7 scratch ======"
WANDB_MODE=disabled $PYTHON val.py \
  --model runs/train/clip_len_07_stride_03/weights/best.pt \
  --data $DATA \
  --split val \
  --channels 5 --clip_length 7 --clip_stride 3 \
  --device 0 --batch 16 \
  --project $PROJECT --name B3_reyolov8s_c7_scratch_val \
  --exist_ok

echo "====== EXPERIMENT B4: Val per-class — ReYOLOv8s C11 scratch ======"
WANDB_MODE=disabled $PYTHON val.py \
  --model runs/train/mtevent_reyolov8s_scratch/weights/best.pt \
  --data $DATA \
  --split val \
  --channels 5 --clip_length 11 --clip_stride 5 \
  --device 0 --batch 16 \
  --project $PROJECT --name B4_reyolov8s_c11_scratch_val \
  --exist_ok

echo "====== EXPERIMENT B5: Val per-class — ReYOLOv8s C21 scratch ======"
WANDB_MODE=disabled $PYTHON val.py \
  --model runs/train/clip_len_21_stride_09/weights/best.pt \
  --data $DATA \
  --split val \
  --channels 5 --clip_length 21 --clip_stride 9 \
  --device 0 --batch 16 \
  --project $PROJECT --name B5_reyolov8s_c21_scratch_val \
  --exist_ok

echo "====== EXPERIMENT B6: Val per-class — GEN1 C11 pretrained (best model) ======"
WANDB_MODE=disabled $PYTHON val.py \
  --model runs/train/mtevent_17cls_v3_adamw_focal2/weights/best.pt \
  --data $DATA \
  --split val \
  --channels 5 --clip_length 11 --clip_stride 5 \
  --device 0 --batch 16 \
  --project $PROJECT --name B6_gen1_c11_pretrained_val \
  --exist_ok

echo "====== EXPERIMENT B7: Val per-class — PEDRo C11 pretrained ======"
WANDB_MODE=disabled $PYTHON val.py \
  --model runs/train/mtevent_pedro_pretrained/weights/best.pt \
  --data $DATA \
  --split val \
  --channels 5 --clip_length 11 --clip_stride 5 \
  --device 0 --batch 16 \
  --project $PROJECT --name B7_pedro_c11_pretrained_val \
  --exist_ok

echo "====== All paper experiments complete. Results in runs/paper_experiments/ ======"
