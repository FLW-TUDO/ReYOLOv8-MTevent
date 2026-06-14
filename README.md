# Benchmarking Recurrent Event-Based Object Detection for Industrial Multi-Class Recognition on MTEvent

Accepted at the [Neuromorphic Field Robotics and Automation Workshop, ICRA 2026](https://nfr-icra2026.com/)  
Lokeshwaran Manohar, Moritz Roidl — TU Dortmund University

**[PAPER](https://arxiv.org/abs/2603.21787)** | **[POSTER](ICRA2026_poster_revised_compact.pdf)**

Built on [ReYOLOv8](https://github.com/silvada95/ReYOLOv8).

---

## Results on MTEvent Validation Split

| Model | Init | Clip | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8s baseline | Scratch | 1 | 0.260 | 0.138 |
| ReYOLOv8s | Scratch | 3 | 0.262 | 0.148 |
| ReYOLOv8s | Scratch | 7 | 0.271 | 0.152 |
| ReYOLOv8s | Scratch | 11 | 0.261 | 0.141 |
| ReYOLOv8s | Scratch | 21 | 0.285 | 0.155 |
| ReYOLOv8s | GEN1 | 3 | 0.293 | 0.157 |
| ReYOLOv8s | GEN1 | 7 | 0.324 | 0.181 |
| ReYOLOv8s | GEN1 | 11 | 0.324 | 0.178 |
| **ReYOLOv8s** | **GEN1** | **21** | **0.329** | **0.164** |
| ReYOLOv8s | PEDRo | 11 | 0.251 | 0.134 |
| ReYOLOv8s | PEDRo | 21 | 0.258 | 0.138 |

---

## Setup

```bash
conda create -n reyolov8 python=3.9 && conda activate reyolov8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Training

```bash
# GEN1 pretrained — best result
WANDB_MODE=disabled python train.py \
  --model weights/gen1/reyolov8s_gen1_rps.pt \
  --data vtei_mtevent_50ms.yaml --hyp default_gen1.yaml \
  --device 0 --batch 16 --epochs 150 \
  --channels 5 --clip_length 21 --clip_stride 5 \
  --optimizer AdamW --cos_lr --nbs 64 --name gen1_c21

# Scratch recurrent
WANDB_MODE=disabled python train.py \
  --model ultralytics/models/v8/Recurrent/ReYOLOV8s.yaml \
  --data vtei_mtevent_50ms.yaml --hyp default_gen1.yaml \
  --device 0 --batch 16 --epochs 150 \
  --channels 5 --clip_length 21 --clip_stride 5 \
  --optimizer AdamW --cos_lr --nbs 64 --name scratch_c21

# Non-recurrent YOLOv8s baseline
WANDB_MODE=disabled python train.py \
  --model ultralytics/models/v8/yolov8s_5ch.yaml \
  --data vtei_mtevent_50ms.yaml --hyp default_gen1.yaml \
  --device 0 --batch 16 --epochs 150 \
  --channels 5 --clip_length 1 --clip_stride 1 \
  --optimizer AdamW --cos_lr --nbs 64 --name yolov8s_baseline
```

GEN1 and PEDRo pretrained weights: [ReYOLOv8 repository](https://github.com/silvada95/ReYOLOv8).

---

## Evaluation

```bash
WANDB_MODE=disabled python val.py \
  --model runs/train/gen1_c21/weights/best.pt \
  --data vtei_mtevent_50ms.yaml --cfg default_gen1.yaml \
  --device 0 --split val --channels 5 --clip_length 21 --clip_stride 5
```

---

## Zero-Shot Human Detection

```bash
python scripts/zeroshot_human_eval.py \
  --data vtei_mtevent_50ms.yaml --device cuda:0 \
  --clip_length 11 --clip_stride 5
```

---

## Citation

```bibtex
@article{manohar2026mtevent,
  title   = {Benchmarking Recurrent Event-Based Object Detection for
             Industrial Multi-Class Recognition on MTEvent},
  author  = {Manohar, Lokeshwaran and Roidl, Moritz},
  journal = {arXiv preprint arXiv:2603.21787},
  year    = {2026}
}
```

## License

GPL-3.0 — see [ReYOLOv8](https://github.com/silvada95/ReYOLOv8) and [Ultralytics](https://github.com/ultralytics/ultralytics).
