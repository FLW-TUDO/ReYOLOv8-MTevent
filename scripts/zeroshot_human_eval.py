"""
Zero-shot human detection evaluation for PEDRo and GEN1 pretrained models.

Both models have nc=2 (car=0, pedestrian=1).
MTEvent human class = 16.
Mapping: model class 1 (pedestrian) <-> GT class 16 (human).

For each model we:
  - run inference through full clip sequence
  - keep only class-1 predictions
  - keep only class-16 GT boxes, remapped to class 1
  - compute AP@0.5 and AP@0.5:0.95
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml, argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from EventVideoDataloader import build_video_val_standalone_dataloader
from ultralytics.nn.autobackend import AutoBackendMemory
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.metrics import box_iou


HUMAN_GT_CLS       = 16   # MTEvent human class index
PEDRO_HUMAN_CLS    = 0    # class 0 = person in PEDRo model (mislabeled 'car' in rps.pt, corrected in rps_fixed.pt)
GEN1_HUMAN_CLS     = 1    # class 1 = pedestrian in GEN1 model
IOU_VEC = torch.linspace(0.5, 0.95, 10)


def process_batch(detections, labels, iouv):
    """
    detections: [N,6]  x1y1x2y2 conf cls   (cls already remapped to 1)
    labels:     [M,5]  cls x1y1x2y2         (cls already remapped to 1)
    returns correct [N,10] bool
    """
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i, thresh in enumerate(iouv):
        x = torch.where((iou >= thresh) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if matches.shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=detections.device)


def evaluate(model_path, data_yaml, human_pred_cls, clip_length=11, clip_stride=5, device='cuda:0'):
    device = torch.device(device)
    iouv = IOU_VEC.to(device)

    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    val_path = data['val']

    cfg = get_cfg(DEFAULT_CFG)
    cfg.workers = 4
    cfg.imgsz  = 320

    video_config = dict(clip_length=clip_length, clip_stride=clip_stride, channels=5)
    dataloader, _ = build_video_val_standalone_dataloader(
        cfg, video_config, batch_size=1, video_path=val_path)

    model = AutoBackendMemory(model_path, device=device, fp16=False)
    model.eval()

    stats = []   # list of (correct[N,10], conf[N], pcls[N], tcls[M])
    n_gt_total = 0
    cls_counter = {}   # tally raw prediction classes

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'eval {Path(model_path).name}'):
            batch['img'] = batch['img'].float()
            hidden_states = {"0": None, "1": None, "2": None, "3": None}
            T_max = batch['img'].shape[1]

            for T in range(T_max):
                imgs_t = (batch['img'][:, T, :, :, :].to(device) / 255.0).float()
                preds, hidden_states = model(imgs_t, hidden_states)
                preds_nms = ops.non_max_suppression(
                    preds, conf_thres=0.001, iou_thres=0.7, max_det=300)

                sequence_mask = batch['vid_pos'] == T

                for si, pred in enumerate(preds_nms):
                    # tally raw predicted classes for debugging
                    if len(pred):
                        for c in pred[:, 5].int().tolist():
                            cls_counter[c] = cls_counter.get(c, 0) + 1
                    # --- GT: keep only human (class 16), remap to 1 ---
                    idx = batch['batch_idx'][sequence_mask] == si
                    gt_cls  = batch['cls'][sequence_mask][idx].squeeze(-1)
                    gt_bbox = batch['bboxes'][sequence_mask][idx]

                    human_mask = gt_cls == HUMAN_GT_CLS
                    gt_cls_h  = gt_cls[human_mask]
                    gt_bbox_h = gt_bbox[human_mask]
                    nl = human_mask.sum().item()
                    n_gt_total += nl

                    # --- Predictions: keep only target class (person/pedestrian) ---
                    if len(pred):
                        pred_mask = pred[:, 5] == human_pred_cls
                        pred = pred[pred_mask]

                    npr = len(pred)
                    correct = torch.zeros(npr, iouv.shape[0], dtype=torch.bool, device=device)

                    if npr == 0:
                        if nl:
                            stats.append((correct,
                                          torch.zeros(0, device=device),
                                          torch.zeros(0, device=device),
                                          gt_cls_h.to(device)))
                        continue

                    if nl:
                        h, w = batch['img'].shape[3], batch['img'].shape[4]
                        tbox = ops.xywh2xyxy(gt_bbox_h).to(device) * torch.tensor(
                            [w, h, w, h], device=device, dtype=torch.float)
                        labelsn = torch.cat(
                            (torch.ones(nl, 1, device=device) * human_pred_cls, tbox), 1)
                        correct = process_batch(pred, labelsn, iouv)

                    stats.append((correct,
                                  pred[:, 4],
                                  pred[:, 5],
                                  gt_cls_h.to(device)))

    print(f"  Raw prediction class counts: {dict(sorted(cls_counter.items()))}")

    if not stats:
        print("No stats collected.")
        return 0.0, 0.0

    tp_all   = torch.cat([s[0] for s in stats])      # [Ntotal, 10]
    conf_all = torch.cat([s[1] for s in stats])      # [Ntotal]

    n_preds = len(conf_all)
    n_tp50  = tp_all[:, 0].sum().item() if n_preds > 0 else 0
    print(f"  Total predictions (target class): {n_preds}")
    print(f"  TP@0.5: {int(n_tp50)}  FP@0.5: {int(n_preds - n_tp50)}")
    if n_preds == 0:
        print("  AP@0.5         : 0.0000")
        print("  AP@0.5:0.95    : 0.0000")
        return 0.0, 0.0

    # sort by confidence descending
    si = conf_all.argsort(descending=True)
    tp_all = tp_all[si]

    # AP per IoU threshold
    ap_per_iou = []
    for j in range(10):
        tp_j = tp_all[:, j].float().cpu().numpy()
        fp_j = 1 - tp_j
        tp_cum = tp_j.cumsum()
        fp_cum = fp_j.cumsum()
        rec  = tp_cum / (n_gt_total + 1e-10)
        prec = tp_cum / (tp_cum + fp_cum + 1e-10)

        # 101-point interpolation
        rec  = np.concatenate(([0.], rec,  [1.]))
        prec = np.concatenate(([1.], prec, [0.]))
        prec = np.flip(np.maximum.accumulate(np.flip(prec)))
        x = np.linspace(0, 1, 101)
        ap_per_iou.append(np.trapz(np.interp(x, rec, prec), x))

    ap50    = ap_per_iou[0]
    ap5095  = float(np.mean(ap_per_iou))

    print(f"  GT human boxes : {n_gt_total}")
    print(f"  AP@0.5         : {ap50:.4f}")
    print(f"  AP@0.5:0.95    : {ap5095:.4f}")
    return ap50, ap5095


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     default='vtei_mtevent_50ms.yaml')
    parser.add_argument('--device',   default='cuda:0')
    parser.add_argument('--clip_length', type=int, default=11)
    parser.add_argument('--clip_stride', type=int, default=5)
    args = parser.parse_args()

    print("\n=== PEDRo pretrained — zero-shot human detection ===")
    print(f"    (using class {PEDRO_HUMAN_CLS} = person in PEDRo model)")
    p50, p95 = evaluate('weights/pedro/reyolov8s_pedro_rps.pt',
                        args.data, PEDRO_HUMAN_CLS,
                        args.clip_length, args.clip_stride, args.device)

    print("\n=== GEN1 pretrained — zero-shot human detection ===")
    print(f"    (using class {GEN1_HUMAN_CLS} = pedestrian in GEN1 model)")
    g50, g95 = evaluate('weights/gen1/reyolov8s_gen1_rps.pt',
                        args.data, GEN1_HUMAN_CLS,
                        args.clip_length, args.clip_stride, args.device)

    print("\n========= SUMMARY =========")
    print(f"{'Model':<10}  AP@0.5   AP@0.5:0.95")
    print(f"{'PEDRo':<10}  {p50:.4f}   {p95:.4f}")
    print(f"{'GEN1':<10}  {g50:.4f}   {g95:.4f}")
