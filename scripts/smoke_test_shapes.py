#!/usr/bin/env python3
"""Generate toy shapes data, train briefly, and run inference (single GPU, no DDP)."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import datasets
import models
from datasets.color_utils import label_to_rgb


def make_optimizer(param_list, optimizer_spec):
    name = optimizer_spec['name']
    args = optimizer_spec['args']
    if name == 'adamw':
        return torch.optim.AdamW(param_list, **args)
    if name == 'adam':
        return torch.optim.Adam(param_list, **args)
    raise ValueError(f'Unsupported optimizer: {name}')


def calc_miou(y_pred, y_true, num_classes=3):
    if y_pred.dim() == 4:
        y_pred = y_pred.squeeze(1)
    if y_true.dim() == 4:
        y_true = y_true.squeeze(1)
    y_pred = y_pred.long().cpu()
    y_true = y_true.long().cpu()
    ious = []
    for cls in range(num_classes):
        inter = union = 0
        for i in range(y_true.shape[0]):
            pred_c = y_pred[i] == cls
            true_c = y_true[i] == cls
            inter += (pred_c & true_c).sum().item()
            union += (pred_c | true_c).sum().item()
        ious.append(inter / union if union else float('nan'))
    valid = [v for v in ious if v == v]
    miou = float(np.mean(valid)) if valid else 0.0
    return miou, *(0.0 if ious[i] != ious[i] else float(ious[i]) for i in range(3))


def generate_data(out_dir: Path, size: int):
    script = ROOT / 'scripts' / 'generate_shapes_dataset.py'
    subprocess.check_call([
        sys.executable, str(script),
        '--out', str(out_dir),
        '--size', str(size),
        '--train-count', '40',
        '--val-count', '10',
    ])


def build_loader(spec):
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    return DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def load_checkpoint(model, ckpt_path: Path):
    if not ckpt_path.is_file():
        print(f'[warn] SAM checkpoint not found: {ckpt_path} (training from random init)')
        return
    state = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}')


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        inp = batch['inp'].to(device)
        gt = batch['gt'].to(device)
        logits = model.infer(inp)
        pred = logits.argmax(dim=1)
        preds.append(pred.cpu())
        gts.append(gt.cpu())
    pred_all = torch.cat(preds, dim=0)
    gt_all = torch.cat(gts, dim=0)
    return calc_miou(pred_all, gt_all)


def save_prediction_grid(model, loader, device, out_path: Path, n=4):
    model.eval()
    batch = next(iter(loader))
    inp = batch['inp'][:n].to(device)
    gt = batch['gt'][:n]
    pred = model.infer(inp).argmax(dim=1).cpu()

    tiles = []
    for i in range(min(n, inp.shape[0])):
        img = inp[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.clip(img * std + mean, 0, 1)
        img_u8 = (img * 255).astype(np.uint8)

        gt_rgb = label_to_rgb(gt[i].numpy())
        pred_rgb = label_to_rgb(pred[i].numpy())

        row = np.concatenate([img_u8, gt_rgb, pred_rgb], axis=1)
        tiles.append(row)
    grid = np.concatenate(tiles, axis=0)
    Image.fromarray(grid).save(out_path)
    print(f'Saved visualization: {out_path}')


def train_epochs(model, train_loader, optimizer, device, epochs):
    model.train()
    model.optimizer = optimizer
    for epoch in range(1, epochs + 1):
        losses = []
        for batch in train_loader:
            inp = batch['inp'].to(device)
            gt = batch['gt'].to(device)
            model.set_input(inp, gt)
            model.optimize_parameters()
            losses.append(model.loss_G.item())
        print(f'epoch {epoch}/{epochs}  loss={np.mean(losses):.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/shapes-demo.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--skip-generate', action='store_true')
    args = parser.parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    out_data = ROOT / 'load' / 'shapes_demo'
    if not args.skip_generate:
        generate_data(out_data, size=config['model']['args']['inp_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_loader = build_loader(config['train_dataset'])
    val_loader = build_loader(config['val_dataset'])
    sample = next(iter(train_loader))
    print('sample inp:', tuple(sample['inp'].shape), 'gt:', tuple(sample['gt'].shape),
          'gt unique:', torch.unique(sample['gt']).tolist())

    model = models.make(config['model']).to(device)
    ckpt = ROOT / config['sam_checkpoint']
    load_checkpoint(model, ckpt)

    optimizer = make_optimizer(model.parameters(), config['optimizer'])
    epochs = args.epochs or config.get('epoch_max', 5)

    miou0, *_ = evaluate(model, val_loader, device)
    print(f'val mIoU before training: {miou0:.4f}')

    train_epochs(model, train_loader, optimizer, device, epochs)

    miou1, iou0, iou1, iou2 = evaluate(model, val_loader, device)
    print(f'val mIoU after training:  {miou1:.4f}  (class0={iou0:.4f}, class1={iou1:.4f}, class2={iou2:.4f})')

    vis_dir = ROOT / 'save' / '_shapes_smoke'
    vis_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = vis_dir / 'model_smoke.pth'
    torch.save(model.state_dict(), ckpt_out)
    save_prediction_grid(model, val_loader, device, vis_dir / 'pred_grid.png')

    # test.py-style inference on saved weights
    model2 = models.make(config['model']).to(device)
    model2.load_state_dict(torch.load(ckpt_out, map_location=device))
    miou2, *_ = evaluate(model2, val_loader, device)
    print(f'val mIoU reloaded checkpoint: {miou2:.4f}')

    ok = miou1 > miou0 or miou1 > 0.35
    print('SMOKE TEST', 'PASSED' if ok else 'FAILED (mIoU did not improve enough)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
