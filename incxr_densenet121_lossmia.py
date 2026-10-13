#!/usr/bin/env python3
"""
IN-CXR + DenseNet-121 + Loss Membership Inference Attack

Purpose
-------
1. Reuse the existing IN-CXR candidate ordering and target membership mask.
2. Train DenseNet-121 from scratch on target members only.
3. Compute per-example cross-entropy loss for all 12,600 candidates.
4. Use -loss as the membership score (higher => more likely member).
5. Report AUC and TPR at FPR = 1%, 0.1%, 0.01%.
6. Save checkpoint, raw scores, ROC curve, and JSON metrics.

Expected existing files
-----------------------
exp/incxr_16/candidates.csv
exp/incxr_16/labels.npy
exp/incxr_16/target_keep.npy

Default IN-CXR image root
-------------------------
data/IN-CXR/

Training defaults match the established IN-CXR target setup:
- DenseNet-121 from scratch
- image_size = 224
- pad = 28
- resize -> reflected pad -> random crop -> horizontal flip
- epochs = 100
- batch_size = 64
- SGD lr = 0.1, momentum = 0.9
- weight_decay = 5e-4 on matrix/tensor weights only
- cosine annealing
- AMP on CUDA
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic where practical.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------
# Candidate path loading
# ---------------------------------------------------------------------

PATH_COLUMN_CANDIDATES = (
    "path",
    "filepath",
    "file_path",
    "image_path",
    "filename",
    "file",
    "image",
)


def _build_unique_basename_map(data_root: Path):
    mapping = {}
    duplicates = set()

    for p in data_root.rglob("*"):
        if not p.is_file():
            continue

        name = p.name
        if name in mapping:
            duplicates.add(name)
        else:
            mapping[name] = p

    for name in duplicates:
        mapping.pop(name, None)

    return mapping


def _resolve_one_path(raw_value: str,
                      csv_dir: Path,
                      data_root: Path,
                      basename_map):
    raw_value = str(raw_value).strip()
    p = Path(raw_value)

    attempts = []

    if p.is_absolute():
        attempts.append(p)
    else:
        attempts.extend([
            p,
            csv_dir / p,
            data_root / p,
        ])

    for candidate in attempts:
        if candidate.is_file():
            return candidate.resolve()

    # Last safe fallback: unique basename match.
    if p.name in basename_map:
        return basename_map[p.name].resolve()

    raise FileNotFoundError(
        f"Could not resolve candidate image path: {raw_value}"
    )


def load_candidate_paths(csv_path: Path,
                         data_root: Path,
                         expected_n: int):
    """
    Reads candidates.csv without silently changing candidate order.
    The target_keep.npy membership mask is meaningful only in this order.
    """
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != expected_n:
        raise ValueError(
            f"{csv_path} has {len(rows)} rows, "
            f"but labels/membership expect {expected_n}."
        )

    fieldnames = reader.fieldnames or []
    lower_to_original = {
        c.lower().strip(): c for c in fieldnames
    }

    path_col = None

    for c in PATH_COLUMN_CANDIDATES:
        if c in lower_to_original:
            path_col = lower_to_original[c]
            break

    # If no conventional column name exists, try to identify a column
    # whose values look like image paths. We do NOT re-sort anything.
    if path_col is None:
        image_exts = {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
        }

        for c in fieldnames:
            vals = [
                str(rows[i].get(c, "")).strip()
                for i in range(min(20, len(rows)))
            ]
            if vals and sum(
                Path(v).suffix.lower() in image_exts for v in vals
            ) >= max(1, len(vals) // 2):
                path_col = c
                break

    if path_col is None:
        raise ValueError(
            "Could not identify the image-path column in candidates.csv. "
            f"Columns found: {fieldnames}"
        )

    basename_map = _build_unique_basename_map(data_root)

    paths = [
        _resolve_one_path(
            row[path_col],
            csv_path.parent,
            data_root,
            basename_map,
        )
        for row in rows
    ]

    return paths, path_col


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class IN_CXR_Candidates(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

        if len(self.paths) != len(self.labels):
            raise ValueError("paths and labels have different lengths")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.paths[idx]

        with Image.open(path) as img:
            # DenseNet expects 3 channels.
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        y = int(self.labels[idx])
        return img, y, idx


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------

def make_transforms(image_size: int, pad: int):
    # Training:
    # resize -> reflected pad -> random crop -> horizontal flip
    train_tf = transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.Pad(
            pad,
            padding_mode="reflect",
        ),
        transforms.RandomCrop(
            (image_size, image_size)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),

        # No ImageNet normalization: model is trained from scratch.
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
    ])

    return train_tf, eval_tf


# ---------------------------------------------------------------------
# Model and optimizer
# ---------------------------------------------------------------------

def build_densenet121(num_classes: int):
    model = models.densenet121(weights=None)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(
        in_features,
        num_classes,
    )
    return model


def make_sgd(model,
             lr: float,
             momentum: float,
             weight_decay: float):
    """
    Apply weight decay to matrix/tensor weights only.
    Biases and normalization vectors get no weight decay.
    """
    decay = []
    no_decay = []

    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim > 1:
            decay.append(p)
        else:
            no_decay.append(p)

    param_groups = [
        {
            "params": decay,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.SGD(
        param_groups,
        lr=lr,
        momentum=momentum,
        nesterov=False,
    )

    return optimizer


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------

def train_one_epoch(model,
                    loader,
                    optimizer,
                    scaler,
                    device,
                    grad_accum: int,
                    use_amp: bool):

    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (x, y, _) in enumerate(loader):
        x = x.to(
            device,
            non_blocking=True,
        )
        y = y.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(x)
            raw_loss = criterion(logits, y)
            loss = raw_loss / grad_accum

        scaler.scale(loss).backward()

        do_step = (
            ((step + 1) % grad_accum == 0)
            or (step + 1 == len(loader))
        )

        if do_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = y.size(0)
        total_loss += raw_loss.item() * bs
        total_correct += (
            logits.argmax(dim=1) == y
        ).sum().item()
        total_seen += bs

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.inference_mode()
def evaluate_all(model,
                 loader,
                 device,
                 n_examples: int,
                 use_amp: bool):

    model.eval()

    losses = np.empty(
        n_examples,
        dtype=np.float64,
    )
    predictions = np.empty(
        n_examples,
        dtype=np.int64,
    )
    true_class_probability = np.empty(
        n_examples,
        dtype=np.float64,
    )

    criterion = nn.CrossEntropyLoss(
        reduction="none"
    )

    for x, y, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        y = y.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(x)
            batch_losses = criterion(
                logits,
                y,
            )

        probs = torch.softmax(
            logits.float(),
            dim=1,
        )

        true_probs = probs.gather(
            1,
            y.view(-1, 1),
        ).squeeze(1)

        batch_pred = logits.argmax(dim=1)

        idx_np = idx.numpy()

        losses[idx_np] = (
            batch_losses.float()
            .cpu()
            .numpy()
        )
        predictions[idx_np] = (
            batch_pred.cpu()
            .numpy()
        )
        true_class_probability[idx_np] = (
            true_probs.cpu()
            .numpy()
        )

    return (
        losses,
        predictions,
        true_class_probability,
    )


# ---------------------------------------------------------------------
# MIA metrics
# ---------------------------------------------------------------------

def tpr_at_or_below_fpr(fpr,
                        tpr,
                        thresholds,
                        target_fpr):
    """
    Conservative empirical operating point:
    choose the point with the largest observed FPR <= requested FPR.

    This is preferable to interpolation at very low FPR because
    interpolation can invent operating points not supported by the
    finite nonmember sample.
    """
    valid = np.where(fpr <= target_fpr)[0]

    if len(valid) == 0:
        return {
            "tpr": 0.0,
            "actual_fpr": 0.0,
            "threshold": float("inf"),
        }

    # Among valid points, maximize TPR.
    best_local = valid[
        np.argmax(tpr[valid])
    ]

    return {
        "tpr": float(tpr[best_local]),
        "actual_fpr": float(
            fpr[best_local]
        ),
        "threshold": float(
            thresholds[best_local]
        ),
    }


def compute_loss_mia(membership,
                     losses):
    """
    Lower CE loss => more likely a member.
    Therefore attack score = -loss.
    """
    membership = np.asarray(
        membership,
        dtype=np.int64,
    )
    score = -np.asarray(
        losses,
        dtype=np.float64,
    )

    auc = roc_auc_score(
        membership,
        score,
    )

    fpr, tpr, thresholds = roc_curve(
        membership,
        score,
        pos_label=1,
        drop_intermediate=False,
    )

    operating_points = {}

    requested = {
        "1%": 0.01,
        "0.1%": 0.001,
        "0.01%": 0.0001,
    }

    n_nonmembers = int(
        (membership == 0).sum()
    )
    min_nonzero_fpr = (
        1.0 / n_nonmembers
        if n_nonmembers > 0
        else math.nan
    )

    for name, target in requested.items():
        point = tpr_at_or_below_fpr(
            fpr,
            tpr,
            thresholds,
            target,
        )

        point["requested_fpr"] = target
        point["empirically_resolvable"] = bool(
            target >= min_nonzero_fpr
        )

        operating_points[name] = point

    # Descriptive best balanced threshold.
    # This is not an independently calibrated deployment threshold.
    best_bacc = -1.0
    best_threshold = None

    finite_thresholds = thresholds[
        np.isfinite(thresholds)
    ]

    for th in finite_thresholds:
        pred_member = (
            score >= th
        ).astype(np.int64)

        bacc = balanced_accuracy_score(
            membership,
            pred_member,
        )

        if bacc > best_bacc:
            best_bacc = float(bacc)
            best_threshold = float(th)

    member_losses = losses[
        membership == 1
    ]
    nonmember_losses = losses[
        membership == 0
    ]

    metrics = {
        "attack": "loss_mia",
        "score_definition": "-cross_entropy_loss",
        "auc": float(auc),
        "best_balanced_accuracy":
            best_bacc,
        "best_balanced_threshold":
            best_threshold,
        "n_members": int(
            (membership == 1).sum()
        ),
        "n_nonmembers": n_nonmembers,
        "minimum_nonzero_empirical_fpr":
            float(min_nonzero_fpr),
        "operating_points":
            operating_points,
        "member_loss_mean":
            float(member_losses.mean()),
        "member_loss_median":
            float(np.median(member_losses)),
        "nonmember_loss_mean":
            float(nonmember_losses.mean()),
        "nonmember_loss_median":
            float(np.median(nonmember_losses)),
        "loss_gap_nonmember_minus_member":
            float(
                nonmember_losses.mean()
                - member_losses.mean()
            ),
    }

    return (
        score,
        fpr,
        tpr,
        thresholds,
        metrics,
    )


# ---------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "data/IN-CXR"
        ),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path(
            "exp/incxr_16"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/incxr_densenet121"
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=28,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=314159265,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    split_dir = args.split_dir
    candidates_csv = (
        split_dir / "candidates.csv"
    )
    labels_path = (
        split_dir / "labels.npy"
    )
    target_keep_path = (
        split_dir / "target_keep.npy"
    )

    for required in (
        candidates_csv,
        labels_path,
        target_keep_path,
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required}"
            )

    labels = np.load(
        labels_path
    ).astype(np.int64)

    target_keep = np.load(
        target_keep_path
    ).astype(bool)

    if labels.ndim != 1:
        labels = labels.reshape(-1)

    if target_keep.ndim != 1:
        target_keep = (
            target_keep.reshape(-1)
        )

    if len(labels) != len(target_keep):
        raise ValueError(
            "labels.npy and target_keep.npy "
            "have different lengths."
        )

    n_candidates = len(labels)
    n_members = int(
        target_keep.sum()
    )
    n_nonmembers = (
        n_candidates - n_members
    )

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Require contiguous labels 0..C-1.
    expected_labels = np.arange(
        num_classes
    )

    if not np.array_equal(
        unique_labels,
        expected_labels,
    ):
        raise ValueError(
            "labels.npy must contain contiguous "
            f"class IDs 0..C-1. Found: "
            f"{unique_labels.tolist()}"
        )

    paths, path_column = (
        load_candidate_paths(
            candidates_csv,
            args.data_root,
            n_candidates,
        )
    )

    # Print split before doing expensive work.
    print()
    print("=" * 72)
    print(
        "IN-CXR — DenseNet-121 "
        "— LOSS MIA"
    )
    print("=" * 72)
    print(
        f"Candidates          : "
        f"{n_candidates}"
    )
    print(
        f"Target members      : "
        f"{n_members}"
    )
    print(
        f"Target nonmembers   : "
        f"{n_nonmembers}"
    )
    print(
        f"Classes             : "
        f"{num_classes}"
    )
    print(
        f"Candidate path col  : "
        f"{path_column}"
    )
    print(
        f"Image size          : "
        f"{args.image_size}"
    )
    print(
        f"Epochs              : "
        f"{args.epochs}"
    )
    print(
        f"Batch size          : "
        f"{args.batch_size}"
    )
    print(
        f"Grad accumulation   : "
        f"{args.grad_accum}"
    )
    print(
        f"Effective batch     : "
        f"{args.batch_size * args.grad_accum}"
    )
    print(
        f"Seed                : "
        f"{args.seed}"
    )

    if n_nonmembers > 0:
        min_fpr = 1.0 / n_nonmembers
        print(
            f"Min nonzero FPR     : "
            f"{100 * min_fpr:.6f}%"
        )

        if 0.0001 < min_fpr:
            print(
                "NOTE: requested FPR 0.01% "
                "is below empirical FPR "
                "resolution for this number "
                "of nonmembers."
            )

    print("=" * 72)

    train_tf, eval_tf = make_transforms(
        args.image_size,
        args.pad,
    )

    train_dataset = IN_CXR_Candidates(
        paths,
        labels,
        transform=train_tf,
    )

    eval_dataset = IN_CXR_Candidates(
        paths,
        labels,
        transform=eval_tf,
    )

    member_indices = np.flatnonzero(
        target_keep
    ).tolist()

    train_subset = Subset(
        train_dataset,
        member_indices,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Eval in exact candidate order.
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    model = build_densenet121(
        num_classes
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    optimizer = make_sgd(
        model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
    )

    start_epoch = 0

    if args.resume is not None:
        ckpt = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            ckpt["model"]
        )

        if (
            not args.eval_only
            and "optimizer" in ckpt
        ):
            optimizer.load_state_dict(
                ckpt["optimizer"]
            )

        if (
            not args.eval_only
            and "scheduler" in ckpt
        ):
            scheduler.load_state_dict(
                ckpt["scheduler"]
            )

        start_epoch = int(
            ckpt.get("epoch", -1)
        ) + 1

        print(
            f"Loaded checkpoint: "
            f"{args.resume}"
        )

    target_dir = (
        args.output_dir / "target"
    )
    loss_dir = (
        args.output_dir / "loss_mia"
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    loss_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        target_dir
        / "target_densenet121.pt"
    )

    history = []

    if not args.eval_only:
        print()
        print(
            f"Training on device: {device}"
        )
        print(
            f"DenseNet-121 params: "
            f"{n_params:,}"
        )
        print(
            f"AMP                : "
            f"{use_amp}"
        )
        print()

        for epoch in range(
            start_epoch,
            args.epochs,
        ):
            t0 = time.time()

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                args.grad_accum,
                use_amp,
            )

            scheduler.step()

            row = {
                "epoch": epoch + 1,
                "train_loss":
                    train_metrics["loss"],
                "train_accuracy":
                    train_metrics["accuracy"],
                "lr":
                    optimizer.param_groups[0]["lr"],
                "seconds":
                    time.time() - t0,
            }

            history.append(row)

            print(
                f"Epoch "
                f"{epoch + 1:03d}/"
                f"{args.epochs:03d}  "
                f"loss="
                f"{row['train_loss']:.6f}  "
                f"acc="
                f"{100 * row['train_accuracy']:.3f}%  "
                f"lr="
                f"{row['lr']:.6g}  "
                f"time="
                f"{row['seconds']:.1f}s"
            )

        torch.save(
            {
                "model":
                    model.state_dict(),
                "optimizer":
                    optimizer.state_dict(),
                "scheduler":
                    scheduler.state_dict(),
                "epoch":
                    args.epochs - 1,
                "architecture":
                    "densenet121",
                "pretrained":
                    False,
                "num_classes":
                    num_classes,
                "num_parameters":
                    n_params,
                "image_size":
                    args.image_size,
                "pad":
                    args.pad,
                "epochs":
                    args.epochs,
                "batch_size":
                    args.batch_size,
                "grad_accum":
                    args.grad_accum,
                "effective_batch_size":
                    args.batch_size
                    * args.grad_accum,
                "lr":
                    args.lr,
                "momentum":
                    args.momentum,
                "weight_decay":
                    args.weight_decay,
                "seed":
                    args.seed,
                "split_dir":
                    str(
                        split_dir.resolve()
                    ),
                "target_keep_path":
                    str(
                        target_keep_path.resolve()
                    ),
            },
            checkpoint_path,
        )

        save_json(
            target_dir
            / "training_history.json",
            history,
        )

        print()
        print(
            f"Saved target checkpoint: "
            f"{checkpoint_path}"
        )

    else:
        # For eval-only without --resume,
        # use standard output checkpoint.
        if args.resume is None:
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    "--eval-only requested but "
                    "no checkpoint exists at "
                    f"{checkpoint_path}"
                )

            ckpt = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            model.load_state_dict(
                ckpt["model"]
            )

            print(
                f"Loaded target checkpoint: "
                f"{checkpoint_path}"
            )

    # -------------------------------------------------------------
    # Evaluate every candidate exactly once in deterministic order
    # -------------------------------------------------------------
    print()
    print("Evaluating all candidates...")

    (
        losses,
        predictions,
        true_probs,
    ) = evaluate_all(
        model,
        eval_loader,
        device,
        n_candidates,
        use_amp,
    )

    membership = (
        target_keep.astype(np.int64)
    )

    member_mask = target_keep
    nonmember_mask = ~target_keep

    member_acc = float(
        (
            predictions[member_mask]
            == labels[member_mask]
        ).mean()
    )

    nonmember_acc = float(
        (
            predictions[nonmember_mask]
            == labels[nonmember_mask]
        ).mean()
    )

    member_loss = float(
        losses[member_mask].mean()
    )

    nonmember_loss = float(
        losses[nonmember_mask].mean()
    )

    target_metrics = {
        "dataset": "IN-CXR",
        "architecture": "densenet121",
        "pretrained": False,
        "num_parameters": int(
            n_params
        ),
        "n_candidates": int(
            n_candidates
        ),
        "n_members": int(
            n_members
        ),
        "n_nonmembers": int(
            n_nonmembers
        ),
        "member_accuracy":
            member_acc,
        "nonmember_accuracy":
            nonmember_acc,
        "accuracy_gap_member_minus_nonmember":
            member_acc - nonmember_acc,
        "member_loss":
            member_loss,
        "nonmember_loss":
            nonmember_loss,
        "loss_gap_nonmember_minus_member":
            nonmember_loss - member_loss,
    }

    save_json(
        target_dir
        / "target_metrics.json",
        target_metrics,
    )

    np.savez_compressed(
        target_dir
        / "target_predictions.npz",
        candidate_index=np.arange(
            n_candidates,
            dtype=np.int64,
        ),
        true_label=labels,
        membership=membership,
        loss=losses,
        prediction=predictions,
        true_class_probability=
            true_probs,
    )

    # -------------------------------------------------------------
    # Loss MIA
    # -------------------------------------------------------------
    (
        attack_scores,
        fpr,
        tpr,
        thresholds,
        mia_metrics,
    ) = compute_loss_mia(
        membership,
        losses,
    )

    np.save(
        loss_dir / "scores.npy",
        attack_scores,
    )

    np.save(
        loss_dir / "labels.npy",
        membership,
    )

    np.savez_compressed(
        loss_dir / "roc_curve.npz",
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )

    np.savez_compressed(
        loss_dir / "per_example.npz",
        candidate_index=np.arange(
            n_candidates,
            dtype=np.int64,
        ),
        membership=membership,
        class_label=labels,
        loss=losses,
        score=attack_scores,
        true_class_probability=
            true_probs,
        prediction=predictions,
    )

    save_json(
        loss_dir / "metrics.json",
        mia_metrics,
    )

    comparison = {
        "dataset": "IN-CXR",
        "model": "DenseNet121",
        "target": target_metrics,
        "loss_mia": mia_metrics,
    }

    save_json(
        args.output_dir
        / "comparison.json",
        comparison,
    )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------
    print()
    print("=" * 72)
    print("TARGET MODEL")
    print("=" * 72)
    print(
        f"Architecture         : "
        f"DenseNet-121"
    )
    print(
        f"Parameters           : "
        f"{n_params:,}"
    )
    print(
        f"Member accuracy      : "
        f"{100 * member_acc:.3f}%"
    )
    print(
        f"Nonmember accuracy   : "
        f"{100 * nonmember_acc:.3f}%"
    )
    print(
        f"Accuracy gap         : "
        f"{100 * (member_acc - nonmember_acc):.3f}%"
    )
    print(
        f"Member loss          : "
        f"{member_loss:.6f}"
    )
    print(
        f"Nonmember loss       : "
        f"{nonmember_loss:.6f}"
    )
    print(
        f"Loss gap             : "
        f"{nonmember_loss - member_loss:.6f}"
    )

    print()
    print("=" * 72)
    print("LOSS MIA")
    print("=" * 72)
    print(
        f"AUC                  : "
        f"{mia_metrics['auc']:.6f}"
    )
    print(
        f"Best bal. accuracy   : "
        f"{mia_metrics['best_balanced_accuracy']:.6f}"
    )

    for key in (
        "1%",
        "0.1%",
        "0.01%",
    ):
        p = (
            mia_metrics[
                "operating_points"
            ][key]
        )

        print(
            f"TPR @ FPR {key:<5}      : "
            f"{100 * p['tpr']:.4f}%  "
            f"(actual FPR="
            f"{100 * p['actual_fpr']:.6f}%)"
        )

        if not p[
            "empirically_resolvable"
        ]:
            print(
                f"  NOTE: {key} FPR is "
                f"below the minimum nonzero "
                f"empirical FPR; this is "
                f"effectively a zero-FP "
                f"operating point."
            )

    print()
    print(
        f"Saved results under : "
        f"{args.output_dir}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
