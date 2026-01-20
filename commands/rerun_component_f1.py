#!/usr/bin/env python3
"""
Recalcula o F1 por componente para todas as iterações da versão indicada em args.yaml.
Executa em paralelo por iteração e sobrescreve apenas os campos de componente nos
arquivos regionais `all_labels_test_metrics.yaml` e no `global_component_metrics.yaml`.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.io_operations import read_tiff, read_yaml  # noqa: E402
from src.metrics import evaluate_f1_by_component  # noqa: E402
from main import _get_component_labels  # noqa: E402


def _overlap_str(overlap: List[float]) -> str:
    """Formata o nome do arquivo de predição (ex.: 0.8)."""
    val = float(sum(overlap))
    text = f"{val:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def _load_pred_path(region_folder: Path, overlap_name: str) -> Path | None:
    pred_dir = region_folder / "raster_prediction"
    candidate = pred_dir / f"join_class_{overlap_name}.TIF"
    if candidate.exists():
        return candidate
    alts = sorted(pred_dir.glob("join_class_*.TIF"))
    return alts[0] if alts else None


def _process_iteration(
    iter_dir: Path,
    region_gt_paths: List[str],
    nb_class: int,
    overlap_name: str,
) -> str:
    """Recalcula métricas de componente para uma iteração."""
    global_gt: List[int] = []
    global_pred: List[int] = []

    for region_idx, gt_path in enumerate(region_gt_paths):
        region_folder = iter_dir / f"region_{region_idx}"
        if not region_folder.exists():
            continue

        pred_path = _load_pred_path(region_folder, overlap_name)
        if pred_path is None or not pred_path.exists():
            continue

        gt = read_tiff(gt_path)
        pred = read_tiff(pred_path)

        comp_metrics = evaluate_f1_by_component(
            pred=pred, gt=gt, num_class=nb_class, average="macro"
        )

        metrics_path = region_folder / "all_labels_test_metrics.yaml"
        existing = {}
        if metrics_path.exists():
            existing = yaml.safe_load(metrics_path.read_text()) or {}

        prefix = f"region_{region_idx}/all_labels_"
        existing.update(
            {
                f"{prefix}Accuracy_component": comp_metrics["Accuracy_component"],
                f"{prefix}avgF1_component": comp_metrics["avgF1_component"],
                f"{prefix}F1_component": comp_metrics["F1_component"],
                f"{prefix}avgPrec_component": comp_metrics["avgPrec_component"],
                f"{prefix}avgRec_component": comp_metrics["avgRec_component"],
                f"{prefix}n_components": comp_metrics["n_components"],
            }
        )

        metrics_path.write_text(
            yaml.safe_dump(existing, sort_keys=False, allow_unicode=False)
        )

        gt_labels, pred_labels = _get_component_labels(pred, gt, nb_class)
        global_gt.extend(gt_labels)
        global_pred.extend(pred_labels)

    if len(global_gt) == 0:
        return iter_dir.name

    labels = list(range(1, nb_class + 1))
    global_metrics = {
        "global/component/n_components": len(global_gt),
        "global/component/Accuracy": float(accuracy_score(global_gt, global_pred)) * 100,
        "global/component/avgF1": float(
            f1_score(global_gt, global_pred, average="macro", zero_division=0, labels=labels)
        )
        * 100,
        "global/component/F1": (
            f1_score(global_gt, global_pred, average=None, zero_division=0, labels=labels) * 100
        ).tolist(),
        "global/component/avgPrec": float(
            precision_score(
                global_gt, global_pred, average="macro", zero_division=0, labels=labels
            )
        )
        * 100,
        "global/component/avgRec": float(
            recall_score(global_gt, global_pred, average="macro", zero_division=0, labels=labels)
        )
        * 100,
    }

    global_path = iter_dir / "global_component_metrics.yaml"
    global_path.write_text(
        yaml.safe_dump(global_metrics, sort_keys=False, allow_unicode=False)
    )

    return iter_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalcula F1 por componente para todas as iterações."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de processos em paralelo (padrão: metade dos CPUs, min 1).",
    )
    args_ns = parser.parse_args()

    cfg = read_yaml(ROOT / "args.yaml")
    data_dir = (ROOT / cfg["data_path"]).resolve()

    iter_dirs = sorted(p for p in data_dir.glob("iter_*") if p.is_dir())
    if not iter_dirs:
        print(f"Nenhuma iteração encontrada em {data_dir}")
        return

    nb_class = int(cfg["nb_class"])
    overlap_name = _overlap_str(cfg["overlap"])
    region_gt_paths = [str((ROOT / p).resolve()) for p in cfg["test_segmentation_paths"]]

    default_workers = max(1, (os.cpu_count() or 2) // 2)
    workers = args_ns.workers or min(default_workers, len(iter_dirs))

    process_fn = partial(
        _process_iteration,
        region_gt_paths=region_gt_paths,
        nb_class=nb_class,
        overlap_name=overlap_name,
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for iter_name in executor.map(process_fn, iter_dirs):
            print(f"[ok] {iter_name}")


if __name__ == "__main__":
    main()

