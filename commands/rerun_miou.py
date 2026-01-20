#!/usr/bin/env python3
"""
Recalcula o Mean IoU por polígono para todas as iterações de um pipeline.

Funciona tanto para:
- Casos multi-region (e.g., /home/luizluz/Documentos/multi-task-fcn/bioflore_data/v02)
- Casos single-region (e.g., /home/luizluz/Documentos/multi-task-fcn/13_amazon_data)

Calcula apenas a métrica global por iteração, não por região.

Usage:
    python commands/rerun_miou.py --data_path /path/to/data
    python commands/rerun_miou.py --data_path bioflore_data/v02
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.io_operations import read_tiff, read_yaml  # noqa: E402
from src.metrics import get_miou_polygon_data  # noqa: E402


def _overlap_str(overlap: List[float]) -> str:
    """Format prediction file name from overlap values (e.g., 0.8)."""
    val = float(sum(overlap))
    text = f"{val:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def _find_overlap_name(pred_dir: Path) -> Optional[str]:
    """Find overlap name from existing prediction files."""
    pred_files = sorted(pred_dir.glob("join_class_*.TIF"))
    if not pred_files:
        return None
    # Extract overlap from filename like "join_class_0.8.TIF"
    name = pred_files[0].stem  # "join_class_0.8"
    overlap_part = name.replace("join_class_", "")
    return overlap_part


def _load_pred_path(folder: Path, overlap_name: str) -> Optional[Path]:
    """Load prediction path from folder."""
    pred_dir = folder / "raster_prediction"
    if not pred_dir.exists():
        return None
    candidate = pred_dir / f"join_class_{overlap_name}.TIF"
    if candidate.exists():
        return candidate
    # Try to find any prediction file
    alts = sorted(pred_dir.glob("join_class_*.TIF"))
    return alts[0] if alts else None


def _detect_structure(iter_dir: Path) -> Tuple[bool, int]:
    """
    Detect if iteration has multi-region or single-region structure.
    
    Returns:
        (is_multi_region, num_regions)
    """
    # Check for region_0 folder
    region_0 = iter_dir / "region_0"
    if region_0.exists() and region_0.is_dir():
        # Count regions
        num_regions = 0
        while (iter_dir / f"region_{num_regions}").exists():
            num_regions += 1
        return True, num_regions
    
    # Check for direct raster_prediction folder
    if (iter_dir / "raster_prediction").exists():
        return False, 1
    
    return False, 0


def _get_test_segmentation_paths(cfg: dict, data_path: Path) -> List[str]:
    """Get test segmentation paths from config, resolving relative paths."""
    if "test_segmentation_paths" in cfg:
        paths = cfg["test_segmentation_paths"]
    elif "test_segmentation_path" in cfg:
        paths = [cfg["test_segmentation_path"]]
    else:
        raise ValueError("No test_segmentation_path(s) found in config")
    
    # Resolve paths
    resolved = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            # Try relative to ROOT first, then data_path
            if (ROOT / p).exists():
                path = ROOT / p
            elif (data_path / p).exists():
                path = data_path / p
            else:
                path = ROOT / p  # Default to ROOT-relative
        resolved.append(str(path.resolve()))
    
    return resolved


def _process_iteration_multi_region(
    iter_dir: Path,
    region_gt_paths: List[str],
    nb_class: int,
    overlap_name: str,
) -> str:
    """Process iteration with multi-region structure."""
    all_iou_per_polygon = []
    all_gt_classes_per_polygon = []
    
    for region_idx, gt_path in enumerate(region_gt_paths):
        region_folder = iter_dir / f"region_{region_idx}"
        if not region_folder.exists():
            continue
        
        pred_path = _load_pred_path(region_folder, overlap_name)
        if pred_path is None or not pred_path.exists():
            continue
        
        if not Path(gt_path).exists():
            print(f"  Warning: GT path not found: {gt_path}")
            continue
        
        gt = read_tiff(gt_path)
        pred = read_tiff(pred_path)
        
        # Get per-polygon IoU data
        iou_data, gt_classes = get_miou_polygon_data(pred, gt, nb_class)
        all_iou_per_polygon.extend(iou_data)
        all_gt_classes_per_polygon.extend(gt_classes)
    
    if len(all_iou_per_polygon) == 0:
        print(f"  Warning: No polygons found in {iter_dir.name}")
        return iter_dir.name
    
    # Compute global mIoU metrics
    avg_miou = float(np.mean(all_iou_per_polygon) * 100)
    n_matched = sum(1 for iou in all_iou_per_polygon if iou > 0)
    
    # Compute mIoU per class
    miou_per_class = []
    for c in range(1, nb_class + 1):
        class_ious = [iou for iou, cls in zip(all_iou_per_polygon, all_gt_classes_per_polygon) if cls == c]
        if class_ious:
            miou_per_class.append(float(np.mean(class_ious) * 100))
        else:
            miou_per_class.append(0.0)
    
    global_miou_metrics = {
        "global/miou/avgMIoU": avg_miou,
        "global/miou/MIoU_per_class": miou_per_class,
        "global/miou/n_polygons_evaluated": len(all_iou_per_polygon),
        "global/miou/n_polygons_matched": n_matched,
    }
    
    # Save to global_miou_metrics.yaml
    global_path = iter_dir / "global_miou_metrics.yaml"
    global_path.write_text(
        yaml.safe_dump(global_miou_metrics, sort_keys=False, allow_unicode=False)
    )
    
    return f"{iter_dir.name} (mIoU={avg_miou:.2f}%, n={len(all_iou_per_polygon)})"


def _process_iteration_single_region(
    iter_dir: Path,
    gt_path: str,
    nb_class: int,
    overlap_name: str,
) -> str:
    """Process iteration with single-region structure."""
    pred_path = _load_pred_path(iter_dir, overlap_name)
    if pred_path is None or not pred_path.exists():
        return f"{iter_dir.name} (no prediction)"
    
    if not Path(gt_path).exists():
        return f"{iter_dir.name} (GT not found)"
    
    gt = read_tiff(gt_path)
    pred = read_tiff(pred_path)
    
    # Get per-polygon IoU data
    iou_data, gt_classes = get_miou_polygon_data(pred, gt, nb_class)
    
    if len(iou_data) == 0:
        return f"{iter_dir.name} (no polygons)"
    
    # Compute mIoU metrics
    avg_miou = float(np.mean(iou_data) * 100)
    n_matched = sum(1 for iou in iou_data if iou > 0)
    
    # Compute mIoU per class
    miou_per_class = []
    for c in range(1, nb_class + 1):
        class_ious = [iou for iou, cls in zip(iou_data, gt_classes) if cls == c]
        if class_ious:
            miou_per_class.append(float(np.mean(class_ious) * 100))
        else:
            miou_per_class.append(0.0)
    
    global_miou_metrics = {
        "global/miou/avgMIoU": avg_miou,
        "global/miou/MIoU_per_class": miou_per_class,
        "global/miou/n_polygons_evaluated": len(iou_data),
        "global/miou/n_polygons_matched": n_matched,
    }
    
    # Save to global_miou_metrics.yaml
    global_path = iter_dir / "global_miou_metrics.yaml"
    global_path.write_text(
        yaml.safe_dump(global_miou_metrics, sort_keys=False, allow_unicode=False)
    )
    
    return f"{iter_dir.name} (mIoU={avg_miou:.2f}%, n={len(iou_data)})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalcula Mean IoU por polígono para todas as iterações."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Caminho para o diretório de dados (ex: bioflore_data/v02 ou /path/to/13_amazon_data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de processos em paralelo (padrão: metade dos CPUs, min 1).",
    )
    args_ns = parser.parse_args()
    
    # Resolve data path
    data_path = Path(args_ns.data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    data_path = data_path.resolve()
    
    if not data_path.exists():
        print(f"Error: Data path not found: {data_path}")
        return
    
    print(f"Processing data path: {data_path}")
    
    # Load config from args.yaml in data directory
    args_yaml = data_path / "args.yaml"
    if not args_yaml.exists():
        print(f"Error: args.yaml not found in {data_path}")
        return
    
    cfg = read_yaml(str(args_yaml))
    nb_class = int(cfg.get("nb_class", 4))
    overlap_name = _overlap_str(cfg.get("overlap", [0.3, 0.5]))
    
    print(f"Config: nb_class={nb_class}, overlap={overlap_name}")
    
    # Find all iteration directories
    iter_dirs = sorted(p for p in data_path.glob("iter_*") if p.is_dir())
    # Skip iter_000 (only distance maps, no predictions)
    iter_dirs = [d for d in iter_dirs if d.name != "iter_000"]
    
    if not iter_dirs:
        print(f"No iteration directories found in {data_path}")
        return
    
    print(f"Found {len(iter_dirs)} iterations to process")
    
    # Detect structure from first iteration with predictions
    is_multi_region = False
    num_regions = 1
    for iter_dir in iter_dirs:
        detected_multi, detected_num = _detect_structure(iter_dir)
        if detected_num > 0:
            is_multi_region = detected_multi
            num_regions = detected_num
            break
    
    print(f"Structure: {'multi-region' if is_multi_region else 'single-region'} ({num_regions} region(s))")
    
    # Get test segmentation paths
    try:
        gt_paths = _get_test_segmentation_paths(cfg, data_path)
        print(f"Test segmentation paths: {gt_paths}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Set up parallel processing
    default_workers = max(1, (os.cpu_count() or 2) // 2)
    workers = args_ns.workers or min(default_workers, len(iter_dirs))
    
    print(f"Using {workers} worker(s)")
    print("-" * 50)
    
    if is_multi_region:
        process_fn = partial(
            _process_iteration_multi_region,
            region_gt_paths=gt_paths,
            nb_class=nb_class,
            overlap_name=overlap_name,
        )
    else:
        process_fn = partial(
            _process_iteration_single_region,
            gt_path=gt_paths[0],
            nb_class=nb_class,
            overlap_name=overlap_name,
        )
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(process_fn, iter_dirs):
            print(f"[ok] {result}")
    
    print("-" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
