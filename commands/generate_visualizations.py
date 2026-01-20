#!/usr/bin/env python3
"""
Gera visualizações de synthetic_selected_labels para todas as iterações de uma versão.

Funciona tanto para:
- Casos multi-region (e.g., /home/luizluz/Documentos/multi-task-fcn/bioflore_data/v02)
- Casos single-region (e.g., /home/luizluz/Documentos/multi-task-fcn/13_amazon_data)

Usage:
    python commands/generate_visualizations.py --data_path /path/to/data
    python commands/generate_visualizations.py --data_path bioflore_data/v02
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.io_operations import read_yaml  # noqa: E402
from visualization import generate_view_for_synthetic_selected_label  # noqa: E402


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
    
    # Check for direct raster_prediction folder or new_labels folder
    if (iter_dir / "raster_prediction").exists() or (iter_dir / "new_labels").exists():
        return False, 1
    
    return False, 0


def _get_train_segmentation_paths(cfg: dict, data_path: Path) -> List[str]:
    """Get train segmentation paths from config, resolving relative paths."""
    if "train_segmentation_paths" in cfg:
        paths = cfg["train_segmentation_paths"]
    elif "train_segmentation_path" in cfg:
        paths = [cfg["train_segmentation_path"]]
    else:
        raise ValueError("No train_segmentation_path(s) found in config")
    
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


def _get_ortho_image_paths(cfg: dict, data_path: Path) -> List[str]:
    """Get orthoimage paths from config, resolving relative paths."""
    if "ortho_images" in cfg:
        paths = cfg["ortho_images"]
    elif "ortho_image" in cfg:
        paths = [cfg["ortho_image"]]
    else:
        raise ValueError("No ortho_image(s) found in config")
    
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
    region_train_paths: List[str],
    region_ortho_paths: List[str],
) -> str:
    """Process iteration with multi-region structure."""
    results = []
    
    for region_idx, (train_path, ortho_path) in enumerate(zip(region_train_paths, region_ortho_paths)):
        region_folder = iter_dir / f"region_{region_idx}"
        if not region_folder.exists():
            continue
        
        # Check if new_labels exist (skip iter_000 which might not have them)
        if iter_dir.name != "iter_000":
            new_labels_path = region_folder / "new_labels"
            if not new_labels_path.exists():
                results.append(f"{iter_dir.name}/region_{region_idx} (no new_labels)")
                continue
        
        try:
            generate_view_for_synthetic_selected_label(
                current_iter_folder=str(region_folder),
                train_segmentation_path=train_path,
                orthoimage_path=ortho_path
            )
            results.append(f"{iter_dir.name}/region_{region_idx}")
        except Exception as e:
            results.append(f"{iter_dir.name}/region_{region_idx} (error: {str(e)})")
    
    if not results:
        return f"{iter_dir.name} (no regions processed)"
    
    return ", ".join(results)


def _process_iteration_single_region(
    iter_dir: Path,
    train_path: str,
    ortho_path: str,
) -> str:
    """Process iteration with single-region structure."""
    # Check if new_labels exist (skip iter_000 which might not have them)
    if iter_dir.name != "iter_000":
        new_labels_path = iter_dir / "new_labels"
        if not new_labels_path.exists():
            return f"{iter_dir.name} (no new_labels)"
    
    try:
        generate_view_for_synthetic_selected_label(
            current_iter_folder=str(iter_dir),
            train_segmentation_path=train_path,
            orthoimage_path=ortho_path
        )
        return iter_dir.name
    except Exception as e:
        return f"{iter_dir.name} (error: {str(e)})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera visualizações de synthetic_selected_labels para todas as iterações de uma versão."
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
    
    print(f"Config loaded successfully")
    
    # Get train segmentation and orthoimage paths
    try:
        train_paths = _get_train_segmentation_paths(cfg, data_path)
        ortho_paths = _get_ortho_image_paths(cfg, data_path)
        print(f"Train segmentation paths: {train_paths}")
        print(f"Orthoimage paths: {ortho_paths}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Find all iteration directories
    iter_dirs = sorted(p for p in data_path.glob("iter_*") if p.is_dir())
    
    if not iter_dirs:
        print(f"No iteration directories found in {data_path}")
        return
    
    print(f"Found {len(iter_dirs)} iterations to process")
    
    # Detect structure from first iteration
    is_multi_region = False
    num_regions = 1
    for iter_dir in iter_dirs:
        detected_multi, detected_num = _detect_structure(iter_dir)
        if detected_num > 0:
            is_multi_region = detected_multi
            num_regions = detected_num
            break
    
    print(f"Structure: {'multi-region' if is_multi_region else 'single-region'} ({num_regions} region(s))")
    
    # Validate number of paths matches number of regions
    if is_multi_region:
        if len(train_paths) != num_regions:
            print(f"Error: Number of train_segmentation_paths ({len(train_paths)}) doesn't match number of regions ({num_regions})")
            return
        if len(ortho_paths) != num_regions:
            print(f"Error: Number of ortho_images ({len(ortho_paths)}) doesn't match number of regions ({num_regions})")
            return
    
    # Set up parallel processing
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor because matplotlib might have issues with multiprocessing
    default_workers = max(1, (os.cpu_count() or 2) // 2)
    workers = args_ns.workers or min(default_workers, len(iter_dirs))
    
    print(f"Using {workers} worker(s)")
    print("-" * 50)
    
    if is_multi_region:
        process_fn = partial(
            _process_iteration_multi_region,
            region_train_paths=train_paths,
            region_ortho_paths=ortho_paths,
        )
    else:
        process_fn = partial(
            _process_iteration_single_region,
            train_path=train_paths[0],
            ortho_path=ortho_paths[0],
        )
    
    # Use ThreadPoolExecutor for visualization tasks (matplotlib works better with threads)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_iter = {executor.submit(process_fn, iter_dir): iter_dir for iter_dir in iter_dirs}
        
        # Process with progress bar
        with tqdm(total=len(iter_dirs), desc="Processing iterations", unit="iter") as pbar:
            for future in as_completed(future_to_iter):
                iter_dir = future_to_iter[future]
                try:
                    result = future.result()
                    tqdm.write(f"[ok] {result}")
                except Exception as e:
                    tqdm.write(f"[error] {iter_dir.name}: {str(e)}")
                finally:
                    pbar.update(1)
    
    print("-" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
