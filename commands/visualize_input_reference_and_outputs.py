#!/usr/bin/env python3
"""
Visualize model input, training references (segmentation + distance), and model outputs.

- Selects a training polygon from train_labels.shp by area rank (default: largest).
- Crops orthoimage, new_labels/all_labels_set.tif, and distance_map for that region.
- Runs inference on the full crop at once (single forward pass): image is resized to model input size, then outputs are resized back.
- Saves each image separately: imagem_entrada, referencia_segmentacao, referencia_distancia,
  saida_segmentacao, saida_distancia.

Usage:
    python commands/visualize_input_reference_and_outputs.py [--output-dir OUTPUT_DIR] [--rank RANK]
    # --rank 0 = largest polygon, 1 = second largest, 2 = third, ...
"""

import argparse
import os
import sys
from os.path import dirname, join, exists

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.windows
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import box
from skimage.color import label2rgb
from skimage.transform import resize as sk_resize

ROOT_PATH = dirname(dirname(__file__))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from src.io_operations import load_args
from src.model import build_model, load_weights
from src.utils import get_device, check_folder as ensure_dir, normalize


def _crop_raster_by_bounds(raster_path: str, bounds: tuple, raster_crs) -> np.ndarray:
    """
    Crop raster to the given bounds (left, bottom, right, top) in raster_crs.
    Bounds are clipped to raster extent.
    Returns array (C, H, W) or (H, W).
    """
    with rasterio.open(raster_path) as src:
        if raster_crs is not None and src.crs != raster_crs:
            from rasterio.warp import transform_bounds
            bounds = transform_bounds(raster_crs, src.crs, *bounds)
        raster_bounds = src.bounds
        # Clip requested bounds to raster
        left = max(bounds[0], raster_bounds.left)
        bottom = max(bounds[1], raster_bounds.bottom)
        right = min(bounds[2], raster_bounds.right)
        top = min(bounds[3], raster_bounds.top)
        if left >= right or bottom >= top:
            raise ValueError("Bounds do not overlap raster")
        window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
        window = window.round_lengths().round_offsets()
        data = src.read(window=window)
    return data


def _select_large_polygon_and_region(shapes_path: str, ortho_paths: list, rank: int = 0):
    """
    Load train_labels.shp, assign each polygon to a region by ortho bounds,
    sort by area descending, and return the polygon at the given rank and its region index.

    rank=0 -> largest, rank=1 -> second largest, rank=2 -> third largest, etc.
    """
    gdf = gpd.read_file(shapes_path)
    if gdf.empty or gdf.geometry is None or len(gdf) == 0:
        raise ValueError("No geometries in train_labels.shp")

    # Get bounds and CRS for each ortho
    region_bounds = []
    region_crs = None
    for ortho_path in ortho_paths:
        with rasterio.open(ortho_path) as src:
            region_bounds.append((src.bounds, src.crs))
            if region_crs is None:
                region_crs = src.crs

    # Reproject shapes to first ortho CRS for intersection
    if gdf.crs is None:
        gdf.set_crs(region_crs, inplace=True)
    gdf = gdf.to_crs(region_crs)

    # Collect (idx, region_idx, area) for every polygon that intersects some ortho
    candidates = []
    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        area = geom.area
        if area <= 0:
            continue
        for region_idx, ((r_left, r_bottom, r_right, r_top), _) in enumerate(region_bounds):
            b = box(r_left, r_bottom, r_right, r_top)
            if geom.intersects(b):
                candidates.append((idx, region_idx, area))
                break

    if not candidates:
        raise ValueError("No polygon from train_labels.shp intersects any orthoimage")

    # Sort by area descending (largest first), then pick the one at rank
    candidates.sort(key=lambda x: x[2], reverse=True)
    if rank < 0 or rank >= len(candidates):
        raise ValueError(
            f"rank={rank} out of range: there are {len(candidates)} intersecting polygons (valid rank: 0..{len(candidates) - 1})"
        )
    best_idx, best_region, _ = candidates[rank]
    return gdf.iloc[best_idx].geometry, best_region, region_bounds[best_region][1]


def _save_ortho_png(arr: np.ndarray, path: str):
    """Save ortho (C, H, W) as RGB PNG."""
    if arr.ndim == 3 and arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.transpose(np.broadcast_to(arr, (3,) + arr.shape[-2:]), (1, 2, 0))
    if rgb.max() <= 1 and rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    plt.imsave(path, rgb)


def _get_label2rgb_colors(num_classes: int) -> list:
    """
    List of RGB colors in [0,1] for label2rgb: one per semantic class (1..num_classes).
    Same list is used for reference and output so colors align (ref 1 = output 0, etc.).
    """
    cmap = plt.get_cmap("tab10", max(num_classes, 10))
    return [np.array(cmap(i)[:3]) for i in range(num_classes)]


def _save_segmentation_png(
    arr: np.ndarray,
    path: str,
    num_classes: int = None,
    colors: list = None,
    output_mode: bool = False,
):
    """
    Save segmentation (H, W) as colorized PNG using skimage.label2rgb.
    - output_mode=False (reference): label 0 = preto (sem anotação), 1..N = classes.
    - output_mode=True (model output): remap 0→1, 1→2, ... so output 0 = mesma cor que ref 1.
    """
    if arr.ndim == 3:
        arr = arr.squeeze()
    if colors is None:
        num_classes = num_classes if num_classes is not None else 4
        colors = _get_label2rgb_colors(num_classes)
    else:
        num_classes = len(colors)
    arr = np.asarray(arr, dtype=np.int64)
    if output_mode:
        # Saída do modelo: valor 0 = classe 1 na ref → remapear para labels 1..N
        label = np.clip(arr + 1, 1, num_classes)
    else:
        label = arr
    # label2rgb: bg_label=0 = preto (ref: sem anotação); cores para 1..N
    rgb = label2rgb(
        label,
        colors=colors,
        bg_label=0,
        bg_color=(0, 0, 0),
        image=None,
    )
    # rgb is float [0,1]; fundo pode ser (0,0,0); garantir opaco para salvamento
    out = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    plt.imsave(path, out)


def _save_referencia_segmentacao_png(arr: np.ndarray, path: str, num_classes: int = 4):
    """Save reference segmentation with white background and 2x size padding (canvas = 2H x 2W)."""
    if arr.ndim == 3:
        arr = arr.squeeze()
    h, w = arr.shape
    cmap = plt.get_cmap("tab10", max(num_classes + 1, 10))
    # Colorize: class 0 = white, others = colormap
    out_small = np.zeros((h, w, 4), dtype=np.uint8)
    out_small[:, :] = (255, 255, 255, 255)  # white background
    for c in range(1, num_classes + 1):
        mask = arr == c
        if mask.any():
            out_small[mask] = (np.array(cmap(c)[:4]) * 255).astype(np.uint8)
    if (arr > num_classes).any():
        out_small[arr > num_classes] = (np.array(cmap(num_classes)[:4]) * 255).astype(np.uint8)
    # Double size canvas with padding (image centered)
    out_h, out_w = 2 * h, 2 * w
    pad_h, pad_w = h // 2, w // 2
    out = np.ones((out_h, out_w, 4), dtype=np.uint8) * 255  # white padding
    out[pad_h : pad_h + h, pad_w : pad_w + w] = out_small
    plt.imsave(path, out)


def _save_depth_png(arr: np.ndarray, path: str):
    """Save single-channel depth as grayscale PNG."""
    if arr.ndim == 3:
        arr = arr.squeeze()
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax <= vmin:
        vmax = vmin + 1
    arr_n = np.clip((arr.astype(float) - vmin) / (vmax - vmin), 0, 1)
    plt.imsave(path, arr_n, cmap="viridis")


def main():
    parser = argparse.ArgumentParser(description="Visualize model input, references and outputs.")
    parser.add_argument(
        "--output-dir",
        default="output_visualization",
        help="Directory to save all output images",
    )
    parser.add_argument(
        "--iter-num",
        type=int,
        default=1,
        help="Iteration number, e.g. 1 -> iter_001, 2 -> iter_002 (default: 1)",
    )
    parser.add_argument(
        "--args-yaml",
        default=None,
        help="Override args yaml (default: bioflore_data/v02/args.yaml)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Polygon rank by area (0=largest, 1=second largest, 2=third, ...). Default: 0",
    )
    parser.add_argument(
        "--crop-multiplier",
        type=float,
        default=3,
        help="Crop size as multiple of polygon bbox (width and height). 2 = 2x each side (default), 1 = exact bbox, 3 = 3x.",
    )
    args_cli = parser.parse_args()

    # 1) Config and paths
    data_path = join(ROOT_PATH, "bioflore_data", "v02")
    iter_dir = join(data_path, f"iter_{args_cli.iter_num:03d}")
    args_yaml = args_cli.args_yaml or join(data_path, "args.yaml")
    shapes_path = join(ROOT_PATH, "bioflore_data", "shapes", "train_labels.shp")
    out_dir = args_cli.output_dir
    if not os.path.isabs(out_dir):
        out_dir = join(ROOT_PATH, out_dir)
    ensure_dir(out_dir)

    if not exists(args_yaml):
        raise FileNotFoundError(f"Args not found: {args_yaml}")
    if not exists(shapes_path):
        raise FileNotFoundError(f"Shapes not found: {shapes_path}")
    if not exists(iter_dir):
        raise FileNotFoundError(f"Iter dir not found: {iter_dir}")

    args = load_args(args_yaml)
    num_regions = getattr(args, "num_regions", len(args.ortho_images))
    ortho_paths = [p if os.path.isabs(p) else join(ROOT_PATH, p) for p in args.ortho_images]

    # Ref iter: iter 1 → distância de iter_000, segmentação do treino original; iter >= 2 → iteração anterior
    if args_cli.iter_num == 1:
        ref_iter_num = 0
        ref_iter_dir = join(data_path, f"iter_{ref_iter_num:03d}")
        if not exists(ref_iter_dir):
            raise FileNotFoundError(f"Ref iter dir not found: {ref_iter_dir} (need iter_000 for distance when iter_num=1)")
        use_train_ref = True
    else:
        ref_iter_num = max(0, args_cli.iter_num - 1)
        ref_iter_dir = join(data_path, f"iter_{ref_iter_num:03d}")
        if not exists(ref_iter_dir):
            raise FileNotFoundError(f"Ref iter dir not found: {ref_iter_dir} (need iter_{ref_iter_num:03d} for references)")
        use_train_ref = False

    # 2) Select polygon by area rank and region
    geom, region_idx, raster_crs = _select_large_polygon_and_region(
        shapes_path, ortho_paths, rank=args_cli.rank
    )
    left, bottom, right, top = geom.bounds
    # Crop window = crop_multiplier × polygon bbox (width and height), centered on polygon
    w, h = right - left, top - bottom
    mult = args_cli.crop_multiplier
    half_w, half_h = mult * w / 2, mult * h / 2
    center_x = (left + right) / 2
    center_y = (bottom + top) / 2
    bounds = (
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    )

    ortho_path = ortho_paths[region_idx]
    region_folder = join(iter_dir, f"region_{region_idx}") if num_regions > 1 else iter_dir
    ref_region_folder = join(ref_iter_dir, f"region_{region_idx}") if num_regions > 1 else ref_iter_dir
    if use_train_ref:
        train_paths = getattr(args, "train_segmentation_paths", None)
        if not train_paths or region_idx >= len(train_paths):
            raise FileNotFoundError("train_segmentation_paths not in args or missing region (need for ref when iter_num=1)")
        labels_path = join(ROOT_PATH, train_paths[region_idx]) if not os.path.isabs(train_paths[region_idx]) else train_paths[region_idx]
        train_dist = join(os.path.dirname(labels_path), "train_distance_map.tif")
        if exists(train_dist):
            distance_map_path = train_dist
        else:
            distance_map_path = join(ref_region_folder, "distance_map", "train_distance_map.tif")
    else:
        labels_path = join(ref_region_folder, "new_labels", "all_labels_set.tif")
        distance_map_path = join(ref_region_folder, "distance_map", "all_labels_distance_map.tif")
    if not exists(labels_path):
        raise FileNotFoundError(f"Labels not found: {labels_path} (ref: {'train' if use_train_ref else f'iter_{ref_iter_num:03d}'})")
    if not exists(distance_map_path):
        raise FileNotFoundError(f"Distance map not found: {distance_map_path} (ref: {'train' if use_train_ref else f'iter_{ref_iter_num:03d}'})")

    # 3) Crop ortho, segmentation ref, distance ref
    crop_ortho = _crop_raster_by_bounds(ortho_path, bounds, raster_crs)
    crop_seg_ref = _crop_raster_by_bounds(labels_path, bounds, raster_crs)
    crop_dist_ref = _crop_raster_by_bounds(distance_map_path, bounds, raster_crs)
    if crop_seg_ref.ndim == 3:
        crop_seg_ref = crop_seg_ref.squeeze()
    if crop_dist_ref.ndim == 3:
        crop_dist_ref = crop_dist_ref.squeeze()

    # Ensure all crops have the same spatial size as the ortho (reference grid)
    target_h, target_w = int(crop_ortho.shape[1]), int(crop_ortho.shape[2])
    if crop_seg_ref.shape != (target_h, target_w):
        crop_seg_ref = sk_resize(
            crop_seg_ref.astype(np.float64),
            (target_h, target_w),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(crop_seg_ref.dtype)
    if crop_dist_ref.shape != (target_h, target_w):
        crop_dist_ref = sk_resize(
            crop_dist_ref.astype(np.float64),
            (target_h, target_w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(crop_dist_ref.dtype)

    # 4) Save input and reference images (each separately)
    _save_ortho_png(crop_ortho, join(out_dir, "imagem_entrada.png"))
    with rasterio.open(ortho_path) as src:
        meta = src.meta.copy()
    # Write crop transform for potential TIF save; for PNG we only save RGB
    crop_ortho_tif = join(out_dir, "imagem_entrada.tif")
    meta.update(height=crop_ortho.shape[1], width=crop_ortho.shape[2], count=crop_ortho.shape[0])
    with rasterio.open(crop_ortho_tif, "w", **meta) as dst:
        dst.write(crop_ortho)

    # Shared colormap (skimage label2rgb) for referencia and saida so they are directly comparable
    seg_colors = _get_label2rgb_colors(args.nb_class)
    _save_segmentation_png(crop_seg_ref, join(out_dir, "referencia_segmentacao.png"), colors=seg_colors, output_mode=False)
    np.save(join(out_dir, "referencia_segmentacao.npy"), crop_seg_ref)  # optional TIF could use array2raster

    _save_depth_png(crop_dist_ref, join(out_dir, "referencia_distancia.png"))
    np.save(join(out_dir, "referencia_distancia.npy"), crop_dist_ref)

    # 5) Single forward pass: resize image to model input size, run model once, resize outputs back
    in_channels = crop_ortho.shape[0]
    orig_h, orig_w = int(crop_ortho.shape[1]), int(crop_ortho.shape[2])
    input_dimension = getattr(args, "input_dimension", 256)
    model_dir = join(iter_dir, args.model_dir)
    checkpoint_path = join(model_dir, args.checkpoint_file)
    if not exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Resize crop to model input size (C, input_dim, input_dim)
    img = crop_ortho.astype(np.float32)
    normalize(img)
    img_small = np.zeros((in_channels, input_dimension, input_dimension), dtype=np.float32)
    for c in range(in_channels):
        img_small[c] = sk_resize(
            img[c],
            (input_dimension, input_dimension),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
    x = torch.from_numpy(img_small).unsqueeze(0).to(get_device(), dtype=torch.float)

    device = get_device()
    model = build_model(
        in_channels=in_channels,
        num_classes=args.nb_class,
        arch=args.arch,
        dropout_rate=args.dropout_rate,
        batch_norm=args.batch_norm,
        pretrained=args.is_pretrained,
        psize=input_dimension,
    )
    model = load_weights(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(x)
    prob = F.softmax(out["out"], dim=1)
    depth_small = torch.sigmoid(out["aux"]).squeeze(1).squeeze(0).cpu().numpy()
    pred_class_small = prob.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize outputs back to original crop size
    pred_class = sk_resize(
        pred_class_small.astype(np.float64),
        (orig_h, orig_w),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)
    depth_map = sk_resize(
        depth_small.astype(np.float64),
        (orig_h, orig_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    # 6) Save model outputs (same seg_colors as referencia; output 0 = class 1 in ref)
    _save_segmentation_png(pred_class, join(out_dir, "saida_segmentacao.png"), colors=seg_colors, output_mode=True)
    np.save(join(out_dir, "saida_segmentacao.npy"), pred_class)
    _save_depth_png(depth_map, join(out_dir, "saida_distancia.png"))
    np.save(join(out_dir, "saida_distancia.npy"), depth_map)

    print(f"Done. Outputs saved in {out_dir}")
    ref_label = "train (original)" if use_train_ref else f"iter_{ref_iter_num:03d}"
    print(f"  Ref (segmentação/distância): {ref_label}  |  Model/saída: iter_{args_cli.iter_num:03d}")
    print(f"  Crop size (pixels): {orig_h} x {orig_w}  (crop_multiplier={args_cli.crop_multiplier})")
    print("  imagem_entrada.png / .tif")
    print("  referencia_segmentacao.png")
    print("  referencia_distancia.png")
    print("  saida_segmentacao.png")
    print("  saida_distancia.png")


if __name__ == "__main__":
    main()
