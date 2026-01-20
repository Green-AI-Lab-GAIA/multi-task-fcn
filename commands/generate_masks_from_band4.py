#!/usr/bin/env python3
"""
Script para gerar máscaras usando a quarta banda (banda 4) de cada orthoimage.

A máscara é gerada da seguinte forma:
- Lê a quarta banda (índice 3) de cada imagem
- Valores > 0 = válido (área de estudo)
- Valores = 0 = inválido (fora da área de estudo)

Uso:
    python commands/generate_masks_from_band4.py --version v04
"""

import sys
import argparse
import numpy as np
from os.path import dirname, join, exists, basename, splitext
from pathlib import Path
from PIL import Image

# Add parent directory to path
ROOT_PATH = dirname(dirname(__file__))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.io_operations import (
    load_args,
    normalize_multi_region_args,
    read_tiff,
    get_image_metadata,
    check_folder,
    array2raster,
    _save_mask_preview,
)
from src.logger import create_logger

logger = create_logger("generate_masks_from_band4")


def generate_mask_from_band4(
    ortho_image_path: str,
    output_path: str = None,
    band_index: int = 3,
    save_preview: bool = True,
    preview_max_size: int = 1024,
) -> np.ndarray:
    """
    Generate a mask from the 4th band (band index 3) of an orthoimage.
    
    Parameters
    ----------
    ortho_image_path : str
        Path to the orthoimage TIFF file
    output_path : str, optional
        Path to save the generated mask TIFF. If None, mask is not saved.
    band_index : int, optional
        Index of the band to use (0-based). Default 3 (4th band).
    save_preview : bool, optional
        If True, saves a low-resolution PNG preview alongside the TIFF. Default True.
    preview_max_size : int, optional
        Maximum dimension (width or height) for the preview image. Default 1024.
        
    Returns
    -------
    np.ndarray
        Binary mask where 1 = valid region (band4 > 0), 0 = invalid (band4 = 0)
    """
    logger.info(f"Generating mask from band {band_index + 1} (index {band_index}) of: {ortho_image_path}")
    
    # Read the orthoimage
    ortho = read_tiff(ortho_image_path)
    
    # Check if image has enough bands
    if ortho.ndim == 3:
        num_bands = ortho.shape[0]
        if band_index >= num_bands:
            raise ValueError(
                f"Image has only {num_bands} bands, but requested band index {band_index} "
                f"(band {band_index + 1}). Available bands: 0-{num_bands - 1}"
            )
        
        # Extract the specified band
        band4 = ortho[band_index, :, :]
        logger.info(f"Multiband image with shape {ortho.shape}, using band {band_index + 1}")
    elif ortho.ndim == 2:
        # Single band image - use it directly
        band4 = ortho
        logger.info(f"Single band image with shape {ortho.shape}")
        if band_index != 0:
            logger.warning(f"Single band image, but requested band index {band_index}. Using the only band.")
    else:
        raise ValueError(f"Unsupported image dimensions: {ortho.ndim}. Expected 2D or 3D array.")
    
    # Create mask: values > 0 = valid, values = 0 = invalid
    mask = (band4 > 0).astype(bool)
    
    # Count initial valid pixels
    initial_valid = np.sum(mask)
    total_pixels = mask.size
    logger.info(f"Initial valid pixels (band4 > 0): {initial_valid:,} / {total_pixels:,} ({100*initial_valid/total_pixels:.2f}%)")
    
    # Convert to uint8 (0 and 1)
    mask = mask.astype(np.uint8)
    
    # Save if output path provided
    if output_path:
        # Get metadata from orthoimage
        metadata = get_image_metadata(ortho_image_path)
        
        # Ensure output directory exists
        output_dir = dirname(output_path)
        if output_dir:
            check_folder(output_dir)
        
        # Save TIFF
        array2raster(output_path, mask, metadata, dtype='uint8')
        logger.info(f"Mask TIFF saved to: {output_path}")
        
        # Save preview PNG
        if save_preview:
            _save_mask_preview(mask, output_path, preview_max_size)
    
    return mask


def main():
    """Gera máscaras usando a quarta banda para todas as regiões da versão especificada."""
    
    parser = argparse.ArgumentParser(
        description="Gera máscaras usando a quarta banda (banda 4) de cada orthoimage."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Versão do dataset (ex: v04, v03)",
    )
    parser.add_argument(
        "--band-index",
        type=int,
        default=3,
        help="Índice da banda a usar (0-based). Padrão: 3 (quarta banda)",
    )
    
    args_ns = parser.parse_args()
    version = args_ns.version
    band_index = args_ns.band_index
    
    # Caminho para args.yaml da versão especificada
    args_yaml_path = join(ROOT_PATH, "bioflore_data", version, "args.yaml")
    
    if not exists(args_yaml_path):
        logger.error(f"args.yaml não encontrado em: {args_yaml_path}")
        logger.info(f"Certifique-se de que o arquivo args.yaml existe em bioflore_data/{version}/")
        return 1
    
    logger.info(f"Carregando configuração de: {args_yaml_path}")
    args = load_args(args_yaml_path)
    
    # Normalizar argumentos para formato multi-região
    args = normalize_multi_region_args(args)
    
    # Obter data_path
    data_path = args.get('data_path', f'bioflore_data/{version}')
    
    # Resolver caminho absoluto se necessário
    if not Path(data_path).is_absolute():
        data_path = join(ROOT_PATH, data_path)
    data_path = str(Path(data_path).resolve())
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"Número de regiões: {args.get('num_regions', 0)}")
    logger.info(f"Usando banda {band_index + 1} (índice {band_index})")
    
    # Verificar se ortho_images existe
    if not args.get('ortho_images'):
        logger.error("Nenhuma orthoimage encontrada em args.yaml")
        return 1
    
    # Resolver caminhos das orthoimages (podem ser relativos)
    ortho_images = []
    for ortho_path in args['ortho_images']:
        if not Path(ortho_path).is_absolute():
            # Tentar resolver como relativo ao ROOT_PATH
            resolved_path = join(ROOT_PATH, ortho_path)
            if exists(resolved_path):
                ortho_images.append(resolved_path)
            elif exists(ortho_path):
                ortho_images.append(ortho_path)
            else:
                logger.warning(f"Orthoimage não encontrada: {ortho_path}")
                ortho_images.append(ortho_path)  # Manter original, função vai tratar erro
        else:
            ortho_images.append(ortho_path)
    
    args['ortho_images'] = ortho_images
    
    # Gerar máscaras para cada região
    num_regions = args.get('num_regions', len(ortho_images))
    logger.info(f"\n{'='*60}")
    logger.info(f"Gerando máscaras para {num_regions} região(ões)")
    logger.info(f"Método: Banda {band_index + 1} (índice {band_index})")
    logger.info(f"Critério: Valores > 0 = válido, Valores = 0 = inválido")
    logger.info(f"{'='*60}\n")
    
    # Criar pasta de máscaras geradas
    masks_folder = join(data_path, "generated_masks")
    check_folder(masks_folder)
    
    success_count = 0
    for region_idx in range(num_regions):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando região {region_idx}")
        logger.info(f"{'='*60}")
        
        ortho_path = args['ortho_images'][region_idx]
        logger.info(f"Orthoimage: {ortho_path}")
        
        # Verificar se orthoimage existe
        if not exists(ortho_path):
            logger.error(f"✗ Orthoimage não encontrada: {ortho_path}")
            continue
        
        # Gerar caminho da máscara
        ortho_name = splitext(basename(ortho_path))[0]
        generated_mask_path = join(masks_folder, f"{ortho_name}_mask.tif")
        
        # Verificar se máscara já existe e avisar
        if exists(generated_mask_path):
            logger.info(f"⚠ Máscara já existe, será regenerada: {generated_mask_path}")
        
        # Gerar máscara usando a quarta banda
        # SEMPRE gera, mesmo se já existir
        try:
            logger.info(f"Gerando máscara a partir da banda {band_index + 1}...")
            mask = generate_mask_from_band4(
                ortho_image_path=ortho_path,
                output_path=generated_mask_path,
                band_index=band_index,
                save_preview=True,
            )
            
            logger.info(f"✓ Máscara gerada com sucesso: {generated_mask_path}")
            
            # Verificar informações da máscara gerada
            mask_meta = get_image_metadata(generated_mask_path)
            
            valid_pixels = (mask > 0).sum()
            total_pixels = mask.size
            valid_percentage = 100 * valid_pixels / total_pixels
            
            logger.info(f"  Shape: {mask.shape}")
            logger.info(f"  Pixels válidos (band4 > 0): {valid_pixels:,} / {total_pixels:,} ({valid_percentage:.2f}%)")
            logger.info(f"  CRS: {mask_meta.get('crs', 'N/A')}")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"✗ Falha ao gerar máscara para região {region_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Resumo: {success_count}/{num_regions} máscaras geradas com sucesso")
    logger.info(f"{'='*60}")
    
    if success_count == num_regions:
        logger.info("\n✓ Todas as máscaras foram geradas com sucesso!")
        logger.info(f"Máscaras salvas em: {join(data_path, 'generated_masks')}")
        return 0
    else:
        logger.warning(f"\n⚠ Apenas {success_count}/{num_regions} máscaras foram geradas")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
