#!/usr/bin/env python3
"""
Script para gerar máscaras para v04 com os novos padrões (sem fill_holes e sem make_convex).

Uso:
    python commands/generate_masks_v04.py
"""

import sys
from os.path import dirname, join, exists
from pathlib import Path

# Add parent directory to path
ROOT_PATH = dirname(dirname(__file__))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.io_operations import (
    load_args,
    normalize_multi_region_args,
    generate_mask_from_orthoimage,
    read_tiff,
    get_image_metadata,
    check_folder,
)
from src.logger import create_logger
from os.path import basename, splitext

logger = create_logger("generate_masks_v04")


def main():
    """Gera máscaras para todas as regiões da v04."""
    
    # Caminho para args.yaml da v04
    args_yaml_path = join(ROOT_PATH, "bioflore_data", "v04", "args.yaml")
    
    if not exists(args_yaml_path):
        logger.error(f"args.yaml não encontrado em: {args_yaml_path}")
        logger.info("Certifique-se de que o arquivo args.yaml existe em bioflore_data/v04/")
        return 1
    
    logger.info(f"Carregando configuração de: {args_yaml_path}")
    args = load_args(args_yaml_path)
    
    # Normalizar argumentos para formato multi-região
    args = normalize_multi_region_args(args)
    
    # Obter data_path
    data_path = args.get('data_path', 'bioflore_data/v04')
    
    # Resolver caminho absoluto se necessário
    if not Path(data_path).is_absolute():
        data_path = join(ROOT_PATH, data_path)
    data_path = str(Path(data_path).resolve())
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"Número de regiões: {args.get('num_regions', 0)}")
    
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
    logger.info(f"Novos padrões: make_convex=False, fill_holes=False")
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
        
        # Gerar máscara usando os novos padrões
        # make_convex=False, fill_holes=False
        # SEMPRE gera, mesmo se já existir
        try:
            logger.info(f"Gerando máscara com novos padrões (make_convex=False, fill_holes=False)...")
            mask = generate_mask_from_orthoimage(
                ortho_image_path=ortho_path,
                output_path=generated_mask_path,
                make_convex=False,  # Novo padrão: False
                fill_holes=False,   # Novo padrão: False
                save_preview=True,
            )
            
            logger.info(f"✓ Máscara gerada com sucesso: {generated_mask_path}")
            
            # Verificar informações da máscara gerada
            mask_meta = get_image_metadata(generated_mask_path)
            
            valid_pixels = (mask > 0).sum()
            total_pixels = mask.size
            valid_percentage = 100 * valid_pixels / total_pixels
            
            logger.info(f"  Shape: {mask.shape}")
            logger.info(f"  Pixels válidos: {valid_pixels:,} / {total_pixels:,} ({valid_percentage:.2f}%)")
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
