#!/usr/bin/env python3
"""
Computa o F1 por componente (por espécie) para versões Amazon de região única.

Ao contrário de `rerun_component_f1.py` (projetado para o layout multi-região
do Bioflore com sub-pastas region_0/, region_1/, …), este script trata versões
onde as predições ficam diretamente em iter_*/raster_prediction/.

O resultado é escrito em:
  - iter_*/all_labels_test_metrics.yaml  →  acrescenta all_labels_F1_component (e demais)
  - iter_*/global_component_metrics.yaml →  criado/sobrescrito (mesmos valores, região única)

Uso:
    python commands/compute_amazon_component_f1.py \\
        --version-dir amazon_data/13_amazon_data \\
        --gt-path     amazon_data/amazon_input_data/segmentation/test_set.tif
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.io_operations import read_tiff  # noqa: E402
from src.metrics import evaluate_f1_by_component  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlap_str(overlap: List[float]) -> str:
    """Converte a lista de overlap para o sufixo usado no nome do arquivo TIF."""
    val = float(sum(overlap))
    text = f"{val:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def _find_pred_path(iter_dir: Path, overlap_name: str) -> Optional[Path]:
    """Localiza o arquivo de predição de classes na pasta raster_prediction.

    Tenta primeiro o nome canônico join_class_{overlap_name}.TIF; se não
    existir, usa o primeiro join_class_*.TIF encontrado (fallback).
    """
    pred_dir = iter_dir / "raster_prediction"
    if not pred_dir.is_dir():
        return None
    candidate = pred_dir / f"join_class_{overlap_name}.TIF"
    if candidate.exists():
        return candidate
    alts = sorted(pred_dir.glob("join_class_*.TIF"))
    return alts[0] if alts else None


# ---------------------------------------------------------------------------
# Per-iteration worker
# ---------------------------------------------------------------------------

def _process_iteration(
    iter_dir: Path,
    gt_path: str,
    nb_class: int,
    overlap_name: str,
) -> str:
    """Calcula métricas de componente para uma iteração e persiste os resultados."""
    pred_path = _find_pred_path(iter_dir, overlap_name)
    if pred_path is None:
        return f"[skip] {iter_dir.name}: sem arquivo de predição"

    gt = read_tiff(gt_path)
    pred = read_tiff(str(pred_path))

    metrics = evaluate_f1_by_component(pred=pred, gt=gt, num_class=nb_class)

    # ── 1. Atualiza all_labels_test_metrics.yaml ──────────────────────────
    yaml_path = iter_dir / "all_labels_test_metrics.yaml"
    existing: dict = {}
    if yaml_path.exists():
        existing = yaml.safe_load(yaml_path.read_text()) or {}

    existing.update(
        {
            "all_labels_Accuracy_component": metrics["Accuracy_component"],
            "all_labels_avgF1_component": metrics["avgF1_component"],
            "all_labels_F1_component": metrics["F1_component"],
            "all_labels_avgPrec_component": metrics["avgPrec_component"],
            "all_labels_avgRec_component": metrics["avgRec_component"],
            "all_labels_n_components": metrics["n_components"],
        }
    )
    yaml_path.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=False))

    # ── 2. Cria/sobrescreve global_component_metrics.yaml ─────────────────
    # Para região única, métricas globais == métricas da única região.
    labels = list(range(1, nb_class + 1))
    global_metrics = {
        "global/component/n_components": metrics["n_components"],
        "global/component/Accuracy": metrics["Accuracy_component"],
        "global/component/avgF1": metrics["avgF1_component"],
        "global/component/F1": metrics["F1_component"],
        "global/component/avgPrec": metrics["avgPrec_component"],
        "global/component/avgRec": metrics["avgRec_component"],
    }
    global_path = iter_dir / "global_component_metrics.yaml"
    global_path.write_text(yaml.safe_dump(global_metrics, sort_keys=False, allow_unicode=False))

    return f"[ok] {iter_dir.name}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Computa F1 por componente (por espécie) para uma versão Amazon de região única."
        )
    )
    parser.add_argument(
        "--version-dir",
        required=True,
        help=(
            "Caminho para a pasta da versão, relativo à raiz do repositório. "
            "Ex.: amazon_data/13_amazon_data"
        ),
    )
    parser.add_argument(
        "--gt-path",
        required=True,
        help=(
            "Caminho para o TIF de ground truth de teste, relativo à raiz do repositório. "
            "Ex.: amazon_data/amazon_input_data/segmentation/test_set.tif"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de processos em paralelo (padrão: metade dos CPUs disponíveis, mín. 1).",
    )
    args_ns = parser.parse_args()

    version_dir = (ROOT / args_ns.version_dir).resolve()
    gt_path = (ROOT / args_ns.gt_path).resolve()

    if not version_dir.is_dir():
        print(f"Erro: pasta da versão não encontrada: {version_dir}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"Erro: arquivo GT não encontrado: {gt_path}")
        sys.exit(1)

    # Lê configurações da versão
    version_args_path = version_dir / "args.yaml"
    if not version_args_path.exists():
        print(f"Erro: args.yaml não encontrado em {version_dir}")
        sys.exit(1)

    version_cfg = yaml.safe_load(version_args_path.read_text())
    nb_class = int(version_cfg["nb_class"])
    overlap_name = _overlap_str(version_cfg["overlap"])

    # Itera apenas dirs que têm raster_prediction (pula iter_000 e similares)
    iter_dirs = sorted(
        p
        for p in version_dir.glob("iter_*")
        if p.is_dir() and (p / "raster_prediction").is_dir()
    )

    if not iter_dirs:
        print(f"Nenhuma iteração com raster_prediction encontrada em {version_dir}")
        sys.exit(1)

    print(
        f"Versão  : {version_dir.name}\n"
        f"GT      : {gt_path}\n"
        f"Classes : {nb_class}\n"
        f"Overlap : {overlap_name}\n"
        f"Iterações: {len(iter_dirs)}\n"
    )

    default_workers = max(1, (os.cpu_count() or 2) // 2)
    workers = args_ns.workers if args_ns.workers is not None else min(default_workers, len(iter_dirs))

    process_fn = partial(
        _process_iteration,
        gt_path=str(gt_path),
        nb_class=nb_class,
        overlap_name=overlap_name,
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(process_fn, iter_dirs):
            print(result)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
