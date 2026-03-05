#!/bin/bash
#
# Script para executar o experimento vDeepLabv3Vanilla
# 
# Este script executa o treinamento do modelo DeepLabv3 ResNet50
# SEM a task auxiliar (distance map) - apenas segmentação
#
# Uso: ./run_vDeepLabv3Vanilla.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Experimento vDeepLabv3Vanilla"
echo "DeepLabv3 ResNet50 - Sem Task Auxiliar"
echo "=========================================="
echo ""

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar se o arquivo de configuração existe
if [ ! -f "args_vDeepLabv3Vanilla.yaml" ]; then
    echo -e "${RED}✗ Arquivo args_vDeepLabv3Vanilla.yaml não encontrado!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Arquivo de configuração encontrado${NC}"
echo ""

# Treinamento
echo -e "${YELLOW}Iniciando treinamento...${NC}"
echo ""
echo "Configuração:"
echo "  - Modelo: deeplabv3_resnet50"
echo "  - Pesos pré-treinados: ImageNet"
echo "  - Task auxiliar: DESATIVADA (lambda_weight=0)"
echo "  - Input: 256x256"
echo "  - Batch size: 32"
echo "  - Iterações: 1 (sem Active Learning)"
echo "  - Diretório: bioflore_data/vDeepLabv3Vanilla/"
echo ""
echo -e "${YELLOW}Pressione CTRL+C para cancelar ou aguarde 5 segundos...${NC}"
sleep 5

python main.py args_vDeepLabv3Vanilla.yaml

echo ""
echo -e "${GREEN}=========================================="
echo "Treinamento concluído!"
echo "==========================================${NC}"
echo ""
echo "Resultados salvos em: bioflore_data/vDeepLabv3Vanilla/"
