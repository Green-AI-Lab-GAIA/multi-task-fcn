#!/bin/bash
#
# Script para executar o experimento vLaura
# 
# Este script executa:
# 1. Validação da configuração
# 2. Treinamento do modelo DeepLabv3+ ResNet9
#
# Uso: ./run_vLaura.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Experimento vLaura - DeepLabv3+ ResNet9"
echo "=========================================="
echo ""

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Validação
echo -e "${YELLOW}[1/2] Validando configuração...${NC}"
if python validate_vLaura.py; then
    echo -e "${GREEN}✓ Validação concluída com sucesso!${NC}"
    echo ""
else
    echo -e "${RED}✗ Validação falhou. Por favor, corrija os erros antes de continuar.${NC}"
    exit 1
fi

# 2. Treinamento
echo -e "${YELLOW}[2/2] Iniciando treinamento...${NC}"
echo ""
echo "Configuração:"
echo "  - Modelo: deeplabv3+_resnet9"
echo "  - Dropout: 0.65"
echo "  - Input: 128x128"
echo "  - Iterações: 1"
echo "  - Diretório: bioflore_data/vLaura/"
echo ""
echo -e "${YELLOW}Pressione CTRL+C para cancelar ou aguarde 5 segundos...${NC}"
sleep 5

python main.py args_vLaura.yaml

echo ""
echo -e "${GREEN}=========================================="
echo "Treinamento concluído!"
echo "==========================================${NC}"
echo ""
echo "Resultados salvos em: bioflore_data/vLaura/"
