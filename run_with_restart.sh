#!/bin/bash
# Script shell para executar main.py com reinício automático

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAIN_SCRIPT="main.py"
ARGS_FILE="${1:-args.yaml}"
MAX_RESTARTS="${2:-100}"
RESTART_DELAY="${3:-5}"

LOG_FILE="main_restart.log"
PYTHON_PATH="${SCRIPT_DIR}/.env/bin/python3"

# Verifica se o Python do venv existe, senão usa o do sistema
if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH="python3"
fi

echo "=========================================="
echo "Script de reinício automático para main.py"
echo "=========================================="
echo "Python: $PYTHON_PATH"
echo "Script: $MAIN_SCRIPT"
echo "Args: $ARGS_FILE"
echo "Max restarts: $MAX_RESTARTS"
echo "Restart delay: ${RESTART_DELAY}s"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

RESTART_COUNT=0

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    RESTART_COUNT=$((RESTART_COUNT + 1))
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Tentativa $RESTART_COUNT/$MAX_RESTARTS" | tee -a "$LOG_FILE"
    echo "Iniciando execução..." | tee -a "$LOG_FILE"
    echo "----------------------------------------"
    
    START_TIME=$(date +%s)
    
    # Executa o script
    $PYTHON_PATH "$MAIN_SCRIPT" "$ARGS_FILE" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ Processo finalizado com sucesso!"
        echo "Tempo total: ${ELAPSED}s"
        echo "==========================================" | tee -a "$LOG_FILE"
        exit 0
    else
        echo ""
        echo "=========================================="
        echo "⚠ Processo terminou com código: $EXIT_CODE"
        echo "Tempo antes da falha: ${ELAPSED}s"
        echo "==========================================" | tee -a "$LOG_FILE"
        
        if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
            echo "Aguardando ${RESTART_DELAY}s antes de reiniciar..." | tee -a "$LOG_FILE"
            sleep $RESTART_DELAY
        else
            echo "Limite de reinícios atingido!" | tee -a "$LOG_FILE"
            exit $EXIT_CODE
        fi
    fi
done

echo "Limite máximo de reinícios atingido!" | tee -a "$LOG_FILE"
exit 1
