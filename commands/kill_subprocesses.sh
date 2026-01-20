#!/bin/bash
# Script para matar processos iniciados por call_main_as_subprocess.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Procurando processos relacionados a call_main_as_subprocess.py e main.py..."

# Encontra processos do call_main_as_subprocess.py
PARENT_PIDS=$(pgrep -f "call_main_as_subprocess.py" | grep -v "^$$")

# Encontra processos do main.py que foram iniciados pelo subprocess
MAIN_PIDS=$(pgrep -f "main.py.*args.yaml" | grep -v "^$$")

# Mostra os processos encontrados
if [ -n "$PARENT_PIDS" ]; then
    echo ""
    echo "Processos do call_main_as_subprocess.py encontrados:"
    ps -p $PARENT_PIDS -o pid,ppid,cmd --no-headers | while read line; do
        echo "  $line"
    done
fi

if [ -n "$MAIN_PIDS" ]; then
    echo ""
    echo "Processos do main.py encontrados:"
    ps -p $MAIN_PIDS -o pid,ppid,cmd --no-headers | while read line; do
        echo "  $line"
    done
fi

if [ -z "$PARENT_PIDS" ] && [ -z "$MAIN_PIDS" ]; then
    echo "Nenhum processo relacionado encontrado."
    exit 0
fi

# Pergunta confirmação
echo ""
read -p "Deseja matar estes processos? (s/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "Operação cancelada."
    exit 0
fi

# Mata os processos
KILLED_COUNT=0

if [ -n "$PARENT_PIDS" ]; then
    echo "Matando processos do call_main_as_subprocess.py..."
    for pid in $PARENT_PIDS; do
        if kill -TERM $pid 2>/dev/null; then
            echo "  ✓ Enviado SIGTERM para PID $pid"
            KILLED_COUNT=$((KILLED_COUNT + 1))
        fi
    done
fi

if [ -n "$MAIN_PIDS" ]; then
    echo "Matando processos do main.py..."
    for pid in $MAIN_PIDS; do
        if kill -TERM $pid 2>/dev/null; then
            echo "  ✓ Enviado SIGTERM para PID $pid"
            KILLED_COUNT=$((KILLED_COUNT + 1))
        fi
    done
fi

# Aguarda um pouco e força kill se necessário
sleep 2

if [ -n "$PARENT_PIDS" ]; then
    for pid in $PARENT_PIDS; do
        if kill -0 $pid 2>/dev/null; then
            echo "  ⚠ Processo $pid ainda está rodando, forçando kill..."
            kill -KILL $pid 2>/dev/null && echo "    ✓ PID $pid morto"
        fi
    done
fi

if [ -n "$MAIN_PIDS" ]; then
    for pid in $MAIN_PIDS; do
        if kill -0 $pid 2>/dev/null; then
            echo "  ⚠ Processo $pid ainda está rodando, forçando kill..."
            kill -KILL $pid 2>/dev/null && echo "    ✓ PID $pid morto"
        fi
    done
fi

echo ""
echo "Concluído! $KILLED_COUNT processo(s) foram terminados."
