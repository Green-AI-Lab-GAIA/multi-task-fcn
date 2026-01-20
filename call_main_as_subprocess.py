#!/usr/bin/env python3
"""
Script para executar main.py com reinício automático caso o processo morra.

Este script detecta quando o main.py morre (por OOM kill, segfault, erro, etc.)
e reinicia automaticamente o processo.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import threading
from datetime import datetime
from os.path import abspath, exists, join, dirname

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main_restart.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Flag global para controlar threads de tail
_stop_tailing = threading.Event()


def tail_log_file(log_file_path, prefix="[LOG]", max_wait=30):
    """
    Faz tail de um arquivo de log e imprime as linhas em tempo real.
    
    Parameters
    ----------
    log_file_path : str
        Caminho para o arquivo de log
    prefix : str
        Prefixo para identificar a origem do log
    max_wait : int
        Tempo máximo em segundos para esperar o arquivo ser criado
    """
    # Espera o arquivo ser criado (pode não existir ainda)
    wait_time = 0
    while not exists(log_file_path) and wait_time < max_wait and not _stop_tailing.is_set():
        time.sleep(0.5)
        wait_time += 0.5
    
    if not exists(log_file_path):
        logger.debug(f"Arquivo de log não encontrado após {max_wait}s: {log_file_path}")
        return
    
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Vai para o final do arquivo (se já existir conteúdo)
            try:
                f.seek(0, 2)
            except:
                pass
            
            while not _stop_tailing.is_set():
                line = f.readline()
                if line:
                    # Remove quebra de linha extra e imprime
                    print(f"{prefix} {line.rstrip()}", flush=True)
                else:
                    # Sem novas linhas, aguarda um pouco
                    time.sleep(0.1)
    except FileNotFoundError:
        # Arquivo foi deletado ou movido
        logger.debug(f"Arquivo de log não encontrado durante tail: {log_file_path}")
    except Exception as e:
        logger.debug(f"Erro ao fazer tail de {log_file_path}: {e}")


def find_python_executable():
    """Encontra o executável Python a ser usado."""
    # Tenta encontrar o Python do ambiente virtual primeiro
    venv_python = abspath(".env/bin/python3")
    if exists(venv_python):
        return venv_python
    
    # Tenta encontrar o Python do ambiente virtual (Windows)
    venv_python_win = abspath(".env/Scripts/python.exe")
    if exists(venv_python_win):
        return venv_python_win
    
    # Usa o Python do sistema
    return sys.executable


def run_main_with_restart(main_script='main.py', args_file='args.yaml', max_restarts=100, 
                          restart_delay=5, timeout=None, tail_logs=True):
    """
    Executa main.py com reinício automático em caso de falha.
    
    Parameters
    ----------
    main_script : str
        Nome do script principal (default: 'main.py')
    args_file : str
        Arquivo de argumentos para passar ao main.py (default: 'args.yaml')
    max_restarts : int
        Número máximo de tentativas de reinício (default: 100)
    restart_delay : int
        Delay em segundos entre tentativas de reinício (default: 5)
    timeout : int, optional
        Timeout em segundos para cada execução (None = sem timeout)
    tail_logs : bool
        Se True, faz tail dos arquivos de log do main.py em tempo real (default: True)
    """
    python_path = find_python_executable()
    main_path = abspath(main_script)
    
    if not exists(main_path):
        logger.error(f"Script não encontrado: {main_path}")
        sys.exit(1)
    
    logger.info(f"Python executable: {python_path}")
    logger.info(f"Main script: {main_path}")
    logger.info(f"Args file: {args_file}")
    logger.info(f"Max restarts: {max_restarts}")
    logger.info(f"Restart delay: {restart_delay}s")
    logger.info(f"Tail logs: {tail_logs}")
    
    # Carrega args para descobrir onde os logs são salvos
    log_files = []
    if tail_logs:
        try:
            from src.io_operations import load_args
            args = load_args(args_file)
            
            # Logs são salvos em ROOT_PATH/log/filename.log
            # O filename é baseado no model_dir ou data_path
            root_path = abspath(dirname(main_path))
            log_dir = join(root_path, "log")
            
            # Descobre o nome do logger (mesmo usado no main.py)
            model_dir = getattr(args, 'model_dir', None)
            data_path = getattr(args, 'data_path', None)
            
            if model_dir:
                version_name = os.path.split(model_dir)[-1]
                if not version_name:
                    version_name = os.path.split(os.path.dirname(model_dir))[-1]
            elif data_path:
                version_name = os.path.split(data_path)[-1]
            else:
                version_name = None
            
            # Procura arquivos de log
            if exists(log_dir):
                # Procura pelo log específico do experimento
                if version_name:
                    log_file = join(log_dir, f"{version_name}.log")
                    if exists(log_file):
                        log_files.append(log_file)
                
                # Também procura por outros arquivos .log recentes
                try:
                    for file in os.listdir(log_dir):
                        if file.endswith('.log'):
                            log_path = join(log_dir, file)
                            # Adiciona se não estiver já na lista
                            if log_path not in log_files:
                                log_files.append(log_path)
                except Exception:
                    pass
            
            if log_files:
                logger.info(f"Arquivos de log detectados: {len(log_files)}")
                for log_file in log_files:
                    logger.info(f"  - {log_file}")
            else:
                logger.info("Nenhum arquivo de log detectado. Apenas stdout/stderr será mostrado.")
                
        except Exception as e:
            logger.debug(f"Não foi possível descobrir arquivos de log automaticamente: {e}")
    
    restart_count = 0
    
    while restart_count < max_restarts:
        restart_count += 1
        
        logger.info("=" * 80)
        logger.info(f"Tentativa {restart_count}/{max_restarts}")
        logger.info(f"Iniciando execução em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Inicializa threads de tail (fora do try para estar disponível no except)
        tail_threads = []
        
        try:
            # Monta o comando
            cmd = [python_path, main_path, args_file]
            
            # Inicia threads de tail para arquivos de log
            if tail_logs and log_files:
                _stop_tailing.clear()
                for log_file in log_files:
                    if exists(log_file):
                        thread = threading.Thread(
                            target=tail_log_file,
                            args=(log_file, f"[{os.path.basename(log_file)}]"),
                            daemon=True
                        )
                        thread.start()
                        tail_threads.append(thread)
                        logger.info(f"Monitorando log: {log_file}")
            
            # Executa o processo
            start_time = time.time()
            print("\n" + "=" * 80)
            print("INÍCIO DA EXECUÇÃO DO MAIN.PY")
            print("=" * 80 + "\n")
            
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(abspath(main_script)),
                timeout=timeout,
                capture_output=False,  # Mostra output em tempo real
                text=True,
                bufsize=1  # Line buffered para output imediato
            )
            
            # Para threads de tail
            _stop_tailing.set()
            for thread in tail_threads:
                thread.join(timeout=1)
            
            elapsed_time = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("FIM DA EXECUÇÃO DO MAIN.PY")
            print("=" * 80 + "\n")
            
            # Verifica o código de retorno
            if result.returncode == 0:
                logger.info("=" * 80)
                logger.info("✓ Processo finalizado com sucesso!")
                logger.info(f"Tempo total de execução: {elapsed_time:.2f}s")
                logger.info("=" * 80)
                return 0
            
            else:
                logger.warning("=" * 80)
                logger.warning(f"⚠ Processo terminou com código de erro: {result.returncode}")
                logger.warning(f"Tempo de execução antes da falha: {elapsed_time:.2f}s")
                logger.warning("=" * 80)
                
                if restart_count < max_restarts:
                    logger.info(f"Aguardando {restart_delay}s antes de reiniciar...")
                    time.sleep(restart_delay)
                    continue
                else:
                    logger.error("Limite de reinícios atingido!")
                    return result.returncode
        
        except subprocess.TimeoutExpired:
            # Para threads de tail
            _stop_tailing.set()
            for thread in tail_threads:
                thread.join(timeout=1)
            
            logger.error("=" * 80)
            logger.error(f"⏱ Timeout! Processo excedeu {timeout}s")
            logger.error("=" * 80)
            
            if restart_count < max_restarts:
                logger.info(f"Aguardando {restart_delay}s antes de reiniciar...")
                time.sleep(restart_delay)
                continue
            else:
                logger.error("Limite de reinícios atingido!")
                return 1
        
        except KeyboardInterrupt:
            # Para threads de tail
            _stop_tailing.set()
            for thread in tail_threads:
                thread.join(timeout=1)
            
            logger.info("\n" + "=" * 80)
            logger.info("⚠ Interrompido pelo usuário (Ctrl+C)")
            logger.info("=" * 80)
            return 130  # Código padrão para SIGINT
        
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"✗ Erro inesperado ao executar processo: {e}")
            logger.error(f"Tipo do erro: {type(e).__name__}")
            logger.error("=" * 80)
            
            if restart_count < max_restarts:
                logger.info(f"Aguardando {restart_delay}s antes de reiniciar...")
                time.sleep(restart_delay)
                continue
            else:
                logger.error("Limite de reinícios atingido!")
                return 1
    
    logger.error("Limite máximo de reinícios atingido!")
    return 1


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Executa main.py com reinício automático em caso de falha',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Execução básica
  python call_main_as_subprocess.py
  
  # Especificar arquivo de argumentos
  python call_main_as_subprocess.py --args args_vDeepLabv3Vanilla.yaml
  
  # Limitar número de reinícios
  python call_main_as_subprocess.py --max-restarts 10
  
  # Aumentar delay entre reinícios
  python call_main_as_subprocess.py --restart-delay 30
        """
    )
    
    parser.add_argument(
        '--script',
        type=str,
        default='main.py',
        help='Script principal a ser executado (default: main.py)'
    )
    
    parser.add_argument(
        '--args',
        type=str,
        default='args.yaml',
        help='Arquivo de argumentos YAML (default: args.yaml)'
    )
    
    parser.add_argument(
        '--max-restarts',
        type=int,
        default=100,
        help='Número máximo de reinícios (default: 100)'
    )
    
    parser.add_argument(
        '--restart-delay',
        type=int,
        default=5,
        help='Delay em segundos entre reinícios (default: 5)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout em segundos para cada execução (default: sem timeout)'
    )
    
    parser.add_argument(
        '--no-tail-logs',
        action='store_true',
        help='Desabilita o tail de arquivos de log (mostra apenas stdout/stderr)'
    )
    
    args = parser.parse_args()
    
    exit_code = run_main_with_restart(
        main_script=args.script,
        args_file=args.args,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay,
        timeout=args.timeout,
        tail_logs=not args.no_tail_logs
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
