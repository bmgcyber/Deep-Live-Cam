"""
Centralized logging for Deep-Live-Cam.
Every module should use get_logger(__name__) instead of print().
Logs go to both console (INFO+) and a rotating file (DEBUG+).
"""

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'dlc.log')

_initialized = False


def _init_logging() -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger('dlc')
    root.setLevel(logging.DEBUG)
    root.propagate = False  # Don't bubble up to the root logger (avoids double-printing)

    # --- Console handler: INFO and above, human-readable ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    root.addHandler(console)

    # --- File handler: DEBUG and above, verbose with line numbers ---
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root.addHandler(file_handler)
    except Exception as e:
        print(f'[LOGGER] Warning: Could not create log file at {LOG_FILE}: {e}')

    root.debug('Logging initialized. Log file: %s', LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger scoped under the 'dlc' namespace.
    Usage:
        from modules.logger import get_logger
        log = get_logger(__name__)
        log.info('Starting...')
        log.debug('Frame shape: %s', frame.shape)
    """
    _init_logging()
    # Strip 'modules.' prefix for cleaner log names
    short_name = name.replace('modules.', '').replace('processors.frame.', '')
    return logging.getLogger(f'dlc.{short_name}')


class Timer:
    """Context manager for timing code blocks and logging the result."""

    def __init__(self, label: str, logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
        self.label = label
        self.logger = logger or get_logger('timer')
        self.level = level
        self.elapsed: float = 0.0

    def __enter__(self) -> 'Timer':
        self.start = time.perf_counter()
        self.logger.log(self.level, '[TIMER] START  %s', self.label)
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start
        self.logger.log(self.level, '[TIMER] FINISH %s — %.3fs', self.label, self.elapsed)


def log_env_summary(logger: Optional[logging.Logger] = None) -> None:
    """Log a full environment summary: Python, GPU, CUDA, ORT providers."""
    log = logger or get_logger('env')
    log.info('=' * 60)
    log.info('ENVIRONMENT SUMMARY')
    log.info('=' * 60)

    import platform
    log.info('OS: %s %s', platform.system(), platform.version())
    log.info('Python: %s (%s)', sys.version, sys.executable)

    # PyTorch / CUDA
    try:
        import torch
        log.info('PyTorch: %s', torch.__version__)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            log.info('GPU: %s | VRAM: %.1f GB | SM: %d.%d',
                     props.name,
                     props.total_memory / 1e9,
                     props.major, props.minor)
            log.info('CUDA runtime: %s', torch.version.cuda)
            log.info('cuDNN: %s', torch.backends.cudnn.version())
        else:
            log.warning('CUDA not available via PyTorch')
    except ImportError:
        log.warning('PyTorch not installed')

    # ONNX Runtime
    try:
        import onnxruntime as ort
        log.info('ONNX Runtime: %s', ort.__version__)
        log.info('ORT providers: %s', ort.get_available_providers())
    except ImportError:
        log.warning('ONNX Runtime not installed')

    # CUDA toolkit directories
    import glob
    cuda_dirs = glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*')
    if cuda_dirs:
        log.info('CUDA toolkits: %s', cuda_dirs)

    log.info('=' * 60)
