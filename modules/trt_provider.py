"""
TensorRT Execution Provider configuration for Deep-Live-Cam.

Provides:
  - build_provider_chain()  : returns a ready-to-use ORT provider list
  - is_trt_available()      : runtime DLL check (no model required)

Why this file exists
--------------------
ORT's fallback behaviour is brutal: if TensorrtExecutionProvider fails to
load its native DLL, ORT falls all the way back to CPU and skips CUDA entirely.
We therefore probe TRT availability at startup, exclude it from the chain if
the DLL isn't present, and log clearly what provider is actually being used.

Engine caching
--------------
TRT compiles each ONNX graph into a hardware-optimised engine the first time.
This takes 2-5 minutes. The compiled engine is cached to disk so subsequent
runs are instant. Cache directory: modules.globals.trt_cache_dir

FP16 mode
---------
Enabled by default (--trt-fp16). The RTX 4070 SUPER has excellent FP16
throughput via Tensor Cores. No quality difference vs FP32 on swap models.

Workspace
---------
ORT/TRT uses a scratch memory pool during optimisation. We default to 4 GB
(--trt-workspace-gb). The 4070 SUPER has 12 GB VRAM so this is conservative;
increase if TRT complains about workspace during engine build.
"""

import ctypes
import os
from typing import List, Tuple, Any

import modules.globals
from modules.logger import get_logger

_log = get_logger(__name__)

_TRT_CHECKED: bool = False
_TRT_AVAILABLE: bool = False


def is_trt_available() -> bool:
    """
    Check whether the TensorRT runtime DLL is loadable.
    Result is cached after the first call.
    """
    global _TRT_CHECKED, _TRT_AVAILABLE
    if _TRT_CHECKED:
        return _TRT_AVAILABLE
    _TRT_CHECKED = True

    import os, glob as _glob
    # Add CUDA 12 bin to PATH if present (helps ORT find CUDA DLLs on Windows)
    cuda_dirs = sorted(_glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*'), reverse=True)
    if cuda_dirs:
        cuda_bin = os.path.join(cuda_dirs[0], 'bin')
        if cuda_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')
            _log.debug('Prepended CUDA 12 bin to PATH: %s', cuda_bin)

    # ORT 1.21+ bundles a preload helper that resolves CUDA/cuDNN/MSVC paths
    try:
        import onnxruntime as ort
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
            _log.debug('ort.preload_dlls() called successfully')
    except Exception as e:
        _log.debug('preload_dlls failed (non-fatal): %s', e)

    # Try to load the TensorRT core inference DLL
    dll_candidates = ['nvinfer_10', 'nvinfer_10.dll', 'libnvinfer.so.10']
    for dll in dll_candidates:
        try:
            ctypes.CDLL(dll)
            _log.info('TensorRT DLL loaded: %s → TRT EP is available', dll)
            _TRT_AVAILABLE = True
            return True
        except OSError:
            _log.debug('TRT DLL not found: %s', dll)

    _log.warning(
        'TensorRT runtime DLL (nvinfer_10.dll) not found in PATH. '
        'TensorRT EP will not be used. '
        'To install: download TensorRT 10.x for CUDA 12 from '
        'https://developer.nvidia.com/tensorrt and add its lib/ to PATH.'
    )
    _TRT_AVAILABLE = False
    return False


def _make_trt_options() -> dict:
    """Build the TensorRT EP options dict from globals."""
    cache_dir = modules.globals.trt_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    workspace_bytes = int(modules.globals.trt_workspace_gb * 1024 ** 3)
    opts = {
        'device_id': 0,
        'trt_fp16_enable': modules.globals.trt_fp16,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_dir,
        'trt_max_workspace_size': workspace_bytes,
        'trt_builder_optimization_level': 3,     # thorough optimisation
        'trt_auxiliary_streams': 2,               # async stream overlap
        'trt_timing_cache_enable': True,           # reuse calibration timing
        'trt_timing_cache_path': cache_dir,
    }
    _log.info('TensorRT EP options: fp16=%s  workspace=%.1fGB  cache=%s',
              modules.globals.trt_fp16, modules.globals.trt_workspace_gb, cache_dir)
    return opts


def _make_cuda_options() -> dict:
    """Build the CUDA EP options dict."""
    return {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'HEURISTIC',
        'do_copy_in_default_stream': True,
    }


def build_provider_chain() -> List[Any]:
    """
    Return an ORT provider list appropriate for the current runtime.

    Priority:
      1. TensorrtExecutionProvider  (if TRT DLL present and requested)
      2. CUDAExecutionProvider      (if CUDA requested)
      3. CPUExecutionProvider       (always as final fallback)

    Each entry is either a plain string (CPU) or a (name, options) tuple.
    """
    import onnxruntime as ort
    available = ort.get_available_providers()
    requested = modules.globals.execution_providers  # already decoded ORT names

    chain = []
    used_labels = []

    # --- TensorRT ---
    if 'TensorrtExecutionProvider' in requested and 'TensorrtExecutionProvider' in available:
        if is_trt_available():
            chain.append(('TensorrtExecutionProvider', _make_trt_options()))
            used_labels.append('TensorRT')
            _log.info('TensorRT EP added to provider chain (first-run engine build may take ~5 min)')
        else:
            _log.warning('TensorRT EP requested but DLL unavailable — falling back to CUDA/CPU')

    # --- CUDA ---
    if 'CUDAExecutionProvider' in requested and 'CUDAExecutionProvider' in available:
        chain.append(('CUDAExecutionProvider', _make_cuda_options()))
        used_labels.append('CUDA')
        _log.info('CUDA EP added to provider chain')

    # --- CPU fallback (always) ---
    chain.append('CPUExecutionProvider')
    used_labels.append('CPU')

    _log.info('Final ORT provider chain: %s', ' → '.join(used_labels))
    return chain


def log_provider_chain_result(session_providers: List[str]) -> None:
    """Log which provider ORT actually chose for a loaded session."""
    if not session_providers:
        _log.warning('Session returned empty provider list')
        return
    active = session_providers[0]
    _log.info('ORT session active provider: %s', active)
    if active == 'CPUExecutionProvider' and len(session_providers) > 1:
        _log.warning(
            'Session fell back to CPU despite requesting %s — '
            'check that the required runtime libraries are installed and in PATH.',
            session_providers[1:]
        )
