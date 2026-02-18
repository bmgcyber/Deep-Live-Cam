"""
Deep-Live-Cam Environment Check & Baseline Benchmark
Run this before starting any development phase to verify your setup.

Usage:
    python tools/env_check.py
    python tools/env_check.py --benchmark   # also runs a quick inference benchmark
"""

import sys
import os
import time
import argparse

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.logger import get_logger, log_env_summary, Timer

log = get_logger('env_check')

PASS = '[PASS]'
FAIL = '[FAIL]'
WARN = '[WARN]'


def check_python() -> bool:
    log.info('--- Python ---')
    ver = sys.version_info
    log.info('Python %d.%d.%d at %s', ver.major, ver.minor, ver.micro, sys.executable)
    if ver < (3, 9):
        log.error('%s Python 3.9+ required, got %d.%d', FAIL, ver.major, ver.minor)
        return False
    log.info('%s Python version OK (%d.%d)', PASS, ver.major, ver.minor)
    return True


def check_ffmpeg() -> bool:
    log.info('--- FFmpeg ---')
    import shutil
    path = shutil.which('ffmpeg')
    if path:
        log.info('%s ffmpeg found at %s', PASS, path)
        return True
    log.warning('%s ffmpeg not found in PATH — video processing will fail', FAIL)
    return False


def check_pytorch() -> bool:
    log.info('--- PyTorch / CUDA ---')
    try:
        import torch
        log.info('PyTorch version: %s', torch.__version__)
        if not torch.cuda.is_available():
            log.warning('%s CUDA not available via PyTorch', WARN)
            return False
        props = torch.cuda.get_device_properties(0)
        log.info('%s GPU: %s', PASS, props.name)
        log.info('   VRAM: %.1f GB', props.total_memory / 1e9)
        log.info('   Compute: SM %d.%d', props.major, props.minor)
        log.info('   CUDA runtime: %s', torch.version.cuda)
        log.info('   cuDNN: %s', torch.backends.cudnn.version())
        return True
    except ImportError:
        log.error('%s PyTorch not installed', FAIL)
        return False


def check_onnxruntime() -> bool:
    log.info('--- ONNX Runtime ---')
    try:
        import onnxruntime as ort
        log.info('ORT version: %s', ort.__version__)
        providers = ort.get_available_providers()
        log.info('Available providers: %s', providers)

        has_cuda = 'CUDAExecutionProvider' in providers
        has_trt = 'TensorrtExecutionProvider' in providers
        has_dml = 'DmlExecutionProvider' in providers

        if has_cuda:
            log.info('%s CUDAExecutionProvider available', PASS)
        else:
            log.warning('%s CUDAExecutionProvider NOT available — inference will be CPU-only!', FAIL)
            log.warning('   Fix: pip install onnxruntime-gpu==1.22.0')

        if has_trt:
            log.info('%s TensorrtExecutionProvider available', PASS)
        else:
            log.info('%s TensorrtExecutionProvider not available (Phase 1 goal)', WARN)

        if has_dml:
            log.info('%s DmlExecutionProvider available (DirectML fallback)', PASS)

        return has_cuda
    except ImportError:
        log.error('%s onnxruntime-gpu not installed', FAIL)
        return False


def check_insightface() -> bool:
    log.info('--- InsightFace ---')
    try:
        import insightface
        log.info('%s insightface %s', PASS, insightface.__version__)
        return True
    except ImportError:
        log.error('%s insightface not installed', FAIL)
        return False


def check_models() -> bool:
    log.info('--- Models ---')
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    required = ['inswapper_128_fp16.onnx', 'inswapper_128.onnx']
    found_any = False
    for model in required:
        path = os.path.join(models_dir, model)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            log.info('%s %s (%.1f MB)', PASS, model, size_mb)
            found_any = True
        else:
            log.warning('%s %s not found at %s', WARN, model, path)

    # buffalo_l is a directory
    buffalo_path = os.path.join(models_dir, 'buffalo_l')
    if os.path.isdir(buffalo_path):
        files = os.listdir(buffalo_path)
        log.info('%s buffalo_l model dir found (%d files)', PASS, len(files))
        log.debug('   buffalo_l contents: %s', files)
        found_any = True
    else:
        log.warning('%s buffalo_l not found at %s', WARN, buffalo_path)

    return found_any


def check_cuda_toolkits() -> None:
    log.info('--- CUDA Toolkit Installations ---')
    import glob
    cuda_dirs = glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*')
    if cuda_dirs:
        for d in sorted(cuda_dirs):
            log.info('  Found: %s', d)
    else:
        log.warning('No CUDA toolkit directories found')


def _try_load_session(model_path: str, providers: list):
    """Try to load an ORT session. Returns (session, active_provider) or (None, None) on failure."""
    import onnxruntime as ort
    import warnings
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 4  # silent
    try:
        # preload_dlls() auto-resolves bundled CUDA/cuDNN paths (ORT 1.21+)
        if hasattr(ort, 'preload_dlls'):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        active = session.get_providers()[0]
        # Check it actually used the requested provider (ORT may silently fallback)
        if active == 'CPUExecutionProvider' and providers[0] != 'CPUExecutionProvider':
            log.debug('Session silently fell back to CPU from %s', providers[0])
        return session, active
    except Exception as e:
        log.debug('Session load failed with providers %s: %s', providers, e)
        return None, None


def benchmark_ort_inference() -> None:
    """Quick ORT inference speed test using a dummy input on inswapper shape.

    Tries each provider in order: TensorRT → CUDA → CPU, benchmarking each
    that successfully loads so you get a full picture of what's usable.
    """
    log.info('--- ORT Inference Benchmark ---')
    import onnxruntime as ort
    import numpy as np

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    model_path = os.path.join(models_dir, 'inswapper_128_fp16.onnx')
    if not os.path.exists(model_path):
        model_path = os.path.join(models_dir, 'inswapper_128.onnx')
    if not os.path.exists(model_path):
        log.warning('%s No inswapper model found, skipping ORT benchmark', WARN)
        return

    available = ort.get_available_providers()
    # Try each provider independently (don't chain TRT+CUDA — ORT falls back to CPU when TRT DLL missing)
    provider_chains_to_try = []
    if 'TensorrtExecutionProvider' in available:
        provider_chains_to_try.append(['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    if 'CUDAExecutionProvider' in available:
        provider_chains_to_try.append(['CUDAExecutionProvider', 'CPUExecutionProvider'])
    provider_chains_to_try.append(['CPUExecutionProvider'])

    log.info('Loading model: %s', os.path.basename(model_path))

    n_runs = 30
    results = {}

    for provider_chain in provider_chains_to_try:
        label = provider_chain[0]
        if label in results:
            continue  # already benchmarked this one
        log.info('Trying provider: %s ...', label)
        session, active = _try_load_session(model_path, provider_chain)
        if session is None:
            log.warning('  %s %s: failed to load session', FAIL, label)
            continue
        actual_label = active
        log.info('  Loaded. Active provider: %s', actual_label)

        inputs = session.get_inputs()
        dummy_inputs = {
            inp.name: np.random.rand(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(np.float32)
            for inp in inputs
        }

        # Warm up
        try:
            for _ in range(3):
                session.run(None, dummy_inputs)

            # Benchmark
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                session.run(None, dummy_inputs)
                times.append(time.perf_counter() - t0)

            avg_ms = (sum(times) / len(times)) * 1000
            fps = 1000 / avg_ms
            results[actual_label] = (avg_ms, fps)
            log.info('  %s avg: %.1f ms -> %.1f FPS theoretical max', actual_label, avg_ms, fps)
        except Exception as e:
            log.warning('  Benchmark run failed for %s: %s', label, e)

    log.info('--- Benchmark Summary ---')
    for provider, (avg_ms, fps) in results.items():
        log.info('  %-35s  avg %.1f ms  =  %.1f FPS', provider, avg_ms, fps)
    if not results:
        log.warning('No providers benchmarked successfully')

    # Placeholder for the rest of the original function body
    return



def main():
    parser = argparse.ArgumentParser(description='Deep-Live-Cam environment check')
    parser.add_argument('--benchmark', action='store_true', help='Run inference benchmark')
    args = parser.parse_args()

    log.info('Deep-Live-Cam Environment Check')
    log.info('=' * 60)

    results = {
        'Python': check_python(),
        'FFmpeg': check_ffmpeg(),
        'PyTorch/CUDA': check_pytorch(),
        'ONNX Runtime': check_onnxruntime(),
        'InsightFace': check_insightface(),
        'Models': check_models(),
    }
    check_cuda_toolkits()

    log.info('=' * 60)
    log.info('SUMMARY:')
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        log.info('  %s %s', status, name)
        if not ok:
            all_pass = False

    if all_pass:
        log.info('All checks passed. App is ready to run.')
    else:
        log.warning('Some checks failed. See above for details.')

    if args.benchmark:
        log.info('')
        benchmark_ort_inference()


if __name__ == '__main__':
    main()
