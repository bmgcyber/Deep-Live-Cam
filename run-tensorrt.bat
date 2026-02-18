@echo off
rem Run Deep-Live-Cam with TensorRT execution provider (fastest on NVIDIA GPUs).
rem On first run, TRT will compile the model into an optimised engine (~5 min).
rem Subsequent runs load the cached engine instantly.
rem
rem Requirements:
rem   - TensorRT 10.x runtime DLLs in PATH (nvinfer_10.dll, nvinfer_plugin_10.dll, etc.)
rem   - Download TensorRT 10.x for CUDA 12 from https://developer.nvidia.com/tensorrt
rem   - Add TensorRT\lib\ to your system PATH, then reboot or open a new terminal
rem
C:\Python312\python.exe run.py --execution-provider tensorrt cuda %*
