# Deep-Live-Cam Improvement Plan

## Prerequisites (Phase 0)
- [ ] Verify Windows Python 3.10+ installed
- [ ] Verify NVIDIA CUDA Toolkit installed (12.x preferred)
- [ ] Verify cuDNN installed
- [ ] Install project dependencies: `pip install -r requirements.txt`
- [ ] Install CUDA-enabled onnxruntime: `pip install onnxruntime-gpu`
- [ ] Download models (inswapper_128_fp16.onnx, buffalo_l) into `models/`
- [ ] Run the app stock to confirm baseline works: `python run.py --execution-provider cuda`
- [ ] Benchmark baseline FPS at 720p on the 4070 SUPER

---

## Phase 1: TensorRT Integration
**Goal:** Compile ONNX models to TensorRT for maximum inference speed on the 4070 SUPER.

### Steps
- [ ] Install TensorRT (pip install tensorrt + NVIDIA TensorRT libs)
- [ ] Install onnxruntime with TensorRT support (`onnxruntime-gpu` already includes TensorRT EP)
- [ ] Add `TensorrtExecutionProvider` to the execution provider chain in `modules/core.py`
- [ ] Configure TensorRT EP options:
  - FP16 mode enabled (4070 SUPER has great FP16)
  - Engine caching (so first-run compilation is saved to disk)
  - Workspace size tuned for 12GB VRAM
- [ ] Handle first-run engine build (takes a few minutes, show progress to user)
- [ ] Add `--execution-provider tensorrt` CLI option
- [ ] Benchmark: compare CUDA vs TensorRT FPS
- [ ] Ensure fallback to CUDA if TensorRT unavailable

### Files to modify
- `modules/core.py` - Add TensorRT provider option
- `modules/processors/frame/face_swapper.py` - Model loading with TRT EP
- `modules/processors/frame/face_enhancer.py` - GFPGAN model loading with TRT EP
- `modules/face_analyser.py` - InsightFace provider config

---

## Phase 2: Virtual Camera Output
**Goal:** Pipe swapped face output as a virtual webcam usable in any app (Zoom, Discord, OBS, etc.)

### Steps
- [ ] Install `pyvirtualcam` (cross-platform: OBS Virtual Cam on Windows, v4l2loopback on Linux)
- [ ] Add virtual camera toggle in UI and CLI (`--virtual-cam`)
- [ ] Create virtual cam writer that runs alongside the preview display
- [ ] Handle resolution/FPS matching between capture and virtual output
- [ ] Add virtual cam status indicator in UI
- [ ] Test with Discord/Zoom/OBS on Windows
- [ ] Ensure clean start/stop (no zombie virtual cams)

### Files to modify
- `modules/ui.py` - Add virtual cam toggle switch
- `modules/ui.py` - Integrate into webcam preview loop
- `modules/globals.py` - Add virtual cam state
- `requirements.txt` - Add pyvirtualcam

### Windows requirement
- User needs OBS Studio installed (provides the virtual camera backend)

---

## Phase 3: UI Modernization
**Goal:** Replace or significantly improve the UI for a modern, intuitive experience.

### Options (decide before starting)
- **Option A: Web UI (Gradio)** - Fast to build, good-looking, works in browser, easy remote access
- **Option B: Web UI (FastAPI + lightweight frontend)** - More control, cleaner, but more work
- **Option C: Overhaul CustomTkinter** - Keep desktop app, just make it not ugly

### Core UI improvements regardless of framework
- [ ] Side-by-side live preview (original vs swapped)
- [ ] Face gallery panel - save/load/switch source faces quickly
- [ ] Drag-and-drop source face images
- [ ] Live FPS/latency/GPU usage stats overlay
- [ ] Resolution and quality presets
- [ ] Clean settings panel (collapsible sections instead of a wall of toggles)
- [ ] Status bar with GPU info, model loaded, virtual cam status

### Files to modify
- `modules/ui.py` - Major rewrite or replacement
- Possibly new `modules/web_ui.py` if going web route
- `run.py` - Add UI selection flag

---

## Phase 4: Temporal Face Smoothing
**Goal:** Eliminate face flicker/jitter between frames for convincing real-time output.

### Steps
- [ ] Implement face position smoothing (exponential moving average on bounding box coordinates)
- [ ] Implement face landmark smoothing (EMA on 106-point landmarks across frames)
- [ ] Add confidence-based blending (low confidence = blend more with previous frame)
- [ ] Handle face lost/found transitions gracefully (fade in/out instead of pop)
- [ ] Add smoothing strength slider in UI
- [ ] Improve the "detect every N frames" logic with proper tracking (optical flow or simple IoU tracker)

### Files to modify
- `modules/processors/frame/face_swapper.py` - Core smoothing logic
- `modules/face_analyser.py` - Tracked face state management
- `modules/ui.py` - Smoothing controls
- `modules/globals.py` - Smoothing parameters

---

## Phase 5: Hotkeys
**Goal:** Keyboard shortcuts for common actions during live use.

### Planned hotkeys
- [ ] `Space` or `F1` - Toggle face swap on/off
- [ ] `F2` - Cycle through saved source faces
- [ ] `F3` - Toggle virtual camera
- [ ] `F5` - Screenshot current frame
- [ ] `Esc` - Stop live preview
- [ ] `+`/`-` - Adjust transparency
- [ ] `M` - Toggle mouth mask

### Steps
- [ ] Add `pynput` or `keyboard` library for global hotkey capture
- [ ] Create hotkey manager module with configurable bindings
- [ ] Wire hotkeys to existing UI toggle functions
- [ ] Add hotkey reference overlay (show/hide with `?` or `H`)
- [ ] Save custom hotkey bindings to config
- [ ] Ensure hotkeys work whether preview window or main window is focused

### Files to modify
- New `modules/hotkeys.py` - Hotkey manager
- `modules/ui.py` - Wire up actions
- `modules/globals.py` - Hotkey state
- `requirements.txt` - Add keyboard/pynput

---

## Notes
- **Windows compatibility** is required throughout - test every change on Windows
- **4070 SUPER (12GB VRAM, Ada Lovelace SM 8.9)** is the target GPU
- Keep Linux/Mac compatibility where reasonable but Windows is primary
- Each phase should be independently testable and mergeable
