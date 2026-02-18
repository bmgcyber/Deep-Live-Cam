"""
Virtual Camera output for Deep-Live-Cam.

Wraps pyvirtualcam to stream the processed (face-swapped) frames as a
virtual webcam device that other apps (Zoom, Discord, OBS, etc.) can select.

Windows backend: OBS Virtual Camera (ships with OBS Studio 26+).
Linux backend:   v4l2loopback kernel module.

Usage
-----
    from modules.virtual_cam import VirtualCamWriter

    vcam = VirtualCamWriter()
    vcam.start(width=960, height=540, fps=30)   # opens the device
    vcam.send_frame(bgr_frame)                   # call per processed frame
    vcam.stop()                                  # closes the device cleanly

Notes
-----
- pyvirtualcam expects RGB uint8 frames; we convert from BGR internally.
- If OBS Virtual Camera is not installed, start() returns False and logs a
  clear error rather than crashing the whole app.
- Thread-safe: send_frame() can be called from the processing thread while
  the main thread drives the UI.
"""

import threading
from typing import Optional

import numpy as np

from modules.logger import get_logger

_log = get_logger(__name__)


class VirtualCamWriter:
    """Manages a single pyvirtualcam output device."""

    def __init__(self) -> None:
        self._cam = None          # pyvirtualcam.Camera instance
        self._lock = threading.Lock()
        self._running = False
        self._width = 0
        self._height = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, width: int, height: int, fps: float = 30.0) -> bool:
        """Open the virtual camera device.

        Returns True on success, False if pyvirtualcam/OBS is unavailable.
        """
        with self._lock:
            if self._running:
                return True

            try:
                import pyvirtualcam
            except ImportError:
                _log.error(
                    'pyvirtualcam is not installed. '
                    'Run: pip install pyvirtualcam'
                )
                return False

            try:
                self._cam = pyvirtualcam.Camera(
                    width=width,
                    height=height,
                    fps=fps,
                    fmt=pyvirtualcam.PixelFormat.RGB,
                    print_fps=False,
                )
                self._width = width
                self._height = height
                self._running = True
                _log.info(
                    'Virtual camera started: %dx%d @ %.0f fps  device=%s',
                    width, height, fps, self._cam.device
                )
                return True
            except Exception as e:
                _log.error(
                    'Failed to open virtual camera: %s\n'
                    '  On Windows, install OBS Studio (which includes OBS Virtual Camera).\n'
                    '  On Linux, load the v4l2loopback kernel module.',
                    e,
                )
                self._cam = None
                return False

    def send_frame(self, bgr_frame: np.ndarray) -> None:
        """Send a BGR frame to the virtual camera.

        Silently no-ops if the device is not running.
        """
        if not self._running:
            return

        with self._lock:
            if self._cam is None:
                return
            try:
                # pyvirtualcam expects RGB; OpenCV gives us BGR
                if bgr_frame.shape[1] != self._width or bgr_frame.shape[0] != self._height:
                    import cv2
                    bgr_frame = cv2.resize(bgr_frame, (self._width, self._height))
                rgb = bgr_frame[:, :, ::-1]   # BGR -> RGB view (no copy)
                self._cam.send(rgb)
            except Exception as e:
                _log.warning('Virtual cam send_frame failed: %s', e)

    def stop(self) -> None:
        """Close the virtual camera device."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            try:
                if self._cam is not None:
                    self._cam.close()
                    _log.info('Virtual camera stopped')
            except Exception as e:
                _log.warning('Error closing virtual camera: %s', e)
            finally:
                self._cam = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def device(self) -> Optional[str]:
        if self._cam is not None:
            return str(self._cam.device)
        return None
