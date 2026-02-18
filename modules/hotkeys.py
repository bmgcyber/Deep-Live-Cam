"""
Global hotkey manager for Deep-Live-Cam.

Captures keyboard shortcuts regardless of which window is focused, using
pynput's background listener. Actions are dispatched via registered
callbacks so this module stays independent of the UI layer.

Default bindings
----------------
    F1      Toggle face swap on/off (opacity 0 <-> 1)
    F2      Toggle virtual camera on/off
    F3      Toggle side-by-side preview
    F4      Toggle face enhancer
    F5      Screenshot current frame (saves to working directory)
    F8      Toggle show FPS
    Esc     Stop / close live preview (sets stop_event)
    +/=     Increase opacity by 10%
    -/_     Decrease opacity by 10%

Usage
-----
    from modules.hotkeys import HotkeyManager

    mgr = HotkeyManager()
    mgr.register(Key.f1, my_callback)  # optional extra bindings
    mgr.start()
    # ... app runs ...
    mgr.stop()

All default actions are wired automatically; the UI just needs to call
`start()` and `stop()`, and optionally register a `screenshot_callback`
and `stop_callback`.

Thread safety
-------------
pynput listener runs in its own daemon thread.  Callbacks are invoked from
that thread, so keep them lightweight and thread-safe.
"""

import threading
from typing import Callable, Dict, Optional

import modules.globals
from modules.logger import get_logger

_log = get_logger(__name__)

try:
    from pynput import keyboard as _kb
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False
    _log.warning('pynput not installed — hotkeys disabled. Run: pip install pynput')


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class HotkeyManager:
    """
    Global hotkey listener.  Runs a daemon thread; safe to start/stop
    multiple times during the app lifecycle.
    """

    def __init__(self) -> None:
        self._listener: Optional[object] = None
        self._lock = threading.Lock()
        self._running = False

        # User-supplied callbacks
        self.screenshot_callback: Optional[Callable] = None
        self.stop_callback: Optional[Callable] = None   # called on Esc
        self.status_callback: Optional[Callable[[str], None]] = None

        # Bindings: Key -> callable
        self._bindings: Dict = {}
        self._setup_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening for hotkeys in a background thread."""
        if not _PYNPUT_AVAILABLE:
            return
        with self._lock:
            if self._running:
                return
            self._listener = _kb.Listener(on_press=self._on_press)
            self._listener.daemon = True
            self._listener.start()
            self._running = True
            _log.info('Hotkey listener started')

    def stop(self) -> None:
        """Stop the hotkey listener."""
        with self._lock:
            if not self._running:
                return
            try:
                if self._listener:
                    self._listener.stop()
            except Exception as e:
                _log.debug('Error stopping hotkey listener: %s', e)
            self._running = False
            self._listener = None
            _log.info('Hotkey listener stopped')

    def register(self, key, callback: Callable) -> None:
        """Register an additional hotkey binding."""
        self._bindings[key] = callback

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_defaults(self) -> None:
        if not _PYNPUT_AVAILABLE:
            return
        Key = _kb.Key
        KeyCode = _kb.KeyCode

        self._bindings = {
            Key.f1: self._toggle_swap,
            Key.f2: self._toggle_virtual_cam,
            Key.f3: self._toggle_side_by_side,
            Key.f4: self._toggle_face_enhancer,
            Key.f5: self._take_screenshot,
            Key.f8: self._toggle_show_fps,
            Key.esc: self._stop_preview,
            KeyCode.from_char('+'): self._opacity_up,
            KeyCode.from_char('='): self._opacity_up,    # same physical key on US layout
            KeyCode.from_char('-'): self._opacity_down,
            KeyCode.from_char('_'): self._opacity_down,
        }

    def _on_press(self, key) -> None:
        callback = self._bindings.get(key)
        if callback:
            try:
                callback()
            except Exception as e:
                _log.warning('Hotkey callback error for %s: %s', key, e)

    def _status(self, msg: str) -> None:
        _log.info('Hotkey: %s', msg)
        if self.status_callback:
            try:
                self.status_callback(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Default actions
    # ------------------------------------------------------------------

    def _toggle_swap(self) -> None:
        if modules.globals.opacity < 0.05:
            modules.globals.opacity = 1.0
            modules.globals.face_swapper_enabled = True
            self._status('Face swap ON (F1)')
        else:
            modules.globals.opacity = 0.0
            self._status('Face swap OFF (F1)')

    def _toggle_virtual_cam(self) -> None:
        modules.globals.virtual_cam = not modules.globals.virtual_cam
        state = 'ON' if modules.globals.virtual_cam else 'OFF'
        self._status(f'Virtual cam {state} (F2)')

    def _toggle_side_by_side(self) -> None:
        modules.globals.side_by_side = not modules.globals.side_by_side
        state = 'ON' if modules.globals.side_by_side else 'OFF'
        self._status(f'Side-by-side {state} (F3)')

    def _toggle_face_enhancer(self) -> None:
        current = modules.globals.fp_ui.get('face_enhancer', False)
        modules.globals.fp_ui['face_enhancer'] = not current
        state = 'ON' if modules.globals.fp_ui['face_enhancer'] else 'OFF'
        self._status(f'Face enhancer {state} (F4)')

    def _take_screenshot(self) -> None:
        if self.screenshot_callback:
            try:
                path = self.screenshot_callback()
                if path:
                    self._status(f'Screenshot saved: {path} (F5)')
                else:
                    self._status('Screenshot failed (F5)')
            except Exception as e:
                _log.warning('Screenshot failed: %s', e)
        else:
            self._status('Screenshot: no callback registered (F5)')

    def _toggle_show_fps(self) -> None:
        modules.globals.show_fps = not modules.globals.show_fps
        state = 'ON' if modules.globals.show_fps else 'OFF'
        self._status(f'Show FPS {state} (F8)')

    def _stop_preview(self) -> None:
        self._status('Stop preview (Esc)')
        if self.stop_callback:
            try:
                self.stop_callback()
            except Exception as e:
                _log.warning('Stop callback error: %s', e)

    def _opacity_up(self) -> None:
        modules.globals.opacity = _clamp(modules.globals.opacity + 0.1, 0.0, 1.0)
        self._status(f'Opacity {int(modules.globals.opacity * 100)}% (+)')

    def _opacity_down(self) -> None:
        modules.globals.opacity = _clamp(modules.globals.opacity - 0.1, 0.0, 1.0)
        self._status(f'Opacity {int(modules.globals.opacity * 100)}% (-)')


# Module-level singleton — shared across the app
hotkey_manager = HotkeyManager()
