"""
Temporal face tracker for Deep-Live-Cam.

Eliminates face jitter/flicker in the live preview by applying Exponential
Moving Average (EMA) to face bounding box coordinates and landmarks across
frames.

How it works
------------
Each frame, detected faces are matched to tracked faces by IoU of their
bounding boxes. Matched faces have their positions blended with the previous
tracked position using EMA. Unmatched tracked faces are kept for a grace
period (LOST_GRACE_FRAMES) before being discarded. Unmatched detected faces
start new tracks.

Configuration
-------------
- smooth_alpha (globals): EMA blending weight for new detections.
  0.0 = fully smooth (always use old position, never updates)
  1.0 = no smoothing (always use raw detection)
  0.3 = recommended default (70% previous, 30% new per frame)

Usage
-----
    from modules.face_tracker import FaceTracker

    tracker = FaceTracker()
    smoothed_faces = tracker.update(detected_faces)   # call each frame
    tracker.reset()                                    # call on source change

The returned face objects have their bbox and landmark_2d_106 attributes
replaced with the smoothed versions. All other face attributes are passed
through from the most recent detection.
"""

import copy
from typing import List, Optional

import numpy as np

import modules.globals
from modules.logger import get_logger

_log = get_logger(__name__)

# Number of frames to keep a track alive after the face disappears
LOST_GRACE_FRAMES = 6

# Minimum IoU to consider a detection-track pair a match
IOU_THRESHOLD = 0.3


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Track:
    """Internal state for a single tracked face."""

    def __init__(self, face) -> None:
        self.face = copy.copy(face)          # most recent detection
        self.bbox = face.bbox.copy().astype(np.float32)
        self.landmarks = (
            face.landmark_2d_106.copy().astype(np.float32)
            if face.landmark_2d_106 is not None else None
        )
        self.lost_frames = 0

    def update(self, face, alpha: float) -> None:
        """Blend new detection into tracked position with EMA."""
        self.face = copy.copy(face)
        self.lost_frames = 0
        # EMA blend on bbox
        self.bbox = alpha * face.bbox.astype(np.float32) + (1.0 - alpha) * self.bbox
        # EMA blend on landmarks (if available)
        if face.landmark_2d_106 is not None and self.landmarks is not None:
            new_lm = face.landmark_2d_106.astype(np.float32)
            self.landmarks = alpha * new_lm + (1.0 - alpha) * self.landmarks
        elif face.landmark_2d_106 is not None:
            self.landmarks = face.landmark_2d_106.copy().astype(np.float32)

    def smoothed_face(self):
        """Return a copy of the tracked face with smoothed bbox/landmarks."""
        f = copy.copy(self.face)
        f.bbox = self.bbox.copy()
        if self.landmarks is not None:
            f.landmark_2d_106 = self.landmarks.copy()
        return f


class FaceTracker:
    """Stateful tracker: maintains a list of active tracks across frames."""

    def __init__(self) -> None:
        self._tracks: List[_Track] = []

    def reset(self) -> None:
        """Clear all tracks (call when source face changes or preview stops)."""
        self._tracks = []

    def update(self, detected_faces: Optional[List]) -> Optional[List]:
        """
        Match detected faces to existing tracks, apply EMA, and return
        a list of smoothed face objects.

        Parameters
        ----------
        detected_faces : list of Face or None

        Returns
        -------
        list of smoothed Face objects, or None if no active tracks remain
        """
        alpha = float(getattr(modules.globals, 'smooth_alpha', 0.3))

        # ---- Matching ----
        matched_track_ids = set()
        matched_det_ids = set()

        if detected_faces:
            for det_idx, det_face in enumerate(detected_faces):
                if det_face is None or det_face.bbox is None:
                    continue
                best_iou = IOU_THRESHOLD
                best_track_idx = -1
                for trk_idx, track in enumerate(self._tracks):
                    if trk_idx in matched_track_ids:
                        continue
                    iou = _iou(det_face.bbox, track.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track_idx = trk_idx

                if best_track_idx >= 0:
                    self._tracks[best_track_idx].update(det_face, alpha)
                    matched_track_ids.add(best_track_idx)
                    matched_det_ids.add(det_idx)
                else:
                    # New face — start a fresh track
                    self._tracks.append(_Track(det_face))
                    matched_track_ids.add(len(self._tracks) - 1)
                    matched_det_ids.add(det_idx)

        # ---- Grace period for lost faces ----
        surviving = []
        for idx, track in enumerate(self._tracks):
            if idx not in matched_track_ids:
                track.lost_frames += 1
            if track.lost_frames <= LOST_GRACE_FRAMES:
                surviving.append(track)
            else:
                _log.debug('FaceTracker: dropping stale track (lost %d frames)', track.lost_frames)
        self._tracks = surviving

        if not self._tracks:
            return None

        return [t.smoothed_face() for t in self._tracks]

    def update_single(self, detected_face) -> Optional[object]:
        """Convenience wrapper for single-face mode."""
        if detected_face is None:
            result = self.update(None)
        else:
            result = self.update([detected_face])
        if result:
            return result[0]
        return None
