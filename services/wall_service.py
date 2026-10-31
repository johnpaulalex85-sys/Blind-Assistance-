"""
Wall / blank-surface detector for walking assistance.

Why this exists
---------------
Object detectors (YOLOv8n/COCO) are trained on *objects* with edges and shape.
COCO has no "wall" class, and a plain wall has no object shape at all, so a
detector can never report it. A wall is instead recognised by what it lacks:
a large, smooth, low-texture region that fills the walking corridor.

The detector is pure OpenCV/NumPy (about 1-2 ms per frame), needs no model
download and runs on every frame. It returns a normal detection dict
({"box", "confidence", "class_name": "wall"}) so the hazard prioritizer,
tracker output, scene builder and web UI treat it like any other object.
"""
import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# Work on a tiny frame: a wall is a large-scale property, and downscaling also
# averages away fine plaster / paint grain and sensor noise.
_ANALYSIS_WIDTH = 160


class WallSegmenter:
    """
    Pretrained semantic-segmentation wall finder (SegFormer-B0 trained on ADE20K,
    which has a "wall" class). No dataset or training is needed from you: the model
    already knows what walls look like in any colour, pattern or lighting.

    Runs in a background thread on the newest frame only (never blocks the live
    loop). Results are a few hundred ms old, which is fine for a wall.
    """
    _IN_W, _IN_H = 384, 288  # multiples of 32, 4:3 like the camera

    def __init__(self, model=None, min_interval: Optional[float] = None):
        import threading
        self.min_interval = min_interval if min_interval is not None else settings.WALL_SEG_INTERVAL_SEC
        self.model = model
        self.wall_ids: List[int] = []
        self._frame = None
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._result = None          # (timestamp, bool mask)
        self._stop = False
        self.ready = False
        if self.model is None:
            self._load()
        else:
            self._find_wall_ids()
        if self.ready:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _load(self):
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation
            self.model = SegformerForSemanticSegmentation.from_pretrained(settings.WALL_SEG_MODEL)
            self.model.eval()
            if settings.DEVICE == "cuda":
                self.model.to("cuda")
            self._find_wall_ids()
            logger.info(f"Wall segmentation model loaded ({settings.WALL_SEG_MODEL}).")
        except Exception as e:
            logger.warning(f"Wall segmentation model unavailable ({e}); using texture-based wall detection only.")
            self.model = None
            self.ready = False

    def _find_wall_ids(self):
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.wall_ids = [int(i) for i, name in id2label.items() if str(name).strip().lower() == "wall"]
        if not self.wall_ids:
            logger.warning("Segmentation model has no 'wall' label; disabling wall segmentation.")
            self.ready = False
        else:
            self.ready = True

    def _label_map(self, small_bgr: np.ndarray) -> np.ndarray:
        """small_bgr (H,W,3) uint8 -> per-pixel class ids at 1/4 resolution."""
        import torch
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        dev = next(self.model.parameters()).device
        with torch.inference_mode():
            logits = self.model(pixel_values=x.to(dev)).logits  # (1, C, H/4, W/4)
        return logits.argmax(1)[0].cpu().numpy()

    def _worker(self):
        import time
        while not self._stop:
            self._event.wait()
            self._event.clear()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            try:
                labels = self._label_map(frame)
                mask = np.isin(labels, self.wall_ids)
                with self._lock:
                    self._result = (time.time(), mask)
            except Exception as e:
                logger.debug(f"Wall segmentation error: {e}")
            time.sleep(self.min_interval)

    def submit(self, frame: np.ndarray):
        """Non-blocking: hand over the newest frame."""
        if not self.ready:
            return
        small = cv2.resize(frame, (self._IN_W, self._IN_H), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._frame = small
        self._event.set()

    def latest_mask(self, max_age: float = 1.5) -> Optional[np.ndarray]:
        import time
        with self._lock:
            r = self._result
        if r is None or time.time() - r[0] > max_age:
            return None
        return r[1]

    def close(self):
        self._stop = True
        self._event.set()


class WallDetector:
    def __init__(
        self,
        min_coverage: Optional[float] = None,
        texture_threshold: Optional[float] = None,
        confirm_frames: Optional[int] = None,
        hold_frames: Optional[int] = None,
        corridor_left_ratio: Optional[float] = None,
        corridor_right_ratio: Optional[float] = None,
    ):
        self.min_coverage = min_coverage if min_coverage is not None else settings.WALL_MIN_COVERAGE
        self.texture_threshold = (
            texture_threshold if texture_threshold is not None else settings.WALL_TEXTURE_THRESHOLD
        )
        self.confirm_frames = confirm_frames if confirm_frames is not None else settings.WALL_CONFIRM_FRAMES
        self.hold_frames = hold_frames if hold_frames is not None else settings.WALL_HOLD_FRAMES
        self.corr_l = corridor_left_ratio if corridor_left_ratio is not None else settings.CORRIDOR_LEFT_RATIO
        self.corr_r = corridor_right_ratio if corridor_right_ratio is not None else settings.CORRIDOR_RIGHT_RATIO
        self.segmenter = WallSegmenter() if (settings.WALL_SEG_ENABLED) else None
        self.reset()

    def reset(self):
        self._hits = 0
        self._miss = 0
        self._confirmed = False
        self._last: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ core
    def _analyse(self, frame: np.ndarray, other_detections: Optional[List[Dict[str, Any]]]):
        """Returns a raw (un-debounced) wall detection dict, or None.

        Two cues, either one is enough:
          1. smooth surface   - plain painted / plastered walls (few edges)
          2. uniform pattern  - brick, tiles, stripes, wallpaper, stone, textured
                                plaster: lots of edges, but the *same* kind of edges
                                everywhere. A real room (furniture, doors, floor,
                                perspective lines) is never statistically uniform.
        """
        found = None
        if self.segmenter is not None and self.segmenter.ready:
            self.segmenter.submit(frame)
            found = self._from_mask(self.segmenter.latest_mask(), frame.shape[1], frame.shape[0])
        if found is None:
            found = self._analyse_smooth(frame, other_detections)
        if found is None and settings.WALL_DETECT_PATTERNED:
            found = self._analyse_patterned(frame, other_detections)
        return found

    def _from_mask(self, mask: Optional[np.ndarray], w0: int, h0: int):
        """Turn the segmenter's wall mask into a detection if it fills the corridor."""
        if mask is None or mask.size == 0:
            return None
        h, w = mask.shape
        coverage = float(mask.mean())
        c1, c2 = int(w * self.corr_l), int(w * self.corr_r)
        corridor = mask[int(h * 0.10):, c1:c2]
        corridor_cov = float(corridor.mean()) if corridor.size else 0.0
        if coverage < settings.WALL_SEG_MIN_COVERAGE or corridor_cov < 0.5:
            return None
        ys, xs = np.where(mask)
        bx1, bx2 = np.percentile(xs, [1, 99])
        by1, by2 = np.percentile(ys, [1, 99])
        box = [round(float(bx1) / w * w0, 1), round(float(by1) / h * h0, 1),
               round(float(bx2 + 1) / w * w0, 1), round(float(by2 + 1) / h * h0, 1)]
        conf = 0.50 + 0.30 * min(1.0, coverage / 0.8) + 0.19 * min(1.0, corridor_cov)
        return {"box": box, "confidence": round(min(conf, 0.99), 3), "class_name": "wall",
                "coverage": round(coverage, 3)}

    def _analyse_patterned(self, frame: np.ndarray, other_detections: Optional[List[Dict[str, Any]]]):
        h0, w0 = frame.shape[:2]
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
        if float(gray.mean()) < 20:
            return None

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # undirected edge orientation, 8 bins
        bins = ((np.arctan2(gy, gx) % np.pi) / np.pi * 8).astype(int) % 8

        rows, cols = 6, 8
        ch, cw = 120 // rows, 160 // cols
        G = np.zeros((rows, cols), np.float32)
        O = np.zeros((rows, cols, 8), np.float32)
        masked = np.zeros((rows, cols), bool)
        for r in range(rows):
            for c in range(cols):
                m = mag[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                G[r, c] = m.mean()
                hist = np.bincount(bins[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].ravel(),
                                   weights=m.ravel(), minlength=8)
                O[r, c] = hist / (hist.sum() + 1e-6)

        # Cells mostly covered by another detected object don't vote.
        if other_detections:
            for d in other_detections:
                b = d.get("box")
                if not b or len(b) != 4 or d.get("class_name") == "wall":
                    continue
                for r in range(rows):
                    for c in range(cols):
                        cx1, cy1 = c * cw * w0 / 160.0, r * ch * h0 / 120.0
                        cx2, cy2 = (c + 1) * cw * w0 / 160.0, (r + 1) * ch * h0 / 120.0
                        iw = max(0.0, min(cx2, b[2]) - max(cx1, b[0]))
                        ih = max(0.0, min(cy2, b[3]) - max(cy1, b[1]))
                        if iw * ih > 0.3 * (cx2 - cx1) * (cy2 - cy1):
                            masked[r, c] = True

        valid = ~masked
        c_lo, c_hi = int(cols * 0.25), int(np.ceil(cols * 0.75))
        if valid.mean() < 0.5 or valid[:, c_lo:c_hi].mean() < 0.6:
            return None

        med_g = float(np.median(G[valid]))
        if med_g < 3.0:          # essentially smooth: handled by the smooth cue
            return None
        med_o = np.median(O[valid], axis=0)
        similarity = np.sum(np.sqrt(O * med_o), axis=2)            # Bhattacharyya, 1.0 = identical
        uniform = (np.abs(G / (med_g + 1e-6) - 1.0) < settings.WALL_PATTERN_STRENGTH_TOL) & \
                  (similarity > settings.WALL_PATTERN_SIMILARITY)
        ok_all = float(uniform[valid].mean())
        ok_corr = float(uniform[:, c_lo:c_hi][valid[:, c_lo:c_hi]].mean())
        if ok_all < 0.55 or ok_corr < 0.75:
            return None

        # Box: the voting cells that agree with the pattern.
        good = uniform & valid
        ys, xs = np.where(good)
        box = [
            round(float(xs.min()) * cw / 160.0 * w0, 1),
            round(float(ys.min()) * ch / 120.0 * h0, 1),
            round(float(xs.max() + 1) * cw / 160.0 * w0, 1),
            round(float(ys.max() + 1) * ch / 120.0 * h0, 1),
        ]
        conf = 0.50 + 0.30 * ok_all + 0.19 * ok_corr
        return {
            "box": box,
            "confidence": round(min(conf, 0.99), 3),
            "class_name": "wall",
            "coverage": round(float(good.mean()), 3),
        }

    def _analyse_smooth(self, frame: np.ndarray, other_detections: Optional[List[Dict[str, Any]]]):
        h0, w0 = frame.shape[:2]
        scale = _ANALYSIS_WIDTH / float(w0)
        small = cv2.resize(frame, (_ANALYSIS_WIDTH, max(8, int(h0 * scale))), interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        # Lens covered / pitch dark: not a wall, don't claim one.
        if mean_brightness < 20:
            return None

        gray_f = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)

        # Texture measure 1: gradient magnitude (edges).
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)

        # Texture measure 2: local standard deviation (blotchy / patterned areas).
        k = (7, 7)
        mu = cv2.blur(gray_f, k)
        var = np.maximum(cv2.blur(gray_f * gray_f, k) - mu * mu, 0.0)
        local_std = np.sqrt(var)

        t = self.texture_threshold
        smooth = ((grad < 4.0 * t) & (local_std < 1.6 * t)).astype(np.uint8)

        # Do not count pixels that belong to another detected object.
        if other_detections:
            sx, sy = w / float(w0), h / float(h0)
            for d in other_detections:
                b = d.get("box")
                if not b or len(b) != 4 or d.get("class_name") == "wall":
                    continue
                x1, y1, x2, y2 = int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)
                smooth[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0

        # Remove speckle, then close small gaps (stains, marks, switches).
        smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        # Largest connected smooth region.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(smooth, connectivity=4)
        if n <= 1:
            return None
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = labels == idx
        area = float(stats[idx, cv2.CC_STAT_AREA])
        coverage = area / float(w * h)

        # How much of the walking corridor (ahead of the user) is that region?
        cx1, cx2 = int(w * self.corr_l), int(w * self.corr_r)
        corridor = region[int(h * 0.10):, cx1:cx2]
        corridor_cov = float(corridor.mean()) if corridor.size else 0.0

        if coverage < self.min_coverage or corridor_cov < 0.55:
            return None

        # Box of the wall region (use percentile extents so a thin leak doesn't stretch it).
        ys, xs = np.where(region)
        bx1, bx2 = np.percentile(xs, [1, 99])
        by1, by2 = np.percentile(ys, [1, 99])
        box = [
            round(float(bx1) / w * w0, 1),
            round(float(by1) / h * h0, 1),
            round(float(bx2 + 1) / w * w0, 1),
            round(float(by2 + 1) / h * h0, 1),
        ]

        # Confidence: grows with coverage and with how cleanly the corridor is filled.
        conf = 0.45 + 0.35 * min(1.0, coverage / 0.8) + 0.2 * min(1.0, corridor_cov)
        return {
            "box": box,
            "confidence": round(min(conf, 0.99), 3),
            "class_name": "wall",
            "coverage": round(coverage, 3),
        }

    # ---------------------------------------------------------------- public
    def detect(
        self,
        frame: np.ndarray,
        other_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Returns [] or [wall_detection]. Debounced to avoid flicker."""
        if frame is None or frame.size == 0:
            return []
        try:
            raw = self._analyse(frame, other_detections)
        except Exception as e:  # never let wall logic break the live loop
            logger.debug(f"Wall detection error: {e}")
            return []

        if raw is not None:
            self._hits += 1
            self._miss = 0
            if self._last is not None:  # smooth the box between frames
                a = 0.6
                raw["box"] = [round(a * n + (1 - a) * o, 1) for n, o in zip(raw["box"], self._last["box"])]
            self._last = raw
            if self._hits >= self.confirm_frames:
                self._confirmed = True
                return [dict(raw)]
            return []

        # No wall this frame.
        self._hits = 0
        self._miss += 1
        if self._confirmed and self._last is not None and self._miss <= self.hold_frames:
            return [dict(self._last)]  # brief dropout (glare, a mark): keep reporting
        self._last = None
        self._confirmed = False
        return []
