# deteccion_seguimiento_balon.py
"""
Detección y seguimiento del balón.
"""

from __future__ import annotations
import os
from operator import itemgetter
from typing import Optional, List, Tuple

import cv2
import numpy as np

from deteccion_jugadores import FeetDetector
from graficos import plt_plot

# Tipos comunes (para anotaciones). Intentamos importar de types.py local.
try:
    from types import Frame, BBox, Point2D  # type: ignore
except Exception:
    Frame = np.ndarray  # type: ignore
    BBox = tuple        # type: ignore
    Point2D = tuple     # type: ignore


# Hiperparámetros globales
MAX_TRACK: int = 5            # número de frames que mantenemos el tracking sin redetección
IOU_BALL_PADDING: int = 30    # margen para calcular intersección balón-jugador


class BallDetectTrack:
    def __init__(self, players: List):
        self.ball_padding: int = 30
        self.check_track: int = MAX_TRACK
        self.do_detection: bool = True
        self.tracker_type: str = "CSRT"
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.legacy.TrackerCSRT_create()
        self.players = players

    @staticmethod
    def circle_detect(img: Frame, plot: bool = False) -> Optional[np.ndarray]:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_blur = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(
            img_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=25, minRadius=5, maxRadius=15
        )

        if circles is not None:
            circles = np.uint16(np.around(circles)).reshape(-1, 3)
            if plot:
                vis = cimg.copy()
                for (cx, cy, r) in circles:
                    cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                    cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255), 3)
                plt_plot(vis, "Detected Circles")
            return circles
        return None

    def ball_detection(self, img_train_dir: str, query_frame: Frame, th: float = 0.98) -> Optional[BBox]:
        templates: List[Frame] = []
        if os.path.isdir(img_train_dir):
            for fname in os.listdir(img_train_dir):
                fpath = os.path.join(img_train_dir, fname)
                if os.path.isfile(fpath):
                    im = cv2.imread(fpath, 0)
                    if im is not None:
                        templates.append(im)
        if not templates:
            return None

        img_gray = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)
        centers = self.circle_detect(img_gray)
        if centers is None:
            return None

        H, W = img_gray.shape[:2]
        margin = 7

        for (cx, cy, r) in centers:
            cx, cy, r = int(cx), int(cy), int(r)
            x1, y1 = cx - r - margin, cy - r - margin
            x2, y2 = cx + r + margin, cy + r + margin

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W - 1, x2); y2 = min(H - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            patch = img_gray[y1:y2, x1:x2]
            for tmpl in templates:
                if patch.shape[0] > tmpl.shape[0] and patch.shape[1] > tmpl.shape[1]:
                    res = cv2.matchTemplate(patch, tmpl, cv2.TM_CCORR_NORMED)
                    if res.size > 0 and float(np.max(res)) >= th:
                        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

        return None

    def ball_tracker(
        self,
        M: np.ndarray,
        M1: np.ndarray,
        frame: Frame,
        map_2d: Frame,
        map_2d_text: Frame,
        timestamp: int
    ) -> Tuple[Frame, Optional[Frame]]:
        bbox: Optional[BBox] = None

        if self.do_detection:
            detected = self.ball_detection("resources/ball/", frame)
            if detected is not None:
                x, y, w, h = (int(detected[0]), int(detected[1]),
                              int(detected[2]), int(detected[3]))
                H, W = frame.shape[:2]
                if w > 0 and h > 0 and 0 <= x <= W - w and 0 <= y <= H - h:
                    self.tracker.init(frame, (x, y, w, h))
                    self.do_detection = False
                    bbox = (x, y, w, h)
        else:
            ok, tbbox = self.tracker.update(frame)
            if ok and tbbox is not None:
                x, y, w, h = (int(round(tbbox[0])), int(round(tbbox[1])),
                              int(round(tbbox[2])), int(round(tbbox[3])))
                H, W = frame.shape[:2]
                if w > 0 and h > 0 and 0 <= x <= W - w and 0 <= y <= H - h:
                    bbox = (x, y, w, h)
                else:
                    self.do_detection = True
                    self.check_track = MAX_TRACK
            else:
                self.do_detection = True
                self.check_track = MAX_TRACK

        if bbox is None:
            return frame, None

        x, y, w, h = bbox
        p1 = (x, y)
        p2 = (x + w, y + h)
        ball_center = np.array([x + w // 2, y + h // 2, 1], dtype=int)

        bbox_iou = (
            int(ball_center[1] - IOU_BALL_PADDING),
            int(ball_center[0] - IOU_BALL_PADDING),
            int(ball_center[1] + IOU_BALL_PADDING),
            int(ball_center[0] + IOU_BALL_PADDING),
        )

        scores = []
        for p in self.players:
            try:
                _ = p.positions[timestamp]
                if p.team != "referee" and p.previous_bb is not None:
                    scores.append((p, FeetDetector.bb_intersection_over_union(bbox_iou, p.previous_bb)))
            except KeyError:
                pass

        if scores:
            for p in self.players:
                p.has_ball = False
            max_score = max(scores, key=itemgetter(1))
            max_score[0].has_ball = True
            cv2.circle(map_2d_text, (max_score[0].positions[timestamp]), 27, (0, 0, 255), 10)

        if self.check_track > 0:
            homo = M1 @ (M @ ball_center.reshape((3, -1)))
            homo = np.int32(homo / homo[-1]).ravel()
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.circle(map_2d, (int(homo[0]), int(homo[1])), 10, (0, 0, 255), 5)
            self.check_track -= 1
        else:
            x1 = max(0, x - self.ball_padding)
            y1 = max(0, y - self.ball_padding)
            x2 = min(frame.shape[1], x + w + self.ball_padding)
            y2 = min(frame.shape[0], y + h + self.ball_padding)
            local = frame[y1:y2, x1:x2]

            if self.ball_detection('resources/ball/', local, 0.5) is not None:
                self.check_track = MAX_TRACK
                self.do_detection = False
            else:
                self.check_track = MAX_TRACK
                self.do_detection = True

        return frame, map_2d


# ==========================
# Alias en español (métodos)
# ==========================
def detectar_circulos(img: Frame, plot: bool = False) -> Optional[np.ndarray]:
    """Función de módulo que llama al método estático para comodidad."""
    return BallDetectTrack.circle_detect(img, plot=plot)

# Alias de instancia (puedes usarlos desde un objeto BallDetectTrack)
BallDetectTrack.detectar_balon = BallDetectTrack.ball_detection
BallDetectTrack.rastreador_balon = BallDetectTrack.ball_tracker
