# main.py
"""
Pipeline principal del proyecto (nombres en español).
"""

import os
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt

from deteccion_seguimiento_balon import BallDetectTrack
from jugador import Player
from deteccion_jugadores import FeetDetector as DetectorPies, COLORS, hsv_a_bgr as hsv2bgr
from geometria_cancha import (
    mosaico,
    anadir_frame,
    binarizar_morfologia,
    rectangularizar_cancha,
    rectificar,
)
from procesador_video import TOPCUT, VideoHandler
from graficos import graficar_imagen

# === NUEVO (2.1): utilidades de la carpeta de la ejecución ===
from run_manager import init_run, path


def obtener_frames(video_path: str, central_frame: int, mod: int) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    index = 0

    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] No se pudo abrir el vídeo: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            cap.release()
            print("[INFO] Released Video Resource")
            break

        if (index % mod) == 0:
            frames.append(frame[TOPCUT:, :])

        if cv2.waitKey(20) == ord('q'):
            break

        index += 1

    cap.release()
    cv2.destroyAllWindows()

    if not frames:
        raise RuntimeError("[ERROR] No se han obtenido frames muestreados del vídeo.")

    print(f"[INFO] Number of frames : {len(frames)}")
    if 0 <= central_frame < len(frames):
        plt.title(f"Central {frames[central_frame].shape}")
        plt.imshow(cv2.cvtColor(frames[central_frame], cv2.COLOR_BGR2RGB))
        plt.show()

    return frames


#####################################################################
if __name__ == '__main__':
    # === NUEVO (2.2): inicializar carpeta de ejecución reproducible ===
    _run_dir = init_run(seed=42)
    print(f"[RUN] outputs -> {_run_dir}")

    video_path = "resources/Short4Mosaicing.mp4"
    pano_path = "resources/pano.png"
    pano_enhanced_path = "resources/pano_enhanced.png"
    map2d_path = "resources/2d_map.png"
    snapshots_dir = "resources/snapshots"

    # ---- 1) Cargar o generar la panorámica base (pano.png) ----
    if os.path.exists(pano_path):
        pano = cv2.imread(pano_path)
        if pano is None:
            raise RuntimeError(f"[ERROR] No se pudo leer la panorámica: {pano_path}")
        print(f"[INFO] Panorámica cargada: {pano_path}")
        # === NUEVO (2.3): copia versionada en runs/<id>/plots ===
        cv2.imwrite(path("plots", "pano.png"), pano)
    else:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"[ERROR] Falta el vídeo: {video_path}")
        central_frame = 36
        frames = obtener_frames(video_path, central_frame, mod=3)

        frames_flipped = [cv2.flip(frames[i], 1) for i in range(central_frame)]
        current_mosaic1 = mosaico(frames[central_frame:], direction=1)
        current_mosaic2 = mosaico(frames_flipped, direction=-1)
        pano = mosaico([cv2.flip(current_mosaic2, 1)[:, :-10], current_mosaic1])

        os.makedirs(os.path.dirname(pano_path), exist_ok=True)
        cv2.imwrite(pano_path, pano)
        print(f"[OK] Panorámica guardada en: {pano_path}")
        # === NUEVO (2.3): copia versionada en runs/<id>/plots ===
        cv2.imwrite(path("plots", "pano.png"), pano)

    # ---- 2) Panorámica mejorada ----
    if os.path.exists(pano_enhanced_path):
        pano_enhanced = cv2.imread(pano_enhanced_path)
        if pano_enhanced is None:
            raise RuntimeError(f("[ERROR] No se pudo leer: {pano_enhanced_path}"))
        graficar_imagen(pano, "Panorama")
        print(f"[INFO] Panorámica mejorada cargada: {pano_enhanced_path}")
        # === NUEVO (2.4): copia versionada en runs/<id>/plots ===
        cv2.imwrite(path("plots", "pano_enhanced.png"), pano_enhanced)
    else:
        pano_enhanced = pano.copy()
        if os.path.isdir(snapshots_dir):
            print(f"[INFO] Mejorando panorámica con snapshots en: {snapshots_dir}")
            for file in sorted(os.listdir(snapshots_dir)):
                fpath = os.path.join(snapshots_dir, file)
                if not os.path.isfile(fpath):
                    continue
                frame = cv2.imread(fpath)
                if frame is None:
                    continue
                frame = frame[TOPCUT:]
                pano_enhanced = anadir_frame(frame, pano, pano_enhanced, plot=False)
        else:
            print(f"[WARN] No existe carpeta de snapshots: {snapshots_dir} (se usará pano base)")

        cv2.imwrite(pano_enhanced_path, pano_enhanced)
        print(f"[OK] Panorámica mejorada guardada en: {pano_enhanced_path}")
        # === NUEVO (2.4): copia versionada en runs/<id>/plots ===
        cv2.imwrite(path("plots", "pano_enhanced.png"), pano_enhanced)

    # ---- 3) Simplificación y esquinas ----
    pano_enhanced = np.vstack((
        pano_enhanced,
        np.zeros((100, pano_enhanced.shape[1], pano_enhanced.shape[2]), dtype=pano.dtype)
    ))

    img = binarizar_morfologia(pano_enhanced, plot=False)

    simplified_court, corners = rectangularizar_cancha(img, plot=False)
    simplified_court = 255 - np.uint8(simplified_court)
    graficar_imagen(simplified_court, "Corner Detection", cmap="gray", additional_points=corners)

    rectified = rectificar(pano_enhanced, corners, plot=False)

    # ---- 4) Correspondencias mapa 2D <-> rectificada ----
    if not os.path.exists(map2d_path):
        raise FileNotFoundError(f"[ERROR] Falta el mapa 2D: {map2d_path}")
    map_2d = cv2.imread(map2d_path)
    if map_2d is None:
        raise RuntimeError(f"[ERROR] No se pudo leer el mapa 2D: {map2d_path}")

    scale = rectified.shape[0] / map_2d.shape[0]
    map_2d = cv2.resize(map_2d, (int(scale * map_2d.shape[1]), int(scale * map_2d.shape[0])))
    _resized = cv2.resize(rectified, (map_2d.shape[1], map_2d.shape[0]))
    map_2d = cv2.resize(map_2d, (rectified.shape[1], rectified.shape[0]))

    # ---- 5) Vídeo y objetos del pipeline ----
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"[ERROR] No se pudo abrir el vídeo para el tracking: {video_path}")

    players: List[Player] = []
    for i in range(1, 6):
        players.append(Player(i, 'green', hsv2bgr(COLORS['green'][2])))
        players.append(Player(i, 'white', hsv2bgr(COLORS['white'][2])))
    players.append(Player(0, 'referee', hsv2bgr(COLORS['referee'][2])))

    feet_detector = DetectorPies(players)
    ball_detect_track = BallDetectTrack(players)
    video_handler = VideoHandler(pano_enhanced, video, ball_detect_track, feet_detector, map_2d)

    # Ejecutar
    video_handler.run_detectors()
