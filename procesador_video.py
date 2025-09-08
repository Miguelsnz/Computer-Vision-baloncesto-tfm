# procesador_video.py
from deteccion_jugadores import *
import cv2
import numpy as np
from graficos import plt_plot
from run_manager import video_path, plots_dir

TOPCUT = 320

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


class VideoHandler:
    def __init__(self, pano, video, ball_detector, feet_detector, map_2d):
        self.M1 = np.load("Rectify1.npy")
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.pano = pano
        self.video = video
        self.kp1, self.des1 = self.sift.compute(pano, self.sift.detect(pano))
        self.feet_detector = feet_detector
        self.ball_detector = ball_detector
        self.map_2d = map_2d

    def run_detectors(self):
        # --- Config de salida de vídeo (OpenCV) ---
        writer = None
        fps = self.video.get(cv2.CAP_PROP_FPS)
        if fps is None or fps == 0:
            fps = 25.0  # fallback si el vídeo no reporta FPS

        time_index = 0
        while self.video.isOpened():
            ok, frame = self.video.read()
            if not ok:
                break

            if 0 <= time_index <= 230:
                print("\r Computing DEMO: " + str(int(100 * time_index / 200)) + "%", flush=True, end="")

                frame = frame[TOPCUT:, :]

                # Homografía frame->pano
                M = self.get_homography(frame, self.des1, self.kp1)

                # Detección de jugadores y proyección a mapa 2D
                frame, self.map_2d, map_2d_text = self.feet_detector.get_players_pos(
                    M, self.M1, frame, time_index, self.map_2d
                )

                # Detección/seguimiento del balón
                frame, ball_map_2d = self.ball_detector.ball_tracker(
                    M, self.M1, frame, self.map_2d.copy(), map_2d_text, time_index
                )

                # Visual combinado: frame + mapa 2D
                vis = np.vstack(
                    (frame, cv2.resize(map_2d_text, (frame.shape[1], frame.shape[1] // 2)))
                )

                # Guardar PNG de depuración
                plt_plot(vis, title=f"vis_{time_index}", out_dir=plots_dir())

                # Inicializar writer cuando tengamos el primer 'vis'
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_path("demo.mp4"), fourcc, fps, (w, h))
                # Escribir frame en el vídeo
                if writer is not None:
                    writer.write(vis)

            time_index += 1

        self.video.release()

        # Cierre seguro del writer (OpenCV)
        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    def get_homography(self, frame, des1, kp1):
        kp2 = self.sift.detect(frame)
        kp2, des2 = self.sift.compute(frame, kp2)

        # Coincidencias FLANN + ratio test
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) < 4:
            # Homografía no fiable si hay muy pocos matches
            return np.eye(3, dtype=np.float32)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is None:
            M = np.eye(3, dtype=np.float32)
        return M

    # Alias en español
    def ejecutar_detectores(self):
        return self.run_detectors()

    def obtener_homografia(self, frame, des1, kp1):
        return self.get_homography(frame, des1, kp1)
