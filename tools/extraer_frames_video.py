# extraer_frames_video.py
# -*- coding: utf-8 -*-
"""
Herramienta sencilla para extraer todos los frames de un vídeo y guardarlos
como imágenes JPEG en la carpeta `data/`.
"""

from __future__ import annotations

import os
import cv2


def extract_videoframe() -> None:
    cam = cv2.VideoCapture("resources/Short4Mosaicing.mp4")

    try:
        if not os.path.exists("data"):
            os.makedirs("data")
    except OSError:
        print("Error: Creating directory of data")

    currentframe = 0

    while True:
        ret, frame = cam.read()

        if ret:
            name = "./data/frame" + str(currentframe) + ".jpg"
            print("Creating..." + name)
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    extract_videoframe()
