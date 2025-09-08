# geometria_cancha.py
# -*- coding: utf-8 -*-
"""
Utilidades para mosaicar, binarizar y rectificar la pista de baloncesto.
"""

from __future__ import annotations
from run_manager import path
from typing import List, Tuple
import cv2
import numpy as np

from graficos import plt_plot  # alias mantiene compat

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def collage(frames: List[np.ndarray], direction: int = 1, plot: bool = False) -> np.ndarray:
    sift = cv2.xfeatures2d.SIFT_create()

    if direction == 1:
        current_mosaic = frames[0]
    else:
        current_mosaic = frames[-1]

    for i in range(len(frames) - 1):
        # FINDING FEATURES
        kp1 = sift.detect(current_mosaic)
        kp1, des1 = sift.compute(current_mosaic, kp1)
        kp2 = sift.detect(frames[i * direction + direction])
        kp2, des2 = sift.compute(frames[i * direction + direction], kp2)

        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # Finding an homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        result = cv2.warpPerspective(
            frames[i * direction + direction],
            M,
            (
                current_mosaic.shape[1] + frames[i * direction + direction].shape[1],
                frames[i * direction + direction].shape[0] + 50,
            ),
        )

        result[:current_mosaic.shape[0], :current_mosaic.shape[1]] = current_mosaic
        current_mosaic = result

        # removing black part of the collage
        for j in range(len(current_mosaic[0])):
            if np.sum(current_mosaic[:, j]) == 0:
                current_mosaic = current_mosaic[:, : j - 50]
                break

        if plot:
            plt_plot(current_mosaic)

    return current_mosaic


def add_frame(frame: np.ndarray, pano: np.ndarray, pano_enhanced: np.ndarray, plot: bool = False) -> np.ndarray:
    sift = cv2.xfeatures2d.SIFT_create()  # sift instance

    # FINDING FEATURES
    kp1 = sift.detect(pano)
    kp1, des1 = sift.compute(pano, kp1)
    kp2 = sift.detect(frame)
    kp2, des2 = sift.compute(frame, kp2)

    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print(f"Number of good correspondences: {len(good)}")
    if len(good) < 70:
        return pano

    # Finding an homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(frame, M, (pano.shape[1], pano.shape[0]))

    if plot:
        plt_plot(result, "Warped new image")

    avg_pano = np.where(
        result < 100,
        pano_enhanced,
        np.uint8(np.average(np.array([pano_enhanced, result]), axis=0, weights=[1, 0.7])),
    )

    if plot:
        plt_plot(avg_pano, "AVG new image")

    return avg_pano


def binarize_erode_dilate(img: np.ndarray, plot: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, img_otsu = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_OTSU)

    if plot:
        plt_plot(img_otsu, "Panorama after Otsu", cmap="gray")

    kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)
    img_otsu = cv2.erode(img_otsu, kernel, iterations=20)
    img_otsu = cv2.dilate(img_otsu, kernel, iterations=20)

    if plot:
        plt_plot(img_otsu, "After Erosion-Dilation", cmap="gray")
    return img_otsu


def rectangularize_court(pano: np.ndarray, plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # BLOB FILTERING & BLOB DETECTION

    # adding a little frame to enable detection of blobs that touch the borders
    pano[-4: -1] = pano[0:3] = 0
    pano[:, 0:3] = pano[:, -4:-1] = 0

    mask = np.zeros(pano.shape, dtype=np.uint8)
    cnts = cv2.findContours(pano, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_court = []

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    threshold_area = 100000
    for c in cnts:
        area = cv2.contourArea(c)
        if area > threshold_area:
            cv2.drawContours(mask, [c], -1, (36, 255, 12), -1)
            contours_court.append(c)

    pano = mask
    if plot:
        plt_plot(pano, "After Blob Detection", cmap="gray")

    # pano = 255 - pano
    contours_court = contours_court[0]
    simple_court = np.zeros(pano.shape)

    # convex hull
    hull = cv2.convexHull(contours_court)
    cv2.drawContours(pano, [hull], 0, 100, 2)
    if plot:
        plt_plot(pano, "After ConvexHull", cmap="gray", additional_points=hull.reshape((-1, 2)))

    # fitting a poly to the hull
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    corners = approx.reshape(-1, 2)
    cv2.drawContours(pano, [approx], 0, 100, 5)
    cv2.drawContours(simple_court, [approx], 0, 255, 3)

    if plot:
        plt_plot(pano, "After Rectangular Fitting", cmap="gray")
        plt_plot(simple_court, "Rectangularized Court", cmap="gray")
        print("simplified contour has", len(approx), "points")

    return simple_court, corners


def homography(rect: np.ndarray, image: np.ndarray, plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    bl, tl, tr, br = rect
    rect = np.array([tl, tr, br, bl], dtype="float32")

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB)) + 700

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    if plot:
        plt_plot(warped)
    return warped, M

def rectify(pano_enhanced: np.ndarray, corners: np.ndarray, plot: bool = False) -> np.ndarray:
    """
    Rectifica la pista a partir de la panorámica mejorada y las esquinas detectadas.
    Guarda:
      - Matrices de homografía en runs/.../metrics/{Rectify1.npy, RectifyL.npy, RectifyR.npy}
      - Paquete conjunto en runs/.../metrics/homographies.npz
      - Imagen rectificada en runs/.../plots/rectified.png
    Si no existe run_manager.path, hace fallback a carpetas locales 'metrics/' y 'plots/'.
    """
    import os

    # --- Resolver función path(subdir, filename) ---
    PATH = None
    try:
        # Preferente: usa run_manager.path (crea subcarpetas dentro del RUN actual)
        from run_manager import path as _rm_path  # type: ignore
        PATH = _rm_path
    except Exception:
        # Fallback: crea subcarpetas locales si no hay run_manager
        def _fallback_path(subdir: str, filename: str) -> str:
            os.makedirs(subdir, exist_ok=True)
            return os.path.join(subdir, filename)
        PATH = _fallback_path  # type: ignore

    # --- Lado izquierdo / derecho y esquinas auxiliares (como en tu versión) ---
    panoL = pano_enhanced[:, :1870]
    panoR = pano_enhanced[:, 1870:]
    cornersL = np.array([corners[0], corners[1], [1865, 55], [1869, 389]])
    cornersR = np.array(
        [
            [0, 389],
            [0, 55],
            [corners[2][0] - 1870, corners[2][1]],
            [corners[3][0] - 1870, corners[3][1]],
        ]
    )

    # --- Homografías ---
    _, M = homography(corners, pano_enhanced)   # frame -> pano_enhanced
    np.save(PATH("metrics", "Rectify1.npy"), M)

    h1, ML = homography(cornersL, panoL)
    np.save(PATH("metrics", "RectifyL.npy"), ML)

    h2, MR = homography(cornersR, panoR)
    np.save(PATH("metrics", "RectifyR.npy"), MR)

    # (Opcional recomendado) paquete conjunto
    np.savez(PATH("metrics", "homographies.npz"),
             Rectify1=M, RectifyL=ML, RectifyR=MR)

    # --- Imagen rectificada (unimos L y R) ---
    rectified = np.hstack((h1, cv2.resize(h2, (h1.shape[1], h1.shape[0]))))
    cv2.imwrite(PATH("plots", "rectified.png"), rectified)

    if plot:
        plt_plot(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB), title="rectified")

    return rectified


# Alias en español (wrappers)
def mosaico(frames: List[np.ndarray], direction: int = 1, plot: bool = False) -> np.ndarray:
    return collage(frames, direction=direction, plot=plot)

def anadir_frame(frame: np.ndarray, pano: np.ndarray, pano_enhanced: np.ndarray, plot: bool = False) -> np.ndarray:
    return add_frame(frame, pano, pano_enhanced, plot=plot)

def binarizar_morfologia(img: np.ndarray, plot: bool = False) -> np.ndarray:
    return binarize_erode_dilate(img, plot=plot)

def rectangularizar_cancha(pano: np.ndarray, plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    return rectangularize_court(pano, plot=plot)

def rectificar(pano_enhanced: np.ndarray, corners: np.ndarray, plot: bool = False) -> np.ndarray:
    return rectify(pano_enhanced, corners, plot=plot)
