# graficos.py
"""
Utilidades de visualización para depuración (WSL/servidor sin GUI).

API:
- graficar_imagen(...)
- plt_plot(...)  -> alias compatible
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Optional, Iterable

import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Tipos comunes
try:
    from types import Frame, Point2D  # type: ignore
except Exception:
    Frame = np.ndarray  # type: ignore
    Point2D = tuple  # type: ignore


def graficar_imagen(
    img: Frame,
    title: Optional[str] = None,
    cmap: str = "viridis",
    additional_points: Optional[Iterable[Point2D]] = None,
    out_dir: str = "plots",
    save: bool = True,
    show: bool = False,
) -> Optional[str]:
    """
    Guarda (y opcionalmente muestra) una imagen de depuración con puntos superpuestos.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Preparar imagen para matplotlib
    if cmap == "gray":
        disp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        m_cmap = "gray"
    else:
        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        m_cmap = None

    # Título y ruta de salida
    if title is None:
        title = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fpath = os.path.join(out_dir, f"debug_plot_{title}.png")

    # Figura
    plt.figure(figsize=(16, 8))
    plt.title(f"{title} | shape={getattr(img, 'shape', None)}")
    plt.imshow(disp, cmap=m_cmap)

    # Puntos extra
    if additional_points is not None:
        for p in additional_points:
            if isinstance(p, (tuple, list)) and len(p) == 2:
                x, y = int(p[0]), int(p[1])
                plt.plot(x, y, "ro", markersize=3)

    plt.tight_layout()

    out_path: Optional[str] = None
    if save:
        plt.savefig(fpath, dpi=150)
        out_path = fpath

    if show:
        try:
            plt.show()
        except Exception:
            pass

    plt.close()
    return out_path


# Compatibilidad hacia atrás
plt_plot = graficar_imagen  # Alias en inglés mantenido
