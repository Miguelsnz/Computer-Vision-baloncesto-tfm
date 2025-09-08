# types.py
"""
Definiciones de tipos comunes usados en todo el proyecto
para mayor legibilidad y consistencia.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional

# === Tipos básicos ===

# Imagen o frame de vídeo (alto x ancho x canales)
Frame = np.ndarray

# Punto 2D (x, y)
Point2D = Tuple[int, int]

# Bounding Box en formato (x, y, w, h)
BBox = Tuple[int, int, int, int]

# Lista de puntos (trayectorias, contornos, etc.)
PointList = List[Point2D]

# Mapa 2D (por ejemplo, la pista proyectada)
CourtMap = np.ndarray

# === Tipos más específicos ===

# Representa la posición del balón en un frame
BallPosition = Optional[Point2D]

# Trayectoria del balón o jugador (lista de puntos en el tiempo)
Trajectory = List[Point2D]

# Diccionario de posiciones de jugadores por frame
#   clave = timestamp (int)
#   valor = posición (x, y)
PlayerPositions = Dict[int, Point2D]
