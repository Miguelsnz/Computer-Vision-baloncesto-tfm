# jugador.py
"""
Definición de la entidad Player (jugador) usada en todo el pipeline.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple


class Player:
    """
    Representa a un jugador (o árbitro) con su identidad, equipo/color
    y el historial de posiciones estimadas en el mapa 2D.
    """

    def __init__(self, ID: int, team: str, color: Tuple[int, int, int]) -> None:
        self.ID: int = ID
        self.team: str = team
        self.color: Tuple[int, int, int] = color

        # timestamp -> (x, y) en el mapa 2D (coordenadas int)
        self.positions: Dict[int, Tuple[int, int]] = {}

        # Última bounding box asociada al jugador, en formato (top, left, bottom, right)
        self.previous_bb: Optional[Tuple[int, int, int, int]] = None

        # Flag de posesión del balón
        self.has_ball: bool = False

    def reset_tracking(self) -> None:
        """Vacía el historial de posiciones y la bounding box; limpia la posesión."""
        self.positions.clear()
        self.previous_bb = None
        self.has_ball = False

    def __repr__(self) -> str:
        return (
            f"Player(ID={self.ID}, team='{self.team}', color={self.color}, "
            f"positions={len(self.positions)} pts, has_ball={self.has_ball})"
        )
