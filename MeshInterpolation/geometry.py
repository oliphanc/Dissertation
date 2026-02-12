from abc import ABC, abstractmethod
import numpy as np


class GeometryInterface(ABC):
    """
    Minimal interface required by MeshInterpolator.
    """

    period: int
    n_blades: int
    tip_clearance: float
    rotation_direction: int

    @abstractmethod
    def pressure_side_cylindrical(self) -> np.ndarray:
        ...

    @abstractmethod
    def suction_side_cylindrical(self) -> np.ndarray:
        ...
