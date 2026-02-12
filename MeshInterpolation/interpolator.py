from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, UnivariateSpline, RBFInterpolator
import matplotlib.pyplot as plt

from .smoothing import elliptical_smoothing
from .geometry import GeometryInterface


class MeshInterpolator:
    """
    Structured mesh generator and flow-field interpolator for turbomachinery
    blade passages with periodic extension.

    This class:
      - builds a body-fitted (θ, z) mesh
      - duplicates the mesh periodically
      - interpolates scattered CFD data onto the structured mesh
    """

    def __init__(
        self,
        n_axial: int = 38,
        n_pitchwise: int = 64,
        n_tip: int = 10,
        cosine_spacing: bool = True,
        smooth_mesh: bool = False,
        interpolation_method: str = "rbf",
    ):
        self.n_axial = n_axial
        self.n_pitchwise = n_pitchwise
        self.n_tip = n_tip
        self.cosine_spacing = cosine_spacing
        self.smooth_mesh = smooth_mesh
        self.interpolation_method = interpolation_method

        self.df: Optional[pd.DataFrame] = None
        self.geometry: Optional[GeometryInterface] = None

        self.location_fields = ["THETA", "Z[m]"]
        self.value_fields = [
            "P[Pa]",
            "Rho[kg/m^3]",
            "Cr[m/s]",
            "Ct[m/s]",
            "Cz[m/s]",
        ]

        self._mesh_ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_geometry(self, geometry: GeometryInterface) -> None:
        """Attach a geometry provider."""
        self.geometry = geometry

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Attach CFD solution data."""
        self.df = df.copy()

    def interpolate(self) -> pd.DataFrame:
        """
        Perform interpolation onto the structured mesh.
        """
        self._check_ready()
        self._generate_mesh()
        self._normalize_points()
        self._interpolate_values()
        return self.interpolated

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------

    def _generate_mesh(self) -> None:
        self._determine_bounds()
        self._fit_blade_splines()
        self._generate_main_block()
        self._generate_tip_block()
        self._combine_blocks()
        self._mesh_ready = True

    def _determine_bounds(self) -> None:
        self.z_min = self.df["Z[m]"].min()
        self.z_max = self.df["Z[m]"].max()
        self.radius = self.df["R[m]"].mean()

        self.delta_theta = 2 * np.pi / self.geometry.period
        self.delta_theta_blade = 2 * np.pi / self.geometry.n_blades

        self.tip_z = np.linspace(
            self.z_min,
            self.z_min + self.geometry.tip_clearance,
            self.n_tip + 1,
        )

    def _fit_blade_splines(self) -> None:
        ps = self.geometry.pressure_side_cylindrical()
        ss = self.geometry.suction_side_cylindrical()

        self.ps_spline = UnivariateSpline(ps[:, 2], ps[:, 1], ext="extrapolate")
        self.ss_spline = UnivariateSpline(ss[:, 2], ss[:, 1], ext="extrapolate")

        self.blade_z = np.linspace(
            self.z_min + self.geometry.tip_clearance,
            self.z_max,
            self.n_axial + 1,
        )

        if self.geometry.rotation_direction < 0:
            self.theta_p = self.ps_spline(self.blade_z)
            self.theta_s = self.ss_spline(self.blade_z)
        else:
            self.theta_s = self.ps_spline(self.blade_z)
            self.theta_p = self.ss_spline(self.blade_z)

    def _generate_main_block(self) -> None:
        self.theta_grid = np.zeros((self.n_axial + 1, self.n_pitchwise + 1))
        self.z_grid = np.zeros_like(self.theta_grid)

        for i in range(self.n_axial + 1):
            gap = (self.theta_s[i] + self.delta_theta_blade) - self.theta_p[i]
            s = np.linspace(0, np.pi, self.n_pitchwise + 1)
            if self.cosine_spacing:
                self.theta_grid[i] = self.theta_p[i] + 0.5 * gap * (1 - np.cos(s))
            else:
                self.theta_grid[i] = np.linspace(
                    self.theta_p[i],
                    self.theta_s[i] + self.delta_theta_blade,
                    self.n_pitchwise + 1,
                )
            self.z_grid[i] = self.blade_z[i]

        if self.smooth_mesh:
            self.theta_grid, self.z_grid = elliptical_smoothing(
                self.theta_grid, self.z_grid
            )

        self.theta_center, self.z_center, self.cell_area = self._cell_centers(
            self.theta_grid, self.z_grid
        )

    def _generate_tip_block(self) -> None:
        tp = self.ps_spline(self.tip_z)
        ts = self.ss_spline(self.tip_z)
        mid = 0.5 * (tp + ts)

        self.tip_theta = np.zeros((self.n_tip + 1, self.n_pitchwise + 1))
        self.tip_z_grid = np.zeros_like(self.tip_theta)

        for i in range(self.n_tip + 1):
            s = np.linspace(0, np.pi, self.n_pitchwise + 1)
            self.tip_theta[i] = mid[i] + 0.5 * self.delta_theta_blade * (1 - np.cos(s))
            self.tip_z_grid[i] = self.tip_z[i]

        (
            self.tip_theta_center,
            self.tip_z_center,
            self.tip_area,
        ) = self._cell_centers(self.tip_theta, self.tip_z_grid)

    def _combine_blocks(self) -> None:
        theta = np.vstack([self.tip_theta_center, self.theta_center])
        z = np.vstack([self.tip_z_center, self.z_center])

        base = np.stack([theta, z])
        shift = np.zeros_like(base)
        shift[0] += self.delta_theta_blade

        self.mesh_points = np.concatenate([base, base + shift], axis=2)
        self.mesh_points = self.mesh_points.reshape(2, -1).T

        areas = np.vstack([self.tip_area, self.cell_area])
        self.areas = np.concatenate([areas, areas], axis=1).ravel()

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _normalize_points(self) -> None:
        pts = self.df[self.location_fields].values
        vals = self.df[self.value_fields].values

        periodic_shift = np.array([self.delta_theta, 0.0])
        self.points = np.vstack([pts, pts + periodic_shift, pts + 2 * periodic_shift])
        self.values = np.vstack([vals, vals, vals])

        self.points[:, 0] *= self.radius
        self.mesh_scaled = self.mesh_points.copy()
        self.mesh_scaled[:, 0] *= self.radius

    def _interpolate_values(self) -> None:
        rbf = RBFInterpolator(self.points, self.values, kernel="thin_plate_spline")
        interp_vals = rbf(self.mesh_scaled)

        self.interpolated = pd.DataFrame(
            interp_vals, columns=self.value_fields
        )
        self.interpolated["THETA"] = self.mesh_points[:, 0]
        self.interpolated["Z[m]"] = self.mesh_points[:, 1]
        self.interpolated["R[m]"] = self.radius
        self.interpolated["dA[m^2]"] = self.areas

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cell_centers(theta: np.ndarray, z: np.ndarray):
        ni, nj = theta.shape[0] - 1, theta.shape[1] - 1
        tc = np.zeros((ni, nj))
        zc = np.zeros_like(tc)
        area = np.zeros_like(tc)

        for i in range(ni):
            for j in range(nj):
                t = theta[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1]]
                zz = z[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1]]
                tc[i, j] = t.mean()
                zc[i, j] = zz.mean()
                area[i, j] = 0.5 * np.abs(
                    np.dot(t, np.roll(zz, -1)) - np.dot(zz, np.roll(t, -1))
                )

        return tc, zc, area

    def _check_ready(self) -> None:
        if self.df is None:
            raise RuntimeError("No DataFrame loaded.")
        if self.geometry is None:
            raise RuntimeError("No geometry provider loaded.")
