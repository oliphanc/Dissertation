from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Tuple

from SIdataFrame import si_dataframe # user-provided function to standardize units and column names in a DataFrame


# ----------------------------------------------------------------------
# Data standardization and derived quantities
# ----------------------------------------------------------------------

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a CFD flow-field DataFrame by:
      - enforcing SI units
      - adding cylindrical coordinates
      - computing derived velocity, massflow, torque, and flow angles

    Parameters
    ----------
    df : pd.DataFrame
        Input flow-field data with Cartesian velocities and geometry.

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with additional derived quantities.
    """

    df = si_dataframe(df, paranthesis=True)

    _add_meridional_velocity(df)
    _add_cylindrical_components(df)
    _enforce_rotation_sign_convention(df)
    _add_velocity_magnitudes(df)
    _add_mass_flow(df)
    _add_torque(df)
    _add_flow_angle(df)

    return df


def _add_meridional_velocity(df: pd.DataFrame) -> None:
    """
    Compute the meridional velocity component Cm using the local face normal.
    """
    normals = df[["Normal[i]", "Normal[j]", "Normal[k]"]]
    velocity = df[["Cx[m/s]", "Cy[m/s]", "Cz[m/s]"]]

    df["Cm[m/s]"] = np.einsum("ij,ij->i", normals, velocity)


def _add_cylindrical_components(df: pd.DataFrame) -> None:
    """
    Add cylindrical coordinates and velocity components.
    """
    x, y = df["X[m]"], df["Y[m]"]

    df["R[m]"] = np.sqrt(x**2 + y**2)
    df["THETA"] = np.arctan2(x, y)

    df["Cr[m/s]"] = (df["Cx[m/s]"] * x + df["Cy[m/s]"] * y) / df["R[m]"]
    df["Ct[m/s]"] = (-df["Cx[m/s]"] * y + df["Cy[m/s]"] * x) / df["R[m]"]

    df["Wt[m/s]"] = (-df["Wx[m/s]"] * y + df["Wy[m/s]"] * x) / df["R[m]"]


def _enforce_rotation_sign_convention(df: pd.DataFrame) -> None:
    """
    Enforce a consistent tangential velocity sign convention such that
    positive Ct aligns with the direction of rotation.
    """
    n_negative = (df["Ct[m/s]"] < 0).sum()
    n_positive = (df["Ct[m/s]"] > 0).sum()

    if n_negative > n_positive:
        df["Ct[m/s]"] *= -1
        df["Wt[m/s]"] *= -1


def _add_velocity_magnitudes(df: pd.DataFrame) -> None:
    """
    Add absolute and relative velocity magnitudes.
    """
    df["C[m/s]"] = np.sqrt(
        df["Cx[m/s]"]**2 + df["Cy[m/s]"]**2 + df["Cz[m/s]"]**2
    )

    df["W[m/s]"] = np.sqrt(
        df["Wx[m/s]"]**2 + df["Wy[m/s]"]**2 + df["Wz[m/s]"]**2
    )


def _add_mass_flow(df: pd.DataFrame) -> None:
    """
    Compute local mass flow rate per cell.
    """
    df["Mass Flow[kg/s]"] = (
        df["Cm[m/s]"] * df["dA[m^2]"] * df["Rho[kg/m^3]"]
    )


def _add_torque(df: pd.DataFrame) -> None:
    """
    Compute local torque contribution per cell.
    """
    df["Torque[Nm]"] = (
        df["R[m]"]
        * df["Ct[m/s]"]
        * df["Cm[m/s]"]
        * df["Rho[kg/m^3]"]
        * df["dA[m^2]"]
    )


def _add_flow_angle(df: pd.DataFrame) -> None:
    """
    Add relative flow angle beta in degrees.
    """
    beta = np.arctan2(df["Wt[m/s]"], df["Cm[m/s]"])
    df["Beta[deg]"] = np.degrees(beta)


# ----------------------------------------------------------------------
# Idealized reference quantities
# ----------------------------------------------------------------------

def calculate_cm_ideal(
    df: pd.DataFrame,
    boundary_conditions: Tuple[float, float],
    machine_type: str,
    gas_constant: float = 287.0382,
) -> float:
    """
    Compute an ideal meridional velocity based on bulk mass flow.

    Parameters
    ----------
    df : pd.DataFrame
        Flow-field data.
    boundary_conditions : (Tt1, Pt1)
        Inlet total temperature [K] and pressure [Pa].
    machine_type : str
        "Compressor" or other (e.g. turbine).
    gas_constant : float, optional
        Specific gas constant [J/(kg·K)].

    Returns
    -------
    float
        Ideal meridional velocity Cm,i.
    """
    total_massflow = df["Mass Flow[kg/s]"].sum()
    total_area = df["dA[m^2]"].sum()

    if machine_type.lower() == "compressor":
        Tt1, Pt1 = boundary_conditions
        p_area_avg = np.average(df["P[Pa]"], weights=df["dA[m^2]"])
        T2_iso = Tt1 * (p_area_avg / Pt1) ** ((1.4 - 1) / 1.4)
        rho_i = p_area_avg / (gas_constant * T2_iso)
    else:
        rho_i = df["Rho[kg/m^3]"].mean()

    cm_ideal = (total_massflow / total_area) / rho_i
    df["Cm/Cm,i"] = df["Cm[m/s]"] / cm_ideal

    return cm_ideal


def calculate_w_ideal(
    df: pd.DataFrame,
    cm_ideal: float,
    beta_deg: float,
) -> float:
    """
    Compute an ideal relative velocity magnitude.

    Parameters
    ----------
    df : pd.DataFrame
        Flow-field data.
    cm_ideal : float
        Ideal meridional velocity.
    beta_deg : float
        Flow angle in degrees.

    Returns
    -------
    float
        Ideal relative velocity magnitude.
    """
    w_ideal = cm_ideal / np.cos(np.radians(beta_deg))
    df["W2/W2,i"] = df["W[m/s]"] / w_ideal * np.sign(df["Cm[m/s]"])

    return w_ideal


def calculate_w2p(
    p00: float,
    p2p: float,
    t00: float,
    u2: float,
    cp: float = 1.005,
    gamma: float = 1.4,
) -> float:
    """
    Compute the ideal relative velocity from constant rothalpy.

    Returns
    -------
    float
        Ideal relative velocity W2p.
    """
    t2p = t00 * (p2p / p00) ** ((gamma - 1) / gamma)
    h2p = cp * t2p
    h0 = cp * t00

    return np.sqrt(2 * ((h0 - h2p) * 1000 + 0.5 * u2**2))


# ----------------------------------------------------------------------
# Geometry utilities
# ----------------------------------------------------------------------

def convert_to_cylindrical(points: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian points to cylindrical coordinates.

    Parameters
    ----------
    points : (N, 3) array
        Cartesian coordinates [x, y, z].

    Returns
    -------
    (N, 3) array
        Cylindrical coordinates [r, theta, z].
    """
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta = np.arctan2(points[:, 0], points[:, 1])
    z = points[:, 2]

    return np.column_stack((r, theta, z))


def is_float(value: Any) -> bool:
    """
    Check whether a value can be safely cast to float.
    """
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
