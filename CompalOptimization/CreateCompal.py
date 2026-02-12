from radial import CCompal
import numpy as np


class MassFlowError(Exception):
    pass


class CompalError(Exception):
    pass


# ============================================================
# COMPAL variable mapping (ONLY place names appear)
# ============================================================

COMPAL_MAP = {
    "total_temperature": "T00",
    "total_pressure": "P00",
    ... # Add more mappings as needed
}


# ============================================================
# Geometry
# ============================================================

def get_geo_variables(geom):
    """
    Return geometry in solver-agnostic, semantic form.
    """

    return {
        "inlet_radius_tip": geom.r1t,
        "inlet_radius_hub": geom.r1h,
        ... # Add more geometry variables as needed
    }


# ============================================================
# Flow
# ============================================================

def calculate_upstream_state(df):
    T = df['P[Pa]'] / (287 * df['Rho[kg/m^3]'])
    temp_ratio = (df['P[Pa]'] / df['Absolute Total Pressure[Pa]']) ** (0.4 / 1.4)

    T0 = T / temp_ratio
    T00 = (T0 * df['Mass Flow[kg/s]']).sum() / df['Mass Flow[kg/s]'].sum()
    P00 = (df['Absolute Total Pressure[Pa]'] * df['dA[m^2]']).sum() / df['dA[m^2]'].sum()

    return T00, P00


def get_flow_variables(inlet_ff, outlet_ff, N):

    T00, P00 = calculate_upstream_state(inlet_ff)

    Mi = abs((inlet_ff['Cm[m/s]'] * inlet_ff['Rho[kg/m^3]'] * inlet_ff['dA[m^2]']).sum())
    Mo = abs((outlet_ff['Cm[m/s]'] * outlet_ff['Rho[kg/m^3]'] * outlet_ff['dA[m^2]']).sum())

    if abs(Mi - Mo) / Mi > 0.1:
        raise MassFlowError("Mass flow imbalance exceeds tolerance.")

    return {
        "total_temperature": T00,
        "total_pressure": P00,
        "mass_flow": 0.5 * (Mi + Mo),
        "rotational_speed": N,
        "area_kinetic_factor": 1.0,
        "blockage": 1.0,
    }


# ============================================================
# Meanline + COMPAL interface
# ============================================================

def create_meanline_input(flow_vars, geo_vars):

    inputs = flow_vars | geo_vars
    inputs["mass_flow"] *= inputs["blade_periodicity"]

    missing = set(COMPAL_MAP.keys()) - inputs.keys()
    if missing:
        raise CompalError(f"Missing inputs: {missing}")

    return inputs


def setup_compal(compal_link, meanline_input):

    for semantic_key, compal_key in COMPAL_MAP.items():
        value = meanline_input[semantic_key]
        status = compal_link._radial.SetParameter(1, compal_key, value)

        if status != 0:
            raise CompalError(f"Failed to set COMPAL parameter: {semantic_key}")


if __name__ == "__main__":
    pass
