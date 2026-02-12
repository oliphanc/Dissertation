from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd

from radial import CCompal  #COM object for COMPAL interface
from CreateCompal import (
    get_geo_variables,
    get_flow_variables,
    create_meanline_input,
    setup_compal,
)
from OptimizeCompal import calculate_mixed_outlet, optimize_compal
from utils.helper import standardize_df


# ============================================================
# Path setup (formerly notebook sys.path cells)
# ============================================================

ROOT = Path.cwd()
sys.path.append(str(ROOT.parent / "MLModelTraining"))
sys.path.append(str(ROOT.parent / "TwoZoneCalculations"))
sys.path.append(str(ROOT.parent / "Lib3.0" / "Lib3.0"))


# ============================================================
# Load CFD data
# ============================================================

with open("inlet_flowfield.pkl", "rb") as f:
    inlet_ff = pickle.load(f)

with open("outlet_flowfield.pkl", "rb") as f:
    outlet_ff = pickle.load(f)

inlet_ff = standardize_df(inlet_ff)
outlet_ff = standardize_df(outlet_ff)


# ============================================================
# Load geometry object
# ============================================================

with open("geometry_object.pkl", "rb") as f:
    geom_object = pickle.load(f)


# ============================================================
# Create COMPAL object
# ============================================================

compal = CCompal()
compal.Init()


# ============================================================
# Extract inputs
# ============================================================

N = 1  # stage index

geo_variables = get_geo_variables(geom_object)
flow_variables = get_flow_variables(inlet_ff, outlet_ff, N)

meanline_input = create_meanline_input(flow_variables, geo_variables)


# ============================================================
# Set COMPAL parameters
# ============================================================

setup_compal(compal, meanline_input)

if not compal.Run():
    raise RuntimeError("Initial COMPAL run failed.")


# ============================================================
# CFD mixed outlet reference
# ============================================================

P2_cfd, Cm2_cfd, Ct2_cfd = calculate_mixed_outlet(outlet_ff)
cfd_reference = np.array([P2_cfd, Cm2_cfd, Ct2_cfd])


# ============================================================
# Optimization
# ============================================================

mr2_solution, opt_solution = optimize_compal(
    compal,
    cfd_reference,
    x0=[0.2, -5.0, 0.0],
)

print("\n=== Optimization Results ===")
print(f"MR2 solution: {mr2_solution.x}")
print(f"Secondary zone solution: {opt_solution.x}")
print(f"Objective value: {opt_solution.fun}")


# ============================================================
# Final COMPAL run with optimized parameters
# ============================================================

compal.SetParameter(1, "MR2", mr2_solution.x[0])
compal.SetParameter(1, "MSEC_M", opt_solution.x[0])
compal.SetParameter(1, "DELTAPin", opt_solution.x[1])
compal.SetParameter(1, "DELTAS", opt_solution.x[2])

compal.Run()

final_results = {
    "MR2": mr2_solution.x[0],
    "chi": opt_solution.x[0],
    "delta_2p": opt_solution.x[1],
    "delta_2s": opt_solution.x[2],
}

with open("optimization_results.pkl", "wb") as f:
    pickle.dump(final_results, f)


if __name__ == "__main__":
    pass
