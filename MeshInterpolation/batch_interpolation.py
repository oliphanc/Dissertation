import pandas as pd
from pathlib import Path

from mesh_interpolation.interpolator import MeshInterpolator
from my_geometry_adapter import GeomTurboAdapter  # user-provided

geometry = GeomTurboAdapter("example.geomTurbo")

interp = MeshInterpolator(
    n_axial=40,
    n_pitchwise=64,
    n_tip=10,
    cosine_spacing=True,
)

interp.load_geometry(geometry)

for csv in Path("data/flowfields").glob("*.csv"):
    df = pd.read_csv(csv)
    interp.load_dataframe(df)
    out = interp.interpolate()
    out.to_csv(f"output/{csv.stem}_interp.csv", index=False)
