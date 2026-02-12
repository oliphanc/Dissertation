"""
===============================================================================
AxCent API Usage Example
===============================================================================

This file demonstrates how to use API calls to the AxCent software for getting
and setting various design parameters, including blade angles, hub and shroud
contours, and thicknesses.
"""

import app
from pathlib import Path

doc = app.document()
filePath = Path(doc.path)
fileName = str(filePath.stem)
des = doc.design
segs = des.segments


# Find the bladed segments. If the indicies are known, they can be used directly.
# Otherwise, iterate through the segments to find the ones that are bladed.
for seg in segs:
    if not seg.is_bladed:
        continue

    # get the segment's hub and shroud points, beta angles, and thicknesses
    # returns tuples of tuples
    shroud_points = seg.shroud.points
    hub_points = seg.hub.points
    shroud_beta = seg.shroud_beta.points
    shroud_thickness = seg.shroud_thickness.points
    hub_beta = seg.hub_beta.points
    hub_thickness = seg.hub_thickness.points

    
    # each point can be modified individually using the associated set_point method
    # for example, decrease R1h by 2%
    R1h = hub_points[0][1]
    R1h_new = R1h * 0.98
    seg.hub.set_point(0, (hub_points[0][0], R1h_new))

    # similarly, we can modify beta
    # for example, increase beta_2t by 5 degrees
    beta_2t = shroud_beta[-1][1]
    beta_2t_new = beta_2t + 5.0
    i = len(shroud_beta) - 1
    seg.shroud_beta.set_point(i, (shroud_beta[-1][0], beta_2t_new))

    
doc.save()