import app
import json
from pathlib import Path

doc = app.document()
filePath = Path(doc.path)
fileName = str(filePath.stem)

outlet_save_path = Path(r"E:\NASA HECC")

des = doc.design
segs = des.segments

diffuser = False
outputs = {'diffuser_hub': [], 
           'diffuser_shroud':[]
           }

for seg in segs:
    if seg.is_bladed:
        outputs['shroud_points'] = seg.shroud.points
        outputs['hub_points'] = seg.hub.points
        outputs['shroud_beta'] = seg.shroud_beta.points
        outputs['shroud_thickness'] = seg.shroud_thickness.points
        outputs['hub_beta'] = seg.hub_beta.points
        outputs['hub_thickness'] = seg.hub_thickness.points
        outputs['EntryTheta'] = (seg.aero.entry_shroud_theta, seg.aero.entry_hub_theta)

        blade = seg.get_camber_points(0,-1)
        z = [it['z'][0] for it in blade]
        r = [it['r'][0] for it in blade]
        outputs['inlet_cant'] = (z[-1] - z[0]) / (r[-1] - r[0])

        if seg.splitter_count > 0:
            #get the starting splitter point
            splitter = seg.get_camber_points(1,-1)
            spliter_LE = [it['m'][0]/it['m'][-1] for it in splitter]
            outputs['splitter'] = [spliter_LE[0], spliter_LE[-1]]
        else:
            outputs['splitter'] = [0.0, 0.0]
        
        diffuser = True
    
    elif diffuser:
        hubPoints = seg.hub.points
        shroudPoints = seg.shroud.points
        outputs['diffuser_hub'].append([hubPoints[0], hubPoints[-1]])
        outputs['diffuser_shroud'].append([shroudPoints[0], shroudPoints[-1]])
    

output_save_file = (outlet_save_path / fileName).with_suffix(".json")
with open(output_save_file, 'w') as f:
    json.dump(outputs, f)

