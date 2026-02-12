import app
from pathlib import Path

doc = app.document()
des = doc.design
segs = des.segments

filePath = Path(doc.path)

###############################################
### Set parameters for the specific machine ###
###############################################

# baseline number of blades
for seg in segs:
    if not seg.is_bladed:
        continue
    baseline_nblades = seg.aero.number_of_main_blades
    break

# blade number variations (iterable)
nblade_variations = list(range(8, 14))

n_segs = len(segs)

for nblades in nblade_variations:

    for seg in segs:
        if not seg.is_bladed:
            continue
        break

    if nblades == baseline_nblades:
        suffix = ""
        parent_dir = ""
    
    else:
        suffix =  f"_{nblades}b"
        parent_dir = "Blade Number Varying"

    seg.aero.number_of_main_blades = nblades

    fileName = str(filePath.stem).split('-')[0] + suffix
    outputPath = filePath.parent / parent_dir / fileName
    outputPath.mkdir(parents=True, exist_ok=True)
    doc.save(str(outputPath) + ".des")

    strset = f'<solver export_type="5" end_index="{2}" solver_type="300" start_index="0" />'
    doc.mdo_import_settings(strset)
    doc.mdo_run_solver()

    geomTurbo = outputPath.with_suffix('.geomTurbo')
    geomTurbo.replace(outputPath / geomTurbo.name)



