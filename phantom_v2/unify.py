import trimesh
import os

stl_dir = "ModelloPhantom"

# Carica e unisci tutti i file
meshes = []
for f in os.listdir(stl_dir):
    if f.endswith('.stl'):
        m = trimesh.load(os.path.join(stl_dir, f), force='mesh')
        meshes.append(m)

combined = trimesh.util.concatenate(meshes)
combined.export("phantom_combined.stl")
print(f"Bounds: {combined.bounds}")
print(f"Extents: {combined.extents}")
