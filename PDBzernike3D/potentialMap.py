"""
Turn PDB coordinates into voxel representation of their surfaces.
This implementation relies on two binaries from msms (see
https://ccsb.scripps.edu/msms/documentation/) which are run with subprocess.
"""

import trimesh
import numpy as np
import pyvista as pv
from pathlib import Path
from gridData import Grid
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d

from PDBzernike3D.voxelise import load_vertices, load_faces, represent


if __name__ == "__main__":
    folder = Path("examples/trypsin_inh")
    code = "1tpa.I"

    # APBS electronic potential grid
    grid = Grid(folder / f"potential.dx")
    gridCoords = np.array(np.meshgrid(*grid.midpoints)).T.reshape(-1, 3)
    potential = trimesh.points.PointCloud(gridCoords)

    # protein surface mesh
    vv = load_vertices(folder / f"{code}.vert")
    ff = load_faces(folder / f"{code}.face")
    surface = trimesh.Trimesh(vertices=vv, faces=ff)

    #
    ww = np.arange(1, 21)[::-1]
    _, potInd = potential.kdtree.query(surface.vertices, 80)
    potValues = {
        tuple(vertex): grid.grid[np.unravel_index(ind, grid.grid.shape)].mean()
        for vertex, ind in zip(surface.vertices, potInd)
    }
    # aggresively smooth over surface neighbouring vertices
    smoothed = {}
    for vertex, value in potValues.items():
        _, verInd = surface.kdtree.query(vertex, 16)
        verVals = [
            potValues[tuple(vertex)]
            for vertex in surface.vertices[verInd]
        ]
        smoothed[tuple(vertex)] = gaussian_filter1d(verVals, 2)[0]
    # convert to colors
    norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    normed = np.nan_to_num(
        norm(
            np.array(list(smoothed.values())).data
        ),
        copy=True,
        posinf=1,
        neginf=-1
    )
    cloud = pv.wrap(surface)
    cloud['data'] = normed
    # cloud.plot(cmap='bwr_r')
