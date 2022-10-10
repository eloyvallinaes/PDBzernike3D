"""
Turn PDB coordinates into voxel representation of their surfaces.
This implementation relies on two binaries from msms (see
https://ccsb.scripps.edu/msms/documentation/) which are run with subprocess.
"""

import trimesh
import numpy as np
from pathlib import Path
from gridData import Grid
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

from PDBzernike3D.voxelise import load_vertices, load_faces, represent


if __name__ == "__main__":
    dir = Path("examples/trypsin_inh")
    code = "1tpa.I"

    # APBS electronic potential grid
    grid = Grid(dir / f"potential.dx")
    gridCoords = np.array(np.meshgrid(*grid.midpoints)).T.reshape(-1, 3)
    potential = trimesh.points.PointCloud(gridCoords)

    # protein surface mesh
    vv = load_vertices(dir / f"{code}.vert")
    ff = load_faces(dir / f"{code}.face")
    surface = trimesh.Trimesh(vertices=vv, faces=ff)

    #
    ww = np.arange(1, 21)[::-1]
    _, potInd = potential.kdtree.query(surface.vertices, 20)
    potValues = []
    for ind in potInd:
        potValues.append(
            np.dot(
                grid.grid[np.unravel_index(ind, grid.grid.shape)],
                ww,
            ) / ww.sum()
        )
    norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    colors = cm.bwr_r(norm(potValues))
    surface.visual.vertex_colors = colors
    # surface.show()

    #
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(
    #     surface.vertices[:, 0],
    #     surface.vertices[:, 1],
    #     surface.vertices[:, 2],
    #     c=colors,
    #     marker="."
    # )
    # plt.show()
