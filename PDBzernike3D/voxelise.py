"""
Turn PDB coordinates into voxel representation of their surfaces.
This implementation relies on two binaries from msms (see
https://ccsb.scripps.edu/msms/documentation/) which are run with subprocess.
"""

import os
import trimesh
import subprocess
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def pdb_to_xyzr(filename: str, outname=None):
    """
    Prepation step to use msms. Turn PDB coordiantes into xyzr representation
    with msms-provided binary.

    :param filename: input PDB file
    :type filename: str
    :param outname: optional output filename, default is filename.stem + ".xyzr"
    :type outname: str or None

    """
    filepath = Path(filename)
    binary = os.path.join(os.path.dirname(__file__), "bin", "pdb_to_xyzr")
    if not outname:
        outname = filepath.with_suffix(".xyzr")

    p = subprocess.run([binary, filepath], capture_output=True)
    with open(outname, "w") as outfile:
        outfile.write(p.stdout.decode())

    return


def msms(filename: str, outname=None):
    """
    MSMS run from binary. Obtain vertices and faces from xyzr file.

    :param filename: xyzr file path.
    :type filename: str
    :param outname: optional output filename stem to which to which MSMS will
    automatically add .vert and .face extensions.
    :type outname: str or None

    """
    filepath = Path(filename)
    binary = os.path.join(os.path.dirname(__file__), "bin", "msms")
    if not outname:
        outname = filepath.with_suffix("")

    p = subprocess.run([
        binary,
        "-if", filename,
        "-of", outname,
        "-density", "5",
        "-hdensity", "50"
        ],
        capture_output=True
    )

    with open(outname.with_suffix(".log"), "w") as logfile:
        logfile.write(p.stdout.decode())

    return


def load_vertices(fname: str):
    """
    Load vertex coordinates in np.ndarray ignoring comments and additional
    header as appropriate.

    :param fname: file path of vertex file.
    :type fname: str
    :return: x y z coordinates of vertices
    :rtype: np.ndarray

    """
    xyz = []
    with open(fname, "r") as vertfile:
        while True:
            line = vertfile.readline()
            if not line:
                break
            elif line.startswith("#") or len(line.split()) < 5:
                continue

            xyz.append(
                np.array(line.split()[:3]).astype(float)
            )
    return np.array(xyz)


def load_faces(fname):
    """
    Load face coordinates in np.ndarray ignoring comments and additional header
    as appropriate.

    :param fname: file path of face file.
    :type fname: str
    :return: i j k indices of face vertices
    :rtype: np.ndarray

    """
    ijk = []
    with open(fname, "r") as facefile:
        while True:
            line = facefile.readline()
            if not line:
                break
            elif line.startswith("#") or len(line.split()) < 5:
                continue

            ijk.append(
                np.array(line.split()[:3]).astype(float) - 1  # one-based index
            )
    return np.array(ijk)


def represent(matrix: np.ndarray, **kwargs):
    """
    Represent in matplotlib a dense boolean matrix of voxels.
    Slow, testing only.

    :param matrix: any 3d dense matrix to interpret as voxels, typically
    obtained from the  trimesh.voxel.base.VoxelGrid.matrix property.
    :type matrix: np.ndarray
    :return: a matplotlib figure instance
    :rtype: plt.figure

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('auto')
    ax.voxels(matrix, **kwargs)
    return fig


if __name__ == "__main__":
    # run msms to obtain triagulated surface
    dir = Path("examples/trypsin_inh")
    code = "1tpa.I"
    pdb_to_xyzr(dir / f"{code}.pdb")
    msms(dir / f"{code}.xyzr")
    # load triagulated surface to TriMesh
    vv = load_vertices(dir / f"{code}.vert")
    ff = load_faces(dir / f"{code}.face")
    mesh = trimesh.Trimesh(vertices=vv, faces=ff)
    mesh.show()
    # # center at origin
    # tx, ty, tz = -mesh.centroid
    # T = (tx, ty, tz)
    # centered_mesh = mesh.apply_translation(T)
    # # turn mesh into voxels
    # voxels = centered_mesh.voxelized(pitch=2)
    # # represent in MPL
    # f = represent(voxels.matrix)
    # f.savefig(dir / "1ris_voxels_p1.png", dpi=300, bbox_inches="tight")
    # # slice in half
    # halved = voxels.matrix.copy()
    # halved[:, :, 15:] = False
    # f = represent(halved)
    # f.savefig(dir / "1ris_halved_p1.png", dpi=300, bbox_inches="tight")
    # # apply hollow method
    # f = represent(voxels.hollow().matrix)
    # f.savefig(dir / "1ris_hollow_p1.png", dpi=300, bbox_inches="tight")
    # # hollow and slice
    # hollow = voxels.hollow().matrix
    # hollow[:, :, 15:] = False
    # f = represent(hollow)
    # f.savefig(dir / "1ris_hollow_sliced_p1.png", dpi=300, bbox_inches="tight")
