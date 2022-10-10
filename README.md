# 3D Zernike Decomposition of Protein Surfaces

Outline:
1. PDB structure of AlphaFold model of protein of interest
2. Obtain Connolly's solvent accessible surface area with MSMS as a triangulated
   surface mesh.
3. Work out protein protonation states and generate input for APBS calculations
4. Run APBS calculations and obtain electrostatic potential solutions on a grid
   containing the protein.
5. Map surface vertices to k-nearest potential grid values
6. Colour surface in red-white-blue ramp and compare with results in pymol


## 
