# Quantifying landscape-flux via single-cell transcriptomics uncovers the underlying mechanism of cell cycle

![image](https://github.com/Zhu-1998/cellcycle/blob/main/Workflow.jpg)
## 1. Downloading and processing the cell cycle scRNA-seq data
The scRNA-seq raw data of U2OS-FUCCI cell cycle from the paper by Mahdessian et al, which are available at GEO with accession GSE146773. Then, we could follow the snakemake pipeline at https://github.com/CellProfiling/FucciSingleCellSeqPipeline to perform scRNA-Seq data preparation and general analysis including filter, dimensionality reduction and clustering analysis.

The scEU-seq data of RPE1-FUCCI cell cycle can be extracted using dynamo’s CLI: dyn.sample_data.scEU_seq_rpe1() or from the paper by Battich et al, which are available at GEO with accession number GSE128365.

## 2. Estimating RNA velocity from scRNA-seq data
For the scRNA-seq for ~1k U2OS-FUCCI data, we can estimate the RNA velocity by scvelo or dynamo. 

For the scEU-seq data for ~3k RPE1-FUCCI cells, we use dyn.sample_data.scEU_seq_rpe1() to acquire the processed data by dynamo, which includes cell cycle clustering and RNA velocity.

## 3. Reconstructing vector field of cell cycle dynamics
We can reconstruct the vector field based RNA velocity with dyn.vf.VectorField() by using dynamo. Then, we can calculate the divergence, curl, acceleration and curvature to perform the differential geometry analysis. We can also calculate the jacobian to perform genetic perturbation and inference gene regulatory interaction.

## 4. Quantifying landscape-flux of cell cycle global dynamics and thermodynamics
We can learn an analytical function of vector field from sparse single cell samples on the entire space robustly by vector_field_function. Then, we could simulate stochastic dynamics by solving the Langevin equation based analytical function and quantify the non-equilibrium landscape-flux of the cell cycle.



