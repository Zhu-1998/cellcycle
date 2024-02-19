# Quantifying landscape-flux via single-cell transcriptomics uncovers the underlying mechanism of cell cycle

![image](https://github.com/Zhu-1998/cellcycle/blob/main/Graphical-abstract.jpg)
## 1. Downloading and processing the cell cycle scRNA-seq data
The scRNA-seq raw data of U2OS-FUCCI cell cycle from the paper by [Mahdessian et al](https://doi.org/10.1038/s41586-021-03232-9), which are available at GEO with accession GSE146773. Then, we could follow the snakemake pipeline at https://github.com/CellProfiling/FucciSingleCellSeqPipeline to perform scRNA-Seq data preparation and general analysis including filter, dimensionality reduction and clustering analysis. The processed data of U2OS-FUCCI cell cycle is in `cell_cycle.h5ad`.

The scEU-seq data of RPE1-FUCCI cell cycle can be extracted using [dynamo](https://github.com/aristoteleo/dynamo-release)’s CLI: `dyn.sample_data.scEU_seq_rpe1()` or from the paper by [Battich et al](https://doi.org/10.1126/science.aax3072), which are available at GEO with accession number GSE128365.

The scRNA-seq raw data of the human fibroblast cell cycle from the paper by [Riba et al](https://doi.org/10.1038/s41467-022-30545-8), which are available at GEO with accession GSE167609. 

## 2. Estimating RNA velocity from scRNA-seq data
For the scRNA-seq for ~1k U2OS-FUCCI data, we can estimate the RNA velocity by [scvelo](https://github.com/theislab/scvelo) or [dynamo](https://github.com/aristoteleo/dynamo-release). 

For the scEU-seq data for ~3k RPE1-FUCCI cells, `dyn.sample_data.scEU_seq_rpe1()` to acquire the processed data, which includes cell cycle clustering and RNA velocity.

For the scRNA-seq data for ~3k human fibroblasts, the annotated data `velocity_anndata_human_fibroblast_DeepCycle_ISMARA.h5ad` with the cell cycle phase and RNA velocity can be downloaded from [Zenodo](https://zenodo.org/records/4719436).

## 3. Reconstructing vector field of cell cycle dynamics
We can reconstruct the vector field based RNA velocity with `dyn.vf.VectorField()` by using [dynamo](https://github.com/aristoteleo/dynamo-release). Then, we can calculate the divergence, curl, acceleration and curvature to perform the differential geometry analysis. We can also calculate the jacobian to perform genetic perturbation and inference gene regulatory interaction.

## 4. Quantifying landscape-flux of cell cycle global dynamics and thermodynamics
We can learn an analytical function of vector field from sparse single cell samples on the entire space robustly by `vector_field_function`. Then, we could simulate stochastic dynamics by solving the Langevin equation based analytical function and quantify the non-equilibrium landscape-flux of the cell cycle.



