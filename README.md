### [Nonlinear Sparse Generalized Canonical Correlation Analysis](https://arxiv.org/abs/2502.18756)

Official Python implementation of NSGCCA, from the following paper:

[Nonlinear Sparse Generalized Canonical Correlation Analysis for Multi-view High-dimensional Data](https://arxiv.org/abs/2502.18756).  \
Rong Wu, Ziqi Chen, Gen Li and Hai Shu. \
New York University \
[[`arXiv`](https://arxiv.org/abs/2502.18756)]

---

We propose three nonlinear, sparse, generalized CCA methods, **HSIC-SGCCA**, **SA-KGCCA**, and **TS-KGCCA**, for variable selection in multi-view high-dimensional data. 
These methods extend existing SCCA-HSIC, SA-KCCA, and TS-KCCA from two-view to multi-view settings. While SA-KGCCA and TS-KGCCA yield multi-convex optimization problems solved via block coordinate descent, HSIC-SGCCA introduces a necessary unit-variance constraint previously ignored in SCCA-HSIC, resulting in a nonconvex, non-multiconvex problem.
We efficiently address this challenge by integrating the block prox-linear method with the  linearized
alternating direction method of multipliers. 
Simulations and TCGA-BRCA data analysis demonstrate that HSIC-SGCCA outperforms competing methods in variable selection.

 ## Installation
Clone this repository and install other required packages:
```
git clone git@github.com:Rows21/NSGCCA
```
 
 ## Datasets
  - [x] Synthetic Datasets [synth_data.py](/NSGCCA/synth_data.py)
  - [x] TCGA Breast Cancer Database in [Realdata](/Realdata/Data_download_preprocess.R) from (https://tcga-data.nci.nih.gov/docs/publications)
 
 (Feel free to post suggestions in issues of recommending latest proposed CCA network for comparison. Currently, the baselines folder is to put comparable models.)
 
 <!-- ✅ ⬜️  -->

 ## Citation
If you find this repository helpful, please consider citing:
```
@article{wu2025nonlinear,
  title={Nonlinear Sparse Generalized Canonical Correlation Analysis for Multi-view High-dimensional Data},
  author={Wu, Rong and Chen, Ziqi and Li, Gen and Shu, Hai},
  journal={arXiv preprint arXiv:2502.18756},
  year={2025}
}
```

 ## Results 
 ### Simulation Studies
 [Figure 2](/Results): The simulation performance for Synthetic Datasets. 
 ### Real-World Studies -- TCGA breast cancer database
 [Data_download_preprocess](/Realdata/Data_download_preprocess.R): TCGA-BRCA preprocessing through R script. <br>
 [Venn Diagram](/Results): The clustering results for TCGA-BRCA. 

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
