## Nonlinear Sparse Generalized Canonical Correlation Analysis

<p align="center">
    <a href='https://arxiv.org/abs/2502.18756'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://creativecommons.org/licenses/by-nc/4.0/'>
      <img src='https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg'>
    </a>
  </p>

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
  - [x] Synthetic Datasets [synth_data.py](/Simulation/proposedmodels/synth_data.py)
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
This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

© 2025 Rong Wu. You are free to share and adapt the material with attribution.
