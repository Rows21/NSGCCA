### [Nonlinear Sparse Generalized Canonical Correlation Analysis]()

Official Python implementation of NSGCCA, from the following paper:

[Nonlinear Sparse Generalized Canonical Correlation Analysis for Multi-view High-dimensional Data]().  \
Rong Wu, Ziqi Chen, Gen Li and Hai Shu. \
New York University \
[[`arXiv`]()]

---

We propose **SNGCCA**.

 ## Installation
Clone this repository and install other required packages:
```
git clone git@github.com:Rows21/NSGCCA
```

 ## Datasets
  - [x] Synthetic Datasets [synth_data.py](/NSGCCA/synth_data.py)
  - [x] TCGA Breast Cancer Database from (https://tcga-data.nci.nih.gov/docs/publications)
 
 (Feel free to post suggestions in issues of recommending latest proposed CCA network for comparison. Currently, the network folder is to put the current SOTA models. We can further add the recommended network in it for training.)
 
 <!-- ✅ ⬜️  -->

 ## Citation
If you find this repository helpful, please consider citing:
```

```

 ## Results 
 ### Simulation Studies

<div class="main">
<div class="tag">
</div>
<div class="images" >
	<div class="mid">
		<img src="/screenshots/Fig2a.png" />
	</div>
	<div class="mid">
		<img src="/screenshots/Fig2b.png" />
	</div>
	<div class="mid">
		<img src="/screenshots/Fig2c.png" />
	</div>
</div>
<div style="clear:both;"></div>
<div style="margin-bottom:30px;">
</div>

### Real-World Studies -- TCGA breast cancer database
<div class="main">
<div class="tag">
</div>
<div class="images" >
	<div class="mid">
		<img src="/screenshots/FigHeat.jpg" />
	</div>
</div>
<div style="clear:both;"></div>
<div style="margin-bottom:30px;">
</div>

<div class="main">
<div class="tag">
</div>
<div class="images" >
	<div class="mid">
		<img src="/screenshots/venndiagram/FigVenna.png" />
	</div>
	<div class="mid">
		<img src="/screenshots/venndiagram/FigVennb.png" />
	</div>
	<div class="mid">
		<img src="/screenshots/venndiagram/FigVennc.png" />
	</div>
</div>
<div style="clear:both;"></div>
<div style="margin-bottom:30px;">
</div>

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
