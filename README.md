# High-Resolution_Load_Profiling
## Breif intro
The algorithm presented in this repository is aimed to provide a more accurate and generalizable algorithm for building load profiling.The algorithm is established as class methods in HRLP.py. An example of how to apply the algorithm for load profiling is described in Jupyter notebook, as well as the baseline algorithms. For more details and the experiment results please refer to the [paper](https://ideaslab.io/publication/pdf/zhan-2019-robust.pdf) published in [Building Simulation 2019](http://buildingsimulation2019.org).

## Framework overview
The framework consists of four main parts: pre-processing, preliminary K-means, Finer DBSCAN and post-processing (Figure 1.a). In figure 1.b, the profiling result of 2 sample buildings are compared with 2 baseline clustering methods K-means and DBSCAN.

<img src="https://user-images.githubusercontent.com/23103678/65244149-7204df80-db1c-11e9-9803-b4c4e9cab717.jpg" width=95%>
