# Graph Neural Tangent Kernels
## An Investigation of the Induced Graph Kernels of Graph Neural Networks

[General Information](#general-info)
[Setup](#setup)
    [Virtual Environment](#venv)
[Third Example](#third-example)
[Fourth Example](#fourth-examplehttpwwwfourthexamplecom)

## Example
## Example2
## Third Example
## [Fourth Example](http://www.fourthexample.com) 

This is the Github Repository accompanying my master thesis.
The purpose of the manuscript is an investigation of the recently proposed Graph Neural Tangent Kernels by Du et al. (2019) \[1\].
Details about the contents of this repository can be found in Appendix A of the manuscript.


__Author__: Philipp Nikolaus\
__Organisation__: ETH Zurich, Seminar for Statistics\
__Submisison Date__: April 17, 2020\
__Supervisors__:
* Prof. Dr. Marloes Maathuis
* Prof. Dr. Karsten Borgwardt
* Dr. Bastian Rieck
* Leslie O'Bray

## Setup
### Virtual Environment
Experiments of this work were run in a Python 3.7 environment according to `requirements.txt`.
To setup an analogous environment:
1. Create a virtual envrionment (for instruction, see e.g. [here](https://docs.python.org/3/library/venv.html))
2. Install the packages listed in `requirements.txt`:
```
pip install -r requirements.txt
```

### Data \[1,2\]
The data (graph datasets and kernel matrices) of this work is available [here](https://www.dropbox.com/sh/2b8f7dbt3dlukij/AACtetIzzhr_LsDpP4eF6VOka?dl=0).
For running experiments paste the contents in the respective subfolders of this repository and unzip the archives.

## Results
### Replication Experiment

|           | IMDB-BINARY   | IMDB-MULTI   | MUTAG        | NCI1         | PROTEINS     | PTC_MR       |
|:----------|:--------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| Du et al. | 76.9 ± 3.6    | 52.8 ± 4.6   | 90.0 ± 8.5   | 84.2 ± 1.5   | 75.6 ± 4.2   | 67.9 ± 6.9   |
| (a.1)     | 76.90 ± 3.83  | 52.87 ± 4.71 | 90.00 ± 9.56 | 83.60 ± 1.34 | 75.77 ± 4.20 | 68.24 ± 7.76 |
| (a.2)     | 73.90 ± 4.06  | 50.67 ± 3.69 | 87.22 ± 9.64 | 83.24 ± 1.25 | 72.52 ± 3.59 | 61.76 ± 7.78 |

### Benchmark Experiment

|    | GNTK         | VH           | EH           | HGKWL_seed0   | HGKSP_seed0   | MLG          | MP           | SP           | WL           | WLOA         | GIN          |
|:-----------|:-------------|:-------------|:-------------|:--------------|:--------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| IMDBBINARY | 73.91 ± 0.83 | 50.20 ± 0.48 | 50.71 ± 0.90 | 73.01 ± 0.60  | 72.39 ± 0.87  | 60.20 ± 0.49 | 72.86 ± 0.90 | 72.19 ± 0.79 | 73.12 ± 0.62 | 73.53 ± 0.74 | 73.47 ± 0.38 |
| IMDBMULTI  | 51.38 ± 0.38 | 34.43 ± 0.82 | 33.79 ± 0.39 | 50.51 ± 0.51  | 50.93 ± 0.48  | 37.73 ± 0.31 | 50.75 ± 0.44 | 51.43 ± 0.34 | 50.32 ± 0.63 | 50.25 ± 0.52 | 51.25 ± 0.31 |
| MUTAG      | 88.09 ± 1.26 | 82.72 ± 1.03 | 84.97 ± 0.75 | 84.80 ± 1.45  | 83.59 ± 1.49  | 84.30 ± 1.53 | 85.99 ± 0.71 | 84.19 ± 1.27 | 84.48 ± 1.65 | 83.86 ± 1.11 | 85.88 ± 0.31 |
| NCI1       | 82.70 ± 0.32 | 64.53 ± 0.37 | 62.60 ± 0.38 | 85.31 ± 0.23  | 74.82 ± 0.27  | 78.64 ± 0.12 | 83.94 ± 0.25 | 75.37 ± 0.30 | 85.91 ± 0.26 | 86.15 ± 0.22 | 77.21 ± 0.57 |
| PROTEINS   | 74.27 ± 0.74 | 71.09 ± 0.21 | 63.99 ± 0.18 | 75.73 ± 0.57  | 74.88 ± 0.27  | 74.54 ± 0.47 | 74.88 ± 0.62 | NA           | 74.77 ± 0.41 | 76.24 ± 0.40 | 75.22 ± 0.45 |
| PTC        | 63.83 ± 0.99 | 56.57 ± 1.27 | 52.67 ± 2.82 | 61.80 ± 0.97  | 59.70 ± 1.88  | 59.56 ± 2.04 | 61.16 ± 1.67 | 56.93 ± 3.04 | 62.62 ± 1.50 | 62.01 ± 1.23 | 63.47 ± 1.68 |

## Running Experiments

Most of our experiments are computationally expensive. Therefore, we pass a `job_type` parameter to our scripts. The parameter can take the values `p` and `b`. If jobs are run with type `p`, they are directly run as python processes within the shell. If jobs are run with type `b`, they will be submitted as batch jobs.

Details about the experiments can be found in the Experiment section of the thesis.

### GNTK \[1\]
To start a GNTK CV experiment for a given dataset, run the following command in the command line:
```
./run_kernel_cv_gntk.sh --dataset DATASET --exp EXPERIMENT --job_type p/b
```
The parameters `DATASET` and `EXPERIMENT` have to be replaced by the name of a dataset in `[IMDBBINARY, IMDBMULTI, MUTAG, NCI1, PROTEINS, PTC]` and the name of an experiment in `[a.1, a.2, b.1]`.

### Other Graph Kernels

To start a GK CV experiment for a given graph kernel and dataset, run the following command in the command line:
```
./run_kernel_cv_gk.sh --dataset DATASET --kernel KERNEL --job_type p/b
```
The parameters `DATASET` and `KERNEL` have to be replaced by the name of a dataset in `[IMDBBINARY, IMDBMULTI, MUTAG, NCI1, PROTEINS, PTC]` and the name of a graph kernel in `[EH, VH, HGKSP_seed0, HGKWL_seed0, MLG, MP, SP, WL]`.

### GIN \[3\]

To train the GIN for a given dataset, run the following command in the command line:
```
./run_train_gin.sh --dataset DATASET --job_type p/b --gpu GPU}
```
The parameters `DATASET` and `GPU` have to be replaced by the name of a dataset in `[IMDBBINARY, IMDBMULTI, MUTAG, NCI1, PROTEINS, PTC]` and a boolean value in `[0,1]` indicating the availability of GPUs.

## References
\[1\] S.S. Du, K. Hou, R.R. Salakhutdinov, B. Poczos, R. Wang, and K. Xu. “Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels” (2019).
http://papers.nips.cc/paper/8809-graph-neural-tangent-kernel-fusing-graph-neural-networks-with-graph-kernels.

\[2\] K. Kersting, N.M. Kriege, C. Morris, P. Mutzel, and M. Neumann. "Benchmark Data Sets for Graph Kernels" (2016). 
http://graphkernels.cs.tu-dortmund.de.

\[3\] K. Xu, W. Hu, J. Leskovec, and S. Jegelka. “How Powerful are Graph Neural Net- works?” (2018). 
https://arxiv.org/abs/1810.00826.