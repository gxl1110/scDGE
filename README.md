# scDGE



`scDGE` is a novel deep clustering framework that dynamically integrates a Graph Attention-based Encoder (GATE) with multiple self-supervised learning tasks. Built on a Mixture of Experts (MoE) architecture, scDGE treats different self-supervised tasks as expert networks to capture multi-level expressive features and topological information. A dynamic gating mechanism further assigns adaptive weights to different experts based on each cell’s topology and gene expression pattern, enabling cell-specific supervision and more robust clustering performance.

We recommend using an environment based on Python 3.9. You can set up the environment as follows:

```bash
conda create -n scdge python=3.9 -y
conda activate scdge
```

First, install `PyTorch` and `PyG` according to your machine’s CUDA version, then install the remaining dependencies:

```bash
pip install scanpy numpy scipy pandas scikit-learn h5py matplotlib seaborn munkres ogb deeprobust
```

If you are using a GPU, please ensure that you install a version of `torch` / `torch-geometric` that is compatible with CUDA, in accordance with the official instructions.

## Data format

The current repository defaults to reading:

```text
data/Human1/Human1.h5
```



## Quick Start

If you wish to use the sample data and pre-trained weights included in the current repository, run the following command:

```bash
python train.py --device cuda --gpu 0
```

If you don’t have a GPU, you can also change it to:

```bash
python train.py --device cpu
```

