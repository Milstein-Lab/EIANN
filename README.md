# EIANN: A framework for training deep neural networks with E/I cell types and biologically-plausible learning rules

[![DOI](https://img.shields.io/badge/DOI-10.1101/2025.05.22.655599-grey.svg?style=for-the-badge&logo=doi&labelColor=green&logoColor=white)](https://doi.org/10.1101/2025.05.22.655599)

This repository contains the code described in the following publication:  
Galloni A.R., Peddada A., Chennawar Y., Milstein A.D. (2025) Cellular and subcellular specialization enables biology-constrained deep learning. [*bioRxiv* 2025.05.22.655599](https://doi.org/10.1101/2025.05.22.655599)





## 🧠 About

EIANN is a PyTorch-based tool to build and train rate-based biological neural networks, containing multiple layers of recurrently connected Excitatory (E) and Inhibitory (I) cell types.  

We provide a simple YAML-based configuration file interface for specifying network architecture, training parameters, learning rules, and cell type constraints (such as enforcing Dale's Law on connections between E and I cell types). We support a range of local rate-based learning rules, including Hebbian, and Oja's rule, and dendrite-based learning rules.



## 💻 Installation

To use EIANN, clone this GitHub Repository and install the requirements by running the following commands in your terminal:

```
git clone https://github.com/Milstein-Lab/EIANN.git
```

```
conda create --name eiann python=3.11.9
```
```
conda activate eiann  
```
```
pip install -r requirements.txt
```
```
pip install -e .
```

## ⏱ Using EIANN

The full documentation for EIANN is available at [https://milstein-lab.github.io/EIANN](https://milstein-lab.github.io/EIANN/).

Models can be created by specifying the model architecture and parameters either in a YAML configuration file or in a Python dictionary.

We provide example YAML configuration files in the `network_config/examples/` directory. You can run these models in the example jupyter notebooks we provide in the `notebooks/` directory: [`explore_MNIST.ipynb`](EIANN/notebooks/explore_MNIST.ipynb) and [`explore_spirals.ipynb`](EIANN/notebooks/explore_spirals.ipynb).

Large-scale hyperparameter optimization and analysis was implemented using a custom Python package called [*Nested*](https://github.com/neurosutras/nested). If you are interested in using Nested to run your own optimization on a computer cluster, contact Aaron Milstein (milstein at cabm.rutgers.edu).