# Installation

EIANN is written in Python. This guide will help you install EIANN and its dependencies on your system.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Milstein-Lab/EIANN.git
   cd EIANN
   ```

2. **Create a conda environment**:
   ```bash
   conda create --name eiann python=3.11.9
   conda activate eiann
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install EIANN in development mode**:
   ```bash
   pip install -e . --use-pep517
   ```


### For Hyperparameter Optimization
If you plan to use the [Nested](https://github.com/neurosutras/nested) package for large-scale hyperparameter optimization:

```{note}
Integration with Nested hyperparameter optimization for cluster computing requires additional setup. Contact Aaron Milstein (milstein at cabm.rutgers.edu) for assistance with cluster-based optimization.
```
<!-- 
## Directory Structure After Installation

After installation, your EIANN directory should look like:

```
EIANN/
├── EIANN/                  # Main package
│   ├── __init__.py
│   ├── models/            # Network models
│   ├── layers/            # Layer implementations  
│   ├── learning_rules/    # Learning rule implementations
│   ├── utils/             # Utility functions
│   └── notebooks/         # Example notebooks
├── network_config/
│   └── examples/          # Example YAML configurations
├── requirements.txt       # Dependencies
├── setup.py              # Installation script
└── README.md
``` -->

## Getting Started

Now that you have EIANN installed, you can:

1. **Explore the examples**: Check out the notebooks in `EIANN/notebooks/`
   - `explore_MNIST.ipynb`: MNIST classification example
   - `explore_spirals.ipynb`: Spiral dataset example

2. **Review configuration files**: Look at the YAML examples in `network_config/examples/`

3. **Continue to the [Quick Start guide](tutorial.ipynb)**


### Getting Help

If you encounter issues:
1. **Check the [GitHub Issues](https://github.com/Milstein-Lab/EIANN/issues)** for known problems
2. **Create a new issue** with details about your problem
3. **Contact the developers**: milstein at cabm.rutgers.edu
