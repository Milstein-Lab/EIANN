# EIANN
Python module extending PyTorch to train networks containing E and I cell types with biologically-plausible learning rules.


## Installation

### 1. Make sure the local project directory is in your PYTHONPATH:
- In Windows: edit environment variables
- In MacOS: edit ~/.bash_profile

### 2. Create a new conda environment:
```
conda create --name eiann python=3.11.9
conda activate eiann  
```

### 3. Install the package &  requirements:
From the root directory of the project, run:
```
pip install -r requirements.txt
```
```
pip install -e . --use-pep517
```

## Testing

### 1. Navigate to the notebooks directory
```
cd EIANN/notebooks
```

### 2. Open a jupyter notebook
```
jupyter notebook
```

### 3. Train example models using the explore_MNIST.ipynb or explore_spirals.ipynb notebooks

For large-scale model optimization and analysis instructions, contact Aaron Milstein (milstein at cabm.rutgers.edu)
