:::{note}
🚧 **Work in Progress:** This documentation site is currently under construction. Content may change frequently.
:::


# EIANN Documentation

**A framework for training neural networks with E/I cell types and biologically-plausible learning rules**

---

## About EIANN

EIANN (Excitatory/Inhibitory Artificial Neural Networks) is a PyTorch-based library designed to build and train rate-based biological neural networks containing multiple layers of recurrently connected **Excitatory (E)** and **Inhibitory (I)** cell types.

EIANN was created to accelerate research within computational neuroscience with a focus on bio-inspired learning rules. We provide a neural network syntax that is intuitive from a neuroscience perspective, organized around neuronal populations and their projections. 

We specifically designed EIANN to make reproducible experiments, hyperparameter optimization, and neural architecture search easier by providing a simple YAML-based configuration file interface for specifying network architecture, training parameters, learning rules, and cell type constraints. EIANN allows you to easily specify biological constraints and mechanisms such as:

- **Dale's Law**: Enforces biologically realistic constraints on connections between E and I cell types
- **Local Learning Rules**: You can create arbirary learning rules at any projection in the network, or just use one of the existing gradient-based, dendrite-based, or Hebbian learning mechanisms.
- **E/I Cell Types**: Explicit modeling of diverse neural populations
- **Recurrent Connections**: Connections can be made either recurrent or feedforward and can arbitrarily connect any two neural populations

In addition to the above, EIANN provides a range of analysis and visualization tools to help you understand your network's behavior and learning dynamics.

## Publications
EIANN is based on research published in:

> **Galloni A.R., Peddada A., Chennawar Y., Milstein A.D.** (2025)  
> *Cellular and subcellular specialization enables biology-constrained deep learning.*  
> bioRxiv 2025.05.22.655599  
> [https://doi.org/10.1101/2025.05.22.655599](https://doi.org/10.1101/2025.05.22.655599)


<!-- ## Quick Example

```python
import eiann
import yaml

# Load network configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create and train network
network = eiann.create_network(config)
network.train(data_loader, epochs=100)

# Analyze results
network.analyze_connectivity()
network.plot_learning_dynamics()
``` -->

## Use Cases

EIANN is particularly well-suited for:

- **Computational Neuroscience Research**: Understanding how biological constraints affect learning
- **Biologically-Inspired AI**: Developing AI systems that incorporate brain-like mechanisms  
- **Educational Applications**: Teaching neural network principles with biological realism


## Getting Started

1. **[Installation](installation.md)**: Set up EIANN in your environment
2. **[Quick Start](quickstart.ipynb)**: Build your first E/I network
3. **[Tutorials](tutorials/mnist_example.ipynb)**: Work through detailed examples
4. **[User Guide](user_guide/basic_usage.ipynb)**: Learn about all features

## Support

- **GitHub**: [https://github.com/Milstein-Lab/EIANN](https://github.com/Milstein-Lab/EIANN)
- **Issues**: [Report bugs and request features](https://github.com/Milstein-Lab/EIANN/issues)

---

*EIANN is developed by the [Milstein Lab] at Rutgers University as part of ongoing research into biologically-constrained deep learning.*