# EIANN Documentation

**A framework for training deep neural networks with E/I cell types and biologically-plausible learning rules**

---

## About EIANN

EIANN (Excitatory/Inhibitory Artificial Neural Networks) is a PyTorch-based framework designed to build and train rate-based biological neural networks containing multiple layers of recurrently connected **Excitatory (E)** and **Inhibitory (I)** cell types.

Unlike traditional artificial neural networks, EIANN incorporates key biological constraints and mechanisms:

- **Dale's Law**: Enforces biologically realistic constraints on connections between E and I cell types
- **Local Learning Rules**: Supports Hebbian learning, Oja's rule, and dendrite-based learning mechanisms
- **E/I Cell Types**: Explicit modeling of excitatory and inhibitory neuron populations
- **Rate-based Dynamics**: Biologically-inspired neural dynamics while maintaining computational efficiency

## Key Features

### 🧠 **Biologically-Constrained Architecture**
- Separate excitatory and inhibitory cell populations
- Dale's Law enforcement for realistic connectivity
- Recurrent connections within and between layers

### 📚 **Local Learning Rules**
- **Hebbian Learning**: "Cells that fire together, wire together"
- **Oja's Rule**: Normalized Hebbian learning for stable synaptic weights
- **Dendrite-based Learning**: Compartmentalized learning mechanisms

### ⚙️ **Easy Configuration**
- Simple YAML-based configuration system
- Specify network architecture, training parameters, and learning rules
- No complex coding required for canonical biological circuit architectures

### 🔬 **Research-Ready**
- Built for computational neuroscience research
- Integration with hyperparameter optimization (via [Nested](https://github.com/neurosutras/nested))
- Extensive analysis and visualization tools

## Scientific Foundation
EIANN is based on research published in:

> **Galloni A.R., Peddada A., Chennawar Y., Milstein A.D.** (2025)  
> *Cellular and subcellular specialization enables biology-constrained deep learning.*  
> bioRxiv 2025.05.22.655599  
> [https://doi.org/10.1101/2025.05.22.655599](https://doi.org/10.1101/2025.05.22.655599)

## What Makes EIANN Different?

Traditional deep learning frameworks like PyTorch and TensorFlow are designed for computational efficiency but ignore biological realism. EIANN bridges this gap by:

1. **Enforcing Biological Constraints**: Dale's Law and E/I balance aren't optional—they're built into the architecture
2. **Local Learning**: Unlike backpropagation, EIANN uses learning rules that could plausibly exist in biological neural networks
3. **Cell Type Specialization**: Different neuron types have different roles, just like in the brain
4. **Research Focus**: Designed specifically for computational neuroscience rather than general machine learning

## Quick Example

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
```

## Use Cases

EIANN is particularly well-suited for:

- **Computational Neuroscience Research**: Understanding how biological constraints affect learning
- **Biologically-Inspired AI**: Developing AI systems that incorporate brain-like mechanisms  
- **Educational Applications**: Teaching neural network principles with biological realism
- **Comparative Studies**: Examining differences between biological and artificial learning

## Getting Started

1. **[Installation](installation.md)**: Set up EIANN in your environment
2. **[Quick Start](quickstart.ipynb)**: Build your first E/I network
3. **[Tutorials](tutorials/mnist_example.ipynb)**: Work through detailed examples
4. **[User Guide](user_guide/basic_usage.ipynb)**: Learn about all features

## Community and Support

- **GitHub**: [https://github.com/Milstein-Lab/EIANN](https://github.com/Milstein-Lab/EIANN)
- **Issues**: [Report bugs and request features](https://github.com/Milstein-Lab/EIANN/issues)
- **Contact**: Aaron Milstein (milstein at cabm.rutgers.edu)

---

*EIANN is developed by the [Milstein Lab] at Rutgers University as part of ongoing research into biologically-constrained deep learning.*