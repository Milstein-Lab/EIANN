# YAML Config Glossary

This glossary describes all the optional parameters that can be specified in the YAML configuration file when building a neural network using the EIANN library.


## 1. Layer Config

Defines the structure and behavior of neuron populations in each layer.

### Structure
```yaml
layer_config:
  LayerName:
    PopulationName:
      size: int
      activation: str
      activation_kwargs: dict
      include_bias: bool
      bias_init: str
      bias_init_args: tuple
      bias_bounds: [float, float]
      bias_learning_rule: str
      bias_learning_rule_kwargs: dict
      custom_update: str
      custom_update_kwargs: dict
      output_pop: bool
```

### Parameters
- `size`: Number of neurons in the population.
- `activation`: Name of the nonlinearity function (e.g. "relu", "sigmoid").
- `activation_kwargs`: Keyword arguments passed to the activation function.
- `include_bias`: Whether to include a trainable bias term.
- `bias_init`: Callable used to initialize the bias.
- `bias_init_args`: Arguments passed to the bias initializer.
- `bias_bounds`: Tuple of min/max values for clipping the bias.
- `bias_learning_rule`: Callable name for the learning rule applied to the bias.
- `bias_learning_rule_kwargs`: Additional parameters for the bias learning rule.
- `output_pop`: Marks this population as the output target for loss calculation.


## 2. Projection Config

Defines all projections (i.e., weighted connections) between neuron populations.

### Structure
```yaml
projection_config:

  PostLayer.PostPopulation:
    PreLayer.PrePopulation:
      weight_init: str
      weight_init_args: tuple
      weight_constraint: str
      weight_constraint_kwargs: dict
      weight_bounds: [float, float]
      direction: str
      update_phase: str
      compartment: str
      learning_rule: str
      learning_rule_kwargs: dict
```

### Parameters
- `weight_init`: Name of the weight initializer function (e.g. "half_kaiming").
- `weight_init_args`: Arguments passed to the initializer. If a single number is provided, it is used as scaling factor when initializing weights.
- `weight_constraint`: Optional constraint on the weight matrix. Options:
  [`normalize_weight`](https://milstein-lab.github.io/EIANN/autoapi/EIANN/learning_rules/weight_functions/index.html#EIANN.rules.weight_functions.normalize_weight),
  [`clone_weight`](https://milstein-lab.github.io/EIANN/autoapi/EIANN/learning_rules/weight_functions/index.html#EIANN.rules.weight_functions.clone_weight),
  [`no_autapses`](https://milstein-lab.github.io/EIANN/autoapi/EIANN/learning_rules/weight_functions/index.html#EIANN.rules.weight_functions.no_autapses)
- `weight_constraint_kwargs`: Parameters for the weight constraint.
- `weight_bounds`: Tuple of [min, max] values to clip weights. "Null" means no clipping.
- `direction`: Forward or Recurrent connection type. Recurrent means the connection uses activities from the previous time step. Options: 'forward','F', 'recurrent','R'
- `update_phase`: Informs the learning rule to determine which phase of activations to use for weight updates. Options: 'forward' (default option),'F', 'backward','B', 'all','A'
- `compartment`: Specifies target compartment ('soma' or 'dendrite').
- `learning_rule`: Name of the function used to update weights (e.g. "Backprop", "BCM", "Hebb_WeightNorm").
- `learning_rule_kwargs`: Keyword arguments passed to the learning rule.


## 3. Training kwargs

Global settings for running the training loop and initializing the network.

### Structure
```yaml
training_kwargs:
  learning_rate: float
  optimizer: str
  optimizer_kwargs: dict
  criterion: str
  criterion_kwargs: dict
  seed: int
  device: str
  tau: int
  forward_steps: int
  backward_steps: int
  verbose: bool
```

### Parameters
- `learning_rate`: Global fallback learning rate for all trainable parameters.
- `optimizer`: Name of optimizer class (default: SGD).
- `optimizer_kwargs`: Arguments passed to the optimizer (e.g. momentum).
- `criterion`: Loss function name (e.g. "MSELoss").
- `criterion_kwargs`: Additional arguments passed to the loss function.
- `seed`: Random seed for reproducibility.
- `device`: Hardware device ("cpu" or "cuda"). [Note: This functionality has not been fully tested for "cuda"]
- `tau`: Decay time constant for unit leak dynamics. tau=1 represents complete decay in each step (i.e. no persistent state)
- `forward_steps`: Number of simulation steps per forward pass.
- `backward_steps`: Number of steps for backprop or backward phase (if applicable). In the case of Backprop, results in truncated BPTT.
- `verbose`: Enables logging or debug outputs during training.