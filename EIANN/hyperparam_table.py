import csv
import click
import os
import re

import EIANN.utils as ut


def flatten_projection_config(projection_config):
    """
    Traverse the projection_config dict and extract weight_init_args and learning_rate from each layer.
    """
    hyperparams = {}

    def recurse(node, path):
        for key, val in node.items():
            if isinstance(val, dict):
                # Detect hyperparameter leaf nodes
                if 'weight_init_args' in val or 'learning_rule_kwargs' in val:
                    path_str = "".join(path + [key])
                    # weight_init_args
                    if 'weight_init_args' in val:
                        hyperparams[f"{path_str} Init Scale"] = val['weight_init_args']
                    # learning_rate
                    lr = val.get('learning_rule_kwargs', {}).get('learning_rate')
                    if lr is not None:
                        hyperparams[f"{path_str} η"] = lr
                    # theta_tau
                    theta = val.get('learning_rule_kwargs', {}).get('theta_tau')
                    if theta is not None:
                        hyperparams[f"{path_str} θΤ"] = theta
                # Continue recursion
                recurse(val, path + [key])

    recurse(projection_config, [])
    
    return hyperparams


def generate_csv(model_hyperparams, output_path):
    """
    Given a mapping of model label -> {hyperparam_name: value}, write out a CSV
    where each column is a model label and each row a hyperparameter.
    """
    # Collect all hyperparam names
    all_names = set()
    for params in model_hyperparams.values():
        all_names.update(params.keys())
    all_names = sorted(all_names)

    labels = list(model_hyperparams.keys())

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hyperparameter'] + labels)
        for name in all_names:
            row = [name]
            for label in labels:
                val = model_hyperparams[label].get(name, '-')
                if isinstance(val, (list, tuple)):
                    val = str(val)
                    if re.match(r'\(\d+\.?\d*,\)', val):
                        val = re.sub(r'\((\d+\.?\d*),\)', r'\1', val)
                row.append(float(val) if val != '-' else val)
            writer.writerow(row)


@click.command()
@click.option('--models_path', type=click.Path(exists=True), required=True, help="Path to the models specification YAML file")
@click.option('--out', type=click.Path(), required=True, help="Output path for the CSV file")
def main(models_path, out):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, out)

    # Read the spec of models
    model_specs = ut.read_from_yaml(models_path)
    model_hyperparams = {}

    for model_key, spec in model_specs.items():

        config_file = spec.get('config')
        if config_file:
            config_file_path = os.path.join(project_root, 'EIANN', 'network_config')
            
            if 'mnist' in config_file.lower():
                config_file_path = os.path.join(config_file_path, 'mnist')
            elif 'spiral' in config_file.lower():
                config_file_path = os.path.join(config_file_path, 'spiral')
            
            config_file_path = os.path.join(config_file_path, config_file)

        # Use the label if provided, otherwise fallback to model_key
        label = spec.get('label', model_key)
        # Read each model's config YAML
        model_cfg = ut.read_from_yaml(config_file_path)
        proj_cfg = model_cfg.get('projection_config', {})
        # Flatten out hyperparameters
        hyperparams = flatten_projection_config(proj_cfg)
        model_hyperparams[label] = hyperparams

    # Generate the CSV
    generate_csv(model_hyperparams, out)
    print(f"Saved hyperparameters CSV to {out}")


if __name__ == "__main__":
    main()

# python hyperparam_table.py --models_path=figure_model_specs.yaml --out=figures/hyperparameters_table.csv

# TODO replace all column names with W, B, Q, R