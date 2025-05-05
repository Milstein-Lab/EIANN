import re
import csv
import click
import os
import openpyxl

import EIANN.utils as ut


def flatten_projection_config(projection_config):
    """
    Traverse the projection_config dict and extract weight_init_args, learning_rate, and theta_tau from each layer.
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
                        hyperparams[f"{path_str} θτ"] = theta
                # Continue recursion
                recurse(val, path + [key])

    recurse(projection_config, [])
    return hyperparams


def shorten_label(name):
    """
    Convert a full hyperparameter key like 'H1EInputE η' into a short abbreviation
    using W, B, Q, Y, R rules plus the unit prefix.
    """
    parts = name.split(' ', 1)
    if len(parts) != 2:
        return name
    prefix, suffix = parts
    # Match destination layer & cell, then source layer & cell
    m = re.match(r'^(Input|Output|H\d+)(E|DendI|SomaI)(Input|Output|H\d+)(E|DendI|SomaI)$', prefix)
    if not m:
        return name
    dst_layer, dst_cell, src_layer, src_cell = m.groups()
    non_excit = ('DendI', 'SomaI')

    def layer_index(layer):
        if layer == 'Input':
            return 0
        if layer == 'Output':
            return float('inf')
        if layer.startswith('H'):
            try:
                return int(layer[1:])
            except ValueError:
                return float('nan')
        return float('nan')

    # Determine connection type
    if dst_cell == 'E' and src_cell == 'E':
        # Excitatory-to-excitatory: forward or backward
        if layer_index(dst_layer) > layer_index(src_layer):
            code = 'W'
        else:
            code = 'B'
        details = f'({dst_layer})'
    elif dst_cell in non_excit and src_cell == 'E':
        # Non-excitatory receiving from excitatory
        code = 'Q'
        details = f'({dst_cell}, {dst_layer})'
    elif dst_cell == 'E' and src_cell in non_excit:
        # Excitatory receiving from non-excitatory
        code = 'Y'
        details = f'({src_cell}, {src_layer})'
    elif dst_cell in non_excit and src_cell in non_excit and dst_layer == src_layer:
        # Recurrent non-excitatory
        code = 'R'
        details = f'({dst_cell}, {dst_layer})'
    else:
        return name

    # Normalize unit label to lowercase (keeps Greek letters intact)
    unit = suffix.lower()
    return f"{unit}, {code} {details}"


def generate_excel(model_hyperparams, output_path):
    """
    Generate an Excel file with columns for each model and rows for each hyperparameter.
    Adds an 'abbr' column for the shortened labels.
    """
    # Collect all hyperparam names
    all_names = set()
    for params in model_hyperparams.values():
        all_names.update(params.keys())
    all_names = sorted(all_names)

    labels = list(model_hyperparams.keys())
    
    wb = openpyxl.Workbook()
    ws = wb.active
    # Header with abbreviation column
    ws.append(['abbr', 'hyperparameter'] + labels)

    for name in all_names:
        abbr = shorten_label(name)
        row = [abbr, name]
        for label in labels:
            val = model_hyperparams[label].get(name, '-')
            if isinstance(val, (list, tuple)):
                # take the single scale value for init scale
                val = val[0] if val else '-'
            # convert numeric strings to floats where appropriate
            if val != '-':
                try:
                    val = float(val)
                except Exception:
                    pass
            row.append(val)
        ws.append(row)

    wb.save(output_path)


@click.command()
@click.option('--models_path', type=click.Path(exists=True), required=True, help="Path to the models specification YAML file")
@click.option('--out', type=click.Path(), required=True, help="Output path for the XLSX file")
def main(models_path, out):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'EIANN', out)

    model_specs = ut.read_from_yaml(models_path)
    model_hyperparams = {}

    for model_key, spec in model_specs.items():
        config_file = spec.get('config')
        if config_file:
            base = os.path.join(project_root, 'EIANN', 'network_config')
            if 'mnist' in config_file.lower():
                base = os.path.join(base, 'mnist')
            elif 'spiral' in config_file.lower():
                base = os.path.join(base, 'spiral')
            config_file = os.path.join(base, config_file)

        label = spec.get('label', model_key)
        model_cfg = ut.read_from_yaml(config_file)
        proj_cfg = model_cfg.get('projection_config', {})
        hyperparams = flatten_projection_config(proj_cfg)
        model_hyperparams[label] = hyperparams

    generate_excel(model_hyperparams, output_path)
    print(f"Saved hyperparameters XLSX to {output_path}")


if __name__ == "__main__":
    main()

# python hyperparam_table.py --models_path=figure_model_specs.yaml --out=figures/hyperparameters_table.xlsx

# normalized weight scale, clone weight scale, temporal discount