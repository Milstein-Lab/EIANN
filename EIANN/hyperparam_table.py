import re
import click
import os
import openpyxl

import EIANN.utils as ut


def flatten_projection_config(projection_config):
    """
    Traverse the projection_config dict and extract weight_init_args, learning_rate, and theta_tau.
    Returns a dict mapping flattened keys to their values.
    """
    hyperparams = {}

    def recurse(node, path):
        for key, val in node.items():
            if isinstance(val, dict):
                if 'weight_init_args' in val or 'learning_rule_kwargs' in val:
                    path_str = "".join(path + [key])
                    # init scale
                    if 'weight_init_args' in val:
                        hyperparams[f"{path_str} Init Scale"] = val['weight_init_args']
                    # learning rate
                    lr = val.get('learning_rule_kwargs', {}).get('learning_rate')
                    if lr is not None:
                        hyperparams[f"{path_str} η"] = lr
                    # theta_tau
                    theta = val.get('learning_rule_kwargs', {}).get('theta_tau')
                    if theta is not None:
                        hyperparams[f"{path_str} θτ"] = theta
                    # temporal_discount
                    td = val.get('learning_rule_kwargs', {}).get('temporal_discount')
                    if td is not None:
                        hyperparams[f"{path_str} Temporal Discount"] = td
                    # clone weight scale
                    if val.get('weight_constraint') == 'clone_weight':
                        clone_scale = val.get('weight_constraint_kwargs', {}).get('scale')
                        if clone_scale is not None:
                            hyperparams[f"{path_str} Clone Weight Scale"] = clone_scale
                    # normalized weight scale
                    if val.get('weight_constraint') == 'normalize_weight':
                        norm_scale = val.get('weight_constraint_kwargs', {}).get('scale')
                        if norm_scale is not None:
                            hyperparams[f"{path_str} Normalized Weight Scale"] = norm_scale
                recurse(val, path + [key])

    recurse(projection_config, [])
    return hyperparams


def shorten_label(name):
    """
    Map full hyperparam key to abbreviated form using W, B, Q, Y, R rules.
    """
    parts = name.split(' ', 1)
    if len(parts) != 2:
        return name
    prefix, suffix = parts
    # regex for dest/source
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
            return int(layer[1:])
        return float('nan')

    # determine code & details
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
    Create an Excel workbook for given model hyperparameters dict.
    Adds columns: abbr, hyperparameter, and one per model label.
    """
    # collect all names
    all_names = sorted({name for params in model_hyperparams.values() for name in params})
    labels = list(model_hyperparams)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['abbr', 'hyperparameter'] + labels)

    for name in all_names:
        abbr = shorten_label(name)
        row = [abbr, name]
        for label in labels:
            val = model_hyperparams[label].get(name, '-')
            if isinstance(val, (list, tuple)):
                val = val[0] if val else '-'
            if val != '-':
                try:
                    val = float(val)
                except:
                    pass
            row.append(val)
        ws.append(row)

    wb.save(output_path)


@click.command()
@click.option('--models_path', type=click.Path(exists=True), required=True, help="Path to the models specification YAML file")
@click.option('--out', type=click.Path(), required=True, help="Base output path (without extension) for the XLSX files")
def main(models_path, out):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # read spec
    model_specs = ut.read_from_yaml(models_path)
    mnist_params = {}
    spiral_params = {}

    for model_key, spec in model_specs.items():
        config_file = spec.get('config', '').lower()
        # determine which group
        target = spiral_params if 'spiral' in config_file else mnist_params

        # build path to config YAML
        base = os.path.join(project_root, 'EIANN', 'network_config')
        if 'mnist' in config_file and 'spiral' not in config_file:
            base = os.path.join(base, 'mnist')
        elif 'spiral' in config_file:
            base = os.path.join(base, 'spiral')
        cfg_path = os.path.join(base, spec['config'])

        label = spec.get('label', model_key)
        model_cfg = ut.read_from_yaml(cfg_path)
        proj_cfg = model_cfg.get('projection_config', {})
        target[label] = flatten_projection_config(proj_cfg)

    # generate two excels
    mnist_out = f"{out}_mnist.xlsx"
    spiral_out = f"{out}_spiral.xlsx"
    if mnist_params:
        generate_excel(mnist_params, mnist_out)
        print(f"Saved MNIST table to {mnist_out}")
    if spiral_params:
        generate_excel(spiral_params, spiral_out)
        print(f"Saved Spiral table to {spiral_out}")

if __name__ == "__main__":
    main()


# Run with:
# python hyperparam_table.py --models_path=figure_model_specs.yaml --out=figures/hyperparameters_table