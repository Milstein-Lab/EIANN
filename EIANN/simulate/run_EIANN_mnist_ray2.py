import os
import time
import torch
import random
import numpy as np
import ray
import click

import EIANN.utils as ut

@ray.remote(num_gpus=0.5, num_cpus=1)
def run_seed(seed_idx, network_config_file_name, data_dir):
    network_seed = 66049 + seed_idx
    data_seed = 257 + seed_idx

    ut.set_all_seeds(seed=network_seed)
    torch.cuda.manual_seed_all(network_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    train_loader, val_loader, test_loader, _ = ut.get_MNIST_dataloaders(data_dir=data_dir, data_seed=data_seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    config_file_path = os.path.join(project_root, "EIANN", "network_config", "mnist", network_config_file_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ut.build_EIANN_from_config(config_file_path, network_seed=network_seed, device=device)

    start = time.time()
    net.train(
        train_loader, 
        val_loader, 
        epochs=1, 
        samples_per_epoch=20_000, 
        val_interval=(0, -1, 100), 
        store_history=False
    )

    val_acc = net.val_accuracy_history[-1]
    val_loss = net.val_loss_history[-1]
    return {
        "seed_idx": seed_idx,
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "run_time": time.time() - start,
        "network_seed": network_seed,
        "data_seed": data_seed,
    }

@click.command()
@click.option('--network-config-file-name', required=True, type=str, help="Network config file name")
@click.option('--data-dir', default="../data/mnist", type=str, help="Directory for MNIST data")
@click.option('--num-seeds', default=5, type=int, help="Number of different seeds to try")
def main(network_config_file_name, data_dir, num_seeds):
    overall_start_time = time.time()

    ray.init()
    seeds = list(range(num_seeds))
    handles = [run_seed.remote(s, network_config_file_name, data_dir) for s in seeds]
    # Asynchronous collection
    pending = handles[:]
    results = []
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        out = ray.get(ready[0])
        print("Finished seed", out["seed_idx"], "acc", out["val_accuracy"])
        results.append(out)

    overall_end_time = time.time()
    print(f"Overall time for {num_seeds} seeds: {overall_end_time - overall_start_time:.2f} seconds")

    print("Summary:")
    for res in results:
        print(f"Seed {res['seed_idx']}: Val Acc = {res['val_accuracy']}, Val Loss = {res['val_loss']}, Time = {res['run_time']:.2f} seconds")

if __name__ == "__main__":
    main()

# bp Dale:
# 36192337: 1 cpu 0.2 gpu per seed, request 6 cpus (GPU-shared)
#   436.63 seconds 

# 36192343: 1 cpu 0.5 gpu per seed, request 6 cpus (GPU-shared)
#   389.19 seconds