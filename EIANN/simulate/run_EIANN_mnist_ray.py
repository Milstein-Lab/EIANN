import os
import time
import torch
import random
import numpy as np
import click
import ray
import traceback
from ray import tune
from ray.air import session
from ray.air.config import RunConfig

import EIANN.utils as ut

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def train_eiann(config):
    """
    Ray Tune trainable function.
    Each trial corresponds to one (seed_idx, config_file) pair.
    """

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.current_device():", torch.cuda.current_device() if torch.cuda.is_available() else None)

    seed_idx = config["seed_idx"]
    network_config_file_name = config["network_config_file_name"]
    data_dir = config["data_dir"]

    # Each trial gets a fractional GPU (e.g. 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute which GPU Ray gave us (optional)
    gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else None

    # Seeds
    network_seed = 66049 + seed_idx
    data_seed = 257 + seed_idx

    # Deterministic setup
    torch.cuda.manual_seed_all(network_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    start_time = time.time()

    try:
        ut.set_all_seeds(seed=network_seed)

        train_loader, val_loader, test_loader, _ = ut.get_MNIST_dataloaders(data_dir=data_dir, data_seed=data_seed)

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels: simulate -> EIANN -> project root
        project_root = os.path.dirname(os.path.dirname(script_dir))
        config_file_path = os.path.join(project_root, "EIANN", "network_config", "mnist", network_config_file_name)
        
        network = ut.build_EIANN_from_config(config_file_path, network_seed=network_seed, device=device)

        network.train(
            train_loader,
            val_loader,
            epochs=1,
            samples_per_epoch=20_000,
            val_interval=(0, -1, 100),
            store_history=False
        )

        val_acc = network.val_accuracy_history[-1]
        val_loss = network.val_loss_history[-1]

        result = {
            "val_accuracy": float(val_acc) if torch.is_tensor(val_acc) else val_acc,
            "val_loss": float(val_loss) if torch.is_tensor(val_loss) else val_loss,
            "run_time": time.time() - start_time,
            "gpu_id": gpu_id,
            "seed_idx": seed_idx,
            "network_seed": network_seed,
            "data_seed": data_seed,
        }

        session.report(result)  # Send metrics to Ray Tune
        torch.cuda.empty_cache()

    except Exception as e:
        traceback.print_exc()
        session.report({"error": str(e)})


@click.command()
@click.option('--network-config-file-name', required=True, type=str, help="Network config file name")
@click.option('--data-dir', default="../data/mnist", type=str, help="Directory for MNIST data")
@click.option('--num-seeds', default=5, type=int, help="Number of different seeds to try")
def main(network_config_file_name, data_dir, num_seeds):

    overall_start_time = time.time()

    ray.init()

    # Define parameter space (each trial = one seed)
    param_space = [
        {
            "seed_idx": i,
            "network_config_file_name": network_config_file_name,
            "data_dir": data_dir
        }
        for i in range(num_seeds)
    ]

    tuner = tune.Tuner(
        tune.with_resources(train_eiann, resources={"cpu": 0, "gpu": 0.5}),
        param_space=tune.grid_search(param_space),
        run_config=RunConfig(name="eiann_mnist_parallel_ray")
    )

    results = tuner.fit()

    print(network_config_file_name)

    overall_end_time = time.time()
    print(f"Overall time for {num_seeds} seeds: {overall_end_time - overall_start_time:.2f} seconds")

    print("==== Summary ====")
    for res in results:
        print(res)

if __name__ == "__main__":
    main()

# interact -p GPU-shared -N 1 --gres=gpu:v100-32:3 -t 01:00:00


# ===== Single-node GPU runs =====

# bp Dale: 
# 36191138: 1 cpu 0.5 gpu per seed, request 12 cpus (GPU-shared)
#   346.74 seconds

# 36191154: 1 cpu 0.5 gpu per seed, request 6 cpus (GPU-shared)
#   382.83 seconds

# 36191200: 2 cpu 0.5 gpu per seed, request 12 cpus (GPU-shared)
#   370.17 seconds

# 36191244: 1 cpu 0.25 gpu per seed, request 12 cpus (GPU-shared)
#   397.13 seconds

# 36191248: 1 cpu 1 gpu per seed, request 12 cpus (GPU-shared)
#   688.59 seconds -> did not request enough gpus -> retried with 36192557

# 36192288: 1 cpu 0.5 gpu per seed (12 seeds), request 12 cpus (GPU-shared)
#   722.51 seconds -> sequential

# 36192294: 0 cpu 0.5 gpu per seed, request 12 cpus (GPU-shared)
#   360.29 seconds

# 36192324: 1 cpu 0.2 gpu per seed, request 6 cpus (GPU-shared)
#   error

# 36192511: 0 cpu 0.5 gpu per seed, request 15 cpus (GPU-shared)
#   356.43 seconds

# 36192512: 0 cpu 0.5 gpu per seed (12 seeds), request 15 cpus (GPU-shared)
#   733.16 seconds

# 36192557: 1 cpu 1 gpu per seed, request 15 cpus 5 gpus (GPU)
#   error for some

# ===== Multi-node GPU runs =====

# bp Dale:
# 36193643: 2 GPU nodes, 0 cpu 0.5 gpu per seed (32 seeds), request 15 cpus 16 gpus (GPU), with ray head only
#   770.49 seconds -> only used 8 gpus

# 36193950: 2 GPU nodes, 0 cpu 0.5 gpu per seed (32 seeds), request 8,8 cpus 8,7 gpus (GPU), with ray head+worker
#   384.80 seconds