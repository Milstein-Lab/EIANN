import os
from mpi4py import MPI

# Get MPI rank early so we can set GPU env before importing torch
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------
# GPU pinning: ensure each MPI rank sees exactly one GPU
# (must run BEFORE importing torch)
# ---------------------------

# If the scheduler already sets CUDA_VISIBLE_DEVICES, it may list multiple devices like "0,1".
# We want each MPI rank to see exactly one device. So parse the list (if present) and pick one id.
visible = os.environ.get("CUDA_VISIBLE_DEVICES")

if visible:
    # Split comma-separated list, keep as strings
    visible_list = [v.strip() for v in visible.split(",") if v.strip() != ""]
    try:
        n_gpus_available = len(visible_list)
    except Exception:
        n_gpus_available = 0
else:
    # Try nvidia-smi fallback
    out = os.popen("nvidia-smi -L 2>/dev/null").read().strip()
    if out:
        n_gpus_available = out.count("\n") + 1 if "\n" in out else 1
        # construct a list of indices as strings: "0","1",...
        visible_list = [str(i) for i in range(n_gpus_available)]
    else:
        n_gpus_available = 0
        visible_list = []

if n_gpus_available <= 0:
    # no GPUs detected; leave CUDA_VISIBLE_DEVICES unset so torch uses CPU
    print(f"[Rank {rank}] No GPUs detected; running on CPU.")
else:
    # Choose the gpu index (physical id from visible_list) for this rank
    chosen_index = rank % n_gpus_available
    chosen_gpu = visible_list[chosen_index]
    # Override CUDA_VISIBLE_DEVICES so this process sees *only* that GPU -> avoid resource contention
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen_gpu
    print(f"[Rank {rank}] Setting CUDA_VISIBLE_DEVICES={chosen_gpu} (selected from {visible_list})")

# Import torch after CUDA_VISIBLE_DEVICES is set
import torch
print(f"[Rank {rank}] torch sees {torch.cuda.device_count()} GPU(s). Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

import click
from time import time
import random
import numpy as np

import EIANN.utils as ut

# Determinism settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# enforce deterministic algorithms (may raise error if non-deterministic op used)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    # older torch may not have this function
    print("Warning: torch.use_deterministic_algorithms not available, proceeding without it.")
    pass

@click.command()
@click.option('--network-config-file-name', required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data/mnist')
@click.option('--num-seeds', default=5, type=int, help="Number of different seeds to try")
@click.option('--debug', default=False, is_flag=True)
def main(network_config_file_name, data_dir, num_seeds, debug):
    print('Using MPI')
    # Use global comm/rank/size from above
    global comm, rank, size

    # Track wall clock time per rank
    wall_start = time()

    # Each MPI rank handles multiple seeds sequentially.
    seeds_per_rank = (num_seeds + size - 1) // size  # Ceiling division
    start_idx = rank * seeds_per_rank
    end_idx = min((rank + 1) * seeds_per_rank, num_seeds)

    print(f"[Rank {rank}/{size}] visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}, processing seeds {start_idx}..{end_idx-1}")

    results_from_this_rank = []
    for i in range(start_idx, end_idx):
        network_seed = 66049 + i
        data_seed = 257 + i

        # Device selection: after setting CUDA_VISIBLE_DEVICES above, device is cuda:0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Rank {rank}, Iter {i}: Device {device}, NetSeed {network_seed}, DataSeed {data_seed}")

        start_time = time()

        # Seed everything in deterministic way
        ut.set_all_seeds(seed=network_seed)
        torch.manual_seed(network_seed)
        np.random.seed(network_seed)
        random.seed(network_seed)

        # Create and seed a generator for the dataloader for this specific run
        data_generator = torch.Generator()
        data_generator.manual_seed(data_seed)

        # Load dataset with deterministic worker_init_fn and num_workers=0 or set worker_init_fn
        train_dataloader, val_dataloader, test_dataloader, data_generator = \
            ut.get_MNIST_dataloaders(data_dir=data_dir, data_seed=data_seed)

        if debug:
            print(f"Rank {rank}, Seed {network_seed}: Loaded Data")

        # Build model (pass device)
        config_file_path = f"EIANN/network_config/mnist/{network_config_file_name}"
        network = ut.build_EIANN_from_config(config_file_path,
                                             network_seed=network_seed,
                                             device=device)
        if debug:
            print(f"Rank {rank}, Seed {network_seed}: Built Network on {device}")

        # Train
        # data_generator.manual_seed(data_seed)
        network.train(train_dataloader, val_dataloader,
                      epochs=1,
                      samples_per_epoch=20_000,
                      val_interval=(0, -1, 100),
                      store_history=False,
                      store_history_interval=(0, -1, 100),
                      store_dynamics=False,
                      store_params=False,
                      status_bar=False)

        run_time = time() - start_time

        result = {
            'rank': rank,
            'seed': network_seed,
            'data_seed': data_seed,
            'device': str(device),
            'config': network_config_file_name,
            'val_accuracy': network.val_accuracy_history[-1],
            'val_loss': network.val_loss_history[-1],
            'run_time': run_time,
        }
        results_from_this_rank.append(result)

        print(f"\nRank {rank}, Seed {network_seed} Results:")
        print(f"  Device: {device}")
        print(f"  Final Val Accuracy: {result['val_accuracy']:.4f}")
        print(f"  Final Val Loss: {result['val_loss']:.6f}")
        print(f"  Run Time: {run_time:.2f} sec\n")

        del network
        torch.cuda.empty_cache()

    # rank-local wall time
    wall_end = time()
    rank_wall = wall_end - wall_start

    # Gather and print summary on rank 0
    all_results = comm.gather(results_from_this_rank, root=0)
    all_wall_times = comm.gather(rank_wall, root=0)

    if rank == 0:
        flattened_results = []
        for rank_results in all_results:
            flattened_results.extend(rank_results)
        flattened_results.sort(key=lambda x: x['seed'])

        print("\n" + "="*60)
        print("SUMMARY OF ALL RUNS")
        print("="*60)
        val_accuracies = [r['val_accuracy'] for r in flattened_results]
        val_losses = [r['val_loss'] for r in flattened_results]
        run_times = [r['run_time'] for r in flattened_results]

        print(f"\nNetwork: {network_config_file_name}")
        print(f"Number of seeds tested: {len(flattened_results)}")
        print(f"\nValidation Accuracy:")
        print(f"  Mean: {sum(val_accuracies)/len(val_accuracies):.4f}")
        print(f"  Std:  {torch.tensor(val_accuracies).std().item():.4f}")
        print(f"  Min:  {min(val_accuracies):.4f} (Seed: {flattened_results[val_accuracies.index(min(val_accuracies))]['seed']})")
        print(f"  Max:  {max(val_accuracies):.4f} (Seed: {flattened_results[val_accuracies.index(max(val_accuracies))]['seed']})")

        print(f"\nValidation Loss:")
        print(f"  Mean: {sum(val_losses)/len(val_losses):.6f}")
        print(f"  Std:  {torch.tensor(val_losses).std().item():.6f}")

        print(f"\nRun Time:")
        print(f"  Mean: {sum(run_times)/len(run_times):.2f} sec")
        print(f"  Total: {sum(run_times):.2f} sec")

        wall_clock = max(all_wall_times)  # longest rank determines MPI job wall time
        print(f"\nTotal wall-clock time: {wall_clock:.2f} sec")

        print("\nIndividual Results:")
        for r in flattened_results:
            print(f"  Net Seed {r['seed']:6d}, Data Seed {r['data_seed']:4d}: Acc={r['val_accuracy']:.4f}, Loss={r['val_loss']:.6f}, Time={r['run_time']:.2f}s")

if __name__ == '__main__':
    main()
