import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import click
from time import time
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def seed_worker(worker_id):
    """Called in each DataLoader worker process"""
    import torch  # Import here too for worker processes
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)

def run_single_seed(seed_idx, network_config_file_name, data_dir, num_gpus, debug=False):
    """
    Run training for a single seed on the assigned GPU.
    
    Parameters
    ----------
    seed_idx : int
        Index of the seed (0-based)
    network_config_file_name : str
        Name of the network config file
    data_dir : str
        Path to data directory
    num_gpus : int
        Total number of GPUs available
    debug : bool
        Whether to print debug messages
        
    Returns
    -------
    dict
        Results dictionary for this seed
    """
    # Calculate which GPU this seed should use (0-based indexing)
    gpu_id = seed_idx % num_gpus
    
    # Set CUDA_VISIBLE_DEVICES to only see the assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Import torch and EIANN.utils ONLY after setting CUDA_VISIBLE_DEVICES
    import torch
    import EIANN.utils as ut
    
    # Determinism settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    
    # Calculate seeds (starting from your specified values)
    network_seed = 66049 + seed_idx
    data_seed = 257 + seed_idx
    
    # Device - use 'cuda' like in the working implementation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Seed {seed_idx}: GPU {gpu_id}, NetSeed {network_seed}, DataSeed {data_seed}, Device {device}")
    
    start_time = time()
    
    try:
        # Seed with network_seed first (like working implementation)
        ut.set_all_seeds(seed=network_seed)
        
        # Load dataset
        train_dataloader, val_dataloader, test_dataloader, data_generator = \
            ut.get_MNIST_dataloaders(data_dir=data_dir)
        
        if debug:
            print(f"Seed {seed_idx}: Loaded Data")
        
        # Build model
        config_file_path = f"EIANN/network_config/mnist/{network_config_file_name}"
        network = ut.build_EIANN_from_config(config_file_path,
                                             network_seed=network_seed,
                                             device=device)
        
        if debug:
            print(f"Seed {seed_idx}: Built Network on {device}")
        
        # Seed data generator before training
        data_generator.manual_seed(data_seed)
        
        # Train
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
            'seed_idx': seed_idx,
            'gpu_id': gpu_id,
            'network_seed': network_seed,
            'data_seed': data_seed,
            'device': str(device),
            'config': network_config_file_name,
            'val_accuracy': network.val_accuracy_history[-1],
            'val_loss': network.val_loss_history[-1],
            'run_time': run_time,
        }
        
        print(f"\nSeed {seed_idx} Results:")
        print(f"  GPU: {gpu_id}")
        print(f"  Device: {device}")
        print(f"  Final Val Accuracy: {result['val_accuracy']:.4f}")
        print(f"  Final Val Loss: {result['val_loss']:.6f}")
        print(f"  Run Time: {run_time:.2f} sec\n")
        
        # Clean up memory
        del network
        del train_dataloader, val_dataloader, test_dataloader
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        import traceback
        print(f"Error in seed {seed_idx}: {str(e)}")
        print(traceback.format_exc())
        return {
            'seed_idx': seed_idx,
            'gpu_id': gpu_id,
            'network_seed': network_seed,
            'data_seed': data_seed,
            'error': str(e)
        }

@click.command()
@click.option('--network-config-file-name', required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../data/mnist')
@click.option('--num-seeds', default=5, type=int, help="Number of different seeds to try")
@click.option('--num-gpus', default=2, type=int, help="Number of GPUs to use")
@click.option('--debug', default=False, is_flag=True)
def main(network_config_file_name, data_dir, num_seeds, num_gpus, debug):
    """
    Run multiple seeds in parallel across multiple GPUs.
    """
    # Import torch here just to check GPU availability in main process
    import torch
    
    print(f"Starting parallel execution: {num_seeds} seeds across {num_gpus} GPUs")
    print(f"Network config: {network_config_file_name}")
    print(f"Data directory: {data_dir}")
    
    # Verify GPU availability
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available")
        num_gpus = available_gpus
    
    print(f"Using {num_gpus} GPUs")
    
    # Print seed assignment
    print("\nSeed to GPU assignment:")
    for i in range(num_seeds):
        gpu_id = i % num_gpus
        network_seed = 66049 + i
        data_seed = 257 + i
        print(f"  Seed {i}: GPU {gpu_id}, NetSeed {network_seed}, DataSeed {data_seed}")
    
    # Run seeds in parallel
    results = []
    
    # Use ProcessPoolExecutor to run seeds in parallel
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Submit all jobs
        future_to_seed = {
            executor.submit(run_single_seed, seed_idx, network_config_file_name, 
                          data_dir, num_gpus, debug): seed_idx 
            for seed_idx in range(num_seeds)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_seed):
            seed_idx = future_to_seed[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed seed {seed_idx}")
            except Exception as exc:
                print(f"Seed {seed_idx} generated an exception: {exc}")
                results.append({
                    'seed_idx': seed_idx,
                    'error': str(exc)
                })
    
    # Sort results by seed index
    results.sort(key=lambda x: x['seed_idx'])
    
    # Print summary
    successful_results = [r for r in results if 'error' not in r]
    
    if successful_results:
        print("\n" + "="*60)
        print("SUMMARY OF ALL RUNS")
        print("="*60)
        
        val_accuracies = [r['val_accuracy'] for r in successful_results]
        val_losses = [r['val_loss'] for r in successful_results]
        run_times = [r['run_time'] for r in successful_results]
        
        print(f"\nNetwork: {network_config_file_name}")
        print(f"Number of successful seeds: {len(successful_results)}")
        print(f"Number of failed seeds: {len(results) - len(successful_results)}")
        
        print(f"\nValidation Accuracy:")
        print(f"  Mean: {sum(val_accuracies)/len(val_accuracies):.4f}")
        print(f"  Std:  {torch.tensor(val_accuracies).std().item():.4f}")
        print(f"  Min:  {min(val_accuracies):.4f} (Seed: {successful_results[val_accuracies.index(min(val_accuracies))]['network_seed']})")
        print(f"  Max:  {max(val_accuracies):.4f} (Seed: {successful_results[val_accuracies.index(max(val_accuracies))]['network_seed']})")
        
        print(f"\nValidation Loss:")
        print(f"  Mean: {sum(val_losses)/len(val_losses):.6f}")
        print(f"  Std:  {torch.tensor(val_losses).std().item():.6f}")
        
        print(f"\nRun Time:")
        print(f"  Mean: {sum(run_times)/len(run_times):.2f} sec")
        print(f"  Total (wall time): {max(run_times):.2f} sec")
        
        print("\nIndividual Results:")
        for r in successful_results:
            print(f"  Seed {r['seed_idx']:2d} (Net {r['network_seed']:6d}, Data {r['data_seed']:4d}): "
                  f"GPU {r['gpu_id']}, Acc={r['val_accuracy']:.4f}, Loss={r['val_loss']:.6f}, Time={r['run_time']:.2f}s")
    
    # Print any errors
    failed_results = [r for r in results if 'error' in r]
    if failed_results:
        print("\nFailed runs:")
        for r in failed_results:
            print(f"  Seed {r['seed_idx']}: {r['error']}")

if __name__ == '__main__':
    main()