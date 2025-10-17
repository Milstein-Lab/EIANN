import click
import torch
from time import time
from mpi4py import MPI
import EIANN.utils as ut

@click.command()
@click.option('--network-config-file-name', required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), 
              default='../data/mnist')
@click.option('--num-seeds', default=5, type=int, help="Number of different seeds to try")
@click.option('--debug', default=False, is_flag=True)
def main(network_config_file_name, data_dir, num_seeds, debug):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Calculate which seeds this rank will process
    seeds_per_rank = (num_seeds + size - 1) // size  # Ceiling division
    start_idx = rank * seeds_per_rank
    end_idx = min((rank + 1) * seeds_per_rank, num_seeds)
    
    # Process multiple seeds per rank
    results_from_this_rank = []
    for i in range(start_idx, end_idx):
        # Calculate seed for this iteration
        network_seed = 66049 + i
        data_seed = 257 + i
        
        # Assign GPU to this worker
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            gpu_id = rank % n_gpus
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Rank {rank}, Iteration {i}: Using GPU {gpu_id}, Network Seed: {network_seed}, Data Seed: {data_seed}")
        else:
            device = torch.device('cpu')
            print(f"Rank {rank}, Iteration {i}: Using CPU, Network Seed: {network_seed}, Data Seed: {data_seed}")
        
        start_time = time()
        
        # Set seeds for this worker
        ut.set_all_seeds(seed=network_seed)
        
        # Load dataset
        train_dataloader, val_dataloader, test_dataloader, data_generator = \
            ut.get_MNIST_dataloaders(data_dir=data_dir)
        if debug:
            print(f"Rank {rank}, Seed {network_seed}: Loaded Data")
        
        # Create network
        config_file_path = f"EIANN/network_config/mnist/{network_config_file_name}"
        network = ut.build_EIANN_from_config(config_file_path, 
                                             network_seed=network_seed, 
                                             device=device)
        if debug:
            print(f"Rank {rank}, Seed {network_seed}: Built Network")
        
        # Train network
        data_generator.manual_seed(data_seed)
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
        
        # Collect results from this iteration
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
        print(f"  Network Seed: {network_seed}, Data Seed: {data_seed}")
        print(f"  Device: {device}")
        print(f"  Final Val Accuracy: {result['val_accuracy']:.4f}")
        print(f"  Final Val Loss: {result['val_loss']:.6f}")
        print(f"  Run Time: {run_time:.2f} sec\n")
    
    # Gather all results to rank 0
    all_results = comm.gather(results_from_this_rank, root=0)
    
    # Rank 0 processes and saves results
    if rank == 0:
        # Flatten results from all ranks
        flattened_results = []
        for rank_results in all_results:
            flattened_results.extend(rank_results)
            
        # Sort by seed for consistent output
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
        
        print("\nIndividual Results:")
        for r in flattened_results:
            print(f"  Net Seed {r['seed']:6d}, Data Seed {r['data_seed']:4d}: Acc={r['val_accuracy']:.4f}, Loss={r['val_loss']:.6f}, Time={r['run_time']:.2f}s")
        
        # Save results to file
        # import json
        # output_file = f"results_{network_config_file_name.replace('.yaml', '')}_{len(flattened_results)}seeds.json"
        # with open(output_file, 'w') as f:
        #     json.dump(flattened_results, f, indent=2)
        # print(f"\nResults saved to: {output_file}")
        # print("="*60 + "\n")

if __name__ == '__main__':
    main()