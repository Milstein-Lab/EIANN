import os
import re
import click

def rename_files(directory):
    for filename in os.listdir(directory):
        if 'complete_optimized' in filename:
            print(f"Skipping: {filename}")
            continue

        if filename.endswith('.pkl'):
            match = re.match(r'(.*?)_(\d{5})_(\d{3})_complete(.*)\.pkl', filename)
            if match:
                base_name = match.group(1)
                net_seed_number = match.group(2)
                data_seed_number = match.group(3)

                extended_tag = '_extended' if 'extended' in directory.lower() else ''
                
                # Construct the new filename with 'complete_optimized'
                new_filename = f"{base_name}_complete_optimized_{net_seed_number}_{data_seed_number}{extended_tag}.pkl"
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_filename}")

@click.command()
@click.option('--dir', default=None, help='spiral or MNIST')
@click.option('--extended', is_flag=True, help='extended or not')
def main(dir, extended):
    # Set the directory to where files are located
    if os.name == "posix":
        username = os.environ.get("USER")
        saved_network_path = f"/Users/{username}/Library/CloudStorage/Box-Box/Milstein-Shared/EIANN exported data/2024 Manuscript V2/"
    elif os.name == "nt":
        username = os.environ.get("USERNAME")
        saved_network_path = f"C:/Users/{username}/Box/Milstein-Shared/EIANN exported data/2024 Manuscript V2/"

    saved_network_path += dir + '/'

    if extended:
        saved_network_path += 'extended/'

    rename_files(saved_network_path)

if __name__ == '__main__':
    main()