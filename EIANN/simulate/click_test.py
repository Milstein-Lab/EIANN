import click


@click.command
@click.option("--data-file-path", '-d', multiple=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(data_file_path):
    print(len(data_file_path), data_file_path)
    if not data_file_path:
        print('getting here')
    
    
if __name__ == '__main__':
    main(standalone_mode=False)