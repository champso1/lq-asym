import autorootcwd
from argparse import ArgumentParser
from utils import TRExFitter, read_yaml
from pathlib import Path

def main():
    parser = ArgumentParser(prog="process.py")
    parser.add_argument("config_file", nargs="?", default="config-new.yaml")
    args = parser.parse_args()
    config_file_name = args.config_file
    config_filepath = Path().joinpath("data_processing", config_file_name).resolve()
    if not config_filepath.exists():
        print(f"Config file '{config_filepath}' doesnt exist")
        exit(1)

    config_file_data = read_yaml(config_filepath)
    trexfitter_config_name = config_file_data["trex_config"]
    trexfitter_config_path = Path().joinpath("trex-fitter", "configs", trexfitter_config_name + ".config")
    trexfitter = TRExFitter(trexfitter_config_path)
    trexfitter.read_region(config_file_data)

if __name__ == "__main__":
    main()
