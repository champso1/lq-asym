#!/usr/bin/env python
import autorootcwd
from argparse import ArgumentParser
from utils import TRExFitter, read_yaml
from pathlib import Path
import platform
from tqdm import tqdm
import numpy as np
import uproot
from collections import defaultdict

START_PATH = ""

class Data:
    def __init__(self,
        x: np.ndarray, y: np.ndarray | None, w: np.ndarray,
        x_names: list[str], y_names: list[str] | None,
        event_numbers: np.ndarray
    ):
        self.x = x
        self.y = y
        self.w = w
        self.x_names = x_names
        self.y_names = y_names
        self.event_numbers = event_numbers

class ProcessedData:
    def __init__(self,
        x_continuous, x_discrete, y, w, mean, std,
        x_names_continuous, x_names_discrete, y_names,
        map_discrete, event_numbers
    ):
        self.x_continuous = x_continuous
        self.x_discrete = x_discrete
        self.y = y
        self.w = w
        self.mean = mean
        self.std, = std,
        self.x_names_continuous = x_names_continuous
        self.x_names_discrete = x_names_discrete
        self.y_names, = y_names,
        self.map_discrete = map_discrete
        self.event_numbers = event_numbers

def convert_features(data, feature_list: list[str]) -> Data:
    w = data["weight"].to_numpy()
    event_numbers = np.ones_like(w)
    n_features = len(feature_list)
    n_samples = w.shape[0]
    x = np.zeros((n_samples, n_features), dtype=np.float32)

    for i,feature in enumerate(feature_list):
        feature_data = data[feature].to_numpy().astype(np.float32)
        x[:, i] = feature_data

    return Data(x=x, y=None, w=w, x_names=feature_list, y_names=None, event_numbers=event_numbers)

# grab the weight and selection expr
# then grab the entire list of samples, and resolve their full filepaths
# also load the features we want to train with
# then we load all the data from all the files corresponding to each sample
# all while applying the corresponsing weight and selection criteria
def read_region(config_file_data: dict, trexfitter: TRExFitter):
    samples = config_file_data["Samples"]
    weight_expr = trexfitter.get_weight_expr()
    selection_expr = trexfitter.get_selection_expr(config_file_data)
    print(f"Weight: {weight_expr}")
    print(f"Selection: {selection_expr}")
    print("Using the following samples:")
    for s in samples:
        print(f"  - {s}")

    features = []
    features_raw = config_file_data["Features"]
    for feature_set in features_raw:
        if "Continuous" in feature_set.keys():
            features = features + feature_set["Continuous"]
        elif "Discrete" in feature_set.keys():
            features = features + feature_set["Discrete"]

    expressions = [*features]
    aliases = {}
    if weight_expr is not None:
        expressions.append("weight")
        aliases["weight"] = weight_expr
    if selection_expr is not None:
        expressions.append("selected")
        aliases["selected"] = selection_expr
    if len(aliases) == 0:
        aliases = None

    print("Using the following features:")
    for feature in features:
        print(f"  - {feature}")

    root_path = "/home/champson/Dropbox/CERN/ntuples/fastframes/"
    empty_files = {}
    all_data = {}
    for s in tqdm(samples,desc="Loading samples..."):
        all_files = []
        empty_file_list = []
        files = trexfitter.replace(trexfitter.get_block("Sample", s, "NtupleFiles")).split(", ")
        files = [root_path + f + ".root:reco" for f in files]
        for f in files:
            with uproot.open(f) as root_file:
                if len(root_file) == 0:
                    empty_file_list.append(f)
                    continue
                all_files.append(f)
        if len(empty_file_list) != 0:
            empty_files[s] = empty_file_list
        all_data[s] = convert_features(uproot.concatenate(all_files, expressions=expressions, aliases=aliases), features)

    print("Found the following empty files:")
    for k,v in empty_files.items():
        print(f"'{k}'")
        v = ["/".join(f.split("/")[-2:]).split(":")[0] for f in v]
        for f in v:
            print(f"  - {f}")

    # x_names = features
    # y_names = samples
    x = np.concatenate([all_data[y_name].x for y_name in samples])
    y = np.concatenate([np.full(all_data[y_name].x.shape[0], i) for i, y_name in enumerate(samples)])
    w = np.concatenate([all_data[y_name].w for y_name in samples])
    event_numbers = np.concatenate([all_data[y_name].event_numbers for y_name in samples])

    return Data(x=x, w=w, y=y, x_names=features, y_names=samples, event_numbers=event_numbers)


def process_data(data: Data, config_file_data: dict[str], discrete_feature_list: list[str]) -> ProcessedData:
    # Split x into categorical and continuous
    i_discrete = [data.x_names.index(feature) for feature in discrete_feature_list]
    i_continuous = [i for i in range(len(data.x_names)) if i not in i_discrete]

    x_discrete = data.x[:, i_discrete].astype(int)
    x_continuous = data.x[:, i_continuous]
    x_names_discrete = [data.x_names[i] for i in i_discrete]
    x_names_continuous = [data.x_names[i] for i in i_continuous]

    map_discrete = defaultdict(dict)
    for i, feature_name in enumerate(x_names_discrete):
        features = x_discrete[:, i]
        min_value = np.min(features)
        features -= min_value
        max_value = np.max(features)

        new_index = 0
        for j in range(max_value + 1):
            matches = features == j

            if matches.sum() > 0:
                map_discrete[feature_name][new_index] = f"{feature_name}={j + min_value}"
                features[matches] = new_index
                new_index += 1

    # Find the mean and std of the continuous features
    mean = np.nanmean(x_continuous, axis=0)
    std = np.nanstd(x_continuous, axis=0)

    i_valid = std != 0

    mean = mean[i_valid]
    std = std[i_valid]
    x_continuous = (x_continuous[:, i_valid] - mean) / std
    x_names_continuous = [name for i, name in enumerate(x_names_continuous) if i_valid[i]]

    assert len(x_names_continuous) == len(mean) == len(std) == x_continuous.shape[1]
    assert len(x_names_discrete) == x_discrete.shape[1]

    return ProcessedData(
        x_discrete=x_discrete,
        x_continuous=x_continuous,
        y=data.y,
        w=data.w,
        mean=mean,
        std=std,
        x_names_discrete=x_names_discrete,
        x_names_continuous=x_names_continuous,
        y_names=data.y_names,
        map_discrete=map_discrete,
        event_numbers=data.event_numbers,
    )

def save_data(data: ProcessedData, config_file_data: dict[str]):
    path = Path().joinpath("data_processing", "output").resolve()

    if not path.exists():
        path.mkdir()

    np.save(path.joinpath("x_categorical.npy"), data.x_discrete)
    np.save(path.joinpath("x_continuous.npy"), data.x_continuous)
    np.save(path.joinpath("y.npy"), data.y)
    np.save(path.joinpath("w.npy"), data.w)
    np.save(path.joinpath("mean.npy"), data.mean)
    np.save(path.joinpath("std.npy"), data.std)
    np.save(path.joinpath("x_names_discrete.npy"), data.x_names_discrete)
    np.save(path.joinpath("x_names_continuous.npy"), data.x_names_continuous)
    np.save(path.joinpath("y_names.npy"), data.y_names)
    np.save(path.joinpath("map_discrete.npy"), data.map_discrete)
    np.save(path.joinpath("event_numbers.npy"), data.event_numbers)

def main():
    global START_PATH
    platform_name = platform.system()
    print(f"Target platform: {platform_name}")
    if platform_name == "Linux":
        START_PATH = "/home/champson/Dropbox/CERN/ntuples/fastframes/"
    elif platform_name == "Windows":
        START_PATH = "C:/Users/Casey/Dropbox/CERN/ntuples/fastframes/"
    else:
        exit(1)

    parser = ArgumentParser(prog="process.py")
    parser.add_argument("config_file", nargs="?", default="config-new.yaml")
    args = parser.parse_args()
    config_file_name = args.config_file
    config_filepath = Path().joinpath("data_processing", config_file_name).resolve()
    if not config_filepath.exists():
        print(f"Config file '{config_filepath}' doesnt exist")
        exit(1)
    config_file_data = read_yaml(config_filepath)
    
    trexfitter_config_name = config_file_data["TRExConfig"]
    trexfitter_config_path = Path().joinpath("trex-fitter", "configs", trexfitter_config_name + ".config")
    trexfitter = TRExFitter(trexfitter_config_path)

    features = config_file_data["Features"]
    discrete_features = None
    for feature in features:
        if isinstance(feature, dict):
            if "Discrete" in feature.keys():
                discrete_features = feature["Discrete"]
    if discrete_features is None:
        print("Failed to find discrete features")
        exit(1)
    
    print(config_file_data)
    data = read_region(config_file_data, trexfitter)
    data = process_data(data, config_file_data, discrete_features)
    save_data(data, config_file_data)
    

if __name__ == "__main__":
    main()
