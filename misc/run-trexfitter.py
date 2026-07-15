import os

basepath = "./pnn_xgb/trex_inputs"
allfiles = [f"{basepath}/{f}" for f in os.listdir(basepath)]

with open("limits.config.in", "r") as f:
    limits_config_in = f.readlines()

with open("probs.config.in", "r") as f:
    probs_config_in = f.readlines()

for f in allfiles:
    mass = f.split("_")[-1].split(".")[0][1:]
    print(f"Running m = {mass} GeV")

    limits_config_in_copy = []
    for i in range(len(limits_config_in)):
        limits_config_in_copy.append(limits_config_in[i].replace("%MASS%", mass))    
    with open("limits.config", "w") as f:
        f.writelines(limits_config_in_copy)

    probs_config_in_copy = []
    for i in range(len(probs_config_in)):
        probs_config_in_copy.append(probs_config_in[i].replace("%MASS%", mass))    
    with open("probs.config", "w") as f:
        f.writelines(probs_config_in_copy)

    os.system("trex-fitter hwfl limits.config")
    os.system("trex-fitter hwd probs.config")
