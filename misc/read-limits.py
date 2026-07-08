import os
import uproot
import numpy as np
from matplotlib import pyplot as plt

basepath = "trexfitter-output"

def get_all_root_files(basepath: str):
    all_root_files = {}
    for d in os.listdir(basepath):
        mass = d[1:]
        all_root_files[mass] = f"{basepath}/{d}/lq-bymass-limits/Limits/Asymptotics/myLimit.root"
    return all_root_files
    
all_root_files = get_all_root_files(basepath)

features = [
    "exp_upperlimit",
    "exp_upperlimit_plus1",
    "exp_upperlimit_plus2",
    "exp_upperlimit_minus1",
    "exp_upperlimit_minus2"
]

def fix_xs_weight(x):
    return x*0.1;

all_stats = {}
for mass,root_file in all_root_files.items():
    print(f"Reading root file {root_file}")
    with uproot.open(root_file) as f:
        stats_df = f["stats"].arrays(features, library="pd")
        stats_df[features] = stats_df[features].apply(fix_xs_weight)
        all_stats[mass] = stats_df[features].values.astype(np.float32)[0]

print(all_stats)

masses_sorted = np.array(sorted(all_stats.keys()))
print(f"sorted masses: {masses_sorted}")

exp   = np.array([all_stats[m][0] for m in masses_sorted], dtype=np.float32)
expp1 = np.array([all_stats[m][1] for m in masses_sorted], dtype=np.float32)
expp2 = np.array([all_stats[m][2] for m in masses_sorted], dtype=np.float32)
expm1 = np.array([all_stats[m][3] for m in masses_sorted], dtype=np.float32)
expm2 = np.array([all_stats[m][4] for m in masses_sorted], dtype=np.float32)


fig,ax = plt.subplots()
ax.fill_between(masses_sorted, expm2, expp2, facecolor="yellow", edgecolor="none", label=r"Exp. $\pm$ 2$\sigma$")
ax.fill_between(masses_sorted, expm1, expp1, facecolor="green",  edgecolor="none", label=r"Exp. $\pm$ 1$\sigma$")
ax.plot(masses_sorted, exp, color="black", linestyle="--", label="Exp. 95% CL Limit")

ax.set_yscale("log")
ax.set_xlabel("LQ Mass [GeV]")
ax.set_ylabel("Cross Section [pb]")
ax.set_title("Upper Limit on Cross Section at 95% CL")
ax.legend(loc="upper right")

plt.savefig("limits.pdf")
