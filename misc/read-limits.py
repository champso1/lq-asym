import os
import uproot
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

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
    return x*0.1

all_stats = {}
for mass,root_file in all_root_files.items():
    with uproot.open(root_file) as f:
        stats_df = f["stats"].arrays(features, library="pd")
        stats_df[features] = stats_df[features].apply(fix_xs_weight)
        all_stats[mass] = stats_df[features].values.astype(np.float32)[0]


masses_sorted = np.array(sorted(all_stats.keys()))
masses = masses_sorted.astype(np.float32)

exp   = np.array([all_stats[m][0] for m in masses_sorted], dtype=np.float32)
expp1 = np.array([all_stats[m][1] for m in masses_sorted], dtype=np.float32)
expp2 = np.array([all_stats[m][2] for m in masses_sorted], dtype=np.float32)
expm1 = np.array([all_stats[m][3] for m in masses_sorted], dtype=np.float32)
expm2 = np.array([all_stats[m][4] for m in masses_sorted], dtype=np.float32)

fig,ax = plt.subplots()
ax.fill_between(masses, expm2, expp2, facecolor="yellow", edgecolor="none", label=r"Exp. $\pm$ 2$\sigma$")
ax.fill_between(masses, expm1, expp1, facecolor="lime",  edgecolor="none", label=r"Exp. $\pm$ 1$\sigma$")
ax.plot(masses, exp, color="black", linestyle="--", label="Exp. 95% CL Limit")

# masses_th = np.array([1500., 2000., 2500.], dtype=np.float32)
# values_th = np.array([1.728e-3, 2.376e-4, 3.491e-5], dtype=np.float32)
# ax.plot(masses_th, values_th, color="red", label="Theory y=1.0")

# print(f"sorted masses: {masses}")
# print(f"theory masses: {masses_th}")

ax.set_yscale("log")
ax.set_xlabel("LQ Mass [GeV]")
ax.set_ylabel("Cross Section [pb]")
ax.set_ylim(bottom=5e-4, top=5e-2)
ax.set_title("Upper Limit on Cross Section at 95% CL")
ax.legend(loc="upper right")

plt.savefig("limits-plot.pdf")

# ====================
#    print latex
# ====================
latex_file_in = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{caption}
\begin{document}
%TABLE%
\end{document}
"""
outfilename = "limits-table.tex"

latex_data = {
    "Masses": masses.astype(int),
    "Expected": exp,
    r"$+1\sigma$": expp1,
    r"$+2\sigma$": expp2,
    r"$-1\sigma$": expm1,
    r"$-2\sigma$": expm2,
}
latex_data_df = pd.DataFrame(latex_data)
latex_table = latex_data_df.to_latex(
    index=False,
    caption=r"95\% Limits on LQ Cross Section",
    label="tab:limits",
    column_format="c||c|cccc",
    position="h!",
    
)
latex_file = latex_file_in.replace("%TABLE%",  latex_table)
with open(outfilename, "w") as f:
    f.write(latex_file)

os.system(f"pdflatex -interaction=batchmode {outfilename}")
