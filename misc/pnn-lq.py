import os
import uproot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import shap
import pandas as pd
import json
import cupy as cp
import optuna


# base = "/home/champson/data/fastframes"
base = "/mnt/d/Documents/Ntuples/fastframes"
# -----------------------------
# lq masses (1.0 yukawa only)
# -----------------------------
signal_file_1 = [ # 1500 GeV
    f"{base}/mc20a/545824.root",
    f"{base}/mc20d/545824.root",
    f"{base}/mc20e/545824.root",
]
signal_file_2 = [ # 2000 GeV
    f"{base}/mc20a/545825.root",
    f"{base}/mc20d/545825.root",
    f"{base}/mc20e/545825.root",
]
signal_file_3 = [ # 2500 GeV
    f"{base}/mc20a/545826.root",
    f"{base}/mc20d/545826.root",
    f"{base}/mc20e/545826.root",
]
signal_file_4 = [ # 1600 GeV
    f"{base}/mc20a/567831.root",
    f"{base}/mc20d/567831.root",
    f"{base}/mc20e/567831.root",
]
signal_file_5 = [ # 1700 GeV
    f"{base}/mc20a/567832.root",
    f"{base}/mc20d/567832.root",
    f"{base}/mc20e/567832.root",
]
signal_file_6 = [ # 1800 GeV
    f"{base}/mc20a/567833.root",
    f"{base}/mc20d/567833.root",
    f"{base}/mc20e/567833.root",
]
signal_file_7 = [ # 1900 GeV
    f"{base}/mc20a/567834.root",
    f"{base}/mc20d/567834.root",
    f"{base}/mc20e/567834.root",
]
signal_file_8 = [ # 2100 GeV
    f"{base}/mc20a/567835.root",
    f"{base}/mc20d/567835.root",
    f"{base}/mc20e/567835.root",
]
signal_file_9 = [ # 2200 GeV
    f"{base}/mc20a/567836.root",
    f"{base}/mc20d/567836.root",
    f"{base}/mc20e/567836.root",
]
signal_file_10 = [ # 2300 GeV
    f"{base}/mc20a/567837.root",
    f"{base}/mc20d/567837.root",
    f"{base}/mc20e/567837.root",
]
signal_file_11 = [ # 2400 GeV
    f"{base}/mc20a/567838.root",
    f"{base}/mc20d/567838.root",
    f"{base}/mc20e/567838.root",
]

# -----------------------------
# backgrounds
# -----------------------------
ttH = [
    f"{base}/mc20a/346343.root",
    f"{base}/mc20d/346343.root",
    f"{base}/mc20e/346343.root",
    
    f"{base}/mc20a/346344.root",
    f"{base}/mc20d/346344.root",
    f"{base}/mc20e/346344.root",
    
    f"{base}/mc20a/346345.root",
    f"{base}/mc20d/346345.root",
    f"{base}/mc20e/346345.root",
]

ttW = [
    f"{base}/mc20a/701261.root",
    f"{base}/mc20d/701261.root",
    f"{base}/mc20e/701261.root",
    
    f"{base}/mc20a/701262.root",
    f"{base}/mc20d/701262.root",
    f"{base}/mc20e/701262.root",
]

ttZ = [
    f"{base}/mc20a/504330.root",
    f"{base}/mc20d/504330.root",
    f"{base}/mc20e/504330.root",
    
    f"{base}/mc20a/504334.root",
    f"{base}/mc20d/504334.root",
    f"{base}/mc20e/504334.root",
    
    f"{base}/mc20a/504342.root",
    f"{base}/mc20d/504342.root",
    f"{base}/mc20e/504342.root",
]

ttbar = [
    f"{base}/mc20a/410470.root",
    f"{base}/mc20d/410470.root",
    f"{base}/mc20e/410470.root",
]


vv = [
    f"{base}/mc20a/701000.root",
    f"{base}/mc20d/701000.root",
    f"{base}/mc20e/701000.root",
    
    f"{base}/mc20a/701005.root",
    f"{base}/mc20d/701005.root",
    f"{base}/mc20e/701005.root",
    
    f"{base}/mc20a/701010.root",
    f"{base}/mc20d/701010.root",
    f"{base}/mc20e/701010.root",

    f"{base}/mc20a/701015.root",
    f"{base}/mc20d/701015.root",
    f"{base}/mc20e/701015.root",
    
    f"{base}/mc20a/701020.root",
    f"{base}/mc20d/701020.root",
    f"{base}/mc20e/701020.root",
    
    f"{base}/mc20a/701025.root",
    f"{base}/mc20d/701025.root",
    f"{base}/mc20e/701025.root",
    
    f"{base}/mc20a/701030.root",
    f"{base}/mc20d/701030.root",
    f"{base}/mc20e/701030.root",
    
    f"{base}/mc20a/701035.root",
    f"{base}/mc20d/701035.root",
    f"{base}/mc20e/701035.root",
    
    f"{base}/mc20a/701055.root",
    f"{base}/mc20d/701055.root",
    f"{base}/mc20e/701055.root",
    
    f"{base}/mc20a/701085.root",
    f"{base}/mc20d/701085.root",
    f"{base}/mc20e/701085.root",
    
    f"{base}/mc20a/701090.root",
    f"{base}/mc20d/701090.root",
    f"{base}/mc20e/701090.root",
    
    f"{base}/mc20a/701105.root",
    f"{base}/mc20d/701105.root",
    f"{base}/mc20e/701105.root",
    
    f"{base}/mc20a/701115.root",
    f"{base}/mc20d/701115.root",
    f"{base}/mc20e/701115.root",
    
    f"{base}/mc20a/701120.root",
    f"{base}/mc20d/701120.root",
    f"{base}/mc20e/701120.root",
    
    f"{base}/mc20a/701125.root",
    f"{base}/mc20d/701125.root",
    f"{base}/mc20e/701125.root",
]

vvv = [
    f"{base}/mc20a/364242.root",
    f"{base}/mc20d/364242.root",
    f"{base}/mc20e/364242.root",
    
    f"{base}/mc20a/364243.root",
    f"{base}/mc20d/364243.root",
    f"{base}/mc20e/364243.root",
    
    f"{base}/mc20a/364244.root",
    f"{base}/mc20d/364244.root",
    f"{base}/mc20e/364244.root",
    
    f"{base}/mc20a/364245.root",
    f"{base}/mc20d/364245.root",
    f"{base}/mc20e/364245.root",
    
    f"{base}/mc20a/364246.root",
    f"{base}/mc20d/364246.root",
    f"{base}/mc20e/364246.root",
    
    f"{base}/mc20a/364247.root",
    f"{base}/mc20d/364247.root",
    f"{base}/mc20e/364247.root",
    
    f"{base}/mc20a/364248.root",
    f"{base}/mc20d/364248.root",
    f"{base}/mc20e/364248.root",
    
    f"{base}/mc20a/364249.root",
    f"{base}/mc20d/364249.root",
    f"{base}/mc20e/364249.root",
]


vh = [
    f"{base}/mc20a/346645.root",
    f"{base}/mc20d/346645.root",
    f"{base}/mc20e/346645.root",
    
    f"{base}/mc20a/346646.root",
    f"{base}/mc20d/346646.root",
    f"{base}/mc20e/346646.root",
]


vgamma = [
    f"{base}/mc20a/700398.root",
    f"{base}/mc20d/700398.root",
    f"{base}/mc20e/700398.root",
    f"{base}/mc20a/700399.root",
    f"{base}/mc20d/700399.root",
    f"{base}/mc20e/700399.root",
    f"{base}/mc20a/700400.root",
    f"{base}/mc20d/700400.root",
    f"{base}/mc20e/700400.root",
#    f"{base}/mc20a/700401.root",
#    f"{base}/mc20d/700401.root",
#    f"{base}/mc20e/700401.root",
    f"{base}/mc20a/700402.root",
    f"{base}/mc20d/700402.root",
    f"{base}/mc20e/700402.root",
    f"{base}/mc20a/700403.root",
    f"{base}/mc20d/700403.root",
    f"{base}/mc20e/700403.root",
    f"{base}/mc20a/700404.root",
    f"{base}/mc20d/700404.root",
    f"{base}/mc20e/700404.root",
]

wjets = [
    f"{base}/mc20a/700338.root",
    f"{base}/mc20d/700338.root",
    f"{base}/mc20e/700338.root",
    f"{base}/mc20a/700341.root",
    f"{base}/mc20d/700341.root",
    f"{base}/mc20e/700341.root",
]

zjets = [
    f"{base}/mc20a/700320.root",
    f"{base}/mc20d/700320.root",
    f"{base}/mc20e/700320.root",
    f"{base}/mc20a/700321.root",
    f"{base}/mc20d/700321.root",
    f"{base}/mc20e/700321.root",
#    f"{base}/mc20a/700322.root",
#    f"{base}/mc20d/700322.root",
#    f"{base}/mc20e/700322.root",
    f"{base}/mc20a/700323.root",
    f"{base}/mc20d/700323.root",
    f"{base}/mc20e/700323.root",
    f"{base}/mc20a/700324.root",
    f"{base}/mc20d/700324.root",
    f"{base}/mc20e/700324.root",
    f"{base}/mc20a/700325.root",
    f"{base}/mc20d/700325.root",
    f"{base}/mc20e/700325.root",
]


# Rare tops


ttHH = [
    f"{base}/mc20a/500460.root",
    f"{base}/mc20d/500460.root",
    f"{base}/mc20e/500460.root",
]

ttWH = [
    f"{base}/mc20a/500461.root",
    f"{base}/mc20d/500461.root",
    f"{base}/mc20e/500461.root",
]

ttWW = [
    f"{base}/mc20a/410081.root",
    f"{base}/mc20d/410081.root",
    f"{base}/mc20e/410081.root",
]

ttWZ = [
    f"{base}/mc20a/500463.root",
    f"{base}/mc20d/500463.root",
    f"{base}/mc20e/500463.root",
]

ttZZ = [
    f"{base}/mc20a/500462.root",
    f"{base}/mc20d/500462.root",
    f"{base}/mc20e/500462.root",
]

ttgamma = [
    f"{base}/mc20a/500462.root",
    f"{base}/mc20d/500462.root",
    f"{base}/mc20e/500462.root",
    
    f"{base}/mc20a/504554.root",
    f"{base}/mc20d/504554.root",
    f"{base}/mc20e/504554.root",
]

# Other rare processes
tZ = [
    f"{base}/mc20a/410560.root",
    f"{base}/mc20d/410560.root",
    f"{base}/mc20e/410560.root",
]

WtZ = [
    f"{base}/mc20a/410408.root",
    f"{base}/mc20d/410408.root",
    f"{base}/mc20e/410408.root",
]

fourTop = [
    f"{base}/mc20a/412043.root",
    f"{base}/mc20d/412043.root",
    f"{base}/mc20e/412043.root",
]


singleTop = [
    f"{base}/mc20a/410644.root",
    f"{base}/mc20d/410644.root",
    f"{base}/mc20e/410644.root",
    
    f"{base}/mc20a/410645.root",
    f"{base}/mc20d/410645.root",
    f"{base}/mc20e/410645.root",
    
    f"{base}/mc20a/410654.root",
    f"{base}/mc20d/410654.root",
    f"{base}/mc20e/410654.root",
    
    f"{base}/mc20a/410655.root",
    f"{base}/mc20d/410655.root",
    f"{base}/mc20e/410655.root",
    
    f"{base}/mc20a/410659.root",
    f"{base}/mc20d/410659.root",
    f"{base}/mc20e/410659.root",
]


threeTop = [
    f"{base}/mc20a/304014.root",
    f"{base}/mc20d/304014.root",
    f"{base}/mc20e/304014.root",
]



background_files = ttH + ttW + ttZ + ttbar + vv + vvv + vh + vgamma + wjets + zjets + ttHH + ttWH + ttWW + ttWZ + ttZZ + ttgamma + tZ + WtZ + fourTop + singleTop + threeTop
signal_files_all = signal_file_1 + signal_file_2 + signal_file_3 + signal_file_4 + signal_file_5 + signal_file_6 + signal_file_7 + signal_file_8 + signal_file_9 + signal_file_10 + signal_file_11

background_files_dict = {
    "ttH": ttH,
    "ttW": ttW,
    "ttZ": ttZ,
    "ttbar": ttbar,
    "vv": vv,
    "vvv": vvv,
    "vh": vh,
    "vgamma": vgamma,
    "wjets": wjets,
    "zjets": zjets,
    "ttHH": ttHH,
    "ttWH": ttWH,
    "ttWW": ttWW,
    "ttWZ": ttWZ,
    "ttZZ": ttZZ,
    "ttgamma": ttgamma,
    "tZ": tZ,
    "WtZ": WtZ,
    "fourTop": fourTop,
    "singleTop": singleTop,
    "threeTop": threeTop,
}

tree_name = "reco"

selected_features = [
    "taus_pt_0_NOSYS",
    "m_eff_NOSYS",
    "HT_leptons_NOSYS",
    "jets_n_NOSYS",
    "pT_balance_lep_tau_MET_NOSYS",
    "min_DeltaR_tau0_SSlepton_NOSYS",
    "b0_pt_NOSYS",
    "MLepMet_NOSYS",
    "HT_NOSYS",
    "jet_0_1_phi_diff_cos_NOSYS",
    "top_reco_mass_l0b0_NOSYS",
    "dEta_maxMjj_frwdjet_NOSYS",
    "lep_deltaz0sinTheta_0_NOSYS",
    "DeltaR_min_lep_jet_fwd_NOSYS",
    "minDeltaR_LJ_1_NOSYS",
    "lep_d0sig_1_NOSYS",
    "nbJets77_NOSYS",
    "min_AbsDeltaPhi_bjet_MET_NOSYS",
    "EtaZeppen_tau_NOSYS",
    "reco_Hplus_visH_hadW_mass_NOSYS",
    "min_DeltaR_tau_bjet_NOSYS",
    "leps_charge_0_NOSYS",
    "pT_vector_balance_NOSYS",
    "had_W_jj_mass_NOSYS",
    "lep1_tau_Phi_diff_cos_NOSYS",
    "MET_centrality_NOSYS",
    "taus_RNNJetScoreSigTrans_0_NOSYS",
    "leps_pt_1_NOSYS",
    "lep_d0sig_0_NOSYS",
    "DeltaPhi_SSLeptonSystem_Tau_NOSYS",
    "DeltaR_L0_L1_NOSYS",
    "DeltaPhi_T0_MET_NOSYS",
    "lep_Z0SinTheta_1_NOSYS",
    "min_chi2_top_reco_NOSYS",
]



signal_by_mass = {
    1500:  signal_file_1,
    2000: signal_file_2,
    2500:  signal_file_3,
    1600:  signal_file_4,
    1700:  signal_file_5,
    1800:  signal_file_6,
    1900:  signal_file_7,
    2100:  signal_file_8,
    2200: signal_file_9,
    2300: signal_file_10,
    2400: signal_file_11,
}

pnn_features = list(dict.fromkeys(selected_features))
# The pNN model will see these features PLUS log(m) appended as the last column.
# Total input width = len(pnn_features) + 1
 
print(f"pNN feature set: {len(pnn_features)} physics features + 1 mass column")

def make_samples_for_background():
    all_lines = []
    intext = """Sample: "%SAMPLE%"
  Type: BACKGROUND
  Group: "%GROUP%"
  Title: "%SAMPLE%"
  FillColor: %COLOR_IDX%
  LineColor: 1
  HistoFile: "hist_m%MASS%"
  HistoName: "%SAMPLE%"
    
"""
    color_idx_background = 3
    color_idx_other = 20
    for name,_ in background_files_dict.items():
        printtxt = intext.replace("%SAMPLE%", name)
        if name in ["ttH", "ttW", "ttZ", "ttbar"]:
            printtxt = printtxt.replace("%GROUP%", name)
            printtxt = printtxt.replace("%COLOR_IDX%", str(color_idx_background))
            color_idx_background += 1
        else:
            printtxt = printtxt.replace("%GROUP%", "Other")
            printtxt = printtxt.replace("%COLOR_IDX%", str(color_idx_other))
            color_idx_other += 1 
        all_lines.append(printtxt)

    with open("trexfitter-samples.txt", "w") as f:
        f.writelines(all_lines)

make_samples_for_background()
# ── Unchanged helpers ─────────────────────────────────────────────────────────
 
def extract_feature(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.max(x)) if len(x) > 0 else np.nan
    return float(x)
 
 
def load_root_files(file_paths, tree_name, features, cut_expr):
    """Load ROOT files and return a plain feature matrix (no mass column)."""
    chunks_X = []
    chunks_w = []
    
    for fp in file_paths:
        print(f"  Loading: {fp}")
        with uproot.open(fp) as f:
            df = f[tree_name].arrays(features, library="pd", cut=cut_expr)
            df_w = f[tree_name].arrays(["weight_total_NOSYS"], library="pd", cut=cut_expr)
        for feat in features:
            df[feat] = df[feat].apply(extract_feature)
        df_w["weight_total_NOSYS"] = df_w["weight_total_NOSYS"].apply(extract_feature)
        df = df.fillna(0)
        df_w = df_w.fillna(0)

        X_data = df[features].values.astype(np.float32)
        w_data = df_w["weight_total_NOSYS"].values.astype(np.float32)
        
        chunks_X.append(X_data)
        chunks_w.append(w_data)
        
    if not chunks_X:
        return np.empty((0, len(features)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks_X, axis=0), np.concatenate(chunks_w, axis=0)
 
 
def append_log_mass(X, mass_value):
    """Append log(mass_value) as a constant last column to every row."""
    log_m = np.full((len(X), 1), np.log(float(mass_value)), dtype=np.float32)
    return np.concatenate([X, log_m], axis=1)
 
 
def find_best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precision[:-1] * recall[:-1]
                 / (precision[:-1] + recall[:-1] + 1e-12))
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]
 
 
def compute_conf_matrix(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return cm, tn, fp, fn, tp
 
 
def save_prf1_vs_threshold_plot(y_true, y_score, title, out_path, n_thr=1000):
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.linspace(0.0, 1.0, n_thr + 1)
    precisions = np.zeros_like(thresholds)
    recalls    = np.zeros_like(thresholds)
    f1s        = np.zeros_like(thresholds)
    sigs       = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        y_pred = (y_score >= t).astype(int)
        precisions[i] = precision_score(y_true, y_pred, zero_division=0)
        recalls[i]    = recall_score(y_true, y_pred, zero_division=0)
        f1s[i]        = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sigs[i] = tp / np.sqrt(fp) if fp > 0 else 0.0
    best_f1_idx  = np.argmax(f1s)
    best_sig_idx = np.argmax(sigs)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, precisions, "b--", label="Precision")
    ax1.plot(thresholds, recalls,    "g-",  label="Recall")
    ax1.plot(thresholds, f1s,        "r-",  label="F1-Score")
    ax1.axvline(thresholds[best_f1_idx], color="purple", linestyle="--",
                label=f"Best F1 thr ({thresholds[best_f1_idx]:.2f})")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, sigs, "-", label="TP/sqrt(FP)")
    ax2.axvline(thresholds[best_sig_idx], color="black", linestyle=":",
                label=f"Best S/sqrt(B) thr ({thresholds[best_sig_idx]:.2f})")
    ax2.set_ylabel("TP/sqrt(FP)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return thresholds[best_f1_idx], f1s[best_f1_idx], thresholds[best_sig_idx], sigs[best_sig_idx]
 
 
# ── 1. Load data ──────────────────────────────────────────────────────────────
# Signal: one array per mass point (no mass column yet)
# Background: one array (no mass column yet)
# ── 1. Load raw data ──────────────────────────────────────────────────────────

cut_expr = "(taus_n_NOSYS >= 1) * (jets_n_NOSYS >= 2) * (nbJets77_NOSYS >= 1) * (abs(Mll01_NOSYS/1.0e3 - 91.2) > 10.0) * (Mll01_NOSYS/1.0e3 > 12.0) * (leps_pt_0_NOSYS/1.0e3 > 25.0) * (leps_pt_1_NOSYS/1.0e3 > 25.0) * (taus_pt_0_NOSYS/1.0e3 >= 50.0)"

print("\n=== Loading background ===")

"""
X_bkg_raw_dict = {}
w_bkg_raw_dict = {}
X_bkg_raw = None
w_bkg_raw = None
for bkg_name, bkg_files in background_files_dict.items():
    X_bkg_raw_dict[bkg_name], w_bkg_raw_dict[bkg_name] = load_root_files(bkg_files, tree_name, pnn_features, cut_expr)
    X_bkg_raw = X_bkg_raw_dict[bkg_name] if X_bkg_raw is None else np.concatenate([X_bkg_raw, X_bkg_raw_dict[bkg_name]], axis=0)
    w_bkg_raw = w_bkg_raw_dict[bkg_name] if w_bkg_raw is None else np.concatenate([w_bkg_raw, w_bkg_raw_dict[bkg_name]], axis=0)
np.savez("pnn_xgb/X_bkg_raw_dict.npz", **X_bkg_raw_dict)
np.savez("pnn_xgb/w_bkg_raw_dict.npz", **w_bkg_raw_dict)
"""

X_bkg_raw_dict = np.load("pnn_xgb/X_bkg_raw_dict.npz")
w_bkg_raw_dict = np.load("pnn_xgb/w_bkg_raw_dict.npz")
X_bkg_raw = None
w_bkg_raw = None
for bkg_name, bkg_files in background_files_dict.items():
    X_bkg_raw = X_bkg_raw_dict[bkg_name] if X_bkg_raw is None else np.concatenate([X_bkg_raw, X_bkg_raw_dict[bkg_name]], axis=0)
    w_bkg_raw = w_bkg_raw_dict[bkg_name] if w_bkg_raw is None else np.concatenate([w_bkg_raw, w_bkg_raw_dict[bkg_name]], axis=0)


print(f"Background events: {len(X_bkg_raw)}")
X_bkg_count = len(X_bkg_raw)

print("\n=== Loading signal by mass ===")
mass_grid  = np.array(sorted(signal_by_mass.keys()), dtype=np.float32)
print(f"Masses: {mass_grid}")

# saving data at first
"""
X_sig_dict = {}
w_sig_dict = {}
for m, files in sorted(signal_by_mass.items()):
    X, w = load_root_files(files, tree_name, pnn_features, cut_expr)
    # Store using the mass as a string key
    X_sig_dict[f"mass_{m}"] = X
    w_sig_dict[f"mass_{m}"] = w
    print(f"  m={m:5d} GeV: {len(X)} events")
np.savez("pnn_xgb/X_sig_list.npz", **X_sig_dict)
np.savez("pnn_xgb/w_sig_list.npz", **w_sig_dict)
"""

X_sig_list_archive = np.load("pnn_xgb/X_sig_list.npz")
w_sig_list_archive = np.load("pnn_xgb/w_sig_list.npz")
X_sig_list = []
w_sig_list = []
for m, files in sorted(signal_by_mass.items()):
    X = X_sig_list_archive[f"mass_{m}"]
    w = w_sig_list_archive[f"mass_{m}"]
    X_sig_list.append((m, X))
    w_sig_list.append((m, w))
    print(f"  m={m:5d} GeV: {len(X)} events")

X_sig_count_bymass = {}
for m,X in X_sig_list:
    X_sig_count_bymass[m] = len(X)

X_sig_raw_all  = np.concatenate([X for _, X in X_sig_list], axis=0)
w_sig_raw_all  = np.concatenate([w for _, w in w_sig_list], axis=0)
m_sig_true_all = np.concatenate([np.full(len(X), m) for m, X in X_sig_list])



# ── 2. Split RAW data BEFORE duplication ─────────────────────────────────────
# This is critical — val and test must use unique background events
# so early stopping and evaluation are not affected by duplication noise

# Split raw background 80/10/10
Xb_temp, Xb_test_raw, wb_temp, wb_test_raw = train_test_split(X_bkg_raw, w_bkg_raw, test_size=0.10, random_state=42)
Xb_train_raw, Xb_val_raw, wb_train_raw, wb_val_raw = train_test_split(Xb_temp, wb_temp, test_size=0.10, random_state=42)

# Split raw signal 80/10/10
Xs_temp, Xs_test_raw, ms_temp, ms_test, ws_temp, ws_test_raw = train_test_split(
    X_sig_raw_all, m_sig_true_all, w_sig_raw_all, test_size=0.10, random_state=42
)
Xs_train_raw, Xs_val_raw, ms_train, ms_val, ws_train_raw, ws_val_raw = train_test_split(
    Xs_temp, ms_temp, ws_temp, test_size=0.10, random_state=42
)

print("\nRaw splits:")
print(f"  Signal     — train: {len(Xs_train_raw)}  val: {len(Xs_val_raw)}  test: {len(Xs_test_raw)}")
print(f"  Background — train: {len(Xb_train_raw)}  val: {len(Xb_val_raw)}  test: {len(Xb_test_raw)}")
print(f"  Sig Weight — train: {len(ws_train_raw)}  val: {len(ws_val_raw)}  test: {len(ws_test_raw)}")
print(f"  Bkg Weight — train: {len(wb_train_raw)}  val: {len(wb_val_raw)}  test: {len(wb_test_raw)}")

rng = np.random.default_rng(42)


# ── 3. Build TRAINING data — duplicate background across all masses ───────────

# for the training weights from the ntuples,
# we do an ABS for some reason...
ws_train_phys_raw = np.abs(ws_train_raw)
wb_train_phys_raw = np.abs(wb_train_raw)

# Signal train: append true log mass
X_sig_train = np.concatenate(
    [Xs_train_raw,
     np.log(ms_train).astype(np.float32).reshape(-1, 1)],
    axis=1
)
y_sig_train = np.ones(len(X_sig_train), dtype=np.int32)

# Per-mass equalisation weights for training signal
w_sig_train = np.ones(len(X_sig_train), dtype=np.float32)
for m, X in X_sig_list:
    mask = (ms_train == m)
    if mask.sum() > 0:
        w_sig_train[mask] = len(mass_grid) / mask.sum()
w_sig_train *= ws_train_phys_raw

# Background train: duplicate across all masses
bkg_chunks = []
bkg_w_chunks = []
for m in mass_grid:
    X_bkg_m = np.concatenate(
        [Xb_train_raw,
         np.full((len(Xb_train_raw), 1), np.log(m), dtype=np.float32)],
        axis=1
    )
    bkg_chunks.append(X_bkg_m)
    bkg_w_chunks.append(wb_train_phys_raw)


    
X_bkg_train = np.concatenate(bkg_chunks, axis=0)
y_bkg_train = np.zeros(len(X_bkg_train), dtype=np.int32)
w_bkg_train = np.full(len(X_bkg_train), 1.0 / len(mass_grid), dtype=np.float32)

w_bkg_chunks_all = np.concatenate(bkg_w_chunks, axis=0)
w_bkg_train *= w_bkg_chunks_all

# Global balance for training
total_sig_w = w_sig_train.sum()
total_bkg_w = w_bkg_train.sum()
if total_sig_w < total_bkg_w:
    w_sig_train *= (total_bkg_w / total_sig_w)
else:
    w_bkg_train *= (total_sig_w / total_bkg_w)

print(f"X_sig_train={len(X_sig_train)},   ws_train_phys_raw={len(ws_train_phys_raw)}")
print(f"X_bkg_train={len(X_sig_train)},   ws_train_phys_raw={len(ws_train_phys_raw)}")

# Stack training data
X_train = np.concatenate([X_sig_train, X_bkg_train], axis=0).astype(np.float32)
y_train = np.concatenate([y_sig_train, y_bkg_train], axis=0).astype(np.int32)
w_train = np.concatenate([w_sig_train, w_bkg_train], axis=0).astype(np.float32)
X_train, y_train, w_train = shuffle(X_train, y_train, w_train, random_state=42)

print(f"\nTraining set: {len(y_train)} rows  "
      f"(signal={y_train.sum()}, background={len(y_train)-y_train.sum()})")
print(f"Weighted signal:     {w_train[y_train==1].sum():.1f}")
print(f"Weighted background: {w_train[y_train==0].sum():.1f}")

# ── 4. Build VALIDATION data — one random mass per background event ───────────
# NOT duplicated — clean signal for reliable early stopping

ws_val_phys_raw = np.abs(ws_val_raw)
wb_val_phys_raw = np.abs(wb_val_raw)

# Signal val: append true log mass
X_sig_val = np.concatenate(
    [Xs_val_raw,
     np.log(ms_val).astype(np.float32).reshape(-1, 1)],
    axis=1
)
y_sig_val = np.ones(len(X_sig_val), dtype=np.int32)
w_sig_val = np.ones(len(X_sig_val), dtype=np.float32)

# Per-mass equalisation for val signal too
for m, X in X_sig_list:
    mask = (ms_val == m)
    if mask.sum() > 0:
        w_sig_val[mask] = len(mass_grid) / mask.sum()
w_sig_val *= ws_val_phys_raw

# Background val: one random mass per event (not duplicated)
m_val_bkg = rng.choice(mass_grid, size=len(Xb_val_raw)).astype(np.float32)
X_bkg_val = np.concatenate(
    [Xb_val_raw,
     np.log(m_val_bkg).reshape(-1, 1)],
    axis=1
)
y_bkg_val = np.zeros(len(X_bkg_val), dtype=np.int32)
w_bkg_val = wb_val_phys_raw

# Global balance for validation
total_sig_val_w = w_sig_val.sum()
total_bkg_val_w = w_bkg_val.sum()
if total_sig_val_w < total_bkg_val_w:
    w_sig_val *= (total_bkg_val_w / total_sig_val_w)
else:
    w_bkg_val *= (total_sig_val_w / total_bkg_val_w)

# Stack validation data
X_val = np.concatenate([X_sig_val, X_bkg_val], axis=0).astype(np.float32)
y_val = np.concatenate([y_sig_val, y_bkg_val], axis=0).astype(np.int32)
w_val = np.concatenate([w_sig_val, w_bkg_val], axis=0).astype(np.float32)
X_val, y_val, w_val = shuffle(X_val, y_val, w_val, random_state=42)

print(f"\nValidation set: {len(y_val)} rows  "
      f"(signal={y_val.sum()}, background={len(y_val)-y_val.sum()})")

# ── 5. Keep raw test arrays for per-mass evaluation ───────────────────────────
# Test background stays as unique raw events — no mass column yet
# Mass column appended fresh inside predict_at_mass() at inference time

X_test_raw = Xb_test_raw                    # (N_bkg_test, 24) unique background
X_sig_test_raw = Xs_test_raw                # (N_sig_test, 24) signal
m_sig_test     = ms_test                    # true mass for each test signal event
w_test_sig_raw = ws_test_raw
w_test_bkg_raw = wb_test_raw
y_test_sig     = np.ones(len(Xs_test_raw),  dtype=np.int32)
y_test_bkg     = np.zeros(len(Xb_test_raw), dtype=np.int32)

print("\nTest set (raw, no mass column):")
print(f"  Signal:     {len(X_sig_test_raw)} events")
print(f"  Background: {len(X_test_raw)} unique events")
print(f"\nFeature vector width for training: {X_train.shape[1]} "
      f"(={len(pnn_features)} physics + 1 log-mass)")

  
def predict_at_mass(model, X_features_raw, mass_value):
    """
    Score every event in X_features_raw under hypothesis mass_value (GeV).
    X_features_raw: (N, n_phys)  — NO mass column
    Returns: (N,) probability array
    """

    # now that I'm loading/reading per bkg sometimes,
    # there are 0-event files so
    # we must now return explicitly a 0-len array
    if len(X_features_raw) == 0:
        return np.array([], dtype=np.float32)
    
    X_with_m = append_log_mass(X_features_raw, mass_value)
    X_gpu = cp.array(X_with_m)
    probs = model.predict_proba(X_gpu)[:, 1].astype(np.float32)

    # also apparently might want to return
    # np array rather than cp array
    if isinstance(probs, cp.ndarray):
        return probs.get().astype(np.float32)
    return probs.astype(np.float32)


"""
def objective(trial):
    params = dict(
        n_estimators         = 20000,
        learning_rate        = trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        max_depth            = trial.suggest_int("max_depth", 4, 12),
        subsample            = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree     = 1.0,
        colsample_bylevel    = trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        gamma                = trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        reg_lambda           = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        alpha                = trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        eval_metric          = "logloss",
        early_stopping_rounds = 100,
        tree_method          = "hist",
        device               = "cuda",
        random_state         = 42,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              sample_weight          = w_train,
              eval_set               = [(X_train, y_train), (X_val, y_val)],
              sample_weight_eval_set = [w_train, w_val],
              verbose                = False,
    )
    
    X_bkg_val_raw = X_val_raw[y_val == 0]
    
    weighted_auc_sum = 0.0
    total_weight     = 0.0
    for m in mass_grid:
        mask_sig = (y_val == 1) & (np.abs(m_val - m) < 1.0)
        if mask_sig.sum() < 5:
            continue
        X_sig_m = X_val_raw[mask_sig]
        X_eval  = np.concatenate([X_sig_m, X_bkg_val_raw], axis=0)
        y_eval  = np.concatenate([np.ones(len(X_sig_m)), np.zeros(len(X_bkg_val_raw))])
        probs = predict_at_mass(model, X_eval, m)
        auc   = roc_auc_score(y_eval, probs)
        w = 1.0
        weighted_auc_sum += w * auc
        total_weight     += w
    mean_auc = weighted_auc_sum / total_weight
    return mean_auc
n_phys  = len(pnn_features)
X_val_raw = X_val[:, :n_phys]
m_val     = np.exp(X_val[:, -1])
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)
print("Best AUC:", study.best_value)
print("Best params:", study.best_params)
exit(0)
"""

# ── 4. Custom eval metric (unchanged from original) ───────────────────────────
 
def f1_eval_sklearn(y_true, y_pred):
    binary_preds = (y_pred > 0.5).astype(int)
    return -f1_score(y_true, binary_preds)
 
 
# ── 5. XGB-pNN model definition ───────────────────────────────────────────────
# One model, same hyper-parameters as your best previous model.
# You can re-run Optuna on this combined model later.
 
def build_pnn_model():
    return xgb.XGBClassifier(
        n_estimators          = 20000,
        learning_rate         = 0.015112599069587925,
        max_depth             = 12,
        subsample             = 0.983362787316825,
        colsample_bytree      = 1.0,
        colsample_bylevel     = 0.5201218423827665,
        gamma                 = 0.0006586894605906733,
        reg_lambda            = 0.04484966066283736,
        alpha                 = 0.001491180705629154,
        eval_metric           = "logloss",
        random_state          = 42,
        early_stopping_rounds = 100,
        tree_method           = "hist",
        device                = "cuda",
    )
 

# ── 6. Train ──────────────────────────────────────────────────────────────────

model_path = "pnn_xgb/pnn_model.json"
model_metrics_path = "pnn_xgb/pnn_model_results.json"
print("\n=== Training XGB-pNN ===")

"""
pnn_model = xgb.XGBClassifier()
pnn_model.load_model(model_path)
pnn_model.set_params(device="cuda")
with open(model_metrics_path, "r") as f:
    result_metrics = json.load(f)
print(f"Loaded pNN model from '{model_path}'")
"""

pnn_model = build_pnn_model()
pnn_model.fit(
    X_train, y_train,
    sample_weight          = w_train,
    eval_set               = [(X_train, y_train), (X_val, y_val)],
    sample_weight_eval_set = [w_train, w_val],
    verbose                = 100,
)
os.makedirs("pnn_xgb", exist_ok=True)
pnn_model.save_model(model_path)
result_metrics = pnn_model.evals_result()
with open(model_metrics_path, "w") as f:
    json.dump(result_metrics, f)
print(f"Saved pNN model → {model_path}")


 
# ── 7. Result metrics ───────────────────────────────────────────────────────
def plot_metrics(result_metrics):
    num_epochs = len(result_metrics["validation_0"]["logloss"])
    xaxis = range(num_epochs)
    train_loss = result_metrics["validation_0"]["logloss"]
    val_loss = result_metrics["validation_1"]["logloss"]
    
    fig,ax = plt.subplots(figsize=(8,6))
    
    ax.plot(xaxis, train_loss, label="Train loss")
    ax.plot(xaxis, val_loss, label="Validation loss")

    ax.legend()
    ax.set_xlabel("Epochs (Trees)")
    ax.set_ylabel("Log Loss")
    ax.set_title("XGBoost LQ pNN Learning Curve")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pnn_xgb/learning-curve.png", dpi=200)
    plt.close()
    
    print("Saved learning curve plot → pnn_xgb/learning_curve.png")

plot_metrics(result_metrics)
 
 
# ── 8. Sanity checks ──────────────────────────────────────────────────────────
"""
# Run these before trusting TRExFitter results.
# Combined test set
X_test_combined = np.concatenate([X_sig_test_raw, X_test_raw], axis=0)
y_test = np.concatenate([y_test_sig, y_test_bkg], axis=0).astype(np.int32)
m_test = np.concatenate([m_sig_test, np.zeros(len(X_test_raw))], axis=0)  # bkg gets 0
w_test = np.concatenate([w_test_sig_raw, w_test_bkg_raw], axis=0)

print("\n=== Sanity check 1: per-mass AUC on test set ===")
# For each mass m, evaluate signal-at-m vs all-background
 
X_bkg_test_raw = X_test_raw   # X_test_raw is already pure background

per_mass_auc = {}
for m, _ in sorted(signal_by_mass.items()):
    X_sig_test_raw = X_test_combined[(y_test == 1) & (np.abs(m_test - m) < 1.0)]
    if len(X_sig_test_raw) == 0:
        print(f"  m={m:5d}: no test events")
        continue
    X_eval = np.concatenate([X_sig_test_raw, X_bkg_test_raw], axis=0)
    y_eval = np.concatenate([np.ones(len(X_sig_test_raw)),
                             np.zeros(len(X_bkg_test_raw))])
    probs  = predict_at_mass(pnn_model, X_eval, m)
    auc    = roc_auc_score(y_eval, probs)
    per_mass_auc[m] = auc
    print(f"  m={m:5d} GeV:  AUC = {auc:.4f}  "
          f"(n_sig={len(X_sig_test_raw)}, n_bkg={len(X_bkg_test_raw)})")
 
print("\n=== Sanity check 2: background score vs mass hypothesis ===")
# Mean bkg score should be roughly flat across mass hypotheses.
# A strong trend → mass input is leaking.
X_bkg_check = X_bkg_test_raw[:5000]   # subsample for speed
for m in sorted(mass_grid):
    mean_bkg_score = predict_at_mass(pnn_model, X_bkg_check, m).mean()
    print(f"  m={m:5.0f} GeV:  mean bkg score = {mean_bkg_score:.4f}")
 
print("\n=== Sanity check 3: mass response curve for one signal mass ===")
# Pick 2000 GeV signal. Score it under ALL mass hypotheses.
# Should peak at 2000 GeV.
probe_mass = 2000
X_probe = X_test_combined[(y_test == 1) & (np.abs(m_test - probe_mass) < 1.0)]
if len(X_probe) > 0:
    print(f"  Signal at {probe_mass} GeV scored under each hypothesis:")
    for m in sorted(mass_grid):
        mean_score = predict_at_mass(pnn_model, X_probe, m).mean()
        bar = "█" * int(mean_score * 40)
        print(f"  hyp m={m:5.0f}:  {mean_score:.4f}  {bar}")
""" 
 
# ── 9. Per-mass evaluation (feeds into TRExFitter) ────────────────────────────
#
# For each mass hypothesis m:
#   - sig_prob for signal events at mass m
#   - sig_prob for ALL background events
#   - Save histograms for TRExFitter input

X_test_combined = np.concatenate([X_sig_test_raw, X_test_raw], axis=0)
y_test = np.concatenate([y_test_sig, y_test_bkg], axis=0).astype(np.int32)
m_test = np.concatenate([m_sig_test, np.zeros(len(X_test_raw))], axis=0)  # bkg gets 0
w_test = np.concatenate([w_test_sig_raw, w_test_bkg_raw], axis=0)
 
print("\n=== Per-mass evaluation for TRExFitter ===")
os.makedirs("pnn_xgb/trex_inputs", exist_ok=True)
X_bkg_test_raw = X_test_raw   # X_test_raw is already pure background
 
trex_results = {}
 
for m in sorted(signal_by_mass.keys()):
    # Signal events whose TRUE mass == m
    X_sig_m    = X_test_combined[(y_test == 1) & (np.abs(m_test - m) < 1.0)]
    y_sig_m    = np.ones(len(X_sig_m), dtype=np.int32)
    w_sig_m    = w_test[(y_test == 1) & (np.abs(m_test - m) < 1.0)]
 
    # All background test events
    X_bkg_m    = X_bkg_test_raw
    y_bkg_m    = np.zeros(len(X_bkg_m), dtype=np.int32)
    w_bkg_m    = wb_test_raw
 
    if len(X_sig_m) == 0:
        print(f"  m={m:5d}: no test signal events, skipping")
        continue
 
    # Score everything at hypothesis m
    p_sig = predict_at_mass(pnn_model, X_sig_m, m)
    p_bkg = predict_at_mass(pnn_model, X_bkg_m, m)
 
    X_combined = np.concatenate([X_sig_m, X_bkg_m], axis=0)
    y_combined = np.concatenate([y_sig_m, y_bkg_m], axis=0)
    p_combined = np.concatenate([p_sig,   p_bkg],   axis=0)
 
    auc      = roc_auc_score(y_combined, p_combined)
    best_thr, best_f1 = find_best_f1_threshold(y_combined, p_combined)
    cm, tn, fp, fn, tp = compute_conf_matrix(y_combined, p_combined, best_thr)
    s_sqrt_b = tp / np.sqrt(fp) if fp > 0 else 0.0
 
    print(f"m={m:5d} GeV | AUC={auc:.4f} | F1={best_f1:.4f} "
          f"| S/√B={s_sqrt_b:.2f} "
          f"| sig_eff={tp/(tp+fn):.3f} | bkg_rej={1-fp/(fp+tn):.3f}")
 
    # Optional: save prf1 plot

    """
    save_prf1_vs_threshold_plot(
        y_combined, p_combined,
        title=f"pNN — m={m} GeV",
        out_path=f"pnn_xgb/trex_inputs/prf1_m{m}.png"
    )
    """
 
    trex_results[m] = dict(auc=auc, best_f1=best_f1, s_sqrt_b=s_sqrt_b,
                           sig_eff=tp/(tp+fn), bkg_rej=1-fp/(fp+tn))

    # Save sig_prob arrays as histograms
    weights_sig_m = w_sig_m
    weights_bkg_m = w_test_bkg_raw
    
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins+1)
    sig_scaled,_ = np.histogram(p_sig, bins=bin_edges, weights=weights_sig_m)
    bkg_scaled,_ = np.histogram(p_bkg, bins=bin_edges, weights=weights_bkg_m)
    # sig_scaled,_ = np.histogram(p_sig, bins=bin_edges)
    # bkg_scaled,_ = np.histogram(p_bkg, bins=bin_edges)

    with uproot.recreate(f"pnn_xgb/trex_inputs/hist_m{m}.root") as f:
        f["Signal"] = (sig_scaled, bin_edges)
        f["Background"] = (bkg_scaled, bin_edges)

        # re-acquire all background and their raw weights
        # to acquire the output histograms to be used for signal separation plot
        for bkg_name,_ in background_files_dict.items():
            X_bkg, W_bkg = X_bkg_raw_dict[bkg_name], w_bkg_raw_dict[bkg_name]
            p_bkg_ind = predict_at_mass(pnn_model, X_bkg, m)
            bkg_ind_scaled,_ = np.histogram(p_bkg_ind, bins=bin_edges, weights=W_bkg)
            f[bkg_name] = (bkg_ind_scaled, bin_edges)
        
 
# ── 10. Summary table ─────────────────────────────────────────────────────────
 
print("\n\n── pNN Summary ──────────────────────────────────────────────────────")
print(f"{'Mass':>6}  {'AUC':>6}  {'F1':>6}  {'S/√B':>6}  {'SigEff':>7}  {'BkgRej':>7}")
for m in sorted(trex_results):
    r = trex_results[m]
    print(f"{m:6d}  {r['auc']:.4f}  {r['best_f1']:.4f}  "
          f"{r['s_sqrt_b']:6.2f}  {r['sig_eff']:.4f}  {r['bkg_rej']:.4f}")
 
print("\nDone. Sig_prob arrays saved to pnn_xgb/trex_inputs/ for TRExFitter.")


# ── 11. SHAP feature importance (top 10) ──────────────────────────────────────

print("\n=== SHAP feature importance ===")
feature_names = pnn_features + ["log_mass"]

# Subsample for speed — 5000 rows is enough for stable SHAP mean |values|
rng_shap = np.random.default_rng(0)
idx = rng_shap.choice(len(X_train), size=min(5000, len(X_train)), replace=False)
X_shap = X_train[idx]

explainer   = shap.TreeExplainer(pnn_model)
shap_values = explainer.shap_values(X_shap)  # (N, n_features)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature":    feature_names,
    "mean_|shap|": mean_abs_shap,
}).sort_values("mean_|shap|", ascending=False).head(10).reset_index(drop=True)

print(importance_df.to_string(index=False))

# Bar plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance_df["feature"][::-1], importance_df["mean_|shap|"][::-1])
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top 10 features — XGB-pNN")
plt.tight_layout()
os.makedirs("pnn_xgb", exist_ok=True)
plt.savefig("pnn_xgb/shap_top10.png", dpi=200)
plt.close()
print("Saved SHAP plot → pnn_xgb/shap_top10.png")

# Also save the full beeswarm summary for reference
shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                  max_display=10, show=False)
plt.tight_layout()
plt.savefig("pnn_xgb/shap_summary.png", dpi=200)
plt.close()
print("Saved SHAP summary plot → pnn_xgb/shap_summary.png")
