import os
import sklearn
import uproot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
# import optuna
# import optuna.visualization as vis
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib
import shap
import pandas as pd
import plotly.express as px
from imblearn.over_sampling import SMOTE



import torch
import torch.nn as nn
import torch.nn.functional as F



base = "C:\\Users\\ASUS\\Desktop\\ntuples_2lSS1tau_v17"

# -----------------------------
# new tbH+ signals
# -----------------------------
signal_file_1 = [
    f"{base}/tbH+_AF3_512185_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_512185_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_512185_mc20e_fastsim.root",
]

signal_file_2 = [
    f"{base}/tbH+_AF3_512186_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_512186_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_512186_mc20e_fastsim.root",
]

signal_file_3 = [
    f"{base}/tbH+_AF3_512187_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_512187_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_512187_mc20e_fastsim.root",
]

signal_file_4 = [
    f"{base}/tbH+_AF3_567608_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567608_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567608_mc20e_fastsim.root",
]

signal_file_5 = [
    f"{base}/tbH+_AF3_567609_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567609_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567609_mc20e_fastsim.root",
]

signal_file_6 = [
    f"{base}/tbH+_AF3_567610_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567610_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567610_mc20e_fastsim.root",
]

signal_file_7 = [
    f"{base}/tbH+_AF3_567611_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567611_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567611_mc20e_fastsim.root",
]

signal_file_8 = [
    f"{base}/tbH+_AF3_567613_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567613_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567613_mc20e_fastsim.root",
]

signal_file_9 = [
    f"{base}/tbH+_AF3_567614_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567614_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567614_mc20e_fastsim.root",
]

signal_file_10 = [
    f"{base}/tbH+_AF3_567615_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567615_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567615_mc20e_fastsim.root",
]

signal_file_11 = [
    f"{base}/tbH+_AF3_567616_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567616_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567616_mc20e_fastsim.root",
]

signal_file_12 = [
    f"{base}/tbH+_AF3_567617_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567617_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567617_mc20e_fastsim.root",
]

signal_file_13 = [
    f"{base}/tbH+_AF3_567618_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567618_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567618_mc20e_fastsim.root",
]

signal_file_14 = [
    f"{base}/tbH+_AF3_567619_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567619_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567619_mc20e_fastsim.root",
]

signal_file_15 = [
    f"{base}/tbH+_AF3_567607_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567607_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567607_mc20e_fastsim.root",
]

signal_file_16 = [
    f"{base}/tbH+_AF3_567612_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567612_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567612_mc20e_fastsim.root",
]


signal_file_17 = [
    f"{base}/tbH+_AF3_567620_mc20a_fastsim.root",
    f"{base}/tbH+_AF3_567620_mc20d_fastsim.root",
    f"{base}/tbH+_AF3_567620_mc20e_fastsim.root",
]

# -----------------------------
# backgrounds
# -----------------------------
ttH = [
    f"{base}/ttH_346343_mc20a_fullsim.root",
    f"{base}/ttH_346343_mc20d_fullsim.root",
    f"{base}/ttH_346343_mc20e_fullsim.root",

    f"{base}/ttH_346344_mc20a_fullsim.root",
    f"{base}/ttH_346344_mc20d_fullsim.root",
    f"{base}/ttH_346344_mc20e_fullsim.root",

    f"{base}/ttH_346345_mc20a_fullsim.root",
    f"{base}/ttH_346345_mc20d_fullsim.root",
    f"{base}/ttH_346345_mc20e_fullsim.root",
]

ttW = [
    f"{base}/ttW_700168_mc20a_fullsim.root",
    f"{base}/ttW_700168_mc20d_fullsim.root",
    f"{base}/ttW_700168_mc20e_fullsim.root",
]

ttZ = [
    f"{base}/ttZ_410276_mc20a_fullsim.root",
    f"{base}/ttZ_410276_mc20d_fullsim.root",
    f"{base}/ttZ_410276_mc20e_fullsim.root",

    f"{base}/ttZ_410277_mc20a_fullsim.root",
    f"{base}/ttZ_410277_mc20d_fullsim.root",
    f"{base}/ttZ_410277_mc20e_fullsim.root",

    f"{base}/ttZ_410278_mc20a_fullsim.root",
    f"{base}/ttZ_410278_mc20d_fullsim.root",
    f"{base}/ttZ_410278_mc20e_fullsim.root",
]

ttbar = [
    f"{base}/ttbar_410470_mc20a_fullsim.root",
    f"{base}/ttbar_410470_mc20d_fullsim.root",
    f"{base}/ttbar_410470_mc20e_fullsim.root",

]


fakeTau_ttbar = [
    f"{base}/FakeTau_ttbar_410470_mc20a_fullsim.root",
    f"{base}/FakeTau_ttbar_410470_mc20d_fullsim.root",
    f"{base}/FakeTau_ttbar_410470_mc20e_fullsim.root",

]
# Fake tau backgrounds
fakeTau_Wjets = [
    f"{base}/FakeTau_Wjets_700338_mc20a_fullsim.root",
    f"{base}/FakeTau_Wjets_700338_mc20d_fullsim.root",
    f"{base}/FakeTau_Wjets_700338_mc20e_fullsim.root",
    f"{base}/FakeTau_Wjets_700339_mc20a_fullsim.root",
    f"{base}/FakeTau_Wjets_700339_mc20d_fullsim.root",
    f"{base}/FakeTau_Wjets_700339_mc20e_fullsim.root",
    f"{base}/FakeTau_Wjets_700340_mc20a_fullsim.root",
    f"{base}/FakeTau_Wjets_700340_mc20d_fullsim.root",
    f"{base}/FakeTau_Wjets_700340_mc20e_fullsim.root",
    f"{base}/FakeTau_Wjets_700341_mc20a_fullsim.root",
    f"{base}/FakeTau_Wjets_700341_mc20d_fullsim.root",
    f"{base}/FakeTau_Wjets_700341_mc20e_fullsim.root",
    f"{base}/FakeTau_Wjets_700342_mc20a_fullsim.root",
    f"{base}/FakeTau_Wjets_700342_mc20d_fullsim.root",
    f"{base}/FakeTau_Wjets_700342_mc20e_fullsim.root",
    f"{base}/FakeTau_Wjets_700343_mc20a_fullsim.root",
    f"{base}/FakeTau_Wjets_700343_mc20d_fullsim.root",
    f"{base}/FakeTau_Wjets_700343_mc20e_fullsim.root",
    f"{base}/FakeTau_Wjets_700344_mc20a_fullsim.root",
    f"{base}/FakeTau_Wjets_700344_mc20d_fullsim.root",
    f"{base}/FakeTau_Wjets_700344_mc20e_fullsim.root",
    f"{base}/FakeTau_Wjets_700345_mc20a_fullsim.root",
    f"{base}/FakeTau_Wjets_700345_mc20d_fullsim.root",
]

fakeTau_Zjets = [
    f"{base}/FakeTau_Zjets_700321_mc20a_fullsim.root",
    f"{base}/FakeTau_Zjets_700321_mc20d_fullsim.root",
    f"{base}/FakeTau_Zjets_700321_mc20e_fullsim.root",
    f"{base}/FakeTau_Zjets_700322_mc20a_fullsim.root",
    f"{base}/FakeTau_Zjets_700322_mc20d_fullsim.root",
    f"{base}/FakeTau_Zjets_700322_mc20e_fullsim.root",
    f"{base}/FakeTau_Zjets_700323_mc20a_fullsim.root",
    f"{base}/FakeTau_Zjets_700323_mc20d_fullsim.root",
    f"{base}/FakeTau_Zjets_700323_mc20e_fullsim.root",
    f"{base}/FakeTau_Zjets_700324_mc20a_fullsim.root",
    f"{base}/FakeTau_Zjets_700324_mc20d_fullsim.root",
    f"{base}/FakeTau_Zjets_700324_mc20e_fullsim.root",
    f"{base}/FakeTau_Zjets_700325_mc20a_fullsim.root",
    f"{base}/FakeTau_Zjets_700325_mc20d_fullsim.root",
]

vv = [
    f"{base}/VV_700589_mc20a_fullsim.root",
    f"{base}/VV_700591_mc20a_fullsim.root",
    f"{base}/VV_700592_mc20a_fullsim.root",
    f"{base}/VV_700592_mc20d_fullsim.root",
    f"{base}/VV_700594_mc20a_fullsim.root",
    f"{base}/VV_700594_mc20d_fullsim.root",
    f"{base}/VV_700594_mc20e_fullsim.root",
    f"{base}/VV_700603_mc20a_fullsim.root",
    f"{base}/VV_700603_mc20d_fullsim.root",
    f"{base}/VV_700603_mc20e_fullsim.root",
    f"{base}/VV_700604_mc20a_fullsim.root",
    f"{base}/VV_700604_mc20d_fullsim.root",
    f"{base}/VV_700604_mc20e_fullsim.root",
]

# vvv = [
#     f"{base}/VVV_364242_mc20a_fullsim.root",
#     f"{base}/VVV_364242_mc20d_fullsim.root",
#     f"{base}/VVV_364242_mc20e_fullsim.root",
#     f"{base}/VVV_364243_mc20a_fullsim.root",
#     f"{base}/VVV_364243_mc20d_fullsim.root",
#     f"{base}/VVV_364243_mc20e_fullsim.root",
#     f"{base}/VVV_364245_mc20a_fullsim.root",
#     f"{base}/VVV_364245_mc20d_fullsim.root",
#     f"{base}/VVV_364245_mc20e_fullsim.root",
#     f"{base}/VVV_364246_mc20a_fullsim.root",
#     f"{base}/VVV_364246_mc20d_fullsim.root",
#     f"{base}/VVV_364246_mc20e_fullsim.root",
#     f"{base}/VVV_364247_mc20a_fullsim.root",
#     f"{base}/VVV_364247_mc20d_fullsim.root",
#     f"{base}/VVV_364247_mc20e_fullsim.root",
# ]

vh = [
    f"{base}/VH_346645_mc20a_fullsim.root",
    f"{base}/VH_346645_mc20d_fullsim.root",
    f"{base}/VH_346645_mc20e_fullsim.root",
    f"{base}/VH_346646_mc20a_fullsim.root",
    f"{base}/VH_346646_mc20d_fullsim.root",
    f"{base}/VH_346646_mc20e_fullsim.root",
]


vgamma = [
    f"{base}/Vgamma_700398_mc20a_fullsim.root",
    f"{base}/Vgamma_700398_mc20d_fullsim.root",
    f"{base}/Vgamma_700398_mc20e_fullsim.root",
    f"{base}/Vgamma_700399_mc20a_fullsim.root",
    f"{base}/Vgamma_700399_mc20d_fullsim.root",
    f"{base}/Vgamma_700399_mc20e_fullsim.root",
    f"{base}/Vgamma_700400_mc20a_fullsim.root",
    f"{base}/Vgamma_700400_mc20d_fullsim.root",
    f"{base}/Vgamma_700400_mc20e_fullsim.root",
    f"{base}/Vgamma_700402_mc20a_fullsim.root",
    f"{base}/Vgamma_700402_mc20d_fullsim.root",
    f"{base}/Vgamma_700402_mc20e_fullsim.root",
    f"{base}/Vgamma_700403_mc20a_fullsim.root",
    f"{base}/Vgamma_700403_mc20d_fullsim.root",
    f"{base}/Vgamma_700403_mc20e_fullsim.root",
    f"{base}/Vgamma_700404_mc20a_fullsim.root",
    f"{base}/Vgamma_700404_mc20d_fullsim.root",
    f"{base}/Vgamma_700404_mc20e_fullsim.root",
]

wjets = [
    f"{base}/Wjets_700338_mc20d_fullsim.root",
    f"{base}/Wjets_700338_mc20e_fullsim.root",
    f"{base}/Wjets_700341_mc20d_fullsim.root",
]




# Rare tops


ttHH = [
    f"{base}/ttHH_500460_mc20a_fastsim.root",
    f"{base}/ttHH_500460_mc20d_fastsim.root",
    f"{base}/ttHH_500460_mc20e_fastsim.root",
]

ttWH = [
    f"{base}/ttWH_500461_mc20a_fastsim.root",
    f"{base}/ttWH_500461_mc20d_fastsim.root",
    f"{base}/ttWH_500461_mc20e_fastsim.root",
]

ttWW = [
    f"{base}/ttWW_410081_mc20a_fullsim.root",
    f"{base}/ttWW_410081_mc20d_fullsim.root",
    f"{base}/ttWW_410081_mc20e_fullsim.root",
]

ttWZ = [
    f"{base}/ttWZ_500463_mc20a_fastsim.root",
    f"{base}/ttWZ_500463_mc20d_fastsim.root",
    f"{base}/ttWZ_500463_mc20e_fastsim.root",
]

ttZZ = [
    f"{base}/ttZZ_500462_mc20a_fastsim.root",
    f"{base}/ttZZ_500462_mc20d_fastsim.root",
    f"{base}/ttZZ_500462_mc20e_fastsim.root",
]

ttgamma = [
    f"{base}/ttgamma_500800_mc20a_fullsim.root",
    f"{base}/ttgamma_500800_mc20d_fullsim.root",
    f"{base}/ttgamma_500800_mc20e_fullsim.root",
    f"{base}/ttgamma_504554_mc20a_fullsim.root",
    f"{base}/ttgamma_504554_mc20d_fullsim.root",
    f"{base}/ttgamma_504554_mc20e_fullsim.root",
]

# Other rare processes
tZ = [
    f"{base}/tZ_410560_mc20a_fullsim.root",
    f"{base}/tZ_410560_mc20d_fullsim.root",
    f"{base}/tZ_410560_mc20e_fullsim.root",
]

WtZ = [
    f"{base}/WtZ_410408_mc20a_fullsim.root",
    f"{base}/WtZ_410408_mc20d_fullsim.root",
    f"{base}/WtZ_410408_mc20e_fullsim.root",
]

fourTop = [
    f"{base}/fourTop_412043_mc20a_fastsim.root",
    f"{base}/fourTop_412043_mc20d_fastsim.root",
    f"{base}/fourTop_412043_mc20e_fastsim.root",
]


singleTop = [
    

    f"{base}/SingleTop_410644_mc20d_fullsim.root",
    f"{base}/SingleTop_410644_mc20e_fullsim.root",


    f"{base}/SingleTop_410645_mc20d_fullsim.root",
    f"{base}/SingleTop_410645_mc20e_fullsim.root",





    f"{base}/SingleTop_410659_mc20a_fullsim.root",
    f"{base}/SingleTop_410659_mc20d_fullsim.root",
    f"{base}/SingleTop_410659_mc20e_fullsim.root",

]


threeTop = [
    f"{base}/threeTop_304014_mc20a_fullsim.root",
    f"{base}/threeTop_304014_mc20d_fullsim.root",
    f"{base}/threeTop_304014_mc20e_fullsim.root",
]



background_files = ttH+ ttW+ ttZ + ttbar + fakeTau_Wjets + fakeTau_Zjets + vv  + vh + vgamma + wjets + ttHH + ttWH + ttWW + ttWZ + ttZZ + ttgamma + tZ + WtZ + fakeTau_ttbar + fourTop + singleTop + threeTop

signal_files_all = signal_file_1 +signal_file_2 + signal_file_3 +signal_file_4 + signal_file_5 + signal_file_6 +signal_file_7 +signal_file_8 +signal_file_9 +signal_file_10 +signal_file_11 +signal_file_12 +signal_file_13 + signal_file_14 + signal_file_15 + signal_file_16 + signal_file_17

signal_low_region = signal_file_1 + signal_file_4 + signal_file_5 + signal_file_15 # 250 - 400 GeV

signal_high_region = signal_file_3 + signal_file_6 + signal_file_7 + signal_file_16  # 500 - 800 GeV 





tree_name = "reco"

selected_features_high = [
    #"tau_quality_0_NOSYS",
    "min_DeltaR_tau0_SSlepton_NOSYS",
    "tau_pt_0_NOSYS",
    "meff_event_NOSYS",
    "sumPsbtag_NOSYS",
    "b0_pt_NOSYS",
    "taus_RNNJetScoreSigTrans_0_NOSYS",
    "lep_d0sig_1_NOSYS",
    "bjet_1_pt_NOSYS",
    "InvMass_tau_min_SSlepton_NOSYS",
    "lep_d0sig_0_NOSYS",
    "DeltaR_L0_L1_NOSYS",
    "HT_leptons_NOSYS",
    "DeltaR_b0_b1_NOSYS",
    "lep_pt_0_NOSYS",
    "reco_Hplus_visH_hadW_mass_NOSYS",
    "minDeltaR_LJ_0_NOSYS",
    "top_reco_mass_l0b0_NOSYS",
    "Mt_tau0_MET_NOSYS",
    #"electron_quality_0_NOSYS",
]



selected_features_low = [
    #"tau_quality_0_NOSYS",
    "sumPsbtag_NOSYS",
    "bjet_1_pt_NOSYS",
    "taus_RNNJetScoreSigTrans_0_NOSYS",
    "lep_d0sig_1_NOSYS",
    "tau_pt_0_NOSYS",
    "b0_pt_NOSYS",
    "dEta_maxMjj_frwdjet_NOSYS",
    "HT_NOSYS",
    #"electron_quality_0_NOSYS", 
    "DeltaR_b0_b1_NOSYS",
    "mT_vis_h_MET_NOSYS",
    "top_reco_mass_l0b0_NOSYS",
    "Mll01_NOSYS",
    "MLepMet_NOSYS",
    "InvMass_tau_min_SSlepton_NOSYS",
    "jet_pt_0_NOSYS",
    "lep_d0sig_0_NOSYS",
    "min_DeltaR_tau0_SSlepton_NOSYS",
    #"electron_quality_1_NOSYS"
]


signal_by_mass = {
    250:  signal_file_1,
    3000: signal_file_2,
    800:  signal_file_3,
    350:  signal_file_4,
    400:  signal_file_5,
    500:  signal_file_6,
    600:  signal_file_7,
    900:  signal_file_8,
    1000: signal_file_9,
    1200: signal_file_10,
    1400: signal_file_11,
    1600: signal_file_12,
    1800: signal_file_13,
    2000: signal_file_14,
    300:  signal_file_15,
    700:  signal_file_16,
    2500: signal_file_17,
}

pnn_features = list(dict.fromkeys(selected_features_low + selected_features_high))
# The pNN model will see these features PLUS log(m) appended as the last column.
# Total input width = len(pnn_features) + 1
 
print(f"pNN feature set: {len(pnn_features)} physics features + 1 mass column")

exit()
 
# ── Unchanged helpers ─────────────────────────────────────────────────────────
 
def extract_feature(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.max(x)) if len(x) > 0 else np.nan
    return float(x)
 
 
def load_root_files(file_paths, tree_name, features):
    """Load ROOT files and return a plain feature matrix (no mass column)."""
    chunks = []
    for fp in file_paths:
        print(f"  Loading: {fp}")
        with uproot.open(fp) as f:
            df = f[tree_name].arrays(features, library="pd")
        for feat in features:
            df[feat] = df[feat].apply(extract_feature)
        df = df.fillna(0)
        chunks.append(df[features].values.astype(np.float32))
    if not chunks:
        return np.empty((0, len(features)), dtype=np.float32)
    return np.concatenate(chunks, axis=0)
 
 
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
    ax1.set_xlabel("Threshold"); ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, sigs, "-", label="TP/sqrt(FP)")
    ax2.axvline(thresholds[best_sig_idx], color="black", linestyle=":",
                label=f"Best S/sqrt(B) thr ({thresholds[best_sig_idx]:.2f})")
    ax2.set_ylabel("TP/sqrt(FP)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center")
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    return thresholds[best_f1_idx], f1s[best_f1_idx], thresholds[best_sig_idx], sigs[best_sig_idx]
 
 
# ── 1. Load data ──────────────────────────────────────────────────────────────
# Signal: one array per mass point (no mass column yet)
# Background: one array (no mass column yet)
# ── 1. Load raw data ──────────────────────────────────────────────────────────

print("\n=== Loading background ===")
X_bkg_raw = load_root_files(background_files, tree_name, pnn_features)
print(f"Background events: {len(X_bkg_raw)}")

print("\n=== Loading signal by mass ===")
mass_grid  = np.array(sorted(signal_by_mass.keys()), dtype=np.float32)
X_sig_list = []
for m, files in sorted(signal_by_mass.items()):
    X = load_root_files(files, tree_name, pnn_features)
    X_sig_list.append((m, X))
    print(f"  m={m:5d} GeV: {len(X)} events")

X_sig_raw_all  = np.concatenate([X for _, X in X_sig_list], axis=0)
m_sig_true_all = np.concatenate([np.full(len(X), m) for m, X in X_sig_list])

# ── 2. Split RAW data BEFORE duplication ─────────────────────────────────────
# This is critical — val and test must use unique background events
# so early stopping and evaluation are not affected by duplication noise

# Split raw background 80/10/10
Xb_temp, Xb_test_raw = train_test_split(X_bkg_raw, test_size=0.10, random_state=42)
Xb_train_raw, Xb_val_raw = train_test_split(Xb_temp, test_size=0.10, random_state=42)

# Split raw signal 80/10/10
Xs_temp, Xs_test_raw, ms_temp, ms_test = train_test_split(
    X_sig_raw_all, m_sig_true_all, test_size=0.10, random_state=42
)
Xs_train_raw, Xs_val_raw, ms_train, ms_val = train_test_split(
    Xs_temp, ms_temp, test_size=0.10, random_state=42
)

print(f"\nRaw splits:")
print(f"  Signal   — train: {len(Xs_train_raw)}  val: {len(Xs_val_raw)}  test: {len(Xs_test_raw)}")
print(f"  Background — train: {len(Xb_train_raw)}  val: {len(Xb_val_raw)}  test: {len(Xb_test_raw)}")

rng = np.random.default_rng(42)

# ── 3. Build TRAINING data — duplicate background across all masses ───────────

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

# Background train: duplicate across all 17 masses
bkg_chunks = []
for m in mass_grid:
    X_bkg_m = np.concatenate(
        [Xb_train_raw,
         np.full((len(Xb_train_raw), 1), np.log(m), dtype=np.float32)],
        axis=1
    )
    bkg_chunks.append(X_bkg_m)

X_bkg_train = np.concatenate(bkg_chunks, axis=0)
y_bkg_train = np.zeros(len(X_bkg_train), dtype=np.int32)
w_bkg_train = np.full(len(X_bkg_train), 1.0 / len(mass_grid), dtype=np.float32)

# Global balance for training
total_sig_w = w_sig_train.sum()
total_bkg_w = w_bkg_train.sum()
if total_sig_w < total_bkg_w:
    w_sig_train *= (total_bkg_w / total_sig_w)
else:
    w_bkg_train *= (total_sig_w / total_bkg_w)

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

# Background val: one random mass per event (not duplicated)
m_val_bkg = rng.choice(mass_grid, size=len(Xb_val_raw)).astype(np.float32)
X_bkg_val = np.concatenate(
    [Xb_val_raw,
     np.log(m_val_bkg).reshape(-1, 1)],
    axis=1
)
y_bkg_val = np.zeros(len(X_bkg_val), dtype=np.int32)
w_bkg_val = np.ones(len(X_bkg_val), dtype=np.float32)

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
y_test_sig     = np.ones(len(Xs_test_raw),  dtype=np.int32)
y_test_bkg     = np.zeros(len(Xb_test_raw), dtype=np.int32)

print(f"\nTest set (raw, no mass column):")
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
    X_with_m = append_log_mass(X_features_raw, mass_value)
    return model.predict_proba(X_with_m)[:, 1].astype(np.float32)


# import optuna
# from sklearn.metrics import roc_auc_score
# import numpy as np
# import xgboost as xgb

# def objective(trial):
#     params = dict(
#         n_estimators         = 20000,
#         learning_rate        = trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
#         max_depth            = trial.suggest_int("max_depth", 4, 12),
#         subsample            = trial.suggest_float("subsample", 0.5, 1.0),
#         colsample_bytree     = 1.0,
#         colsample_bylevel    = trial.suggest_float("colsample_bylevel", 0.5, 1.0),
#         gamma                = trial.suggest_float("gamma", 1e-4, 5.0, log=True),
#         reg_lambda           = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
#         alpha                = trial.suggest_float("alpha", 1e-4, 10.0, log=True),
#         use_label_encoder    = False,
#         eval_metric          = "logloss",
#         early_stopping_rounds = 100,
#         tree_method          = "hist",
#         device               = "cuda",
#         random_state         = 42,
#     )
#     model = xgb.XGBClassifier(**params)
#     model.fit(
#         X_train, y_train,
#         eval_set               = [(X_val, y_val)],
#         verbose                = 100,
#     )
#     X_bkg_val_raw = X_val_raw[y_val == 0]
#     mass_weights = {
#         250:  3.0, 300:  3.0, 350:  2.0, 400:  2.0, 500:  1.5,
#         600:  1.0, 700:  1.0, 800:  1.0, 900:  1.0, 1000: 1.0,
#         1200: 1.0, 1400: 1.0, 1600: 1.0, 1800: 1.0, 2000: 1.0,
#         2500: 1.0, 3000: 1.0,
#     }
#     weighted_auc_sum = 0.0
#     total_weight     = 0.0
#     for m in mass_grid:
#         mask_sig = (y_val == 1) & (np.abs(m_val - m) < 1.0)
#         if mask_sig.sum() < 5:
#             continue
#         X_sig_m = X_val_raw[mask_sig]
#         X_eval  = np.concatenate([X_sig_m, X_bkg_val_raw], axis=0)
#         y_eval  = np.concatenate([np.ones(len(X_sig_m)), np.zeros(len(X_bkg_val_raw))])
#         probs = predict_at_mass(model, X_eval, m)
#         auc   = roc_auc_score(y_eval, probs)
#         w = mass_weights.get(m, 1.0)
#         weighted_auc_sum += w * auc
#         total_weight     += w
#     mean_auc = weighted_auc_sum / total_weight
#     return mean_auc

# n_phys  = len(pnn_features)
# X_val_raw = X_val[:, :n_phys]
# m_val     = np.exp(X_val[:, -1])

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100, show_progress_bar=True)
# print("Best AUC:", study.best_value)
# print("Best params:", study.best_params)


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
        learning_rate         = 0.006089739621234152,
        max_depth             = 8,
        subsample             = 0.9029562810593745,
        colsample_bytree      = 1.0,
        colsample_bylevel     = 0.9270543452734271,
        gamma                 = 0.001145678134391421,
        reg_lambda            = 0.22838811632234002,
        alpha                 = 9.627164038297993,
        use_label_encoder     = False,
        eval_metric           = "logloss",
        random_state          = 42,
        early_stopping_rounds = 100,
        tree_method           = "hist",
        predictor             = "gpu_predictor",
        device                = "cuda",
    )
 

# ── 6. Train ──────────────────────────────────────────────────────────────────
 
print("\n=== Training XGB-pNN ===")
pnn_model = build_pnn_model()
pnn_model.fit(
    X_train, y_train,
    sample_weight          = w_train,
    eval_set               = [(X_val, y_val)],
    sample_weight_eval_set = [w_val],
    verbose                = 100,
)
 
os.makedirs("pnn_xgb", exist_ok=True)
pnn_model.save_model("pnn_xgb/pnn_model.json")
print("Saved pNN model → pnn_xgb/pnn_model.json")
 
 
# ── 7. Inference helper ───────────────────────────────────────────────────────

 
 
# ── 8. Sanity checks ──────────────────────────────────────────────────────────
# Run these before trusting TRExFitter results.
# Combined test set
X_test_combined = np.concatenate([X_sig_test_raw, X_test_raw], axis=0)
y_test = np.concatenate([y_test_sig, y_test_bkg], axis=0).astype(np.int32)
m_test = np.concatenate([m_sig_test, np.zeros(len(X_test_raw))], axis=0)  # bkg gets 0

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
# Pick 500 GeV signal. Score it under ALL mass hypotheses.
# Should peak at 500 GeV.
probe_mass = 500
X_probe = X_test_combined[(y_test == 1) & (np.abs(m_test - probe_mass) < 1.0)]
if len(X_probe) > 0:
    print(f"  Signal at {probe_mass} GeV scored under each hypothesis:")
    for m in sorted(mass_grid):
        mean_score = predict_at_mass(pnn_model, X_probe, m).mean()
        bar = "█" * int(mean_score * 40)
        print(f"  hyp m={m:5.0f}:  {mean_score:.4f}  {bar}")
 
 
# ── 9. Per-mass evaluation (feeds into TRExFitter) ────────────────────────────
#
# For each mass hypothesis m:
#   - sig_prob for signal events at mass m
#   - sig_prob for ALL background events
#   - Save histograms for TRExFitter input
 
#print("\n=== Per-mass evaluation for TRExFitter ===")
#os.makedirs("pnn_xgb/trex_inputs", exist_ok=True)
 
trex_results = {}
 
for m in sorted(signal_by_mass.keys()):
    # Signal events whose TRUE mass == m
    X_sig_m    = X_test_combined[(y_test == 1) & (np.abs(m_test - m) < 1.0)]
    y_sig_m    = np.ones(len(X_sig_m), dtype=np.int32)
 
    # All background test events
    X_bkg_m    = X_bkg_test_raw
    y_bkg_m    = np.zeros(len(X_bkg_m), dtype=np.int32)
 
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
 
    print(f"\n  m={m:5d} GeV | AUC={auc:.4f} | F1={best_f1:.4f} "
          f"| S/√B={s_sqrt_b:.2f} "
          f"| sig_eff={tp/(tp+fn):.3f} | bkg_rej={1-fp/(fp+tn):.3f}")
 
    # Save sig_prob arrays as numpy files for TRExFitter histogram building
 
    # Optional: save prf1 plot
    save_prf1_vs_threshold_plot(
        y_combined, p_combined,
        title=f"pNN — m={m} GeV",
        out_path=f"pnn_xgb/trex_inputs/prf1_m{m}.png"
    )
 
    trex_results[m] = dict(auc=auc, best_f1=best_f1, s_sqrt_b=s_sqrt_b,
                           sig_eff=tp/(tp+fn), bkg_rej=1-fp/(fp+tn))
 
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
