import os, json
import numpy as np
import uproot
import xgboost as xgb

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def extract_feature(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.max(x)) if len(x) > 0 else np.nan
    return float(x)


def load_pnn_model(model_dir):
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, "pnn_model.json"))

    with open(os.path.join(model_dir, "features.json")) as f:
        pnn_features = json.load(f)

    with open(os.path.join(model_dir, "mass_grid.json")) as f:
        mass_grid = json.load(f)   # list of floats, e.g. [250, 300, 350, ...]

    return model, pnn_features, mass_grid


def predict_at_mass(model, X_raw, mass_value):
    """Append log(mass) column and score. X_raw has NO mass column."""
    log_m   = np.full((len(X_raw), 1), np.log(float(mass_value)), dtype=np.float32)
    X_with_m = np.concatenate([X_raw, log_m], axis=1)
    return model.predict_proba(X_with_m)[:, 1].astype(np.float32)


def create_friend_ntuple_pnn(file_path, tree_name, pnn_features, mass_grid,
                              model, outdir="friend_ntuples_pnn"):
    print(f"Processing: {file_path}")
    with uproot.open(file_path) as f:
        tree = f[tree_name]
        if tree.num_entries == 0:
            print(f"Skipping empty tree: {file_path}")
            return
        df_all  = tree.arrays(library="pd")
        df_feat = tree.arrays(pnn_features, library="pd")

    for feat in pnn_features:
        df_feat[feat] = df_feat[feat].apply(extract_feature)
    df_feat = df_feat.fillna(0)

    X_raw = df_feat[pnn_features].values.astype(np.float32)  # (N, 24) — no mass col

    # One branch per mass hypothesis
    for m in mass_grid:
        probs = predict_at_mass(model, X_raw, m)
        branch_name = f"ml_prob_m{int(m)}"   # e.g. ml_prob_m250, ml_prob_m500
        df_all[branch_name] = probs

    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, "friend_" + os.path.basename(file_path))
    with uproot.recreate(out_file) as f:
        f[tree_name] = df_all

    print(f"Saved: {out_file}  ({len(mass_grid)} ml_prob branches written)")



def main(model_dir, files, tree_name="reco"):
    model, pnn_features, mass_grid = load_pnn_model(model_dir)

    for fp in files:
        create_friend_ntuple_pnn(
            file_path      = fp,
            tree_name      = tree_name,
            pnn_features   = pnn_features,
            mass_grid      = mass_grid,
            model          = model,
        )


if __name__ == "__main__":
    model_dir = "pnn_xgb"

    control_ntuples = "/home/aafandiz/fast_frame_taux/taux_fastframes/ntuples_2l1tau_v17"
    files = [
        os.path.join(control_ntuples, f)
        for f in os.listdir(control_ntuples)
        if os.path.isfile(os.path.join(control_ntuples, f))
        and not os.path.isfile(os.path.join("friend_ntuples_pnn", "friend_" + f))
    ]

    main(model_dir, files)