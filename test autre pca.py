from __future__ import annotations
import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib.patches import Ellipse

# =========================
# CONFIG (Adapté à ton Mac)
# =========================
DATA_DIR = '/Users/mac/Library/Mobile Documents/com~apple~CloudDocs/Session H26/TPOP Projet 1 /Données'
FILE_GLOB = "**/*.TXT" 

# --- AJOUT : TON FICHIER TEST TERRAIN ---
# Chemin vérifié vers ton fichier MIX-mid-3.TXT
NEW_SAMPLE_PATH = '/Users/mac/Library/Mobile Documents/com~apple~CloudDocs/Session H26/TPOP Projet 1 /Données/B2-Bas-2.TXT'

# Choix prétraitements
USE_BASELINE_ALS = True
USE_SMOOTHING = True
USE_NORMALIZATION = True     
USE_STANDARDIZE_FOR_PCA = True  
USE_BLANK_SUBTRACTION = True 

# PCA
N_COMPONENTS = 5

# Domaine spectral (Focus sur les pics chimiques)
WAVENUMBER_MIN = 450   
WAVENUMBER_MAX = 1650  

LABEL_RULES = [
    (r"B2", "Urine_Base"),
    (r"Pseudo-seul", "Pseudo_Pure"),
    (r"trace", "Pseudo_Traces"),
    (r"MIX-haut|MIX-mid", "Pseudo_Concentrée")
]

# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Spectrum:
    path: str
    name: str
    label: str
    x: np.ndarray  
    y: np.ndarray  

# =========================
# IO + PARSING
# =========================
def infer_label_from_name(filename: str, rules=LABEL_RULES) -> str:
    low = filename.lower()
    for pattern, lab in rules:
        if re.search(pattern.lower(), low):
            return lab
    return "Unknown"

def read_txt_spectrum(path: str):
    # Charge les colonnes 2 (Raman Shift) et 3 (Intensité)
    data = np.loadtxt(path, delimiter=",", usecols=(1, 2))
    x, y = data[:, 0], data[:, 1]
    if x[0] > x[-1]: x, y = x[::-1], y[::-1]
    return x, y

def load_library(data_dir: str, file_glob: str) -> List[Spectrum]:
    paths = [p for p in glob.glob(os.path.join(DATA_DIR, FILE_GLOB), recursive=True) 
         if "testo" not in p.lower() and "calibration" not in p.lower()]
    if not paths:
        raise FileNotFoundError(f"No .txt files found in: {data_dir}")
    spectra = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        label = infer_label_from_name(name)
        x, y = read_txt_spectrum(p)
        spectra.append(Spectrum(path=p, name=name, label=label, x=x, y=y))
    return spectra

# =========================
# PREPROCESSING
# =========================
def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.001, niter: int = 15) -> np.ndarray:
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def preprocess_y(x, y, use_baseline, use_smoothing, use_norm, wn_min, wn_max):
    # CROP : On s'assure que le masque s'applique aux deux vecteurs en même temps
    mask = (x >= (wn_min or x.min())) & (x <= (wn_max or x.max()))
    if not np.any(mask): mask = np.ones(len(x), dtype=bool)
    
    x_proc = x[mask].copy()
    y_proc = y[mask].astype(float).copy()

    if use_baseline:
        y_proc = y_proc - baseline_als(y_proc)
    if use_smoothing:
        win = min(15, len(y_proc) - 1 if (len(y_proc) % 2 == 0) else len(y_proc))
        if win >= 3: 
            y_proc = savgol_filter(y_proc, window_length=win, polyorder=2)
    if use_norm:
        y_proc = (y_proc - np.mean(y_proc)) / (np.std(y_proc) + 1e-12)
        
    return x_proc, y_proc

# =========================
# ALIGNMENT / RESAMPLING
# =========================
def build_common_grid(spectra: List[Spectrum], n_points: int = 1500) -> np.ndarray:
    xmin = max(sp.x.min() for sp in spectra)
    xmax = min(sp.x.max() for sp in spectra)
    return np.linspace(xmin, xmax, n_points)

# =========================
# MAIN ANALYSIS
# =========================
def main():
    print(f"Loading library from: {DATA_DIR}")
    spectra = load_library(DATA_DIR, FILE_GLOB)
    grid = build_common_grid(spectra, n_points=1500)

    X_list, labels, names = [], [], []
    for sp in spectra:
        # SÉCURITÉ : On récupère x et y déjà coupés et synchronisés
        x_p, y_p = preprocess_y(sp.x, sp.y, USE_BASELINE_ALS, USE_SMOOTHING, 
                                USE_NORMALIZATION, WAVENUMBER_MIN, WAVENUMBER_MAX)
        X_list.append(np.interp(grid, x_p, y_p))
        labels.append(sp.label)
        names.append(sp.name)
    
    X = np.vstack(X_list)

    # --- SOUSTRACTION DU BLANK (B2) ---
    blank_mean = 0
    if USE_BLANK_SUBTRACTION:
        idx_b = [i for i, l in enumerate(labels) if "Control_Blank" in l]
        if idx_b:
            blank_mean = np.mean(X[idx_b], axis=0)
            X = X - blank_mean
            print(f"-> Soustraction du Blank moyen ({len(idx_b)} fichiers)")

    # Pipeline PCA
    steps = [("pca", PCA(n_components=N_COMPONENTS, random_state=42))]
    if USE_STANDARDIZE_FOR_PCA:
        steps.insert(0, ("scaler", StandardScaler()))
    pipe = Pipeline(steps)
    scores = pipe.fit_transform(X)

   # --- CALCUL DES ELLIPSES ET TRI AUTOMATIQUE ---
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # 1. Détection des aberrations pour le groupe de contrôle
    target_label = "Urine_Base" if "Urine_Base" in labels else "Control_Blank"
    idx_control = [i for i, l in enumerate(labels) if l == target_label]
    b_scores = scores[idx_control]
    b_names = np.array(names)[idx_control]
    
    # Filtre statistique (IQR)
    b_center_raw = np.mean(b_scores[:, :2], axis=0)
    dists_raw = np.linalg.norm(b_scores[:, :2] - b_center_raw, axis=1)
    limit = np.percentile(dists_raw, 75)
    clean_mask = dists_raw <= limit
    
    ignored = b_names[~clean_mask]
    print(f"\n--- FILTRE D'ABERRATIONS ---")
    print(f"Fichiers Urine_Base masqués (trop dispersés) : {list(ignored)}")

    # 2. Dessin des points propres et des ellipses
    for lab in np.unique(labels):
        mask = np.array(labels) == lab
        # Pour l'urine, on ne prend que les points "clean"
        if lab == target_label:
            idx_to_plot = np.array(idx_control)[clean_mask]
            s_plot = scores[idx_to_plot]
        else:
            s_plot = scores[mask]
        
        # Dessin des points
        scatter = plt.scatter(s_plot[:,0], s_plot[:,1], label=lab, s=60, edgecolors='k', alpha=0.7)
        color = scatter.get_facecolor()[0]

        # Dessin de l'ellipse de confiance (95%)
        if len(s_plot) > 2:
            cov = np.cov(s_plot[:, :2], rowvar=False)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * 2 * np.sqrt(vals) # Rayon 2-sigma
            ell = Ellipse(xy=np.mean(s_plot[:, :2], axis=0), width=width, height=height, 
                          angle=theta, color=color, alpha=0.1, label=f"Zone {lab}")
            ax.add_artist(ell)

    # 3. Projection de l'Échantillon Terrain (Étoile)
    if NEW_SAMPLE_PATH and os.path.exists(NEW_SAMPLE_PATH):
        xt, yt = read_txt_spectrum(NEW_SAMPLE_PATH)
        xtp, ytp = preprocess_y(xt, yt, USE_BASELINE_ALS, USE_SMOOTHING, 
                                USE_NORMALIZATION, WAVENUMBER_MIN, WAVENUMBER_MAX)
        yti = np.interp(grid, xtp, ytp) - blank_mean
        st = pipe.transform([yti])
        
        # Verdict basé sur le groupe propre
        b_clean_scores = b_scores[clean_mask]
        b_center_clean = np.mean(b_clean_scores[:, :2], axis=0)
        dist_test = np.linalg.norm(st[0, :2] - b_center_clean)
        
        # Seuil statistique
        dists_clean = np.linalg.norm(b_clean_scores[:, :2] - b_center_clean, axis=1)
        seuil = np.mean(dists_clean) + 2 * np.std(dists_clean)
        
        verdict = "POSITIF" if dist_test > seuil else "NEGATIF"
        plt.scatter(st[0,0], st[0,1], color='red' if verdict=="POSITIF" else 'lime', 
                    marker='*', s=500, label=f"TEST: {verdict}", edgecolors='k', zorder=100)
        
        print(f"VERDICT FINAL : {verdict} (Dist: {dist_test:.2f} | Seuil: {seuil:.2f})")

    plt.xlabel(f"PC1 ({pipe.named_steps['pca'].explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pipe.named_steps['pca'].explained_variance_ratio_[1]:.1%})")
    plt.title("PCA Raman - Analyse de Détection avec Zones de Confiance")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()