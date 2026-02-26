"""
PCA Raman Library Analyzer - Version COLLÈGUE + OPTIONS TERRAIN
"""

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

# =========================
# CONFIG (Adapté à ton Mac)
# =========================
DATA_DIR = '/Users/mac/Library/Mobile Documents/com~apple~CloudDocs/Session H26/TPOP Projet 1 /Données'
FILE_GLOB = "**/*.TXT" 

# --- AJOUT : TON FICHIER TEST TERRAIN ---
# Remplace None par le chemin de ton fichier pour un test spontané
NEW_SAMPLE_PATH = '/Users/mac/Library/Mobile Documents/com~apple~CloudDocs/Session H26/TPOP Projet 1 /MIX-mid-3.TXT'

# Choix prétraitements
USE_BASELINE_ALS = True
USE_SMOOTHING = True
USE_NORMALIZATION = True     
USE_STANDARDIZE_FOR_PCA = True  
USE_BLANK_SUBTRACTION = True # <-- NOUVEAU : Soustrait le B2 moyen

# PCA
N_COMPONENTS = 5

# Domaine spectral (Focus sur les pics chimiques)
WAVENUMBER_MIN = 450   
WAVENUMBER_MAX = 1650  

LABEL_RULES = [
    (r"B2", "Control_Blank"),
    (r"Pseudo-seul", "Pseudo_Pure"),
    (r"testo-1", "Testo_Pure"),
    (r"trace", "Mix_Traces"),
    (r"MIX-haut|MIX-mid|testo_melange", "Mix_Concentrated"),
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
    data = np.loadtxt(path, delimiter=",", usecols=(1, 2))
    x, y = data[:, 0], data[:, 1]
    if x[0] > x[-1]: x, y = x[::-1], y[::-1]
    return x, y

def load_library(data_dir: str, file_glob: str) -> List[Spectrum]:
    paths = glob.glob(os.path.join(data_dir, file_glob), recursive=True)
    paths = [p for p in paths if os.path.isfile(p) and "calibration" not in p.lower()]
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
    mask = (x >= (wn_min or x.min())) & (x <= (wn_max or x.max()))
    if not np.any(mask): mask = np.ones(len(x), dtype=bool)
    x, y = x[mask], y.astype(float)[mask]

    if use_baseline:
        y = y - baseline_als(y)
    if use_smoothing:
        win = min(15, len(y) - 1 if (len(y) % 2 == 0) else len(y))
        if win >= 3: y = savgol_filter(y, window_length=win, polyorder=2)
    if use_norm:
        y = (y - np.mean(y)) / (np.std(y) + 1e-12)
    return x, y

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

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    for lab in np.unique(labels):
        mask = np.array(labels) == lab
        plt.scatter(scores[mask, 0], scores[mask, 1], label=lab, s=50, alpha=0.7)

    # --- MODE TERRAIN : PROJECTION ---
    if NEW_SAMPLE_PATH and os.path.exists(NEW_SAMPLE_PATH):
        xt, yt = read_txt_spectrum(NEW_SAMPLE_PATH)
        _, ytp = preprocess_y(xt, yt, USE_BASELINE_ALS, USE_SMOOTHING, 
                              USE_NORMALIZATION, WAVENUMBER_MIN, WAVENUMBER_MAX)
        yti = np.interp(grid, xt, ytp) - blank_mean
        st = pipe.transform([yti])
        
        # Calcul Distance pour Verdict
        b_scores = scores[np.array(labels) == "Control_Blank"]
        b_center = np.mean(b_scores[:, :2], axis=0)
        dist = np.linalg.norm(st[0, :2] - b_center)
        seuil = np.mean(np.linalg.norm(b_scores[:, :2] - b_center, axis=1)) + 2 * np.std(np.linalg.norm(b_scores[:, :2] - b_center, axis=1))
        
        verdict = "POSITIF" if dist > seuil else "NEGATIF"
        plt.scatter(st[0,0], st[0,1], color='red', marker='*', s=250, label=f"TEST: {verdict}", edgecolors='k')
        print(f"\nRESULTAT : {verdict} (Dist: {dist:.2f} | Seuil: {seuil:.2f})")

    plt.xlabel(f"PC1 ({pipe.named_steps['pca'].explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pipe.named_steps['pca'].explained_variance_ratio_[1]:.1%})")
    plt.title("PCA Raman - Bibliothèque de Référence vs Échantillon Terrain")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()