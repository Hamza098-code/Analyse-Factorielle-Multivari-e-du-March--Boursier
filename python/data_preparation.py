"""
=============================================================================
Préparation des Données – Nettoyage, Standardisation, Discrétisation
=============================================================================
Ce module charge les données brutes, effectue :
  1. Nettoyage et vérification de la qualité
  2. Statistiques descriptives et matrice de corrélation
  3. Standardisation Z-score pour l'ACP
  4. Discrétisation par tertiles pour l'ACM

Auteur  : Équipe de Recherche
Projet  : Analyse Factorielle Multivariée – Marché Boursier Marocain
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# ─── Chemins ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR   = os.path.join(PROJECT_ROOT, "outputs", "tables")


# =============================================================================
#  1. Chargement et nettoyage
# =============================================================================
def charger_donnees() -> pd.DataFrame:
    """
    Charge le fichier CSV combiné depuis data/raw/.

    Retourne
    --------
    pd.DataFrame
        Données brutes chargées.
    """
    path = os.path.join(RAW_DIR, "donnees_combinees.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"  Chargement : {len(df)} observations, {len(df.columns)} colonnes")
    return df


def verifier_qualite(df: pd.DataFrame):
    """Vérifie les valeurs manquantes, doublons et types."""
    print("\n── Contrôle qualité ──")

    # Valeurs manquantes
    na_count = df.isnull().sum()
    na_total = na_count.sum()
    print(f"  Valeurs manquantes : {na_total}")
    if na_total > 0:
        print(na_count[na_count > 0])

    # Doublons
    dup = df.duplicated().sum()
    print(f"  Doublons           : {dup}")

    # Types
    print(f"  Types détectés     : {dict(df.dtypes.value_counts())}")


# =============================================================================
#  2. Statistiques descriptives
# =============================================================================
def statistiques_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule et sauvegarde les statistiques descriptives.

    Retourne
    --------
    pd.DataFrame
        Tableau des statistiques descriptives.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[num_cols].describe().T
    stats["CV"] = (stats["std"] / stats["mean"]).round(4)  # coefficient de variation

    os.makedirs(TABLES_DIR, exist_ok=True)
    stats.to_csv(os.path.join(TABLES_DIR, "statistiques_descriptives.csv"))
    print("  ✔ statistiques_descriptives.csv sauvegardé")

    return stats


def matrice_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule et sauvegarde la matrice de corrélation.

    Retourne
    --------
    pd.DataFrame
        Matrice de corrélation (Pearson).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr().round(4)

    corr.to_csv(os.path.join(TABLES_DIR, "matrice_correlation.csv"))
    print("  ✔ matrice_correlation.csv sauvegardé")

    return corr


# =============================================================================
#  3. Standardisation Z-score (pour l'ACP)
# =============================================================================
def standardiser_acp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les variables numériques (moyenne=0, écart-type=1).

    La standardisation est nécessaire pour l'ACP car les variables ont
    des unités et des échelles très différentes (%, indices, Mrd MAD…).

    Retourne
    --------
    pd.DataFrame
        Données standardisées avec colonne Date conservée.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[num_cols])
    df_std = pd.DataFrame(scaled, columns=num_cols)

    # conserver la colonne Date
    df_std.insert(0, "Date", df["Date"].values)

    os.makedirs(PROC_DIR, exist_ok=True)
    df_std.to_csv(os.path.join(PROC_DIR, "donnees_acp.csv"), index=False)
    print("  ✔ donnees_acp.csv sauvegardé (standardisé Z-score)")

    return df_std


# =============================================================================
#  4. Discrétisation par tertiles (pour l'ACM)
# =============================================================================
def discretiser_acm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les variables continues en variables catégorielles
    via une discrétisation par tertiles.

    Stratégie
    ---------
    Pour chaque variable X :
      - X ≤ P33  →  "Faible"
      - P33 < X ≤ P67  →  "Moyen"
      - X > P67  →  "Élevé"

    Ceci garantit ~60 observations par catégorie (équilibre),
    ce qui est crucial pour la stabilité de l'ACM.

    Retourne
    --------
    pd.DataFrame
        Données catégorielles pour l'ACM.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    labels = ["Faible", "Moyen", "Élevé"]

    df_cat = pd.DataFrame()
    df_cat["Date"] = df["Date"].values

    for col in num_cols:
        # Calcul des seuils (tertiles)
        q33 = df[col].quantile(0.3333)
        q67 = df[col].quantile(0.6667)

        # Classification
        conditions = [
            df[col] <= q33,
            (df[col] > q33) & (df[col] <= q67),
            df[col] > q67,
        ]
        df_cat[f"{col}_cat"] = np.select(conditions, labels, default="Moyen")

    os.makedirs(PROC_DIR, exist_ok=True)
    df_cat.to_csv(os.path.join(PROC_DIR, "donnees_acm.csv"), index=False)
    print("  ✔ donnees_acm.csv sauvegardé (catégories : Faible/Moyen/Élevé)")

    # Résumé de la distribution des catégories
    print("\n── Distribution des catégories ──")
    cat_cols = [c for c in df_cat.columns if c.endswith("_cat")]
    for col in cat_cols[:5]:  # afficher les 5 premières
        counts = df_cat[col].value_counts()
        print(f"  {col}: {dict(counts)}")
    if len(cat_cols) > 5:
        print(f"  … et {len(cat_cols) - 5} variables supplémentaires")

    return df_cat


# =============================================================================
#  5. Point d'entrée
# =============================================================================
def main():
    print("=" * 65)
    print("  PRÉPARATION DES DONNÉES")
    print("=" * 65)

    # Chargement
    print("\n[1/5] Chargement des données brutes…")
    df = charger_donnees()

    # Qualité
    print("\n[2/5] Vérification de la qualité…")
    verifier_qualite(df)

    # Statistiques
    print("\n[3/5] Statistiques descriptives…")
    stats = statistiques_descriptives(df)
    print(stats[["mean", "std", "min", "max"]].round(2).to_string())

    # Corrélation
    print("\n[4/5] Matrice de corrélation…")
    corr = matrice_correlation(df)

    # Corrélations notables
    num_cols = df.select_dtypes(include=[np.number]).columns
    print("\n── Corrélations les plus fortes (|r| > 0.5) ──")
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                print(f"  {num_cols[i]} ↔ {num_cols[j]} : r = {r:.3f}")

    # Standardisation ACP
    print("\n[5a/5] Standardisation pour l'ACP…")
    df_acp = standardiser_acp(df)

    # Discrétisation ACM
    print("\n[5b/5] Discrétisation pour l'ACM…")
    df_acm = discretiser_acm(df)

    print("\n" + "=" * 65)
    print("  PRÉPARATION TERMINÉE ✔")
    print("=" * 65)

    return df, df_acp, df_acm


if __name__ == "__main__":
    main()
