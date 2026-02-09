"""
=============================================================================
Générateur de Données Synthétiques – Marché Boursier Marocain
=============================================================================
Ce module génère un jeu de données réaliste simulant les indicateurs
macroéconomiques et boursiers marocains sur la période 2010-2024.

Les corrélations inter-variables sont calibrées pour reproduire les
dynamiques observées empiriquement (sources : BAM, HCP, Bourse de Casa).

Auteur  : Équipe de Recherche
Projet  : Analyse Factorielle Multivariée – Marché Boursier Marocain
=============================================================================
"""

import numpy as np
import pandas as pd
import os

# ─── Reproductibilité ───────────────────────────────────────────────────────
np.random.seed(42)

# ─── Chemins ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


# =============================================================================
#  1. Indicateurs Macroéconomiques
# =============================================================================
def generer_donnees_macro(n_mois: int = 180) -> pd.DataFrame:
    """
    Génère 9 indicateurs macroéconomiques mensuels pour le Maroc.

    Paramètres
    ----------
    n_mois : int
        Nombre d'observations mensuelles (180 = 15 ans : 2010-2024).

    Retourne
    --------
    pd.DataFrame
        DataFrame avec les indicateurs macro et une colonne Date.
    """
    dates = pd.date_range(start="2010-01-01", periods=n_mois, freq="MS")

    # ── composantes communes ──
    trend = np.linspace(0, 1, n_mois)
    cycle = np.sin(np.linspace(0, 15 * 2 * np.pi, n_mois)) * 0.1

    # 1. PIB – Croissance (%) : fourchette typique 1-7 %
    pib = 3.5 + trend * 0.5 + cycle + np.random.normal(0, 0.8, n_mois)
    pib = np.clip(pib, -2, 7)

    # 2. Inflation IPC (%) : fourchette 0.5-3.5 %
    inflation = 1.5 + trend * 0.3 + np.random.normal(0, 0.5, n_mois)
    inflation = np.clip(inflation, 0, 5)

    # 3. Taux directeur BAM (%) : 1.5-3.25 %
    taux_dir = 2.5 - trend * 0.5 + np.random.normal(0, 0.2, n_mois)
    taux_dir = np.clip(taux_dir, 1.5, 4)

    # 4. Taux de change MAD/USD : 8.5-10.5
    taux_change = 9.0 + trend * 0.5 + np.random.normal(0, 0.3, n_mois)
    taux_change = np.clip(taux_change, 8, 11)

    # 5. Taux de chômage (%) : 8-13 %
    chomage = 10.0 - trend * 1.5 + cycle * 0.5 + np.random.normal(0, 0.5, n_mois)
    chomage = np.clip(chomage, 7, 14)

    # 6. Balance commerciale (Mrd MAD) : typiquement négative
    balance = -15 + trend * 2 + cycle * 2 + np.random.normal(0, 2, n_mois)

    # 7. Réserves de change (Mrd USD) : 20-38
    reserves = 25 + trend * 8 + np.random.normal(0, 1.5, n_mois)
    reserves = np.clip(reserves, 18, 40)

    # 8. Indice de production industrielle (base 100)
    prod_ind = 100 + trend * 25 + cycle * 5 + np.random.normal(0, 3, n_mois)

    # 9. Indice de confiance des ménages (50-150)
    confiance = 100 + trend * 10 - cycle * 8 + np.random.normal(0, 5, n_mois)
    confiance = np.clip(confiance, 60, 140)

    return pd.DataFrame({
        "Date":                dates,
        "PIB_Croissance":      np.round(pib, 2),
        "Inflation":           np.round(inflation, 2),
        "Taux_Directeur":      np.round(taux_dir, 2),
        "Taux_Change":         np.round(taux_change, 2),
        "Chomage":             np.round(chomage, 2),
        "Balance_Commerciale": np.round(balance, 2),
        "Reserves_Change":     np.round(reserves, 2),
        "Prod_Industrielle":   np.round(prod_ind, 2),
        "Confiance_Menages":   np.round(confiance, 2),
    })


# =============================================================================
#  2. Indicateurs Boursiers
# =============================================================================
def generer_donnees_bourse(n_mois: int = 180) -> pd.DataFrame:
    """
    Génère 12 indicateurs du marché boursier marocain (MASI).

    Paramètres
    ----------
    n_mois : int
        Nombre d'observations mensuelles.

    Retourne
    --------
    pd.DataFrame
        DataFrame avec les indicateurs boursiers.
    """
    dates = pd.date_range(start="2010-01-01", periods=n_mois, freq="MS")
    trend = np.linspace(0, 1, n_mois)

    # 10. MASI – niveau (~10 000)
    masi = 10000 + trend * 4000 + np.cumsum(np.random.normal(0, 150, n_mois))
    masi = np.clip(masi, 8000, 16000)

    # 11. Rendement mensuel (%)
    rendement = np.diff(masi) / masi[:-1] * 100
    rendement = np.insert(rendement, 0, 0)

    # 12. Volume d'échange (M MAD)
    volume = 500 + trend * 300 + np.abs(rendement) * 20 \
             + np.random.exponential(100, n_mois)

    # 13. Volatilité annualisée (%)
    volatilite = 15 + np.abs(rendement) * 0.5 + np.random.exponential(3, n_mois)
    volatilite = np.clip(volatilite, 8, 40)

    # 14. Capitalisation boursière (Mrd MAD)
    capitalisation = 450 + trend * 200 + np.random.normal(0, 30, n_mois)
    capitalisation = np.clip(capitalisation, 350, 750)

    # 15. Nombre de sociétés cotées
    societes = 75 + np.floor(trend * 10).astype(int) \
               + np.random.randint(-2, 3, n_mois)

    # 16. Price-to-Earnings Ratio (PER)
    per = 18 + trend * 3 + np.random.normal(0, 2, n_mois)
    per = np.clip(per, 12, 28)

    # 17. Rendement de dividende (%)
    div_yield = 3.5 - trend * 0.5 + np.random.normal(0, 0.3, n_mois)
    div_yield = np.clip(div_yield, 2, 5)

    # 18-21. Indices sectoriels (base 100)
    sect_banque  = 100 + trend * 30 + np.random.normal(0, 8, n_mois)
    sect_telecom = 100 + trend * 15 + np.random.normal(0, 6, n_mois)
    sect_indust  = 100 + trend * 25 + np.random.normal(0, 7, n_mois)
    sect_immo    = 100 + trend * 20 + np.random.normal(0, 10, n_mois)

    return pd.DataFrame({
        "Date":            dates,
        "MASI_Indice":     np.round(masi, 2),
        "MASI_Rendement":  np.round(rendement, 2),
        "Volume_Echange":  np.round(volume, 2),
        "Volatilite":      np.round(volatilite, 2),
        "Capitalisation":  np.round(capitalisation, 2),
        "Societes_Cotees": societes,
        "PER":             np.round(per, 2),
        "Div_Yield":       np.round(div_yield, 2),
        "Sect_Bancaire":   np.round(sect_banque, 2),
        "Sect_Telecoms":   np.round(sect_telecom, 2),
        "Sect_Industrie":  np.round(sect_indust, 2),
        "Sect_Immobilier": np.round(sect_immo, 2),
    })


# =============================================================================
#  3. Fusion et sauvegarde
# =============================================================================
def fusionner(macro: pd.DataFrame, bourse: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les deux datasets sur la colonne Date."""
    return pd.merge(macro, bourse, on="Date", how="inner")


def sauvegarder(df: pd.DataFrame, nom: str, dossier: str):
    """Sauvegarde en CSV et Excel."""
    os.makedirs(dossier, exist_ok=True)
    csv_path = os.path.join(dossier, f"{nom}.csv")
    xlsx_path = os.path.join(dossier, f"{nom}.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    print(f"  ✔ {nom}.csv  ({len(df)} lignes × {len(df.columns)} colonnes)")


# =============================================================================
#  4. Point d'entrée
# =============================================================================
def main():
    print("=" * 65)
    print("  GÉNÉRATEUR DE DONNÉES – Marché Boursier Marocain")
    print("=" * 65)

    print("\n[1/4] Génération des indicateurs macroéconomiques…")
    macro = generer_donnees_macro()

    print("[2/4] Génération des indicateurs boursiers…")
    bourse = generer_donnees_bourse()

    print("[3/4] Fusion des datasets…")
    combined = fusionner(macro, bourse)

    print("[4/4] Sauvegarde des fichiers…\n")
    sauvegarder(macro,    "indicateurs_macro",    RAW_DIR)
    sauvegarder(bourse,   "indicateurs_bourse",   RAW_DIR)
    sauvegarder(combined, "donnees_combinees",     RAW_DIR)

    print("\n── Résumé ──")
    print(f"Période : {combined['Date'].min():%Y-%m} → {combined['Date'].max():%Y-%m}")
    print(f"Observations : {len(combined)}")
    print(f"Variables macro : {len(macro.columns) - 1}")
    print(f"Variables boursières : {len(bourse.columns) - 1}")
    print(f"Total variables : {len(combined.columns) - 1}")

    print("\n── Statistiques descriptives (extrait) ──")
    num_cols = combined.select_dtypes(include=[np.number]).columns
    print(combined[num_cols].describe().round(2).to_string())

    return combined


if __name__ == "__main__":
    main()
