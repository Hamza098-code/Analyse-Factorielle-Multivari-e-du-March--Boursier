"""
=============================================================================
Analyse en Composantes Principales (ACP) – Python / scikit-learn
=============================================================================
Ce module réalise l'ACP complète :
  1. Chargement données standardisées
  2. Calcul des valeurs propres, loadings, communalités
  3. Critère de Kaiser et seuil de variance (80 %)
  4. Visualisations : éboulis, biplot, cercle des corrélations, heatmap
  5. Interprétation économique des composantes

Formules clés (LaTeX) :
  Matrice de covariance : Σ = (1/(n-1)) X^T X
  Valeurs propres       : Σ v_k = λ_k v_k
  Part d'inertie        : τ_k = λ_k / Σ λ_j × 100
  Loadings              : l_{jk} = v_{jk} √λ_k
  Communalité           : h²_j = Σ_k l²_{jk}

Auteur  : Équipe de Recherche
Projet  : Analyse Factorielle Multivariée – Marché Boursier Marocain
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Chemins ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR   = os.path.join(PROJECT_ROOT, "outputs", "tables")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "outputs", "figures")

# ─── Style graphique ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_style("whitegrid")


# =============================================================================
#  Classe ACP
# =============================================================================
class AnalyseACP:
    """
    Analyse en Composantes Principales pour données économiques marocaines.

    L'ACP permet de :
      • Réduire la dimensionnalité en conservant la variance
      • Identifier les facteurs latents sous-jacents
      • Extraire des composantes orthogonales interprétables
    """

    def __init__(self, n_composantes: int = None):
        self.n_composantes = n_composantes
        self.pca = None
        self.noms_variables = None
        self.resultats = {}

    # ── Ajustement ────────────────────────────────────────────────────────
    def ajuster(self, X: pd.DataFrame):
        """Ajuste le modèle ACP aux données standardisées."""
        self.noms_variables = X.columns.tolist()
        self.pca = PCA(n_components=self.n_composantes)
        self.pca.fit(X)
        print(f"  ACP ajustée : {self.pca.n_components_} composantes extraites")
        return self

    def transformer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Projette les données dans l'espace des composantes."""
        scores = self.pca.transform(X)
        cols = [f"CP{i+1}" for i in range(scores.shape[1])]
        return pd.DataFrame(scores, columns=cols)

    def ajuster_transformer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ajuste et transforme en une étape."""
        self.ajuster(X)
        return self.transformer(X)

    # ── Valeurs propres ───────────────────────────────────────────────────
    def valeurs_propres(self) -> pd.DataFrame:
        """
        Retourne les valeurs propres et la variance expliquée.

        λ_k = variance expliquée par la k-ème composante
        τ_k = λ_k / Σ λ_j × 100  (part d'inertie)
        """
        vp = self.pca.explained_variance_
        ratio = self.pca.explained_variance_ratio_
        cumul = np.cumsum(ratio)

        df = pd.DataFrame({
            "Composante":       [f"CP{i+1}" for i in range(len(vp))],
            "Valeur_Propre":    np.round(vp, 4),
            "Variance_pct":     np.round(ratio * 100, 2),
            "Cumul_pct":        np.round(cumul * 100, 2),
        })
        self.resultats["valeurs_propres"] = df
        return df

    # ── Loadings (saturations) ────────────────────────────────────────────
    def loadings(self) -> pd.DataFrame:
        """
        Calcule les saturations factorielles (loadings).

        l_{jk} = v_{jk} × √λ_k

        Les loadings représentent la corrélation entre les variables
        originales et les composantes principales.
        """
        L = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        cols = [f"CP{i+1}" for i in range(L.shape[1])]
        df = pd.DataFrame(L, index=self.noms_variables, columns=cols)
        self.resultats["loadings"] = df
        return df

    # ── Communalités ──────────────────────────────────────────────────────
    def communalites(self) -> pd.DataFrame:
        """
        Calcule les communalités.

        h²_j = Σ_k l²_{jk}

        La communalité mesure la proportion de variance d'une variable
        expliquée par l'ensemble des composantes retenues.
        """
        L = self.loadings()
        h2 = (L ** 2).sum(axis=1)
        df = pd.DataFrame({
            "Variable":     self.noms_variables,
            "Communalite":  np.round(h2.values, 4),
        })
        self.resultats["communalites"] = df
        return df

    # ── Critères de sélection ─────────────────────────────────────────────
    def critere_kaiser(self) -> int:
        """Critère de Kaiser : retient les composantes avec λ > 1."""
        n = int(np.sum(self.pca.explained_variance_ > 1))
        print(f"  Critère de Kaiser → {n} composantes (λ > 1)")
        return n

    def seuil_variance(self, seuil: float = 0.80) -> int:
        """Nombre de composantes pour expliquer ≥ seuil % de variance."""
        cumul = np.cumsum(self.pca.explained_variance_ratio_)
        n = int(np.argmax(cumul >= seuil)) + 1
        print(f"  Seuil {seuil*100:.0f}% → {n} composantes ({cumul[n-1]*100:.1f}%)")
        return n

    # ── Interprétation ────────────────────────────────────────────────────
    def interpreter(self, n_top: int = 5) -> dict:
        """Identifie les variables dominantes par composante."""
        L = self.loadings()
        interp = {}
        for i, cp in enumerate(L.columns[:5]):
            tri = L[cp].abs().sort_values(ascending=False)
            top = tri.head(n_top)
            interp[cp] = {
                "variables_dominantes": L.loc[top.index, cp].to_dict(),
                "variance_expliquee":   self.pca.explained_variance_ratio_[i] * 100,
            }
        return interp

    # ── Sauvegarde ────────────────────────────────────────────────────────
    def sauvegarder(self):
        """Sauvegarde tous les résultats dans outputs/tables/."""
        os.makedirs(TABLES_DIR, exist_ok=True)
        self.valeurs_propres().to_csv(
            os.path.join(TABLES_DIR, "acp_valeurs_propres.csv"), index=False)
        self.loadings().to_csv(
            os.path.join(TABLES_DIR, "acp_loadings.csv"))
        self.communalites().to_csv(
            os.path.join(TABLES_DIR, "acp_communalites.csv"), index=False)
        print("  ✔ Résultats ACP sauvegardés dans outputs/tables/")


# =============================================================================
#  Visualisations ACP
# =============================================================================

def graphique_eboulis(acp: AnalyseACP):
    """
    Graphique des éboulis (scree plot).
    Montre la décroissance des valeurs propres et le coude.
    """
    vp = acp.pca.explained_variance_
    ratio = acp.pca.explained_variance_ratio_ * 100
    cumul = np.cumsum(ratio)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(1, len(vp) + 1)

    # Barres : variance individuelle
    bars = ax1.bar(x, ratio, color="#2196F3", alpha=0.7, label="Variance individuelle (%)")
    ax1.set_xlabel("Composante Principale")
    ax1.set_ylabel("Variance Expliquée (%)", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    # Ligne de Kaiser (λ = 1)
    kaiser_line = 100 / len(vp)  # seuil moyen
    ax1.axhline(y=kaiser_line, color="red", linestyle="--", alpha=0.5, label=f"Seuil moyen ({kaiser_line:.1f}%)")

    # Courbe cumulative
    ax2 = ax1.twinx()
    ax2.plot(x, cumul, "o-", color="#FF5722", linewidth=2, label="Cumul (%)")
    ax2.axhline(y=80, color="#FF5722", linestyle=":", alpha=0.5, label="Seuil 80%")
    ax2.set_ylabel("Variance Cumulée (%)", color="#FF5722")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    # Légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Graphique des Éboulis – ACP\nIndicateurs Macroéconomiques & Boursiers Marocains")
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "acp_eboulis.png"), dpi=150)
    plt.close()
    print("  ✔ acp_eboulis.png")


def cercle_correlations(acp: AnalyseACP):
    """
    Cercle des corrélations (variables dans le plan CP1-CP2).
    Les flèches représentent les loadings des variables.
    """
    L = acp.loadings()
    if L.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Cercle unité
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.5)

    # Flèches des variables
    for var in L.index:
        x, y = L.loc[var, "CP1"], L.loc[var, "CP2"]
        ax.annotate(
            var, xy=(x, y), fontsize=8,
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5),
            xytext=(0, 0), textcoords="data",
        )
        # dessiner la flèche manuellement
        ax.arrow(0, 0, x * 0.95, y * 0.95,
                 head_width=0.02, head_length=0.02,
                 fc="#1565C0", ec="#1565C0", alpha=0.7)

    # Axes
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")

    vr = acp.pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"CP1 ({vr[0]:.1f}%)")
    ax.set_ylabel(f"CP2 ({vr[1]:.1f}%)")
    ax.set_title("Cercle des Corrélations – ACP\nProjection des variables sur CP1-CP2")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "acp_cercle_correlations.png"), dpi=150)
    plt.close()
    print("  ✔ acp_cercle_correlations.png")


def biplot_acp(acp: AnalyseACP, scores: pd.DataFrame, dates: pd.Series):
    """
    Biplot : individus + variables dans le plan CP1-CP2.
    Les observations sont colorées par année.
    """
    L = acp.loadings()
    if L.shape[1] < 2 or scores.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Couleur par année
    annees = pd.to_datetime(dates).dt.year
    scatter = ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1],
                         c=annees, cmap="viridis", alpha=0.6, s=30)
    plt.colorbar(scatter, label="Année")

    # Vecteurs des variables (échelle ajustée)
    scale = max(scores.iloc[:, 0].abs().max(), scores.iloc[:, 1].abs().max()) * 0.8
    for var in L.index:
        x = L.loc[var, "CP1"] * scale
        y = L.loc[var, "CP2"] * scale
        ax.arrow(0, 0, x, y, head_width=scale * 0.02,
                 fc="red", ec="red", alpha=0.8)
        ax.text(x * 1.1, y * 1.1, var, fontsize=7, color="red",
                ha="center", va="center")

    vr = acp.pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"CP1 ({vr[0]:.1f}%)")
    ax.set_ylabel(f"CP2 ({vr[1]:.1f}%)")
    ax.set_title("Biplot ACP – Individus et Variables\nMarché Boursier Marocain (2010-2024)")
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "acp_biplot.png"), dpi=150)
    plt.close()
    print("  ✔ acp_biplot.png")


def heatmap_loadings(acp: AnalyseACP, n_cp: int = 5):
    """Heatmap des loadings pour les n premières composantes."""
    L = acp.loadings()
    L_sub = L.iloc[:, :min(n_cp, L.shape[1])]

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(L_sub, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Loading"})
    ax.set_title(f"Heatmap des Loadings – ACP\n(Top {n_cp} composantes)")
    ax.set_ylabel("Variables")
    ax.set_xlabel("Composantes Principales")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "acp_heatmap_loadings.png"), dpi=150)
    plt.close()
    print("  ✔ acp_heatmap_loadings.png")


def heatmap_correlation(df: pd.DataFrame):
    """Heatmap de la matrice de corrélation."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7},
                cbar_kws={"label": "Corrélation de Pearson"})
    ax.set_title("Matrice de Corrélation\nIndicateurs Macro & Boursiers – Maroc")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "matrice_correlation.png"), dpi=150)
    plt.close()
    print("  ✔ matrice_correlation.png")


# =============================================================================
#  Pipeline ACP complète
# =============================================================================
def executer_acp():
    """Exécute l'analyse ACP complète."""
    print("=" * 65)
    print("  ANALYSE EN COMPOSANTES PRINCIPALES (ACP)")
    print("=" * 65)

    # ── 1. Charger les données ────────────────────────────────────────────
    print("\n[1/6] Chargement des données standardisées…")
    df_std = pd.read_csv(os.path.join(PROC_DIR, "donnees_acp.csv"), parse_dates=["Date"])
    df_raw = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "donnees_combinees.csv"),
                         parse_dates=["Date"])
    num_cols = df_std.select_dtypes(include=[np.number]).columns.tolist()
    X = df_std[num_cols]
    print(f"  {X.shape[0]} observations × {X.shape[1]} variables")

    # ── 2. Matrice de corrélation ─────────────────────────────────────────
    print("\n[2/6] Matrice de corrélation…")
    heatmap_correlation(df_raw)

    # ── 3. Ajustement ACP ─────────────────────────────────────────────────
    print("\n[3/6] Ajustement du modèle ACP…")
    acp = AnalyseACP()
    scores = acp.ajuster_transformer(X)

    # ── 4. Valeurs propres et critères ────────────────────────────────────
    print("\n[4/6] Analyse des valeurs propres…")
    vp = acp.valeurs_propres()
    print("\n" + vp.to_string(index=False))
    acp.critere_kaiser()
    acp.seuil_variance(0.80)

    # ── 5. Interprétation économique ──────────────────────────────────────
    print("\n[5/6] Interprétation des composantes…")
    interp = acp.interpreter()
    print("\n── Interprétation Économique ──")
    for cp, info in interp.items():
        print(f"\n  {cp} ({info['variance_expliquee']:.1f}% de variance) :")
        for var, loading in info["variables_dominantes"].items():
            signe = "+" if loading > 0 else "-"
            print(f"    {signe} {var} : {loading:.3f}")

    # ── 6. Visualisations ─────────────────────────────────────────────────
    print("\n[6/6] Génération des visualisations…")
    graphique_eboulis(acp)
    cercle_correlations(acp)
    biplot_acp(acp, scores, df_std["Date"])
    heatmap_loadings(acp)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    acp.sauvegarder()
    scores.insert(0, "Date", df_std["Date"].values)
    scores.to_csv(os.path.join(PROC_DIR, "scores_acp.csv"), index=False)
    print("  ✔ scores_acp.csv sauvegardé")

    print("\n" + "=" * 65)
    print("  ACP TERMINÉE ✔")
    print("=" * 65)

    return acp, scores


def main():
    return executer_acp()


if __name__ == "__main__":
    main()
