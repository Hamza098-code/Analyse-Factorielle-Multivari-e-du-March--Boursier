# 📊 Analyse Factorielle Multivariée – Marché Boursier Marocain

> Étude des relations entre indicateurs macroéconomiques et dynamique du MASI par ACP et ACM.

---

## 🏗️ Structure du projet

```
Project/
├── python/
│   ├── data_generator.py       # Génération de données synthétiques (21 vars, 180 obs)
│   ├── data_preparation.py     # Nettoyage, standardisation Z-score, discrétisation tertiles
│   ├── pca_analysis.py         # ACP complète (scikit-learn) + visualisations
│   └── requirements.txt        # Dépendances Python
├── R/
│   ├── mca_analysis.R          # ACM complète (FactoMineR) + visualisations
│   └── pca_validation.R        # Validation croisée ACP Python vs R
├── data/
│   ├── raw/                    # Données brutes (CSV + Excel)
│   └── processed/              # Données standardisées (ACP) + catégorielles (ACM)
├── outputs/
│   ├── figures/                # Graphiques (éboulis, biplot, cercle corrélations, cartes ACM)
│   └── tables/                 # Tableaux statistiques (valeurs propres, loadings, contributions)
├── report/
│   └── rapport_final.md        # Rapport de recherche (≤ 20 pages)
└── README.md                   # Ce fichier
```

## 🔧 Installation et exécution

### Prérequis

- **Python** ≥ 3.9
- **R** ≥ 4.0 avec packages : FactoMineR, factoextra, tidyverse

### Python

```bash
cd Project/python
pip install -r requirements.txt

# 1. Générer les données
python data_generator.py

# 2. Préparer les données
python data_preparation.py

# 3. Exécuter l'ACP
python pca_analysis.py
```

### R

```bash
cd Project/R

# ACM
Rscript mca_analysis.R

# Validation croisée ACP
Rscript pca_validation.R
```

## 📐 Variables (21)

| # | Variable | Type | Source |
|---|----------|------|--------|
| 1-9 | PIB, Inflation, Taux directeur, Change, Chômage, Balance comm., Réserves, Production ind., Confiance | Macro | HCP, BAM |
| 10-21 | MASI (niveau, rendement), Volume, Volatilité, Capitalisation, Sociétés cotées, PER, Div Yield, 4 secteurs | Bourse | BC |

## 📈 Méthodes

| Méthode | Données | Logiciel | Objectif |
|---------|---------|----------|----------|
| **ACP** | Continues (Z-score) | Python (scikit-learn) | Facteurs latents quantitatifs |
| **ACM** | Catégorielles (tertiles) | R (FactoMineR) | Associations qualitatives |

---

*Février 2026 – Projet universitaire*
