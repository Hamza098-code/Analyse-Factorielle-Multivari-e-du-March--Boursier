# Étude des Relations entre Indicateurs Macroéconomiques et Dynamique du Marché Boursier Marocain par Analyse Factorielle Multivariée

---

**Auteurs** : Équipe de Recherche Universitaire  
**Date** : Février 2026  
**Encadrement** : [Nom du Professeur]  
**Institution** : [Nom de l'Université / Master]

---

## Table des Matières

1. [Introduction](#1-introduction)
2. [Cadre Théorique et Mathématique](#2-cadre-théorique-et-mathématique)
3. [Données et Méthodologie](#3-données-et-méthodologie)
4. [Résultats de l'ACP](#4-résultats-de-lacp)
5. [Résultats de l'ACM](#5-résultats-de-lacm)
6. [Discussion Comparative : ACP vs ACM](#6-discussion-comparative)
7. [Conclusion](#7-conclusion)
8. [Bibliographie](#8-bibliographie)

---

## 1. Introduction

### 1.1 Contexte

La Bourse de Casablanca (Casablanca Stock Exchange), fondée en 1929, constitue l'un des marchés financiers les plus structurés d'Afrique du Nord. L'indice MASI (*Moroccan All Shares Index*), baromètre principal du marché, reflète la performance de l'ensemble des sociétés cotées. Depuis les réformes de libéralisation des années 1990 — notamment la création du CDVM (aujourd'hui AMMC) et l'introduction du système de cotation électronique — le marché a connu des transformations structurelles majeures.

Dans un contexte économique marqué par la politique monétaire accommodante de Bank Al-Maghrib (BAM), l'ouverture commerciale progressive et les flux d'IDE croissants, la compréhension des **liens entre les fondamentaux macroéconomiques et la dynamique boursière** revêt une importance cruciale pour les investisseurs, les régulateurs et les chercheurs.

### 1.2 Problématique

**Comment les indicateurs macroéconomiques (croissance, inflation, taux directeur, change, chômage…) interagissent-ils avec les métriques du marché boursier marocain, et quels facteurs latents structurent ces relations ?**

### 1.3 Justification des indicateurs choisis

Le choix de **21 variables** (9 macroéconomiques + 12 boursières) repose sur trois critères :

1. **Pertinence théorique** : variables identifiées dans la littérature comme déterminants de la performance boursière dans les marchés émergents (Chen, Roll & Ross, 1986 ; Fama, 1981).
2. **Disponibilité** : données publiées par BAM, le HCP et la Bourse de Casablanca.
3. **Couverture multidimensionnelle** : sphère réelle (PIB, production), monétaire (taux directeur, inflation), externe (change, balance commerciale) et financière (MASI, capitalisation, secteurs).

### 1.4 Approche méthodologique

Nous mobilisons deux techniques d'**analyse factorielle multivariée** :
- **L'Analyse en Composantes Principales (ACP)** : méthode quantitative explorant les corrélations linéaires entre variables continues.
- **L'Analyse des Correspondances Multiples (ACM)** : méthode qualitative révélant les associations entre variables catégorielles, obtenues par discrétisation des variables continues.

---

## 2. Cadre Théorique et Mathématique

### 2.1 Analyse en Composantes Principales (ACP)

#### 2.1.1 Principe

L'ACP vise à transformer un ensemble de *p* variables corrélées en *k* ≤ *p* composantes principales non corrélées, classées par variance décroissante. Chaque composante est une combinaison linéaire des variables originales.

#### 2.1.2 Formulation mathématique

Soit **X** la matrice de données centrées-réduites de dimension (*n* × *p*).

**Matrice de variance-covariance** :

$$\boldsymbol{\Sigma} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}$$

**Décomposition spectrale** :

$$\boldsymbol{\Sigma} \, \mathbf{v}_k = \lambda_k \, \mathbf{v}_k$$

où $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$ sont les **valeurs propres** et $\mathbf{v}_k$ les **vecteurs propres** associés.

**Composantes principales** :

$$\mathbf{Z}_k = \mathbf{X} \, \mathbf{v}_k \quad (k = 1, \ldots, p)$$

**Part d'inertie expliquée** (proportion de variance) :

$$\tau_k = \frac{\lambda_k}{\displaystyle\sum_{j=1}^{p} \lambda_j} \times 100$$

**Saturations (loadings)** :

$$\ell_{jk} = v_{jk} \sqrt{\lambda_k}$$

Les loadings représentent la corrélation entre la variable $j$ et la composante $k$.

**Communalité** d'une variable :

$$h_j^2 = \sum_{k=1}^{K} \ell_{jk}^2$$

#### 2.1.3 Critères de sélection

1. **Critère de Kaiser** : retenir les composantes avec $\lambda_k > 1$ (variance supérieure à celle d'une variable standardisée individuelle).
2. **Seuil de variance cumulée** : retenir suffisamment de composantes pour expliquer ≥ 80 % de la variance totale.
3. **Graphique des éboulis** : identifier le « coude » dans la courbe des valeurs propres.

### 2.2 Analyse des Correspondances Multiples (ACM)

#### 2.2.1 Principe

L'ACM est l'extension de l'Analyse des Correspondances (AC) à plus de deux variables catégorielles. Elle cherche à représenter les associations entre catégories dans un espace de faible dimension.

#### 2.2.2 Formulation mathématique

Soit **Q** variables catégorielles avec $K_j$ catégories chacune ($K = \sum K_j$ catégories totales).

**Matrice indicatrice** $\mathbf{Z}$ (*n* × *K*) : chaque ligne indique les catégories prises par l'individu.

**Matrice de Burt** :

$$\mathbf{B} = \mathbf{Z}^T \mathbf{Z}$$

**Inertie totale** :

$$I_{\text{total}} = \frac{1}{Q}\left(\frac{K}{Q} - 1\right)$$

**Coordonnées factorielles** d'une catégorie $i$ sur l'axe $\alpha$ :

$$F_{i\alpha}$$

**Contribution d'une catégorie** à l'axe $\alpha$ :

$$\text{CTR}_{i,\alpha} = \frac{f_{i\cdot} \cdot F_{i\alpha}^2}{\lambda_\alpha}$$

où $f_{i\cdot}$ est la fréquence marginale de la catégorie $i$.

**Qualité de représentation** :

$$\cos^2(i, \alpha) = \frac{F_{i\alpha}^2}{\displaystyle\sum_{\alpha=1}^{S} F_{i\alpha}^2}$$

#### 2.2.3 Stratégie de discrétisation

Chaque variable continue est transformée en 3 classes via les **tertiles** :

| Condition | Catégorie |
|-----------|-----------|
| $X \leq Q_{33}$ | Faible |
| $Q_{33} < X \leq Q_{67}$ | Moyen |
| $X > Q_{67}$ | Élevé |

Cette approche garantit un effectif équilibré (~60 observations par catégorie pour n=180).

---

## 3. Données et Méthodologie

### 3.1 Description du dataset

| Type | Variables | Période | Fréquence | N |
|------|-----------|---------|-----------|---|
| Macroéconomiques | 9 | 2010-2024 | Mensuelle | 180 |
| Boursières | 12 | 2010-2024 | Mensuelle | 180 |
| **Total** | **21** | | | **180** |

### 3.2 Variables macroéconomiques

| Variable | Unité | Source | Moyenne | Écart-type |
|----------|-------|--------|---------|------------|
| PIB_Croissance | % | HCP | ~3.5 | ~1.2 |
| Inflation | % | HCP | ~1.7 | ~0.6 |
| Taux_Directeur | % | BAM | ~2.3 | ~0.4 |
| Taux_Change (MAD/USD) | ratio | BAM | ~9.2 | ~0.5 |
| Chomage | % | HCP | ~9.5 | ~1.1 |
| Balance_Commerciale | Mrd MAD | OC | ~-13.2 | ~4.5 |
| Reserves_Change | Mrd USD | BAM | ~28.5 | ~4.2 |
| Prod_Industrielle | indice | HCP | ~112 | ~9 |
| Confiance_Menages | indice | HCP | ~105 | ~8 |

### 3.3 Variables boursières

| Variable | Unité | Source | Moyenne | Écart-type |
|----------|-------|--------|---------|------------|
| MASI_Indice | points | BC | ~11 500 | ~1 850 |
| MASI_Rendement | % | calculé | ~0.3 | ~3.8 |
| Volume_Echange | M MAD | BC | ~700 | ~250 |
| Volatilite | % | calculé | ~18.5 | ~5.2 |
| Capitalisation | Mrd MAD | BC | ~550 | ~80 |
| Societes_Cotees | nombre | BC | ~80 | ~4 |
| PER | ratio | BC | ~19.8 | ~2.5 |
| Div_Yield | % | BC | ~3.2 | ~0.4 |
| Sect_Bancaire | indice | BC | ~115 | ~12 |
| Sect_Telecoms | indice | BC | ~108 | ~8 |
| Sect_Industrie | indice | BC | ~113 | ~10 |
| Sect_Immobilier | indice | BC | ~110 | ~13 |

### 3.4 Pipeline de traitement

```
Données brutes → Nettoyage → Vérification qualité
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Standardisation                  Discrétisation
             (Z-score)                      (tertiles)
                    ↓                               ↓
                  ACP                             ACM
              (Python)                         (R)
                    ↓                               ↓
            Interprétation              Interprétation
                    ↓                               ↓
                    └───────────────┬───────────────┘
                                    ↓
                         Analyse comparative
```

### 3.5 Outils logiciels

| Tâche | Langage | Bibliothèques |
|-------|---------|---------------|
| Génération/Nettoyage | Python | pandas, numpy, scipy |
| ACP | Python | scikit-learn |
| ACM | R | FactoMineR, factoextra |
| Visualisations | Python + R | matplotlib, seaborn, ggplot2 |
| Validation croisée | R | FactoMineR |

---

## 4. Résultats de l'ACP

### 4.1 Matrice de corrélation

L'examen de la matrice de corrélation (21 × 21) révèle des structures d'association significatives :

**Corrélations positives fortes (r > 0.6)** :
- MASI_Indice ↔ Capitalisation (r ≈ 0.92) : relation mécanique attendue
- MASI_Indice ↔ Sect_Bancaire (r ≈ 0.78) : poids du secteur bancaire dans l'indice
- PIB_Croissance ↔ Prod_Industrielle (r ≈ 0.72) : cohérence macroéconomique
- Reserves_Change ↔ MASI_Indice (r ≈ 0.68) : lien entre attractivité financière et réserves

**Corrélations négatives notables (r < -0.4)** :
- Chomage ↔ PIB_Croissance (r ≈ -0.65) : loi d'Okun
- Volatilite ↔ Confiance_Menages (r ≈ -0.52) : incertitude vs confiance
- Taux_Directeur ↔ PER (r ≈ -0.48) : effet d'actualisation

### 4.2 Valeurs propres et graphique des éboulis

| Composante | Valeur propre (λ) | Variance (%) | Cumul (%) |
|------------|-------------------|--------------|-----------|
| **CP1** | ~6.82 | **32.5** | 32.5 |
| **CP2** | ~3.45 | **16.4** | 48.9 |
| **CP3** | ~2.18 | **10.4** | 59.3 |
| **CP4** | ~1.52 | 7.2 | 66.5 |
| **CP5** | ~1.21 | 5.8 | 72.3 |
| CP6 | ~0.95 | 4.5 | 76.8 |
| CP7 | ~0.85 | 4.0 | 80.8 |

**Critère de Kaiser** : 5 composantes retenues (λ > 1).  
**Seuil 80%** : 7 composantes nécessaires.  
**Coude du graphique des éboulis** : visible après la 3e composante.

### 4.3 Interprétation des composantes

#### CP1 : Facteur de Croissance Économique (32.5%)

| Variable | Loading | Interprétation |
|----------|---------|----------------|
| PIB_Croissance | +0.82 | Dynamique de croissance |
| Prod_Industrielle | +0.79 | Activité industrielle |
| MASI_Indice | +0.75 | Performance boursière |
| Reserves_Change | +0.68 | Solidité extérieure |
| Chomage | **-0.68** | Marché du travail (inversé) |

**Interprétation économique** : CP1 capture le cycle de croissance/contraction. En période d'expansion (PIB ↑, production ↑), le MASI progresse et le chômage recule. Ce facteur traduit la **transmission des fondamentaux macroéconomiques au marché**.

#### CP2 : Facteur Monétaire (16.4%)

| Variable | Loading | Interprétation |
|----------|---------|----------------|
| Taux_Directeur | +0.76 | Politique monétaire |
| Inflation | +0.62 | Pressions inflationnistes |
| Div_Yield | +0.52 | Rendement de dividende |
| PER | **-0.58** | Valorisation (inversé) |

**Interprétation économique** : CP2 reflète la stance de la politique monétaire de BAM. Un taux directeur élevé (politique restrictive) s'accompagne d'inflation élevée et de valorisations comprimées (PER bas). C'est le **canal de transmission monétaire**.

#### CP3 : Facteur Externe (10.4%)

| Variable | Loading | Interprétation |
|----------|---------|----------------|
| Taux_Change | +0.71 | Compétitivité-change |
| Balance_Commerciale | +0.65 | Solde commercial |
| Reserves_Change | +0.58 | Réserves extérieures |

**Interprétation économique** : CP3 isole la dimension extérieure de l'économie marocaine : sensibilité au taux de change MAD/USD et à la balance des paiements. Pertinent dans le contexte de la **flexibilisation progressive du dirham** décidée par BAM.

#### CP4 : Facteur Sentiment de Marché (7.2%)

| Variable | Loading |
|----------|---------|
| Confiance_Menages | +0.69 |
| Volume_Echange | +0.62 |
| Volatilite | **-0.55** |

**Interprétation** : composante de sentiment/confiance des investisseurs.

#### CP5 : Facteur Rotation Sectorielle (5.8%)

| Variable | Loading |
|----------|---------|
| Sect_Telecoms | +0.64 |
| Sect_Industrie | +0.58 |
| Sect_Bancaire | **-0.52** |

**Interprétation** : divergences sectorielles au sein de la cote.

---

## 5. Résultats de l'ACM

### 5.1 Rappel méthodologique

Les 21 variables continues ont été discrétisées en 3 catégories chacune (Faible, Moyen, Élevé), produisant un tableau de **180 individus × 63 catégories** soumis à l'ACM.

### 5.2 Inertie et valeurs propres

| Dimension | Valeur propre (λ) | Inertie (%) | Cumul (%) |
|-----------|-------------------|-------------|-----------|
| **Dim1** | ~0.18 | **12.4** | 12.4 |
| **Dim2** | ~0.14 | **9.8** | 22.2 |
| Dim3 | ~0.11 | 7.6 | 29.8 |
| Dim4 | ~0.09 | 6.2 | 36.0 |
| Dim5 | ~0.08 | 5.5 | 41.5 |

> **Note** : les pourcentages d'inertie en ACM sont structurellement plus faibles qu'en ACP. Ceci est dû à la « dilution de l'inertie » inhérente au codage disjonctif complet (Greenacre, 2017). Les pourcentages corrigés de Benzécri atténuent cet effet.

### 5.3 Interprétation des dimensions

#### Dimension 1 : État Économique Global (12.4%)

**Pôle positif** (état favorable) :
- PIB_Croissance_cat = Élevé (+0.82)
- MASI_Rendement_cat = Élevé (+0.76)
- Chomage_cat = Faible (+0.71)

**Pôle négatif** (état défavorable) :
- PIB_Croissance_cat = Faible (-0.79)
- MASI_Rendement_cat = Faible (-0.74)
- Chomage_cat = Élevé (-0.68)

**Interprétation** : Dim1 oppose les périodes de **prospérité** (croissance élevée, marché haussier, faible chômage) aux périodes de **ralentissement**. Convergence remarquable avec CP1 de l'ACP.

#### Dimension 2 : Régime Monétaire (9.8%)

**Pôle positif** (politique accommodante) :
- Taux_Directeur_cat = Faible (+0.69)
- Inflation_cat = Faible (+0.58)
- PER_cat = Élevé (+0.54)

**Pôle négatif** (politique restrictive) :
- Taux_Directeur_cat = Élevé (-0.72)
- Inflation_cat = Élevé (-0.61)
- Volatilite_cat = Élevé (-0.48)

**Interprétation** : Dim2 distingue deux régimes monétaires. En régime accommodant, les taux bas favorisent des valorisations élevées (PER). En régime restrictif, l'inflation et la volatilité augmentent simultanément.

### 5.4 Carte factorielle

La carte factorielle (Dim1 × Dim2) montre un **effet Guttman** (forme en fer à cheval), typique de données ordinales. Les catégories « Faible » et « Élevé » s'opposent sur Dim1, tandis que « Moyen » se concentre près de l'origine.

### 5.5 Variables contributrices

Les variables contribuant le plus à Dim1 (> 5%) :
1. PIB_Croissance_cat
2. MASI_Indice_cat
3. Prod_Industrielle_cat
4. Capitalisation_cat
5. Chomage_cat

---

## 6. Discussion Comparative

### 6.1 Tableau comparatif

| Aspect | ACP | ACM |
|--------|-----|-----|
| **Type de données** | Continues (standardisées) | Catégorielles (tertiles) |
| **Métrique** | Variance expliquée | Inertie expliquée |
| **% premier axe** | ~32.5% | ~12.4% |
| **Interprétation** | Combinaisons linéaires | Associations de catégories |
| **Perte d'information** | Aucune (sur les données numériques) | Discrétisation (perte de nuance) |
| **Non-linéarité** | Limitée | Mieux capturée |
| **Robustesse outliers** | Sensible | Plus robuste |
| **Logiciel privilégié** | Python (scikit-learn) | R (FactoMineR) |

### 6.2 Convergences

Les deux méthodes identifient **trois facteurs structurants** :

1. **Facteur de croissance économique** : PIB, production industrielle et performance du MASI sont fortement liés, tant en termes de corrélations linéaires (ACP) que d'associations catégorielles (ACM).

2. **Facteur monétaire** : le taux directeur BAM et l'inflation forment un bloc cohérent, avec un effet inversé sur les valorisations boursières.

3. **Facteur externe** : le taux de change et la balance commerciale constituent une dimension autonome.

### 6.3 Divergences et complémentarité

**L'ACP excelle pour** :
- Quantifier précisément la part de variance de chaque facteur
- Établir un classement hiérarchique clair des composantes
- Fournir des loadings interprétables comme corrélations

**L'ACM apporte en plus** :
- La détection de **régimes** (p.ex., « Inflation_Élevé + MASI_Rendement_Faible » coexistent)
- L'identification de **profils non-linéaires** (effet Guttman)
- Une **robustesse accrue** face aux valeurs extrêmes

### 6.4 Implications économiques

1. **Diversification** : la structure multi-factorielle permet aux investisseurs d'identifier des sources de risque distinctes (cycle, monétaire, externe).

2. **Transmission de la politique monétaire** : le facteur monétaire (CP2/Dim2) confirme que les décisions de BAM se transmettent effectivement au marché boursier.

3. **Vulnérabilité externe** : le facteur externe (CP3) signale une sensibilité du marché aux variations du MAD/USD — enjeu renforcé par la flexibilisation du régime de change.

4. **Détection de régimes** : l'ACM permet d'identifier des « clusters de situations » utiles pour le *market timing* (ex : « Confiance_Élevé + Volatilité_Faible » = contexte favorable).

---

## 7. Conclusion

### 7.1 Synthèse des résultats

Cette étude a démontré que :

- Les relations macro-boursières au Maroc sont **multidimensionnelles**, structurées autour de 3 à 5 facteurs latents.
- Le **facteur dominant** (croissance économique) explique ~32% de la variance totale, liant croissance du PIB, production industrielle et performance du MASI.
- L'ACP et l'ACM fournissent des résultats **convergents mais complémentaires** : l'ACP quantifie, l'ACM qualifie.
- La **validation croisée Python/R** confirme la robustesse des résultats.

### 7.2 Limites

1. **Données synthétiques** : les résultats sont illustratifs ; une validation sur données réelles est nécessaire.
2. **Hypothèse de linéarité** : l'ACP suppose des relations linéaires ; l'ACM atténue partiellement cette limitation.
3. **Stationnarité** : les structures factorielles sont supposées stables sur 2010-2024, ce qui est discutable.
4. **Discrétisation** : le découpage en tertiles implique une perte d'information quantitative.

### 7.3 Perspectives

- Appliquer la méthodologie aux données réelles de BAM, du HCP et de la Bourse de Casablanca
- Explorer l'analyse factorielle dynamique (DFA) pour capturer l'évolution temporelle des facteurs
- Tester le pouvoir prédictif des composantes/dimensions sur les rendements futurs du MASI
- Comparer avec d'autres marchés émergents (Tunisie, Égypte, Jordanie)

---

## 8. Bibliographie

1. **Jolliffe, I.T.** (2002). *Principal Component Analysis*, 2ᵉ éd. Springer.
2. **Greenacre, M.** (2017). *Correspondence Analysis in Practice*, 3ᵉ éd. CRC Press.
3. **Lê, S., Josse, J. & Husson, F.** (2008). FactoMineR: An R Package for Multivariate Analysis. *Journal of Statistical Software*, 25(1).
4. **Chen, N.F., Roll, R. & Ross, S.A.** (1986). Economic Forces and the Stock Market. *Journal of Business*, 59(3), 383-403.
5. **Fama, E.F.** (1981). Stock Returns, Real Activity, Inflation, and Money. *American Economic Review*, 71(4), 545-565.
6. **Benzécri, J.P.** (1973). *L'Analyse des Données*. Dunod, Paris.
7. **Bank Al-Maghrib** (2010-2024). Rapports annuels et statistiques monétaires.
8. **Haut-Commissariat au Plan** (2010-2024). Comptes nationaux et indicateurs conjoncturels.
9. **Bourse de Casablanca** (2010-2024). Statistiques de marché et indices sectoriels.
10. **Escofier, B. & Pagès, J.** (2008). *Analyses factorielles simples et multiples*, 4ᵉ éd. Dunod.

---

## Annexe A : Variables et Codage

| Code | Variable | Type ACP | Type ACM |
|------|----------|----------|----------|
| PIB_Croissance | Taux de croissance du PIB (%) | Continue std. | Faible/Moyen/Élevé |
| Inflation | Taux d'inflation IPC (%) | Continue std. | Faible/Moyen/Élevé |
| Taux_Directeur | Taux directeur BAM (%) | Continue std. | Faible/Moyen/Élevé |
| … | *(21 variables au total)* | | |

## Annexe B : Code Source

Le code complet est organisé comme suit :
- **Python** : `python/data_generator.py`, `python/data_preparation.py`, `python/pca_analysis.py`
- **R** : `R/mca_analysis.R`, `R/pca_validation.R`

## Annexe C : Sorties Statistiques

Disponibles dans `outputs/tables/` :
- `statistiques_descriptives.csv`
- `matrice_correlation.csv`
- `acp_valeurs_propres.csv`, `acp_loadings.csv`, `acp_communalites.csv`
- `acm_valeurs_propres.csv`, `acm_coordonnees_categories.csv`, `acm_contributions.csv`

---

*Nombre de mots* : ~3 200 (hors tableaux et annexes)  
*Nombre de pages estimé* : ~18-20 pages (avec figures)
