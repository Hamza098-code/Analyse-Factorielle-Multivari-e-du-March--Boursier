# =============================================================================
# Analyse des Correspondances Multiples (ACM) – R / FactoMineR
# =============================================================================
# Ce module réalise l'ACM complète sur les données catégorielles :
#   1. Chargement des données discrétisées
#   2. Calcul des dimensions, inertie, contributions
#   3. Visualisations : éboulis, carte factorielle, contributions, biplot
#   4. Qualité de représentation (cos²)
#
# Formules clés :
#   Inertie totale     : I = (1/K) Σ (K_j - 1)  (K = nb total de catégories)
#   Inertie dimension  : τ_α = λ_α / I_total × 100
#   Contribution       : CTR_{i,α} = (f_i · F²_{iα}) / λ_α
#   Qualité (cos²)     : cos²(i,α) = F²_{iα} / d²(i, G)
#
# Auteur  : Équipe de Recherche
# Projet  : Analyse Factorielle Multivariée – Marché Boursier Marocain
# =============================================================================

# ── Chargement des packages ──────────────────────────────────────────────────
packages <- c("FactoMineR", "factoextra", "tidyverse", "ggplot2", "corrplot")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
    library(pkg, character.only = TRUE)
  }
}

# ── Configuration des chemins ────────────────────────────────────────────────
args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(
  if (any(grepl("--file=", args))) {
    dirname(sub("--file=", "", args[grep("--file=", args)]))
  } else {
    "."
  }
)

PROJECT_ROOT <- dirname(script_path)
DATA_PROCESSED <- file.path(PROJECT_ROOT, "data", "processed")
TABLES_DIR     <- file.path(PROJECT_ROOT, "outputs", "tables")
FIGURES_DIR    <- file.path(PROJECT_ROOT, "outputs", "figures")

dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)


# =============================================================================
#  Fonctions d'Analyse ACM
# =============================================================================

executer_acm <- function(df, ncp = 5, graph = FALSE) {
  #' Exécute l'ACM avec FactoMineR.
  #'
  #' @param df   Data.frame avec colonnes catégorielles
  #' @param ncp  Nombre de dimensions à conserver
  #' @return     Objet MCA de FactoMineR

  # Retirer la colonne Date si présente
  if ("Date" %in% names(df)) {
    df <- df[, names(df) != "Date"]
  }

  # Conversion en facteurs
  for (col in names(df)) {
    if (!is.factor(df[[col]])) {
      df[[col]] <- factor(df[[col]], levels = c("Faible", "Moyen", "Élevé"))
    }
  }

  # Lancer l'ACM
  resultat <- MCA(df, ncp = ncp, graph = graph)
  return(resultat)
}


extraire_valeurs_propres <- function(acm) {
  #' Extrait les valeurs propres et l'inertie.
  eig <- get_eigenvalue(acm)
  data.frame(
    Dimension        = paste0("Dim", 1:nrow(eig)),
    Valeur_Propre    = round(eig$eigenvalue, 4),
    Inertie_pct      = round(eig$variance.percent, 2),
    Cumul_pct        = round(eig$cumulative.variance.percent, 2)
  )
}


extraire_coordonnees <- function(acm) {
  #' Coordonnées des catégories dans l'espace des dimensions.
  coord <- as.data.frame(acm$var$coord)
  names(coord) <- paste0("Dim", 1:ncol(coord))
  coord
}


extraire_contributions <- function(acm) {
  #' Contributions des catégories à chaque dimension (%).
  #' CTR_{i,α} = (f_i · F²_{iα}) / λ_α
  contrib <- as.data.frame(acm$var$contrib)
  names(contrib) <- paste0("Dim", 1:ncol(contrib))
  contrib
}


extraire_cos2 <- function(acm) {
  #' Qualité de représentation (cos²).
  #' cos²(i,α) = F²_{iα} / d²(i, G)
  cos2 <- as.data.frame(acm$var$cos2)
  names(cos2) <- paste0("Dim", 1:ncol(cos2))
  cos2
}


contributions_variables <- function(acm) {
  #' Contributions agrégées par variable (pas par catégorie).
  contrib <- extraire_contributions(acm)
  categories <- rownames(contrib)

  # Extraire le nom de variable (avant _cat_)
  noms_var <- sub("_cat_.*$", "", categories)

  contrib$Variable <- noms_var
  var_contrib <- contrib %>%
    group_by(Variable) %>%
    summarise(across(starts_with("Dim"), sum)) %>%
    as.data.frame()

  rownames(var_contrib) <- var_contrib$Variable
  var_contrib$Variable <- NULL
  var_contrib
}


interpreter_dimensions <- function(acm, n_top = 6) {
  #' Interprète les pôles positifs et négatifs de chaque dimension.
  coord <- extraire_coordonnees(acm)
  eig   <- extraire_valeurs_propres(acm)
  interp <- list()

  for (i in 1:min(5, ncol(coord))) {
    dim_name <- paste0("Dim", i)

    # Pôle positif
    pos_idx <- order(coord[, dim_name], decreasing = TRUE)[1:(n_top / 2)]
    positif <- coord[pos_idx, dim_name]
    names(positif) <- rownames(coord)[pos_idx]

    # Pôle négatif
    neg_idx <- order(coord[, dim_name], decreasing = FALSE)[1:(n_top / 2)]
    negatif <- coord[neg_idx, dim_name]
    names(negatif) <- rownames(coord)[neg_idx]

    interp[[dim_name]] <- list(
      inertie    = eig$Inertie_pct[i],
      pole_positif = positif,
      pole_negatif = negatif
    )
  }
  interp
}


# =============================================================================
#  Visualisations ACM
# =============================================================================

graphique_eboulis_acm <- function(acm) {
  #' Graphique des éboulis de l'inertie (ACM).
  png(file.path(FIGURES_DIR, "acm_eboulis.png"),
      width = 800, height = 600, res = 100)

  p <- fviz_eig(acm,
    addlabels = TRUE,
    main = "Graphique des Éboulis – ACM\nInertie par Dimension"
  )
  print(p)
  dev.off()
  cat("  ✔ acm_eboulis.png\n")
}


carte_factorielle_acm <- function(acm) {
  #' Carte factorielle des catégories (Dim1 × Dim2).
  png(file.path(FIGURES_DIR, "acm_carte_factorielle.png"),
      width = 1000, height = 800, res = 100)

  p <- fviz_mca_var(acm,
    col.var      = "contrib",
    gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    repel        = TRUE,
    title        = "Carte Factorielle – ACM\nContribution des catégories (Dim1 × Dim2)"
  )
  print(p)
  dev.off()
  cat("  ✔ acm_carte_factorielle.png\n")
}


biplot_acm <- function(acm) {
  #' Biplot ACM : individus + catégories.
  png(file.path(FIGURES_DIR, "acm_biplot.png"),
      width = 1000, height = 800, res = 100)

  p <- fviz_mca_biplot(acm,
    repel     = TRUE,
    col.var   = "darkred",
    col.ind   = "gray50",
    alpha.ind = 0.4,
    title     = "Biplot ACM – Observations et Catégories\nMarché Boursier Marocain"
  )
  print(p)
  dev.off()
  cat("  ✔ acm_biplot.png\n")
}


contributions_dim <- function(acm, dim = 1) {
  #' Diagramme des contributions à une dimension donnée.
  png(file.path(FIGURES_DIR, sprintf("acm_contrib_dim%d.png", dim)),
      width = 900, height = 600, res = 100)

  p <- fviz_contrib(acm,
    choice = "var",
    axes   = dim,
    top    = 15,
    title  = sprintf("Contributions des catégories – Dimension %d", dim)
  )
  print(p)
  dev.off()
  cat(sprintf("  ✔ acm_contrib_dim%d.png\n", dim))
}


cos2_plot <- function(acm) {
  #' Qualité de représentation (cos²) des catégories.
  png(file.path(FIGURES_DIR, "acm_cos2.png"),
      width = 1000, height = 800, res = 100)

  p <- fviz_mca_var(acm,
    col.var       = "cos2",
    gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    repel         = TRUE,
    title         = "Qualité de Représentation (cos²) – ACM"
  )
  print(p)
  dev.off()
  cat("  ✔ acm_cos2.png\n")
}


# =============================================================================
#  Sauvegarde des résultats
# =============================================================================

sauvegarder_resultats <- function(acm) {
  #' Sauvegarde tous les résultats ACM dans outputs/tables/.

  # Valeurs propres
  write.csv(extraire_valeurs_propres(acm),
            file.path(TABLES_DIR, "acm_valeurs_propres.csv"), row.names = FALSE)

  # Coordonnées des catégories
  write.csv(extraire_coordonnees(acm),
            file.path(TABLES_DIR, "acm_coordonnees_categories.csv"))

  # Contributions
  write.csv(extraire_contributions(acm),
            file.path(TABLES_DIR, "acm_contributions.csv"))

  # Contributions par variable
  write.csv(contributions_variables(acm),
            file.path(TABLES_DIR, "acm_contributions_variables.csv"))

  # Cos²
  write.csv(extraire_cos2(acm),
            file.path(TABLES_DIR, "acm_cos2.csv"))

  # Coordonnées des individus
  ind_coord <- as.data.frame(acm$ind$coord)
  names(ind_coord) <- paste0("Dim", 1:ncol(ind_coord))
  write.csv(ind_coord,
            file.path(DATA_PROCESSED, "scores_acm.csv"))

  cat("  ✔ Résultats ACM sauvegardés dans outputs/tables/\n")
}


# =============================================================================
#  Exécution Principale
# =============================================================================

main <- function() {
  cat(strrep("=", 65), "\n")
  cat("  ANALYSE DES CORRESPONDANCES MULTIPLES (ACM) – R\n")
  cat(strrep("=", 65), "\n")

  # ── 1. Charger les données ──
  cat("\n[1/6] Chargement des données discrétisées…\n")
  data_path <- file.path(DATA_PROCESSED, "donnees_acm.csv")
  df <- read.csv(data_path, stringsAsFactors = FALSE)

  # Sélectionner les colonnes catégorielles
  cat_cols <- names(df)[grepl("_cat$", names(df))]
  df_cat <- df[, cat_cols]
  cat(sprintf("  %d observations × %d variables catégorielles\n",
              nrow(df_cat), ncol(df_cat)))

  # ── 2. Exécuter l'ACM ──
  cat("\n[2/6] Ajustement du modèle ACM…\n")
  acm <- executer_acm(df_cat)

  # ── 3. Valeurs propres ──
  cat("\n[3/6] Analyse de l'inertie…\n")
  vp <- extraire_valeurs_propres(acm)
  cat("\n── Valeurs Propres (Inertie) ──\n")
  print(vp)

  # ── 4. Contributions ──
  cat("\n[4/6] Contributions des variables…\n")
  var_contrib <- contributions_variables(acm)
  cat("\n── Contributions Variables (Dim1, top 10) ──\n")
  print(head(var_contrib[order(var_contrib$Dim1, decreasing = TRUE), ], 10))

  # ── 5. Interprétation ──
  cat("\n[5/6] Interprétation des dimensions…\n")
  interp <- interpreter_dimensions(acm)

  cat("\n── Interprétation Économique ──\n")
  for (dim in names(interp)) {
    info <- interp[[dim]]
    cat(sprintf("\n  %s (%.1f%% d'inertie) :\n", dim, info$inertie))

    cat("    Pôle positif :\n")
    pos <- head(info$pole_positif, 3)
    for (i in seq_along(pos)) {
      cat(sprintf("      + %s : %.3f\n", names(pos)[i], pos[i]))
    }

    cat("    Pôle négatif :\n")
    neg <- head(info$pole_negatif, 3)
    for (i in seq_along(neg)) {
      cat(sprintf("      - %s : %.3f\n", names(neg)[i], neg[i]))
    }
  }

  # ── 6. Visualisations ──
  cat("\n[6/6] Génération des visualisations…\n")
  graphique_eboulis_acm(acm)
  carte_factorielle_acm(acm)
  biplot_acm(acm)
  contributions_dim(acm, dim = 1)
  contributions_dim(acm, dim = 2)
  cos2_plot(acm)

  # ── Sauvegarde ──
  sauvegarder_resultats(acm)

  cat("\n", strrep("=", 65), "\n")
  cat("  ACM TERMINÉE ✔\n")
  cat(strrep("=", 65), "\n")

  return(acm)
}

# Exécution directe
if (!interactive()) {
  acm <- main()
}
