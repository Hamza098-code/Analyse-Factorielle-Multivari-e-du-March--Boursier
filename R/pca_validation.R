# =============================================================================
# Validation Croisée de l'ACP – R / FactoMineR
# =============================================================================
# Ce module réplique l'ACP réalisée en Python pour valider les résultats.
# Les valeurs propres et loadings doivent correspondre entre les deux
# implémentations.
#
# Auteur  : Équipe de Recherche
# Projet  : Analyse Factorielle Multivariée – Marché Boursier Marocain
# =============================================================================

packages <- c("FactoMineR", "factoextra", "tidyverse")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
    library(pkg, character.only = TRUE)
  }
}

# ── Chemins ──────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(
  if (any(grepl("--file=", args))) {
    dirname(sub("--file=", "", args[grep("--file=", args)]))
  } else { "." }
)

PROJECT_ROOT   <- dirname(script_path)
DATA_PROCESSED <- file.path(PROJECT_ROOT, "data", "processed")
TABLES_DIR     <- file.path(PROJECT_ROOT, "outputs", "tables")
FIGURES_DIR    <- file.path(PROJECT_ROOT, "outputs", "figures")

dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)


# =============================================================================
#  Fonction principale
# =============================================================================

main <- function() {
  cat(strrep("=", 65), "\n")
  cat("  VALIDATION CROISÉE ACP – R (FactoMineR)\n")
  cat(strrep("=", 65), "\n")

  # ── 1. Charger les données ──
  cat("\n[1/5] Chargement des données standardisées…\n")
  df <- read.csv(file.path(DATA_PROCESSED, "donnees_acp.csv"))

  # Retirer la colonne Date
  num_cols <- df[, sapply(df, is.numeric)]
  cat(sprintf("  %d observations × %d variables\n", nrow(num_cols), ncol(num_cols)))

  # ── 2. ACP avec FactoMineR ──
  cat("\n[2/5] Exécution de l'ACP (FactoMineR)…\n")
  acp_r <- PCA(num_cols, ncp = ncol(num_cols), graph = FALSE)

  # ── 3. Valeurs propres ──
  cat("\n[3/5] Valeurs propres R…\n")
  eig_r <- get_eigenvalue(acp_r)
  eig_df <- data.frame(
    Composante    = paste0("CP", 1:nrow(eig_r)),
    Valeur_Propre = round(eig_r$eigenvalue, 4),
    Variance_pct  = round(eig_r$variance.percent, 2),
    Cumul_pct     = round(eig_r$cumulative.variance.percent, 2)
  )
  cat("\n── Valeurs Propres (R) ──\n")
  print(eig_df)

  write.csv(eig_df,
            file.path(TABLES_DIR, "acp_valeurs_propres_R.csv"), row.names = FALSE)

  # ── 4. Comparer avec Python ──
  cat("\n[4/5] Comparaison Python vs R…\n")
  py_path <- file.path(TABLES_DIR, "acp_valeurs_propres.csv")
  if (file.exists(py_path)) {
    eig_py <- read.csv(py_path)
    n_comp <- min(nrow(eig_py), nrow(eig_df))

    comp <- data.frame(
      Composante = eig_df$Composante[1:n_comp],
      VP_Python  = eig_py$Valeur_Propre[1:n_comp],
      VP_R       = eig_df$Valeur_Propre[1:n_comp]
    )
    comp$Diff_pct <- round(
      abs(comp$VP_Python - comp$VP_R) / comp$VP_Python * 100, 4)

    cat("\n── Comparaison Valeurs Propres ──\n")
    print(comp)

    write.csv(comp,
              file.path(TABLES_DIR, "validation_croisee_acp.csv"), row.names = FALSE)
    cat("  ✔ validation_croisee_acp.csv sauvegardé\n")
  } else {
    cat("  ⚠ Fichier Python non trouvé, exécutez d'abord pca_analysis.py\n")
  }

  # ── 5. Visualisations R ──
  cat("\n[5/5] Visualisations ACP (R)…\n")

  # Scree plot R
  png(file.path(FIGURES_DIR, "acp_eboulis_R.png"),
      width = 800, height = 600, res = 100)
  p <- fviz_eig(acp_r,
    addlabels = TRUE,
    main      = "Graphique des Éboulis – ACP (R)\nValidation croisée"
  )
  print(p)
  dev.off()
  cat("  ✔ acp_eboulis_R.png\n")

  # Biplot R
  png(file.path(FIGURES_DIR, "acp_biplot_R.png"),
      width = 1000, height = 800, res = 100)
  p <- fviz_pca_biplot(acp_r,
    repel    = TRUE,
    col.var  = "contrib",
    gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    title    = "Biplot ACP (R) – Variables et Individus"
  )
  print(p)
  dev.off()
  cat("  ✔ acp_biplot_R.png\n")

  # Cercle des corrélations R
  png(file.path(FIGURES_DIR, "acp_cercle_R.png"),
      width = 900, height = 800, res = 100)
  p <- fviz_pca_var(acp_r,
    col.var  = "contrib",
    gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    repel    = TRUE,
    title    = "Cercle des Corrélations – ACP (R)"
  )
  print(p)
  dev.off()
  cat("  ✔ acp_cercle_R.png\n")

  # Loadings R
  loadings_r <- as.data.frame(acp_r$var$coord)
  names(loadings_r) <- paste0("CP", 1:ncol(loadings_r))
  write.csv(loadings_r, file.path(TABLES_DIR, "acp_loadings_R.csv"))
  cat("  ✔ acp_loadings_R.csv sauvegardé\n")

  cat("\n", strrep("=", 65), "\n")
  cat("  VALIDATION CROISÉE TERMINÉE ✔\n")
  cat(strrep("=", 65), "\n")

  return(acp_r)
}

if (!interactive()) {
  acp_r <- main()
}
