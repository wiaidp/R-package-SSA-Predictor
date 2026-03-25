# ============================================================
# Tutorial 7.1: M-SSA in a Multivariate Nowcast Simulation Framework
# ============================================================
#
# Overview:
#   - Demonstrates customization of the Hodrick-Prescott (HP) filter
#     in a five-dimensional multivariate setting.
#   - Extends Tutorial 2.1 (univariate HP customization) to the
#     multivariate case via M-SSA.
#
# Simulation Design:
#   - The tutorial is structured around a simulation exercise.
#   - The model specification is derived from an application of M-SSA
#     to forecasting German GDP
#     (BIP: Brutto Inlands Produkt — Gross Domestic Product).
#     Empirical data are available in Tutorial 7.3.
#   - The data-generating process (DGP) is based on a VAR(1) model
#     fitted to German macroeconomic data.
#     See Tutorials 7.2 and 7.3 for full details.
#
# Objectives:
#   - Demonstrate that M-SSA performs as expected when the DGP is assumed known.
#
#   Exercise 1:
#     - Verifies that finite-sample performance converges to theoretical expectations,
#       thereby confirming that the formulas derived in the M-SSA paper are valid.
#
#   Exercise 2:
#     - Transcribes the paper's formulas into R code and provides
#       additional performance checks under nearly asymptotic conditions.
#
#   Exercise 3:
#     - Wraps the proposed code into compact, reusable R functions,
#       which are subsequently used in Tutorials 7.2 and 7.3.
#
# Methodological Context:
#   - As presented at the 2025 Conference on Real-Time Data Analysis,
#     Methods, and Applications (Prague, 2025):
#     SSA and M-SSA are "empirical forecasting" frameworks that extend
#     the classical mean-square error (MSE) paradigm by additionally
#     addressing:
#       * Smoothness: controls the rate of zero-crossings of a
#         (mean-centered) predictor/nowcast/forecast.
#       * Timeliness: addresses phase delay (lag/right-shift) of a predictor.
#
#   - M-SSA extends these concepts to a multivariate setting, where
#     multiple input series are combined to predict a single target variable.
#
#   - In this multivariate context, controlling zero-crossings is more
#     complex, since each individual series contributes its own crossing
#     behavior to the aggregated predictor.
#
#   - The mathematical solution to this challenge is both elegant and effective:
#       * The lag-one autocorrelation function (ACF) for a multivariate
#         design can be derived analytically (see the M-SSA paper).
#       * The rate of zero-crossings is directly related to the lag-one ACF
#         (see the SSA and M-SSA papers).
#       * Exercise 1 empirically validates this theoretical relationship.
#
# ============================================================
# References
# ============================================================
#
# Wildi, M. (2026b). The Accuracy-Smoothness Dilemma in Prediction:
#   A Novel Multivariate M-SSA Forecast Approach.
#   Journal of Time Series Analysis, http://doi.org/10.1111/jtsa.70058 
#   arXiv: https://doi.org/10.48550/arXiv.2602.13722
#
# Heinisch, K., Van Norden, S., and Wildi, M. (2026).
#   Smooth and Persistent Forecasts of German GDP:
#   Balancing Accuracy and Stability.
#   IWH Discussion Papers, 1/2026.
#   Halle Institute for Economic Research.
#   https://doi.org/10.18717/dp99kr-7336
#
# ============================================================



# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------

# Clear the workspace to ensure a clean environment
rm(list = ls())

# ------------------------------------------------------------
# Load Required Libraries
# ------------------------------------------------------------

# mFilter: provides standard filtering tools, including the HP filter
library(mFilter)

# MTS: supports multivariate time series analysis;
#      used here to fit and simulate from VAR/VARMA models
library(MTS)

# ------------------------------------------------------------
# Load M-SSA Source Files
# ------------------------------------------------------------

# Core M-SSA functions (multivariate signal extraction and filter design)
source(paste(getwd(), "/R/functions_MSSA.r", sep = ""))

# Auxiliary signal extraction functions developed for the JBCY paper
# (depends on the mFilter package)
source(paste(getwd(), "/R/HP_JBCY_functions.r", sep = ""))

# ============================================================
# Exercise 1
# ============================================================

# ------------------------------------------------------------
# 1.1 Specify the Data-Generating Process (DGP)
# ------------------------------------------------------------
#
# Model Origin:
#   - The following VAR(1) model was obtained by fitting a parsimonious
#     specification to quarterly German macroeconomic data.
#     See Tutorials 7.2 and 7.3 for the full estimation procedure.
#   - Since the estimation span can be adjusted, the specific VAR(1)
#     coefficients below may differ slightly from those in Tutorials 7.2/7.3.
#   - However, any valid multivariate stationary model can serve as the DGP
#     to illustrate M-SSA.
#
# Variable Set (5-dimensional design):
#   - BIP         : German GDP (Brutto Inlands Produkt)
#   - ip          : Industrial production
#   - ifo_c       : ifo business climate indicator
#   - ESI         : Economic Sentiment Indicator
#   - spr_10y_3m  : Interest rate spread (10-year vs. 3-month)
#   - All series are log-transformed (except the spread),
#     first-differenced (no cointegration assumed), and standardized
#     to account for differences in scale.
#
# Data Span:
#   - The quarterly series begin at the introduction of the Euro.
#   - The in-sample estimation window (Tutorials 7.2/7.3) ends in December 2007,
#     leaving the 2008 financial crisis entirely out-of-sample.
#   - The resulting in-sample span is relatively short (< 20 years of quarterly data),
#     which makes it challenging to verify asymptotic properties empirically.
#     This motivates the simulation experiment conducted here.
#
# Model Specification Choices:
#   - Overfitting is a concern given the short sample.
#   - A sparsely parameterized VAR(1) is selected (no MA terms),
#     partly because VARMA(1,1) models are not always invertible.
#   - The sparse structure is reflected by the many zero entries in Phi below.
#   - Note: M-SSA is relatively robust to overfitting and to the
#     choice of in-sample estimation window.
#   - Estimation is performed using the MTS package (VARMA, R. Tsay).
#     See Tutorials 7.2 and 7.3 for details.

# Dimension of the multivariate system
n <- 5

# AR order: MA terms are excluded due to potential invertibility issues
#           in VARMA(1,1) specifications
p <- 1

# VAR(1) coefficient matrix Phi (n x n):
#   - Row i contains the lagged coefficients for equation i.
#   - Example: BIP (row 1) depends only on lagged ifo_c;
#              ip (row 2) depends on lagged BIP, ip, and ifo_c; etc.
#   - Zero entries reflect the sparse parameterization.
#   - Cross-series dependencies beyond Phi are further shaped by the
#     residual covariance matrix Sigma (see below).
#   - For a complete picture of dynamic interdependencies,
#     consult the impulse response functions (MA-inversion) further below.
#   - M-SSA treats this model as the true DGP and derives optimal
#     predictors accordingly.
#   - A better-fitting model could, in principle, be substituted to
#     yield improved predictors — but evaluating the VAR(1)'s fit is
#     not the purpose of this tutorial; we accept it as given.
Phi <- matrix(rbind(
  c( 0.0000000,  0.00000000, 0.4481816,    0, 0.0000000),
  c( 0.2387036, -0.33015450, 0.5487510,    0, 0.0000000),
  c( 0.0000000,  0.00000000, 0.4546929,    0, 0.3371898),
  c( 0.0000000,  0.07804158, 0.4470288,    0, 0.3276132),
  c( 0.0000000,  0.00000000, 0.0000000,    0, 0.3583553)
), nrow = n)
colnames(Phi) <- rownames(Phi) <- c("BIP", "ip", "ifo_c", "ESI", "spr_10y_3m")
Phi

# Residual covariance matrix Sigma (n x n):
#   - Captures contemporaneous correlations among model innovations.
#   - If both Phi and Sigma were diagonal, the VAR(1) would reduce to
#     n independent univariate models running in parallel — which is
#     precisely the univariate SSA setting.
Sigma <- matrix(rbind(
  c( 0.755535544,  0.49500481,  0.11051024,  0.007546104, -0.16687913),
  c( 0.495004806,  0.65832962,  0.07810020,  0.025101191, -0.25578971),
  c( 0.110510236,  0.07810020,  0.66385111,  0.502140497,  0.08539719),
  c( 0.007546104,  0.02510119,  0.50214050,  0.639843288,  0.05908741),
  c(-0.166879134, -0.25578971,  0.08539719,  0.059087406,  0.84463448)
), nrow = n)

# ── Short simulation: 25 years of quarterly data (T = 100) ──────────────────
# Used for visual inspection; mimics realistic empirical sample sizes.
len <- 100
set.seed(31)
x_mat <- VARMAsim(len, arlags = c(p), phi = Phi, sigma = Sigma)$series

# Attach original variable names
colnames(x_mat) <- c("BIP", "ip", "ifo_c", "ESI", "spr_10y_3m")

# Plot the simulated series:
#   - Expect mutually correlated (cross-sectional) and
#     autocorrelated (longitudinal) behavior.
#   - Look for "crisis episodes": sustained periods of negative growth.
par(mfrow = c(1, 1))
ts.plot(x_mat[(len - 99):len, ])

# ── Long simulation: T = 100,000 observations ────────────────────────────────
# Used to verify that finite-sample estimates converge to their
# theoretical (asymptotic) counterparts, validating the M-SSA formulas.
len <- 100000
set.seed(871)
x_mat <- VARMAsim(len, arlags = c(p), phi = Phi, sigma = Sigma)$series

# Attach original variable names
colnames(x_mat) <- c("BIP", "ip", "ifo_c", "ESI", "spr_10y_3m")


# ------------------------------------------------------------
# 1.2 Specify the Acausal (Two-Sided) Target Filter
# ------------------------------------------------------------
#
# Target Signal:
#   - The prediction target is the output of a two-sided
#     Hodrick-Prescott (HP) filter applied to BIP.
#   - HP is not necessarily the optimal signal design for this application;
#     alternative targets could be substituted (see Tutorials 3, 4, and 6).
#   - The key point is that M-SSA accepts any user-specified target filter
#     and derives the corresponding optimal causal predictor, given the DGP.
#
# HP Parameter Choice:
#   - The standard quarterly HP specification uses lambda = 1600.
#   - Here we use lambda = 160, a less aggressive smoothing parameter.
#   - Rationale: HP(1600) removes too much economically relevant variation
#     for GDP prediction purposes (see Tutorials 7.2 and 7.3 for a full
#     discussion; see also Phillips and Jin (2021) for a related critique).
lambda_HP <- 160

# Filter length:
#   - L = 31 corresponds to roughly 4 years of quarterly lags (one-sided).
#   - Technical note: only the right half of the two-sided filter is passed
#     to M-SSA; the full symmetric filter is reconstructed internally by
#     "mirroring" (reflecting) the right half.
#   - L must be odd so that the mirrored reconstruction yields a perfectly
#     symmetric filter with a single peak at its center.
L <- 31

# Compute the HP filter object using the mFilter-based wrapper:
#   - hp_symmetric        : truncated symmetric (two-sided) HP filter of length L
#   - hp_classic_concurrent : standard one-sided (concurrent) HP filter
#   - hp_one_sided        : right half of the two-sided HP filter (length L),
#                           used as input to M-SSA
HP_obj               <- HP_target_mse_modified_gap(L, lambda_HP)
hp_symmetric         <- HP_obj$target
hp_classic_concurrent <- HP_obj$hp_trend
hp_one_sided         <- HP_obj$hp_mse

# Plot the truncated symmetric filter:
#   - The truncation at length L = 31 is visibly imperfect,
#     suggesting that L alone is insufficient for full filter representation.
ts.plot(hp_symmetric,
        main = paste("Symmetric filter of length ", L, sep = ""))

# Plot the right half (one-sided) filter passed to M-SSA:
#   - M-SSA reconstructs the full two-sided filter of length 2*L - 1
#     by mirroring this one-sided input.
#   - The extended length resolves the truncation issue noted above.
ts.plot(hp_one_sided,
        main = paste("Right tail of length ", L,
                     " of the two-sided filter of length ", 2 * L - 1,
                     sep = ""),
        ylab = "")

# Verify the mirroring reconstruction explicitly:
#   - Reflects hp_one_sided around its first element to form
#     the complete symmetric two-sided filter of length 2*L - 1.
#   - This step confirms why L must be odd.
reconstructed_two_sided_hp <- c(hp_one_sided[L:2], hp_one_sided[1:L])
ts.plot(reconstructed_two_sided_hp,
        main = paste("Reconstructed two-sided filter of length ",
                     2 * L - 1, sep = ""),
        ylab = "")


# ------------------------------------------------------------
# 1.3 MA Inversion of the VAR(1): Impulse Response Functions
# ------------------------------------------------------------
#
# Background:
#   - The M-SSA optimization criterion is grounded in the Wold decomposition
#     of a stationary process (see the M-SSA paper for details).
#   - The Wold decomposition requires the MA(∞) representation of the VAR(1),
#     which is obtained here via MA inversion using the MTS package.

# Compute MA-coefficient matrices (impulse response weights) up to lag L
xi_psi <- PSIwgt(Phi = Phi, Theta = NULL, lag = L, plot = F, output = F)
xi_p   <- xi_psi$psi.weight

# Reformat the MA-inversion output into the layout expected by M-SSA:
#   - The result xi is an (n x n*L) matrix.
#   - Columns (i-1)*L + 1 through i*L contain the impulse response weights
#     associated with the i-th white noise (innovation) series, for lags 1 to L.
#   - That is, reading left to right: L weights for innovation 1,
#     followed by L weights for innovation 2, ..., up to innovation n.
xi <- matrix(nrow = n, ncol = n * L)
for (i in 1:n) {
  for (j in 1:L)
    xi[, (i - 1) * L + j] <- xi_p[, i + (j - 1) * n]
}

# ── Plot the impulse response functions ─────────────────────────────────────
# Each panel shows how one target variable responds over time to shocks
# originating from each of the n innovation series.
par(mfrow = c(1, n))
for (i in 1:n) {
  # Collect the first min(10, L) impulse response weights for variable i,
  # one block per innovation series
  mplot <- xi[i, 1:min(10, L)]
  for (j in 2:n) {
    mplot <- cbind(mplot, xi[i, (j - 1) * L + 1:min(10, L)])
  }
  colnames(mplot) <- colnames(x_mat)
  
  colo <- rainbow(ncol(mplot))
  ts.plot(mplot, col = colo,
          main = paste("MA inversion: ", colnames(x_mat)[i], sep = ""))
  for (i in 1:ncol(mplot)) {
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}
# Interpretation:
#   - The impulse response functions serve two purposes:
#       (i)  Explainability: they reveal the dynamic structure of the VAR(1)
#            and support economic narrative.
#       (ii) Verification: they provide an intuitive check on model behavior.
#   - Example: the rightmost panel (spread) shows that the spread responds
#     only to its own lagged innovations, indicating that spread is a leading
#     variable and is not contemporaneously driven by the other series.
#   - More generally, the plots reveal which series act as leading indicators
#     for others within the VAR(1) system.


# ------------------------------------------------------------
# 1.4 Target Filter Specification for M-SSA
# ------------------------------------------------------------
#
# General Setup:
#   - A target filter must be specified for each of the n = 5 series.
#   - Here, the target for each series is the two-sided HP filter applied
#     to that specific series (nowcast: unshifted).
#   - Alternatively, the target could be shifted forward (forecast) or
#     backward (backcast) to suit different prediction horizons.
#
# Filter Matrix Structure:
#   - The target filter matrix has dimension n x (n*L):
#       * n rows, one per target series.
#       * n*L columns, comprising n blocks of L coefficients each.
#   - Row i collects the filters applied to all n input series when
#     predicting target series i.
#   - Block j (columns (j-1)*L+1 through j*L) in row i contains
#     the length-L sub-filter applied to input series j.
#   - In the present setup, the off-diagonal blocks are zero:
#     each series is targeted by HP applied to itself only.

# ── Specify the target filter for series 1 (BIP / GDP) ──────────────────────
# Row 1: HP coefficients for series 1, followed by zeros for series 2 to n.
gamma_target <- c(hp_one_sided, rep(0, (n - 1) * L))

# Visualize the target filter for series 1 (BIP):
#   - Vertical lines delimit the L-coefficient block for each input series.
#   - Only the first block is non-zero (HP applied to BIP).
#   - Note: this is still the one-sided (right-tail) representation;
#     the left tail will be obtained by mirroring, as described below.
par(mfrow = c(1, 1))
ts.plot(gamma_target)
abline(v = (1:n) * L)

# ── Specify the target filters for series 2 to n ────────────────────────────
# Each row i places hp_one_sided in block i and zeros elsewhere,
# indicating that target series i is specified by HP applied to series i only.
for (i in 2:n)
  gamma_target <- rbind(gamma_target,
                        c(rep(0, (i - 1) * L), hp_one_sided, rep(0, (n - i) * L)))

# Visualize all n target filters overlaid:
#   - Each color corresponds to one target series.
#   - Vertical lines mark the boundaries between per-series filter blocks.
#   - The diagonal structure confirms that each target relies only on
#     its own HP filter, with zero weights assigned to the other series.
par(mfrow = c(1, 1))
ts.plot(t(gamma_target), col = rainbow(n))
abline(v = (1:n) * L + 1)

# ── Instruct M-SSA to reconstruct symmetric (two-sided) targets ─────────────
# The filter rows in gamma_target are one-sided (right-tail only).
# Setting symmetric_target = TRUE tells M-SSA to mirror each row at its
# center point, producing the full symmetric two-sided target of length 2*L - 1.
symmetric_target <- TRUE

# Illustrate the mirroring procedure explicitly:
#   - The one-sided right tail hp_one_sided[1:L] is reflected to form
#     the full symmetric filter: [hp_one_sided[L:2], hp_one_sided[1:L]].
#   - The resulting filter has length 2*L - 1 and is symmetric around lag 0.
#   - Important: this mirroring is only exact when L is an odd number
#     (see the earlier note in Section 1.2).
ts.plot(c(hp_one_sided[L:2], hp_one_sided[1:L]),
        main = paste("Reconstructed two-sided filter of length ",
                     2 * L - 1, sep = ""),
        ylab = "")

# Note: an equivalent target specification — passing the full two-sided filter
#       directly rather than relying on mirroring — is demonstrated further below.


# ------------------------------------------------------------
# 1.5 Forecast Horizon and M-SSA Optimization Settings
# ------------------------------------------------------------

# ── Forecast Horizon (delta) ─────────────────────────────────────────────────
#
# The parameter delta controls the prediction horizon:
#   delta = 0  : nowcast (target aligned with the current period)
#   delta > 0  : h-step-ahead forecast (target shifted forward by delta periods)
#   delta < 0  : backcast (target shifted backward; rarely the primary use case)
#
# Caution for backcasts (delta < 0):
#   - The holding-time (HT) constraint may be too restrictive relative to
#     the backward shift, causing M-SSA to un-smooth the data.
#   - The resulting filters may appear irregular or counterintuitive.

# Select the desired prediction horizon:
delta <- 0   # Nowcast
delta <- 4   # One-year-ahead forecast (4 quarters for quarterly data)
# Uncomment the relevant line; both are shown here for illustration.


# ── Holding-Time (HT) Constraints ───────────────────────────────────────────
#
# The holding time (HT) is the expected mean duration between consecutive
# zero-crossings of the (mean-centered) predictor output.
#
# Key properties:
#   - A larger HT produces fewer zero-crossings, yielding a smoother predictor
#     that generates fewer spurious signals (false alarms).
#   - However, increasing the HT comes at the cost of greater phase delay
#     (lag/right-shift), reflecting the Accuracy-Smoothness-Timeliness (AST)
#     trilemma discussed in previous tutorials.
#
# One HT value must be specified per series (n values in total).
# The values below are derived from the German macroeconomic application
# (Tutorials 7.2 and 7.3) and correspond to mean durations of roughly
# 1.5 to 2 years between consecutive zero-crossings of the predictor.
# Increasing these values would produce a smoother (less reactive) predictor.
ht_mssa_vec <- c(6.380160, 6.738270, 7.232453, 7.225927, 7.033768)
names(ht_mssa_vec) <- colnames(x_mat)

# Convert each HT value to the corresponding lag-one ACF (rho),
# which is the form in which the smoothness constraint enters the M-SSA criterion.
# See previous tutorials for the mathematical relationship between HT and rho.
rho0 <- compute_rho_from_ht(ht_mssa_vec)$rho


# ── Numerical Optimization Settings ─────────────────────────────────────────
#
# The following parameters control the behavior of the M-SSA optimizer.

# Allow negative lambda values (un-smoothing):
#   - If TRUE, the optimizer may produce filters with more zero-crossings
#     than the benchmark (i.e., actively rougher predictors).
#   - Default is FALSE: smoothing only (non-negative lambda).
with_negative_lambda <- FALSE

# Lower bound for the nu parameter in the optimization:
#   - "rhomax" is the recommended default; it anchors the search
#     to the feasible region of the smoothness constraint.
lower_limit_nu <- "rhomax"

# Grid resolution for the bisection-based optimization:
#   - The algorithm uses a halving (triangulation) strategy,
#     so effective resolution is 2^split_grid.
#   - This is substantially faster than brute-force grid search.
#   - split_grid = 20 provides a good balance between speed and precision
#     for most applications.
#   - Convergence quality can be verified further below.
#   - Increase this parameter in the absence of convergence
split_grid <- 20


# ── Run M-SSA Optimization ───────────────────────────────────────────────────
MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target,
                      rho0, with_negative_lambda, xi,
                      lower_limit_nu, Sigma, symmetric_target)


# ── Inspect and Plot the M-SSA Filter Coefficients ───────────────────────────
#
# M-SSA returns a rich output object containing optimal filters and
# performance metrics. These are examined in detail in Exercise 2.
#
# Here we extract and visualize the M-SSA filter matrix:
#   - bk_x_mat has dimensions (n*L) x n.
#   - Column i contains the stacked filter coefficients used to predict target i.
#   - Each column is divided into n blocks of length L:
#     block j contains the coefficients applied to input series j.
bk_x_mat <- MSSA_obj$bk_x_mat
colnames(bk_x_mat) <- colnames(x_mat)

# Plot the filter coefficients for each target series:
#   - Each panel shows the n sub-filters (one per input series) for target i.
#   - Colors distinguish the contribution of each input series.
par(mfrow = c(1, n))
for (i in 1:n) {
  # Assemble the sub-filter matrix for target i:
  # each column contains the L coefficients for one input series
  mplot <- bk_x_mat[1:L, i]
  for (j in 2:n) {
    mplot <- cbind(mplot, bk_x_mat[(j - 1) * L + 1:L, i])
  }
  colnames(mplot) <- colnames(x_mat)
  
  colo <- rainbow(n)
  ts.plot(mplot,
          main = paste("M-SSA filter for target: ", colnames(x_mat)[i], sep = ""),
          col = colo)
  for (i in 1:n)
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

# Interpretation of the filter plots:
#   - All five M-SSA filters assign notable weight to the spread series,
#     consistent with spread acting as a leading indicator in the VAR(1).
#   - The spread filter (rightmost panel) depends only on its own lags,
#     as it is not driven by the other series within the VAR(1).
#   - The longitudinal and cross-sectional weight distributions
#     directly reflect the dependency structure encoded in the VAR(1).
#   - Larger HT constraints (smoother predictors) manifest as a
#     slower decay of filter coefficients across lags.



# ------------------------------------------------------------
# 1.6 Applying M-SSA to Simulated Data and Verifying Performance
# ------------------------------------------------------------
#
# This section applies the M-SSA filter derived in 1.5 to the simulated
# data and evaluates three key performance metrics:
#   (1) Mean-square forecast error (MSE)
#   (2) Target correlation (the M-SSA optimization criterion)
#   (3) Smoothness: empirical vs. imposed holding time (HT)
#
# The long simulated series (len = 100,000) allows finite-sample estimates
# to converge closely to their theoretical counterparts, thereby validating
# the M-SSA formulas.

# ── Select the Target Series ─────────────────────────────────────────────────
# Choose the m-th series as the prediction target (m = 1, ..., n).
m <- m_check <- 3
if (m > n)
  print(paste("Warning: m must be <= n =", n, sep = ""))

# ── Extract and Plot the M-SSA Filter for Target m ───────────────────────────
# Assemble the n sub-filters (one per input series) used to predict target m.
# Each sub-filter has length L; bk is an (L x n) matrix.
bk <- NULL
for (j in 1:n)
  bk <- cbind(bk, bk_x_mat[((j - 1) * L + 1):(j * L), m])
colnames(bk) <- colnames(x_mat)

# Visualize the sub-filter coefficients for target m:
#   - Each colored line corresponds to one input series.
#   - The relative magnitudes reflect each series' contribution
#     to predicting target m, as encoded in the VAR(1).
par(mfrow = c(1, 1))
colo <- rainbow(ncol(bk))
ts.plot(bk,
        main = paste("M-SSA filter coefficients for target: ",
                     colnames(x_mat)[m], sep = ""),
        col = colo)
for (i in 1:ncol(bk))
  mtext(colnames(bk)[i], col = colo[i], line = -i)

# ── Compute the M-SSA Predictor for Target m ─────────────────────────────────
# At each time point j, apply each sub-filter to the corresponding input
# series and sum across all n series to obtain the scalar predictor y[j].
y <- rep(NA, len)
for (j in L:len)
  y[j] <- sum(apply(bk * (x_mat[j:(j - L + 1), ]), 2, sum))

# ── Construct the Acausal (Two-Sided) Target Signal ───────────────────────────
# Apply the two-sided HP filter to the m-th series to obtain the target z.
# Only the right tail (hp_one_sided) is specified; the left tail is obtained
# by mirroring, yielding a symmetric filter of length 2*L - 1.
gammak <- hp_one_sided[1:L]

par(mfrow = c(1, 1))
ts.plot(gammak, main = "Right tail of the two-sided target filter")

# Mirror the target filter at the center and apply to the  m-th series (target)
z <- rep(NA, len)
for (j in L:(len - L))
  z[j] <- gammak %*% x_mat[j:(j - L + 1), m] +
  gammak[-1] %*% x_mat[(j + 1):(j + L - 1), m]
# z is the m-th target: it is obtained by applying the two-sided HP to the m-th series

# ── Shift the Target by the Forecast Horizon delta ───────────────────────────
# For delta > 0 (forecast): shift z forward so that the target leads the data.
# For delta < 0 (backcast): shift z backward.
# For delta = 0 (nowcast):  no shift applied.
if (delta > 0) {
  zdelta <- c(z[(delta + 1):len], rep(NA, delta))
} else if (delta < 0) {
  zdelta <- c(rep(NA, abs(delta)), z[1:(len - abs(delta))])
} else {
  zdelta <- z
}
names(zdelta) <- names(y) <- rownames(x_mat)

# ── Visual Inspection: Target vs. M-SSA Predictor ────────────────────────────
# Plot the last 200 observations (equivalent to ~50 years of quarterly data).
# Zero-crossings of the M-SSA predictor are marked by vertical dashed lines.
mplot <- cbind(zdelta, y)[(len - 200):len, ]
ts.plot(mplot,
        col = c("black", "blue"),
        main = "Target (black) and M-SSA predictor (blue): zero-crossings marked by vertical lines")
abline(h = 0)
abline(v = 1 + which(sign(mplot[2:nrow(mplot), 2] *
                            mplot[1:(nrow(mplot) - 1), 2]) < 0),
       lty = 3, col = "blue")

# Observations from the plot:
#   - Approximately four artificial "recessions" (local minima of the target)
#     appear over ~150 observations, implying roughly one recession per
#     150 / (4 * 4) ≈ 10 years — consistent with the HP(160) parameterization.
#   - True Recession durations are shorter than expansions, a stylized fact that
#     the symmetric VAR(1) cannot capture; a Hamilton regime-switching model
#     would be required for asymmetric cycle modeling.
#   - The two-sided target filter cannot be evaluated near the sample boundaries
#     (values remain NA at both ends).
#   - The vertical dashed lines mark zero-crossings of the M-SSA predictor,
#     whose mean spacing is controlled by the HT constraint. Increasing HT (rho0)
#     in the M-SSA call will lead to fewer crossings.


# ── Performance Metric 1: Empirical Holding Time ─────────────────────────────
# Compute the sample mean duration between consecutive zero-crossings of y.
# For large samples, this converges to the theoretically imposed HT.
compute_empirical_ht_func(mplot[, 2])
# Compare with the imposed HT constraint for series m:
ht_mssa_vec[m]
# Pretty close given a relatively short sample length

# ── Performance Metric 2: Mean-Square Forecast Error (MSE) ───────────────────
#
# Technical notes:
#   - M-SSA maximizes the target correlation between zdelta and y,
#     which is equivalent to minimizing MSE up to affine transformations
#     (static level and scale adjustments).
#   - M-SSA computes the optimal scale adjustment internally (assuming a
#     correctly specified model), so the blue line in the plot above is
#     already optimally scaled.
#   - Level adjustment is not applied by M-SSA, as the data are assumed
#     zero-centered (consistent with the zero-crossing framework).
#   - If an explicit level adjustment is needed, it can be obtained by
#     regressing y on zdelta.
#   - Design philosophy: M-SSA prioritizes dynamic forecasting properties
#     (zero-crossings, phase lead, growth direction) over static calibration.
mean((zdelta - y)^2, na.rm = TRUE)


# ── Performance Metric 3: Target Correlation ─────────────────────────────────
#
# The target correlation is the M-SSA optimization criterion:
#   - M-SSA maximizes this quantity subject to the HT (smoothness) constraint.
#   - Maximizing target correlation is equivalent to minimizing MSE between
#     the target and the M-SSA output, up to an affine transformation.
#   - For large samples, the empirical correlation converges to the
#     theoretical criterion value stored in MSSA_obj$crit_rhoy_target.
#
# Empirical target correlation (element [1,2] of the correlation matrix):
cor(na.exclude(cbind(zdelta, y)))[1, 2]

# Theoretical (criterion) target correlation for series m (see Wildi 2026b):
#   - MSSA_obj$crit_rhoy_target is a length-n vector, one value per series.
#   - The empirical value above should approach this for large len.
MSSA_obj$crit_rhoy_target[m]


# ── Performance Metric 4: Smoothness — Empirical vs. Imposed HT ──────────────
# Re-compute the empirical HT for the full predictor series y,
# and compare with the corresponding imposed constraint.
# For increasing sample size len, the empirical HT converges to the imposed HT.
compute_empirical_ht_func(y)
ht_mssa_vec[m]


# ============================================================
# Exercise 1 Summary
# ============================================================
#
# The simulation experiment above confirms the following:
#
#   1. Target Correlation:
#      M-SSA maximizes the correlation between the target signal and the
#      predictor, subject to the imposed HT (smoothness) constraint.
#      The empirical correlation converges to the theoretical criterion
#      value for large samples.
#
#   2. Smoothness (Holding Time):
#      The empirical mean duration between zero-crossings of the predictor
#      converges to the imposed HT constraint for large samples.
#
#   3. Relevance for Empirical Forecasting:
#      Both target correlation and holding time are meaningful in practice:
#        - Target correlation governs forecast accuracy.
#        - Holding time governs signal stability and alarm reliability.
#
#   4. Timeliness:
#      Phase advancement (left-shift) can be achieved by increasing the
#      forecast horizon delta (see Tutorials 2.1 and 3 on the AST trilemma).
# ============================================================



# =============================================================================
# Exercise 2: Advanced M-SSA Analysis
# =============================================================================
# This script extends the basic M-SSA workflow by:
#   - Computing theoretical (expected) performance measures
#   - Verifying the Holding Time (HT) constraint
#   - Examining additional M-SSA outputs in detail
#
# Theoretical background: Wildi (2026b), which derives closed-form expressions
# for the expected performance criteria of M-SSA.
# =============================================================================


# -----------------------------------------------------------------------------
# Section 2.1: Validation of Theoretical Performance Measures
# -----------------------------------------------------------------------------
# Key conceptual points:
#
# Target correlation:
#     - M-SSA maximizes correlation with the *causal* MSE benchmark,
#       not directly with the acausal (two-sided) target.
#     - Both formulations yield the same M-SSA solution (see Wildi 2026b)
#       but different criterion values.
#
# Two equivalent interpretations of the M-SSA objective:
#     - Causal framing:  Replicate the MSE predictor's tracking of the
#                        acausal target, subject to an HT smoothness constraint.
#                        -> This is a smoothing problem.
#     - Acausal framing: Match the two-sided HP filter subject to an HT
#                        constraint.
#                        -> This is a prediction problem.
#     Despite the conceptual difference, both lead to the same solution.
#
# When the imposed HT matches that of the MSE benchmark, M-SSA exactly
#     reproduces the classical MSE estimate. In practice, we often prefer
#     smoother solutions (larger HT), which reduces correlation with MSE but
#     increases noise suppression.
#
# Correlation hierarchy:
#     corr(M-SSA, MSE) > corr(M-SSA, acausal target)
#     because the acausal target incorporates future observations.
# -----------------------------------------------------------------------------

# Correlation between the M-SSA output and the classical M-MSE (Wiener-Kolmogorov) predictor.
# - High values indicate that M-SSA closely approximates M-MSE.
# - This correlation decreases monotonically as the imposed HT (rho0) increases,
#   since stronger smoothing moves the M-SSA solution further from the M-MSE benchmark.
MSSA_obj$crit_rhoyz

# Correlation between M-SSA output and the effective acausal target (e.g. two-sided HP filter).
# Lower than crit_rhoyz because the acausal target exploits future observations.
MSSA_obj$crit_rhoy_target


# -----------------------------------------------------------------------------
# Section 2.2: Verification of the Holding Time (HT) Constraint
# -----------------------------------------------------------------------------
# The HT constraint is equivalent to specifying a lag-one autocorrelation (ACF).
# Checking that the solution's lag-one ACF matches the imposed rho0 serves as a
# diagnostic for successful numerical optimization.
#
# Note: If discrepancies are large, increase the grid resolution (split_grid).
# The solution is unique due to strict monotonicity of the objective (see paper).
# -----------------------------------------------------------------------------

# Lag-one ACF of the M-SSA solution (empirical, from the optimized filter).
MSSA_obj$crit_rhoyy

# Lag-one ACF corresponding to the imposed HT constraint (target value).
rho0

# Both values should agree closely; large deviations signal numerical convergence issues.

# Equivalent diagnostic using HT directly (ACF and HT are in bijection):
compute_holding_time_from_rho_func(MSSA_obj$crit_rhoyy)  # HT of M-SSA solution
compute_holding_time_from_rho_func(rho0)                  # HT of imposed constraint

# Optimal nu parameter from the discrete difference equation.
# Values greater than 2 indicate stronger smoothing relative to the MSE benchmark
# (i.e., fewer zero-crossings in the M-SSA output).
MSSA_obj$nu_opt


# -----------------------------------------------------------------------------
# Section 2.3: M-SSA Filter Coefficients
# -----------------------------------------------------------------------------
# M-SSA provides two representations of the optimal filter:
#
#   bk_mat   : Filter applied to MA-inverted (whitened) VAR residuals.
#              This is the direct solution to the M-SSA optimization problem.
#              Useful for interpretation (e.g., impulse response analysis).
#
#   bk_x_mat : Filter applied to the observed data.
#              Obtained from bk_mat via deconvolution with the MA representation.
#              This is what is applied in practice.
#
# Note: bk_mat typically decays more slowly than bk_x_mat because the
# MA-inverted residuals are noisier than the original VAR(1) observations.
# The approximation improves as filter length L increases.
# -----------------------------------------------------------------------------

# --- Filter applied to MA-inverted residuals (bk_mat) ---
bk_mat <- MSSA_obj$bk_mat
par(mfrow = c(1, n))
for (i in 1:n) {
  # Extract filter coefficients for output series i across all n input series
  mplot <- bk_mat[1:L, i]
  for (j in 2:n) {
    mplot <- cbind(mplot, bk_mat[(j - 1) * L + 1:L, i])
  }
  ts.plot(mplot,
          main = paste("M-SSA filter: applied to epsilon_t", colnames(x_mat)[i]),
          col  = rainbow(n))
}

# --- Filter applied to observed data (bk_x_mat) ---
bk_x_mat <- MSSA_obj$bk_x_mat
par(mfrow = c(1, n))
for (i in 1:n) {
  # Extract filter coefficients for output series i across all n input series
  mplot <- bk_x_mat[1:L, i]
  for (j in 2:n) {
    mplot <- cbind(mplot, bk_x_mat[(j - 1) * L + 1:L, i])
  }
  ts.plot(mplot,
          main = paste("M-SSA filter applied to x_t", colnames(x_mat)[i]),
          col  = rainbow(n))
}

# --- Verify deconvolution relationship: bk_mat * xi = bk_x_mat ---
# Deconvolving bk_mat with the MA representation xi should exactly recover bk_x_mat.
deconv_M <- t(M_deconvolute_func(t(bk_mat), xi)$deconv)

# Maximum absolute deviation (should be negligible, i.e. near machine epsilon).
max(abs(deconv_M - bk_x_mat))


# -----------------------------------------------------------------------------
# Section 2.4: Classical M-MSE (Wiener-Kolmogorov) Filter Coefficients
# -----------------------------------------------------------------------------
# M-SSA also returns the classical M-MSE filter for reference and comparison.
# If the imposed HT in M-SSA equals the HT of the M-MSE filter, M-SSA exactly
# reproduces the M-MSE solution. Larger imposed HT yields smoother M-SSA filters
# that typically decay more slowly than their M-MSE counterparts.
# -----------------------------------------------------------------------------

# --- M-MSE filter applied to MA-inverted residuals (gammak_mse) ---
gammak_mse <- MSSA_obj$gammak_mse
par(mfrow = c(1, n))
for (i in 1:n) {
  mplot <- gammak_mse[1:L, i]
  for (j in 2:n) {
    mplot <- cbind(mplot, gammak_mse[(j - 1) * L + 1:L, i])
  }
  ts.plot(mplot,
          main = paste("M-MSE filter applied to epsilon_t", colnames(x_mat)[i]),
          col  = rainbow(n))
}

# --- M-MSE filter applied to observed data (gammak_x_mse) ---
gammak_x_mse <- MSSA_obj$gammak_x_mse
par(mfrow = c(1, n))
for (i in 1:n) {
  mplot <- gammak_x_mse[1:L, i]
  for (j in 2:n) {
    mplot <- cbind(mplot, gammak_x_mse[(j - 1) * L + 1:L, i])
  }
  ts.plot(mplot,
          main = paste("M-MSE filter applied to x_t", colnames(x_mat)[i]),
          col  = rainbow(n))
}


# -----------------------------------------------------------------------------
# Section 2.5: Acausal (Two-Sided) Target Filter
# -----------------------------------------------------------------------------
# The acausal target (e.g. two-sided HP filter) is shown for reference.
# The right tail is mirrored onto the left to form a symmetric two-sided filter.
# In the univariate case each series is filtered only by itself (diagonal structure).
# -----------------------------------------------------------------------------

# --- Acausal target in the original data domain ---
#  Note: we show the right tail only: this is mirrored to obtain the acausal target
par(mfrow = c(1, 1))
ts.plot(t(gamma_target),
        col  = rainbow(n),
        main = c("Acausal target filter (data domain)",
                 "Right tail mirrored to form two-sided filter",
                 "Each series weighted by itself only (diagonal structure)"))
abline(v = (1:n * (nrow(t(gamma_target)) / n)))

# --- Acausal target transformed to the residual domain (gamma_target_long) ---
# Obtained by convolving gamma_target with the MA representation xi.
# After convolution:
#   - The original filter symmetry may be distorted.
#   - Off-diagonal (cross-series) weights appear when the VAR is non-diagonal.
gamma_target_long <- MSSA_obj$gammak_target
par(mfrow = c(1, 1))
ts.plot(gamma_target_long,
        col  = rainbow(n),
        main = c("Acausal target filter (residual domain, after convolution with MA)",
                 "Symmetry may be distorted; cross-series weights possible"))
abline(v = (1:n * (nrow(gamma_target_long) / n)))


# -----------------------------------------------------------------------------
# Section 2.6: Variance of the Acausal Target
# -----------------------------------------------------------------------------
# var_target: theoretical variance of the two-sided HP output under correct
# model specification.
#
# Expected relationship:
#   diag(var_target) >= diag(var_MSE)
# because the acausal filter uses future innovations, producing a higher-variance
# signal, while M-MSE minimizes variance by construction.
# -----------------------------------------------------------------------------

var_target <- MSSA_obj$var_target
# The variances lie on the diagonal of the variance-covariance matrix
diag(var_target)

# -----------------------------------------------------------------------------
# Section 2.7: System Matrices for M-SSA Optimization
# -----------------------------------------------------------------------------
# Construct the system matrices used in the M-SSA quadratic program.
# These are built from Kronecker products of the VAR residual covariance matrix
# Sigma with identity and lag-one structure matrices (see Wildi 2026b, Section 2).
#
#   M_tilde : Weighted inner-product matrix for the smoothness penalty.
#   I_tilde : Weighted identity matrix for the correlation objective.
# -----------------------------------------------------------------------------

M_obj   <- M_func(L, Sigma)
M_tilde <- M_obj$M_tilde  # Key Input to Lag-1 ACF: Smoothness / HT constraint
I_tilde <- M_obj$I_tilde  # Key Input to Variance computation


# -----------------------------------------------------------------------------
# Section 2.8: Variance Decomposition
# -----------------------------------------------------------------------------
# Expected variance ordering (due to zero-shrinkage toward the mean):
#   var(M-SSA) <= var(M-MSE) <= var(acausal target)
# This follows because acausal target uses more information
# (future observations), and stronger smoothing in M-SSA shrinks variance further.
# -----------------------------------------------------------------------------

# Theoretical variance of the acausal target (computed internally by M-SSA).
diag(var_target)

# Theoretical variance of the classical M-MSE predictor (Wiener-Kolmogorov).
# Formula derived from the M-SSA paper: diag(Gamma_mse' * I_tilde * Gamma_mse).
diag(t(gammak_mse) %*% I_tilde %*% gammak_mse)

# Theoretical variance of the M-SSA output.
# In general: var(M-SSA) <= var(M-MSE) <= var(target).
diag(t(bk_mat) %*% I_tilde %*% bk_mat)


# -----------------------------------------------------------------------------
# Section 2.9: Verification Against Sample Estimates
# -----------------------------------------------------------------------------
# For sufficiently long samples, sample variances converge to their theoretical
# counterparts. The following pairs should match asymptotically.
# -----------------------------------------------------------------------------

# Theoretical vs. sample variance of M-SSA output for series m.
diag(t(bk_mat) %*% I_tilde %*% bk_mat)[m]  # Theoretical
var(na.exclude(y))                            # Sample estimate

# Theoretical vs. sample variance of the acausal target for series m.
diag(var_target)[m]                           # Theoretical
var(na.exclude(zdelta))                       # Sample estimate


# -----------------------------------------------------------------------------
# Section 2.10: Lag-One ACF and Holding Time (HT) — Theoretical Computation
# -----------------------------------------------------------------------------
# The lag-one ACF of any linear filter output can be expressed as a ratio of
# quadratic forms involving M_tilde (lag-one cross-product matrix) and I_tilde
# (variance matrix). See the M-SSA paper for derivations.
#
# ACF and HT are in bijective correspondence for Gaussian processes, so each
# uniquely determines the other.
# -----------------------------------------------------------------------------

# Theoretical lag-one ACF of the classical M-MSE predictor for each series.
rho_mse <- gammak_mse[, 1] %*% M_tilde %*% gammak_mse[, 1] /
  gammak_mse[, 1] %*% I_tilde %*% gammak_mse[, 1]
for (i in 2:n)
  rho_mse <- c(rho_mse,
               gammak_mse[, i] %*% M_tilde %*% gammak_mse[, i] /
                 gammak_mse[, i] %*% I_tilde %*% gammak_mse[, i])
rho_mse
# Theoretical lag-one ACF of the M-SSA output for each series.
rho_ssa <- bk_mat[, 1] %*% M_tilde %*% bk_mat[, 1] /
  bk_mat[, 1] %*% I_tilde %*% bk_mat[, 1]
for (i in 2:n)
  rho_ssa <- c(rho_ssa,
               bk_mat[, i] %*% M_tilde %*% bk_mat[, i] /
                 bk_mat[, i] %*% I_tilde %*% bk_mat[, i])
# Under successful optimization, rhos_ssa matches rho0
# Typically rho_ssa>rho_mse (smoothing)
rho0
rho_ssa

# Theoretical HT of the M-SSA output derived from rho_ssa.
# Sample HTs computed in the simulation experiment should match these values.
ht_comp <- apply(matrix(rho_ssa, nrow = 1), 1,
                 compute_holding_time_from_rho_func)[[1]]$ht
ht_comp




# -----------------------------------------------------------------------------
# Section 2.11: Target Correlation — M-SSA vs. M-MSE Benchmark
# -----------------------------------------------------------------------------
# M-SSA maximizes the correlation between its output and the M-MSE benchmark,
# subject to the imposed HT constraint. This is equivalent to targeting the
# acausal filter directly (both yield the same M-SSA solution; see paper).
#
# Key relationships:
#   - If HT(M-SSA) == HT(M-MSE): M-SSA replicates M-MSE exactly -> corr = 1.
#   - If HT(M-SSA) >  HT(M-MSE): M-SSA is smoother -> corr < 1.
#   - M-SSA maximizes this correlation subject to the HT constraint.
# -----------------------------------------------------------------------------

# Self-correlation of the M-MSE benchmark (trivially equals 1 for all series).
crit_mse <- gammak_mse[, 1] %*% I_tilde %*% gammak_mse[, 1] /
  gammak_mse[, 1] %*% I_tilde %*% gammak_mse[, 1]
for (i in 2:n)
  crit_mse <- c(crit_mse,
                gammak_mse[, i] %*% I_tilde %*% gammak_mse[, i] /
                  gammak_mse[, i] %*% I_tilde %*% gammak_mse[, i])
crit_mse  # Should be 1 for all series (serves as a sanity check)

# Correlation of M-SSA with the M-MSE benchmark (the maximized objective).
crit_ssa <- gammak_mse[, 1] %*% I_tilde %*% bk_mat[, 1] /
  (sqrt(bk_mat[, 1] %*% I_tilde %*% bk_mat[, 1]) *
     sqrt(gammak_mse[, 1] %*% I_tilde %*% gammak_mse[, 1]))
for (i in 2:n)
  crit_ssa <- c(crit_ssa,
                gammak_mse[, i] %*% I_tilde %*% bk_mat[, i] /
                  (sqrt(bk_mat[, i] %*% I_tilde %*% bk_mat[, i]) *
                     sqrt(gammak_mse[, i] %*% I_tilde %*% gammak_mse[, i])))
crit_ssa
# Compile results into a labelled summary matrix.
criterion_mat <- rbind(crit_mse, crit_ssa)
colnames(criterion_mat) <- paste("Series", 1:n)
rownames(criterion_mat) <- c("M-MSE (self-correlation)", "M-SSA vs. M-MSE")
criterion_mat

# Cross-check: second row of criterion_mat should match MSSA_obj$crit_rhoyz,
# which is the correlation criterion computed internally by M-SSA and maximized
# by the optimization.
# M-SSA maximizes the correlation with the MSE predictor
MSSA_obj$crit_rhoyz


# =============================================================================
# Summary of Exercise 2
# =============================================================================
#
# Rich M-SSA output:
#     M-SSA returns additional filters (including M-MSE) and performance metrics
#     beyond the primary M-SSA and M-MSE filters.
#
# Theoretical vs. sample consistency:
#     Sample estimates of variances, ACFs, HTs, and correlations converge to
#     their theoretical counterparts as sample length increases (under correct
#     model specification).
#
# Optimization principle:
#     - M-SSA maximizes correlation with the M-MSE benchmark subject to an HT
#       constraint. Targeting M-MSE or the acausal filter are equivalent
#       formulations yielding the same solution.
#     - The target correlation is invariant to static level and scale shifts,
#       making it equivalent to minimum MSE up to such adjustments.
#     - Under Gaussianity, maximizing the target correlation is equivalent to
#       maximizing sign accuracy (the probability that M-SSA and the target
#       share the same sign), since the two are linked by a strictly monotone
#       transformation (arcsin law).
#
# M-SSA as a generalization of M-MSE:
#     M-SSA reproduces classical M-MSE signal extraction when the imposed HT
#     matches the HT of the M-MSE filter.
#
# Flexible target specification:
#     - Supports backcasting (delta < 0), nowcasting (delta = 0), and
#       forecasting (delta > 0).
#     - The two-sided HP filter used here can be replaced by Hamilton,
#       Baxter-King, Beveridge-Nelson, or identity (for h-step forecasting).
#
# Stationarity assumption:
#     The DGP is assumed stationary. In practice, growth rates (first
#     differences) are used, which are approximately stationary. M-SSA relies
#     on the Wold decomposition, straightforwardly obtained via MA inversion
#     for VARMA processes.
#
# Robustness (previewing Tutorials 7.2 and 7.3):
#     Applied to German macro data, M-SSA proves robust against:
#       - Pandemic outliers (singular observations).
#       - In-sample span: pre-crisis M-SSA (data up to Jan-2007) performs
#         nearly as well as the full-sample version.
#       - VARMA misspecification (provided heavy overfitting is avoided).
#     The VAR(1) is likely misspecified for the macro application; Tutorial 7.2
#     introduces a simple and effective correction for this context.
#
# Next steps (Exercise 3):
#     Sample estimates of target correlations will be shown to converge to
#     the theoretical values derived here, for both the acausal and the
#     causal (M-MSE) target formulations.
# =============================================================================





# =============================================================================
# Exercise 3: Wrapping M-SSA into Functions and Verifying Convergence
# =============================================================================
# This exercise consolidates the code from Exercises 1 and 2 into modular
# functions, each responsible for a distinct task in the M-SSA workflow.
# We then verify that sample performance estimates converge to their
# theoretical counterparts as sample length increases.
#
# Function overview:
#   1. HP_target_sym_T   : Constructs the two-sided HP target filter.
#   2. MA_inv_VAR_func   : Computes the MA inversion of the VAR model.
#   3. MSSA_main_func    : Runs the M-SSA optimization.
#   4. filter_func       : Applies M-SSA, M-MSE, and target filters to data.

# These functions will be used extensively in tutorials 7.2-7.5 on GDP forecasting
# =============================================================================


# -----------------------------------------------------------------------------
# Function 1: HP_target_sym_T
# -----------------------------------------------------------------------------
# Constructs the one-sided (causal) HP target filter for each of the n series,
# and signals to M-SSA that each filter should be mirrored at its centre to
# form a symmetric (two-sided) target.
#
# Arguments:
#   n         : Number of series.
#   lambda_HP : HP smoothing parameter.
#   L         : Filter length (number of lags).
#
# Returns:
#   gamma_target      : (n x nL) matrix of one-sided target filter coefficients.
#                       Row i contains the filter for series i, with non-zero
#                       weights only in positions [(i-1)*L+1 : i*L].
#   symmetric_target  : Logical flag (TRUE) instructing M-SSA to mirror each
#                       filter to obtain the full two-sided target.
# -----------------------------------------------------------------------------
HP_target_sym_T <- function(n, lambda_HP, L)
{
  HP_obj <- HP_target_mse_modified_gap(L, lambda_HP)
  
  hp_symmetric        <- HP_obj$target    # Full two-sided HP filter
  hp_classic_concurrent <- HP_obj$hp_trend  # Concurrent (one-sided) HP filter
  hp_one_sided        <- HP_obj$hp_mse   # One-sided MSE-optimal HP filter
  
  # Build the block-diagonal target matrix: series i is filtered only by itself.
  # Row 1: [hp_one_sided | 0 ... 0]
  gamma_target <- c(hp_one_sided, rep(0, (n - 1) * L))
  for (i in 2:n)
    gamma_target <- rbind(gamma_target,
                          c(rep(0, (i - 1) * L),
                            hp_one_sided,
                            rep(0, (n - i) * L)))
  
  # Instruct M-SSA to mirror each one-sided filter at its centre, yielding
  # the full symmetric (two-sided) target.
  symmetric_target <- TRUE
  
  return(list(gamma_target     = gamma_target,
              symmetric_target = symmetric_target))
}


# -----------------------------------------------------------------------------
# Function 2: MA_inv_VAR_func
# -----------------------------------------------------------------------------
# Computes the MA representation (Wold decomposition) of a VAR model via
# MA inversion. M-SSA requires this because its optimization criterion is
# formulated in terms of white-noise (MA-inverted) residuals.
#
# Arguments:
#   Phi   : VAR coefficient matrix.
#   Theta : MA coefficient matrix (NULL for pure VAR).
#   L     : Filter / truncation length.
#   n     : Number of series.
#   Plot  : Logical; if TRUE, plots the first min(10, L) MA weights per series.
#
# Returns:
#   xi : (n x nL) matrix of MA weights, structured for M-SSA.
#        Columns [(i-1)*L+1 : i*L] contain the lag-0 to lag-(L-1) weights
#        of the i-th white-noise innovation on all n output series.
# -----------------------------------------------------------------------------
MA_inv_VAR_func <- function(Phi, Theta, L, n, Plot = FALSE)
{
  # Compute MA coefficient matrices up to lag L via PSI weights.
  xi_psi <- PSIwgt(Phi = Phi, Theta = NULL, lag = L, plot = FALSE, output = FALSE)
  xi_p   <- xi_psi$psi.weight
  
  # Reorganise xi_p into the (n x nL) structure expected by M-SSA:
  #   Columns [(i-1)*L+1 : i*L] hold the weights of the i-th innovation
  #   on all n series at lags 0, 1, ..., L-1.
  xi <- matrix(nrow = n, ncol = n * L)
  for (i in 1:n)
    for (j in 1:L)
      xi[, (i - 1) * L + j] <- xi_p[, i + (j - 1) * n]
  
  if (Plot)
  {
    # Plot the first min(10, L) MA weights for each output series.
    par(mfrow = c(1, n))
    for (i in 1:n)
    {
      mplot <- xi[i, 1:min(10, L)]
      for (j in 2:n)
        mplot <- cbind(mplot, xi[i, (j - 1) * L + 1:min(10, L)])
      ts.plot(mplot,
              col  = rainbow(ncol(mplot)),
              main = paste("MA inversion: series", colnames(x_mat)[i]))
    }
  }
  
  return(list(xi = xi))
}


# -----------------------------------------------------------------------------
# Function 3: MSSA_main_func
# -----------------------------------------------------------------------------
# Runs the M-SSA optimization for a given set of HT constraints and returns
# the optimized real-time filter together with the full M-SSA output object.
#
# Arguments:
#   delta            : Forecast horizon (negative = backcast, 0 = nowcast,
#                      positive = forecast).
#   ht_vec           : Vector of imposed Holding Times, one per output series.
#   xi               : (n x nL) MA-inversion matrix from MA_inv_VAR_func.
#   symmetric_target : Logical; TRUE if target filters are to be mirrored.
#   gamma_target     : (n x nL) target filter matrix.
#   Sigma            : (n x n) VAR residual covariance matrix.
#   Plot             : Logical; if TRUE, plots the optimized real-time filters.
#
# Returns:
#   bk_x_mat  : (nL x n) optimized M-SSA filter in the data domain.
#   MSSA_obj  : Full M-SSA output object (filters, diagnostics, criteria).
# -----------------------------------------------------------------------------
MSSA_main_func <- function(delta, ht_vec, xi, symmetric_target,
                           gamma_target, Sigma, Plot = FALSE)
{
  # Convert imposed HTs to equivalent lag-one ACF values (bijective mapping).
  rho0 <- compute_rho_from_ht(ht_vec)$rho
  
  # --- Numerical optimization settings ---
  
  # Allow search over unsmoothing (lambda < 0, more zero-crossings than M-MSE)?
  # FALSE = smoothing only (default and recommended).
  with_negative_lambda <- FALSE
  
  # Starting point for the nu search: "rhomax" initialises at the smoothest
  # feasible solution, which aids convergence.
  lower_limit_nu <- "rhomax"
  
  # Half-way triangulation grid: effective resolution is 2^split_grid.
  # Value of 20 gives fast convergence in most applications.
  split_grid <- 20
  
  # M-SSA expects gamma_target with rows = target series, columns = lags.
  gamma_target <- t(gamma_target)
  
  # --- Run M-SSA optimization ---
  MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size,
                        gamma_target, rho0,
                        with_negative_lambda, xi,
                        lower_limit_nu, Sigma,
                        symmetric_target)
  
  # Extract the real-time (causal) filter in the data domain.
  # The full MSSA_obj also contains M-MSE filters, ACF/HT diagnostics,
  # target correlations, and variance estimates (see Exercise 2).
  bk_x_mat <- MSSA_obj$bk_x_mat
  
  if (Plot)
  {
    par(mfrow = c(1, n))
    for (i in 1:n)
    {
      mplot <- bk_x_mat[1:L, i]
      for (j in 2:n)
        mplot <- cbind(mplot, bk_x_mat[(j - 1) * L + 1:L, i])
      ts.plot(mplot,
              main = paste("M-SSA filter (data domain): output",
                           colnames(x_mat)[i]),
              col  = rainbow(n))
    }
  }
  
  return(list(bk_x_mat = bk_x_mat,
              MSSA_obj  = MSSA_obj))
}


# -----------------------------------------------------------------------------
# Function 4: filter_func
# -----------------------------------------------------------------------------
# Applies three filters to the observed data matrix x_mat:
#   (a) M-SSA filter      -> mssa_mat
#   (b) M-MSE filter      -> mmse_mat
#   (c) Acausal target    -> target_mat
#
# The target is either one-sided (symmetric_target = FALSE) or two-sided
# (symmetric_target = TRUE, right half mirrored at the centre).
# In both cases the data is shifted by delta to align with the forecast horizon.
#
# Arguments:
#   x_mat            : (T x n) matrix of observed data.
#   bk_x_mat         : (nL x n) M-SSA filter coefficients (data domain).
#   gammak_x_mse     : (nL x n) M-MSE filter coefficients (data domain).
#   gamma_target     : (nL x n) target filter coefficients.
#   symmetric_target : Logical; TRUE = mirror target filter (two-sided).
#   delta            : Forecast horizon (shift applied to target computation).
#
# Returns:
#   mssa_mat   : (T x n) M-SSA filter output.
#   mmse_mat   : (T x n) M-MSE filter output.
#   target_mat : (T x n) acausal target output.
# -----------------------------------------------------------------------------
filter_func <- function(x_mat, bk_x_mat, gammak_x_mse,
                        gamma_target, symmetric_target, delta)
{
  len <- nrow(x_mat)
  n   <- dim(bk_x_mat)[2]
    
    # --- (a) M-SSA filter output ---
    mssa_mat <- NULL
  for (m in 1:n)
  {
    # Assemble (L x n) coefficient matrix for output series m.
    bk <- NULL
    for (j in 1:n)
      bk <- cbind(bk, bk_x_mat[((j - 1) * L + 1):(j * L), m])
    
    # Apply filter via inner product at each time point.
    y <- rep(NA, len)
    for (j in L:len)
      y[j] <- sum(apply(bk * (x_mat[j:(j - L + 1), ]), 2, sum))
    
    mssa_mat <- cbind(mssa_mat, y)
  }
  
  # --- (b) M-MSE (Wiener-Kolmogorov) filter output ---
  mmse_mat <- NULL
  for (m in 1:n)
  {
    # Assemble (L x n) coefficient matrix for output series m.
    gamma_mse <- NULL
    for (j in 1:n)
      gamma_mse <- cbind(gamma_mse, gammak_x_mse[((j - 1) * L + 1):(j * L), m])
    
    ymse <- rep(NA, len)
    for (j in L:len)
      ymse[j] <- sum(apply(gamma_mse * (x_mat[j:(j - L + 1), ]), 2, sum))
    
    mmse_mat <- cbind(mmse_mat, ymse)
  }
  
  # --- (c) Acausal target output ---
  target_mat <- NULL
  for (m in 1:n)
  {
    # Assemble (L x n) coefficient matrix for target series m.
    gammak <- NULL
    for (j in 1:n)
      gammak <- cbind(gammak, gamma_target[(j - 1) * L + 1:L, m])
    
    z <- rep(NA, len)
    
    if (symmetric_target)
    {
      # Two-sided filter: mirror the right half at the centre and shift by delta.
      # The causal half runs backward (j to j-L+1) and the anti-causal half
      # runs forward (j+1 to j+L-1), both shifted by delta.
      for (j in (L - delta):(len - L - delta + 1))
        z[j] <- sum(apply(gammak  * x_mat[delta + j:(j - L + 1),      ], 2, sum)) +
          sum(apply(gammak[-1, ] * x_mat[delta + (j + 1):(j + L - 1), ], 2, sum))
    } else
    {
      # One-sided filter: apply directly with delta shift (no mirroring).
      for (j in (L - delta):(len - delta))
        z[j] <- sum(apply(gammak * (x_mat[delta + j:(j - L + 1), ]), 2, sum))
    }
    
    names(z) <- names(y) <- rownames(x_mat)
    target_mat <- cbind(target_mat, z)
  }
  
  colnames(mssa_mat) <- colnames(mmse_mat) <- colnames(target_mat) <- colnames(x_mat)
  
  return(list(mssa_mat   = mssa_mat,
              target_mat = target_mat,
              mmse_mat   = mmse_mat))
}


# -----------------------------------------------------------------------------
# Apply the above functions to the simulation experiment from Exercises 1 & 2
# -----------------------------------------------------------------------------


#------------------------------------------------------------------------
# Apply the previously defined functions to the simulation experiment
# described above. Steps 1–4 cover: target specification, MA-inversion,
# M-SSA filter design, and filter application to data.

# ----- 1. Target Specification -----

# Regularization parameter for the Hodrick-Prescott (HP) filter
lambda_HP <- 160

# Filter length (approximately 4 years of monthly data).
# Must be an odd number to ensure symmetric mirroring around the center point
# (see earlier comments on the mirroring convention).
L <- 31

# Compute the HP target object: returns filter coefficients and symmetry flag
target_obj <- HP_target_sym_T(n, lambda_HP, L)

# Extract the transposed matrix of target filter coefficients (one row per series)
gamma_target <- t(target_obj$gamma_target)

# Boolean flag: TRUE if the target is symmetric (i.e., the right tail of the
# one-sided target can be mirrored to the left to form the two-sided filter)
symmetric_target <- target_obj$symmetric_target

# Plot the target filter coefficients as applied to the original (non-inverted) data.
# Each colored line corresponds to one of the n series.
# The right tail of each one-sided target will be mirrored about the center point
# to construct the full two-sided HP filter.
par(mfrow = c(1, 1))
ts.plot(
  gamma_target,
  col  = rainbow(n),
  main = "Target coefficients (original data): right tail will be mirrored to obtain the two-sided HP filter"
)
# Add vertical lines to delimit the coefficient segments for each series
abline(v = 1 + (1:n * (nrow(gamma_target) / n)))

# Confirm whether the symmetric mirroring applies for this target
symmetric_target


# ----- 2. MA-Inversion via VAR Model -----

# Compute the MA-representation (infinite-order moving average) of the
# multivariate process by inverting the fitted VAR/VMA model.
# Inputs: VAR coefficient matrix (Phi), MA coefficient matrix (Theta),
#         filter half-length (L), number of series (n), and sample length (T).
MA_inv_obj <- MA_inv_VAR_func(Phi, Theta, L, n, T)

# Extract the MA coefficient array (dimensions: n x (L*n))
xi <- MA_inv_obj$xi
dim(xi)


# ----- 3. M-SSA Filter Design -----

# Forecast horizon (delta):
#   delta = 0  -> nowcast (real-time level estimation)
#   delta = 4  -> one year ahead forecast for quarterly data
delta <- 0   # nowcast
delta <- 4   # one-year-ahead forecast (quarterly data)

# Compute the M-SSA filter coefficients and associated performance criteria.
# Inputs: forecast horizon (delta), target holding-time vector (ht_mssa_vec),
#         MA coefficients (xi), symmetry flag, target coefficients (gamma_target),
#         noise covariance matrix (Sigma), and sample length (T).
MSSA_main_obj <- MSSA_main_func(delta, ht_mssa_vec, xi, symmetric_target,
                                gamma_target, Sigma, T)

# Attach the benchmark (population) filter coefficient matrix for later reference
MSSA_main_obj$bk_x_mat <- bk_x_mat

# Extract the core M-SSA output object
MSSA_obj <- MSSA_main_obj$MSSA_obj

# Extract the benchmark M-MSE (minimum mean-square error) filter coefficients
gammak_x_mse <- MSSA_obj$gammak_x_mse


# ----- 4. Apply M-SSA Filter to Data -----

# Apply the designed M-SSA filter to the multivariate data matrix x_mat.
# Note: for long samples this step can be slow due to R's for-loop overhead.
filt_obj <- filter_func(x_mat, bk_x_mat, gammak_x_mse, gamma_target,
                        symmetric_target, delta)

# Extract filter output matrices:
mssa_mat   <- filt_obj$mssa_mat    # M-SSA filtered output (one column per series)
target_mat <- filt_obj$target_mat  # Acausal (two-sided) target output
mmse_mat   <- filt_obj$mmse_mat    # M-MSE benchmark filter output


#------------------------
# Verification Checks
# The results below should match the previously computed scalar outputs
# (y and zdelta for series m_check); differences should be effectively zero.

# Maximum absolute deviation between scalar y and the m_check column of mssa_mat
max(abs(y - mssa_mat[, m_check]), na.rm = T)

# Maximum absolute deviation between scalar zdelta and the m_check column of target_mat
max(abs(zdelta - target_mat[, m_check]), na.rm = T)


# ----- Mean-Square Errors -----

# Compute the sample squared error between the acausal target and M-SSA
# for each of the n series (NA values excluded).
apply(na.exclude((target_mat - mssa_mat)^2), 2, mean)


# ----- Sample vs. Expected Target Correlations -----

# Verify that sample correlations between the acausal target and M-SSA output
# converge toward their theoretical (expected) values as the sample length grows.
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[, i], mssa_mat[, i])))[1, 2])

# Corresponding theoretical (population) target correlation values for comparison:
# In applications, empirical estimates converge to expected values when the VAR model 
#   is correctly specified
MSSA_obj$crit_rhoy_target


# ----- Sample vs. Expected M-MSE / M-SSA Correlations -----

# Similarly, verify that sample correlations between the M-MSE benchmark and
# M-SSA output converge to the theoretical criterion values (the M-SSA objective).
for (i in 1:n)
  print(cor(na.exclude(cbind(mmse_mat[, i], mssa_mat[, i])))[1, 2])

# Corresponding theoretical (population) correlation values (M-SSA objective function):
MSSA_obj$crit_rhoyz


# ----- Holding-Time (HT) Verification -----

# M-SSA maximizes the target correlation subject to a holding-time (HT) constraint.
# Compare empirical (sample) HTs against the imposed (target) HT vector:

# Empirical HTs computed from the M-SSA output for each series
apply(mssa_mat, 2, compute_empirical_ht_func)

# Target HT values that were imposed as constraints during filter design
ht_mssa_vec

# Note: convergence of empirical HTs to true values is typically slow 
# Very long samples are required to observe tight fit

# ----- Source Utility Functions -----

# All functions used above are also available in the external utility script.
# This script will be relied upon in tutorials 7.2 and 7.3.
source(paste(getwd(), "/R/M_SSA_utility_functions.r", sep = ""))

















