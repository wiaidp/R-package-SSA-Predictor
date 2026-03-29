# This is currently under construction

# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 8: M-SSA Smoothing
# ══════════════════════════════════════════════════════════════════════════════
#
# OVERVIEW
# ────────
# M-SSA can target either acausal or causal filters, subject to a
# holding-time (HT) constraint:
#
#   • Acausal target → M-SSA acts as a PREDICTOR
#                      (target lies in the future relative to t)
#   • Causal target  → M-SSA acts as a SMOOTHER
#                      (target lies at or before t)
#
# Previous tutorials emphasised prediction; this tutorial focuses on smoothing.
#
# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING TARGET
# ────────────────
# The target is x_{t + delta}, the original (unfiltered) series at lag -delta,
# where -T ≤ delta ≤ 0.
#
# Contrast with related problems:
#   • Forecasting       : delta > 0  (target lies in the future)
#   • Signal extraction : target is a filtered version of x_t
#                         (e.g., the HP trend), not x_t itself
#
# ══════════════════════════════════════════════════════════════════════════════
# ROLE OF THE HT CONSTRAINT IN SMOOTHING
# ───────────────────────────────────────
# The holding-time (HT) constraint governs the smoothness of the M-SSA output:
#
#   • delta = 0  (nowcast)  : M-SSA produces a real-time estimate of x_t,
#                             subject to the specified HT constraint.
#   • HT > HT(x_t)          : M-SSA yields a smoother estimate of x_{t + delta};
#                             mean-crossings occur less frequently than in x_t.
#   • HT < HT(x_t)          : M-SSA yields a noisier estimate of x_{t + delta};
#                             generally not of practical interest.
#
# ══════════════════════════════════════════════════════════════════════════════
# NOVEL SMOOTHING CONCEPT
# ───────────────────────
# M-SSA smoothing controls the rate of mean-crossings (zero-crossings for a
# zero-mean series) directly via the HT constraint, providing an interpretable
# and operationally meaningful smoothness criterion.
#
# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ─────────
# M-SSA smoothing is benchmarked against classic Whittaker–Henderson (WH)
# graduation, which generalises the HP filter. The comparison focuses on
# smoothing performance rather than forecasting accuracy.
#
# ══════════════════════════════════════════════════════════════════════════════
# THEORETICAL REFERENCES
# ──────────────────────
#
#   Wildi, M. (2024).
#   "Business Cycle Analysis and Zero-Crossings of Time Series:
#    A Generalized Forecast Approach."
#   https://doi.org/10.1007/s41549-024-00097-5
#
# Wildi, M. (2026b). The Accuracy-Smoothness Dilemma in Prediction:
#   A Novel Multivariate M-SSA Forecast Approach.
#   Journal of Time Series Analysis, http://doi.org/10.1111/jtsa.70058 
#   arXiv: https://doi.org/10.48550/arXiv.2602.13722
#
# ══════════════════════════════════════════════════════════════════════════════
# This tutorial is based entirely on Wildi (2026b), Section 4.2. Additional 
#   applications are given in Wildi (2024)



# ══════════════════════════════════════════════════════════════════════════════
# INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

# Clear the workspace to ensure a clean environment
rm(list = ls())


# ══════════════════════════════════════════════════════════════════════════════
# LOAD REQUIRED LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# HP filter and other standard time-series filters
library(mFilter)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD CUSTOM M-SSA FUNCTION LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# Core M-SSA filter construction and optimisation routines
source(paste(getwd(), "/R/functions_MSSA.r", sep = ""))

# HP-filter utilities used in the JBCY paper (depends on mFilter)
source(paste(getwd(), "/R/HP_JBCY_functions.r", sep = ""))

# M-SSA utility functions: data preparation, plotting helpers, and wrappers
source(paste(getwd(), "/R/M_SSA_utility_functions.r", sep = ""))


# ══════════════════════════════════════════════════════════════════════════════
# GOAL
# ════
# Benchmark M-SSA smoothing against Whittaker–Henderson (WH) graduation,
# which generalises the HP filter (assuming second-order differences, d = 2).
# See Wildi (2026b), Section 4.2 for full theoretical background.
#
# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND  (Wildi 2026b, Section 4.2)
# ═══════════
# WH graduation / HP filter
# ─────────────────────────
#   • The HP filter is the solution to the WH problem when the penalty term
#     emphasises squared second-order differences (curvature).
#   • Consequence: for a given level of curvature, HP is the closest possible
#     smoother to the original series x_t (i.e., HP minimises MSE subject to
#     a curvature constraint, and equivalently minimises curvature subject to
#     a given MSE).
#
# M-SSA smoothing
# ───────────────
#   • M-SSA controls smoothness via the holding-time (HT) constraint rather
#     than curvature, offering an alternative smoothness criterion based on
#     the rate of mean-crossings.
#   • The rate of mean-crossings (or sign changes for a zero-mean process) is
#     a more natural smoothness measure in applications where the sign of the
#     output is the primary driver of decision-making — for example,
#     distinguishing true alarms from false alarms (see Exercise 6.3 in
#     Tutorial 7.4: Analysis of True vs. False Alarms through the ROC Curve).
#   • In such applications, controlling mean-crossings directly via the HT
#     constraint is more compelling than minimising curvature, as it aligns
#     the smoothness criterion with the operational objective.
#
# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON STRATEGY 
# ════════════════════════════════════
# Two complementary designs are evaluated against WH (or HP):
#
#   Exercise 1 (Strategy 1) — Match MSE tracking of x_t:
#     Design M-SSA so that its MSE tracking of x_t equals that of HP.
#     Then compare the HT and curvature of both smoothers.
#     Expected result: M-SSA achieves a longer HT (smoother output)
#     for the same tracking accuracy, but larger curvature.
#
#   Exercise 2 (Strategy 2) — Match holding time of HP:
#     Design M-SSA so that its HT equals that of HP.
#     Then compare the MSE tracking and curvature of both smoothers.
#     Expected result: M-SSA achieves better MSE tracking (closer to x_t)
#     for the same HT but larger curvature.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1: M-SSA vs. HP (Strategy 1)
# ══════════════════════════════════════════════════════════════════════════════

#------------------------------------------------------------------
# 1.1 Specify HP 
#------------------------------------------------------------------

# --- HP filter parameters ---
lambda_HP <- 14400    # Monthly HP smoothing parameter (standard business-cycle value)
L         <- 201      # Filter length (number of coefficients)

# Compute the bi-infinite HP target filter and its MSE-optimal approximation
HP_obj   <- HP_target_mse_modified_gap(L, lambda_HP)
hp_target <- HP_obj$target      # Bi-infinite HP filter coefficients (length L)
hp_trend  <- HP_obj$hp_trend    # Associated HP trend estimate

#------------------------------------------------------------------
# 1.2 Specify M-SSA
#------------------------------------------------------------------
# --- M-SSA design settings ---
Sigma            <- NULL   # NULL → assumes univariate design (Sigma = is the variance-covariance matrix of the innovation process)
xi               <- NULL   # NULL → white noise
symmetric_target <- FALSE  # Causal (one-sided) target; not symmetric

# --- Target specification ---
# gamma_target = 1 → allpass target (identity filter).
#   This definition contrasts with signal extraction/nowcasting where the target filter is not an identity
# MSSA_func will automatically zero-pad gamma_target to length L if needed.
gamma_target <- 1

# delta specifies the M-SSA smoother for time point t+delta
#   M-SSA matches x_{t+delta} best possible (max. target correlation or sign accuracy) 
#   subject to the ht constraint
# Backshift: delta = -(L-1)/2 
#   This will generate a symmetric causal smoothing filter centered at t-(L-1)/2
delta <- -(L - 1) / 2

# --- Bisection grid resolution ---
# The optimiser uses a halving (bisection) strategy rather than brute-force
# grid search, so effective resolution scales as 2^split_grid.
#   • split_grid = 20 balances speed and precision for most applications.
#   • Convergence quality can be verified in the diagnostics below.
#   • Increase split_grid if convergence is not achieved.
split_grid <- 20

#------------------------------------------------------------------
# 1.3 Specify the HT: Strategy 1
#------------------------------------------------------------------

# Strategy 1: impose the same HT as the two-sided HP filter.
#             Under identical HT, M-SSA will outperform HP in target correlation.
# Note: we consider the two-sided (symmetric) HP since M-SSA is also two-sided symmetric (since delta <- -(L - 1) / 2)


# Extract the HT and corresponding rho from the bi-infinite HP target filter
rho1 <- compute_holding_time_func(hp_target)$rho_ff1   # rho implied by HP's HT
ht1  <- compute_holding_time_func(hp_target)$ht        # Holding time of HP filter
# Interpretation of ht1: 
#   The two sided HP exhibits a mean duration of ~60 between consecutive sign changes when 
#   applied to white noise
# We apply the same HT to M-SSA and verify how closely M-SSA (and HP) track x_{t+delta}.
# Expected outcome: M-SSA outperforms HP (smaller MSE)
ht1

#------------------------------------------------------------------
# 1.4 Compute M-SSA
#------------------------------------------------------------------

# Design M-SSA filter with the HP-matched HT constraint
# NOTE: omitting xi assumes white-noise input (xi = NULL → xi = Identity);
#       MSSA_func will warn that gamma_target is zero-padded to length L.
SSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1)

# Extract the optimised filter coefficients
bk_mat <- SSA_obj$bk_mat

# --- Visual inspection of the filter ---
# M-SSA is symmetric because delta <- -(L - 1) / 2
par(mfrow = c(1, 1))
ts.plot(bk_mat)   # Plot filter coefficients

# --- Optimisation diagnostics ---
SSA_obj$crit_rhoy_target   # Achieved target correlation (should exceed HP)


# ══════════════════════════════════════════════════════════════════════════════
# 1.5 Outcome
# ══════════════════════════════════════════════════════════════════════════════

# --- Simulate long white-noise input ---
lenq   <- 100000
set.seed(86)
x <- rnorm(lenq)

# --- Apply both symmetric (two-sided) filters to x ---
y_ssa     <- filter(x, bk_mat,   sides = 2)   # M-SSA smoother output
y_hp_two  <- filter(x, hp_target, sides = 2)  # HP smoother output

# --- Compute MSE of each smoother relative to the original series ---
# MSE_scale re-scales the M-SSA output so its level is comparable to x
MSE_scale      <- as.double(bk_mat[(L + 1) / 2] / t(bk_mat) %*% bk_mat)
mse_ssa_smooth <- mean((x - MSE_scale * y_ssa)^2, na.rm = TRUE)
mse_hp_smooth  <- mean((x - y_hp_two)^2,          na.rm = TRUE)

# --- Correlation matrix: x, M-SSA output, HP output ---
filter_mat <- na.exclude(cbind(x, y_ssa, y_hp_two))
cor_mat<-cor(filter_mat)
cor_mat

#-----------------------------------------------------
# Interpretation
# 1. The sample target correlation of M-SSA and x ((1,2)-element) matches the maximized M-SSA objective
cor_mat[1,2]
SSA_obj$crit_rhoy_target   
# Since M-SSA maximizes this target correlation (equivalently: the sign accuracy), M-SSA must be optimal

# 2. HP-loss: ~10% 
# M-SSA
cor_mat[1,2]
# HP
cor_mat[1,3]
#------------------------------------------------------
# Main Take-Aways:
# 1. For given HT constraint, M-SSA maximizes target correlation or sign accuracy
# 2. Imposing the HT of HP to M-SSA then implies that M-SSA must outperform HP when tracking x_{t+delta}
# 3. However, M-SSA must have worse (larger) curvature
# Note: The above results can be straightforwardly extended from white noise data  to arbitrary dependent processes 
#----------------------------------------------------------

# ══════════════════════════════════════════════════════════════════════════════
# Exercise 2: M-SSA vs. WH Smoothing (Strategy 2)
# ══════════════════════════════════════════════════════════════════════════════

# 2.1 Compute M-SSA
# Strategy 2: find the HT such that M-SSA achieves the same target correlation
#             as HP. Under identical correlation, M-SSA will outperform HP in HT.

ht1_1  <- 75                                    # Target HT to match HP correlation
rho1_1 <- compute_rho_from_ht(ht1_1)$rho        # Corresponding rho value

# Design M-SSA filter with the correlation-matched HT constraint
SSA_obj_1 <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1_1)

# Extract filter coefficients and verify correlation
bk_mat_1 <- SSA_obj_1$bk_mat
ts.plot(bk_mat_1)
SSA_obj_1$crit_rhoy_target   # Should approximately equal cor(filter_mat)[1, 3]
# Check: M-SSA replicates (sample) target correlation of HP
cor(filter_mat)[1, 3]

# ══════════════════════════════════════════════════════════════════════════════
# 2.2 Compute Curvatures
# ══════════════════════════════════════════════════════════════════════════════
# All three filters are scaled to unit variance before comparison.
# HP minimises mean squared second differences by construction (WH criterion).

# Scale all filters to unit variance
scaled_filters <- scale(cbind(bk_mat, bk_mat_1, hp_target),
               center = FALSE, scale = TRUE) / sqrt(L - 1)

# Apply each scaled filter to x
yhat_mat <- NULL
for (i in 1:ncol(mplot))
  yhat_mat <- cbind(yhat_mat, filter(x, scaled_filters[, i], sides = 2))

# Assign descriptive column names
colnames(scaled_filters)<-colnames(yhat_mat) <- c(
  paste("SSA(", round(ht1,   2), ",", delta, ")", sep = ""),
  paste("SSA(", round(ht1_1, 2), ",", delta, ")", sep = ""),
  "HP"
)

# Compute mean squared second-order differences for each smoother
# HP is expected to achieve the smallest value by the WH optimality criterion
sq_se_dif <- apply(
  apply(apply(na.exclude(yhat_mat), 2, diff), 2, diff)^2,
  2, mean
)
sq_se_dif

# Outcome: as expected, HP minimizes the curvature among all three designs


# ══════════════════════════════════════════════════════════════════════════════
# 2.3 Plot and Sample Performances
# ══════════════════════════════════════════════════════════════════════════════

# --- Time-series plot of a 1 000-observation window ---
colo <- c("blue", "violet", "black")
ts.plot(na.exclude(yhat_mat)[1000:2000, ], col = colo)
abline(h = 0)
for (i in 1:ncol(yhat_mat))
  mtext(colnames(yhat_mat)[i],col=colo[i],line=-i)

# --- Autocorrelation functions ---
# M-SSA filters: slowly decaying ACF (long memory in the smooth output)
acf(na.exclude(yhat_mat)[, 1], lag.max = 100)
acf(na.exclude(yhat_mat)[, 2], lag.max = 100)

# HP filter: faster decay with cyclical structure; half-period ≈ 57 lags,
# consistent with its holding time
acf(na.exclude(yhat_mat)[, 3], lag.max = 100)

# Diagnostics: comparison of sample performances

# Empirical holding times: M-SSA should equal or exceed HP's HT
apply(na.exclude(yhat_mat), 2, compute_empirical_ht_func)

# Tracking ability (correlation with x): M-SSA should equal or exceed HP
cor(na.exclude(cbind(x, yhat_mat)))[1, ]


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 3:  COMPUTE THE FULL SSA SMOOTHER FAMILY (delta SWEEP)
# ══════════════════════════════════════════════════════════════════════════════
# Sweep delta from the fully causal end (-(L-1)/2) to nowcast (0),
# designing one M-SSA filter per lag shift. This may be time-consuming;
# set recompute_calculations = TRUE only when a fresh run is needed.

# Re-extract HP holding-time parameters for the sweep
rho1 <- compute_holding_time_func(hp_target)$rho_ff1
ht1  <- compute_holding_time_func(hp_target)$ht
  
gamma_target <- 1   # Allpass target
filt_mat     <- NULL
  
# Design one M-SSA filter for each lag delta
for (delta in (-(L - 1) / 2):0)
{
  print(delta)
  SSA_obj  <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1)
  filt_mat <- cbind(filt_mat, SSA_obj$bk_mat)
}
  


# ══════════════════════════════════════════════════════════════════════════════
# 3.2 PLOT THE FULL SSA SMOOTHER FAMILY
# ══════════════════════════════════════════════════════════════════════════════

# --- Scale all filters to unit variance for a fair visual comparison ---
filt_mat <- scale(filt_mat, center = FALSE, scale = TRUE) / sqrt(L - 1)

# Verify unit L2 norm after scaling (each column sum of squares ≈ 1)
apply(filt_mat^2, 2, sum)

# --- Plot all smoother coefficient profiles, coloured by delta ---
plot(
  filt_mat[, 1],
  col  = rainbow(ncol(filt_mat))[1],
  main = "SSA Smoothers",
  axes = FALSE, type = "l",
  ylab = "", xlab = expression(delta)
)
for (i in 2:ncol(filt_mat))
  lines(filt_mat[, i], col = rainbow(ncol(filt_mat))[i])

# Mark the filter centre (lag 0 position)
abline(v = (L - 1) / 2 + 1)

axis(1,
     at     = 1 + c(1, 50 * 1:(nrow(filt_mat) / 50)),
     labels = c(0,    50 * 1:(nrow(filt_mat) / 50)))
axis(2)
box()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: OVERLAY PLOT — SYMMETRIC SSA VS HP FILTER COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════════

coloh <- c("blue", "violet", "black")

plot(
  mplot[, 1],
  main  = "Symmetric SSA and HP Smoothers",
  axes  = FALSE, type = "l",
  xlab  = "", ylab = "",
  ylim  = c(min(na.exclude(mplot)), max(na.exclude(mplot))),
  col   = coloh[1], lwd = 2, lty = 1
)

# Annotate each filter in its own colour
mtext(paste("SSA(", round(ht1,   2), ",", delta, ")   ", sep = ""),
      col = coloh[1], line = -1)
mtext(paste("SSA(", round(ht1_1, 2), ",", delta, ")   ", sep = ""),
      col = coloh[2], line = -2)

lines(mplot[, 2], col = coloh[2], lwd = 2, lty = 1)
lines(mplot[, 3], col = coloh[3], lwd = 2, lty = 1)

mtext(paste("HP(", lambda_HP, ")   ", sep = ""),
      col = coloh[3], line = -3)

axis(1,
     at     = c(1, 50 * 1:(nrow(mplot) / 50)),
     labels = c(0,    50 * 1:(nrow(mplot) / 50)))
axis(2)
box()







# --- Compute the spectral radius of M-tilde (see Wildi (2026b)---
# rho_max is the largest eigenvalue of M_tilde; it defines the upper bound
# for the HT constraint parameter rho for a filter of length L.
M_obj    <- M_func(L, Sigma)
M_tilde  <- M_obj$M_tilde
rho_max  <- max(eigen(M_tilde)$values)   # Maximum achievable rho
v1       <- eigen(M_tilde)$vectors[, 1]  # Leading eigenvector of M_tilde (to the max eigenvalue rho_max)












