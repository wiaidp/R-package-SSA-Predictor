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
# TWO MAIN EXERCISES:

# Exercise 1: Univariate SSA-Smoothing
# -Contrast SSA-smoothing with classical Whittaker–Henderson (WH) graduation and HP

# Exercise 2: Multivariate M-SSA Smoothing

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT TO EXERCISES 
# ══════════════════════════════════════════════════════════════════════════════
# WH graduation / HP filter
# ─────────────────────────
#   • The HP filter is the solution to the WH problem when the penalty term
#     emphasises squared second-order differences (curvature).
#   • Consequence: for a given level of curvature, HP is the closest possible
#     smoother to the original series x_t (i.e., HP minimises MSE subject to
#     a curvature constraint).
#
# (M-)SSA Smoothing: Concept 
# ───────────────
#   • (M-)SSA controls smoothness via the holding-time (HT) constraint rather
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
# Smoothing: Curvature vs. Backcasting
# ───────────────
#   • WH graduation is a classic smoothing approach: it tracks a target subject
#     to a regularisation term that penalises `noise'. In this sense, smoothness
#     is defined through curvature in HP (minimize squared second-order differences).
#
#      → Exercises 1&2 compare SSA's HT against WH-HP's curvature approaches. 
#
#   • Alternatively, the term "smoothing" also refers to backcasting:
#     the retrospective refinement of historical estimates of a smooth
#     component (e.g., a trend or cycle) by incorporating all data up to the
#     most recent observation. The additional information available at the end
#     of the sample allows the filter to suppress spurious noise more
#     effectively, so that historical estimates gain in smoothness relative to
#     their real-time counterparts.
#
#      →Exercise 3 explores M-SSA in the context of this backcasting notion of
#       smoothness, examining how the filter evolves as the lag delta increases
#       and more future observations become available.

# ══════════════════════════════════════════════════════════════════════════════

# Clear the workspace to ensure a clean environment
rm(list = ls())


# LOAD REQUIRED LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# HP filter and other standard time-series filters
library(mFilter)


# LOAD CUSTOM M-SSA FUNCTION LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# Core M-SSA filter construction and optimisation routines
source(paste(getwd(), "/R/functions_MSSA.r", sep = ""))

# HP-filter utilities used in the JBCY paper (depends on mFilter)
source(paste(getwd(), "/R/HP_JBCY_functions.r", sep = ""))

# M-SSA utility functions: data preparation, plotting helpers, and wrappers
source(paste(getwd(), "/R/M_SSA_utility_functions.r", sep = ""))



# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: 
# GOAL
# ════
# Benchmark (univariate) SSA smoothing against Whittaker–Henderson (WH) graduation,
# which generalises the HP filter (assuming second-order differences, d = 2).
# See Wildi (2026b), Section 4.2 for full theoretical background.
#
# BACKGROUND  (Wildi 2026b, Section 4.2)
# ═══════════

# COMPARISON STRATEGY 
# ════════════════════════════════════
# Two complementary designs are evaluated against WH (or HP):
#
#   Exercise 1.1 — Match holding time of HP:
#     Design SSA  so that its HT equals that of HP.
#     Then compare the MSE tracking and curvature of both smoothers.
#     Expected result: SSA  achieves better MSE tracking (closer to x_t)
#     for the same HT but larger curvature.

#   Exercise 1.2 — Match MSE tracking of x_t:
#     Design SSA  so that its MSE tracking of x_t equals that of HP.
#     Then compare the HT and curvature of both smoothers.
#     Expected result:  SSA achieves a longer HT (smoother output)
#     for the same tracking accuracy, but larger curvature.



# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1.1: SSA vs. HP — Match Holding TimeS
# ══════════════════════════════════════════════════════════════════════════════
# Design SSA with the same HT as the symmetric HP filter, then compare
# tracking accuracy (target correlation) and curvature of both smoothers.
# Expected outcome: SSA outperforms HP in target correlation (and hence
# sign accuracy / MSE) for the same degree of smoothness.
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.1  Specify the HP Filter
# ─────────────────────────────────────────────────────────────────────────────

lambda_HP <- 14400   # Monthly HP smoothing parameter (standard business-cycle value)
L         <- 201     # Filter length (number of coefficients)

# Compute the bi-infinite HP target filter and its MSE-optimal finite approximation
HP_obj    <- HP_target_mse_modified_gap(L, lambda_HP)
hp_target <- HP_obj$target    # Bi-infinite HP filter coefficients (length L)
hp_trend  <- HP_obj$hp_trend  # Associated HP trend estimate


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.2  Specify SSA Design Settings
# ─────────────────────────────────────────────────────────────────────────────

# Sigma: variance–covariance matrix of the innovation process.
#        NULL → univariate design; identity matrix assumed.
Sigma <- NULL

# xi: spectral density of the input process.
#     NULL → white-noise assumption (xi = Identity).
xi <- NULL

# symmetric_target: FALSE → causal (one-sided) target filter.
symmetric_target <- FALSE

# --- Target filter ---
# gamma_target = 1 → allpass (identity) target: SSA tracks x_t itself.
# This contrasts with signal-extraction or nowcasting settings where the
# target is a non-trivial filtered version of x_t (e.g., the HP trend).
# MSSA_func zero-pads gamma_target to length L automatically if needed.
gamma_target <- 1

# --- Lag parameter delta ---
# The SSA filter targets x_{t + delta}, i.e., it tracks the series value
# delta steps ahead of (or behind) the current observation t.
#   delta < 0  → smoother  (target lies in the past; causal design)
#   delta = 0  → nowcaster (real-time estimate of x_t)
#   delta > 0  → predictor (target lies in the future)
#
# Setting delta = -(L-1)/2 places the target at the centre of the filter
# support, yielding a symmetric (two-sided) smoothing filter — directly
# comparable to the symmetric HP filter used as benchmark.
delta <- -(L - 1) / 2

# --- Bisection optimiser resolution ---
# The optimiser halves the search interval at each step, so effective
# resolution scales as 2^split_grid (much faster than brute-force search).
#   • split_grid = 20 provides a good balance of speed and precision.
#   • Increase if convergence diagnostics indicate insufficient resolution.
split_grid <- 20


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.3  Specify the HT Constraint — Match HP (Strategy 1)
# ─────────────────────────────────────────────────────────────────────────────
# Impose the same HT as the two-sided HP filter.
# Both filters are symmetric (delta = -(L-1)/2), so a like-for-like
# comparison is valid.
# Under an identical HT constraint, SSA is guaranteed to outperform HP
# in target correlation (and hence sign accuracy / MSE).

# Extract the HT and the corresponding rho from the bi-infinite HP filter
rho1 <- compute_holding_time_func(hp_target)$rho_ff1  # rho implied by HP's HT
ht1  <- compute_holding_time_func(hp_target)$ht       # Holding time of HP filter

# Interpretation of ht1:
#   The symmetric HP filter applied to white noise produces sign changes with
#   a mean inter-crossing duration of approximately 60 time steps.
#   Imposing this HT on SSA ensures both smoothers operate at the same
#   level of smoothness; any gain in tracking accuracy then reflects the
#   superior optimality of SSA within the HT-constrained class.

ht1   # Inspect the HP holding time


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.4  Compute the SSA Filter
# ─────────────────────────────────────────────────────────────────────────────
# Design the SSA filter subject to the HP-matched HT constraint.
# Note: xi = NULL assumes white-noise input; MSSA_func will issue a warning
#       confirming that gamma_target has been zero-padded to length L.

SSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1)

# Extract the optimised (univariate) filter coefficients
bk_mat <- SSA_obj$bk_mat

# --- Visual inspection ---
# The filter should be symmetric, consistent with delta = -(L-1)/2
par(mfrow = c(1, 1))
ts.plot(bk_mat, main = "SSA Filter Coefficients (Strategy 1: Match HT)")

# --- Optimisation diagnostics ---
# crit_rhoy_target: maximised target correlation achieved by SSA.
#                   This should exceed the HP target correlation.
SSA_obj$crit_rhoy_target


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.5  Results: Tracking Accuracy on Simulated White Noise
# ─────────────────────────────────────────────────────────────────────────────

# Simulate a long white-noise series for empirical evaluation
lenq <- 100000
set.seed(86)
x <- rnorm(lenq)

# Apply both symmetric filters to x
y_ssa    <- filter(x, bk_mat,    sides = 2)  # M-SSA smoother output
y_hp_two <- filter(x, hp_target, sides = 2)  # HP smoother output

# --- MSE relative to the original series ---
# MSE_scale normalises the SSA output to the same amplitude as x before
# computing MSE, ensuring a fair comparison with HP.
MSE_scale      <- as.double(bk_mat[(L + 1) / 2] / t(bk_mat) %*% bk_mat)
mse_ssa_smooth <- mean((x - MSE_scale * y_ssa)^2, na.rm = TRUE)
mse_hp_smooth  <- mean((x - y_hp_two)^2,          na.rm = TRUE)

# --- Correlation matrix: x, SSA output, HP output ---
filter_mat <- na.exclude(cbind(x, y_ssa, y_hp_two))
cor_mat    <- cor(filter_mat)
cor_mat


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.6  Interpretation of Results
# ─────────────────────────────────────────────────────────────────────────────

# 1. SSA target correlation matches the optimised objective (internal check):
#    The sample correlation cor(x, y_ssa) should equal crit_rhoy_target,
#    confirming that the optimisation converged correctly.
cor_mat[1, 2]              # Empirical correlation: x vs. M-SSA output
SSA_obj$crit_rhoy_target   # Optimised objective value

# 2. HP incurs a loss in target correlation of approximately 10% relative to
#    SSA, despite operating under the same HT constraint.
#    SSA target correlation:
cor_mat[1, 2]
#    HP target correlation:
cor_mat[1, 3]


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.7  Main Take-Aways
# ─────────────────────────────────────────────────────────────────────────────
# 1. For a given HT constraint, SSA maximises target correlation and hence
#    sign accuracy (equivalently, minimises MSE after suitable calibration).
#
# 2. Imposing the HP holding time on SSA guarantees that SSA outperforms
#    HP in tracking x_{t + delta}.
#
# 3. The gain in tracking accuracy comes at the cost of larger curvature
#    (greater mean squared second-order differences) relative to HP, which
#    minimises curvature by construction (WH optimality).
#
# 4. These results extend straightforwardly from white noise to arbitrary
#    stationary dependent processes by specifying the appropriate xi (Wold decomposition).
# ─────────────────────────────────────────────────────────────────────────────




# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1.2: SSA vs. HP — Match target Correlation (MSE)
# ══════════════════════════════════════════════════════════════════════════════

# 1.2.1 Compute SSA
# Strategy 2: find the HT such that SSA achieves the same target correlation
#             as HP. Under identical correlation,  SSA will outperform HP in HT.

ht1_1  <- 75                                    # Target HT to match HP correlation
rho1_1 <- compute_rho_from_ht(ht1_1)$rho        # Corresponding rho value

# Design  filter with the correlation-matched HT constraint
SSA_obj_1 <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1_1)

# Extract filter coefficients and verify correlation
bk_mat_1 <- SSA_obj_1$bk_mat
ts.plot(bk_mat_1)
SSA_obj_1$crit_rhoy_target   # Should approximately equal cor(filter_mat)[1, 3]
# Check:  replicates (sample) target correlation of HP
cor(filter_mat)[1, 3]

# ══════════════════════════════════════════════════════════════════════════════
# 1.2.2 Compute Curvatures
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
# 1.2.3 Plot and Sample Performances
# ══════════════════════════════════════════════════════════════════════════════

# --- Time-series plot of a 1 000-observation window ---
colo <- c("blue", "violet", "black")
ts.plot(na.exclude(yhat_mat)[1000:2000, ], col = colo)
abline(h = 0)
for (i in 1:ncol(yhat_mat))
  mtext(colnames(yhat_mat)[i],col=colo[i],line=-i)

# --- Autocorrelation functions ---
#  filters: slowly decaying ACF (long memory in the smooth output)
acf(na.exclude(yhat_mat)[, 1], lag.max = 100)
acf(na.exclude(yhat_mat)[, 2], lag.max = 100)

# HP filter: faster decay with cyclical structure; half-period ≈ 57 lags,
# consistent with its holding time
acf(na.exclude(yhat_mat)[, 3], lag.max = 100)

# Diagnostics: comparison of sample performances

# Empirical holding times:  should equal or exceed HP's HT
apply(na.exclude(yhat_mat), 2, compute_empirical_ht_func)

# Tracking ability (correlation with x):  should equal or exceed HP
cor(na.exclude(cbind(x, yhat_mat)))[1, ]


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 2:  COMPUTE THE FULL SSA SMOOTHER FAMILY (delta SWEEP)
# ══════════════════════════════════════════════════════════════════════════════
# Sweep delta from the fully causal end (-(L-1)/2) to nowcast (0),
# designing one  filter per lag shift. This may be time-consuming;
# set recompute_calculations = TRUE only when a fresh run is needed.

# 2.1
# Re-extract HP holding-time parameters for the sweep
rho1 <- compute_holding_time_func(hp_target)$rho_ff1
ht1  <- compute_holding_time_func(hp_target)$ht
  
gamma_target <- 1   # Allpass target
filt_mat     <- NULL
  
# Design one  filter for each lag delta
for (delta in (-(L - 1) / 2):0)
{
  print(delta)
  SSA_obj  <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1)
  filt_mat <- cbind(filt_mat, SSA_obj$bk_mat)
}
  


# ══════════════════════════════════════════════════════════════════════════════
# 2.2 PLOT THE FULL SSA SMOOTHER FAMILY
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
# 2.3 OVERLAY PLOT — SYMMETRIC SSA VS HP FILTER COEFFICIENTS
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












