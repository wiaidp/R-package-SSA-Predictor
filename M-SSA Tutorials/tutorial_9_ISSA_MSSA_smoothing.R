# This tutorial is under construction

# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 9: I-SSA and M-SSA SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCES
# ──────────────────────
#
# Wildi, M. (2024). Business Cycle Analysis and Zero-Crossings of Time Series:
#    A Generalized Forecast Approach.  Journal of Business Cycle Research,
#   https://doi.org/10.1007/s41549-024-00097-5

# Wildi, M. (2026a). Sign Accuracy, Mean-Squared Error and the Rate of Zero Crossings:
#     a Generalized Forecast Approach, https://doi.org/10.48550/arXiv.2601.06547
#
# Wildi, M. (2026b). The Accuracy-Smoothness Dilemma in Prediction:
#   A Novel Multivariate M-SSA Forecast Approach.
#   Journal of Time Series Analysis, http://doi.org/10.1111/jtsa.70058 
#   arXiv: https://doi.org/10.48550/arXiv.2602.13722
#
# Parts in this  tutorial are  based on Wildi (2026b), Section 4.2. Additional 
#   applications are given in Wildi (2024). Extensions to non-stationary series (I-SSA) 
#   are presented in Wildi (2026a).

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

source(file.path(getwd(), "R/simple_sign_accuracy.r"))

# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1: Random Walk
# Backcasting: Symmetric Smoother with HT of Two-Sided HP
# ══════════════════════════════════════════════════════════════════════════════
# We assume that x_t follows a random walk (i.e., first differences are white
# noise). A very long series is simulated to ensure sample estimates converge
# closely to their expected values (asymptotic regime).
a1 <- b1 <- 0
len <- 10000000
set.seed(12)
eps <- rnorm(len)
x   <- cumsum(eps)

# ─────────────────────────────────────────────────────────────────────────────
# 1.1  Model Setup for I-SSA
# ─────────────────────────────────────────────────────────────────────────────

# Filter design parameters
L         <- 201
lambda_hp <- 14400

# Compute system matrices and filters
filter_obj <- compute_system_filters_func(L, lambda_hp, a1, b1)

B           <- filter_obj$B            # Cointegration matrix (see cited literature)
M           <- filter_obj$M            # Lag-one autocovariance generating matrix
gamma_tilde <- filter_obj$gamma_tilde  # Transformed (HP) target
gamma_mse   <- filter_obj$gamma_mse    # MSE-optimal filter (one-sided estimate of HP target)
Xi_tilde    <- filter_obj$Xi_tilde     # Convolution operator (see Section 5.3: Wold
#   decomposition of first differences convolved
#   with the integration operator)
Sigma       <- filter_obj$Sigma        # Integration operator (see Section 5.3)
Delta       <- filter_obj$Delta        # Differencing operator
Xi          <- filter_obj$Xi           # Wold MA representation in matrix form
#   (see equation 22 in Wildi 2026a)
hp_target   <- filter_obj$hp_two       # Two-sided HP target filter
hp_trend    <- filter_obj$hp_trend     # Classic one-sided HP (HP-C): benchmark
#   for I-SSA customisation

# ─────────────────────────────────────────────────────────────────────────────
# 1.2  I-SSA Settings
# ─────────────────────────────────────────────────────────────────────────────

# 1.2.1  Holding-Time (HT) Constraint Calibration
# ─────────────────────────────────────────────────────────────────────────────
# The HT constraint is defined on first differences (see equation 29,
# Wildi 2026a). The HT of the first differences of the two-sided HP applied
# to x_t equals the HT of HP itself.
rho1 <- compute_holding_time_func(hp_target)$rho_ff1
ht1  <- compute_holding_time_func(hp_target)$ht
# In first differences, a zero-crossing of the two-sided HP occurs on average
# once every 60 observations.
ht1
# Important: HP enters I-SSA through the HT constraint only.
# The HP filter itself is not targeted explicitly; only its HT is matched.

# 1.2.2  Specify Smoothing Lag and Targets
# ─────────────────────────────────────────────────────────────────────────────
delta <- -(L - 1) / 2 - 1

# Target in levels: we want to track the random walk x_t directly,
# so the target filter is the identity (a unit spike at lag -delta).
# This differs from Exercise 6, where the target was the one-sided HP of x_t.
target_filter <- c(rep(0, -delta - 1), 1, rep(0, L + delta))

# Target in first differences: apply the summation operator to the level target.
# This is a finite-length proxy of the effective first-difference target used
# in optimisation. The proxy is equivalent to the effective target when L is
# sufficiently large, because MA coefficients decay fast enough under the
# cointegration constraint (see Section 5.3 in Wildi 2026a for details).
target_filter_diff <- cumsum(target_filter)

par(mfrow = c(2, 1))
ts.plot(target_filter,
        main = paste("Target filter: backcast x_t with lag =", -delta),
        ylab = "", xlab = "Lag")
ts.plot(target_filter_diff,
        main = "Finite-length target filter in differences (summation/integration of level target)",
        ylab = "", xlab = "Lag")

# The targets do not hint at HP: HP enters into I-SSA through its HT only

# ─────────────────────────────────────────────────────────────────────────────
# 1.3  Compute I-SSA Solution (Wildi 2026a, Sections 5.3–5.4)
# ─────────────────────────────────────────────────────────────────────────────
# Numerical optimisation (optim) determines the optimal Lagrange multiplier
# lambda ensuring compliance with the HT constraint.
# Initialising at lambda = 0 corresponds to the unconstrained MSE benchmark.
# Note: lambda here is the SSA Lagrange multiplier and is unrelated to the HP
# regularisation parameter lambda_hp.

# 1.3.1  I-SSA Optimisation
# ─────────────────────────────────────────────────────────────────────────────
lambda <- 0

opt_obj <- optim(
  lambda,
  b_optim,
  lambda,
  target_filter,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  target_filter_diff,
  rho1
)

# Optimal Lagrange multiplier
lambda_opt <- opt_obj$par

# Compute the I(1) cointegrated I-SSA solution based on lambda_opt
bk_obj <- bk_int_func(
  lambda_opt,
  target_filter,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  target_filter_diff,
  rho1
)

# 1.3.2  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# Verify convergence: bk_obj$rho_yy should match rho1.
# If the values differ, increase the number of iterations in optim.
bk_obj$rho_yy   # Empirical lag-1 ACF of filter output (should equal rho1)
rho1            # Imposed HT constraint

# The correlation with the target is not well defined for non-stationary
# processes in general. However, the finite-length inversion used here (see
# equation 29, left side, Wildi 2026a) yields a well-defined and pertinent
# expression for the purpose of numerical optimisation.
bk_obj$rho_yz   # Correlation with target (finite-length approximation)

# The MSE is well defined under the cointegration constraint;
# without it the MSE would be infinite for a non-stationary process.
# Sample MSE performance is expected to converge to this theoretical value.
bk_obj$mse_yz   # Theoretical MSE: tracking accuracy of I-SSA for x_{t+delta}

# Extract the filter applied to the data
b_x <- bk_obj$b_x

# Verify the cointegration constraint: the difference should vanish (≈ 0),
# ensuring a finite MSE on non-stationary levels.
sum(b_x) - sum(target_filter)

# ─────────────────────────────────────────────────────────────────────────────
# 1.3.3  Plot Smoothers
# ─────────────────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "black", "blue", "red")

# Trim the two-sided HP filter to length L
hp_two <- hp_target[((L / 2) + 1):(length(hp_target) - (L / 2) + 1)]

mplot <- cbind(hp_two, target_filter, b_x)
colnames(mplot) <- c("HP-two", "Target", "I-SSA")

# Full filter coefficient profiles
plot(mplot[, 1], main = "Trend filters", axes = FALSE, type = "l",
     ylab = "", xlab = "Lags", col = colo[1], lwd = 1,
     ylim = range(mplot))
abline(h = 0)
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:nrow(mplot), labels = 0:(nrow(mplot) - 1))
axis(2); box()

# Zoom in on a 100-lag window centred on -delta
mplot <- mplot[(-delta - 50):(-delta + 50), ]

plot(mplot[, 1], axes = FALSE, type = "l", col = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot[, "I-SSA"])), ylab = "", xlab = "Lags")
abline(h = 0)
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:101, labels = (-delta - 50):(-delta + 50))
axis(2); box()

# Unlike the SSA smoother in Tutorial 8 (which targets stationary series),
# I-SSA adopts a cyclical shape when applied to a non-stationary series
# (random walk). This reflects the strong low-frequency content of the random
# walk: to track x_{t+delta} optimally, I-SSA emphasizes low frequencies
# accordingly. We now verify sample performance: I-SSA should outperform HP
# in terms of MSE and match HP in terms of HT in first differences.


# ─────────────────────────────────────────────────────────────────────────────
# 1.4  Check Performances
# ─────────────────────────────────────────────────────────────────────────────

# 1.4.1  Apply Smoothers to Data
# ─────────────────────────────────────────────────────────────────────────────
# sides = 2 applies an acausal two-sided convolution, consistent with the
# symmetric backcast design (delta = -(L-1)/2).
y_ssa <- filter(x, b_x,   sides = 2)  # I-SSA smoother applied to x_t (two-sided)
y_hp  <- filter(x, hp_two, sides = 2)  # HP  smoother applied to x_t (two-sided)

# Since both filters are two-sided, the target is the unshifted series x
target <- x

# Visual inspection: I-SSA appears to track x_t more closely than HP on average
par(mfrow = c(1, 1))
ts.plot(cbind(target, y_ssa, y_hp)[5000:5500, ], col = c("black", "blue", "violet"))

# ─────────────────────────────────────────────────────────────────────────────
# 1.4.2  Tracking Accuracy: MSE
# ─────────────────────────────────────────────────────────────────────────────
# MSE quantifies how closely each smoother tracks the target on average.
# Expected result: I-SSA achieves a lower MSE than HP under the same HT constraint.
mse_ssa_smooth <- mean((target - y_ssa)^2, na.rm = TRUE)
mse_hp_smooth  <- mean((target - y_hp)^2,  na.rm = TRUE)

mse_hp_smooth   # HP  MSE (sample)
mse_ssa_smooth  # I-SSA MSE (sample)

# The sample MSE of I-SSA should converge to the theoretical value below.
# Any discrepancy reflects Monte Carlo sampling variability.
bk_obj$mse_yz   # Theoretical MSE under the cointegration constraint

# I-SSA outperforms HP by approximately 50% in MSE reduction.

# Note: target correlation and sign accuracy are not meaningful criteria for
# non-stationary time series and are therefore omitted here.

# ─────────────────────────────────────────────────────────────────────────────
# 1.4.3  Smoothness
# ─────────────────────────────────────────────────────────────────────────────

# 1.4.3.1  Holding Time
# ─────────────────────────────────────────────────────────────────────────────
# HT is evaluated on first differences in I-SSA.
# Both I-SSA and HP should match the imposed HT target ht1.
ht1                                     # Target HT (= HP holding time in first differences)
compute_empirical_ht_func(diff(y_ssa))  # Empirical HT of I-SSA output (first differences)
compute_empirical_ht_func(diff(y_hp))   # Empirical HT of HP  output (first differences)

output_mat_diff <- cbind(x, y_ssa, y_hp)

# 1.4.3.2  Curvature (Root Mean Squared Second-Order Differences)
# ─────────────────────────────────────────────────────────────────────────────
# HP minimises curvature by construction (WH optimality) and therefore
# exhibits slightly smaller RMSD2 than I-SSA under the same HT constraint.
# In contrast to Tutorial 8, the difference in curvature is less pronounced
# here, reflecting the smoother weight profile of I-SSA on non-stationary data.
sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat_diff), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif

# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways
# ─────────────────────────────────────────────────────────────────────────────
# Exercise 1 demonstrates symmetric I-SSA backcasting of a non-stationary
# random walk, with the HT constraint matched to that of the two-sided HP
# filter in first differences.
#
# 1. For identical HT in first differences, I-SSA reduces MSE on levels by
#    approximately 50% relative to HP.
#
# 2. HP exhibits slightly smaller curvature than I-SSA under the same HT
#    constraint, as expected from WH optimality. The gap is less pronounced
#    here than in Tutorial 8.
#
# 3. Unlike SSA smoothing applied to white noise (Tutorial 8), I-SSA smoothing
#    of a non-stationary level sequence adopts a cyclical shape. This reflects
#    the strong low-frequency content of the random walk: I-SSA emphasizes low
#    frequencies in order to track x_{t+delta} optimally.
#
# 4. In contrast to Tutorial 8, I-SSA targets the non-stationary level x_t
#    directly, while SSA in Tutorial 8 targets growth in stationary first
#    differences. I-SSA imposes the HT constraint on stationary
#    first differences, ensuring a well-defined and interpretable smoothness
#    criterion. Both approaches are logically consistent, statistically
#    efficient, and data-driven: neither imprints extraneous structure on the
#    smoothed output.
# ─────────────────────────────────────────────────────────────────────────────






# ══════════════════════════════════════════════════════════════════════════════
# Exercise 2: Same as Exercise 1 but for the Nowcast Smoother (delta = 0)
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Specify Smoothing Lag and Targets
# ─────────────────────────────────────────────────────────────────────────────
delta <- 0

# Target in levels: track the random walk x_t directly (identity target).
# This differs from Exercise 6, where the target was the one-sided HP of x_t.
target_filter <- c(rep(0, -delta), 1, rep(0, L + delta - 1))

# Target in first differences: apply the summation operator to the level target.
# This is a finite-length proxy of the effective first-difference target used
# in optimisation. The proxy is equivalent to the effective target when L is
# sufficiently large, because MA coefficients decay fast enough under the
# cointegration constraint (see Section 5.3 in Wildi 2026a for details).
target_filter_diff <- cumsum(target_filter)

par(mfrow = c(2, 1))
ts.plot(target_filter,
        main = paste("Target filter: nowcast x_t (lag =", -delta,")"),
        ylab = "", xlab = "Lag")
ts.plot(target_filter_diff,
        main = "Finite-length target filter in differences (summation/integration of level target)",
        ylab = "", xlab = "Lag")

# The targets do not hint at HP: HP enters into I-SSA through its HT only

# ─────────────────────────────────────────────────────────────────────────────
# 2.2  Compute I-SSA Solution (Wildi 2026a, Sections 5.3–5.4)
# ─────────────────────────────────────────────────────────────────────────────
# Numerical optimisation (optim) determines the optimal Lagrange multiplier
# lambda ensuring compliance with the HT constraint.
# Initialising at lambda = 0 corresponds to the unconstrained MSE benchmark.
# Note: lambda here is the SSA Lagrange multiplier and is unrelated to the HP
# regularisation parameter lambda_hp.

# 2.2.1  I-SSA Optimisation
# ─────────────────────────────────────────────────────────────────────────────
lambda <- 0

opt_obj <- optim(
  lambda,
  b_optim,
  lambda,
  target_filter,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  target_filter_diff,
  rho1
)

# Optimal Lagrange multiplier
lambda_opt <- opt_obj$par

# Compute the I(1) cointegrated I-SSA solution based on lambda_opt
bk_obj <- bk_int_func(
  lambda_opt,
  target_filter,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  target_filter_diff,
  rho1
)

# 2.2.2  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# Verify convergence: bk_obj$rho_yy should match rho1.
# If the values differ, increase the number of iterations in optim.
bk_obj$rho_yy   # Empirical lag-1 ACF of filter output (should equal rho1)
rho1            # Imposed HT constraint

# The correlation with the target is not well defined for non-stationary
# processes in general. However, the finite-length inversion used here (see
# equation 29, left side, Wildi 2026a) yields a well-defined and pertinent
# expression for the purpose of numerical optimisation.
bk_obj$rho_yz   # Correlation with target (finite-length approximation)

# The MSE is well defined under the cointegration constraint;
# without it the MSE would be infinite for a non-stationary process.
# Sample MSE performance is expected to converge to this theoretical value.
# Note: the nowcast MSE is substantially larger than the backcast MSE in
# Exercise 1, as the filter has no access to future observations.
bk_obj$mse_yz   # Theoretical MSE: tracking accuracy of I-SSA for x_{t+delta}

# Extract the filter applied to the data
b_x <- bk_obj$b_x

# Verify the cointegration constraint: the difference should vanish (≈ 0),
# ensuring a finite MSE on non-stationary levels.
sum(b_x) - sum(target_filter)

# ─────────────────────────────────────────────────────────────────────────────
# 2.2.3  Plot Smoothers
# ─────────────────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "black", "blue", "red")

# Trim the two-sided HP filter to length L
hp_two <- hp_target[((L / 2) + 1):(length(hp_target) - (L / 2) + 1)]

mplot <- cbind(hp_two, target_filter, b_x)
colnames(mplot) <- c("HP-two", "Target", "I-SSA")

# Full filter coefficient profiles
plot(mplot[, 1], main = "Trend filters", axes = FALSE, type = "l",
     ylab = "", xlab = "Lags", col = colo[1], lwd = 1,
     ylim = range(mplot))
abline(h = 0)
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:nrow(mplot), labels = 0:(nrow(mplot) - 1))
axis(2); box()

# Zoom in on the first 100 lags (relevant for the nowcast filter)
mplot <- mplot[1:100, ]

plot(mplot[, 1], axes = FALSE, type = "l", col = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot[, "I-SSA"])), ylab = "", xlab = "Lags")
abline(h = 0)
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:101, labels = 0:100)
axis(2); box()

# The I-SSA nowcast filter exhibits the characteristic nose shape near lag 0.
# This arises from the zero boundary constraint b_{-1} = 0 (the coefficient
# at lag -1 is not shown since the filter starts at lag 0); see Theorem 1 in
# Wildi 2026a and 2026b for details.
# Importantly, this nowcast smoother achieves the same HT as the two-sided HP,
# and a much larger HT than the classic one-sided HP.

# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Check Performances
# ─────────────────────────────────────────────────────────────────────────────

# 2.3.1  Apply Smoothers to Data
# ─────────────────────────────────────────────────────────────────────────────
# sides = 1 applies a causal one-sided convolution for the nowcast filters;
# the two-sided HP uses sides = 2 as a benchmark.
y_ssa    <- filter(x, b_x,      sides = 1)  # I-SSA nowcast smoother
y_hp_one <- filter(x, hp_trend, sides = 1)  # Classic one-sided HP (HP-C)
y_hp_two <- filter(x, hp_two,   sides = 2)  # Two-sided HP (acausal benchmark)

# The nowcast target is the contemporaneous level x_t (delta = 0)
target <- x

# Visual inspection: I-SSA (blue) is very smooth but exhibits a lag relative to 
# x_t
par(mfrow = c(1, 1))
ts.plot(cbind(target, y_ssa, y_hp_one, y_hp_two)[5000:5500, ],
        col = c("black", "blue", "red", "violet"))

# ─────────────────────────────────────────────────────────────────────────────
# 2.3.2  Tracking Accuracy: MSE
# ─────────────────────────────────────────────────────────────────────────────
# MSE quantifies how closely each smoother tracks the target on average.
# Note: target correlation and sign accuracy are not meaningful for non-stationary
# time series and are therefore omitted.
mse_ssa_smooth     <- mean((target - y_ssa)^2,    na.rm = TRUE)
mse_hp_one_smooth  <- mean((target - y_hp_one)^2, na.rm = TRUE)
mse_hp_two_smooth  <- mean((target - y_hp_two)^2, na.rm = TRUE)

mse_hp_one_smooth  # HP one-sided MSE (sample)
mse_hp_two_smooth  # HP two-sided MSE (sample)
mse_ssa_smooth     # I-SSA MSE (sample)

# Surprisingly, the one-sided HP achieves the lowest MSE among the three
# smoothers, even though the two-sided HP can rely on future observations. 
# This demonstrates once more that the two-sided HP is two smooth for 
# applications with a single unit root (typical economic series). 

# The sample MSE of I-SSA should converge to the theoretical value below.
bk_obj$mse_yz      # Theoretical MSE under the cointegration constraint

# ─────────────────────────────────────────────────────────────────────────────
# 2.3.3  Smoothness
# ─────────────────────────────────────────────────────────────────────────────

# 2.3.3.1  Holding Time
# ─────────────────────────────────────────────────────────────────────────────
# HT is evaluated on first differences.
# I-SSA and the two-sided HP should match ht1; the one-sided HP should exhibit
# a much smaller HT, reflecting its weaker smoothing constraint.
ht1                                       # Target HT (= two-sided HP holding time)
compute_empirical_ht_func(diff(y_ssa))    # Empirical HT of I-SSA output
compute_empirical_ht_func(diff(y_hp_two)) # Empirical HT of two-sided HP output
compute_empirical_ht_func(diff(y_hp_one)) # Empirical HT of one-sided HP output (much smaller)


# 2.3.3.2  Curvature (Root Mean Squared Second-Order Differences)
# ─────────────────────────────────────────────────────────────────────────────
# HP minimises curvature by construction (WH optimality) and therefore
# exhibits slightly smaller RMSD2 than I-SSA under the same HT constraint.
# In contrast to Tutorial 8, the difference is less pronounced here.

output_mat_diff <- cbind(x, y_ssa, y_hp_one, y_hp_two)

sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat_diff), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif

# Note that I-SSA has much smaller curvature than the one-sided HP. 
# This result constrasts with tutorial 8 and is partly due to the cyclical 
# shape of the SSA smoother here.


# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways
# ─────────────────────────────────────────────────────────────────────────────
# In contrast to Exercise 1, Exercise 2 emphasized a nowcast smoothing problem.
#
# 1. The classic HP nowcast filter has a substantially shorter HT
#    than the two-sided (symmetric) HP smoother.
#
# 2. By contrast, I-SSA can be constrained to produce a nowcast smoother whose
#    HT matches that of the two-sided HP smoother.
#
# 3. However, achieving this longer HT comes at a cost: the resulting I-SSA
#    nowcast smoother exhibits a considerably larger MSE, reflecting the
#    trade-off between smoothness, retardation (right-shift/lag), and accuracy.
#
# Building on these findings, we next investigate how I-SSA performs relative
# to the one-sided HP filter when both are subject to the same (shorter) HT
# constraint — i.e., the HT of the classic HP nowcast. Under this setting,
# I-SSA is expected to outperform the one-sided HP filter in terms of MSE.
# ─────────────────────────────────────────────────────────────────────────────




# ══════════════════════════════════════════════════════════════════════════════
# Exercise 3: Same as Exercise 2 (Nowcast) but Matching the HT of One-Sided HP
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Specify I-SSA Settings
# ─────────────────────────────────────────────────────────────────────────────

# 3.1.1  Smoothing Lag and Targets
# ─────────────────────────────────────────────────────────────────────────────
# Nowcast: as in exercise 2
delta <- 0

# Target : the same as in exercise 2 above
target_filter <- c(rep(0, -delta), 1, rep(0, L + delta - 1))
target_filter_diff <- cumsum(target_filter)

par(mfrow = c(2, 1))
ts.plot(target_filter,
        main = paste("Target filter: nowcast x_t (lag =", -delta,")"),
        ylab = "", xlab = "Lag")
ts.plot(target_filter_diff,
        main = "Finite-length target filter in differences (summation/integration of level target)",
        ylab = "", xlab = "Lag")

# The targets do not hint at HP: HP enters into I-SSA through its HT only

# 3.1.2  Holding-Time (HT) Constraint Calibration
# ─────────────────────────────────────────────────────────────────────────────
# Here we depart from exercise 2.
# We match the HT of the one-sided HP rather than the two-sided HP.
# The one-sided HP has a much smaller HT, reflecting weaker smoothing.
rho1 <- compute_holding_time_func(hp_trend)$rho_ff1
ht1  <- compute_holding_time_func(hp_trend)$ht
ht1

# ─────────────────────────────────────────────────────────────────────────────
# 3.2  Compute I-SSA Solution (Wildi 2026a, Sections 5.3–5.4)
# ─────────────────────────────────────────────────────────────────────────────

# 3.2.1  I-SSA Optimisation
# ─────────────────────────────────────────────────────────────────────────────
lambda <- 0

opt_obj <- optim(
  lambda,
  b_optim,
  lambda,
  target_filter,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  target_filter_diff,
  rho1
)

# Optimal Lagrange multiplier
lambda_opt <- opt_obj$par

# Compute the I(1) cointegrated I-SSA solution based on lambda_opt
bk_obj <- bk_int_func(
  lambda_opt,
  target_filter,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  target_filter_diff,
  rho1
)

# 3.2.2  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# Verify convergence: bk_obj$rho_yy should match rho1.
# If the values differ, increase the number of iterations in optim.
bk_obj$rho_yy   # Empirical lag-1 ACF of filter output (should equal rho1)
rho1            # Imposed HT constraint

bk_obj$rho_yz   # Correlation with target (finite-length approximation)

# The theortical MSE:
# Sample MSE should converge to this number.
# By matching the smaller HT of the one-sided HP, the nowcast MSE is
# expected to be smaller than in Exercise 2.
bk_obj$mse_yz   # Theoretical MSE: tracking accuracy of I-SSA for x_{t+delta}

# Extract the filter applied to the data
b_x <- bk_obj$b_x

# Verify the cointegration constraint: the difference should vanish (≈ 0),
# ensuring a finite MSE on non-stationary levels.
sum(b_x) - sum(target_filter)

# ─────────────────────────────────────────────────────────────────────────────
# 3.2.3  Plot Filters
# ─────────────────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "red", "black", "blue")

# Trim the two-sided HP filter to length L
hp_two <- hp_target[((L / 2) + 1):(length(hp_target) - (L / 2) + 1)]

mplot <- cbind(hp_two, hp_trend, target_filter, b_x)
colnames(mplot) <- c("HP-two", "HP-one", "Target", "I-SSA")

# Full filter coefficient profiles
plot(mplot[, 1], main = "Trend filters", axes = FALSE, type = "l",
     ylab = "", xlab = "Lags", col = colo[1], lwd = 1,
     ylim = range(mplot))
abline(h = 0)
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:nrow(mplot), labels = 0:(nrow(mplot) - 1))
axis(2); box()

# Zoom in on the first 100 lags (relevant for the nowcast filter)
mplot <- mplot[1:100, ]

plot(mplot[, 1], axes = FALSE, type = "l", col = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot[, "I-SSA"])), ylab = "", xlab = "Lags")
abline(h = 0)
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:101, labels = 0:100)
axis(2); box()

# The I-SSA nowcast filter exhibits the characteristic nose shape near lag 0.
# This arises from the zero boundary constraint b_{-1} = 0 (the coefficient
# at lag -1 is not shown since the filter starts at lag 0); see Theorem 1 in
# Wildi 2026a and 2026b for details.
# This nowcast smoother achieves the same HT as the one-sided HP.
# Remarkably, the I-SSA filter coefficients decay faster to zero than the 
# one-sided HP for identical smoothness (HT).

# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Check Performances
# ─────────────────────────────────────────────────────────────────────────────

# 3.3.1  Apply Smoothers to Data
# ─────────────────────────────────────────────────────────────────────────────
# sides = 1 applies a causal one-sided convolution for the nowcast filters;
# the two-sided HP uses sides = 2 as an acausal benchmark.
y_ssa    <- filter(x, b_x,      sides = 1)  # I-SSA nowcast smoother
y_hp_one <- filter(x, hp_trend, sides = 1)  # Classic one-sided HP (HP-C)
y_hp_two <- filter(x, hp_two,   sides = 2)  # Two-sided HP (acausal benchmark)

# The nowcast target is the contemporaneous level x_t (delta = 0)
target <- x

# Visual inspection: I-SSA is smoother than the one-sided HP but exhibits
# less lag than the two-sided HP
par(mfrow = c(1, 1))
ts.plot(cbind(target, y_ssa, y_hp_one, y_hp_two)[5000:5500, ],
        col = c("black", "blue", "red", "violet"))

# ─────────────────────────────────────────────────────────────────────────────
# 3.3.2  Tracking Accuracy: MSE
# ─────────────────────────────────────────────────────────────────────────────
# MSE quantifies how closely each smoother tracks the target on average.
# Note: target correlation and sign accuracy are not meaningful for non-stationary
# time series and are therefore omitted.
mse_ssa_smooth    <- mean((target - y_ssa)^2,    na.rm = TRUE)
mse_hp_one_smooth <- mean((target - y_hp_one)^2, na.rm = TRUE)
mse_hp_two_smooth <- mean((target - y_hp_two)^2, na.rm = TRUE)

mse_hp_one_smooth  # HP one-sided MSE (sample)
mse_hp_two_smooth  # HP two-sided MSE (sample)
mse_ssa_smooth     # I-SSA MSE (sample)

# Notably, the two-sided HP performs worst despite being acausal. This is
# because its large HT constraint imposes excessive smooting for typical I(1) 
# series. I-SSA, constrained to match the smaller HT of the one-sided
# HP, achieves a substantially lower MSE than in Exercise 2, now outperforming 
# the one-sided HP by a reduction of more than 50%.

# The sample MSE of I-SSA should converge to the theoretical value below.
bk_obj$mse_yz      # Theoretical MSE under the cointegration constraint

# ─────────────────────────────────────────────────────────────────────────────
# 3.3.3  Smoothness
# ─────────────────────────────────────────────────────────────────────────────

# 3.3.3.1  Holding Time
# ─────────────────────────────────────────────────────────────────────────────
# HT is evaluated on first differences.
# I-SSA and the one-sided HP should match ht1; the two-sided HP will exhibit
# a much larger HT.
ht1                                       # Target HT (= one-sided HP holding time)
compute_empirical_ht_func(diff(y_ssa))    # Empirical HT of I-SSA output
compute_empirical_ht_func(diff(y_hp_two)) # Empirical HT of two-sided HP output (much larger)
compute_empirical_ht_func(diff(y_hp_one)) # Empirical HT of one-sided HP output (should match ht1)

output_mat_diff <- cbind(x, y_ssa, y_hp_one, y_hp_two)

# 3.3.3.2  Curvature (Root Mean Squared Second-Order Differences)
# ─────────────────────────────────────────────────────────────────────────────
# The two-sided HP achieves the smallest curvature by construction (WH optimality).
# I-SSA exhibits slightly larger curvature than the one-sided HP but remains
# comparable (in contrast to bigger differences in tutorial 6).
sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat_diff), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif


# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways
# ─────────────────────────────────────────────────────────────────────────────
# In contrast to Exercise 2, Exercise 3 imposes the HT constraint of the
# one-sided HP filter.
#
# 1. Unlike Exercise 2 — where matching the two-sided HP's longer HT inflated
#    the MSE — I-SSA now substantially outperforms the classical one-sided HP
#    nowcast/smoother in terms of MSE on levels, while simultaneously matching
#    its degree of smoothness as measured by the HT in first differences.
#
# 2. Unlike Tutorial 8 — which operated on stationary (differenced) data —
#    I-SSA here directly tracks the target series x_t on non-stationary levels,
#    demonstrating extension of the smoothing concept to non-stationary series.
# ─────────────────────────────────────────────────────────────────────────────






# ══════════════════════════════════════════════════════════════════════════════
# Exercise 4: I-SSA Smoothing of Macro Indicator
# Target: US Industrial Production Index (INDPRO)
# ══════════════════════════════════════════════════════════════════════════════
# Similar to Tutorial 6, but targets the raw index I_t directly rather than
# its HP-filtered trend HP(I_t). Extends Exercise 3 by replacing the synthetic
# random-walk input with the empirical macroeconomic series INDPRO.
# Note:
# - The I-SSA filter applied here was optimised under a random-walk assumption
#   (i.e., white-noise first differences) and has not been re-fitted to the
#   empirical autocorrelation structure of INDPRO.
# - This is sub-optimal: the differenced index exhibits positive serial
#   correlation (see ACF in Section 4.3), which deviates from the white-noise
#   assumption underlying the current I-SSA design.
# - We acknowledge this limitation without correction, noting that re-fitting
#   I-SSA to the true dependence structure of INDPRO would likely yield further
#   improvements in MSE performance beyond those already documented above.

# ────────────────────────────────────────────────────────────────
# 4.1 Load Data
# ────────────────────────────────────────────────────────────────

reload_data <- FALSE

if (reload_data) {
  # Download INDPRO from FRED and cache locally
  getSymbols("INDPRO", src = "FRED")
  save(INDPRO, file = file.path(getwd(), "Data", "INDPRO"))
} else {
  # Load from local cache to avoid repeated API calls
  load(file = file.path(getwd(), "Data", "INDPRO"))
}

tail(INDPRO)


# ────────────────────────────────────────────────────────────────
# 4.2 Sample Selection and Transformations
# ────────────────────────────────────────────────────────────────

start_year <- 1962
end_year   <- 3000

# Log-transform to stabilize variance (avoid non-stationarity due to drifting 
# scale)
y      <- as.double(log(INDPRO[paste0(start_year, "/", end_year)]))
y_xts  <- log(INDPRO[paste0(start_year, "/", end_year)])


# ────────────────────────────────────────────────────────────────
# 4.3 Exploratory Plots: Raw Data, Log-Levels, and First Differences
# ────────────────────────────────────────────────────────────────
par(mfrow = c(2, 2))

# Panel 1: Raw index in original units
plot(as.double(INDPRO), main = "INDPRO (levels)", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = "black", lwd = 1)
axis(1, at = 1:length(INDPRO), labels = index(INDPRO))
axis(2); box()

# Panel 2: Log-transformed index
plot(as.double(y_xts), main = "Log-INDPRO", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = "black", lwd = 1)
axis(1, at = 1:length(y_xts), labels = index(y_xts))
axis(2); box()

# Panel 3: First differences of log-INDPRO (approximate monthly growth rates)
plot(as.double(diff(y_xts)), main = "Diff-log INDPRO (growth rate)", axes = FALSE,
     type = "l", xlab = "", ylab = "", col = "black", lwd = 1)
abline(h = 0)
axis(1, at = 1:length(diff(y_xts)), labels = index(diff(y_xts)))
axis(2); box()

# Panel 4: ACF of first differences — persistent positive autocorrelation
# suggests AR(1)-like dynamics, confirming INDPRO is smoother than a random walk
acf(na.exclude(diff(y_xts)), main = "ACF of diff-log INDPRO")

# Strip xts attributes: plain numeric vector required for filter()
x <- as.double(y_xts)


# ─────────────────────────────────────────────────────────────────────────────
# 4.4 Apply I-SSA Smoother to INDPRO and Evaluate Performance
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 4.4.1 Apply Filters to Data
# ─────────────────────────────────────────────────────────────────────────────
# sides = 1: causal (one-sided) convolution — used for real-time nowcast filters.
# sides = 2: acausal (two-sided) convolution — used for the benchmark HP smoother.
y_ssa    <- filter(x, b_x,      sides = 1)  # I-SSA nowcast smoother
y_hp_one <- filter(x, hp_trend, sides = 1)  # Classical one-sided HP nowcast (HP-C)
y_hp_two <- filter(x, hp_two,   sides = 2)  # Two-sided HP smoother (acausal benchmark)

# The nowcast target is the contemporaneous log-level x_t (zero-lag, delta = 0)
target <- x

# Visual inspection over full sample:
# I-SSA is smoother than the one-sided HP and shows less lag than the two-sided HP
par(mfrow = c(1, 1))
mplot <- cbind(target, y_ssa, y_hp_one, y_hp_two)
ts.plot(mplot, col = c("black", "blue", "red", "violet"))
legend("topleft",
       legend = c("Target (log-INDPRO)", "I-SSA", "HP one-sided", "HP two-sided"),
       col    = c("black", "blue", "red", "violet"), lty = 1)

# Zoom into observations starting in 2000 (discard remote past to reduce 
# non-stationarity in earlier data)
# Search for 2000 observation
year_2000<-which(index(y_xts)>"2000-01-01")[1]
par(mfrow = c(1, 1))
mplot <- cbind(target, y_ssa, y_hp_one, y_hp_two)[year_2000:length(target), ]
ts.plot(mplot, col = c("black", "blue", "red", "violet"))
mtext("INDPRO",line=-1)
mtext("HP two sided",col="violet",line=-2)
mtext("HP one sided",col="red",line=-3)
mtext("I-SSA",col="blue",line=-4)
# I-SSA tracks the index substantially more closely than the one-sided HP,
# while maintaining a comparable degree of smoothness.


# ─────────────────────────────────────────────────────────────────────────────
# 4.4.2 Tracking Accuracy: Mean Squared Error (MSE)
# ─────────────────────────────────────────────────────────────────────────────
# MSE measures how closely each filter tracks the target log-level on average.
# Note: sign accuracy and target correlation are not meaningful for non-stationary
# series and are therefore omitted here.

# Full-sample MSE
mse_ssa_smooth    <- mean((target - y_ssa)^2,    na.rm = TRUE)
mse_hp_one_smooth <- mean((target - y_hp_one)^2, na.rm = TRUE)
mse_hp_two_smooth <- mean((target - y_hp_two)^2, na.rm = TRUE)

mse_hp_one_smooth  # One-sided HP MSE (full sample)
mse_hp_two_smooth  # Two-sided HP MSE (full sample)
mse_ssa_smooth     # I-SSA MSE (full sample)
# I-SSA outperforms both HP designs.

# Theoretical MSE benchmark under the cointegration constraint:
# scales the normalised MSE from the I-SSA design by the empirical variance
# of the first differences (since the series is non-stationary).
# Empirical and expected values differ due to non-stationarity in long sample
# and the fact that INDPRO does not conform to a random-walk
bk_obj$mse_yz * var(diff(target), na.rm = TRUE)

#-------------------------
# Recompute MSE based on trimmed-sample MSE (observations from 2000 onward):
# Removes the remote past, which introduces additional non-stationarity
# and inflates MSE due to filter warm-up / initialisation effects.
mse_ssa_smooth    <- mean((target - y_ssa)[year_2000:length(target)]^2,    na.rm = TRUE)
mse_hp_one_smooth <- mean((target - y_hp_one)[year_2000:length(target)]^2, na.rm = TRUE)
mse_hp_two_smooth <- mean((target - y_hp_two)[year_2000:length(target)]^2, na.rm = TRUE)

mse_hp_one_smooth  # One-sided HP MSE (trimmed sample)
mse_hp_two_smooth  # Two-sided HP MSE (trimmed sample)
mse_ssa_smooth     # I-SSA MSE (trimmed sample)
# After trimming, the sample MSE of I-SSA aligns closely with the theoretical value.
# Theoretical MSE over trimmed sample
bk_obj$mse_yz * var(diff(target[year_2000:length(target)]), na.rm = TRUE)

# ─────────────────────────────────────────────────────────────────────────────
# 4.4.3 Smoothness Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

# 4.4.3.1 Holding Time (HT)
# ─────────────────────────────────────────────────────────────────────────────
# HT counts the average number of periods between consecutive zero-crossings
# of the first-differenced filter output. A larger HT indicates a smoother,
# less oscillatory signal.

# a. Full-sample HT on raw (non-centred) first differences
# ─────────────────────────────────────────────────────────
ht1                                        # Design target HT (= one-sided HP holding time)
compute_empirical_ht_func(diff(y_ssa))     # Empirical HT: I-SSA
compute_empirical_ht_func(diff(y_hp_two))  # Empirical HT: two-sided HP (largest by construction)
compute_empirical_ht_func(diff(y_hp_one))  # Empirical HT: one-sided HP (should be close to ht1)

# Issue: a non-zero mean in the first-differenced series (due to a trending
# level) suppresses zero-crossings, inflating HT estimates. Centering the
# series before computing HT is therefore necessary.

# b. Full-sample HT on centred first differences
# ─────────────────────────────────────────────────────────
ht1                                                   # Design target HT
compute_empirical_ht_func(scale(diff(y_ssa)))         # HT: I-SSA (centred)
compute_empirical_ht_func(scale(diff(y_hp_two)))      # HT: two-sided HP (centred)
compute_empirical_ht_func(scale(diff(y_hp_one)))      # HT: one-sided HP (centred)

# Centering reduces but does not fully resolve the discrepancy with ht1.
# Issue: INDPRO exhibits structural non-stationarity across the full sample —
# growth was steep pre-2000 and broadly flat post-2000. Removing a single
# global mean does not adequately account for this regime shift, leaving
# residual drift that continues to distort zero-crossing counts.
# Remedy: restrict the analysis to post-2000 observations, where the trend
# is approximately constant and a single mean adjustment is appropriate.

# c. Post-2000 HT on centred first differences
# ─────────────────────────────────────────────────────────
# Over this sub-sample the drift is roughly stable, so demeaning aligns
# the series with zero-crossing-based HT evaluation.
ht1                                                                              # Design target HT
compute_empirical_ht_func(scale(diff(y_ssa)[year_2000:length(target)]))         # HT: I-SSA (centred, post-2000)
compute_empirical_ht_func(scale(diff(y_hp_two)[year_2000:length(target)]))      # HT: two-sided HP (centred, post-2000)
compute_empirical_ht_func(scale(diff(y_hp_one)[year_2000:length(target)]))      # HT: one-sided HP (centred, post-2000)

# Now I-SSA and one-sided HP match closely (up to finite sample error).
# Post-2000 sample HTs align more closely with the design target ht1.
# The residual discrepancy is attributable to INDPRO being intrinsically
# smoother than a random walk — its positive serial correlation and the
# prolonged swings associated with major recessions naturally produce
# longer intervals between zero-crossings than the random-walk assumption
# underlying the I-SSA design would predict.

# 4.4.3.2 Curvature: Root Mean Squared Second-Order Differences (RMSD2)
# ─────────────────────────────────────────────────────────────────────────────
# Second-order differences approximate the discrete second derivative;
# smaller RMSD2 values indicate a less curved output.
# The two-sided HP minimises curvature by construction (Whittaker-Henderson optimality).
# I-SSA is expected to exhibit larger curvature than the two-sided HP
# but remain broadly comparable to the one-sided HP.

output_mat <- cbind(x, y_ssa, y_hp_one, y_hp_two)

sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif  # RMSD2 for: raw series, I-SSA, one-sided HP, two-sided HP

# Same but post 2000 data
sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat[year_2000:length(target),]), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif  # RMSD2 for: raw series, I-SSA, one-sided HP, two-sided HP


# ─────────────────────────────────────────────────────────────────────────────
# 4.5 Recession Tracking
# ─────────────────────────────────────────────────────────────────────────────

# We plot differences of the smoothers with vertical lines indicating zero-crossings 
# of differences, i.e., turning points (TP) in levels: local max and min of the level tracker.
mplot<-apply(output_mat[year_2000:length(target),-1],2,diff)
colnames(mplot)<-c("I-SSA","HP one-sided","HP two-sided")
rownames(mplot)<-as.character(index(y_xts))[(nrow(y_xts)-nrow(mplot)+1):nrow(y_xts)]
colo<-c("blue","red","violet")
plot(mplot[,1],
     main="Recession tracking", axes=F, type="l", xlab="", ylab="",
     col=colo[1], lwd=1,
     ylim=c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col=colo[1], line=-1)

for (i in 1:ncol(mplot))
{
  lines(mplot[,i], col=colo[i], lwd=1, lty=1)
  mtext(colnames(mplot)[i], col=colo[i], line=-i)
  abline(v=which(mplot[1:(nrow(mplot)-1),i]*mplot[2:(nrow(mplot)),i] <0),col=colo[i])
  
}
abline(h=0)
axis(1, at=c(1, 4*1:(nrow(mplot)/4)),
     labels=rownames(mplot)[c(1, 4*1:(nrow(mplot)/4))])
axis(2)
box()

# I-SSA reacts faster than the one-sided HP at cyclical turning points and
# crisis episodes, but this timeliness comes at a cost: I-SSA also generates
# spurious downturn signals during prolonged expansions. This is partly
# attributable to the cyclical component embedded in the I-SSA filter design
# (see the discussion in Tutorial 8).
#
# As documented in Tutorial 8, I-SSA produces a higher rate of turning-point
# (TP) signals than the one-sided HP. To obtain a fairer comparison, we
# constrain I-SSA to match the TP rate of the one-sided HP, penalising
# excessive signal variability. This will require increased smoothness, 
# potentially harming MSE performances and speed (advancement).
#
# The approach taken here differs from Exercise 2 of Tutorial 8 in terms of
# the integration level at which each component operates:
#
# - HP filtering is applied to non-stationary log-levels of INDPRO.
# - Turning points (TPs) are defined on the stationary first differences
#   of the filter output.
# - The holding-time (HT) constraint is likewise imposed on stationary
#   first differences.
#
# In contrast, Tutorial 8 Exercise 2 operated one integration level lower:
# HP was applied directly to a stationary series (white noise), and the HT
# constraint was imposed on the non-invertible first differences of that
# white-noise input — a fundamentally different setting that does not carry
# over to the non-stationary, levels-based framework considered here.

# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways
# ─────────────────────────────────────────────────────────────────────────────
# Exercise 4 extends Exercise 3 from a synthetic random-walk input to the
# empirical US Industrial Production Index (INDPRO).
#
# 1. The I-SSA filter was optimised under a random-walk (white-noise differences)
#    assumption, which does not fully capture the positive serial correlation
#    observed in differenced INDPRO. The reported MSE gains are therefore
#    conservative: re-fitting I-SSA to the observed dependence structure of the
#    index would likely yield further performance improvements.
#
# 2. Despite this model mismatch, I-SSA strongly outperforms the classical
#    one-sided HP nowcast in terms of MSE tracking of the log-index, while
#    maintaining similar smoothness in terms of HT in differences. The gain
#    is driven by improved timeliness (a left-shift relative to the one-sided
#    HP) and more accurate tracking of dynamic swings at business-cycle peaks 
#    and troughs.
#
# 3. The combination of reduced lag, superior MSE performance, and robust
#    tracking of dynamic swings makes I-SSA a compelling, data-driven
#    alternative to the classical one-sided HP smoother for real-time
#    macroeconomic monitoring — even when the underlying model is only an
#    approximation of the true data-generating process.
#
# 4. However, real-time recession tracking by I-SSA suffers from more spurious
#    alarms (false TPs) during longer expansions. 
#
# To address the last point, we now constrain the HT of I-SSA to match the 
# frequency of TPs of the one-sided HP.
# ─────────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────
# Exercise 5 Replicate TP-frequency of HP by SSA
# ─────────────────────────────────────────────────────────────────────────────
# This is once again unusal because we use I-SSA for series that are stationary.
# 1. Define HP in diffs: HT in diffs = TP rate on level
# 2. Specify I-SSA that targets cumsum(x_t)=eps_t on levels and imposes 
#     HT on differences x_t=eps_t-eps_{t-1}
# In contrast to exercise 2, the resulting I-SSA replicates TP-rate without 
#   additional cumsum, is stationary and tracks eps_t optimally.










# ══════════════════════════════════════════════════════════════════════════════
# Exercise 5: M-SSA Smoothing
# ══════════════════════════════════════════════════════════════════════════════

