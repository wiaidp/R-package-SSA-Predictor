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




















# ══════════════════════════════════════════════════════════════════════════════
# Exercise 4: I-SSA Smoothing for IP
# ══════════════════════════════════════════════════════════════════════════════
# Similar to tutorial 6 but we target I_t instead of HP(I_t)

# ─────────────────────────────────────────────────────────────────────────────
# Exercise 2.1 Create exercise (random-walk???)
# ─────────────────────────────────────────────────────────────────────────────

# We assume white noise after first differences: random-walk
a1 <- 0.
b1 <- 0
# ────────────────────────────────────────────────────────────────
# Model setup for maximal monotone predictor
# ────────────────────────────────────────────────────────────────
# Cointegration constraint ensures finite MSE for integrated processes

# Assume ARIMA(1,1,0) for log-INDPRO (based on ACF)
# Parameters fixed to avoid instability from extreme (COVID) observations
a1 <- 0.3
b1 <- 0
# This alignes with pre-pandemic estimate of AR(1)
p=1;q=0
arima_obj<-arima(diff(y_xts)["/2020"],order=c(p,0,q))
arima_obj
# Diagnostics are not perfect but ACF of residuals are close to zero
tsdiag(arima_obj)
# Need vola estimate for later calibration
sigma_ip<-sqrt(arima_obj$sigma2)

# Filter design parameters
L         <- 101
lambda_hp <- 14400

# Compute system matrices and filters
filter_obj <- compute_system_filters_func(L, lambda_hp, a1, b1)

B            <- filter_obj$B            # Cointegration matrix, see cited literature
M            <- filter_obj$M            # Autocovariance generator
gamma_tilde  <- filter_obj$gamma_tilde  # Transformed (HP) target
gamma_mse    <- filter_obj$gamma_mse    # MSE-optimal filter (one-sided estimate of HP target)
Xi_tilde     <- filter_obj$Xi_tilde     # Convolution operator (see section 5.3: Wold-decomposition of first differences convolved with integration operator)
Sigma        <- filter_obj$Sigma        # Integration operator (see section 5.3)
Delta        <- filter_obj$Delta        # Differencing operator
Xi           <- filter_obj$Xi           # Wold MA representation in matrix notation, see equation 22 in Wildi 2026a
hp_two       <- filter_obj$hp_two       # Two-sided HP target
hp_trend     <- filter_obj$hp_trend     # Classic one-sided HP (HP-C): benchmark for I-SSA customization

par(mfrow=c(1,1))
ts.plot(cbind(gamma_tilde,gamma_mse),col=c("black","brown"))
mtext("Optimal MSE filter applied to data in levels",line=-1,col="brown")
mtext("Serves as target in constrained optimization",line=-2)

# Plot Wold decomposition of differenced process: first column in matrix Xi
ts.plot(Xi[,1],main="Wold decomposition of ARMA(1,1)")




# ────────────────────────────────────────────────────────────────
# Holding-Time (HT) Constraint Calibration
# ────────────────────────────────────────────────────────────────
# The HT constraint is defined on first differences (see eq. 29, Wildi 2026a)

# Derivation of the HT of the first-differenced HP predictor:
#
#   Xi       : convolution matrix (eq. 22, Wildi 2026a)
#   Xi_tilde : Xi composed with Sigma (the integration operator); see section 5.3
#
#   Xi_tilde %*% hp_trend provides a finite MA representation of the HP predictor in levels.
#   Although this representation is not the predictor itself (which is non-stationary),
#   it is suitable for differencing: finite and infinite MA filters behave equivalently
#   under first differencing, so the stationary differences are well-defined.
#
#   Applying the differencing operator Delta yields:
#       Delta %*% Xi_tilde %*% hp_trend = Xi %*% hp_trend
#   (both expressions are algebraically identical)
#
#   Therefore, Xi %*% hp_trend is the appropriate input for computing the holding time
#   of the differenced HP trend (i.e., the classic concurrent trend nowcast in levels).

HT_HP_obj<-compute_holding_time_func(Xi %*% hp_trend)

# HT: expected duration (in months) between consecutive mean-crossings of the
#   filtered process, where the filter is described by the MA inversion Xi
#   (in this application: an AR(1) process)
HT_HP_obj$ht

# First-order autocorrelation (rho1): stands in a one-to-one (bijective) correspondence
#   with the HT above; see eq. 18, Wildi (2026a) for the analytical relationship
HT_HP_obj$rho_ff1

# Since HT and rho1 are in bijective correspondence, imposing rho1 is equivalent
#   to imposing the HT constraint. We therefore use rho1 to enforce the HT
#   constraint in I-SSA (see eq. 18, Wildi 2026a).
rho1 <- as.double(rho_hp_concurrent<-HT_HP_obj$rho_ff1)

# Interpretation of the I-SSA constraint:
#   I-SSA is constrained to match the holding time of the one-sided (causal) HP filter,
#   as measured in stationary first differences. Concretely:
#     1. Apply the causal HP filter to the data in levels.
#     2. Compute the first differences of the filtered series.
#     3. Compute the mean duration between consecutive sign changes of these differences.
#   This mean duration is the target HT (stored in HT_HP_obj$ht), and rho1 is its
#   bijective counterpart used to impose the constraint in I-SSA.
#   Two remarks on HT_HP_obj$ht:
#     - It is derived from a theoretical formula and is therefore exact only if
#       the assumed model (Xi) is the true data-generating process.
#     - We will compare this theoretical value against the empirical estimate
#       obtained by applying the filters directly to the data, as a model diagnostic.
#
#   By the optimality properties of I-SSA, a trend nowcast that replicates the HT of HP
#   (via the rho1 constraint) is expected to outperform the classic HP filter in terms
#   of MSE on non-stationary levels. Moreover, no other linear predictor subject to the
#   same HT constraint can improve upon I-SSA if the model (Xi) is correctly specified.

# The following code verifies optimality empirically and quantifies the MSE gain.



# ────────────────────────────────────────────────────────────────
# Reference: HT of differenced MSE-optimal predictor (MSE-optimal in levels)
# ────────────────────────────────────────────────────────────────
# Using the same derivation logic:
#   Xi %*% gamma_mse characterizes the differenced representation
#   of the MSE-optimal predictor in levels.

rho_mse <- as.double(compute_holding_time_func(Xi %*% gamma_mse)$rho_ff1)

# Typically, rho_mse is small → frequent zero-crossings
# This reflects the higher noise of MSE-optimal predictors in levels.
# MSE optimality trades off smoothness for timeliness:
# such predictors are generally more reactive but noisier.
rho_mse
# Compare to HP: the latter is much smoother
rho1
# The plots below will illustrate the smoothness differences and their impact 
#   on recession signaling


# ────────────────────────────────────────────────────────────────
# Motivation for I-SSA
# ────────────────────────────────────────────────────────────────
# - MSE-optimal level predictors are efficient but very noisy
# - They generate excessive sign changes in first differences
#   → unreliable signals of slowdowns/accelerations
#
# - One-sided HP (applied to levels) is smoother:
#   → fewer crossings in differences
#   → better tracking of business-cycle turning points
#   → but MSE performances of the trend nowcast are inferior 
#   
#
# - I-SSA combines both advantages:
#   → replicates HP smoothness (via HT constraint): generate fewer spurious false alarms in first differences
#   → improves tracking of the two-sided HP in levels (lower MSE)
#
# Result:
# A predictor that closely tracks the level while producing
# sparse and informative sign changes in first differences.


# ────────────────────────────────────────────────────────────────
# Compute I-SSA solution (Wildi, 2026a, Sections 5.3–5.4)
# ────────────────────────────────────────────────────────────────
# Use numerical optimization (optim) to determine the optimal
# Lagrange multiplier λ ensuring compliance with the HT constraint.
# Initialization at λ = 0 corresponds to the MSE benchmark.

# Do not confuse this lambda with lambda_hp (the lambda regularization parameter of HP)
lambda <- 0

# Classical numerical optimization procedure optim in R
opt_obj <- optim(
  lambda,
  b_optim,
  lambda,
  gamma_mse,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  gamma_tilde,
  rho1
)

# Optimal Lagrange multiplier
lambda_opt <- opt_obj$par

# Compute I(1) cointegrated I-SSA solution based on lambda_opt
bk_obj <- bk_int_func(
  lambda_opt,
  gamma_mse,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  gamma_tilde
)

# Diagnostics
bk_obj$rho_yy   # should match rho1 (HT constraint): this verifies convergence of the optimization (if not: increase the number of iterations)
rho1
bk_obj$rho_yz   # correlation with target
bk_obj$mse_yz*sigma_ip^2   # MSE with respect to MSE-optimal predictor (rescaled with residual variance from AR(1) model)
# This MSE vanishes if I-SSA replicates exactly the MSE predictor 
# (set lambda_opt<-0 and verify that bk_obj$mse_yz=0)
# I-SSA optimization principle:
# Match the classic MSE predictor as close as possible under the HT constraint.


# Extract filters
b_x   <- bk_obj$b_x     # applied to data
b_eps <- bk_obj$b_eps   # applied to innovations

# If Xi = I, then b_x = b_eps
par(mfrow = c(1, 2))
ts.plot(b_eps, main = "b applied to epsilon")
ts.plot(b_x,   main = "b applied to INDPRO")

# Constraint checks
sum(b_x) - sum(gamma_mse)     # cointegration (≈ 0): ensures a finite MSE on non-stationary levels
bk_obj$rho_yy - rho1          # HT constraint (≈ 0). If this is not small, the numerical optimization did not converge


# ────────────────────────────────────────────────────────────────
# Plot filters
# ────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "green", "blue", "red")

mplot <- cbind(
  hp_two,
  c(gamma_mse, rep(0, L - 1)),
  c(b_x,       rep(0, L - 1)),
  c(hp_trend,  rep(0, L - 1))
)
colnames(mplot) <- c("HP-two", "MSE", "I-SSA", "HP-C")

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

# Zoom on first 30 lags
mplot <- mplot[1:30, ]

plot(mplot[, 1], axes = FALSE, type = "l", col = colo[1], lwd = 1,
     ylim = c(min(mplot[, "HP-C"]), max(mplot[, "I-SSA"])))
abline(h = 0)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}

axis(1, at = 1:nrow(mplot), labels = 0:(nrow(mplot) - 1))
axis(2); box()





# ─────────────────────────────────────────────────────────────────────────────
# Exercise 4.2 Replicate TP-frequency of HP by SSA
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

