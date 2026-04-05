# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 9: I-SSA and M-SSA SMOOTHING
# Introducing the I-SSA Trend
# ══════════════════════════════════════════════════════════════════════════════

# This tutorial extends the I-SSA smoothing framework in two directions:
# from univariate stationary processes (Tutorial 8) to non-stationary
# processes (I-SSA: Exercises 1–4), and further to multivariate smoothing
# applications (M-SSA: Exercise 5). For a comprehensive discussion of the
# distinction between smoothing and prediction, the reader is referred
# to Tutorial 8.

# We shall introduce a new trend definition, the I-SSA trend, based on I-SSA 
# smoothing.

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
# ROC curve calculation
library(pROC)
# NBER recession datings for the US
library(tis)


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
# ROC plot
source(paste(getwd(), "/R/ROCplots.r", sep = ""))


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
# HP Lambda Selection:
# Lambda is the penalty weight assigned to the curvature term in the
# Whittaker-Henderson (WH/HP) graduation criterion (see Tutorials 2.0 and 8).
# A larger lambda enforces greater smoothness at the cost of reduced fidelity
# to the observed series; a smaller lambda allows the trend to track the data
# more closely at the cost of increased roughness.
#
# The conventional value of 14,400 is the standard choice for monthly time
# series, calibrated to yield a trend extraction broadly equivalent to that
# of the HP filter applied to quarterly data with lambda = 1,600.
# Exercise 5 will apply the resulting HP and I-SSA smoothers to the monthly
# US Industrial Production Index (INDPRO).
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

# Target in first differences: apply the summation operator to the level
# target filter. This yields a finite-length proxy of the effective
# first-difference target used in optimisation. The proxy converges to the
# true target as filter length L grows, because MA coefficients decay
# sufficiently fast under the cointegration constraint (see Section 5.3 in
# Wildi 2026a for details).
target_filter_diff <- cumsum(target_filter)

# The following code in curly brackets is shown for completeness.
if (F)
{
# The following generalises the cumsum above to cases where the Wold
# decomposition Xi is no longer white noise.
# - In the random-walk case, both expressions are equivalent.
# - In all other cases, the expression below applies; using cumsum alone
#   would yield an incorrect result.
# - Background: we need a finite MA-inversion of the target on levels: this 
#   is obtained by the convolution of the Wold-decomposition with the 
#   integrator. The finite-length convolution can be obtained through the 
#   matrix product Xi_tilde %*% target_filter where Xi_tilde <- (Sigma) %*% Xi,
#   see Wildi 2026a, section 5.3.
#
# Notes:
# - Sigma is the finite-length summation (integration) operator that maps
#   first differences back to levels.
# - A finite-length integrator suffices here because:
#     a) target_filter_diff is used solely for optimisation, not for
#        effective smoothing of the data.
#     b) For optimisation purposes, the relevant MA-inversion decays
#        rapidly to zero under the cointegration constraint (hence a 
#        finite-length target is sufficient).
#     c) The cointegration constraint is invisible here: it is implemented 
#        directly and automatically within the function bk_int_func(). The 
#        constraint warrants a finite MSE even though non-stationary levels 
#        may diverge asymptotically.
  Xi_tilde <- (Sigma) %*% Xi
  target_filter_diff <- Xi_tilde %*% target_filter
}

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

# Zoom in on a 2*width-lag window centred on -delta
width<-80
mplot_short <- mplot[(-delta - width):(-delta + width), ]

plot(mplot_short[, 1], axes = FALSE, type = "l", col = colo[1], lwd = 1,
     ylim = c(min(mplot_short), max(mplot_short[, "I-SSA"])), ylab = "", xlab = "Lags")
abline(h = 0)
for (i in 1:ncol(mplot_short)) {
  lines(mplot_short[, i], col = colo[i])
  mtext(colnames(mplot_short)[i], line = -i, col = colo[i])
}
axis(1, at = 1:(2*width+1), labels = (-delta - width):(-delta + width))
axis(2); box()

# Unlike the SSA smoother in Tutorial 8 — which targets stationary series —
# I-SSA adopts a distinct shape when applied to a non-stationary input
# (random walk):
#
# - The central bell-shaped mass of the I-SSA filter is more pronounced than
#   that of the HP filter, concentrating more weight than HP on observations 
#   close to the nowcast point (intuitively: better tracking).
# - The absorbing side-lobes flanking the central mass are also stronger
#   than those of HP.
#
# We now compute and compare the tracking accuracy and smoothness of both
# smoothers.  

# ─────────────────────────────────────────────────────────────────────────────
# 1.4  Check Performances
# ─────────────────────────────────────────────────────────────────────────────
# The key hypothesis is as follows:
#
# For an identical HT in first differences — equivalently, for
# an identical rate of turning points (TPs) on levels — I-SSA tracks the
# target x_{t+delta} more closely than HP, as measured by MSE.

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
mse_ssa_smooth <- mean((target - y_ssa)^2, na.rm = TRUE)
mse_hp_smooth  <- mean((target - y_hp)^2,  na.rm = TRUE)

# Outcome: I-SSA achieves achieves a reduction in MSE of over 50% relative to HP.
mse_hp_smooth   # HP  MSE (sample)
mse_ssa_smooth  # I-SSA MSE (sample)

# The sample MSE of I-SSA should converge to the theoretical value below.
# Any discrepancy reflects Monte Carlo sampling variability.
bk_obj$mse_yz   # Theoretical MSE under the cointegration constraint

# Notes: 
#   - Target correlation and sign accuracy are not meaningful criteria for
#     non-stationary time series and are therefore omitted here.
#   - Wildi (2026a) proposes pseudo target correlation and pseudo sign accuracy 
#     based on finite MA-inversions on levels (omitted).

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


# 1.4.3.2  Curvature (Root Mean Squared Second-Order Differences)
# ─────────────────────────────────────────────────────────────────────────────
# Compute square root of mean-squared second order differences of the smoothers
output_mat <- cbind(x, y_ssa, y_hp)

sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif

# Outcome: 
# HP minimises curvature by construction (WH optimality) and therefore
# exhibits slightly smaller RMSD2 than I-SSA under the same HT constraint.
# In contrast to Tutorial 8, the difference in curvature is less pronounced
# here, reflecting the smoother weight profile of I-SSA on non-stationary data.
  

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
#
# ─────────────────────────────────────────────────────────────────────────────
# DISCUSSION
# ─────────────────────────────────────────────────────────────────────────────
#
# The following analysis  extends and complements the discussion on smoothing 
# provided in tutorial 8.
# 
# Optimality:
# ─────────────────────────────────────────────────────────────────────────────
# The I-SSA smoother shape (pattern) in the above plot is not an arbitrary 
# design choice but the direct consequence of optimising tracking accuracy of 
# x_{t+delta} under the given (HT and cointegration) constraints. 
# In particular, it does not impose an extraneous artificial signature on the 
# data-generating process.
#
# Minimally Invasive (Amorphous) Smoothness Constraint:
# ─────────────────────────────────────────────────────────────────────────────
# The HT constraint is amorphous in the sense that it does not favour any
# particular functional form or shape for the smoothed series (e.g., locally 
# linear, polynomial, or spline-based trends). Instead, it operates indirectly 
# by regulating the first-order autocorrelation structure of the filter output
# in first differences, requiring it to comply with a prescribed memory
# level — and nothing more.
#
# The HT constraint asks that the smoother exhibit sufficiently long memory 
# between direction changes (in levels), without prescribing how that memory 
# should manifest in the shape of the output.
#
# As a result, the HT constraint is minimally invasive: it enforces the
# desired degree of smoothness while leaving the smoother free to determine
# the optimal shape of the trend from the data — introducing no structural
# assumptions beyond those already embedded in the target specification 
# (the identity) and the cointegration constraint (cancellation of the 
# unit root).
#
# Interpretability:
# ─────────────────────────────────────────────────────────────────────────────
# The HT constraint in first differences regulates the rate of turning points
# (TPs) of the smoother in levels:
#
# - TPs mark potentially important events in the evolution of a time series,
#   such as business-cycle peaks and troughs.
#
# - I-SSA anchors TPs to the criterion of optimally tracking the level of
#   the series — a logically compelling and statistically grounded approach
#   that ties signal features directly to the data-generating process.
#
# - As demonstrated in Tutorial 8, for an identical TP rate, TPs derived
#   from minimising curvature (HP) are more evenly spaced in time than those
#   derived from optimal tracking of the series level (I-SSA). This spurious
#   regularity is an artifact of the curvature constraint: by penalising
#   second-order differences, HP imposes a quasi-periodic rhythm on the
#   extracted trend that reflects the filter's structural bias rather than
#   any genuine feature of the underlying data-generating process.
#
# Data-Generating Process (DGP)
# ─────────────────────────────────────────────────────────────────────────────
# A potential — and admittedly unfair — advantage of I-SSA over HP is that
# the I-SSA filter is optimised under knowledge of the true DGP. In the
# exercises above, this DGP was assumed to be a random walk.
#
# We argue, however, that this knowledge is not essential in practice,
# because most macroeconomic time series exhibit dynamics that are broadly
# consistent with a random-walk approximation. The random-walk default is
# therefore a robust and widely applicable baseline for non-stationary
# economic data.
#
# To substantiate the random-walk (RW) default model, we apply the I-SSA filter
# optimised for the RW — without any re-fitting or adjustment — directly to
# the monthly US Industrial Production Index (INDPRO), an important
# real-sector business-cycle indicator (see Exercise 4).
#
# In a Nutshell:
# ─────────────────────────────────────────────────────────────────────────────
# I-SSA rests on a logically consistent and statistically efficient
# design principle: smoothness is encoded as a memory constraint on the
# smoother — one that is deliberately unstructured, imposing no parametric
# form on the underlying signal. Coefficients are then chosen to minimise
# tracking error subject to that constraint alone, free of any artificial
# assumptions about the data-generating process. As a result, turning points
# (TPs) are determined solely by the intrinsic growth dynamics of the series,
# and their timing is not an artefact of externally imposed regularity.
#
#
# Nowcasting:
# ─────────────────────────────────────────────────────────────────────────────
# Exercise 1 focused on symmetric backcasting smoothers. In a nowcasting
# framework — where only past and current observations are available — the
# limitations of HP become more pronounced, further widening the performance
# gap between HP and I-SSA, even under model misspecification.
#
#
#
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

output_mat <- cbind(x, y_ssa, y_hp_one, y_hp_two)

sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat), 2, diff), 2, diff)^2,
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


# 3.3.3.2  Curvature (Root Mean Squared Second-Order Differences)
# ─────────────────────────────────────────────────────────────────────────────
output_mat <- cbind(x, y_ssa, y_hp_one, y_hp_two)
sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif

# Outcome:
# The two-sided HP achieves the smallest curvature by construction (WH optimality).
# I-SSA exhibits slightly larger curvature than the one-sided HP but remains
# comparable (in contrast to bigger differences in tutorial 6).

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

# ─────────────────────────────────────────────────────────────────────────────
# Model Misspecification
# ─────────────────────────────────────────────────────────────────────────────
# A potential — and admittedly unfair — advantage of I-SSA over HP in the above 
# exercises, is that the I-SSA filter is optimised under knowledge of the true 
# DGP, a random walk (RW).
#
# We argue, however, that this knowledge is not essential in practice,
# because many economic time series exhibit dynamics that are broadly
# consistent with a random-walk approximation. The random-walk default is
# therefore a robust and widely applicable baseline for non-stationary
# economic data (assuming absence of seasonal effects which would require 
# specific treatment not analysed here).
#
# To substantiate the random-walk (RW) default model, we now apply the I-SSA 
# smoother of the above exercise 3 — optimised for the RW — without any 
# re-fitting or adjustment directly to the monthly US Industrial Production 
# Index (INDPRO), an important business-cycle indicator.
#
#
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
nrow(output_mat)
dates<-as.character(index(y_xts))
rownames(output_mat)<-dates
colnames(output_mat)<-c("INDPRO","I-SSA","HP one-sided","HP two-sided")
tail(output_mat)

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
# 4.5 NBER Recession Detection
# ─────────────────────────────────────────────────────────────────────────────
# We evaluate the ability of each nowcast to detect NBER-dated recessions using
# Receiver Operating Characteristic (ROC) curves and the Area Under the Curve
# (AUC). First differences of the filter outputs serve as recession signals.
# AUC summarises overall detection skill — a value of 1 indicates perfect
# discrimination; 0.5 indicates no skill beyond random guessing.
#
# ─────────────────────────────────────────────────────────────────────────────
# 4.5.1 NBER dating
# ─────────────────────────────────────────────────────────────────────────────

# Retrieve official NBER business-cycle turning-point dates
tail(nberDates())

# Construct a binary recession indicator aligned to the filter output grid:
# 1 = NBER recession month, 0 = expansion month
NBER_recessions <- rep(0, nrow(output_mat))
names(NBER_recessions) <- rownames(output_mat)

# Assign recession episodes based on official NBER peak-to-trough dates
recession_dates <- c(
  which(names(NBER_recessions) >= "1970-01-01" & names(NBER_recessions) <= "1970-11-30"),  # 1969–70
  which(names(NBER_recessions) >= "1973-12-01" & names(NBER_recessions) <= "1975-03-31"),  # 1973–75
  which(names(NBER_recessions) >= "1980-02-01" & names(NBER_recessions) <= "1980-07-31"),  # 1980
  which(names(NBER_recessions) >= "1981-08-01" & names(NBER_recessions) <= "1982-11-30"),  # 1981–82
  which(names(NBER_recessions) >= "1990-08-01" & names(NBER_recessions) <= "1991-03-31"),  # 1990–91
  which(names(NBER_recessions) >= "2001-04-01" & names(NBER_recessions) <= "2001-11-30"),  # 2001
  which(names(NBER_recessions) >= "2008-01-01" & names(NBER_recessions) <= "2009-06-30"),  # 2008–09 (GFC)
  which(names(NBER_recessions) >= "2020-01-01" & names(NBER_recessions) <= "2020-04-30")   # 2020 (COVID-19)
)
NBER_recessions[recession_dates] <- 1

# Quick visual check of the recession indicator
plot(NBER_recessions,
     main = "NBER US Recession Dating", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = "black", lwd = 1)
axis(1, at    = c(1, 4 * 1:(length(NBER_recessions) / 4)),
     labels = names(NBER_recessions)[c(1, 4 * 1:(length(NBER_recessions) / 4))])
axis(2); box()


# ─────────────────────────────────────────────────────────────────────────────
# 4.5.2 Compare NBER Dating with Differenced Smoothers 
# ─────────────────────────────────────────────────────────────────────────────

# The recession indicator serves as the binary classification target
target <- NBER_recessions

# Build the ROC data matrix: binary target + first-differenced filter outputs.
# First differences convert non-stationary levels to stationary growth signals
# that can be thresholded to produce recession/expansion calls.
ROC_data_all <- cbind(target, apply(output_mat[, c("I-SSA", "HP one-sided")], 2, diff))
rownames(ROC_data_all) <- rownames(apply(output_mat, 2, diff))
colnames(ROC_data_all) <- c("Target", colnames(output_mat[, c("I-SSA", "HP one-sided")]))

# Restrict to post-2000 sub-sample for comparability with HT analysis and to
# avoid the high-growth pre-2000 period, which has a different drift regime
# (mitigate non-stationarity)
ROC_data <- ROC_data_all[year_2000:nrow(ROC_data_all), ]
head(ROC_data)
tail(ROC_data)
# Compare NBER datings with differenced nowcast smoothers: 
#   Both nowcast smoothers track official NBER datings well
ts.plot(scale(ROC_data),col=c("black","blue","red"),main="Recessions vs. differenced smoothers")
mtext("I-SSA",col="blue",line=-1)
mtext("HP one-sided",col="red",line=-2)



# ─────────────────────────────────────────────────────────────────────────────
# 4.5.3 ROC AnalysisC
# ─────────────────────────────────────────────────────────────────────────────
smoothROC <- TRUE   # Apply smoothing to ROC curves for cleaner visualisation
showROC   <- TRUE   # Display ROC plots
lbls      <- "Hit"  # Axis label: hit rate (sensitivity) vs. false alarm rate
lg_cex    <- 0.5    # Legend text size

par(mfrow = c(1, 1))


# Compute ROC curves and AUCs
# Transform into a data frame (requested for ROC plot)
ROC_data <- as.data.frame(ROC_data)

# Plot ROC curves and compute AUC for each filter
# As expected (from previous plot), both nowcast smoothers track recession 
# datings well in first differences.
showLegend <- TRUE
AUC <- ROCplots(ROC_data, showROC, main = "ROC Analysis: Recession Detection", lbls = lbls,
                smoothROC, colours = NULL, lwd = 2,
                showLegend, lg_cex = 1, lg_ncol = 1)

# Summarise AUC values
AUC_table <- AUC$AUC
names(AUC_table) <- colnames(ROC_data[, c("I-SSA", "HP one-sided")])

# AUC results: higher values indicate better recession discrimination
AUC_table


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
#    maintaining comparable smoothness — as measured either by the HT of
#    first differences or, equivalently, by an equal frequency of turning
#    points (TPs) on levels. The MSE gain is driven by two complementary
#    factors: improved timeliness (a left-shift of the filter output relative
#    to the one-sided HP) and more accurate tracking of dynamic swings at
#    business-cycle peaks and troughs.
#
# 3. The combination of reduced lag, superior MSE performance, and robust
#    tracking of dynamic swings makes I-SSA a compelling, data-driven
#    alternative to the classical one-sided HP smoother for real-time
#    macroeconomic monitoring. Unlike the HP filter — whose smoothness
#    criterion explicitly penalises curvature (second-order differences) —
#    I-SSA is amorphous in the sense that it does not impose aa extraneous 
#    structural form on the extracted trend, allowing it to adapt more
#    flexibly to the shape of the underlying signal by tracking non-stationary 
#    levels. Importantly, these advantages persist even when the assumed model 
#    (random walk) is only an approximation of the true data-generating process.
#
# 4. A ROC analysis confirms that both nowcast smoothers track official NBER 
#    datings fairly closely. 

# ─────────────────────────────────────────────────────────────────────────────
# A Note on Model Misspecification in I-SSA
# ─────────────────────────────────────────────────────────────────────────────
# I-SSA permits the specification of either a theoretical or empirically
# fitted model via the Wold decomposition Xi, directly shaping the
# interpretability and optimality of the resulting smoother.
#
# In Exercise 4, INDPRO departs from the posited (default) random-walk 
# assumption:
# the indicator is smoother and more regular than a random walk, exhibiting
# positive serial correlation in growth rates. Nevertheless, the practical
# impact of this misspecification remains limited. The empirical turning-point
# (TP) rate differs modestly from the rate implied by the random-walk
# assumption, introducing a mild positive bias — where a positive bias
# indicates that the data is smoother than the assumed DGP.
#
# Crucially, this TP-rate bias affects HP and I-SSA symmetrically: it is a
# shared consequence of calibrating smoothers under one DGP and applying them
# to data generated by another. Despite this common bias, I-SSA markedly
# outperforms HP in terms of MSE at an equivalent TP rate, confirming that
# the performance advantage of I-SSA is robust to moderate departures from
# the random-walk default.

# ─────────────────────────────────────────────────────────────────────────────
# Default Models in I-SSA
# ─────────────────────────────────────────────────────────────────────────────
# Two natural default settings can be distinguished based on the dynamics
# of the data:
#
# - Non-stationary (trending) series: the I-SSA smoother targets the TREND LEVEL
#   The random walk is recommended as the default DGP assumption, as applied 
#   throughout the above exercises.
#
# - Stationary series: when stationarity is achieved by differencing a
#   non-stationary series, the smoother targets TREND GROWTH (i.e., the
#   rate of change of the underlying level series). The white-noise
#   hypothesis is recommended as the default DGP assumption in this
#   setting, as illustrated in Tutorial 8.
#
# - Notably, the optimal growth estimate is not obtained by differencing the
#   optimal level estimate; conversely, the optimal level estimate is not
#   recovered by integrating (cumulating) the optimal growth estimate. Each
#   target requires its own dedicated smoother specification in order
#   to achieve efficiency and interpretability.
#
# In both cases, the default model may be replaced by one estimated directly
# from the data. Doing so serves two purposes: it can improve filter
# performance by more accurately capturing the true dependence structure of
# the series, and it ensures that sample-based diagnostics — including
# empirical MSE, target correlation, sign accuracy, and the HT — 
# converge towards their theoretical expected values,
# thereby guaranteeing the unbiasedness of sample performance estimates.

# ─────────────────────────────────────────────────────────────────────────────
# I-SSA Trend and the HT
# ─────────────────────────────────────────────────────────────────────────────
# For a non-stationary series, I-SSA smoothing can be interpreted as an
# estimator of a latent trend at time points t = 1, ..., T within the sample
# (causal target x_1, ..., x_T). The I-SSA trend is implicitly defined by:
#
#   i)  Specifying the desired rate of turning points (TPs), expressed via
#       the holding time (HT) of the smoother's first differences.
#   ii) Tracking the data x_{t+delta} as closely as possible, subject to
#       the specified TP rate.
#
# This formulation imposes minimal structure on the resulting I-SSA trend:
# its shape is determined entirely by the data and the HT constraint,
# without conditioning on artificial assumptions about the underlying
# data-generating process or a particular (idealized) appearance of the 
# smoothed series.

# In this setting, HT serves as a hyperparameter that must be chosen by the
# analyst prior to estimation. It can be set on the basis of:
#
#   - A priori knowledge of the data-generating process (e.g., the expected
#     duration of a business cycle), or
#   - Specific research objectives that motivate a particular TP frequency
#     (e.g., high-frequency trading versus long-term systematic investment).
#
# Alternatively, the HT can be calibrated against a benchmark smoother — such as
# HP, as in the above exercises. In this case, I-SSA replicates the TP rate
# of the benchmark while tracking the underlying series more closely,
# yielding a I-SSA trend that can be viewed as a refined counterpart to the
# benchmark: one with superior data fidelity and a reduced artificial
# imprint on the smoothing outcome. I-SSA can equally be used to deliberately
# lengthen or shorten the HT relative to a chosen benchmark.
#
# Finally, rather than tracking the raw data x_{t+delta} (smoothing), I-SSA can 
# be directed to track an arbitrary (extraneous) target trend, see tutorial 6. 
# Modifying filter characteristics in this way is referred to as customisation 
# (see previous exerrcises). 
# In this case, the resulting I-SSA trend inherits its meaning and 
# interpretation directly from the benchmark, by virtue of optimal tracking. 
# It should be noted, however, that customisation is generally not a 
# smoothing exercise but a prediction exercise.




################################################################################
################################################################################
# Exercise 5 extends the univariate (stationary) SSA smoothing framework to its
# multivariate counterpart, M-SSA, as introduced in Tutorial 7. The sole
# modification relative to Tutorial 7 is the choice of target: here the
# target is the raw observed series x_{t+delta} rather than an acausal
# filter applied to the data.
################################################################################
################################################################################

# ══════════════════════════════════════════════════════════════════════════════
# Exercise 5: M-SSA Smoothing
# Nowcasting with Mild Smoothing
# ══════════════════════════════════════════════════════════════════════════════
# This example is drawn from Wildi (2026b), Section 4.2. It is based on a
# three-dimensional VAR(1) process.

# ─────────────────────────────────────────────────────────────────────────────
# 5.1. Three-Dimensional VAR(1)
# ─────────────────────────────────────────────────────────────────────────────
n <- 3
# Specify a random innovation covariance matrix Sigma and an AR(1) coefficient
# matrix A.
set.seed(1)
Sigma_sqrt <- matrix(rnorm(n * n), ncol = n)
Sigma <- Sigma_sqrt %*% t(Sigma_sqrt)
Sigma
# Sigma is full rank: all eigenvalues are strictly positive.
# If Sigma is rank-deficient, the problem simplifies; see Wildi (2026b).
eigen(Sigma)$values
A <- rbind(c(0.7, 0.4, -0.2), c(-0.6, 0.9, 0.3), c(0.5, 0.2, -0.3))
A
# Check stationarity: all eigenvalues of A must be strictly less than one
# in absolute value.
abs(eigen(A)$values)
# Optional: replace the VAR(1) with a VARMA(1,1) specification.
if (F)
{
  A <- diag(rep(0, n))
  B <- 0.05 * rbind(c(0.5, 0.4, -0.4), c(0.6, -0.9, 0.3), c(0.8, -0.7, -0.8))
}

# ─────────────────────────────────────────────────────────────────────────────
# 5.2. M-SSA Settings
# ─────────────────────────────────────────────────────────────────────────────
L <- 51
# Nowcasting: the target coincides with the current observation (delta = 0).
# A symmetric multivariate backcast will be explored below.
delta <- delta1 <- 0
# Target filter matrix: Gamma is the n x nL identity mapping, where each
# series targets its own contemporaneous value.
gamma_target <- matrix(rep(0, n^2 * L), nrow = n)
gamma_target[1, 1] <- gamma_target[2, 1 + L] <- gamma_target[3, 1 + 2 * L] <- 1
# Structure of gamma_target (n x nL matrix):
#   - Row i specifies the filter coefficients for the i-th target series.
#   - Columns (j-1)*L+1 : j*L of row i hold the coefficients applied to the
#     j-th input series when estimating target i.
#   - In the nowcasting case, all entries are zero except at position
#     (i, (i-1)*L+1), which equals one (Kronecker delta structure).
gamma_target
# Verify the Kronecker delta structure for each series:
i <- 1
gamma_target[i, (i - 1) * L + 1]
i <- 2
gamma_target[i, (i - 1) * L + 1]
i <- 3
gamma_target[i, (i - 1) * L + 1]
# Confirm identity structure along the block diagonal.
cbind(gamma_target[, 1], gamma_target[, L + 1], gamma_target[, 2 * L + 1])
# Note: in Tutorial 7 (M-SSA), gamma_target[i, ((i-1)*L)+1:L] was populated
# with HP filter coefficients, making HP the smoothing target. Here, the
# target is the raw series X_{t+delta} with delta = 0, i.e., a 3-dimensional
# VAR(1) observed contemporaneously.

# ─────────────────────────────────────────────────────────────────────────────
# 5.3. Wold Decomposition
# ─────────────────────────────────────────────────────────────────────────────
# M-SSA requires the Wold decomposition of the process into orthogonal
# innovations epsilon_t. Below, the MA coefficient matrices are computed
# for the VAR(1) specification.

# Method a: Analytic recursion for VAR(1)
B <- diag(rep(0, n))
if (F)
{
  A <- diag(rep(0, n))
  B <- 0.05 * rbind(c(0.5, 0.4, -0.4), c(0.6, -0.9, 0.3), c(0.8, -0.7, -0.8))
}
det(A)
Ak   <- A
Akm1 <- diag(rep(1, n))
xi   <- matrix(nrow = dim(A)[1], ncol = L * dim(A)[1])
xi[, L * (0:(dim(xi)[1] - 1)) + 1] <- diag(rep(1, n))
for (i in 2:L)
{
  # Row k of xi holds the MA weights for the k-th target series.
  # Columns 1:L correspond to the weights on the first noise series
  # eps_{1,t}, eps_{1,t-1}, ...; columns L+1:2L to the second noise
  # series eps_{2,t}, eps_{2,t-1}, ...; and so on.
  # This parametrisation follows the IJFOR paper convention.
  for (j in 1:n)
    xi[, i + (j - 1) * L] <- Ak[, j] + (Akm1 %*% B)[, j]
  Ak   <- Ak %*% A
  Akm1 <- Akm1 %*% A
}

# Method b: General MA inversion for arbitrary VARMA processes.
B    <- NULL
# Use non-orthogonalised innovations (consistent with Sigma).
orth <- F
irf  <- M_MA_inv(A, B, Sigma, L - 1, orth)$irf
# Populate Xi (capital X) from the impulse response function.
Xi <- 0*xi
for (i in 1:n)
{
  Xi[, (i - 1) * L + 1:L] <- irf[(i - 1) * n + 1:n, 1:L]
}
# Verification: both MA inversions should agree to numerical precision.
max(abs(xi - Xi))

# ─────────────────────────────────────────────────────────────────────────────
# 5.4. Holding-Time and M-SSA Default Settings
# ─────────────────────────────────────────────────────────────────────────────
# Specify the desired holding times (HTs) for each of the three series, see Wildi 2026b.
ht_vec <- matrix(c(min(8,  L / 2), min(6, L / 2), min(10, L / 2)), nrow = 1)
# Convert the specified HTs to first-order autocorrelations (rho), as M-SSA
# is parameterised in terms of rho rather than HT directly.
rho0 <- apply(ht_vec, 1, compute_rho_from_ht)[[1]]$rho

# Hyperparameters for controlling numerical computation.
# Setting with_negative_lambda = TRUE extends the search to the unsmoothing
# regime (generating more zero-crossings than the benchmark). 
with_negative_lambda <- T
# Numerical optimisation settings for M-SSA.
# The setting chosen here allows for extreme smoothing (see example below)
lower_limit_nu <- "rhomax"
# The target is asymmetric (Gamma is the identity), so this flag has no
# effect on estimation; either TRUE or FALSE is admissible here.
symmetric_target <- F
# Use bisection-based optimisation with 2^split_grid effective resolution.
# This is substantially faster than brute-force grid search and exploits
# the monotonicity of the first-order ACF when |nu| > 2 * rho_max(L).
# For |nu| < 2 * rho_max(L), grid search should be used instead, as it
# enumerates all solutions to the HT constraint (at higher computational cost).
split_grid <- 20

# ─────────────────────────────────────────────────────────────────────────────
# 5.5. M-SSA Estimation
# ─────────────────────────────────────────────────────────────────────────────
MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho0,
                      with_negative_lambda, xi, lower_limit_nu, Sigma,
                      symmetric_target)

# Target correlations of M-SSA with the causal M-MSE smoother, for each
# series i = 1, ..., n. If the HT of M-SSA matches that of M-MSE exactly,
# these correlations equal one (M-SSA replicates M-MSE).
MSSA_obj$crit_rhoyz
# Target correlations of M-SSA with the acausal target. In a smoothing
# context the target is always causal, so these values coincide with
# crit_rhoyz above.
MSSA_obj$crit_rhoy_target
# First-order ACFs implied by the optimised smoother. If numerical
# optimisation has converged, these should match rho0 closely.
MSSA_obj$crit_rhoyy
# Verify convergence: the two sets of values should agree.
rho0
# Optimal regularisation parameter nu for each series. Values of nu > 2
# indicate active smoothing (fewer zero-crossings than M-MSE).
MSSA_obj$nu_opt

# ─────────────────────────────────────────────────────────────────────────────
# Smoother Coefficients
# ─────────────────────────────────────────────────────────────────────────────
# 1. M-SSA filter applied to innovations epsilon_t (Wold decomposition).
#    Primarily used for diagnostic purposes; also relevant when the process
#    is non-invertible (see Exercise 2.7 in Tutorial 8).
bk_mat <- MSSA_obj$bk_mat
# 2. M-SSA filter applied directly to the observed series x_t (deconvolution).
#    This is the operationally relevant smoother in nearly all applications.
bk_x_mat <- MSSA_obj$bk_x_mat

# ─────────────────────────────────────────────────────────────────────────────
# M-MSE Reference Smoother
# ─────────────────────────────────────────────────────────────────────────────
# M-SSA optimisation principle: for a given HT (encoded via rho0), M-SSA
# seeks the smoother that is as close as possible to the classical M-MSE
# smoother. See Wildi (2026b) for the formal statement.
#
# 1. M-MSE filter applied to innovations epsilon_t from the Wold decomposition.
gammak_mse <- MSSA_obj$gammak_mse
# Verification: in the nowcasting case (delta = 0, identity target), M-MSE
# reduces to the Wold decomposition itself.
max(abs(t(gammak_mse) - xi))
# 2. M-MSE filter applied to the observed series x_t.
#    Since the target is x_t and delta = 0, the optimal MSE solution is the
#    identity: the best causal approximation of x_t is x_t itself.
gammak_x_mse <- MSSA_obj$gammak_x_mse


# ─────────────────────────────────────────────────────────────────────────────
# 5.6. Performance: Theoretical (Expected) Values
# ─────────────────────────────────────────────────────────────────────────────

# 5.6.1. Compute True First-Order ACFs
# Compute the system matrices required for theoretical performance evaluation.
# Notation follows Wildi (2026b).
M_obj    <- M_func(L, Sigma)
M_tilde  <- M_obj$M_tilde
I_tilde  <- M_obj$I_tilde

# Theoretical lag-one ACF for each series, derived from the filter coefficients.
# The lag-one ACF is in bijective correspondence with the holding time (HT);
# see Wildi (2026b), Section 2.
# Series 1:
rho_mse_1 <- (gammak_mse[, 1]) %*% M_tilde %*% gammak_mse[, 1] /
  gammak_mse[, 1]  %*% I_tilde  %*% gammak_mse[, 1]
rho_ssa_1 <- bk_mat[, 1] %*% M_tilde %*% bk_mat[, 1] /
  bk_mat[, 1] %*% I_tilde  %*% bk_mat[, 1]
# Expected: rho_ssa_1 > rho_mse_1, confirming that M-SSA is smoother than M-MSE.
rho_mse_1
rho_ssa_1
# Series 2:
rho_mse_2 <- gammak_mse[, 2] %*% M_tilde %*% gammak_mse[, 2] /
  gammak_mse[, 2] %*% I_tilde  %*% gammak_mse[, 2]
rho_ssa_2 <- bk_mat[, 2] %*% M_tilde %*% bk_mat[, 2] /
  bk_mat[, 2] %*% I_tilde  %*% bk_mat[, 2]
# Series 3:
rho_mse_3 <- gammak_mse[, 3] %*% M_tilde %*% gammak_mse[, 3] /
  gammak_mse[, 3] %*% I_tilde  %*% gammak_mse[, 3]
rho_ssa_3 <- bk_mat[, 3] %*% M_tilde %*% bk_mat[, 3] /
  bk_mat[, 3] %*% I_tilde  %*% bk_mat[, 3]

# 5.6.2. Convert ACFs to Holding Times (Equation 4 in Wildi 2026b)
# Verify that the optimised M-SSA HTs match the imposed targets in ht_vec.
# Increasing split_grid in the MSSA_func call improves the fit.
compute_holding_time_from_rho_func(rho_ssa_1)$ht
compute_holding_time_from_rho_func(rho_ssa_2)$ht
compute_holding_time_from_rho_func(rho_ssa_3)$ht
# Target HTs for reference:
ht_vec
# M-MSE HTs (all smaller than the corresponding M-SSA HTs, confirming that
# M-SSA imposes greater smoothness):
compute_holding_time_from_rho_func(rho_mse_1)$ht
compute_holding_time_from_rho_func(rho_mse_2)$ht
compute_holding_time_from_rho_func(rho_mse_3)$ht

# 5.6.3. Objective Functions (Target Correlations); see Wildi (2026b), Section 2
# M-MSE trivially achieves a target correlation of one with itself, since the
# target is defined as the M-MSE output (equivalently, x_{t+delta} with delta=0).
# M-SSA maximises the target correlation in the second row subject to the HT
# constraint. The values computed here should match MSSA_obj$crit_rhoyz.
crit_mse_1 <- gammak_mse[, 1] %*% I_tilde %*% gammak_mse[, 1] /
  gammak_mse[, 1] %*% I_tilde  %*% gammak_mse[, 1]
crit_ssa_1 <- gammak_mse[, 1] %*% I_tilde %*% bk_mat[, 1] /
  (sqrt(bk_mat[, 1]    %*% I_tilde %*% bk_mat[, 1]) *
     sqrt(gammak_mse[, 1] %*% I_tilde %*% gammak_mse[, 1]))

crit_mse_2 <- gammak_mse[, 2] %*% I_tilde %*% gammak_mse[, 2] /
  gammak_mse[, 2] %*% I_tilde  %*% gammak_mse[, 2]
crit_ssa_2 <- gammak_mse[, 2] %*% I_tilde %*% bk_mat[, 2] /
  (sqrt(bk_mat[, 2]    %*% I_tilde %*% bk_mat[, 2]) *
     sqrt(gammak_mse[, 2] %*% I_tilde %*% gammak_mse[, 2]))

crit_mse_3 <- gammak_mse[, 3] %*% I_tilde %*% gammak_mse[, 3] /
  gammak_mse[, 3] %*% I_tilde  %*% gammak_mse[, 3]
crit_ssa_3 <- gammak_mse[, 3] %*% I_tilde %*% bk_mat[, 3] /
  (sqrt(bk_mat[, 3]    %*% I_tilde %*% bk_mat[, 3]) *
     sqrt(gammak_mse[, 3] %*% I_tilde %*% gammak_mse[, 3]))

# Verify: M-SSA target correlations should match MSSA_obj$crit_rhoyz.
c(crit_ssa_1, crit_ssa_2, crit_ssa_3)
MSSA_obj$crit_rhoyz

# Summary table of target correlations:
# Row 1 (M-MSE): all entries are one in this smoothing exercise.
# Row 2 (M-SSA): maximised values subject to the HT constraint.
criterion_mat <- rbind(c(crit_mse_1, crit_mse_2, crit_mse_3),
                       c(crit_ssa_1, crit_ssa_2, crit_ssa_3))
colnames(criterion_mat) <- c(paste("Series ", 1:n, paste = ""))
rownames(criterion_mat) <- c("MSE", "SSA")
criterion_mat

# ─────────────────────────────────────────────────────────────────────────────
# 5.7. Sample Performances
# ─────────────────────────────────────────────────────────────────────────────
# Simulate a long realisation of the VAR(1) process and compute sample
# performance measures. Convergence of sample values to their theoretical
# counterparts validates the M-SSA framework as a pertinent multivariate
# extension of univariate I-SSA smoothing.
len <- 10000

# Note: depending on len simulation may take some time to go through.
# Source the simulation utilities used in Wildi (2026b).
source(paste(getwd(), "/R/M_SSA_paper_functions.r", sep = ""))
# Simulate the VAR(1) process and evaluate filter performance.
setseed <- 16
sample_obj     <- sample_series_performances_smooth_func(
  A, Sigma, len, bk_mat, bk_x_mat,
  t(gammak_mse), L, setseed, MSSA_obj)
perf_mat_sample <- sample_obj$perf_mat_sample
perf_mat_true   <- sample_obj$perf_mat_true
bk_mat          <- sample_obj$bk_mat
gammak_mse      <- sample_obj$gammak_mse
y_mat           <- sample_obj$y_mat
zdelta_mat      <- sample_obj$zdelta_mat
z_mse_mat       <- sample_obj$z_mse_mat
x_mat           <- sample_obj$x_mat

# Display theoretical values alongside sample estimates (in parentheses).
# Convergence of sample to theoretical values confirms the validity of the
# M-SSA approach. Increasing len improves agreement at the cost of
# longer computation time.
perf_mat <- matrix(
  paste(round(perf_mat_true[,   c("Sign accuracy", "Cor. with MSE", "ht", "ht MSE")], 5),
        " (",
        round(perf_mat_sample[, c("Sign accuracy", "Cor. with MSE", "ht", "ht MSE")], 5),
        ")", sep = ""),
  ncol = 4)
colnames(perf_mat) <- c("Sign accuracy", "Cor. with data", "HT M-SSA", "HT of data")
perf_mat

# ─────────────────────────────────────────────────────────────────────────────
# 5.8. Plots
# ─────────────────────────────────────────────────────────────────────────────

# Plot 1: M-SSA filter weights (bk_x_mat) for each of the three targets.
# Each panel shows the lag structure of the weights applied to all three
# input series when constructing one smoothed output series.
mplot<-cbind(bk_x_mat[1:L,1],bk_x_mat[L+1:L,1],bk_x_mat[2*L+1:L,1])
colnames(mplot)<-c("Series 1","Series 2","Series 3")
colo<-c("blue","red","darkgreen","violet","black")
par(mfrow=c(1,3))
plot(mplot[,1],main="SSA: first target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


mplot<-cbind(bk_x_mat[1:L,2],bk_x_mat[L+1:L,2],bk_x_mat[2*L+1:L,2])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: second target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

mplot<-cbind(bk_x_mat[1:L,3],bk_x_mat[L+1:L,3],bk_x_mat[2*L+1:L,3])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: third target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


# Plot 2: Cross-correlation functions (CCFs) against Series 2 (left panel)
# and a short realisation of the VAR(1) process (right panel).
# See Wildi (2026b), Section 4.2, for a discussion of this example.
# Hint: The smoother weights in the above plot is related to the CCF and sample 
# realizations in the below plot (explainability).

par(mfrow=c(1,2))

colo<-c("blue","red","darkgreen")

lag_max<-20
j<-1
k<-2
pc<-peak_cor_func(x_mat,j,k,lag_max)
j<-2
k<-2
pc<-cbind(pc,peak_cor_func(x_mat,j,k,lag_max))
j<-3
k<-2
pc<-cbind(pc,peak_cor_func(x_mat,j,k,lag_max))
plot(pc[,1],lty=1,main=paste("CCF against series ",k,sep=""),axes=F,type="l",ylab="",xlab="Lag-structure",ylim=c(min(pc),max(pc)))
for (i in 1:ncol(pc))
  lines(pc[,i],col=colo[i],lwd=1,lty=1)
abline(h=0)
abline(v=lag_max)
axis(1,at=1:nrow(pc),labels=-lag_max+1:nrow(pc))
axis(2)
box()

ts.plot(x_mat[850:875,],col=colo,main="VAR(1)")
for (i in 1:3)
  mtext(paste("Series ",i,sep=""),col=colo[i],line=-i,xlab="time",ylab="")



# Plot 3: Observed data (black), M-SSA smoother (cyan), and M-MSE smoother
# (violet) over a short window, with vertical dashed lines marking
# zero-crossings of the M-SSA output.
# In the nowcasting case (delta = 0, identity target), M-MSE reduces to
# the identity: z_mse = x_t (bloth lines overlap). The plot therefore 
# illustrates how M-SSA tracks the data subject to the imposed HTe constraint.

# Select a short sub-sample of the very longth simulation path
anf<-10*L
enf<-11*L
y_mat[anf:enf,]
z_mse_mat[anf:enf,]
zdelta_mat[anf:enf,]
colo<-c("cyan","violet","black")
par(mfrow=c(2,2))
# MSE, target and x are identical: gammak is an identity and delta=0
# We here demonstrate the multivariate SSA trend specification
#   yt matches xt conditional on holding-time constraint
for (i in 1:n)
{
  mplot<-cbind(y_mat[anf:enf,i],z_mse_mat[anf:enf,i],zdelta_mat[anf:enf,i])
  
  ts.plot(scale(mplot,center=F,scale=T),col=colo,main=paste("Series ",i,sep=""),lwd=2)
  abline(h=0)
  abline(v=1+which(mplot[2:nrow(mplot),1]*mplot[1:(nrow(mplot)-1),1]<0),col=colo[1],lty=2)
  #abline(v=1+which(mplot[2:nrow(mplot),2]*mplot[1:(nrow(mplot)-1),2]<0),col=colo[2],lty=2)
  lines(scale(mplot,center=F,scale=T)[1],col=colo[1])
}



# ══════════════════════════════════════════════════════════════════════════════
# Exercise 6: M-SSA Smoothing — Three Special Cases
#   6.1  Symmetric multivariate backcast smoother
#   6.2  Replicating M-MSE (identity nowcast)
#   6.3  Extreme smoothing
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 6.1. Symmetric M-SSA Backcast Smoother
# ─────────────────────────────────────────────────────────────────────────────
# Extend Exercise 5 to the symmetric backcasting setting by shifting the
# target to the centre of the filter window. Only delta needs to change;
# all other settings remain identical to Exercise 5.
delta <- delta2 <- -(L - 1) / 2

# Retain the same holding-time targets as in Exercise 5.
ht_vec <- matrix(c(min(8, L / 2), min(6, L / 2), min(10, L / 2)), nrow = 1)
# Convert HTs to first-order ACFs for input to M-SSA.
rho0 <- apply(ht_vec, 1, compute_rho_from_ht)[[1]]$rho

MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho0,
                       with_negative_lambda, xi, lower_limit_nu, Sigma,
                       symmetric_target)

# Extract key performance summaries.
crit_sym     <- MSSA_obj$crit_rhoyz   # Target correlations with M-MSE.
ht_sym       <- MSSA_obj$crit_rhoyy   # Achieved first-order ACFs.
MSSA_obj2$nu_opt                        # Optimal regularisation parameters.

# Extract filter coefficient matrices.
bk_mat_sym   <- MSSA_obj$bk_mat       # M-SSA applied to innovations epsilon_t.
bk_x_mat_sym <- MSSA_obj$bk_x_mat    # M-SSA applied to observed series x_t.
gammak_mse_sym <- MSSA_obj$gammak_mse # M-MSE applied to innovations epsilon_t.

# Plot filter weights for each target series.
mplot<-cbind(bk_x_mat_sym[1:L,1],bk_x_mat_sym[L+1:L,1],bk_x_mat_sym[2*L+1:L,1])
colnames(mplot)<-c("Series 1","Series 2","Series 3")
colo<-c("blue","red","green","violet","black")
par(mfrow=c(1,3))
plot(mplot[,1],main="SSA: first target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


mplot<-cbind(bk_x_mat_sym[1:L,2],bk_x_mat_sym[L+1:L,2],bk_x_mat_sym[2*L+1:L,2])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: second target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

mplot<-cbind(bk_x_mat_sym[1:L,3],bk_x_mat_sym[L+1:L,3],bk_x_mat_sym[2*L+1:L,3])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: third target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Outcome: the symmetric M-SSA backcast smoother is virtually
# indistinguishable from a univariate SSA smoother applied to each series
# independently. The intuition is straightforward: when future observations
# are available (backcasting), the additional series convey negligible
# incremental information for constructing the optimal smoother, and the
# multivariate structure offers no material gain over the univariate solution.

# ─────────────────────────────────────────────────────────────────────────────
# 6.2. Replicating M-MSE (Identity Nowcast)
# ─────────────────────────────────────────────────────────────────────────────
# Repeat the nowcasting exercise from Exercise 5, but now calibrate the HT
# targets to match the empirical HTs of the data. When the imposed HT equals
# the HT of the data, M-SSA reduces to M-MSE (the identity in this case).
delta <- 0

# Estimate the empirical HT of each series from the simulated sample.
# Note: a longer sample (larger len) yields more accurate HT estimates.
ht_vec <- matrix(c(compute_empirical_ht_func(x_mat[, 1])$empirical_ht,
                   compute_empirical_ht_func(x_mat[, 2])$empirical_ht,
                   compute_empirical_ht_func(x_mat[, 3])$empirical_ht),
                 nrow = 1)
colnames(ht_vec) <- paste("Series ", 1:3, sep = "")
ht_vec

# Convert empirical HTs to first-order ACFs for M-SSA input.
rho0 <- apply(ht_vec, 1, compute_rho_from_ht)[[1]]$rho

MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho0,
                      with_negative_lambda, xi, lower_limit_nu, Sigma,
                      symmetric_target)

# Expected outcome: target correlations are (approximately) equal to one,
# since the imposed HT matches the HT of the data and M-SSA replicates M-MSE.
MSSA_obj$crit_rhoyz

# Extract filter coefficient matrices.
bk_mat_sym     <- MSSA_obj$bk_mat       # M-SSA applied to innovations.
bk_x_mat_sym   <- MSSA_obj$bk_x_mat    # M-SSA applied to observed series.
gammak_mse_sym <- MSSA_obj$gammak_mse  # M-MSE applied to innovations.

# Plot filter weights for each target series.
mplot<-cbind(bk_x_mat_sym[1:L,1],bk_x_mat_sym[L+1:L,1],bk_x_mat_sym[2*L+1:L,1])
colnames(mplot)<-c("Series 1","Series 2","Series 3")
colo<-c("blue","red","green","violet","black")
par(mfrow=c(1,3))
plot(mplot[,1],main="SSA: first target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


mplot<-cbind(bk_x_mat_sym[1:L,2],bk_x_mat_sym[L+1:L,2],bk_x_mat_sym[2*L+1:L,2])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: second target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

mplot<-cbind(bk_x_mat_sym[1:L,3],bk_x_mat_sym[L+1:L,3],bk_x_mat_sym[2*L+1:L,3])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: third target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Outcome: the M-SSA nowcast filter weights are approximately those
# of the identity, confirming that M-SSA replicates M-MSE when the imposed
# HT matches the empirical HT of the data. Increasing the sample length (len)
# yields more accurate empirical HT estimates and brings M-SSA closer to the
# exact identity solution.


# ─────────────────────────────────────────────────────────────────────────────
# 6.3. Extreme Smoothing
# ─────────────────────────────────────────────────────────────────────────────
# Impose the nearly maximum admissible degree of smoothing by setting all HTs 
# equal to the filter length L. At this extreme setting, the M-SSA smoother is 
# no longer approximately exponentially decaying in its lag structure.
#
# Practical note: extreme smoothing is not recommended in applications, as
# it leaves very few effective degrees of freedom. If such strong smoothing
# is required, the filter length L should be increased so that the target
# HT satisfies HT < L/2.

# Extreme HT
ht_vec <- matrix(c(L, L, L), nrow = 1)

# Convert the extreme HTs to first-order ACFs for M-SSA input.
rho0 <- apply(ht_vec, 1, compute_rho_from_ht)[[1]]$rho

MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho0,
                      with_negative_lambda, xi, lower_limit_nu, Sigma,
                      symmetric_target)

# Expected outcome: target correlations are close to zero, reflecting the
# severe trade-off between smoothness and tracking accuracy at the HT boundary.
MSSA_obj$crit_rhoyz

# Extract filter coefficient matrices.
bk_mat_sym     <- MSSA_obj$bk_mat       # M-SSA applied to innovations.
bk_x_mat_sym   <- MSSA_obj$bk_x_mat    # M-SSA applied to observed series.
gammak_mse_sym <- MSSA_obj$gammak_mse  # M-MSE applied to innovations.

# Plot filter weights for each target series.
mplot<-cbind(bk_x_mat_sym[1:L,1],bk_x_mat_sym[L+1:L,1],bk_x_mat_sym[2*L+1:L,1])
colnames(mplot)<-c("Series 1","Series 2","Series 3")
colo<-c("blue","red","green","violet","black")
par(mfrow=c(1,3))
plot(mplot[,1],main="SSA: first target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


mplot<-cbind(bk_x_mat_sym[1:L,2],bk_x_mat_sym[L+1:L,2],bk_x_mat_sym[2*L+1:L,2])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: second target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

mplot<-cbind(bk_x_mat_sym[1:L,3],bk_x_mat_sym[L+1:L,3],bk_x_mat_sym[2*L+1:L,3])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: third target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# ─────────────────────────────────────────────────────────────────────────────
# Technical Note: Smoothers at the HT Boundary
# ─────────────────────────────────────────────────────────────────────────────
# When the imposed HT approaches the upper limit (L+1), rho0 approaches the 
# largest eigenvalue of the
# system matrix M. In this limiting regime, the optimal smoother converges to
# the eigenvector associated with the maximum eigenvalue of M; see Wildi
# (2026b), Section 3.
eigen_obj <- eigen(M)
# Largest eigenvalue of M (upper bound on achievable rho0):
max(eigen_obj$values)
# Imposed rho0 (close to the maximum eigenvalue):
rho0

# Plot the eigenvector corresponding to the maximum eigenvalue of M.
# This vector defines the theoretically smoothest possible filter of length L,
# and the extreme-smoothing M-SSA coefficients should approximate it in the 
# above plot.
par(mfrow = c(1, 1))
ts.plot(eigen_obj$vectors[, 1],
        main = paste("Smoothest admissible filter of length ", L,
                     " (eigenvector of max eigenvalue of M)", sep = ""),
        xlab = "Lag", ylab = "")














# ─────────────────────────────────────────────────────────────────────────────
# 6.3 Extreme Smoothing
# ─────────────────────────────────────────────────────────────────────────────


# Impose very strong smoothing: bk_x_mat will no longer be approximately
# exponentially decaying. We are at the boundary of smoothness for an MA
# smoother of length L=51
# This is not recommended in applications since the degrees of freedom are very small
# Recommendation: increase the filter length L so that HT < L/2 at least

ht_vec <- matrix(c(L, L, L), nrow = 1)

# Transform into lag-one ACF for M-SSA
rho0 <- apply(ht_vec, 1, compute_rho_from_ht)[[1]]$rho


MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho0,
                      with_negative_lambda, xi, lower_limit_nu, Sigma,
                      symmetric_target)

# Target correlations are close to zero
MSSA_obj$crit_rhoyz


bk_mat_sym<-MSSA_obj$bk_mat
# SSA as applied to xt (deconvolution)
bk_x_mat_sym<-MSSA_obj$bk_x_mat
# MSE as applied to epsilont
gammak_mse_sym<-MSSA_obj$gammak_mse



mplot<-cbind(bk_x_mat_sym[1:L,1],bk_x_mat_sym[L+1:L,1],bk_x_mat_sym[2*L+1:L,1])
colnames(mplot)<-c("Series 1","Series 2","Series 3")
colo<-c("blue","red","green","violet","black")
par(mfrow=c(1,3))
plot(mplot[,1],main="SSA: first target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


mplot<-cbind(bk_x_mat_sym[1:L,2],bk_x_mat_sym[L+1:L,2],bk_x_mat_sym[2*L+1:L,2])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: second target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

mplot<-cbind(bk_x_mat_sym[1:L,3],bk_x_mat_sym[L+1:L,3],bk_x_mat_sym[2*L+1:L,3])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: third target",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Technical note:
# The smoothers are at the limit of the HT constraint
# Specifically; rho0 is close to the maximal eigenvalue of M
eigen_obj<-eigen(M)
max(eigen_obj$values)
rho0

# Therefore, the above smoothers must be close to the eigenvector of the max eigenvalue
# see Wildi 2026b, section 3
par(mfrow=c(1,1))
ts.plot(eigen_obj$vectors[,1],main=paste("This smoother maximizes the HT for a smoother/filter of length ",L,sep=""),xlab="Lag",ylab="")










