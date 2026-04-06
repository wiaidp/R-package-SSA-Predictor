# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 9: I-SSA SMOOTHING
# Introducing the I-SSA Trend
# ══════════════════════════════════════════════════════════════════════════════

# This tutorial extends the SSA smoothing framework of tutorial 8 to 
# non-stationary processes (I-SSA). For a comprehensive discussion of the
# distinction between smoothing and prediction, the reader is referred
# to Tutorial 8.

# ─────────────────────────────────────────────────────────────────────────────
# The I-SSA Trend: A New Trend Definition
# ─────────────────────────────────────────────────────────────────────────────
# We introduce a new trend concept — the I-SSA trend — grounded in the
# I-SSA smoothing framework. It is distinguished from classical trend
# definitions by the following key properties:
#
# - Minimal structural imposition: the shape of the I-SSA trend is
#   determined entirely by the data and the holding-time (HT) constraint.
#   No artificial structure is imposed on the data-generating process,
#   and no idealised appearance is prescribed for the smoothed series.
#
# - Economically meaningful turning points (TPs): TPs of a trend component
#   typically signal important transitions in the underlying process and
#   are directly relevant to decision makers. The HT constraint governs
#   the frequency of such TPs in a transparent and interpretable way,
#   linking the smoothing outcome directly to specific research objectives
#   e.g., business-cycle analysis (crisis tracking) or algorithmic trading.
#
# - Logical consistency and statistical efficiency: the I-SSA trend
#   complies with the imposed TP frequency while tracking the
#   non-stationary data as closely as possible. This dual requirement —
#   respecting the TP frequency and minimising tracking error —
#   yields a trend design that is both logically consistent and
#   statistically efficient.
#
# Taken together, these properties make the I-SSA trend a compelling
# alternative to classical trend definitions (HP, ideal trend, 
# canonical trend or Wiener Kolmogorov trend extraction).


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
# Backcasting: Symmetric I-SSA Smoother/Trend with HT of Two-Sided HP
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
# 1.1  Setup for I-SSA
# ─────────────────────────────────────────────────────────────────────────────

# Filter design parameters
L <- 201

# HP Lambda Selection:
# Lambda is the penalty weight assigned to the curvature term in the
# Whittaker-Henderson (WH/HP) graduation criterion (see Tutorials 2.0
# and 8). A larger lambda enforces greater smoothness at the cost of
# reduced fidelity to the observed series; a smaller lambda allows the
# trend to track the data more closely at the cost of increased roughness.
#
# The conventional value of 14,400 is the standard choice for monthly
# series, calibrated to yield a trend broadly equivalent to that of the
# HP filter applied to quarterly data with lambda = 1,600. Exercise 5
# applies the resulting HP and I-SSA smoothers to the monthly US
# Industrial Production Index (INDPRO).
lambda_hp <- 14400

HP_obj   <- HP_target_mse_modified_gap(2 * (L - 1) + 1, lambda_hp)
hp_two   <- HP_obj$target
hp_one   <- hp_trend <- HP_obj$hp_trend
# hp_target is used solely to derive the HT constraint for I-SSA.
# The HP filter itself plays no further role in the I-SSA computation.
hp_target <- hp_two

# ─────────────────────────────────────────────────────────────────────────────
# 1.2. I-SSA Settings
# ─────────────────────────────────────────────────────────────────────────────

# 1.2.1. Holding-Time (HT) Constraint Calibration
# ─────────────────────────────────────────────────────────────────────────────
# For non-stationary series, crossings about the mean are not well
# defined. We therefore work with first differences and assume the
# differenced series to be stationary, see tutorial 8. 
# Accordingly, the HT constraint in
# I-SSA is defined on first differences (see Equation 29 in Wildi 2026a).
# Because first differences of a random walk are white noise, the HT is
# obtained directly from compute_holding_time_func(), which assumes
# white-noise input.
rho1 <- compute_holding_time_func(hp_target)$rho_ff1
ht1  <- compute_holding_time_func(hp_target)$ht
# In first differences, a zero-crossing of the two-sided HP smoother
# occurs on average once every ht1 observations.
ht1
# Important: HP enters I-SSA solely through the HT constraint.
# The HP filter itself is not used as a target; only its TP rate is matched.

# 1.2.2. Smoothing Lag and Target Specification
# ─────────────────────────────────────────────────────────────────────────────
# delta = 0          : nowcast (target is the current observation).
# delta = -(L-1)/2   : symmetric backcast (target is the centre of the
#                      filter window).
delta <- -(L - 1) / 2

# ─────────────────────────────────────────────────────────────────────────────
# 1.3. I-SSA Trend Estimation
# ─────────────────────────────────────────────────────────────────────────────

# 1.3.1. Calling ISSA_Trend_func()
# ─────────────────────────────────────────────────────────────────────────────
# ISSA_Trend_func() builds on the I-SSA framework from Tutorial 8. The key
# difference is that the target is x_{t+delta} (smoothing of the observed
# series) rather than a filtered version of x_t. The function returns an
# I-SSA trend filter of length L whose TP rate — measured as zero-crossings
# in first differences — matches the specified HT constraint (of the 
# two-sided HP).
#
# The assumed DGP for first differences is an ARMA(p,q) with AR coefficients
# a1 (vector of length p) and MA coefficients b1 (vector of length q).
# lambda_init initialises the numerical optimisation; lambda_init = 0
# corresponds to the MSE benchmark (the identity, i.e., no smoothing imposed).
lambda_init   <- 0
ht_constraint <- ht1

# Call 1: explicit specification of all arguments.
ISSA_obj <- ISSA_Trend_func(ht_constraint, L, delta, a1, b1, lambda_init)
bk_obj   <- ISSA_obj$bk_obj

# Call 2: default arguments. When a1, b1, and lambda_init are omitted,
# ISSA_Trend_func() defaults to a random-walk DGP (white-noise first
# differences). This call is equivalent to Call 1 above.
ISSA_obj <- ISSA_Trend_func(ht_constraint, L, delta)
bk_obj   <- ISSA_obj$bk_obj

# Call 3: HT constraint supplied as a lag-one ACF.
# The HT constraint may be specified in either of two equivalent forms:
#   - ht_constraint > 1 : mean duration between zero-crossings (holding
#     time in observation units); a confirmatory message is issued.
#   - 0 < ht_constraint < 1 : lag-one ACF (rho); a confirmatory message
#     is issued.
# Supplying rho1 instead of ht1 yields an identical result.
ht_constraint <- rho1
ISSA_obj <- ISSA_Trend_func(ht_constraint, L, delta)
bk_obj   <- ISSA_obj$bk_obj

# All three calls above are equivalent and produce identical I-SSA trends.

# 1.3.2  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# Verify convergence: bk_obj$rho_yy should match rho1.
# If the values differ, increase the number of iterations in optim or 
# use a better choice for lambda_init.
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
# 3. I-SSA targets the non-stationary level x_t
#    directly, while SSA in Tutorial 8 targets growth in stationary first
#    differences. I-SSA imposes the HT constraint on stationary
#    first differences, ensuring a well-defined and interpretable smoothness
#    criterion. 
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
# The HT constraint asks that the smoother exhibits sufficiently long memory 
# between direction changes (in levels), without prescribing how that memory 
# should manifest in the shape of the output.
#
# As a result, the HT constraint is minimally invasive: it enforces the
# desired degree of smoothness while leaving the smoother free to determine
# the optimal shape of the trend from the data.
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
# framework — where the smoother is one-sided — the
# limitations of HP become more pronounced, further widening the performance
# gap between HP and I-SSA, even under model misspecification.
#
#
#
# ══════════════════════════════════════════════════════════════════════════════
# Exercise 2: Same as Exercise 1 but for the Nowcast Smoother (delta = 0)
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Specify Smoothing Lag and run I-SSA Trend
# ─────────────────────────────────────────────────────────────────────────────
delta <- 0

ISSA_obj<-ISSA_Trend_func(ht_constraint,L,delta)

bk_obj<-ISSA_obj$bk_obj

# 2.2 Diagnostics
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
colnames(mplot) <- c("HP-two", "Target", "I-SSA trend")

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
     ylim = c(min(mplot), max(mplot[, "I-SSA trend"])), ylab = "", xlab = "Lags")
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

# 3.1.2  Holding-Time (HT) Constraint Calibration
# ─────────────────────────────────────────────────────────────────────────────
# Here we depart from exercise 2.
# We match the HT of the one-sided HP rather than the two-sided HP.
# The one-sided HP has a much smaller HT, reflecting weaker smoothing.
rho1 <- compute_holding_time_func(hp_trend)$rho_ff1
ht1  <- compute_holding_time_func(hp_trend)$ht
ht1

ht_constraint<-ht1

# 3.2 I-SSA Trend
# ─────────────────────────────────────────────────────────────────────────────
# 3.2.1 Run ISSA_Trend_func
# ─────────────────────────────────────────────────────────────────────────────


ISSA_obj<-ISSA_Trend_func(ht_constraint,L,delta)

bk_obj<-ISSA_obj$bk_obj


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
colnames(mplot) <- c("HP-two", "HP-one", "Target", "I-SSA trend")

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
     ylim = c(min(mplot), max(mplot[, "I-SSA trend"])), ylab = "", xlab = "Lags")
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
# ONE-SIDED HP filter.
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
#   (i.e., white-noise first differences) and is not re-fitted to the
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
# The I-SSA trend tracks the index substantially more closely than the 
# one-sided HP, while maintaining a comparable degree of smoothness. 
# In particular, the I-SSA trend is faster (left-shifted) at cycle peaks and 
# dips. 


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
# Empirical and expected values differ due to non-stationarity 
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
# level) biases HT estimates. Centering the
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
# 1. The I-SSA trend was optimised under a random-walk (white-noise differences)
#    assumption, which does not fully capture the positive serial correlation
#    observed in differenced INDPRO. The reported MSE gains are therefore
#    conservative: re-fitting I-SSA to the observed dependence structure of the
#    index would likely yield further performance improvements.
#
# 2. Despite this model mismatch, the I-SSA trend strongly outperforms the classical
#    one-sided HP nowcast trend in terms of MSE tracking of the log-index, while
#    maintaining comparable smoothness — as measured either by the HT of
#    first differences or, equivalently, by an equal frequency of turning
#    points (TPs) on levels. The MSE gain is driven by two complementary
#    factors: improved timeliness (a left-shift of the filter output relative
#    to the one-sided HP) and more accurate tracking of dynamic swings at
#    business-cycle peaks and troughs.
#
# 3. The combination of reduced lag, superior MSE performance, and robust
#    tracking of dynamic swings makes I-SSA (trend) a compelling, data-driven
#    alternative to the classical one-sided HP smoother for real-time
#    macroeconomic monitoring. Unlike the HP filter — whose smoothness
#    criterion explicitly penalises curvature (second-order differences) —
#    I-SSA is amorphous in the sense that it does not impose an extraneous 
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
# Crucially, this TP-rate bias affects HP and I-SSA equally: it is a
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
# - Non-stationary (trending) series: the I-SSA smoother targets the TREND LEVEL.
#   The random walk is recommended as the default DGP assumption, as applied 
#   throughout the above exercises.
#
# - Stationary series: when stationarity is achieved by differencing a
#   non-stationary series, the smoother targets TREND GROWTH (i.e., the
#   rate of change of the underlying level series). The white-noise
#   hypothesis is recommended as the default DGP assumption in this
#   setting, as illustrated in Tutorial 8.
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
# (see previous exercises). 
# In this case, the resulting I-SSA trend inherits its meaning and 
# interpretation directly from the benchmark, by virtue of optimal tracking. 
# It should be noted, however, that customisation is generally not a 
# smoothing exercise but a prediction exercise.



