# ========================================================================
# Tutorial 6: I-SSA
# Extension of SSA to non-stationary (I-ntegrated) processes
# ========================================================================
# Topics:
#   - Introduction of I-SSA
#   - Double-Stroke: Joint unified treatment of level nowcasting and 
#     growth-sign signaling (recession tracking) within a single consistent 
#     design
#   - Maximal monotone predictor
# ========================================================================

# ────────────────────────────────────────────────────────────────
# Context
# ────────────────────────────────────────────────────────────────
# SSA is typically formulated for stationary time series.
#
# SSA optimization principle:
#   - Maximize sign accuracy (or target correlation)
#   - Subject to a holding-time (HT) constraint
#
# Holding-time (HT):
#   - Mean duration between consecutive mean crossings of the predictor
#   - For zero-mean processes: mean duration between sign changes
#
# Issue:
#   - Integrated (non-stationary) processes are not mean-reverting
#   - ⇒ HT is undefined for non-stationary levels
#   - ⇒ Neither target correlation nor sign accuracy are well-defined
#       for non-stationary levels

# ────────────────────────────────────────────────────────────────
# Extension to Non-Stationary Series
# ────────────────────────────────────────────────────────────────
# Notation:
#   - x_t       : observed data; assumed I(1), i.e. Δx_t = x_t - x_{t-1} = u_t
#                 is stationary. Extensions to I(2) are covered in Wildi (2026a).
#   - z_{t+δ}   : acausal (two-sided) linear target relying on future values
#                 x_{t+k}, k > 0. Here: output of an acausal trend filter.
#   - y_t       : I-SSA predictor (causal nowcast of z_{t+δ}).
#
# Working in stationary first differences:
#   - HT is defined on Δy_t = y_t - y_{t-1}  (stationary)
#   - A pseudo target correlation and pseudo sign accuracy can be defined
#     via finite-length MA inversions of y_t (details omitted here)
#
# Primal and dual formulations of I-SSA:
#   Primal : For a given HT in first differences, y_t maximises tracking of
#            z_{t+δ} in non-stationary levels (minimises level MSE).
#   Dual   : For a given level-tracking accuracy (e.g. MSE), I-SSA maximises
#            HT in first differences Δy_t.
#
# Maximal monotone property:
#   - If Δy_t is zero-mean, zero-crossings of Δy_t are turning points (TPs) of
#     y_t which is maximally monotone: no other linear predictor of z_t has
#     a longer mean duration between consecutive TPs than I-SSA.
#   - If E[Δy_t] = μ ≠ 0, HT regulates the mean duration between above-average
#     (Δy_t > μ) and below-average (Δy_t < μ) growth episodes (crossover
#     points). I-SSA is then maximally monotone about the linear trend T_t
#     with slope μ, i.e. the average duration of crossings of T_t is maximal.

# ────────────────────────────────────────────────────────────────
# References
# ────────────────────────────────────────────────────────────────
# Application to business-cycle analysis (stationary first differences):
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5
#
# Application to business-cycle analysis in original (non-stationary) levels:
#   Wildi, M. (2026a)
#     Sign Accuracy, Mean-Squared Error and the Rate of Zero Crossings:
#     a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547

# ────────────────────────────────────────────────────────────────
# This tutorial is based on Section 5 of Wildi (2026a)
# ────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────
# Initialize session: clear workspace and load required packages
# ────────────────────────────────────────────────────────────────
rm(list = ls())

library(xts)        # Extended time-series objects and utilities
library(mFilter)    # HP and BK trend/cycle filters
library(quantmod)   # Data retrieval (e.g. from FRED)
library(pROC)       # ROC curve calculation
library(tis)        # NBER recession datings for the US


# ────────────────────────────────────────────────────────────────
# Load custom I-SSA and signal-extraction functions
# ────────────────────────────────────────────────────────────────
source(file.path(getwd(), "R", "simple_sign_accuracy.r"))   # Core SSA routines
source(file.path(getwd(), "R", "ISSA_functions.r"))         # Core I-SSA routines
source(file.path(getwd(), "R", "HP_JBCY_functions.r"))      # HP-filter helpers and JBCY utilities
source(paste(getwd(), "/R/ROCplots.r", sep = ""))           # ROC plot


# ========================================================================
# Exercise 1: Nowcasting via the "Double-Stroke" Principle
# ========================================================================
#
# Objective:
#   Design a nowcast of the HP trend that simultaneously achieves:
#     • Accurate trend tracking in LEVELS  (fidelity / low MSE)
#     • Smooth growth tracking in DIFFERENCES  (regularity / high HT)
#
# The "double-stroke" principle refers to satisfying both goals with a
# single, consistent filter design — rather than applying two separate,
# potentially conflicting approaches.
#
# Practical outputs of such a nowcast:
#   (a) An estimate of the current trend level
#   (b) A signal for the current economic state (above/below average growth,
#       or recession/expansion phases)
#
# ────────────────────────────────────────────────────────────────
# Specific challenge:
#   Construct an I-SSA nowcast of the two-sided HP(14400) trend for monthly
#   US industrial production (INDPRO) such that the nowcast:
#     (i)  Matches the holding-time (HT) of the classic one-sided HP filter
#          when evaluated in first differences, and
#     (ii) Achieves a lower MSE (better level tracking) than the classic
#          one-sided HP nowcast.
#
# ────────────────────────────────────────────────────────────────
# Benchmarks:
#
#   1) MSE-optimal predictor (based on an ARIMA(1,1,0) model)
#      • Minimises level MSE unconstrainedly
#      • Typically very noisy (low HT) — unsuitable for phase signaling
#
#   2) One-sided concurrent HP filter (HP-C)
#      • Standard business-cycle monitoring tool
#      • Adequate smoothness for phase signaling, but suboptimal level MSE
#
#   I-SSA target: match HP-C smoothness (same HT in differences) while
#   improving level-tracking accuracy (lower MSE than HP-C).
#
# ────────────────────────────────────────────────────────────────
# Key evaluation questions:
#   Q1: How much MSE does I-SSA sacrifice relative to the unconstrained
#       MSE-optimal predictor (cost of imposing the HT constraint)?
#   Q2: How much MSE does I-SSA gain relative to HP-C for the same
#       degree of smoothness (benefit of I-SSA optimisation)?
# ========================================================================


# ────────────────────────────────────────────────────────────────
# 1.1 INDPRO: Data Loading and Preparation
# ────────────────────────────────────────────────────────────────

# ············································
# 1.1.1 Load data (option to refresh from FRED)
# ············································
# Set reload_data = TRUE to download a fresh copy from FRED;
# FALSE uses the previously saved local file.
reload_data <- FALSE

if (reload_data) {
  getSymbols("INDPRO", src = "FRED")
  save(INDPRO, file = file.path(getwd(), "Data", "INDPRO"))
} else {
  load(file = file.path(getwd(), "Data", "INDPRO"))
}

tail(INDPRO)   # Quick sanity-check: inspect the last few observations


# ············································
# 1.1.2 Sample selection and transformations
# ············································
# Restrict to the analysis window and take natural logarithms.
# Log-transformation stabilises variance and makes growth rates
# approximately additive (log-differences ≈ percentage changes).
start_year <- 1982
end_year   <- 2024

y      <- as.double(log(INDPRO[paste0(start_year, "/", end_year)]))   # Numeric vector (log-levels)
y_xts  <- log(INDPRO[paste0(start_year, "/", end_year)])               # xts object  (log-levels)


# ············································
# 1.1.3 Plot raw data, log-levels, and first differences
# ············································
par(mfrow = c(2, 2))

# Panel 1: Raw INDPRO index (levels)
plot(as.double(INDPRO), main = "INDPRO", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = "black", lwd = 1)
axis(1, at = 1:length(INDPRO), labels = index(INDPRO))
axis(2); box()

# Panel 2: Log-INDPRO (log-levels) — trend is approximately linear
plot(as.double(y_xts), main = "Log-INDPRO", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = "black", lwd = 1)
axis(1, at = 1:length(y_xts), labels = index(y_xts))
axis(2); box()

# Panel 3: Log-differences (monthly growth rates) — approximately stationary
plot(as.double(diff(y_xts)), main = "Diff-log", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = "black", lwd = 1)
abline(h = 0)
axis(1, at = 1:length(diff(y_xts)), labels = index(diff(y_xts)))
axis(2); box()

# Panel 4: ACF of log-differences — positive first-order autocorrelation
# suggests an AR(1) model for the differenced series (ARIMA(1,1,0) overall)
acf(na.exclude(diff(y_xts)), main = "ACF of diff-log")



# Store series length and the full log-level vector for downstream use
len     <- length(y)
x_tilde <- as.double(y_xts)

# Note:
#   Wildi (2024) applies SSA to log-differences (stationary). 
#   An optimal nowcast of the non-stationary level is not considered.
#   Here, I-SSA operates on log-LEVELS (non-stationary) and is designed
#   to track the two-sided HP trend in levels directly.


# ────────────────────────────────────────────────────────────────
# 1.2 HP Target Specification
# ────────────────────────────────────────────────────────────────
# Define the two-sided HP trend filter that I-SSA will approximate.

# Filter length: 201 coefficients (captures the bulk of HP weight mass)
L <- 201

# Standard HP smoothing parameter for monthly data
lambda_hp <- 14400

# Compute HP filter coefficients and related objects
HP_obj <- HP_target_mse_modified_gap(L, lambda_hp)

# hp_mse : Right half (causal side) of the two-sided HP filter,
#           derived as the MSE-optimal one-sided approximation under
#           a white-noise input assumption
hp_mse <- gamma <- HP_obj$hp_mse

# hp_two : Full symmetric classic two-sided HP filter
hp_two <- HP_obj$target

# hp_c   : Classic one-sided concurrent HP filter (HP-C),
#           i.e. the standard real-time approximation
hp_c <- HP_obj$hp_trend

# Visualise all three filter representations
par(mfrow = c(1, 1))
ts.plot(cbind(hp_two, hp_mse, hp_c),
        xlab = "Lags", col = rainbow(3))
mtext("Two-sided HP (shifted right for display)", col = rainbow(3)[1], line = -1)
mtext("Classic one-sided concurrent HP (HP-C)",   col = rainbow(3)[3], line = -2)
mtext("MSE-optimal one-sided HP (white-noise input)", col = rainbow(3)[2], line = -3)


# ────────────────────────────────────────────────────────────────
# 1.3 Model Setup for I-SSA
# ────────────────────────────────────────────────────────────────
# I-SSA requires a parametric model for the differenced series.
# ACF evidence (exercise 1.1.3) supports an ARIMA(1,1,0) for log-INDPRO,
# i.e. an AR(1) for the log-differences.

# AR(1) and MA(0) coefficients — fixed a priori to avoid distortion
# from the extreme COVID-19 observations present in the full sample
a1 <- 0.3
b1 <- 0

# Fit ARIMA(1,1,0) on the pre-pandemic sample for reference and to
# obtain the innovation variance estimate (σ²); the pre-2020 sample
# avoids outlier-driven parameter instability
p <- 1; q <- 0
arima_obj <- arima(diff(y_xts)["/2020"], order = c(p, 0, q))
# This is close to AR(1) specified above
arima_obj

# Inspect residual diagnostics: ACF of residuals should be near zero.
# Minor deviations are acceptable given the deliberately simple model
# (parsimony reduces overfitting risk)
tsdiag(arima_obj)

# Extract residual standard deviation for later MSE rescaling
sigma_ip <- sqrt(arima_obj$sigma2)


# ────────────────────────────────────────────────────────────────
# 1.4 Holding-Time (HT) Constraint Derivation
# ────────────────────────────────────────────────────────────────
# The Holding-Time (HT) constraint is imposed on the first differences of the
# I-SSA output (see Eq. 29, Wildi 2026a). Specifically, I-SSA is required to
# match the HT of HP-C, ensuring that both filters signal economic cycle phases
# (i.e., identify Crossover/Turning Points) at the same average rate.

# ── Step 1: Wold (MA-∞) Representation of the AR(1) Model ─────
# Compute the Wold coefficients ξ_j = a1^j for the AR(1) model fitted to
# log-differences of INDPRO, truncated at lag L - 1.
# These coefficients represent the impulse response of the AR(1) process,
# expressing it as an infinite moving average of white-noise innovations.
xi <- c(1, ARMAtoMA(ar = a1, ma = b1, lag.max = L - 1))

# ── Step 2: Convolve HP-C with the Wold Coefficients ──────────
# Convolving HP-C with the Wold coefficients ξ yields the equivalent filter
# applied directly to white-noise innovations. This transformation is required
# because the HT computation formulas assume a white-noise input; applying
# HP-C to the AR(1) process is therefore equivalent to applying the convolved
# filter to its underlying innovations.
hp_c_convolved_with_xi <- conv_two_filt_func(xi, hp_c)$conv

# Plot:
#   - Red:    HP-C applied directly to the AR(1) process.
#   - Orange: The convolved filter (hp_c_convolved_with_xi) applied to
#             the white-noise innovations of the AR(1) process.
ts.plot(cbind(hp_c, hp_c_convolved_with_xi),
        col  = c("red", "orange"),
        main = "Verification: HP-C vs. Convolved HP-C")
mtext("Original HP-C (applied to AR(1))",         line = -1, col = "red")
mtext("HP-C convolved with ξ (applied to innovations)", line = -2, col = "orange")
# The outputs of both filters are identical.

# ── Step 3: Compute the HT of HP-C under the AR(1) Model ──────
# The HT is defined as the expected number of months between consecutive
# mean-crossings of the HP-C output when the underlying data-generating
# process is the fitted AR(1) model for INDPRO log-differences.
# This value is subsequently enforced as the HT constraint in I-SSA
# optimisation, so that I-SSA and HP-C signal Turning Points at the same rate.
HT_HP_obj     <- compute_holding_time_func(hp_c_convolved_with_xi)
ht_constraint <- HT_HP_obj$ht   # HT of HP-C under the AR(1) model
ht_constraint

# ── Reference: HT of HP-C Applied Directly to White Noise ─────
# For comparison, we also compute the HT of HP-C when applied to a pure
# white-noise input (i.e., ignoring the AR(1) autocorrelation structure).
# This serves as a baseline and is expected to be shorter than ht_constraint,
# because white noise has no persistence and therefore produces more frequent
# sign changes (mean-crossings) than a positively autocorrelated AR(1) process.
ht_hp_applied_to_wn <- compute_holding_time_func(hp_c)$ht
ht_hp_applied_to_wn

# Summary:
#   - ht_hp_applied_to_wn: HT of HP-C under white-noise input (shorter).
#   - ht_constraint:       HT of HP-C under the AR(1) model (longer if a1>0).
# The latter ht_constraint is relevant for imposing the HT to I-SSA, 
# assuming the AR(1) model fits the data: first differences of INDPRO.


# ────────────────────────────────────────────────────────────────
# 1.5 I-SSA Optimisation
# ────────────────────────────────────────────────────────────────
# ············································
# 1.5.1 Specify I-SSA Design
# ············································

# Nowcast: predict the contemporaneous trend z_t
delta <- 0  
# Warning: avoid forecast horizons larger than (L-1)/2
if (delta>(L-1)/2)
{
  print(paste("Warning: forecast horizon delta=",delta," is large.",sep="")) 
  print("This may result in poorly conditioned MSE and I-SSA estimates ")
}
# ── Target specification ──────────────────────────────────────────
# Option 1 (used by default):
# 
gamma_target     <- hp_two
# Setting symmetric_target <- TRUE will shift δ to the centre of the 
# causal target filter hp_two: this transforms the causal hp_two to an 
# effectively acausal two-sided design, as required.
symmetric_target <- TRUE

if (FALSE) {
  # Option 2 (alternative, equivalent formulation):
  #   Supply the full causal (right-shifted) two-sided HP directly.
  #   δ must be shifted to the centre of the causal filter to align
  #   the nowcast with the correct time index

  symmetric_target <- FALSE
  delta          <- delta + (length(gamma_target) - 1) / 2
}

# ── Lagrange multiplier initialisation ───────────────────────────
# λ = 0 corresponds to the unconstrained MSE predictor (no HT penalty);
# the I-SSA optimiser will update λ to satisfy the HT constraint
lambda_start <- 0

# ············································
# 1.5.2 Run I-SSA
# ············································

# ── Run I-SSA ─────────────────────────────────────────────────────
# The constraint on smoothness can be specified in two equivalent forms:
#
#   • As a holding-time (HT):
#       Values of ht_constraint > 1 are interpreted as an effective HT,
#       i.e. the expected number of periods between consecutive mean-
#       crossings of the I-SSA output in first differences.
#
#   • As a lag-one autocorrelation (ACF):
#       Values in the range -1 ≤ ht_constraint ≤ 1 are interpreted as a
#       lag-one ACF of the I-SSA first differences. 
#
# The function automatically distinguishes between the two forms based on
# the magnitude of ht_constraint and issues a confirmatory message
# indicating which interpretation has been applied.
#
# An error is raised if ht_constraint < -1, since a lag-one ACF below -1
# is inadmissible (outside the valid autocorrelation range).
ISSA_obj <- ISSA_func(ht_constraint, L, delta, gamma_target,
                      symmetric_target, a1, b1, lambda_start)

# ············································
# 1.5.3 Unpack
# ············································

# Unpack results
bk_obj     <- ISSA_obj$bk_obj    # Main I-SSA object
gamma_mse  <- ISSA_obj$gamma_mse # MSE-optimal filter coefficients (λ = 0 benchmark)
b_x        <- ISSA_obj$b_x       # Optimised I-SSA filter coefficients
lambda_opt<-ISSA_obj$lambda_opt  # optimal Lagrangian multiplier: lambda_opt=0 corresponds to the MSE benchmark.
ht_issa<-bk_obj$ht_issa          # HT of optimized I-SSA: should match ht_constraint

# ── Theoretical MSE ───────────────────────────────────────────────
# Expected MSE of the I-SSA nowcast, computed relative to the MSE-optimal
# (unconstrained) one-sided predictor rather than the two-sided HP target.
#
# Rationale for this reference choice:
#   Both objectives — minimising MSE relative to the two-sided HP and
#   minimising MSE relative to the one-sided MSE predictor — are
#   theoretically equivalent optimisation criteria (see Section 2,
#   Wildi 2026a). I-SSA reports MSE relative to the one-sided MSE
#   predictor by default because this metric has a natural interpretation:
#     • MSE = 0  ⟺  I-SSA exactly replicates the unconstrained MSE predictor
#                   (HT constraint is non-binding)
#     • MSE > 0  ⟺  smoothness constraint is active; the reported value
#                   quantifies the accuracy cost of imposing the HT constraint
#
# Rescaling by σ² (the AR(1) innovation variance) converts the theoretical
# MSE from normalised innovation units to the original INDPRO innovations,
# making it directly comparable to the sample MSE computed below.
bk_obj$mse_yz * sigma_ip^2
# The sample MSE between the MSE-optimal predictor and I-SSA (exercise 1.10.1)
# should be close to this theoretical value; discrepancies indicate
# either finite-sample variability or model misspecification.

# ── Pseudo Target Correlation ─────────────────────────────────────
# Because the two-sided HP target is non-stationary (integrated), a
# conventional Pearson correlation between the target and the nowcast is
# ill-defined.
#
# I-SSA instead uses a pseudo target correlation derived via a finite-length
# MA inversion of the integrated process, which maps the non-stationary
# optimisation problem onto a well-defined stationary objective
# (Wildi 2026a, eq. 29, LHS). We do not discuss this any further.
bk_obj$rho_yz


# ············································
# 1.5.4 Validation Checks
# ············································

# ── Convergence check ─────────────────────────────────────────────
# If optimisation converged, the achieved HT should equal ht_constraint
# to numerical precision. A large discrepancy indicates convergence
# failure; remedies: increase optim() iterations or provide a better
# starting value for lambda_start (lambda_start<0 indicates smoothing).
abs(ht_issa - ht_constraint)

# ── Cointegration Check ────────────────────────────────────────
# For I-SSA applied to non-stationary I(1) levels, a finite Mean Squared
# Error (MSE) requires that I-SSA tracks the target filter (here HP-C, or
# its MSE-optimal predictor) without systematic bias. Any non-zero bias
# would lead to an asymptotically unbounded MSE.
#
# Unbiasedness is guaranteed when the two filters are cointegrated, i.e.,
# when their coefficient sums are equal. Equivalently, this requires that
# the transfer functions of both filters agree at the unit-root (zero)
# frequency — ensuring that both filters apply the same long-run scaling
# to the integrated input series.
#
# The cointegration check below verifies this condition:
# the difference between the coefficient sums should be zero (or negligibly
# small up to numerical precision).
sum(b_x)-sum(gamma_mse)

# This cointegration constraint is a key structural element in the I-SSA 
# optimization criterion, see equation 30 in Wildi 2026a.


# ────────────────────────────────────────────────────────────────
# 1.6 Plot filters
# ────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "green", "blue", "red")

mplot <- cbind(
  hp_two,
  gamma_mse,
  b_x,
  hp_c
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
     ylim = c(min(mplot[, "HP-C"]), max(mplot[, "I-SSA"])),main="First 30 lags")
abline(h = 0)
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:nrow(mplot), labels = 0:(nrow(mplot) - 1))
axis(2); box()

# The MSE-optimal filter concentrates most of its weight on the most recent
# observation — a direct consequence of the data following a random walk
# (or near-random-walk) process. Because successive filter outputs are
# dominated by the latest data point, consecutive differences of the filter
# output are highly volatile, resulting in a very short HT in first
# differences, as confirmed below.
#
# The I-SSA filter exhibits a more complex weight-decay profile than the
# classic HP-C filter:
#   • Convergence of the coefficients to zero is faster than HP-C.
#   • Near lag 0, the filter displays a characteristic "nose" shape
#     (a local dip or inflection before the main peak), which is a
#     consequence of the implicit zero-boundary condition
#     embedded in the (I-)SSA optimisation framework — see Theorem 1
#     in Wildi (2026a).

# ────────────────────────────────────────────────────────────────
# 1.7 Filter and Plot Data 
# ────────────────────────────────────────────────────────────────
# Apply side = 1 to one-sided designs and side = 2 to two-sided HP target
y_ssa            <- filter(x_tilde, b_x,       side = 1)
y_hp_concurrent  <- filter(x_tilde, hp_c,  side = 1)
y_mse            <- filter(x_tilde, gamma_mse, side = 1)
y_hp_two         <- filter(x_tilde, hp_two,    side = 2)

# ────────────────────────────────────────────────────────────────
# Plot Nowcasts and Target: in Levels
# ────────────────────────────────────────────────────────────────

colo <- c("black", "violet", "green", "blue", "red")

# Plot: levels
par(mfrow = c(1, 1))
# Skip burn-in phase
anf <- L 
enf <- length(x_tilde)

mplot <- cbind(x_tilde, y_hp_two, y_mse, y_ssa, y_hp_concurrent)[anf:enf, ]
colnames(mplot) <- c("Data", "Target: HP-two", "MSE: HP-one", "I-SSA", "HP-C")

plot(mplot[, 1], main = "Data and trends", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = colo[1], lwd = 1)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[anf:length(y_xts)])
axis(2); box()

# Examining the filter outputs over the sample period reveals two notable
# empirical properties of I-SSA relative to the classic HP-C nowcast:
#
#   • Timeliness: I-SSA tracks turning points (peaks and troughs) in the
#     HP trend more promptly than HP-C. 
#
#   • Smoothness: Despite its improved timeliness, I-SSA is visually as
#     smooth as HP-C — that is, it does not generate additional (spurious)
#     turning-point signals. This smoothness is enforced through the HT 
#     constraint, which requires the I-SSA first differences to cross their 
#     mean at the same average frequency as the HP-C first differences.
#

# ────────────────────────────────────────────────────────────────
# Plot Nowcasts and Target: in First Differences
# ────────────────────────────────────────────────────────────────

# Skip burn-in phase
anf <- L
enf <- length(x_tilde) - 1

# ············································
# Plot: first differences
# ············································
mplot <- apply(
  cbind(x_tilde, y_hp_two, y_mse, y_ssa, y_hp_concurrent),
  2,
  diff
)[anf:enf, ]
colnames(mplot) <- c("Diff-Data", "Target: HP-two", "MSE: HP-one", "I-SSA", "HP-C")


# ············································
# Three panels plot
# ············································
par(mfrow = c(3, 1))
# Panel 1: MSE vs target
select_vec <- c(2, 3)
plot(mplot[, select_vec[1]], main = "Zero Crossings MSE in First Differences",
     axes = FALSE, type = "l", col = colo[select_vec[1]], lwd = 1,
     ylim = c(-0.013, 0.006))
abline(v = 1 + which(sign(mplot[-1, select_vec[2]]) != sign(mplot[-nrow(mplot), select_vec[2]])),
       col = colo[select_vec[2]], lty = 2)
abline(h = 0)

for (i in seq_along(select_vec)) {
  lines(mplot[, select_vec[i]], col = colo[select_vec[i]])
  mtext(colnames(mplot)[select_vec[i]], line = -i, col = colo[select_vec[i]])
}
axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[(anf + 1):length(y_xts)])
axis(2); box()

# Panel 2: HP-C vs target
select_vec <- c(2, 5)
plot(mplot[, select_vec[1]], main = "Zero Crossings HP-C in First Differences",
     axes = FALSE, type = "l", col = colo[select_vec[1]], lwd = 1,
     ylim = c(-0.013, 0.006))
abline(v = 1 + which(sign(mplot[-1, select_vec[2]]) != sign(mplot[-nrow(mplot), select_vec[2]])),
       col = colo[select_vec[2]], lty = 2)
abline(h = 0)

for (i in seq_along(select_vec)) {
  lines(mplot[, select_vec[i]], col = colo[select_vec[i]])
  mtext(colnames(mplot)[select_vec[i]], line = -i, col = colo[select_vec[i]])
}
axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[(anf + 1):length(y_xts)])
axis(2); box()

# Panel 3: I-SSA vs target
select_vec <- c(2, 4)
plot(mplot[, select_vec[1]], main = "Zero Crossings I-SSA in First Differences",
     axes = FALSE, type = "l", col = colo[select_vec[1]], lwd = 1,
     ylim = c(-0.013, 0.006))
abline(v = 1 + which(sign(mplot[-1, select_vec[2]]) != sign(mplot[-nrow(mplot), select_vec[2]])),
       col = colo[select_vec[2]], lty = 2)
abline(h = 0)

for (i in seq_along(select_vec)) {
  lines(mplot[, select_vec[i]], col = colo[select_vec[i]])
  mtext(colnames(mplot)[select_vec[i]], line = -i, col = colo[select_vec[i]])
}
axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[(anf + 1):length(y_xts)])
axis(2); box()

# Examining the first-difference panels reveals a clear contrast between
# the three competing nowcasts:
#
#   • MSE-optimal nowcast (green, top panel):
#       Highly erratic in first differences, with frequent and irregular
#       sign changes. While it minimises level MSE, its growth-rate signal
#       is too noisy to be practically useful for economic phase detection
#       (recession vs. expansion), generating numerous false alarms.
#
#   • HP-C and I-SSA (lower panels):
#       Both filters produce smooth first-difference profiles, with a
#       crossing frequency governed by the imposed HT constraint. Their
#       growth-rate signals dip below the zero-growth line consistently
#       at NBER-dated recessions, demonstrating reliable phase detection.
#       Occasional false signals occur during prolonged expansions — a
#       known limitation shared by both filters — but these are infrequent
#       relative to the MSE-optimal predictor.
#
# This comparison underscores the practical value of the HT constraint:
# by controlling the mean crossing frequency in first differences, I-SSA
# retains the phase-signaling reliability of HP-C while improving upon
# its level-tracking accuracy — confirming the double-stroke principle
# in an empirical setting.

# We now verify these assertions by computing sample performances

# ────────────────────────────────────────────────────────────────
# 1.8 Sample Performance Evaluation
# ────────────────────────────────────────────────────────────────

# ············································
# 1.8.1 Tracking Accuracy (Level MSE)
# ············································
# Compute the sample MSE of each nowcast relative to the two-sided HP
# trend (the infeasible but optimal target).
#
# Expected ranking (lowest to highest MSE):
#   MSE-optimal predictor  <  I-SSA  <  HP-C
#
# Key question: I-SSA is designed to outperform HP-C in MSE terms while
# matching it in smoothness (HT) — do the sample results confirm this?

# ── Target Construction ────────────────────────────────────────
# Shift the two-sided HP trend by delta periods to align it with the
# nowcast outputs. A positive delta shifts the target forward (trims
# the head); a negative delta shifts it backward (trims the tail).
if (delta >= 0)
  y_target <- c(y_hp_two[(1 + delta):length(y_hp_two)], rep(NA, delta))
if (delta < 0)
  y_target <- c(rep(NA, -delta), y_hp_two[1:(length(y_hp_two) + delta)])

# ── Sample MSE Computation ─────────────────────────────────────
mse_mse  <- mean((y_target - y_mse)^2,          na.rm = TRUE)  # MSE-optimal nowcast (theoretical lower bound)
mse_issa <- mean((y_target - y_ssa)^2,           na.rm = TRUE)  # I-SSA nowcast
mse_hpc  <- mean((y_target - y_hp_concurrent)^2, na.rm = TRUE)  # Concurrent (one-sided) HP-C

# ── MSE Ratios Relative to HP-C ────────────────────────────────
# All MSEs are normalised by the HP-C MSE, so HP-C has a ratio of 1.
# Values below 1 indicate outperformance relative to HP-C.
mse_mat <- matrix(c(mse_mse / mse_hpc, 1, mse_issa / mse_hpc), nrow = 1)
colnames(mse_mat) <- c("MSE-Optimal", "HP-C", "I-SSA")
rownames(mse_mat) <- "MSE Ratio (relative to HP-C)"
round(mse_mat, 2)

# ── Interpretation ─────────────────────────────────────────────
# The results confirm the expected ranking:
#   - MSE-optimal nowcast: achieves the lowest MSE, approximately one-third
#     of the HP-C MSE, representing the best achievable level tracking
#     accuracy (theoretical lower bound).
#   - I-SSA: substantially outperforms HP-C, reducing the MSE by
#     approximately one-half relative to HP-C, while matching it in
#     smoothness (HT constraint).
#   - HP-C: serves as the reference benchmark (ratio = 1.00 by construction).


# Cross-check: compare the sample MSE between the MSE-optimal predictor
# and I-SSA against its theoretical counterpart derived under the AR(1) model.
# Close agreement validates both model adequacy and the I-SSA optimisation.
mean((y_mse - y_ssa)^2, na.rm = TRUE)   # Sample estimate
bk_obj$mse_yz * sigma_ip^2              # Expected MSE computed internally by I-SSA (rescaled to AR(1) innovation variance)
# Note: the reference is not the two-sided HP (as in mse_issa) but the 
# optimal MSE predictor.

# ············································
# 1.8.2 Smoothness: Holding Time in First Differences
# ············································
# The HT is computed on first differences of each nowcast to measure
# the mean duration between consecutive mean-crossings of the growth signal.
# Series are mean-centred (standardized) before computing zero-crossings.

# MSE-optimal nowcast — expected to have the shortest HT (noisiest growth signal)
# Sample HT
compute_empirical_ht_func(scale(diff(y_mse)[anf:enf]))$empirical_ht
# Compute expected (theoretical) HT of MSE nowcast
# i) Convolve gamma_mse with Wold decomposition of AR(1)
mse_convolved_with_xi <- conv_two_filt_func(xi, gamma_mse)$conv
# ii) Apply compute_holding_time_func
compute_holding_time_func(mse_convolved_with_xi)$ht
# Sample and expected numbers should agree up to finite sample variation. 
# If not, the principal reasons could be model misspecification or uncentered 
# data.


# I-SSA nowcast — HT constrained to match HP-C by design
compute_empirical_ht_func(scale(diff(y_ssa)[anf:enf]))$empirical_ht
ht_issa         # Theoretical HT under AR(1) : should be close to the sample value above
# Differences are due to sample variance and/or model misspecification

# Classic one-sided HP-C — benchmark for smoothness
compute_empirical_ht_func(scale(diff(y_hp_concurrent)[anf:enf]))$empirical_ht
ht_constraint   # Theoretical HT under AR(1) : should be close to the sample value above
# Differences are due to sample variance and/or model misspecification

# Two-sided HP target — substantially smoother than all causal predictors.
# (Potentially too smooth, see tutorial 2.0)
compute_empirical_ht_func(scale(diff(y_hp_two)))$empirical_ht


# ────────────────────────────────────────────────────────────────
# 1.9 Summary Table: MSE and Holding Time
# ────────────────────────────────────────────────────────────────
# Consolidate tracking accuracy and smoothness metrics into a single table
# for direct comparison across the three competing nowcasts.

mat_perf <- matrix(nrow = 2, ncol = 3)

# Row 1: Sample MSE relative to the two-sided HP target
mat_perf[1, ] <- c(
  mean((y_hp_two - y_mse)^2, na.rm = TRUE),
  mean((y_hp_two - y_ssa)^2, na.rm = TRUE),
  mean((y_hp_two - y_hp_concurrent)^2, na.rm = TRUE)
)

# Row 2: Sample HT computed on first differences (unscaled, within analysis window)
mat_perf[2, ] <- c(
  compute_empirical_ht_func(diff(y_mse)[anf:enf])$empirical_ht,
  compute_empirical_ht_func(diff(y_ssa)[anf:enf])$empirical_ht,
  compute_empirical_ht_func(diff(y_hp_concurrent)[anf:enf])$empirical_ht
)

colnames(mat_perf) <- c("MSE-Optimal Nowcast", "I-SSA", "HP-C")
rownames(mat_perf) <- c("Sample MSE", "Sample Holding Time")

mat_perf

# ── Interpretation ────────────────────────────────────────────────
# The summary table highlights three key findings:
#
#   MSE-optimal nowcast:
#     • Achieves the lowest sample MSE (best level-tracking accuracy)
#     • Very short HT — the growth signal is highly erratic, making it
#       impractical for real-time recession/expansion monitoring
#
#   HP-C (classic one-sided HP):
#     • Higher HT — smooth growth signal suitable for phase detection
#     • Sample MSE is approximately 200% larger than the MSE-optimal
#       benchmark, reflecting substantial level-tracking inefficiency
#
#   I-SSA:
#     • Matches or marginally exceeds HP-C in smoothness (similar HT),
#       confirming that the HT constraint is effectively enforced
#     • Sample MSE is approximately 50% larger than the MSE-optimal
#       benchmark and roughly half the MSE incurred by HP-C.
#
#   Overall conclusion:
#     I-SSA occupies a sweet spot (an efficient position on the 
#     smoothness–accuracy frontier), simultaneously delivering:
#       (a) Large smoothness gains over the MSE-optimal predictor at
#           moderate MSE cost, and
#       (b) MSE improvements over HP-C at identical (or
#           slightly better) smoothness — validating the double-stroke
#           principle in an empirical setting.
#
# ── Outlook: Exercises 2, 3 and 4 ─────────────────────────────
# Exercises 3 and 4 will explore alternative smoothness trade-offs,
# examining settings that impose respectively more and less smoothness
# than the baseline established in Exercise 1. An evaluation
# criterion will be the ability of the nowcast to identify NBER-dated
# recessions, quantified via a Receiver Operating Characteristic (ROC)
# curve analysis and the corresponding Area Under the Curve (AUC).
#
# Before proceeding to Exercises 3 and 4, Exercise 2 examines a
# degenerate case: targeting a symmetric backcast under
# the conditions of Exercise 1. While this configuration is not
# practically relevant, it provides useful theoretical insight into
# the behaviour of the I-SSA optimisation framework.



# ========================================================================
# Exercise 2: As Exercise 1 but a Symmetric Backcast
# ========================================================================
# ────────────────────────────────────────────────────────────────
# 2.1
# ────────────────────────────────────────────────────────────────
# The right half of the two-sided HP filter has length:
(L - 1) / 2

# A symmetric backcast is obtained by setting δ to the negative of this
# half-length, which centres the I-SSA prediction horizon at the midpoint
# of the two-sided HP filter resulting in a symmetric backcast:
delta <- -(L - 1) / 2

# Note on boundary behaviour:
#   Any value δ ≤ -(L-1)/2 produces an identical solution, because the
#   I-SSA filter cannot exploit information beyond the centre of the
#   symmetric target. The implementation automatically clamps δ to
#   -(L-1)/2 if a more negative value is supplied, and issues a brief
#   informational message to flag the adjustment.

if (FALSE) {
  # Example: this more negative value triggers the automatic correction
  # and yields the same filter coefficients as delta = -(L-1)/2
  delta <- -150
}

# The remaining settings are copied from exercise 1
gamma_target     <- hp_two
symmetric_target <- TRUE
lambda_start <- 0

# ────────────────────────────────────────────────────────────────
# 2.2 Run I-SSA
# ────────────────────────────────────────────────────────────────

ISSA_obj <- ISSA_func(ht_constraint, L, delta, gamma_target,
                      symmetric_target, a1, b1, lambda_start)

# Unpack results
bk_obj     <- ISSA_obj$bk_obj    # Main I-SSA object
gamma_mse  <- ISSA_obj$gamma_mse # MSE-optimal filter coefficients (λ = 0 benchmark)
b_x        <- ISSA_obj$b_x       # Optimised I-SSA filter coefficients
lambda_opt<-ISSA_obj$lambda_opt  # optimal Lagrangian multiplier: lambda_opt=0 corresponds to the MSE benchmark.
ht_issa<-bk_obj$ht_issa          # HT of optimized I-SSA: should match ht_constraint

# The theoretical MSE is much smaller than in exercise 1 because we have `future' 
#  data available in the backcast
bk_obj$mse_yz * sigma_ip^2
# But why does it not vanish exactly in this case?
# The answer is given in the plot


# ────────────────────────────────────────────────────────────────
# 2.3 Plot filters
# ────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "green", "blue", "red")

mplot <- cbind(
  hp_two,
  gamma_mse,
  b_x,
  hp_c
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

# Inspecting the filter coefficient plots reveals three distinct profiles:
#
#   1. MSE-optimal filter:
#        Coefficients (green) are identical to the two-sided HP weights 
#        (violet), as expected — the unconstrained MSE predictor reproduces 
#        the two-sided HP exactly when the backcast horizon aligns with the 
#        center point of HP.
#
#   2. I-SSA filter:
#        Displays an unusual oscillating pattern superimposed on the
#        smooth MSE/HP baseline. This behaviour is not a numerical
#        artefact but a theoretically necessary consequence of the
#        constraint configuration (see points 3–5 below).
#
#   3. Origin of the oscillating pattern:
#        The imposed HT constraint is substantially smaller than the
#        natural HT of the MSE-optimal predictor (which itself inherits
#        the high smoothness of the two-sided HP). Enforcing a much
#        shorter HT forces I-SSA to generate more frequent mean-crossings
#        in first differences than the MSE benchmark would naturally
#        produce (un-smoothing).
#
#   4. Competing requirements:
#        I-SSA must simultaneously satisfy two conflicting objectives:
#          • Track the two-sided HP target as closely as possible
#            (minimise level MSE), and
#          • Produce a prescribed — here artificially elevated — number
#            of crossings in first differences (satisfy the low-HT
#            constraint).
#
#   5. Resolution via oscillation:
#        The only way to reconcile accurate level-tracking with a
#        high crossing frequency is to inject a regular alternating
#        (oscillatory) component into the filter coefficients. This
#        component contributes negligibly to level MSE (the oscillations
#        largely cancel when convolved with a smooth trend) but
#        systematically increases the zero-crossing rate of the first-
#        difference output, satisfying the HT constraint.
#
# Practical implication:
#   This pathological behaviour signals that the HT constraint is set
#   well outside the feasible range for meaningful backcasting — the
#   constraint demands more spurious high-frequency noise than the
#   optimal MSE backcast supports. In applied work, HT constraints
#   should be set above (or equal to) the natural HT of the
#   MSE-optimal predictor to avoid such degenerate solutions.

# Note: a positive value of the optimized Lagrangian multiplier signals 
# un-smoothing (more crossings than MSE benchmark) by I-SSA
lambda_opt


# ========================================================================
# Exercise 3: Dual I-SSA — Maximal Monotone for a Given MSE Budget
# ========================================================================
#
# Recap of Exercise 1 (Primal formulation):
#   The HT of HP-C (in first differences) was imposed as a constraint,
#   and I-SSA minimised level MSE subject to that smoothness requirement.
#   Result: I-SSA matched HP-C in smoothness while achieving better level-
#   tracking accuracy (lower MSE).
#
# Exercise 3 (Dual formulation):
#   The MSE of HP-C is now treated as the budget constraint, and I-SSA
#   maximises HT (smoothness in first differences) subject to that MSE
#   ceiling. Result: I-SSA matches HP-C in level-tracking accuracy while
#   achieving greater smoothness:
#     -longer HT in difference
#     -less turning points (TP) in level
# Exercise 3 emphasizes the maximal-monotone property of the I-SSA nowcast
#   for the given MSE budget.
#
# Implementation note:
#   I-SSA is currently implemented in primal form only; a direct MSE
#   constraint cannot be specified. The dual problem is therefore solved
#   indirectly via the following iterative strategy:
#
#     1. Start from the primal HT constraint used in Exercise 1.
#     2. Increase the HT constraint by a trial increment.
#     3. Re-run I-SSA and check whether the resulting sample MSE matches
#        the HP-C sample MSE (within acceptable sampling error).
#     4. Repeat until the MSE condition is satisfied.
#
# Rule of thumb:
#   Empirical experience suggests that increasing the primal HT constraint
#   by approximately 50% is a reliable starting point for matching the
#   HP-C MSE level. This heuristic is applied below and then verified
#   against the sample MSE of HP-C.
#
# All other hyperparameters (filter length L, model parameters a1/b1,
# delta, target specification) are held fixed from Exercise 1.
# ========================================================================
# NOTE: This exercise depends on the Wold decomposition xi defined in Exercise 1.
# If Exercise 1 has not been run yet, please do so before proceeding.

# Building on the setup from Exercise 1, we modify only the HT parameter.

# Set the HT to a value approximately 50% greater than the one used 
# in Exercise 1.
ht_constraint <- 15


# ────────────────────────────────────────────────────────────────
# 3.1 I-SSA Optimisation
# ────────────────────────────────────────────────────────────────
# Same settings as in exrecise 1
delta <- 0
gamma_target     <- hp_two
symmetric_target <- TRUE
lambda_start <- 0

ISSA_obj <- ISSA_func(ht_constraint, L, delta, gamma_target,
                      symmetric_target, a1, b1, lambda_start)

bk_obj     <- ISSA_obj$bk_obj    # Main I-SSA object
gamma_mse  <- ISSA_obj$gamma_mse # MSE-optimal filter coefficients (λ = 0 benchmark)
b_x        <- ISSA_obj$b_x       # Optimised I-SSA filter coefficients
lambda_opt<-ISSA_obj$lambda_opt  # optimal Lagrangian multiplier: lambda_opt=0 corresponds to the MSE benchmark.
ht_issa<-bk_obj$ht_issa          # HT of optimized I-SSA: should match ht_constraint

# ── Theoretical MSE (measured against the MSE optimal nowcast ─────
bk_obj$mse_yz * sigma_ip^2
# ── Convergence check ─────────────────────────────────────────────
abs(ht_issa - ht_constraint)
# Cointegration check: difference should vanish
sum(b_x-gamma_mse)


# ────────────────────────────────────────────────────────────────
# 3.2 Plot filters
# ────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "green", "blue", "red")

mplot <- cbind(
  hp_two,
  c(gamma_mse, rep(0, L - 1)),
  c(b_x,       rep(0, L - 1)),
  c(hp_c,  rep(0, L - 1))
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

# I-SSA coefficients decay slower than in exercise 1 due to stronger smoothing

# ────────────────────────────────────────────────────────────────
# 3.3 Filter Data and Plot in Levels
# ────────────────────────────────────────────────────────────────
y_ssa            <- filter(x_tilde, b_x,       side = 1)
y_hp_concurrent  <- filter(x_tilde, hp_c,  side = 1)
y_mse            <- filter(x_tilde, gamma_mse, side = 1)
y_hp_two         <- filter(x_tilde, hp_two,    side = 2)


colo <- c("black", "violet", "green", "blue", "red")

# Plot: levels
par(mfrow = c(1, 1))
# Define the plotting window: skip the first L observations (burn-in period)
# and plot through to the end of the sample.
anf <- L 
enf <- length(x_tilde)
mplot <- cbind(x_tilde, y_hp_two, y_mse, y_ssa, y_hp_concurrent)[anf:enf, ]
colnames(mplot) <- c("Data", "Target: HP-two", "MSE: HP-one", "I-SSA", "HP-C")

plot(mplot[, 1], main = "Data and trends", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = colo[1], lwd = 1)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[anf:length(y_xts)])
axis(2); box()

# First informal inspection:
# I-SSA is very smooth but not systematically faster than HP-C 
# (as was the case in exercise 1)
# We will emphasize TPs in the next plot, but previously we check 
# sample performances.


# ────────────────────────────────────────────────────────────────
# 3.4 Sample Performance Evaluation
# ────────────────────────────────────────────────────────────────

# ············································
# 3.4.1 Tracking Accuracy (Level MSE)
# ············································
# Compute the sample MSE of each nowcast relative to the two-sided HP trend,
# which serves as the infeasible but optimal benchmark.

# Calibration logic (dual formulation of I-SSA):
#   For a given MSE budget matched to HP-C, I-SSA is designed to maximise HT.
#   As a consistency check, the MSE of I-SSA should therefore not exceed that
#   of HP-C, and should ideally be close to it. This would indicate that the
#   rule of thumb used in calibration (increasing ht_constraint by 50%) is
#   reasonably well tuned.

# Benchmark target: two-sided HP trend shifted by delta
if (delta >= 0)
  y_target <- c(y_hp_two[(1 + delta):length(y_hp_two)], rep(NA, delta))
if (delta < 0)
  y_target <- c(rep(NA, -delta), y_hp_two[1:(length(y_hp_two) + delta)])

mean((y_target - y_ssa)^2, na.rm = TRUE)            # I-SSA nowcast
mean((y_target - y_hp_concurrent)^2, na.rm = TRUE)  # Standard one-sided HP-C

# Since I-SSA still delivers a lower MSE than HP-C, the resulting gain in
# smoothness should be interpreted as conservative: additional smoothness could
# likely be achieved by calibrating the MSE constraint more tightly to HP-C.

# ············································
# 3.4.2 Smoothness: Holding Time in First Differences
# ············································

# I-SSA: HT maximised subject to the HP-C MSE budget
sample_ht_issa <- compute_empirical_ht_func(scale(diff(y_ssa)[anf:enf]))$empirical_ht
sample_ht_issa

# HP-C: smoothness benchmark
sample_ht_hpc <- compute_empirical_ht_func(scale(diff(y_hp_concurrent)[anf:enf]))$empirical_ht
sample_ht_hpc

# Percentage improvement in HT achieved by I-SSA over HP-C
paste(round(100 * (sample_ht_issa - sample_ht_hpc) / sample_ht_hpc, 2), "%", sep = "")

# We now emphasize TPs of the two nowcasts in the following two-panel plot.

# ──────────────────────────────────────────────────────────────────────────
# 3.5 Plot: Highlight Turning Points (TPs) — I-SSA Maximal Monotone Property
# ──────────────────────────────────────────────────────────────────────────

colo <- c("blue", "red")

# Define the plotting window: skip the first L observations (burn-in period)
# and plot through to the end of the sample.
par(mfrow = c(2, 1))
anf <- L 
enf <- length(x_tilde)
# Combine I-SSA and HP-C outputs into a single matrix for joint plotting
mplot_tp <- cbind(y_ssa, y_hp_concurrent)[anf:enf, ]
colnames(mplot_tp) <- c("I-SSA", "HP-C")

# ── Top Panel: I-SSA only, with its Turning Points marked ──────
plot(mplot_tp[, 1],
     main   = "I-SSA with Turning Points (blue dashed vertical lines)",
     axes   = FALSE, type = "l",
     xlab   = "", ylab  = "",
     col    = colo[1], lwd = 1,
     ylim   = c(min(mplot_tp), max(mplot_tp)))

# Detect Turning Points (TPs) for HP-C and I-SSA.
# A TP occurs where the sign of successive first differences reverses,
# i.e., where diff[t] * diff[t+1] < 0 (a local peak or trough).
TP_hp   <- which(diff(mplot_tp[1:(nrow(mplot_tp) - 1), "HP-C"])  *
                   diff(mplot_tp[2:nrow(mplot_tp),       "HP-C"])  < 0)
TP_issa <- which(diff(mplot_tp[1:(nrow(mplot_tp) - 1), "I-SSA"]) *
                   diff(mplot_tp[2:nrow(mplot_tp),       "I-SSA"]) < 0)
# Overlay I-SSA Turning Points as blue dashed vertical lines
abline(v = 1 + TP_issa, col = "blue", lty = 2)

# Add time-axis labels and box
axis(1, at = 1:nrow(mplot_tp),
     labels = index(y_xts)[anf:length(y_xts)])
axis(2); box()

# ── Bottom Panel: I-SSA and HP-C overlaid, with TPs of both series ──
plot(mplot_tp[, 1],
     main   = "I-SSA (blue) and HP-C (red) with Turning Points",
     axes   = FALSE, type = "l",
     xlab   = "", ylab  = "",
     col    = colo[1], lwd = 1,
     ylim   = c(min(mplot_tp), max(mplot_tp)))

# Draw both series and add in-plot legend via marginal text
for (i in 1:ncol(mplot_tp)) {
  lines(mplot_tp[, i], col = colo[i])
  mtext(colnames(mplot_tp)[i], line = -i, col = colo[i])
}
# Detect TPs for both filters (same logic as top panel)
TP_hp   <- which(diff(mplot_tp[1:(nrow(mplot_tp) - 1), "HP-C"])  *
                   diff(mplot_tp[2:nrow(mplot_tp),       "HP-C"])  < 0)
TP_issa <- which(diff(mplot_tp[1:(nrow(mplot_tp) - 1), "I-SSA"]) *
                   diff(mplot_tp[2:nrow(mplot_tp),       "I-SSA"]) < 0)
# Overlay TPs: I-SSA (blue dashed) and HP-C (red dotted)
abline(v = 1 + TP_issa, col = "blue", lty = 2)
abline(v = 1 + TP_hp,   col = "red",  lty = 3)
# Add time-axis labels and box
axis(1, at = 1:nrow(mplot_tp),
     labels = index(y_xts)[anf:length(y_xts)])
axis(2); box()

# ── Turning Point Counts ────────────────────────────────────────
length(TP_hp)    # Number of TPs for HP-C
length(TP_issa)  # Number of TPs for I-SSA (expected to be fewer)
# HP-C generates more TPs than I-SSA despite achieving similar (slightly
# worse) MSE tracking accuracy. This illustrates the maximal monotone property
# of I-SSA: among all linear predictors (nowcasts) with the same MSE tracking 
# error, I-SSA maximises the mean duration between consecutive TPs, given a 
# sufficiently long sample and a reasonably well-specified model.

# ── Interpretation of the Two Panels ───────────────────────────
# Top panel:    I-SSA series with its own TPs marked (blue dashed lines).
# Bottom panel: Both I-SSA and HP-C, with HP-C TPs (red dotted) overlaid
#               on I-SSA TPs (blue dashed).
#
# ──────────────────────────────────────────────────────────────────────────
# Main Take-Aways:
# ──────────────────────────────────────────────────────────────────────────
#
# 1. Maximal Monotone Property:
#    For equal or slightly better MSE, I-SSA produces fewer TPs than HP-C.
#    No other linear predictor achieves a longer mean duration between TPs
#    under the same conditions (sufficiently long sample, well-specified model,
#    identical MSE tracking error), see Wildi 2026a.
#
# 2. Timeliness at TPs:
#    Despite its greater smoothness (fewer TPs), I-SSA does not suffer from
#    excessive lag at turning points. Across the sample, I-SSA is:
#      - Coincident with HP-C at 5 TPs,
#      - Leading HP-C at 1 TP, and
#      - Lagging HP-C at 4 TPs.
#    This confirms that smoothness gains are not achieved at the cost of
#    excessive timeliness losses.

# In the last exercise we consider an inverse formulation, imposing 
# a smaller (rather than larger)  HT than HP-C. 

# ========================================================================
# Exercise 4: I-SSA Double-Stroke Revisited
# ========================================================================

# We here illustrate that I-SSA can outperform HP-C in MSE tracking accuracy 
# on stationary levels, as well as in recession signaling on stationary 
# first differences (growth). For this purpose we design I-SSA with a HT in 
# between the optimal MSE level tracker and HP-C's own HT. We then compute 
# MSE performances as well as recession signaling based on the ROC curve.

# NOTE: This exercise depends on the Wold decomposition xi defined in Exercise 1.
# If Exercise 1 has not been run yet, please do so before proceeding.

# Building on the setup from Exercise 1, we modify only the HT parameter.

# Set the HT to a value midway the HT of HP_c (~10) and the MSE optimal 
# nowcast (~2) in Exercise 1.
ht_constraint <- (10+2)/2
ht_constraint

# ────────────────────────────────────────────────────────────────
# 4.1 I-SSA Optimisation
# ────────────────────────────────────────────────────────────────
# Same settings as in exercise 1
delta <- 0
gamma_target     <- hp_two
symmetric_target <- TRUE
lambda_start <- 0

ISSA_obj <- ISSA_func(ht_constraint, L, delta, gamma_target,
                      symmetric_target, a1, b1, lambda_start)

bk_obj     <- ISSA_obj$bk_obj    # Main I-SSA object
gamma_mse  <- ISSA_obj$gamma_mse # MSE-optimal filter coefficients (λ = 0 benchmark)
b_x        <- ISSA_obj$b_x       # Optimised I-SSA filter coefficients
lambda_opt<-ISSA_obj$lambda_opt  # optimal Lagrangian multiplier: lambda_opt=0 corresponds to the MSE benchmark.
ht_issa<-bk_obj$ht_issa          # HT of optimized I-SSA: should match ht_constraint

# ── Theoretical MSE ───────────────────────────────────────────────
bk_obj$mse_yz * sigma_ip^2
# ── Convergence check ─────────────────────────────────────────────
abs(ht_issa - ht_constraint)
# Cointegration check: difference should vanish
sum(b_x-gamma_mse)


# ────────────────────────────────────────────────────────────────
# 4.2 Plot filters
# ────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "green", "blue", "red")
mplot <- cbind(
  hp_two,
  c(gamma_mse, rep(0, L - 1)),
  c(b_x,       rep(0, L - 1)),
  c(hp_c,  rep(0, L - 1))
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

# I-SSA coefficients decay faster than in the previous exercises (smaller HT)

# ────────────────────────────────────────────────────────────────
# 4.3 Filter Data and Plot in Levels
# ────────────────────────────────────────────────────────────────
y_ssa            <- filter(x_tilde, b_x,       side = 1)
y_hp_concurrent  <- filter(x_tilde, hp_c,  side = 1)
y_mse            <- filter(x_tilde, gamma_mse, side = 1)
y_hp_two         <- filter(x_tilde, hp_two,    side = 2)


colo <- c("black", "violet", "green", "blue", "red")

# Plot: levels
par(mfrow = c(1, 1))
# Define the plotting window: skip the first L observations (burn-in period)
# and plot through to the end of the sample.
anf <- L 
enf <- length(x_tilde)
mplot <- cbind(x_tilde, y_hp_two, y_mse, y_ssa, y_hp_concurrent)[anf:enf, ]
colnames(mplot) <- c("Data", "Target: HP-two", "MSE: HP-one", "I-SSA", "HP-C")

plot(mplot[, 1], main = "Data and trends", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = colo[1], lwd = 1)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}
axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[anf:length(y_xts)])
axis(2); box()

# ── Preliminary Visual Inspection ──────────────────────────────
# Before computing formal performance metrics, an initial visual inspection
# of the two filters suggests the following:
#   - I-SSA is less smooth than HP-C but responds faster and adapts more
#     readily to narrow, short-lived recessionary dips in the series.
#
# We now proceed to compute and compare formal sample performance statistics
# for both filters.

# ────────────────────────────────────────────────────────────────
# 4.4 Sample Performance Evaluation
# ────────────────────────────────────────────────────────────────

# ············································
# 4.4.1 Tracking Accuracy (Level MSE)
# ············································
# Compute the sample MSE of each nowcast relative to the two-sided HP
# trend (the infeasible but optimal benchmark target).
#
# Design intention (dual formulation of I-SSA):
#   For a given HT, I-SSA maximises tracking accuracy.
#   Since the imposed HT is smaller than HP-C, I-SSA should substantially 
#   outperform HP-C in terms of MSE.

# Target: two-sided HP shifted by delta
if (delta>=0)
  y_target<-c(y_hp_two[(1+delta):length(y_hp_two)],rep(NA,delta))
if (delta<0)
  y_target<-c(rep(NA,-delta),y_hp_two[1:(length(y_hp_two)+delta)])

mse_issa<-mean((y_target - y_ssa)^2, na.rm = TRUE)            # I-SSA nowcast
mse_hpc<-mean((y_target - y_hp_concurrent)^2, na.rm = TRUE)  # Classic one-sided HP-C

mse_issa
mse_hpc
# MSE Ratio (relative to HP-C): even better than in exercise 1.
round(mse_issa/mse_hpc,2)


# ············································
# 4.4.2 Smoothness: Holding Time in First Differences
# ············································

# I-SSA: HT maximised subject to the HP-C MSE budget
sample_ht_issa <- compute_empirical_ht_func(scale(diff(y_ssa)[anf:enf]))$empirical_ht
sample_ht_issa

# HP-C: smoothness benchmark
sample_ht_hpc <- compute_empirical_ht_func(scale(diff(y_hp_concurrent)[anf:enf]))$empirical_ht
sample_ht_hpc

# Smaller HT (the cost of improved MSE):
sample_ht_issa/sample_ht_hpc






# ─────────────────────────────────────────────────────────────────────────────
# 4.5 NBER Recession Detection
# ─────────────────────────────────────────────────────────────────────────────
# We evaluate the ability of each trend nowcast to detect NBER-dated recessions 
# using Receiver Operating Characteristic (ROC) curves and the Area Under the 
# Curve (AUC). First differences of the filter outputs serve as recession 
# signals. AUC summarises overall detection skill — a value of 1 indicates 
# perfect discrimination; 0.5 indicates no skill beyond random guessing.
#
# ─────────────────────────────────────────────────────────────────────────────
# 4.5.1 NBER dating
# ─────────────────────────────────────────────────────────────────────────────

output_mat <- cbind(y_ssa, y_hp_concurrent, y_target)
dates<-as.character(index(y_xts))
rownames(output_mat)<-dates
colnames(output_mat)<-c("I-SSA","HP one-sided","HP two-sided")
tail(output_mat)

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
# 4.5.2 Compare NBER Dating with Differenced nowcasts 
# ─────────────────────────────────────────────────────────────────────────────

# The recession indicator serves as the binary classification target
target <- NBER_recessions


# Build the ROC data matrix: binary target + first-differenced filter outputs.
#  -Remove first element of target to match length of first differences.
# First differences convert non-stationary levels to stationary growth signals
# that can be thresholded to produce recession/expansion calls.
ROC_data_all <- cbind(target[-1], apply(output_mat[, c("I-SSA", "HP one-sided")], 2, diff))
rownames(ROC_data_all) <- rownames(apply(output_mat, 2, diff))
colnames(ROC_data_all) <- c("Target", colnames(output_mat[, c("I-SSA", "HP one-sided")]))

# We can either use full data set
ROC_data <- ROC_data_all
# Or emphasize post 2000 data (the AUC is only marginally affected)
year_2000<-which(index(y_xts)>"2000-01-01")[1]
# Restrict to post-2000 sub-sample for comparability with HT analysis and to
# avoid the high-growth pre-2000 period, which has a different drift regime
# (mitigate non-stationarity)
ROC_data <- ROC_data_all[year_2000:nrow(ROC_data_all), ]
head(ROC_data)
tail(ROC_data)
# Compare NBER datings with differenced trend nowcasts: 
#   Both nowcasts track official NBER datings well.
#   I-SSA seems systematically faster at recoveries.
ts.plot(scale(ROC_data,center=F,scale=T),col=c("black","blue","red"),main="Recessions vs. differenced smoothers")
mtext("I-SSA",col="blue",line=-1)
mtext("HP one-sided",col="red",line=-2)



# ─────────────────────────────────────────────────────────────────────────────
# 4.5.3 ROC AnalysisC
# ─────────────────────────────────────────────────────────────────────────────
smoothROC <- T   # Apply smoothing to ROC curves for cleaner visualisation
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


# ──────────────────────────────────────────────────────────────────────────
# Main Take-Aways: I-SSA Double-Stroke Design
# ──────────────────────────────────────────────────────────────────────────
#
# 1. Tracking Accuracy:
#    I-SSA can outperform HP-C in level tracking accuracy (i.e., yield a better
#    level nowcast) and in recession signaling based on first differences 
#    (growth rates).
#
# 2. HT Constraint — Rule of Thumb:
#    As a practical guideline, the HT constraint can be set
#    to a value between the optimal MSE level tracker and the chosen benchmark
#    (here HP-C), offering a principled starting point for calibration.
#
# 3. Smoothness vs. Adaptiveness Trade-Off:
#    The resulting I-SSA filter will be less smooth (i.e., exhibit more Turning
#    Points in levels) but faster and more adaptive to structural changes in the
#    series. This trade-off can be advantageous in terms of the balance between
#    false alarms and hit rates, as reflected in the ROC curve and AUC metric.
#
# 4. Continued Relevance of the Maximal Monotone Design (Exercise 3):
#    Despite the benefits of the Double-Stroke design, the maximal monotone
#    I-SSA from Exercise 3 remains relevant in its own right. Its strong noise
#    suppression — manifested as long mean durations between Turning Points —
#    makes it well suited as a confirmatory analytical tool, complementing the
#    more adaptive Double-Stroke variant.


# ============================================================
# Summary Tutorial 6
# ============================================================
# 1. Extension to Non-Stationary Series:
#    We introduced the Integrated SSA (I-SSA),
#    which extends the original SSA framework to non-stationary I(1)
#    (integrated) time series. An extension to I(2) processes is derived in 
#    Wildi 2026a.
#
# 2. Maximal Monotone: 
#    I-SSA generalises the HT-based smoothness criterion of SSA to the
#    property of maximal monotonicity in non-stationary levels, ensuring the 
#    longest expected time between consecutive crossover or turning points 
#    among all linear predictors operating under the same MSE-tracking 
#    constraint.
#
# 3. Double-Stroke:
#    I-SSA addresses two objectives within a single, coherent indicator:
#    (a) Optimised tracking accuracy in levels (level nowcasting), and
#    (b) Efficient detection of downturns and upturns in first differences
#        (i.e., growth rate signals), achieved by imposing a Holding-Time
#        (HT) constraint on the filter output.
#    Traditionally, these objectives have been pursued through separate,
#    purpose-built filters whose designs may be mutually inconsistent.
#    For instance, a filter optimised purely for MSE-based level tracking
#    may produce excessive noise or spurious Turning Points in first
#    differences — as illustrated by the MSE-optimal design in the
#    exercises above — and, conversely, a filter designed for smooth
#    Turning Point detection may sacrifice level tracking accuracy.
#    By jointly addressing both objectives within a single optimisation
#    framework, I-SSA offers a principled alternative that mitigates this
#    inconsistency, potentially reducing or eliminating the need for
#    parallel, specialised filter designs.
#
# ============================================================
#  Outlook: I-SSA Trend
# ============================================================
#    Tutorial 9 will explore these themes further from a complementary
#    perspective. A key distinction is that it omits any specification of
#    a pre-defined target filter — such as the HP filter used here — which
#    served as an artificial structuring element in the present Tutorial 6.
#    Removing this constraint allows the I-SSA framework to be developed
#    in a more self-contained and assumption-lean manner, relying solely
#    on the properties of the data and the optimisation criteria, rather
#    than on the choice of a reference benchmark. This leaner formulation
#    gives rise to the so-called I-SSA Trend — a novel, data-driven trend
#    concept that emerges directly from the I-SSA optimisation framework
#    without reference to any pre-specified filter target.

# Meanwhile, Tutorial 7 introduces M-SSA, a generalization of SSA to a 
# multivariate stationary framework.  




