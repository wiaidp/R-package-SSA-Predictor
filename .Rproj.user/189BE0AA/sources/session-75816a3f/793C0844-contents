# =============================================================================
# TUTORIAL 2.0: BUSINESS-CYCLE ANALYSIS (BCA) — FOUNDATION FOR SSA INTEGRATION
# =============================================================================
# NOTE: This tutorial is independent of the SSA.
#
# FOCUS: The Hodrick-Prescott (HP) filter as a target specification for
#        trend and cycle extraction.
# Readers/coders interested in SSA could skip this tutorial

#
# FILTER PERSPECTIVE:
#   - Primary emphasis is placed on ONE-SIDED filters (also known as
#     'causal', 'concurrent', or 'real-time' filters), which are central
#     to our forecasting and prediction objectives within SSA.
#   - TWO-SIDED (symmetric) filters are also examined, as the one-sided
#     filters are formally derived from them.
# =============================================================================


# =============================================================================
# OUTLINE
# =============================================================================
#
# 1. BACKGROUND: ORIGINS OF THE APPROACH
#    - Whittaker-Henderson smoothing and signal extraction
#
# 2. ONE- AND TWO-SIDED HP TREND FILTERS
#    - Derivation and properties of the HP trend filter
#
# 3. ONE- AND TWO-SIDED HP GAP FILTERS
#    - Typical application in BCA (cyclical component extraction)
#
# 4. FREQUENCY-DOMAIN ANALYSIS
#    - Amplitude functions and time-shift (phase delay) properties
#
# 5. SPECTRAL DENSITY OF THE EXTRACTED CYCLE — METHOD 1
#    - HP trend filter applied to first-differenced data
#
# 6. SPECTRAL DENSITY OF THE EXTRACTED CYCLE — METHOD 2
#    - Original HP gap filter applied to data in levels
#
# 7. COMPARISON: HP GAP (LEVELS) vs. HP TREND (DIFFERENCES)
#    - The HP trend applied to differences offers notable advantages
#      over the standard HP gap, with only minor trade-offs
#    - This design serves as the basis for SSA integration in Tutorial 2.1
#
# 8. SUMMARY
#    - Key findings and recommendations
#
# =============================================================================



#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions
rm(list=ls())

library(xts)
# Load the library mFilter
# Standard R-package for HP and other filters 
library(mFilter)
# McElroys package for HP
source(paste(getwd(),"/R utility functions/hpFilt.r",sep=""))
# Load all relevant SSA-functions
source(paste(getwd(),"/R/ssa.r",sep=""))


# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R utility functions/HP_JBCY_functions.r",sep=""))

# Load data from FRED with library quantmod
library(quantmod)
# Download Non-farm payroll and INDPRO
getSymbols('PAYEMS',src='FRED')
getSymbols('INDPRO',src='FRED')

# We now develop points 1-8 listed above 
################################################################################
################################################################################
# =============================================================================
# 1. BACKGROUND
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 DERIVATION OF THE HP FILTER: WHITTAKER-HENDERSON SMOOTHING
# -----------------------------------------------------------------------------
# The HP trend is obtained by solving the following penalized least-squares
# (Whittaker-Henderson) optimization problem:
#
#   min_{y_t} sum_{t=1}^{T} (x_t - y_t)^2
#              + lambda * sum_{t=d+1}^{T} ((1-B)^d * y_t)^2
#
# Interpretation:
#   - The FIRST term penalizes deviations of the trend y_t from the data x_t
#     (fidelity to the data).
#   - The SECOND term penalizes roughness of y_t via d-th order differences
#     (smoothness of the trend).
#   - lambda controls the smoothness-fidelity trade-off:
#       * Large lambda  => smoother trend (penalizes roughness more heavily)
#       * Small lambda  => trend tracks data more closely
#   - Setting d = 2 yields the classical HP trend filter.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# IMPLEMENTATION: HP FILTER VIA THE mFilter PACKAGE
# -----------------------------------------------------------------------------

# --- Filter Length -----------------------------------------------------------
# Specify the number of filter coefficients.
# IMPORTANT: L must be an odd number so that the two-sided (symmetric) HP
#            filter can be properly centered around the current time point t.
L <- 201

# Enforce odd filter length with a warning if violated
if (L / 2 == as.integer(L / 2)) {
  print("Warning: Filter length should be an odd number.")
  print("An even L prevents the two-sided HP filter from being centered.")
  L <- L + 1
}

# --- Smoothing Parameter -----------------------------------------------------
# Lambda calibration follows the convention for MONTHLY data.
# Standard values by data frequency:
#   Monthly:   lambda = 14,400
#   Quarterly: lambda =  1,600  (Hodrick & Prescott, 1997)
#   Annual:    lambda =    100
lambda_monthly <- 14400

# --- Compute HP Filter Object ------------------------------------------------
# HP_target_mse_modified_gap() wraps mFilter and additionally derives
# several HP-based designs discussed later in this tutorial.
par(mfrow = c(1, 1))
HP_obj <- HP_target_mse_modified_gap(L, lambda_monthly)

# --- Extract the Two-Sided (Symmetric) HP Trend Filter -----------------------
# hp_target contains the coefficients of the BI-INFINITE two-sided HP trend.
# Properties:
#   - SYMMETRIC (two-sided): uses observations both before AND after time t
#     => acausal / non-real-time filter (not suitable for real-time estimation)
#   - Serves as the TARGET from which one-sided (causal) approximations
#     are derived (see exercises 2,3 below)
#   - This is a FINITE approximation of length L to the bi-infinite filter
hp_target <- HP_obj$target

# --- Plot Filter Coefficients ------------------------------------------------
ts.plot(hp_target,
        main = paste("Two-sided HP trend filter  |  lambda =",
                     lambda_monthly, sep = " "))

# --- Verify Lowpass Property -------------------------------------------------
# Filter coefficients sum to 1, confirming this is a LOWPASS filter:
#   => Passes low-frequency components (trend) with unit gain
#   => Attenuates high-frequency components (cycles, noise)
# See amplitude functions in exercise 4 for a full frequency-domain analysis.
sum(hp_target)

# =============================================================================
# 1.2 ALTERNATIVE DERIVATION OF THE HP FILTER: MSE SIGNAL EXTRACTION
# =============================================================================
# The HP filter has an important statistical interpretation:
#
# It is the OPTIMAL MEAN-SQUARED ERROR (MSE) signal extraction filter for
# the trend component in Harvey's (1989) 'smooth trend' state-space model.
#
# THE IMPLICIT (LATENT) DATA-GENERATING MODEL:
# ----------------------------------------------------------------------------
#   x_t = T_t + sqrt(lambda) * epsilon_t
#
#   where:
#     x_t     = observed data
#     T_t     = latent trend ~ ARIMA(0,2,0)  [i.e., (1-B)^2 * T_t = eta_t]
#     epsilon_t = i.i.d. white noise (observation disturbance), typically standardized
#     eta_t   = i.i.d. white noise (level disturbance), typically standardized (same variance as epsilon_t)
#     lambda  = smoothing parameter (signal-to-noise ratio)
#
# IMPLICATION FOR THE CYCLE:
#   The implied 'cycle' x_t - T_t = sqrt(lambda) * epsilon_t is pure
#   white noise — a well-known limitation of the HP framework (see exercise 7
#   for a discussion of this and other drawbacks).
#
# ARIMA REPRESENTATION:
#   The reduced form of this model follows an ARIMA(0,2,2) process.
#   The MA(2) coefficients are determined analytically by lambda,
#   as derived in McElroy (2006).
# =============================================================================

# --- Simulate Data from the Implicit HP Model --------------------------------
# We simulate a realization of the latent HP model using lambda = 14,400
# (monthly calibration established in exercise 1.1).
set.seed(23)
len <- 12000

# x_t = ARIMA(0,2,0) trend + scaled white noise
# cumsum(cumsum(.)) generates the I(2) trend component T_t
x <- cumsum(cumsum(rnorm(len))) + rnorm(len) * sqrt(lambda_monthly)
ts.plot(x,
        main = "Simulated HP implicit model: I(2) trend + scaled noise")

# --- Inspect First Differences -----------------------------------------------
# First differences show a slowly drifting, non-stationary pattern,
# confirming the I(2) nature of the DGP.
ts.plot(diff(x),
        main = "First differences of x: non-stationary drift visible")

# --- ACF of First Differences ------------------------------------------------
# Key features:
#   - Negative lag-1 autocorrelation: induced by first-differencing the
#     white noise component (MA signature)
#   - Weak but persistent positive autocorrelations beyond lag 1:
#     residual non-stationarity from the I(2) trend — differences alone
#     are insufficient to achieve stationarity
acf(diff(x),
    main = "ACF of first differences: residual non-stationarity")

# --- ACF of Second Differences -----------------------------------------------
# After second-order differencing:
#   - Autocorrelations decay to zero beyond lag 2
#   - Lags 1 and 2 reflect the MA(2) signature from double-differencing
#     the noise component — consistent with ARIMA(0,2,2) structure
acf(diff(diff(x)),
    main = "ACF of second differences: MA(2) pattern at lags 1-2")

# =============================================================================
# ARIMA(0,2,2) REPRESENTATION
# =============================================================================
# The ACF pattern motivates fitting an ARIMA(0,2,2) model.
# Per McElroy (2006), the MA(2) parameters are EXACT analytical functions
# of lambda — we verify this by comparing estimated vs. theoretical values.

# --- Estimate ARIMA(0,2,2) from Simulated Data --------------------------------
arima(x, order = c(0, 2, 2))

# --- Compute Exact Theoretical MA Parameters via McElroy (2006) --------------
# hpFilt() parameterizes the filter using q = 1/lambda (inverse signal-to-noise)
q <- 1 / lambda_monthly
hp_filt_obj <- hpFilt(q, L)

# HP filter coefficients (left/causal tail only)
head(hp_filt_obj$filter_coef)

# ARIMA(0,2,2) MA parameter vector returned by hpFilt():
#   Element 1: normalizing constant
#   Elements 2-3: theoretical MA(1) and MA(2) coefficients
hp_filt_obj$ma_model

# Extract the two MA parameters for direct comparison
ma <- hp_filt_obj$ma_model[2:3]
ma

# --- Verification ------------------------------------------------------------
# The ARIMA(0,2,2) estimates from the simulated data should fall within
# the 95% confidence interval of the theoretical MA parameters.
# This confirms McElroy's (2006) analytical result for this lambda.
arima(x, order = c(0, 2, 2))
#-------------------------------
# =============================================================================
# 1.3 SIMULATING FROM THE IMPLICIT HP MODEL: DOES IT RESEMBLE ECONOMIC DATA?
# =============================================================================
# If real economic series are statistically indistinguishable from simulations
# of the implicit HP model, then HP would be the OPTIMAL filter for trend
# extraction. However, this would simultaneously imply that the business cycle
# is pure white noise — an economically implausible conclusion.
#
# We investigate this by:
#   (i)  Simulating realizations of the ARIMA(0,2,2) implicit model
#   (ii) Comparing them visually and statistically with key US macro indicators
# =============================================================================

# =============================================================================
# SINGLE REALIZATION: MODEL DIAGNOSTICS
# =============================================================================
set.seed(1)

# Reference series: Monthly US Industrial Production Index (INDPRO)
#   Source: https://fred.stlouisfed.org/series/INDPRO
#   Available from: January 1920
#   Length: approximately 100 years * 12 months = 1,200 observations
#
# NOTE ON INITIALIZATION:
#   Integrated (I(2)) processes have INFINITE memory — the starting values
#   have a permanent effect on all subsequent observations.
#   We initialize at zero here (x_0 = x_{-1} = 0) to match
#   a standard double-difference recursion: x_1 = 2*x_0 - x_{-1}
len <- 1200
x <- cumsum(cumsum(rnorm(len))) + rnorm(len) * sqrt(lambda_monthly)

# --- Verify ARIMA(0,2,2) Fit -------------------------------------------------
# Estimated MA parameters should be close to the theoretical values
# derived from lambda (see exercise 1.2 and McElroy, 2006)
arima(x, order = c(0, 2, 2))

# --- Model Diagnostics -------------------------------------------------------
# Standardized residuals, ACF, and Ljung-Box statistics should show
# no systematic misspecification
tsdiag(arima(x, order = c(0, 2, 2)))

# --- Visual Inspection -------------------------------------------------------
ts.plot(x, main = "Single realization of ARIMA(0,2,2) implicit HP model")
# Observation: the series does not resemble a typical economic time series —
# it lacks directional structure, exhibits arbitrary drift, and the
# 'cycle' component is merely noise.

# =============================================================================
# MULTIPLE REALIZATIONS: ENSEMBLE COMPARISON
# =============================================================================
anzsim <- 5
mat_sim <- NULL
set.seed(96)

for (i in 1:anzsim) {
  x <- cumsum(cumsum(rnorm(len))) + rnorm(len) * sqrt(lambda_monthly)
  mat_sim <- cbind(mat_sim, x)
}

# --- Plot Levels -------------------------------------------------------------
# Simulated series appear 'flexing' (large stochastic trend) and noisy
# (the cycle is pure white noise, visible as high-frequency noise around the trend)
ts.plot(mat_sim,
        col  = rainbow(anzsim),
        main = "Multiple realizations of ARIMA(0,2,2) implicit HP model")

# --- Plot First Differences --------------------------------------------------
# First differences remain non-stationary (slowly drifting) due to the
# remaining I(1) component. The noise from double-differencing the white
# noise cycle dominates the short-run dynamics.
ts.plot(apply(mat_sim, 2, diff),
        col  = rainbow(anzsim),
        main = "First differences of ARIMA(0,2,2) realizations")
abline(h = 0)

# =============================================================================
# EFFECT OF INITIALIZATION ON I(2) PROCESSES
# =============================================================================
# Integrated processes never 'forget' their starting values.
# Here we illustrate the effect of a non-zero initialization:
#   x_0 = 1,000  (instead of 0),  x_{-1} = 0
# This introduces a permanent linear drift of 1,000 per period.
set.seed(1)
x <- cumsum(1000 + cumsum(rnorm(len))) + rnorm(len) * sqrt(lambda_monthly)

ts.plot(x,
        main = "Effect of non-zero initialization: x_0 = 1,000, x_{-1} = 0")
# Result: the series appears almost linear — the initialization induces
# a deterministic-looking drift that persists indefinitely.

ts.plot(diff(x),
        main = "First differences with non-zero initialization")
# First differences are shifted upward by the initialization value (1,000)
# but continue to drift slowly due to the remaining I(1) component.

# =============================================================================
# VISUAL COMPARISON: SIMULATED vs. REAL US MACRO INDICATORS
# =============================================================================
# Indicators used:
#   PAYEMS: US Non-Farm Payroll Employment (monthly, FRED)
#   INDPRO: US Industrial Production Index  (monthly, FRED)
#
# We compare three transformations:
#   (a) Levels             — to assess overall structural plausibility
#   (b) Log levels         — to stabilize variance (log-linearization)
#   (c) Log first diff.    — approximate monthly growth rates ('returns')

# --- (a) Levels --------------------------------------------------------------
par(mfrow = c(2, 2))
ts.plot(mat_sim,
        xlab = "", col = rainbow(anzsim),
        main = "Simulated: ARIMA(0,2,2) levels")
plot(PAYEMS, main = "Non-Farm Payroll (levels)")
plot(INDPRO,  main = "Industrial Production (levels)")
# Key differences:
#   - Simulated series lack recession episodes and directional structure
#   - Growth sign is random; drift can grow arbitrarily large in absolute value
#   - Real series show persistent upward trends with identifiable business-cycle features

# --- (b) Log Levels ----------------------------------------------------------
# Log transformation is recommended for the real series to stabilize variance:
#   variance of real data scales with the level, whereas the simulated
#   noise variance is level-independent by construction.
par(mfrow = c(2, 2))
ts.plot(mat_sim,
        xlab = "", col = rainbow(anzsim),
        main = "Simulated: ARIMA(0,2,2) levels")
plot(log(PAYEMS), main = "Log Non-Farm Payroll")
plot(log(INDPRO),  main = "Log Industrial Production")
# Log-transformed real series display more regular (long-term) growth than simulated data.
# Adjusting initial values could partly reconcile the levels, but would
# compromise smoothness — no parameterization of the implicit model
# can fully replicate the structure of real economic series.

# --- (c) Log First Differences (approximate growth rates) -------------------
par(mfrow = c(2, 2))
ts.plot(apply(mat_sim, 2, diff),
        xlab = "", col = rainbow(anzsim),
        main = "Simulated: first differences")
abline(h = 0)
plot(diff(log(PAYEMS)), main = "Monthly growth: Non-Farm Payroll")
plot(diff(log(INDPRO)),  main = "Monthly growth: Industrial Production")
# Simulated first differences are substantially noisier 
# (less persistent) than the real macro growth rates (ignoring non-stationarity and extreme COVID outliers).

# =============================================================================
# FOCUSED COMPARISON: 1990–2019 (PRE-PANDEMIC WINDOW)
# =============================================================================
# Subsetting to a common 30-year window removes COVID-19 distortions and
# allows a cleaner structural comparison on matching sample lengths.
par(mfrow = c(2, 2))

ts.plot(apply(mat_sim, 2, diff)[(len - 29 * 12):(len - 1), ],
        xlab = "", col = rainbow(anzsim),
        main = "Simulated: first differences (1990–2019 equivalent)")
abline(h = 0)

ts.plot(diff(log(PAYEMS["1990/2019"])),
        main = "Non-Farm Payroll growth (1990–2019)",
        xlab = "", ylab = "")
abline(h = 0)

ts.plot(diff(log(INDPRO)["1990/2019"]),
        main = "Industrial Production growth (1990–2019)",
        xlab = "", ylab = "")
abline(h = 0)
# Simulated data lacks cyclical structure and is substantially noisier
# than either real series.

# =============================================================================
# ZERO-CROSSING ANALYSIS: HOLDING TIMES
# =============================================================================
# The empirical holding time (average run length between sign changes)
# quantifies persistence in the sign of first differences.
#
# Benchmark:
#   - Pure white noise => holding time = 2 (sign changes every 2 periods on average)
#   - Positively autocorrelated series => holding time > 2
#   - Negatively autocorrelated series => holding time < 2

# Simulated data (one realization, 1990-equivalent window)
compute_empirical_ht_func(diff(mat_sim[(len - 29 * 12):(len - 1), 1]))

# Real macro indicators (1990–2019)
compute_empirical_ht_func(as.double(diff(PAYEMS["1990/2019"])))
compute_empirical_ht_func(as.double(diff(INDPRO["1990/2019"])))

# Results interpretation:
#   - Simulated series: holding time < 2 => NEGATIVELY autocorrelated
#     (dominated by differenced white noise from the implicit cycle)
#   - Real macro indicators: holding time > 2 => POSITIVELY autocorrelated
#     (genuine business-cycle persistence)
#   - Non-Farm Payroll is notably smoother than INDPRO, suggesting
#     stronger low-frequency persistence — differences in data-generating
#     structure that require explicit model fitting
#     (addressed in Tutorials 2.1, 3, and 4)

# =============================================================================
# 2. HP TREND FILTER: TWO-SIDED AND ONE-SIDED DESIGNS
# =============================================================================
# RECAP FROM EXERCISE 1.3:
#   Comparing the implicit HP model simulations with real macro data revealed
#   two key discrepancies:
#     (a) Simulated data lacks economic structure (no recessions/expansions)
#     (b) Simulated data is substantially noisier than real macro series
#
# IMPLICATION FOR FILTER DESIGN:
#   If the implicit model were the true DGP, the HP trend filter would be
#   well-specified: its sole purpose would be aggressive noise suppression,
#   with no need to preserve cyclical structure.
#
#   Applied to REAL macro data, however, this strong smoothing becomes
#   problematic: it can attenuate or entirely wash out economically
#   meaningful features such as recession troughs and expansion peaks.
#   See Phillips and Jin (2021) for a formal treatment of this issue.
# =============================================================================

# =============================================================================
# 2.1 TWO-SIDED (SYMMETRIC) HP TREND FILTER
# =============================================================================

# --- Smoothing Strength: Holding Time ----------------------------------------
# The holding time measures the MEAN DURATION between consecutive zero-crossings
# of the filter output when applied to white noise input.
# A large holding time indicates strong low-frequency pass-through
# (aggressive smoothing / high noise suppression).
compute_holding_time_func(hp_target)$ht
# Result: very large holding time => extremely strong smoothing effect.
# This confirms that the two-sided HP trend is a (possibly too) powerful lowpass filter.

# LIMITATION OF THE TWO-SIDED FILTER:
#   The symmetric filter requires observations on BOTH sides of time t.
#   => It cannot be applied at or near the END of the sample.
#   => It is unsuitable for real-time or concurrent estimation.
#   A one-sided (causal) filter is required for these applications.

# =============================================================================
# 2.2 ONE-SIDED (CAUSAL) HP TREND FILTER
# =============================================================================
# The optimal one-sided HP trend filter is derived under the assumption that
# the DGP follows the ARIMA(0,2,2) model implied by the chosen lambda.
# 'Optimal' here means minimum MSE among all causal filters, given that DGP.
hp_trend <- HP_obj$hp_trend

par(mfrow = c(1, 1))
ts.plot(hp_trend,
        main = paste("Optimal one-sided HP trend  |  ARIMA(0,2,2) DGP  |  lambda =",
                     lambda_monthly, sep = " "))

# --- Verify Lowpass Property -------------------------------------------------
# Coefficients sum to 1 => unit gain at frequency zero => lowpass filter
# (see exercise 4 for full frequency-domain analysis)
sum(hp_trend)

# --- Holding Time: One-sided vs. Two-sided ------------------------------------
compute_holding_time_func(hp_trend)$ht
# The one-sided filter has a CONSIDERABLY SHORTER holding time than the
# two-sided filter. This reflects 'noise leakage':
#   - The causal filter has no future information available
#   - It cannot suppress noise as aggressively as the symmetric filter
#   - This trade-off between real-time availability and noise suppression (or lag)
#     is a fundamental property of causal signal extraction

# =============================================================================
# 2.3 FILTER APPLICATION: SIMULATED DATA
# =============================================================================
set.seed(18)
len <- 1200

# Simulate one realization of the implicit HP model (ARIMA(0,2,2) DGP)
x <- cumsum(cumsum(rnorm(len))) + rnorm(len) * sqrt(lambda_monthly)

# --- Apply Both Filters ------------------------------------------------------
# side = 1: one-sided (causal)   — uses only past and current observations
# side = 2: two-sided (centered) — uses past, current, and future observations
y_hp_concurrent <- filter(x, hp_trend,  sides = 1)
y_hp_symmetric  <- filter(x, hp_target, sides = 2)

# --- Plot: Side-by-Side Comparison -------------------------------------------
ts.plot(y_hp_concurrent,
        main = "HP trend: two-sided (black) vs. one-sided (blue)",
        col  = "blue")
lines(y_hp_symmetric)
abline(h = 0)
mtext("Two-sided HP (symmetric)", col = "black", line = -1)
mtext("One-sided HP (causal)",    col = "blue",  line = -2)
# The two outputs closely overlap for most of the sample.
# The key advantage of the one-sided filter is that it EXTENDS TO THE SAMPLE
# END, whereas the two-sided filter produces NAs in the boundary region.

# =============================================================================
# 2.4 FILTER APPROXIMATION ERROR
# =============================================================================
# The approximation error measures how closely the one-sided filter
# tracks the infeasible two-sided benchmark.
error <- y_hp_symmetric - y_hp_concurrent

ts.plot(error,
        main = "Approximation error: two-sided minus one-sided HP trend")
# The error appears stationary and cyclical in character.
# By construction (MSE optimality of the one-sided filter under the
# ARIMA(0,2,2) DGP), no alternative causal filter can achieve a
# smaller approximation error. The two trend series are COINTEGRATED.

# --- Mean Squared Error ------------------------------------------------------
MSE_error <- mean(error^2, na.rm = TRUE)
MSE_error
# This is the MINIMUM achievable MSE for any one-sided filter applied
# to data generated by the ARIMA(0,2,2) model with this lambda.

# =============================================================================
# 2.5 THE HP GAP: RESIDUAL FROM THE TREND
# =============================================================================
# The HP gap is defined as: gap_t = x_t - trend_t
# Under the implicit model, this should recover the white noise cycle component.
gap <- x - y_hp_symmetric

ts.plot(gap,
        main = "HP gap (two-sided): x_t minus HP trend")

# --- Verify Noise Variance ---------------------------------------------------
# The empirical variance of the gap should approximate lambda_monthly
# (the noise variance in the implicit model).
# Convergence to the true value improves with sample length.
mean(gap^2, na.rm = TRUE)
# Compare with: lambda_monthly = 14,400: the mean converges to lambda_monthly asymptotically

# --- ACF of the Gap ----------------------------------------------------------
# Under the implicit model, the gap should be WHITE NOISE (i.i.d.).
# Significant autocorrelation would indicate model misspecification
# or the presence of genuine cyclical structure beyond white noise.
acf(na.exclude(gap),
    main = "ACF of HP gap: testing for white noise residuals")

################################################################################
################################################################################
# =============================================================================
# 3. HP GAP FILTER
# =============================================================================
# In standard BCA applications, the CYCLE is extracted using the HP GAP filter,
# defined as the complement of the HP trend:
#
#   HP_gap = Identity - HP_trend
#   => gap_t = x_t - trend_t
#
# INTERPRETATION UNDER THE IMPLICIT MODEL:
#   Since x_t = T_t + sqrt(lambda) * epsilon_t, the gap x_t - T_t recovers
#   the white noise component sqrt(lambda) * epsilon_t.
#
# POLICY RELEVANCE:
#   - Positive gap (x_t > trend): economy is above potential
#     => overheating, possibly requiring contractionary policy
#   - Negative gap (x_t < trend): economy is below potential
#     => slack, possibly requiring expansionary policy
#   - If the gap were truly white noise, no systematic policy response
#     would be warranted (purely random fluctuations around trend)
#
# NOTE: The implausibility of a white noise cycle is one of the key
#       limitations of the HP framework identified in Exercise 1.3.
# =============================================================================

# =============================================================================
# 3.1 TWO-SIDED (SYMMETRIC) HP GAP FILTER
# =============================================================================
# Construct the two-sided HP gap as: Identity - HP_trend (two-sided)
# The identity filter is represented as a unit spike at the center lag
hp_gap_sym <- c(rep(0, (L - 1) / 2), 1, rep(0, (L - 1) / 2)) - hp_target

ts.plot(hp_gap_sym,
        main = "Two-sided HP gap filter coefficients")

# --- Apply to Simulated Data -------------------------------------------------
y_hp_gap_symmetric <- filter(x, hp_gap_sym, sides = 2)

ts.plot(y_hp_gap_symmetric,
        main = "Two-sided HP gap output (implicit model: should approximate white noise)")
# Under the ARIMA(0,2,2) DGP, the gap should recover the white noise component.
# The HP gap filter effectively CANCELS THE UNIT ROOTS of the ARIMA(0,2,2):
# the output is stationary, despite the I(2) nature of the input.

# --- ACF: Verify White Noise -------------------------------------------------
acf(na.exclude(y_hp_gap_symmetric),
    main = "ACF of two-sided HP gap output")
# Near-zero autocorrelations at all lags confirm approximate white noise behavior
# under the correctly specified implicit model.

# =============================================================================
# 3.2 ONE-SIDED (CAUSAL) HP GAP FILTER
# =============================================================================
# Construct the one-sided HP gap as: Identity - HP_trend (one-sided)
# The identity filter is a unit spike at lag zero (current observation only)
hp_gap <- c(1, rep(0, L - 1)) - hp_trend

ts.plot(hp_gap,
        main = "One-sided HP gap filter coefficients")

# --- Apply to Simulated Data -------------------------------------------------
y_hp_gap_concurrent <- filter(x, hp_gap, sides = 1)

ts.plot(y_hp_gap_concurrent,
        main = "One-sided HP gap output (implicit model: should approximate white noise)")
# The causal HP gap also cancels the unit roots and produces a stationary output,
# but with somewhat more noise leakage than its two-sided counterpart
# (consistent with the holding-time comparison in Exercise 2.2).

# --- ACF: Verify White Noise -------------------------------------------------
acf(na.exclude(y_hp_gap_concurrent),
    main = "ACF of one-sided HP gap output")

# =============================================================================
# 3.3 MSE COMPARISON: ONE-SIDED vs. TWO-SIDED HP GAP
# =============================================================================
# The MSE between the two gap outputs equals MSE_error from Exercise 2.4.
# REASON: subtracting the same identity filter from both trend filters
#         cancels out, leaving only the trend approximation error.
#
#   (gap_sym - gap_concurrent) = (x - trend_sym) - (x - trend_concurrent)
#                               = trend_concurrent - trend_sym
#                               = -error  (from Exercise 2.4)
mean((y_hp_gap_concurrent - y_hp_gap_symmetric)^2, na.rm = TRUE)
# This equals MSE_error computed in Exercise 2.4 — confirming the identity above.
#
# By the MSE optimality of the one-sided HP trend under the ARIMA(0,2,2) DGP,
# no alternative causal gap filter can achieve a smaller approximation error
# relative to the two-sided benchmark.

# =============================================================================
# 4. FREQUENCY-DOMAIN ANALYSIS: AMPLITUDE AND TIME-SHIFT FUNCTIONS
# =============================================================================
# We now characterize all four filters in the FREQUENCY DOMAIN by computing:
#   - AMPLITUDE function: gain applied to each frequency component
#     (1 = pass through unchanged, 0 = fully suppressed)
#   - TIME-SHIFT function: phase delay divided by frequency
#     (non-zero shift => filter output is time-displaced relative to input)
#
# Two-sided symmetric filters have zero time-shift (phase = 0 at all freq.)
# One-sided causal filters generally introduce non-zero time-shifts.
# =============================================================================

# --- Frequency Grid ----------------------------------------------------------
# K equidistant ordinates spanning [0, pi]
# Higher K => finer frequency resolution
K <- 600

# --- Compute Transfer, Amplitude and Shift Functions -------------------------
amp_obj_hp_trend_concurrent <- amp_shift_func(K, hp_trend,   F)
amp_obj_hp_trend_sym         <- amp_shift_func(K, hp_target,  F)
amp_obj_hp_gap_sym           <- amp_shift_func(K, hp_gap_sym, F)
amp_obj_hp_gap_concurrent    <- amp_shift_func(K, hp_gap,     F)

# =============================================================================
# 4.1 AMPLITUDE FUNCTIONS: ALL FOUR FILTERS
# =============================================================================
par(mfrow = c(1, 1))

mplot <- cbind(
  amp_obj_hp_trend_concurrent$amp,
  amp_obj_hp_trend_sym$amp,
  amp_obj_hp_gap_sym$amp,
  amp_obj_hp_gap_concurrent$amp
)
colnames(mplot) <- c(
  "HP trend (one-sided)",
  "HP trend (two-sided)",
  "HP gap  (two-sided)",
  "HP gap  (one-sided)"
)

colo <- c("blue", rainbow(ncol(mplot)))

# --- Base plot: one-sided HP trend -------------------------------------------
plot(mplot[, 1],
     type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude functions: HP trend and HP gap filters",
     ylim = c(min(mplot), max(mplot)),
     col  = colo[1])

# Mark the peak amplitude frequency of the one-sided HP trend
# (corresponds to the dominant business-cycle periodicity)
abline(v   = which(mplot[, 1] == max(mplot[, 1])),
       col = colo[1],
       lty = 2)

mtext(colnames(mplot)[1], line = -1, col = colo[1])

# --- Overlay remaining filters -----------------------------------------------
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

# --- Axes --------------------------------------------------------------------
axis(1,
     at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# =============================================================================
# 4.2 INTERPRETATION OF AMPLITUDE FUNCTIONS
# =============================================================================
# HP TREND FILTERS (lowpass):
#   - Both pass low frequencies (trend) and suppress high frequencies (noise)
#   - Two-sided HP trend: very steep roll-off => extremely strong smoothing
#       * Near-zero amplitude at business-cycle and higher frequencies
#       * Consistent with the large holding time observed in Exercise 2.1
#   - One-sided HP trend: more gradual roll-off => some noise leakage
#       * Still effective at suppressing high-frequency components
#       * Trade-off: real-time availability vs. noise suppression
#
# HP GAP FILTERS (highpass):
#   - Complementary to the trend filters (amplitude = 1 - trend amplitude)
#   - Suppress low-frequency trend components; pass higher frequencies
#
# PEAK AMPLITUDE AND BUSINESS-CYCLE PERIODICITY:
#   The one-sided HP trend reaches its peak amplitude (vertical dashed line)
#   at the periodicity computed below. This corresponds roughly to the
#   canonical 5-8 year business-cycle frequency band.

# Compute the periodicity (in months) at which the one-sided HP trend
# achieves its maximum amplitude
2 * (K - 1) / (which(mplot[, 1] == max(mplot[, 1])) - 1)
# Result: approximately 85 months (~7 years) => consistent with
# the typical duration of business cycles in monthly macro data.


# =============================================================================
# 4.3 TIME-SHIFT (PHASE-SHIFT) FUNCTIONS
# =============================================================================
# The time-shift function measures the DELAY (in time periods) introduced
# by each filter at each frequency:
#
#   time-shift(omega) = phase(omega) / omega
#
# where phase(omega) is the argument of the complex transfer function.
#
# INTERPRETATION:
#   - time-shift = 0: no delay => filter output is synchronous with input
#   - time-shift > 0: filter output LAGS the input (delayed detection)
#   - time-shift < 0: filter output LEADS the input (anticipative)
#
# KEY DISTINCTION:
#   All filters as stored are ONE-SIDED (causal) coefficient vectors.
#   When the two-sided filters are APPLIED in practice (centered at time t),
#   their phase is zero by symmetry => no time-shift in actual use.
#   The non-zero shifts shown below for two-sided designs reflect only the
#   one-sided representation of those coefficient vectors.
# =============================================================================

# --- Assemble Time-Shift Matrix ----------------------------------------------
mplot <- cbind(
  amp_obj_hp_trend_concurrent$shift,
  amp_obj_hp_trend_sym$shift,
  amp_obj_hp_gap_sym$shift,
  amp_obj_hp_gap_concurrent$shift
)
colnames(mplot) <- c(
  "HP trend (one-sided)",
  "HP trend (two-sided)",
  "HP gap  (two-sided)",
  "HP gap  (one-sided)"
)

# --- Plot: All Four Filters --------------------------------------------------
plot(mplot[, 1],
     type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time-shift (periods)",
     main = "Time-shift functions: HP trend and HP gap filters",
     ylim = c(min(mplot), max(mplot)),
     col  = colo[1])

mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

axis(1,
     at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# =============================================================================
# INTERPRETATION: ALL FOUR FILTERS
# =============================================================================
# TWO-SIDED DESIGNS (HP trend and HP gap, symmetric):
#   - Stored as one-sided vectors => apparent shift = (L-1)/2 periods
#     (half the filter length), reflecting only the storage convention.
#   - When APPLIED as centered (acausal) filters, their phase is zero
#     by symmetry => no time-shift in actual BCA applications.
#
# ONE-SIDED HP GAP (causal):
#   - Exhibits a NEGATIVE time-shift => anticipative behavior.
#   - However, the one-sided HP gap provides no noise suppression
#     (it passes high-frequency components with near-unit amplitude,
#     see Exercise 4.1)
#
# ONE-SIDED HP TREND (causal):
#   - The most relevant filter for real-time applications.
#   - Analyzed in isolation below.
# =============================================================================

# --- Plot: One-Sided HP Trend Only -------------------------------------------
plot(mplot[, 1],
     type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time-shift (periods)",
     main = "Time-shift function: one-sided HP trend filter",
     ylim = c(min(mplot[, 1]), max(mplot[, 1])),
     col  = colo[1])

mtext(colnames(mplot)[1], line = -1, col = colo[1])

axis(1,
     at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# =============================================================================
# 4.4 KEY PROPERTY: ZERO TIME-SHIFT AT FREQUENCY ZERO
# =============================================================================
# The one-sided HP trend has a VANISHING TIME-SHIFT AT FREQUENCY ZERO.
#
# This is a remarkable and practically important property:
#
#   (i)  TIMELINESS:
#          A near-zero shift at low frequencies means the filter detects
#          peaks and troughs of the trend with minimal delay.
#          This is unusual for causal lowpass designs, which typically
#          introduce substantial lag at trend frequencies.
#
#   (ii) THEORETICAL NECESSITY:
#          The zero shift at frequency zero is not a coincidence — it is
#          a mathematical requirement for I(2) processes.
#          If the one-sided filter had a non-zero phase at omega = 0,
#          the approximation error between the one-sided and two-sided
#          trend estimates would be I(1), i.e., NON-STATIONARY.
#          Stationarity of the error is required for the MSE optimality
#          criterion to be well-defined (see Exercise 2.4).
#
#   (iii) IMPLICATION FOR SSA:
#          This zero-shift property is a key target criterion when
#          designing SSA-based real-time filters in Tutorial 2.1.
#          SSA filters that preserve this property will inherit the
#          timeliness advantage of the optimal one-sided HP trend.
# =============================================================================


# =============================================================================
# 5. Spectral Densities of HP-Trend Filters
# =============================================================================
# Goal:
#   Compare the spectral densities of cycles extracted by one-sided and 
#   two-sided HP-trend filters when applied to first differences (returns) 
#   of macroeconomic data.
#
# This section is organized as follows:
#   5.1 Modelling the Data Generating Process (DGP)
#   5.2 Derivation of the spectral density of the extracted cycles


# -----------------------------------------------------------------------------
# 5.1 Modelling the DGP: Motivating the I(1) Assumption
# -----------------------------------------------------------------------------
# We assume that typical (possibly log-transformed) economic time series follow
# an I(1) process. Below, we provide three lines of motivation for this assumption.

# --- (a) Visual inspection of returns ---
# Plot the returns (first differences of log-levels) and their ACFs.
# If the series is I(1), returns should fluctuate around a stable mean
# with no persistent autocorrelation structure.
par(mfrow=c(2,2))
plot(diff(log(INDPRO)["1960/2019"]), xlab="", ylab="", 
     main="Returns: Indpro")
plot(diff(log(PAYEMS)["1960/2019"]), xlab="", ylab="", 
     main="Returns: Non-Farm Payroll")
acf(na.exclude(diff(log(INDPRO)["1960/2019"])), 
    main="ACF: Indpro Returns")
acf(na.exclude(diff(log(PAYEMS)["1960/2019"])), 
    main="ACF: Non-Farm Payroll Returns")
# Observation:
#   Returns show no pronounced level shifts, broadly consistent with I(1) levels.
#   Note: pre-1990 data may behave differently due to the Great Moderation.

# --- (b) Stationary ARMA models fit the differenced data well ---
# Simple ARMA(2,1) models are estimated on the returns of both series.
# Satisfactory diagnostics (residual whiteness) further support the 
# I(1) assumption.
model_indpro <- arima(diff(log(INDPRO)["1960/2019"]), order=c(2,0,1))
model_payems <- arima(diff(log(PAYEMS)["1960/2019"]), order=c(2,0,1))
tsdiag(model_indpro)
tsdiag(model_payems)

# --- (c) Theoretical motivation ---
# Economic theory suggests that a broad range of macroeconomic and financial
# variables — including stock prices, futures prices, long-term interest rates,
# oil prices, consumption, inflation, tax rates, and money supply growth —
# should behave as (near) martingales:
#   Fama (1965), Samuelson (1965), Sargent (1976), Hamilton (2009),
#   Hall (1978), Mankiw (1987).

# Conclusion:
#   We proceed under the I(1) assumption, which stands in contrast to the
#   ARIMA(0,2,2) DGP implicitly assumed by HP to justify filter optimality.
#
# Implications for applying HP-trend to differenced (I(0)) data:
#   (a) The one-sided HP filter does not severely distort business-cycle 
#       extraction from differenced data: its amplitude peaks near a 
#       7-year periodicity.
#   (b) The two-sided HP filter over-smooths at business-cycle frequencies,
#       damping them excessively — see also Phillips and Jin (2021).

# -----------------------------------------------------------------------------
# Wold Decomposition of the Fitted ARMA Models
# -----------------------------------------------------------------------------
# To derive the spectral density of the extracted cycles, we need to convolve
# the HP filter coefficients with the Wold (MA-infinity) representation of
# the differenced data. This is because the HP filter is applied to returns,
# which can be expressed as filtered white noise via the Wold decomposition.

# Compute the Wold (MA-infinity) coefficients for each fitted model:
ar <- model_indpro$coef[1:2]
ma <- model_indpro$coef[3]
  xi_indpro <- c(1, ARMAtoMA(lag.max=L-1, ar=ar, ma=ma))

ar <- model_payems$coef[1:2]
ma <- model_payems$coef[3]
  xi_payems <- c(1, ARMAtoMA(lag.max=L-1, ar=ar, ma=ma))

# Plot the Wold coefficients for both series side by side:
par(mfrow=c(1,1))
ts.plot(cbind(xi_indpro, xi_payems),
        main="Wold Decomposition Coefficients: Indpro vs. PAYEMS",
        col=c("violet","orange"))
mtext("Wold decomposition: Indpro",  col="violet", line=-1)
mtext("Wold decomposition: PAYEMS",  col="orange", line=-2)

# -----------------------------------------------------------------------------
# Convolution of HP Filter with Wold Decomposition
# -----------------------------------------------------------------------------
# The convolution of the HP filter weights with the Wold coefficients yields
# a composite filter. When applied to white-noise residuals, this composite
# filter reproduces the same output as applying HP directly to the returns.

hp_one_sided_conv_indpro <- conv_two_filt_func(xi_indpro, hp_trend)$conv
hp_two_sided_conv_indpro <- conv_two_filt_func(xi_indpro, hp_target)$conv
hp_one_sided_conv_payems <- conv_two_filt_func(xi_payems, hp_trend)$conv
hp_two_sided_conv_payems <- conv_two_filt_func(xi_payems, hp_target)$conv

# --- Verification ---
# We verify the convolution equivalence for Indpro:
#   Applying hp_one_sided_conv to model residuals should match
#   applying hp_trend directly to the returns.
#   Note: the intercept is added to re-center the zero-mean residuals.
y_hp_conv <- filter(model_indpro$residuals, hp_one_sided_conv_indpro, side=1) +
  model_indpro$coef["intercept"]
y_hp      <- filter(diff(log(INDPRO)["1960/2019"]), hp_trend, side=1)

# Both series should overlap exactly — confirming the convolution identity:
ts.plot(cbind(y_hp_conv, y_hp),
main="Verification: Convolution vs. Direct HP Filter (series should overlap)")


# =============================================================================
# 5.2 Spectral Densities of the Extracted Cycles
# =============================================================================
# The spectral density of a cycle extracted by hp_trend equals the squared 
# amplitude of the composite filter hp_one_sided_conv.
#
# Key assumptions and conventions:
#   - The composite filter is applied to white-noise residuals, whose spectral
#     density is flat (constant). Hence the cycle's spectral density is
#     proportional to the squared amplitude of the composite filter.
#   - We omit the scaling factor sigma^2 / (2*pi) for clarity.
#   - Spectral densities are normalized (scaled) to facilitate visual comparison.
#   - Two vertical reference lines mark the business-cycle frequency band:
#     periodicities of 2 years (higher freq.) and 10 years (lower freq.).

# --- One-sided HP trend ---
amp_obj_one_sided_hp_conv_indpro <- amp_shift_func(K, hp_one_sided_conv_indpro, F)
amp_obj_one_sided_hp_conv_payems <- amp_shift_func(K, hp_one_sided_conv_payems, F)

par(mfrow=c(1,1))
plot(scale(amp_obj_one_sided_hp_conv_indpro$amp^2, center=F, scale=T),
     type="l", col="violet", axes=F, xlab="Frequency", ylab="",
     main="Scaled Spectral Density: Cycle Extracted by One-Sided HP Trend")
lines(scale(amp_obj_one_sided_hp_conv_payems$amp^2, center=F, scale=T), col="orange")
abline(v=K/12)   # 2-year periodicity boundary
abline(v=K/60)   # 10-year periodicity boundary
mtext("One-sided HP trend applied to differences of Indpro", col="violet", line=-1)
mtext("One-sided HP trend applied to differences of PAYEMS", col="orange", line=-2)
axis(1, at=1+0:6*K/6,
     labels=expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Interpretation:
#   - As expected, the shape of the spectral density depends on the assumed DGP;
#     this DGP-dependence is often overlooked or ignored in applied work.
#   - For Indpro: the spectral density peaks within the business-cycle band,
#     suggesting reasonable cycle extraction from a BCA perspective.
#   - For PAYEMS: the peak is shifted toward lower frequencies, indicating
#     that the extracted cycle may be overly smooth (too long-period).
#   - Since filters are applied to differenced (I(0)) data, they are extracting
#     the *differenced* business cycle. A non-flat spectral density in the
#     business-cycle band is desirable; whether it should vanish at frequency
#     zero depends on further modeling considerations (see below).

# --- Two-sided HP trend ---
amp_obj_two_sided_hp_conv_indpro <- amp_shift_func(K, hp_two_sided_conv_indpro, F)
amp_obj_two_sided_hp_conv_payems <- amp_shift_func(K, hp_two_sided_conv_payems, F)

par(mfrow=c(1,1))
plot(scale(amp_obj_two_sided_hp_conv_indpro$amp^2, center=F, scale=T),
     type="l", col="violet", axes=F, xlab="Frequency", ylab="",
     main="Scaled Spectral Density: Cycle Extracted by Two-Sided HP Trend")
lines(scale(amp_obj_two_sided_hp_conv_payems$amp^2, center=F, scale=T), col="orange")
abline(v=K/12)   # 2-year periodicity boundary
abline(v=K/60)   # 10-year periodicity boundary
mtext("Two-sided HP trend applied to differences of Indpro", col="violet", line=-1)
mtext("Two-sided HP trend applied to differences of PAYEMS", col="orange", line=-2)
axis(1, at=1+0:6*K/6,
     labels=expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Interpretation:
#   - The spectral densities of the two-sided filter are shifted toward lower
#     frequencies relative to the one-sided case.
#   - This indicates that the two-sided HP trend over-smooths the data,
#     effectively suppressing business-cycle frequency components —
#     consistent with the findings of Phillips and Jin (2021).


# =============================================================================
# 6. Spectral Densities of the Original HP-Gap Filter
# =============================================================================
# Goal:
#   Compare the spectral densities of cycles extracted by one-sided HP-gap 
#   filters applied to macroeconomic data in levels.
#
# Key distinction from Section 5:
#   The HP-trend filter (Section 5) was applied to first differences (log-diff),
#   whereas the HP-gap filter is applied directly to data in levels.
#
# This section is organized as follows:
#   6.1 Modelling the DGP          — see Section 5.1 (I(1) assumption)
#   6.2 Handling non-stationarity  — transform the gap filter to work with 
#       differences
#   6.3 Derivation of spectral densities via the Wold decomposition


# -----------------------------------------------------------------------------
# 6.1 DGP Assumption
# -----------------------------------------------------------------------------
# As motivated in Section 5.1, we assume all series follow an I(1) process.
# See Section 5.1 for the full motivation.


# -----------------------------------------------------------------------------
# 6.2 Transforming the HP-Gap Filter for Differenced Data
# -----------------------------------------------------------------------------
# The standard HP-gap filter is designed to operate on data in levels.
# Since our DGP is I(1), applying filters directly to non-stationary levels
# complicates spectral analysis. We therefore transform the HP-gap filter
# into an equivalent filter that operates on first differences,
# while producing identical output to the original gap filter applied to levels.
#
# This transformation is performed via conv_with_unitroot_func:
#   Reference: Wildi, M. (2024), https://doi.org/10.1007/s41549-024-00097-5
hp_gap_diff <- conv_with_unitroot_func(hp_gap)$conv

# --- Visual comparison: original vs. transformed HP-gap filter ---
# Plot both filter coefficient sequences side by side to illustrate
# how the transformation redistributes the filter weights.
ts.plot(cbind(hp_gap, hp_gap_diff),
        col=c("red","darkred"),
        main=paste("HP-gap: Original (levels) vs. Transformed (differences), lambda=",
                   lambda_monthly, sep=""))
mtext("Original HP-gap: applied to levels",      col="red",     line=-1)
mtext("Transformed HP-gap: applied to differences", col="darkred", line=-2)

# --- Comparison with the HP-trend filter (both applied to differences) ---
# After scaling, the transformed HP-gap filter bears a notable resemblance
# to the HP-trend filter — both now operate on first-differenced data.
ts.plot(scale(cbind(hp_trend, hp_gap_diff), scale=T, center=F),
        col=c("blue","darkred"),
        main=paste("Transformed HP-gap vs. HP-trend (both applied to differences), lambda=",
                   lambda_monthly, sep=""))
mtext("HP-trend: applied to differences",          col="blue",    line=-1)
mtext("Transformed HP-gap: applied to differences", col="darkred", line=-2)

# --- Verification: equivalence of original and transformed filters ---
# We verify that applying hp_gap_diff to first differences produces output
# identical to applying the original hp_gap to log-levels.
y_hp_gap      <- filter(log(INDPRO)["1960/2019"],            hp_gap,      side=1)
y_hp_gap_diff <- filter(diff(log(INDPRO)["1960/2019"]),      hp_gap_diff, side=1)

# The two series should overlap exactly.
# Consequently, their spectral densities must also be identical.
ts.plot(cbind(y_hp_gap, y_hp_gap_diff),
        main="Verification: Original HP-gap (levels) vs. Transformed HP-gap (differences) — series should overlap")


# -----------------------------------------------------------------------------
# 6.3 Spectral Densities of the HP-Gap Cycle
# -----------------------------------------------------------------------------
# Now that the HP-gap has been expressed as a filter on stationary differences,
# we can derive the spectral density of the extracted cycle by convolving
# hp_gap_diff with the Wold decomposition of each series.
# (This mirrors the approach used for HP-trend in Section 5.2.)

# --- Convolve the transformed HP-gap with the Wold decompositions ---
hp_gap_diff_one_sided_conv_indpro <- conv_two_filt_func(xi_indpro, hp_gap_diff)$conv
hp_gap_diff_one_sided_conv_payems <- conv_two_filt_func(xi_payems, hp_gap_diff)$conv

# --- Compute amplitude functions (squared = spectral density, up to scaling) ---
amp_obj_hp_gap_diff_conv_indpro <- amp_shift_func(K, hp_gap_diff_one_sided_conv_indpro, F)
amp_obj_hp_gap_diff_conv_payems <- amp_shift_func(K, hp_gap_diff_one_sided_conv_payems, F)

# --- Plot scaled spectral densities ---
# Vertical reference lines mark the business-cycle frequency band:
#   K/12  → 2-year periodicity (upper frequency boundary)
#   K/60  → 10-year periodicity (lower frequency boundary)
par(mfrow=c(1,1))
plot(scale(amp_obj_hp_gap_diff_conv_indpro$amp^2, center=F, scale=T),
     type="l", axes=F, xlab="Frequency", col="darkred", ylab="",
     main="Scaled Spectral Density: Cycle Extracted by One-Sided HP-Gap Filter")
lines(scale(amp_obj_hp_gap_diff_conv_payems$amp^2, center=F, scale=T), col="brown")
mtext("Spectral density: HP-gap cycle, Indpro", line=-1, col="darkred")
mtext("Spectral density: HP-gap cycle, PAYEMS", line=-2, col="brown")
abline(v=K/12)   # 2-year periodicity boundary
abline(v=K/60)   # 10-year periodicity boundary
axis(1, at=1+0:6*K/6,
     labels=expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Interpretation:
#   - The HP-gap applied to PAYEMS produces a smoother cycle than when applied to Indpro.
#   - This reflects the DGP: the ARMA model fitted to PAYEMS acts as a more effective
#     low-pass filter than the one fitted to Indpro, meaning its Wold decomposition
#     more aggressively attenuates high-frequency innovations.
#   - In plain terms: PAYEMS is inherently smoother than Indpro, so applying
#     the same filter to both will yield a smoother cycle for the smoother series.
#   - This illustrates a fundamental point: spectral properties of extracted cycles
#     depend critically on the DGP of the input series, not just the filter design.


# -----------------------------------------------------------------------------
# 6.4 Direct Comparison: HP-Gap Cycles for Indpro vs. PAYEMS
# -----------------------------------------------------------------------------
# We now visually compare the two extracted HP-gap cycles over the sample period.

y_hp_gap_payems <- filter(log(PAYEMS)["1960/2019"], hp_gap, side=1)
mplot <- scale(cbind(y_hp_gap, y_hp_gap_payems), center=F, scale=T)

plot(mplot[,1],
     main="HP-Gap Cycles: Indpro vs. PAYEMS",
     col="darkred", axes=F, xlab="", ylab="", type="l")
lines(mplot[,2], col="green")
mtext("HP-gap cycle: Indpro", line=-1, col="darkred")
mtext("HP-gap cycle: PAYEMS", line=-2, col="green")
abline(h=0)
axis(1, at=12*(1:(nrow(mplot)/12)),
     labels=index(diff(log(INDPRO)["1960/2019"]))[12*(1:(nrow(mplot)/12))])
axis(2)
box()

# Observations:
#   - The PAYEMS-based cycle is substantially smoother than the Indpro-based cycle.
#   - The twin recessions of the early 1980s appear as two distinct contractions
#     in the Indpro cycle, but merge into a single "double-dip" recession in
#     the PAYEMS cycle.
#   - The PAYEMS cycle systematically lags behind the Indpro cycle, reflecting
#     the sluggishness of labor market adjustments relative to industrial output.
#   - Post-financial crisis behavior diverges: the PAYEMS cycle shows a more
#     monotonic recovery trajectory, whereas Indpro exhibits more oscillation.
#
# Caution:
#   Applying the same HP filter (lambda = 14400) to series with qualitatively
#   different dynamic properties is not advisable. The extracted cycles reflect
#   both the filter design and the underlying DGP, and conflating these two
#   sources of smoothing can lead to misleading interpretations.





# =============================================================================
# 7. Comparison: HP-Trend Applied to Differences vs. Original HP-Gap Applied to Levels
# =============================================================================
# Goal:
#   Directly compare the one-sided HP-trend filter (applied to first differences)
#   with the original HP-gap filter (applied to levels) in terms of:
#     7.1 Spectral densities of the extracted cycles
#     7.2 Filter outputs, zero-crossings, and holding times


# -----------------------------------------------------------------------------
# 7.1 Comparison of Spectral Densities
# -----------------------------------------------------------------------------
# We overlay the scaled spectral densities of the two extracted cycles for Indpro.
# (A qualitatively similar result would be obtained for PAYEMS.)
#
# Vertical reference lines mark the business-cycle frequency band:
#   K/12  → 2-year periodicity (upper frequency boundary)
#   K/60  → 10-year periodicity (lower frequency boundary)

par(mfrow=c(1,1))
plot(scale(amp_obj_hp_gap_diff_conv_indpro$amp^2, center=F, scale=T),
     type="l", axes=F, xlab="Frequency", col="darkred", ylab="",
     main=paste("Spectral Densities: One-Sided HP-Gap vs. HP-Trend, lambda=",
                lambda_monthly, sep=""))
lines(scale(amp_obj_one_sided_hp_conv_indpro$amp^2, center=F, scale=T), col="violet")
mtext("Spectral density: HP-gap applied to levels of Indpro",       line=-1, col="darkred")
mtext("Spectral density: HP-trend applied to differences of Indpro", line=-2, col="violet")
abline(v=K/12)   # 2-year periodicity boundary
abline(v=K/60)   # 10-year periodicity boundary
axis(1, at=1+0:6*K/6,
     labels=expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Interpretation:
#   The two spectral densities are broadly similar, with one key difference:
#   - HP-trend (applied to differences): its spectral density does NOT vanish at
#     frequency zero, meaning low-frequency (trend) information passes through.
#     The extracted cycle is slightly smoother and more heavily weighted toward
#     lower frequencies — an effect comparable in magnitude to the difference
#     between the HP-gap cycles of PAYEMS and Indpro observed in Section 6.
#   - HP-gap (applied to levels): its spectral density vanishes at frequency zero,
#     ensuring the extracted cycle has zero mean.
#
# Which cycle is more appropriate? The answer depends on the analytical purpose:
#
#   HP-trend (differences):
#     + Better suited for tracking long expansion episodes, since low-frequency
#       (level) information is preserved rather than suppressed.
#     + Less prone to generating false turning-point signals during prolonged
#       expansions (e.g., it will not prematurely signal a downturn mid-expansion).
#     - Acts as a growth tracker rather than a classic cycle indicator:
#         * Its mean level equals the drift of the original series (non-zero
#           unless the series is flat).
#         * It does not automatically revert to zero.
#         * It is not strictly stationary: if the drift or slope of the original
#           series changes over time, the level of the extracted "cycle" shifts
#           accordingly.
#
#   HP-gap (levels):
#     + Possesses all the classical cycle characteristics: zero mean, stationarity,
#       and automatic mean reversion.
#     + Better resolves short expansion episodes (e.g., the twin recessions
#       of the early 1980s).
#     + Smaller phase shift / lag: the gap filter tends to anticipate zero-crossings
#       of the trend.
#     - More prone to generating spurious zero-crossings during long expansions,
#       potentially signalling recessions several years before they materialise.
#     - The "automatic" mean reversion built into the gap filter may be misleading:
#       if no genuine mean-reverting mechanism exists in the data — or if
#       counter-cyclical forces suppress it — then a low-pass design that retains
#       frequency-zero information (as in HP-trend) may be more appropriate for
#       analysing growth phases.


# -----------------------------------------------------------------------------
# 7.2 Comparison of Filter Outputs and Zero-Crossings
# -----------------------------------------------------------------------------
# We now plot the time-domain outputs of both filters applied to Indpro
# to visually corroborate the frequency-domain findings above.

mplot <- scale(cbind(y_hp_gap, y_hp), center=F, scale=T)

plot(mplot[,1],
     main="One-Sided HP-Trend vs. HP-Gap Applied to Indpro",
     col="darkred", axes=F, xlab="", ylab="", type="l")
lines(mplot[,2], col="violet")
mtext("HP-gap: applied to levels",      line=-1, col="darkred")
mtext("HP-trend: applied to differences", line=-2, col="violet")
abline(h=0)
axis(1, at=12*(1:(nrow(mplot)/12)),
     labels=index(diff(log(INDPRO)["1960/2019"]))[12*(1:(nrow(mplot)/12))])
axis(2)
box()

# Observations (consistent with the frequency-domain analysis above):
#
#   HP-gap cycle:
#     - Generates numerous zero-crossings during the expansions preceding the
#       dot-com bust and the 2008 financial crisis, acting as early warning signals.
#     - In both cases, the early zero-crossings anticipate the recessions by
#       several years — potentially useful but also prone to false alarms.
#     - Correctly resolves the twin recessions of the early 1980s as two
#       distinct downturns.
#
#   HP-trend cycle:
#     - More conservative: tracks long expansions better without premature
#       zero-crossings.
#     - Generally lags behind the HP-gap cycle at recession onset and recovery.
#     - Surprisingly, also resolves the twin recessions of the early 1980s
#       as two separate events.
#
#   Shared observation:
#     - Both filters identify a trough around 2016, coinciding with the sharp
#       decline in crude oil prices, which significantly affected petroleum
#       extraction and related industrial activity in the US.
#       Note: PAYEMS (or GDP) is less sensitive to this event.


# -----------------------------------------------------------------------------
# 7.3 Zero-Crossing Analysis: Empirical vs. Expected Holding Times
# -----------------------------------------------------------------------------
# The holding time (HT) of a cycle is the average duration between consecutive
# zero-crossings. A longer holding time implies a smoother, lower-frequency cycle.

# --- Empirical holding times (from the actual filter output) ---
compute_empirical_ht_func(y_hp)       # HP-trend applied to differences
compute_empirical_ht_func(y_hp_gap)   # HP-gap applied to levels

# --- Model-implied (expected) holding times ---
# We use the convolved filters applied to model residuals (white noise),
# which represent the theoretically correct spectral structure under the assumed DGP.
compute_holding_time_func(hp_one_sided_conv_indpro)$ht          # HP-trend
compute_holding_time_func(hp_gap_diff_one_sided_conv_indpro)$ht  # HP-gap

# Interpretation:
#   - The empirical and model-implied holding times are broadly consistent but
#     do not match exactly. Possible explanations include:
#       (a) Finite-sample variability in empirical holding-time estimates.
#       (b) Model misspecification, for example non-zero means for HP-trend
#           (off-centered cycles inflate or deflate zero-crossing counts).
#     Both factors are likely contributing.
#   - Across both measures, the HP-trend cycle consistently exhibits longer
#     holding times than the HP-gap cycle. Under the assumed DGP, we expect
#     approximately 30% fewer zero-crossings in the long run for HP-trend,
#     confirming that it extracts a smoother cycle than the HP-gap filter.


# =============================================================================
# 7.3 Comparison of Filter Coefficients
# =============================================================================

# -----------------------------------------------------------------------------
# 7.3.1 Both Filters Applied to First Differences
# -----------------------------------------------------------------------------
# We compare the coefficient sequences of hp_trend and hp_gap_diff, both
# expressed as filters on first-differenced (stationary) data.
# Since the two filters operate on different scales, we normalise before plotting.

ts.plot(scale(cbind(hp_trend, hp_gap_diff), center=F, scale=T),
        col=c("blue","darkred"),
        main="HP-Trend vs. Transformed HP-Gap: Both Applied to Differences")
mtext("HP-trend",          col="blue",    line=-1)
mtext("Transformed HP-gap", col="darkred", line=-2)

# Observation:
#   The two coefficient sequences are broadly similar in shape.
#   Key difference: the HP-gap coefficients dip more clearly below zero, 
#   which is a necessary consequence of the filter's sum being exactly zero.
#   A vanishing sum corresponds to a zero amplitude at frequency zero,
#   ensuring that the original concurrent gap filter cancels the double
#   unit root implicit in the I(2) differencing operator.
sum(hp_gap_diff)


# -----------------------------------------------------------------------------
# 7.3.2 Both Filters Expressed as Applied to Levels
# -----------------------------------------------------------------------------
# We now bring both filters onto the same footing by expressing them as
# filters operating on data in levels (rather than differences).
#
# For hp_trend (originally applied to differences), this transformation is
# straightforward: applying a first-difference operator to the filter
# coefficients yields an equivalent filter for use on levels.
hp_trend_sum <- c(hp_trend, 0) - c(0, hp_trend)

# --- Verification: transformed filter replicates original output ---
# Applying hp_trend_sum to log-levels should reproduce the output of
# hp_trend applied to log-differences.
y_hp     <- filter(diff(log(INDPRO)["1960/2019"]), hp_trend,     side=1)
y_hp_sum <- filter(     log(INDPRO)["1960/2019"],  hp_trend_sum, side=1)

# Both series should overlap exactly:
ts.plot(cbind(y_hp, y_hp_sum),
        main="Verification: Transformed HP-Trend (levels) Replicates HP-Trend (differences)")

# --- Visual comparison of coefficient sequences (both on levels) ---
# We rescale for easier visual comparison.
ts.plot(scale(cbind(hp_gap, hp_trend_sum[1:L]), center=F, scale=T),
        col=c("red","darkblue"),
        main="Original HP-Gap vs. Transformed HP-Trend: Both Applied to Levels")
mtext("Transformed HP-trend", col="darkblue", line=-1)
mtext("Original HP-gap",      col="red",      line=-2)
abline(h=0)

# Observation:
#   The two coefficient sequences remain broadly similar even when both are
#   expressed as level filters. A key structural difference remains:
#   the transformed HP-trend coefficients cancels only a single unit root,
#   whereas the original HP-gap cancels a double unit root (I(2)).

# The sum vanishes: cancellation of a single unit-root. 
sum(hp_trend_sum)


# -----------------------------------------------------------------------------
# 7.3.3 Amplitude Functions: Original HP-Gap vs. Transformed HP-Trend
# -----------------------------------------------------------------------------
# We compare the amplitude (frequency response) functions of both level-domain
# filters to further characterise their frequency-selective properties.

amp_obj_gap       <- amp_shift_func(K, hp_gap,       F)
amp_obj_trend_sum <- amp_shift_func(K, hp_trend_sum, F)

par(mfrow=c(1,1))
# Vertical reference lines mark the business-cycle frequency band:
#   K/12  → 2-year periodicity (upper frequency boundary)
#   K/60  → 10-year periodicity (lower frequency boundary)
plot(scale(amp_obj_gap$amp, center=F, scale=T),
     type="l", col="red", axes=F, xlab="Frequency", ylab="",
     main="Scaled Amplitude Functions: Original HP-Gap vs. Transformed HP-Trend")
lines(scale(amp_obj_trend_sum$amp, center=F, scale=T), col="darkblue")
abline(v=K/12)   # 2-year periodicity boundary
abline(v=K/60)   # 10-year periodicity boundary
mtext("Amplitude: original HP-gap (applied to levels)",      col="red",      line=-1)
mtext("Amplitude: transformed HP-trend (applied to levels)", col="darkblue", line=-2)
axis(1, at=1+0:6*K/6,
     labels=expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Interpretation:
#   Both filters are high-pass in nature — they suppress low frequencies
#   and pass business-cycle and higher frequencies.
#
#   Key differences:
#   - Original HP-gap: amplitude approaches zero with a second-order zero
#     at frequency zero, reflecting the double unit-root cancellation
#     required for I(2) data.
#   - Transformed HP-trend: amplitude approaches zero with a first-order
#     (linear) zero at frequency zero, as it cancels only a single unit root.
#   - Transformed HP-trend is marginally smoother at business-cycle
#     frequencies: its amplitude peaks slightly higher in the business-cycle
#     band, implying slightly greater pass-through of those frequencies.


# =============================================================================
# 7.4 Spurious Cycles: Simulation Experiment
# =============================================================================
# Reference: Wildi, M. (2024), https://doi.org/10.1007/s41549-024-00097-5
#
# Goal:
#   Illustrate the concept of spurious cycles by applying hp_trend and
#   hp_gap_diff to a controlled artificial time series.
#
# Setup:
#   - In first differences: the artificial series consists of a deterministic,
#     time-varying level (the "signal") contaminated with white noise.
#   - In levels: the integrated series exhibits a non-stationary, changing
#     growth rate — representative of typical macroeconomic indicators.

set.seed(35)
len <- 1000

# Construct a piecewise deterministic level (signal) with smooth transitions:
mu1 <- c(rep(-1, 200), seq(-1, 1, by=0.05), rep(1, 100),
         seq(1, 0, by=-0.005), rep(1, 198))
mu  <- c(mu1, mu1)[1:len]

# Superimpose white noise on the deterministic level:
eps <- rnorm(len) + mu

# Apply both filters to the noisy series (normalised for comparability):
x_trende <- filter(eps, scale(hp_trend,    center=F, scale=T) / sqrt(L-1), sides=1)
x_gape   <- filter(eps, scale(hp_gap_diff, center=F, scale=T) / sqrt(L-1), sides=1)


par(mfrow=c(2,2))

# --- Panel 1: Series in first differences ---
# Representative of a macroeconomic indicator in returns (log-differenced).
# The dashed line shows the underlying deterministic level (signal).
mplot <- na.exclude(cbind(eps[(L+1):len]))
plot(mplot[,1],
     main="Simulated Series: First Differences",
     axes=F, type="l", xlab="", ylab="",
     ylim=c(min(mplot), max(mplot)), col="grey")
lines(mu[L:len], lty=2, lwd=2)   # True underlying signal
abline(h=0)
axis(1, at=1:nrow(mplot), labels=-1+1:nrow(mplot))
axis(2)
box()

# --- Panel 2: Series in levels ---
# The cumulative sum of the differenced series, representative of
# a non-stationary macroeconomic indicator (e.g., log industrial production).
mplot <- na.exclude(cbind(eps[(L+1):len]))
plot(cumsum(mplot[,1]),
     main="Simulated Series: Levels (Cumulative Sum)",
     axes=F, type="l", xlab="", ylab="", col="grey")
abline(h=0)
axis(1, at=1:nrow(mplot), labels=-1+1:nrow(mplot))
axis(2)
box()

# --- Panel 3: Filter outputs ---
# Both filters are applied to the differenced series.
# The dashed line shows the (rescaled) true underlying signal for reference.
mplot <- na.exclude(cbind(x_trende, x_gape))
plot(mplot[,1],
     main="Filter Outputs: HP-Trend vs. Transformed HP-Gap",
     axes=F, type="l", xlab="", ylab="",
     ylim=c(min(mplot), max(mplot)), col="brown")
lines(mplot[,2], col="red")
mtext("HP-trend (tracks level changes)", col="brown", line=-1)
mtext("Transformed HP-gap (spurious cycle)", col="red",   line=-2)
lines(3.28 * mu[L:len], lty=2, lwd=2)   # Rescaled true signal
abline(h=0)
axis(1, at=1:nrow(mplot), labels=-1+1:nrow(mplot))
axis(2)
box()

# Interpretation of simulation results:
#
#   HP-gap (red):
#     - Suppresses the changing level (signal) and remains centred near zero.
#     - Produces oscillatory output whose periodicity is determined by the
#       frequency at which its amplitude peaks (governed by lambda).
#     - This oscillation is a filter artifact: it is unrelated to any salient
#       feature of the data and constitutes a spurious cycle.
#
#   HP-trend (brown):
#     - Tracks the changing level of the differenced data (equivalently, the
#       changing growth rate of the level series).
#     - Residual high-frequency fluctuations around the signal are attributable
#       to noise leakage through the filter's imperfect high-frequency attenuation.
#     - The extracted output reflects a genuine feature of the data.
#
# Conceptual contrast:
#   - For HP-gap:  the oscillatory "cycle" is the artifact; the level shifts
#                  are the signal that is being "inadvertently" suppressed.
#   - For HP-trend: the level-tracking output is the signal; the residual
#                   oscillations are the noise.
#
# Open questions for reflection:
#   - Which filter output is less prone to spurious behaviour?
#   - Which filter is more likely to extract economically meaningful features
#     from macroeconomic time series?





# =============================================================================
# 8. SUMMARY
# =============================================================================
# This tutorial examined the HP filter from multiple perspectives — implicit
# model assumptions, filter design, frequency-domain properties, and spectral
# characteristics — to evaluate its suitability for real-time BCA.
#
# The key findings are organized below by theme.
# =============================================================================

# -----------------------------------------------------------------------------
# 8.1 IMPLICIT MODEL: A POOR FIT FOR BUSINESS-CYCLE DATA
# -----------------------------------------------------------------------------
# The theoretical DGP underlying the HP filter (I(2) trend + white noise cycle)
# is fundamentally incompatible with observed macro data:
#
#   - The implicit 'cycle' is pure white noise => no genuine cyclical structure
#   - Simulated data lacks recession/expansion episodes and directional growth
#   - Real macro series are I(1), not I(2), contradicting the HP optimality
#     conditions (see Exercise 1.3)

# -----------------------------------------------------------------------------
# 8.2 TWO-SIDED HP FILTERS: OVER-SMOOTHING
# -----------------------------------------------------------------------------
# Both the two-sided HP trend and HP gap are excessively smooth:
#
#   - The standard lambda values proposed in the literature (e.g., 14,400
#     for monthly data) are likely too large for real macro applications
#   - Strong smoothing attenuates genuine cyclical signals, washing out
#     critical recession troughs and expansion peaks
#   - See Phillips and Jin (2021) for a formal treatment of this issue

# -----------------------------------------------------------------------------
# 8.3 ONE-SIDED HP TREND: A STRONG REAL-TIME BENCHMARK
# -----------------------------------------------------------------------------
# The causal (one-sided) HP trend offers several important advantages:
#
#   TIMELINESS:
#     - Vanishing time-shift at frequency zero (Exercise 4.4) means peaks
#       and troughs are detected with minimal delay
#     - Typically faster than Hamilton's (2018) regression-based filter
#       (see Tutorial 3 for a direct comparison)
#
#   BUSINESS-CYCLE ALIGNMENT:
#     - Peak amplitude at ~85 months (~7 years) matches 
#       business-cycle frequencies (Exercise 4.2)
#     - Capable of resolving closely spaced recessions (e.g., the
#       twin recessions of the early 1980s)
#
#   REAL-TIME SUITABILITY:
#     - Less smooth than the two-sided filter but still suppresses
#       high-frequency noise effectively
#     - The vanishing time-shift makes it a demanding benchmark

# -----------------------------------------------------------------------------
# 8.4 SPECTRAL DENSITY: DGP-DEPENDENCE OF THE EXTRACTED CYCLE
# -----------------------------------------------------------------------------
# The spectral density of the HP-filtered cycle depends on the Wold
# decomposition of the input series (Exercise 5):
#
#   - Applying the SAME HP filter to INDPRO and PAYEMS produces
#     qualitatively different cycles:
#       * PAYEMS cycle is substantially smoother than INDPRO cycle
#       * Differences reflect distinct persistence structures in each series
#   - This DGP-dependence is a fundamental but largely unaddressed problem
#     in standard BCA practice, where a single lambda is applied uniformly
#     across all series regardless of their time-series properties

# -----------------------------------------------------------------------------
# 8.5 HP TREND ON DIFFERENCES vs. HP GAP ON LEVELS
# -----------------------------------------------------------------------------
# The two approaches share structural similarities but differ
# in important ways:
#
#   SIMILARITIES (Exercises 6-7):
#     - Similar filter coefficient shapes in levels and differences
#     - Similar amplitude functions across the frequency band, except at omega=0 
#         (bandpass HP-gap has vanishing A(0)=0)
#     - Both effectively cancel unit roots (up to their respective orders)
#
#   FILTER ALGEBRA:
#     - HP gap:              gap_t   = x_t - trend_t
#                            => (Identity - HP_trend) applied to levels
#                            => cancels unit roots up to ORDER TWO (I(2))
#     - HP trend on diffs:   cycle_t = trend_t - trend_{t-1}
#                            => HP_trend applied to (1-B)*x_t
#                            => cancels a SINGLE unit root (I(1))
#
#   ADVANTAGES OF HP TREND ON DIFFERENCES:
#     - Smoother cycle: fewer false signals 
#     - Better tracking of prolonged expansion episodes
#     - Does not generate spurious cycle alarms mid-expansion
#     - More consistent with the I(1) nature of macro data (Section 5.1)
#
#   DISADVANTAGES OF HP TREND ON DIFFERENCES:
#     - Systematic lag at the START and END of recessions (when compared to HP-gap)
#     - Slightly slower to detect cyclical turning points (despite a vanishing shift)
#
#   HP GAP CHARACTERISTICS:
#     - 'Fast' filter: negative time-shift => anticipative behavior
#     - Can be TOO anticipative: systematically drops below zero during
#       prolonged expansions, generating false contraction signals

# -----------------------------------------------------------------------------
# 8.6 OVERALL RECOMMENDATION AND OUTLOOK
# -----------------------------------------------------------------------------
# HP trend applied to differences is the MORE CONSERVATIVE and RELIABLE design:
#
#   - Tracks expansions and recessions more accurately than the original HP gap
#   - Mitigates spurious cycle problem 
#   - Better aligned with the I(1) properties of real macro data
#   - Provides a strong real-time benchmark
#
# NEXT STEPS — Tutorial 2.1:
#   SSA will be applied on top of the HP trend filter to address the
#   fundamental smoothness-timeliness trade-off:
#
#     smoothness  <=>  larger time-shift  (delayed turning point detection)
#     timeliness  <=>  more noise leakage (rougher cycle estimate)
#
#   SSA optimization will seek filter designs that improve on the HP trend
#   benchmark along both dimensions simultaneously.
# =============================================================================


