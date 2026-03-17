# =============================================================================
# TUTORIAL: BUSINESS-CYCLE ANALYSIS (BCA) — FOUNDATION FOR SSA INTEGRATION
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
source(paste(getwd(),"/R/hpFilt.r",sep=""))
# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))


# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

# Load data from FRED with library quantmod
library(quantmod)
# Download Non-farm payroll and INDPRO
getSymbols('PAYEMS',src='FRED')
getSymbols('INDPRO',src='FRED')

# We now develop points 1-8 listed above 
#########################################################################################################
#########################################################################################################
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

####################################################################################################################
####################################################################################################################
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

###################################################################################################################
#################################################################################################################
# 5. Spectral densities HP-trend
# -We want to compare the spectral densities of the extracted cycles
# -For that purpose we compute cycles based on two- and one-sided HP-trend filters, as applied to differences of the data
# -This topic requires 
#   5.1. modelling of the DGP
#   5.2. derivation of the spectral density


# 5.1 Let us assume, that typical (eventually log-transformed) economic time series are compatible with an I(1)-DGP
# We try to motivate this I(1)-assumption
# a. Let's look at returns:
par(mfrow=c(2,2))
plot(diff(log(INDPRO)["1960/2019"]),xlab="",ylab="",main="Returns Indpro")
plot(diff(log(PAYEMS)["1960/2019"]),xlab="",ylab="",main="Returns non-farm payroll")
acf(na.exclude(diff(log(INDPRO)["1960/2019"])),main="ACF Indpro")                
acf(na.exclude(diff(log(PAYEMS)["1960/2019"])),main="ACF non-farm payroll")                
# Returns do not seem to be subject to pronounced changes of their levels (but data prior 1990 looks different: great moderation)

# b. Also, simple stationary ARMA-models seem to fit the data 
model_indpro<-arima(diff(log(INDPRO)["1960/2019"]),order=c(2,0,1))
model_payems<-arima(diff(log(PAYEMS)["1960/2019"]),order=c(2,0,1))
tsdiag(model_indpro)
tsdiag(model_payems)

# c. Finally, economic theory suggests that a broad range of economic series, such as  stock prices, futures prices, 
#   long-term interest rates, oil prices, consumption spending, inflation, tax rates, or money supply growth rates  
#   should follow (near) martingales, see Fama (1965), Samuelson (1965), Sargent (1976), Hamilton (2009), Hall (1978) 
#   and Mankiw (1987)

# Let us therefore assume that the data is I(1), in contradiction to the ARIMA(0,2,2)-hypothesis of the implicit DGP justifying optimality of HP
# In principle, we could apply HP-trend to returns: the above amplitude functions suggest that 
#   a. the one-sided filter would not overtly conflict with an extraction of the (differenced) business-cycle from the data (its peak amplitude corresponds to a periodicity of seven years)
#   b. the two-sided filter would damp cycle-frequencies too heavily (over-smoothing), see also Phillips and Jin (2021)

# Let us now compute the spectral density of the components (`cycles`) extracted by both filters when applied to differenced macro-data.
# For this purpose we need to compute the convolution of HP and xi, the Wold-decomposition of the differenced data
# We can compute the weights of the Wold-decomposition of the above models
ar<-model_indpro$coef[1:2]
ma<-model_indpro$coef[3]
xi_indpro<-c(1,ARMAtoMA(lag.max=L-1,ar=ar,ma=ma))
ar<-model_payems$coef[1:2]
ma<-model_payems$coef[3]
xi_payems<-c(1,ARMAtoMA(lag.max=L-1,ar=ar,ma=ma))
par(mfrow=c(1,1))
ts.plot(cbind(xi_indpro,xi_payems),main="MA-inversions (Wold decomposition) for Macro-Indicators",col=c("violet","orange"))
mtext("Wold-decomposition Indpro",col="violet",line=-1)
mtext("Wold-decomposition PAYEMS",col="orange",line=-2)

# We can now compute the convolution of HP and Wold-decompositions
hp_one_sided_conv_indpro<-conv_two_filt_func(xi_indpro,hp_trend)$conv
hp_two_sided_conv_indpro<-conv_two_filt_func(xi_indpro,hp_target)$conv
hp_one_sided_conv_payems<-conv_two_filt_func(xi_payems,hp_trend)$conv
hp_two_sided_conv_payems<-conv_two_filt_func(xi_payems,hp_target)$conv


# Interpretation: hp_one_sided_conv applied to model-residuals generates the same output as hp_trend applied to returns
# We briefly check this assertion for Indpro (check for PAYEMS is similar)
#   Note that we must shift by the mean ("intercept") since residuals are centered at zero
y_hp_conv<-filter(model_indpro$residuals,hp_one_sided_conv_indpro,side=1)+model_indpro$coef["intercept"]
y_hp<-filter(diff(log(INDPRO)["1960/2019"]),hp_trend,side=1)
# Both series overlap, as expected.
ts.plot(cbind(y_hp_conv,y_hp),main="Convolution applied to residuals vs. hp applied to returns: both series overlap")

#-------------------
# 5.2 Spectral densities
# The spectral density of the `cycle` extracted by hp_trend corresponds to the squared amplitude of hp_one_sided_conv
#   -Recall that hp_one_sided_conv is applied to model residuals and that the spectral density of white noise (residuals) is flat
#   -We ignore the scaling by sigma^2/(2*pi) corresponding to the residuals' spectral density
#   -We scale the spectral densities in order to simplify comparisons 
amp_obj_one_sided_hp_conv_indpro<-amp_shift_func(K,hp_one_sided_conv_indpro,F)
amp_obj_one_sided_hp_conv_payems<-amp_shift_func(K,hp_one_sided_conv_payems,F)
par(mfrow=c(1,1))
# Let's scale the spectral densities for easier comparison
#   -We also add two vertical bars corresponding to business-cycle frequencies: 2-10 years periodicities
plot(scale(amp_obj_one_sided_hp_conv_indpro$amp^2,center=F,scale=T),type="l",col="violet",axes=F,xlab="Frequency",ylab="",main=paste("Scaled spectral density of cycle extracted by one-sided HP trend",sep=""))
lines(scale(amp_obj_one_sided_hp_conv_payems$amp^2,center=F,scale=T),col="orange")
abline(v=K/12)
abline(v=K/60)
mtext("Scaled spectral density of one-sided HP-trend applied to differences of Indpro",col="violet",line=-1)
mtext("Scaled spectral density of one-sided HP-trend applied to differences of Payems",col="orange",line=-2)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# Outcome:
# We see that the spectral density, trivially, depends on the DGP (this evidence is frequently lacking, omitted or neglected in applications)
#   -The Indpro-cycle spectral density is not overtly conflicting with a BCA-perspective 
#   -The PAYEMS-cycle is likely a bit `too smooth' (the peak is left-shifted towards lower frequencies)
# Note: we here apply filters to differenced data. Accordingly, the above filters are supposed to `extract' the differenced cycle
#   -We don't want the spectral density of the differenced cycle to be flat in the business-cycle band: in this sense the trend-filters seem to perform well.
#   -But maybe we'd like the density to vanish towards frequency zero? That depends... (see below)

# We now look at the spectral densities of two-sided trend filters
amp_obj_two_sided_hp_conv_indpro<-amp_shift_func(K,hp_two_sided_conv_indpro,F)
amp_obj_two_sided_hp_conv_payems<-amp_shift_func(K,hp_two_sided_conv_payems,F)
par(mfrow=c(1,1))
# Let's scale the spectral densities for easier comparison
#   -We also add two vertical bars corresponding to business-cycle frequencies: 2-10 years periodicities
plot(scale(amp_obj_two_sided_hp_conv_indpro$amp^2,center=F,scale=T),type="l",col="violet",axes=F,xlab="Frequency",ylab="",main=paste("Scaled spectral density of cycle extracted by two-sided HP trend",sep=""))
lines(scale(amp_obj_two_sided_hp_conv_payems$amp^2,center=F,scale=T),col="orange")
abline(v=K/12)
abline(v=K/60)
mtext("Scaled spectral density of two-sided HP-trend applied to differences of Indpro",col="violet",line=-1)
mtext("Scaled spectral density of two-sided HP-trend applied to differences of Payems",col="orange",line=-2)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# The spectral density are left-shifted: the two-sided filters smooth out business-cycle frequencies, see Phillips and Jin (2021)

##################################################################################################
################################################################################################
# 6. Spectral densities of original HP-gap
# -We want to compare the spectral densities of the extracted cycles
# -For that purpose we compute cycles based on two- and one-sided gaps, applied to data in levels (HP-trend was applied to differences)
# -This topic requires 
#   6.1. modelling of the DGP (see 5.1 above)
#   6.2  stationarity (gaps are applied to non-stationary data in levels)
#   6.3. derivation of the spectral density

# 6.1: see 5.1 above

# 6.2 Transformation: from levels to differences
#   -The gap is applied to levels (not differences)
#   -We here transform the gap-filter such that one can work with differenced data
#  For that purpose we can rely on conv_with_unitroot_func, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
hp_gap_diff<-conv_with_unitroot_func(hp_gap)$conv

# Compare both gap filters
ts.plot(cbind(hp_gap,hp_gap_diff),col=c("red","darkred"),main=paste("HP-gap: original and transformed filters, lambda=",lambda_monthly,sep=""))
mtext("Original HP-gap as applied to levels",col="red",line=-1)  
mtext("Transformed HP-gap as applied to differences",col="darkred",line=-2)  

# After scaling, we can note a degree of familiarity between the transformed gap and the trend-filter (both are applied to differences)
ts.plot(scale(cbind(hp_trend,hp_gap_diff),scale=T,center=F),col=c("blue","darkred"),main=paste("Transformed HP-gap vs. HP-trend, lambda=",lambda_monthly,sep=""))
mtext("HP-trend as applied to differences",col="blue",line=-1)  
mtext("Transformed HP-gap as applied to differences",col="darkred",line=-2)  

# We check that the output of hp_gap_diff as applied to differences replicates the original gap when applied to levels 
y_hp_gap<-filter((log(INDPRO)["1960/2019"]),hp_gap,side=1)
y_hp_gap_diff<-filter(diff(log(INDPRO)["1960/2019"]),hp_gap_diff,side=1)
# Both series overlap
#   -Therefore the corresponding spectral densities must overlap, too
ts.plot(cbind(y_hp_gap,y_hp_gap_diff),main="Original HP-gap applied to levels vs. modified gap applied to returns: both series overlap")

# We are now closer to our goal: computing the spectral density of the original HP-gap cycle
# We just have to convolve hp_gap_diff with the Wold-decomposition of the series
hp_gap_diff_one_sided_conv_indpro<-conv_two_filt_func(xi_indpro,hp_gap_diff)$conv
hp_gap_diff_one_sided_conv_payems<-conv_two_filt_func(xi_payems,hp_gap_diff)$conv

# The spectral densities are obtained by squaring the amplitude functions of the convolved transformed filters (up to an arbitrary scaling)
amp_obj_hp_gap_diff_conv_indpro<-amp_shift_func(K,hp_gap_diff_one_sided_conv_indpro,F)
amp_obj_hp_gap_diff_conv_payems<-amp_shift_func(K,hp_gap_diff_one_sided_conv_payems,F)
par(mfrow=c(1,1))
plot(scale(amp_obj_hp_gap_diff_conv_indpro$amp^2,center=F,scale=T),type="l",axes=F,xlab="Frequency",col="darkred",ylab="",main=paste("Spectral density of cycle extracted by one-sided HP filters",sep=""))
lines(scale(amp_obj_hp_gap_diff_conv_payems$amp^2,center=F,scale=T),col="brown")
mtext("Spectral density original HP-gap Indpro",line=-1,col="darkred")
mtext("Spectral density original HP-gap Payems",line=-2,col="brown")
abline(v=K/12)
abline(v=K/60)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# HP-gap applied to Payems generates a smoother cycle 
#   -The corresponding Wold-decomposition un-whitens innovations in the Wold-decomposition more effectively
#   -The ARMA-model of Payems is a more effective lowpass than the ARMA-model of Indpro
#   -In simpler terms: PAYEMS looks (and is) smoother than Indpro
#   -Therefore, applying the same filter to different series will generate a smoother cycle for the smoother series

# Let us compare both cycles:
y_hp_gap_payems<-filter((log(PAYEMS)["1960/2019"]),hp_gap,side=1)
mplot<-scale(cbind(y_hp_gap,y_hp_gap_payems),center=F,scale=T)
plot(mplot[,1],main="Cycles of HP-gap applied to Indpro and PAYEMS",col="darkred",axes=F,xlab="",ylab="",type="l")
lines(mplot[,2],col="green")
mtext("HP-gap applied to Indpro",line=-1,col="darkred")
mtext("HP-gap applied to PAYEMS",line=-2,col="green")
abline(h=0)
axis(1,at=12*(1:(nrow(mplot)/12)),labels=index(diff(log(INDPRO)["1960/2019"]))[12*(1:(nrow(mplot)/12))])
axis(2)
box()
# Differences: 
#   -the cycle based on non-farm payroll is substantially smoother, 
#   -it transforms the twin-recessions in the early 80s into a single `double-dip' recession, 
#   -it is systematically lagging behind the cycle extracted from Indpro 
#   -it behaves differently after the financial crisis (more or less monotonous decay after the recession-rebound)
# Applying the same HP-filter (based on lambda=14400) to qualitatively different series is not recommended (sadly, this is not our topic of interest)

#################################################################################################
#################################################################################################
# 7. Comparison: HP-trend applied to differences vs. original HP-gap (applied to levels)
# 7.1 Compare spectral densities 
# We now compare the spectral densities of HP-trend applied to differences with the original HP-gap applied to levels
# We do this for Indpro (a similar outcome would be obtained for Payems)
par(mfrow=c(1,1))
plot(scale(amp_obj_hp_gap_diff_conv_indpro$amp^2,center=F,scale=T),type="l",axes=F,xlab="Frequency",col="darkred",ylab="",main=paste("Spectral density of cycles extracted by one-sided HP-gap and HP-trend, lambda=",lambda_monthly,sep=""))
lines(scale(amp_obj_one_sided_hp_conv_indpro$amp^2,center=F,scale=T),col="violet")
mtext("Spectral density original HP-gap applied to levels of Indpro",line=-1,col="darkred")
mtext("Spectral density HP-trend applied to differences of Indpro",line=-2,col="violet")
abline(v=K/12)
abline(v=K/60)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# Quite similar...
# -The main difference: the density of HP-trend does not vanish at frequency zero
# -The cycle extracted by HP-trend is also a bit smoother (emphasizes more heavily lower frequencies): 
#   -This particular effect seems comparable to the difference between gap-filters applied to Payems and Indpro in a previous plot above

# Which `cycle' is `better'? The answer depends on the purpose of the analysis
# -The cycle extracted by HP-trend in first differences can track long expansion episodes better, because level-information (at frequency zero) passes through
#   -In contrast, the gap-filter will generate (more false) alarms in between consecutive recessions, at least if the expansion separating them is longer than usual
#   -But the gap-filter will track very short expansion episodes better (the twin recessions in the early eighties)
# -In general, the shift or lag of the gap filter is smaller (the shift-function computed above indicated anticipation)
#   -Pros: the gap generally anticipates zero-crossings of the trend
#   -Cons: sometimes the anticipation can be  `too much`: we expect the gap to drop below the zero-line up to several years before the effective recession starts
#------------------------
# 7.2 Compare filter outputs and zero-crossings
# Let's look at the data: compare outputs of one-sided HP-trend and HP-gap

mplot<-scale(cbind(y_hp_gap,y_hp),center=F,scale=T)
plot(mplot[,1],main="One-sided HP-trend vs. HP-gap applied to Indpro",col="darkred",axes=F,xlab="",ylab="",type="l")
lines(mplot[,2],col="violet")
mtext("HP-gap applied to levels",line=-1,col="darkred")
mtext("HP-trend applied to returns",line=-2,col="violet")
abline(h=0)
axis(1,at=12*(1:(nrow(mplot)/12)),labels=index(diff(log(INDPRO)["1960/2019"]))[12*(1:(nrow(mplot)/12))])
axis(2)
box()

# The plot confirms the above conjectures based on frequency-domain analysis of the filters
# -The gap-cycle generates numerous zero-crossings along the expansions before the dotcom-bubble and the financial crisis  
# -In both cases early zero-crossings of the gap-cycle anticipate the recessions by several years
# -In contrast, the trend-cycle is much more conservative: it tracks longer expansions better but it generally lags behind the gap-cycle at start and end of recessions
# -Surprisingly, the trend filter is able to resolve/separate the twin-recessions in the early 80s.
# -Both filters indicate a trough around 2016, at a time  when the price for crude oil declined sharply, hence affecting petrol extraction as well as collateral industrial activity in the US.
#   -PAYEMS (or GDP) is less affected by this singular event
# -HP-trend applied to differences is a growth-tracker: 
#   -its mean-level, the drift of the original series, is not zero (unless the original series is flat) 
#   -it is not automatically reverting to the zero-line
#   -it is not even stationary: if the drift or slope of the original series changes, then the level of the `cycle` will change accordingly
# -HP-gap, on the other hand, has all requested cycle-characteristics (zero-mean, stationary, mean-reverting)
# We argue that this fundamental characteristics of a cycle, and in particular its `automatic' mean-reversion, are spurious
#   -If a particular phenomenon does not have an explicit built-in mean-reverting cycle mechanism, or if important 
#     actors work succesfully (counter-cyclically) against its appearance, or if the mean-reverting mechanism is weak, 
#     then a lowpass design, letting frequency zero pass through, is likely a more suitable design for analysing 
#     growth-phases.   


# Consider zero-crossings: empirical holding-time
compute_empirical_ht_func(y_hp)
compute_empirical_ht_func(y_hp_gap)
# The trend filter is slightly smoother 
# Compare with expected holding times, assuming the model of the data is correct: we use the convolved filters which are applied to model residuals (which is supposed to be white noise)
compute_holding_time_func(hp_one_sided_conv_indpro)$ht
compute_holding_time_func(hp_gap_diff_one_sided_conv_indpro)$ht
# The discrepancy between empirical and expected holding-times suggests 
#   a. random-sample errors (of empirical ht) 
#   b. model misspecification (for example non-vanishing means i.e. off-centered cycles)
# Probably both... But the numbers confirm that the cycle extracted by HP-trend is smoother (on the long run we expect ~30% less zero-crossings)

#-------------------------------------
# 7.3 Compare filter coefficients
# 7.3.1 Both filters as applied to returns
# Let us also compare hp_gap_diff and hp_trend, both applied to differences
# Since the scales differ we normalize the filters
ts.plot(scale(cbind(hp_trend,hp_gap_diff),center=F,scale=T),col=c("blue","darkred"),main="HP-trend and transformed HP-gap: both applied to differences")
mtext("HP-trend",col="blue",line=-1)
mtext("Transformed HP-gap",col="darkred",line=-2)
# We see a degree of familiarity. 
#   -The gap-coefficients drop below zero because their sum must vanish (vanishing amplitude at frequency zero: the original concurrent gap must cancel a second-order unit-root)
sum(hp_gap_diff)

# 7.3.2 Both filters as applied to levels
# We could also transform hp_trend (applied to differences) into a filter that replicates its cycle when applied to levels
# The transformation is very simple: apply first differences to filter coefficients:
hp_trend_sum<-c(hp_trend,0)-c(0,hp_trend)

# We can check that the transformed filter replicates the original one
y_hp<-filter(diff(log(INDPRO)["1960/2019"]),hp_trend,side=1)
y_hp_sum<-filter((log(INDPRO)["1960/2019"]),hp_trend_sum,side=1)
# Both series overlap
ts.plot(cbind(y_hp,y_hp_sum),main="Transformed HP-trend applied to levels replicates original HP-trend applied to returns")

# We can now compare the original HP-gap (as applied to levels) with the transformed HP-trend (also applied to levels, too)
# We rescale coefficients for easier visual comparison
ts.plot(scale(cbind(hp_gap,hp_trend_sum[1:L]),center=F,scale=T),col=c("red","darkblue"),main="Original HP-gap vs. transformed HP-trend: both applied to levels")
mtext("Transformed HP-trend",col="darkblue",line=-1)
mtext("Original HP-gap",col="red",line=-2)
abline(h=0)
# Once again: a degree of familiarity is apparent 
# Note that the coefficients of the transformed HP-trend now add to zero: the filter must cancel a single unit-root of the data in levels
sum(hp_trend_sum)

# We can also compare amplitude functions of original HP-gap and transformed HP-trend (as applied to levels)
amp_obj_gap<-amp_shift_func(K,hp_gap,F)
amp_obj_trend_sum<-amp_shift_func(K,hp_trend_sum,F)
par(mfrow=c(1,1))
# Let's scale the spectral densities for easier comparison
#   -We also add two vertical bars corresponding to business-cycle frequencies: 2-10 years periodicities
plot(scale(amp_obj_gap$amp,center=F,scale=T),type="l",col="red",axes=F,xlab="Frequency",ylab="",main=paste("Scaled amplitude of original gap and transformed trend",sep=""))
lines(scale(amp_obj_trend_sum$amp,center=F,scale=T),col="darkblue")
abline(v=K/12)
abline(v=K/60)
mtext("Amplitude HP-gap original (applied to levels)",col="red",line=-1)
mtext("Amplitude transformed trend (applied to levels)",col="darkblue",line=-2)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# Once again, the degree of familiarity is patent
#   -Both filters are highpass
#   -Original HP-gap approaches zero more smoothly (second order zero to cancel a double unit-root)
#   -Transformed HP-trend approaches zero more linearly (first order zero: cancels a single unit-root only)
#   -Transformed HP-trend is a hair smoother (the amplitude is a bit larger at business-cycle frequencies, where it peaks)

#-----------------------------------------
# Spurious cycles, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# -The following simulation experiment applies hp_trend and hp_gap_diff to an artificial time series
# -The time series in differences corresponds to a deterministic changing level (signal) overlaid with noise
# -The series in levels is a non-stationary series with changing growth-rate


set.seed(35)
len<-1000
# Deterministic changing level 
mu1<-c(rep(-1,200),seq(-1,1,by=0.05),rep(1,100),seq(1,0,by=-0.005),rep(1,198))
mu<-c(mu1,mu1)[1:len]
# Noise+level
eps<-rnorm(len)+mu

# Apply hp_trend and hp_gap_diff to the data 
x_trende<-filter(eps,scale(hp_trend,center=F,scale=T)/sqrt(L-1),sides=1)
x_gape<-filter(eps,scale(hp_gap_diff,center=F,scale=T)/sqrt(L-1),sides=1)


par(mfrow=c(2,2))
# Plot data together with level: representative of economic data in first differences
mplot<-na.exclude(cbind(eps[(L+1):len]))
plot(mplot[,1],main="Series in differences",axes=F,type="l",xlab="",ylab="",ylim=c(min(mplot),max(mplot)),col="grey")
#lines(c(rep(3.28,200),rep(NA,200)),lty=2)
#lines(c(rep(NA,400),rep(3.28,200)),lty=2)
lines(mu[L:len],lty=2,lwd=2)
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Plot integrated data: representative of economic indicator in levels
mplot<-na.exclude(cbind(eps[(L+1):len]))
plot(cumsum(mplot[,1]),main="Series in levels",axes=F,type="l",xlab="",ylab="",col="grey")
#lines(c(rep(3.28,200),rep(NA,200)),lty=2)
#lines(c(rep(NA,400),rep(3.28,200)),lty=2)
#lines(mu[L:800],lty=2)
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Plot filter outputs
mplot<-na.exclude(cbind(x_trende,x_gape))
plot(mplot[,1],main="Filter outputs",axes=F,type="l",xlab="",ylab="",ylim=c(min(mplot),max(mplot)),col="brown")
lines(mplot[,2],col="red")
mtext("Original HP-trend",col="brown",line=-1)
mtext("Transformed HP-gap",col="red",line=-2)
#lines(c(rep(3.28,200),rep(NA,200)),lty=2)
#lines(c(rep(NA,400),rep(3.28,200)),lty=2)
lines(3.28*mu[L:len],lty=2,lwd=2)
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


# Outcomes:
# -hp_gap_diff (red line) cancels the signal (changing level) and stays centered at the zero line
# -hp_trend (brown line) tracks the changing level (changing growth rate of data in levels)
# -The output of hp_gap_diff is a spurious cycle whose duration is determined by the frequency at which its amplitude peaks.
#   -the corresponding `cycle' is an artifact of the bandpass characteristics of the filter, as determined by lambda.
#   -the corresponding cycle is not related to a salient feature of the data.
# -In contrast, the output of hp_trend `extracts` a salient feature of the data, namely the changing level


# The undesirable `cyclical` movements of hp_trend are just `noise' which is due to high-frequency leakage of the filter
#   -The level shifts of the data are the proper `signal'
# In contrast, the `cyclical' gap of hp_gap_diff is the proper signal and the level-shifts of the data are a nuisance.

# Questions:
# Which filter output is `less spurious'? 
# Which filter output is more likely to extract relevant (salient) features from economic data?



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
#     - Peak amplitude at ~85 months (~7 years) matches canonical
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
#     - Similar amplitude functions across the frequency band, except at omega=0 (bandpass HP-gap has vanishing A(0)=0)
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


