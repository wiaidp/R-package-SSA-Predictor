
# =============================================================================
# Tutorial 3: SSA Applied to Hamilton's Regression Filter (HF)
# =============================================================================
# # James D. Hamilton, 2017. "Why You Should Never Use the Hodrick-Prescott Filter," 
# NBER Working Papers 23429, National Bureau of Economic Research, Inc.

# Overview:
#   This tutorial demonstrates how SSA customization can be applied to
#   Hamilton's regression filter (HF) to improve its real-time business cycle
#   analysis (BCA) properties along two key dimensions: smoothness and timeliness.
#
# Datasets:
#   - Example 1:        Quarterly US GDP
#   - Examples 2 & 3:  Monthly US non-farm payroll employment (PAYEMS)
#
# Sample coverage:
#   Both long samples (post-WWII for GDP) and shorter samples (post-Great 
#   Moderation, ~1990 onwards for PAYEMS) are analysed. 
#
# Recommendation: to mitigate non-stationarity avoid working on unnecessarily 
# long historical samples when fitting parameters to data.

# =============================================================================
# Summary of Main Findings
# =============================================================================
#
# 1. HF AS A LOW-PASS FILTER:
#      When expressed in terms of first differences (see Exercises 1.4 and 1.9),
#      HF behaves as a low-pass filter. This property confers two important
#      advantages for business cycle analysis (BCA):
#
#        (a) Mitigation of spurious cycles:
#              Low-pass behaviour suppresses the artificial oscillations that
#              band-pass filters tend to impose on the data, regardless of
#              whether such oscillations are present in the underlying series.
#
#        (b) Advantage over classic band-pass designs:
#              Filters such as Baxter-King (BK), Christiano-Fitzgerald (CF),
#              and the HP-gap are all band-pass in nature and therefore prone
#              to spurious cycle generation. HF avoids this pathology by design.
#              See Tutorials 2.0 (Example 7), 4, and 5 for detailed evidence.
#
#
# 2. HF ACCOMMODATES ARBITRARY INTEGRATION ORDERS:
#      By increasing the AR lag order p in the regression specification, HF can
#      accommodate series integrated to an arbitrary order. This makes it
#      well-suited for macroeconomic indicators whose growth rates evolve slowly
#      over long historical samples — a common feature of post-WWII data.
#
#
# 3. HF IS SAMPLE-DEPENDENT:
#      HF is defined by regression coefficients estimated on a specific sample.
#      As a result, the extracted cycle is inherently sample-specific:
#        - Estimating the regression on the full post-WWII sample versus the
#          post-1990 sub-sample yields substantially different filter coefficients
#          and, consequently, different cycle definitions.
#        - Users should be aware that HF comparisons across studies may reflect
#          differences in estimation samples as much as differences in the data.
#
#
# 4. HF PRODUCES NON-STATIONARY CYCLES OUT-OF-SAMPLE:
#      (Note: this finding does not contradict Finding 2 above.)
#
#      HF is estimated as an unrestricted OLS regression, without imposing
#      any constraint on the unit-root structure of the data. As a consequence:
#        - IN-SAMPLE:  the HF cycle is stationary by construction (OLS residuals
#                      are centred and well-behaved within the estimation window).
#        - OUT-OF-SAMPLE: the HF filter weights do not sum to zero, so the filter
#                      fails to cancel the unit root(s) in the data. The extracted
#                      cycle therefore becomes non-stationary beyond the estimation
#                      window.
#
#      Two remedies are discussed in this tutorial:
#        (i)  Regular re-estimation: updating the regression with new observations
#             restores in-sample stationarity, at the cost of real-time revisions
#             to historical cycle estimates.
#        (ii) Unit-root-constrained regression: imposing a zero-sum constraint on
#             the filter coefficients ensures unit-root cancellation out-of-sample
#             for a single (or more) integrated root. See Exercises 1.2 and 2.3.
#
#
# 5. HF IS NOT ON THE EFFICIENT ATS FRONTIER:
#      The Accuracy-Timeliness-Smoothness (ATS) frontier characterises the set
#      of filters that cannot be improved in one dimension without sacrificing
#      another. HF does not lie on this frontier:
#        - SSA can strictly improve the smoothness (noise suppression) of HF
#          without meaningfully sacrificing accuracy or timeliness.
#          See Example 2.9 for empirical confirmation.
#        - An analogous result holds for HP-based filters; see Tutorial 2.1.
#        - This inefficiency provides a clear and well-motivated rationale for
#          applying SSA customization to HF.
#
#
# 6. SSA CUSTOMIZATION STRATEGIES:
#      Two complementary SSA customization approaches are demonstrated,
#      targeting different points on the ATS frontier:
#
#        (a) Smoothness improvement only:
#              Target: reduce high-frequency noise without affecting timeliness.
#              Method: impose a holding time 50% larger than HF's empirical
#                      holding time as a constraint within the SSA optimisation.
#              Reference: Exercises 1.6 and 2.7.
#
#        (b) Simultaneous smoothness and timeliness improvement:
#              Target: reduce noise AND advance turning-point detection.
#              Method: impose the enlarged holding-time constraint while
#                      simultaneously extending the SSA forecast horizon,
#                      shifting the filter output ahead of real time.
#              Reference: Exercises 1.8 and 2.9.
#
#
# 7. PANDEMIC ROBUSTNESS:
#      The COVID-19 pandemic generated extreme outliers in economic aggregates
#       — observations so large that they approximate impulse inputs to
#      the filters. This provides a natural experiment for studying filter
#      dynamics empirically, without simulation:
#        - The post-pandemic "phantom peak" in each filter's output directly
#          reflects the shape and decay rate of its impulse response function.
#        - Filters with faster-decaying impulse responses recover more quickly
#          from the shock, producing shorter-lived distortions.
#        - SSA forecast filters are shown to recover materially faster than HF,
#          an advantage that is especially relevant in the presence of singular
#          macro events.
#      See Exercise 2.11 for the full analysis.
# =============================================================================


# -----------------------------------------------------------------------------
# Broader Motivation
# -----------------------------------------------------------------------------
# The goal of this tutorial is not to advocate for (or against) a particular 
# business cycle tool or filter design. Rather, it illustrates a general and 
# broadly applicable principle:
#
#   Any causal linear filter — including HF — can be replicated and
#   systematically improved by SSA with respect to two practically
#   important BCA priorities:
#     1. Smoothness:  suppression of high-frequency noise, reducing false signals.
#     2. Timeliness:  earlier and more reliable detection of cyclical turning points.
#
# HF serves here as a convenient, well-known baseline platform and a concrete
# showcase for the SSA optimisation principle. Throughout the tutorial, a range
# of quantitative performance measures — including holding times, amplitude
# functions, phase-shift functions, and empirical zero-crossing counts — is
# provided to confirm the practical value of the SSA approach.
# =============================================================================



# ── BACKGROUND ────────────────────────────────────────────────────
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

#   Wildi, M. (2026a) Sign Accuracy, Mean-Squared Error and the Rate
#     of Zero Crossings: a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547

# ─────────────────────────────────────────────────────────────────

# Clear the workspace and load required packages and functions.
rm(list = ls())

library(xts)

# The 'neverhpfilter' package implements the Hamilton filter ("never use the
# HP-filter"). It is used here for background information and plotting only.
library(neverhpfilter)

# Load data from FRED using the quantmod library.
library(quantmod)

# Load data from FRED using the alfred library (no API key required).
install.packages("alfred")
library(alfred)

# Load all relevant SSA functions.
source(paste(getwd(), "/R/ssa.r", sep = ""))

# Load the tau-statistic: quantifies time-shift performance (lead/lag).
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load signal extraction functions used for the JBCY paper (requires mFilter).
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))

# Load the Hamilton regression function with a unit-root constraint.
source(paste(getwd(), "/R utility functions/hamilton_unit_root.r", sep = ""))


# ──────────────────────────────────────────────────────────────────────────
# Example 1: QUARTERLY GDP
# ──────────────────────────────────────────────────────────────────────────
# Application of SSA to the Hamilton Filter (HF) using quarterly U.S. GDP data.
# We first demonstrate HF using the 'neverhpfilter' package, which also
# provides the data source (GDPC1).
data(GDPC1)

# Apply the Hamilton filter: decompose log-GDP (scaled by 100) into trend
# and cycle. h = 8 quarters (2-year forecast horizon); p = 4 lags
# (quarterly autoregressive order).
gdp_trend <- yth_filter(
  100 * log(GDPC1), h = 8, p = 4, output = c("x", "trend")
)

# Plot the original log-GDP series together with the estimated trend.
plot.xts(
  gdp_trend,
  grid.col    = "white",
  legend.loc  = "topleft",
  main        = "Log of GDP and trend"
)

# Extract the cyclical and irregular (random) components from the Hamilton
# filter.
gdp_cycle <- yth_filter(
  100 * log(GDPC1), h = 8, p = 4, output = c("cycle", "random")
)

# Note: The cycle and random components often exhibit strong correlation,
# which contradicts the standard business-cycle assumption of independence.
# The cycle is centred around zero. HF removes up to p unit roots, where p
# is the autoregressive order used in the regression.
plot.xts(
  gdp_cycle,
  grid.col   = "white",
  legend.loc = "topleft",
  main       = "Cycle and irregular"
)

# ──────────────────────────────────────────────────────────────────────────
# We now abandon 'neverhpfilter' and re-implement the Hamilton filter from
# scratch. This custom implementation allows us to:
#   (1) Target the Hamilton filter design in SSA.
#   (2) Modify filter characteristics beyond the original specification 
#       (customization).

# Retrieve quarterly real GDP data directly from FRED.
# No API key is needed for basic use with the alfred package.
reload_data <- FALSE

if (reload_data) {
  GDPC1 <- get_fred_series("GDPC1", series_name = "GDP")
  GDPC1 <- as.xts(GDPC1)
  save(GDPC1, file = file.path(getwd(), "Data", "GDPC1"))
} else {
  load(file = file.path(getwd(), "Data", "GDPC1"))
}
head(GDPC1)
tail(GDPC1)

# Convert the xts object to a plain numeric (double) vector.
# Reason: xts objects carry hidden metadata and index conventions that make
# direct application of SSA error-prone and unpredictable (e.g., applying a
# linear filter to an xts object can yield unexpected results).
#
# We also truncate the sample at end-2019 to exclude the COVID-19 pandemic.
# Pandemic-era outliers severely distort the Hamilton regression coefficients.
# The pandemic's impact on the filter is analysed separately in the final
# example.
y   <- as.double(log(GDPC1["/2019"]))
len <- length(y)

# ──────────────────────────────────────────────────────────────────────────
# 1.1 Replicate Hamilton Filter
# ──────────────────────────────────────────────────────────────────────────
# Hamilton's recommended settings for quarterly macroeconomic data:
h <- 2 * 4  # Forecast horizon: 8 quarters (2 years ahead).
p <- 4      # Number of autoregressive lags in the regression.

# Construct the regressor matrix for OLS:
# y_{t+h} is regressed on y_t, y_{t-1}, ..., y_{t-p+1}.
# The first column corresponds to y_t (the contemporaneous lag relative to
# the forecast origin).
explanatory <- y[(p):(len - h)]
for (i in 1:(p - 1))
  explanatory <- cbind(explanatory, y[(p - i):(len - h - i)])

# Define the dependent variable: log-GDP observed h periods ahead.
target <- y[(h + p):len]

# Fit the Hamilton regression over the full available sample.
# Note: The estimation window reaches back to the post-WWII period. A shorter
# sub-sample starting in 1990 is explored below with monthly data. 
lm_obj <- lm(y[(h + p):len] ~ explanatory)

summary(lm_obj)
# Outcomes:
# 1. Typically only the first lag coefficient is statistically significant — a
#     common finding when applying HF to non-stationary macroeconomic series.
# 2, The sum of the lag coefficients deviates from 1, so
#     (1 - sum(coefficients)) ≠ 0.
#     Consequence: the forecast residual y_{t+h} - ŷ_{t+h} is stationary
#     in-sample but may become non-stationary out-of-sample,
#     meaning the forecast and the actual future value are not `cointegrated'.

# Construct the Hamilton filter:
#   • Structure: [1, 0, ..., 0 (h-1 zeros), -β_1, -β_2, ..., -β_p]
#   • The leading 1 corresponds to y_{t+h}; the remaining entries are the
#     negative OLS coefficients.
hamilton_filter <- c(1, rep(0, h - 1), -lm_obj$coefficients[1 + 1:p])
par(mfrow = c(1, 1))
ts.plot(
  hamilton_filter,
  main = paste(
    "Hamilton filter: GDP from ",
    index(GDPC1["/2019"])[1], " to 2019",
    sep = ""
  )
)

# Extract the regression intercept (used later for replication and
# comparison). 
#   - Mean-centring of the cycle is skipped.
#   - SSA optimization (target correlation) is indifferent to 
#     level adjustments; the HT constraints assumes crossings of 
#     the mean level. 
intercept <- lm_obj$coefficients[1]
  
# --- Replicate the Hamilton Filter Output ---
# Construct the data matrix for filter application.
# Columns 1 through h correspond to the h-step-ahead and intermediate
# observations. Because the Hamilton filter coefficients are zero for lags
# 1 through h-1, those columns do not affect the output; they are filled by
# repeating 'target' h times for convenience. Columns h+1 through h+p
# contain the lagged regressors.
data_mat <- cbind(matrix(rep(target, h), ncol = h), explanatory)

# Apply the Hamilton filter: compute forecast residuals (= cycle estimate).
# Subtract the intercept to match the OLS residuals exactly.
residuals <- data_mat %*% hamilton_filter - intercept

# Verify replication: the computed residuals should be numerically identical
# to the OLS residuals in lm_obj$residuals. Both series must overlap
# perfectly in the plot.
ts.plot(
  cbind(residuals, lm_obj$residuals),
  main = "Replication of Hamilton cycle: both series overlap"
)

# ──────────────────────────────────────────────────────────────────────────
# 1.2 Unit-Root Adjustment
# ──────────────────────────────────────────────────────────────────────────

# Diagnostic check: the lag coefficients should sum close to 1, as required
# for the filter to cancel the stochastic trend (unit root) in log-GDP.
# Deviation from 1 indicates a risk of residual non-stationarity
# out-of-sample.
sum(lm_obj$coefficients[1 + 1:p])

# The sum of all Hamilton filter coefficients should be exactly zero to
# guarantee cancellation of a single unit root out-of-sample (cointegration
# condition for an I(1) process).
sum(hamilton_filter)

# The sum of the Hamilton filter coefficients is not exactly zero. Applying
# the filter out-of-sample would therefore generate a non-stationary cycle.
# To avoid this non-stationarity when replicating HF within SSA, it is
# useful to impose a unit-root adjustment. The adjustment is minor and affects 
# the cycle dynamics marginally. 

# --- Unit-Root Adjustment of the Hamilton Filter ---
Diff      <- TRUE
Ham_obj   <- HamiltonFilter_Restricted(y, p, h, Diff)

# Verify that the slope coefficients sum to exactly 1 under the constraint.
sum(Ham_obj$coefficients[-1])

# Construct the adjusted Hamilton filter coefficient vector.
hamilton_filter_adjusted <- c(1, rep(0, h - 1), -Ham_obj$coefficients[-1])

# Verify that the adjusted filter coefficients sum to exactly zero,
# confirming proper unit-root cancellation.
sum(hamilton_filter_adjusted)

# Comparison:
# Plot original and adjusted HF
par(mfrow = c(1, 1))
ts.plot(
  cbind(hamilton_filter,hamilton_filter_adjusted),col=c("black","grey"),
  main = paste(
    "Original and adjusted HF",
    sep = ""
  )
)
mtext("Original",line=-1)
mtext("Adjusted",col="grey",line=-2)


# ──────────────────────────────────────────────────────────────────────────
# 1.3 Plot and Compare Cycles
# ──────────────────────────────────────────────────────────────────────────

# Compute the adjusted cycle: apply the adjusted filter to the data matrix.
# No intercept subtraction needed here, as the zero-sum constraint absorbs 
# the mean.
residuals_adjusted <- data_mat %*% hamilton_filter_adjusted

# --- Visual Comparison of Cycle Estimates ---

# Plot 1: Three cycle variants overlaid.
#   Red:    Original Hamilton cycle (OLS forecast residuals).
#   Orange: Hamilton cycle shifted upward by the regression intercept.
#   Blue:   Unit-root-adjusted, uncentred cycle.
# The three series differ primarily in level; their dynamic patterns are
# virtually identical.
par(mfrow = c(1, 1))
ts.plot(
  cbind(residuals, residuals + intercept, residuals_adjusted),
  col  = c("red", "orange", "blue"),
  main = "Cycles"
)
mtext("Unit-root adjusted uncentred cycle",   col = "blue",   line = -1)
mtext("Hamilton Cycle",                       col = "red",    line = -2)
mtext("Hamilton cycle shifted by intercept",  col = "orange", line = -3)
abline(h = 0)

# Plot 2: Three-panel comparison of log-GDP and cycle variants.
par(mfrow = c(2, 2))
ts.plot(y, main = "Log(GDPC1)")
# Overlay original (red) and adjusted (blue) cycles.
ts.plot(
  cbind(residuals, residuals_adjusted),
  col  = c("red", "blue"),
  main = "Cycles"
)
mtext("Hamilton Cycle",                     col = "red",  line = -1)
mtext("Unit-root adjusted uncentred cycle", col = "blue", line = -2)
abline(h = 0)
# Plot the pointwise difference between the two cycle estimates over time.
ts.plot(residuals - residuals_adjusted, main = "Cycle difference")


# ==========================================================================
# MAIN TAKE-AWAY:
#   - The adjusted and original cycles share NEARLY IDENTICAL DYNAMICS;
#     the primary difference is a slowly evolving level offset (bottom plot).
#   - SSA CUSTOMIZATION WILL THEREFORE HAVE COMPARABLE EFFECTS ON EITHER
#     CYCLE VARIANT, SO THE CHOICE BETWEEN THEM DOES NOT MATERIALLY AFFECT
#     SSA-BASED CONCLUSIONS.
#   - For technical reasons (zero-sum filter constraint, cointegration
#     property), we proceed with the adjusted, uncentred cycle as the
#     basis for SSA customization.
# ==========================================================================


# ──────────────────────────────────────────────────────────────────────────
# 1.4 Transformation: From Levels to Differences
# ──────────────────────────────────────────────────────────────────────────
#
# To apply SSA to the Hamilton filter (for customisation), we must first
# transform the problem so that the filter operates on stationary
# (differenced) data.
# Theoretical background: Wildi, M. (2024),
# https://doi.org/10.1007/s41549-024-00097-5
#

# Construct the equivalent difference-domain filter 'ham_diff'.
# Set the filter length L: must be at least as long as the adjusted
# Hamilton filter.
L <- 20
L <- max(length(hamilton_filter_adjusted), L)

# Zero-pad the adjusted Hamilton filter to length L if necessary.
if (L > length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L <- c(
    hamilton_filter_adjusted,
    rep(0, L - length(hamilton_filter_adjusted))
  )

# Derive 'ham_diff' by convolving the (zero-padded) Hamilton filter with
# the summation (unit-root) filter. 
ham_diff <- conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv

# Plot and compare the two equivalent filter representations.
par(mfrow = c(2, 1))
ts.plot(ham_diff,
        main = "Hamilton filter expressed in terms of first differences")
ts.plot(hamilton_filter_adjusted_L,
        main = "Hamilton filter expressed in terms of levels")

# Verify numerical equivalence: both filter representations should produce
# identical cycle estimates when applied to their respective inputs.

# Compute first differences of log-GDP (stationary input for ham_diff).
x <- diff(y)
len_diff <- length(x)

# Apply 'ham_diff' to the differenced series to obtain the cycle estimate.
residual_diff <- na.exclude(filter(x, ham_diff, side = 1))

# Align series lengths by prepending NAs (cycle series are shorter due to
# filter initialization at the start of the sample).
original_hamilton_cycle <- c(rep(NA, length(x) - length(residuals)), residuals)

# Visual check: both approaches should produce overlapping cycle estimates.
par(mfrow = c(1, 1))
ts.plot(residual_diff, col = "blue",
        main = "HF applied to levels vs. differences: outputs should overlap")
lines(residuals_adjusted[(L - p - h + 2):length(residuals)], col = "red")

# With this equivalence established, we can apply SSA to HF by working
# entirely in the difference domain — using 'ham_diff' as the target filter.
# Note: the same transformation is required for BK (Tutorial 4) and HP-gap 
# (Tutorials 2 and 5).

# ──────────────────────────────────────────────────────────────────────────
# 1.4 Holding Times
# ──────────────────────────────────────────────────────────────────────────
#
# Before applying SSA, we characterize the zero-crossing frequency of the
# Hamilton filter via its theoretical holding time (computed under white noise 
# input).

# Compute the theoretical holding time of 'ham_diff'.
ht_ham_diff_obj <- compute_holding_time_func(ham_diff)

# Result: approximately one zero-crossing every 1.25 years.
ht_ham_diff_obj$ht

# Compute the empirical holding time directly from the adjusted Hamilton cycle.
compute_empirical_ht_func(residuals_adjusted)

# The empirical holding time is substantially longer than the theoretical value.
# Reason: the adjusted cycle is un-centered (slowly drifting positive mean),
# which suppresses zero-crossings and biases the empirical holding time.
par(mfrow = c(1, 1))
ts.plot(residuals_adjusted)
abline(h = 0)

# Centering the cycle narrows the gap between empirical and true HT: 
# 1 crossing all 2 years
compute_empirical_ht_func(scale(residuals_adjusted))

# ──────────────────────────────────────────────────────────────────────────
# 1.5 Autocorrelation Structure
# ──────────────────────────────────────────────────────────────────────────
#
# ACF of the differenced GDP.
acf(x, main = "ACF of first-differenced log-GDP")

# The ACF reveals weak serial dependence in the GDP growth rate series.
# For illustration and simplicity we here assume white noise:
#   - The resulting SSA design is revision-free (filter coefficients are
#     fixed).
#   - This simplification is also adopted in Wildi (2024).
#
# Set xi = NULL to indicate white noise (i.e., x_t = epsilon_t; no Wold
# decomposition needed).
xi <- NULL


# ──────────────────────────────────────────────────────────────────────────
# 1.6 Apply SSA to the Hamilton Filter
# ──────────────────────────────────────────────────────────────────────────
# 1.6.1 SSA settings
# --------------------------------------------------------------------------

# Theoretical holding time of the Hamilton filter (~1.25 years).
ht_ham_diff_obj$ht

# Our objective is a smoother SSA filter relative to HF. We increase the
# holding-time target by 50%: the SSA output will automatically exhibit
# fewer zero-crossings (i.e., smoother).
ht <- 1.5 * ht_ham_diff_obj$ht

# Convert the target holding time to the corresponding autocorrelation
# parameter rho1, which is the direct input to the SSA optimisation
# function.
rho1 <- compute_rho_from_ht(ht)

# Specify the SSA target filter: 
# - Since we work in first differences, we supply 'ham_diff' as
#   the target.
gammak_generic <- ham_diff

# Set forecast horizon: nowcast (h = 0).
# Note: HF is a causal filter. If h=0, SSA does not predict HF; it smooths it.
# If the HT constraint matches the HT of HF, and h=0, SSA replicates HF exactly.
forecast_horizon <- 0

# 1.6.2 Run SSA
# --------------------------------------------------------------------------

# Run the SSA optimisation. If xi is not supplied, the function defaults
# to the white noise assumption.
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1)

# The SSA function returns two filters:
#   ssa_x:   the primary filter applied directly to the observed data x_t.
#   ssa_eps: the convolved filter applied to the white noise innovations
#            epsilon_t (used mainly to verify the holding-time constraint).
# Under the white noise assumption (x_t = epsilon_t), both are identical.
SSA_filt_ham_diff <- SSA_obj_ham_diff$ssa_x

# Sanity check: confirm that ssa_x and ssa_eps are numerically
# identical (difference should be zero under white noise).
max(abs(SSA_obj_ham_diff$ssa_x - SSA_obj_ham_diff$ssa_eps))

# 1.6.3 Plot filters
# --------------------------------------------------------------------------

# Plot the target (Hamilton-diff) and SSA filters for visual comparison.
par(mfrow = c(1, 1))
mplot <- cbind(ham_diff, SSA_filt_ham_diff)
ts.plot(
  mplot,
  ylim = c(min(mplot), max(mplot)),
  col  = c("black", "blue"),
  main = "Target filter and SSA filter"
)
mtext(
  "Hamilton filter (applied to differenced data)",
  col  = "black",
  line = -1
)
mtext(
  paste(
    "SSA: holding time increased by ",
    100 * (ht / ht_ham_diff_obj$ht - 1), "%",
    sep = ""
  ),
  col  = "blue",
  line = -2
)
abline(h = 0)

# 1.6.4 Check numerical convergence
# --------------------------------------------------------------------------

# Verify convergence of the SSA optimisation: the effective holding time
# of the SSA filter should match the imposed target 'ht'.
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff)

# Effective holding time of the SSA filter:
ht_obj$ht

# Target holding time:
ht

# Both values should agree up to rounding errors, confirming convergence
# to the global optimum. 

# ──────────────────────────────────────────────────────────────────────────
# 1.7 Filter the Series and Compute Performance Measures
# ──────────────────────────────────────────────────────────────────────────
#
# Apply the SSA filter to the differenced log-GDP series.
SSA_out <- filter(x, SSA_filt_ham_diff, side = 1)

# Compare theoretical (imposed) and empirical holding times for SSA.
# They differ because the output series is not centred.
ht
compute_empirical_ht_func(SSA_out)

# Repeat for the Hamilton filter output.
ham_out <- filter(x, ham_diff, side = 1)
ht_ham_diff_obj$ht
compute_empirical_ht_func(ham_out)

# Compare empirical HTs of centred cycles. 
# - SSA now shows ~30% fewer crossings, as desired. 
# - Sample estimates deviate from true values due to misspecification 
#   (white noise assumption, non-stationarity).
# -The series runs from 1947 onward and is subject to multiple breaks/changes
compute_empirical_ht_func(scale(SSA_out))
compute_empirical_ht_func(scale(ham_out))

# Visual comparison of SSA and Hamilton filter outputs.
# Both cycles are closely aligned; SSA generates fewer crossings at the mean 
# line (blue horizontal).
mplot <- cbind(SSA_out, ham_out)
colo  <- c("blue", "red")
par(mfrow = c(1, 1))
ts.plot(
  mplot[, 1],
  col  = colo[1],
  main = "SSA vs. Hamilton (level offset not yet corrected)"
)
mtext("SSA",      col = colo[1], line = -1)
mtext("Hamilton", col = colo[2], line = -2)
lines(mplot[, 2], col = colo[2])
abline(h = mean(mplot[, 1], na.rm = T), col = "blue")


# ──────────────────────────────────────────────────────────────────────────
# 1.8 Forecasting: Gaining Timeliness Without Sacrificing Smoothness
# ──────────────────────────────────────────────────────────────────────────
# We explore whether SSA can LEAD the Hamilton cycle by increasing
# the forecast horizon, while maintaining the same smoothness (ht constraint).

# Set a 1-year (4-quarter) forecast horizon; all other settings remain 
# unchanged.
forecast_horizon <- 4
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)
SSA_filt_ham_diff_forecast <- SSA_obj_ham_diff$ssa_x

# Apply the forecast filter to the differenced data.
SSA_out_forecast <- filter(x, SSA_filt_ham_diff_forecast, side = 1)

# Compare empirical holding times: SSA-forecast remains  smoother
# than the Hamilton filter (fewer zero-crossings), despite the increased 
# forecast horizon.
compute_empirical_ht_func(SSA_out_forecast)
compute_empirical_ht_func(ham_out)
# Scale series: SSA has ~30% less crossings (up to sample variation)
compute_empirical_ht_func(scale(SSA_out_forecast))
compute_empirical_ht_func(scale(ham_out))

# Visual comparison of all three series (standardized for comparability).
mplot <- scale(cbind(SSA_out,SSA_out_forecast,ham_out))

colo <- c("blue", "darkgreen", "red")
par(mfrow = c(1, 1))
ts.plot(mplot[, 1], col = colo[1],
        ylim = c(min(mplot, na.rm = T), max(mplot, na.rm = T)))
lines(mplot[, 2], col = colo[2])
lines(mplot[, 3], col = colo[3])
mtext("SSA nowcast",                                    col = colo[1], line = -1)
mtext(paste("SSA forecast: horizon = ", forecast_horizon, " quarters", sep = ""),
      col = colo[2], line = -2)
mtext("Hamilton filter",                                col = colo[3], line = -3)
abline(h = 0)
# Key observations: 
# 1. The standardized cycles are slowly drifting downwards: this could be 
#    corrected straighftorwardly by incorporating the level drift in the 
#    bottom plot of exercise 1.3 (omitted).
# 2. The SSA forecast is shifted to the LEFT relative to the SSA nowcast 
#    and the Hamilton filter: it leads while preserving the same degree 
#    of smoothness (frequency of zero-crossings).


# ==============================================================================
# MAIN TAKE-AWAY:
# SSA demonstrates superior performance in timeliness (relative lead) and 
# smoothness (less crossings) compared to HF
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────
# 1.9 Amplitude and Phase-Shift Functions
# ──────────────────────────────────────────────────────────────────────────
# Frequency-domain filter characteristics provide a formal, data-independent
# complement to the above `time-domain' comparisons.
# Two key diagnostics:
#   - Amplitude function: measures gain at each frequency.
#     Values close to zero at high frequencies indicate effective noise 
#     suppression (less noisy crossings).
#   - Phase-shift function: measures the lag (or lead) of the filter at each 
#     frequency. A smaller phase-shift in the passband indicates a relative 
#     lead.

# Set the number of equidistant frequency ordinates on [0, pi].
# A finer grid (larger K) yields higher resolution.
K <- 600

# Compute amplitude and phase-shift functions for all three filters.
# - All filters are expressed as applied to first-differenced data.
amp_obj_SSA_now <- amp_shift_func(K, as.vector(SSA_filt_ham_diff),          F)
amp_obj_SSA_for <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_forecast),  F)
amp_obj_ham     <- amp_shift_func(K, ham_diff,                               F)

par(mfrow = c(2, 1))
# --- Panel 1: Amplitude Functions ---
mplot <- cbind(amp_obj_SSA_now$amp, amp_obj_SSA_for$amp, amp_obj_ham$amp)
# Normalize each amplitude function to 1 at frequency zero.
# This is always valid for lowpass filters and facilitates direct visual comparison
# of the relative attenuation across filters at higher frequencies.
mplot[, 1] <- mplot[, 1] / mplot[1, 1]
mplot[, 2] <- mplot[, 2] / mplot[1, 2]
mplot[, 3] <- mplot[, 3] / mplot[1, 3]
colnames(mplot) <- c(paste("SSA(", round(ht, 1), ", nowcast)", sep = ""),
                     paste("SSA(", round(ht, 1), ", h=", forecast_horizon, ")", sep = ""),
                     "Hamilton")
plot(mplot[, 1], type = "l", axes = F, xlab = "Frequency", ylab = "",
     main = "Amplitude Functions: SSA vs. Hamilton Filter",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# --- Panel 2: Phase-Shift Functions ---
# Phase-shift = phase angle divided by frequency.
# Interpretation: the number of periods by which the filter output lags (positive)
# or leads (negative) the true signal at each frequency.
# We focus on the passband (frequencies where amplitude > 0.5), as phase-shift
# is most meaningful where the filter passes signal rather than attenuating it.
mplot <- cbind(amp_obj_SSA_now$shift, amp_obj_SSA_for$shift, amp_obj_ham$shift)
colnames(mplot) <- c(paste("SSA(", round(ht, 1), ", nowcast)", sep = ""),
                     paste("SSA(", round(ht, 1), ", h=", forecast_horizon, ")", sep = ""),
                     "Hamilton")
plot(mplot[, 1], type = "l", axes = F, xlab = "Frequency", ylab = "",
     main = "Phase-Shift Functions: SSA vs. Hamilton Filter",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

#----------------------------------------
# Discussion of Results
#
# Amplitude Functions:
#
# 1. 'ham_diff', applied to first differences, behaves as a LOWPASS filter.
#    (The original Hamilton filter, applied to log-levels, is a bandpass.)
#
# 2. Both SSA amplitude functions lie BELOW that of 'ham_diff' at higher 
#    frequencies. 
#
# Interpretation:
#  - SSA more aggressively attenuates high-frequency noise components.
#  - This directly reduces the number of noisy (spurious) zero-crossings
#    in the cycle estimate relative to HF.
#  - This behavior — SSA damping high-frequency noise more effectively than
#    the benchmark filter — is a characteristic property of SSA designs.
#  - See Wildi 2026a, section 4.2, for background. 
#
# Phase-Shift Functions:
#
# 3. SSA nowcast phase-shift is marginally larger than HF in the passband.
#    This is negligible in practice, i.e., the SSA nowcast and HF 
#    are nearly time-aligned.
#
# 4. SSA forecast phase-shift is the SMALLEST of the three filters in the 
#    passband, confirming the relative lead of the SSA forecast over HF 
#    observed in the time-domain (filter outputs).



# ──────────────────────────────────────────────────────────────────────────
# Example 2: Monthly Data
# ──────────────────────────────────────────────────────────────────────────
# SSA applied to Hamilton filtering using monthly PAYEMS (non-farm payrolls)
# Illustrate the Hamilton filter on quarterly PAYEMS using the neverhpfilter package
library(neverhpfilter)

data(PAYEMS)
log_Employment <- 100 * log(xts::to.quarterly(PAYEMS["1947/2016-6"], OHLC = FALSE))

employ_trend <- yth_filter(log_Employment, h = 8, p = 4,
                           output = c("x", "trend"), family = gaussian)

plot.xts(employ_trend, grid.col = "white", legend.loc = "topleft",
         main = "Log employment and trend")

# In practice, cycle and irregular components are often correlated,
# contradicting standard decomposition assumptions
employ_cycle <- yth_filter(log_Employment, h = 8, p = 4,
                           output = c("cycle", "random"), family = gaussian)
par(mfrow = c(1,1))
plot.xts(employ_cycle, grid.col = "white", legend.loc = "topright",
         main = "Cycle and irregular components")
abline(h = 0)

# Log-returns are non-stationary: they exhibit drift and structural changes
# (e.g., pre/post 1960, pre/post 1990, WWII, Great Moderation).
plot(diff(log(PAYEMS)))
abline(h = 0)


# ──────────────────────────────────────────────────────────────────────────
# 2.1: Data selection: Post-1990 Analysis
# ──────────────────────────────────────────────────────────────────────────
# Novelties: 
# I)  Integrate a model of the data generating process for SSA
#     Motivation: first differences of PAYEMS are strongly autocorrelated
# II) Select a sub-sample starting in 1990 to escape severe non-stationarity 
#     issues
# Consequences: 
# The subsample selection affects two key components:
#   1. Hamilton filter: regression parameters are estimated on post-1990 data,
#      yielding a potentially different cycle definition.
#   2. ARMA model: the dependence structure (and hence the Wold decomposition xi)
#      reflects the smoother dynamics post 1990 (Great Moderation).
#
# NOTE: The pandemic period is excluded here (sample ends 2019).
#       Pandemic effects are analyzed separately in 2.9??? at the end of the tutorial.

# Load log-transformed PAYEMS

reload_data <- FALSE

if (reload_data) {
  PAYEMS <- get_fred_series("PAYEMS", series_name = "GDP")
  PAYEMS<-as.xts(PAYEMS)
  save(PAYEMS, file = file.path(getwd(), "Data", "PAYEMS"))
} else {
  load(file = file.path(getwd(), "Data", "PAYEMS"))
}
head(PAYEMS)
tail(PAYEMS)

# Select post-1990, pre-pandemic window
# An analysis of the COVID outliers will be analyzed subsequently.
y   <- as.double(log(PAYEMS["1990::2019"]))
ts.plot(y, main = "Log(PAYEMS): 1990–2019")
len <- length(y)

# ──────────────────────────────────────────────────────────────────────────
# 2.2 Hamilton Filter 
# ──────────────────────────────────────────────────────────────────────────
# Hamilton's original settings for quarterly data:
#   - Forecast horizon h = 2 years = 8 quarters
#   - AR order p = 4 (to account for up to 4 unit roots if present)
# Adaptation to monthly PAYEMS:
#   - h is scaled to 2 * 12 = 24 months
#   - p = 4 is retained 
h <- 2 * 12
p <- 4

# Construct the design matrix of lagged regressors and the forecast target
explanatory <- cbind(
  y[(p):(len - h)],
  y[(p - 1):(len - h - 1)],
  y[(p - 2):(len - h - 2)],
  y[(p - 3):(len - h - 3)]
)
target <- y[(h + p):len]

# Fit the Hamilton regression: h-step-ahead level on p lags
lm_obj <- lm(y[(h + p):len] ~ explanatory)

summary(lm_obj)
# Plot cycle
ts.plot(lm_obj$residuals, main = "Hamilton Cycle Residuals (Post-1990)")

# --- Construct the Hamilton Filter Coefficient Vector ---
# The filter maps the current and lagged levels to the cycle (regression residual).
hamilton_filter <- c(1, rep(0, h - 1), -lm_obj$coefficients[1 + 1:p])
intercept       <- lm_obj$coefficients[1]
  
# Replicate regression residuals using the filter for verification
# The data matrix stacks the h-step target (repeated h times) with the regressors
data_mat  <- cbind(matrix(rep(target, h), ncol = h), explanatory)
residuals <- data_mat %*% hamilton_filter - intercept

# Confirm exact replication of regression residuals
par(mfrow = c(1, 1))
ts.plot(cbind(residuals, lm_obj$residuals),
        main = "Replication Check: Hamilton Filter vs. Regression Residuals")

# ──────────────────────────────────────────────────────────────────────────
# 2.3 Adjusted Hamilton Filter 
# ──────────────────────────────────────────────────────────────────────────

# --- Cointegration Adjustment ---
# Impose a unit root to the filter, see example 1 above.
# Problem: the coefficients do not sum to zero 
sum(hamilton_filter)

# --- Unit-Root Adjustment of the Hamilton Filter ---
# Impose unit-root to regression
Diff=T
Ham_obj<-HamiltonFilter_Restricted(y, p, h, Diff )
# Verify: sum = 1
sum(Ham_obj$coefficients[-1])
#Ham_obj$cycle_xts
#Ham_obj$trend_xts 
hamilton_filter_adjusted<-c(1,rep(0,h-1),-Ham_obj$coefficients[-1])
sum(hamilton_filter_adjusted)
#Ham_obj$verification 

# Compute the cointegration-adjusted cycle
residuals_adjusted <- data_mat %*% hamilton_filter_adjusted


# --- Visualize Level, Cycle, and Cycle Difference ---
# The two cycle definitions (regression residuals vs. adjusted residuals) differ
# primarily in level; their shapes are nearly identical.
# See Examples 2–3 for a full discussion of this difference.
par(mfrow = c(2, 2))
ts.plot(y,                                         main = "Log(PAYEMS): 1990–2019")
ts.plot(cbind(residuals, residuals_adjusted),
        col  = c("red", "blue"),
        main = "Hamilton Cycle: Classic vs. Adjusted")
mtext("Classic cycle (regression residuals)",   col = "red",  line = -1)
mtext("Adjusted un-centered cycle",             col = "blue", line = -2)
ts.plot(residuals - residuals_adjusted,            main = "Difference Between Cycle Definitions")

# ==========================================================================
# MAIN TAKE-AWAY:
#   - The adjusted and original cycles share NEARLY IDENTICAL DYNAMICS;
#     the primary difference is a slowly evolving level offset (bottom plot).
#   - SSA CUSTOMIZATION WILL THEREFORE HAVE COMPARABLE EFFECTS ON EITHER
#     CYCLE VARIANT, SO THE CHOICE BETWEEN THEM DOES NOT MATERIALLY AFFECT
#     SSA-BASED CONCLUSIONS.
#   - For technical reasons (zero-sum filter constraint, cointegration
#     property), we proceed with the adjusted, uncentred cycle as the
#     basis for SSA customization.
# ==========================================================================

# ──────────────────────────────────────────────────────────────────────────
# 2.4 Transformation: Level to First Differences
# ──────────────────────────────────────────────────────────────────────────
# Apply the same level-to-differences transformation as in Example 1,
# but with a larger filter length L = 100.
#
# Motivation: post-1990 log-returns exhibit slower ACF decay (longer memory),
# consistent with the Great Moderation (lower volatility, more persistent dynamics).
# A larger L is required to ensure that the Wold decomposition xi converges to zero.
# See Section 2.3 and Proposition 4 in Wildi 2024 for background.
L <- 100

# Ensure L is at least as long as the Hamilton filter (required for valid convolution)
L <- max(length(hamilton_filter_adjusted), L)
if (L > length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L <- c(hamilton_filter_adjusted,
                                  rep(0, L - length(hamilton_filter_adjusted)))

# Convolve the adjusted Hamilton filter with the unit-root (summation) filter
ham_diff <- conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv

# Visualize both filter representations for comparison
par(mfrow = c(2, 1))
ts.plot(ham_diff,                       main = "Hamilton Filter: Applied to First Differences x_t")
ts.plot(hamilton_filter_adjusted_L,     main = "Hamilton Filter: Applied to Log-Level y_t")

# --- Verify Filter Equivalence ---
# Applying ham_diff to log-returns x_t should reproduce the same cycle as
# applying hamilton_filter_adjusted_L to the log-level y_t.
x        <- diff(y)
len_diff <- length(x)
residual_diff <- na.exclude(filter(x, ham_diff, side = 1))

# Align series lengths by prepending NAs (filter outputs are shorter than input)
original_hamilton_cycle <- c(rep(NA, length(x) - length(residuals)),    residuals)

# Confirmation plot: both approaches should produce identical cycle estimates
par(mfrow = c(1, 1))
ts.plot(residual_diff, col = "blue",
        main = "Replication Check: HF Applied to Differences vs. Levels")
lines(residuals_adjusted[(L - p - h + 2):length(residuals)], col = "red")

# With ham_diff verified, we can now apply SSA in the differences domain.

# ──────────────────────────────────────────────────────────────────────────
# 2.5 Holding Time Analysis
# ──────────────────────────────────────────────────────────────────────────

# Compute the theoretical holding time of ham_diff under the white noise assumption
ht_ham_diff_obj <- compute_holding_time_func(ham_diff)

ht_ham_diff_obj$ht

# Sample HT
compute_empirical_ht_func(residuals_adjusted)
# The sample holding time (from filtered output) is much longer than the 
# theoretical value.
# Root causes: 
# i)  mean is not vanishing
# ii) x_t (log-returns) are not white noise — they exhibit strong positive 
#     autocorrelation.

# After centering: the rate of mean crossings is closer to expected HT 
# (however, we have not yet accounted for the autocorrelation in data)
compute_empirical_ht_func(scale(residuals_adjusted))

# Visual confirmation that log-returns are not white noise
par(mfrow = c(1, 1))
ts.plot(x, main = "Log-Returns of PAYEMS (Post-1990)")
abline(h = 0)
# Strong dependence, i.e., slowly decaying ACF:
acf(x)

# ──────────────────────────────────────────────────────────────────────────
# 2.6 Autocorrelation Analysis and ARMA Model Fitting
# ──────────────────────────────────────────────────────────────────────────
# OPTIONAL: Split the sample in half to assess out-of-sample robustness.
#   - SSA is largely insensitive to moderate ARMA misspecification,
#     provided the model is not severely overfitted.
#   - Both in-sample and out-of-sample parameterizations yield nearly identical results.
#   - The comments below refer to the full-sample estimation (try_out_of_sample = FALSE).
try_out_of_sample <- F

if (try_out_of_sample) {
  # Use the first half of x for ARMA estimation (out-of-sample robustness check)
  in_sample_length <- length(x) / 2
} else {
  # Use the full sample for ARMA estimation
  in_sample_length <- length(x)
}

# ACF of post-1990 log-returns: 
acf(x[1:in_sample_length], main = "ACF: Slowly Decaying (Post-1990, Longer Memory)")

# Fit an ARMA(1,1) model to capture the persistent but weak autocorrelation structure
ar_order  <- 1
ma_order  <- 1
estim_obj <- arima(x[1:in_sample_length], order = c(ar_order, 0, ma_order))

estim_obj

# Residual diagnostics: confirm that the ARMA(1,1) adequately captures the ACF structure
tsdiag(estim_obj)

# --- Compute the Wold Decomposition (MA-Infinity Representation) ---
# xi contains the infinite-order MA coefficients (impulse response weights).
# L = 100 was chosen to ensure xi decays sufficiently close to zero by lag L.
xi <- c(1, ARMAtoMA(
  ar      = estim_obj$coef[1:ar_order],
  ma      = estim_obj$coef[ar_order + 1:ma_order],
  lag.max = L - 1
))

# Visualize xi: 
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition xi: Slowly Decaying Impulse Response (Post-1990)")

# Convolve xi with ham_diff to obtain the composite filter applied to innovations epsilon_t.
# This convolved filter is used solely for computing the holding time constraint.
ham_conv        <- conv_two_filt_func(xi, ham_diff)$conv
ht_ham_conv_obj <- compute_holding_time_func(ham_conv)

# Compare expected and sample HTs (the latter based on centered series):
ht_ham_conv_obj$ht
compute_empirical_ht_func(scale(residuals_adjusted))

# RESULT: 
# After adjustment of autocorrelation and non-vansishing mean, expected and 
# sample Holding Times (mean crossings) are in close agreement

# ──────────────────────────────────────────────────────────────────────────
# 2.7 Apply SSA (Post-1990, ARMA-Informed)
# ──────────────────────────────────────────────────────────────────────────
#---------------------------------------------------------------------------
# 2.7.1 SSA settings
#---------------------------------------------------------------------------
# The theoretical holding time of ham_conv is slightly under one year
ht_ham_conv_obj$ht

# Target a 50% longer holding time: SSA should generate approximately 30% fewer crossings
# (under correct model specification and stationarity)
ht   <- 1.5 * ht_ham_conv_obj$ht

# Compute the autocorrelation parameter rho1 corresponding to the target holding time
rho1 <- compute_rho_from_ht(ht)

# Define the benchmark filter: SSA aims to improve upon ham_diff (applied to x_t)
gammak_generic <- ham_diff

# Set forecast horizon to zero (nowcast only; forecasting addressed in step 4.7)
forecast_horizon <- 0

#---------------------------------------------------------------------------
# 2.7.2 Run SSA
#---------------------------------------------------------------------------

# Fit SSA with the ARMA(1,1) Wold decomposition xi supplied
# This ensures SSA correctly accounts for the autocorrelation structure of x_t
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# ssa_eps: filter in the innovation (epsilon_t) domain
#   => Used primarily to verify convergence of the optimization to the global maximum
SSA_filt_ham_diff_eps <- SSA_obj_ham_diff$ssa_eps

# ssa_x: filter in the observable data (x_t) domain
#   => Operationally relevant filter for real-time signal extraction
SSA_filt_ham_diff_x <- SSA_obj_ham_diff$ssa_x

#---------------------------------------------------------------------------
# 2.7.3 Plot filters
#---------------------------------------------------------------------------

# --- Compare Filters in the x_t Domain ---
# ham_diff and ssa_x are both applied to x_t: direct apples-to-apples comparison
mplot <- cbind(ham_diff, SSA_filt_ham_diff_x)
par(mfrow = c(1, 1))
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"),
        main = "Filter Coefficients Applied to Log-Diff Series x_t")
mtext("Hamilton filter",   col = "black", line = -1)
mtext("SSA filter (ssa_x)", col = "blue",  line = -2)

# --- Compare Filters in the Epsilon_t Domain ---
# ham_conv and ssa_eps are both applied to innovations epsilon_t:
# the natural comparison domain within the Wold representation
mplot <- cbind(ham_conv, SSA_obj_ham_diff$ssa_eps)
par(mfrow = c(1, 1))
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"),
        main = "Convolved Filter Coefficients Applied to Innovations epsilon_t")
mtext("Hamilton (convolved with xi)", col = "black", line = -1)
mtext("SSA (ssa_eps)",                col = "blue",  line = -2)

#---------------------------------------------------------------------------
# 2.7.4 Check numerical convergence
#---------------------------------------------------------------------------

# --- Verify Global Convergence of the SSA Optimization ---
# The holding time of ssa_eps must match the targeted ht (up to numerical rounding).
# Agreement between the two numbers confirms convergence to the global maximum.
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff_eps)
ht_obj$ht  # Computed holding time of ssa_eps
ht         # Targeted holding time


# ──────────────────────────────────────────────────────────────────────────
# 2.8 Filter the Series and Evaluate Performance
# ──────────────────────────────────────────────────────────────────────────

# --- 4.6.1 SSA Filter Output ---
# Apply the SSA nowcast filter (in x_t space) to log-returns using one-sided filtering
SSA_out <- filter(x, SSA_filt_ham_diff_x, side = 1)

# The empirical holding time exceeds the targeted ht.
# Root cause: x_t (log-returns) are still non-stationary.
compute_empirical_ht_func(scale(SSA_out))
ht   # Targeted holding time for reference

# --- 4.6.2 Hamilton Filter Output ---
# Apply the Hamilton filter (in x_t space) to log-diff x
ham_out <- filter(x, ham_diff, side = 1)
# SSA generates 30% less mean-crossings, as intended
compute_empirical_ht_func(scale(ham_out))
# The appropriate theoretical benchmark for Hamilton's empirical ht is ht_ham_conv
# (which accounts for autocorrelation via the ARMA model), not ht_ham_diff.
ht_ham_conv_obj$ht

# --- Visual Comparison of SSA and Hamilton Outputs ---
# Both outputs appear visually similar, but Hamilton exhibits greater high-frequency
# leakage, producing noisier zero-crossings relative to SSA.
mplot <- scale(na.exclude(cbind(SSA_out, ham_out)))
colo  <- c("blue", "red")
ts.plot(mplot[, 1], col = colo[1])
lines(mplot[, 2], col = colo[2])
abline(h = 0)
mtext("Hamilton filter output", col = "red",  line = -1)
mtext("SSA filter output",      col = "blue", line = -2)

# Empirical holding time comparison:
# SSA achieves approximately 50% longer holding time (30% less mean crossings) than Hamilton,
# consistent with the targeted improvement (difference is within sampling error).
compute_empirical_ht_func(scale(SSA_out))
compute_empirical_ht_func(scale(ham_out))



# ──────────────────────────────────────────────────────────────────────────
# 2.9 Forecasting: Timeliness and Lead/Lag Analysis
# ──────────────────────────────────────────────────────────────────────────
# =============================================================================
# OBJECTIVE
# =============================================================================
# Goal:
#   Construct faster (more timely) concurrent SSA filters by augmenting the
#   forecast horizon. 
#
# Design strategy:
#   - The holding-time constraint is held fixed throughout, ensuring that the
#     noise-suppression properties of the filter are preserved regardless of
#     the chosen forecast horizon.
#   - Extending the forecast horizon improves timeliness (reduces phase lag)
#     while maintaining smoothness — achieving both desirable properties
#     simultaneously.
#
# Trade-off:
#   - The dual gain in smoothness and timeliness comes at a cost: accuracy
#     (as measured by target correlation) deteriorates disproportionately
#     as the forecast horizon is extended.
#   - In other words, the filter becomes faster and smoother, but at the
#     expense of tracking the target signal less faithfully.
#
# Note:
#   - The forecast horizon addresses timeliness indirectly. The novel 
#     look-ahead DFP and PCS are Pareto optimal and trace the accuracy-
#     timeliness frontier (tutorial in preparation).
#
# =============================================================================
# 2.9.1 Forecast Horizon Specification
# =============================================================================
# We define two forecast horizons to compare against the concurrent (nowcast)
# filter (delta = 0):
#   - delta =  6: six months ahead  (half-year horizon)
#   - delta = 12: twelve months ahead (full-year horizon)
#
# Note: SSA_func accepts a vector of horizons and returns a filter matrix
#       with one column per forecast horizon.
forecast_horizon <- c(6, 12)


# =============================================================================
# 2.9.2 SSA Filter Estimation: Forecast Horizons
# =============================================================================
# Fit SSA filters for both forecast horizons using:
#   - Filter length L
#   - ARMA(1,1) Wold decomposition coefficients xi
#   - Autocovariance sequence gammak_generic
#   - HT constraint parameter rho1
#   - xi Wold decomposition of ARMA(1,1)
#
# The returned object contains filters expressed in both the x_t domain
# and the epsilon_t (innovations) domain.
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract the estimated filter coefficients in the x_t domain
# (one column per forecast horizon):
SSA_filt_ham_diff_x_forecast <- SSA_obj_ham_diff$ssa_x


# =============================================================================
# 2.9.3 Visualization: Filter Coefficient Morphing Across Horizons
# =============================================================================
# As the forecast horizon increases, the filter shifts progressively more weight
# toward recent observations and away from the remote past — a morphing toward
# a leading (anticipating) shape.
#
# Crucially, the holding-time constraint is held fixed across all horizons,
# so smoothness (noise suppression) is preserved as timeliness improves.
par(mfrow = c(1, 1))
ts.plot(SSA_filt_ham_diff_x_forecast,
        col  = c("orange", "darkgreen"),
        main = "SSA Filter Coefficients: Nowcast vs. Forecast Horizons",
        ylim = c(min(SSA_filt_ham_diff_x), max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x, col = "blue")
mtext(paste("SSA forecast: delta =", forecast_horizon[1]), col = "orange",    line = -1)
mtext(paste("SSA forecast: delta =", forecast_horizon[2]), col = "darkgreen", line = -2)
mtext("SSA nowcast: delta = 0",                            col = "blue",      line = -3)


# =============================================================================
# 2.9.4 Convergence Verification: Holding Times Across Forecast Horizons
# =============================================================================
# We apply compute_holding_time_func to each column of ssa_eps (the filter
# expressed in the epsilon_t innovations domain).
#
# If the holding times match the target ht across all forecast horizons,
# this confirms that the optimizer converged to the global optimum
# for each horizon.
apply(SSA_obj_ham_diff$ssa_eps, 2, compute_holding_time_func)
ht   # Target holding time (for reference)


# =============================================================================
# 2.9.5 Apply Forecast Filters to the First-Differenced Series
# =============================================================================
# Compute the filter output for each forecast horizon by convolving the
# estimated filter coefficients with the first-differenced input series x.

# Six-month-ahead SSA forecast filter output:
SSA_out_forecast_6  <- filter(x, SSA_filt_ham_diff_x_forecast[, 1], side = 1)

# Twelve-month-ahead SSA forecast filter output:
SSA_out_forecast_12 <- filter(x, SSA_filt_ham_diff_x_forecast[, 2], side = 1)


# =============================================================================
# Visualization: All Filter Outputs
# =============================================================================
# We overlay all filter outputs (nowcast + two forecast horizons + Hamilton)
# to assess the timeliness-smoothness trade-off visually.
#
# Expected pattern:
#   - SSA forecast outputs are left-shifted (leading) relative to the Hamilton
#     filter, with the degree of lead increasing with the forecast horizon.
#   - Smoothness (cycle duration) is approximately equal across all SSA variants,
#     reflecting the binding holding-time constraint.
mplot <- scale(na.exclude(cbind(SSA_out, SSA_out_forecast_6, SSA_out_forecast_12, ham_out)))
colo  <- c("blue", "orange", "darkgreen", "red")

par(mfrow = c(1, 1))
mplot <- scale(na.exclude(cbind(SSA_out, SSA_out_forecast_6, SSA_out_forecast_12, ham_out)))
colo  <- c("blue", "orange", "darkgreen", "red")
par(mfrow = c(1, 1))
ts.plot(mplot[, 1], col = colo[1], ylim = c(min(mplot), max(mplot)))
lines(mplot[, 2], col = colo[2])
lines(mplot[, 3], col = colo[3])
lines(mplot[, 4], col = colo[4])
mtext("SSA nowcast (delta = 0)",                           col = colo[1], line = -1)
mtext(paste("SSA forecast: delta =", forecast_horizon[1]), col = colo[2], line = -2)
mtext(paste("SSA forecast: delta =", forecast_horizon[2]), col = colo[3], line = -3)
mtext("Hamilton filter",                                    col = colo[4], line = -4)
abline(h = 0)

# =============================================================================
# 2.9.6 Empirical Holding Times: SSA Variants vs. Hamilton Benchmark
# =============================================================================
# We compute empirical holding times (average duration between zero-crossings)
# for all filter outputs.
#
# Expected result:
#   - All SSA variants should achieve longer holding times than Hamilton,
#     confirming that the holding-time constraint enforces
#     stronger noise suppression across all forecast horizons.
#   - Holding times should be approximately equal (up to random sampling 
#     error) across SSA variants, since all share the same constraint.
compute_empirical_ht_func(scale(SSA_out))             # SSA nowcast  (delta = 0)
compute_empirical_ht_func(scale(SSA_out_forecast_6))  # SSA forecast (delta = 6)
compute_empirical_ht_func(scale(SSA_out_forecast_12)) # SSA forecast (delta = 12)
compute_empirical_ht_func(scale(ham_out))             # Hamilton benchmark


# =============================================================================
# 2.10 Amplitude and Phase-Shift Functions
# =============================================================================
# PURPOSE:
#   Provide formal frequency-domain diagnostics that confirm the empirical
#   findings of Section 2.7. Two complementary functions are examined:
#
#   Amplitude function:
#     - Quantifies the filter's gain at each frequency.
#     - High-frequency amplitude close to zero indicates strong noise suppression,
#       which corresponds directly to longer holding times (smoother output).
#
#   Phase-shift function:
#     - Quantifies timing distortion introduced by the filter at each frequency.
#     - A more negative (smaller) phase shift in the passband (low frequencies)
#       indicates a relative lead — i.e., the filter output anticipates
#       turning points in the target signal.
#
# Together, these diagnostics characterise the smoothness-timeliness trade-off
# across the SSA nowcast, SSA forecast variants, and the Hamilton benchmark.


# -----------------------------------------------------------------------------
# Setup: Frequency Grid and Color Palette
# -----------------------------------------------------------------------------
# Number of equidistant frequency ordinates over [0, pi]:
K <- 600
# Colors assigned consistently across all plots:
#   Column 1 — SSA nowcast  (delta = 0)
#   Column 2 — SSA forecast (delta = 6 months)
#   Column 3 — SSA forecast (delta = 12 months)
#   Column 4 — Hamilton benchmark
colo <- c("black", "blue", "darkgreen", "red")
# -----------------------------------------------------------------------------
# Compute Amplitude and Phase-Shift Objects for All Four Filters
# -----------------------------------------------------------------------------
# All filters are expressed in the x_t domain (i.e., applied to log-return series).
# amp_shift_func returns both the amplitude and phase-shift at each frequency ordinate.
amp_obj_SSA_now    <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x),               F)
amp_obj_SSA_for_6  <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 1]), F)
amp_obj_SSA_for_12 <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 2]), F)
amp_obj_ham        <- amp_shift_func(K, ham_diff,                                      F)
# -----------------------------------------------------------------------------
# Panel Layout: Amplitude (top panel) and Phase-Shift (bottom panel)
# -----------------------------------------------------------------------------
par(mfrow = c(2, 1))
# PLOT 1: Amplitude Functions
# =============================================================================
# Assemble amplitude matrix (one column per filter):
mplot <- cbind(
  amp_obj_SSA_now$amp,
  amp_obj_SSA_for_6$amp,
  amp_obj_SSA_for_12$amp,
  amp_obj_ham$amp
)
# Normalize each amplitude curve to unity at frequency zero (DC component).
# This enables a scale-free comparison of noise-suppression profiles across
# filters with potentially different overall gains.
mplot[, 1] <- mplot[, 1] / mplot[1, 1]
mplot[, 2] <- mplot[, 2] / mplot[1, 2]
mplot[, 3] <- mplot[, 3] / mplot[1, 3]
mplot[, 4] <- mplot[, 4] / mplot[1, 4]
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                   ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1],  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2],  ")", sep = ""),
  "Hamilton"
)
# Plot all normalized amplitude functions.
# Interpretation: a smaller amplitude at high frequencies implies stronger
# attenuation of noise, consistent with a longer empirical holding time.
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Amplitude Functions: Hamilton vs. SSA Variants (Post-1990)",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
# Overlay amplitude curves for remaining filters:
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()
# PLOT 2: Phase-Shift Functions
# =============================================================================
# Assemble phase-shift matrix (one column per filter):
mplot <- cbind(
  amp_obj_SSA_now$shift,
  amp_obj_SSA_for_6$shift,
  amp_obj_SSA_for_12$shift,
  amp_obj_ham$shift
)
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                   ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1],  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2],  ")", sep = ""),
  "Hamilton"
)
# Plot all phase-shift functions.
# Interpretation: a smaller phase shift at low (passband) frequencies
# indicates that the filter output leads the target signal — i.e., turning
# points are anticipated. A larger shift implies a delay in signal detection.
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Phase-Shift Functions: Hamilton vs. SSA Variants (Post-1990)",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
# Overlay phase-shift curves for remaining filters:
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

#
# =============================================================================
# DISCUSSION: Frequency-Domain Diagnostics — Amplitude and Phase-Shift
# =============================================================================
#
# The two plots above reveal a clear and consistent pattern across three
# evaluation criteria: smoothness, timeliness, and accuracy.
#
# -----------------------------------------------------------------------------
# Smoothness:
# -----------------------------------------------------------------------------
#   - In the top panel (amplitude functions), all SSA designs exhibit smaller
#     amplitudes than the Hamilton filter (HF) at high frequencies.
#   - Stronger attenuation of high-frequency components translates directly
#     into fewer zero-crossings and smoother filter output — confirming that
#     the holding-time constraint is binding and effective across all SSA variants.
#
# -----------------------------------------------------------------------------
# Timeliness:
# -----------------------------------------------------------------------------
#   - In the bottom panel (phase-shift functions), increasing the forecast
#     horizon progressively reduces the phase shift at low (passband) frequencies.
#   - A smaller phase shift implies that the filter output leads
#     the target HF signal — i.e., turning points are detected earlier as the
#     forecast horizon grows.
#
# -----------------------------------------------------------------------------
# Accuracy:
# -----------------------------------------------------------------------------
#   - As the forecast horizon increases, the SSA amplitude functions deviate
#     more strongly from the Hamilton filter target at low frequencies,
#     indicating growing passband distortion and a loss of accuracy.
#   - Zero-shrinkage pulls SSA filter coefficients toward zero, further
#     attenuating the passband gain. This effect is not visible in the plots
#     due to the unit normalization applied at frequency zero.
#   - Notably, the SSA nowcast (black curve) is nearly indistinguishable
#     from the Hamilton filter across the entire passband:
#       * Amplitude functions are virtually identical at low frequencies.
#       * Phase-shift functions are virtually identical at low frequencies,
#         implying equivalent timeliness between the two.
#     The SSA nowcast thus matches Hamilton in both accuracy and timeliness,
#     while strictly dominating it in smoothness (stronger high-frequency
#     attenuation).
#
# -----------------------------------------------------------------------------
# Key Implication:
# -----------------------------------------------------------------------------
#   - The fact that SSA nowcast (black) outperforms HF in smoothness 
#     (amplitude in stop band) without sacrificing accuracy (amplitude in 
#     passband) or timeliness (phase-shift in passband) demonstrates that HF is
#     suboptimal: it does not lie on the efficient Accuracy-Timeliness-
#     Smoothness (ATS) frontier.
#   - Put differently, a better filter exists — one that achieves the same
#     signal-tracking (amplitude in passband) and timing performance 
#     (time-shift in passband) as HF while attenuating noise (amplitude in 
#     stop band) more effectively.
#   - Extending the forecast horizon beyond the nowcast trades accuracy for
#     timeliness (for fixed HT smoothness) , tracing out a frontier of 
#     Pareto-improving designs relative to Hamilton along the smoothness-
#     timeliness dimension.
# Note: 
#   - Similar findings applied to HP (tutorial 2.1) where SSA could replicate
#     HP smoothness (HT) while improving MSE substantially.



# =============================================================================
# 2.11 Out-of-Sample Application Including the Pandemic Period
# =============================================================================
# PURPOSE:
#   Apply all filters estimated on the 1990–2019 subsample — without any
#   re-estimation — to the extended series that includes the pandemic period
#   (2020 onwards). This out-of-sample exercise serves two purposes:
#     (1) Assess whether filter properties established in-sample hold up when
#         exposed to extreme, unprecedented observations.
#     (2) Exploit the pandemic shock as a natural impulse experiment
#         to visualise each filter's impulse response function directly 
#         in the data.
#
# KEY INSIGHT:
#   The extreme pandemic outliers appear as impulses in the log-differenced
#   series. By linearity, each filter responds to such an impulse with an output
#   proportional to its impulse response function. 


# -----------------------------------------------------------------------------
# Load Extended Sample: 1990 to Present (Including Pandemic)
# -----------------------------------------------------------------------------
y <- as.double(log(PAYEMS["1990/"]))
x <- diff(y)

# Visualise log-returns: the pandemic shock (early 2020) manifests as extreme
# outliers — a large negative return followed by a large positive rebound.
par(mfrow = c(1, 1))
ts.plot(x, main = "Log-Returns of PAYEMS: 1990–Present (Including Pandemic)")
abline(h = 0)


# -----------------------------------------------------------------------------
# Apply All Pre-Estimated Filters Out-of-Sample (No Re-Estimation)
# -----------------------------------------------------------------------------
# Filter coefficients are held fixed at their 1990–2019 in-sample estimates.
# Any differences in output across filters reflect differences in filter design,
# not differences in the estimation sample.

# 1. SSA nowcast filter (delta = 0):
SSA_out <- filter(x, SSA_filt_ham_diff_x, side = 1)

# 2. Hamilton filter (benchmark):
ham_out <- filter(x, ham_diff, side = 1)

# 3. SSA 6-month-ahead forecast filter (delta = 6):
SSA_out_forecast_6 <- filter(x, SSA_filt_ham_diff_x_forecast[, 1], side = 1)

# 4. SSA 12-month-ahead forecast filter (delta = 12):
SSA_out_forecast_12 <- filter(x, SSA_filt_ham_diff_x_forecast[, 2], side = 1)


# -----------------------------------------------------------------------------
# Visualise All Filter Outputs Over the Extended Sample
# -----------------------------------------------------------------------------
# Expected pattern:
#   The large negative pandemic impulse (early 2020) is followed by a secondary
#   positive "phantom" peak in each filter output. This phantom peak is a filter
#   artifact — a direct consequence of the negative lobe in each filter's impulse
#   response — and does not reflect any real economic signal.
#   Its magnitude and timing vary across filter designs, reflecting differences
#   in coefficient shape and rate of decay.

par(mfrow = c(1, 1))
mplot <- na.exclude(cbind(SSA_out, SSA_out_forecast_6, SSA_out_forecast_12, ham_out))
colo  <- c("blue", "orange", "darkgreen", "red")
par(mfrow = c(1, 1))
ts.plot(mplot[, 1], col = colo[1],
        ylim = c(min(mplot, na.rm = TRUE), max(mplot, na.rm = TRUE)),
        main = "Out-of-Sample Filter Outputs: 1990–Present (Including Pandemic)")
lines(mplot[, 2], col = colo[2])
lines(mplot[, 3], col = colo[3])
lines(mplot[, 4], col = colo[4])
mtext("SSA nowcast (delta = 0)",                           col = colo[1], line = -1)
mtext(paste("SSA forecast: delta =", forecast_horizon[1]), col = colo[2], line = -2)
mtext(paste("SSA forecast: delta =", forecast_horizon[2]), col = colo[3], line = -3)
mtext("Hamilton filter",                                    col = colo[4], line = -4)
abline(h = 0)


# -----------------------------------------------------------------------------
# Revisit Filter Coefficients to Interpret the Pandemic Response
# -----------------------------------------------------------------------------
# The shape of the phantom post-pandemic peak in each filtered output is 
# determined by the corresponding filter's coefficient sequence (impulse response).
# Plotting the coefficients side by side explains:
#   (a) The relative magnitude of the phantom peak across filter designs.
#   (b) The temporal location of the phantom peak, which shifts with the
#       forecast horizon due to the morphing of coefficient weights.
par(mfrow = c(1, 1))
ts.plot(SSA_filt_ham_diff_x_forecast,
        col  = c("orange", "darkgreen"),
        main = "SSA Filter Coefficients: Forecast Horizons vs. Nowcast",
        ylim = c(min(SSA_filt_ham_diff_x), max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x, col = "blue")
mtext(paste("Forecast horizon delta =", forecast_horizon[1]), col = "orange",    line = -1)
mtext(paste("Forecast horizon delta =", forecast_horizon[2]), col = "darkgreen", line = -2)
mtext("Nowcast (delta = 0)",                                   col = "blue",      line = -3)


# -----------------------------------------------------------------------------
# Interpretation: Forecast Filters Recover More Rapidly from Extreme Outliers
# -----------------------------------------------------------------------------
# The phantom post-pandemic peak is a filter artifact whose duration is governed
# by the rate at which the filter's impulse response decays to zero.
#
# Key finding — advantage of longer forecast horizons in the presence of outliers:
#   - Forecast filters (delta > 0) assign less weight to the remote past and
#     consequently decay to zero faster than the nowcast or Hamilton filter.
#   - As a result, they "forget" extreme singular observations — such as the
#     pandemic shock — more quickly, limiting the duration of filter-induced
#     distortions in the output.
#   - Specifically: the 12-month-ahead SSA forecast filter clears the pandemic
#     outlier approximately 1.5 years after its occurrence.
#   - By contrast, the SSA nowcast and Hamilton filter require an additional
#     10–11 months to fully dissipate the impulse response, sustaining the
#     phantom peak for a materially longer period.
#




###################################################################################################
# SUMMARY 
###################################################################################################
#
# PROPOSED FRAMEWORK:
#   - A variant of the Hamilton filter (HF) was constructed onto which SSA can be grafted
#     via the level-to-differences transformation (ham_diff).
#   - A simple regression-based adjustment was proposed to map the SSA output back to
#     the original (classic) Hamilton cycle scale and level for direct comparison.
#   - After transformation, cycles were indistinguishable.
#
# ROBUSTNESS TO MODEL MISSPECIFICATION:
#   - SSA is broadly robust to misspecification of the data's dependence structure.
#   - Full-sample analyses (Examples 2–3, WWII to 2023):
#       * The long span introduces substantial structural non-stationarity (changing dynamics).
#       * White noise and ARMA(1,1) specifications yielded broadly similar filter designs.
#       * Empirical holding times were generally upward-biased, but correcting for
#         autocorrelation (via the ARMA model) consistently reduced this bias.
#       * The targeted 50% ht improvement was not always fully realized on the full sample.
#         This reflects non-stationarity rather than a failure of the SSA methodology.
#         A stronger constraint (e.g., 100% increase) could be imposed if desired.
#   - Post-1990 subsample (Example 4):
#       * Restricting to the Great Moderation era substantially reduced structural
#         misspecification, yielding tighter agreement between theoretical and empirical
#         holding times across all SSA variants.
#
# ROBUSTNESS TO ARMA OVERFITTING:
#   - SSA is also robust to moderate overfitting of the ARMA model.
#   - For typical applications, a parsimonious ARMA(1,1) is sufficient to capture
#     the autocorrelation structure relevant for the holding time constraint.
#
# SSA NOWCAST PERFORMANCE:
#   - SSA nowcasts consistently outperform the Hamilton target in terms of smoothness
#     (fewer and cleaner zero-crossings).
#   - Empirical improvements are broadly commensurate with theoretical specifications,
#     especially when the sample is sufficiently stationary and the model is not severely
#     misspecified.
#
# SSA FORECAST PERFORMANCE:
#   - SSA forecast filters achieve the same smoothness as the nowcast while simultaneously
#     outperforming the Hamilton target in terms of timeliness (left shift at zero-crossings).
#   - This comes at no cost in noise suppression — the holding time constraint is unchanged.
#   - However, simultaneous improvement in speed, smoothness, and MSE accuracy is
#     impossible: this is the filter design trilemma.
#     => See Tutorial 0.1 for a formal exposition of the trilemma.
#   - SSA optimally positions the filter within the limits of the trilemma:
#     it achieves the best possible MSE for any given combination of holding time and phase-shift.
#
# OUTLOOK:
#   - A more refined and operationally effective treatment of timeliness (left shift)
#     is obtained by the novel Look-Ahead DFP/PCS predictors (tutorial in preparation) 
###################################################################################################
















