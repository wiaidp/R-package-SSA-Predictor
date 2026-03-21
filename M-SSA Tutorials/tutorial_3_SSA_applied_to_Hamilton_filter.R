# ========================================================================
# Tutorial 3: SSA Applied to Hamilton's Regression Filter (HF)
# ========================================================================
# This tutorial demonstrates how SSA can be applied to Hamilton's regression
# filter (HF) to improve its real-time business cycle analysis (BCA) properties.
#
# Two datasets are used for illustration:
#   - Example 1: Quarterly US GDP
#   - Examples 2 & 3: Monthly US non-farm payroll employment
# Both long samples (starting post-WWII) and shorter samples (starting at the
# Great Moderation, ~1985) are analysed to assess robustness across regimes.
#
# ------------------------------------------------------------------------
# Main findings:
# ------------------------------------------------------------------------
#
#   1. HF is a low-pass filter (when expressed in terms of returns/differences):
#        - It mitigates the generation of spurious business cycles.
#        - This is a key advantage over band-pass designs such as Baxter-King (BK),
#          Christiano-Fitzgerald (CF), and the HP-gap filter, which are all prone
#          to spurious cycles — see Tutorials 2 (Example 7), 4, and 5.
#
#   2. HF removes the residual weak drift in the differenced series:
#        - It can handle arbitrary integration orders by increasing the AR lag
#          order p in the regression specification.
#
#   3. HF is inherently sample-dependent (defined by an in-sample regression):
#        - The out-of-sample cycle is non-stationary because the estimated
#          regression coefficients do not sum to one (the filter does not cancel unit roots out-of-sample).
#        - The regression must be continuously updated as new observations arrive,
#          which introduces undesirable real-time revisions to historical estimates.
#
#   4. HF has a relatively short holding-time and a relatively large phase-lag:
#        - Short holding-time => noisy zero-crossings (many false turning-point signals).
#        - Large phase-lag => retardation (turning points are detected with delay).
#        - Both deficiencies make HF a natural candidate for SSA enhancement.
#
#   5. SSA nowcast target: approximately 30% fewer zero-crossings than HF
#        (i.e., a ~30% larger holding-time), suppressing noise while retaining
#        the low-pass character of the original filter.
#
#   6. SSA forecasts (6-month and 12-month horizons):
#        - Remain smooth (holding-time constraint is satisfied).
#        - Are leading relative to HF (reduced phase-lag / improved timeliness).
#
# ------------------------------------------------------------------------
# Broader motivation:
# ------------------------------------------------------------------------
# The goal of this tutorial is not to advocate for any specific Business Cycle tool.
# Rather, it illustrates a general principle:
#
#   Any linear filter — including HF — can be replicated and systematically
#   improved by SSA with respect to two practical BCA priorities:
#     1. Smoothness: suppression of high-frequency noise (fewer false signals).
#     2. Timeliness: earlier detection of cyclical turning points.
#
# In this context, HF serves as a convenient baseline platform and a showcase
# for the SSA optimisation principle. A range of quantitative performance
# measures is provided to confirm the practical value of the approach.
# ========================================================================

# ── BACKGROUND ────────────────────────────────────────────────────
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

# Theoretical background:
#   Wildi, M. (2026a) Sign Accuracy, Mean-Squared Error and the Rate
#     of Zero Crossings: a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547

# ─────────────────────────────────────────────────────────────────


# Make a clean-sheet, load packages and functions

rm(list=ls())

library(xts)
# This package implements the `never hpfilter', i.e., the Hamilton filter:  
# It is used here for background information only (generate some plots): we deploy own code for the Hamilton filter 
# The Hamilton filter is a breeze to implement: simple linear regression.
#   But the filter can change substantially, depending on the sample window: see examples 2 and 3
library(neverhpfilter)
# Load data from FRED with library quantmod
library(quantmod)


# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

# ==================================================================================
# Example 1
# ==================================================================================
# Application of SSA to the Hamilton Filter (HF) using quarterly U.S. GDP data.
# We first demonstrate HF using the R-package 'neverhpfilter',
# which also provides the data source (GDPC1).
data(GDPC1)

# Apply the Hamilton filter: decompose log-GDP (scaled by 100) into trend and cycle.
# h = 8 quarters (2-year forecast horizon), p = 4 lags (quarterly AR order).
gdp_trend <- yth_filter(100*log(GDPC1), h = 8, p = 4, output = c("x", "trend"))

# Plot the original log-GDP series together with the estimated trend.
plot.xts(gdp_trend, grid.col = "white", legend.loc = "topleft", main = "Log of GDP and trend")

# Extract the cyclical and irregular (random) components from the Hamilton filter.
gdp_cycle <- yth_filter(100*log(GDPC1), h = 8, p = 4, output = c("cycle", "random"))

# Note: The cycle and random components often exhibit strong correlation,
# which contradicts standard business-cycle assumptions of independence.
# The cycle is centered around zero. HF removes up to p unit roots,
# where p is the autoregressive order used in the regression.
plot.xts(gdp_cycle, grid.col = "white", legend.loc = "topleft", main = "Cycle and irregular")

#---------------------------------------------------------
# We now abandon the 'neverhpfilter' package and re-implement the Hamilton filter from scratch.
# This custom implementation allows us to:
#   (1) Graft SSA onto the Hamilton filter design.
#   (2) Modify filter characteristics beyond the original specification.

# Retrieve quarterly real GDP data directly from FRED.
getSymbols('GDPC1', src = 'FRED')

# Convert the xts object to a plain numeric (double) vector.
# Reason: xts objects carry hidden metadata and conventions that make
# direct application of SSA error-prone, unintuitive, and unpredictable
# (e.g., applying a linear filter to an xts object can yield unexpected results).
#
# We also truncate the sample at end-2019 to exclude the COVID-19 pandemic period.
# Pandemic-era outliers severely distort the Hamilton regression coefficients.
# The pandemic's impact on the filter is analyzed separately in the final example.
y <- as.double(log(GDPC1["/2019"]))
len <- length(y)

#------------------
# 1.1 Hamilton Filter
# Hamilton's recommended settings for quarterly macroeconomic data:
h <- 2 * 4   # forecast horizon: 8 quarters (2 years ahead)
p <- 4       # number of autoregressive lags in the regression

# Construct the regressor matrix for the OLS regression:
# y_{t+h} is regressed on y_t, y_{t-1}, ..., y_{t-p+1}
# The first column corresponds to y_t (contemporaneous lag relative to the forecast origin).
explanatory <- y[(p):(len - h)]
for (i in 1:(p - 1))
  explanatory <- cbind(explanatory, y[(p - i):(len - h - i)])

# Define the dependent variable: log-GDP observed h periods ahead.
target <- y[(h + p):len]

# Fit the Hamilton regression over the full available sample.
# Note: The estimation window is very long, reaching back to the post-WWII period.
# A shorter subsample starting in 1990 is explored in Example 4.
# This long-sample issue does not qualitatively affect the relative advantages of SSA over HF.
lm_obj <- lm(y[(h + p):len] ~ explanatory)

# Interpretation of regression output:
# Typically, only the first lag coefficient is statistically significant —
# a common finding when applying HF to non-stationary macroeconomic series.
# This is problematic because it implies the sum of the lag coefficients
# deviates from 1, so (1 - sum(coefficients)) ≠ 0.
# Consequence: the forecast residual y_{t+h} - ŷ_{t+h} is stationary in-sample
# (due to overfitting) but may become non-stationary out-of-sample,
# meaning the forecast and the actual future value are not 'cointegrated'
# (the MSE of the forecast diverges asymptotically).
# Root cause: the growth rate (drift) of log-GDP is itself time-varying,
# so first differences are non-stationary (second-order non-stationarity).
# Using p = 4 lags absorbs this in-sample, but the fix does not carry over out-of-sample.
# As a result, the model must be continuously re-estimated as new data arrives,
# which introduces forecast revisions over time.
summary(lm_obj)

# Construct the Hamilton filter coefficient vector.
# Structure: [1, 0, ..., 0 (h-1 zeros), -β_1, -β_2, ..., -β_p]
# The leading 1 corresponds to the h-step-ahead observation y_{t+h},
# and the remaining entries are the negated OLS slope coefficients.
hamilton_filter <- c(1, rep(0, h - 1), -lm_obj$coefficients[1 + 1:p])
par(mfrow=c(1,1))
ts.plot(hamilton_filter, main = paste("Hamilton filter: GDP from ",
                                      index(GDPC1["/2019"])[1], " to 2019", sep = ""))

# Extract the regression intercept (used later for replication and comparison).
# Mean-centering of the cycle is skipped here.
intercept <- lm_obj$coefficients[1]
  
  # --- Replicate the Hamilton Filter Output ---
  # Construct the data matrix for filter application.
  # Columns 1 through h correspond to the h-step-ahead and intermediate observations.
  # Since the Hamilton filter coefficients are zero for lags 1 through h-1,
  # those columns do not affect the output; we fill them by repeating 'target' h times
  # for convenience. Columns h+1 through h+p contain the lagged regressors.
  data_mat <- cbind(matrix(rep(target, h), ncol = h), explanatory)

# Apply the Hamilton filter: compute forecast residuals (= cycle estimate).
# Subtract the intercept to match the OLS residuals exactly.
residuals <- data_mat %*% hamilton_filter - intercept

# Verify replication: the computed residuals should be numerically identical
# to the OLS residuals stored in lm_obj$residuals. Both series should overlap perfectly.
ts.plot(cbind(residuals, lm_obj$residuals),
        main = "Replication of Hamilton cycle: both series overlap")

# Diagnostic check: the lag coefficients should sum close to 1,
# as required for the filter to cancel the stochastic trend (unit root) in log-GDP.
# Deviation from 1 indicates residual non-stationarity risk out-of-sample.
sum(lm_obj$coefficients[1 + 1:p])

# The sum of all Hamilton filter coefficients should be exactly zero
# to guarantee cancellation of a (single) unit root out-of-sample (cointegration condition if the process is I(1)).
sum(hamilton_filter)

# The sum is not exactly zero due to in-sample overfitting in the OLS regression.
# This is a mild but notable drawback of HF compared to HP:
#   HP gap filter coefficients sum to exactly zero by construction,
#   though HP tends to generate spurious cyclical fluctuations (see Tutorial 2, Example 7).

# --- Unit-Root Adjustment of the Hamilton Filter ---
# We correct the filter so that its coefficients sum to exactly zero,
# enforcing the cointegration constraint (no long-run bias out-of-sample).
# Approach: distribute the non-zero sum evenly across the p AR-lag coefficients.
# Alternative redistribution schemes are possible, but since the correction
# is much smaller than the OLS sampling error, the choice is inconsequential.
# These adjustments do not materially affect relative SSA vs. HF performance comparisons.
hamilton_filter_adjusted <- hamilton_filter
hamilton_filter_adjusted[(h + 1):(h + p)] <-
  hamilton_filter_adjusted[(h + 1):(h + p)] - sum(hamilton_filter) / p

# Confirm that the adjusted filter now sums to exactly zero.
sum(hamilton_filter_adjusted)

# Compute the adjusted cycle: apply the corrected filter to the data matrix.
# No intercept subtraction needed here, as the zero-sum constraint absorbs the mean.
residuals_adjusted <- data_mat %*% hamilton_filter_adjusted

# Optional: center the adjusted cycle around zero.
# Skipped here; we retain the un-centered adjusted cycle for the SSA customization.
if (F)
  residuals_adjusted <- residuals_adjusted - mean(residuals_adjusted)

# --- Visual Comparison of Cycle Estimates ---

# Plot 1: Three cycle variants overlaid.
#   Red:    Original Hamilton cycle (OLS forecast residuals).
#   Orange: Hamilton cycle shifted upward by the regression intercept.
#   Blue:   Unit-root-adjusted, un-centered cycle.
# The three series differ primarily in level; their dynamic patterns are virtually identical.
par(mfrow = c(1, 1))
ts.plot(cbind(residuals, residuals + intercept, residuals_adjusted),
        col = c("red", "orange", "blue"), main = "Cycles")
mtext("Unit-root adjusted un-centered cycle", col = "blue",   line = -1)
mtext("Hamilton Cycle",                        col = "red",    line = -2)
mtext("Hamilton cycle shifted by regression intercept", col = "orange", line = -3)
abline(h = 0)

# Plot 2: Four-panel comparison of log-GDP and cycle variants.
par(mfrow = c(2, 2))
ts.plot(y, main = "Log(GDPC1)")

# Overlay original (red) and adjusted (blue) cycles.
ts.plot(cbind(residuals, residuals_adjusted), col = c("red", "blue"), main = "Cycles")
mtext("Hamilton Cycle",                       col = "red",  line = -1)
mtext("Unit-root adjusted un-centered cycle", col = "blue", line = -2)
abline(h = 0)

# Plot the pointwise difference between the two cycle estimates over time.
ts.plot(residuals - residuals_adjusted, main = "Cycle difference")

# Analysis:
# - The level of the adjusted cycle drifts slowly over time,
#   reflecting the gradual slowdown in the growth rate (drift) of log-GDP.
# - Consequently, the level difference between the two cycles is largest
#   in the early sample (when GDP growth was stronger) and shrinks
#   toward the end of the sample (when growth was weaker).
# - We give a slight preference to the adjusted cycle, as it is conceptually
#   closer to the NBER definition of recessions (periods of negative output growth):
#     * The original Hamilton cycle is negative roughly 50% of the time,
#       requiring an upward level shift to serve as a recession indicator.
#     * The adjusted cycle is already 'uplifted' relative to the original,
#       though it may sit somewhat too high during the pre-financial-crisis period.
# - The cycle difference (bottom panel) resembles a rescaled copy of log-GDP itself,
#   directly reflecting the time-varying drift in the underlying series.
# - Since both cycles are statistical constructs rather than observable quantities,
#   the choice between them is ultimately subjective and application-dependent.

################################################################################
# MAIN TAKE-AWAY:
#   - The adjusted and original cycles share nearly identical dynamics;
#     the primary difference is a slowly evolving level offset.
#   - SSA customization will therefore have comparable effects on either cycle variant,
#     so the choice between them does not materially affect SSA-based conclusions.
#   - For technical reasons (zero-sum filter constraint, cointegration property),
#     we proceed with the adjusted, un-centered cycle as the basis for SSA grafting.
################################################################################

# Store the level difference between the original and adjusted cycles.
# This offset is used later to back-transform SSA results back to the
# original Hamilton cycle scale, enabling direct comparison.
cycle_diffh <- residuals - residuals_adjusted

#---------------------------------------------------
# 1.2 Transformation: From Levels to Differences
#
# To graft SSA onto the Hamilton filter, we must first transform the problem
# so that the filter operates on stationary (differenced) data.
# Theoretical background: Wildi, M. (2024), https://doi.org/10.1007/s41549-024-00097-5
#
# Motivation:
#   - The concept of zero-crossings (and holding times) is not well-defined
#     for non-stationary (integrated) series (an extension to integrated processes is given in Wildi (2026a))
#   - We therefore re-express HF as an equivalent filter applied to first differences
#     rather than levels.
#   - This transformation applies broadly to all bandpass-cycle filters that operate
#     on data in levels: CF, BK (Tutorial 4), HP-gap (Tutorials 2 and 5).
#
# Key ideas:
#   - The filter INPUT (log-GDP) is non-stationary (integrated).
#   - The filter OUTPUT (cycle estimate) is approximately stationary.
#   - Therefore, the bandpass filter must annihilate the unit root at frequency zero.
#     (HF can remove more than one unit root, depending on the AR order p.)
#   - We can always construct an equivalent filter — called 'ham_diff' below —
#     that, when applied to first differences, replicates the output of the original
#     Hamilton filter applied to levels.
#   - See Section 2.3 and Proposition 4 of the JBCY paper for the formal derivation.

# Construct the equivalent difference-domain filter 'ham_diff'.
# Set the filter length L: must be at least as long as the adjusted Hamilton filter.
L <- 20
L <- max(length(hamilton_filter_adjusted), L)

# Zero-pad the adjusted Hamilton filter to length L if necessary.
if (L > length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L <- c(hamilton_filter_adjusted,
                                  rep(0, L - length(hamilton_filter_adjusted)))

# Derive 'ham_diff' by convolving the (zero-padded) Hamilton filter with
# the summation (unit-root) filter. This encodes the assumption of one unit root
# and yields a filter suitable for application to first-differenced data.
ham_diff <- conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv

# Plot and compare the two equivalent filter representations.
# Note: 'ham_diff' coefficients decay to zero for lags beyond the length of
# 'hamilton_filter_adjusted', confirming that L = length(hamilton_filter_adjusted)
# would suffice as the minimum filter length.
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
cycle_diff <- c(rep(NA, length(x) - length(cycle_diffh)), cycle_diffh)
original_hamilton_cycle <- c(rep(NA, length(x) - length(residuals)), residuals)

# Visual check: both approaches should produce overlapping cycle estimates.
par(mfrow = c(1, 1))
ts.plot(residual_diff, col = "blue",
        main = "HF applied to levels vs. differences: outputs should overlap")
lines(residuals_adjusted[(L - p - h + 2):length(residuals)], col = "red")

# With this equivalence established, we can now graft SSA onto HF by working
# entirely in the difference domain — using 'ham_diff' as the target filter.
# The same transformation is required for BK (Tutorial 4) and HP-gap (Tutorials 2 and 5).

#-------------------------------------------------------------------------------
# 1.3 Holding Times
#
# Before applying SSA, we characterize the zero-crossing frequency of the
# Hamilton filter via its theoretical holding time (computed under white noise input).

# Compute the theoretical holding time of 'ham_diff'.
ht_ham_diff_obj <- compute_holding_time_func(ham_diff)

# Result: approximately one zero-crossing every 1.5 years.
ht_ham_diff_obj$ht

# Compute the empirical holding time directly from the adjusted Hamilton cycle.
compute_empirical_ht_func(residuals_adjusted)

# The empirical holding time is substantially longer than the theoretical value.
# Reason: the adjusted cycle is un-centered (slowly drifting positive mean),
# which suppresses zero-crossings and inflates the empirical holding time.
par(mfrow = c(1, 1))
ts.plot(residuals_adjusted)
abline(h = 0)

# In principle, a level shift (translation) of the series does not affect
# the SSA computation itself. However, the theoretical holding time no longer
# reflects the empirical zero-crossing rate, introducing a bias — see
# Tutorial 1, Example 7, for a detailed discussion.
# This bias will be corrected when we back-transform the SSA output to match
# the original Hamilton cycle later in this example.

#-----------------------------------------------------------------------
# 1.4 Autocorrelation Structure
#
# Section 2 of the JBCY paper extends the SSA framework to autocorrelated input data.
# We first inspect the ACF of the differenced series to assess dependency.
acf(x, main = "ACF of first-differenced log-GDP")

# The ACF reveals weak serial dependence in the GDP growth rate series.
# In such cases, we recommend defaulting to the white noise assumption
# (Section 1 of the JBCY paper) for the following reasons:
#   - The white noise assumption is broadly appropriate for many economic series
#     (consistent with Granger's typical spectral shape for macroeconomic data).
#   - The resulting SSA design is revision-free (filter coefficients are fixed).
#   - This simplification is also adopted in Section 4 of the JBCY paper when
#     applying HP and SSA to INDPRO (see Tutorial 5).
#
# Set xi = NULL to indicate white noise (i.e., x_t = epsilon_t, no Wold decomposition needed).
xi <- NULL

# Note: Examples 3 and 4 demonstrate how to incorporate a fitted time-series model
# when the data exhibits meaningful autocorrelation.

#--------------------------------------
# 1.5 Applying SSA to the Hamilton Filter
#
# We are now ready to graft SSA onto HF. The setup satisfies the key requirements:
#   - Stationarity: achieved via the transformation to 'ham_diff'.
#   - Wold decomposition: specified by xi (white noise in this case).
# Remaining caveat: the adjusted cycle is un-centered and slowly drifting,
# so the theoretical holding time is biased (see Section 1.3 and Tutorial 1, Example 7).

# Recall the theoretical holding time of the Hamilton filter (~1.5 years).
ht_ham_diff_obj$ht

# Our objective is a smoother SSA filter relative to HF.
# We increase the holding-time target by 50%: the SSA output will automatically
# exhibit fewer zero-crossings (i.e., smoother, less noisy cycle estimates).
ht <- 1.5 * ht_ham_diff_obj$ht

# Convert the target holding time to the corresponding autocorrelation parameter rho1,
# which is the direct input accepted by the SSA optimization function.
rho1 <- compute_rho_from_ht(ht)

# Specify the SSA target filter: SSA should approximate the Hamilton-diff filter.
# Since we work in the difference domain, we supply 'ham_diff' as the target.
gammak_generic <- ham_diff

# Set forecast horizon: nowcast (h = 0 means no look-ahead).
forecast_horizon <- 0

# Run the SSA optimization.
# If xi is not supplied, the function defaults to white noise input.
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# The SSA function returns two filters:
#   ssa_x:   the primary filter, applied directly to the observed data x_t.
#   ssa_eps: the convolved filter that would be applied to the white noise
#            innovations epsilon_t (used mainly to verify the holding-time constraint).
# Under the white noise assumption (x_t = epsilon_t), both filters are identical.
SSA_filt_ham_diff <- SSA_obj_ham_diff$ssa_x

# Optional sanity check: confirm that ssa_x and ssa_eps are numerically identical
# (difference should be zero under white noise).
if (F) {
  SSA_obj_ham_diff$ssa_x - SSA_obj_ham_diff$ssa_eps
}

# Plot the target (Hamilton-diff) and SSA filters for visual comparison.
par(mfrow = c(1, 1))
mplot <- cbind(ham_diff, SSA_filt_ham_diff)
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"),
        main = "Target filter and SSA filter")
mtext("Hamilton filter (difference domain)",          col = "black", line = -1)
mtext(paste("SSA: holding time increased by ",
            100 * (ht / ht_ham_diff_obj$ht - 1), "%", sep = ""),
      col = "blue", line = -2)
abline(h = 0)

# Verify convergence of the SSA optimization:
# The effective holding time of the SSA filter should match the imposed target 'ht'.
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff)

# Effective holding time of SSA filter:
ht_obj$ht

# Target holding time:
ht

# Both values should agree up to rounding errors, confirming convergence to the
# global optimum. In practice, convergence is reliable for standard forecast problems
# of this type, as the optimization landscape is well-behaved.

#--------------------------------------------------
# 1.6 Filter the Series and Compute Performance Measures
#
# Our version of HF preserves the slowly changing drift (slope) in log-GDP
# rather than removing it. Whether this is desirable depends on the application;
# we consider it appropriate here.
# Below, we also provide a simple back-transformation that aligns the SSA output
# with the original (approximately zero-centered) Hamilton cycle.

# Apply the SSA filter to the differenced log-GDP series.
SSA_out <- filter(x, SSA_filt_ham_diff, side = 1)

# Compare theoretical (imposed) and empirical holding times for SSA.
# They differ because the output series is not centered around zero
# (the drifting mean suppresses zero-crossings, inflating the empirical ht).
ht
compute_empirical_ht_func(SSA_out)

# Repeat for the Hamilton filter output, for reference.
ham_out <- filter(x, ham_diff, side = 1)
ht_ham_diff_obj$ht
compute_empirical_ht_func(ham_out)

# Result: the empirical holding time of SSA is approximately 50% larger than that
# of HF, broadly consistent with the imposed constraint (the remaining discrepancy
# is due to the level-bias discussed in Section 1.3).

# Visual comparison of SSA and Hamilton filter outputs.
# Both cycles are closely aligned; the Hamilton filter is slightly noisier
# (more spectral leakage outside the target band).
# Note: the level offset between the two series has not yet been corrected here.
mplot <- cbind(SSA_out, ham_out)
colo <- c("blue", "red")
par(mfrow = c(1, 1))
ts.plot(mplot[, 1], col = colo[1],
        main = "SSA vs. Hamilton (level offset not yet corrected)")
mtext("SSA",      col = colo[1], line = -1)
mtext("Hamilton", col = colo[2], line = -2)
lines(mplot[, 2], col = colo[2])
abline(h = 0)

# --- Back-Transformation: Align SSA with the Original Hamilton Cycle ---
#
# To enable direct comparison with the original (OLS-based) Hamilton cycle,
# we apply a two-step adjustment to the SSA output:
#
#   Step 1: Account for the level offset between the adjusted and original cycles,
#           captured by 'cycle_diff' (= original Hamilton cycle - adjusted cycle).
#   Step 2: Correct for scaling differences between SSA_out, cycle_diff, and
#           original_hamilton_cycle, which arise from different filter normalizations.
#
# Rather than computing the exact scaling analytically from filter properties,
# we use a simple OLS regression of the original Hamilton cycle on both components.
# This is justified because the scaling correction is minor relative to other sources
# of uncertainty.
lm_obj <- lm(original_hamilton_cycle ~ cycle_diff + SSA_out - 1)
coef <- lm_obj$coef

# Construct the back-transformed SSA series: a weighted combination of the
# level-offset correction and the scaled SSA output.
scale_shifted_SSA <- coef[1] * cycle_diff + coef[2] * SSA_out

# Plot: back-transformed SSA vs. original Hamilton cycle.
# The SSA series is shorter (NAs at the start) due to filter initialization.
ts.plot(scale_shifted_SSA, col = colo[1],
        main = "Back-transformed SSA cycle vs. original Hamilton cycle")
mtext("Shifted and scaled SSA",   col = colo[1],   line = -1)
mtext("Original Hamilton cycle",  col = "black",   line = -2)
lines(original_hamilton_cycle)
abline(h = 0)

# Compare empirical holding times after back-transformation.
# SSA produces approximately 30% fewer zero-crossings than the Hamilton filter,
# confirming the desired smoothness improvement.
# Note: the empirical holding times are now much closer to the theoretical values
# because the back-transformation has corrected key sources of mis-specification
# (level offset and scaling), as discussed in Tutorial 1, Exercise 7.
compute_empirical_ht_func(original_hamilton_cycle)
compute_empirical_ht_func(scale_shifted_SSA)

# Compute the relative lead/lag between SSA and HF via the Tau-statistic.
# Reference: Wildi, M. (2024), https://doi.org/10.1007/s41549-024-00097-5
#
# Methodology:
#   - One series is shifted relative to the other across a range of lead/lag values.
#   - At each shift, the absolute difference in zero-crossing timings between
#     the two series is computed.
#   - The shift minimizing this timing difference is the estimated lead or lag.
#
# Result: the minimum is achieved at lead/lag = 0, confirming that the SSA nowcast
# and the Hamilton cycle are temporally synchronized (no lead or lag).
mat <- cbind(scale_shifted_SSA, original_hamilton_cycle)
lead_lag_obj <- compute_min_tau_func(mat)
# SSA improves smoothness (larger HT) without sacrificing timeliness in this case

#-----------------------------------------------
# 1.7 Forecasting: Gaining Timeliness Without Sacrificing Smoothness
#
# The previous section confirmed that the SSA nowcast (forecast_horizon = 0)
# is synchronized with the Hamilton cycle while being smoother.
# We now explore whether SSA can also LEAD the Hamilton cycle by increasing
# the forecast horizon, while maintaining the same smoothness (ht constraint).

# Set a 1-year (4-quarter) forecast horizon; all other settings remain unchanged.
forecast_horizon <- 4
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)
SSA_filt_ham_diff_forecast <- SSA_obj_ham_diff$ssa_x

# Apply the forecast filter to the differenced data.
SSA_out_forecast <- filter(x, SSA_filt_ham_diff_forecast, side = 1)

# Compare empirical holding times: SSA-forecast remains approximately 30% smoother
# than the Hamilton filter (fewer zero-crossings), despite the increased forecast horizon.
compute_empirical_ht_func(SSA_out_forecast)
compute_empirical_ht_func(ham_out)

# Visual comparison of all three series (standardized to unit variance for comparability).
# Key observation: the SSA forecast is shifted to the LEFT relative to both
# the SSA nowcast and the Hamilton filter — i.e., it leads — while preserving
# the same degree of smoothness.
mplot <- cbind(SSA_out          / sd(SSA_out,          na.rm = T),
               SSA_out_forecast / sd(SSA_out_forecast, na.rm = T),
               ham_out          / sd(ham_out,          na.rm = T))
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

# --- Back-Transformation: Align SSA Forecast with Original Hamilton Cycle ---
#
# Apply the same OLS-based back-transformation as in Section 1.5,
# now using the forecast filter output (SSA_out_forecast) instead of the nowcast.
lm_obj <- lm(original_hamilton_cycle ~ cycle_diff + SSA_out_forecast - 1)
coef <- lm_obj$coef
scale_shifted_SSA <- coef[1] * cycle_diff + coef[2] * SSA_out_forecast

# Plot: back-transformed SSA forecast vs. original Hamilton cycle.
# The SSA forecast is visibly smoother and shifted to the left (leading).
ts.plot(scale_shifted_SSA, col = colo[2],
        main = "Back-transformed SSA forecast vs. original Hamilton cycle")
mtext("Shifted and scaled SSA forecast", col = colo[2], line = -1)
mtext("Original Hamilton cycle",         col = "black", line = -2)
lines(original_hamilton_cycle)
abline(h = 0)

# Confirm smoothness improvement: SSA forecast has fewer
# zero-crossings than the original Hamilton cycle.
compute_empirical_ht_func(original_hamilton_cycle)
compute_empirical_ht_func(scale_shifted_SSA)

# Compute the Tau-statistic lead/lag for the SSA forecast.
# Result: the minimum is now achieved at lead = -1 quarter,
# confirming that the SSA forecast leads the Hamilton cycle by one quarter —
# consistent with the visual impression from the previous plot.
mat <- cbind(scale_shifted_SSA, original_hamilton_cycle)
lead_lag_obj <- compute_min_tau_func(mat)


########################################################################################################
# MAIN TAKE-AWAY:
# SSA demonstrates superior performance in timeliness (relative lead) and smoothness (less crossings)
########################################################################################################

# We verify this statement based on frequency-domain characteristics: amplitude and time-shift

#----------------------------------------
# 1.8 Amplitude and Phase-Shift Functions
#
# Frequency-domain filter characteristics provide a formal, data-independent
# complement to the empirical comparisons in Sections 1.5 and 1.6.
# Two key diagnostics:
#   - Amplitude function: measures gain at each frequency.
#     Values close to zero at high frequencies indicate effective noise suppression
#     (less spectral leakage into the cycle estimate).
#   - Phase-shift function: measures the lag (or lead) of the filter at each frequency.
#     A smaller phase-shift in the passband indicates a relative lead over the benchmark.

# Set the number of equidistant frequency ordinates on [0, pi].
# A finer grid (larger K) yields smoother, more readable curves.
K <- 600

# Compute amplitude and phase-shift functions for all three filters.
# All filters are expressed in the difference domain (applied to first-differenced data).
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
# 2. Both SSA amplitude functions lie BELOW that of 'ham_diff' at higher frequencies.
#    Interpretation:
#      - SSA more aggressively attenuates high-frequency noise components.
#      - This directly reduces the number of noisy (spurious) zero-crossings
#        in the cycle estimate relative to HF.
#      - This behavior — SSA damping high-frequency noise more effectively than
#        the benchmark filter — is a characteristic property of SSA smoothing designs
#        and motivates the term 'noisy crossings' for benchmark filter outputs.
#
# Phase-Shift Functions:
#
# 3. SSA nowcast phase-shift is marginally larger than HF in the passband.
#    This is negligible in practice and consistent with the empirical finding
#    of near-perfect synchronization between the SSA nowcast and HF (Section 1.5).
#
# 4. SSA forecast phase-shift is the SMALLEST of the three filters in the passband,
#    confirming the relative lead of the SSA forecast over HF observed empirically
#    and measured by the Tau-statistic in Section 1.6.
#
# 5. The phase-shift of HF is substantial.
#    Root cause: the 2-year (8-quarter) forecast horizon embedded in the HF regression
#    equation introduces a structural lag into the filter.
#    Note: HP-trend applied to returns has a smaller lag (imposing a second-order unit root forces 
#     the phase-shift to vanish at frequency zero).
#
# Implication for the 'never use HP' debate:
#   - HP-GAP applied to levels: avoid — generates spurious cycles (confirmed in Tutorial 2).
#   - HP-TREND applied to returns: performs well — smaller lag and smoother output than HF
#     (see Tutorial 2 for a direct comparison).
#   The advice 'never use HP' therefore depends critically on WHICH HP design is used
#   and HOW it is applied.








########################################################################################################################
#################################################################################################################
# Example 2
# Application of SSA to HF based on monthly PAYEMS series (non-farm payroll) 

# The data starts pre WWII: series is non-stationary.
# Example 2 considers the whole data-sample, assuming log-returns to be white noise
# Example 3 fits an ARMA-model to the data
# Example 4 emphasizes post-1990 data only (great moderation) and it also analyzes the pandemic effect


# Let's briefly illustrate the Hamilton filter as applied to quarterly transformed PAYEMS: R-package neverhpfilter
library(neverhpfilter)


data(PAYEMS)
log_Employment <- 100*log(xts::to.quarterly(PAYEMS["1947/2016-6"], OHLC = FALSE))

employ_trend <- yth_filter(log_Employment, h = 8, p = 4, output = c("x", "trend"), family = gaussian)

plot.xts(employ_trend, grid.col = "white", legend.loc = "topleft", main = "Log of Employment and trend")

# Often the cycle and the random components correlate strongly which contradicts the classic assumptions
employ_cycle <- yth_filter(log_Employment, h = 8, p = 4, output = c("cycle", "random"), family = gaussian)
par(mfrow=c(1,1))
plot.xts(employ_cycle, grid.col = "white", legend.loc = "topright", main="Log of Employment cycle and random")
abline(h=0)

# Returns are non stationary: slowly drifting and series pre/post 1960 as well as pre/post 1990 differ  in term of 
#   dependence structure and variance (WWII, `great moderation') 
# In example 2, here, we ignore these issues: our main purpose is to illustrate SSA. 
# SSA is more or less insensitive to the misspecification: it will outperform the benchmark irrespective of `false` models
plot(diff(log(PAYEMS)))
abline(h=0)


#---------------------------------------------------------
# We now skip the above environment and R-package and implement the filter with own code, based on original monthly data
# This will allow to engraft SSA onto HF
getSymbols('PAYEMS',src='FRED')

# Discard Pandemic: extreme outliers affect regression of Hamilton filter
# Make double: xts objects are subject to lots of automatic/hidden assumptions which make an application of SSA 
# more cumbersome, unpredictable and hazardous
y<-as.double(log(PAYEMS["/2019"]))
len<-length(y)
#-----------------------------------------------------
# 2.1 Hamilton filter
# Settings proposed by Hamilton for quarterly data: two years ahead forecast and AR-order 4 to remove multiple unit-roots (if necessary)
h<-2*4
p<-4
# We here adapt to monthly PAYEMS: p remains the same (accounts for integration order, see Hamilton paper) 
h<-2*12
p<-4

explanatory<-cbind(y[(p):(len-h)],y[(p-1):(len-h-1)],y[(p-2):(len-h-2)],y[(p-3):(len-h-3)])
target<-y[(h+p):len]

lm_obj<-lm(y[(h+p):len]~explanatory)

# Only the first coefficient is significant, as is often the case in applications to non-stationary economic series.
# This is a potential problem because the sum of the parameters is not one and therefore 1-sum(coefficients)!=0.
# As a result the forecast residual y_{t+h}-\hat{y}_{t+h} is stationary in-sample (due to overfitting) but non-stationary out-of-sample. 
#   Stated differently: the forecast and the future observation are not cointegrated.
# The growth-rate (drift) of the series is changing over time i.e. first differences are non-stationary.
# Selecting p=4 removes this second-order non-stationarity in-sample; but not out-of-sample.
# Therefore the model has to be up-dated continuously over time as new data becomes available which leads to revisions.
summary(lm_obj)

# Plot cycle
ts.plot(lm_obj$residuals)

# Specify filter
hamilton_filter<-c(1,rep(0,h-1),-lm_obj$coefficients[1+1:p])
# Plot filter
ts.plot(hamilton_filter,main="HF filter coefficients")

# We could center the cycle by relying on the intercept
intercept<-lm_obj$coefficients[1]
# Apply HF to data
# We can fill any numbers for the leads from t+1,...,t+h-1 since the hamilton filter coefficients vanish there: we just repeat the target h-time
data_mat<-cbind(matrix(rep(target,h),ncol=h),explanatory)
residuals<-data_mat%*%hamilton_filter-intercept
# We just replicated the regression residuals (Hamilton filter)
ts.plot(cbind(residuals,lm_obj$residuals),main="Replication of regression residuals by HF (both series overlap)")

# The sum of the coefficients nearly vanishes: this is because the series is non-stationary and therefore the filter must 
#     remove the trend
sum(hamilton_filter)
# We now correct the filter such that the sum of the coefficients is zero:  cointegration constraint 
#   (we just distribute the difference evenly on coefficients)
hamilton_filter_adjusted<-hamilton_filter
hamilton_filter_adjusted[(h+1):(h+p)]<-hamilton_filter_adjusted[(h+1):(h+p)]-sum(hamilton_filter)/p
# Now the sum is zero
sum(hamilton_filter_adjusted)
# Compute corresponding adjusted residuals
residuals_adjusted<-data_mat%*%hamilton_filter_adjusted
# Center: not mandatory (we here rely on uncentered adjusted residuals or cycle estimates)
if (F)
  residuals_adjusted<-residuals_adjusted-mean(residuals_adjusted)

# See comments to cycles in example 1 above
par(mfrow=c(2,2))
ts.plot(y,main="Log(PAYEMS)")
ts.plot(cbind(residuals,residuals_adjusted),col=c("red","blue"),main="Cycles")
mtext("Classic cycle",col="red",line=-1)
mtext("Adjusted un-centered cycle",col="blue",line=-2)
ts.plot(residuals-residuals_adjusted,main="Cycle difference")

# For the purpose of engrafting SSA onto HF, we here consider the adjusted un-centered cycle 
#   But we also transform SSA back, to match the original (centered) cycle, see corresponding adjustments and plots below
#   For that purpose we need the difference between both cycles:
cycle_diffh<-residuals-residuals_adjusted


#---------------------------------------------------
# 2.2 Transformation: from levels to differences
# In order to plug SSA on HF we have to transform the empirical setting such that the data 
#   is stationary, see corresponding discussion in example 1 above
# -We can always find a new filter, called ham_diff, which, when applied to differences (instead of levels), replicates the output of the original bandpass, as applied to levels
# See section 2.3 and proposition 4 in JBCY paper for background.
L<-50
# Filter length: at least length of Hamilton filter
L<-max(length(hamilton_filter_adjusted),L)
if (L>length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L<-c(hamilton_filter_adjusted,rep(0,L-length(hamilton_filter_adjusted)))


# Convolution with summation filter (unit-root assumption)
ham_diff<-conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv
# Compare both filters
par(mfrow=c(2,1))
ts.plot(ham_diff,main="Hamilton filter as applied to first differences")
ts.plot(hamilton_filter_adjusted_L,main="Hamilton filter as applied to level")

# Apply new filter ham_diff to returns and verify that its filter output is the same as hamilton_filter_adjusted_L
x<-diff(y)
len_diff<-length(x)
residual_diff<-na.exclude(filter(x,ham_diff,side=1))

# Compare outputs
# Add NAs at the start of cycle_diff and Hamilton cycle (series are shorter due to applying Hamilton filter)
cycle_diff<-c(rep(NA,length(x)-length(cycle_diffh)),cycle_diffh)
original_hamilton_cycle<-c(rep(NA,length(x)-length(residuals)),residuals)
# Check: ham_diff, as applied to first differences, generates the same cycle as original Hamilton filter applied to levels
par(mfrow=c(1,1))
ts.plot(residual_diff,col="blue",main="Replication of Hamilton filter: application to level vs. first differences")
lines(residuals_adjusted[(L-p-h+2):length(residuals)],col="red")

# With this transformation in place we are in a position to engraft SSA onto HF

#-------------------------------------------------------------------------------
# 2.3 Holding-times
# We first compute the holding-time of the Hamilton-filter ham_diff
ht_ham_diff_obj<-compute_holding_time_func(ham_diff)
# Quite short: more than two zero-crossings per year on average
ht_ham_diff_obj$ht
# This number will be used for reference in example 4 below
ht_ham_example2<-ht_ham_diff_obj$ht
# But the empirical holding time is much longer: see discussion in example 1 above and example 7 in tutorial 1
compute_empirical_ht_func(residuals_adjusted)

# Explanation:  xt (log-returns) are not white noise
par(mfrow=c(1,1))
ts.plot(x,ylim=c(-0.05,0.05),main="Log-returns PAYEMS")
abline(h=0)

#-----------------------------------------------------------------------
# 2.4 Autocorrelation
# Section 2 in the JBCY paper proposes an extension of SSA to autocorrelated data.
# The main tool for checking autocorrelation is the ACF:
acf(x,main="ACF")
# The ACF suggests dependency among the returns of PAYEMS
# We here (in example 2) deliberately assume a wrong model, namely white noise, to illustrate that SSA is quite robust against 
#   misspecification
xi<-NULL
# Examples 3 and 4 further down show how to work with a fitted time series model 

# We are now in a position to engraft SSA onto HF
#   -We can work with stationary series thanks to our transformation to ham_diff
#   -We can work with the Wold decomposition specified by xi: in our case white noise
#   -But the data is non-stationary (slowly drifting and variance is changing too) and un-centered and the holding-time is biased, recall tutorial 1, exercise 7


#--------------------------------------
# 2.5 Apply SSA
# The above holding-time was approximately half a year 
ht_ham_diff_obj$ht 
# This number is biased because the data is not white noise
# But our main purpose is to derive a smoother (SSA-) design: smoother relative to HF 
# Therefore, we can augment ht of the Hamilton filter by 50% 
# As a result, the SSA filter output will be automatically smoother: irrespective of biases or model misspecifications
ht<-1.5*ht_ham_diff_obj$ht 
# This is the corresponding rho1 for the holding time constraint: our SSA-function accepts rho1 as input (not ht)
rho1<-compute_rho_from_ht(ht)
# Specify the target: SSA should track Hamilton-cycle (beeing smoother)
gammak_generic<-ham_diff
# Forecast horizon: nowcast (see forecasts below)
forecast_horizon<-0
# SSA of Hamilton-diff
# Note: if we do not supply xi then the SSA-function assumes white noise
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1)
# SSA computes two filters: ssa_x and ssa_eps: in our case we assume xt=epsilont (white noise) and therefore both 
#   filters are identical
SSA_filt_ham_diff<-SSA_obj_ham_diff$ssa_x
# Check that both filters are identical: difference should vanish
if (F)
{
  SSA_obj_ham_diff$ssa_x-SSA_obj_ham_diff$ssa_eps
}

# Compare target and SSA
par(mfrow=c(1,1))
mplot<-cbind(ham_diff,SSA_filt_ham_diff)
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"))
mtext("HF (as applied to differences)",col="black",line=-1)
mtext(paste("SSA: ht increased by ",100*(ht/ht_ham_diff_obj$ht-1),"%",sep=""), col="blue",line=-2)

# Check holding time constraint
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff)
ht_obj$ht 
ht
# Both numbers agree, confirming that the optimization converged to the global optimum

#--------------------------------------------------
# 2.6 Filter series and compute performance numbers
# 2.6.1. SSA
SSA_out<-filter(x,SSA_filt_ham_diff,side=1)
# Empirical ht much larger than targeted ht, see explanation above (series is non-stationary)
compute_empirical_ht_func(SSA_out)
ht  
# 2.6.2. Hamilton: SSA generates less crossings, as expected
ham_out<-filter(x,ham_diff,side=1)
compute_empirical_ht_func(ham_out)
ht_ham_diff_obj$ht 

# Although the series look similar, HF has more (noise-) leakage, see amplitude functions below, 
#   and therefore it generates  more `noisy' crossings (in the long run)
mplot<-na.exclude(cbind(SSA_out,ham_out))
colo=c("blue","red")
ts.plot(mplot[,1],col=colo[1])
lines(mplot[,2],col=colo[2])
abline(h=0)
mtext("Hamilton (as applied to differences)",col="red",line=-1)
mtext("SSA", col="blue",line=-2)

# 2.6.3 Empirical holding times: SSA approximately 50% larger than Hamilton in the long run (if the model is not misspecified)
#   Our model assumptions here are wrong: see example 4 below
#   One could increase ht in the function call in order to improve smoothness further, if desired
# We can clearly see the non-stationarity (changing shape) of the cycles.
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(ham_out)


# We can also compare the original Hamilton cycle with SSA shifted for the changing level (and scaled to original cycle)
# see example 1 above for details
lm_obj<-lm(original_hamilton_cycle~cycle_diff+SSA_out-1)
coef<-lm_obj$coef
# We now adjust SSA
scale_shifted_SSA<-coef[1]*cycle_diff+coef[2]*SSA_out
# We can plot adjusted SSA and original Hamilton cycle: data is missing at the start of scale_shifted_SSA shorter because it has been filtered
ts.plot(scale_shifted_SSA,col=colo[1],main="Shifted SSA cycle and original Hamilton cycle",ylim=c(min(original_hamilton_cycle,na.rm=T),max(original_hamilton_cycle,na.rm=T)))
mtext("Shifted and scaled SSA",col=colo[1],line=-1)
mtext("Original Hamilton cycle",col="black",line=-2)
lines(original_hamilton_cycle)
abline(h=0)

# Although both series look similar, SSA generates slightly less crossings (but not 30% less)
# We could either increase ht in SSA (making it smoother) or, better yet, split the series into 
#   shorter (more consistent or less stationary) time-frames: see example 4 below
compute_empirical_ht_func(scale_shifted_SSA)
compute_empirical_ht_func(original_hamilton_cycle)


#-----------------------------------------------
# 2.7 Forecasting
# Let us now address timeliness or lead/lags: 
# We augment the forecast horizon (delta in the JBCY paper) to obtain a faster (leading) SSA filter 
# We do not give-up in terms of smoothness or noise suppression since we do not change the ht-constraint

# We provide two different forecast horizons and compare with nowcast
# Compute half-a-year and full-year forecasts: we can supply a vector with the intended forecast horizons
#   SSA will return a matrix of filters: each column corresponds to the fixed forecast horizon 
forecast_horizon<-c(6,12)
# SSA of Hamilton-diff
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_ham_diff_x_forecast<-SSA_obj_ham_diff$ssa_x
# This is now a matrix with two columns corresponding to the forecast horizons
head(SSA_filt_ham_diff_x_forecast)

# Plot and compare SSA filters
# Morphing: the forecast filters assign less weight to the remote past (faster/leading) but 
#   the particular filter patterns will ensure smoothness (ht unchanged)
par(mfrow=c(1,1))
ts.plot(SSA_filt_ham_diff_x_forecast,col=c("orange","darkgreen"),main="SSA forecasts vs nowcast",ylim=c(min(SSA_filt_ham_diff),max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff,col="blue")
mtext(paste("Forecast horizon ",forecast_horizon[1],sep=""),col="orange",line=-1)
mtext(paste("Forecast horizon ",forecast_horizon[2],sep=""),col="darkgreen",line=-2)
mtext("Nowcast",col="blue",line=-3)

#----------------------
# Filter series: 
# 6 months ahead
SSA_out_forecast_6<-filter(x,SSA_filt_ham_diff_x_forecast[,1],side=1)
# 12 months ahead
SSA_out_forecast_12<-filter(x,SSA_filt_ham_diff_x_forecast[,2],side=1)

# The SSA-forecast is now shifted to the left; smoothness is the same.
# The scale or variance of the forecast is smaller because the estimation problem is more difficult (zero-shrinkage)
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out))
colo=c("blue","orange","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)

# We here scale to unit variance in order to ease visual inspection
mplot<-scale(na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out)),scale=T,center=F)
colo=c("blue","orange","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)




# ht of SSA designs is approximately 50% larger in the long run (if the model is not misspecified)
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(SSA_out_forecast_6)
compute_empirical_ht_func(SSA_out_forecast_12)
compute_empirical_ht_func(ham_out)



# The nowcast SSA is synchronized with HF; the forecasts are left-shifted (leading) and smoother
max_lead=8
# SSA nowcast is synchronized with HF
shift_obj<-compute_min_tau_func(mplot[,c(1,4)],max_lead)
# SSA half-year forecast is leading HF by ~3 months 
shift_obj<-compute_min_tau_func(mplot[,c(2,4)],max_lead)
# SSA full-year forecast is leading HF by ~6 months
shift_obj<-compute_min_tau_func(mplot[,c(3,4)],max_lead)

# Finally, we can analyze smoothness and timeliness issues by looking at amplitude and phase-shift functions
# These results offer an alternative description, based on formal filter characteristics
# They confirm the above empirical results (obtained by comparing filtered series)


#----------------------------------------
# 2.8 Compute amplitude and phase-shift functions
# Amplitude at higher frequencies: closer to zero means less noise leakage
# Phase-shift in passband: a smaller shift means a relative lead

# Select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude and phase-shifts for SSA, SSA-forecast and ham-diff: all filters are applied to returns

colo=c("black","blue","darkgreen","red")
amp_obj_SSA_now<-amp_shift_func(K,as.vector(SSA_filt_ham_diff),F)
amp_obj_SSA_for_6<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,1]),F)
amp_obj_SSA_for_12<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,2]),F)
amp_obj_ham<-amp_shift_func(K,ham_diff,F)

par(mfrow=c(2,1))
mplot<-cbind(amp_obj_SSA_now$amp,amp_obj_SSA_for_6$amp,amp_obj_SSA_for_12$amp,amp_obj_ham$amp)
# Scale SSA such that all amplitudes are normalized to one at frequency zero
mplot[,1]<-mplot[,1]/mplot[1,1]
mplot[,2]<-mplot[,2]/mplot[1,2]
mplot[,3]<-mplot[,3]/mplot[1,3]
mplot[,4]<-mplot[,4]/mplot[1,4]
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")

# 1. Plot amplitude
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude Hamilton-Filter",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# 2. Plot phase-shift
mplot<-cbind(amp_obj_SSA_now$shift,amp_obj_SSA_for_6$shift,amp_obj_SSA_for_12$shift,amp_obj_ham$shift)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()


# Discussion: see the discussion at the end of example 1 above 
# Note that the leads measured by the tau-statistic (in the time-domain and at zero-crossings) match 
#  the phase-shift differences in the passband closely
# Once again, the positive phase-shift or lag of HF is substantial
#   -It is larger than the classic HP-concurrent trend, applied to returns, whose shift must vanish at frequency zero
#   -The size of the lag is due to the forecast horizon (2 years) in the regression equation 
#   -It is not clear at this point "why you should never use the HP"
#     -Don't use HP-gap, applied to the original data: yes, see confirmation in tutorial 2.
#     -But HP-trend applied to returns performs well: it is quite smooth  and its lag is smaller than HF, see tutorial 2.



########################################################################################################################
#################################################################################################################
# Example 3: same as example 2 but we fit a model to the data
# Note: one has to run example 2 at least once before example 3 in order to initialize all settings 

# 3.1-3.3: We assume that steps 2.1 to 2.3 were done (otherwise run the corresponding code lines)
#--------------------------------------------  
# 3.4 Autocorrelation
# We here fit an ARMA-model to the data
# ACF suggests autocorrelation
acf(x,main="ACF")
ar_order<-1
ma_order<-1
estim_obj<-arima(x,order=c(ar_order,0,ma_order))
# We have the typical cancelling AR and MA-roots which can fit a weak but long lasting ACF 
estim_obj
# Diagnostics are OK for the purpose at hand
tsdiag(estim_obj)
# Compute the MA-inversion of the ARMA: Wold-decomposition
xi<-c(1,ARMAtoMA(ar=estim_obj$coef[1:ar_order],ma=estim_obj$coef[ar_order+1:ma_order],lag.max=L-1))
# Remark: L should be sufficiently large for xi to decay (converge) to zero: L=50 is fine
par(mfrow=c(1,1))
ts.plot(xi,main="Wold-decomposition: xi")
# Convolve xi and ham_diff: this is the filter that would be applied to the innovations epsilont in the Wold 
#   decomposition of xt
# If the above model were true, i.e. if epsilont were white noise, then the holding-time of this filter would match 
#   the observed empirical holding time of ham_diff
ham_conv<-conv_two_filt_func(xi,ham_diff)$conv
ts.plot(ham_conv,main="Convolved Hamilton filter")
# Compute ht
ht_ham_conv_obj<-compute_holding_time_func(ham_conv)
ht_ham_conv_obj$ht
# Compare with ht of ham_diff (based on white noise assumption)
#   ham_conv is much smoother because the ARMA-filter for the DGP is a lowpass  
ht_ham_diff_obj$ht
# Compare with empirical ht: residuals_adjusted is the output of ham_diff
compute_empirical_ht_func(residuals_adjusted)
# Part of the ht bias of ham_diff (assuming white noise) has been addressed and resolved by ham_conv by modelling the ACF-structure of the data
#   -ht of ham_conv is closer to the empirical ht. 
#   -But there is some bias left in the ht of ham_conv: the data is still un-centered and non-stationary 
# We use the holding time of ham_conv when specifying ht for SSA 


# We are now in a position to plug SSA on HF
#   -We can work with stationary series thanks to our transformation to ham_diff
#   -We can work with the Wold decomposition specified by xi 

#--------------------------------------
# 3.5 Apply SSA: 
# We here rely on ham_conv for the ht-constraint
ht_ham_conv_obj$ht
# We can lengthen by 50%: SSA will generate ~30% less crossings (if the model is not misspecified)
ht<-1.5*ht_ham_conv_obj$ht
# This is the corresponding rho1 for the holding time constraint
rho1<-compute_rho_from_ht(ht)
# Target: we want to `improve' upon ham_diff (which is the filter that is applied to xt)
# By supplying xi to SSA_func the algorithm `knows' that xt is not white noise
gammak_generic<-ham_diff
# Forecast horizon: nowcast
forecast_horizon<-0
# SSA of Hamilton-diff
# Note: this call to SSA is not correct!!!! We did not supply the Wold decomposition xi in the function call! 
#  Let's see what happens 
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1)

SSA_filt_ham_diff<-SSA_obj_ham_diff$ssa_eps

# Compare target and SSA: looks odd (SSA assumes the data to be white noise: in this case ht in the constraint 
#   is too large: the filter smooths excessively)
par(mfrow=c(1,1))
mplot<-cbind(ham_diff,SSA_filt_ham_diff)
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"))
mtext("Hamilton (as applied to differences)",col="black",line=-1)
mtext("Incorrect SSA (erroneously assuming white noise)", col="blue",line=-2)

# Check holding time constraint
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff)
# It matches our constraint. But the constraint is too heavy (it is based on ham_conv which assumes autocorrelation)
ht_obj$ht 
ht
 
# Let's do it the right way now: add xi (Wold decomposition) in the function call
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)
# ssa_eps is the filter which is applied to epsilont: this is mainly used for verifying the ht-constraint (convergence of optimization to global maximum)
SSA_filt_ham_diff_eps<-SSA_obj_ham_diff$ssa_eps
# ssa_x is the filter which is applied to the data (xt) 
#   In the previous call both filters were identical because we forgot to supply xi (now both filters differ)
SSA_filt_ham_diff_x<-SSA_obj_ham_diff$ssa_x


# Compare target and SSA: we must compare ham_diff with ssa_x (both are applied to xt)
#   In contrast to the previous call and plot, SSA seems now to be `OK' 
mplot<-cbind(ham_diff,SSA_filt_ham_diff_x)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"),main="Filters as applied to first differences xt")
mtext("Hamilton ",col="black",line=-1)
mtext("SSA: assuming autocorrelation ", col="blue",line=-2)

# Alternatively, we could compare ham_conv and ssa_eps (both are applied to epsilont)
mplot<-cbind(ham_conv,SSA_obj_ham_diff$ssa_eps)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"),main="Convolved filters as applied to innovations epsilont")
mtext("Hamilton",col="black",line=-1)
mtext("SSA", col="blue",line=-2)

# Check holding time constraint: 
#   Both numbers are identical (up to rounding error) confirming convergence of the optimization to the global maximum
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff_eps)
ht_obj$ht 
ht
# Note that the holding-time of ssa_x is much smaller because ssa_x is applied to xt which is smoother than epsilont
# This feature explains part of the ht-bias (mismatch of expected and empirical hts, see example 7 in tutorial 1)
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff_x)
ht_obj$ht 



#--------------------------------------------------
# 3.6 Filter series and compute performance measures
# 1. SSA
SSA_out<-filter(x,SSA_filt_ham_diff_x,side=1)
# Empirical ht quite larger than targeted ht, see explanation above (series is non-stationary)
compute_empirical_ht_func(SSA_out)
ht  
# 2. Hamilton
ham_out<-filter(x,ham_diff,side=1)
compute_empirical_ht_func(ham_out)
# Have to compare with ht of ham_conv (not ham_diff)
ht_ham_conv_obj$ht 

# Although the series look quite similar, HF has more (noise-) leakage and therefore it generates 
#   more `noisy' crossings 
mplot<-na.exclude(cbind(SSA_out,ham_out))
colo=c("blue","red")
ts.plot(mplot[,1],col=colo[1])
lines(mplot[,2],col=colo[2])
abline(h=0)
mtext("Hamilton (as applied to differences)",col="red",line=-1)
mtext("SSA", col="blue",line=-2)

# Empirical holding times: ht of SSA is larger but not 50% larger  
# Problem: non-stationarity, cycle is changing (see example 4 below which emphasizes data post-1990)
# Hint: try ht<-2*ht_ham_conv_obj$ht instead of ht<-1.5*ht_ham_conv_obj$ht before SSA-call
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(ham_out)


# We can also compare the original Hamilton cycle with SSA shifted for the changing level (and scaled to original cycle)
# see example 1 above for details
lm_obj<-lm(original_hamilton_cycle~cycle_diff+SSA_out-1)
coef<-lm_obj$coef
# We now adjust SSA
scale_shifted_SSA<-coef[1]*cycle_diff+coef[2]*SSA_out
# We can plot adjusted SSA and original Hamilton cycle: data is missing at the start of scale_shifted_SSA shorter because it has been filtered
ts.plot(scale_shifted_SSA,col=colo[1],main="Shifted SSA cycle and original Hamilton cycle",ylim=c(min(original_hamilton_cycle,na.rm=T),max(original_hamilton_cycle,na.rm=T)))
mtext("Shifted and scaled SSA",col=colo[1],line=-1)
mtext("Original Hamilton cycle",col="black",line=-2)
lines(original_hamilton_cycle)
abline(h=0)

# Although both series look similar, SSA generates slightly less crossings (but not 30% less)
# We could either increase ht in SSA (making it smoother) or, better yet, split the series into 
#   shorter time-frames in order to mitigate misspecification (for example discard data prior 1990): see example 4 below
compute_empirical_ht_func(scale_shifted_SSA)
compute_empirical_ht_func(original_hamilton_cycle)

 


#-----------------------------------------------
# 3.7 Forecasting
# Let us now address timeliness or lead/lags: 
# We augment the forecast horizon (delta in the JBCY paper) to obtain a faster (leading) SSA filter 
# We do not give-up in terms of smoothness or noise suppression since we do not change the ht-constraint

# We compute two forecast horizons and compare with nowcast
# Compute half a year and full year forecasts: we can supply a vector with the intended forecast horizons
#   SSA will return a matrix of filters: each column corresponds to the intended forecast horizon 
forecast_horizon<-c(6,12)
# SSA of Hamilton-diff
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_ham_diff_x_forecast<-SSA_obj_ham_diff$ssa_x

# Plot and compare SSA filter
# Morphing: the forecast filter assign less weight to remote past (faster/leading) but 
#   the particular filter pattern will ensure smoothness (ht unchanged)
# Note that these forecast filters differ from example 2 because we here assume an ARMA(1,1) model of the data
par(mfrow=c(1,1))
ts.plot(SSA_filt_ham_diff_x_forecast,col=c("orange","darkgreen"),main="SSA forecasts vs nowcast",ylim=c(min(SSA_filt_ham_diff_x),max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x,col="blue")
mtext(paste("Forecast horizon ",forecast_horizon[1],sep=""),col="orange",line=-1)
mtext(paste("Forecast horizon ",forecast_horizon[2],sep=""),col="darkgreen",line=-2)
mtext("Nowcast",col="blue",line=-3)

#----------------------
# Filter series: 
# 6 months ahead
SSA_out_forecast_6<-filter(x,SSA_filt_ham_diff_x_forecast[,1],side=1)
# 12 months ahead
SSA_out_forecast_12<-filter(x,SSA_filt_ham_diff_x_forecast[,2],side=1)

# The SSA-forecast is now shifted to the left; smoothness is the same.
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out))
colo=c("blue","orange","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)


# ht of SSA designs is approximately 50% larger: stronger noise suppression by SSA
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(SSA_out_forecast_6)
compute_empirical_ht_func(SSA_out_forecast_12)
compute_empirical_ht_func(ham_out)



# The forecast SSA filter has a lead at zero-crossings relative to HF (and less crossings)
max_lead=8
# SSA nowcast is synchronized with HF
shift_obj<-compute_min_tau_func(mplot[,c(1,4)],max_lead)
# SSA half-year forecast is leading HF by a quarter
shift_obj<-compute_min_tau_func(mplot[,c(2,4)],max_lead)
# SSA full-year forecast is leading HF by two quarters
shift_obj<-compute_min_tau_func(mplot[,c(3,4)],max_lead)

# Finally, we can analyze smoothness and timeliness issues by looking at amplitude and phase-shift functions
# These results offer an alternative description, based on formal filter characteristics
# They confirm the above empirical results (obtained by comparing filtered series)


#----------------------------------------
# 3.8 Compute amplitude and phase-shift functions
# Amplitude at higher frequencies: closer to zero means less noise leakage
# Phase-shift in passband: a smaller shift means a relative lead

# Select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude and phase-shifts for SSA, SSA-forecast and ham-diff: all filters are applied to returns

colo=c("black","blue","darkgreen","red")
amp_obj_SSA_now<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x),F)
amp_obj_SSA_for_6<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,1]),F)
amp_obj_SSA_for_12<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,2]),F)
amp_obj_ham<-amp_shift_func(K,ham_diff,F)

par(mfrow=c(2,1))
mplot<-cbind(amp_obj_SSA_now$amp,amp_obj_SSA_for_6$amp,amp_obj_SSA_for_12$amp,amp_obj_ham$amp)
# Scale SSA such that all amplitudes are normalized to one at frequency zero
mplot[,1]<-mplot[,1]/mplot[1,1]
mplot[,2]<-mplot[,2]/mplot[1,2]
mplot[,3]<-mplot[,3]/mplot[1,3]
mplot[,4]<-mplot[,4]/mplot[1,4]
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")

# 1. Plot amplitude
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude Hamilton-Filter",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# 2. Plot phase-shift
mplot<-cbind(amp_obj_SSA_now$shift,amp_obj_SSA_for_6$shift,amp_obj_SSA_for_12$shift,amp_obj_ham$shift)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()


# Discussion: see the discussion at the end of example 1 above  
# General comments
# -SSA seems to be quite robust against misspecification of the dependence structure
#   -Examples 2 (white noise) and 3 (ARMA-model) lead to filters with similar/comparable performances
# -The data covers a very long history and is subject to non-stationarity which can lead to systematic biases of ht
#   -Biases could be addressed, to some extent, by adjusting SSA to the original Hamilton cycle  
#   -In all cases, SSA outperformed the target in terms of smoothness; in some cases the gain was less than projected, though
#     (the problem could be addressed by increasing ht in the SSA-constraint or by shortening the data, see example 4)
#   -In all cases, the forecast filters outperformed the target in terms of smoothness and lead (left shift)
# Once again, the positive phase-shift or lag of HF is substantial
#   -It is larger than the classic HP-concurrent trend, applied to returns, whose shift must vanish at frequency zero
#   -The size of the lag is due to the forecast horizon (2 years) in the regression equation 
#   -It is not clear at this point "why you should never use the HP" (which HP?)
#     -Don't use HP-gap, applied to the original data: yes, see confirmation in tutorial 2.
#     -But HP-trend applied to returns performs well: it is pretty smooth and its lag is smaller than HF, see tutorial 2.






##############################################################################################################
#############################################################################################################
# Example 4: same as example 3 but we use data past 1990 only.
# Our selection affects: 
#   1. The definition of the Hamilton cycle because the regression parameters will change 
#   2. The ARMA-model for deriving the Wold-decomposition xi

# Use data from 1990 up to 2019 (skip pandemic)
# Pandemic is analyzed in example 4.9, at the end of the tutorial
y<-as.double(log(PAYEMS["1990::2019"]))
ts.plot(y)
len<-length(y)
#--------------------------
# 4.1 Hamilton filter
# Settings proposed by Hamilton for quarterly data: two years ahead forecast and AR-order 4 to remove multiple unit-roots (if necessary)
h<-2*4
p<-4
# We here adapt to monthly PAYEMS: p remains the same (accounts for integration order, see Hamilton paper) 
h<-2*12
p<-4

explanatory<-cbind(y[(p):(len-h)],y[(p-1):(len-h-1)],y[(p-2):(len-h-2)],y[(p-3):(len-h-3)])
target<-y[(h+p):len]

lm_obj<-lm(y[(h+p):len]~explanatory)

# Substantially different regression parameters (as compared to example 3 above).
# In contrast to HP, which is fixed, Hamilton filter depends on data-fitting (data mining).
# This problem is sometimes overlooked in the literature (revision errors)
summary(lm_obj)
# Cycle appears  noisier (than in examples 2,3)
ts.plot(lm_obj$residuals)

# Specify filter
hamilton_filter<-c(1,rep(0,h-1),-lm_obj$coefficients[1+1:p])
intercept<-lm_obj$coefficients[1]
# We can fill any numbers for the leads from t+1,...,t+h-1 since the hamilton filter coefficients vanish there: we just repeat the target h-time
data_mat<-cbind(matrix(rep(target,h),ncol=h),explanatory)

residuals<-data_mat%*%hamilton_filter-intercept
# We just replicated the regression residuals (Hamilton filter)
par(mfrow=c(1,1))
ts.plot(cbind(residuals,lm_obj$residuals),main="Replication of regression residuals by hamilton_filter")

# The sum of filter coefficients nearly vanishes: this is because the series is non-stationary and therefore the filter must 
#   remove the trend 
# But there is no cointegration imposed: the out-of-sample cycle will be non-stationary
# Therefore HF must be regularly up-dated, as new data is available, which leads to revisions
sum(hamilton_filter)
# We now correct the filter such that the sum of the coefficients is zero: cointegration constraint 
#   (we just distribute the difference evenly on coefficients)
hamilton_filter_adjusted<-hamilton_filter
hamilton_filter_adjusted[(h+1):(h+p)]<-hamilton_filter_adjusted[(h+1):(h+p)]-sum(hamilton_filter)/p
# Now sum is zero
sum(hamilton_filter_adjusted)
# Compute corresponding adjusted residuals
residuals_adjusted<-data_mat%*%hamilton_filter_adjusted
# Center: not mandatory (we here rely on uncentered adjusted residuals or cycle estimates)
if (F)
  residuals_adjusted<-residuals_adjusted-mean(residuals_adjusted)

# Both cycles (regression residuals) differ in terms of `levels' , see discussions in previous examples
par(mfrow=c(2,2))
ts.plot(y,main="Log(PAYEMS)")
ts.plot(cbind(residuals,residuals_adjusted),col=c("red","blue"),main="Cycles")
mtext("Classic cycle",col="red",line=-1)
mtext("Adjusted un-centered cycle",col="blue",line=-2)
ts.plot(residuals-residuals_adjusted,main="Cycle difference")
# See discussion about cycle differences in example 2 above
# For the purpose of engrafting SSA onto HF, we here consider the adjusted and un-centered cycle 
#   But we also transform SSA back, to match the original cycle, see corresponding adjustments and plots below
#   For that purpose we need the difference between both cycles:
cycle_diffh<-residuals-residuals_adjusted


#---------------------------------------------------
# 4.2 Transformation: from levels to first differences, see previous examples
# We here select a larger filter-length L than in the previous examples because the weights of the Wold-decomposition 
#   decay more slowly: longer memory of the data (smoother pattern) after 1990 (great moderation)
# See section 2.3 and proposition 4 in JBCY paper for background.
L<-100
# Filter length: at least length of HF
L<-max(length(hamilton_filter_adjusted),L)
if (L>length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L<-c(hamilton_filter_adjusted,rep(0,L-length(hamilton_filter_adjusted)))

# Convolution with summation filter (unit-root assumption)
ham_diff<-conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv
# Plot and compare both filters
par(mfrow=c(2,1))
ts.plot(ham_diff,main="HF as applied to first differences")
ts.plot(hamilton_filter_adjusted_L,main="HF as applied to level")

# Apply new filter ham_diff to returns and verify that it's filter output is the same as hamilton_filter_adjusted_L
x<-diff(y)
len_diff<-length(x)
residual_diff<-na.exclude(filter(x,ham_diff,side=1))

# Compare outputs
# Add NAs at the start of cycle_diff and Hamilton cycle (series are shorter due to applying HF)
cycle_diff<-c(rep(NA,length(x)-length(cycle_diffh)),cycle_diffh)
original_hamilton_cycle<-c(rep(NA,length(x)-length(residuals)),residuals)
# Check: ham_diff, as applied to first differences, generates the same cycle as original HF applied to levels
par(mfrow=c(1,1))
ts.plot(residual_diff,col="blue",main="Replication of HF: application to level vs. first differences")
lines(residuals_adjusted[(L-p-h+2):length(residuals)],col="red")

# With this transformation in place, we can now engraft SSA onto HF
#-------------------------------------------------------------------------------
# 4.3 Holding-times
# We first compute the holding-time of  ham_diff (as applied to differences)
ht_ham_diff_obj<-compute_holding_time_func(ham_diff)
# Much shorter holding time than in examples 2 (and 3): the regression equation now fits post-1990 data
ht_ham_diff_obj$ht 
ht_ham_example2

# But the empirical holding time is much longer: see discussion in examples 1-3 above
compute_empirical_ht_func(residuals_adjusted)

# Explanation:  xt (log-returns) are not white noise 
par(mfrow=c(1,1))
ts.plot(x,main="Log-returns PAYEMS")
abline(h=0)
#----------------------------------------------------------------------------
# 4.4 Autocorrelation
# One can split the data-sample into two-halves to assess out-of-sample performances
#   -SSA relies on ARMA-model to fit the dependence structure
# SSA is pretty insensitive as long as the ARMA-model is not severely overfitted
# Try either one (the comments refer to full-sample though both outcomes are nearly identical)
try_out_of_sample<-F
if (try_out_of_sample)
{
# Halve the data sample  
  in_sample_length<-length(x)/2
} else
{  
# Full sample  
  in_sample_length<-length(x)
}
# ACF suggests strong autocorrelation (`great moderation' effect?)
acf(x[1:in_sample_length],main="ACF: slowly decaying (longer memory)")
ar_order<-1
ma_order<-1
estim_obj<-arima(x[1:in_sample_length],order=c(ar_order,0,ma_order))
# We have the typical `cancelling' AR and MA-roots which can fit a weak but long lasting ACF 
estim_obj
tsdiag(estim_obj)
  
# Compute the MA-inversion of the ARMA: Wold-decomposition
xi<-c(1,ARMAtoMA(ar=estim_obj$coef[1:ar_order],ma=estim_obj$coef[ar_order+1:ma_order],lag.max=L-1))
# Remark: L should be sufficiently large for xi to decay (converge) to zero: this motivated the choice of L<-100
par(mfrow=c(1,1))
ts.plot(xi,main="Wold decomposition xi: slowly decaying (longer memory)")
# Convolve xi and ham_diff: filter applied to innovations in Wold decomposition (this filter is used for determining the holding-time only)
ham_conv<-conv_two_filt_func(xi,ham_diff)$conv
ht_ham_conv_obj<-compute_holding_time_func(ham_conv)


#--------------------------------------
# 4.5 Apply SSA
# The expected holding-time of ham_conv is slightly less than a year (ignoring the bias)
ht_ham_conv_obj$ht 
# We can lengthen by 50%: SSA will generate ~30% less crossings
ht<-1.5*ht_ham_conv_obj$ht
# This is the corresponding rho1 for the holding time constraint
rho1<-compute_rho_from_ht(ht)
# Target: we want to `improve' upon ham_diff: the filter applied to xt 
# By supplying xi to the SSA function-call, the algorithm `knows' that the data is not white noise`
gammak_generic<-ham_diff
# Forecast horizon: nowcast
forecast_horizon<-0

SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# ssa_eps is the filter which is applied to epsilont: it is mainly used to verify convergence of the optimization to the global maximum
SSA_filt_ham_diff_eps<-SSA_obj_ham_diff$ssa_eps
# ssa_x is the main filter of interest: it is applied to the data (xt)
SSA_filt_ham_diff_x<-SSA_obj_ham_diff$ssa_x

# Compare target and SSA: we must compare ham_diff with ssa_x (both are applied to xt)
mplot<-cbind(ham_diff,SSA_filt_ham_diff_x)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"),main="Filters as applied to first differences xt")
mtext("Hamilton ",col="black",line=-1)
mtext("SSA ", col="blue",line=-2)

# Alternatively, we could compare ham_conv and ssa_eps (both are applied to epsilont)
mplot<-cbind(ham_conv,SSA_obj_ham_diff$ssa_eps)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"),main="Convolved filters as applied to innovations epsilont")
mtext("Hamilton",col="black",line=-1)
mtext("SSA", col="blue",line=-2)


# Check optimization towards global maximum
# The holding-time of the solution should match the imposed number
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff_eps)
# Matching numbers confirm that the optimization has converged 
ht_obj$ht 
ht


#--------------------------------------------------
# 4.6 Filter series and compute performance measures
# 4.6.1. SSA
SSA_out<-filter(x,SSA_filt_ham_diff_x,side=1)
# Empirical ht quite larger than targeted ht, see explanation above (series is non-stationary)
compute_empirical_ht_func(SSA_out)
ht  
# 4.6.2. Hamilton
ham_out<-filter(x,ham_diff,side=1)
compute_empirical_ht_func(ham_out)
ht_ham_conv_obj$ht 

# Although the series look quite similar, HF has more (noise-) leakage and therefore it generates 
#   more `noisy' crossings 
mplot<-na.exclude(cbind(SSA_out,ham_out))
colo=c("blue","red")
ts.plot(mplot[,1],col=colo[1])
lines(mplot[,2],col=colo[2])
abline(h=0)
mtext("Hamilton (as applied to differences)",col="red",line=-1)
mtext("SSA", col="blue",line=-2)

# Empirical holding times: SSA approximately 50% larger than HF (difference commensurate with sampling error)
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(ham_out)


# We can also compare the original Hamilton cycle with SSA shifted for the changing level (and scaled to original cycle)
# see example 1 above for details
lm_obj<-lm(original_hamilton_cycle~cycle_diff+SSA_out-1)
coef<-lm_obj$coef
# We now adjust SSA
scale_shifted_SSA<-coef[1]*cycle_diff+coef[2]*SSA_out
# We can plot adjusted SSA and original Hamilton cycle: data is missing at the start of scale_shifted_SSA shorter because it has been filtered
ts.plot(scale_shifted_SSA,col=colo[1],main="Shifted SSA cycle and original Hamilton cycle",ylim=c(min(original_hamilton_cycle,na.rm=T),max(original_hamilton_cycle,na.rm=T)))
mtext("Shifted and scaled SSA",col=colo[1],line=-1)
mtext("Original Hamilton cycle",col="black",line=-2)
lines(original_hamilton_cycle)
abline(h=0)

# Although both series look similar, SSA generates ~30% less crossings (even less than that) 
# Since model misspecification is smaller (than in previous examples), empirical and expected holding times are in better agreement 
compute_empirical_ht_func(scale_shifted_SSA)
compute_empirical_ht_func(original_hamilton_cycle)
# Note that scale_shifted_SSA is shorter than the original_hamilton_cycle (because of filtering)
# We here align time frames accordingly
# The correction does not affect our findings, though
compute_empirical_ht_func(original_hamilton_cycle[L:length(original_hamilton_cycle)])


#-----------------------------------------------
# 4.7 Forecasting
# Let us now address timeliness or lead/lags: 
# We augment the forecast horizon (delta in the JBCY paper) to obtain a faster (leading) SSA filter 
# We do not give-up in terms of smoothness or noise suppression since we do not change the ht-constraint

# We compute two forecast horizons and compare with nowcast
# Compute half a year and full year forecasts: we can supply a vector with the intended forecast horizons
#   SSA will return a matrix of filters: each column corresponds to the intended forecast horizon 
forecast_horizon<-c(6,12)
# SSA of Hamilton-diff
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_ham_diff_x_forecast<-SSA_obj_ham_diff$ssa_x

# Plot and compare SSA filter
# Morphing: the forecast filters assign less weight to remote past (faster/leading) but 
#   the particular filter patterns will ensure smoothness (ht unchanged)
par(mfrow=c(1,1))
ts.plot(SSA_filt_ham_diff_x_forecast,col=c("orange","darkgreen"),main="SSA forecasts vs nowcast",ylim=c(min(SSA_filt_ham_diff_x),max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x,col="blue")
mtext(paste("Forecast horizon ",forecast_horizon[1],sep=""),col="orange",line=-1)
mtext(paste("Forecast horizon ",forecast_horizon[2],sep=""),col="darkgreen",line=-2)
mtext("Nowcast",col="blue",line=-3)

# Check ht: matching numbers confirm convergence of the optimization to the global maximum
# Note that we supply ssa_eps: the filter which would be applied to epsilont (not xt)
apply(SSA_obj_ham_diff$ssa_eps,2,compute_holding_time_func)
ht

#----------------------
# Filter series: 
# 6 months ahead
SSA_out_forecast_6<-filter(x,SSA_filt_ham_diff_x_forecast[,1],side=1)
# 12 months ahead
SSA_out_forecast_12<-filter(x,SSA_filt_ham_diff_x_forecast[,2],side=1)

# The SSA-forecast is now shifted to the left; smoothness is the same.
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out))
colo=c("blue","orange","darkgreen","red")
par(mfrow=c(1,1))
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)

# ht of SSA designs is approximately 50% larger: stronger noise suppression by SSA
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(SSA_out_forecast_6)
compute_empirical_ht_func(SSA_out_forecast_12)
compute_empirical_ht_func(ham_out)



# The forecast SSA filter has a lead at zero-crossings relative to HF (and less crossings)
max_lead=10
# SSA nowcast is synchronized with HF
shift_obj<-compute_min_tau_func(mplot[,c(1,4)],max_lead)
# SSA half-year forecast is leading Hamilton
shift_obj<-compute_min_tau_func(mplot[,c(2,4)],max_lead)
# SSA full-year forecast is leading Hamilton even more
shift_obj<-compute_min_tau_func(mplot[,c(3,4)],max_lead)

# Finally, we can analyze smoothness and timeliness issues by looking at amplitude and phase-shift functions
# These results offer an alternative description, based on formal filter characteristics
# They confirm the above empirical results (obtained by comparing filtered series)


#----------------------------------------
# 4.8 Compute amplitude and phase-shift functions
# Amplitude at higher frequencies: closer to zero means less noise leakage
# Phase-shift in passband: a smaller shift means a relative lead

# Select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude and phase-shifts for SSA, SSA-forecast and ham-diff: all filters are applied to returns

colo=c("black","blue","darkgreen","red")
amp_obj_SSA_now<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x),F)
amp_obj_SSA_for_6<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,1]),F)
amp_obj_SSA_for_12<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,2]),F)
amp_obj_ham<-amp_shift_func(K,ham_diff,F)

par(mfrow=c(2,1))
mplot<-cbind(amp_obj_SSA_now$amp,amp_obj_SSA_for_6$amp,amp_obj_SSA_for_12$amp,amp_obj_ham$amp)
# Scale SSA such that all amplitudes are normalized to one at frequency zero
mplot[,1]<-mplot[,1]/mplot[1,1]
mplot[,2]<-mplot[,2]/mplot[1,2]
mplot[,3]<-mplot[,3]/mplot[1,3]
mplot[,4]<-mplot[,4]/mplot[1,4]
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")

# 1. Plot amplitude: smaller amplitude at higher frequencies means stronger noise suppression (longer holding time)
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude Hamilton-Filter",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# 2. Plot phase-shift: smaller shift at lower frequencies means a lead
mplot<-cbind(amp_obj_SSA_now$shift,amp_obj_SSA_for_6$shift,amp_obj_SSA_for_12$shift,amp_obj_ham$shift)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()


# Discussion: see the discussion at the end of example 1 above  
# Findings
# -by discarding the remote past (from WWII up to 1990), the shortened time series allows for a less inconsistent modelling of the 
#     data (non-stationarity is less pronounced from 1990-2023).
# -As a result, expected and empirical holding times match better than in the previous examples 2 and 3 (entire data set).
# -Also, empirical holding times of SSA match the intended 50% increase (over target) better than in examples 2 and 3.
#   -Sometimes a bit smaller: unadjusted cycles.
#   -sometimes quite a bit larger: adjusted cycles. 
#   -deviations are compatible with random sampling errors.
# The positive phase-shift or lag of HF is smaller than in the previous examples.
#   -HF depends on the selection of the data-window (for estimating parameters of the regression equation)
#   -The lag is still larger than the classic HP-concurrent trend, applied to returns, whose shift must vanish at frequency zero.
#   -The size of the lag is due to the forecast horizon (2 years) in the regression equation 
#   -It is not clear at this point "why you should never use the HP"
#     -Don't use HP-gap, applied to the original data: yes, see confirmation in tutorial 2.
#     -But HP-trend applied to returns performs well: it is pretty smooth and its lag is smaller, see tutorial 2.
#     -Moreover, HP-trend does not depend on the data-window or sample-size (there are pros and cons to this argument)


#----------------------------------------------------------------------------------------------
# 4.9 To conclude we apply the above filters unchanged (no re-estimation) out-of-sample, including the pandemic.
# With a potentially instructive outcome...

# Compute long sample
y<-as.double(log(PAYEMS["1990/"]))
x<-diff(y)
# The very strong pandemic outliers will act as `impulses`, triggering the impulse response of the filter, 
#   i.e., the proper (sign inverted) filter coefficients
par(mfrow=c(1,1))
ts.plot(x)

# Apply all filters unchanged (out of sample)
# 1. SSA nowcast
SSA_out<-filter(x,SSA_filt_ham_diff_x,side=1)
# 2. Hamilton
ham_out<-filter(x,ham_diff,side=1)
# 3. Both SSA forecasts
SSA_out_forecast_6<-filter(x,SSA_filt_ham_diff_x_forecast[,1],side=1)
# 12 months ahead
SSA_out_forecast_12<-filter(x,SSA_filt_ham_diff_x_forecast[,2],side=1)

# Interesting:
# The pandemic dip is mirrored by a later peak whose timing depends on the SSA-design
# Explanation: negative Impulse responses of the filters
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out))
colo=c("blue","orange","darkgreen","red")
par(mfrow=c(1,1))
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot,na.rm=T),max(mplot,na.rm=T)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)

# In order to understand the observed pattern, it is instructive to have a look at filter coefficients, once again
# The (negative) Pandemic impulse replicates the (sign inverted) pattern of the corresponding filters 
par(mfrow=c(1,1))
ts.plot(SSA_filt_ham_diff_x_forecast,col=c("orange","darkgreen"),main="SSA forecasts vs nowcast",ylim=c(min(SSA_filt_ham_diff_x),max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x,col="blue")
mtext(paste("Forecast horizon ",forecast_horizon[1],sep=""),col="orange",line=-1)
mtext(paste("Forecast horizon ",forecast_horizon[2],sep=""),col="darkgreen",line=-2)
mtext("Nowcast",col="blue",line=-3)

# We can now easily understand the secondary 'fake' peak of the filtered series as well as its different location on the time-axis
# Advantage of forecast filters
#   -a longer forecast horizon leads to a more rapid zero-decay of filter coefficients
#   -therefore, this type of filter `forgets' more rapidly extreme or singular observations (than nowcast or target)
#   -As an example, the one-year ahead forecast could be applied 1.5 years after the outliers occurred
#   -In contrast, the nowcast or HF would need another 10-11 months to `forget' the singularity
# It is not clear, at this point, "why you should never use the HP" (which HP?)
#   -The impulse response of the classic HP-trend decays faster than HF 
#   -Therefore the effects of gross outliers (pandemic) would be less pronounced (smeared)


###################################################################################################
###################################################################################################
# Summary
# -We proposed a variant of HF onto which SSA could be grafted
# -We also proposed a (simple) adjustment for SSA to match the original (Hamilton) cycle
# -SSA is quite robust against model-misspecification
#   -The very long sample in example 2 and 3, ranging from WWII up to 2023, is subject to changes in the time series dynamics (non-stationarity)
#     -White noise and ARMA specifications generated fairly similar results on the long time span 
#     -Empirical holding times are generally strongly biased. But correcting for misspecifications generally reduced this bias (see results for adjusted SSA vs. original Hamilton cycle)
#     -Imposing a 50% larger ht did not improve relative smoothness (of SSA against target) accordingly. However, 
#       one could simply increase ht in the SSA-constraint further (for example requiring 100% increase)
#     -These differences and deviations from target were mainly due to non-stationarity (model misspecification)
#   -The shorter sample, starting in 1990 (great moderation), led to more consistent results
# -SSA is also quite robust against overfitting, at least as long as the ARMA-model is not severely overfitted
#   -For typical applications, a simple ARMA(1,1) is often sufficiently flexible to extract the relevant features from the data   
# -SSA nowcasts generally improve upon the target in terms of smoothness (less noisy crossings). 
#   -Empirical improvements are commensurate with theoretical specifications, at least in the long run and assuming that the model is not severely misspecified  
# -SSA forecasts can retain the same smoothness while outperforming the target in terms of timeliness, too (lead/left-shift)
#   -One cannot improve speed and smoothness without loosing in terms of MSE-performances
#   -The tradeoff is a trilemma, see tutorial 0.1 on trilemma
#   -SSA allows to position the filter-design in the limits delineated by the trilemma: best MSE for given ht and shift
# -A more refined and effective handling of timeliness (left shift) is proposed in due time
