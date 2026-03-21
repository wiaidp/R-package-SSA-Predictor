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





# ==============================================================================================
# Example 2: Monthly Data
# ==============================================================================================
# SSA applied to Hamilton filtering using monthly PAYEMS (non-farm payrolls)

# The sample starts before WWII; the series is clearly non-stationary.
# In this example we use the full sample and assume log-returns are white noise.
# Example 3 will instead fit an ARMA model.
# Example 4 focuses on post-1990 data (Great Moderation) and includes the pandemic period.


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
# Here we ignore these features to focus on SSA.
# SSA is relatively robust to model misspecification and often outperforms
# standard benchmarks even under incorrect assumptions.
plot(diff(log(PAYEMS)))
abline(h = 0)


# ---------------------------------------------------------
# Re-implement the Hamilton filter on monthly data (no package)
# This allows us to combine SSA with the Hamilton filter
getSymbols('PAYEMS', src = 'FRED')

# Exclude pandemic period: extreme outliers distort regression estimates
# Convert to numeric vector (xts objects can introduce unwanted behavior for SSA)
y <- as.double(log(PAYEMS["/2019"]))
len <- length(y)

# -----------------------------------------------------
# 2.1 Hamilton filter
# For quarterly data: h = 8 (2 years ahead), p = 4
# For monthly data: keep p = 4 (integration order), set h = 24 (2 years)
h <- 2 * 12
p <- 4

explanatory <- cbind(y[(p):(len-h)],
                     y[(p-1):(len-h-1)],
                     y[(p-2):(len-h-2)],
                     y[(p-3):(len-h-3)])
target <- y[(h+p):len]

lm_obj <- lm(target ~ explanatory)

# Typically only the first lag is significant in non-stationary macro series.
# This implies sum(coefficients) ≠ 1, so the forecast error is stationary in-sample
# but not out-of-sample (lack of cointegration).
# The drift changes over time, i.e., first differences are also non-stationary.
# Using p = 4 helps address higher-order integration in-sample,
# but requires continuous re-estimation as new data arrive.
summary(lm_obj)

# Plot cycle (regression residuals)
ts.plot(lm_obj$residuals)

# Construct Hamilton filter coefficients
hamilton_filter <- c(1, rep(0, h-1), -lm_obj$coefficients[1+1:p])
ts.plot(hamilton_filter, main = "Hamilton filter coefficients")

# Intercept can be used to center the cycle
intercept <- lm_obj$coefficients[1]
  
  # Apply filter: leads (t+1,...,t+h-1) can be filled arbitrarily since weights are zero
  data_mat <- cbind(matrix(rep(target, h), ncol = h), explanatory)
residuals <- data_mat %*% hamilton_filter - intercept

# Verify equivalence with regression residuals
ts.plot(cbind(residuals, lm_obj$residuals),
        main = "Hamilton filter replication")

# For non-stationary series, filter coefficients should sum to zero (trend removal)
sum(hamilton_filter)

# Enforce zero-sum constraint (cointegration condition)
# Distribute adjustment evenly across lag coefficients
hamilton_filter_adjusted <- hamilton_filter
hamilton_filter_adjusted[(h+1):(h+p)] <-
  hamilton_filter_adjusted[(h+1):(h+p)] - sum(hamilton_filter)/p

sum(hamilton_filter_adjusted)

# Compute adjusted cycle
residuals_adjusted <- data_mat %*% hamilton_filter_adjusted

# Optional centering
if (F)
  residuals_adjusted <- residuals_adjusted - mean(residuals_adjusted)

# Compare original and adjusted cycles
par(mfrow = c(2,2))
ts.plot(y, main = "Log(PAYEMS)")
ts.plot(cbind(residuals, residuals_adjusted), col = c("red","blue"),
        main = "Cycles")
mtext("Original cycle", col = "red", line = -1)
mtext("Adjusted (uncentered)", col = "blue", line = -2)
ts.plot(residuals - residuals_adjusted, main = "Cycle difference")

##########################################################################
# TAKEAWAY
# The original and adjusted cycles are very similar; the main difference is a slowly varying level component.
# SSA can be applied to either specification with comparable results.
# We use the adjusted cycle for technical convenience.
###########################################################################

# We proceed with the adjusted (uncentered) cycle for SSA.
# The difference between both versions is stored for later back-transformation.
cycle_diffh <- residuals - residuals_adjusted


# ---------------------------------------------------
# 2.2 Transformation: levels → differences
# To apply SSA, we require stationarity.
# We construct a filter (ham_diff) that, when applied to first differences,
# reproduces the Hamilton filter applied to levels.
# See JBCY paper, Section 2.3, Proposition 4.

L <- 50
L <- max(length(hamilton_filter_adjusted), L)

if (L > length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L <-
  c(hamilton_filter_adjusted,
    rep(0, L - length(hamilton_filter_adjusted)))

# Convolution with summation filter (unit-root assumption)
ham_diff <- conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv

# Compare filters
par(mfrow = c(2,1))
ts.plot(ham_diff, main = "Filter on first differences")
ts.plot(hamilton_filter_adjusted_L, main = "Filter on levels")

# Apply ham_diff to log-returns
x <- diff(y)
len_diff <- length(x)
residual_diff <- na.exclude(filter(x, ham_diff, side = 1))

# Align series for comparison
cycle_diff <- c(rep(NA, length(x) - length(cycle_diffh)), cycle_diffh)
original_hamilton_cycle <- c(rep(NA, length(x) - length(residuals)), residuals)

# Verify equivalence: differences vs levels
par(mfrow = c(1,1))
ts.plot(residual_diff, col = "blue",
        main = "Hamilton filter: levels vs differences")
lines(residuals_adjusted[(L-p-h+2):length(residuals)], col = "red")

# With this transformation, SSA can be applied consistently


# -------------------------------------------------------------------------------
# 2.3 Holding times
# Compute holding time of ham_diff
ht_ham_diff_obj <- compute_holding_time_func(ham_diff)

# Theoretical holding time: relatively short (frequent zero-crossings)
ht_ham_diff_obj$ht

# Store for later comparison (Example 4)
ht_ham_example2 <- ht_ham_diff_obj$ht

# Empirical holding time is much longer (see Example 1 and Tutorial 1)
compute_empirical_ht_func(residuals_adjusted)

# Reason: log-returns are not white noise
par(mfrow = c(1,1))
ts.plot(x, ylim = c(-0.05, 0.05), main = "Log-returns PAYEMS")
abline(h = 0)


# -----------------------------------------------------------------------
# 2.4 Autocorrelation
# SSA can be extended to autocorrelated data (JBCY, Section 2).
# Inspect dependence via the ACF:
acf(x, main = "ACF of log-returns")

# The ACF indicates serial dependence in returns.
# Here we deliberately assume white noise to illustrate SSA robustness.
# Examples 3 and 4 will incorporate explicit time-series models.
xi <- NULL

# Summary:
# - Stationarity achieved via transformation (ham_diff)
# - Wold representation specified via xi (here: white noise)
# - Data still exhibit drift, time-varying variance, and biased holding times

#--------------------------------------
# 2.5 Apply SSA
# The theoretical holding time of the Hamilton filter (in differences)
ht_ham_diff_obj$ht 

# This value is biased since the data are not white noise.
# Our goal is to construct a smoother filter than Hamilton (via SSA).
# We therefore increase the holding time by 50%.
# This enforces additional smoothness, regardless of model misspecification.
ht <- 1.5 * ht_ham_diff_obj$ht 

# Convert holding time into rho1 (required input for SSA)
rho1 <- compute_rho_from_ht(ht)

# Target filter: SSA should approximate the Hamilton cycle, but more smoothly
gammak_generic <- ham_diff

# Forecast horizon: 0 corresponds to a nowcast
forecast_horizon <- 0

# Apply SSA to the Hamilton-difference filter
# If xi is not provided, SSA assumes white noise
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1)

# SSA returns two filters (ssa_x and ssa_eps).
# Under the white-noise assumption (xt = eps_t), they coincide.
SSA_filt_ham_diff <- SSA_obj_ham_diff$ssa_x

# Optional check: the two filters should be identical
if (F)
{
  SSA_obj_ham_diff$ssa_x - SSA_obj_ham_diff$ssa_eps
}

# Compare Hamilton filter and SSA approximation
par(mfrow = c(1,1))
mplot <- cbind(ham_diff, SSA_filt_ham_diff)
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black","blue"))
mtext("Hamilton (differences)", col = "black", line = -1)
mtext(paste("SSA (ht +", 100*(ht/ht_ham_diff_obj$ht - 1), "%)", sep=""),
      col = "blue", line = -2)

# Verify that the holding time constraint is satisfied
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff)
ht_obj$ht 
ht
# Agreement confirms successful optimization


#--------------------------------------------------
# 2.6 Filter series and evaluate performance
# 2.6.1 SSA output
SSA_out <- filter(x, SSA_filt_ham_diff, side = 1)

# Empirical holding time is larger than theoretical (due to non-stationarity)
compute_empirical_ht_func(SSA_out)
ht  

# 2.6.2 Hamilton output (more zero-crossings expected)
ham_out <- filter(x, ham_diff, side = 1)
compute_empirical_ht_func(ham_out)
ht_ham_diff_obj$ht 

# Although both series are similar, the Hamilton filter exhibits more noise leakage,
# leading to more frequent crossings over time
mplot <- na.exclude(cbind(SSA_out, ham_out))
colo <- c("blue","red")
ts.plot(mplot[,1], col = colo[1])
lines(mplot[,2], col = colo[2])
abline(h = 0)
mtext("Hamilton", col = "red", line = -1)
mtext("SSA", col = "blue", line = -2)

# Empirical holding times:
# SSA is smoother (fewer crossings), roughly consistent with the imposed increase in ht
# Note: results are affected by model misspecification (see Example 4)
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(ham_out)

# Adjust SSA back to the scale of the original Hamilton cycle
# (accounts for level shifts and scaling differences)
lm_obj <- lm(original_hamilton_cycle ~ cycle_diff + SSA_out - 1)
coef <- lm_obj$coef

scale_shifted_SSA <- coef[1]*cycle_diff + coef[2]*SSA_out

# Compare adjusted SSA with original Hamilton cycle
ts.plot(scale_shifted_SSA, col = colo[1],
        main = "Adjusted SSA vs Hamilton cycle",
        ylim = c(min(original_hamilton_cycle, na.rm = TRUE),
                 max(original_hamilton_cycle, na.rm = TRUE)))
mtext("Adjusted SSA", col = colo[1], line = -1)
mtext("Hamilton", col = "black", line = -2)
lines(original_hamilton_cycle)
abline(h = 0)

# SSA produces fewer crossings, but not dramatically fewer in this misspecified setting.
# Further smoothing could be achieved by increasing ht or by analyzing subsamples.
compute_empirical_ht_func(scale_shifted_SSA)
compute_empirical_ht_func(original_hamilton_cycle)


#-----------------------------------------------
# 2.7 Forecasting (timeliness vs smoothness)
# Increasing the forecast horizon (delta) produces leading indicators.
# Smoothness is preserved since the holding-time constraint remains unchanged.

# Define forecast horizons (in months)
forecast_horizon <- c(6, 12)

# Apply SSA with forecasting
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

SSA_filt_ham_diff_x_forecast <- SSA_obj_ham_diff$ssa_x

# Each column corresponds to a different forecast horizon
head(SSA_filt_ham_diff_x_forecast)

# Compare nowcast and forecast filters
# Forecast filters place less weight on distant past observations (more responsive)
par(mfrow = c(1,1))
ts.plot(SSA_filt_ham_diff_x_forecast,
        col = c("orange","darkgreen"),
        main = "SSA: forecasts vs nowcast",
        ylim = c(min(SSA_filt_ham_diff),
                 max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff, col = "blue")
mtext(paste("Forecast horizon", forecast_horizon[1]), col = "orange", line = -1)
mtext(paste("Forecast horizon", forecast_horizon[2]), col = "darkgreen", line = -2)
mtext("Nowcast", col = "blue", line = -3)

#----------------------
# Apply filters to data
SSA_out_forecast_6  <- filter(x, SSA_filt_ham_diff_x_forecast[,1], side = 1)
SSA_out_forecast_12 <- filter(x, SSA_filt_ham_diff_x_forecast[,2], side = 1)

# Forecasted series lead the nowcast and are equally smooth.
# Variance is smaller due to increased estimation uncertainty (shrinkage).
mplot <- na.exclude(cbind(SSA_out,
                          SSA_out_forecast_6,
                          SSA_out_forecast_12,
                          ham_out))
colo <- c("blue","orange","darkgreen","red")
ts.plot(mplot[,1], col = colo[1], ylim = c(min(mplot), max(mplot)))
lines(mplot[,2], col = colo[2])
lines(mplot[,3], col = colo[3])
lines(mplot[,4], col = colo[4])
mtext("SSA nowcast", col = colo[1], line = -1)
mtext(paste("SSA forecast: delta=", forecast_horizon[1]), col = colo[2], line = -2)
mtext(paste("SSA forecast: delta=", forecast_horizon[2]), col = colo[3], line = -3)
mtext("Hamilton", col = colo[4], line = -4)
abline(h = 0)

# Normalize variance for visual comparison
mplot <- scale(na.exclude(cbind(SSA_out,
                                SSA_out_forecast_6,
                                SSA_out_forecast_12,
                                ham_out)),
               scale = TRUE, center = FALSE)
ts.plot(mplot[,1], col = colo[1], ylim = c(min(mplot), max(mplot)))
lines(mplot[,2], col = colo[2])
lines(mplot[,3], col = colo[3])
lines(mplot[,4], col = colo[4])
mtext("SSA nowcast", col = colo[1], line = -1)
mtext(paste("SSA forecast: delta=", forecast_horizon[1]), col = colo[2], line = -2)
mtext(paste("SSA forecast: delta=", forecast_horizon[2]), col = colo[3], line = -3)
mtext("Hamilton", col = colo[4], line = -4)
abline(h = 0)

# Empirical holding times: SSA designs remain smoother than Hamilton
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(SSA_out_forecast_6)
compute_empirical_ht_func(SSA_out_forecast_12)
compute_empirical_ht_func(ham_out)

# Lead-lag analysis
max_lead <- 8

# Nowcast: aligned with Hamilton
compute_min_tau_func(mplot[,c(1,4)], max_lead)

# 6-month forecast: leads by ~3 months
compute_min_tau_func(mplot[,c(2,4)], max_lead)

# 12-month forecast: leads by ~6 months
compute_min_tau_func(mplot[,c(3,4)], max_lead)

# Amplitude and phase-shift functions provide a complementary,
# frequency-domain view of smoothness and timeliness,
# confirming the results observed in the time domain.

########################################################################################################
# MAIN TAKE-AWAY:
# SSA demonstrates superior performance in timeliness (relative lead) and smoothness (less crossings)
########################################################################################################

# We verify this statement based on frequency-domain characteristics: amplitude and time-shift

#----------------------------------------
# 2.8 Compute Amplitude and Phase-Shift Functions
#
# PURPOSE:
#   - Amplitude function: measures noise leakage at higher frequencies;
#     values closer to zero indicate better suppression of high-frequency noise.
#   - Phase-shift function: measures timing distortion in the passband;
#     smaller (or negative) shifts indicate a relative lead over the target signal.

# --- Grid Setup ---
# Define the number of equidistant frequency ordinates over [0, pi]
K <- 600

# --- Color Palette ---
# Colors assigned to: SSA-nowcast, SSA-forecast(h1), SSA-forecast(h2), Hamilton
colo <- c("black", "blue", "darkgreen", "red")

# --- Compute Amplitude and Phase-Shift for Each Filter ---
# All filters are applied to returns (first differences of log-prices)
# Arguments: grid size K, filter coefficients (as vector), and a centering flag (FALSE)
amp_obj_SSA_now    <- amp_shift_func(K, as.vector(SSA_filt_ham_diff), F)
amp_obj_SSA_for_6  <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 1]), F)
amp_obj_SSA_for_12 <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 2]), F)
amp_obj_ham        <- amp_shift_func(K, ham_diff, F)

# --- Panel Layout: 2 rows (Amplitude on top, Phase-Shift on bottom) ---
par(mfrow = c(2, 1))

# ============================================================
# PLOT 1: Amplitude Functions
# ============================================================

# Combine amplitude functions from all four filters into a single matrix
mplot <- cbind(
  amp_obj_SSA_now$amp,
  amp_obj_SSA_for_6$amp,
  amp_obj_SSA_for_12$amp,
  amp_obj_ham$amp
)

# Normalize each amplitude to 1 at frequency zero (DC component)
# This ensures comparability across filters regardless of overall gain
mplot[, 1] <- mplot[, 1] / mplot[1, 1]
mplot[, 2] <- mplot[, 2] / mplot[1, 2]
mplot[, 3] <- mplot[, 3] / mplot[1, 3]
mplot[, 4] <- mplot[, 4] / mplot[1, 4]

# Assign descriptive column names encoding filter type, half-length, and forecast horizon
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1], ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2], ")", sep = ""),
  "Hamilton"
)

# Plot the first amplitude curve (SSA nowcast), then overlay remaining curves
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Amplitude: Hamilton-Filter vs. SSA Variants",
     ylim = c(min(mplot), max(mplot)), col = colo[1])

# Add legend-style annotation for the first filter
mtext(colnames(mplot)[1], line = -1, col = colo[1])

# Overlay amplitude curves for remaining filters and annotate each
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}

# Add frequency axis with pi-fraction labels and complete the plot frame
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# ============================================================
# PLOT 2: Phase-Shift Functions
# ============================================================

# Combine phase-shift functions from all four filters into a single matrix
mplot <- cbind(
  amp_obj_SSA_now$shift,
  amp_obj_SSA_for_6$shift,
  amp_obj_SSA_for_12$shift,
  amp_obj_ham$shift
)

# Reuse same column names as for the amplitude plot
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1], ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2], ")", sep = ""),
  "Hamilton"
)

# Plot the first phase-shift curve, then overlay remaining curves
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Phase-Shift: Hamilton-Filter vs. SSA Variants",
     ylim = c(min(mplot), max(mplot)), col = colo[1])

# Annotate first filter
mtext(colnames(mplot)[1], line = -1, col = colo[1])

# Overlay phase-shift curves for remaining filters and annotate each
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}

# Add frequency axis and complete the plot frame
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# ============================================================
# DISCUSSION
# ============================================================
# - The leads measured by the tau-statistic (time-domain, zero-crossings) closely
#   match the phase-shift differences observed in the passband — confirming consistency
#   between time-domain and frequency-domain diagnostics.
#
# - The Hamilton filter (HF) exhibits a substantial positive phase-shift (lag) in the passband:
#     * This lag exceeds that of the classic HP-concurrent trend applied to returns,
#       whose phase-shift must vanish at frequency zero by construction.
#     * The large lag is structurally driven by the 2-year forecast horizon embedded
#       in the Hamilton regression equation.
#
# - Clarification on "why you should never use the HP":
#     * Avoid the HP-gap— see tutorial 2 for confirmation.
#     * However, the HP-trend applied to log-differences performs well: it is smoother and
#       incurs a smaller lag than the Hamilton filter — see tutorial 2 for details.









# ============================================================
# Example 3: Extension of Example 2 — Fitting an ARMA Model to the Data
# ============================================================
# NOTE: Example 2 must be run at least once before Example 3 to ensure
#       all required objects and settings are properly initialized.

# Steps 3.1–3.3 assume that steps 2.1–2.3 have already been executed.
# If not, run the corresponding code blocks from Example 2 before proceeding.

#--------------------------------------------
# 3.4 Autocorrelation Analysis and ARMA Model Fitting
#--------------------------------------------

# Inspect the ACF of the return series x to assess autocorrelation structure
acf(x, main = "ACF of Return Series")

# Fit an ARMA(1,1) model to the return series x
ar_order <- 1
ma_order <- 1
estim_obj <- arima(x, order = c(ar_order, 0, ma_order))

# Note: AR and MA roots typically cancel each other in ARMA(1,1) fits to financial returns,
#       producing a parsimonious representation of weak but persistent autocorrelation.
estim_obj

# Diagnostic checks: residual ACF, p-values of Ljung-Box test, etc.
# Results are acceptable for the purposes of this analysis.
tsdiag(estim_obj)

# Compute the Wold (MA-infinity) decomposition via MA-inversion of the fitted ARMA model.
# xi represents the infinite-order moving average coefficients (impulse response weights).
xi <- c(1, ARMAtoMA(
  ar  = estim_obj$coef[1:ar_order],
  ma  = estim_obj$coef[ar_order + 1:ma_order],
  lag.max = L - 1
))
# NOTE: L must be large enough for xi to decay (converge) to zero.
#       L = 50 is sufficient in practice for typical ARMA(1,1) parameterizations.

par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition: Impulse Response Coefficients (xi)")

# Convolve xi with ham_diff to obtain the composite filter applied to the innovations epsilon_t.
# Interpretation: if the ARMA model is correctly specified (i.e., epsilon_t is white noise),
#   then the holding time of this convolved filter should match the empirical holding time
#   of ham_diff applied to the observed data x_t.
ham_conv <- conv_two_filt_func(xi, ham_diff)$conv
ts.plot(ham_conv, main = "Convolved Hamilton Filter (xi * ham_diff)")
# Looks strange, isn't it?

# Compute the theoretical holding time of the convolved filter ham_conv
ht_ham_conv_obj <- compute_holding_time_func(ham_conv)
ht_ham_conv_obj$ht

# Compare with the holding time of ham_diff under the white noise assumption.
# ham_conv is MUCH SMOOTHER than ham_diff because the ARMA filter acts as a lowpass,
# elongating the effective holding time.
ht_ham_diff_obj$ht

# Compare both theoretical holding times against the empirical holding time
# of residuals_adjusted (the output of the Hamilton filter applied to x).
compute_empirical_ht_func(residuals_adjusted)

# Summary of holding time comparison:
#   - ham_conv has a holding time closer to the empirical ht than ham_diff,
#     because it accounts for the autocorrelation structure of x_t.
#   - Some residual bias remains in ham_conv: the data is not fully stationary
#     (it is un-centered and exhibits long-run non-stationarity).
#   => We use ht_ham_conv as the basis for the SSA holding time constraint in step 3.5.


#--------------------------------------
# 3.5 Apply SSA with ARMA-Informed Holding Time Constraint
#--------------------------------------
# Having established stationarity (via differencing to ham_diff) and
# the Wold decomposition (xi), we are now equipped to apply SSA optimally.

# Retrieve the theoretical holding time based on the convolved filter
ht_ham_conv_obj$ht

# Target a 50% longer holding time than ham_conv:
# SSA with a correct model specification should generate ~30% fewer zero-crossings
# (i.e., ~50% longer holding time) relative to the benchmark.
ht <- 1.5 * ht_ham_conv_obj$ht

# Compute the autocorrelation parameter rho1 corresponding to the target holding time
rho1 <- compute_rho_from_ht(ht)

# Define the target filter: ham_diff is the filter applied to x_t that SSA aims to improve upon
gammak_generic <- ham_diff

# Set forecast horizon to zero (nowcast)
forecast_horizon <- 0

# ---- INCORRECT SSA CALL (for illustration purposes) ----
# Warning: xi (Wold decomposition) is intentionally omitted here to demonstrate
#          the consequences of incorrectly assuming white noise innovations.
#          With no xi supplied, SSA treats x_t as white noise, causing the ht
#          constraint to be excessively tight and the resulting filter to over-smooth.
SSA_obj_ham_diff   <- SSA_func(L, forecast_horizon, gammak_generic, rho1)
SSA_filt_ham_diff  <- SSA_obj_ham_diff$ssa_eps

# Visualize the incorrect SSA output vs. the Hamilton filter:
# The SSA filter is clearly over-smoothed relative to ham_diff
par(mfrow = c(1, 1))
mplot <- cbind(ham_diff, SSA_filt_ham_diff)
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"))
mtext("Hamilton filter (applied to differences x_t)",        col = "black", line = -1)
mtext("Incorrect SSA (white noise assumption — over-smoothed)", col = "blue",  line = -2)

# Verify holding time constraint for the incorrect SSA:
# The constraint is met numerically, but it is misspecified —
# the target ht was derived from ham_conv (which assumes autocorrelation),
# yet SSA was run without supplying xi, creating an internal inconsistency.
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff)
ht_obj$ht  # Matches ht by construction, but ht itself is mis-targeted
ht

# ---- CORRECT SSA CALL: supply xi (Wold decomposition) ----
# With xi provided, SSA correctly accounts for the autocorrelation structure of x_t.
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# ssa_eps: filter coefficients as applied to white noise innovations epsilon_t
#   => Used primarily to verify the ht constraint (convergence to global optimum)
SSA_filt_ham_diff_eps <- SSA_obj_ham_diff$ssa_eps

# ssa_x: filter coefficients as applied to the observed return series x_t
#   => This is the operationally relevant filter for real-time signal extraction
#   => Differs from ssa_eps because it has been convolved with xi^{-1}
SSA_filt_ham_diff_x <- SSA_obj_ham_diff$ssa_x

# Compare ham_diff and ssa_x — both applied to x_t (correct apples-to-apples comparison)
# The correct SSA now tracks ham_diff much more faithfully than the white-noise version
mplot <- cbind(ham_diff, SSA_filt_ham_diff_x)
par(mfrow = c(1, 1))
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"),
        main = "Filters Applied to Return Series x_t")
mtext("Hamilton filter",                          col = "black", line = -1)
mtext("SSA (with autocorrelation model — correct)", col = "blue",  line = -2)

# Alternative comparison: ham_conv vs. ssa_eps — both applied to innovations epsilon_t
# This is the natural domain for comparing filters in the Wold representation
mplot <- cbind(ham_conv, SSA_obj_ham_diff$ssa_eps)
par(mfrow = c(1, 1))
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"),
        main = "Convolved Filters Applied to Innovations epsilon_t")
mtext("Hamilton (convolved with xi)", col = "black", line = -1)
mtext("SSA (in innovation space)",    col = "blue",  line = -2)

# Verify holding time constraint for the CORRECT SSA call:
# ht of ssa_eps should match the targeted ht (up to numerical rounding),
# confirming that the optimization converged to the global maximum.
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff_eps)
ht_obj$ht  # Should be approximately equal to ht
ht

# Note on ht of ssa_x vs. ssa_eps:
#   The holding time of ssa_x (applied to x_t) is considerably shorter than that of ssa_eps,
#   because x_t is smoother (more autocorrelated) than white noise epsilon_t.
#   This explains part of the holding time bias observed in practice
#   (see Example 7 in Tutorial 1 for a detailed discussion).
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff_x)
ht_obj$ht


#--------------------------------------------------
# 3.6 Filter the Series and Evaluate Performance
#--------------------------------------------------

# --- 1. SSA Filter Output ---
# Apply the SSA filter (in x_t space) to the original series using one-sided filtering
SSA_out <- filter(x, SSA_filt_ham_diff_x, side = 1)

# The empirical ht of SSA_out exceeds the targeted ht.
# Reason: x_t is non-stationary; the cycle frequency/amplitude shifts over time,
#         violating the stationarity assumption embedded in the ht constraint.
compute_empirical_ht_func(SSA_out)
ht  # Targeted holding time for reference

# --- 2. Hamilton Filter Output ---
ham_out <- filter(x, ham_diff, side = 1)
compute_empirical_ht_func(ham_out)

# The appropriate benchmark for Hamilton's empirical ht is ht_ham_conv (not ht_ham_diff),
# since ham_conv correctly reflects the autocorrelation structure of x_t.
ht_ham_conv_obj$ht

# Visual comparison of SSA and Hamilton filter outputs
# Both series appear similar, but Hamilton has greater high-frequency leakage,
# generating noisier zero-crossings.
mplot <- na.exclude(cbind(SSA_out, ham_out))
colo  <- c("blue", "red")
ts.plot(mplot[, 1], col = colo[1])
lines(mplot[, 2], col = colo[2])
abline(h = 0)
mtext("Hamilton filter output", col = "red",  line = -1)
mtext("SSA filter output",      col = "blue", line = -2)

# Empirical holding times: SSA holds longer than Hamilton, but the 50% target is not fully achieved.
# Root cause: structural non-stationarity of the business cycle (changing cycle frequency post-1990).
# Suggestion: Try ht <- 2 * ht_ham_conv_obj$ht for a more aggressive smoothing target,
#             or restrict the sample to post-1990 data (see Example 4 for this approach).
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(ham_out)


# --- 3. Adjust SSA Cycle to Match the Original Hamilton Cycle Scale and Level ---
# Regress the original Hamilton cycle on: (i) the differenced cycle and (ii) SSA_out
# to estimate the level adjustment and scaling factors (cf. Example 1 for methodology)
lm_obj <- lm(original_hamilton_cycle ~ cycle_diff + SSA_out - 1)
coef   <- lm_obj$coef

# Construct the level- and scale-adjusted SSA series
scale_shifted_SSA <- coef[1] * cycle_diff + coef[2] * SSA_out

# Plot the adjusted SSA cycle against the original Hamilton cycle.
# Note: scale_shifted_SSA has missing values at the start due to filter initialization lag.
ts.plot(scale_shifted_SSA, col = colo[1],
        main  = "Level-Adjusted SSA vs. Original Hamilton Cycle",
        ylim  = c(min(original_hamilton_cycle, na.rm = TRUE),
                  max(original_hamilton_cycle, na.rm = TRUE)))
mtext("Adjusted and scaled SSA cycle", col = colo[1], line = -1)
mtext("Original Hamilton cycle",       col = "black", line = -2)
lines(original_hamilton_cycle)
abline(h = 0)

# Both series track each other closely; SSA generates slightly fewer zero-crossings
# than the Hamilton cycle, but the improvement falls short of the ~30% theoretical target.
#
# Two paths forward to improve the result:
#   (a) Increase ht in the SSA call to impose stronger smoothing, or
#   (b) Restrict the estimation sample to post-1990 data to reduce structural
#       misspecification from the earlier, more volatile cycle regime (see Example 4).
compute_empirical_ht_func(scale_shifted_SSA)
compute_empirical_ht_func(original_hamilton_cycle)



#-----------------------------------------------
# 3.7 Forecasting: Timeliness and Lead/Lag Analysis
#-----------------------------------------------
# OBJECTIVE:
#   Augment the forecast horizon (delta in the JBCY paper) to obtain a faster (leading) SSA filter.
#   Crucially, the holding time constraint is kept fixed, so smoothness/noise suppression
#   is preserved while timeliness is improved — there is no smoothness-timeliness trade-off here.

# Define two forecast horizons for comparison against the nowcast (delta = 0):
#   - 6 months ahead (half-year)
#   - 12 months ahead (full year)
# SSA_func accepts a vector of forecast horizons and returns a matrix of filters,
# where each column corresponds to one forecast horizon.
forecast_horizon <- c(6, 12)

# Fit SSA with the ARMA(1,1)-informed Wold decomposition xi and the two forecast horizons
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract the filters applied to the observed return series x_t (one column per horizon)
SSA_filt_ham_diff_x_forecast <- SSA_obj_ham_diff$ssa_x

# --- Visualize Filter Coefficients ---
# Morphing effect: forecast filters assign less weight to the remote past,
# making them faster (more leading) while maintaining the same smoothness (ht unchanged).
# Note: these forecast filters differ from those in Example 2 because we now
#       incorporate the ARMA(1,1) model of the data via xi.
par(mfrow = c(1, 1))
ts.plot(SSA_filt_ham_diff_x_forecast,
        col  = c("orange", "darkgreen"),
        main = "SSA Forecast Filters vs. Nowcast Filter Coefficients",
        ylim = c(min(SSA_filt_ham_diff_x), max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x, col = "blue")
mtext(paste("Forecast horizon delta =", forecast_horizon[1]), col = "orange",    line = -1)
mtext(paste("Forecast horizon delta =", forecast_horizon[2]), col = "darkgreen", line = -2)
mtext("Nowcast (delta = 0)",                                   col = "blue",      line = -3)

#----------------------
# Apply Filters to the Return Series x
#----------------------

# 6-month-ahead SSA forecast filter output
SSA_out_forecast_6  <- filter(x, SSA_filt_ham_diff_x_forecast[, 1], side = 1)

# 12-month-ahead SSA forecast filter output
SSA_out_forecast_12 <- filter(x, SSA_filt_ham_diff_x_forecast[, 2], side = 1)

# Plot all filtered outputs: SSA nowcast, SSA forecasts, and Hamilton filter
# Expected pattern: SSA forecast outputs are left-shifted (leading) relative to Hamilton;
#                   smoothness is approximately equal across SSA variants.
mplot <- na.exclude(cbind(SSA_out, SSA_out_forecast_6, SSA_out_forecast_12, ham_out))
colo  <- c("blue", "orange", "darkgreen", "red")
ts.plot(mplot[, 1], col = colo[1], ylim = c(min(mplot), max(mplot)))
lines(mplot[, 2], col = colo[2])
lines(mplot[, 3], col = colo[3])
lines(mplot[, 4], col = colo[4])
mtext("SSA nowcast (delta = 0)", col = colo[1], line = -1)
mtext(paste("SSA forecast: delta =", forecast_horizon[1]),    col = colo[2], line = -2)
mtext(paste("SSA forecast: delta =", forecast_horizon[3]),    col = colo[3], line = -3)
mtext("Hamilton filter",                                       col = colo[4], line = -4)
abline(h = 0)

# --- Empirical Holding Times ---
# SSA designs achieve approximately 50% longer holding times than Hamilton,
# confirming stronger noise suppression without sacrificing timeliness.
compute_empirical_ht_func(SSA_out)           # SSA nowcast
compute_empirical_ht_func(SSA_out_forecast_6)  # SSA 6-month forecast
compute_empirical_ht_func(SSA_out_forecast_12) # SSA 12-month forecast
compute_empirical_ht_func(ham_out)             # Hamilton benchmark

# --- Lead/Lag Analysis via Tau-Statistic (Zero-Crossing Shifts) ---
# Compute the minimum-tau shift between each SSA output and the Hamilton filter output.
# A negative shift indicates that SSA leads Hamilton at zero-crossings.
max_lead <- 8

# SSA nowcast vs. Hamilton: expected to be approximately synchronized (no lead)
shift_obj <- compute_min_tau_func(mplot[, c(1, 4)], max_lead)

# SSA 6-month forecast vs. Hamilton: expected lead of approximately one quarter
shift_obj <- compute_min_tau_func(mplot[, c(2, 4)], max_lead)

# SSA 12-month forecast vs. Hamilton: expected lead of approximately two quarters
shift_obj <- compute_min_tau_func(mplot[, c(3, 4)], max_lead)

# NOTE: The frequency-domain analysis in section 3.8 (amplitude and phase-shift functions)
#       provides a formal, complementary characterization of the same smoothness and
#       timeliness properties observed empirically above.


#----------------------------------------
# 3.8 Amplitude and Phase-Shift Functions
#----------------------------------------
# PURPOSE:
#   - Amplitude function: quantifies noise leakage at high frequencies;
#     values closer to zero indicate better suppression of unwanted high-frequency variation.
#   - Phase-shift function: quantifies timing distortion in the passband;
#     a more negative (or smaller) shift indicates a relative lead over the target signal.

# --- Grid Setup ---
# Number of equidistant frequency ordinates over [0, pi]
K <- 600

# --- Color Palette ---
# Colors assigned to: SSA-nowcast, SSA-forecast(6m), SSA-forecast(12m), Hamilton
colo <- c("black", "blue", "darkgreen", "red")

# --- Compute Amplitude and Phase-Shift for All Four Filters ---
# All filters are expressed in x_t space (applied to return series)
amp_obj_SSA_now    <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x),              F)
amp_obj_SSA_for_6  <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 1]), F)
amp_obj_SSA_for_12 <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 2]), F)
amp_obj_ham        <- amp_shift_func(K, ham_diff,                                      F)

# --- Panel Layout: Amplitude (top) and Phase-Shift (bottom) ---
par(mfrow = c(2, 1))

# ============================================================
# PLOT 1: Amplitude Functions
# ============================================================

# Assemble amplitude matrix
mplot <- cbind(
  amp_obj_SSA_now$amp,
  amp_obj_SSA_for_6$amp,
  amp_obj_SSA_for_12$amp,
  amp_obj_ham$amp
)

# Normalize each amplitude to 1 at frequency zero (DC component)
# This ensures all filters are compared on the same scale regardless of overall gain
mplot[, 1] <- mplot[, 1] / mplot[1, 1]
mplot[, 2] <- mplot[, 2] / mplot[1, 2]
mplot[, 3] <- mplot[, 3] / mplot[1, 3]
mplot[, 4] <- mplot[, 4] / mplot[1, 4]

# Assign descriptive column names encoding filter type, half-length, and forecast horizon
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1], ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2], ")", sep = ""),
  "Hamilton"
)

# Plot amplitude functions
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Amplitude: Hamilton Filter vs. SSA Variants (ARMA Model)",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

# Overlay remaining amplitude curves
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}

# Frequency axis with pi-fraction labels
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# ============================================================
# PLOT 2: Phase-Shift Functions
# ============================================================

# Assemble phase-shift matrix
mplot <- cbind(
  amp_obj_SSA_now$shift,
  amp_obj_SSA_for_6$shift,
  amp_obj_SSA_for_12$shift,
  amp_obj_ham$shift
)

# Reuse same column names as amplitude plot for consistency
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1], ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2], ")", sep = ""),
  "Hamilton"
)

# Plot phase-shift functions
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Phase-Shift: Hamilton Filter vs. SSA Variants (ARMA Model)",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

# Overlay remaining phase-shift curves
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}

# Frequency axis with pi-fraction labels
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# ============================================================
# DISCUSSION
# ============================================================
#
# ROBUSTNESS TO MODEL MISSPECIFICATION:
#   - SSA is remarkably robust to misspecification of the dependence structure.
#   - Examples 2 (white noise assumption) and 3 (ARMA(1,1) model) yield filters
#     with broadly similar/comparable empirical performance.
#
# NON-STATIONARITY AND HOLDING TIME BIAS:
#   - The full sample spans a very long history, introducing structural non-stationarity
#     that systematically biases the empirical holding time upward.
#   - In all cases, SSA outperformed the Hamilton target in terms of smoothness;
#     however, the magnitude of the gain was sometimes less than projected.
#   - Remedies: (a) increase the ht constraint in the SSA call, or
#               (b) shorten the estimation window (see Example 4 for the post-1990 subsample).
#   - In all cases, the forecast SSA filters outperformed Hamilton in both
#     smoothness (longer holding time) and timeliness (left shift at zero-crossings).
#
# HAMILTON FILTER LAG:
#   - The Hamilton filter exhibits a substantial positive phase-shift (lag) in the passband.
#   - This lag exceeds that of the classic HP-concurrent trend applied to returns,
#     whose phase-shift must vanish at frequency zero by construction.
#   - The source of the lag is structural: the Hamilton regression uses a 2-year
#     forecast horizon, which induces a systematic delay in the extracted cycle.
#
# CLARIFICATION ON "NEVER USE THE HP":
#   - The advice applies specifically to the HP-gap:
#     this produces a spurious and unreliable cycle — see Tutorial 2 for confirmation.
#   - The HP-trend applied to returns performs well: it is smooth and its passband lag
#     is smaller than that of the Hamilton filter — see Tutorial 2 for details.


# ============================================================
# Example 4: Post-1990 Subsample Analysis
# ============================================================
# PURPOSE:
#   Replicates Example 3 using data from 1990 onwards only.
#   Restricting to the post-1990 subsample (the "Great Moderation" era) addresses
#   the structural non-stationarity bias observed in the full-sample analyses
#   of Examples 2 and 3.
#
# The subsample selection affects two key components:
#   1. Hamilton filter: regression parameters are re-estimated on post-1990 data,
#      yielding a different cycle definition than in Examples 2–3.
#   2. ARMA model: the dependence structure (and hence the Wold decomposition xi)
#      is re-estimated, reflecting the smoother dynamics of the Great Moderation.
#
# NOTE: The pandemic period is excluded here (sample ends 2019).
#       Pandemic effects are analyzed separately in Example 4.9 at the end of the tutorial.

# Load log-transformed PAYEMS for the post-1990, pre-pandemic window
y   <- as.double(log(PAYEMS["1990::2019"]))
ts.plot(y, main = "Log(PAYEMS): 1990–2019")
len <- length(y)

#--------------------------
# 4.1 Hamilton Filter (Post-1990 Parameterization)
#--------------------------
# Hamilton's original settings for quarterly data:
#   - Forecast horizon h = 2 years = 8 quarters
#   - AR order p = 4 (to account for up to 4 unit roots if present)
# Adaptation to monthly PAYEMS:
#   - h is scaled to 2 * 12 = 24 months
#   - p = 4 is retained (sufficient to remove integration-induced autocorrelation)
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

# Note: Regression parameters differ substantially from Examples 2 and 3 (full sample).
# This highlights a key limitation of the Hamilton filter: unlike HP (which is parameter-free),
# Hamilton depends on data-fitted regression coefficients, making it susceptible to
# instability and revision errors across different sample periods.
summary(lm_obj)

# The residual cycle appears noisier than in Examples 2–3,
# reflecting the tighter fit of the regression to post-1990 data.
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

# --- Cointegration Adjustment ---
# The raw filter coefficients sum to approximately zero (required for trend removal
# from a non-stationary series), but the sum is not exactly zero.
# Without an exact zero-sum constraint, the out-of-sample cycle will be non-stationary,
# necessitating frequent filter updates as new data arrives (causing revisions).
sum(hamilton_filter)

# Impose the exact cointegration constraint:
# Distribute the residual sum evenly across the p AR coefficients.
hamilton_filter_adjusted <- hamilton_filter
hamilton_filter_adjusted[(h + 1):(h + p)] <-
  hamilton_filter_adjusted[(h + 1):(h + p)] - sum(hamilton_filter) / p

# Verify that the adjusted filter sums to exactly zero
sum(hamilton_filter_adjusted)

# Compute the cointegration-adjusted cycle
residuals_adjusted <- data_mat %*% hamilton_filter_adjusted

# Optional centering (disabled by default; un-centered cycle is used throughout)
if (F)
  residuals_adjusted <- residuals_adjusted - mean(residuals_adjusted)

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

# Save the cycle difference for use in the level-shift adjustment when comparing
# the SSA output back to the original (classic) Hamilton cycle.
cycle_diffh <- residuals - residuals_adjusted


#---------------------------------------------------
# 4.2 Transformation: Level to First Differences
#---------------------------------------------------
# Apply the same level-to-differences transformation as in Examples 2–3,
# but with a larger filter length L = 100.
#
# Motivation: post-1990 log-returns exhibit slower ACF decay (longer memory),
# consistent with the Great Moderation (lower volatility, more persistent dynamics).
# A larger L is required to ensure that the Wold decomposition xi converges to zero.
# See Section 2.3 and Proposition 4 in the JBCY paper for the theoretical background.
L <- 100

# Ensure L is at least as long as the Hamilton filter (required for valid convolution)
L <- max(length(hamilton_filter_adjusted), L)
if (L > length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L <- c(hamilton_filter_adjusted,
                                  rep(0, L - length(hamilton_filter_adjusted)))

# Convolve the adjusted Hamilton filter with the unit-root (summation) filter
# to obtain the equivalent filter applied to first differences (returns)
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
cycle_diff             <- c(rep(NA, length(x) - length(cycle_diffh)),  cycle_diffh)
original_hamilton_cycle <- c(rep(NA, length(x) - length(residuals)),    residuals)

# Confirmation plot: both approaches should produce identical cycle estimates
par(mfrow = c(1, 1))
ts.plot(residual_diff, col = "blue",
        main = "Replication Check: HF Applied to Differences vs. Levels")
lines(residuals_adjusted[(L - p - h + 2):length(residuals)], col = "red")

# With ham_diff verified, we can now apply SSA in the differences domain.

#-------------------------------------------------------------------------------
# 4.3 Holding Time Analysis
#-------------------------------------------------------------------------------

# Compute the theoretical holding time of ham_diff under the white noise assumption
ht_ham_diff_obj <- compute_holding_time_func(ham_diff)

# The holding time is substantially shorter than in Examples 2–3:
# the regression is now fit to post-1990 data, producing a higher-frequency cycle.
ht_ham_diff_obj$ht
ht_ham_example2  # Reference value from Example 2

# The empirical holding time (from filtered output) is much longer than the theoretical value.
# Root cause: x_t (log-returns) are not white noise — they exhibit positive autocorrelation.
# See Examples 1–3 for a detailed discussion of this bias.
compute_empirical_ht_func(residuals_adjusted)

# Visual confirmation that log-returns are not white noise
par(mfrow = c(1, 1))
ts.plot(x, main = "Log-Returns of PAYEMS (Post-1990)")
abline(h = 0)

#----------------------------------------------------------------------------
# 4.4 Autocorrelation Analysis and ARMA Model Fitting
#----------------------------------------------------------------------------
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

# ACF of post-1990 log-returns: slower decay than full-sample ACF,
# consistent with the Great Moderation (reduced volatility and longer memory).
acf(x[1:in_sample_length], main = "ACF: Slowly Decaying (Post-1990, Longer Memory)")

# Fit an ARMA(1,1) model to capture the persistent but weak autocorrelation structure
ar_order  <- 1
ma_order  <- 1
estim_obj <- arima(x[1:in_sample_length], order = c(ar_order, 0, ma_order))

# Note: The near-cancellation of AR and MA roots is typical for financial returns
#       and produces a parsimonious representation of slowly decaying autocorrelation.
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

# Visualize xi: the slower decay relative to Examples 2–3 confirms longer memory
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition xi: Slowly Decaying Impulse Response (Post-1990)")

# Convolve xi with ham_diff to obtain the composite filter applied to innovations epsilon_t.
# This convolved filter is used solely for computing the holding time constraint.
ham_conv        <- conv_two_filt_func(xi, ham_diff)$conv
ht_ham_conv_obj <- compute_holding_time_func(ham_conv)


#--------------------------------------
# 4.5 Apply SSA (Post-1990, ARMA-Informed)
#--------------------------------------

# The theoretical holding time of ham_conv is slightly under one year
# (ignoring the non-stationarity bias discussed in Examples 1–3)
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

# Fit SSA with the ARMA(1,1) Wold decomposition xi supplied
# This ensures SSA correctly accounts for the autocorrelation structure of x_t
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# ssa_eps: filter in the innovation (epsilon_t) domain
#   => Used primarily to verify convergence of the optimization to the global maximum
SSA_filt_ham_diff_eps <- SSA_obj_ham_diff$ssa_eps

# ssa_x: filter in the observable data (x_t) domain
#   => Operationally relevant filter for real-time signal extraction
SSA_filt_ham_diff_x <- SSA_obj_ham_diff$ssa_x

# --- Compare Filters in the x_t Domain ---
# ham_diff and ssa_x are both applied to x_t: direct apples-to-apples comparison
mplot <- cbind(ham_diff, SSA_filt_ham_diff_x)
par(mfrow = c(1, 1))
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"),
        main = "Filter Coefficients Applied to Return Series x_t")
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

# --- Verify Global Convergence of the SSA Optimization ---
# The holding time of ssa_eps must match the targeted ht (up to numerical rounding).
# Agreement between the two numbers confirms convergence to the global maximum.
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff_eps)
ht_obj$ht  # Computed holding time of ssa_eps
ht         # Targeted holding time


#--------------------------------------------------
# 4.6 Filter the Series and Evaluate Performance
#--------------------------------------------------

# --- 4.6.1 SSA Filter Output ---
# Apply the SSA nowcast filter (in x_t space) to log-returns using one-sided filtering
SSA_out <- filter(x, SSA_filt_ham_diff_x, side = 1)

# The empirical holding time exceeds the targeted ht.
# Root cause: x_t (log-returns) are non-stationary over the full sample,
# violating the stationarity assumption embedded in the ht constraint.
compute_empirical_ht_func(SSA_out)
ht   # Targeted holding time for reference

# --- 4.6.2 Hamilton Filter Output ---
# Apply the Hamilton filter (in x_t space) to log-returns
ham_out <- filter(x, ham_diff, side = 1)
compute_empirical_ht_func(ham_out)

# The appropriate theoretical benchmark for Hamilton's empirical ht is ht_ham_conv
# (which accounts for autocorrelation via the ARMA model), not ht_ham_diff.
ht_ham_conv_obj$ht

# --- Visual Comparison of SSA and Hamilton Outputs ---
# Both outputs appear visually similar, but Hamilton exhibits greater high-frequency
# leakage, producing noisier zero-crossings relative to SSA.
mplot <- na.exclude(cbind(SSA_out, ham_out))
colo  <- c("blue", "red")
ts.plot(mplot[, 1], col = colo[1])
lines(mplot[, 2], col = colo[2])
abline(h = 0)
mtext("Hamilton filter output", col = "red",  line = -1)
mtext("SSA filter output",      col = "blue", line = -2)

# Empirical holding time comparison:
# SSA achieves approximately 50% longer holding time than Hamilton,
# consistent with the targeted improvement (difference is within sampling error).
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(ham_out)

# --- Level- and Scale-Adjusted SSA Cycle ---
# To enable a direct comparison with the original (classic) Hamilton cycle,
# we regress original_hamilton_cycle on cycle_diff and SSA_out to estimate
# the required level shift and scale factor (cf. Example 1 for methodology).
lm_obj <- lm(original_hamilton_cycle ~ cycle_diff + SSA_out - 1)
coef   <- lm_obj$coef

# Construct the adjusted SSA cycle aligned to the original Hamilton cycle
scale_shifted_SSA <- coef[1] * cycle_diff + coef[2] * SSA_out

# Plot the adjusted SSA cycle against the original Hamilton cycle.
# Note: scale_shifted_SSA has missing values at the start due to filter initialization.
ts.plot(scale_shifted_SSA, col = colo[1],
        main = "Level-Adjusted SSA Cycle vs. Original Hamilton Cycle",
        ylim = c(min(original_hamilton_cycle, na.rm = TRUE),
                 max(original_hamilton_cycle, na.rm = TRUE)))
mtext("Adjusted and scaled SSA cycle", col = colo[1], line = -1)
mtext("Original Hamilton cycle",       col = "black", line = -2)
lines(original_hamilton_cycle)
abline(h = 0)

# Result: SSA generates approximately 30% fewer zero-crossings than the original Hamilton cycle.
# Agreement between empirical and expected holding times is better here than in Examples 2–3,
# because the post-1990 subsample is more stationary (reduced structural misspecification).
compute_empirical_ht_func(scale_shifted_SSA)
compute_empirical_ht_func(original_hamilton_cycle)

# Note on time-frame alignment:
# scale_shifted_SSA is shorter than original_hamilton_cycle due to filter initialization lag.
# We align the comparison window by restricting original_hamilton_cycle to the same span.
# This correction does not materially affect the conclusions.
compute_empirical_ht_func(original_hamilton_cycle[L:length(original_hamilton_cycle)])


#-----------------------------------------------
# 4.7 Forecasting: Timeliness and Lead/Lag Analysis
#-----------------------------------------------
# OBJECTIVE:
#   Augment the forecast horizon (delta in the JBCY paper) to construct faster (leading) SSA filters.
#   The holding time constraint is kept fixed, so noise suppression is preserved
#   while timeliness is improved — no smoothness-timeliness trade-off is incurred.

# Define two forecast horizons for comparison against the nowcast (delta = 0):
#   - 6 months ahead (half-year)
#   - 12 months ahead (full year)
# SSA_func accepts a vector of horizons and returns a filter matrix
# (one column per forecast horizon).
forecast_horizon <- c(6, 12)

# Fit SSA with the ARMA(1,1) Wold decomposition xi and both forecast horizons
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract the filters in the x_t domain (one column per forecast horizon)
SSA_filt_ham_diff_x_forecast <- SSA_obj_ham_diff$ssa_x

# --- Visualize Filter Coefficient Morphing ---
# As the forecast horizon increases, the filter assigns progressively less weight
# to the remote past (morphing towards a leading shape), while the holding time
# constraint ensures that smoothness is maintained across all horizons.
par(mfrow = c(1, 1))
ts.plot(SSA_filt_ham_diff_x_forecast,
        col  = c("orange", "darkgreen"),
        main = "SSA Forecast Filter Coefficients vs. Nowcast",
        ylim = c(min(SSA_filt_ham_diff_x), max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x, col = "blue")
mtext(paste("Forecast horizon delta =", forecast_horizon[1]), col = "orange",    line = -1)
mtext(paste("Forecast horizon delta =", forecast_horizon[2]), col = "darkgreen", line = -2)
mtext("Nowcast (delta = 0)",                                   col = "blue",      line = -3)

# --- Verify Global Convergence for All Forecast Horizons ---
# Apply compute_holding_time_func to each column of ssa_eps (filter in epsilon_t domain).
# Matching values across all horizons confirm that the optimization converged
# to the global maximum for each forecast horizon independently.
apply(SSA_obj_ham_diff$ssa_eps, 2, compute_holding_time_func)
ht  # Targeted holding time for reference

#----------------------
# Apply Forecast Filters to the Return Series x
#----------------------

# 6-month-ahead SSA forecast filter output
SSA_out_forecast_6  <- filter(x, SSA_filt_ham_diff_x_forecast[, 1], side = 1)

# 12-month-ahead SSA forecast filter output
SSA_out_forecast_12 <- filter(x, SSA_filt_ham_diff_x_forecast[, 2], side = 1)

# --- Plot All Filter Outputs ---
# Expected pattern: SSA forecast outputs are left-shifted (leading) relative to Hamilton;
# smoothness is approximately equal across all SSA variants.
mplot <- na.exclude(cbind(SSA_out, SSA_out_forecast_6, SSA_out_forecast_12, ham_out))
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

# --- Empirical Holding Times ---
# All SSA variants achieve approximately 50% longer holding times than Hamilton,
# confirming stronger noise suppression without sacrificing timeliness.
compute_empirical_ht_func(SSA_out)            # SSA nowcast
compute_empirical_ht_func(SSA_out_forecast_6)   # SSA 6-month forecast
compute_empirical_ht_func(SSA_out_forecast_12)  # SSA 12-month forecast
compute_empirical_ht_func(ham_out)              # Hamilton benchmark

# --- Lead/Lag Analysis via Tau-Statistic (Zero-Crossing Shifts) ---
# Compute the minimum-tau shift between each SSA output and Hamilton.
# A negative shift indicates that SSA leads Hamilton at zero-crossings.
max_lead <- 10

# SSA nowcast vs. Hamilton: expected to be approximately synchronized
shift_obj <- compute_min_tau_func(mplot[, c(1, 4)], max_lead)

# SSA 6-month forecast vs. Hamilton: expected to lead Hamilton by approximately one quarter
shift_obj <- compute_min_tau_func(mplot[, c(2, 4)], max_lead)

# SSA 12-month forecast vs. Hamilton: expected to lead Hamilton by approximately two quarters
shift_obj <- compute_min_tau_func(mplot[, c(3, 4)], max_lead)

# NOTE: The frequency-domain analysis in section 4.8 provides a formal, complementary
#       characterization of the same smoothness and timeliness properties observed above.


#----------------------------------------
# 4.8 Amplitude and Phase-Shift Functions
#----------------------------------------
# PURPOSE:
#   - Amplitude function: quantifies noise leakage at high frequencies;
#     values closer to zero indicate stronger suppression of high-frequency noise
#     and correspond to longer holding times.
#   - Phase-shift function: quantifies timing distortion in the passband;
#     a more negative (or smaller) shift at low frequencies indicates a relative lead.
#
# These frequency-domain diagnostics formally confirm the empirical findings in section 4.7.

# --- Grid Setup ---
# Number of equidistant frequency ordinates over [0, pi]
K <- 600

# --- Color Palette ---
# Colors assigned to: SSA-nowcast, SSA-forecast(6m), SSA-forecast(12m), Hamilton
colo <- c("black", "blue", "darkgreen", "red")

# --- Compute Amplitude and Phase-Shift for All Four Filters ---
# All filters are in the x_t domain (applied to log-return series)
amp_obj_SSA_now    <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x),               F)
amp_obj_SSA_for_6  <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 1]), F)
amp_obj_SSA_for_12 <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 2]), F)
amp_obj_ham        <- amp_shift_func(K, ham_diff,                                       F)

# --- Panel Layout: Amplitude (top) and Phase-Shift (bottom) ---
par(mfrow = c(2, 1))

# ============================================================
# PLOT 1: Amplitude Functions
# ============================================================

# Assemble amplitude matrix
mplot <- cbind(
  amp_obj_SSA_now$amp,
  amp_obj_SSA_for_6$amp,
  amp_obj_SSA_for_12$amp,
  amp_obj_ham$amp
)

# Normalize each amplitude to 1 at frequency zero (DC component)
# Enables scale-free comparison of noise suppression across all filters
mplot[, 1] <- mplot[, 1] / mplot[1, 1]
mplot[, 2] <- mplot[, 2] / mplot[1, 2]
mplot[, 3] <- mplot[, 3] / mplot[1, 3]
mplot[, 4] <- mplot[, 4] / mplot[1, 4]

# Descriptive column names encoding filter type, half-length, and forecast horizon
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1], ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2], ")", sep = ""),
  "Hamilton"
)

# Plot amplitude functions:
# Smaller amplitude at high frequencies => stronger noise suppression => longer holding time
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Amplitude: Hamilton Filter vs. SSA Variants (Post-1990)",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

# Overlay remaining amplitude curves
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

# ============================================================
# PLOT 2: Phase-Shift Functions
# ============================================================

# Assemble phase-shift matrix
mplot <- cbind(
  amp_obj_SSA_now$shift,
  amp_obj_SSA_for_6$shift,
  amp_obj_SSA_for_12$shift,
  amp_obj_ham$shift
)

# Reuse same column names as amplitude plot for consistency
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1], ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2], ")", sep = ""),
  "Hamilton"
)

# Plot phase-shift functions:
# Smaller (more negative) shift at low frequencies => relative lead in the passband
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Phase-Shift: Hamilton Filter vs. SSA Variants (Post-1990)",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

# Overlay remaining phase-shift curves
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

# ============================================================
# DISCUSSION: Post-1990 Subsample vs. Full Sample (Examples 2–3)
# ============================================================
#
# IMPROVED STATIONARITY AND HOLDING TIME CONSISTENCY:
#   - Discarding pre-1990 data (WWII through the Great Inflation era) substantially
#     reduces structural non-stationarity, yielding a more homogeneous estimation environment.
#   - As a result, the agreement between theoretical and empirical holding times improves
#     considerably compared to Examples 2 and 3 (full sample).
#   - Empirical holding times of SSA more closely match the intended 50% increase over
#     the Hamilton benchmark. Residual deviations are consistent with random sampling error:
#       * Slightly below target for unadjusted cycles.
#       * Slightly above target for level-adjusted cycles.
#
# HAMILTON FILTER LAG (POST-1990):
#   - The positive phase-shift (lag) of the Hamilton filter is smaller in the post-1990
#     subsample than in the full-sample analyses of Examples 2–3.
#   - This confirms that the Hamilton filter's lag is data-window dependent,
#     reflecting its reliance on regression parameters estimated from the chosen sample.
#   - Despite being smaller, the Hamilton lag still exceeds that of the classic HP-concurrent
#     trend applied to returns, whose phase-shift vanishes at frequency zero by construction.
#   - The residual lag is structurally driven by the 2-year forecast horizon in the
#     Hamilton regression equation.
#
# CLARIFICATION ON "NEVER USE THE HP":
#   - The advice applies specifically to the HP-gap on the original (level) series:
#     this produces a spurious and unreliable business cycle — confirmed in Tutorial 2.
#   - The HP-trend applied to returns performs well: it is smooth, its passband lag is
#     smaller than Hamilton's, and — unlike Hamilton — it does not depend on the
#     chosen estimation window or sample size.
#     (There are both pros and cons to parameter-free filters; see Tutorial 2 for details.)


#----------------------------------------------------------------------------------------------
# 4.9 Out-of-Sample Application Including the Pandemic Period
#----------------------------------------------------------------------------------------------
# PURPOSE:
#   Apply all filters estimated on the 1990–2019 subsample unchanged (no re-estimation)
#   to the extended series including the pandemic period (2020 onwards).
#
# KEY INSIGHT:
#   The extreme pandemic outliers act as near-perfect impulses in the return series.
#   Because the filters are linear, each outlier triggers a response proportional to
#   the filter's impulse response function (sign-inverted filter coefficients).
#   This provides a rare and instructive opportunity to visualize filter dynamics directly
#   in the observed output — without any simulation.

# --- Load Extended Sample (1990 to Present, Including Pandemic) ---
y <- as.double(log(PAYEMS["1990/"]))
x <- diff(y)

# Visualize log-returns: the pandemic shock (early 2020) appears as extreme outliers
par(mfrow = c(1, 1))
ts.plot(x, main = "Log-Returns of PAYEMS: 1990–Present (Including Pandemic)")
abline(h = 0)

#----------------------
# Apply All Pre-Estimated Filters Out-of-Sample (No Re-Estimation)
#----------------------

# 1. SSA nowcast filter
SSA_out <- filter(x, SSA_filt_ham_diff_x, side = 1)

# 2. Hamilton filter
ham_out <- filter(x, ham_diff, side = 1)

# 3. SSA 6-month-ahead forecast filter
SSA_out_forecast_6 <- filter(x, SSA_filt_ham_diff_x_forecast[, 1], side = 1)

# 4. SSA 12-month-ahead forecast filter
SSA_out_forecast_12 <- filter(x, SSA_filt_ham_diff_x_forecast[, 2], side = 1)

# --- Visualize All Filter Outputs (Including Pandemic Period) ---
# Notable pattern: the pandemic dip (large negative impulse) is followed by a secondary
# positive peak whose magnitude and timing differ across filter designs.
# This is a direct consequence of the negative lobe in each filter's impulse response:
# the filters temporarily "mirror" the pandemic shock before returning to baseline.
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

# --- Revisit Filter Coefficients to Interpret the Pandemic Response ---
# The secondary post-pandemic peak in each filtered output directly reflects the shape
# of the corresponding filter's coefficients (sign-inverted impulse response).
# Comparing filter coefficients explains both:
#   (a) the magnitude of the secondary peak, and
#   (b) its location on the time axis (which differs across forecast horizons).
par(mfrow = c(1, 1))
ts.plot(SSA_filt_ham_diff_x_forecast,
        col  = c("orange", "darkgreen"),
        main = "SSA Filter Coefficients: Forecast Horizons vs. Nowcast",
        ylim = c(min(SSA_filt_ham_diff_x), max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x, col = "blue")
mtext(paste("Forecast horizon delta =", forecast_horizon[1]), col = "orange",    line = -1)
mtext(paste("Forecast horizon delta =", forecast_horizon[2]), col = "darkgreen", line = -2)
mtext("Nowcast (delta = 0)",                                   col = "blue",      line = -3)

# --- Interpretation: Forecast Filters Recover Faster from Extreme Outliers ---
#
# The secondary "phantom" peak in the filtered output is a filter artifact, not a real signal.
# Its timing and height are fully determined by the filter's impulse response structure.
#
# Advantage of longer forecast horizons in the presence of outliers:
#   - Forecast filters decay to zero more rapidly than the nowcast or Hamilton filter.
#   - Consequently, they "forget" extreme or singular observations (e.g., the pandemic shock)
#     more quickly, reducing the duration of filter-induced distortions.
#   - Example: the 12-month-ahead SSA forecast filter clears the pandemic outlier
#     approximately 1.5 years after its occurrence.
#   - By contrast, the SSA nowcast and Hamilton filter require an additional 10–11 months
#     to fully dissipate the impulse response, prolonging the phantom peak.
#
# Remark on the HP filter in this context:
#   - The classic HP-trend filter's impulse response decays faster than Hamilton's.
#   - Therefore, gross outliers such as the pandemic would produce less pronounced and
#     shorter-lived phantom peaks under HP — one practical advantage of HP over Hamilton
#     that is rarely discussed in the "never use HP" literature.


###################################################################################################
# SUMMARY OF EXAMPLES 1–4
###################################################################################################
#
# PROPOSED FRAMEWORK:
#   - A variant of the Hamilton filter (HF) was constructed onto which SSA can be grafted
#     via the level-to-differences transformation (ham_diff).
#   - A simple regression-based adjustment was proposed to map the SSA output back to
#     the original (classic) Hamilton cycle scale and level for direct comparison.
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
#     will be presented in a forthcoming tutorial.
###################################################################################################
















