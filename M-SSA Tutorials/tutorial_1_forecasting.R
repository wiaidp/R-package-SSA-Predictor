# ════════════════════════════════════════════════════════════════════
# TUTORIAL 1: SSA — HOLDING-TIME, SMOOTHNESS AND FORECASTING
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────
# This tutorial presents applications of the univariate SSA
# framework, with a primary focus on classic forecasting
# (signal extraction and nowcasting are addressed later).
#
# A unifying theme throughout is the holding-time (ht) concept:
# its role as a principled, systematic, and predictable means of
# controlling the smoothness of a forecast (or signal extraction
# filter).

# ── TUTORIAL OUTLINE ──────────────────────────────────────────────
#
#   Example 1 — Theoretical vs. empirical holding-time
#       → Relates the expected (theoretical) ht to the effective
#         (empirically measured) ht
#
#   Example 2 — Feasibility
#       → Examines the conditions and limits of feasible ht targets
#
#   Example 3 — Smoothness improvement
#       → Enhances the smoothness of a simple one-step ahead
#         predictor via SSA
#
#   Example 4 — Replication of the MSE predictor
#       → Demonstrates that SSA can exactly replicate the classical
#         MSE predictor as a special case
#
#   Example 5 — Flexible interface
#       → Explores the SSA interface by interchanging the roles of
#         the data-generating process and the target filter
#
#   Example 6 — Unsmoothing
#       → SSA is configured to generate more zero-crossings than
#         the benchmark predictor, i.e., a deliberate reduction
#         in smoothness
#
#   Example 7 — Model misspecification
#       → Analyzes cases where expected and empirical ht diverge,
#         and shows how simple adjustments can resolve the mismatch
#
#   Example 8 — HP filter replication
#       → Replicates Hodrick-Prescott filter designs via SSA
#         → See Wildi, M. (2024)
#            https://doi.org/10.1007/s41549-024-00097-5

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

rm(list=ls())

library(xts)
# Load the library mFilter
# HP and BK filters
library(mFilter)

# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


# ================================================================
# Example 1
# ================================================================

# Illustrate the holding time concept; see Wildi (2024, 2026a) for theoretical 
# background.

# Generate a realization of length 12,000 from an AR(1) process.
# A long series is required to obtain reliable empirical estimates.
len <- 12000
a1  <- 0.8
set.seed(1)
x <- arima.sim(n = len, list(ar = a1))

# Plot the simulated time series.
ts.plot(x)

# Inspect the autocorrelation and partial autocorrelation functions
# to verify the expected structure of an AR(1) process.
acf(x)
acf(x, type = "partial")

# Fit an AR(1) model to the simulated data.
ar_obj <- arima(x, order = c(1, 0, 0))

# Check model diagnostics; residuals should resemble white noise.
tsdiag(ar_obj)


# ── Holding Time ──────────────────────────────────────────────────────────────

# Compute the empirical holding time, defined as the average number of
# time steps between consecutive sign changes (zero-crossings) of the series.
empirical_ht <- len / length(which(sign(x[2:len]) != sign(x[1:(len - 1)])))
empirical_ht

# The same calculation is available via a dedicated convenience function.
empirical_ht <- compute_empirical_ht_func(x)
empirical_ht

# Compute the theoretical holding time using the exact closed-form expression
# from the cited literature (Section 2).
# This requires the MA(∞) representation of the process; either the true
# parameter a1 or its sample estimate may be used.
# ARMAtoMA() computes the MA filter coefficients for any stationary ARMA model.
xi <- c(1, ARMAtoMA(ar = a1, lag.max = len - 1))

# Pass the MA filter coefficients to compute the theoretical holding time.
# Note: an AR(1) process admits an exact MA(∞) representation, so the full
# sequence of filter coefficients characterises the process completely.
ht_obj <- compute_holding_time_func(xi)

# Theoretical holding time of the AR(1) process.
# Compare this value with the empirical estimate computed above.
ht_obj$ht

# The function also returns the lag-one autocorrelation of the filtered output,
# which equals a1 for a pure AR(1) process.
ht_obj$rho_ff1

# If the lag-one ACF is known, the holding time can be computed directly
# from that single quantity.
rho <- a1
compute_holding_time_from_rho_func(rho)

# Conversely, if the holding time is known, the corresponding lag-one ACF
# can be recovered by inverting the relationship.
ht <- ht_obj$ht
compute_rho_from_ht(ht)


# ── Finite-Sample Analysis ────────────────────────────────────────────────────

# Repeat the analysis on a much shorter realization (n = 100) of the same
# AR(1) process to examine the effect of finite-sample variability.
len <- 100
a1  <- 0.8
set.seed(1)
x <- arima.sim(n = len, list(ar = a1))

# Plot the short series.
ts.plot(x)

# ACF and PACF for the short series — compare with the long-series counterparts.
acf(x)
acf(x, type = "partial")

# Fit an AR(1) model to the short series.
ar_obj <- arima(x, order = c(1, 0, 0))

# Check model diagnostics for the short-series fit.
tsdiag(ar_obj)

# Extract the estimated AR(1) coefficient.
ahat <- ar_obj$coef["ar1"]
rho  <- ahat

# True holding time based on the known population parameter a1.
compute_holding_time_from_rho_func(a1)

# Estimated holding time based on the sample estimate of the lag-one ACF.
compute_holding_time_from_rho_func(rho)

# Empirical holding time based on observed zero-crossings in the short series.
compute_empirical_ht_func(x)

# All three estimates converge to the true value as the sample size grows.
# The discrepancies visible here are due to finite-sample estimation error.

# Notes on sampling error (see also Tables 3 and 4 in Wildi, 2024):
#   - In practice, sampling error has little impact on conclusions because
#     it largely cancels when performance is evaluated in relative terms.
#   - SSA focuses on relative performance — i.e., improvement over a chosen
#     benchmark — rather than on the absolute value of any criterion.


# ================================================================
# Example 2: Infeasible Settings  
# ================================================================
#
# A MA filter of length L imposes an upper bound on the achievable holding
# time (HT): the HT of the filtered output cannot be made arbitrarily large
# for a fixed L (see Wildi, 2024, 2026a).
# Requesting an HT that exceeds this bound leads to an infeasible
# specification. This example illustrates such a case and its consequences.


# ── Setup ────────────────────────────────────────────────────────────────────
#
# We retain the AR(1) process and its MA(∞) representation from Example 1.
len <- 120
xi  <- c(1, ARMAtoMA(ar = a1, lag.max = len - 1))


# ── Filter Length ─────────────────────────────────────────────────────────────
#
# Proposition 3 in Wildi (2024) establishes that the filter length L must be
# large enough to accommodate the desired HT. Specifically, the imposed HT
# must not exceed the maximum achievable HT of a MA(L) filter, which equals
# L + 1.
# Here, L is deliberately chosen too small in order to trigger the
# inconsistency and observe its effect.
L <- 5


# ── Target Specification: Identity Filter ────────────────────────────────────
#
# The target is set to the identity filter for simplicity.
# This choice will be explained in detail in subsequent examples.
gammak_generic <- 1


# ── Forecast Horizon ──────────────────────────────────────────────────────────
#
# One-step-ahead forecasting is used throughout this example.
forecast_horizon <- 1


# ── Holding-Time Constraint ───────────────────────────────────────────────────
#
# The HT value below is intentionally set larger than what a filter of length
# L = 5 can achieve (the max attainable HT is L+1=6 which is smaller than 7), 
# thereby creating an infeasible specification.
# See Proposition 3 in Wildi (2024) for the formal statement of this bound.
ht <- 7


# ── Converting HT to Lag-One ACF ─────────────────────────────────────────────
#
# SSA_func() requires the lag-one autocorrelation (rho1) as input rather than
# the HT directly. The function compute_rho_from_ht() performs this conversion.
rho1 <- compute_rho_from_ht(ht)
rho1


# ── Maximum Achievable Lag-One ACF for a MA(L) Filter ────────────────────────
#
# rhomax_func(L) returns the largest lag-one ACF that a MA filter of length L
# can produce (see Proposition 3 in Wildi, 2024).
# If rhomax_func(L) < rho1, no valid SSA solution exists for the chosen L,
# because the filter is too short to meet the imposed smoothness constraint.
rhomax_func(L)


# ── SSA Optimisation ──────────────────────────────────────────────────────────
#
# Because rho1 exceeds rhomax_func(L) in this example, the call below will
# raise an error, confirming that the specification is infeasible.
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)


# ── Note: Data vs. Model Specification ───────────────────────────────────────
#
# The observed series x_t is never passed directly to SSA_func().
# All information about the data-generating process is encoded in xi:
#   - xi == NULL  →  white noise is assumed as the input process.
#   - xi != NULL  →  xi contains the MA filter coefficients from the Wold
#                    decomposition of x_t, fully characterising its dynamics.

# ================================================================
# Resolving the Inconsistency: Two Options
# ================================================================
#
# Returning to the infeasible specification from Example 2, the problem
# can be resolved by either:
#   (a) Increasing L so that the filter is long enough to support the
#       imposed HT, or
#   (b) Decreasing the imposed HT so that it falls within the bound
#       determined by the current L.
#
# Option (a) is demonstrated first; option (b) follows immediately after.


# ── Option (a): Increase the Filter Length ───────────────────────────────────

# Filter length increased from 5 to 15 to accommodate the imposed HT of 7.
L <- 15

# Target set to the identity filter; details are provided in later examples.
gammak_generic <- 1

# One-step-ahead forecast horizon.
forecast_horizon <- 1

# HT is kept at the same value as in the infeasible example.
ht <- 7

# Convert HT to the lag-one ACF, which is the format required by SSA_func().
rho1 <- compute_rho_from_ht(ht)
rho1

# Verify feasibility: rhomax_func(L) must exceed rho1 for a solution to exist.
# With L = 15, rhomax_func(L) > rho1, so the specification is now consistent
# and a valid SSA solution can be obtained.
rhomax_func(L)

# Run the SSA optimisation with the revised (feasible) settings.
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# SSA_obj$ssa_x contains the optimal filter coefficients of length L = 15.
# These weights are applied directly to the observed series x_t.
# For a full description of the return values of SSA_func(), see Tutorial 0.3.
SSA_obj$ssa_x


# ── Option (b): Decrease the Holding-Time Constraint ─────────────────────────

# Revert to the original (short) filter length.
L <- 5

# Reduce the HT to a value that is attainable by a filter of length L = 5.
# An HT of 4 satisfies the bound imposed by Proposition 3 in Wildi (2024).
ht <- 4

# Convert the revised HT to the corresponding lag-one ACF.
rho1 <- compute_rho_from_ht(ht)
rho1

# Confirm feasibility: rhomax_func(L) should now exceed rho1.
rhomax_func(L)

# Run the SSA optimisation with the reduced HT and short filter.
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Optimal filter coefficients for the feasible short-filter specification.
SSA_obj$ssa_x


# ── General Remark: Choosing L Relative to the HT Constraint ─────────────────
#
# When rho1 is close to rhomax(L), the SSA optimisation may produce a
# degenerate or counterintuitive filter. This is a boundary effect: as rho1
# approaches rhomax(L), the feasible solution space contracts, leaving little
# room for meaningful optimisation.
#
# Recommendation: choose L large enough relative to the imposed HT so that
# rho1 remains comfortably within the feasible range.
#
# Practical rule of thumb:
#
#                         L >= 2 * HT
#
# Adhering to this guideline ensures that rho1 lies well below rhomax(L),
# producing well-behaved and interpretable filter coefficients.


# ================================================================
# Example 3: One-Step-Ahead Forecasting
# ================================================================
#
# This example demonstrates one-step-ahead forecasting for the AR(1) process
# introduced in the previous examples. The SSA filter is optimised under a
# smoothness constraint and its performance is verified empirically.


# ── Data-Generating Process ───────────────────────────────────────────────────

# AR(1) coefficient (retained from previous examples).
a1 <- 0.8

# Compute the MA(∞) representation (Wold decomposition) of the AR(1) process,
# truncated at lag len - 1. This encodes all information about the
# data-generating process that SSA_func() requires; no observed sample is
# passed directly.
# In practice, a1 may be replaced by a finite-sample estimate without
# materially affecting the results.
len <- 100
xi  <- c(1, ARMAtoMA(ar = a1, lag.max = len - 1))


# ── SSA Settings ──────────────────────────────────────────────────────────────

# Holding-Time Constraint
# The target HT is set larger than the native HT of the AR(1) process
# (see Example 1), so that the SSA output is smoother than the raw series
# (i.e., it crosses zero less frequently).
ht <- 6

# Convert HT to the lag-one ACF, which is the input format required by
# SSA_func().
rho1 <- compute_rho_from_ht(ht)

# Filter Length
# L should satisfy two conditions simultaneously:
#   - Large enough to capture the relevant filter dynamics.
#     The practical rule of thumb L >= 2 * HT (see Example 2) is applied here.
#   - Small enough to remain well below the available sample length.
# A chosen L is adequate when the filter coefficients decay sufficiently
# close to zero at the far lags.
# Note: a larger L does not cause overfitting of the SSA filter itself;
# overfitting can only arise from an overfitted Wold decomposition xi.
L <- 20

# Target Specification
# For forecasting, the target is the identity filter (gammak_generic = 1):
# SSA seeks a causal real-time filter whose output best approximates x_t
# shifted forward by forecast_horizon steps.
# In signal extraction settings the target would instead be a non-trivial
# filter applied to x_t (e.g., a lowpass filter) — see the examples below.
gammak_generic <- 1

# One-step-ahead forecast horizon.
forecast_horizon <- 1

# Run the SSA optimisation with the settings defined above.
# SSA_func() checks whether the length of the supplied target matches L.
# If not, a warning is issued and the target is automatically zero-padded
# to length L before optimisation proceeds.
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

ssa_x <- SSA_obj$ssa_x

# Plot the optimised SSA filter coefficients to inspect their shape and decay.
ts.plot(ssa_x)


# ── Performance Checks ────────────────────────────────────────────────────────

# 1. Holding-Time Verification
#    Apply the optimised filter to a long AR(1) realisation and compare the
#    resulting empirical HT with the imposed constraint.
#    Both values should agree as the sample size grows large.

len <- 100000
set.seed(1)
x <- arima.sim(n = len, list(ar = a1))

# Apply the SSA filter to the long series (one-sided, i.e., causal).
yhat <- filter(x, ssa_x, sides = 1)

# Empirical HT of the filtered output.
empirical_ht <- compute_empirical_ht_func(yhat)
empirical_ht

# The imposed HT constraint for comparison.
ht


# ── 2. Optimisation Convergence Check ────────────────────────────────────────
#
# Compare the lag-one ACF of the optimised filter output (crit_rhoyy) with
# the target rho1. Close agreement indicates that the optimisation converged
# to the global maximum.
# If a substantial discrepancy is observed, increase split_grid (the number
# of grid iterations; default = 20, which is sufficient for nearly all
# standard applications).
SSA_obj$crit_rhoyy
rho1


# ── 3. SSA Criterion Values ───────────────────────────────────────────────────
#
# Two equivalent criteria are available (see Proposition 4, Wildi 2004 JBCY).
# Both yield the same optimal filter but measure performance from different
# perspectives.

# 3.1 Criterion 1: Correlation with the One-Step-Ahead MSE Forecast
#     crit_rhoyz measures the correlation between the SSA filter output and
#     the minimum MSE predictor of x_t.
SSA_obj$crit_rhoyz

# Empirical verification: construct the one-step-ahead MSE predictor and
# compute its sample correlation with the filter output.
MSE_forecast <- a1 * x
cor(yhat, MSE_forecast, use = "pairwise.complete.obs")


# 3.2 Criterion 2: Correlation with the Effective Target
#     crit_rhoy_target measures the correlation between the SSA filter output
#     and the series shifted forward by forecast_horizon steps (i.e., the
#     future observations that the filter is trying to track).
SSA_obj$crit_rhoy_target

# Empirical verification: compute the sample correlation between the filter
# output and the forward-shifted series.
cor(yhat,
    c(x[(1 + forecast_horizon):len], rep(NA, forecast_horizon)),
    use = "pairwise.complete.obs")


# ── Interpretation of the Two Criteria ───────────────────────────────────────
#
# - Sample correlations converge to their respective criterion values as the
#   sample size increases, consistent with asymptotic theory.
#
# - Both criteria are equivalent in the sense that they identify the same
#   optimal SSA filter (see Proposition 4, Wildi 2004 JBCY).
#
# - Criterion 1 yields a higher correlation because the MSE predictor is
#   causal — it conditions only on past and present observations.
#
# - Criterion 2 yields a lower correlation because the forward-shifted series
#   implicitly requires knowledge of future observations (here: the value one
#   step ahead), which are unavailable to the causal filter.
#
# - When the HT imposed on SSA matches the native HT of the MSE solution,
#   SSA exactly replicates the MSE predictor — see Tutorial 0.3 and
#   Example 4 below for a detailed illustration.




#================================================================
# Example 4: Replicating the MSE Solution via SSA
#================================================================
#
# This example mirrors Example 3, with one key modification:
# we set rho1 = a1 in the holding-time constraint, so that SSA
# exactly replicates the lag-one ACF — and therefore the filter — of the MSE solution.
#
# Key points:
#   - Setting rho1 = a1 matches the lag-one ACF of the one-step ahead MSE filter
#   - This causes SSA to exactly replicate the MSE solution

# AR(1) coefficient
a1 <- 0.8

# Wold Decomposition
#   Truncated MA(inf) representation of the AR(1) process (see Example 3).
#   Reminder: SSA_func does not require observed data — all process information
#   is encoded in xi.
len <- 100
xi <- c(1, ARMAtoMA(ar = a1, lag.max = len - 1))

# Holding-Time Constraint
#   For an AR(1) process, the lag-one ACF of the one-step ahead MSE filter is
#   exactly a1. Setting rho1 = a1 therefore instructs SSA to match the smoothness
#   of the MSE solution, which causes SSA to replicate it exactly.
rho1 <- a1

# Filter Length
L <- 20

# Target Specification: Identity Filter
#   As in Example 3, the target is the identity filter (gammak_generic = 1),
#   appropriate for forecasting the original series x_t.
gammak_generic <- 1

# Forecast Horizon: one-step ahead
forecast_horizon <- 1

# SSA Optimization
#   Two warnings may be issued by SSA_func:
#     Warning 1: the target filter gammak_generic is shorter than L —
#                this is expected and handled automatically via zero-padding.
#     Warning 2: the SSA solution is very close to the MSE solution after
#                optimization — this is the intended outcome in this example.
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

#----------------------------------------------------------------
# Results
#----------------------------------------------------------------

ssa_x <- SSA_obj$ssa_x

# Plot the optimized SSA filter
#   The resulting filter should closely replicate the MSE predictor.
ts.plot(ssa_x)

# Convolved Filter (applied to epsilon_t)
#   ssa_eps is the SSA filter prior to deconvolution — i.e., expressed in terms
#   of the white noise innovations epsilon_t rather than x_t.
#   For an AR(1) process with coefficient a1, the optimal filter coefficients
#   follow a geometric decay pattern governed by a1.
ssa_eps <- SSA_obj$ssa_eps
ts.plot(ssa_eps)

# Verify the geometric decay rate
#   Successive ratios of ssa_eps should equal a1, confirming that the
#   convolved filter replicates the exponential decay of the MSE solution.
ssa_eps[2:L] / ssa_eps[1:(L - 1)]




#================================================================
# Example 5: Exchanging the Roles of xi and gammak_target
#================================================================
#
# Background:
#   In the previous examples, we assumed:
#     - x_t  follows an AR(1) process  (xi encodes the Wold decomposition)
#     - z_t  is the identity target     (z_{t+delta} = x_t shifted forward)
#
#   An equivalent alternative formulation is:
#     - x_t  = epsilon_t  (white noise input; xi = NULL)
#     - z_t  = AR(1) filter applied to x_t  (non-trivial target)
#
#   Both formulations represent the same underlying forecasting problem.
#   This example demonstrates how to implement the alternative design.
#   See Wildi, M. (2024)

#----------------------------------------------------------------
# SSA Settings
#----------------------------------------------------------------

# AR(1) coefficient
a1 <- 0.8

# Wold Decomposition
#   Since x_t = epsilon_t (white noise), the Wold decomposition is trivial.
#   Setting xi = NULL instructs SSA_func to assume white noise input.
xi <- NULL

# Holding-Time Constraint
#   As in Example 4, we match the lag-one ACF of the one-step ahead MSE filter,
#   which equals a1 for an AR(1) process.
rho1 <- 0.8

# Filter Length
L <- 20

# Target Specification
#   We now supply the AR(1) filter (via its MA inversion) as the target.
#   This replaces the identity target used in the previous examples.
#   Note: SSA_func checks whether the target length matches L. If not, a warning
#   is issued and the target is automatically zero-padded to length L.
gammak_generic <- c(1, ARMAtoMA(ar = a1, lag.max = len - 1))

# Forecast Horizon: one-step ahead
forecast_horizon <- 1

# SSA Optimization
#   Standard call supplying xi explicitly:
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)
#   Equivalent call omitting xi — when xi is not supplied, white noise is assumed
#   by default (xi = NULL):
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1)

#----------------------------------------------------------------
# Results
#----------------------------------------------------------------

ssa_x <- SSA_obj$ssa_x

# Plot the optimized SSA filter
#   Unlike Example 4, where ssa_x was simply a1 * identity (a scalar multiple
#   of the unit vector), ssa_x now represents the full finite-sample MA forecast
#   filter, with coefficients decaying geometrically.
ts.plot(ssa_x)

# Verify the geometric decay rate of ssa_x
#   Successive ratios should converge to a1, confirming exponential decay.
ssa_x[2:L] / ssa_x[1:(L - 1)]

# Convolved Filter (applied to epsilon_t)
#   Since x_t = epsilon_t in this formulation, ssa_eps is identical to ssa_x —
#   no deconvolution step is required or applied.
ssa_eps <- SSA_obj$ssa_eps

# Verify the geometric decay rate of ssa_eps
#   As with ssa_x, successive ratios should equal a1.
#   Note: in Example 4, ssa_eps differed from ssa_x because x_t followed an
#   AR(1) process (xi != NULL), requiring an explicit deconvolution step.
ssa_eps[2:L] / ssa_eps[1:(L - 1)]




#================================================================
# Example 6: Unsmoothing — Imposing a Smaller Holding-Time Constraint
#================================================================
#
# This example mirrors Example 3, but imposes a holding-time (HT) smaller than
# the native HT of the AR(1) process. As a result, the SSA filter must actively
# introduce additional zero-crossings to meet the constraint — the opposite of
# the typical smoothing use case.

#----------------------------------------------------------------
# SSA Settings
#----------------------------------------------------------------

# AR(1) coefficient
a1 <- 0.8

# Wold Decomposition
#   Truncated MA(inf) representation of the AR(1) process (see Example 3).
#   Reminder: SSA_func does not require observed data — all process information
#   is encoded in xi.
len <- 100
xi <- c(1, ARMAtoMA(ar = a1, lag.max = len - 1))

# Holding-Time Constraint
#   We deliberately impose an HT smaller than the native HT of the AR(1) filter,
#   forcing the SSA filter to generate additional zero-crossings (unsmoothing).
ht <- 3

# Lag-One ACF
#   SSA_func requires rho1 rather than HT directly; we convert accordingly.
rho1 <- compute_rho_from_ht(ht)

# Filter Length
L <- 20

# Target Specification: Identity Filter
#   As in Example 3, the target is the identity filter (gammak_generic = 1),
#   appropriate for forecasting the original series x_t.
gammak_generic <- 1

# Forecast Horizon: one-step ahead
forecast_horizon <- 1

# SSA Optimization
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

#----------------------------------------------------------------
# Results
#----------------------------------------------------------------

ssa_x <- SSA_obj$ssa_x

# Plot the optimized SSA filter
#   The alternating sign pattern of the coefficients reflects the additional
#   zero-crossings introduced to satisfy the smaller HT constraint.
ts.plot(ssa_x)

#----------------------------------------------------------------
# Performance Checks
#----------------------------------------------------------------

# 1. Holding-Time Verification
#   We verify the imposed HT empirically using a long simulated AR(1) series.
len <- 100000
set.seed(1)
x <- arima.sim(n = len, list(ar = a1))

# Apply the optimized SSA filter to the simulated series
yhat <- filter(x, ssa_x, sides = 1)

# Empirical HT of the filter output — should match the imposed constraint
empirical_ht <- compute_empirical_ht_func(yhat)
empirical_ht
ht

#----------------------------------------------------------------
# 2. Optimization Convergence Check
#   The lag-one ACF of the optimized filter (crit_rhoyy) should match rho1.
#   Agreement confirms convergence to the global maximum.
SSA_obj$crit_rhoyy
rho1

#----------------------------------------------------------------
# 3. SSA Criterion Values
#   Two equivalent criteria are available (see Proposition 4 in the JBCY paper).
#   Both yield the same optimal filter, but measure performance against
#   different targets:
#     a. Correlation with the MSE predictor: how well does SSA approximate
#        the MSE solution, subject to the HT constraint?
#     b. Correlation with the effective target: how well does SSA approximate
#        the forward-shifted series (the quantity being forecast)?

# 3a. Theoretical correlation with the one-step ahead MSE forecast
SSA_obj$crit_rhoyz
# Empirical verification: compute the MSE predictor and correlate with filter output
MSE_forecast <- a1 * x
cor(yhat, MSE_forecast, use = 'pairwise.complete.obs')

# 3b. Theoretical correlation with the effective target (series shifted one step forward)
SSA_obj$crit_rhoy_target
# Empirical verification: correlate filter output with the forward-shifted series
cor(yhat, c(x[2:len], x[len]), use = 'pairwise.complete.obs')



# ==================================================================================
# Example 7: Holding Time — Strict Interpretation, Misspecification, and
#             Smoothness Hyperparameter
# ==================================================================================
#
# The holding time (HT) measures how long the output of a zero-mean filter tends
# to remain on the same side of zero, i.e., the persistence of its sign.
# Two versions are compared throughout this example:
#   - Analytical HT : derived from the filter coefficients under a white noise
#                     input assumption.
#   - Empirical HT  : computed directly from the filtered output series by
#                     counting zero-crossings.


# ── 7.1 Correctly Specified Model: x_t = ε_t (White Noise Input) ─────────────
#
# When the input is white noise, the analytical HT and the empirical HT should
# agree up to finite-sample variation.

len <- 12000  # Length of the simulated time series.

# Define an equally weighted moving average (MA) filter of length L.
# Each coefficient equals 1/L, so the filter computes a simple rolling mean.
L <- 10
b <- rep(1/L, L)

set.seed(65)
x <- rnorm(len)  # Simulate a white noise input series.

# Apply the MA filter causally (one-sided convolution).
yhat <- filter(x, b, side = 1)
yhat <- na.exclude(yhat)  # Remove leading NAs introduced by the filter lag.

# Compare the two HT estimates.
# Any discrepancy is attributable to finite-sample variation, not model error.
compute_holding_time_func(b)$ht  # Analytical HT derived from filter coefficients.
compute_empirical_ht_func(yhat)  # Empirical HT counted from the filtered series.


# ── 7.2 Misspecified Model: x_t = ARMA(1,1) Process ─────────────────────────
#
# The analytical formula for HT assumes white noise input. When the input is
# autocorrelated (e.g., an ARMA process), this assumption is violated and the
# two HT estimates will no longer agree.

L  <- 10
b  <- rep(1/L, L)  # Same equally weighted MA filter as in Section 7.1.
a1 <- 0.4          # AR(1) coefficient of the input process.
b1 <- 0.3          # MA(1) coefficient of the input process.

set.seed(65)
x <- arima.sim(n = len, model = list(ar = a1, ma = b1))  # Simulate ARMA(1,1) input.

# Apply the MA filter to the autocorrelated series.
yhat <- filter(x, b, side = 1)
yhat <- na.exclude(yhat)

# The analytical HT is now incorrect because compute_holding_time_func() assumes
# white noise input, which is violated here.
compute_holding_time_func(b)$ht  # Analytical HT (white noise assumption — misspecified).
compute_empirical_ht_func(yhat)  # Empirical HT from the ARMA-driven filtered series.


# ── Reconciling Analytical and Empirical HT Under Misspecification ────────────
#
# To resolve the mismatch, we identify the filter that receives white noise ε_t
# as input but produces output identical to applying b to x_t.
#
# By the Wold decomposition, any stationary ARMA process can be written as an
# infinite-order MA:
#
#   x_t = ξ(L) · ε_t,   where ξ contains the MA inversion coefficients.
#
# Applying filter b to x_t is therefore equivalent to applying the convolution
# (b ∗ ξ) directly to ε_t:
#
#   b(L) · x_t  =  b(L) · ξ(L) · ε_t  =  [b ∗ ξ](L) · ε_t.
#
# Passing this convolved filter to compute_holding_time_func() recovers the
# correct analytical HT, since the input to that filter is white noise by
# construction.

# Step 1: Compute the MA inversion (Wold representation) of the ARMA(1,1) process.
xi <- c(1, ARMAtoMA(ar = a1, ma = b1, lag.max = len - 1))

# Step 2: Convolve the Wold filter ξ with the MA filter b.
# The resulting filter ssa_eps maps ε_t to the same output as b applied to x_t.
ssa_eps <- conv_two_filt_func(xi, b)$conv

# Visual check: plot the original filter b alongside the convolved filter ssa_eps.
# The convolved filter is broader, reflecting the additional autocorrelation
# structure inherited from the ARMA input process.
ts.plot(ssa_eps[1:30], col = "red",
        main = "Original filter b (black) vs. convolved filter ssa_eps (red)")
lines(b, col = "black")

# Step 3: Compute the analytical HT using the corrected convolved filter.
# The result should now match the empirical HT from the ARMA-driven series.
compute_holding_time_func(ssa_eps)$ht  # Corrected analytical HT.
compute_empirical_ht_func(yhat)        # Empirical HT for comparison.


# ── Numerical Verification ────────────────────────────────────────────────────
#
# Confirm algebraically that applying b to x_t is equivalent to applying ssa_eps
# to ε_t by comparing the two filtered series directly.

# Simulate the same ARMA(1,1) process manually to retain direct access to ε_t.
set.seed(43)
x <- eps <- rnorm(len)
for (i in 2:len)
  x[i] <- a1 * x[i - 1] + eps[i] + b1 * eps[i - 1]

# Filter x_t with the original MA filter b.
yhat_x   <- filter(x,   b,             side = 1)

# Filter ε_t with the first 30 coefficients of the convolved filter ssa_eps.
# Truncation at lag 30 is a practical approximation; the remaining coefficients
# are negligible in magnitude.
yhat_eps <- filter(eps, ssa_eps[1:30], side = 1)

# Align both series by removing NAs and plot over the first 1000 observations.
# The two series should overlap almost perfectly, confirming the equivalence.
mplot <- na.exclude(cbind(yhat_x, yhat_eps))

par(mfrow = c(1, 1))
ts.plot(mplot[1:1000, ], lty = 1:2,
        main = paste("b applied to x_t (solid) vs. convolved filter applied",
                     "to epsilon_t (dashed): series overlap"))


# ── 7.3 Misspecified Model: x_t = ARMA(1,1) + μ (Non-Zero Mean) ──────────────
#
# We extend Section 7.2 by adding a non-zero mean μ to the input series.
# This has an important implication for the HT: shifting the distribution
# upward moves the zero-crossing threshold away from the centre of mass,
# reducing the frequency of zero-crossings and thereby *increasing* the
# empirical HT relative to the zero-mean case in Section 7.2.

L  <- 10
b  <- rep(1/L, L)  # Equally weighted MA filter of length L.
a1 <- 0.4          # AR(1) coefficient of the ARMA(1,1) input process.
b1 <- 0.3          # MA(1) coefficient of the ARMA(1,1) input process.
mu <- 1            # Non-zero mean shift applied to the simulated series.

set.seed(655)
# Simulate an ARMA(1,1) process shifted upward by μ.
# The filtered output will oscillate around μ rather than zero.
x    <- mu + arima.sim(n = len, model = list(ar = a1, ma = b1))
yhat <- filter(x, b, side = 1)
yhat <- na.exclude(yhat)  # Remove leading NAs introduced by the filter lag.

# The discrepancy between the two estimates is now larger than in Section 7.2,
# because the mean shift reduces zero-crossings independently of the
# autocorrelation structure.
compute_holding_time_func(b)$ht  # Analytical HT (zero-mean white noise assumed — misspecified).
compute_empirical_ht_func(yhat)  # Empirical HT: inflated by both autocorrelation and mean shift.

# ── Key Remark ────────────────────────────────────────────────────────────────
#
# When μ ≠ 0, SSA controls the mean-crossing rate rather than the zero-crossing
# rate. For a zero-mean process the two notions coincide, but in general the
# relevant threshold is the process mean, not zero.


# ── Diagnostic Plot: Effect of the Mean Shift on Zero-Crossings ───────────────
#
# Plot the first 100 observations of x to illustrate how far the series sits
# above zero. Two reference lines are added:
#   - Red dashed line  at 0  : the zero-crossing threshold used in the HT formula.
#   - Blue dashed line at μ  : the true centre of the process.
# As μ increases, zero-crossings become rarer; for μ > 4 they may vanish
# entirely over a typical sample.

ts.plot(x[1:100], main = "Simulated ARMA(1,1) + μ series (first 100 observations)")
abline(h = 0,  col = "red",  lty = 2, lwd = 1.5)  # Zero line  — reference for HT calculation.
abline(h = mu, col = "blue", lty = 2, lwd = 1.5)  # Mean line  — true centre of the process.


# ── Reconciling Analytical and Empirical HT Under Mean-Shifted Misspecification ──
#
# Two corrections are required to restore agreement between the analytical and
# empirical HT:
#
#   1. Centre the data by subtracting μ, so that zero-crossings of the centred
#      series correspond to crossings of the process mean — which is what the
#      HT formula actually measures.
#
#   2. Correct for the ARMA autocorrelation structure by convolving the filter b
#      with the Wold MA inversion of x_t, exactly as in Section 7.2.
#
# After both corrections are applied, the analytical HT should match the
# empirical HT of the centred, filtered series up to sampling error.

# Step 1: Compute the MA inversion (Wold representation) of the ARMA(1,1) process.
xi <- c(1, ARMAtoMA(ar = a1, ma = b1, lag.max = len - 1))

# Step 2: Convolve ξ with the original filter b to obtain the corrected filter
# ssa_eps. This filter maps white noise ε_t to the same output as b applied to
# the centred series (x_t − μ).
ssa_eps <- conv_two_filt_func(xi, b)$conv

# Visual check: compare the original filter b (black) with the corrected filter
# ssa_eps (red). The convolved filter is broader, reflecting the additional
# autocorrelation memory inherited from the ARMA input process.
ts.plot(ssa_eps[1:30], col = "red",
        main = "Original filter b (black) vs. corrected filter ssa_eps (red)")
lines(b, col = "black")

# Analytical HT of the corrected filter.
# Note: this does not yet match the empirical HT of yhat computed in Section 7.3,
# because yhat was derived from the uncentred series x. Centring is applied in
# the numerical verification below.
compute_holding_time_func(ssa_eps)$ht


# ── Numerical Verification ────────────────────────────────────────────────────
#
# Confirm that applying b to the centred series (x_t − μ) is equivalent to
# applying ssa_eps directly to ε_t, by comparing the two filtered outputs.

# Simulate the same ARMA(1,1) process manually to retain direct access to ε_t.
set.seed(43)
y <- eps <- rnorm(len)
for (i in 2:len)
  y[i] <- a1 * y[i - 1] + eps[i] + b1 * eps[i - 1]

x <- y + mu  # Reintroduce the mean shift to recover the original series.

# Approach A: filter the centred series (x − mean(x)) with b.
# Subtracting mean(x) estimates and removes μ; any residual discrepancy
# relative to Approach B reflects finite-sample estimation error of μ.
yhat_x   <- filter(x - mean(x), b,             side = 1)

# Approach B: filter ε_t directly with the corrected filter ssa_eps.
# The filter is truncated at lag 30; coefficients beyond this lag are
# negligibly small for this ARMA specification and can be safely ignored.
yhat_eps <- filter(eps,          ssa_eps[1:30], side = 1)

# Align both series by removing NAs introduced by the one-sided filter lag.
mplot <- na.exclude(cbind(yhat_x, yhat_eps))

# Plot both filtered series over the first 1000 observations.
# The two lines should overlap almost perfectly, confirming the equivalence
# of the two filtering approaches.
ts.plot(mplot[1:1000, ], lty = 1:2,
        main = paste("b applied to centred x_t (solid) vs.",
                     "corrected filter applied to epsilon_t (dashed): series overlap"))

# Final check: the empirical HT of the centred filtered output (Approach A)
# should now agree with the analytical HT of the corrected filter ssa_eps
# (Approach B), up to finite-sample variation.
compute_empirical_ht_func(yhat_x)      # Empirical HT after centring.
compute_holding_time_func(ssa_eps)$ht  # Analytical HT of the corrected filter.


# ── Conclusions from Sections 7.1–7.3 ────────────────────────────────────────
#
# 1. The analytical HT formula is derived under the assumption that filter b is
#    applied to zero-mean white noise. Two forms of misspecification violate this:
#
#    (a) Autocorrelated input: if x_t follows an ARMA process, convolve b with
#        the Wold MA inversion ξ of x_t to obtain the correctly specified filter
#        before applying the HT formula.
#
#    (b) Non-zero mean: if x_t has mean μ ≠ 0, centre the data before filtering.
#        The HT then measures the mean duration between crossings of the μ-line
#        rather than the zero-line. Under misspecification this literal
#        interpretation is lost, but the smoothing effect remains valid
#        (see point 3 below).
#
# 2. When both corrections are applied simultaneously — centring and convolution
#    with the Wold filter — the analytical and empirical HTs agree up to
#    finite-sample variation.
#
# 3. Even under misspecification, increasing HT in the SSA optimisation always
#    produces a smoother filter output:
#      - The smoothing effect acts at the μ-line and at any other level.
#      - However, the HT value is biased relative to the true zero-crossing rate
#        of the uncentred series.
#
# 4. HT can therefore be interpreted as a smoothing hyperparameter that governs
#    noise suppression independently of whether the model is correctly specified.
#    Tutorials 2 onwards illustrate these practical smoothing effects in detail.




#================================================================
# Example 8
# Replicating SSA filters for business-cycle analysis
#================================================================
#-----------------------------------------------------------------------------
#
# Reference: Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
#
# Key modelling assumption:
#   SSA filters here assume white noise input (xi = NULL), meaning we do NOT
#   fit a parametric time-series model to the data. This is intentional:
#   it avoids over-fitting ("data mining") and keeps the filter design robust.
#
# Overview of what this example produces:
#   1. A suite of HP-filter designs (two-sided target, one-sided concurrent,
#      one-sided MSE-optimal, gap filters).
#   2. Two SSA filter sets derived from those HP designs:
#      a. A "smooth" SSA filter with an *increased* holding-time (ht = 12)
#         that generates ~40% fewer noisy zero-crossings than the classic HP.
#      b. A "fast" SSA filter that matches the holding-time of the classic
#         concurrent HP but forecasts 18 months ahead.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# 8.1 HP filter targets and hyperparameters
#-----------------------------------------------------------------------------

L              <- 201     # Filter length (number of coefficients); monthly data context
lambda_monthly <- 14400   # Standard HP smoothing parameter for monthly data
# (corresponds to lambda = 1600 for quarterly data, scaled by ~3^4)

# Compute the full set of HP-related filter designs in one call.
# The returned object contains both the two-sided ideal target and several
# one-sided (causal) approximations used as SSA targets below.
HP_obj <- HP_target_mse_modified_gap(L, lambda_monthly)

# Two-sided (symmetric) HP trend filter — the ideal smoother, not realisable
# in real time because it uses future observations.  Used as the benchmark target.
target <- HP_obj$target
# Note that the filter is effectively one-sided (must be shifted forward to become acausal two-sided)
ts.plot(target,main="Causal Version of two-sided HP-trend")

# One-sided gap filter: approximates (data - HP trend) in real time.
# Positive values indicate the series is above the long-run trend (expansion);
# negative values indicate contraction.
hp_gap <- HP_obj$hp_gap
ts.plot(hp_gap,main="One-sided HP-gap")
# Modified one-sided gap filter designed to operate on *first differences* of
# the data.  When applied to differenced data it produces output identical to
# hp_gap applied to the original (level) series.  Useful when the data are
# differenced for stationarity before filtering.
modified_hp_gap <- HP_obj$modified_hp_gap

# Classic one-sided (concurrent) HP trend filter.
# This is the MSE-optimal real-time estimate of the two-sided target *if* the
# data follow an ARIMA(0,2,2) process — the model implicitly assumed by the HP filter.
hp_trend <- HP_obj$hp_trend
ts.plot(hp_trend,main="Classical causal one-sided HP: optimal if data is an ARIMA(0,2,2)")
# Alternative one-sided HP trend based on direct (least-squares) truncation of
# the two-sided symmetric filter.  This is MSE-optimal under the white noise
# assumption (consistent with xi = NULL used in SSA below).
hp_mse <- HP_obj$hp_mse
ts.plot(hp_mse,main="Optimal Causal one-sided HP if data is white noise")

# Any of the above designs can serve as an SSA target; see Tutorial 2.

#-----------------------------------------------------------------------------
# 8.2 SSA filter design — smooth variant with increased holding-time (ht = 12)
#
# Problem with the classic concurrent HP trend (hp_trend):
#   The filter exhibits unwanted noise leakage, producing spurious zero-crossings
#   (false alarms about turning points) that are undesirable in practice.
#
# Solution:
#   Ask SSA to approximate hp_trend while imposing a *larger* holding-time.
#   A higher ht forces the filter output to stay on the same side of zero longer,
#   effectively suppressing high-frequency noise and reducing false alarms.
#-----------------------------------------------------------------------------

# Holding-time of the classic concurrent HP trend (baseline reference)
compute_holding_time_func(hp_trend)$ht


# Target holding-time for the smooth SSA variant.
# Setting ht = 12 (months) means the filter output is expected to cross zero
# roughly once every 12 months on average under white noise.
ht <- 12

# Ratio of target ht to the HP trend's own ht: shows the degree of extra smoothing.
# A ratio > 1 means fewer crossings; ~1.56 implies roughly 40% fewer noisy alarms.
ht / compute_holding_time_func(hp_trend)$ht

# Convert the target holding-time to the equivalent autocorrelation parameter rho1
# used internally by SSA_func to enforce the ht constraint.
rho1 <- compute_rho_from_ht(ht)

# Forecast horizons for which SSA will compute optimal filters simultaneously.
# horizon = 0  → concurrent (nowcast) filter
# horizon = 18 → 18-month-ahead forecast filter
# SSA_func accepts a vector and returns one optimised filter per horizon.
forecast_horizon_vec <- c(0, 18)

# Choose the MSE-optimal truncated HP trend as the SSA approximation target.
# Using hp_mse (rather than hp_trend) is consistent with the white noise
# assumption (xi = NULL) adopted throughout this example.
gammak_generic <- hp_mse

# Compute SSA filters for all horizons in forecast_horizon_vec.
# Because xi is omitted (xi = NULL by default), SSA assumes white noise input,
# so ssa_x and ssa_eps are identical — no MA-inversion correction is needed.
SSA_obj  <- SSA_func(L, forecast_horizon_vec, gammak_generic, rho1)
ssa_eps  <- SSA_obj$ssa_eps

# Label each column to identify its holding-time and forecast horizon clearly.
colnames(ssa_eps) <- paste0("SSA(", round(ht, 2), ",", forecast_horizon_vec, ")")

#-----------------------------------------------------------------------------
# 8.3 SSA filter design — fast variant matching HP holding-time, 18-month forecast
#
# For comparison we also compute an SSA filter that:
#   - Matches the holding-time of the classic concurrent HP trend (no extra smoothing)
#   - Forecasts 18 months ahead
# This allows a direct comparison: same noise level as HP, but with a longer
# forecast horizon.
#-----------------------------------------------------------------------------

# Holding-time of the classic concurrent HP trend (used as the ht constraint here)
ht_short     <- compute_holding_time_func(hp_trend)$ht

# Convert this holding-time to its equivalent rho1 parameter for SSA_func
rho1_short   <- compute_rho_from_ht(ht_short)

# We only need the 18-month-ahead forecast for this variant
forecast_short <- forecast_horizon_vec[2]
  
# Compute the fast SSA forecast filter with HP-equivalent holding-time
SSA_obj  <- SSA_func(L, forecast_short, gammak_generic, rho1_short)
ssa_eps1 <- SSA_obj$ssa_eps

# Label the filter column consistently with the smooth variant above
colnames(ssa_eps1) <- paste0("SSA(", round(ht_short, 2), ",", forecast_horizon_vec[2], ")")

# Combine all SSA filters (smooth nowcast, smooth 18m forecast, fast 18m forecast)
# into a single matrix for convenient plotting and comparison in subsequent steps.
ssa_eps <- cbind(ssa_eps, ssa_eps1)


#-----------------------------------------------------------------------------
# 8.4 Visualisation of HP and SSA filter coefficients
#
# This plot replicates Figure 9 in:
#   Wildi, M. (2024) 
#
# Left panel:  HP filter family — compares the two-sided symmetric target,
#              the classic one-sided HP trend, and both gap filter variants.
# Right panel: SSA filter family — compares the concurrent (nowcast) and
#              forecast SSA filters at the smooth and fast holding-time settings.
#
# All filter coefficient vectors are scaled to unit length 
# (unit variance when fed with standardized white noise) for visual
# comparability (center = FALSE, scale = TRUE in scale()).
# The x-axis shows the lag index (0 = most recent observation).
#-----------------------------------------------------------------------------

# Colour scheme ---------------------------------------------------------------
# HP filter family: warm tones (brown, red)
colo_hp_all <- c("brown", "red")
# SSA filter family: cool/distinct tones (orange, blue, violet)
colo_SSA    <- c("orange", "blue", "violet")
# Combined palette (used if all filters are plotted together)
colo_all    <- c(colo_hp_all, colo_SSA)

#-----------------------------------------------------------------------------
# Left panel: HP filter family
#-----------------------------------------------------------------------------

par(mfrow = c(1, 2))   # Side-by-side layout: HP filters (left), SSA filters (right)

# Stack the four HP-related filters column-wise and scale each to unit maximum.
# Scaling makes the shape (lag structure) comparable regardless of filter magnitude.
mplot <- scale(cbind(hp_trend, target, hp_gap, modified_hp_gap),
               center = FALSE, scale = TRUE)
colnames(mplot) <- c("HP trend",           # Classic one-sided concurrent HP trend
                     "Target symmetric",   # Two-sided ideal target (non-causal benchmark)
                     "HP-gap (original)",  # One-sided gap filter applied to levels
                     "HP-gap (modified)")  # One-sided gap filter applied to first differences

colo <- c(colo_hp_all[1],   # HP trend       → brown
          "black",           # Symmetric target → black
          "darkgreen",       # HP-gap original  → dark green
          colo_hp_all[2])    # HP-gap modified  → red

# Initialise the plot with the first filter series; remaining series added in loop
plot(mplot[, 1],
     main  = "",
     axes  = FALSE,
     type  = "l",
     xlab  = "Lag structure",
     ylab  = "Filter coefficients (scaled)",
     ylim  = c(min(mplot), max(mplot)),
     col   = colo[1])

# Overlay all HP filters and add colour-coded legend via mtext
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  # Place each filter label at the top of the plot, stacked by line offset
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

# x-axis: re-label tick marks so they show lag 0, 1, 2, ... instead of index 1, 2, 3, ...
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

#-----------------------------------------------------------------------------
# Right panel: SSA filter family
#
# We select the first three columns of ssa_eps, which correspond to:
#   col 1: SSA concurrent filter  (horizon =  0, smooth ht = 12)
#   col 2: SSA forecast filter    (horizon = 18, smooth ht = 12)
#   col 3: SSA forecast filter    (horizon = 18, fast   ht = hp_trend ht)
#
# Comparing columns 2 and 3 isolates the effect of the holding-time constraint
# on the 18-month-ahead forecast filter shape.
# Comparing columns 1 and 2 shows how the filter adapts to increasing forecast horizon.
#-----------------------------------------------------------------------------

select_vec <- 1:3   # Select the three SSA filter columns described above

# Scale SSA filters to unit maximum for shape comparison (same rationale as HP panel)
mplot <- scale(ssa_eps[, select_vec], center = FALSE, scale = TRUE)

# Initialise the right-panel plot
plot(mplot[, 1],
     main  = "",
     axes  = FALSE,
     type  = "l",
     xlab  = "Lag structure",
     ylab  = "Filter coefficients (scaled)",
     ylim  = c(min(mplot), max(mplot)),
     col   = colo_SSA[1])

# Overlay all selected SSA filters and add colour-coded labels
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo_SSA[i])
  mtext(colnames(mplot)[i], col = colo_SSA[i], line = -i)
}

# x-axis: re-label to show lag 0, 1, 2, ... consistent with the left panel
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

#-----------------------------------------------------------------------------
# Note on downstream use:
#   These filters are applied to the monthly US Industrial Production (INDPRO)
#   series in Tutorial 5, where their real-time business-cycle dating performance
#   is evaluated empirically.
#   Results replicate Figure 9 in Wildi, M. (2024).
#-----------------------------------------------------------------------------






























