# ════════════════════════════════════════════════════════════════════
# TUTORIAL 1: SSA — HOLDING-TIME, SMOOTHNESS AND FORECASTING
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────
# This tutorial presents applications of the univariate SSA
# framework, with a primary focus on classic forecasting
# (rather than signal extraction or nowcasting).
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

#---------------------------------------------------------
# Example 1
# Illustrate the holding-time, see Wildi, M. (2024), (2026a)


# Let xt be a realization of length 12000 of an AR(1)-process
# We need a long series in order to obtain an accurate empirical holding-time estimate
len<-12000
a1<-0.8
set.seed(1)
x<-arima.sim(n = len, list(ar = a1))

ts.plot(x)
# Typical patterns for acf and pacf
acf(x)
acf(x,type="partial")
# Estimation
ar_obj<-arima(x,order=c(1,0,0))
# Diagnostics are OK
tsdiag(ar_obj)



# Holding time
# Lets have a look at the holding time of the AR(1) process
# We first compute the empirical holding-time i.e. the mean-duration between consecutive zero-crossings of the data
empirical_ht<-len/length(which(sign(x[2:len])!=sign(x[1:(len-1)])))
empirical_ht
# We have implemented the formula in a function
empirical_ht<-compute_empirical_ht_func(x)
empirical_ht

# We now rely on the exact holding-time expression, see cited literature, section 2
# For that purpose we need the MA-inversion of the process: we can use the true a1 or the MSE estimate for that purpose
# The function ARMAtoMA can invert arbitrary stationary ARMA-specifications
xi<-c(1,ARMAtoMA(ar=a1,lag.max=len-1))
# Note: we can interpret the AR(1)-process as MA(infty)-filter applied to epsilont
# We can then plug the filter (MA-inversion) into the function compute_holding_time_func
ht_obj<-compute_holding_time_func(xi)
# This is the 'true' or expected holding time of the AR(1)-filter (or process): compare with the empirical one above
ht_obj$ht
# The function also computes the lag-one acf which is of course a1 in this case
ht_obj$rho_ff1

# If the lag-one acf is known, then we can compute the holding-time ht with the function compute_holding_time_from_rho_func
rho<-a1
compute_holding_time_from_rho_func(rho)

# If the holding-time ht is known, then we can compute the lag-one acf with the function compute_rho_from_ht
ht<-ht_obj$ht
compute_rho_from_ht(ht)


# We can analyze finite sample issues
# For that purpose consider a much shorter sample of the above process
len<-100
a1<-0.8
set.seed(1)
x<-arima.sim(n = len, list(ar = a1))

ts.plot(x)
# Typical patterns for acf and pacf
acf(x)
acf(x,type="partial")
# Estimation
ar_obj<-arima(x,order=c(1,0,0))
# Diagnostics OK
tsdiag(ar_obj)

ahat<-ar_obj$coef["ar1"]

rho<-ahat
# Compare with true/expected holding-time above
compute_holding_time_from_rho_func(rho)
#   - See also Tables 3 and 4 in Wildi (2024) for a detailed discussion of sampling error.
#       * In practice, sampling error is largely irrelevant because it cancels out
#         when performance is assessed in relative terms.
#       * SSA is fundamentally concerned with relative performance — i.e., gains
#         over a chosen benchmark — rather than absolute criterion values.


#================================================================
# Example 2: Inconsistent Settings
#================================================================
#
# A MA filter of length L imposes an upper bound on the achievable holding-time (HT).
# Requesting an HT that exceeds this bound leads to an inconsistent specification.
# This example briefly illustrates such a case and its consequences.
#
#----------------------------------------------------------------
# Setup
#----------------------------------------------------------------
#
# We retain the AR(1) process and filter from the previous example.
len<-120
xi<-c(1,ARMAtoMA(ar=a1,lag.max=len-1))
# Filter Length Selection
#   Proposition 3 in Wildi (2024) establishes that the filter length L must be chosen
#   large enough to accommodate the desired HT: specifically, the imposed HT must not
#   exceed the maximum achievable HT of a MA(L) filter.
#   Here, we deliberately choose L too small to trigger this inconsistency.
L <- 5

# Target Specification: Identity Filter
#   We set the target to the identity filter for simplicity — this choice will be
#   explained in detail below.
gammak_generic <- 1

# Forecast Horizon: One-Step Ahead
forecast_horizon <- 1

# Holding-Time Constraint
#   The following HT is intentionally too large for a filter of length L = 5,
#   creating an inconsistent specification (see Proposition 3 in Wildi (2024)).
ht <- 7

# Converting HT to Lag-One ACF
#   SSA_func requires rho1 (lag-one ACF) rather than HT directly.
#   We use compute_rho_from_ht() to convert:
rho1 <- compute_rho_from_ht(ht)
rho1

# Maximum Achievable Lag-One ACF for MA(L)
#   rhomax_func(L) returns the maximum lag-one ACF attainable by a MA filter of
#   length L (see Proposition 3 in Wildi (2024)).
#   If rhomax_func(L) < rho1, no valid SSA solution exists for the given L —
#   the filter length is insufficient to meet the imposed smoothness constraint.
rhomax_func(L)

# SSA Optimization
#   Two optimization routines are available:
#     1. Brute-force grid search : robust for edge cases and exotic configurations
#     2. Fast triangulation      : efficient for typical applications, provided the
#                                  imposed HT does not exceed the L-dependent upper bound
#
#   In this example, the call below will produce an error message, since rho1 exceeds
#   rhomax — confirming that the specification is inconsistent.
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Important: Data Sample vs. Model Specification
#   Note that the observed data x_t is never passed directly to SSA_func.
#   All relevant information about the data-generating process is encoded in xi:
#     - If xi == NULL : white noise is assumed
#     - If xi != NULL : xi contains the MA coefficient weights from the
#                       Wold decomposition of x_t
#
#================================================================
# Resolving the Inconsistency: Two Options
#================================================================
#
# Returning to the previous example, the inconsistency can be resolved by either:
#   (a) Increasing L to accommodate the imposed HT, or
#   (b) Decreasing HT to stay within the bounds of the current L
#
# We first demonstrate option (a): increasing L.
#
#----------------------------------------------------------------
# Revised Settings
#----------------------------------------------------------------

# Filter length: increased from 5 to 15 to accommodate the imposed HT
L <- 15

# Target: identity filter (details provided below)
gammak_generic <- 1

# Forecast horizon: one-step ahead
forecast_horizon <- 1

# Holding-time: unchanged from the previous (inconsistent) example
ht <- 7

# Converting HT to lag-one ACF (required input format for SSA_func)
rho1 <- compute_rho_from_ht(ht)
rho1

# Maximum achievable lag-one ACF for MA(L = 15)
#   Since rhomax_func(L) > rho1, a valid SSA solution now exists.
#   The filter length L = 15 is sufficient to meet the imposed smoothness constraint.
rhomax_func(L)

# SSA Optimization
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Optimal SSA Filter Coefficients
#   SSA_obj$ssa_x contains the optimal SSA filter of length L = 15,
#   expressed as weights to be applied to x_t.
#   For a full reference of SSA_func return values, see Tutorial 0.3.
SSA_obj$ssa_x


# Let's now decrease ht (with the short filter length)
L<-5
# Now ht is OK
ht<-4
# Note that we need to supply rho1 (instead of ht) to SSA_func below
rho1<-compute_rho_from_ht(ht)
rho1
rhomax_func(L)

SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_obj$ssa_x

#----------------------------------------------------------------
# General Remark: Choosing L Relative to the Holding-Time Constraint
#----------------------------------------------------------------
#
# When rho1 is close to rhomax(L), the SSA optimization may yield a degenerate
# or unintuitive filter solution (correct but strange looking). This is a boundary effect: as rho1 approaches
# rhomax(L), the feasible solution space contracts, leaving little room for
# meaningful optimization.
#
# Recommendation: choose L sufficiently large relative to the imposed HT.
#   As a practical rule of thumb:
#
#                         L >= 2 * ht
#
#   This ensures that rho1 remains well within the feasible range of rhomax(L),
#   producing well-behaved and interpretable filter solutions.




#---------------------------------------------------------------------------------------
#================================================================
# Example 3: One-Step Ahead Forecasting
#================================================================
#
# We demonstrate one-step ahead forecasting for the AR(1) process
# specified in the previous example.

# AR(1) coefficient
a1 <- 0.8

# Wold Decomposition
#   We compute the MA(inf) representation (Wold decomposition) of the AR(1) process,
#   truncated at lag len - 1. This encodes all data-generating process information
#   required by SSA_func — no observed data sample is needed.
#   Note: in practice, a1 could be replaced by its finite-sample estimate.
len <- 100
xi <- c(1, ARMAtoMA(ar = a1, lag.max = len - 1))

#----------------------------------------------------------------
# SSA Settings
#----------------------------------------------------------------

# Holding-Time Constraint
#   We target an HT larger than the native HT of the AR(1) filter (see Example 1),
#   so that the SSA output is smoother (fewer zero-crossings) than the raw process.
ht <- 6

# Lag-One ACF
#   SSA_func requires rho1 rather than HT directly; we convert accordingly.
rho1 <- compute_rho_from_ht(ht)

# Filter Length
#   L should be:
#     - Large enough to capture the filter dynamics (see Example 2: L >= 2 * ht)
#     - Small enough to remain below the available sample length
#   If the filter coefficients decay sufficiently fast to zero, the chosen L is adequate.
#   Note: larger L does not cause overfitting of the SSA filter itself — overfitting
#   can only arise if xi (the Wold decomposition) is overfitted.
L <- 20

# Target Specification
#   For forecasting, the target is the identity filter (gammak_generic = 1):
#   SSA seeks a causal filter whose output best approximates x_t itself.
#   In signal extraction settings, the target would instead be a non-trivial
#   filter applied to x_t (e.g., a lowpass filter) — see the examples below.
gammak_generic <- 1

# Forecast Horizon: one-step ahead
forecast_horizon <- 1

# SSA Optimization
#   We retain the default settings for the numerical optimization routine.
#   Note: SSA_func checks whether the length of the supplied target matches L.
#   If not, a warning is issued and the target is automatically zero-padded
#   to length L.
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

ssa_x <- SSA_obj$ssa_x

# Plot the optimized SSA filter coefficients
ts.plot(ssa_x)

#----------------------------------------------------------------
# Performance Checks
#----------------------------------------------------------------

# 1. Holding-Time Verification
#   We verify the imposed HT constraint empirically by applying the optimized
#   filter to a long simulated AR(1) series and computing the resulting HT.
len <- 100000
set.seed(1)
x <- arima.sim(n = len, list(ar = a1))

# Apply the optimized SSA filter to the simulated series
yhat <- filter(x, ssa_x, sides = 1)

# Empirical HT of the filter output
empirical_ht <- compute_empirical_ht_func(yhat)
empirical_ht
# Compare with the imposed constraint — both should agree (match asymptotically):
ht

#----------------------------------------------------------------
# 2. Optimization Convergence Check
#   We compare the lag-one ACF of the optimized filter (crit_rhoyy) with
#   the imposed rho1. If both values agree (up to rounding), the optimization
#   successfully converged to the global maximum.
#   If a substantial discrepancy is observed, increase split_grid
#   (the number of iterations; default = 20, sufficient for nearly all applications).
#   In well-posed, non-exotic cases, both values match almost exactly.
SSA_obj$crit_rhoyy
rho1

#----------------------------------------------------------------
# 3. SSA Criterion Values
#   Two equivalent criteria are available (see Proposition 4 in the 2004 JBCY paper).
#   Both yield the same optimal filter, but measure performance differently.

# 3.1 Criterion 1: Correlation with the One-Step Ahead MSE Forecast
#   crit_rhoyz measures the correlation between the SSA filter output and
#   the MSE predictor. 
SSA_obj$crit_rhoyz

# Empirical verification: compute the one-step ahead MSE predictor and correlate
MSE_forecast <- a1 * x
cor(yhat, MSE_forecast, use = 'pairwise.complete.obs')

# 3.2 Criterion 2: Correlation with the Effective Target
#   crit_rhoy_target measures the correlation between the SSA filter output and
#   the series shifted forward by forecast_horizon (i.e., the future observations).
SSA_obj$crit_rhoy_target

# Empirical verification: correlate filter output with the forward-shifted series
cor(yhat, c(x[(1 + forecast_horizon):len], rep(NA, forecast_horizon)), use = 'pairwise.complete.obs')

# Interpretation of the Two Criteria:
#   - The sample estimates converge to their respective criterion values as the
#     sample size increases — as expected from asymptotic theory.
#   - Both criteria are equivalent in the sense that they lead to the same
#     SSA filter (see Proposition 4 in the JBCY paper).
#   - Criterion 1 (MSE target) yields a higher correlation because the MSE
#     predictor is causal — it uses only past and present observations.
#   - Criterion 2 (effective target) yields a lower correlation because the
#     forward-shifted series implicitly assumes knowledge of future observations
#     (here: the one-step ahead value).
#   - When the HT imposed on SSA matches the native HT of the MSE solution,
#     SSA exactly replicates MSE — see Tutorial 0.3 and Example 4 below.

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


#================================================================
# Example 7
# Holding-time: strict interpretation, misspecification, smoothing hyperparameter
#================================================================
#
# The "holding-time" (ht) measures how long a filter output tends to stay on
# the same side of zero (i.e., its sign persistence). Two versions are compared:
#   - Expected ht: derived analytically from the filter coefficients (assumes white noise input)
#   - Empirical ht: computed directly from the filtered output series

#-----------------------------------------------------------------------------
# 7.1 Correct model: xt = epsilon_t (white noise)
#
# When the input is truly white noise, the analytical expected ht
# and the empirically observed ht should agree up to sampling error.
#-----------------------------------------------------------------------------

len <- 12000  # Length of the simulated time series

# Define an equally-weighted moving average (MA) filter of length L
# Each coefficient equals 1/L, so the filter computes a simple rolling mean
L  <- 10
b  <- rep(1/L, L)

set.seed(65)
x <- rnorm(len)  # Simulate white noise input

# Apply the MA filter to x using a one-sided (causal) convolution
yhat <- filter(x, b, side = 1)
yhat <- na.exclude(yhat)  # Remove the leading NAs introduced by the filter lag

# Compare expected ht (from filter coefficients) with empirical ht (from filtered series).
# Both should be close; any difference is due to finite-sample variation.
compute_holding_time_func(b)$ht       # Analytical expected holding-time
compute_empirical_ht_func(yhat)       # Empirical holding-time from simulated data

#-----------------------------------------------------------------------------
# 7.2 Misspecified model: xt = ARMA(1,1) process
#
# The analytical formula for expected ht assumes white noise input.
# When the input is autocorrelated (e.g., ARMA), that assumption is violated,
# so the expected ht and the empirical ht will no longer agree.
#-----------------------------------------------------------------------------

L  <- 10
b  <- rep(1/L, L)   # Same equally-weighted MA filter as above
a1 <- 0.4           # AR(1) coefficient
b1 <- 0.3           # MA(1) coefficient

set.seed(65)
x <- arima.sim(n = len, model = list(ar = a1, ma = b1))  # Simulate ARMA(1,1) input

# Apply the same MA filter to the ARMA-generated data
yhat <- filter(x, b, side = 1)
yhat <- na.exclude(yhat)

# Compare expected ht vs. empirical ht.
# They now differ because the white noise assumption underlying compute_holding_time_func() is violated.
compute_holding_time_func(b)$ht       # Analytical ht (incorrectly assumes white noise input)
compute_empirical_ht_func(yhat)       # Empirical ht from ARMA-filtered data

#-----------------------------------------------------------------------------
# Reconciling expected and empirical ht under misspecification
#
# To fix the mismatch we need to find the "correct" filter: the one whose input
# IS white noise (epsilon_t) but whose output is identical to applying b to xt.
#
# By the Wold decomposition, any ARMA process can be written as an infinite MA:
#   xt = xi(L) * epsilon_t,  where xi contains the MA-inversion coefficients.
#
# The correct filter is therefore the convolution of xi with b:
#   (b applied to xt) = (b * xi applied to epsilon_t)
# Using this convolved filter with the white-noise formula recovers the true ht.
#-----------------------------------------------------------------------------

# Step 1: Compute MA-inversion (Wold representation) of the ARMA(1,1) process.
xi <- c(1, ARMAtoMA(ar = a1, ma = b1, lag.max = len - 1))

# Step 2: Convolve the MA-inversion filter xi with the original filter b.
# The result, ssa_eps, is the filter that maps white noise epsilon_t to the
# same output as b applied to xt.
ssa_eps <- conv_two_filt_func(xi, b)$conv

# Visual check: compare the original filter b (black) with the convolved filter ssa_eps (red).
# The convolved filter captures additional autocorrelation structure not present in b.
ts.plot(ssa_eps[1:30], col = "red",
        main = "Original filter b (black) vs. convolved filter ssa_eps (red)")
lines(b, col = "black")

# Compute the analytical expected ht using the corrected (convolved) filter.
# This should now match the empirical ht of yhat computed above.
compute_holding_time_func(ssa_eps)$ht

#-----------------------------------------------------------------------------
# Numerical verification: confirm that b applied to xt equals ssa_eps applied to epsilon_t
#-----------------------------------------------------------------------------

# Simulate the same ARMA(1,1) process manually to have direct access to epsilon_t
set.seed(43)                          # Note: use set.seed() as a function call, not assignment
x   <- eps <- rnorm(len)
for (i in 2:len)
  x[i] <- a1 * x[i - 1] + eps[i] + b1 * eps[i - 1]

# Filter xt with the original filter b
yhat_x   <- filter(x,   b,              side = 1)

# Filter epsilon_t with the first 30 coefficients of the convolved filter ssa_eps.
# Truncating at lag 30 is a practical approximation; the tail coefficients are negligible.
yhat_eps <- filter(eps, ssa_eps[1:30],  side = 1)

# Align both series by removing NAs (introduced by the filter lags)
mplot <- na.exclude(cbind(yhat_x, yhat_eps))

# Plot both filtered series over the first 1000 observations.
# They should be virtually identical, confirming that the two representations are equivalent.
ts.plot(mplot[1:1000, ], lty = 1:2,
        main = "b applied to xt (solid) vs. convolved filter applied to epsilon_t (dashed)")

#-----------------------------------------------------------------------------
# 7.3 Misspecified model: xt = ARMA(1,1) + mu (non-zero mean)
#
# We extend the misspecification of Section 7.2 by adding a non-zero mean mu.
# This has an important consequence for the holding-time: shifting the series
# upward moves the zero-crossing line away from the centre of the distribution,
# reducing the frequency of zero-crossings and therefore *increasing* the
# empirical ht relative to the zero-mean case.
#-----------------------------------------------------------------------------

L  <- 10
b  <- rep(1/L, L)   # Equally-weighted MA filter of length L
a1 <- 0.4           # AR(1) coefficient of the ARMA(1,1) input process
b1 <- 0.3           # MA(1) coefficient of the ARMA(1,1) input process
mu <- 1             # Non-zero mean shift applied to the simulated series

set.seed(655)
# Simulate an ARMA(1,1) process shifted upward by mu.
# The resulting series oscillates around mu instead of zero.
x    <- mu + arima.sim(n = len, model = list(ar = a1, ma = b1))
yhat <- filter(x, b, side = 1)   # Apply one-sided MA filter
yhat <- na.exclude(yhat)          # Remove leading NAs introduced by filter lag

# Compare analytical expected ht with empirical ht.
# The discrepancy is now *larger* than in Section 7.2 (zero-mean ARMA case),
# because the upward shift reduces zero-crossings in the filtered output.
compute_holding_time_func(b)$ht   # Analytical ht (assumes zero-mean white noise input)
compute_empirical_ht_func(yhat)   # Empirical ht: inflated due to the mean shift

#-----------------------------------------------------------------------------
# Diagnostic plot: visualise the effect of the mean shift on zero-crossings
#-----------------------------------------------------------------------------

# Plot the first 100 observations of x to see how far the series sits above zero.
# The horizontal line at 0 is the reference for zero-crossings;
# the line at mu shows the true centre of the process.
# As mu increases, zero-crossings become rarer (for mu > 4 they may vanish entirely).
ts.plot(x[1:100], main = "Simulated ARMA(1,1) + mu series (first 100 obs)")
abline(h = 0,  col = "red",  lty = 2, lwd = 1.5)   # Zero line (reference for ht calculation)
abline(h = mu, col = "blue", lty = 2, lwd = 1.5)   # Mean line (true centre of the process)

#-----------------------------------------------------------------------------
# Reconciling expected and empirical ht under mean-shifted misspecification
#
# Two corrections are required:
#   1. Centre the data (subtract mu) so that zero-crossings again correspond
#      to crossings of the process mean — this is what the ht formula measures.
#   2. Correct the filter for the ARMA autocorrelation structure, exactly as
#      in Section 7.2 (convolve b with the Wold/MA-inversion of xt).
#
# After both corrections, the analytical expected ht should match the
# empirical ht of the centred, filtered series.
#-----------------------------------------------------------------------------

# Step 1: Compute the MA-inversion (Wold representation) of the ARMA(1,1) process.
xi <- c(1, ARMAtoMA(ar = a1, ma = b1, lag.max = len - 1))

# Step 2: Convolve xi with the original filter b to obtain the corrected filter ssa_eps.
# ssa_eps maps white noise epsilon_t to the same output as b applied to (xt - mu).
ssa_eps <- conv_two_filt_func(xi, b)$conv

# Visual check: compare the original filter b with the corrected filter ssa_eps.
# The convolved filter is more spread out, reflecting the ARMA memory of xt.
ts.plot(ssa_eps[1:30], col = "red",
        main = "Original filter b (black) vs. corrected filter ssa_eps (red)")
lines(b, col = "black")

# Analytical ht of the corrected filter.
# Note: this does NOT yet match the empirical ht of yhat above, because yhat
# was computed from the uncentred series x.  Centering is still required (see below).
compute_holding_time_func(ssa_eps)$ht

#-----------------------------------------------------------------------------
# Numerical verification: confirm equivalence of the two filtering approaches
#-----------------------------------------------------------------------------

# Simulate the ARMA(1,1) process manually to retain direct access to epsilon_t.
# Note: set.seed() must be called as a function, not used as an assignment.
set.seed(43)
y   <- eps <- rnorm(len)
for (i in 2:len)
  y[i] <- a1 * y[i - 1] + eps[i] + b1 * eps[i - 1]
x <- y + mu   # Reintroduce the mean shift

# Approach A: filter the *centred* series (x - mean(x)) with b.
# Subtracting mean(x) estimates and removes mu; any residual difference from
# ssa_eps arises from finite-sample estimation error of mu by mean(x).
yhat_x   <- filter(x - mean(x), b,             side = 1)

# Approach B: filter epsilon_t directly with the corrected filter ssa_eps.
# Truncating at lag 30 is a practical approximation; coefficients beyond
# lag 30 are negligibly small for this ARMA specification.
yhat_eps <- filter(eps,          ssa_eps[1:30], side = 1)

# Align both series by removing NAs
mplot <- na.exclude(cbind(yhat_x, yhat_eps))

# Plot both filtered series for the first 1000 observations.
# The two lines should be virtually identical, confirming the equivalence.
ts.plot(mplot[1:1000, ], lty = 1:2,
        main = paste("b applied to centred xt (solid)",
                     "vs. corrected filter applied to epsilon_t (dashed)"))

# Final check: empirical ht of the centred filtered series should now match
# the analytical ht of the corrected filter ssa_eps (up to sampling error).
compute_empirical_ht_func(yhat_x)          # Empirical ht after centering
compute_holding_time_func(ssa_eps)$ht      # Analytical ht of corrected filter

#-----------------------------------------------------------------------------
# Conclusions from Sections 7.1 – 7.3
#
# 1. The analytical expected ht formula assumes that filter b is applied to
#    zero-mean white noise.  Two forms of misspecification break this:
#
#    a) Autocorrelation: if xt is ARMA, convolve b with the Wold MA-inversion
#       xi of xt to obtain the correct filter before applying the ht formula.
#
#    b) Non-zero mean: if xt has mean mu != 0, centre the data first (subtract mu).
#       The ht then measures mean duration between crossings of the mu-line,
#       not the zero-line. Under misspecification, this literal interpretation
#       is lost, but the smoothing effect remains valid (see point 3 below).
#
# 2. When both corrections are applied (centering + convolution), the
#    analytical and empirical holding-times agree up to sampling error.
#
# 3. Even under misspecification, increasing ht in the SSA optimisation
#    always produces a smoother filter:
#      - The smoothing effect operates at the mu-line and at any other level.
#      - However, the ht value is biased relative to the true zero-crossing rate.
#
# 4. Therefore, ht can be treated as a *smoothing hyperparameter* that controls
#    noise suppression regardless of whether the model is correctly specified.
#    Tutorials 2, 3, and 4 illustrate these practical smoothing effects.
#-----------------------------------------------------------------------------

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
# (corresponds to lambda = 1600 for quarterly data, scaled by 3^4)

# Compute the full set of HP-related filter designs in one call.
# The returned object contains both the two-sided ideal target and several
# one-sided (causal) approximations used as SSA targets below.
HP_obj <- HP_target_mse_modified_gap(L, lambda_monthly)

# Two-sided (symmetric) HP trend filter — the ideal smoother, not realisable
# in real time because it uses future observations.  Used as the benchmark target.
target <- HP_obj$target

# One-sided gap filter: approximates (data - HP trend) in real time.
# Positive values indicate the series is above the long-run trend (expansion);
# negative values indicate contraction.
hp_gap <- HP_obj$hp_gap

# Modified one-sided gap filter designed to operate on *first differences* of
# the data.  When applied to differenced data it produces output identical to
# hp_gap applied to the original (level) series.  Useful when the data are
# differenced for stationarity before filtering.
modified_hp_gap <- HP_obj$modified_hp_gap

# Classic one-sided (concurrent) HP trend filter.
# This is the MSE-optimal real-time estimate of the two-sided target *if* the
# data follow an ARIMA(0,2,2) process — the model implicitly assumed by the HP filter.
hp_trend <- HP_obj$hp_trend

# Alternative one-sided HP trend based on direct (least-squares) truncation of
# the two-sided symmetric filter.  This is MSE-optimal under the white noise
# assumption (consistent with xi = NULL used in SSA below).
hp_mse <- HP_obj$hp_mse

# Any of the above designs can serve as an SSA target; see Tutorials 2 and 5.

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
# All filter coefficient vectors are scaled to unit maximum for visual
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
































#-----------------------------------------------------------------
# Example 8
# We here replicate the SSA-filters in the business-cycle analysis, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# SSA-filters assume white noise i.e. xi=NULL: we do not `mine' the data by fitting a model

# HP and hyperparameter
L<-201
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)

# Two sided filter
target=HP_obj$target
# One-sided gap filter: observation minus HP-trend
hp_gap=HP_obj$hp_gap
# One sided gap filter when applied to first differences of the data
#  modified_hp_gap applied to first differences replicates the output of the original gap applied to the original data  
modified_hp_gap=HP_obj$modified_hp_gap
# Classic one-sided (concurrent) HP-trend: this is an optimal (MSE) estimate of the two-sided filter if the data follows an ARIMA(0,2,2) specification 
hp_trend=HP_obj$hp_trend
# Alternative one-sided HP-trend based on truncating the two-sided filter
#   It is an optimal (MSE) estimate of the two-sided filter if the data is white noise (which is consistent with xi=NULL)
hp_mse=HP_obj$hp_mse
# We can use any of the above HP-designs as targets for SSA, see tutorials 2 and 5
#---------------------------
# SSA and hyperparameters
# The classic one-sided HP-trend hp_trend is subject to some undesirable noise-leakage: the filter generates 
#   unwanted noisy crossings
# Its holding time is:
compute_holding_time_func(hp_trend)$ht
# We therefore ask SSA to target hp_trend while simultaneously improving noise-suppression (less noisy alarms)  
# For this purpose we impose a larger holding-time, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
ht<-12
# The resulting SSA filter will generate roughly 40% less (noisy) crossings:
ht/compute_holding_time_func(hp_trend)$ht
rho1<-compute_rho_from_ht(ht)
# We compute a nowcast of the classic concurrent HP as well as a 18-months ahead forecast, both subject to ht
# We can supply a vector with the desired forecast horizons: SSA will compute optimal filters for each forecast horizon
forecast_horizon_vec<-c(0,18)
# We here want SSA to approximate the one-sided MSE HP-trend
gammak_generic<-hp_mse

# By omitting xi in the call we assume the data to be white noise
SSA_obj<-SSA_func(L,forecast_horizon_vec,gammak_generic,rho1)

# In this case ssa_x and ssa_eps are the same filters
ssa_eps<-SSA_obj$ssa_eps
colnames(ssa_eps)<-paste("SSA(",round(ht,2),",",forecast_horizon_vec,")",sep="")


# We also compute a concurrent SSA filter which replicates ht of the classic one-sided HP and which forecasts 
#  it 18-months ahead
ht_short<-compute_holding_time_func(hp_trend)$ht
rho1_short<-compute_rho_from_ht(ht_short)
forecast_short<-forecast_horizon_vec[2]

# Compute fast SSA forecast filter with same holding time as HP-trend
SSA_obj<-SSA_func(L,forecast_short,gammak_generic,rho1_short)

ssa_eps1<-SSA_obj$ssa_eps
colnames(ssa_eps1)<-paste("SSA(",round(ht_short,2),",",forecast_horizon_vec[2],")",sep="")

ssa_eps<-cbind(ssa_eps,ssa_eps1)

#--------------------------------------
# Plot filters: this plot replicates fig.4 in JBCY paper
colo_hp_all<-c("brown","red")
colo_SSA<-c("orange","blue","violet")
colo_all<-c(colo_hp_all,colo_SSA)


par(mfrow=c(1,2))
mplot<-scale(cbind(hp_trend,target,hp_gap,modified_hp_gap),center=F,scale=T)
colnames(mplot)<-c("HP trend","Target symmetric","HP-gap (original)","HP-gap (modified)")
colo<-c(colo_hp_all[1],"black","darkgreen",colo_hp_all[2])
plot(mplot[,1],main="",axes=F,type="l",xlab="Lag-structure",ylab="filter-coefficients",ylim=c(min(mplot),max(mplot)),col=colo[1])
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()
# Select forecast horizons 0 and 18
select_vec<-1:3
mplot<-scale(ssa_eps[,select_vec],center=F,scale=T)
plot(mplot[,1],main="",axes=F,type="l",xlab="",ylab="",ylim=c(min(mplot),max(mplot)),col=colo_SSA[1])
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo_SSA[i])
  mtext(colnames(mplot)[i],col=colo_SSA[i],line=-i)
}  
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# We employ these filters in tutorial 5, when applied to the monthly INDPRO series
# The above plot replicates results in Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5





