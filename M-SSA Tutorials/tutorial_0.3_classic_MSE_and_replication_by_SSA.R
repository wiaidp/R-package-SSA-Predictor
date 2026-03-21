# Tutorial 0.3
#
# This tutorial covers three main topics:
#
# 1. Derivation of the classical Mean-Square Error (MSE) predictor
#    for a simple signal extraction problem:
#      - Example 1: signal extraction based on a white noise process.
#      - Example 2: signal extraction based on an autocorrelated process.
#
# 2. Introduction to the SSA criterion
#    (Wildi, M. (2024), (2026a)):
#      - Example 3: replication of the MSE predictor using SSA.
#      - Example 4: sensitivity analysis of SSA to its input parameters.
#
# 3. Role of the MSE predictor in subsequent tutorials:
#      - As a benchmark for comparing predictor performances.
#      - As a base predictor on which SSA can be applied to trade off
#          smoothness against timeliness.


rm(list=ls())

# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


rm(list=ls())

# Load SSA-related functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic (quantifies lead/lag performance)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))
# Load signal extraction functions (used in JBCY paper; depends on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


#==============================================================================================
# Example 1: White noise input (x_t = ε_t)
#==============================================================================================
# This example extends Example 1 from the previous tutorial.

# Define a symmetric target filter
# (symmetry is not required in general; see Tutorials 2–5)
gamma <- c(0.25,0.5,0.75,1,0.75,0.5,0.25)

# Plot the symmetric (two-sided) target filter
plot(gamma, axes=F, type="l",
     xlab="Lag structure", ylab="Filter coefficients",
     main="Simple signal extraction (smoothing) filter")
axis(1, at=1:length(gamma),
     labels=(-(length(gamma)+1)/2)+1:length(gamma))
axis(2)
box()

# Simulate white noise input
set.seed(231)
len <- 120
sigma <- 1
epsilon <- sigma * rnorm(len)
x <- epsilon

# Verify absence of autocorrelation
acf(x)

# Apply filters:
# side = 2 → symmetric (two-sided, acausal)
# side = 1 → one-sided (causal, real-time)
y_sym <- filter(x, gamma, side=2)
y_one_sided <- filter(x, gamma, side=1)

tail(cbind(y_sym,y_one_sided))

# The symmetric filter is centered but undefined near the sample end (NAs).
# The one-sided filter is available in real time but delayed (right-shifted).
ts.plot(cbind(y_sym,y_one_sided),
        col=c("black","black"), lty=1:2,
        main="Two-sided vs one-sided filter")

# ── NOWCASTING AT THE SAMPLE END ────────────────────────────────
# In practice, the symmetric output y_sym is often required at
# or near the sample boundary.

# Definition:
#   • Nowcast: estimate of y_sym at t = len

# Problem:
#   • The symmetric filter at t = len depends on future values
#     x_{len+1}, x_{len+2}, x_{len+3}, which are not observed
#   • These must be replaced by forecasts

# MSE principle:
#   • Substitute missing future values with their MSE-optimal forecasts
#   • This yields the MSE-optimal nowcast

# Special case (white noise):
#   • Optimal forecasts of future values are zero
#   • ⇒ The MSE-optimal nowcast reduces to a truncated one-sided filter
# ───────────────────────────────────────────────────────────────

b_MSE <- gamma[((length(gamma)+1)/2):length(gamma)]

plot(b_MSE, axes=F, type="l",
     xlab="Lag structure", ylab="Filter coefficients",
     main="MSE nowcast filter")
axis(1, at=1:((length(gamma)+1)/2),
     labels=-1+1:((length(gamma)+1)/2))
axis(2)
box()

# Apply filters:
#   y_mse = causal predictor (one-sided filter)
#   y_sym = target (two-sided filter)
y_mse <- filter(x, b_MSE, side=1)
y_sym <- filter(x, gamma,  side=2)

ts.plot(cbind(y_sym, y_mse),
        col=c("black","green"), lty=1:2,
        main="Target (black) vs MSE predictor (green)")
abline(h=0)

# The MSE predictor appears noisier than the target.
# We quantify this via holding time (ht):
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)

# Interpretation:
#   - The predictor exhibits more frequent zero-crossings
#   - These additional crossings can be interpreted as spurious signals

# Compare with theoretical holding times
compute_holding_time_func(gamma)$ht
compute_holding_time_func(b_MSE)$ht

# Result:
#   - The MSE predictor generates substantially more crossings
#   - Empirical estimates converge to theoretical values as sample size increases

# ── TIMELINESS: LAG VIA TAU STATISTIC ──────────────────────────
# The predictor is also right-shifted (lagging) relative to the target.

# Arrange data: column 1 = target, column 2 = predictor
data_mat <- cbind(y_sym, y_mse)

# compute_min_tau_func:
#   - shifts the predictor relative to the target
#   - computes distances between nearest zero-crossings
#   - identifies the shift minimizing total distance
#   → estimate of lead (negative) or lag (positive)
max_lead <- 4
compute_min_tau_func(data_mat, max_lead)

# Result: approximately one-period lag of the MSE predictor

# ── MEAN-SQUARE ERROR (MSE) ────────────────────────────────────

# Empirical MSE
mean((y_sym - y_mse)^2, na.rm=T)

# Theoretical MSE:
#   - Forecast errors arise from replacing future ε_t by zero
#   - The error equals the contribution of omitted future innovations

# Weights applied to future innovations
gamma[1:3]

# Expected MSE
sigma^2 * sum(gamma[1:3]^2)

# Relative MSE (scaled by variance of target)
sum(gamma[1:3]^2) / sum(gamma^2)

# Interpretation:
#   - The nowcast error variance is about 30% of the target variance

# ── FREQUENCY-DOMAIN ANALYSIS ──────────────────────────────────
# The amplitude and phase-shift functions characterize filter properties:

#   • Amplitude → noise suppression (frequency attenuation)
#   • Phase shift → timeliness (lead/lag across frequencies)

K <- 600
amp_obj_mse <- amp_shift_func(K, as.vector(b_MSE), F)
amp_mse   <- amp_obj_mse$amp
shift_mse <- amp_obj_mse$shift
amp_obj_target <- amp_shift_func(K, gamma, F)
amp_target  <- amp_obj_target$amp

par(mfrow=c(2,1))

plot(amp_mse, type="l", axes=F,
     xlab="Frequency", ylab="",
     main="Amplitude of MSE filter",
     col="green", ylim=c(0,max(amp_target)))
lines(amp_target)
mtext("Target amplitude",line=-1)
mtext("MSE amplitude",line=-2,col="green")
abline(h=0)
axis(1, at=1+0:6*K/6,
     labels=expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

plot(shift_mse, type="l", axes=F,
     xlab="Frequency", ylab="",
     main="Phase shift of MSE filter",
     col="green")
lines(rep(0,K+1))
axis(1, at=1+0:6*K/6,
     labels=expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Interpretation:
#   - Amplitude: 
#       -Like the target, the MSE filter behaves as a low-pass filter
#       -Amplitude of MSE smaller in passband (at lower frequencies): typical zero-shrinkage of MSE
#       -Amplitude of MSE larger in stopband: typical noise leakage of MSE (more noisy zero-crossings)
#   - Phase shift:
#       -Shift of two-sided (symmartic) filter is zero
#       -Shift of MSE: approximately one-period delay in the passband,
#         consistent with above time-domain estimates (based on function compute_min_tau_func())

# Advantage of frequency-domain diagnostics:
#   - Depend only on filter coefficients (data-independent)
#   - Provide a structural characterization of filter performance

# Preview:
#   - Faster SSA predictors → smaller phase shifts (Tutorials 2, 3, 5)
#   - Smoother SSA predictors → stronger high-frequency attenuation
#       (with exceptions; see Tutorial 4)

# ── SPECTRAL DENSITY ───────────────────────────────────────────
# For white noise input:
#   - Input spectrum is flat
#   - Output spectrum = sigma^2 × (amplitude)^2

par(mfrow=c(1,1))

plot(sigma^2 * amp_mse^2, type="l", axes=F,
     xlab="Frequency", ylab="",
     main="Spectral density of MSE predictor",
     col="green",
     ylim=c(0,max(sigma^2 * amp_mse^2)))
abline(h=0)
axis(1, at=1+0:6*K/6,
     labels=expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# This spectral representation will be used later to assess whether
# standard business cycle filters (HP, BK, Hamilton, BN) introduce
# spurious cyclical components.

# ── DISCUSSION ─────────────────────────────────────────────────
# The MSE predictor is optimal in terms of mean-square error:
#   no alternative linear filter achieves a lower MSE.

# However, it exhibits two practical drawbacks:
#   - Lag: the predictor is delayed relative to the target
#   - Noise: the predictor has shorter holding time (more crossings)

# This leads to key questions:
#   - Can timeliness and smoothness be improved?
#   - What is the MSE cost of such improvements?

# The SSA framework addresses these trade-offs explicitly
#   and is introduced in the next example.





#==============================================================================================
# Example 2: x_t follows an ARMA process (no longer white noise)
#==============================================================================================
# The target filter gamma remains the same as in Example 1.
gamma <- c(0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25)

# Generate a new time series from an ARMA(1,1) data generating process (DGP)
set.seed(76)
len <- 1200

# ARMA(1,1) parameters: AR coefficient (a1), MA coefficient (b1), innovation std dev (sigma)
a1    <- 0.4
b1    <- 0.3
sigma <- 1

# Draw standard normal innovations scaled by sigma
epsilon <- sigma * rnorm(len)
x       <- epsilon

# Simulate the ARMA(1,1) process manually (without arima.sim) using the recursion:
#   x_t = a1 * x_{t-1} + epsilon_t + b1 * epsilon_{t-1}
for (i in 2:len) {
  x[i] <- a1 * x[i - 1] + epsilon[i] + b1 * epsilon[i - 1]
}

# Inspect the empirical autocorrelation function (ACF):
# x_t now exhibits serial dependence, confirming it is no longer white noise.
acf(x)

# Set the filter length L: must be large enough for the filter weights
# to decay to (effectively) zero, ensuring negligible truncation error.
L <- 50

# ── Wold (MA) representation ─────────────────────────────────────────────────
# The target filter gamma can be applied either directly to x_t, or to the
# innovation process epsilon_t. In the latter case, gamma must be convolved
# with the Wold (infinite MA) representation of x_t.
#
# We therefore compute the Wold MA coefficients by inverting the ARMA model,
# retaining the first L terms.
xi <- c(1, ARMAtoMA(ar = a1, ma = b1, lag.max = L - 1))

par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition of ARMA(1,1)")

# Verify the MA inversion: reconstructing x_t via the Wold representation
# should reproduce the original ARMA series exactly.
x_wold <- filter(epsilon, xi, side = 1)
ts.plot(cbind(x, x_wold), main = "ARMA vs. Wold: both series overlap")

# ── Convolving the target filter with the Wold coefficients ──────────────────
# Compute the convolution of the target filter gamma with the Wold coefficients xi.
# This yields gamma_conv: the target filter re-expressed in terms of epsilon_t.
gamma_conv <- conv_two_filt_func(xi, gamma)$conv
ts.plot(gamma_conv, main = "Target filter expressed in terms of epsilon_t (via Wold convolution)")
# Note: the convolved filter is no longer symmetric (unlike the original gamma).

# ── Verify equivalence of filter outputs ─────────────────────────────────────
# Confirm that applying gamma to x_t produces the same output as applying
# gamma_conv to epsilon_t. One-sided filters are used here to avoid
# edge-alignment issues introduced by the two-sided filter function.
y_sym  <- filter(x,       gamma,      side = 1)
y_conv <- filter(epsilon, gamma_conv, side = 1)

ts.plot(cbind(y_sym, y_conv),
        main = "Target applied to x_t overlaps convolved target applied to epsilon_t")

# ── Four purposes of the Wold convolution ────────────────────────────────────
# Expressing the target filter through its Wold convolution serves four main purposes:
#
# 1. Forecasting / backcasting:
#      Future innovations epsilon_{t+h} are unobserved and replaced by their
#      optimal MSE forecast of zero. Truncating these terms in gamma_conv yields
#      the MSE-optimal nowcast (or any forecast or backcast horizon).
#
# 2. Holding-time formula:
#      The analytical holding-time formula in Wildi (2024,2026a) requires white-noise
#      input. Working with epsilon_t satisfies this requirement directly.
#
# 3. Spectral density of the predictor:
#      The spectral density of the filtered output y_t follows immediately from
#      the squared amplitude function of the convolved filter gamma_conv.
#
# 4. Expected MSE:
#      The expected MSE of the predictor is straightforward to compute from
#      the squared coefficients of gamma_conv (see below).

# ── MSE-optimal nowcast filter ───────────────────────────────────────────────
# Derive the MSE nowcast filter expressed in terms of epsilon_t.
# Future innovations epsilon_{t+1}, epsilon_{t+2}, epsilon_{t+3} are unobservable
# and set to zero (their optimal MSE forecast), i.e., we discard the first three entries
# of gamma_conv
gamma_conv_mse_unpadded <- gamma_conv[4:L]
# zero-padded to preserve the original length L.
gamma_conv_mse <- c(gamma_conv_mse_unpadded, rep(0, 3))

# Plot: 
#   -the acausal target starts at lag=-3; 
#   -the MSE starts at lag=0; 
#   -for lags>=0 MSE and target overlap
plot(c(gamma_conv,rep(0,3)), main = "Target and MSE nowcast filter expressed in terms of epsilon_t",type="l",axes=F,ylab="",xlab="")
lines(c(rep(NA,3),gamma_conv_mse),col="green",lty=2)
mtext("Target",line=-1)
mtext("MSE",line=-2,col="green")
axis(1, at = 1 + 0:10 * L / 10,
     labels = -3+ 0:10 * L / 10)
axis(2)
box()


# ── Recover the MSE filter in terms of x_t ───────────────────────────────────
# In practice the filter is generally applied to x_t, not epsilon_t.
# Deconvolving the Wold coefficients xi from gamma_conv_mse recovers the
# MSE filter coefficients expressed directly in terms of x_t.
gamma_mse <- deconvolute_func(gamma_conv_mse, xi)$dec_filt
# Plot: 
#   -the acausal target starts at lag=-3; 
#   -the MSE starts at lag=0; 
#   -MSE assigns a larger weight to the most recent observation (lag 0)
plot(c(gamma,rep(NA,L-7)), main = "Target and MSE nowcast filter expressed in terms of epsilon_t",type="l",axes=F,ylab="",xlab="",ylim=c(0,max(gamma_mse)))
lines(c(rep(NA,3),gamma_mse[1:(L-3)]),col="green",lty=2)
mtext("Target",line=-1)
mtext("MSE",line=-2,col="green")
axis(1, at = 1 + 0:10 * L / 10,
     labels = -3+ 0:10 * L / 10)
axis(2)
box()


# Verify equivalence: applying gamma_mse to x_t should match gamma_conv_mse applied to epsilon_t.
y_mse      <- filter(x,       gamma_mse,      side = 1)
y_conv_mse <- filter(epsilon, gamma_conv_mse, side = 1)

ts.plot(cbind(y_conv_mse, y_mse),
        main = "MSE filter applied to x_t equals convolved MSE filter applied to epsilon_t")


# ── Visual comparison: target vs. MSE predictor ──────────────────────────────
# Apply the symmetric target filter to x_t (two-sided, using both leads and lags).
y_sym <- filter(x, gamma, side = 2)

ts.plot(cbind(y_sym, y_mse), col = c("black", "green"), lty = 1:2,
        main = "Target (black) vs. MSE predictor (green)")
abline(h = 0)

# ── Empirical holding times ───────────────────────────────────────────────────
# Compute the empirical holding times (average number of periods between
# consecutive zero-crossings) for the target and the MSE predictor.
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)
# The MSE predictor crosses zero more frequently than the target,
# indicating an excess of spurious (noisy) signals.

# ── Theoretical holding times ─────────────────────────────────────────────────
# In large samples, empirical holding times converge to their theoretical
# counterparts, provided the model is correctly specified.
#
# However, supplying gamma and gamma_mse (filters in terms of x_t) to the
# holding-time formula produces incorrect results, because the formula
# requires white-noise input.
compute_holding_time_func(gamma)$ht
compute_holding_time_func(gamma_mse)$ht

# Correct approach: supply the convolved filters (expressed in terms of epsilon_t).
# This satisfies the white-noise input requirement of the holding-time formula.
# Note: empirical ht above converge to true ht for large sample size 
compute_holding_time_func(gamma_conv)$ht
compute_holding_time_func(gamma_conv_mse)$ht
# Result: the MSE predictor generates approximately 40% more zero-crossings than
# the target in the long run, implying that a substantial share of its signals
# are spurious.

# ── Timeliness: tau statistic ─────────────────────────────────────────────────
# The MSE predictor appears right-shifted (lagging) relative to the target.
# The tau statistic from the JBCY paper quantifies the average lead/lag at
# zero-crossings by comparing target and predictor.
data_mat <- cbind(y_sym, y_mse)
max_lead  <- 4
# The plot produced below suggests the target leads the MSE predictor by
# approximately one time unit.
compute_min_tau_func(data_mat, max_lead)

# ── Mean-square error (MSE) ───────────────────────────────────────────────────
# Empirical MSE: average squared difference between target and MSE predictor.
mean((y_sym - y_mse)^2, na.rm = TRUE)

# Theoretical MSE: equals sigma^2 times the sum of squared coefficients
# corresponding to the unobservable future innovations (lags 1 to 3 of gamma_conv).
sigma^2 * sum(gamma_conv[1:3]^2)

# Relative MSE: expresses the approximation error as a fraction of the
# target output's total variance.
sum(gamma_conv[1:3]^2) / sum(gamma_conv^2)
# The relative MSE (~17%) is smaller than in Example 1, because x_t is smoother
# than white noise, making the target filter easier to approximate.
#
# Smoothness and timeliness of the MSE predictor can be improved via SSA
# (Selective Spectral Analysis) at the cost of a higher MSE;
# this trade-off is explored in Tutorials 1–5.

# ── Frequency-domain diagnostics ─────────────────────────────────────────────
# Examine the amplitude and phase-shift functions of the MSE predictor.
# These characterise two key properties of the filter:
#   • Noise suppression : how strongly the filter attenuates high-frequency components.
#   • Timeliness        : the lag (or lead) introduced by the filter at each frequency.
# Both are computed using gamma_mse (the MSE filter expressed in terms of x_t).
K         <- 600
amp_obj_mse <- amp_shift_func(K, as.vector(gamma_mse), FALSE)
amp_mse     <- amp_obj_mse$amp
shift_mse   <- amp_obj_mse$shift

par(mfrow = c(2, 1))

# Amplitude function: values close to 1 indicate faithful signal replication;
# values near 0 indicate strong attenuation.
plot(amp_mse, type = "l", axes = FALSE, xlab = "Frequency", ylab = "",
     main = "Amplitude function of the MSE filter", col = "green",
     ylim = c(0, max(amp_mse)))
abline(h = 0)
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Phase-shift function: positive values indicate a lag; negative values a lead.
# Ideally the phase shift is zero (or small) in the signal pass-band.
plot(shift_mse, type = "l", axes = FALSE, xlab = "Frequency", ylab = "",
     main = "Phase-shift function of the MSE filter", col = "green")
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()


























# Interpretation:
#   - Noise leakage at higher frequencies is more pronounced than in Example 1.
#       This is because x_t is smoother than epsilon_t and therefore requires
#       less attenuation from the predictor filter.
#   - The phase shift in the passband is slightly smaller than in Example 1,
#       though it remains close to one time unit.

# To compute the spectral density of the predictor y_mse, we must use
#   gamma_conv_mse - the filter expressed in terms of epsilon_t - rather
#   than gamma_mse. 
par(mfrow=c(1,1))
amp_obj_mse<-amp_shift_func(K,as.vector(gamma_conv_mse),F)
amp_conv_mse<-amp_obj_mse$amp
plot(sigma^2*amp_conv_mse^2,type="l",axes=F,xlab="Frequency",ylab="",main="Spectral density of predictor",col="green",ylim=c(0,max(sigma^2*amp_conv_mse^2)))
abline(h=0)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# The spectrum is rather weak towards higher frequencies: besides the squaring effect (we look at the squared amplitude function), 
#   this effect is also due to the fact that the ARMA-filter, linking xt to epsilont, is a lowpass in this example: 
#   xt is smoother than epsilont
amp_obj_xi<-amp_shift_func(K,as.vector(xi),F)
amp_arma<-amp_obj_xi$amp
par(mfrow=c(1,1))
plot(amp_arma,type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude of ARMA filter",col="green",ylim=c(0,max(amp_arma)))
abline(h=0)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

#####################################################################################################
#####################################################################################################
# Background and overview of the SSA framework
# (Building on the ARMA signal extraction problem introduced in Example 2)
#
# Setup:
#   - Let z_t = sum_{k=-inf}^{inf} gamma_k x_{t-k} denote the target.
#       In the examples above, z_t = y_sym is the output of the acausal two-sided filter.
#   - Let x_t = sum_{j=0}^{inf} xi_j epsilon_{t-j} be a stationary (or integrated) process
#       with Wold representation xi.
#   - The goal is to estimate or predict z_{t+delta} for integer delta
#       (delta > 0: forecast, delta = 0: nowcast, delta < 0: backcast).
#
# The three-step procedure introduced in Example 2 applies to SSA as well:
#   1. Wold decomposition: fit a SARIMA model to x_t and invert it to obtain the
#        MA(inf) representation xi using ARMAtoMA.
#   2. Convolution and truncation: convolve gamma with xi and replace future
#        innovations epsilon_{t+h} (h > 0) by zero. This yields the MSE filter
#        expressed in terms of epsilon_t.
#   3. Deconvolution: recover the MSE filter expressed in terms of x_t by
#        deconvolving xi from the result of step 2.
#
# SSA inputs:
#   The target z_{t+delta} is specified by:
#     1. gamma_k : the target filter (forecast, nowcast, or signal extraction;
#                    no restrictions other than finite length).
#     2. delta   : the forecast horizon (positive), nowcast (zero), or backcast (negative).
#     3. xi      : the Wold decomposition of x_t. If omitted, SSA defaults to white
#                    noise (x_t = epsilon_t).
#   Beyond the MSE inputs, SSA additionally requires:
#     4. rho1    : the lag-one ACF of the predictor, encoding the holding-time constraint.
#     5. L       : the filter length (3 <= L <= n-1). A larger L yields a better predictor
#                    when xi is correctly specified, but filter coefficients typically decay
#                    quickly, so truncation at moderate L rarely affects performance.
#
# SSA criterion:
#   For given inputs, SSA computes the optimal finite-length filter subject to the
#   holding-time constraint. Optimality is defined as:
#     1. Maximum sign accuracy: the filter best matches the signs of z_{t+delta}.
#     2. Maximum target correlation: the filter correlates most strongly with z_{t+delta}.
#   Both criteria are equivalent under Gaussianity (Wildi, M. (2024),
#   https://doi.org/10.1007/s41549-024-00097-5). The equivalence is robust to
#   moderate departures from Gaussianity (e.g. t-distributions down to nu=2,
#   equity returns, and macroeconomic data).
#
# SSA outputs:
#   Filter coefficients:
#     1. ssa_eps : optimal SSA filter applied to epsilon_t (holding-time compliant).
#     2. ssa_x   : optimal SSA filter applied to x_t (holding-time compliant).
#                    This is the primary output for practical use.
#     3. mse_eps : MSE filter applied to epsilon_t (not holding-time compliant).
#     4. mse_x   : MSE filter applied to x_t (not holding-time compliant).
#   Theoretical criterion values:
#     5. crit_rhoyz      : correlation of the SSA predictor with the MSE predictor.
#     6. crit_rhoy_target: correlation of the SSA predictor with the two-sided target.
#     7. crit_rhoyy      : lag-one ACF of the optimised predictor; should match the
#                            imposed rho1 (a discrepancy indicates non-convergence).
#
# In large samples and under correct model specification, the empirical counterparts
#   of crit_rhoyz, crit_rhoy_target, and crit_rhoyy converge to their theoretical values.
#
# Examples 3 and 4 illustrate:
#   1. The SSA function call: inputs and outputs.
#   2. Convergence of sample estimates to the theoretical criterion values.

#############################################################################################
#############################################################################################
# Example 3: Replicating the MSE Predictor with SSA
#
# This example builds on the framework established in Example 2.
# As outlined in the tutorial introduction, SSA exactly replicates the MSE
#   predictor when the holding-time constraint ht is set to match the holding
#   time of the MSE predictor itself.
#
# Hyperparameter specification:
#   - To replicate the MSE predictor, we set ht to the theoretical holding time
#       of the MSE predictor.
#   - The holding-time formula requires the filter expressed in terms of epsilon_t;
#       we therefore use the convolved filter gamma_conv_mse for this computation.
ht<-compute_holding_time_func(gamma_conv_mse)$ht
# Instead of ht, we provide the bijective `twin', namely the lag-one acf, see Wildi, M. (2024) 
# We can transform ht in rho1 with the function compute_rho_from_ht 
rho1<-compute_rho_from_ht(ht)
# Alternatively we could set rho1 directly (the above function computes ht as well as the lag-one acf)
rho1<-compute_holding_time_func(gamma_conv_mse)$rho_ff1
# Filter length (should be at least twice the holding time)
L<-max(L,2*round(ht,0))
# Nowcast i.e. delta=0
delta<-0
# Recall that the symmetric two-sided target filter gamma is right-shifted by
#   (length(gamma)-1)/2 = 3 time units when expressed as a one-sided causal filter.
# To recover the acausal two-sided target, we must left-shift gamma by
#   (length(gamma)-1)/2 lags. Equivalently, this corresponds to forecasting
#   the causal filter gamma by (length(gamma)-1)/2 + delta steps ahead.
forecast_horizon<-(length(gamma)-1)/2+delta
forecast_horizon
# We let SSA know that the data is an ARMA: recall that the ARMA-filter pre-whitens the noise (it's a lowpass in this example) 
xi<-xi
# Target specification:
#   - gamma is specified as a symmetric one-sided (not two-sided) filter and will be left-shifted
#       by forecast_horizon within SSA_func to recover the acausal two-sided target.
#   - gamma is expressed in terms of x_t, not epsilon_t.
#   - By supplying the Wold decomposition xi to SSA_func, the function automatically
#       convolves the target filter with xi internally.
gammak_generic<-gamma

# Apply SSA
# Since gamma is applied to xt, we have to supply information about the link between xt and epsilont in terms of xi
# Warning messages inform that zeroes are appended to shorter filters and that SSA solution is very close to MSE after optimization
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# Filter as applied to xt
ssa_x<-SSA_obj$ssa_x
# Convolved filter as applied to epsilont
ssa_eps<-SSA_obj$ssa_eps

# SSA replicates MSE
mplot<-cbind(ssa_x,gamma_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()


# Same for the convolved designs (filters applied to epsilont)
ssa_eps=SSA_obj$ssa_eps
mplot<-cbind(ssa_eps,gamma_conv_mse)
# Both filters overlap: SSA just replicated MSE up to arbitrary scaling
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to epsilont: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()


# SSA also computes the MSE-filters directly: as applied to xt and epsilont
mplot<-cbind(SSA_obj$mse_x,gamma_mse)
# Both filters overlap: SSA just replicated MSE up to arbitrary scaling
plot(mplot[,1],main="MSE by SSA and original MSE as applied to xt: both filters overlap",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()
# MSE as applied to epsilont
mplot<-cbind(SSA_obj$mse_eps,gamma_conv_mse)
# Both filters overlap: SSA just replicated MSE up to arbitrary scaling
plot(mplot[,1],main="MSE by SSA and original MSE as applied to epsilont: both filters overlap",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# In applications one typically relies on ssa_x: but the package also returns all other filters (for diagnostics, benchmarking, validation,...)

#------------------------------
# Performance Checks: Convergence of sample estimates to theoretical criterion values
#
# A. Criterion Values
#
# There are two equivalent optimization criteria:
#   1. Maximize the correlation with the MSE solution (here: gamma_conv_mse)
#   2. Maximize the correlation with the effective target (here: the two-sided filter)
#
# Both criteria are mathematically equivalent — see:
#   Wildi, M. (2024), (2026a) section 2
#
# A.1 Below, we compute the first criterion:
#   We leverage the convolved filters (applied to epsilon_t) to derive the correlations.
#   Under the assumption of white noise input, the exact cross-correlation formula is:
t(ssa_eps)%*%gamma_conv_mse/sqrt(t(ssa_eps)%*%ssa_eps*t(gamma_conv_mse)%*%gamma_conv_mse)
# Since SSA replicates MSE, the cross-correlation is trivially one
# SSA returns this criterion value
SSA_obj$crit_rhoyz
# A.2 Second Criterion: Maximize Correlation with the Acausal Two-Sided Target
#
#   To switch from the first criterion to the second, we simply correct for the
#   difference in squared norms (or variances) between the MSE solution and the
#   two-sided target filter — see Wildi, M. (2024), (2026a).
#
#   Crucially, this scaling ratio is independent of ssa_eps, which means both
#   criteria are equivalent up to a constant scaling factor.
length_ratio<-sqrt(sum(gamma_conv_mse^2)/sum(gamma_conv^2))
t(ssa_eps)%*%gamma_conv_mse/sqrt(t(ssa_eps)%*%ssa_eps*t(gamma_conv_mse)%*%gamma_conv_mse)*length_ratio
# SSA also returns the corresponding criterion value
SSA_obj$crit_rhoy_target
# The correlation with the effective target is smaller than one because the acausal filter relies on future (unobserved) data
# Therefore, crit_rhoy_target <= crit_rhoyz 

# We can verify pertinence of the above criterion values by computing empirical correlations
y_ssa<-filter(x,ssa_x,side=1)
# First criterion
cor(na.exclude(cbind(y_mse,y_ssa)))[1,2]
# Second criterion
cor(na.exclude(cbind(y_sym,y_ssa)))[1,2]
# These numbers converge to the theoretical values for large sample sizes

# B. Holding-Times and Lag-One ACFs: Additional Performance Checks
#
#   Beyond optimization criteria, we can — and should — also verify holding-times (HT)
#   or, equivalently, the lag-one ACFs of the filters.
#   See Proposition 2 in the JBCY paper for the theoretical background.
#
# The imposed holding-time is:
ht
# Its equivalent lag-one ACF representation is:
rho1
# After optimization, SSA achieves the following lag-one ACF:
SSA_obj$crit_rhoyy
# If both values match, this confirms that the optimization successfully reached the global maximum.
#   Note: The tightness of this approximation can be improved by increasing the number of
#   iterations in the numerical optimization. The default is 20 iterations, which is
#   sufficient for typical (non-exotic) applications — no need to adjust this in practice.

# Checking Holding-Times via compute_holding_time_func()
#
#   We use compute_holding_time_func() to verify HT numerically.
#   Important: we must supply the convolved filter ssa_eps, since the HT computation
#   assumes white noise input.
compute_holding_time_func(ssa_eps)$ht

#   The filter applied directly to x_t yields a substantially smaller HT:
compute_holding_time_func(ssa_x)$ht
#   This is expected: the ARMA filter already imposes a smoothing effect on epsilon_t,
#   as confirmed by its own HT:
compute_holding_time_func(xi)$ht
#   Accordingly, ssa_x extends the native HT of the ARMA filter (a lowpass in this example)
#   to satisfy the imposed constraint. Note that HTs of convolved filters do not combine
#   additively — HT is a non-linear function of the lag-one ACF (see Proposition 2, JBCY paper).

# Empirical Verification
#
#   Empirical lag-one ACF of the SSA output — should match the imposed rho1:
acf(na.exclude(y_ssa))$acf[2]
rho1
# Let's have a look at the empirical ht
compute_empirical_ht_func(y_ssa)
# It matches our constraint (error can be made arbitrarily small when increasing the sample size)
ht
#   Any discrepancy can be made arbitrarily small by increasing the sample size,
#   since xi is assumed to be the true (correctly specified) model.
#
#   If empirical estimates differ substantially from theoretical values, this signals
#   model misspecification in xi. However, SSA is generally fairly robust to misspecification —
#   ample empirical evidence is provided in the subsequent tutorials.
#
#   Remark: The statistics above can also be reinterpreted as model diagnostics for xi,
#   offering an alternative to classical tests such as the Ljung-Box test.
#########################################################################################
#########################################################################################
# =============================================================================
# Example 4: Alternative Target Specifications
# =============================================================================
#
# This example builds on Example 3, but replaces the symmetric (acausal) filter
# with the MSE filter as the optimization target.
# For the theoretical background, see:
#   Wildi, M. (2024), (2026a)
#
# The target can be specified in two equivalent ways:
#   - gamma_mse      : the MSE filter as applied to x_t
#   - gamma_conv_mse : the MSE filter as applied to epsilon_t (convolved form)
#
#   Important: the choice of input representation requires care —
#   the filter supplied must be consistent with the data passed to SSA
#   (i.e., gamma_mse paired with x_t, or gamma_conv_mse paired with epsilon_t).
#
# -----------------------------------------------------------------------------
# A. First Case: gamma_mse — The MSE Filter Applied to x_t
# -----------------------------------------------------------------------------
gammak_generic<-gamma_mse
# This target is causal and does not have to be shifted (in contrast to gamma in exercise 3). We set:
forecast_horizon<-delta
# Nowcast
forecast_horizon
# Since gamma_mse is applied to xt, we have to supply information about the link between xt and epsilont in terms of xi
xi<-xi
# Don't forget xi in the function call: otherwise SSA assumes xt=epsilont white noise, by default
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

ssa_x=SSA_obj$ssa_x
mse_x<-SSA_obj$mse_x
mplot<-cbind(ssa_x,gamma_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

mplot<-cbind(ssa_eps,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to epsilont: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# B. Second case: target is gamma_conv_mse which is applied to epsilont
gammak_generic<-gamma_conv_mse
# There is a subtle change in the function-call since we now omit xi
# By default (omission of xi), SSA assumes the data to be white noise
# This is correct since gamma_conv_mse is the convolved target and is applied to epsilont 
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1)

ssa_x=SSA_obj$ssa_x
ssa_eps=SSA_obj$ssa_eps
mplot<-cbind(ssa_eps,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# Since we assume xt=epsilont (white noise) ssa_x and ssa_eps are identical in this case
mplot<-cbind(ssa_x,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to epsilont: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# We could obtain the same result as above by setting
xi_null<-NULL
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi_null)

ssa_x=SSA_obj$ssa_x
ssa_eps=SSA_obj$ssa_eps
mplot<-cbind(ssa_eps,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# Or we could obtain the same result by specifying the identity
xi_id<-1
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi_id)

ssa_x=SSA_obj$ssa_x
ssa_eps=SSA_obj$ssa_eps
# We do not have to re-scale the new filter
mplot<-cbind(ssa_eps,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

#================================================================
# Summary
#================================================================
#
# Filter Decomposition
#   - Splitting the estimation problem into Gamma (target filter applied to x_t) and
#     Xi (Wold decomposition of x_t) is a modelling convenience, not a necessity.
#   - The fundamental object of interest — from a theoretical standpoint — is always
#     the convolved filter design, applied to epsilon_t:
#       * The convolved design is required for deriving optimal filters and
#         computing theoretical performance measures.
#       * Once obtained, the optimal x_t-filter can be recovered via simple deconvolution.
#       * The x_t-filter is practically convenient, as it can be applied directly
#         to the observed data x_t (rather than to the unobserved innovations epsilon_t).
#
# SSA and the MSE Solution
#   - SSA can exactly replicate the MSE solution under appropriate settings:
#       * The MSE solution can be substituted for the effective target without
#         affecting the SSA output.
#       * The MSE filter can be supplied either in its x_t-form (gamma_mse) or
#         in its convolved form (gamma_conv_mse), provided the function call is
#         adjusted consistently.
#
# Empirical vs. Theoretical Consistency
#   - Theoretical criteria and holding-times (HT) align with empirical estimates
#     when the model Xi is correctly specified.
#   - Substantial discrepancies between empirical and theoretical values indicate
#     model misspecification.
#
#----------------------------------------------------------------
# Final Remarks
#----------------------------------------------------------------
#
#   - The constraint rho1 can be freely modified in the above example:
#       * If rho1 = rho(MSE): SSA replicates the MSE solution (as shown here).
#       * If rho1 > rho(MSE): SSA produces a smoother output — the typical
#         application case in practice.
#
#   - See the following tutorials for more meaningful and practically
#     relevant applications.
#================================================================
