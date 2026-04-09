# =============================================================================
# Tutorial 0.3
# =============================================================================
# This tutorial covers three main topics:
#
# 1. Derivation of the classical Mean-Square Error (MSE) predictor
#    for a simple signal extraction problem:
#      - Example 1: signal extraction based on a white noise process.
#      - Example 2: signal extraction based on an autocorrelated process.
#
# 2. Introduction to the SSA criterion
#    (Wildi, M. (2024), (2026a)):
#      - Replication of the MSE predictor by SSA (Example 3).
#      - Swapping acausal target (prediction) and MSE predictor (smoothing) 
#         in SSA objective (example 4)
#
# 3. Role of the MSE predictor in subsequent tutorials:
#      - As a benchmark for comparing predictor performances.
#      - As a base predictor on which SSA can be applied to trade off
#          smoothness against timeliness (customization).

# ── BACKGROUND ────────────────────────────────────────────────────
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

# Theoretical background:
#   Wildi, M. (2026a) Sign Accuracy, Mean-Squared Error and the Rate
#     of Zero Crossings: a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547

# =============================================================================


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


# =============================================================================
# Example 1: White noise input (x_t = ε_t)
# =============================================================================
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





# =============================================================================
# Example 2: x_t follows an ARMA process (no longer white noise)
# =============================================================================
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
# at the cost of a higher MSE;
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
     ylim = c(0, max(amp_target)))
abline(h = 0)
lines(amp_target)
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Phase-shift function: positive values indicate a lag; negative values a lead.
# Ideally the phase shift is zero (or small), i.e., equal to the target in the signal pass-band.
plot(shift_mse, type = "l", axes = FALSE, xlab = "Frequency", ylab = "",
     main = "Phase-shift function of the MSE filter", col = "green")
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Interpretation:
#   - Amplitude: 
#     - Noise leakage is more pronounced than in Example 1 (amplitude larger in stopband): 
#       the ARMA process generates a smoother series (stronger autocorrelation, lowpass, see plot below), 
#       leaving less residual noise for the MSE filter to suppress.
#   - Phase shift:
#       -Shift is smaller than in example 1, since noise suppression is weaker

# ── Spectral density of the MSE predictor ────────────────────────────────────
# To compute the spectral density of y_mse, we must use gamma_conv_mse
# (the MSE filter expressed in terms of the white-noise innovations epsilon_t)
# rather than gamma_mse (the filter expressed in terms of x_t).
# This is because the spectral density formula sigma^2 * |A(omega)|^2 is only
# valid when the input is white noise with variance sigma^2.
par(mfrow = c(1, 1))

amp_obj_mse  <- amp_shift_func(K, as.vector(gamma_conv_mse), FALSE)
amp_conv_mse <- amp_obj_mse$amp

plot(sigma^2 * amp_conv_mse^2, type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "",
     main = "Spectral density of MSE predictor",
     col  = "green", ylim = c(0, max(sigma^2 * amp_conv_mse^2)))
abline(h = 0)
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# The spectral density is strongly concentrated at low frequencies and
# decays rapidly towards higher frequencies. Two effects contribute to this:
#
# 1. Squaring effect: the spectral density is proportional to the squared
#      amplitude |A(omega)|^2, which amplifies the attenuation already
#      present in the amplitude function.
#
# 2. ARMA low-pass effect: the Wold filter xi (mapping epsilon_t to x_t)
#      acts as a low-pass filter in this example, meaning that x_t is
#      inherently smoother than epsilon_t. The convolved filter gamma_conv_mse
#      therefore inherits this low-pass character, further suppressing
#      high-frequency spectral power.

# ── Amplitude of the Wold (ARMA) filter ──────────────────────────────────────
# To confirm the low-pass character of the ARMA process, we plot the
# amplitude function of the Wold filter xi. A monotonically decreasing
# amplitude confirms that the ARMA filter attenuates high frequencies,
# producing the smoother series x_t observed in the time domain.
amp_obj_xi <- amp_shift_func(K, as.vector(xi), FALSE)
amp_arma   <- amp_obj_xi$amp

par(mfrow = c(1, 1))
plot(amp_arma, type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "",
     main = "Amplitude function of the Wold (ARMA) filter",
     col  = "green", ylim = c(0, max(amp_arma)))
abline(h = 0)
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

#=============================================================================================
# Background and overview of the SSA framework
#=============================================================================================
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
#   https://doi.org/10.1007/s41549-024-00097-5). The equivalence is fairly robust to
#   rather strong departures from Gaussianity (e.g. t-distributions down to nu=4,
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






# =============================================================================
# Example 3: Replicating the MSE Predictor with SSA
# =============================================================================
#
# This example builds on the framework established in Example 2.
# SSA exactly replicates the MSE predictor when the holding-time constraint
# is set to match the theoretical holding time of the MSE predictor itself.
# This serves as a correctness check: if SSA cannot replicate MSE as a
# special case, something is wrong with the implementation or the inputs.

# ── Step 1: Compute the holding-time constraint ───────────────────────────────
#
# The holding-time formula requires the filter expressed in terms of epsilon_t.
# We therefore use gamma_conv_mse (the MSE filter convolved with the Wold
# coefficients xi) rather than gamma_mse (the filter in terms of x_t).
ht <- compute_holding_time_func(gamma_conv_mse)$ht
ht

# The SSA function accepts the holding-time constraint as its
# bijective counterpart rho1 (the lag-one ACF of the predictor output).
# The two representations are equivalent; see Wildi, M. (2024) for details.
# compute_rho_from_ht() converts ht to rho1:
rho1 <- compute_rho_from_ht(ht)
rho1

# Alternatively, rho1 can be extracted directly from compute_holding_time_func(),
# which returns both ht and rho1 simultaneously:
rho1 <- compute_holding_time_func(gamma_conv_mse)$rho_ff1
rho1

# ── Step 2: Set the filter length ────────────────────────────────────────────
#
# L must be at least twice the holding time to ensure the filter has enough
# lags to capture the imposed smoothness constraint without truncation bias.
L <- max(L, 2 * round(ht, 0))

# ── Step 3: Set the forecast horizon ─────────────────────────────────────────
#
# delta = 0 corresponds to a nowcast (estimating the current target value).
delta <- 0

# The symmetric two-sided target filter gamma, when written as a one-sided
# causal filter, is right-shifted by (length(gamma) - 1) / 2 = 3 time units.
# To recover the acausal two-sided target, the causal gamma must be left-shifted by the
# same amount. Equivalently, this corresponds to forecasting the one-sided
# causal filter (length(gamma) - 1) / 2 + delta steps ahead.
forecast_horizon <- (length(gamma) - 1) / 2 + delta
forecast_horizon

# ── Step 4: Specify inputs to SSA_func ───────────────────────────────────────
#
# Supply the Wold decomposition xi so that SSA_func is aware that x_t follows
# an ARMA process. The ARMA filter acts as a low-pass in this example,
# pre-smoothing epsilon_t before it enters the target filter.
# SSA_func will internally convolve gamma with xi.
xi <- xi

# gamma is specified as a causal filter in terms of x_t
# The causal filter is symmetric around lag=(length(gamma) - 1) / 2 = 3 (instead of lag 0).
# SSA_func will left-shift gamma by forecast_horizon to recover the intended acausal
# two-sided target, and will convolve it with xi to express it in terms of
# epsilon_t — both steps are handled internally.
gammak_generic <- gamma

# ── Step 5: Apply SSA ─────────────────────────────────────────────────────────
#
# Note: warning messages may inform that shorter filters are zero-padded to
# length L, and that the SSA solution is very close to MSE after optimisation
# (which is expected here, since ht matches that of the MSE predictor).
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract the SSA filter expressed in terms of x_t (primary output for
# practical use) and in terms of epsilon_t (used for diagnostics).
ssa_x   <- SSA_obj$ssa_x
ssa_eps <- SSA_obj$ssa_eps

# ── Verification: SSA replicates MSE ─────────────────────────────────────────
#
# When the holding-time constraint matches that of the MSE predictor, the SSA
# filter should coincide with the MSE filter (up to an arbitrary scaling factor).

# Filter in terms of x_t: SSA vs. MSE
mplot <- cbind(ssa_x, gamma_mse)
plot(mplot[, 1], main = "SSA and MSE filters applied to x_t: both overlap",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "Filter weight")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()

# Convolved filters in terms of epsilon_t: SSA vs. MSE
ssa_eps <- SSA_obj$ssa_eps
mplot   <- cbind(ssa_eps, gamma_conv_mse)
plot(mplot[, 1], main = "SSA and MSE filters applied to epsilon_t: both overlap",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "Filter weight")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()

# SSA_func also computes the MSE filter internally and returns it separately.
# These internal MSE filters should match the reference filters derived in Example 2.

# Internal MSE filter in terms of x_t vs. reference gamma_mse
mplot <- cbind(SSA_obj$mse_x, gamma_mse)
plot(mplot[, 1], main = "Internal MSE (SSA) and reference MSE applied to x_t: both overlap",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "Filter weight")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()

# Internal MSE filter in terms of epsilon_t vs. reference gamma_conv_mse
mplot <- cbind(SSA_obj$mse_eps, gamma_conv_mse)
plot(mplot[, 1], main = "Internal MSE (SSA) and reference MSE applied to epsilon_t: both overlap",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "Filter weight")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()

# In practice, ssa_x is the primary output. The remaining filters (ssa_eps,
# mse_x, mse_eps) are returned for diagnostics, benchmarking, and validation.

#############################################################################################
# Performance Checks: Convergence of Sample Estimates to Theoretical Criterion Values
#############################################################################################

# ── A. Criterion values ───────────────────────────────────────────────────────
#
# SSA optimises one of two equivalent criteria (see Wildi, M. (2024), (2026a), Section 2):
#
#   Criterion 1: Maximise correlation between the SSA predictor and the MSE predictor.
#   Criterion 2: Maximise correlation between the SSA predictor and the two-sided target.
#
# Both criteria are mathematically equivalent and yield the same optimal filter, see Wildi (2024,2026a).

# A.1 – Criterion 1: Correlation with the MSE predictor
#
# Using the convolved filters (in terms of epsilon_t) and the white-noise
# cross-correlation formula:
t(ssa_eps) %*% gamma_conv_mse /
  sqrt(t(ssa_eps) %*% ssa_eps * t(gamma_conv_mse) %*% gamma_conv_mse)

# Since SSA replicates MSE here, the correlation is trivially 1.
# SSA_func returns this value directly:
SSA_obj$crit_rhoyz

# A.2 – Criterion 2: Correlation with the acausal two-sided target
#
# Switching from Criterion 1 to Criterion 2 requires correcting for the
# difference in squared norms between the MSE filter and the two-sided target
# filter (see Wildi, M. (2024), (2026a)).
# This scaling ratio is independent of ssa_eps, confirming that both criteria
# are equivalent up to a constant factor.
length_ratio <- sqrt(sum(gamma_conv_mse^2) / sum(gamma_conv^2))
t(ssa_eps) %*% gamma_conv_mse /
  sqrt(t(ssa_eps) %*% ssa_eps * t(gamma_conv_mse) %*% gamma_conv_mse) * length_ratio

# SSA_func returns this value directly:
SSA_obj$crit_rhoy_target

# Note: crit_rhoy_target < crit_rhoyz, because the two-sided target relies on
# future (unobserved) data (whereas MSE is a causal filter).

# A.3 – Empirical verification
#
# In large samples, empirical correlations converge to the theoretical values
# returned by SSA_func. We verify this by computing correlations from the
# filtered output y_ssa.
y_ssa <- filter(x, ssa_x, side = 1)

# Empirical Criterion 1: correlation between SSA and MSE predictors
cor(na.exclude(cbind(y_mse, y_ssa)))[1, 2]

# Empirical Criterion 2: correlation between SSA predictor and two-sided target
cor(na.exclude(cbind(y_sym, y_ssa)))[1, 2]

# Both values should be close to the theoretical counterparts above.
# Any residual discrepancy decreases as the sample size increases.

# ── B. Holding-time and lag-one ACF verification ─────────────────────────────
#
# Beyond the optimisation criteria, we verify that the imposed holding-time
# constraint is satisfied. See Proposition 2 in Wildi, M. (JBCY paper) for
# the theoretical background.

# Imposed holding time and its lag-one ACF equivalent:
ht
rho1

# Lag-one ACF of the optimised predictor as returned by SSA_func:
SSA_obj$crit_rhoyy
# If this matches rho1, the optimisation successfully reached the global maximum.
# The closeness of this approximation improves with the number of optimisation
# iterations (default: 20), which is sufficient for typical applications.

# Numerical verification via compute_holding_time_func():
# Important: supply ssa_eps (filter in terms of epsilon_t), since the
# holding-time formula assumes white-noise input.
compute_holding_time_func(ssa_eps)$ht

# The filter in terms of x_t yields a substantially smaller holding time:
compute_holding_time_func(ssa_x)$ht
# This is expected: the ARMA filter (a low-pass in this example) already
# imposes smoothing on epsilon_t, as confirmed by its own holding time:
compute_holding_time_func(xi)$ht
# Accordingly, ssa_x extends the native smoothing of the ARMA filter to
# satisfy the imposed constraint. Note that holding times of convolved filters
# do not combine additively — HT is a non-linear function of the lag-one ACF
# (see Proposition 2, JBCY paper).

# ── C. Empirical lag-one ACF and holding-time ─────────────────────────────────
#
# The empirical lag-one ACF of the SSA output should match the imposed rho1:
acf(na.exclude(y_ssa))$acf[2]
rho1

# The empirical holding time should match the imposed ht:
compute_empirical_ht_func(y_ssa)
ht
# Any discrepancy decreases as the sample size increases, provided xi is
# correctly specified.

# ── D. Model diagnostics remark ───────────────────────────────────────────────
#
# If empirical estimates differ substantially from theoretical values, this
# signals model misspecification in xi. SSA is generally robust to moderate
# misspecification — empirical evidence is provided in the subsequent tutorials.
#
# Remark: The statistics computed above can be reinterpreted as informal
# model diagnostics for xi, offering a practical alternative to classical
# tests such as the Ljung-Box test.





# =============================================================================
# Example 4: Alternative Target Specifications
# =============================================================================
#
# This example builds on Example 3, but replaces the symmetric (acausal) filter
# with the MSE-optimal filter as the optimization target.
#
# For the theoretical background, see:
#   Wildi, M. (2024), (2026a)
# -----------------------------------------------------------------------------
# Interpretation:
#
# Although both target specifications yield the same SSA solution, their conceptual
# interpretations differ:
#
#   I. gamma_mse (causal MSE target):
#      SSA acts as a smoother of the causal MSE filter.
#      The MSE filter serves as the baseline, and SSA customizes its
#      output by imposing a holding-time constraint on the signal.
#
#   II. gamma (acausal filter, after shift by forecast_horizon):
#      SSA acts as a predictor of the acausal (two-sided) target filter.
#      The optimization seeks the best causal approximation to the
#      symmetric target, subject to a holding-time constraint.
#
#   The same solution is regarded either as a smoother (case I) or a predictor (case II)

# -----------------------------------------------------------------------------
# Furthermore, the MSE target  can be specified in two equivalent ways:
#
#   A. gamma_mse      : the MSE filter expressed in terms of x_t
#   B. gamma_conv_mse : the MSE filter expressed in terms of epsilon_t
#                       (convolved form, obtained by pre-multiplying gamma_mse
#                        by the MA coefficients xi)
#
# Important: the filter specification and the input data passed to SSA must
# be consistent:
#   - gamma_mse      requires x_t as input       (xi must be supplied)
#   - gamma_conv_mse requires epsilon_t as input  (xi can be omitted)
# see details below.

# =============================================================================


# =============================================================================
# 4.A. First Case: gamma_mse — MSE Filter Expressed in Terms of x_t
# =============================================================================

# Set the optimization target to the MSE filter applied to x_t
gammak_generic <- gamma_mse

# gamma_mse is a causal filter, so no time-shift is needed
# (unlike the symmetric target in Example 3, which required shifting).
# The forecast horizon is set directly to delta (i.e., nowcast: horizon = 0).
forecast_horizon <- delta

# Confirm the forecast horizon (should be 0 for a nowcast)
forecast_horizon

# Specify the MA coefficient vector xi, which defines the link between x_t
# and epsilon_t via the moving-average representation: x_t = xi(B) * epsilon_t.
# This is required because gamma_mse operates on x_t, not epsilon_t.
xi <- xi

# Call SSA with xi explicitly supplied.
# Omitting xi would cause SSA to assume x_t = epsilon_t (white noise), which
# would be incorrect here since x_t has MA structure.
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract the SSA filter expressed in terms of x_t and the MSE criterion value
ssa_x  <- SSA_obj$ssa_x
mse_x  <- SSA_obj$mse_x

# --- Plot 1: SSA vs. gamma_mse, both expressed in terms of x_t ---
# The SSA filter should closely replicate the MSE filter when applied to x_t.
mplot <- cbind(ssa_x, gamma_mse)
plot(mplot[, 1],
     main = "SSA vs. MSE filter applied to x_t: SSA replicates MSE",
     axes = F, type = "l", xlab = "Lag", ylab = "Filter weights")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()

# --- Plot 2: SSA vs. gamma_conv_mse, both expressed in terms of epsilon_t ---
# Converting both filters to the epsilon_t representation allows cross-checking
# consistency between the two equivalent target specifications.
mplot <- cbind(ssa_eps, gamma_conv_mse)
plot(mplot[, 1],
     main = "SSA vs. MSE filter applied to epsilon_t: SSA replicates MSE",
     axes = F, type = "l", xlab = "Lag", ylab = "Filter weights")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()


# =============================================================================
# 4.B. Second Case: gamma_conv_mse — MSE Filter Expressed in Terms of epsilon_t
# =============================================================================

# Set the optimization target to the convolved MSE filter applied to epsilon_t
gammak_generic <- gamma_conv_mse

# Call SSA without supplying xi (i.e., xi is omitted from the function call).
# By default, SSA treats the input as white noise (x_t = epsilon_t), which is
# correct here since gamma_conv_mse is already expressed in terms of epsilon_t.
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1)

# Extract SSA filters in both representations
ssa_x   <- SSA_obj$ssa_x
ssa_eps <- SSA_obj$ssa_eps

# --- Plot 3: SSA (epsilon_t representation) vs. gamma_conv_mse ---
# Both filters are in the epsilon_t domain; SSA should replicate gamma_conv_mse.
mplot <- cbind(ssa_eps, gamma_conv_mse)
plot(mplot[, 1],
     main = "SSA vs. MSE filter applied to epsilon_t: SSA replicates MSE",
     axes = F, type = "l", xlab = "Lag", ylab = "Filter weights")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()

# --- Plot 4: SSA (x_t representation) vs. gamma_conv_mse ---
# When xi is omitted, SSA assumes x_t = epsilon_t, so ssa_x and ssa_eps are
# identical. This plot confirms that result.
mplot <- cbind(ssa_x, gamma_conv_mse)
plot(mplot[, 1],
     main = "SSA vs. MSE filter applied to x_t (white noise assumed): SSA replicates MSE",
     axes = F, type = "l", xlab = "Lag", ylab = "Filter weights")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()


# -----------------------------------------------------------------------------
# 4.B1. Equivalent call: explicitly passing xi = NULL
# -----------------------------------------------------------------------------
# Passing xi = NULL is equivalent to omitting xi entirely.
# Both result in SSA treating the input as white noise (x_t = epsilon_t).
xi_null <- NULL
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi_null)

ssa_x   <- SSA_obj$ssa_x
ssa_eps <- SSA_obj$ssa_eps

# Results should be identical to those obtained in section B above
mplot <- cbind(ssa_eps, gamma_conv_mse)
plot(mplot[, 1],
     main = "SSA vs. MSE filter (xi = NULL): SSA replicates MSE",
     axes = F, type = "l", xlab = "Lag", ylab = "Filter weights")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()


# -----------------------------------------------------------------------------
# 4.B2. Equivalent call: explicitly passing xi = 1 (identity/scalar MA)
# -----------------------------------------------------------------------------
# Setting xi = 1 is the scalar identity for the MA representation,
# meaning x_t = 1 * epsilon_t, i.e., x_t = epsilon_t (white noise).
# This produces the same result as omitting xi or passing xi = NULL.
xi_id <- 1
SSA_obj <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi_id)

ssa_x   <- SSA_obj$ssa_x
ssa_eps <- SSA_obj$ssa_eps

# No rescaling is needed since xi = 1 leaves the filter structure unchanged.
# Results should again be identical to those in sections B and B1.
mplot <- cbind(ssa_eps, gamma_conv_mse)
plot(mplot[, 1],
     main = "SSA vs. MSE filter (xi = 1, identity): SSA replicates MSE",
     axes = F, type = "l", xlab = "Lag", ylab = "Filter weights")
lines(mplot[, 2])
axis(1, at = 1:L, labels = -1 + 1:L)
axis(2)
box()



#================================================================
# Summary
#================================================================
#
# Convolution:
#   - Splitting the estimation problem into Gamma (target filter applied to x_t)
#     and Xi (Wold decomposition of x_t) is a modelling convenience, not a necessity.
#   - The fundamental object of interest, from a theoretical standpoint, is always
#     the convolved filter design, applied to epsilon_t:
#       * The convolved design is required for deriving optimal filters and
#         computing theoretical performance measures.
#       * Once obtained, the optimal x_t-filter can be recovered via simple deconvolution.
#       * The x_t-filter is practically convenient, as it can be applied directly
#         to the observed data x_t (rather than to the unobserved innovations epsilon_t).
#   - Therefore, one can specify either:
#       (i) the convolved target, applied to epsilon_t (omitting the Wold decomposition xi in the SSA call),
#           or
#      (ii) the original target, applied to x_t, and inform SSA about the data-generating process
#           by supplying the Wold decomposition xi.
#   - In the background, SSA always operates with convolved designs (the theory assumes white noise data).
#
# Target Swap: 
#   - The SSA criterion is indifferent to swapping acausal target and causal MSE predictor as targets 
#       for SSA (see literature).
#   - When targeting the acausal design, SSA acts as a predictor.
#   - When targeting the causal MSE, SSA acts as a smoother for MSE:
#       * customization of the MSE predictor by SSA
#       * SSA remains as close as possible to the benchmark MSE while respecting the ht constraint
#
# Replication: 
#   - SSA can exactly replicate the MSE solution under appropriate settings:
#       * Use the acausal target and impose the ht of MSE in the constraint
#         (equivalently, rho1 equal to the first-order ACF of the MSE predictor)
#       * Use the causal MSE as target with the same constraint
#
# Consistency:
#   - Theoretical criteria (target correlation or sign accuracy) and holding times (HT)
#     align with empirical estimates when the model xi (Wold decomposition) is correctly specified.
#   - Substantial discrepancies between empirical and theoretical values indicate
#     model misspecification (wrong xi, non-zero mean) or the absence of numerical convergence
#     (the latter is rare and can be remedied by increasing the number of iterations).
#
#----------------------------------------------------------------
# Final Remarks
#----------------------------------------------------------------
#
#   - The constraint rho1 can be freely modified in the above example:
#       * If rho1 = rho(MSE): SSA replicates the MSE solution (as shown here).
#       * If rho1 > rho(MSE): SSA yields a smoother output — common in practical
#         applications.
#       * If rho1 < rho(MSE): SSA yields less smooth output — with more zero-crossings.
#
#   - See the tutorials for practically (more) relevant applications.
#================================================================









