# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 4
#
# Replicating the Baxter–King (BK) Band-Pass Filter via the
# Smooth Sign Accuracy (SSA) Approach, and Exploring Filter Customisation
# ══════════════════════════════════════════════════════════════════════════════
#
# Key insights:
#
# 1. Limitations of the BK filter:
#    - BK is not well suited for business cycle analysis (BCA).
#    - Like the HP gap (Tutorial 2), the bandpass filter is prone to generating
#      spurious cycles.
#    - The dominant periodicity in the output is largely determined by the
#      passband's lower bound, not by the underlying data.
#
# 2. Limited gains from SSA in this context:
#    - BK output is already highly smooth (nearly sinusoidal), leaving little
#      room for SSA to improve signal extraction.
#    - Applying longer SSA windows can produce unusual and potentially
#      undesirable filter behaviour (like attenuating passband components).
#
# Practical recommendations:
#    - Use bandpass filters for BCA with caution (same concerns as HP-gap;
#      see Tutorial 2).
#    - Inspect amplitude functions to understand filter properties and
#      identify tendencies toward spurious-cycle generation.
#    - Verify that SSA meaningfully improves the signal, e.g., by reducing
#      noisy zero-crossings caused by high-frequency leakage.
#
# Summary:
#    This tutorial illustrates and highlights several counterintuitive findings.
#
# ── REFERENCES ───────────────────────────────────────────────────────────────
#
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5
#
#   Wildi, M. (2026a)
#     Sign Accuracy, Mean-Squared Error and the Rate of Zero Crossings:
#     a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547
#
# ═════════════════════════════════════════════════════════════════════════════


# ── Setup ─────────────────────────────────────────────────────────────────────

rm(list = ls())   # Clear workspace

library(xts)       # Time-series handling
library(mFilter)   # HP and BK filters
library(quantmod)  # Data retrieval (FRED)

# Load SSA helper functions
source(paste(getwd(), "/R/simple_sign_accuracy.r", sep = ""))

# Load the tau-statistic (lead/lag performance measure)
source(paste(getwd(), "/R/Tau_statistic.r", sep = ""))

# Load signal-extraction utilities (used in the JBCY paper; requires mFilter)
source(paste(getwd(), "/R/HP_JBCY_functions.r", sep = ""))


# ── Data retrieval ────────────────────────────────────────────────────────────

getSymbols('PAYEMS', src = 'FRED')

# Exclude the pandemic period: extreme observations distort filtering results.
# Convert to a plain numeric vector; xts attributes complicate SSA computations.
y   <- as.double(log(PAYEMS["/2019"]))
len <- length(y)


# ══════════════════════════════════════════════════════════════════════════════
# Example 1: Apply the BK filter to log-levels
# (Example 2 will apply BK to log-differences, i.e., growth rates / "returns")
# ══════════════════════════════════════════════════════════════════════════════

# ── 1.1  Attempt via mFilter ──────────────────────────────────────────────────
#
# mFilter requires a ts object to generate filter coefficients.
# A series of odd length is needed for a centred, symmetric (two-sided) filter.

len_f <- 1201
x     <- ts(1:len_f, frequency = 96)

# Bandpass period bounds (converted from quarterly to monthly frequency):
#   Original BK: pl = 6 quarters, pu = 32 quarters
#   Monthly equivalent: multiply by 3
pl <- 3 * 6    # Lower period bound: 18 months
pu <- 3 * 32   # Upper period bound: 96 months

L <- 201   # Filter length (must be odd for a symmetric, centred filter)

# Apply mFilter with the BK specification
filt_obj <- mFilter(as.ts(x[1:L]), filter = "BK", pl = pl, pu = pu, drift = TRUE)
filt     <- filt_obj$fmatrix   # Filter matrix: one coefficient vector per time point

# mFilter returns an L × L matrix, with a distinct filter for each observation
dim(filt)

# Extract the two-sided (symmetric) BK filter from the central column,
# which corresponds to the midpoint of the sample
bk_target <- filt[, (L + 1) / 2]

# The ideal (infinite-length) bandpass weights sum to zero, ensuring that
# any unit-root (non-stationary) component is removed from the filtered series
sum(bk_target)

# Extract the one-sided (concurrent / real-time) filter from the first column,
# constructed under a white noise assumption for future observations
fil_c <- filt[, 1]
sum(fil_c)

bk_wn <- fil_c
ts.plot(cbind(bk_target, bk_wn), col = c("black", "violet"))
mtext("Two-sided BK: centre shifted right (causal representation)", line = -1)
mtext("One-sided causal BK (mFilter output)", col = "violet", line = -2)

# ── WARNING ───────────────────────────────────────────────────────────────────
# The mFilter BK output is incorrect: the one-sided filter produced by mFilter
# is unreliable. We therefore construct the BK filter from scratch below.
# ─────────────────────────────────────────────────────────────────────────────


# ── 1.2  Build the BK filter from scratch ────────────────────────────────────

bk_weights <- function(L, pl, pu) {
  # Arguments:
  #   L  : Total number of filter weights (must be odd to ensure symmetry
  #          around lag 0)
  #   pl : Lower period bound (in observations); e.g., pl = 18 retains
  #          cycles lasting at least 18 months
  #   pu : Upper period bound (in observations); e.g., pu = 96 retains
  #          cycles lasting at most 96 months
  #
  # Returns:
  #   A numeric vector of L symmetric BK filter weights that sum to
  #   (approximately) zero, suitable for band-pass filtering of potentially
  #   non-stationary (unit-root) series.
  
  if (L %% 2 == 0) stop("L must be odd to ensure a symmetric filter.")
  
  k <- (L - 1) / 2   # Maximum lag / lead index
  j <- -k:k           # Symmetric index vector: −k, …, 0, …, k
  
  # Convert period bounds to angular frequencies (radians per observation).
  # Period p maps to angular frequency w = 2π / p.
  # w1 is the lower cutoff (long cycles); w2 is the upper cutoff (short cycles).
  w1 <- 2 * pi / pu   # Cutoff frequency corresponding to the upper period bound
  w2 <- 2 * pi / pl   # Cutoff frequency corresponding to the lower period bound
  
  # Allocate the weight vector
  b <- numeric(length(j))
  
  # Ideal band-pass weights via the inverse Fourier transform of the
  # rectangular frequency window [w1, w2]:
  #   b(0) = (w2 − w1) / π
  #   b(j) = [sin(w2·j) − sin(w1·j)] / (π·j),   j ≠ 0
  for (i in seq_along(j)) {
    if (j[i] == 0) {
      b[i] <- (w2 - w1) / pi
    } else {
      b[i] <- (sin(w2 * j[i]) - sin(w1 * j[i])) / (pi * j[i])
    }
  }
  
  # Enforce the zero-sum constraint by subtracting the mean.
  # This guarantees the filter assigns zero gain at frequency 0, preventing
  # non-stationary (unit-root) components from leaking into the output.
  b <- b - mean(b)
  
  return(b)
}

# ── Compute the two-sided BK filter ──────────────────────────────────────────

L_two_sided <- L    # Filter length (odd); already set to 201 above
pl          <- 18   # Lower period bound: 18 months (retain cycles ≥ 18 months)
pu          <- 96   # Upper period bound: 96 months (retain cycles ≤ 96 months)

bk_target <- bk_weights(L_two_sided, pl, pu)

# Verify the zero-sum property: must hold to remove a unit root
sum(bk_target)   # Expected: ≈ 0

plot(bk_target, type = "l", main = "BK filter weights (two-sided, from scratch)")


# ── 1.3  Construct the one-sided (causal) MSE filters ────────────────────────
#
# To build a one-sided filter, future (unobserved) values must be forecast.
# Two forecasting assumptions are considered below.
#
# Strategy: compute a longer two-sided BK of length (2L − 1), then truncate
# the left (future-looking) half, replacing it with the chosen forecasts.

bk_target_long <- bk_weights(2 * L - 1, pl, pu)   # Extended two-sided filter

# ── Assumption A: White Noise (WN) ───────────────────────────────────────────
# Under WN, the optimal forecast for all future values is zero.
# → Simply truncate: set future coefficients to zero (right half only).
# Note: WN is stationary, so the one-sided WN filter weights need NOT sum to zero.

bk_wn <- bk_target_long[L:(2 * L - 1)]
sum(bk_wn)   # Not necessarily ≈ 0 (WN is stationary — no unit-root cancellation needed)

ts.plot(cbind(bk_target, bk_wn), col = c("black", "violet"))
mtext("Two-sided BK (centred, acausal)", line = -1)
mtext("One-sided BK — White Noise assumption", col = "violet", line = -2)

# ── Assumption B: Random Walk (RW) ───────────────────────────────────────────
# Under RW, the optimal forecast for all future values is the last observation.
# → Collapse all future-looking coefficients onto the current lag (lag 0),
#   by assigning their combined weight to the first coefficient.
# Note: the RW has a unit root, so the one-sided RW filter weights MUST sum to zero.
# Caveat: the RW assumption is at odds with the existence of business cycles,
# but it is a widely used baseline and generally outperforms the white noise
# alternative.


bk_rw    <- bk_target_long[L:(2 * L - 1)]
bk_rw[1] <- sum(bk_target_long[1:(2 * L - 1)])   # Absorb all future weights into lag 0
sum(bk_rw)   # Expected: ≈ 0 (unit-root cancellation preserved)

ts.plot(cbind(bk_target, bk_rw), col = c("black", "violet"))
mtext("Two-sided BK (centred, acausal)", line = -1)
mtext("One-sided BK — Random Walk assumption", col = "violet", line = -2)


# ── 1.4  Visual comparison of all three filter kernels ───────────────────────

par(mfrow = c(2, 2))
ts.plot(bk_target, main = "Two-sided BK (symmetric)")
ts.plot(bk_wn,     main = "One-sided BK (White Noise assumption)")
ts.plot(bk_rw,     main = "One-sided BK (Random Walk assumption)")
par(mfrow = c(1, 1))


# ── 1.5  Amplitude functions: comparing filters against the ideal passband ────
#
# The amplitude function shows what fraction of each frequency component
# passes through the filter. An ideal bandpass has amplitude = 1 in [w1, w2]
# and 0 elsewhere.

K <- 600   # Frequency resolution: higher K → finer grid

par(mfrow = c(1, 1))

plot(
  amp_shift_func(K, bk_target, F)$amp,
  type  = "l",
  axes  = FALSE,
  xlab  = "Frequency",
  ylab  = "Amplitude",
  col   = "blue",
  main  = "Amplitude functions: two-sided vs. one-sided BK filters"
)
lines(amp_shift_func(K, bk_rw, F)$amp, col = "red")
lines(amp_shift_func(K, bk_wn, F)$amp, col = "orange")

mtext("Two-sided BK",   col = "blue",   line = -1)
mtext("One-sided — RW", col = "red",    line = -2)
mtext("One-sided — WN", col = "orange", line = -3)

# Mark the passband boundaries (18 and 96 months)
abline(v = 2 * K / pl + 1)   # Lower period bound (pl = 18 months)
abline(v = 2 * K / pu + 1)   # Upper period bound (pu = 96 months)

axis(1,
     at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# ── Interpretation ────────────────────────────────────────────────────────────
# - All three filters pass cycles in the 18–96 month band, as intended.
# - Ripples (Gibbs phenomenon): finite truncation of the ideal infinite filter
#     introduces oscillations; increasing L reduces them in the two-sided case.
# - Two-sided BK and one-sided RW both show zero amplitude at frequency 0,
#     confirming unit-root cancellation (weights sum to zero).
# - One-sided WN does NOT vanish at frequency 0, reflecting that WN is
#     stationary and no unit-root cancellation is required.
# - Differences between the WN and RW one-sided filters are modest: both
#     suffer from amplitude shrinkage within the passband due to zero-padding.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 1.2  Transform the BK filter for use on first-differenced data
# ══════════════════════════════════════════════════════════════════════════════
#
# SSA requires a stationary input series. Since log-levels of PAYEMS are
# non-stationary (unit root), the filter must be reformulated to operate on
# first differences rather than levels.
#
# This transformation is standard practice for bandpass filters applied to
# integrated (I(1)) series. For the theoretical underpinning see Section 2.3
# and Proposition 4 of the JBCY paper.

# Note: we supply the one-sided RW MSE filter bk_rw

bk_diff <- conv_with_unitroot_func(bk_rw)$conv

par(mfrow = c(2, 1))
ts.plot(bk_diff, main = "BK filter coefficients — formulated for first differences")
ts.plot(bk_rw,   main = "BK filter coefficients — formulated for levels")

# ── Verify equivalence of the two formulations ───────────────────────────────
# Filtering levels with bk_rw and filtering first differences with bk_diff
# must produce identical cycles. Any discrepancy would indicate a coding error.

x          <- diff(y)
cycle      <- na.exclude(filter(y, bk_rw,   side = 1))   # Filter applied to levels
cycle_diff <- na.exclude(filter(x, bk_diff, side = 1))   # Filter applied to differences

par(mfrow = c(1, 1))
ts.plot(cycle_diff, col = "blue",
        main = "Equivalence check: BK on differences (blue) vs. levels (red)")
lines(cycle[2:length(cycle)], col = "red")
# Expected result: the two series overlap exactly.


# ══════════════════════════════════════════════════════════════════════════════
# 1.3  Holding time and data properties
# ══════════════════════════════════════════════════════════════════════════════

# ── Theoretical holding time (white noise input assumption) ───────────────────
# The holding time measures the average number of periods between successive
# zero-crossings; larger values indicate a smoother output series, see Wildi (2024, 2026a).

ht_bk_diff_obj <- compute_holding_time_func(bk_diff)

# The BK filter produces very long holding times (very smooth output),
# considerably smoother than the Hamilton filter (Tutorial 3) or the
# concurrent HP filter (Tutorial 2).
ht_bk_diff_obj$ht

# ── Empirical holding time (data-driven) ──────────────────────────────────────
# The empirical holding time, computed from the actual filtered cycle, is even
# longer than the theoretical value. This is because the input (differenced
# log-PAYEMS) is not white noise — it exhibits positive autocorrelation that
# further smooths the filtered output.

compute_empirical_ht_func(cycle_diff)

# Centering the cycle (removing its mean) halves the empirical holding time:
#   Centering emphasizes mean-crossings (instead of zero crossings)
#   SSA emphasizes mean crossings
# The HT is huge!
compute_empirical_ht_func(cycle_diff - mean(cycle_diff))

# ── Inspect stationarity and autocorrelation of the input series ──────────────
# The plot and ACF below confirm that differenced log-PAYEMS is stationary but
# positively autocorrelated — it is not white noise, which explains why the
# empirical holding time exceeds the theoretical (white noise) benchmark.

par(mfrow = c(2, 1))
ts.plot(x, ylim = c(-0.05, 0.05), main = "Log-differenced PAYEMS (monthly growth rate)")
abline(h = 0)
acf(x, main = "ACF of log-differenced PAYEMS")
par(mfrow = c(1, 1))



# ══════════════════════════════════════════════════════════════════════════════
# 1.4  Autocorrelation structure and ARMA model fit
# ══════════════════════════════════════════════════════════════════════════════
#
# Fit an ARMA(1,1) model to the differenced log-PAYEMS series (x).
# An ARMA(1,1) is well suited for series with a weak but slowly decaying ACF,
# which is the typical pattern observed here. The near-cancellation of the AR
# and MA roots produces a small but persistent autocorrelation structure.

ar_order <- 1
ma_order <- 1
estim_obj <- arima(x, order = c(ar_order, 0, ma_order))

# Inspect estimated coefficients and residual diagnostics.
# The near-cancellation of AR and MA roots is the signature of a slowly
# decaying ACF that cannot be captured well by a (parsimonious) pure AR or pure MA model.
estim_obj
tsdiag(estim_obj)

# ── Wold decomposition (MA-inversion of the fitted ARMA) ─────────────────────
# Convert the ARMA model to its infinite moving-average (Wold) representation.
# The resulting sequence xi = {1, xi_1, xi_2, ...} defines how each white-noise
# innovation epsilon_t propagates forward through the data-generating process.
# xi is used downstream to compute the theoretically correct holding time and
# to supply the correct spectral weighting inside SSA.

xi <- c(1, ARMAtoMA(
  ar      = estim_obj$coef[1:ar_order],
  ma      = estim_obj$coef[ar_order + 1:ma_order],
  lag.max = L - 1
))

par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold (MA) coefficients of fitted ARMA(1,1)")

# ── Holding time based on Wold decomposition ─────────────────────────────────
# Convolve xi with bk_diff to obtain the composite filter as seen from the
# white-noise innovations epsilon_t. The holding time computed from this
# convolved filter should match the empirical holding time of the BK cycle,
# because epsilon_t is (approximately) white noise.

bk_conv         <- conv_two_filt_func(xi, bk_diff)$conv
ht_bk_conv_obj  <- compute_holding_time_func(bk_conv)

# Theoretical holding time (derived from the convolved filter)
ht_bk_conv_obj$ht

# Empirical holding time (computed from the actual filtered cycle).
# The cycle is centred to satisfy the zero-mean assumption underlying the HT formula.
compute_empirical_ht_func(cycle_diff - mean(cycle_diff))
# Post-adjustment (i.e., mean centering and autocorrelation), empirical and expected HT estimates align closely;
# remaining discrepancies are consistent with random variation.

# ── Summary ───────────────────────────────────────────────────────────────────
# We now have all ingredients needed to apply SSA to the BK filter:
#   1. Stationarity: achieved by working with bk_diff (filter on first differences)
#      rather than bk_rw (filter on levels).
#   2. Spectral weighting: provided by xi (Wold decomposition of the ARMA model).


# ══════════════════════════════════════════════════════════════════════════════
# 1.5  Diagnostics: squared amplitude functions and spurious-cycle detection
# ══════════════════════════════════════════════════════════════════════════════
#
# Three representations of the same filter are compared. All three produce
# identical output cycles, but they differ in the input series they expect:
#
#   bk_rw   → applied to log-levels  (non-stationary)
#   bk_diff → applied to xt = log-differences (ARMA process)
#   bk_conv → applied to epsilon_t (white-noise innovations of xt)
#
# Because epsilon_t is (approximately) white noise with a flat spectral density,
# the squared amplitude of bk_conv directly reflects the spectral density of
# the common filter output — i.e., the BK cycle itself.

K <- 600   # Frequency resolution

amp_obj_rw  <- amp_shift_func(K, bk_rw,   F)
amp_obj_x   <- amp_shift_func(K, bk_diff, F)
amp_obj_eps <- amp_shift_func(K, bk_conv, F)

# Scale amplitudes for visual comparability (column-wise, no centering)
mplot <- scale(
  cbind(amp_obj_rw$amp^2, amp_obj_x$amp^2, amp_obj_eps$amp^2),
  scale  = TRUE,
  center = FALSE
)
colnames(mplot) <- c("BK — levels", "BK — differences (xt)", "BK — innovations (ε_t)")
colo <- c("blue", "darkgreen", "red")

par(mfrow = c(1, 1))
plot(
  mplot[, 1], type = "l", axes = FALSE,
  xlab = "Frequency", ylab = "Scaled squared amplitude",
  main = "Squared scaled amplitude functions: one-sided BK (three input representations)",
  ylim = c(min(mplot), max(mplot)),
  col  = colo[1]
)
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# ── Interpretation ────────────────────────────────────────────────────────────
#
# Spurious cycle (red / ε_t curve):
#   Since ε_t is white noise its spectrum is flat. The squared amplitude of
#   bk_conv therefore equals the spectral density of the BK cycle output.
#   The pronounced peak in this curve reveals that the BK cycle is dominated
#   by a single frequency determined by the LOWER cutoff of the passband (pl),
#   NOT by any cyclical feature present in the data. This is a spurious cycle
#   — an artefact of the filter specification, not a data-driven signal.
#   It also explains the remarkable smoothness of the BK cycle: nearly
#   sinusoidal output with a very long holding time.
#
# Amplitude at frequency zero:
#   - bk_rw (blue) vanishes at frequency 0: the zero-sum constraint successfully
#       cancels the unit root of the log-level series.
#   - bk_diff and bk_conv (green and red) do NOT vanish at frequency 0, because
#       the unit root of the levels series is factored out when differencing.
#       Any non-zero content at frequency 0 in the returns (e.g. a time-varying
#       drift) will therefore leak through these filters.
#   - The drift in log-PAYEMS is not constant over time (changing trend growth),
#       so this leakage is practically relevant and will manifest as unusual
#       long-horizon forecasts (discussed further below).
#
# Implications for BCA:
#   1. The Hamilton filter on levels (Tutorial 3) is a lowpass: it does NOT
#        inherently generate spurious cycles and is preferable for BCA.
#   2. The HP lowpass applied to log-differences (Tutorial 2, JBCY paper) shares this
#        desirable property.
#   3. The HP gap applied to levels DOES generate a spurious cycle (Tutorial 2) 
#       and is therefore not recommended for BCA.
#
# Conclusion:
#   The BK filter is not a reliable BCA tool. The analysis from here onward is
#   instructive for understanding SSA mechanics, NOT for BCA. We are examining
#   what happens when SSA is applied to an already very smooth signal.


# ══════════════════════════════════════════════════════════════════════════════
# 1.6  Apply SSA to the BK filter
# ══════════════════════════════════════════════════════════════════════════════
#
# Context: BK already produces a very smooth cycle.
# Applying SSA here is instructive but does NOT improve BCA. Since BK output
# has few noisy zero-crossings to begin with, SSA cannot meaningfully reduce
# them. The exercise below illustrates what happens when SSA is engrafted onto
# an already smooth filter.

# Current theoretical holding time of the BK cycle
ht_bk_conv_obj$ht   # ≈ 2.5 years

# Target a modest 20 % increase in holding time.
# Note: because BK's holding time is already very long, even a 20 % increase
# requires a relatively long filter length L.
ht   <- 1.2 * ht_bk_conv_obj$ht

# Translate the holding-time target into the corresponding autocorrelation
# constraint parameter rho1 required by SSA
rho1 <- compute_rho_from_ht(ht)

# Target filter: bk_diff (the BK filter formulated for first-differenced input).
# If we were targeting epsilon_t directly, bk_conv would be used instead.
gammak_generic   <- bk_diff

# Forecast horizon: nowcast (no look-ahead)
forecast_horizon <- 0

# Run SSA. Supplying xi (Wold decomposition) ensures the optimisation accounts
# for the ARMA autocorrelation structure of xt, not just white-noise dynamics.
SSA_obj_bk_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract the two SSA filter representations:
#   ssa_eps → filter applied to white-noise innovations ε_t
#   ssa_x   → filter applied to log-differences xt
SSA_filt_bk_diff_eps <- SSA_obj_bk_diff$ssa_eps
SSA_filt_bk_diff_x   <- SSA_obj_bk_diff$ssa_x


# ── Compare BK and SSA filters in the xt domain ──────────────────────────────
# Both bk_diff and SSA_filt_bk_diff_x are applied to xt; comparing their
# coefficients directly shows how SSA modifies the filter shape.

mplot <- cbind(bk_diff, SSA_filt_bk_diff_x)
par(mfrow = c(1, 1))
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"))
mtext("BK  (applied to differences)",  col = "black", line = -1)
mtext("SSA filter (applied to differences)", col = "blue",  line = -2)


# ── Compare BK and SSA filters in the ε_t domain ─────────────────────────────
# Both bk_conv and SSA_filt_bk_diff_eps are applied to ε_t; this comparison
# is more informative for spectral analysis because ε_t is approximately white
# noise, making the squared amplitude directly interpretable as a power spectrum.

mplot <- cbind(bk_conv, SSA_filt_bk_diff_eps)
par(mfrow = c(1, 1))
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"))
mtext("BK target (applied to innovations ε_t)",  col = "black", line = -1)
mtext("SSA filter (applied to innovations ε_t)", col = "blue",  line = -2)


# ── Verify the holding-time constraint ───────────────────────────────────────
# The holding time must be checked using the ε_t representation (ssa_eps),
# because the HT formula assumes white-noise input. Checking against ssa_x
# gives a misleading result when xt is autocorrelated.

# HT from ssa_x: will NOT match target ht (xt is not white noise)
ht_obj <- compute_holding_time_func(SSA_filt_bk_diff_x)
ht_obj$ht   # Expected: does NOT equal ht
ht

# HT from ssa_eps: WILL match target ht (ε_t is approximately white noise)
ht_obj <- compute_holding_time_func(SSA_filt_bk_diff_eps)
ht_obj$ht   # Expected: ≈ ht
ht


# ── Verify internal consistency: ssa_eps = conv(ssa_x, xi) ───────────────────
# By construction, convolving ssa_x with the Wold coefficients xi must reproduce
# ssa_eps. This identity confirms that the two filter representations are
# mutually consistent and that no numerical errors were introduced.

ts.plot(
  scale(
    cbind(
      conv_two_filt_func(SSA_filt_bk_diff_x, xi)$conv[1:L],
      SSA_filt_bk_diff_eps
    ),
    center = FALSE, scale = TRUE
  ),
  col  = c("blue", "red"),
  main = "Consistency check: conv(ssa_x, xi) and ssa_eps (should overlap exactly)"
)
# Expected: the two scaled series lie on top of each other.


# ══════════════════════════════════════════════════════════════════════════════
# 1.7  Filter the series and compare empirical holding times
# ══════════════════════════════════════════════════════════════════════════════

# Apply the SSA and BK filters to the log-differenced series
SSA_out <- filter(x, SSA_filt_bk_diff_x, side = 1)
bk_out  <- filter(x, bk_diff,            side = 1)

# ── Empirical holding times ───────────────────────────────────────────────────

# Compute empirical mean crossings (not zero crossings)
compute_empirical_ht_func(scale(SSA_out))   # Empirical HT of SSA output
ht                                   # SSA target HT

compute_empirical_ht_func(scale(bk_out))    # Empirical HT of BK output
ht_bk_conv_obj$ht                    # Theoretical BK HT

# ── Visual comparison ─────────────────────────────────────────────────────────
# The plot confirms that the BK cycle is already extremely smooth: SSA adds
# virtually no additional smoothness, and the two series are nearly identical.

mplot <- na.exclude(cbind(SSA_out, bk_out))
colo  <- c("blue", "red")

par(mfrow = c(1, 1))
ts.plot(mplot[, 1], col = colo[1],
        main = "SSA vs. BK filtered cycle (applied to log-differences)")
lines(mplot[, 2], col = colo[2])
abline(h = 0)
mtext("BK",  col = "red",  line = -1)
mtext("SSA", col = "blue", line = -2)



# ══════════════════════════════════════════════════════════════════════════════
# 1.8  Timeliness: using the forecast horizon to reduce lag
# ══════════════════════════════════════════════════════════════════════════════
#
# The previous plot suggested that the SSA nowcast possible slightly lags the BK.
# Increasing the forecast horizon (delta in the JBCY paper) instructs SSA to
# correlate with a lead-shifted version of the target, producing a faster
# (less lagging) filter without sacrificing smoothness.

forecast_horizon <- 12   # 12-month ahead forecast

SSA_obj_bk_diff           <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)
SSA_filt_bk_diff_x_forecast <- SSA_obj_bk_diff$ssa_x

# Apply the forecast filter
SSA_out_forecast <- filter(x, SSA_filt_bk_diff_x_forecast, side = 1)
bk_out           <- filter(x, bk_diff, side = 1)

# ── Raw comparison (unshifted) ────────────────────────────────────────────────
# The forecast output appears to lead the BK cycle by approximately one year.
# This is expected: a 12-month ahead filter anticipates future values, so its
# output at time t corresponds to what BK would show at t + 12.

mplot <- na.exclude(cbind(SSA_out, SSA_out_forecast, bk_out))
colo  <- c("blue", "darkgreen", "red")

par(mfrow = c(1, 1))
ts.plot(mplot[, 1], col = colo[1], ylim = c(min(mplot), max(mplot)),
        main = "SSA nowcast vs. 12-month forecast vs. BK (unshifted)")
lines(mplot[, 2], col = colo[2])
lines(mplot[, 3], col = colo[3])
abline(h = 0)
mtext("SSA nowcast",                                col = colo[1], line = -1)
mtext(paste("SSA forecast: delta =", forecast_horizon), col = colo[2], line = -2)
mtext("BK",                                         col = colo[3], line = -3)

# ── Shifted comparison ────────────────────────────────────────────────────────
# Shift the forecast output forward by forecast_horizon periods so that it is
# time-aligned with the nowcast and BK. The three series should now track
# each other closely, confirming that the forecast filter is working correctly.

mplot_shift <- na.exclude(cbind(
  SSA_out[(1 + forecast_horizon):length(SSA_out_forecast)],
  SSA_out_forecast[1:(length(SSA_out) - forecast_horizon)],
  bk_out[(1 + forecast_horizon):length(SSA_out_forecast)]
))

par(mfrow = c(1, 1))
ts.plot(mplot_shift[, 1], col = colo[1], ylim = c(min(mplot_shift), max(mplot_shift)),
        main = paste("Shifted comparison: SSA forecast advanced by", forecast_horizon, "months"))
lines(mplot_shift[, 2], col = colo[2])
lines(mplot_shift[, 3], col = colo[3])
abline(h = 0)
mtext("SSA nowcast",                                col = colo[1], line = -1)
mtext(paste("SSA forecast: delta =", forecast_horizon), col = colo[2], line = -2)
mtext("BK",                                         col = colo[3], line = -3)



# ══════════════════════════════════════════════════════════════════════════════
# 1.9  Long forecast horizon: 4-year ahead (half the upper passband period)
# ══════════════════════════════════════════════════════════════════════════════
#
# Setting the forecast horizon to pu/2 (half the longest cycle in the passband)
# is a natural benchmark: at this horizon the target cycle has completed 
# a half-turn, so a perfect forecast filter would produce output that is ~180°
# out of phase with the nowcast.
# This extreme horizon stress-tests the SSA framework and exposes the impact
# of model misspecification at long horizons.

forecast_horizon <- pu / 2   # 48 months (4 years) for pu = 96

SSA_obj_bk_diff_long           <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)
SSA_filt_bk_diff_x_forecast_long <- SSA_obj_bk_diff_long$ssa_x

# Apply the long-horizon forecast filter
SSA_out_forecast_long <- filter(x, SSA_filt_bk_diff_x_forecast_long, side = 1)

# ── Visual comparison (unshifted) ─────────────────────────────────────────────
# The long-horizon forecast output is smaller in scale (zero-shrinkage: the
# 4-year ahead prediction problem is much harder) and appears roughly out of
# phase with the BK target — consistent with the half-turn argument above.

mplot <- na.exclude(cbind(SSA_out, SSA_out_forecast_long, bk_out))
colo  <- c("blue", "darkgreen", "red")

par(mfrow = c(1, 1))
ts.plot(mplot[, 1], col = colo[1], ylim = c(min(mplot), max(mplot)),
        main = paste("4-year ahead SSA forecast (delta =", forecast_horizon, "months)"))
lines(mplot[, 2], col = colo[2])
lines(mplot[, 3], col = colo[3])
abline(h = 0)
mtext("SSA nowcast",                                    col = colo[1], line = -1)
mtext(paste("SSA forecast: delta =", forecast_horizon), col = colo[2], line = -2)
mtext("BK",                                             col = colo[3], line = -3)

# ── Filter coefficient comparison ─────────────────────────────────────────────
# Comparing the long-horizon forecast filter (green) with the nowcast filter
# (blue) reveals the phase shift introduced by the 4-year ahead horizon.
# The coefficient pattern of the forecast filter mirrors that of the nowcast,
# consistent with an approximate 180° phase rotation of the dominant cycle.

ts.plot(SSA_filt_bk_diff_x_forecast_long,
        col = "darkgreen",
        ylim = c(min(SSA_filt_bk_diff_x_forecast_long), max(SSA_filt_bk_diff_x)),
        main = "Filter coefficients: SSA nowcast (blue) vs. 4-year ahead forecast (green)")
lines(SSA_filt_bk_diff_x, col = "blue")


# To better understand the  long-term out-of-phase forecast, we examine the
# amplitude and phase-shift functions of the filters involved.
#----------------------------------------
# 1.10. Compute amplitude and phase-shift functions

K <- 600
amp_obj_SSA_now <- amp_shift_func(K, as.vector(SSA_filt_bk_diff_x), F)
amp_obj_SSA_for <- amp_shift_func(K, as.vector(SSA_filt_bk_diff_x_forecast), F)
amp_obj_bk      <- amp_shift_func(K, bk_diff, F)

par(mfrow = c(2, 1))
mplot <- cbind(amp_obj_SSA_now$amp, amp_obj_SSA_for$amp, amp_obj_bk$amp)

# Scale all amplitude functions so that their peak values are equal,
# enabling direct visual comparison across filters.
mplot[, 2] <- mplot[, 2] / max(mplot[, 2]) * max(mplot[, 1])
mplot[, 3] <- mplot[, 3] / max(mplot[, 3]) * max(mplot[, 1])
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon, ")", sep = ""),
  "BK"
)

# The amplitude functions reveal that the BK design is poorly suited for BCA:
# the one-sided filter exhibits a strong, narrow peak at the lower boundary of
# the passband, effectively generating a spurious cycle at frequency 2*pi/pu.
colo <- c("blue", "darkgreen", "red")
plot(mplot[, 1], type = "l", axes = F, xlab = "Frequency", ylab = "",
     main = paste("Amplitude BK", sep = ""),
     ylim = c(min(mplot), max(mplot)), col = colo[1])
lines(mplot[, 2], col = colo[2])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
if (ncol(mplot) > 1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

mplot <- cbind(amp_obj_SSA_now$shift, amp_obj_SSA_for$shift, amp_obj_bk$shift)
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon, ")", sep = ""),
  "BK"
)

plot(mplot[, 1], type = "l", axes = F, xlab = "Frequency", ylab = "",
     main = paste("Shift (phase-lag) ", sep = ""),
     ylim = c(-6, 20), col = colo[1])
lines(mplot[, 2], col = colo[2])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
if (ncol(mplot) > 1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Key observations from the amplitude and phase-shift functions:
#
# 1. SSA amplitude functions: atypical and noteworthy findings
#   - The original BK filter is already very smooth:
#       - Unlike HP (tutorial 2) or the Hamilton filter (tutorial 3),
#         BK does not require additional noise suppression.
#       - The BK amplitude function is very close to zero at higher frequencies,
#         indicating strong attenuation of high-frequency content.
#       - Consequently, SSA cannot reduce the number of zero-crossings further
#         by additional high-frequency damping — BK already handles this effectively.
#       - Counter-intuitively, the SSA amplitude is marginally larger than BK
#         at higher frequencies — an atypical and unexpected result.
#   - How does SSA then achieve fewer zero-crossings?
#
# 2. SSA reduces zero-crossings by attenuating lower frequencies within
#    the BK passband:
#   - The SSA amplitude functions fall below those of BK on both sides of the
#     upper bandpass boundary at 2*pi/18.
#       - Again, this is atypical and unexpected behavior.
#       - However, it is statistically purposeful: it represents the most
#         effective strategy for the filter to further reduce zero-crossings.
#   - We conclude that the noise-suppression role of SSA is unnecessary in this
#     BK context and is, to some extent, counterproductive, as it distorts the
#     passband in an undesirable way.
#
# Interpretation of the anomalous long-horizon forecasts (forecast_horizon = pu/2):
#
#   1. Due to the narrow amplitude peak at 2*pi/pu, the BK filter output closely
#      resembles a regular, sinusoidal-like spurious cycle at frequency 2*pi/pu:
#       - This type of signal is very smooth.
#       - Regular periodic patterns are relatively straightforward to forecast:
#         one simply adds a phase shift corresponding to the forecast horizon.
#       - A forecast horizon of pu/2 corresponds to half the spurious cycle length;
#         consequently, the 4-year-ahead forecast is systematically out-of-phase.
# =================================================================================
# BK: an instructive and counter-intuitive cautionary example!
# =================================================================================

