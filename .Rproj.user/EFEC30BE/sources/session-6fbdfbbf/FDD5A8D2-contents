# ════════════════════════════════════════════════════════════════════
# TUTORIAL 0.1 — SSA: THE FORECAST TRILEMMA
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────
# This tutorial presents and discusses a fundamental forecast trilemma
# underpinning the SSA framework.
#
# Theoretical background:
#   Wildi, M. (2026a) Sign Accuracy, Mean-Squared Error and the Rate
#     of Zero Crossings: a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547

# Application to business-cycle analysis
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

# ─────────────────────────────────────────────────────────────────

rm(list=ls())

# Load all relevant SSA-functions
source(paste(getwd(),"/R/SSA.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R utility functions/Tau_statistic.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R utility functions/HP_JBCY_functions.r",sep=""))


#============================================================
# EXercise: FORECAST TRILEMMA AND THE MSE APPROACH
#============================================================
# Data-Generating Process (DGP):
#   x_t = ε_t  (white noise, iid)
#   z_t = Γ(x_t), where Γ is a symmetric equally weighted
#         moving-average filter (the TARGET filter)
#============================================================


#------------------------------------------------------------
# TARGET FILTER DEFINITION
#------------------------------------------------------------
# A symmetric (two-sided, acausal) equally weighted MA filter
# of length L = 11. Each coefficient equals 1/L.
# This serves as the benchmark "ideal" smoother.
L     <- 11
gamma <- rep(1/L, L)

# Plot the target filter coefficients against their lag indices
plot(gamma, axes = F, type = "l",
     xlab = "Lag",
     ylab = "Filter coefficients",
     main = "Equally weighted acausal (two-sided) smoothing filter")
axis(1, at      = 1:length(gamma),
     labels  = (-(length(gamma) + 1)/2) + 1:length(gamma))
axis(2)
box()


#------------------------------------------------------------
# SIMULATE WHITE NOISE INPUT
#------------------------------------------------------------
set.seed(231)
len     <- 120
sigma   <- 1
epsilon <- sigma * rnorm(len)
x       <- epsilon

# Verify absence of autocorrelation (expected for white noise)
acf(x)


#------------------------------------------------------------
# APPLY SYMMETRIC AND ONE-SIDED FILTERS
#------------------------------------------------------------
# side = 2 → symmetric (two-sided, acausal): uses future values
# side = 1 → one-sided (causal, real-time):  uses only past values
y_sym       <- filter(x, gamma, side = 2)
y_one_sided <- filter(x, gamma, side = 1)

tail(cbind(y_sym, y_one_sided))

# Key observations:
#   y_sym       : centered output; produces NAs at both sample ends
#                 due to missing future values
#   y_one_sided : available up to sample end but is right-shifted
#                 (delayed) relative to y_sym
ts.plot(cbind(y_sym, y_one_sided),
        col = c("black", "black"), lty = 1:2,
        main = "Two-sided (solid) vs one-sided (dashed) filter output")


#------------------------------------------------------------
# MSE-OPTIMAL PREDICTOR
#------------------------------------------------------------
# Goal: estimate y_sym at the current sample end (t = len)
#
# Forecast horizon conventions:
#   delta = 0  → nowcast  (estimate at current time)
#   delta > 0  → forecast (estimate future values)
#   delta < 0  → backcast (estimate past values)
#
# MSE-optimal strategy:
#   Replace unknown future values x_{t+j} with their forecasts.
#   For white noise, E[x_{t+j}] = 0  ∀ j > 0.
#   ⇒ The MSE filter is simply the TRUNCATED one-sided version
#     of gamma (zeroing out future coefficients).
delta <- 0
b_mse <- gamma[((L + 1)/2 + delta):L]
plot(c(rep(NA,delta+(L-1)/2),b_mse), axes = F, type = "l",
     xlab = "Lag",
     ylab = "Filter coefficients",
     main = "MSE (one-sided) nowcast",col="green")
axis(1, at      = 1:length(gamma),
     labels  = (-(length(gamma) + 1)/2) + 1:length(gamma))
axis(2)
box()


# Apply MSE predictor and recompute symmetric target
y_mse <- filter(x, b_mse, side = 1)
y_sym <- filter(x, gamma, side = 2)

short_sample <- cbind(y_sym, y_mse)


#------------------------------------------------------------
# VISUAL COMPARISON: TARGET VS MSE PREDICTOR
#------------------------------------------------------------
# Zero-crossings (sign changes) are marked as vertical lines.
# More frequent crossings → noisier / less smooth output.
ts.plot(short_sample,
        col = c("black", "green"), lty = 1:2,
        main = "Target (black/solid) vs MSE predictor (green/dashed)")
abline(h = 0)

# Zero-crossings of MSE predictor (green)
abline(v = 1 + which(sign(y_mse[2:len]) != sign(y_mse[1:(len-1)])),
       col = "green")

# Zero-crossings of symmetric target (black)
abline(v = 1 + which(sign(y_sym[2:len]) != sign(y_sym[1:(len-1)])),
       col = "black")

# Observation:
#   The MSE predictor is visibly delayed (right-shifted) and
#   noisier (more frequent zero-crossings) than the target.


#------------------------------------------------------------
# SMOOTHNESS METRIC: EMPIRICAL HOLDING TIME (HT)
#------------------------------------------------------------
# Holding time (ht) = mean duration between consecutive
# zero-crossings. Higher ht → smoother signal.
#
# Function: compute_empirical_ht_func(x)
#   Input : univariate time series x
#   Output: empirical_ht (average inter-crossing interval)
compute_empirical_ht_func <- function(x) {
  x          <- na.exclude(x)
  len        <- length(x) - 1
  empirical_ht <- len / length(which(sign(x[2:len]) != sign(x[1:(len-1)])))
  return(list(empirical_ht = empirical_ht))
}

# Use a longer sample for numerically stable ht estimates
set.seed(16)
len     <- 12000
epsilon <- sigma * rnorm(len)
x       <- epsilon
y_mse   <- filter(x, b_mse, side = 1)
y_sym   <- filter(x, gamma, side = 2)

# Empirical holding times
compute_empirical_ht_func(y_sym)   # Target:    smoother → higher ht
compute_empirical_ht_func(y_mse)   # Predictor: noisier  → lower  ht

# Relative smoothness: target vs predictor
# Ratio > 1 confirms the predictor has ~34% more zero-crossings
compute_empirical_ht_func(y_sym)$empirical_ht /
  compute_empirical_ht_func(y_mse)$empirical_ht


#------------------------------------------------------------
# THEORETICAL HOLDING TIME (WHITE NOISE INPUT)
#------------------------------------------------------------
# Computes the TRUE (population) holding time of a filter
# applied to white noise input, based on the lag-1
# autocorrelation of the filtered output.
# See: Wildi (2024), Equation 3.
#
# Function: compute_holding_time_func(b)
#   Input : filter coefficient vector b
#   Output: ht        (theoretical holding time)
#           rho_ff1   (lag-1 autocorrelation of filtered output)
compute_holding_time_func <- function(b) {
  if (length(b) > 1) {
    # Lag-1 autocorrelation of the filtered output
    rho_ff1 <- b[1:(length(b)-1)] %*% b[2:length(b)] / sum(b^2)
    # Theoretical holding time (primary formula)
    ht <- 1 / (2 * (0.25 - asin(rho_ff1) / (2*pi)))
    # Equivalent alternative expression (disabled)
    if (F)
      ht <- pi / acos(rho_ff1)
  } else {
    # Degenerate case: scalar filter → minimal smoothing
    ht      <- 2
    rho_ff1 <- 0
  }
  return(list(ht = ht, rho_ff1 = rho_ff1))
}

# Theoretical ht for target and MSE predictor
compute_holding_time_func(gamma)$ht   # Target smoother
compute_holding_time_func(b_mse)$ht   # MSE predictor

# Note: empirical ht converges to theoretical ht as n → ∞


#------------------------------------------------------------
# TIMELINESS METRIC: EMPIRICAL SHIFT (ZERO-CROSSING LAG)
#------------------------------------------------------------
# Estimate the relative delay of the MSE predictor vs target
# by finding the shift that minimizes the distance between
# their respective zero-crossing patterns.
# The minimizing shift ≈ lead/lag in periods (Wildi, 2024).
filter_mat <- cbind(y_sym, y_mse)
compute_min_tau_func(filter_mat)

# Result: predictor lags target by approximately 3 periods

# Align predictor by shifting it 3 periods to the left
short_sample_shifted <- cbind(
  short_sample[1:(nrow(short_sample) - 3), 1],   # target (unshifted)
  short_sample[4:nrow(short_sample),          2]  # predictor (advanced)
)

# Visual check: zero-crossings should align after shifting
ts.plot(short_sample_shifted,
        col = c("black", "green"), lty = 1:2,
        main = "Target vs MSE predictor (shifted 3 periods left)")
abline(h = 0)


#------------------------------------------------------------
# TIMELINESS METRIC: FREQUENCY-DOMAIN PHASE SHIFT
#------------------------------------------------------------
# The phase shift of the MSE predictor's transfer function
# quantifies delay as a function of frequency.
# Positive shift → filter output lags behind the input signal.
K     <- 600
shift <- amp_shift_func(K, b_mse, F)$shift

plot(shift, type = "l", axes = F,
     xlab = "Frequency",
     ylab = "Phase shift (periods)",
     main = "Frequency-domain phase shift of MSE predictor")
axis(1, at     = 1 + 0:6 * K/6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Interpretation:
#   Low-frequency components (0 ≤ ω ≤ π/3) are delayed by
#   2.5 periods — consistent with the
#   time-domain zero-crossing estimate above.


#------------------------------------------------------------
# ACCURACY METRIC: MEAN SQUARED ERROR (MSE)
#------------------------------------------------------------
# Empirical MSE (sample estimate)
mean((y_sym - y_mse)^2, na.rm = TRUE)

# Theoretical MSE (analytical formula):
#   Error arises from replacing future x_{t+j} with 0.
#   Since x_t ~ WN(0, σ²), the squared contributions accumulate
#   over the (L-1)/2 + delta truncated future coefficients.
sum(sigma^2 * gamma[1:((L-1)/2 + delta)]^2)

# Note: empirical MSE converges to theoretical MSE as n → ∞


#============================================================
# TRILEMMA SUMMARY TABLE: MSE vs SHIFT vs HOLDING TIME
#============================================================
# Evaluates all three forecast performance criteria across
# the full range of forecast horizons (delta).
#
# Forecast trilemma: accuracy (MSE), timeliness (shift), and
# smoothness (ht) form an interdependent triplet — improving
# one necessarily degrades at least one of the others.
#============================================================

delta_vec <- -((L-1)/2):((L-1)/2)
ht <- mse <- shift <- NULL

for (i in 1:length(delta_vec)) {
  delta <- delta_vec[i]
  
  # MSE-optimal filter for this forecast horizon
  b_mse <- gamma[((L+1)/2 + delta):L]
  
  # 1. SMOOTHNESS: theoretical holding time
  ht <- c(ht, compute_holding_time_func(b_mse)$ht)
  
  # 2. TIMELINESS: low-frequency phase shift (ω ≈ 0)
  shift <- c(shift, amp_shift_func(K, b_mse, F)$shift[1])
  
  # 3. ACCURACY: theoretical MSE
  if (((L-1)/2 + delta) > 0) {
    # Positive horizon: future values are unknown → MSE > 0
    mse <- c(mse, sum(sigma^2 * gamma[1:((L-1)/2 + delta)]^2))
  } else {
    # Non-positive horizon: all values observed → perfect fit
    mse <- c(mse, 0)
  }
}

# Assemble results into a labelled summary table
table           <- rbind(ht, shift, mse)
rownames(table) <- c("Holding Time (smoothness)",
                     "Shift (timeliness)",
                     "MSE (accuracy)")
colnames(table) <- paste0("delta=", delta_vec)

# Display trilemma table
# Interpretation:
#   → Moving right (larger delta): MSE ↑, ht ↓, |shift| ↓
#   → Moving left  (smaller delta): MSE ↓, ht ↑, |shift| ↑
#   → No column simultaneously optimizes all three criteria
table


# ============================================================
# DISCUSSION: MSE FILTER & FORECAST TRILEMMA
# ============================================================
# The classical MSE filter optimizes predictive accuracy alone,
# ignoring smoothness and timeliness. As the forecast horizon
# (delta) increases:
#   → MSE increases        (bottom row of output table)
#   → Holding time drops   (filter becomes noisier)
#   → Delay/shift reduces  (signal appears more timely)
#
# These trade-offs illustrate the core tension known as the
# FORECAST TRILEMMA: accuracy, smoothness, and timeliness
# cannot all be simultaneously improved.
# ============================================================


# ============================================================
# SSA 
# ============================================================
# SSA extends the classical MSE framework by explicitly
# controlling smoothness via the holding time (ht) parameter.
#
# Two equivalent formulations:
#   - Primal form : Minimize MSE subject to a fixed ht constraint
#   - Dual form   : Maximize ht subject to a fixed MSE constraint
#
# Together, these trace out an EFFICIENT FRONTIER in the
# (ht, MSE) space — analogous to Markowitz in portfolio theory —
# allowing practitioners to navigate the smoothness-accuracy
# trade-off in a principled way.
# ============================================================



# ============================================================
# NOTE ON TIMELINESS CONTROL
# ============================================================
# SSA controls Timeliness INDIRECTLY via the forecast horizon 
# parameter (delta).
#
# More direct and specialized approaches are available:
#   → Look-Ahead DFP and PCS 
#   → These trace out an efficient frontier between MSE and timeliness
#   → A corresponding tutorial is in preparation. 
#
# ============================================================

