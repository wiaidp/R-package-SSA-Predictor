# ════════════════════════════════════════════════════════════════════
# TUTORIAL 0.1 — SSA: THE FORECAST TRILEMMA
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────
# This tutorial presents and discusses the fundamental forecast
# trilemma underpinning the SSA framework.
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
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

#------------------------------------------------------------
# Example 0: Forecast trilemma and the MSE approach

# Data-generating process:
# x_t = ε_t (white noise), and z_t = Γ(x_t) is obtained by a
# simple two-sided symmetric equally weighted moving-average filter

# Target filter
L <- 11
gamma <- rep(1/L, L)

# Plot the symmetric (two-sided) target filter
plot(gamma, axes = F, type = "l",
     xlab = "Lag structure", ylab = "Filter coefficients",
     main = "Equally weighted acausal smoothing filter")
axis(1, at = 1:length(gamma),
     labels = (-(length(gamma)+1)/2) + 1:length(gamma))
axis(2)
box()

# Simulate white noise input
set.seed(231)
len <- 120
sigma <- 1
epsilon <- sigma * rnorm(len)
x <- epsilon

# Check: no autocorrelation
acf(x)

# Apply filters:
# side = 2 → symmetric (two-sided, non-causal)
# side = 1 → one-sided (causal, real-time)
y_sym <- filter(x, gamma, side = 2)
y_one_sided <- filter(x, gamma, side = 1)

tail(cbind(y_sym, y_one_sided))

# The symmetric filter is centered but produces NAs at the sample end.
# The one-sided filter is available up to the end but is delayed (right-shifted).
ts.plot(cbind(y_sym, y_one_sided),
        col = c("black", "black"), lty = 1:2,
        main = "Two-sided vs one-sided filter")

# We want to estimate y_sym at the sample end (t = len):
#   delta = 0  → nowcast
#   delta > 0  → forecast
#   delta < 0  → backcast
delta <- 0

# MSE-optimal predictor:
# Replace unknown future values x_{t+j} by their forecasts.
# For white noise, the optimal forecast is 0.
# ⇒ The MSE filter is the truncated one-sided version of gamma.
b_mse <- gamma[((L+1)/2 + delta):L]

# Apply filters
y_mse <- filter(x, b_mse, side = 1)
y_sym <- filter(x, gamma, side = 2)

short_sample <- cbind(y_sym, y_mse)

# Plot and mark zero-crossings (sign changes)
ts.plot(short_sample, col = c("black", "green"), lty = 1:2,
        main = "Target (black) vs MSE predictor (green)")
abline(h = 0)
abline(v = 1 + which(sign(y_mse[2:len]) != sign(y_mse[1:(len-1)])),
       col = "green")
abline(v = 1 + which(sign(y_sym[2:len]) != sign(y_sym[1:(len-1)])),
       col = "black")

# The predictor is delayed and noisier (more zero-crossings).
# Smoothness proxy: mean distance between zero-crossings (holding time, ht)

# We now compute the mean distance between consecutive zero-crossings.
# The following function computes the average distance between zero-crossings.
compute_empirical_ht_func<-function(x)
{ 
  x<-na.exclude(x)
  len<-length(x)-1
  empirical_ht<-len/length(which(sign(x[2:len])!=sign(x[1:(len-1)])))
  return(list(empirical_ht=empirical_ht))
}
# This function is loaded with the above source command. 

# Use a longer sample for stable estimates
set.seed(16)
len <- 12000
epsilon <- sigma * rnorm(len)
x <- epsilon
y_mse <- filter(x, b_mse, side = 1)
y_sym <- filter(x, gamma, side = 2)

# Empirical holding time: 
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)

# Target (y_sym) has larger ht → fewer crossings (smoother)
# Predictor has ~34% more crossings
compute_empirical_ht_func(y_sym)$empirical_ht /
  compute_empirical_ht_func(y_mse)$empirical_ht

# Theoretical holding time (white noise case; see Wildi, 2024)
# The following function is loaded with the above source command.
# It computes the expected (true) HT of a `filter' with weights b.
# It assumes that the filter is applied to white noise.
# See Wildi, 2024 (equation 3).
compute_holding_time_func<-function(b)
{
  if (length(b)>1)
  {  
    rho_ff1<-b[1:(length(b)-1)]%*%b[2:length(b)]/sum(b^2)
    # Mean holding-time
    ht<-1/(2*(0.25-asin(rho_ff1)/(2*pi)))
    # Alternative expression  
    if (F)
      ht<-pi/acos(rho_ff1)
  } else
  {
    ht<-2
    rho_ff1<-0
  }
  return(list(ht=ht,rho_ff1=rho_ff1))
}

compute_holding_time_func(gamma)$ht
compute_holding_time_func(b_mse)$ht

#-------------------------------------------------------------------
# Empirical ht converges to theoretical ht as sample size increases
#-------------------------------------------------------------------

# Measure relative delay (lag) via zero-crossing alignment
filter_mat <- cbind(y_sym, y_mse)

# The function shifts the second series relative to the first
# and minimizes total distance between corresponding crossings
# The minimizing shift estimates lead/lag (Wildi, 2024)
compute_min_tau_func(filter_mat)

# Result suggests ≈ 3-period lag of the predictor

# Align predictor accordingly
short_sample_shifted <- cbind(
  short_sample[1:(nrow(short_sample)-3), 1],
  short_sample[4:nrow(short_sample), 2]
)

# Visual check after alignment
ts.plot(short_sample_shifted,
        col = c("black", "green"), lty = 1:2,
        main = "Predictor shifted 3 periods to the left")
abline(h = 0)

# Frequency-domain view: phase shift of the transfer function
K <- 600
shift <- amp_shift_func(K, b_mse, F)$shift

plot(shift, type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Phase shift of MSE predictor")
axis(1, at = 1 + 0:6*K/6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6,
                         4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Low-frequency components (0<=omega<=2pi/6) are delayed by 2.5 periods,
# consistent with time-domain estimates

# Mean squared error (empirical)
mean((y_sym - y_mse)^2, na.rm = T)

# True MSE:
# Error arises from replacing future x_t by 0.
# Since x_t is white noise:
sum(sigma^2 * gamma[1:((L-1)/2 + delta)]^2)

#-------------------------------------------------------------------
# Empirical MSE converges to theoretical MSE with larger samples
#-------------------------------------------------------------------

# Forecast performance criteria:
# - MSE (accuracy)
# - Shift (timeliness)
# - Holding time (smoothness)
# → Forecast trilemma: cannot optimize all three simultaneously

# Evaluate performance across forecast horizons
delta_vec <- -((L-1)/2):((L-1)/2)

ht <- mse <- shift <- NULL

for (i in 1:length(delta_vec)) {
  delta <- delta_vec[i]
  
  # MSE-optimal filter for given horizon
  b_mse <- gamma[((L+1)/2 + delta):L]
  
  # 1. Smoothness (ht)
  ht <- c(ht, compute_holding_time_func(b_mse)$ht)
  
  # 2. Timeliness (shift)
  shift <- c(shift, amp_shift_func(K, b_mse, F)$shift[1])
  
  # 3. MSE
  if (((L-1)/2 + delta) > 0) {
    mse <- c(mse,
             sum(sigma^2 * gamma[1:((L-1)/2 + delta)]^2))
  } else {
    # Perfect reconstruction: no future values needed
    mse <- c(mse, 0)
  }
}

table <- rbind(ht, shift, mse)
rownames(table) <- c("holding-time", "shift", "mse")
colnames(table) <- paste("delta=", delta_vec, sep = "")

table

# Discussion:
# - The MSE filter optimizes accuracy only.
# - Increasing forecast horizon (delta):
#     → increases MSE (bottom row in table)
#     → decreases holding time (more noise)
#     → reduces delay or shift (appears more timely)
# - Trade-offs illustrate the forecast trilemma

# SSA approach:
# - Controls smoothness (ht) and timeliness (shift)
# - Minimizes MSE subject to these constraints

# Key trade-offs:
# - Higher smoothness or stronger lead → higher MSE
# - Lower MSE → less smooth or more delayed estimates
# - For fixed ht: shift can be traded against MSE
# - For fixed shift: ht can be traded against MSE

# References:
# - Wildi (2024), (2026a)
# - McElroy and Wildi (2018): ATS trilemma in frequency-domain (MDFA)

# Notes:
# 1. Timeliness is currently controlled via delta (forecast horizon).
#    More direct approaches are available: Look-Ahead DFP/PCS
# 2. SSA resolves the trilemma.
