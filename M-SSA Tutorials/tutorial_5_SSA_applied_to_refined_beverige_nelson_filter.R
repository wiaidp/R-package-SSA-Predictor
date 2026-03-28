
# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 5: Refined Beveridge-Nelson (BN) Filter Analysis
# ══════════════════════════════════════════════════════════════════════════════
#
# This tutorial analyzes the refined Beveridge-Nelson (BN) filter proposed in:
#
#   Kamber, G., Morley, J., & Wong, B. (2024).
#   "Trend-Cycle Decomposition in the Presence of Large Shocks."
#   CAMA Working Papers 2024-24, Centre for Applied Macroeconomic Analysis,
#   Crawford School of Public Policy, The Australian National University.
#   Revised August 2024.
#
# The authors' original R-code is available at:
#   https://drive.google.com/file/d/15P2OOV7aPl8qcwAsvSltaO2QfmuxID_N/view
#   https://bnfiltering.com/
#
# ── DESCRIPTION OF THE AUTHORS' ORIGINAL CODE ────────────────────────────────
#
# This code computes the BN filter output gap using the automatic
# signal-to-noise selection criteria described in:
#
#   Kamber, Morley & Wong (2018) [KMW2018]:
#   "Intuitive and Reliable Estimates of the Output Gap from a
#    Beveridge-Nelson Filter."
#   Review of Economics and Statistics, 100(3), 550–566.
#   https://doi.org/10.1162/rest_a_00691
#
# The code has been extended in KMW2024 to include four refinements
# relative to the original KMW2018 implementation:
#
#   1) Alternative automatic delta selection: uses the local minimum of the
#      variance of trend shocks, rather than the local maximum of the
#      amplitude-to-noise ratio.
#
#   2) Iterative dynamic mean adjustment: uses trend estimates (rather than
#      overall growth) to avoid undue influence of outlier cyclical observations.
#
#   3) Dynamic estimation of BN cycle variance: uses the same rolling window as
#      dynamic demeaning, yielding more accurate 95% error bands.
#
#   4) Iterative backcasting: exploits the reversibility of the restricted AR
#      process to backcast output growth prior to the first observation,
#      enabling computation of the BN cycle from the first observation in
#      levels (rather than from the second).
#
# File structure:
#   - bnf_run.R  : Main file; contains data input and estimation choices.
#   - bnf_fcn.R  : Contains all functions called from the main file.
#
# Additional options include:
#   - Dynamic demeaning (KMW2018) or user-specified structural break dates
#     (e.g., via Bai-Perron test) to accommodate shifts in long-run drift.
#   - An option to impose no drift in levels, suitable for variables such as
#     the unemployment rate or inflation.
#
# Required inputs for the BN filter:
#   (i)   First-differenced series (of the variable to be detrended).
#   (ii)  Lag order of the restricted AR model used in estimation.
#   (iii) Indicator for whether iterative backcasting is employed.
#   (iv)  Signal-to-noise ratio delta.
#
# Error bands are computed according to the formula in the online appendix
# of KMW2018 (also available at https://doi.org/10.1162/rest_a_00691).
#
#
# ── THEORETICAL SSA BACKGROUND ───────────────────────────────────────────────────
#
#   Wildi, M. (2024).
#   "Business Cycle Analysis and Zero-Crossings of Time Series:
#    A Generalized Forecast Approach."
#   https://doi.org/10.1007/s41549-024-00097-5
#
#   Wildi, M. (2026a).
#   "Sign Accuracy, Mean-Squared Error and the Rate of Zero Crossings:
#    A Generalized Forecast Approach."
#   https://doi.org/10.48550/arXiv.2601.06547
#
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────


# Make a clean-sheet, load packages and functions

rm(list=ls())

library(xts)
# Load data from FRED with library quantmod
library(quantmod)


# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))




# ══════════════════════════════════════════════════════════════════════════════
# 1. Set up the BN Filter: using Kamber, Morley & Wong code (2024)
# ══════════════════════════════════════════════════════════════════════════════

# Clear the workspace and free memory.
# Comment out if other packages or objects are already loaded.
rm(list = ls(all = TRUE))
gc()

# Load all required BN filter functions from the companion functions file.
source(paste(getwd(), "/R/bnf_fcns.R", sep = ""))

# Load the US real GDP dataset.
# The dataset is stored as a CSV file without a header row.
usdata <- read.csv(
  file             = paste(getwd(), '/Data/USGDP_updated.csv', sep = ""),
  header           = FALSE,
  stringsAsFactors = FALSE
)

# Convert the GDP series to a time-series (ts) object.
# The sample ends in Q2 2023 at quarterly frequency.
# Type help(ts) for further documentation.
gdp <- ts(data = usdata, end = c(2023, 2), frequency = 4)

# Log-transform the series and multiply by 100 (i.e., express in log-points).
# This is equivalent to: log(gdp) * 100
y <- transform_series(y = gdp, take_log = TRUE, pcode = "p1")

cat("Example: log US real GDP\n\n")

# ── Estimation Settings ───────────────────────────────────────────────────────

p         <- 12    # Lag order of the AR(p) model. A large value allows
# for a low signal-to-noise ratio delta.

ib        <- 1     # Iterative backcasting:
#   0 = unconditional mean (KMW2018 default)
#   1 = iterative backcasting (KMW2024 refinement)

iterative <- 100   # Maximum number of iterations for iterative dynamic
# demeaning. Set to a value > 1 to enable; set to 0 to
# use a fixed sample mean instead.

window    <- 40    # Rolling window length for dynamic demeaning and/or
# dynamic error bands (40 quarters = 10 years).

# Signal-to-noise ratio (delta) selection method:
#   0 = fixed delta (use fixed_delta below)
#   1 = automatic: maximize amplitude-to-noise ratio (KMW2018)
#   2 = automatic: minimize variance of trend shocks (KMW2024 refinement)
delta_select <- 2

fixed_delta  <- 0.01   # Fixed delta, used only if delta_select = 0.
d0           <- 0.005  # Lower bound of the grid search for delta.
dt           <- 0.0005 # Grid increment for delta search.

# ── Demeaning and Error Band Settings ────────────────────────────────────────

demean         <- "nd"   # Default: no drift. Overridden below based on
# the value of `iterative`.
dynamic_bands  <- 0      # Default: fixed standard error bands.

if (iterative == 0) {
  demean <- "sm"         # Use full-sample mean for drift (KMW2018 default).
} else if (iterative > 0) {
  demean        <- "dm"  # Use dynamic demeaning (KMW2024 refinement).
  dynamic_bands <- 1     # Use dynamic error bands to match demeaning window.
}

# To allow for structural breaks in the long-run drift (e.g., identified via
# Bai-Perron), uncomment the following two lines and specify break dates:
# demean <- "pm"
# breaks <- c(50, 75, 100, 125, 150)

# If structural break demeaning is selected, disable iterative demeaning.
if (demean == "pm") { iterative <- 0 }

# To override the automatic error band choice, uncomment the line below:
# dynamic_bands <- 1   # 1 = dynamic error bands; 0 = fixed error bands.

# ── Run BN Filter and Plot Output ────────────────────────────────────────────

bnf <- bnf(y)

# Plot the estimated US output gap with 95% confidence bands.
plot(bnf, main = "US Output Gap", col = "red", plot_ci = TRUE)

cat("\nPrinting out cycle data...\n")
print(bnf)   # Comment this line to suppress cycle data output to the console.
cat('\n')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Replicating the Refined BN Filter and Preparing for SSA customization
# ══════════════════════════════════════════════════════════════════════════════
# 2.1 Extracting and Replicating the Refined BN Filter Weights
#
# Goal: Recover the implicit filter weights of the refined BN procedure
# directly from the filter output, without requiring full analytical
# derivation from the underlying model.
#
# Rationale: The refined BN filter is a composite of multiple modifications.
# While it is possible in principle to derive filter weights analytically from
# the cited literature, a more practical approach is to recover them directly
# via matrix inversion of the filtered output.
#
# ── Step 1: Combine raw and filtered data ────────────────────────────────────

mat <- cbind(y, bnf$cycle)
tail(mat)   # Inspect the last few observations of both series.

# ── Step 2: Construct the regression (Toeplitz) matrix ───────────────────────
# Each row contains L consecutive lagged values of y, forming the design
# matrix used to recover the filter coefficients via least squares inversion.

len     <- length(y)
L       <- as.integer(len / 2)   # Filter length: half the sample size.

# Initialize with the most recent L observations, then append lagged columns.
reg_mat <- y[len:(len - L + 1)]
for (i in 1:(L - 1))
  reg_mat <- cbind(reg_mat, y[-i + len:(len - L + 1)])
dim(reg_mat)   # Verify dimensions: should be L x L.

# ── Step 3: Recover filter weights by matrix inversion ───────────────────────
# Solve the linear system: reg_mat %*% rbn = bnf$cycle[len:(len-L+1)]
# The acronym 'rbn' stands for Refined Beveridge-Nelson filter.

rbn <- as.double(solve(reg_mat) %*% bnf$cycle[len:(len - L + 1)])
par(mfrow=c(1,1))
ts.plot(rbn, main = "Refined BN Filter Coefficients: Default Settings (KMW2024)")

# ── Step 4: Verify filter replication ────────────────────────────────────────
# Convolve the recovered filter weights with y and compare against the
# original bnf$cycle output. Exact overlap confirms successful replication.

y_check <- rep(NA, len)
for (i in L:len)
  y_check[i] <- rbn %*% y[i:(i - L + 1)]

ts.plot(
  cbind(y_check, bnf$cycle),
  main = "Replication Check: Recovered Filter Output vs. Original BN Cycle"
)
# Both series should overlap exactly if replication is successful.
tail(cbind(y_check, bnf$cycle))   # Confirm agreement at the end of sample.

# ══════════════════════════════════════════════════════════════════════════════
# 2.2 Frequency-Domain Analysis: Amplitude and Phase-Shift Functions
# ══════════════════════════════════════════════════════════════════════════════

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


# ── Frequency-domain analysis of the refined BN filter ─────────────
# Compute and plot the amplitude and phase-shift functions of the recovered
# filter weights rbn to characterize its frequency-domain properties.

K <- 600

amp_shift_obj <- amp_shift_func(K, rbn, F)

amp   <- amp_shift_obj$amp
shift <- amp_shift_obj$shift

# Plot amplitude and phase-shift side by side for visual inspection.
# The amplitude reveals which frequency components are passed or suppressed.
# The phase-shift indicates the time delay introduced at each frequency.
par(mfrow = c(2, 1))
plot(amp, type = "l", axes = F, xlab = "Frequency", ylab = "",
     main = "Amplitude: Refined BN Filter (KMW2024 Default Settings)")
axis(1, at = 1 + 0:6 * K / 6,
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

plot(shift, type = "l", axes = F, xlab = "Frequency", ylab = "Phase-Shift",
     main = "Time-Shift", ylim = c(-3, 5))
axis(1, at = 1 + 0:6 * K / 6,
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
abline(h = 0)   # Reference line at zero shift (no time delay).
box()



# ══════════════════════════════════════════════════════════════════════════════
# 2.2  Transform the filter for use on first-differenced data
# ══════════════════════════════════════════════════════════════════════════════
# The function below converts a filter designed for levels into an equivalent
# filter that operates on first differences, producing identical output.
# The transformation computes the cumulative sum of the original coefficients:
# for each lag i, the new coefficient equals the sum of all original
# coefficients from lag 1 to lag i.
# Purpose of the filter transformation:
#   Working in (stationary) first differences allows replication and customization by SSA

conv_with_unitroot_func <- function(filt) {
  L    <- length(filt)
  conv <- filt
  for (i in 1:L)
    conv[i] <- sum(filt[1:i])
  return(list(conv = conv))
}

# Equivalent filter in first differences: when applied to differenced data, 
#   rbn_d replicates the original filter output of rbn (applied to original level data)
rbn_d <- conv_with_unitroot_func(rbn)$conv

# ── Equivalence check ─────────────────────────────────────────────────────────
# Verify that applying the transformed filter (rbn_d) to first-differenced data
# yields the same cycle as applying the original filter (rbn) to levels.

y_d <- rep(NA, len)
for (i in (L + 1):len)
  y_d[i] <- rbn_d %*% diff(y)[-1 + i:(i - L + 1)]
par(mfrow=c(1,1))
ts.plot(cbind(y_d, bnf$cycle), main = "Equivalence check: both outputs should overlap exactly")


# ══════════════════════════════════════════════════════════════════════════════
# 2.3  Amplitude and phase-shift functions of the transformed filter
# ══════════════════════════════════════════════════════════════════════════════

amp_shift_obj <- amp_shift_func(K, rbn_d, F)
amp           <- amp_shift_obj$amp
shift         <- amp_shift_obj$shift

par(mfrow = c(2, 1))

plot(amp, type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude: rbn_d (applied to differences)")
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2); box()

plot(shift, type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Shift (periods)",
     main = "Phase shift: rbn_d", ylim = c(-3, 5))
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2); abline(h = 0); box()

# The filter in first differences is a bandpass, like HP-gap (tutorial 2) and BK (tutorial 4)
#   Bandpass filters have a propensity to generate spurious cycles (vanishing amplitude at frequency zero)

# ── Dominant frequency ────────────────────────────────────────────────────────
# Identify the frequency at which the amplitude peaks, then convert it to a
# period. With quarterly data the dominant cycle is approximately 17 years,
# suggesting the filter captures very low-frequency (long-run) fluctuations.

peak_idx <- which(amp == max(amp))
peak_idx                         # Index of peak frequency
2 * K / (peak_idx - 1)           # Corresponding period (quarters)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Apply the filter to simulated data: spurious-cycle diagnostics
# ══════════════════════════════════════════════════════════════════════════════

# ── 3.1  Original filter applied to a random walk (levels) ───────────────────
# A random walk has no cyclical structure. Any cyclical pattern in the filtered
# output is therefore entirely spurious — an artefact of the filter, not of
# the data.

set.seed(123)
len_rw <- 1200
x      <- cumsum(rnorm(len_rw))   # Simulated random walk

par(mfrow = c(1, 1))
ts.plot(x, main = "Simulated random walk (levels)")

# Apply the levels filter
output <- rep(NA, len_rw)
for (i in L:len_rw)
  output[i] <- rbn %*% x[i:(i - L + 1)]

ts.plot(output, main = "Filter output: spurious cycle generated from a random walk")


# ── 3.2  Transformed filter applied to white noise (differences) ─────────────
# Applying rbn_d to i.i.d. noise is mathematically equivalent to applying rbn
# to a random walk (up to negligible finite-sample convolution error). The
# spurious cycle therefore reappears, confirming that it is a property of the
# filter, not of the integration order of the input.

set.seed(123)
len_rw <- 1200
x      <- rnorm(len_rw)   # i.i.d. white noise (first differences of the random walk above)

par(mfrow = c(1, 1))
ts.plot(x, main = "Simulated white noise (first differences of random walk)")

output <- rep(NA, len_rw)
for (i in L:len_rw)
  output[i] <- rbn_d %*% x[i:(i - L + 1)]

ts.plot(output, main = "Filter output: same spurious cycle (rbn_d on white noise)")


# ── 3.3  Transformed filter applied to noise with a slowly changing level ────
# Adding a low-frequency cosine to the noise introduces a slowly evolving
# deterministic component (analogous to gradual shifts between expansions and
# recessions). The filter fails to track this component: band-pass filters are
# not designed to follow slowly drifting levels, so this feature leaks through
# as distortion rather than being cleanly separated.

omega <- 2 * pi / len_rw
level <- cos((1:len_rw) * omega)   # One full cycle over the entire sample

# Plot series
ts.plot(x + level, main = "White noise with slowly changing deterministic level")

# Filter series with rbn_d
output <- rep(NA, len_rw)
for (i in L:len_rw)
  output[i] <- rbn_d %*% (x + level)[i:(i - L + 1)]

# Plot: the long-term shift has been removed
ts.plot(output,
        main = "Filter output: band-pass cannot track a slowly changing level")

# Bandpass filters zero out frequency-zero amplitude, stripping away
# gradual level shifts. Lowpass filters (Hamilton, ideal trend, HP-trend)
# pass these shifts through, which we consider informative for
# understanding growth dynamics.


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Derive the equivalent trend filter
# ══════════════════════════════════════════════════════════════════════════════

# ── 4.1  Trend filter via spectral complement ─────────────────────────────────
# Because rbn is a gap (high-pass) filter, its spectral complement — obtained
# by subtracting its coefficients from the unit-impulse sequence — yields the
# corresponding trend (low-pass) filter. The two filters partition the signal:
# trend + gap = original series.

rbn_trend <- c(1, rep(0, length(rbn) - 1)) - rbn

# The trend filter coefficients sum to one, confirming it preserves the
# long-run level (unit gain at frequency zero).
sum(rbn_trend)

# The coefficient plot shows an unusual negative lobe around lag 40 (≈ 10 years),
# suggesting the filter may be overfitting high-frequency detail in the sample
# used to derive rbn.
ts.plot(rbn_trend, main = "rbn trend filter coefficients")


# ── 4.2  Frequency-domain analysis of the trend filter ───────────────────────

amp_shift_obj <- amp_shift_func(K, rbn_trend, F)
amp           <- amp_shift_obj$amp
shift         <- amp_shift_obj$shift

par(mfrow = c(2, 1))

# The amplitude function is irregular (non-monotone), consistent with the
# suspected overfitting. Nevertheless, the peak amplitude falls within the
# business-cycle frequency range, so the filter does capture the intended
# low-frequency component.
plot(amp, type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude: rbn trend filter")
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2); box()

# Like the classic concurrent HP filter, the phase shift vanishes at frequency
# zero. This confirms that rbn (and hence rbn_trend) cancels a double unit root,
# producing an unbiased trend estimate for I(2)-type processes.
plot(shift, type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Shift (periods)",
     main = "Phase shift: rbn trend filter", ylim = c(-3, 5))
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2); abline(h = 0); box()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Compare rbn / rbn_trend with the HP gap and HP trend filters
# ══════════════════════════════════════════════════════════════════════════════

library(mFilter)
source(paste(getwd(), "/R/HP_JBCY_functions.r",   sep = ""))
source(paste(getwd(), "/R/simple_sign_accuracy.r", sep = ""))

# ── 5.1  Compute HP(1600) filters (quarterly design) ─────────────────────────
# Use the same filter length as rbn, ensuring a like-for-like comparison.
# HP(1600) is the standard quarterly smoothing parameter (Hodrick–Prescott).

L <- length(rbn)

# HP requires an odd filter length for proper centering.
if (L %% 2 == 0) {
  message("Filter length is even; incrementing by 1 to ensure correct HP centering.")
  L <- L + 1
}

lambda_monthly <- 1600
HP_obj   <- HP_target_mse_modified_gap(L, lambda_monthly)
hp_trend <- HP_obj$hp_trend
hp_gap   <- HP_obj$hp_gap

colo <- c("black", "red")
par(mfrow = c(2, 1))

# Gap filters: rbn and HP(1600) are qualitatively similar, both targeting
# business-cycle frequencies.
ts.plot(cbind(rbn, hp_gap),
        main = "Gap filter coefficients: rbn (black) vs. HP (red)", col = colo)

# Trend filters: rbn_trend shows a notable negative lobe around lag 40 (≈ 10
# years) absent in hp_trend, suggesting stronger — possibly excessive — noise
# suppression at intermediate frequencies.
ts.plot(cbind(rbn_trend, hp_trend),
        main = "Trend filter coefficients: rbn (black) vs. HP (red)", col = colo)


# ── 5.2  Compare amplitude and shift of the trend filters ────────────────────

amp_shift_obj <- amp_shift_func(K, rbn_trend, F)
amp_rbn       <- amp_shift_obj$amp
shift_rbn     <- amp_shift_obj$shift

amp_shift_obj <- amp_shift_func(K, hp_trend, F)
amp_hp        <- amp_shift_obj$amp
shift_hp      <- amp_shift_obj$shift

par(mfrow = c(2, 1))

# rbn_trend has a more irregular (non-monotone) amplitude function and suppresses
# high-frequency noise more aggressively than HP. The trade-off: HP is faster
# (less phase lag) but allows more high-frequency leakage.
plot(amp_rbn, col = colo[1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude: rbn trend (black) vs. HP trend (red)",
     ylim = c(0, 1.3))
lines(amp_hp, col = colo[2])
abline(h = 0)
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2); box()

# rbn_trend introduces more phase lag than HP trend across most frequencies.
# This is the direct consequence of its stronger noise suppression: smoother
# filters are inherently slower to respond to turning points.
plot(shift_rbn, col = colo[1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Shift (periods)",
     main = "Phase shift: rbn trend (black) vs. HP trend (red)")
lines(shift_hp, col = colo[2])
abline(h = 0)
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2); box()

# ══════════════════════════════════════════════════════════════════════════════
# 6. Apply SSA — refer to Tutorial 2.1 (classic HP) for the complete SSA workflow
# ══════════════════════════════════════════════════════════════════════════════

# This step is left as an exercise. Given the comparable amplitude and time-shift
# characteristics, customizing rbn (trend or gap) via SSA is expected to yield
# results similar to those obtained with the HP filter (trend or gap).









































# 2.2 Transform filter in first differences: the transformed filter replicates the original one when applied to differenced data


conv_with_unitroot_func<-function(filt)
{
  conv<-filt
  L<-length(filt)
  for (i in 1:L)
  {
    conv[i]<-sum(filt[1:i])
  }  
  return(list(conv=conv))
}

rbn_d<-conv_with_unitroot_func(rbn)$conv

# Check: verify that output of transformed and original filters match
# New filter_d is applied to differenced data
y_d<-rep(NA,len)
for (i in (L+1):len)
  y_d[i]<-rbn_d%*%diff(y)[-1+i:(i-L+1)]

# Check
ts.plot(cbind(y_d,bnf$cycle),main="Both outputs overlap")

# 2.3 Compute amplitude and shift of transformed filter

amp_shift_obj<-amp_shift_func(K,rbn_d,F)

amp<-amp_shift_obj$amp
shift<-amp_shift_obj$shift

par(mfrow=c(1,2))

plot(amp,type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude filter as applied to diff")
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()
plot(shift,type="l",axes=F,xlab="Frequency",ylab="Shift",main="Shift",ylim=c(-3,5))
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
abline(h=0)
box()

# Peak of amplitude
which(amp==max(amp))
# Periodicity of corresponding frequency: roughly 17 years (data quarterly)
2*K/(which(amp==max(amp))-1)

#----------------------------------------------------------
# 3 Apply the  filter to a random-walk

# 3.1 Original filter (in level)
set.seed(123)
len_rw<-1200
x<-cumsum(rnorm(len_rw))
par(mfrow=c(1,1))
ts.plot(x,main="Random-walk")

# Apply filter
output<-rep(NA,len_rw)
for (i in L:len_rw)
  output[i]<-rbn%*%x[i:(i-L+1)]

# The filter generates a spurious cycle
ts.plot(output,main="Spurious cycle")


# 3.2 Apply differenced filter to noise

set.seed(123)
len_rw<-1200
x<-(rnorm(len_rw))
par(mfrow=c(1,1))
ts.plot(x,main="Noise")

# Apply filter: output is the same as above (up to negligible finite sample convolution error)
output<-rep(NA,len_rw)
for (i in L:len_rw)
  output[i]<-rbn_d%*%x[i:(i-L+1)]


# The filter generates a spurious cycle
ts.plot(output,main="Spurious cycle")


# Add slowly changing deterministic level to noise
omega<-2*pi/len_rw
level<-cos((1:len_rw)*omega)

ts.plot(x+level)

# Filter data
output<-rep(NA,len_rw)
for (i in L:len_rw)
  output[i]<-rbn_d%*%(x+level)[i:(i-L+1)]


# The filter cannot track salient feature (changing level: recessions/expansions)
ts.plot(output,main="Bandpass cannot track changing level")


#------------------------------------------------------------------------
# 4. Define an equivalent trend filter:
# 4.1 rbn is a `gap` filter much like HP_gap
# To obtain the equivalent trend filter we just use 1-rbn

rbn_trend<-c(1,rep(0,length(rbn)-1))-rbn

# Coefficients add to one
sum(rbn_trend)

# Looks `strange' (overfitting?)
ts.plot(rbn_trend,main="rbn trend filter")

#---------------
# 4.2 Analyze in frequency-domain
amp_shift_obj<-amp_shift_func(K,rbn_trend,F)

amp<-amp_shift_obj$amp
shift<-amp_shift_obj$shift

par(mfrow=c(1,2))
# Amplitude is unsmooth: overfitting?
# Peak amplitude matches business-cycle frequencies
plot(amp,type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude filter rbn trend")
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()
# Like classic concurrent HP the shift vanishes at frequency zero (rbn cancels a double unit-root)
plot(shift,type="l",axes=F,xlab="Frequency",ylab="Shift",main="Shift",ylim=c(-3,5))
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
abline(h=0)
box()

#---------------------------------------------------------------------
# 5. Compare rbn with HP-gap and rbn_trend with HP_trend

library(mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))
# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))

# 5.1 Compute HP(1600): quarterly data
# Same length as rbn
L<-length(rbn)
# Should be an odd number
if (L/2==as.integer(L/2))
{
  print("Filter length should be an odd number")
  print("If L is even then HP cannot be adequately centered")
  L<-L+1
}  
# Specify lambda: querterly design to match rbn filter
lambda_monthly<-1600
par(mfrow=c(1,1))
# This function relies on mFilter and it derives additional HP-designs to be discussed further down
HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)

hp_trend<-HP_obj$hp_trend
hp_gap<-HP_obj$hp_gap

colo<-c("black","red")
par(mfrow=c(1,2))
# Gap filters look similar
ts.plot(cbind(rbn,hp_gap),main="Original gap filters",col=colo)
# Trend filters are a bit different: strange negative lob of rbn_trend towards lag 40 (10 years)
ts.plot(cbind(rbn_trend,hp_trend),main="Trend filters",col=colo)

#------------------------
# 5.2 Compare amplitude functions of trend filters

amp_shift_obj<-amp_shift_func(K,rbn_trend,F)

amp_rbn<-amp_shift_obj$amp
shift_rbn<-amp_shift_obj$shift

amp_shift_obj<-amp_shift_func(K,hp_trend,F)

amp_hp<-amp_shift_obj$amp
shift_hp<-amp_shift_obj$shift


par(mfrow=c(1,2))
# Amplitude rbn_trend is unsmooth (overfitting) and stronger noise suppression than hp_trend
plot(amp_rbn,col=colo[1],type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude trend filters",ylim=c(0,1.3))
lines(amp_hp,col=colo[2])
abline(h=0)
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()
# Larger shift of rbn_trend due to stronger noise suppression: HP is faster but noisier
plot(shift_rbn,col=colo[1],type="l",axes=F,xlab="Frequency",ylab="",main="Shift trend filters")
lines(shift_hp,col=colo[2])
abline(h=0)
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()

#----------------------------------------
# 6. Apply SSA: we refer to tutorial 2.1 (classic HP)

