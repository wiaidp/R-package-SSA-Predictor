
# =============================================================================
# TUTORIAL 2.1: SSA APPLIED TO THE HODRICK-PRESCOTT FILTER
# =============================================================================
# =============================================================================
# BROADER PERSPECTIVE: HP AS A PLATFORM FOR SSA (CUSTOMIZATION OF HP BY SSA)
# =============================================================================
#
#   Any linear filter can be replicated and improved by SSA with respect to:
#     1. SMOOTHNESS  — noise suppression, holding-time control
#     2. TIMELINESS  — phase advancement, earlier turning point detection

# We here replicate and customize HP
#
# RELATED TUTORIALS:
#   - Tutorial 3: SSA applied to Hamilton's (2018) regression filter
#                 (proposed as an alternative to HP)
#   - Tutorial 4: SSA applied to the Baxter-King bandpass filter
#   - Tutorial 5: SSA applied to the `refined' Beveridge-Nelson filter

# =============================================================================
# Background on HP: see Tutorial 2.0 (not necessary to understand results presented here)
# =============================================================================
# SCOPE AND COVERAGE
# =============================================================================
# All examples in this tutorial use SIMULATED stationary data.
# Advantage: the true DGP is known, enabling exact theoretical verification.
#
# Applications to real data are deferred to later tutorials:
#   - Tutorial 5: HP and SSA applied to US Industrial Production (INDPRO)
#   - Tutorial 7: Multivariate SSA (M-SSA) applied to German macro data

# =============================================================================
# EXAMPLES COVERED IN THIS TUTORIAL: TARGET FILTER × DATA TYPE
# =============================================================================
#
#   Example | Target Filter                        | Input Data
#   --------|--------------------------------------|----------------------
#   1       | One-sided MSE-HP (WN-optimal)        | White noise
#   2       | Classic one-sided HP (ARIMA(0,2,2))  | White noise
#   5a      | Classic one-sided HP                 | Autocorrelated data
#   5b      | One-sided MSE-HP                     | Autocorrelated data
#   6       | Two-sided HP                         | Autocorrelated data
#   7       | Classic HP gap (levels)              | Non-stationary data
#   8       | SSA targeting HP-MSE                 | Forecast trilemma
#


# =============================================================================
# MAIN FINDINGS
# =============================================================================


# -----------------------------------------------------------------------------
# FINDING 1: SSA CONTROLS NOISE LEAKAGE (Examples 1, 2, 5)
# -----------------------------------------------------------------------------
# SSA can be applied to tame the noisy zero-crossings of HP-concurrent:
#
#   Standard SSA constraint:
#     - Holding time increased by 50%  (smoother cycle)
#     - Equivalently: 33% fewer noisy zero-crossings than the HP benchmark
#
#   This smoothness constraint is applied consistently across all examples.

# -----------------------------------------------------------------------------
# FINDING 2: SSA FORECASTS ARE TIMELY AND SMOOTH (Examples 5, 6)
# -----------------------------------------------------------------------------
#
#   - SSA concurrent designs can simultaneously achieve:
#       * Greater SMOOTHNESS (noise suppression) than HP-concurrent
#       * Earlier DETECTION (timeliness) of peaks and troughs

# -----------------------------------------------------------------------------
# FINDING 3: ACTING ON THE FORECAST TRILEMMA THROUGH SSA (Example 8)
# -----------------------------------------------------------------------------
# Example 8 visualizes the fundamental trade-off in real-time filter design
# for SSA targeting HP-MSE (see Tutorial 0.1 for theoretical background):
#
#   ACCURACY   <=>  closeness to the two-sided benchmark
#   SMOOTHNESS <=>  fewer noisy zero-crossings (larger holding time)
#   TIMELINESS <=>  smaller phase delay (earlier turning point detection)
#
#   No filter can simultaneously maximize all three objectives.
#   SSA makes this trade-off explicit and controllable via its
#   optimization criterion. 

#   SSA delineates the efficient Accuracy-Smoothness frontier, see Wildi (2026a), (2026b) 

#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions
rm(list=ls())

library(xts)
# Load the library mFilter
# HP and BK filters
library(mFilter)
# Plot for heat map of Trilemma
library(ggplot2)
library("gplots")

# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


# ============================================================
# Introduction — See also Tutorial 2.0
# ============================================================
# This script covers:
#   Derivation of the Hodrick-Prescott (HP) filter and brief analysis of the classic one-sided (concurrent) HP trend filter
# ============================================================

# ------------------------------------------------------------
# a. Derivation of the HP Filter
# ------------------------------------------------------------

# We use the R package 'mFilter' to compute the HP filter.

# Specify the filter length L.
# L must be an odd number so that the two-sided (symmetric) HP filter
# can be properly centered around its midpoint.
L <- 201

# Safety check: if L is even, increment by 1 to enforce the odd-length requirement.
if (L / 2 == as.integer(L / 2))
{
  print("Filter length must be an odd number.")
  print("An even L prevents adequate centering of the two-sided HP filter.")
  L <- L + 1
}

# Specify the smoothing parameter lambda for a monthly sampling frequency.
# The standard monthly value (lambda = 14,400) was proposed by Hodrick and Prescott.
lambda_monthly <- 14400

par(mfrow = c(1, 1))
# Compute the HP filter object, which contains both the two-sided target
# and the one-sided (concurrent) filter coefficients.
HP_obj <- HP_target_mse_modified_gap(L, lambda_monthly)

# --- Two-sided (symmetric) HP filter ---
# This is the bi-infinite HP filter, here truncated to length L.
# It serves as the ideal (non-causal) benchmark for the trend component.
hp_target <- HP_obj$target
ts.plot(hp_target, main = paste("HP(", lambda_monthly, ") two-sided target", sep = ""))

# --- Concurrent HP gap filter ---
# Applied to a series in levels, this high-pass filter isolates the cyclical component
# as the difference between the series and its estimated trend.
hp_gap <- HP_obj$hp_gap
ts.plot(hp_gap, main = "Concurrent HP gap (high-pass)")

# --- Classic one-sided (concurrent) HP trend filter ---
# Derived under the assumption that the observed series follows an I(2) process.
# See McElroy (2006) for the theoretical derivation.
hp_trend <- HP_obj$hp_trend
ts.plot(hp_trend, main = "One-sided (concurrent) HP trend")

# --- Holding time and lag-1 autocorrelation of the concurrent HP trend ---
# The holding time (ht) measures the average duration between zero-crossings
# of the filter output, and serves as a proxy for the smoothness of the filter.
htrho_obj <- compute_holding_time_func(hp_trend)
rho_hp    <- htrho_obj$rho_ff1   # lag-1 autocorrelation of the filter output
ht_hp     <- htrho_obj$ht        # holding time of the concurrent HP trend
ht_hp

# Compare holding times of the one-sided (concurrent) and two-sided (symmetric) filters.
# The two-sided HP is substantially smoother, yielding a much longer holding time.
# This large discrepancy is discussed in detail in the JBCY paper (Wildi (2004)).
compute_holding_time_func(hp_target)$ht

# ------------------------------------------------------------
# b. Analysis of the Classic Concurrent HP Trend
# ------------------------------------------------------------
# We examine the smoothness discrepancy identified above and illustrate
# its practical consequences by comparing filter outputs on simulated data.

len <- L + 1000
set.seed(67)

# Simulate a white noise process as the input series.
# In the JBCY paper, the filter is applied to log-returns of INDPRO,
# which resemble white noise
a1 <- 0
x  <- arima.sim(n = len, list(ar = a1))

# Apply both filters to the simulated series.
y_hp_concurrent <- filter(x, hp_trend,  sides = 1)   # one-sided (causal) filter
y_hp_symmetric  <- filter(x, hp_target, sides = 2)   # two-sided (non-causal) filter

# --- Visual comparison ---
# The one-sided filter output (red) is considerably noisier than the two-sided output (black),
# exhibiting far more zero-crossings (marked by vertical dashed lines).
# This reflects the weaker noise-suppression of the concurrent filter.
ts.plot(y_hp_concurrent,
        main = "HP filter outputs: two-sided (black) vs. one-sided (red)\nVertical lines mark zero-crossings of the one-sided filter",
        col  = "red")
lines(y_hp_symmetric)
abline(h = 0)
abline(v   = which(y_hp_concurrent[2:len] * y_hp_concurrent[1:(len - 1)] < 0),
       col = "red", lty = 3)
mtext("Two-sided HP", col = "black", line = -1)
mtext("One-sided HP",  col = "red",   line = -2)

# --- Empirical holding times ---
# For sufficiently long samples, empirical holding times converge to their theoretical values.
compute_empirical_ht_func(y_hp_concurrent)   # one-sided: shorter ht (noisier)
compute_empirical_ht_func(y_hp_symmetric)    # two-sided: longer  ht (smoother)

# Theoretical holding times for reference:
ht_hp                                        # one-sided HP trend
compute_holding_time_func(hp_target)$ht      # two-sided HP target

# Key observations from the time-series plot:
#  - The two-sided HP can remain far from zero over extended episodes,
#    indicating a tendency toward over-smoothing.
#  - This over-smoothing may wash out short but severe recession dips,
#    causing the estimated cycle to understate the impact of crises.
#    See Phillips and Jin (2021) for a thorough discussion.
#  - The one-sided HP, with its weaker noise suppression, tracks recession dips
#    more faithfully, at the cost of a noisier output.
#  - Frequency-domain analysis provides additional diagnostic information (see below).

# --- Amplitude and time-shift of the classic concurrent HP trend ---
K       <- 600
amp_obj <- amp_shift_func(K, as.vector(hp_trend), F)

par(mfrow = c(1, 2))

# Amplitude function
plot(amp_obj$amp,
     type  = "l", axes = F,
     xlab  = "Frequency", ylab = "",
     main  = paste("Amplitude — HP concurrent trend", sep = ""),
     ylim  = c(0, max(amp_obj$amp)))
mtext("Amplitude of classic concurrent HP trend", line = -1)
abline(h = 0)
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Time-shift function
plot(amp_obj$shift,
     type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = paste("Time-shift — HP concurrent trend", sep = ""))
mtext("Time-shift of classic concurrent HP trend", line = -1)
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Frequency-domain observations:
#  - The amplitude peaks near a periodicity of ~7 years, consistent with
#    the conventional definition of the business cycle.
#  - The time-shift is small and approaches zero as frequency tends to zero (see tutorial 2.0 for background).

# ============================================================
# Summary and Implications for Business Cycle Analysis
# ============================================================

# Two-sided HP filter:
#  - May be an overly smooth target for BCA; standard lambda values proposed
#    in the literature are arguably too large. See Phillips and Jin (2021).
#  - Alternatively, the underlying model may be viewed as severely misspecified
#    for BCA purposes. See Tutorial 2.0 for discussion.

# One-sided (concurrent) HP filter:
#  - Less smooth than its two-sided counterpart, enabling better tracking of
#    short but severe recession dips.
#  - Peak amplitude aligns well with business-cycle frequencies (~7 years).
#  - Near-zero time-shift makes it a demanding real-time benchmark:
#    it is typically faster than Hamilton's regression filter. See Tutorial 3.

# Proposed strategy:
#  - Target the two-sided HP via SSA in Examples 4 and 6.
#  - Target the one-sided HP via SSA in Examples 1, 2, 3, 5, and 8.
#  - Example 1 specifically addresses the one-sided HP-MSE filter under a
#    white-noise data assumption.
#  - Example 4 demonstrates that this target is equivalent to the two-sided HP
#    when data are white noise
# ============================================================


######################################################################################################################
######################################################################################################################
# Example 1:
# Target: HP-MSE (Mean Squared Error optimal concurrent HP filter under white noise assumption)


#-----------------------------------------------------------------------------------
# 1.1 Concurrent MSE Estimate of the Bi-Infinite HP Filter Assuming White Noise Input (see tutorial 2.0 for background)

hp_mse <- hp_mse_example7 <- HP_obj$hp_mse

par(mfrow = c(1, 1))
ts.plot(hp_mse)

# Compute the lag-one autocorrelation (rho) and holding time (ht) for hp_mse
# Holding time (ht): average number of time steps between consecutive zero-crossings
#   of the filter output — a measure of smoothness/persistence
htrho_obj <- compute_holding_time_func(hp_mse)
rho_hp    <- htrho_obj$rho_ff1   # lag-one autocorrelation of hp_mse output
ht_mse    <- htrho_obj$ht        # holding time of hp_mse output

# Display the holding time: this serves as our reference baseline for SSA design
ht_mse


#-----------------------------------------------------------------------------------
# 1.2 Setting Up SSA 
#
# Objective: design an SSA filter that:
#   (a) tracks the HP-MSE target as closely as possible (maximizes correlation), and
#   (b) produces a smoother output than hp_mse by imposing a larger holding time constraint
#
# Step 1: Choose the target holding time
#   We select ht > ht_mse to reduce the number of zero-crossings (signal reversals)
#   relative to the MSE concurrent filter — making the real-time estimate less "choppy"
ht_mse
ht <- 1.5 * ht_mse   # SSA output will have ~33% fewer zero-crossings than hp_mse

# Step 2: Convert the holding time constraint to a lag-one autocorrelation (rho1)
#   SSA requires the constraint in terms of rho1; ht and rho1 are in one-to-one correspondence
rho1 <- compute_rho_from_ht(ht)

# Confirm the implied reduction in zero-crossings (should equal 1.5)
ht / ht_mse

# Step 3: Specify the forecast horizon
#   forecast_horizon = 0 corresponds to a nowcast (real-time concurrent estimate, no lookahead)
forecast_horizon <- 0

# Step 4: Specify the input process model
#   xi = NULL: assumes the input xt is white noise (default setting)
#   This is consistent with how hp_mse was derived
xi <- NULL

# Step 5: Specify the target filter
gammak_generic <- hp_mse


#-----------------------------------------------------------------------------------
# Run SSA optimization targeting HP-MSE
SSA_obj_HP <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Equivalent shorthand call: omitting xi defaults to xi = NULL (white noise input)
SSA_obj_HP <- SSA_func(L, forecast_horizon, gammak_generic, rho1)


# Extract SSA filter coefficients
# Note: When xi = NULL (white noise input), the two SSA filter representations
#   ssa_eps (expressed in terms of innovations) and ssa_x (expressed in terms of xt)
#   are identical — see Tutorial 1 for a detailed explanation
ssa_x          <- SSA_obj_HP$ssa_x
SSA_filt_HP    <- SSA_example1 <- ssa_eps <- SSA_obj_HP$ssa_eps


#-----------------------------------------------------------------------------------
# 1.3 Plot: Compare Filter Weights
#
# Left panel:  Symmetric (two-sided) HP target vs. HP-MSE concurrent filter
# Right panel: SSA-HP filter vs. HP-MSE concurrent filter
#   — illustrates how SSA reshapes the concurrent filter to meet the ht constraint

par(mfrow = c(1, 2))

# Left panel: symmetric HP target vs. HP-MSE concurrent filter
mplot <- cbind(hp_target, hp_mse)
colnames(mplot) <- c("Symmetric", "Concurrent")

colo<-c("black","brown","blue")
par(mfrow=c(1,2))
mplot<-cbind(hp_target,hp_mse)
colnames(mplot)<-c("Symmetric","Concurrent")

plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=2)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()



# Right panel: SSA-HP filter vs. HP-MSE concurrent filter
#   SSA shifts/redistributes weight to satisfy the smoothness (ht) constraint
#   while remaining as close as possible (in correlation) to the HP-MSE target
mplot<-cbind(SSA_filt_HP,hp_mse)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP")

plot(mplot[,1],main=paste("Concurrent",sep=""),axes=F,type="l",xlab="",ylab="",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[3],lwd=2)
mtext(colnames(mplot)[1],col=colo[3],line=-1)
lines(mplot[,2],col=colo[2],lwd=2)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


#-----------------------------------------------------------------------------------
# 1.4 Validation Checks
#
# We verify that the SSA optimization delivers on its stated properties using a
# long Monte Carlo simulation. All sample statistics should converge to their
# theoretical (population) counterparts as the sample length increases.
#
# Specifically, we check:
#   (i)   The empirical holding time matches the imposed constraint
#   (ii)  The empirical lag-one acf of the filter output matches rho1
#   (iii) The sample correlation of SSA output with the HP-MSE target matches
#         the theoretical criterion value

len <- 100000
set.seed(16)

# Simulate a long white noise series (AR(1) with coefficient a1 = 0 reduces to white noise)
a1 <- 0
x  <- arima.sim(n = len, list(ar = a1))

# Apply the SSA-HP filter to the simulated series (one-sided, causal filtering)
yhat <- filter(x, SSA_filt_HP, side = 1)


# Check 1: Holding time
# Compare empirical holding time of filter output with the imposed ht constraint
# Good convergence confirms SSA successfully controls signal smoothness
empirical_ht <- compute_empirical_ht_func(yhat)
empirical_ht   # empirical value
ht             # imposed constraint — both should be approximately equal


# Check 2: Lag-one autocorrelation
# SSA imposes rho1 as a hard constraint; the optimized filter should reproduce it exactly
# Perfect agreement here indicates the optimization converged to the global optimum
SSA_obj_HP$crit_rhoyy   # theoretical (optimized) lag-one acf of SSA output
rho1                    # imposed constraint — both should match closely


# Check 3: Correlation with HP-MSE target
# Note: Correlations with the two-sided (symmetric) HP target are examined in Examples 4 and 6

crit_example1 <- SSA_obj_HP$crit_rhoyz
crit_example1   # theoretical correlation between SSA output and HP-MSE target

# a. Compute HP-MSE nowcast (target filter output) on the simulated data
MSE_nowcast <- filter(x, hp_mse, side = 1)

# b. Compute sample correlation between SSA output and HP-MSE nowcast
cor(yhat, MSE_nowcast, use = 'pairwise.complete.obs')
# Should be very close to crit_example1 — confirms the analytical criterion is empirically accurate


#-----------------------------------------------------------------------------------
# Summary of Example 1:
#
# - The HP-MSE filter is the MSE-optimal concurrent HP filter under a white noise input assumption
# - SSA maximizes the correlation of the real-time predictor with this target,
#   subject to a user-specified smoothness constraint (holding time ht = 1.5 * ht_mse)
# - All three validation checks confirm that the SSA optimization is:
#     * interpretable: the ht constraint has a clear, measurable effect on output smoothness
#     * reliable:      sample properties converge to their theoretical counterparts
#     * practically relevant: the imposed trade-off between timeliness and smoothness is
#                              precisely controlled and empirically verifiable






##############################################################################################################
##############################################################################################################
# Example 2: SSA Targeting the Classic Concurrent HP Filter
#
# This example mirrors Example 1 with one key difference:
#   - Example 1 targeted the HP-MSE filter (MSE-optimal under white noise assumption)
#   - Example 2 targets the classic HP concurrent filter (implicitly assumes ARIMA(0,2,2) input)
#
# By comparing both examples, we can assess the sensitivity of the SSA design to the
# choice of target filter, and understand the practical implications of each target assumption.


#-----------------------------------------------------------------------------------
# 2.1 HP Filter Reference Values (computed in Example 1 — reproduced here for convenience)
#   ht_hp: holding time of the classic HP concurrent filter output under white noise
#   This serves as the smoothness baseline for the SSA design in this example


#-----------------------------------------------------------------------------------
# 2.2 SSA Setup and Hyperparameter Choices

# Display baseline holding time of the classic HP concurrent filter
ht_hp

# Step 1: Choose the target holding time
#   We want the SSA output to be smoother than the classic HP concurrent filter,
#   i.e., to produce fewer zero-crossings (signal reversals) on average.
#   We set ht = 12, which should yield approximately 33% fewer crossings than HP concurrent.
ht <- 12

# Step 2: Convert the holding time constraint to a lag-one autocorrelation (rho1)
#   SSA requires the smoothness constraint in the form of a lag-one acf value.
#   The function compute_rho_from_ht() performs this conversion (ht and rho1 are in 1-to-1 correspondence).
rho1 <- compute_rho_from_ht(ht)

# Confirm the implied reduction in zero-crossings relative to classic HP concurrent (~1.33 = ~33% fewer)
ht / ht_hp

# Step 3: Specify the forecast horizon
#   forecast_horizon = 0: nowcast (real-time concurrent estimate, no lookahead)
forecast_horizon <- 0

# Step 4: Specify the target filter
#   Unlike Example 1 (which used hp_mse), here we supply the classic HP concurrent filter.
#   This implicitly assumes an ARIMA(0,2,2) data-generating process for xt.
#   SSA will maximize correlation with this target subject to the ht constraint.
gammak_generic <- hp_trend

# Step 5: Specify the input process model
#   xi = NULL: assumes white noise input (default setting).
#   Note: there is a mild inconsistency here — the classic HP concurrent filter was designed
#   under an ARIMA(0,2,2) assumption, but we simulate white noise. This deliberate mismatch
#   allows a clean comparison with Example 1 under identical simulation conditions.
xi <- NULL


#-----------------------------------------------------------------------------------
# Run SSA optimization targeting the classic HP concurrent filter
SSA_obj_HP <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract SSA filter coefficients
# Since xt is white noise (xi = NULL), deconvolution reduces to an identity transformation,
# so ssa_eps and ssa_x are identical — see Tutorial 1 for a full explanation
ssa_x       <- SSA_obj_HP$ssa_x
SSA_filt_HP <- ssa_eps <- SSA_obj_HP$ssa_eps


#-----------------------------------------------------------------------------------
# Plot: SSA filter weights vs. classic HP concurrent filter weights
#
# Visual note: the SSA filter typically exhibits a characteristic "tip" or spike
# at the most recent lag (lag 0). This arises from an implicit boundary constraint
# in the SSA optimization: the filter coefficient at lag -1 is constrained to vanish.
# See Theorem 1 in the JBCY paper for a formal derivation of this property.

mplot <- cbind(ssa_x, hp_trend)
colnames(mplot) <- c(paste("SSA(", ht, ",", forecast_horizon, ")", sep = ""), "HP-concurrent")
colo <- c("blue", "green")

par(mfrow=c(1,1))
plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


#-----------------------------------------------------------------------------------
# 2.3 Simulation Validation: Compare SSA Output with Classic HP Concurrent
#
# We filter a long white noise series through both the SSA and HP concurrent filters,
# then verify that:
#   (i)  SSA output has the correct (imposed) holding time
#   (ii) HP concurrent output has the expected baseline holding time ht_hp
#   (iii) SSA produces approximately 33% fewer zero-crossings than HP concurrent

len    <- 100000
set.seed(1)

# Simulate a long white noise series (AR coefficient a1 = 0 reduces AR(1) to white noise)
a1 <- 0
x  <- arima.sim(n = len, list(ar = a1))

# Apply SSA-HP filter (one-sided, causal)
yhat <- filter(x, SSA_filt_HP, side = 1)

# Verify SSA holding time: empirical value should match imposed constraint ht = 12
ht                                  # imposed constraint
compute_empirical_ht_func(yhat)     # empirical holding time — should be close to ht

# Apply classic HP concurrent filter
HP_concurrent <- filter(x, hp_trend, side = 1)

# Verify HP concurrent holding time: empirical value should match baseline ht_hp
ht_hp                                        # theoretical baseline
compute_empirical_ht_func(HP_concurrent)     # empirical holding time — should be close to ht_hp

# As expected, SSA generates approximately 33% fewer zero-crossings than HP concurrent
# (ratio ht/ht_hp ≈ 1.33 confirmed empirically above)


#-----------------------------------------------------------------------------------
# Plot: Time-domain comparison of SSA output vs. HP concurrent output
#
# Both series are displayed on a shared scale for visual comparison.
# Note: SSA filter output has arbitrary scaling; the key comparison is the
#       frequency and clustering pattern of zero-crossings.
#
# What to look for in the plot:
#   - At up- and downswings (filter output well away from zero): SSA tracks HP closely,
#     reflecting the SSA optimality principle (maximizing correlation with the target)
#   - Near the zero line: HP concurrent tends to generate clusters of noisy crossings,
#     while SSA maintains better control — the smoothness constraint actively suppresses
#     these spurious reversals
#   - See the amplitude (frequency response) analysis for a formal explanation of
#     why these differences are concentrated near zero-crossings

mplot <- na.exclude(cbind(yhat, HP_concurrent))
colnames(mplot) <- c(paste("SSA(", ht, ",", forecast_horizon, ")", sep = ""), "HP-concurrent")

par(mfrow = c(1, 1))
anf <- 1000
enf <- 1500

ts.plot(mplot[anf:enf,],col=colo)
mtext(paste("SSA(",ht,",",forecast_horizon,")",sep=""),col=colo[1],line=-1)
mtext("HP-concurrent",col=colo[2],line=-2)
abline(h=0)

#-----------------------------------------------------------------------------------
# 2.4 Shift Analysis: Relative Timing of SSA vs. HP Concurrent
#
# The time-series plot in section 2.3 hints that the classic HP concurrent filter
# is very slightly left-shifted (i.e., leads SSA by a small fraction of a time unit).
# We now quantify this timing difference formally using the tau-statistic.
#
# The tau-statistic (proposed in Wildi, M. (2024),
# and explained in previous tutorials) measures relative shift at zero-crossings:
#
#   - The statistic is computed as a function of a time-shift parameter tau
#   - The location of the minimum (trough) identifies the relative timing:
#       * Trough to the LEFT  of zero  --> the series in column 1 (SSA) leads HP concurrent
#       * Trough to the RIGHT of zero  --> the series in column 1 (SSA) lags HP concurrent
#
# Interpretation of results:
#   The slight asymmetry of the trough location to the right of zero indicates that
#   SSA marginally lags the HP concurrent filter — by approximately half a time unit.
#   This is the price paid for the additional smoothness (larger ht) imposed on SSA.

shift_tau_obj <- compute_min_tau_func(mplot)


#-----------------------------------------------------------------------------------
# 2.5 Frequency-Domain Analysis: Amplitude and Phase-Shift
#
# We now examine the amplitude (gain) and phase-shift functions of both filters.
# These provide a formal frequency-domain explanation of the time-domain observations
# made in sections 2.3 and 2.4:
#   - Why does SSA produce fewer zero-crossings than HP concurrent?  --> Amplitude function
#   - Why does SSA marginally lag HP concurrent?                     --> Phase-shift function

K <- 600
amp_obj_SSA <- amp_shift_func(K, as.vector(SSA_filt_HP), F)
amp_obj_HP  <- amp_shift_func(K, hp_trend, F)

par(mfrow = c(1, 1))

#-----------------------------------------------------------------------------------
# Amplitude functions
#
# Background: The HP filter is a lowpass filter applied to differenced data (returns).
#   Its amplitude function describes how much of each frequency component is retained
#   in the filter output.
#
# Key observation:
#   The SSA amplitude function lies closer to zero in the stopband (high frequencies)
#   compared to HP concurrent. This means SSA suppresses high-frequency noise more
#   aggressively — directly explaining why SSA produces fewer zero-crossings.
#
# This stronger high-frequency attenuation is a characteristic signature of SSA
# smoothing designs, and is the formal basis for the term "noisy crossings":
#   the extra crossings of HP concurrent are driven by residual high-frequency content
#   that SSA has filtered out via the ht constraint.

mplot <- scale(cbind(amp_obj_SSA$amp, amp_obj_HP$amp), scale = T, center = F)
colnames(mplot) <- c(paste("SSA(", ht, ",", forecast_horizon, ")", sep = ""), "HP-concurrent")

plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
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


#-----------------------------------------------------------------------------------
# Time-shift functions
#
# The time-shift function measures the time delay (in units of time steps) introduced
# by the filter at each frequency. A larger (more negative) phase-shift means a greater lag.
#
# Key observation:
#   SSA exhibits a slightly larger time-shift than HP concurrent,
#   This confirms — now in the frequency domain — the ~half-unit lag identified by
#   the tau-statistic in section 2.4.
#
# Interpretation:
#   The small additional lag of SSA relative to HP concurrent is the direct consequence
#   of the stronger smoothing imposed by the ht constraint: to suppress more
#   high-frequency noise, the filter redistributes weight toward longer lags,
#   which introduces a modest timing cost.

mplot<-cbind(amp_obj_SSA$shift,amp_obj_HP$shift)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# The larger phase-lag of SSA implies a slight lag relative to HP-concurrent: roughly half a time-unit (confirming the above tau-statistic in the time-domain)
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



#-----------------------------------------------------------------------------------
# Summary of Example 2 (sections 2.4–2.5):
#
# The time-domain (tau-statistic) and frequency-domain (amplitude/phase) analyses
# together provide a consistent and complete picture of the SSA–HP tradeoff:
#
#   Smoothness gain:  SSA produces ~33% fewer zero-crossings than HP concurrent
#                     --> explained by stronger high-frequency attenuation in the stopband
#
#   Timing cost:      SSA lags HP concurrent by approximately half a time unit
#                     --> explained by the larger phase-shift in the passband
#
# This smoothness–timeliness tradeoff is the fundamental design tension addressed
# by the SSA optimization framework: the ht constraint directly controls where on
# this tradeoff curve the filter is placed.


###########################################################################################################
###########################################################################################################
# Example 3: Addressing Timing Lags via the Forecast Horizon Parameter
#
# Background:
#   In Example 2 we observed that the SSA filter (with ht = 12) lags the classic HP
#   concurrent filter by approximately half a time unit — the timing cost of imposing
#   stronger smoothing. In practice, a real-time analyst may wish to compensate for
#   this lag by building a lead directly into the SSA filter design.
#
# Approach:
#   SSA allows the user to request a lead by setting forecast_horizon > 0.
#   Here we consider a one-year-ahead forecast (forecast_horizon = 12 months).
#   See Wildi, M. (2024), for the
#   theoretical foundation.
#
# Key design choice — holding time is kept fixed (ht = 12, same as Example 2):
#   By holding ht constant across Examples 2 and 3, we isolate the pure effect of
#   changing forecast_horizon. The resulting SSA filter will:
#     - Retain the same degree of smoothness (noise suppression) as Example 2
#     - Shift from a slight lag (Example 2) to a slight lead (Example 3)
#
# Fundamental tradeoff (the SSA trilemma):
#   SSA keeps smoothness fixed and exchanges lead/lag against MSE performance.
#   Asking for both a lead AND strong smoothing necessarily degrades MSE accuracy
#   relative to Example 2 (a nowcast). This three-way tension between smoothness,
#   timeliness, and accuracy is the SSA trilemma — see Tutorial 0.1 and Example 8
#   below for a detailed treatment.


#-----------------------------------------------------------------------------------
# 3.1 SSA Setup and Hyperparameter Choices

# Step 1: Forecast horizon
#   forecast_horizon = 12: one-year-ahead forecast
#   Compared to Example 2 (forecast_horizon = 0), this shifts the SSA filter output
#   forward in time, converting the half-unit lag into a lead
forecast_horizon <- 12

# Step 2: Holding time (unchanged from Example 2)
#   Keeping ht = 12 ensures that the smoothness level is held constant,
#   so that any differences from Example 2 are attributable solely to the
#   change in forecast horizon
ht <- 12

# Step 3: Convert holding time to lag-one autocorrelation (rho1) for SSA input
rho1 <- compute_rho_from_ht(ht)

# Confirm the implied zero-crossing reduction relative to HP concurrent (~33% fewer)
ht / ht_hp

# Step 4: Specify the target filter
#   As in Example 2, we target the classic HP concurrent filter
gammak_generic <- hp_trend

# Step 5: Input process assumption
#   White noise (xi = NULL, default) — same as Examples 1 and 2


#-----------------------------------------------------------------------------------
# Run SSA optimization with one-year forecast horizon
SSA_obj_HP <- SSA_func(L, forecast_horizon, gammak_generic, rho1)

# Extract SSA filter coefficients
# Since xt is white noise (xi = NULL), ssa_eps and ssa_x are identical
# (deconvolution reduces to an identity transformation — see Tutorial 1)
ssa_x       <- SSA_obj_HP$ssa_x
SSA_filt_HP <- ssa_eps <- SSA_obj_HP$ssa_eps


#-----------------------------------------------------------------------------------
# Plot: SSA filter weights (forecast_horizon = 12) vs. HP concurrent filter weights
#
# What to look for:
#
# 1. Scale difference (zero-shrinkage):
#    The SSA filter weights are visibly smaller in magnitude than the HP concurrent weights.
#    This "zero-shrinkage" effect is an inherent consequence of simultaneously demanding
#    a lead AND strong smoothing:
#      - Both requirements pull the filter away from the MSE-optimal design
#      - The optimizer compensates by shrinking the overall filter gain
#      - Rescaling (inflating) the SSA filter weights is possible but would worsen
#        MSE performance further; the current scaling is already optimal
#    Note: zero-crossings and correlations are scale-invariant, so shrinkage does
#    not affect the smoothness or timing properties of the filter output.
#
# 2. Boundary constraint tip:
#    As in Example 2, the SSA filter exhibits a characteristic spike at the most
#    recent lag (lag 0), arising from the implicit constraint that the filter
#    coefficient at lag -1 must vanish. See Theorem 1 in the JBCY paper.

mplot <- cbind(ssa_x, hp_trend)
colnames(mplot) <- c(paste("SSA(", ht, ",", forecast_horizon, ")", sep = ""), "HP-concurrent")
colo <- c("blue", "green")

par(mfrow=c(1,1))
# The typical noise-shape or tip of SSA is due to an implicit boundary constraint which states that the coefficient at lag -1 vanishes, see theorem 1 in JBCY paper
plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


#-----------------------------------------------------------------------------------
# 3.2 Simulation Validation: Compare SSA Output with Classic HP Concurrent
#
# We filter a long white noise series through both filters and verify:
#   (i)   SSA output matches the imposed holding time constraint (ht = 12)
#   (ii)  HP concurrent output matches its baseline holding time (ht_hp)
#   (iii) SSA now leads HP concurrent — in contrast to the slight lag in Example 2

len <- 100000
set.seed(1)

# Simulate a long white noise series (AR coefficient a1 = 0 reduces AR(1) to white noise)
a1 <- 0
x  <- arima.sim(n = len, list(ar = a1))

# Apply SSA filter (one-sided, causal)
yhat <- filter(x, SSA_filt_HP, side = 1)

# Verify SSA holding time: empirical value should match imposed constraint ht = 12
ht                               # imposed constraint
compute_empirical_ht_func(yhat)  # empirical holding time — should be close to ht

# Apply classic HP concurrent filter
HP_concurrent <- filter(x, hp_trend, side = 1)

# Verify HP concurrent holding time: empirical value should match baseline ht_hp
ht_hp                                        # theoretical baseline
compute_empirical_ht_func(HP_concurrent)     # empirical holding time — should be close to ht_hp

# As expected, SSA is smoother: it produces ~33% fewer zero-crossings than HP concurrent
# (same smoothness level as Example 2, since ht is unchanged)


#-----------------------------------------------------------------------------------
# Plot: Time-domain comparison of SSA (forecast_horizon=12) vs. HP concurrent
#
# Both series are rescaled to unit variance for visual comparability.
# Note: SSA filter output has arbitrary scaling (zero-shrinkage); rescaling does not
#       affect zero-crossings or correlations, which are scale-invariant.
#
# What to look for:
#   1. Zero-crossing frequency:
#      SSA generates ~33% fewer zero-crossings than HP concurrent (same as Example 2),
#      confirmed by the holding time check above. Extra HP crossings cluster near the
#      zero line where the signal hovers — SSA's stronger stopband attenuation suppresses
#      these spurious reversals.
#
#   2. Timing (the key difference from Example 2):
#      SSA now visibly LEADS HP concurrent, rather than lagging it as in Example 2.
#      This lead is noteworthy because the classic HP concurrent filter is itself
#      considered a relatively fast real-time estimator — SSA with forecast_horizon=12
#      manages to anticipate it by a meaningful margin.
#
#   3. Tracking during trends:
#      At clear up- and downswings (filter output well away from zero), SSA tracks
#      the HP target closely, reflecting the SSA optimality principle (maximizing
#      correlation with the target).



mplot<-na.exclude(scale(cbind(yhat,HP_concurrent),scale=T,center=F))
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# Plot a short sample of the series
par(mfrow=c(1,1))

anf<-3000
enf<-3300

ts.plot(mplot[anf:enf,],col=colo)
mtext(paste("SSA(",ht,",",forecast_horizon,")",sep=""),col=colo[1],line=-1)
mtext("HP-concurrent",col=colo[2],line=-2)
abline(h=0)




#-----------------------------------------------------------------------------------
# 3.3 Shift Analysis: Confirming the Lead of SSA over HP Concurrent
#
# We apply the tau-statistic (Wildi, 2024) to quantify the relative timing.
# Recall from Examples 2 and 2.4:
#   - Trough to the LEFT  of zero --> series in column 1 (SSA) LEADS HP concurrent
#   - Trough to the RIGHT of zero --> series in column 1 (SSA) LAGS  HP concurrent
#
# Expected result: the trough should now be left-shifted (SSA leads),
#   confirming that forecast_horizon = 12 has successfully converted the
#   half-unit lag of Example 2 into a measurable lead.

shift_tau_obj <- compute_min_tau_func(mplot)


#-----------------------------------------------------------------------------------
# 3.4 Frequency-Domain Analysis: Amplitude and Phase-Shift
#
# We examine the amplitude and phase-shift functions to provide a formal
# frequency-domain explanation of the time-domain findings in sections 3.2 and 3.3.
# We compare against the HP concurrent results from Example 2 to isolate the
# effect of changing forecast_horizon from 0 to 12.

K <- 600
amp_obj_SSA <- amp_shift_func(K, as.vector(SSA_filt_HP), F)
amp_obj_HP  <- amp_shift_func(K, hp_trend, F)

par(mfrow = c(1, 1))

#-----------------------------------------------------------------------------------
# Amplitude functions
#
#
# Key observations (comparing SSA with forecast_horizon=12 to HP concurrent):
#
#   1. Stopband attenuation (high frequencies):
#      SSA's amplitude is again closer to zero in the stopband than HP concurrent —
#      the same stronger noise suppression seen in Example 2. This confirms that
#      the ht constraint continues to drive the smoothness property regardless of
#      the forecast horizon.
#
#   2. Bandpass morphing:
#      Unlike the lowpass-shaped amplitude of Example 2's SSA, the amplitude here
#      takes on a bandpass-like shape. This is a direct consequence of requesting
#      a lead: to anticipate the target, the filter must emphasize the mid-frequency
#      components that carry the most predictive timing information, at the expense
#      of some low-frequency gain.
#
#   3. Zero-shrinkage:
#      The overall amplitude level of SSA is reduced relative to HP concurrent
#      (see also section 3.1), reflecting the scale penalty of simultaneously
#      demanding a lead and strong smoothing.

mplot<-scale(cbind(amp_obj_SSA$amp,amp_obj_HP$amp),scale=T,center=F)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# Once again, the amplitude function of SSA is closer to zero in stop band: stronger noise suppression, less noisy crossings
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
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

#-----------------------------------------------------------------------------------
# Time-shift functions
#
#
# Key observations:
#
#   1. SSA now leads HP concurrent:
#      The SSA time-shift is SMALLER  than that
#      of HP concurrent across most of the passband. This is the frequency-domain
#      confirmation of the lead observed in sections 3.2 and 3.3.
#
#   2. Contrast with Example 2:
#      In Example 2 (forecast_horizon=0), SSA had a slightly larger time-shift
#      than HP concurrent (a small lag). Here (forecast_horizon=12), the ordering
#      is reversed — SSA leads. Changing forecast_horizon is the mechanism that
#      controls this reversal.
#
#   3. Amplitude-shift linkage:
#      For minimum-phase filters, amplitude and time-shift functions are linked
#      bijectively (via the Hilbert transform). Consequently, the stronger noise
#      suppression imposed by the ht constraint also influences the shape of the
#      time-shift function — the two cannot be adjusted fully independently.
#
# Note: Very large (absolute) phase-shift values at the boundary frequencies are
#   clipped to -5 for visual clarity; the first entry is set to 0.
mplot<-cbind(amp_obj_SSA$shift,amp_obj_HP$shift)
mplot[which(mplot[,1]<(-5)),1]<--5
mplot[1,1]<-0
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")

plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase shift",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
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

#-----------------------------------------------------------------------------------
# Summary of Example 3 (sections 3.2–3.4):
#
# By increasing forecast_horizon from 0 (Example 2) to 12 (Example 3), while
# keeping ht fixed, SSA successfully converts a half-unit lag into a measurable lead:
#
#   Smoothness:     Unchanged — SSA still produces ~33% fewer crossings than HP concurrent
#                   --> confirmed by holding time checks and stopband attenuation
#
#   Timing:         SSA now LEADS HP concurrent (vs. slight lag in Example 2)
#                   --> confirmed by tau-statistic (trough left of zero) and
#                       smaller passband phase-shift
#
#   MSE cost:       Requesting a lead degrades MSE accuracy relative to Example 2
#                   (zero-shrinkage in amplitude is the visible symptom)
#                   --> this is the smoothness–timeliness–accuracy trilemma in action
#
# Effects of forecast_horizon > 0 on filter properties (summary):
#   1. Zero-shrinkage: overall amplitude level is reduced
#   2. Bandpass morphing: amplitude shifts from lowpass toward bandpass shape
#   3. Phase lead: phase-shift becomes smaller (less negative) in the passband
#
# Effects of the ht constraint on filter properties (summary):
#   1. Stronger stopband attenuation: amplitude closer to zero at high frequencies
#   2. Phase-shift influence: via the amplitude-shift bijection for minimum-phase filters

########################################################################################################
##########################################################################################################
# Example 4: Playing with the Target and Forecast Horizon
#
# Overview:
#   This example builds on Example 1, but replaces the one-sided (causal) MSE target with the
#   symmetric (two-sided) HP filter as the optimization target for SSA.
#
# Key challenge:
#   - The hp_target provided by the R-package mFilter is a causal (one-sided) filter.
#     However, for a symmetric two-sided HP filter, we need a non-causal (two-sided) implementation.
#   - To work around this, we do NOT shift the target directly. Instead, we encode the non-causality
#     by specifying a suitable forecast horizon that corresponds to the center of the symmetric filter.
#     This effectively "shifts" the filter indirectly.

#--------------------------------------------------------------------------------------------------
# 4.1 Compute the Symmetric HP Filter and Compare Concurrent MSE and SSA Designs
#--------------------------------------------------------------------------------------------------

# Filter length: we use twice the length (L=401) compared to Examples 1 & 2 (L=201).
# The reason: a symmetric two-sided filter is split into left and right halves, each of length
# (L-1)/2. Using L=401 ensures both halves have length 200, matching the L=201 filter in Example 1.
L_sym <- 401

# The filter length must be an odd number so that the symmetric HP filter is correctly centered.
# If L_sym is even, the filter center is ambiguous, leading to incorrect centering.
if (L_sym / 2 == as.integer(L_sym / 2)) {
  print("Filter length should be an odd number.")
  print("If L_sym is even, the HP filter cannot be centered correctly.")
  L_sym <- L_sym + 1  # Increment by 1 to make it odd
}

# Set the HP smoothing parameter for monthly data.
# lambda=14400 is the standard choice for monthly business cycle analysis (analogous to lambda=1600
# for quarterly data, scaled by 4^2 = 16 for the frequency conversion).
lambda_monthly <- 14400

# Compute the symmetric HP filter and its associated MSE-based design.
# HP_target_mse_modified_gap() returns a list containing:
#   - $target: the bi-infinite (symmetric, two-sided) HP filter coefficients
#   - Other components related to the MSE design
HP_obj <- HP_target_mse_modified_gap(L_sym, lambda_monthly)

# Extract the bi-infinite (symmetric two-sided) HP filter coefficients.
# This is our NEW optimization target, replacing the one-sided MSE filter used in Example 1.
hp_target <- HP_obj$target

# Note: The filter as returned by mFilter is causal (one-sided). 
# We will account for its symmetric (non-causal) nature via the forecast_horizon parameter below.

# Visual inspection of the HP target filter coefficients
par(mfrow = c(1, 1))
ts.plot(hp_target,
        main = "Symmetric HP Filter Coefficients",
        ylab = "Filter Weights",
        xlab = "Lag")

#--------------------------------------------------------------------------------------------------
# 4.2 SSA Design: Hyperparameters and Filter Specification
#--------------------------------------------------------------------------------------------------

# Holding time (ht): controls the smoothness/speed of the SSA filter.
# We use the same holding time as in Example 1, scaled from the MSE benchmark.
ht <- 1.5 * ht_mse

# Compute the lag-one autocorrelation (rho1) corresponding to the chosen holding time.
# SSA_func() requires rho1 as input (not ht directly).
# compute_rho_from_ht() performs the conversion: higher ht => slower filter => larger rho1.
rho1 <- compute_rho_from_ht(ht)

# Specify the assumed data-generating process (DGP) for the input series.
# xi=NULL means we assume white noise (the default in SSA_func).
# If the data were autocorrelated (e.g., AR process), xi would contain the AR coefficients.
xi <- NULL

# Supply the symmetric HP filter as the optimization target for SSA.
# gammak_generic holds the filter coefficients of the target (here: two-sided HP filter).
gammak_generic <- hp_target

# Specify the forecast horizon to encode the non-causality of the symmetric target.
# The symmetric HP filter is centered at position (L_sym+1)/2 in the coefficient vector.
# Since the filter is two-sided, the effective "center lag" is at (L_sym-1)/2.
# By setting forecast_horizon = (L_sym-1)/2, we instruct SSA to treat the filter as
# if it requires predicting (L_sym-1)/2 steps into the future, correctly reflecting
# the non-causal (symmetric) nature of the target.
forecast_horizon <- (L_sym - 1) / 2

# Set the one-sided (causal) filter length for SSA, matching Example 1.
L <- 201

# Compute the SSA filter targeting the symmetric HP filter.
# Parameters:
#   L                 : one-sided filter length (number of filter coefficients)
#   forecast_horizon  : encodes the non-causal shift of the symmetric target
#   gammak_generic    : target filter coefficients (symmetric HP)
#   rho1              : lag-one autocorrelation of the input (derived from ht)
# Missing parameters (xi, etc.) use SSA_func defaults.
SSA_obj_HP <- SSA_func(L, forecast_horizon, gammak_generic, rho1)

# Extract the SSA filter coefficients for further analysis and comparison
SSA_filt_HP <- SSA_example4 <- SSA_obj_HP$ssa_x

#--------------------------------------------------------------------------------------------------
# 4.3 Plot: Comparison of SSA Filters from Example 1 and Example 4
#--------------------------------------------------------------------------------------------------
# Key result: The SSA filters from Example 1 and Example 4 are IDENTICAL!
#
# Why? Because:
#   - In Example 1, the target is the one-sided MSE predictor of the HP filter (hp_mse).
#   - In Example 4, the target is the symmetric HP filter (hp_target).
#   - When the input is white noise (xt = epsilon_t), the MSE predictor of the symmetric
#     HP filter IS exactly hp_mse.
#   - Therefore, SSA optimization is unaffected by this substitution.
#     (See Wildi, M. (2024) for the theoretical proof:
#      the two-sided target can be replaced by its MSE predictor without changing the SSA solution.)

colo <- c("black", "brown", "blue")
par(mfrow = c(1, 1))

# Combine filter coefficients from both examples for side-by-side comparison
mplot <- cbind(SSA_example1, SSA_example4)
colnames(mplot) <- c("Example 1", "Example 4")

# Plot Example 1 SSA filter (black)
par(mfrow=c(1,1))
mplot<-cbind(SSA_example1,SSA_example4)
colnames(mplot)<-c("Example 1","Example 4")

# Both filters overlap: they are identical
plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

#--------------------------------------------------------------------------------------------------
# 4.4 Criterion Values: Analytical and Empirical Correlations
#--------------------------------------------------------------------------------------------------
# SSA computes two distinct criterion values:
#
# Criterion 1: crit_rhoyz
#   - Correlation between the SSA filter output and the one-sided MSE filter output (hp_mse).
#   - This is the SAME as in Example 1, confirming that both SSA filters are identical.
#
# Criterion 2: crit_rhoy_target
#   - Correlation between the SSA filter output and the EFFECTIVE target (symmetric HP filter).
#   - This correlation is SMALLER than crit_rhoyz because the symmetric (non-causal) target
#     requires predicting future values. A causal filter (SSA) cannot perfectly replicate
#     a non-causal filter, so tracking performance is inherently reduced.

# Display both criterion values for comparison
SSA_obj_HP$crit_rhoyz     # Should match the criterion from Example 1
crit_example1             # Reference value from Example 1

SSA_obj_HP$crit_rhoy_target  # Correlation with the symmetric (non-causal) HP target

#--------------------------------------------------------------------------------------------------
# Empirical Validation: Verify Analytical Criteria via Simulation
#--------------------------------------------------------------------------------------------------

# a. Generate a very long time series for accurate empirical correlation estimates.
#    Using len=1,000,000 observations ensures Monte Carlo error is negligible.
len <- 1000000
set.seed(14)  # Set seed for reproducibility
x <- arima.sim(n = len, list(ar = a1))  # Simulate AR(1) process with coefficient a1

# b. Compute filter outputs for all three filters under comparison:

# SSA-HP filter output (causal, one-sided)
yhat <- filter(x, SSA_filt_HP, side = 1)

# Symmetric HP filter output (non-causal, two-sided):
# side=2 instructs filter() to apply the filter symmetrically (centered convolution)
HP_symmetric <- filter(x, hp_target, side = 2)

# Concurrent MSE HP filter output (causal, one-sided):
# side=1 instructs filter() to use only past and current values
HP_concurrent <- filter(x, hp_mse, side = 1)

# c. Compute the empirical cross-correlation matrix between all three filter outputs.
#    We scale (standardize) all series but do NOT center them (center=F), since
#    the filters have zero mean by construction.
mplot <- na.exclude(scale(cbind(yhat, HP_concurrent, HP_symmetric),
                          scale  = T,
                          center = F))
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", forecast_horizon, ")", sep = ""),
  "HP-MSE (one-sided)",
  "HP-Symmetric (effective target)"
)

# Print the empirical correlation matrix
cor(mplot)

# Interpretation of results:
#   - Row 1 of the correlation matrix shows the empirical correlations of the SSA filter output
#     with HP-MSE and HP-Symmetric respectively.
#   - These empirical values should closely match the analytical criteria:
#       * cor(SSA, HP-MSE)       ≈ crit_rhoyz       (correlation with one-sided MSE filter)
#       * cor(SSA, HP-Symmetric) ≈ crit_rhoy_target  (correlation with two-sided target)
#   - Note: The MSE filter has a slightly HIGHER correlation with the symmetric HP target
#     than SSA does, by design — the MSE filter is specifically constructed to maximize
#     this target correlation (at the cost of timeliness/phase adjustment).



######################################################################################################################
######################################################################################################################
# Example 5: Working with Autocorrelated Data (Instead of White Noise)
#
# Overview:
#   Previous examples assumed white noise input (xt = epsilon_t). In practice, macroeconomic
#   time series are autocorrelated (even after first differences). This example demonstrates how to correctly apply SSA when
#   the input data follows an ARMA process.
#
# Key idea:
#   - The HP-concurrent filter is first applied to an ARMA(1,1) series.
#   - The holding time (a measure of cycle smoothness / zero-crossing frequency) must be
#     recomputed to account for autocorrelation via the Wold decomposition.
#   - SSA is then "engrafted" onto the HP-concurrent filter, using the corrected holding time
#     and the Wold decomposition (MA-inversion) of the ARMA process as additional input.
#   - Two SSA filter representations are distinguished:
#       * ssa_x   : the filter applied directly to the observed series xt
#       * ssa_eps : the equivalent filter applied to the white noise innovations epsilon_t

######################################################################################################################
# 5.1 Generate and Filter Autocorrelated Data
######################################################################################################################

set.seed(4)       # Set seed for reproducibility of the simulated series
len <- 1200       # Sample length: 1200 observations (e.g., 100 years of monthly data)

# ARMA(1,1) coefficients for the data-generating process:
#   xt = a1 * x_{t-1} + epsilon_t + b1 * epsilon_{t-1}
a1 <- 0.3         # AR(1) coefficient: mild positive autocorrelation
b1 <- 0.2         # MA(1) coefficient: mild positive moving average component

# Simulate an ARMA(1,1) time series with the specified coefficients
x <- arima.sim(n = len, list(ar = a1, ma = b1))

# Estimate the ARMA(1,1) parameters from the simulated data.
# In practice, the true parameters are unknown and must be estimated from data.
# Here we use the known model order (1,0,1) for estimation.
estim_obj <- arima(x, order = c(1, 0, 1))

# Diagnostic check: inspect residual plots and Ljung-Box test statistics.
# A well-fitting model should show:
#   - Residuals resembling white noise (no visible patterns)
#   - ACF of residuals within confidence bands
#   - Non-significant p-values in the Ljung-Box test
tsdiag(estim_obj)   # -> Diagnostics look OK

# Apply the HP-concurrent filter (hp_trend) to the observed series xt.
# side=1 ensures the filter is causal (one-sided), using only past and current values.
# na.exclude() removes the leading NAs introduced by the filter warm-up period.
y_hp <- na.exclude(filter(x, hp_trend, side = 1))

# Visual inspection of the HP-filtered output (trend/cycle estimate)
ts.plot(y_hp,
        main = "HP-Concurrent Filter Output (ARMA(1,1) Input)",
        ylab = "Filtered Series",
        xlab = "Time")

######################################################################################################################
# 5.2 Holding Time: Correcting for Autocorrelation via the Wold Decomposition
######################################################################################################################
# The holding time (ht) measures the expected number of observations between consecutive
# zero-crossings of the filter output. A larger ht => smoother output with fewer sign changes.
#
# IMPORTANT: The standard holding time formula assumes xt = epsilon_t (white noise).
# When xt is autocorrelated, this formula is INCORRECT and will underestimate the true ht.
# We must account for the autocorrelation structure via the Wold decomposition.

# Visual inspection of the HP-concurrent filter coefficients
ts.plot(hp_trend,
        main = "HP-Concurrent Filter Coefficients",
        ylab = "Filter Weights",
        xlab = "Lag")

# Compute the theoretical holding time of the HP-concurrent filter under white noise assumption.
# This gives the "naive" ht that ignores the ARMA autocorrelation of xt.
ht_hp_trend_obj <- compute_holding_time_func(hp_trend)
ht_hp <- ht_hp_trend_obj$ht
ht_hp   # Print the theoretical (white noise-based) holding time

# Compare with the empirical holding time computed directly from the filtered output y_hp.
# The empirical ht is LARGER than the theoretical ht under white noise, because:
#   - xt is positively autocorrelated (ARMA(1,1) with a1>0, b1>0)
#   - Positive autocorrelation in xt induces additional smoothness in the filter output
#   - The naive formula (assuming white noise) fails to capture this extra smoothness
compute_empirical_ht_func(y_hp)   # Should be noticeably larger than ht_hp

# ------ Wold Decomposition (MA-Inversion) ------
# To correctly compute the holding time for autocorrelated data, we decompose xt into
# its Wold representation: xt = xi(L) * epsilon_t, where xi(L) is the MA(inf) polynomial.
#
# For an ARMA(1,1) process, the Wold decomposition is:
#   xt = (1 + b1*L) / (1 - a1*L) * epsilon_t
#
# Key considerations:
#   - L (filter length) should be large enough that xi has decayed sufficiently close to zero.
#     If xi has NOT decayed to zero at lag L, finite-length convolutions are poor approximations
#     of the corresponding infinite-length transformations (see Section 2 of the JBCY paper).
#   - For integrated processes (e.g., I(1)), xi does NOT decay to zero. These cases are handled
#     separately in Tutorials 3 (Hamilton filter), 4 (Baxter-King filter), and 5 (HP-gap filter).
#   - One can use EITHER the true parameters (a1, b1) or the estimated parameters from estim_obj.
#     Here we use the true parameters since we are working with simulated data.

# Step 1: Compute the MA(inf) Wold decomposition of the ARMA(1,1) process.
# ARMAtoMA() converts the ARMA coefficients to the equivalent MA(inf) representation.
# We prepend 1 to represent the coefficient at lag 0 (the contemporaneous term).
xi_data <- c(1, ARMAtoMA(ar = a1, ma = b1, lag.max = L - 1))

# Visual check: verify that xi_data has decayed sufficiently close to zero by lag L.
# If the series is still large at the right edge, consider increasing L.
par(mfrow = c(1, 1))
ts.plot(xi_data,
        main = "Wold Decomposition (MA-Inversion) of ARMA(1,1)",
        ylab = "xi Coefficients",
        xlab = "Lag")

# Step 2: Convolve xi_data with hp_trend to obtain the equivalent filter applied to epsilon_t.
# The convolved filter hp_conv represents the combined effect of:
#   (a) The ARMA autocorrelation structure (xi_data)
#   (b) The HP-concurrent filter (hp_trend)
# Applied to the white noise innovations epsilon_t, hp_conv produces the same output as
# hp_trend applied to xt. (See Section 2 of the JBCY paper for derivation.)
hp_conv <- conv_two_filt_func(xi_data, hp_trend)$conv

# Compute the holding time of the convolved filter (relative to white noise epsilon_t).
# This is the CORRECTED holding time that properly accounts for ARMA autocorrelation.
ht_hp_conv_obj <- compute_holding_time_func(hp_conv)
ht_hp_conv_obj$ht   # Print the corrected theoretical holding time

# Verify: the corrected theoretical ht should now match the empirical ht from the filtered data.
# Any remaining discrepancy is due to finite sample error (the empirical ht converges to the
# theoretical value as sample length increases).
compute_empirical_ht_func(y_hp)   # Should now closely match ht_hp_conv_obj$ht

# Remark on practical implications:
#   Applying the same fixed HP-concurrent design to macroeconomic indicators with different
#   autocorrelation structures (e.g., INDPRO vs. non-farm payroll) produces qualitatively
#   different business cycles — even if lambda is the same. This was highlighted in Tutorial 2.0.
#   The Wold decomposition-based correction is essential for consistent cross-series comparisons.

######################################################################################################################
# 5.3 SSA Design: Hyperparameters and Filter Specification for Autocorrelated Data
######################################################################################################################

# Target holding time for SSA:
#   We want SSA to produce a SMOOTHER output than HP-concurrent (i.e., fewer zero-crossings).
#   Strategy: set ht = 1.5 * (corrected HP holding time), which increases the holding time
#   by 50%, corresponding to approximately 33% fewer zero-crossings than HP-concurrent.
#
# IMPORTANT: We use ht_hp_conv_obj$ht (the CORRECTED holding time) as the baseline, NOT
#   the naive ht_hp (which assumes white noise). Using the uncorrected value would lead to
#   an incorrect smoothness specification when xt is autocorrelated.
ht <- 1.5 * ht_hp_conv_obj$ht

# Convert the holding time to the lag-one autocorrelation (rho1) required by SSA_func().
# SSA_func() takes rho1 as input (not ht directly).
# compute_rho_from_ht() performs the conversion: larger ht => slower filter => larger rho1.
rho1 <- compute_rho_from_ht(ht)

# Verify the holding time ratio: SSA should have 50% longer holding time than HP
# (equivalently ~33% fewer zero-crossings)
ht / ht_hp_conv_obj$ht   # Should print 1.5

# Set the forecast horizon: delta=0 means we are estimating the concurrent (nowcast) value.
# No future observations are used; only past and present data are available.
forecast_horizon <- 0

# Specify the optimization target: we "engraft" SSA onto the classic HP-concurrent filter.
# SSA will find the causal filter that best approximates hp_trend while respecting
# the smoothness constraint encoded in rho1.
gammak_generic <- hp_trend

# Provide the Wold decomposition (MA-inversion) of the ARMA process to SSA_func().
# This is CRITICAL when xt is autocorrelated:
#   - If xi=NULL (default), SSA_func() assumes xt = epsilon_t (white noise). This is WRONG here.
#   - By providing xi_data, SSA_func() correctly accounts for the ARMA autocorrelation
#     in both the criterion and the filter design.
xi <- xi_data

# Compute the SSA filter for the autocorrelated ARMA(1,1) data.
# Parameters:
#   L                : one-sided filter length (number of causal coefficients)
#   forecast_horizon : nowcast (delta=0), no future values used
#   gammak_generic   : target filter (HP-concurrent)
#   rho1             : lag-one ACF derived from the target holding time
#   xi               : Wold decomposition of ARMA(1,1) — accounts for autocorrelation
SSA_obj_HP <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract the two filter representations returned by SSA_func():
#
# ssa_x (filter applied to observed series xt):
#   - This is the operationally relevant filter for real data applications.
#   - Applied directly to xt to obtain the SSA output.
#
# ssa_eps (filter applied to white noise innovations epsilon_t):
#   - Equivalent representation in the innovation domain.
#   - ssa_eps = convolution(ssa_x, xi_data): it combines the SSA filter with the Wold decomp.
#   - Useful for theoretical analysis and criterion evaluation.
#   - When xt = epsilon_t (white noise), ssa_x == ssa_eps (they coincide).
SSA_filt_HP <- ssa_x <- SSA_example5 <- SSA_obj_HP$ssa_x   # Filter for observed data xt
ssa_eps <- SSA_obj_HP$ssa_eps                                # Filter for innovations epsilon_t

######################################################################################################################
# 5.4 Plot: Compare Filters in Both the xt and epsilon_t Domains
######################################################################################################################

par(mfrow = c(2, 1))

# ------ Panel 1: Filters as Applied to xt (Observed Series) ------
# These are the operationally relevant filters for real-world applications.
# Note the characteristic "noise-shape" or "tip" in the SSA filter near lag 0:
#   This is a consequence of an implicit boundary constraint in SSA, which enforces
#   that the filter coefficient at lag -1 vanishes (see Theorem 1 in the JBCY paper).
par(mfrow=c(2,1))
# Filters applied to xt (these are mostly relevant in applications)
mplot<-cbind(SSA_filt_HP,hp_trend)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP-concurrent")
colo<-c("blue","green")
# The typical noise-shape or tip of SSA is due to an implicit boundary constraint which states that the coefficient at lag -1 vanishes, see theorem 1 in JBCY paper
plot(mplot[,1],main=paste("HP(",lambda_monthly,"): filters as applied to xt",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()
# ------ Panel 2: Filters as Applied to epsilon_t (White Noise Innovations) ------
# These are the innovation-domain representations (ssa_eps and hp_conv).
# Comparing in the epsilon_t domain allows direct assessment of filter properties
# under white noise, since all autocorrelation has been "removed" by the Wold decomposition.
# Note: the SSA "tip" near lag 0 is again visible in ssa_eps.
mplot<-cbind(ssa_eps,hp_conv)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP-concurrent")
colo<-c("blue","green")
# The typical noise-shape or tip of SSA is due to an implicit boundary constraint which states that the coefficient at lag -1 vanishes, see theorem 1 in JBCY paper
plot(mplot[,1],main=paste("HP(",lambda_monthly,"): filters as applied to epsilont",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


######################################################################################################################
# 5.4 Filter Series and Compare Classic Concurrent HP with SSA
#
# Overview:
#   We now apply both the SSA filter and the classic HP-concurrent filter to a long simulated
#   ARMA(1,1) series and compare their outputs empirically.
#
# Goals:
#   1. Verify that the empirical holding time of SSA matches the target (ht = 1.5 * ht_HP).
#   2. Confirm that SSA produces ~50% fewer zero-crossings than HP-concurrent.
#   3. Visually inspect the qualitative difference between the two filter outputs.
######################################################################################################################

# Generate a long ARMA(1,1) series for accurate empirical evaluation.
# Using len=100,000 observations ensures that empirical statistics (e.g., holding times,
# zero-crossing counts) converge closely to their theoretical population values,
# minimizing Monte Carlo estimation error.
len <- 100000
set.seed(1)   # Set seed for reproducibility
x <- arima.sim(n = len, list(ar = a1, ma = b1))   # ARMA(1,1) with coefficients a1, b1 from Section 5.1

#------------------------------------------------------
# Compute and Validate SSA Filter Output
#------------------------------------------------------

# Apply the SSA filter (ssa_x, the filter for observed series xt) to the simulated data.
# side=1 ensures causal (one-sided) filtering: only past and current values of x are used.
yhat <- filter(x, SSA_filt_HP, side = 1)

# Validate the holding time of the SSA output:
#   - ht          : the TARGET holding time we specified when designing the SSA filter (Section 5.3)
#   - compute_empirical_ht_func(yhat): the EMPIRICAL holding time measured from the filter output
# These two values should match closely, confirming that SSA correctly implements the
# smoothness constraint. Any small discrepancy is due to finite sample variability.
ht                                  # Target (theoretical) holding time for SSA
compute_empirical_ht_func(yhat)     # Empirical holding time of SSA output — should ≈ ht

#------------------------------------------------------
# Compute and Validate HP-Concurrent Filter Output
#------------------------------------------------------

# Apply the classic HP-concurrent filter (hp_trend) to the same simulated series.
# side=1 ensures one-sided (causal) filtering, consistent with a real-time nowcasting setting.
HP_concurrent <- filter(x, hp_trend, side = 1)

# Validate the holding time ratio between SSA and HP-concurrent:
#   - ht_hp_conv_obj$ht          : corrected theoretical holding time of HP-concurrent
#                                  (accounts for ARMA autocorrelation via Wold decomposition)
#   - compute_empirical_ht_func(): empirical holding time of HP-concurrent output
#
# Expected result:
#   empirical ht(SSA) ≈ 1.5 * empirical ht(HP-concurrent)
#   i.e., SSA produces approximately 33% fewer zero-crossings than HP-concurrent.
ht_hp_conv_obj$ht                        # Corrected theoretical holding time of HP-concurrent
compute_empirical_ht_func(HP_concurrent) # Empirical holding time of HP-concurrent — should ≈ ht_hp_conv_obj$ht

#------------------------------------------------------
# Prepare Data for Plotting
#------------------------------------------------------

# Combine both filter outputs into a matrix, removing leading NAs from filter warm-up.
# scale() is NOT applied here: we want to compare the raw filter outputs directly,
# preserving their relative amplitudes for visual comparison.
mplot <- na.exclude(cbind(yhat, HP_concurrent))
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", forecast_horizon, ")", sep = ""),
  "HP-Concurrent"
)

#------------------------------------------------------
# Plot: Visual Comparison of SSA and HP-Concurrent Outputs
#------------------------------------------------------

par(mfrow = c(1, 1))

# Define the plotting window: a short subsample of 500 observations is sufficient
# to visually assess zero-crossing behavior and smoothness differences.
# Two candidate windows are defined below; the second window [1000:1500] is used.
# (The first window [500:1000] is overridden — adjust anf/enf to explore other segments.)
anf <- 500   # Start index (overridden below)
enf <- 1000  # End index (overridden below)
anf <- 1000  # Active start index: observation 1000
enf <- 1500  # Active end index: observation 1500

# Plot both filter outputs over the selected subsample.
# Key observations to look for:
#   1. SSA (blue) produces visibly fewer zero-crossings than HP-concurrent (green).
#      The reduction is approximately 33% (corresponding to ht being 50% larger).
#   2. Extra zero-crossings in HP-concurrent tend to CLUSTER near the zero line:
#      these occur when the HP output hovers close to zero and oscillates around it
#      due to noise, rather than reflecting genuine trend reversals.
#      SSA's smoothness constraint suppresses these spurious crossings.
#   3. Both filters generally agree on the major turning points and cycle direction,
#      confirming that SSA retains the essential HP trend/cycle information.
ts.plot(mplot[anf:enf,],col=colo)
mtext(paste("SSA(",round(ht,2),",",forecast_horizon,")",sep=""),col=colo[1],line=-1)
mtext("HP-concurrent",col=colo[2],line=-2)
abline(h=0)

# Note on further analysis:
#   Timeliness (lead/lag behavior) could also be examined here — analogous to Example 4 —
#   by computing phase shifts or empirical lead/lag correlations between SSA and HP-concurrent.
#   This is left as an exercise for the reader.


######################################################################################################################
######################################################################################################################
# Example 6: Optimal One-Sided Filter for Tracking the Symmetric (Two-Sided) HP Filter
#            with Autocorrelated Data
#
# Overview:
#   This example extends Example 5 by changing the OPTIMIZATION TARGET from the classic
#   HP-concurrent filter (one-sided) to the SYMMETRIC (two-sided) HP filter.
#   The goal is to find the optimal causal (one-sided) filter that tracks the symmetric
#   HP filter output as closely as possible, given that the input data is ARMA(1,1).
#
# Motivation and context:
#   - The classic HP-concurrent filter is optimal ONLY if xt ~ ARIMA(0,2,2), where the two
#     MA parameters are determined by lambda (see Tutorial 2.0). Economic data rarely conforms
#     to this assumption.
#   - In practice, nearly all users of HP-concurrent implicitly assume it is the best real-time
#     estimate of the two-sided HP filter. This assumption is generally INCORRECT.
#   - By targeting the symmetric HP filter directly and accounting for the true data model
#     (ARMA(1,1)), we obtain a genuinely optimal concurrent filter — one that actually does
#     what practitioners believe HP-concurrent is doing.
#
# What this example does NOT address:
#   - The majority of HP use-cases, where practitioners are satisfied with HP-concurrent
#     as a pertinent BCA tool (even if it does not optimally track the two-sided target).
#   - Examples 2, 3, and 5 "engraft" SSA onto HP-concurrent for users who prefer the
#     HP-concurrent as their design benchmark.
#
# What this example DOES address:
#   - Analysts specifically interested in OPTIMALLY TRACKING the symmetric (two-sided) HP
#     filter in real time, using the true (or estimated) data model.
#   - Beyond the optimal MSE filter, we also present SSA extensions that are smoother
#     and/or faster (timeliness-aware), offering a richer set of design choices.
#
# WARNING: This example is more technically involved than previous ones, because the
#   target (symmetric HP), the data model (ARMA), and the forecast horizon (nowcast of
#   a non-causal filter) are all intertwined. Read the comments carefully.

######################################################################################################################
# 6.1 Specify the Symmetric HP Filter as the Optimization Target
######################################################################################################################

# Filter length for the symmetric HP filter.
# We use L_sym=401 (twice the one-sided filter length L=201, minus 1) so that when the
# symmetric filter is split into left and right halves, each half has length (L_sym-1)/2 = 200,
# matching the one-sided filter length L=201 used in the SSA design.
L_sym <- 401

# The filter length must be ODD so that the symmetric HP filter has a well-defined center.
# An even length would result in an ambiguous center position, breaking the symmetry.
if (L_sym / 2 == as.integer(L_sym / 2)) {
  print("Filter length should be an odd number.")
  print("If L_sym is even, the HP filter cannot be correctly centered.")
  L_sym <- L_sym + 1   # Increment by 1 to enforce odd length
}

# Standard HP smoothing parameter for monthly data.
# lambda=14400 is the monthly analogue of lambda=1600 for quarterly data
# (scaled by 4^2=16 for the frequency conversion from quarterly to monthly).
lambda_monthly <- 14400

# Compute the symmetric HP filter and associated MSE-based design objects.
# HP_target_mse_modified_gap() returns:
#   - $target : the symmetric (two-sided) HP filter coefficients of length L_sym
#   - Other components related to the MSE design
HP_obj <- HP_target_mse_modified_gap(L_sym, lambda_monthly)

# Extract the symmetric HP filter coefficients.
# IMPORTANT: As returned by mFilter, this filter is stored as a CAUSAL (one-sided) object,
# with its center at position (L_sym+1)/2. We must account for this when specifying
# the forecast horizon below — this is the "tricky" part of this example.
hp_target <- HP_obj$target

# Visual inspection of the symmetric HP filter coefficients
ts.plot(hp_target,
        main = "Symmetric HP Filter Coefficients (as stored: causal representation)",
        ylab = "Filter Weights",
        xlab = "Lag")

#------------------------------------------------------
# Forecast Horizon: Encoding the Non-Causality of the Target
#------------------------------------------------------
# This is the most technically subtle part of Example 6. Read carefully.
#
# The variable h represents the EFFECTIVE forecast horizon of the concurrent estimate:
#   h = 0  : nowcast  — estimate the current (time t) value of the symmetric filter output
#   h > 0  : forecast — estimate h steps ahead
#   h < 0  : backcast — estimate h steps behind the current time point
h <- 0   # We want a nowcast (real-time estimate at time t)

# Why does the forecast horizon matter here?
#
# The symmetric HP filter hp_target is stored as a CAUSAL object by mFilter:
#   - Its CENTER (peak coefficient) is at position (L_sym+1)/2 in the coefficient vector.
#   - Visually: the brown line in the plot below peaks at lag (L_sym+1)/2.
#
# For a NOWCAST of the two-sided (acausal) symmetric filter:
#   - We need to shift the causal filter to the LEFT so that its center aligns with lag 0.
#   - This shift is exactly (L_sym-1)/2 positions to the left.
#   - Visually: the violet line in the plot below peaks at lag 0 (the nowcast position).
#
# The effective forecast horizon that SSA_func() requires is therefore:
#   forecast_horizon = (L_sym-1)/2 + h
#
# Interpretation:
#   - The (L_sym-1)/2 term accounts for the left-shift needed to center the symmetric filter.
#   - The h term adds any additional lead (h>0) or lag (h<0) relative to nowcast.
#   - For h=0 (nowcast): forecast_horizon = (L_sym-1)/2 = 200 (for L_sym=401).
forecast_horizon <- (L_sym - 1) / 2 + h

# Construct both the causal and acausal representations for visualization:
#
# causal_sym: the symmetric HP filter padded with zeros on the left to length 2*L_sym-1.
#   Represents the filter as stored by mFilter — center at position (L_sym+1)/2.
causal_sym <- c(rep(0, (L_sym - 1) / 2), hp_target)

# acausal_sym: the symmetric HP filter shifted LEFT by forecast_horizon positions.
#   Represents the target as SSA sees it — center at lag 0 (the nowcast).
#   Constructed by:
#     (a) Dropping the first max(0, (L_sym+1)/2 - forecast_horizon - 1) zeros on the left
#     (b) Appending forecast_horizon zeros on the right (future lags are zero for nowcast)
#     (c) Truncating to the same length as causal_sym for plotting
acausal_sym <- c(
  rep(0, max(0, (L_sym + 1) / 2 - forecast_horizon - 1)),
  hp_target[max(1, 1 + (forecast_horizon - (L_sym - 1) / 2)):L_sym],
  rep(0, forecast_horizon)
)
acausal_sym <- acausal_sym[1:length(causal_sym)]

# Plot causal vs. acausal representations to visualize the left-shift:
#   - Brown line: causal symmetric HP as stored by mFilter (center at (L_sym+1)/2)
#   - Violet line: acausal target as seen by SSA (center shifted to lag 0)
#   - Brown vertical line: center of causal filter
#   - Violet vertical line: center of acausal filter (lag 0 = nowcast position)
plot(causal_sym,
     col  = "brown",
     axes = F,
     type = "l",
     xlab = "lead                                                                  lag",
     ylab = "",
     main = "Acausal vs. Causal HP: The Optimization Target is Acausal (Two-Sided)")
lines(acausal_sym, col = "violet")
abline(v = (L_sym + 1) / 2, col = "violet")   # Center of acausal filter (lag 0)
abline(v = L_sym, col = "brown")               # Center of causal filter

# Annotation: explain the shift
mtext(paste("Target acausal filter: center is at lag 0.",
            " For SSA we shift the causal HP (brown) LEFT by forecast_horizon=",
            forecast_horizon, sep = ""),
      line = -1, col = "violet")
mtext(paste("                                                   ",
            "Causal symmetric HP as calculated by mFilter: center is at lag ",
            (L_sym + 1) / 2, sep = ""),
      line = -3, col = "brown")
axis(1, at = 1:length(acausal_sym), labels = -forecast_horizon + 1:length(acausal_sym))
axis(2)
box()

#------------------------------------------------------
# MSE Benchmark: Optimal One-Sided Filter for Tracking the Symmetric HP
#------------------------------------------------------
# Before applying SSA, we compute the optimal MSE filter as a natural benchmark.
# This filter minimizes the mean squared error between its output and the symmetric HP output,
# given the true ARMA(1,1) data model.
#
# IMPORTANT: hp_mse (computed in Example 1) is NOT optimal here because it assumes
# white noise input (xt = epsilon_t). Since xt is ARMA(1,1), we must account for the
# autocorrelation structure via the Wold decomposition xi_data.
#
# Construction of the MSE filter (two steps):
#   Step a: Convolve xi_data (Wold decomposition) with hp_target (symmetric HP filter).
#           This gives the combined filter applied to the white noise innovations epsilon_t.
#           The convolution maps the problem into the innovation domain where epsilon_t is i.i.d.
#   Step b: Truncate the convolved filter at lag 0 (corresponding to position forecast_horizon+1).
#           Coefficients beyond lag 0 would require future values of epsilon_t, whose best
#           forecast is zero (since epsilon_t is white noise). Truncation implements this.

# Step a: Convolve Wold decomposition xi_data with symmetric HP target hp_target
hp_conv_mse_d <- conv_two_filt_func(xi_data, hp_target)$conv

# Step b: Truncate at lag 0 (position forecast_horizon+1 in the convolved filter).
# The resulting filter hp_conv_mse is the MSE-optimal filter applied to epsilon_t.
# Length: L_sym - forecast_horizon = L_sym - (L_sym-1)/2 - h = (L_sym+1)/2 - h coefficients.
hp_conv_mse <- hp_conv_mse_d[(forecast_horizon + 1):L_sym]

# Compute the holding time of the MSE-optimal filter.
# This serves as the NATURAL BASELINE for SSA design:
#   - SSA should improve smoothness relative to the MSE benchmark.
#   - Using ht_hp_conv_mse as baseline enables stringent, meaningful comparisons.
ht_hp_conv_mse_obj <- compute_holding_time_func(hp_conv_mse)
ht_hp_conv_mse <- ht_hp_conv_mse_obj$ht

# Print and compare holding times:
#   ht_hp_conv_mse : holding time of the MSE-optimal filter for ARMA(1,1) data
#   ht_mse         : holding time of hp_mse (MSE filter assuming WHITE NOISE from Example 1)
#
# Expected: ht_hp_conv_mse > ht_mse
# Reason: The positive autocorrelation in the ARMA(1,1) data (a1=0.3, b1=0.2) adds extra
# smoothness to the filter output, increasing the holding time relative to the white noise case.
ht_hp_conv_mse   # Holding time of MSE-optimal filter (ARMA data) — should be > ht_mse
ht_mse           # Holding time of hp_mse (white noise assumption) — reference from Example 1

######################################################################################################################
# 6.2 SSA Design: Optimal Tracking of the Symmetric HP Filter with Autocorrelated Data
######################################################################################################################

# Target holding time for SSA:
#   We augment the MSE-optimal holding time by 50%, requesting a smoother filter than MSE.
#   This corresponds to approximately 33% fewer zero-crossings than the MSE benchmark.
#
# NOTE: The 50% increase is a design choice. Alternative values can be used based on:
#   - A priori knowledge about desired cycle smoothness
#   - Specific priorities (e.g., emphasizing timeliness over smoothness, or vice versa)
#   Using ht_hp_conv_mse as baseline (rather than an ad hoc value) enables principled comparisons.
ht <- 1.5 * ht_hp_conv_mse

# Convert the holding time to the lag-one autocorrelation rho1 required by SSA_func().
# Higher ht => slower, smoother filter => larger rho1.
rho1 <- compute_rho_from_ht(ht)

# Specify the symmetric HP filter as the optimization target.
# In contrast to Examples 2, 3, and 5 (where the target was the one-sided HP-concurrent),
# here we directly target the two-sided symmetric filter. SSA_func() handles the
# non-causality via the forecast_horizon parameter specified above.
gammak_generic <- hp_target

# The forecast horizon was computed above: forecast_horizon = (L_sym-1)/2 + h.
# It encodes the left-shift needed to align the causal hp_target with the acausal target.
# (No change needed here — just making explicit that the earlier value is used.)
forecast_horizon <- forecast_horizon

# One-sided filter length: same as Examples 1–5 for comparability.
L <- 201

# Supply the Wold decomposition (MA-inversion) of the ARMA(1,1) process.
# This is ESSENTIAL here (unlike Example 4 which assumed white noise):
#   - xi_data encodes the autocorrelation structure of xt.
#   - Without xi_data, SSA_func() would assume xt = epsilon_t (white noise), which is incorrect
#     and would produce a suboptimal filter that does not account for ARMA autocorrelation.
xi <- xi_data

# Compute the SSA filter targeting the symmetric HP filter with ARMA(1,1) data.
# Parameters:
#   L                : one-sided (causal) filter length
#   forecast_horizon : encodes the non-causal shift of the symmetric target (= 200 for h=0)
#   gammak_generic   : symmetric HP filter coefficients (the two-sided target)
#   rho1             : lag-one ACF corresponding to the target holding time ht
#   xi               : Wold decomposition of ARMA(1,1) — accounts for autocorrelation
SSA_obj_HP <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

#------------------------------------------------------
# Verify MSE Filter: Compare SSA_func Output with Own Calculation
#------------------------------------------------------
# SSA_func() internally computes the MSE-optimal filter as part of its optimization.
# We verify consistency by comparing it with our own manually computed MSE filter (hp_conv_mse).
#
# mse_eps: the MSE-optimal filter as returned by SSA_func(), applied to epsilon_t.
#   This should be identical to hp_conv_mse[1:L] computed above.
#   If the two lines in the plot below overlap perfectly, our manual calculation is confirmed.
mse_eps <- SSA_obj_HP$mse_eps
ts.plot(cbind(hp_conv_mse[1:L], mse_eps),
        main = "MSE Estimate: SSA_func vs. Manual Calculation (lines should overlap perfectly)",
        col  = c("black", "red"),
        ylab = "Filter Weights",
        xlab = "Lag")

#------------------------------------------------------
# Extract SSA and MSE Filters for Applied Use
#------------------------------------------------------

# ssa_x: the SSA filter applied to the OBSERVED series xt.
#   This is the operationally relevant filter for real-data applications.
#   Incorporates both the SSA smoothness optimization and the ARMA autocorrelation correction.
SSA_filt_HP <- SSA_obj_HP$ssa_x

# mse_x: the MSE-optimal filter applied to the observed series xt.
#   Automatically computed by SSA_func() as the optimization benchmark.
#   This is the best achievable filter in terms of MSE for the given data model and target.
HP_MSE_x <- SSA_obj_HP$mse_x

# -----------------------------------------------------------------------
# Plot SSA filter designs from examples 4, 5, and 6 side-by-side for comparison.
# This helps visualize how different assumptions (noise model, holding-time)
# affect the resulting filter shapes.
# -----------------------------------------------------------------------

colo<-c("black","brown","blue")
par(mfrow=c(1,1))
mplot<-cbind(SSA_example4,SSA_example5,SSA_filt_HP)
colnames(mplot)<-c("Example 4","Example 5","Example 6")

# Why do the filters from examples 4 (black) and 6 (blue) look so similar?
#   - Example 4: used the same symmetric two-sided HP target, but assumed white noise
#     input (xi = NULL) and a shorter holding-time constraint (ht = 12).
#   - Example 6: assumes an ARMA-process as input and imposes a larger holding-time
#     of approximately 17.
#   - Despite these different assumptions, the filter from example 4, when applied
#     to the ARMA-process, happens to produce a holding-time very close to 17.
#   - This coincidence causes both designs to be nearly identical (minor differences
#     appear only near lag 0).
#   - Larger differences would emerge by changing the ARMA specification and/or the
#     holding-time constraint.
#
# Why does example 5 (brown) have a noticeably different scale?
#   - Example 5 targets the one-sided (causal) HP filter hp_trend, which is a simpler
#     estimation problem than nowcasting the acausal two-sided HP.
#   - Examples 4 and 6 nowcast the two-sided HP, which is inherently more uncertain.
#   - The smaller filter weights (zero-shrinkage effect) in examples 4 and 6 reflect
#     this increased estimation uncertainty.


plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
lines(mplot[,3],col=colo[3],lwd=2,lty=1)
mtext(colnames(mplot)[3],col=colo[3],line=-3)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)

# -----------------------------------------------------------------------
# Now that the optimal filters have been identified and computed, we:
#   1. Filter a long simulated series to obtain accurate empirical estimates.
#   2. Compare empirical and expected performance metrics (holding-times, correlations).
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# 6.3 Filter the simulated series
# -----------------------------------------------------------------------

# Simulate a very long series from the specified ARMA process to ensure that
# empirical estimates (holding-times, correlations) are statistically accurate.
len <- 1000000
set.seed(14)
x <- arima.sim(n = len, list(ar = a1, ma = b1))

# Apply the SSA-HP filter (one-sided, causal) to the simulated series.
# This is the SSA-optimal filter for nowcasting the two-sided HP given the ARMA input.
yhat <- filter(x, SSA_filt_HP, side = 1)

# Verify that the empirical holding-time matches the target holding-time ht.
# Good agreement here confirms that the SSA constraint is satisfied in practice.
ht
compute_empirical_ht_func(yhat)

# Apply the MSE-optimal filter for the same ARMA process.
# This provides a correlation-maximising benchmark for comparison with SSA.
HP_mse <- filter(x, HP_MSE_x, side = 1)

# Verify that the empirical holding-time of the MSE filter matches its theoretical value.
# Note: the SSA filter produces ~50% more holding-time than MSE, as intended by design.
ht_hp_conv_mse
compute_empirical_ht_func(HP_mse)

# Compute the output of the two-sided (symmetric) HP filter — this is the estimation target.
# side = 2 is required because the filter is symmetric and uses both past and future observations.
HP_symmetric <- filter(x, hp_target, side = 2)

# Compute the output of the classic one-sided HP concurrent filter for reference.
# side = 1 since this is a causal (one-sided) filter.
HP_concurrent <- filter(x, hp_trend, side = 1)

# The classic HP concurrent filter produces more zero-crossings (less smooth output) than SSA.
compute_empirical_ht_func(HP_concurrent)
# As shown below, the classic HP concurrent is also inferior to SSA in terms of
# correlation with the symmetric target — misspecification hurts on both dimensions because of severe misspecification.

# -----------------------------------------------------------------------
# Plot all filter outputs over a short representative sample window
# -----------------------------------------------------------------------
colo <- c("blue", "green", "black", "brown")

# Combine all filter outputs into a matrix for joint plotting (remove NAs at boundaries).
mplot <- na.exclude(cbind(yhat, HP_mse, HP_symmetric, HP_concurrent))
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", forecast_horizon, ")", sep = ""),
  "HP MSE",
  "HP-symmetric (effective target)",
  "Classic HP concurrent"
)

# Define the sample window to display (short segment for clarity).
# Note: the second anf/enf assignment overrides the first — only rows 1000–1500 are plotted.
par(mfrow = c(1, 1))
anf <- 500;  enf <- 1000   # Initial window (overridden below)
anf <- 1000; enf <- 1500   # Active window: rows 1000 to 1500

# Visual inspection confirms that SSA produces roughly 30% fewer zero-crossings
# compared to the MSE filter or the classic HP concurrent filter.
ts.plot(mplot[anf:enf,],col=colo)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
if (ncol(mplot)>1)
  for (i in 2:ncol(mplot))
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
abline(h=0)

# -----------------------------------------------------------------------
# Discussion: Filter scaling and the role of misspecification
# -----------------------------------------------------------------------
# The classic HP concurrent filter appears off-scale in the plot above.
# This is due to severe misspecification: hp_trend implicitly assumes the input
# follows an ARIMA(0,2,2) process (strongly trending), which is inappropriate here.
#
# A useful diagnostic for filter scaling is the sum of filter coefficients,
# which equals the filter's amplitude at frequency zero.
# It therefore reveals how the filter scales the dominant low-frequency (trend) component.

# Two-sided HP target: coefficients sum to 1 (unit gain at frequency zero — passes trends unchanged).
sum(hp_target)

# Classic concurrent HP: also sums to 1, consistent with its ARIMA(0,2,2) assumption.
# Under that model, the one-sided filter must track the trend without rescaling.
sum(hp_trend)

# SSA filter: sum is markedly less than 1 (zero-shrinkage toward 0.5).
# This reflects the fact that the ARMA input is less strongly trending than ARIMA(0,2,2),
# so shrinking the output reduces estimation error.
sum(SSA_filt_HP)

# MSE-optimal filter: also sums to markedly less than 1, for the same reason.
# When the ARIMA(0,2,2) model is misspecified (e.g. input has strong noise), shrinkage improves MSE.
sum(hp_mse)

# Summary of zero-gain implications:
#   - hp_trend assumes strong trending: unit gain preserves the dominant low-frequency component.
#   - SSA and hp_mse recognise that the data is not strongly trending:
#       * They shrink the output (coefficients sum to ~0.5 instead of 1).
#       * This shrinkage is optimal under the true ARMA model.
#   - The apparent scaling mismatch of hp_trend in the plot is a direct consequence
#     of this misspecification.
#   - Note: simply rescaling hp_trend by 0.5 would partially fix the amplitude, but
#     the filter would still be suboptimal — see correlation analysis below.

# -----------------------------------------------------------------------
# 6.4 Empirical vs. expected performance: correlations and holding-times
# -----------------------------------------------------------------------

# Compute the full empirical correlation matrix across all filter outputs.
# This summarises pairwise agreement between all designs and the true target.
cor_mat <- cor(mplot)
cor_mat

# Focus on correlations of the one-sided filters with the symmetric target (row 3 = HP-symmetric).
# These are the primary performance metrics for nowcasting quality.
cor_mat[3, ]

# MSE marginally outperforms SSA in correlation with the target (MSE is the correlation-maximising benchmark).
# However, SSA achieves ~50% larger holding-times — a better smoothness-accuracy trade-off.
cor_mat[3, 2]   # MSE vs. target
cor_mat[3, 1]   # SSA vs. target

# MSE also outperforms the classic HP concurrent in correlation with the target.
cor_mat[3, 2]   # MSE vs. target
cor_mat[3, 4]   # Classic HP concurrent vs. target

# Key finding: SSA outperforms the classic HP concurrent on BOTH dimensions —
#   * Higher correlation with the symmetric target (better accuracy), AND
#   * More zero-crossings (better smoothness / fewer false signals).
# This confirms that the ARIMA(0,2,2) assumption underlying hp_trend is severely
# misspecified for typical economic time series.
cor_mat[3, 1]   # SSA vs. target
cor_mat[3, 4]   # Classic HP concurrent vs. target

# -----------------------------------------------------------------------
# Verify that empirical SSA performance matches the theoretical criterion values
# reported by the SSA optimisation routine.
# These checks validate the internal consistency of the SSA framework.
# -----------------------------------------------------------------------
cor_mat[1, ]   # Row 1: correlations of SSA output with all other series

# A. Empirical correlation of SSA with the symmetric target should match crit_rhoy_target
#    (the SSA objective value for correlation with target, from Proposition 4 in the JBCY paper).
cor_mat[1, 3]                        # Empirical correlation: SSA vs. target
SSA_obj_HP$crit_rhoy_target          # Theoretical value from SSA optimisation

# B. Empirical correlation of SSA with the MSE filter should match crit_rhoyz
#    (the SSA criterion value for correlation with MSE benchmark, per Proposition 4).
cor_mat[1, 2]                        # Empirical correlation: SSA vs. MSE
SSA_obj_HP$crit_rhoyz                # Theoretical value from SSA optimisation



# ========================================================================
# Example 7: Analysis of the Classic HP-Gap Filter (Counter-Example)
# ========================================================================
# This example is a counter-example and is not related to SSA.
# We analyse the classic HP-gap filter (see also Tutorial 2.0) and
# demonstrate why its use in Business Cycle Analysis (BCA) is inadvisable.
# This conclusion aligns with Hamilton's recommendation: "never use the HP
# (gap) filter", as discussed in Tutorial 3.
# ========================================================================

# ------------------------------------------------------------------------
# 7.1 Compute the HP-gap filter
#     We use the same filter length as in Example 6 above.
# ------------------------------------------------------------------------

L_sym <- 401

# The filter length must be odd to ensure the symmetric HP filter is correctly
# centred. If L_sym is even, the filter cannot be centred symmetrically around
# lag 0, which would introduce a spurious phase shift.
if (L_sym / 2 == as.integer(L_sym / 2))
{
  print("Filter length should be an odd number")
  print("If L_sym is even then HP cannot be correctly centered")
  L_sym <- L_sym + 1
}

# Set the HP smoothing parameter for monthly data.
# lambda = 14400 is the standard monthly calibration (Hodrick-Prescott, 1997).
lambda_monthly <- 14400

# Compute the HP target, HP-gap, and HP-trend (classic concurrent) filters.
HP_obj <- HP_target_mse_modified_gap(L_sym, lambda_monthly)

# Bi-infinite (two-sided symmetric) HP-trend: serves as the estimation target.
# This represents the ideal smoother we wish to approximate in real time.
hp_target <- HP_obj$target
ts.plot(hp_target)

# HP-gap filter: the complement of the HP-trend, extracting the cyclical component.
# Applied to levels, this is a high-pass filter.
hp_gap <- HP_obj$hp_gap
ts.plot(hp_gap)

# Classic one-sided (concurrent) HP-trend filter for real-time estimation.
hp_trend <- HP_obj$hp_trend
ts.plot(hp_trend)

# Verify the fundamental identity: HP-gap = delta_0 - HP-trend,
# where delta_0 is a unit impulse (i.e., the identity operator in levels).
# The maximum absolute deviation should be zero (or machine precision).
max(abs((c(1, rep(0, L_sym - 1)) - hp_trend) - hp_gap))

# ------------------------------------------------------------------------
# 7.2 Transformation: expressing the HP-gap in terms of first differences
# ------------------------------------------------------------------------
# The HP-gap filter is defined for data in levels, which makes direct
# frequency-domain analysis difficult (levels are typically non-stationary).
# To simplify the analysis, we derive an equivalent filter — gap_diff — that
# produces identical output to HP-gap when applied to first differences of the
# input series (see Tutorial 2.0 and Proposition 4 in the JBCY paper).
# This transformation allows us to interpret the filter's spectral properties
# in terms of a (near-stationary) differenced series.

gap_diff <- conv_with_unitroot_func(hp_gap)$conv

par(mfrow = c(2, 1))
# Note: the coefficients of gap_diff decay to zero beyond the length of the
# original hp_gap filter, so the effective filter length is well-defined.
ts.plot(gap_diff, main = "HP-gap as applied to first differences")
ts.plot(hp_gap,   main = "HP-gap as applied to levels")

# ------------------------------------------------------------------------
# Verify that hp_gap applied to levels and gap_diff applied to first 
# differences produce numerically identical outputs.
# ------------------------------------------------------------------------
set.seed(252)
len <- L + 2000

# Simulate a random walk in levels (unit-root process) to feed into hp_gap.
y <- cumsum(rnorm(len))

# Compute first differences to feed into gap_diff.
# A leading zero is prepended to align the length of x with y.
x      <- c(0, diff(y))
len_diff <- length(x)

# Apply the transformed filter to differences.
yhat_diff <- filter(x, gap_diff, side = 1)
# Apply the original HP-gap filter to levels.
yhat_gap  <- filter(y, hp_gap,   side = 1)

# Overlay both outputs: they should be indistinguishable (exact numerical match).
par(mfrow = c(1, 1))
ts.plot(yhat_diff, col = "blue",
        main = "Transformed HP-gap applied to differences replicates original HP-gap applied to levels")
lines(yhat_gap, col = "red")

# ------------------------------------------------------------------------
# 7.3 Spectral analysis of the HP-gap filter via gap_diff
# ------------------------------------------------------------------------
# First differences of typical macroeconomic data are approximately white noise
# (see Granger, 1966, for the characteristic spectral shape of economic series).
# Under this approximation, the squared amplitude of gap_diff is proportional
# to the spectral density of the HP-gap filter output (by the convolution theorem).
# Analysing gap_diff therefore provides direct insight into the spectral content
# of the extracted HP-gap cycle — this is the main motivation for the transformation.

# Set the frequency grid resolution: K equidistant ordinates over [0, pi].
K <- 600

# Compute the amplitude function of gap_diff over the frequency grid.
amp_gap_diff <- amp_shift_func(K, as.vector(gap_diff), F)$amp

# Plot the amplitude function of gap_diff.
# This reveals which frequencies are passed, attenuated, or amplified by the filter.
plot(amp_gap_diff,
     type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = paste("Amplitude HP-gap as applied to differences", sep = ""))
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Plot the squared amplitude function of gap_diff.
# Under the white-noise approximation for first differences, this equals the
# true spectral density of the HP-gap cycle when the input is a random walk —
# a good proxy for spectral behaviour with typical non-stationary economic data.
par(mfrow = c(1, 1))
plot(amp_gap_diff^2,
     type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = paste("Squared amplitude HP-gap as applied to differences", sep = ""))
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Key findings from the spectral analysis:
#   1. Band-pass behaviour after differencing:
#      Applied to levels the HP-gap is a high-pass filter (Tutorial 2.0), but
#      applied to first differences gap_diff behaves as a band-pass filter
#      (amplitude vanishes at frequency zero). This occurs because the one-sided
#      HP-gap implicitly assumes a second-order unit root: after one round of
#      differencing, gap_diff must still remove the remaining unit root.
#   2. Spurious cycle risk:
#      The filter concentrates spectral energy around a peak frequency, which
#      corresponds to a specific periodicity. This peak drives a spurious cycle
#      in the filter output, independent of any true cyclical feature in the data.

# ------------------------------------------------------------------------
# Identify the dominant (peak) frequency and its implied periodicity.
# ------------------------------------------------------------------------

# Frequency (in radians) at which the amplitude of gap_diff is maximised.
omega_gap <- pi * (which(amp_gap_diff == max(amp_gap_diff)) - 1) / K

# Convert the peak frequency to a period expressed in months.
2 * pi / omega_gap
# Result: approximately 72 months (6 years).
# This is shorter than many historical expansions (e.g., the Great Moderation),
# meaning the HP-gap tends to signal mid-expansion cycle peaks that are
# artefacts of the filter rather than genuine turning points.
# See Tutorials 2.0 and 5 for detailed empirical evidence.

# ------------------------------------------------------------------------
# Corroborate the spectral findings using the periodogram and ACF.
# ------------------------------------------------------------------------

# Periodogram of the HP-gap output: confirms damping of both low and high
# frequencies, with a spectral peak near the spurious cycle frequency.
per_obj <- per(na.exclude(yhat_gap), T)

# ACF of the HP-gap output: the autocorrelation structure reflects the
# imposed periodicity and confirms the presence of the spurious cycle.
acf(na.exclude(yhat_gap))

# ------------------------------------------------------------------------
# Visualise the extracted HP-gap cycle.
# ------------------------------------------------------------------------
par(mfrow = c(1, 1))
ts.plot(yhat_gap)
abline(h = 0)
# Observations:
#   - The cycle contains numerous noisy zero-crossings, which would make
#     real-time business cycle assessment unreliable and prone to false signals.
#   - The Baxter-King band-pass filter shares similar drawbacks; see Tutorial 4.

# ------------------------------------------------------------------------
# Conclusions
# ------------------------------------------------------------------------
# The above findings confirm and reinforce Hamilton's recommendation:
#   "Never use the HP-gap filter for Business Cycle Analysis."
#
# However, the HP-trend (lowpass) filter applied to first differences offers
# a practical and well-behaved alternative:
#   - It reliably tracks expansions and recessions (neither over- nor under-smoothed).
#   - It avoids the spurious cycle that afflicts the HP-gap.
#   - It does not generate false alarms mid-expansion or mid-recession.
#   - It lags slightly behind HP-gap at turning points — a minor and acceptable trade-off.
#   - Its real-time properties can be further refined using SSA to match
#     specific practitioner priorities (see Examples 1–6 above).
# See Tutorial 2.0 for a full comparative analysis of HP-trend vs. HP-gap applied
# to differences.


# ========================================================================
# Example 8: Visualisation of the Prediction Trilemma for the HP Filter
# ========================================================================
# Background:
#   MSE, timeliness, and smoothness define a forecast trilemma (Tutorial 0.1).
#   No filter can simultaneously optimise all three — improving one dimension
#   necessarily degrades at least one of the others.
#
# The three trilemma dimensions are operationalised as follows:
#   a. MSE (accuracy): measured via the SSA criteria crit_rhoyz and
#      crit_rhoy_target, which quantify the correlation of the SSA predictor
#      with the causal MSE benchmark and the acausal target, respectively.
#      Higher correlation <=> lower MSE.
#   b. Timeliness: measured by the forecast horizon (number of steps ahead).
#      A larger forecast horizon means the filter is more timely.
#   c. Smoothness: measured by the holding-time constraint (ht).
#      A larger ht means fewer zero-crossings, i.e. a smoother output.
#
# Approach:
#   We use the classic concurrent HP-trend estimate (hp_trend) as the SSA
#   target and compute crit_rhoy_target and crit_rhoyz across a grid of
#   holding-time values (ht) and forecast horizons.
#   The trilemma is then visualised as a heat map: criterion values as a
#   function of ht (rows) and forecast horizon (columns).
# ========================================================================

# ------------------------------------------------------------------------
# 8.1 Set up and run (or load) the SSA loop
# ------------------------------------------------------------------------

# NOTE: The grid search below is computationally intensive (~5 minutes on a
# single 3 GHz core). Set compute_length_loop = FALSE to skip the computation
# and load pre-saved results instead.
compute_length_loop <- FALSE

# Specify the SSA target: the two-sided symmetric HP filter.
gammak_generic <- hp_target

if (compute_length_loop)
{
  # Display the holding-time of the MSE-optimal filter as a reference point.
  # SSA results will be compared against this benchmark.
  ht_mse
  
  # Define the grid of holding-time values to evaluate.
  # The range spans from one quarter of the MSE holding-time up to twice its value,
  # covering both smoother and less-smooth designs relative to the MSE benchmark.
  ht_vec <- seq(max(2, ht_mse / 4), 2 * ht_mse, by = 0.1)
  
  # Define the grid of forecast horizons to evaluate.
  # We shift by (L_sym - 1)/2 to convert from the one-sided causal representation
  # to the acausal (centred) two-sided target. The shift is removed from the
  # column labels at the end of the loop.
  delta_vec <- 0:24 + (L_sym - 1) / 2
  
  # Initialise a progress bar to track loop execution.
  pb <- txtProgressBar(min = 0, max = length(ht_vec), initial = 0, style = 3)
  
  # Allocate result matrices:
  #   MSE_mat    : correlation of SSA with the causal MSE benchmark (crit_rhoyz)
  #   target_mat : correlation of SSA with the acausal two-sided target (crit_rhoy_target)
  # Rows index ht values; columns index forecast horizons.
  MSE_mat    <- matrix(ncol = length(delta_vec), nrow = length(ht_vec))
  target_mat <- matrix(ncol = length(delta_vec), nrow = length(ht_vec))
  
  # Main grid search: iterate over all (ht, forecast horizon) combinations.
  # For each combination, compute the SSA filter and record both criterion values.
  for (i in 1:length(ht_vec))
  {
    setTxtProgressBar(pb, i)
    for (j in 1:length(delta_vec))
    {
      # Convert the holding-time to the corresponding AR(1) parameter rho1.
      rho1 <- compute_rho_from_ht(ht_vec[i])
      forecast_horizon <- delta_vec[j]
      
      # Compute the SSA filter assuming white noise input (xi = NULL).
      # This simplifies computation and isolates the effect of ht and horizon.
      SSA_obj_HP <- SSA_func(L, forecast_horizon, gammak_generic, rho1)
      
      # Store the correlation of SSA with the causal MSE predictor.
      # This is the preferred benchmarking measure: it shows how much accuracy
      # is sacrificed relative to the best possible causal filter.
      MSE_mat[i, j] <- SSA_obj_HP$crit_rhoyz
      
      # Store the correlation of SSA with the acausal (two-sided) target.
      # This is always smaller than crit_rhoyz, since the acausal target is
      # inherently harder to predict from one-sided (past-only) data.
      target_mat[i, j] <- SSA_obj_HP$crit_rhoy_target
    }
  }
  close(pb)
  
  # Attach row and column labels for interpretability.
  # Row names: rounded holding-time values.
  # Column names: forecast horizons (with the artificial (L_sym-1)/2 shift removed).
  rownames(MSE_mat) <- rownames(target_mat) <- round(ht_vec, 2)
  colnames(MSE_mat) <- colnames(target_mat) <- delta_vec - (L_sym - 1) / 2
  
  # Save results to disk for future use (avoids re-running the loop).
  save(MSE_mat,    file = paste(getwd(), "/Data/Trilemma_mse_heat_map",    sep = ""))
  save(target_mat, file = paste(getwd(), "/Data/Trilemma_target_heat_map", sep = ""))
  
} else {
  # Load pre-computed results from disk.
  load(file = paste(getwd(), "/Data/Trilemma_mse_heat_map",    sep = ""))
  load(file = paste(getwd(), "/Data/Trilemma_target_heat_map", sep = ""))
}

# ------------------------------------------------------------------------
# Inspect the result matrices before plotting.
# ------------------------------------------------------------------------

# MSE_mat: rows = ht (smoothness), columns = forecast horizon (timeliness).
# Values are correlations between the SSA filter and the causal MSE benchmark.
# Higher values indicate smaller MSE relative to the best causal predictor.
head(MSE_mat)
tail(MSE_mat)

# target_mat: same structure as MSE_mat, but values are correlations between
# the SSA filter and the acausal two-sided target.
# These are uniformly smaller than MSE_mat values (acausal target is harder to match).
head(target_mat)
tail(target_mat)

# ------------------------------------------------------------------------
# 8.2 Heat map visualisations of the trilemma
# ------------------------------------------------------------------------

# 8.2.0 Define a colour scheme for the heat maps.
# We use a subset of the rainbow palette (excluding deep red/violet) and reverse
# it so that darker colours represent higher correlation (better performance).
lcol  <- 100
coloh <- rainbow(lcol)[1:(10 * lcol / 11)]
colo  <- coloh[length(coloh):1]

# ------------------------------------------------------------------------
# 8.2.1 Heat map: correlation of SSA with the acausal (effective) target
# ------------------------------------------------------------------------
# Axes: x = forecast horizon (timeliness), y = holding time (smoothness).
# Colour: darker = higher correlation with the two-sided HP target (lower MSE).
# The matrix is reversed row-wise so that larger ht appears at the top.
heatmap.2(target_mat[nrow(target_mat):1, ],
          dendrogram  = "none",
          scale       = "none",
          col         = colo,
          trace       = "none",
          density.info = "none",
          Rowv        = FALSE,
          Colv        = FALSE,
          ylab        = "Smoothness: holding time",
          xlab        = "Timeliness: forecast horizon",
          main        = "Trilemma: Correlation with Effective Acausal Target")

# Interpretation of the heat map:
#   - Fixed ht, increasing forecast horizon:
#       Correlations decrease monotonically. Gaining timeliness (more steps ahead)
#       for a fixed smoothness level always increases MSE.
#   - Fixed forecast horizon, varying ht:
#       Correlations peak at the MSE-optimal ht for that horizon. Above and below
#       this peak the SSA smoothness constraint is binding: correlation is sacrificed
#       to enforce the ht requirement (smoother or less-smooth than MSE-optimal).
#   - Simultaneous increase in ht and forecast horizon (diagonal movement):
#       MSE degrades disproportionately — both trilemma penalties compound.
#   - Improving both accuracy (higher correlation) and timeliness (larger horizon)
#       disproportionately penalises smoothness (requires larger ht trade-off).

# ------------------------------------------------------------------------
# 8.2.2 Log-transformed heat map for enhanced visual resolution
# ------------------------------------------------------------------------
# Applying a log transform compresses the range of correlations and reveals
# finer structure that is not visible in the raw correlation heat map.
heatmap.2(log(target_mat[nrow(target_mat):1, ]),
          dendrogram   = "none",
          scale        = "none",
          col          = colo,
          trace        = "none",
          density.info = "none",
          Rowv         = FALSE,
          Colv         = FALSE,
          ylab         = "Smoothness: holding time",
          xlab         = "Timeliness: forecast horizon",
          main         = "Trilemma: Log-Correlation with Effective Acausal Target")
# The log scale accentuates differences among low-correlation regions and makes
# the trilemma trade-off structure more legible across the full grid.

# ------------------------------------------------------------------------
# 8.2.3 Line plot: correlation vs. ht for selected forecast horizons (slices)
# ------------------------------------------------------------------------
# Plotting selected columns of target_mat as line plots allows a more precise
# quantitative reading of the trilemma trade-offs than the heat map alone.

# Select the forecast horizons to display (column indices in target_mat).
select_vec <- 22:24
mplot <- target_mat[, select_vec]
coli  <- rainbow(length(select_vec))

par(mfrow=c(1,1))
plot(mplot[,1],col=colo,ylim=c(min(mplot),max(mplot)),axes=F,type="l",xlab="Holding time",ylab="Correlation",main="Trilemma for HP: Correlations as a Function of ht and a Selection of Forecast Horizons")
mtext(paste("Forecast horizon ",colnames(MSE_mat)[select_vec[1]],sep=""),col=coli[1],line=-1)
# Add lines for the remaining selected forecast horizons.
if (length(select_vec)>1)
  for (i in 2:length(select_vec))
  {
    lines(mplot[,i],col=coli[i])
    mtext(paste("Forecast horizon ",colnames(MSE_mat)[select_vec[i]],sep=""),col=coli[i],line=-i)
  }
# Draw a horizontal reference line at a chosen fixed correlation level.
# This illustrates how a given accuracy requirement translates into different
# (ht, forecast horizon) combinations — the core of the trilemma trade-off.
cor_val=0.105
abline(h=cor_val)
# Draw vertical lines at the ht values where each correlation curve crosses
# the reference level, i.e., the black horizontal line (transition from above to below cor_val).
for (i in 1:ncol(mplot))
  abline(v=which(mplot[1:(nrow(mplot)-1),i]>cor_val&mplot[2:nrow(mplot),i]<cor_val),col=coli[i])
axis(1,at=1:nrow(MSE_mat),labels=rownames(MSE_mat))
axis(2)
box()


# Interpretation of the slice plot:
#   - Each curve's peak corresponds to the MSE-optimal ht for that forecast horizon.
#     At the peak, the SSA smoothness constraint is exactly met by the MSE predictor.
#   - To the right of the peak: SSA is smoother than the MSE predictor (ht constraint
#     is binding); correlation decreases as smoothness is enforced.
#   - To the left of the peak: SSA is less smooth than the MSE predictor; correlation
#     also decreases (the ht constraint forces less smoothing than is optimal).
#   - Reading across curves at the reference level (cor_val = 0.105):
#       A fixed accuracy target of 0.105 can be achieved at approximately:
#         ht ~ 20  for forecast horizon 21,
#         ht ~ 16  for forecast horizon 22,
#         ht ~ 9   for forecast horizon 23.
#       This illustrates the smoothness–timeliness trade-off: gaining timeliness
#       (larger forecast horizon) requires accepting a shorter holding-time (less smooth).
#
#   Note: In Examples 1–6, we traded timeliness (larger forecast horizon) against
#   accuracy (lower correlation) for a fixed ht — one face of the trilemma.
#   See also Tutorial 5 for applied examples of this trade-off.

# ------------------------------------------------------------------------
# 8.2.4 Heat map: correlation with the causal MSE predictor
# ------------------------------------------------------------------------
# Unlike the acausal target heat map (8.2.1), this plot benchmarks SSA against
# the best possible causal (one-sided) predictor rather than the two-sided target.
# It emphasises the smoothing dimension of the trilemma (Section 2.4, JBCY paper)
# rather than the prediction dimension.
heatmap.2(MSE_mat[nrow(MSE_mat):1, ],
          dendrogram   = "none",
          scale        = "none",
          col          = colo,
          trace        = "none",
          density.info = "none",
          Rowv         = FALSE,
          Colv         = FALSE,
          ylab         = "Smoothness: holding time",
          xlab         = "Timeliness: forecast horizon",
          main         = "Correlation with Causal MSE Predictor")

# ------------------------------------------------------------------------
# 8.2.5 Column-scaled heat map: isolating the effect of ht on accuracy
# ------------------------------------------------------------------------
# Scaling within each column (fixed forecast horizon) removes the absolute
# effect of the forecast horizon on correlation levels.
# This makes the within-column variation (driven purely by ht) visible.
# The darkest ridge in each column marks the MSE-optimal ht for that horizon.
heatmap(target_mat,
        col   = colo,
        scale = "column",
        Rowv  = NA,
        Colv  = NA,
        ylab  = "Smoothness: holding time",
        xlab  = "Timeliness: forecast horizon",
        main  = "Trilemma: Correlation with Effective Acausal Target (Scaled Along ht)")

# ------------------------------------------------------------------------
# 8.2.6 Column-scaled heat map for the causal MSE criterion
# ------------------------------------------------------------------------
# Applying the same column-scaling to MSE_mat yields an identical plot to 8.2.5.
# This is not a coincidence: for a fixed forecast horizon, crit_rhoyz and
# crit_rhoy_target differ only by a constant scaling factor (Proposition 5,
# JBCY paper). Column-wise scaling cancels this factor, making both heat maps
# numerically and visually equivalent.
heatmap(MSE_mat,
        scale = "column",
        col   = colo,
        Rowv  = NA,
        Colv  = NA,
        ylab  = "Smoothness: holding time",
        xlab  = "Timeliness: forecast horizon",
        main  = "Trilemma: Correlation with Causal MSE Predictor (Scaled Along ht)")



if (F)
{  
  # We can also rely on ggplot for drawing a heat-map: the code is set-up in a function heat_map_func
  # We can apply the same scaling as above: to bring forward the ht-effect better
  scale_column<-T
  # We can select MSE-predictor or effective target
  select_acausal_target<-T
  
  heat_map_func(scale_column,select_acausal_target,MSE_mat,target_mat)
}

# ========================================================================
# Final Comment: Timeliness — Alternatives, Extensions and Future Directions
# ========================================================================
#
# Timeliness via forecast horizon (current approach):
#   - In Examples 3 and 8, timeliness (advancement / left-shift of the filter output
#     relative to the target) is controlled by increasing the forecast horizon h.
#   - Larger h shifts the SSA predictor further ahead in time, at the cost of
#     reduced accuracy (lower correlation with the target) and/or reduced
#     smoothness (lower holding-time) — the trilemma trade-off visualised above.
#
# ------------------------------------------------------------------------
# A more powerful route to timeliness: the Look-Ahead DFP/PCS approach
# ------------------------------------------------------------------------
#   - The DFP (Decoupling From Present) and the PCS (Peak Correlation Shifting))
#     offer fundamentally different and more powerful mechanisms for
#     achieving timeliness than simply augmenting the forecast horizon h.
#   - Key properties of the Look-Ahead DFP/PCS framework:
#       * It can generate leads that extend BEYOND the range achievable by
#         increasing the forecast horizon alone.
#       * For any given level of target correlation (accuracy), DFP/PCS
#         produces the MAXIMUM attainable lead among all linear predictors —
#         it is lead-optimal by construction (though the lead is measured differently in DFP and PCS).
#       * It defines a novel Accuracy–Timeliness efficient frontier:
#         the set of Pareto-optimal (accuracy, lead) combinations that cannot
#         be simultaneously improved upon by any other linear filter.
#   - This frontier provides a principled and quantitative basis for
#     navigating the accuracy–timeliness dimension of the prediction trilemma,
#     complementing the Accuracy-Smoothness trade-off inherent to (M-)SSA.

#   -The MDFA tutorial (https://wiaidp.github.io/MDFA-tutorial) proposes an alternative frequency-domain trilemma
#
# ------------------------------------------------------------------------
# Outlook:
# ------------------------------------------------------------------------
#   - A dedicated tutorial covering the Look-Ahead DFP/PCS approach,
#     its theoretical foundations, and its application to BCA is currently
#     in preparation.
# ========================================================================
