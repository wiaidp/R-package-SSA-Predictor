
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
# FINDING 3: ACTING ON THE FORECAST TRILEMMA TROUGH SSA (Example 8)
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
# Example 5
# Working with autocorrelated data (instead of noise)
# We here engraft SSA onto the classic HP-concurrent and apply the filter(s) to autocorrelated data
# 5.1 Generate data 
set.seed(4)
len<-1200
a1<-0.3
b1<-0.2
# Generate series
x<-arima.sim(n = len, list(ar = a1, ma = b1))
# Estimate coefficients
estim_obj<-arima(x,order=c(1,0,1))
# Check diagnostics: OK
tsdiag(estim_obj)
# Filter data: apply HP-concurrent
y_hp<-na.exclude(filter(x,hp_trend,side=1))
ts.plot(y_hp)

#-----------------
# 5.2 Holding time: we use the classic HP-concurrent (hp_trend)
ts.plot(hp_trend)
ht_hp_trend_obj<-compute_holding_time_func(hp_trend)
ht_hp<-ht_hp_trend_obj$ht 
ht_hp
# Compare with empirical holding time: the latter is larger
compute_empirical_ht_func(y_hp)
# This is because xt is not white noise!
# However, the computation of the holding-time assumes white noise
# Therefore, we have to decompose xt in a white noise sequence: Wold decomposition, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# Once decomposed, the holding-time is calculated properly

# Step 1: Compute the MA-inversion of the ARMA (Wold-decomposition or MA-inversion)
# One can insert true or estimated parameters 
xi_data<-c(1,ARMAtoMA(ar=a1,ma=b1,lag.max=L-1))
# Have a look at xi: if xi has not decayed to zero then one should increase the filter-length L
# Note:
#   -L should be sufficiently large so that finite-length convolutions and deconvolutions are good proxies (of infinite length transformations), see section 2 in JBCY paper
#   -For integrated processes xt, the MA-inversion xi does not decay to zero: this case will be analyzed in tutorials 3 (Hamilton filter), 4 (Baxter King filter) and 5 (HP-gap filter): see proposition 4 in JBCY paper for background 
par(mfrow=c(1,1))
ts.plot(xi_data)
# Step 2: Convolve xi_data and hp, see section 2 of JBCY paper: 
#   The resulting convolved filter is applied to epsilont (innovation in Wold decomposition)
#   Therefore the expected holding-time (of the convolved filter) is 
hp_conv<-conv_two_filt_func(xi_data,hp_trend)$conv
ht_hp_conv_obj<-compute_holding_time_func(hp_conv)
ht_hp_conv_obj$ht
# Now the expected holding time (above) matches the empirical one (below), at least up to finite sample error (the empirical number corresponds to the expected ht for increasing sample length)
compute_empirical_ht_func(y_hp)

# We stressed in tutorial 2.0 that an application of the same fixed HP-design to macro-indicators with different 
#   autocorrelation structure leads to qualitatively different cycles (INDPRO and non-farm payroll)

#------------------
# 5.3 SSA and hyperparameters
# Holding time: we want SSA to lessen the number of zero-crossings when compared to HP 
# We here increase ht by 50% when compared to HP
# Note: since the data xt is not white noise, we do not provide the original holding time of HP (ht_hp), 
#   which assumes xt=epsilont is white noise. Instead we here use the effective and corrected holding-time 
#   ht_hp_conv_obj$ht, see the previous discussion and analysis above 
ht<-1.5*ht_hp_conv_obj$ht
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# Our selection here means that SSA will have 33% less crossings:
ht/ht_hp_conv_obj$ht
# Forecast horizon: nowcast i.e. delta=0
forecast_horizon<-0
# Target: classic HP-concurrent
gammak_generic<-hp_trend
# Autocorrelated data: we provide the Wold decomposition to the SSA function
xi<-xi_data

# SSA of HP-target: we must include xi in the function call (otherwise it is assumed that xt=epsilont is white noise, by default)
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# Since xt is not white noise, the two SSA filters ssa_eps and ssa_x are not the same: 
# ssa_x is the filter which is applied to xt
SSA_filt_HP<-ssa_x<-SSA_example5<-SSA_obj_HP$ssa_x
# ssa_eps is applied to epsilont: it is the convolution of ssa_x and Wold-decomposition xi_data
ssa_eps<-SSA_obj_HP$ssa_eps

# Plot and compare filters: 
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
# We can also compare the filters which are applied to epsilont (model residuals)
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


#--------------------------------------------
# 5.4 Filter series and compare classic concurrent HP with SSA

len<-100000
set.seed(1)
x<-arima.sim(n = len, list(ar = a1,ma=b1))
# Compute filter output of SSA-HP filter
yhat<-filter(x,SSA_filt_HP,side=1)
# Compare expected and empirical holding-times: they match, as desired
ht
compute_empirical_ht_func(yhat)

# Compute concurrent HP (classic HP nowcast)
HP_concurrent<-filter(x,hp_trend,side=1)
# Compare expected and empirical holding-times: the empirical holding time of SSA is (approximately) 50% larger, 
#   as desired
ht_hp_conv_obj$ht
compute_empirical_ht_func(HP_concurrent)

# Plot both series:
mplot<-na.exclude(cbind(yhat,HP_concurrent))
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# Plot a short sample of the series
par(mfrow=c(1,1))
anf<-500
enf<-1000
anf<-1000
enf<-1500

# SSA generates ~50% less crossings
# The additional crossings of HP are typically clustered at time points where the filter output seems to hover at the zero line
ts.plot(mplot[anf:enf,],col=colo)
mtext(paste("SSA(",ht,",",forecast_horizon,")",sep=""),col=colo[1],line=-1)
mtext("HP-concurrent",col=colo[2],line=-2)
abline(h=0)

# We could also address timeliness (lead/lags) as in example 4 above but we leave this as an exercise 


##########################################################################################################
##########################################################################################################
# Example 6
# Same as example 5 but we now target specifically the output of the symmetric HP-filter
# As a result, we will obtain a one-sided filter which tracks the output of the symmetric filter better than the classic HP-concurrent or HP-MSE (the latter is assuming white noise)
# Recall that the classic HP-concurrent is optimal if the series is an ARIMA(0,2,2), where the two MA-parameters are determined by lambda, see tutorial 2.0
# This example is a bit tricky because everything is merged and entangled!

# Let us emphasize that:
# -This exercise does not refer to the majority of use-cases of HP: 
#   -Most (nearly all) practitioners assume that the classic concurrent HP is the optimal estimate of the two-sided filter at the sample end (real-time business BCA)
#   -This assumption is wrong because economic data does not conform to an ARMA(0,2,2)-process, see tutorial 2.0
#   -Therefore, by applying the classic HP-concurrent, users are not effectively tracking the two-sided target (at least not optimally)
# -We advocated in tutorial 2.0 that the classic HP-concurrent has some desirable characteristics for BCA
#   -Therefore, we plugged SSA on the classic HP-concurrent in the previous exercises 2,3 and 5
#   -But, clearly, its tracking ability (of the two-sided target) is not optimal in typical applications 
#   -When applying HP-concurrent, practitioners use a pertinent BCA-tool; but the tool is not doing what it is claimed to do (and what the analyst think it does)
# -Exercise 6, here, addresses specifically an optimal one-sided filter for tracking the two-sided HP, assuming the true 
#   (or the empirical) model of the data (which is not an ARIMA(0,2,2))
# -This exercise is meant for analysts mainly interested in tracking the two-sided filter by an optimal concurrent design
# -Besides the optimal MSE we also present SSA extensions which are smoother and/or faster 

# 6.1 Specify symmetric filter as target
L_sym<-401
# Should be an odd number: otherwise HP is not centered correctly
if (L_sym/2==as.integer(L_sym/2))
{
  print("Filter length should be an odd number")
  print("If L_sym is even then HP cannot be correctly centered")
  L_sym<-L_sym+1
}  
# HP monthly design
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L_sym,lambda_monthly)
# Symmetric HP: this is not yet our new target, because this filter is one-sided causal
# But we want the two-sided acausal as target for SSA (instead of one-sided MSE in example 1)
hp_target=HP_obj$target
ts.plot(hp_target)

# Forecast horizon: here things become a bit tricky
# The variable h is now our effective forecast horizon
# For a nowcast of the symmetric filter we set h<-0 (h>0 for forecast, h<0 for backcast)
h<-0
# Here is the tricky part:
# The symmetric filter hp_target is causal: its center is at lag (L_sym+1)/2, see brown line in plot below
# For a nowcast of the two-sided acausal filter we have to shift the causal filter to the left, see violet line
#   The peak of the violet line is now shifted to lag h=0 (nowcast)
# Therefore our effective  forecast horizon is (L_sym-1)/2+h 
forecast_horizon<-(L_sym-1)/2+h
causal_sym<-c(rep(0 ,(L_sym-1)/2 ),hp_target)
acausal_sym<-c(rep(0,max(0,(L_sym+1)/2-forecast_horizon-1)),hp_target[max(1,1+(forecast_horizon-(L_sym-1)/2)):L_sym],rep(0 ,forecast_horizon ))
acausal_sym<-acausal_sym[1:length(causal_sym)]
plot(causal_sym,col="brown",axes=F,type="l",xlab="lead                                                                  lag",ylab="",main="Acausal vs. causal HP: the target is acausal")
lines(acausal_sym,col="violet")
abline(v=(L_sym+1)/2,col="violet")
abline(v=L_sym,col="brown")
mtext(paste("Target acausal filter: center is at lag 0. For SSA we shift the causal HP (brown line) to the left by forecast_horizon=",forecast_horizon,sep=""), line=-1,col="violet")
mtext(paste("                                                   Causal symmetric HP as calculated by mFilter: center is at lag ",(L_sym+1)/2,sep=""), line=-3,col="brown")
axis(1,at=1:length(acausal_sym),labels=-forecast_horizon +1:length(acausal_sym))
axis(2)
box()

# Since the target is the symmetric filter, the natural benchmark for SSA is the MSE estimate of the target 
#  (this is also computed by our SSA function, see below. We here replicate these calculations, for illustration)
# Proceeding for obtaining the MSE-filter (recall that hp_mse is not optimal here because xt is not white noise):
#   a. Compute the convolution of symmetric filter hp_trend and Wold-decomposition xi: this way, HP is applied to epsilont 
#   b. Truncate the convolution at lag forecast_horizon+1 (which is lag 0 in un-shifted data) because forecasts of future epsilont are zero
# Step a: convolve xi_data and symmetric hp_target
hp_conv_mse_d<-conv_two_filt_func(xi_data,hp_target)$conv
# Step b: truncate at lag 0 corresponding to forecast_horizon
hp_conv_mse<-hp_conv_mse_d[(forecast_horizon+1):L_sym]
# We can now compute the holding-time of the optimal MSE filter: this is a natural benchmark for SSA 
# -Idea: SSA should improve smoothness of best (MSE) nowcast
# -Of course, the holding-time could be set differently (based on a priori knowledge or on particular priorities). 
# -But having a natural (optimal) benchmark at disposal allows for more stringent/interesting comparisons
ht_hp_conv_mse_obj<-compute_holding_time_func(hp_conv_mse)
ht_hp_conv_mse<-ht_hp_conv_mse_obj$ht
# ht of optimal MSE (assuming the data is not white noise): 
ht_hp_conv_mse
# We can compare with the holding-time ht_mse of hp_mse above (assuming white noise): this is slightly smaller because the data is weakly positively autocorrelated
ht_mse
#-----------------------------------------------
# 6.2 Apply SSA: 
# SSA: augment ht_hp_conv_mse of optimal MSE benchmark by 50% 
ht<-1.5*ht_hp_conv_mse
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# In contrast to example 5 we now specify the symmetric target filter
gammak_generic<-hp_target
# Forecast horizon was discussed above 
forecast_horizon<-forecast_horizon
# We use the same filter length as in example 5
L<-201
# In contrast to example 4, we now supply the Wold-decomposition (MA-inversion) of the data generating process
xi<-xi_data
# SSA of HP-target
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# SSA_func also computes the MSE estimate which we already computed above
#   This filter is supposed to be applied to epsilont, not xt (convolution of target with xi)
mse_eps<-SSA_obj_HP$mse_eps
ts.plot(cbind(hp_conv_mse[1:L],mse_eps),main="MSE estimate: SSA_func vs. own calculation (both filters overlap)")

# retrieve the SSA-filter applied to xt (we here ignore ssa_eps as applied to epsilont since we shall filter xt)
SSA_filt_HP<-SSA_obj_HP$ssa_x
# Retrieve benchmark MSE filter: this is automatically calculated by SSA 
HP_MSE_x<-SSA_obj_HP$mse_x


# Plot and compare with SSA designs of previous examples 4 and 5
colo<-c("black","brown","blue")
par(mfrow=c(1,1))
mplot<-cbind(SSA_example4,SSA_example5,SSA_filt_HP)
colnames(mplot)<-c("Example 4","Example 5","Example 6")

# The filters of examples 4 and 6 look similar. Why is that?
#   -Example 4 relied on the symmetric target too, but it assumed xt=epsilont white noise (xi=NULL) and a smaller holding time ht=12
#   -Here, in example 6, we assume an ARMA-process and a larger holding-time of approximately 17
#   -As it happens (fortuitously) the solution of example 4 when applied to the ARMA-process has a holding-time very close to 17
#   -Therefore, both designs are close to each other (up to some differences towards lag 0)
#   -More significant differences could be observed by changing the above specifications (the ARMA-process and/or the holding-time constraint)
# The scale of example 5 is quite different
#   -This is because the estimation problem of example 5 is much simpler since we target the one-sided filter hp_trend
#   -In contrast, examples 4 and 6 nowcast the acausal two-sided HP
#   -The smaller scaling (zero-shrinkage) in the latter examples reflects the increased uncertainty in this case
plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
lines(mplot[,3],col=colo[3],lwd=2,lty=1)
mtext(colnames(mplot)[3],col=colo[3],line=-3)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Now that the relevant filters have been sorted out and computed, we can  filter the series, compare 
#   performances and proceed to our checks of empirical and expected numbers (holding times, correlations)
#-------------------------------------------
# 6.3 Filter series 
# Generate very long series in order to obtain accurate empirical estimates
len<-1000000
set.seed(14)
x<-arima.sim(n = len, list(ar = a1,ma=b1))
# Compute filter output of SSA-HP filter: optimal as applied to xt
yhat<-filter(x,SSA_filt_HP,side=1)
# Compare expected and empirical holding-times: they match, as desired
ht
compute_empirical_ht_func(yhat)

# Compute filter output of  MSE 
HP_mse<-filter(x,HP_MSE_x,side=1)
# Compare expected and empirical holding-times: both match
ht_hp_conv_mse
compute_empirical_ht_func(HP_mse)
# The empirical holding time of SSA is (approximately) 50% larger, as desired

# Compute output of target, i.e., symmetric HP: we now have to set side=2 (symmetry)
HP_symmetric<-filter(x,hp_target,side=2)

# Out of curiosity we also compute the output of the classic HP-concurrent: side=1 since the filter is one-sided
HP_concurrent<-filter(x,hp_trend,side=1)
# The classic HP concurrent generates more crossings than SSA
compute_empirical_ht_func(HP_concurrent)
# Interestingly, as we shall see below, the classic HP concurrent looses not only in terms of smoothness but also in terms of correlation with the target
#   -The underlying implicit ARIMA(0,2,2) assumption is a misspecification!

# Plot all filter outputs
colo<-c( "blue","green","black","brown" )
# Plot both series: 
mplot<-na.exclude(cbind(yhat,HP_mse,HP_symmetric,HP_concurrent))
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP MSE","HP-symmetric (effective target)","Classic HP concurrent")
# Plot a short sample of the series
par(mfrow=c(1,1))
anf<-500
enf<-1000
anf<-1000
enf<-1500
# SSA generates ~30% less crossings than MSE or HP concurrent
ts.plot(mplot[anf:enf,],col=colo)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
if (ncol(mplot)>1)
  for (i in 2:ncol(mplot))
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
abline(h=0)

# Discussion:
#   -The scale of the classic HP-concurrent is off the mark: the filter assumes xt to be an ARIMA(0,2,2)-process
#     -Severe misspecification
# One possibility to look at scalings (of filter outputs) is to look at the sum of filter coefficients
#   -The sum of filter coefficients corresponds to the amplitude function of a filter at frequency zero (eventually up to sign)  
#   -Therefore, the sum is informative about the scaling of low-frequency components by the filter
# Two-sided target: the coefficients sum to one
sum(hp_target)
# Classic concurrent: sum is one, too
sum(hp_trend)
# SSA: sum is markedly smaller than one
sum(SSA_filt_HP)
# hp_mse: sum is markedly smaller than one
sum(hp_mse)
# We infer that hp_trend (classic concurrent HP) matches hp_target, at least in terms of amplitude at frequency zero: 
#   -this is because the implicit model (ARIMA(0,2,2)) assumes the data to be strongly trending
#   -in order to track the two-sided filter, the one-sided hp_trend must pass the dominating 
#     (low-frequency) trend, without changing scales: the filter coefficients sum to one.
# However, if the data is not strongly trending, i.e. if the ARIMA(0,2,2) is severely misspecified, then there is 
#     no necessity to track the (missing) trend: the filter coefficients do not have to add to one anymore
#   -In particular, in the presence of noise (strong high-frequency components), MSE-performances generally improve
#     by shrinking the predictor (zero-shrinkage): 
#       -the coefficients of SSA sum to ~0.5 (instead of one)
#       -hp_mse assumes the data to be white noise and its coefficients sum to ~0.5
# The `odd' scaling of hp_trend in the above plot is due to severe misspecification. 
#   -Part of this misspecification could be addressed by simply re-scaling the filter output (say by 0.5)
#   -But even after re-scaling the classic HP-concurrent is not optimal, see below

#------------------------------------------
# 6.4 Compute empirical and true (expected) performance numbers and compare all one-sided designs
# Compute empirical cross correlations of all designs: 
cor_mat<-cor(mplot)
cor_mat
# Let's first look at correlations of the various one-sided designs with the target: 
#  Numbers in the third row of the correlation matrix
cor_mat[3,]
# -MSE marginally outperforms SSA (MSE is best possible; but SSA is smoother: 50% larger empirical holding-times in above simulation):
cor_mat[3,2]
cor_mat[3,1]
# -Of course, MSE also outperforms the classic one-sided HP
cor_mat[3,2]
cor_mat[3,4]
# -Interestingly, SSA outperforms HP-concurrent both in terms of smoothness (fewer crossings) as well 
#   as in terms of correlation with target 
# As stated: the classic HP-concurrent filter is generally not optimal for applications: typical economic data
#   (in levels or in differences) is less smooth than assumed by hp_trend
cor_mat[3,1]
cor_mat[3,4]

# Let's now look at empirical correlations of SSA with target and MSE-benchmark: 
#  Numbers in first row of the above correlation matrix
cor_mat[1,]
# -These numbers correspond to the criterion values of SSA: 
#   They should match crit_rhoy_target (correlation with target) and crit_rhoyz (correlation with MSE: see proposition 4 in JBCY paper)
# A. crit_rhoy_target
cor_mat[1,3]
SSA_obj_HP$crit_rhoy_target
# B. crit_rhoyz
cor_mat[1,2]
SSA_obj_HP$crit_rhoyz


##########################################################################################################
##########################################################################################################
# Example 7
# This is more of a counter-example and it is not related to SSA.
# We here analyze the classic HP-gap filter, see also tutorial 2.0.
# Based on its characteristics we advise against its use in BCA, in agreement with Hamilton: "never use the HP (gap)...", see tutorial 3

# HP gap
# 7.1 Compute HP-gap: we use the same length as in example 6 above
L_sym<-401
# Should be an odd number: otherwise HP is not centered correctly
if (L_sym/2==as.integer(L_sym/2))
{
  print("Filter length should be an odd number")
  print("If L_sym is even then HP cannot be correctly centered")
  L_sym<-L_sym+1
}  
# HP monthly design
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L_sym,lambda_monthly)
# Bi-infinite HP: this is our new target (instead of one-sided MSE in example 1)
hp_target=HP_obj$target
ts.plot(hp_target)
# Gap
hp_gap<-HP_obj$hp_gap
ts.plot(hp_gap)
# Classic concurrent
hp_trend<-HP_obj$hp_trend
ts.plot(hp_trend)
# Gap is 1-hp_trend
# Check: the difference of the following filters should vanish everywhere
max(abs((c(1,rep(0,L_sym-1))-hp_trend)-hp_gap))

#---------------------------------------------
# 7.2 Transformation: from levels to first differences
# The gap filter is applied to data in levels: this renders a direct analysis cumbersome and difficult
# Instead, we here derive a filter whose output is the same as HP-gap when applied to first differences, see tutorial 2.0
# This transformation will simplify our analysis, see proposition 4 in JBCY paper for background.
gap_diff<-conv_with_unitroot_func(hp_gap)$conv
par(mfrow=c(2,1))
# Note that the coefficients of the new filter ham_diff vanish for lags larger than length of Hamilton_filter_adjusted and therefore we could set L=length(hamilton_filter_adjusted)
ts.plot(gap_diff,main="HP-gap as applied to first differences")
ts.plot(hp_gap,main="HP-gap as applied to level")

# We now verify that the outputs of both filters are identical
set.seed(252)
len<-L+2000
# Generate random-walk: data in levels will be fed to hp_gap
y<-cumsum(rnorm(len))
# Difference data: this will be fed to gap_diff: lengthen series with a zero to match y
x<-c(0,diff(y))
len_diff<-length(x)
# Compute new cycle based on new filter ham_diff applied to returns
yhat_diff<-filter(x,gap_diff,side=1)
yhat_gap<-filter(y,hp_gap,side=1)

# Check: both series are identical
par(mfrow=c(1,1))
ts.plot(yhat_diff,col="blue",main="Transformed HP-gap applied to differences replicates original HP-gap applied to levels")
lines(yhat_gap,col="red")
#-------------------------------------------
# 7.3 We can now compute the amplitude  of HP-gap or, better, gap_diff as applied to first differences
# First differences of typical economic data are generally close to white noise (typical spectral shape, see Granger (1966))
# Therefore the squared amplitude of gap_diff is close to the spectral density of the filter output (convolution theorem)
#   This is the main reason for considering gap_diff: we can derive the spectral density of the extracted cycle

# We now select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude of gap_diff
amp_gap_diff<-amp_shift_func(K,as.vector(gap_diff),F)$amp

# Plot amplitude function
plot(amp_gap_diff,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP-gap as applied to differences",sep=""))
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# Plot squared amplitude function: this is the true spectral density of the original HP-gap `cycle', when applied to a random-walk
# This is a good proxy of the true spectral density of the HP-gap `cycle' when applied to typical (non-stationary) economic data in levels
par(mfrow=c(1,1))
plot(amp_gap_diff^2,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Squared amplitude HP-gap as applied to differences",sep=""))
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# Outcomes:
#   1. The original HP-gap is a highpass when applied to levels, see tutorial 2.0; but it is a bandpass when applied to first differences (the above amplitude vanishes at frequency zero)
#     The one-sided HP-gap assumes a second order unit-root: after first differences, gap_diff must still remove a remaining unit-root
#   2. HP-gap tends to generate a spurious cycle whose periodicity corresponds to the frequency of the peak-amplitude in the above plot

# Let's briefly check the second claim
# This is the frequency of the peak-amplitude
omega_gap<-pi*(which(amp_gap_diff==max(amp_gap_diff))-1)/K
# The periodicity in months: 
2*pi/omega_gap
# Approximately 6 years: a bit short for a completed cycle when compared against the duration of longer expansions (great moderation) 
#   Therefore HP-gap tends to generate spurious alarms in midst of expansions, see tutorials 2.0 and 5

# The periodogram of the filter output confirms our findings: low and high frequencies are damped, 
#   thus generating a spurious cycle at the peak frequency
per_obj<-per(na.exclude(yhat_gap),T)
# The acf confirms these findings
acf(na.exclude(yhat_gap))
# Finally: here's the cyle
par(mfrow=c(1,1))
ts.plot(yhat_gap)
abline(h=0)
# There are lots of noisy zero-crossings, too, which would hamper a real-time assessment of the cycle, see tutorials 2.0 and 5
#   As we shall see, the Baxter and King filter is subject to similar issues, see tutorial 4
# These findings confirm and reinforce Hamilton's statement: never use the HP (-gap) filter for BCA
# But we may add: try the HP-concurrent lowpass instead, as applied to differences, see tutorials 2.0 and 5
# Tutorial 2.0 shows that HP-trend applied to differences is a more conservative design (than the original HP-gap)
#   -It tracks expansions and recessions well (neither too smooth nor too noisy: a smart compromise); 
#   -It does not generate spurious alarms years ahead of (or past) effective recessions; 
#   -It is slightly lagging behind HP-gap at start and end of recessions;
#   -Its real-time characteristics can be modified by SSA in order to match specific priorities, see the above examples


##########################################################################################################
##########################################################################################################
# Example 8
# Visualization of Prediction  Trilemma for HP
# Background:
# -MSE, timeliness and smoothness define a forecast trilemma, see tutorial 0.1
# -We measure all three terms as follows in this exercise:
# a. MSE: we compute the SSA-criteria crit_rhoyz and crit_rhoy_target 
#   -The criteria measure the correlations of the SSA-predictor with causal and acausal targets, see examples above
#   -Large correlations mean: small MSE 
# b. Timeliness: forecast horizon
# c. Smoothness: ht in holding-time constraint

# For visualization, we then represent or plot all three terms of the trilemma in a heat-map

# For this study we consider the classic concurrent trend estimate hp_trend as a target of SSA. 
# We then compute crit_rhoy_target and crit_rhoyz for a range of ht and forecast horizons.
# The forecast trilemma is visualized by a heat map of the criterion value(s) as a function of ht and 
#   forecast horizon.

#----------------------------------------
# 8.1 Setting-up SSA

# The following computations are a bit lengthy (~5 Min. on 3GHz single core)
#   One can skip the loop and load a file with the results

compute_length_loop<-F
# Specify target
gammak_generic<-hp_target

if (compute_length_loop)
{  
# Holding time of hp_mse 
  ht_mse
# Compute SSA and MSE for a selection of ht
  ht_vec<-seq(max(2,ht_mse/4), 2*ht_mse, by = 0.1)
# Compute SSA and MSE for a selection of forecasts horizons
# Note: we must shift the causal symmetric HP by (L_sym-1)/2 to the left in order to obtain the acausal two-sided target
  delta_vec<-0:24+(L_sym-1)/2
  
  pb = txtProgressBar(min = 0, max = length(ht_vec), initial = 0,style=3) 
  
  MSE_mat<-target_mat<-matrix(ncol=length(delta_vec),nrow=length(ht_vec))
# Loop through all combinations of ht and forecast horizon: compute the SSA filter and collect 
#   crit_rhoy_target (correlation of SSA with effective target) as well as crit_rhoyz (correlation with causal MSE benchmark)  
  for (i in 1:length(ht_vec))
  {
    setTxtProgressBar(pb,i)
    for (j in 1:length(delta_vec))
    {  
      rho1<-rho1<-compute_rho_from_ht(ht_vec[i])
      forecast_horizon<-delta_vec[j]
# Skip xi: we assume white noise    
      SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1)
# Correlation with (caual) MSE predictor: this is the preferred measure here because we can benchmark SSA
#   directly against MSE 
      MSE_mat[i,j]<-SSA_obj_HP$crit_rhoyz
# Or correlation with (acausal) target 
      target_mat[i,j]<-SSA_obj_HP$crit_rhoy_target
    }
  }
  close(pb)
# Row-names correspond to holding-times; column-names are forecast horizons  
  rownames(MSE_mat)<-rownames(target_mat)<-round(ht_vec,2)
# Forecast horizon: we remove the artificial shift (L_sym-1)/2 
  colnames(MSE_mat)<-colnames(target_mat)<-delta_vec-(L_sym-1)/2
# Save results
  save(MSE_mat,file=paste(getwd(),"/Data/Trilemma_mse_heat_map",sep=""))
  save(target_mat,file=paste(getwd(),"/Data/Trilemma_target_heat_map",sep=""))
} else
{  
# Load pre-computed results  
  load(file=paste(getwd(),"/Data/Trilemma_mse_heat_map",sep=""))
  load(file=paste(getwd(),"/Data/Trilemma_target_heat_map",sep=""))
}

# 1. MSE_mat collects the correlations crit_rhoyz of SSA with the causal MSE benchmark predictor of the target
#   Row names correspond to ht (holding-time constraint)
#   Column names correspond to the forecast horizon: from a nowcast up to 24-steps ahead
head(MSE_mat)
tail(MSE_mat)
# 2. target_mat collects the correlations crit_rhoy_target of SSA with the effective (acausal two-sided) target 
#     -In our case: the two-sided filter shifted by 0,1,...,24
#   Naturally, these correlations are smaller than in MSE_mat 
head(target_mat)
tail(target_mat)
#---------------------------------------
# 8.2 Heat maps
# We can now represent and visualize the trilemma in a heat map
# 8.2.0 Specify color scheme
lcol<-100
coloh<-rainbow(lcol)[1:(10*lcol/11)]
colo<-coloh[length(coloh):1]
# 8.2.1. Heat-map of correlations with acausal (effective) target
heatmap.2(target_mat[nrow(target_mat):1,], dendrogram="none",scale = "none", col = colo,trace = "none", density.info = "none",Rowv = F, Colv = F,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Trilemma: Correlation with Effective Acausal Target")

# Interpretations of heat-map
# -For fixed ht, the correlations decrease with increasing forecast horizon
#   -Enhancing timeliness (larger forecast horizon) for fixed smoothness (ht) means an increase of MSE (decrease of correlation)
# -For fixed forecast horizon, the correlations peak somewhere: 
#   -The peak value corresponds to the classic MSE-predictor of the target for that forecast horizon and the corresponding ht is the holding-time of the classic MSE predictor
#   -Above and below the peak-value, the holding-time constraint in SSA is `activated': SSA maximizes the correlation subject to ht
#   -Enhancing smoothness, relative to the MSE-benchmark and for fixed forecast horizon, means a decrease of correlation or increase of MSE
# -Increasing simultaneously ht and the forecast horizon (along a diagonal) affects correlations (MSE) disproportionately
#   -See SSA-forecasts vs. SSA-nowcasts in the above examples 
# -Improving correlations (smaller MSE) and timeliness (larger forecast horizon) affects smoothness (ht) disproportionately

# 8.2.2 The resolution in the above plot can be improved by `squeezing' the correlations with the non-linear log-transform
heatmap.2(log(target_mat[nrow(target_mat):1,]), dendrogram="none",scale = "none", col = colo,trace = "none", density.info = "none",Rowv = F, Colv = F,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Trilemma: Log-Correlation with Effective Acausal Target")
# We can now observe additional structure

# 8.2.3 We can plot slices (selected columns) of the above heat-map 
select_vec<-22:24
mplot<-target_mat[,select_vec]
coli<-rainbow(length(select_vec))
par(mfrow=c(1,1))
plot(mplot[,1],col=colo,ylim=c(min(mplot),max(mplot)),axes=F,type="l",xlab="Holding time",ylab="Correlation",main="Trilemma for HP: Correlations as a Function of ht and a Selection of Forecast Horizons")
mtext(paste("Forecast horizon ",colnames(MSE_mat)[select_vec[1]],sep=""),col=coli[1],line=-1)
if (length(select_vec)>1)
  for (i in 2:length(select_vec))
  {
    lines(mplot[,i],col=coli[i])
    mtext(paste("Forecast horizon ",colnames(MSE_mat)[select_vec[i]],sep=""),col=coli[i],line=-i)
  }
cor_val=0.105
abline(h=cor_val)
for (i in 1:ncol(mplot))
  abline(v=which(mplot[1:(nrow(mplot)-1),i]>cor_val&mplot[2:nrow(mplot),i]<cor_val),col=coli[i])
axis(1,at=1:nrow(MSE_mat),labels=rownames(MSE_mat))
axis(2)
box()
# For a given forecast horizon, the peak of the correlation value corresponds to the MSE-predictor 
# To the left and to the right of the peak-value, SSA maximizes the correlation subject to ht
# SSA is smoother (than the classic MSE predictor) if ht is to the right of the peak; otherwise it is `unsmoother'
# For a given fixed correlation, one can trade smoothness (holding-time) for timeliness (forecast horizon)
#   -As an example, a fixed correlation value of 0.105 (horizontal black line in graph) intersects the correlation 
#     curves corresponding to forecast horizons 21, 22 and 23 at the holding-times ht~20, ht~16 and ht~9, in decreasing size

# In our applications of SSA to HP in example 3, we traded timeliness (larger forecast horizon) against MSE (smaller correlation) for fixed ht, see also tutorial 5

# 8.2.4 It is also possible to draw a heat-map for the correlations with the causal MSE-predictor
#   -This heat-map does not emphasize prediction (the target is causal); it is about smoothing (see section 2.4 in JBCY paper)
heatmap.2(MSE_mat[nrow(MSE_mat):1,], dendrogram="none",scale = "none", col = colo,trace = "none", density.info = "none",Rowv = F, Colv = F,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Correlation with causal MSE-Predictor")


# 8.2.5 We can emphasize more specifically the effect of ht on correlations (MSE) for a fixed forecast horizon
# For this purpose, we can scale the data in the column direction of the heat map 
#   -Scaling along the column means that the absolute effect of the forecast horizon is diminished
#   -As above, the MSE-benchmark predictor corresponds to the peak criterion value for each forecast criterion (darkest ridge in plot)
heatmap(target_mat,col=colo,scale="column",Rowv = NA, Colv = NA,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Trilemma: Correlation with  Effective Acausal Target (scaled along ht)")

# 8.2.6 We can apply the same scaling to the correlations with the causal MSE predictor
heatmap(MSE_mat, scale = "column", col = colo,Rowv = NA, Colv = NA,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Trilemma: Correlation with Causal MSE Predictor (scaled along ht)")
# Same plot because for given forecast horizon the two criteria differ only by a scaling, see proposition 5 in JBCY paper
#   Therefore, scaling along columns will cancel the difference between both criteria






if (F)
{  
# We can also rely on ggplot for drawing a heat-map: the code is set-up in a function heat_map_func
# We can apply the same scaling as above: to bring forward the ht-effect better
  scale_column<-T
# We can select MSE-predictor or effective target
  select_acausal_target<-T

  heat_map_func(scale_column,select_acausal_target,MSE_mat,target_mat)
}


