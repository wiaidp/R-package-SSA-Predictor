#######################################################################################
# Tutorial 7.4: M-SSA Components Predictor for German GDP (BIP-) Forecasting
#######################################################################################
#
# Nomenclature: 
#   The acronym BIP refers to German GDP (Brutto-Inlands-Produkt)
#
# Overview:
#   This tutorial extends the M-SSA framework from tutorial 7.3 in four directions:
#
#   1. Forecasting BIP (MSE-optimal):
#      - The original M-SSA predictor (tutorial 7.3) was designed to track the smoothed
#        trend growth rate of BIP (HP-BIP), using equally-weighted, standardized components
#      - It was not calibrated to minimize MSE when targeting raw BIP directly
#      - Here we introduce the 'M-SSA components predictor': an extended design that
#        replaces equal weighting with an optimal (MSE-minimizing) weighting step,
#        allowing explicit targeting of raw BIP
#
#   2. Interpretability:
#      - Use the individual M-SSA components to assess the reliability and
#        trustworthiness of the aggregate predictor's current outlook
#      - Key question: when the predictor signals a turning point, do the
#        individual components agree or diverge?
#
#   3. Explainability:
#      - Decompose forecast performance to identify which construction steps
#        generate the observed gains over simpler benchmarks
#      - Key question: at which forecast horizon, and through which mechanism,
#        does the M-SSA components predictor outperform?
#
#   4. M-SSA Smoothing:
#      - Added value of increased smoothness in terms of zero-crossing rate (Holding Time HT).
#        Intuition: maximizing sign accuracy and controlling  the rate of sign changes 
#        within M-SSA will provide a better balance between true and false alarms when
#        detecting cyclical turning points.  
#
#########################################################################################

# Main Ideas
#
# 1. Addressing noise:
#                Direct forecasts regress future (forward-shifted) BIP on the original
#                indicators (BIP, ip, ifo, ESI, spread).
#    Problem:    Noise in the raw indicators obscures the underlying dependence structure,
#                weakening the predictive signal at longer forecast horizons.
#    Solution:   Instead of regressing future BIP on raw indicators, regress it on
#                filtered indicators. Three filtering approaches are considered:
#                - Univariate HP filtering:            does not recover predictive power
#                                                      (see Exercise 4).
#                - M-MSE (multivariate, no smoothness constraint):
#                                                      restores statistical significance
#                                                      out-of-sample at longer forecast
#                                                      horizons, unlike direct forecasts.
#                - M-SSA (multivariate, with smoothness constraint):
#                                                      matches M-MSE in out-of-sample MSE
#                                                      performance while being substantially
#                                                      smoother (see Exercise 5).
#    Preference: Since M-SSA and M-MSE achieve comparable MSE, the smoother predictor
#                (M-SSA) is preferred: its less frequent and more reliable zero-crossings
#                translate into a better balance between true and false alarms when
#                detecting cyclical turning points (see Exercise 6).
#
# 2. Equal vs. optimal weighting:
#    Equal weighting:   Tutorial 7.3 proposed an equally weighted average of M-SSA
#                       components, where each component tracks the HP-filtered version
#                       of one indicator:
#                       M-SSA-BIP tracks HP(BIP), M-SSA-ip tracks HP(ip), ...,
#                       M-SSA-spread tracks HP(spread).
#    Optimal weighting: Here, instead of equal weighting, optimal regression weights are
#                       derived by regressing future BIP on the M-SSA-filtered indicators,
#                       letting the data determine the relative importance of each component.
#
# 3. Calibration:
#    - M-SSA is designed to track the HP-filtered version of each indicator, not BIP
#      directly.
#    - In the previous tutorial, all series were standardised to facilitate plotting
#      and interpretation; as a result, the level and scale of the M-SSA predictors
#      are not calibrated to match the original BIP series.
#    - The regression step introduced in this tutorial serves a dual purpose:
#        i)  derive optimal combination weights for the M-SSA components, and
#        ii) simultaneously calibrate level and scale of the combined predictor to match BIP
#            (rather than the HP-filtered targets used during M-SSA optimisation).

#######################################################################################
# Structure of the Tutorial: 6 Exercises
#######################################################################################
#
# Exercise 1:
#   - Derive M-SSA components and replicate the original equally-weighted M-SSA predictor (tutorial 7.3)
#   - Use components to assess forecast reliability (interpretability)
#   - Introduce the new optimal weighting step to directly target BIP (MSE sense)
#   - Out-of-sample performance evaluation vs. mean, direct forecast, and
#     original M-SSA predictor
#
# Exercise 2:
#   - Analyze real-time revisions of the new M-SSA components predictor
#   - Assess how the predictor's outlook changes as new data arrive
#
# Exercise 3:
#   - Skipped
#
# Exercise 4:
#   - Explainability: identify why the M-SSA components predictor outperforms
#     specifically at multiple-quarters-ahead forecast horizons
#   - Decompose the source of forecast gains across construction steps:
#     Can univariate filters compete with M-SSA? 
#
# Exercise 5:
#   - Introduce the 'M-MSE components predictor': same framework as M-SSA but
#     without the holding-time (HT) constraint (less smooth, more reactive)
#   - Compare forecast performance against the mean benchmark and the
#     M-SSA components predictor: 
#     -Does the multivariate M-MSE compete with M-SSA?
#     -Does the HT constraint of M-SSA pay off? 
#     -If yes: in which terms?
#
# Exercise 6:
#   - Compute final M-SSA and M-MSE components predictors using full data,
#     with Pandemic observations excluded from parameter estimation to avoid
#     distortion by singular outlier dynamics
#   - Identify and quantify the contribution of the larger holding-time (HT) 
#     constraint in M-SSA to forecast gains relative to M-MSE.
#
# A word of caution: the settings explored here probe the limits of what the M-SSA
#   predictor can achieve when forecasting German GDP multiple quarters ahead.
# While such an approach is legitimate when the true data-generating process is known
#   (as optimality of the filter is then certified), caution is mandatory in the
#   present application, because the data-model used (simple VAR(1))
#   is almost surely (with probability one) misspecified. 

# Trade-off between conservative and aggressive forecast settings:
#   - Conservative (small f_excess, see below): better interpretability and greater robustness,
#     but the predictor tends to lag at larger forecast horizons, limiting its
#     usefulness for multi-quarter-ahead forecasting.
#   - Aggressive (large f_excess): harder to interpret and less robust to model
#     misspecification, but produces a stronger lead that enables effective tracking
#     of forward-shifted BIP at larger shift values (one year or more).
#
# Advantage of the M-SSA framework: 
#   M-SSA allows the practitioner to actively manage various trade-offs by tuning design 
#   parameters (e.g., f_excess, holding time) explicitly, rather than accepting a fixed balance 
#   imposed by the model.

# ============================================================
# References
# ============================================================
#
# Wildi, M. (2026b). The Accuracy-Smoothness Dilemma in Prediction:
#   A Novel Multivariate M-SSA Forecast Approach.
#   Journal of Time Series Analysis, http://doi.org/10.1111/jtsa.70058 
#   arXiv: https://doi.org/10.48550/arXiv.2602.13722
#
# Heinisch, K., Van Norden, S., and Wildi, M. (2026).
#   Smooth and Persistent Forecasts of German GDP:
#   Balancing Accuracy and Stability.
#   IWH Discussion Papers, 1/2026.
#   Halle Institute for Economic Research.
#   https://doi.org/10.18717/dp99kr-7336
#
# ============================================================
######################################################################################
# Concluding Remark:
#   The M-SSA framework presented here relies on a deliberately simple VAR(1)
#   specification. While this parsimonious model already delivers meaningful
#   forecasting performance, it leaves room for further refinement.
#   Tutorial 7.5 introduces a more sophisticated BVAR(3) model — a Bayesian
#   VAR with three lags — which incorporates prior information to regularize
#   parameter estimation and better capture the dynamic interdependencies
#   among the indicator series. The richer BVAR(3) specification is shown
#   to yield further improvements in out-of-sample forecasting performance
#   relative to the VAR(1) baseline established here.
#######################################################################################

# Start with a clean workspace
rm(list = ls())

# --- Load Required R Libraries ---

# HP and other standard time series filters
library(mFilter)

# Multivariate time series: VARMA estimation for macro indicator dynamics
# Used here primarily for VAR-based simulation and spectrum computation
library(MTS)

# HAC-consistent standard errors (Newey-West) for inference under
# heteroscedasticity and autocorrelation
library(sandwich)

# Extended time series objects (xts): flexible date-indexed data handling
library(xts)

# Diebold-Mariano test for equal predictive accuracy across competing forecasts
library(multDM)

# Ridge regression: regularized OLS for the optimal weighting step
library(MASS)

# LASSO and elastic net regression: sparse regularization alternative
# for the optimal weighting step
library(glmnet)

# ROC and AUC
library(pROC)



# ==================================================================
# Load Custom M-SSA Function Libraries
# ==================================================================

# Core M-SSA filter construction and optimization routines
source(paste(getwd(), "/R/functions_MSSA.r", sep = ""))

# ROC plots
source(paste(getwd(), "/R/ROCplots.r", sep = ""))

# HP filter utilities used in the JBCY paper (relies on mFilter)
source(paste(getwd(), "/R/HP_JBCY_functions.r", sep = ""))

# M-SSA utility functions: data preparation, plotting helpers, wrappers
source(paste(getwd(), "/R/M_SSA_utility_functions.r", sep = ""))


# ==================================================================
# Load Data and Select Indicators
# ==================================================================
# See Tutorials 7.2 and 7.3 for a detailed discussion of the data,
# indicator selection, and publication lag assumptions.

load(file = paste(getwd(), "\\Data\\macro", sep = ""))

# Publication lag assumption:
#   - In practice, BIP is published with approximately a 1-quarter delay.
#   - Tutorial 7.3 adopted a more conservative 2-quarter lag as a pragmatic
#     buffer against data revision effects that were not explicitly modelled.
#   - Based on consultation with domain experts, the 1-quarter lag is deemed
#     sufficiently accurate for the present analysis, and we revert to this
#     more realistic assumption here.
#   - All other indicators (ip, ifo_c, ESI, spr_10y_3m) are assumed to be
#     available contemporaneously, i.e., without any publication delay.
lag_vec <- c(1, rep(0, ncol(data) - 1))

# ------------------------------------------------------------------
# Plot all indicators (excluding BIP from the first column of data)
# ------------------------------------------------------------------
# Note: BIP (red) and industrial production ip (orange) appear right-shifted
# relative to the other indicators due to their publication lags:
#   - BIP: 1-quarter publication lag
#   - ip:  approximately 2-month publication lag
#     (visible more clearly in the monthly data; see data_monthly)
par(mfrow = c(1, 1))
mplot <- data[, -1]
colo  <- rainbow(ncol(data) - 1)
main_title <- "Quarterly macro indicators"

# Base plot: first indicator in the selection
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     lwd = c(2, rep(1, ncol(data) - 1)),
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay all remaining indicators
for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
     labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
axis(2)
box()

# ------------------------------------------------------------------
# Select macro indicators for M-SSA
# ------------------------------------------------------------------
# Five key macro-financial indicators (same selection as Tutorial 7.3):
#   BIP       : German GDP growth (log-differenced, quarterly)
#   ip        : Industrial production growth
#   ifo_c     : ifo Business Climate Index (current conditions: ifo_e(xpectations) does not improve performances) 
#   ESI       : Economic Sentiment Indicator (European Commission)
#   spr_10y_3m: Term spread (10-year minus 3-month government bond yield)
select_vec_multi <- c("BIP", "ip", "ifo_c", "ESI", "spr_10y_3m")
x_mat <- data[, select_vec_multi]
rownames(x_mat) <- rownames(data)

# Number of series and total observations
n   <- dim(x_mat)[2]
len <- dim(x_mat)[1]

# ------------------------------------------------------------
#
# Target Signal:
#   - The prediction target is the output of a two-sided
#     Hodrick-Prescott (HP) filter applied to BIP.
#   - HP is not necessarily the optimal signal design for this application;
#     alternative targets could be substituted (see Tutorials 3, 4, and 6).
#   - The key point is that M-SSA accepts any user-specified target filter
#     and derives the corresponding optimal causal predictor, given the DGP.
#
# HP Parameter Choice:
#   - The standard quarterly HP specification uses lambda = 1600.
#   - Here we use lambda = 160, a less aggressive smoothing parameter.
#   - Rationale: HP(1600) removes too much economically relevant variation
#     for BIP prediction purposes (see Tutorials 7.2 and 7.3 for a full
#     discussion; see also Phillips and Jin (2021) for a related critique).
lambda_HP <- 160

# Filter length:
#   - L = 31 corresponds to roughly 4 years of quarterly lags (one-sided).
#   - Technical note: only the right half of the two-sided filter is passed
#     to M-SSA; the full symmetric filter is reconstructed internally by
#     "mirroring" (reflecting) the right half.
#   - L must be odd so that the mirrored reconstruction yields a perfectly
#     symmetric filter with a single peak at its center.
L <- 31

# Compute the HP filter object using the mFilter-based wrapper:
#   - hp_symmetric        : truncated symmetric (two-sided) HP filter of length L
#   - hp_classic_concurrent : standard one-sided (concurrent) HP filter
#   - hp_one_sided        : right half of the two-sided HP filter (length L),
HP_obj                <- HP_target_mse_modified_gap(L, lambda_HP)
hp_symmetric          <- HP_obj$target
hp_classic_concurrent <- HP_obj$hp_trend
hp_one_sided          <- HP_obj$hp_mse
hp_two                <-c(hp_one_sided[L:2],hp_one_sided) # Specify two-sided acausal HP



  
  
# ==================================================================
# Exercise 1: Working with M-SSA Sub-Components
# ==================================================================
# Background:
#   - The aggregate M-SSA predictor from Tutorial 7.3 is the equally-weighted
#     average of individual standardized M-SSA component outputs, one per indicator series.
#   - This exercise investigates whether individual components or re-weighted
#     combinations of components can improve forecast performance relative
#     to the equally-weighted aggregate, particularly at longer horizons.
#   - The design (hyperparameters) follows Tutorial 7.3, Exercise 1 exactly,
#     to ensure a fair comparison (accounting for the smaller publication lag here).


# =================================================================
# Exercise 1.0: Replicate the Original M-SSA Predictor (Tutorial 7.3)
# =================================================================
# This serves as the baseline benchmark against which all component-based
# designs proposed in this tutorial will be evaluated.

# HP smoothing parameter: moderately adaptive filter targeting mid-term
# trend growth. See Tutorial 7.3 for a full discussion of lambda_HP.
lambda_HP <- 160

# Filter length: L = 31 quarters (≈ 8 years) is sufficient for lambda_HP = 160
# because HP filter weights decay fast enough at this smoothing level.
# Must be an odd number to ensure filter symmetry (see Tutorial 7.1).
L <- 31

# In-sample end date for VAR estimation.
# The design is relatively insensitive to this choice due to the
# parsimonious VAR parameterization (p = 1, q = 0).
# Note: shorter samples lead to numerical instability (errors): 2008 is 
# at the limit of feasibility of the MTS package
date_to_fit <- "2008"

# VAR model orders: minimal AR(1) specification to avoid overfitting
p <- 1
q <- 0

# HT constraints: calibrated in Tutorial 7.2; approximately twice the
# MSE-optimal benchmark values, yielding fewer than half as many
# zero-crossings as the unconstrained M-MSE predictor.
# See Tutorial 7.3.
ht_mssa_vec <- c(6.380160, 6.738270, 7.232453, 7.225927, 7.033768)
names(ht_mssa_vec) <- colnames(x_mat)

# Forecast horizons: M-SSA is separately optimized for each h in h_vec.
# h = 0: nowcast; h = 6: six-quarter-ahead forecast.
h_vec <- 0:6

# Forecast excess: a slightly larger excess (5) is applied to BIP itself
# (first element) relative to the other indicators (4), reflecting the
# lag of BIP and the need for additional anticipation.
# See Tutorial 7.2, Exercise 2 for background on forecast excess.
# A NOTE OF CAUTION:
#   - The chosen values for f_excess represent a relatively aggressive setting: the filter is
#     effectively forced to "look ahead" beyond the nominal forecast horizon.
#   - The multivariate design facilitates this by exploiting lead-lag relationships
#     between indicators; it also has greater freedom to manipulate the filter phase
#     in order to reach further into the future.
#   - This forced "phase-playing" (manipulation) is theoretically justified when the true model is
#     known, since the filter can then anticipate future dynamics in an optimal sense.
#   - Under the unavoidable model misspecification present in practice, however,
#     overly aggressive forward-looking designs may produce unexpected or
#     unstable forecast behaviour and should therefore be applied WITH CAUTION.
# In short: these settings probe the limits of what the M-SSA predictor can achieve
#           when forecasting German GDP multiple quarters ahead.
# Conversely, selecting a small f_excess (near zero) may result in a negative correlation
#   between the M-SSA predictor and forward-shifted BIP at larger shift values
#   (one year or more). This is because M-SSA is designed to predict HP-BIP, the
#   smooth low-frequency trend component of BIP, rather than BIP directly. Without
#   a sufficient forecast excess, the filter cannot bridge the gap between the
#   smooth target it is optimised for and the `faster evolving' (noisier) BIP series it 
#   is ultimately evaluated against in this tutorial.
f_excess <- c(5, rep(4, length(select_vec_multi) - 1))
f_excess

# ------------------------------------------------------------------
# Run the M-SSA wrapper function (documented in Tutorial 7.2)
# Computes M-SSA predictors for each forecast horizon h in h_vec
# ------------------------------------------------------------------
mssa_indicator_obj <- compute_mssa_BIP_predictors_func(
  x_mat, lambda_HP, L, date_to_fit, p, q,
  ht_mssa_vec, h_vec, f_excess, lag_vec, select_vec_multi
)


# ------------------------------------------------------------------
# Retrieve outputs from the wrapper
# ------------------------------------------------------------------

# Forward-shifted HP-BIP targets (one column per horizon in h_vec):
# the two-sided HP filter applied to BIP, shifted forward by
# (forecast horizon + publication lag) quarters.
# This is the explicit optimization target of the original M-SSA design.
target_shifted_mat <- mssa_indicator_obj$target_shifted_mat

# Aggregate M-SSA predictor (one column per horizon in h_vec):
# equally-weighted average of all standardized M-SSA component outputs across
# the five selected indicator series.
predictor_mssa_mat <- mssa_indicator_obj$predictor_mssa_mat
tail(predictor_mssa_mat)

# M-SSA component array (3-dimensional: [series, time, horizon]):
#   - For each forecast horizon h and each indicator series, this array
#     contains the M-SSA output when the two-sided HP filter is applied
#     to that specific indicator as the local optimization target.
#   - The aggregate predictor_mssa_mat is the row-wise mean of this array
#     across the series dimension.
#   - Individual components are analyzed in Exercise 1.1 below.
mssa_array <- mssa_indicator_obj$mssa_array
# Inspect the most recent M-SSA outputs for two selected indicator series
tail(mssa_array["BIP",,])
tail(mssa_array["ifo_c",,])
# Averaging the individual M-SSA components across all indicator series
# yields the aggregate predictor_mssa_mat:
tail(predictor_mssa_mat)
# In the absence of strong prior information favouring any single indicator,
# equal-weight averaging is a robust and well-established strategy for
# combining predictors. However, rather than imposing equal weights a priori,
# one could instead derive 'optimal' weights via linear regression — allowing
# the data to determine each indicator's relative contribution to forecasting
# the target variable. This regression-based weighting scheme is the central
# methodological topic of the present tutorial.

# M-MSE component array (same structure as mssa_array):
#   - Same as mssa_array but without the HT smoothness constraint:
#     classic multivariate MSE-optimal signal extraction.
#   - Used as a reference benchmark in Exercise 5.
#   - This is noisier than M-SSA
mmse_array <- mssa_indicator_obj$mmse_array
tail(mmse_array["BIP",,])

# ------------------------------------------------------------------
# Compute performance metrics for the original M-SSA predictor
# These serve as the baseline benchmark throughout this tutorial.
# ------------------------------------------------------------------

# Out-of-sample start date: placing the entire 2008 financial crisis
# and subsequent period in the out-of-sample evaluation window
# Note: the VAR is based on data up 2008 (shorter samples trigger errors) 
in_out_separator <- "2007"

# Direct forecast indicator selection (same as Tutorial 7.3):
# results are not the focus of this tutorial but a selection is required
# by compute_perf_func
select_direct_indicator <- c("ifo_c", "ESI")
# Select all indicators: possible overfitting issues
select_direct_indicator <- colnames(x_mat)
# Equally weighted combination
predictor<-predictor_mssa_mat

perf_obj <- compute_perf_func(
  x_mat, target_shifted_mat, predictor,
  predictor_mmse_mat, in_out_separator, select_direct_indicator, h_vec
)

# ------------------------------------------------------------------
# Extract the key performance metric for downstream comparisons:
# HAC-adjusted p-values from regressions of the out-of-sample M-SSA
# predictor on forward-shifted BIP (see Tutorial 7.3, Exercise 1.2.2).
# Note: publication lag is 1 quarter here.
#
# Matrix structure:
#   - Rows   (i): forward-shift of BIP target = h_vec[i] + publication lag
#   - Columns (j): M-SSA predictor optimized for forecast horizon h_vec[j]
#   - Small p-values on or near the diagonal indicate that the predictor
#     optimized for horizon h performs best at the matching forward-shift.
# ------------------------------------------------------------------
# Targeting forward-shifted HP-GDP:
#   p_value_HAC_HP_BIP_oos are the HAC-adjusted p-values obtained from a regression of 
#   the smooth HP-BIP, shifted forward by shift=0,...,5, on the predictors in predictor_mssa_mat,
#   optimized for h=0,...,6 (out-of-sample). 
p_value_HAC_HP_BIP_oos <- perf_obj$p_value_HAC_HP_BIP_oos
round(p_value_HAC_HP_BIP_oos, 3)
#   Result: out-of-sample P-values are smallest close to the diagonal (h = shift), confirming a tight
#   alignment between the optimization horizon and the evaluation forward shift.


# Targeting forward-shifted raw GDP:
#   The diagonal pattern is less sharp due to higher noise in the unfiltered target.
#   However, for shift >= 2, p-values tend to decrease from left to right (increasing h),
#   suggesting that M-SSA filters optimized for longer horizons track forward-shifted
#   BIP more effectively than shorter-horizon designs.
p_value_HAC_BIP_oos <- perf_obj$p_value_HAC_BIP_oos
round(p_value_HAC_BIP_oos, 3)

# ------------------------------------------------------------------
# Illustrative example: one-year-ahead forecast significance
# ------------------------------------------------------------------
# Row i = 5: BIP shifted forward by h_vec[5] + publication lag (effective 1-year forecast)
i <- 5
# Column j = 7: M-SSA predictor optimized for h = h_vec[7] = 6 quarters
j <- 7
p_value_HAC_BIP_oos[i, j]
# Result: the original equally-weighted M-SSA predictor optimized for h = 6
# shows evidence of predictability of BIP four quarters ahead (including publication lag).
# Exercise 1.1 below shows that a component-based re-weighting improves this.

# ------------------------------------------------------------------
# Technical note: HP-BIP vs. raw BIP as the regression target
# ------------------------------------------------------------------
# The M-SSA predictor tracks forward-shifted HP-BIP (smooth trend growth)
# more tightly than forward-shifted raw BIP (noisy), as expected given that
# HP-BIP is its explicit optimization target.
perf_obj$p_value_HAC_HP_BIP_oos[i, j]
# Result: the same predictor is strongly significant when predicting
# HP-BIP four quarters ahead — confirming that M-SSA successfully
# captures mid-term trend growth, even if raw BIP significance is weaker.

# ------------------------------------------------------------------
# Key design distinction for this tutorial:
# ------------------------------------------------------------------
# - The original M-SSA predictor (Tutorial 7.3) targets HP-BIP:
#   it is optimized to track smooth mid-term trend growth.
# - The newly proposed M-SSA component predictor (this tutorial)
#   re-weights the individual component outputs to target raw BIP
#   more directly, accepting some loss of smoothness in exchange
#   for improved raw BIP predictability at longer horizons.


# ==================================================================
# Exercise 1.1: What Are M-SSA Sub-Components?
# ==================================================================
#
# Background:
#   - The original M-SSA predictor (tutorial 7.3) is constructed as the equally-weighted
#     cross-sectional mean of the standardized M-SSA outputs across all input indicators:
#     BIP, ip, ifo, ESI, and spread
#   - Each output tracks the two-sided HP filter applied to its respective indicator,
#     using all indicators jointly as explanatory variables (multivariate design)
#   - Here we briefly replicate the aggregate predictor from its components, and
#     verify that equal weighting exactly recovers the original M-SSA predictor
# =================================================================

# Select forecast horizon for illustration (j_now = 1 corresponds to the nowcast)
j_now <- 1
h_vec[j_now]   # Confirm the selected forecast horizon

# Inspect the M-SSA sub-components for the selected horizon (most recent observations)
tail(t(mssa_array[,, j_now]))

# --- Plot M-SSA Sub-Components ---
# Each column of mssa_array[,,j_now] is the real-time (causal) M-SSA output for one indicator,
# optimized to track the corresponding acausal two-sided HP target:
#   - HP-BIP (shifted forward by h + publication lag)
#   - HP-ip, HP-ifo, HP-ESI, HP-spread (shifted forward by h)
# All five outputs share the same set of explanatory variables (BIP, ip, ifo, ESI, spread),
# making this a genuinely multivariate filtering exercise

mplot <- t(mssa_array[,, j_now])
colo  <- rainbow(length(select_vec_multi))

main_title <- paste0("M-SSA sub-components: causal outputs tracking two-sided HP targets ",
                     "(h = ", h_vec[j_now], ")")
par(mfrow = c(1, 1))
plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], lwd = 1, lty = 2,
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 2)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

abline(h = 0)
abline(v = which(rownames(mplot) <= date_to_fit)[length(which(rownames(mplot) <= date_to_fit))],
       lwd = 2, lty = 2)
axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
     labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
axis(2)
box()

# --- Replicate the Original M-SSA Predictor from Its Components ---
# The aggregate M-SSA predictor is the cross-sectional mean of the standardized sub-components
agg_std_comp <- apply(scale(t(mssa_array[,, j_now])), 1, mean)

# Numerical verification: maximum absolute deviation from the original predictor
# Should be zero (or negligibly small) up to floating-point precision
max(abs(agg_std_comp - predictor_mssa_mat[, j_now]), na.rm = T)

# Plot: cross-sectional mean of standardized components vs. original M-SSA predictor
mplot <- cbind(agg_std_comp, predictor_mssa_mat[, j_now])
rownames(mplot) <- rownames(x_mat)
colnames(mplot) <- c(
  "Cross-sectional mean of standardized sub-components",
  "Original M-SSA predictor (tutorial 7.3)"
)
colo       <- c("blue", rainbow(length(select_vec_multi)))
main_title <- paste0("Replication of M-SSA predictor from equally-weighted components (h = ",
                     h_vec[j_now], ")")
par(mfrow = c(1, 1))
plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], lwd = 2,
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 2)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
abline(v = which(rownames(mplot) <= date_to_fit)[length(which(rownames(mplot) <= date_to_fit))],
       lwd = 2, lty = 2)
axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
     labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
axis(2)
box()
# Both series overlap exactly, confirming the replication

# Remarks on equal weighting:
#   - Equal weighting treats each M-SSA component as equally informative for tracking
#     dynamic changes in BIP growth — a simple and robust assumption, but potentially suboptimal
#   - A natural alternative is to replace equal weighting with regression-based optimal weights,
#     obtained by regressing the components on forward-shifted BIP
#   - This would explicitly target BIP-MSE forecast performance rather than HP-BIP tracking
#   - The resulting 'M-SSA components predictor' is derived in exercise 1.3 below
#   - First, however, we demonstrate a complementary use of the components: interpretability

# =================================================================
# Exercise 1.2: Interpretability — Gauging the M-SSA Predictor via Its Components
# =================================================================
# Motivation:
#   - The individual M-SSA sub-components reveal which indicators are driving the
#     current dynamics of the aggregate predictor
#   - When components agree (all point in the same direction), the aggregate signal
#     is more reliable; when they diverge, greater caution is warranted
#   - This provides a practical tool for assessing the trustworthiness of any given
#     forecast or nowcast signal before acting on it
# =================================================================
# Illustration: M-SSA nowcast (h = h_vec[j_now]) and its sub-components

par(mfrow = c(1, 1))

# Standardize the aggregate predictor and all sub-components for visual comparability
mplot <- scale(cbind(predictor_mssa_mat[, j_now], scale(t(mssa_array[,, j_now]))))
rownames(mplot) <- rownames(x_mat)
colnames(mplot) <- c(
  paste0("M-SSA predictor (h = ", h_vec[j_now], ")"),
  paste0("Component: ", select_vec_multi)
)
colo       <- c("blue", rainbow(length(select_vec_multi)))
main_title <- c(
  paste0("M-SSA nowcast (solid blue) and sub-components (dashed) — h = ", h_vec[j_now]),
  "Vertical dashed line marks end of in-sample span"
)
plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], lwd = 2,
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 2)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
abline(v = which(rownames(mplot) <= date_to_fit)[length(which(rownames(mplot) <= date_to_fit))],
       lwd = 2, lty = 2)
axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
     labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
axis(2)
box()

# =================================================================
# --- Interpretation (as of January 2025 data) ---
#
# Trough identification:
#   - All five sub-components consistently date the most recent trough in German GDP
#     growth to late 2023 — a robust, cross-indicator signal
#
# Current dynamics (Jan 2025):
#   - The aggregate nowcast has just crossed above zero (long-run average growth),
#     tentatively suggesting the trough in BIP levels may have passed
#   - However, the sub-components tell a more nuanced story:
#
#     Spread (violet dashed):  strongest upward signal; historically a leading indicator,
#                               though its reliability has been questioned in recent cycles
#     ifo / ESI:               marginally above zero; consistent with stabilization
#                               but not yet signalling a robust recovery
#     ip / BIP:                still at or below zero; awaiting further confirmation
#                               before aligning with the leading indicators
#
# Implication for forecast reliability:
#   - The divergence across components signals uncertainty: the recovery signal
#     is currently driven primarily by financial/survey indicators, while hard
#     activity data (ip, BIP) have yet to confirm
#   - Users should treat the positive nowcast with appropriate caution until
#     broader component agreement emerges
#
# Caveat:
#   - Announced and/or unexpected structural shocks (trade tariffs, geopolitical
#     disruptions) are not yet reflected in the Jan 2025 data vintage and could
#     materially alter the outlook

# ==================================================================
# Exercise 1.3: Targeting Raw BIP Directly via M-SSA Components
# ==================================================================
#
# Motivation:
#   - The original M-SSA predictor (Tutorial 7.3) is either standardized (default) 
#     or scaled to track centered HP-BIP (smooth trend growth). In any case, 
#     its level and scale are not calibrated against raw BIP, so MSE-based 
#     performance against raw BIP is not its primary strength.
#   - To predict future raw BIP explicitly and evaluate MSE performance,
#     we replace the equal-weighting scheme with a regression-based
#     combination of selected M-SSA components on forward-shifted BIP.
#   - This mirrors the direct forecast approach from Tutorial 7.3
#     (Exercise 1.2.1), but uses M-SSA-filtered components as regressors
#     rather than the raw unfiltered indicators. The M-SSA filtering
#     pre-processes the indicators to emphasize the relevant signal at
#     the target forecast horizon before the regression is run.


# =================================================================
# 1.3.1 Component Selection
# =================================================================
# Not all M-SSA components are equally useful for MSE-based BIP prediction 
#   in particular in short in-sample spans (~40 observations when accounting 
#   for filter initialization `burn-in'):
#
#   - ESI, ifo_c, and spread components: primarily informative in a
#     dynamic/directional context (recessions, turning points). They are
#     the main drivers of the original equally-weighted M-SSA predictor
#     but are less suited as direct regressors for raw BIP.
#
#   - BIP and ip components: natural candidates for MSE-based BIP
#     prediction, since they directly filter the two most closely related
#     series. Note that ESI, ifo_c and spread remain indirectly informative
#     through their role in the VAR (and hence M-SSA) used to construct these components
#     (see Tutorial 7.2, Exercise 1).
#
# Available design choices (from simplest to most complex):
#
#   sel_vec_pred <- "BIP"          (recommended default in short samples)
#     + Simplest and most interpretable design
#     + Reasonably strong out-of-sample performance
#     + Robust across time periods and data vintages
#     + Small revisions when new data arrive
#
#   sel_vec_pred <- c("BIP", "ip") (more aggressive alternative)
#     + Can improve MSE performance marginally
#     - Less robust than BIP alone in short samples
#     - Larger revisions
#     - ip regression coefficient is negative, producing stronger
#       left-shifts that are harder to interpret economically
#
# We adopt the simplest design as the baseline:
sel_vec_pred <- "BIP"
sel_vec_pred

# Regression of M-SSA BIP on (single) forward-shifted BIP serves two calibration purposes:
#
#   - Level (intercept):
#       Re-anchor M-SSA BIP to the empirical mean of forward-shifted BIP
#
#   - Scale (slope):
#       Re-scale to match the empirical variance of forward-shifted BIP
#
# Implications for performance metrics:
#
#   - MSE/rRMSE: directly affected by the calibration, since MSE is sensitive to both
#       level and scale mismatches between predictor and target
#
#   - Target correlations and HAC-adjusted statistics: less affected (correlation
#       and t-statistics are invariant to affine transformations of the predictor)

# ------------------------------------------------------------------
# Illustrative example: regression setup for a specific horizon
# ------------------------------------------------------------------

# Forward-shift of BIP target (quarters ahead, excluding publication lag):
# shift = 2 corresponds to a 2-quarter-ahead forecast.
# Below, all shifts from 0 to 5 quarters are analyzed systematically.
shift <- 2

# M-SSA component horizon index (indexes into h_vec):
# k = 5 corresponds to h_vec[5] = 4 quarters (one-year-ahead M-SSA component).
# Below, all (shift, horizon) combinations are evaluated.
k <- 5
h_vec[k] # Confirm: should be 4

# ------------------------------------------------------------------
# Construct the regression data matrix:
#   - Column 1: BIP shifted forward by (shift + publication lag) quarters
#   - Columns 2+: selected M-SSA components for horizon h_vec[k]
# Note: ifo_c, ESI and spread are important indirect inputs via the VAR,
# even though they do not appear explicitly as regressors here.
# ------------------------------------------------------------------
dat <- cbind(
  c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
  matrix(t(mssa_array[sel_vec_pred, , k]), ncol = length(sel_vec_pred))
)
rownames(dat) <- rownames(x_mat)
colnames(dat) <- c(
  paste("BIP shifted forward by lag_vec + shift = ", shift + lag_vec[1], sep = ""),
  paste("M-SSA component ", sel_vec_pred, ": h=", h_vec[k], sep = "")
)
tail(dat)

# Remove rows with NAs (arising from the forward-shift and filter boundaries)
# before fitting the regression
dat <- na.exclude(dat)


# ==================================================================
# 1.3.2 Regression: M-SSA Components on Forward-Shifted BIP
# ==================================================================
# We regress forward-shifted BIP (column 1) on the selected M-SSA
# components (columns 2+).
#
# In-sample span: for illustration we use data up to end of 2010.
# In Exercise 1.3.3 below, we extend this to a full expanding-window
# scheme starting in Q1-2008 and ending in Q4-2025.

i_time <- which(rownames(dat) > 2010)[1]
  
# Display the last few rows of the in-sample span as a sanity check
tail(dat[1:i_time, ])

# Fit the regression
lm_obj <- lm(dat[1:i_time, 1] ~ dat[1:i_time, 2:ncol(dat)])
summary(lm_obj)
# The M-SSA component is strongly statistically significant (p=0.00469).
# HAC-adjusted standard errors would not overturn this conclusion,
# given the magnitude of the t-statistics.

# ------------------------------------------------------------------
# Out-of-sample prediction for time point (i_time + shift + lag_vec[1])
# ------------------------------------------------------------------

# Alignment note:
#   Due to the GDP publication lag and the forward shift of the target series, the
#   out-of-sample prediction at each iteration is made for time point:
#       i_time + shift + lag_vec[1]
#
# Concrete example (forward-shift = 4 quarters, publication lag = 1 quarter):
#   - Forecast date:       Q1-2024  (the point at which the prediction is made)
#   - Forecast target:     Q1-2025  (one year ahead, i.e., shift = 4 quarters forward)
#   - Last available GDP:  Q4-2023  (one quarter publication lag)
#   - Last usable in-sample regression equation:
#                          Q4-2022  (because the regression target is forward-shifted by
#                                    shift = 4 quarters, the last equation with a realized
#                                    target is dated 4 quarters before Q4-2023)
#   - Total lag between forecast date and last usable regression equation:
#                          5 quarters = shift (4) + publication lag (1)
#   - Total lag between observation of forecast error (Q2-2025 due to publication lag) 
#     and last usable in-sample regression equation: 
#                         10 quarters = 2*shift (8) + 2* publication lag (2)
#
# Out-of-sample predictor for time point i_time + shift + lag_vec[1]
oos_pred <- lm_obj$coef[1] +
  lm_obj$coef[2:ncol(dat)] %*% dat[i_time + shift + lag_vec[1], 2:ncol(dat)]

# Out-of-sample forecast error at the target time point
#   Note that the first column dat[,1] is BIP forward-shifted by shift+lag_vec[1]. 
#   So dat[i_time + shift + lag_vec[1], 1] represents BIP forward-shifted by
#   2*(shift + lag_vec[1]) with respect to the last available in-sample regression equation
oos_error <- dat[i_time + shift + lag_vec[1], 1] - oos_pred
# This error will be observed 2*(shift + lag_vec[1]) quarters after 
# the last available in-sample regression equation in t=i_time
oos_error

# ------------------------------------------------------------------
# Next step: extend the above single-period evaluation to a full
# expanding-window out-of-sample analysis in Exercise 1.3.3 below.
# ------------------------------------------------------------------


# ==================================================================
# 1.3.3 Out-of-Sample Performance: Expanding-Window Evaluation (post-2007)
# ==================================================================
#
# The evaluation span begins in 2008, ensuring that the entire financial crisis
# falls within the out-of-sample window — a demanding test of forecast robustness.
#
# Note on sample length:
#   - The in-sample window is short at the start (compounded by filter initialization losses)
#   - MSE performance is therefore expected to be weaker early in the evaluation span
#     and more representative towards the sample end, as the estimation window grows

# --- Regression Settings ---

# In-sample / out-of-sample split (carried over from exercise 1.2)
in_out_separator <- in_out_separator

# Sample alignment:
#   - M-SSA loses the first L observations due to filter initialization
#   - Setting align_sample = T removes the same observations from the mean and direct forecast
#     benchmarks, ensuring all methods are evaluated on an identical evalution sample
align_sample <- T

# Regression type for the optimal weighting step:
#   - OLS:   unpenalized; may overfit when the number of components is large
#   - Ridge: L2 penalty; shrinks all coefficients uniformly towards zero
#   - LASSO: L1 penalty; performs implicit variable selection (sparse solution)
reg_type <- "OLS"
reg_type <- "LASSO"
reg_type <- "Ridge"

# Regularization penalty (applies to Ridge and LASSO only)
# Larger values impose stronger shrinkage towards zero
lambda_reg <- 10

# --- Run Expanding-Window Performance Evaluation ---
# For each time point after in_out_separator, the function:
#   1. Runs a OLS/Ridge/LASSO regression of M-SSA components on forward-shifted BIP
#   2. Computes the out-of-sample forecast and records the forecast error
perf_obj <- optimal_weight_predictor_func(
  dat, in_out_separator, F, shift, lag_vec, align_sample, reg_type, lambda_reg
)

# --- Retrieve and Inspect Performance Metrics ---

# Out-of-sample forecast errors: M-SSA components predictor
# (shorter than the full data due to publication lag and forward-shift of BIP)
tail(perf_obj$epsilon_oos)

# Out-of-sample forecast errors of naive mean benchmark
tail(perf_obj$epsilon_mean_oos)

# HAC-adjusted p-value: regression of out-of-sample M-SSA components predictor on forward-shifted BIP
# Small values indicate statistically significant predictive content
perf_obj$p_value

# Same, but excluding Pandemic observations
# Comparing the two reveals the extent to which significance is driven by the Pandemic episode
perf_obj$p_value_without_covid

# Out-of-sample MSE: M-SSA components predictor
perf_obj$MSE_oos

# Same, excluding Pandemic: quantifies the Pandemic's contribution to overall MSE
perf_obj$MSE_oos_without_covid

# Out-of-sample MSE: naive mean benchmark (expanding window)
# Expected to be slightly larger than M-SSA components MSE, at least for shorter forward-shifts
# (the mean becomes increasingly hard to beat at longer horizons)
perf_obj$MSE_mean_oos
perf_obj$MSE_mean_oos_without_covid

# Relative Root MSE (rRMSE): M-SSA components predictor vs. mean benchmark
# Values below 1 indicate M-SSA outperforms the mean
sqrt(perf_obj$MSE_oos / perf_obj$MSE_mean_oos)

# Same, excluding Pandemic: 
#   COVID tends to distort efficiency gains (underestimation of effective gains)
sqrt(perf_obj$MSE_oos_without_covid / perf_obj$MSE_mean_oos_without_covid)

#-----------------------------------------------------------------------------------
# Technical Note: Performance Metrics
#
# 1. HAC-Adjusted P-values, e.g., perf_obj$p_value
#
#   Procedure:
#     - The out-of-sample predictor is regressed on the forward-shifted target (BIP or HP-BIP)
#       over the entire out-of-sample span in a single full-sample regression.
#     - This regression implicitly calibrates the (standardized or HP-BIP scaled) predictor to the level and
#       scale of the target series before assessing predictive content.
#
#   Interpretation:
#     - After level and scale calibration, the HAC-adjusted p-value measures how much of
#       the dynamic variation in the target is explained by the predictor.
#     - A small p-value indicates that the predictor captures the temporal pattern of the
#       target significantly better than chance.
#
#   Consistency with M-SSA design:
#     - This metric is well-aligned with the M-SSA optimization criterion, which deliberately
#       abstracts from `static' level and scale calibration:
#         (i)  M-SSA maximizes target correlation — equivalently, sign accuracy — subject
#              to a sign-change rate constraint (holding time).
#         (ii) M-SSA emphasizes dynamic properties (up/down swings) of the predictor
#              and treats level and scale as static nuisance parameters to be handled separately 
#              (e.g., through regression).
#
#
# 2. MSE and rRMSE, e.g., perf_obj$MSE_oos
#
#   Procedure:
#     - Unlike the HAC p-value approach, level and scale calibration of M-SSA is performed in a
#       truly out-of-sample fashion: regression coefficients are re-estimated recursively,
#       using only data available at each point in time, and forecast errors are computed
#       on the held-out observation (to be predicted).
#     - The MSE is then derived from the `true' out-of-sample MSE.
#
#   Interpretation:
#     - MSE and rRMSE jointly account for both dynamic accuracy (up/down swings) and
#       static accuracy (level and scale calibration), providing a complete
#       assessment of out-of-sample forecast performance.
#
#   Scope:
#     - This procedure extends the M-SSA forecasting framework to settings where level and
#       scale form an integral part of the forecast task — for example, when forecasting
#       effective BIP growth rates rather than a standardized signal.
#
#   Application:
#     - Researchers focused on the dynamic pattern of BIP — i.e., the directional signal
#       (expansion vs. contraction phases, turning points, cyclical swings) — may regard
#       HAC-adjusted p-values as the more relevant performance metric, since these directly
#       measure the predictor's ability to track the temporal dynamics of the target,
#       independently of level and scale.
#     - Researchers focused on effective BIP growth forecasting — where accurate level and
#       scale calibration are integral to the forecast task — will find MSE and rRMSE to be
#       the more appropriate metrics, as these jointly penalize both dynamic and static
#       forecast errors.
#-----------------------------------------------------------------------------------

# ==================================================================
# 1.3.4 Full Performance Matrix: 
#       All Combinations of Forward-Shift and Forecast Horizon
# ==================================================================
# We now compute the above performance metrics exhaustively across all combinations of:
#   - shift_vec: forward shifts of the GDP target (rows of the performance matrices)
#   - h_vec:     M-SSA forecast horizons, i.e., the horizon for which each filter is
#                optimized in-sample (columns of the performance matrices)
#
# This yields a (6 x 7) performance matrix for each metric, providing a comprehensive
# view of the horizon-shift interaction pattern across the full grid of design choices.
#
# Consistency check:
#   By construction, an M-SSA filter optimized in-sample for horizon h should deliver
#   its best out-of-sample forecast performance when predicting HP-BIP (and eventually BIP) at a
#   forward shift of shift = h (or shift ≈ h for near-diagonal entries).
#   This alignment between the optimization horizon and the evaluation shift defines
#   the expected diagonal pattern in the performance matrices, and serves as an
#   internal out-of-sample validation of the M-SSA design principle.

# Exercise 1.0 above demonstrated that a clear diagonal pattern emerges in the
# (equally-weighted) M-SSA output when the smooth HP-filtered BIP serves as the optimization
# target. However, this pattern becomes obscured when the raw (unfiltered)
# BIP is used as the target instead, as the noise present in the unfiltered
# series contaminates the M-SSA output and blurs the underlying structure.


# --- Specifications ---

# M-SSA component used as predictor: BIP sub-component only.
# Restricting the predictor to the BIP-based M-SSA component reduces the
# risk of overfitting in the subsequent optimal weighting step, which is
# a relevant concern given the relatively short in-sample estimation span 
# (~40 observations after filter initialization).
# In Exercise 6, the full set of M-SSA components across all indicator
# series is incorporated, where the longer full-sample period provides
# a more reliable basis for estimating a larger number of weights.
sel_vec_mssa_comp <- "BIP"

# Macro-indicators for the direct forecast benchmark
# Parsimonious two-indicator design (ifo_c, ESI) chosen to avoid overfitting
# The two series are well-known nowcast indicators
sel_vec_direct_forecast <- c("ifo_c", "ESI")

# Sample alignment (same rationale as in 1.3.3)
align_sample <- T

# Regression type and penalty (same options as in 1.3.3)
reg_type <- "OLS"
reg_type <- "LASSO"
reg_type <- "Ridge"
lambda_reg <- 10

# Forward-shifts of BIP: 0 to 5 quarters (plus publication lag)
shift_vec <- 0:5

# All indicators retained for reference (not used in the main loop below)
sel_indicator_out_sample <- select_vec_multi

# --- Initialize Output Containers ---

# Performance matrices: rows = forward-shifts, columns = forecast horizons
MSE_oos_mssa_comp_mat                  <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
MSE_oos_mssa_comp_without_covid_mat    <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
p_mat_mssa                             <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
p_mat_mssa_components                  <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
p_mat_mssa_components_without_covid    <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
p_mat_direct                           <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
p_mat_direct_without_covid             <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
rRMSE_mSSA_comp_mean                   <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
rRMSE_mSSA_comp_direct                 <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
rRMSE_mSSA_comp_mean_without_covid     <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
rRMSE_mSSA_comp_direct_without_covid   <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
rRMSE_mSSA_direct_mean                 <- matrix(nrow = length(shift_vec), ncol = length(h_vec))
rRMSE_mSSA_direct_mean_without_covid   <- matrix(nrow = length(shift_vec), ncol = length(h_vec))

# Arrays storing final (full-sample) and real-time (expanding-window) predictors
# Dimensions: [forward-shift, forecast horizon, time]
# Used for plotting and revision analysis in exercise 2
final_components_preditor_array <- array(
  dim      = c(length(shift_vec), length(h_vec), nrow(x_mat)),
  dimnames = list(paste0("shift=", shift_vec), paste0("h=", h_vec), rownames(x_mat))
)
oos_components_preditor_array <- array(
  dim      = c(length(shift_vec), length(h_vec), nrow(x_mat)),
  dimnames = list(paste0("shift=", shift_vec), paste0("h=", h_vec), rownames(x_mat))
)

# Array storing expanding-window regression weights over time
# Dimensions: [forward-shift, forecast horizon, time, regression coefficients]
# Used to track systematic vs. noisy revisions in exercise 2.2
track_weights_array <- array(
  dim      = c(length(shift_vec), length(h_vec), nrow(x_mat), length(sel_vec_mssa_comp) + 1),
  dimnames = list(
    paste0("shift=", shift_vec),
    paste0("h=",     h_vec),
    rownames(x_mat),
    c("Intercept", sel_vec_mssa_comp)
  )
)


# Progress bar: tracks loop completion in the R console
pb <- txtProgressBar(min = min(h_vec), max = max(h_vec) - 1, style = 3)

# --- Double Loop: All Forward-Shifts x Forecast Horizons ---
for (shift in shift_vec)  # outer loop: forward-shift of BIP target
{
  setTxtProgressBar(pb, shift)
  
  for (j in h_vec)  # inner loop: M-SSA forecast horizon
  {
    # Map forecast horizon j to array index k (nowcast j=0 corresponds to column k=1)
    k <- j + 1
    
    # A. Construct data matrix for the optimal weighting regression
    #    - Column 1: forward-shifted BIP (target), with NAs padding the end
    #    - Column 2+: M-SSA sub-component(s) optimized for horizon j=k-1
    if (length(sel_vec_mssa_comp) > 1) {
# R handles the case with multiple (M-SSA) predictors differently: have to transpose the matrix t(mssa_array)     
      dat <- cbind(
        c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
        t(mssa_array[sel_vec_mssa_comp,, k])
      )
    } else {
# Single predictor      
      dat <- cbind(
        c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
        mssa_array[sel_vec_mssa_comp,, k]
      )
    }
    rownames(dat) <- rownames(x_mat)
    colnames(dat) <- c(colnames(x_mat)[1], sel_vec_mssa_comp)
    
    # B. Run the expanding-window optimal weighting function
    perf_obj <- optimal_weight_predictor_func(
      dat, in_out_separator, F, shift, lag_vec, align_sample, reg_type, lambda_reg
    )
    
    # C. Store HAC-adjusted p-values (with and without Pandemic)
    p_mat_mssa_components[shift + 1, k]              <- perf_obj$p_value
    p_mat_mssa_components_without_covid[shift + 1, k] <- perf_obj$p_value_without_covid
    
    # D. Store out-of-sample MSE (with and without Pandemic)
    MSE_oos_mssa_comp_mat[shift + 1, k]             <- MSE_oos_mssa_comp             <- perf_obj$MSE_oos
    MSE_oos_mssa_comp_without_covid_mat[shift + 1, k] <- MSE_oos_mssa_comp_without_covid <- perf_obj$MSE_oos_without_covid
    MSE_mean_oos                                     <- perf_obj$MSE_mean_oos
    MSE_mean_oos_without_covid                       <- perf_obj$MSE_mean_oos_without_covid
    
    # E. Store final (full-sample) and real-time (expanding-window) predictors
    #    - final_in_sample_preditor: fitted on the full sample; used for visualization
    #    - cal_oos_pred: re-estimated at each time point; used for out-of-sample evaluation
    #    Both are right-aligned within the full time dimension of x_mat
    final_in_sample_preditor <- perf_obj$final_in_sample_preditor
    final_components_preditor_array[
      shift + 1, j + 1,
      (nrow(x_mat) - length(final_in_sample_preditor) + 1):nrow(x_mat)
    ] <- final_in_sample_preditor
    
    cal_oos_pred <- perf_obj$cal_oos_pred
    oos_components_preditor_array[
      shift + 1, j + 1,
      (nrow(x_mat) - length(cal_oos_pred) + 1):nrow(x_mat)
    ] <- cal_oos_pred
    
    # F. Store expanding-window regression weights for revision analysis (exercise 2.2)
    #    Note: track_weights is overwritten at each iteration; only the final combination
    #    (maximum shift and maximum horizon) is retained after the loop completes
    track_weights <- perf_obj$track_weights
    track_weights_array[
      shift + 1, j + 1,
      (nrow(x_mat) - nrow(track_weights) + 1):nrow(x_mat),
    ] <- track_weights


    #----------------
    # B. Direct Forecasts
    #
    # Key differences from M-SSA (Section A above):
    #   - The data matrix 'dat' is constructed using 'x_mat' (raw indicators) rather than
    #     'mssa_array' (M-SSA components).
    #   - All available indicators are included via 'sel_vec_direct_forecast'. While this
    #     choice could be modified, results are largely robust as long as ifo and ESI are present.
    #   - Unlike in the M-SSA loop above, 'dat' does not depend on the component index 'j',
    #     since no SSA decomposition is involved here.
    
    # Construct data matrix: dependent variable is the forward-shifted target series (BIP),
    # aligned by 'shift' and 'lag_vec<a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[1]</a>'; explanatory variables are the selected raw indicators.
    dat <- cbind(
      c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
      x_mat[, sel_vec_direct_forecast]
    )
    rownames(dat) <- rownames(x_mat)
    
    # Two sample variants for the data matrix:
    if (align_sample)
    {
      # Variant 1: Restrict to the same estimation sample as M-SSA.
      # The first L observations are dropped to account for filter initialization loss.
      dat <- dat[L:nrow(dat), ]
      # Verify aligned sample length matches M-SSA
      nrow(na.exclude(dat))
    } else
    {
      # Variant 2: Use the full available sample, including the first L observations.
      # Preferred when all data is available and no alignment with M-SSA is required.
      dat <- dat[1:nrow(dat), ]
    }
    
    # Fit the direct forecast model and retrieve performance metrics
    perf_obj <- optimal_weight_predictor_func(
      dat, in_out_separator, F, shift, lag_vec, align_sample, reg_type, lambda_reg
    )
    
    # Extract out-of-sample performance metrics:
    # HAC-adjusted p-values and forecast MSE, evaluated both over the full sample
    # and over the COVID-19 pandemic-excluded subsample.
    p_mat_direct[shift + 1, k]                <- perf_obj$p_value
    p_mat_direct_without_covid[shift + 1, k]  <- perf_obj$p_value_without_covid
    MSE_oos_direct                            <- perf_obj$MSE_oos
    MSE_oos_direct_without_covid              <- perf_obj$MSE_oos_without_covid
    
    # Compute relative Root Mean Squared Errors (rRMSEs) across three benchmark comparisons:
    #
    # a. M-SSA component predictor vs. direct forecast
    #    (rRMSE < 1: M-SSA outperforms direct; rRMSE > 1: direct forecast outperforms M-SSA)
    rRMSE_mSSA_comp_direct[shift + 1, k]      <- sqrt(MSE_oos_mssa_comp / MSE_oos_direct)
    
    # b. M-SSA component predictor vs. naive mean benchmark
    #    (rRMSE < 1: M-SSA outperforms the mean benchmark)
    rRMSE_mSSA_comp_mean[shift + 1, k]        <- sqrt(MSE_oos_mssa_comp / MSE_mean_oos)
    
    # c. Direct forecast vs. naive mean benchmark
    #    (rRMSE < 1: direct forecast outperforms the mean benchmark)
    rRMSE_mSSA_direct_mean[shift + 1, k]      <- sqrt(MSE_oos_direct / MSE_mean_oos)
    
    # Repeat comparisons a, b, c excluding the COVID-19 pandemic period
    rRMSE_mSSA_comp_direct_without_covid[shift + 1, k] <-
      sqrt(MSE_oos_mssa_comp_without_covid / MSE_oos_direct_without_covid)
    rRMSE_mSSA_comp_mean_without_covid[shift + 1, k]   <-
      sqrt(MSE_oos_mssa_comp_without_covid / MSE_mean_oos_without_covid)
    rRMSE_mSSA_direct_mean_without_covid[shift + 1, k] <-
      sqrt(MSE_oos_direct_without_covid / MSE_mean_oos_without_covid)
  }
}

# Close the progress bar after loop completion
close(pb)

#-----------------------
# Assign column and row names to all output matrices
# Columns correspond to forecast horizons h; rows correspond to forward-shift values.

colnames(p_mat_mssa_components) <-
  colnames(p_mat_direct) <-
  colnames(p_mat_mssa_components_without_covid) <-
  colnames(rRMSE_mSSA_comp_direct) <-
  colnames(rRMSE_mSSA_comp_mean) <-
  colnames(rRMSE_mSSA_comp_direct_without_covid) <-
  colnames(rRMSE_mSSA_comp_mean_without_covid) <-
  colnames(rRMSE_mSSA_direct_mean) <-
  colnames(rRMSE_mSSA_direct_mean_without_covid) <-
  colnames(p_mat_direct_without_covid) <-
  colnames(MSE_oos_mssa_comp_mat) <-
  colnames(MSE_oos_mssa_comp_without_covid_mat) <- paste("h=", h_vec, sep = "")

rownames(p_mat_mssa_components) <-
  rownames(p_mat_direct) <-
  rownames(p_mat_mssa_components_without_covid) <-
  rownames(rRMSE_mSSA_comp_direct) <-
  rownames(rRMSE_mSSA_comp_mean) <-
  rownames(rRMSE_mSSA_comp_direct_without_covid) <-
  rownames(rRMSE_mSSA_comp_mean_without_covid) <-
  rownames(rRMSE_mSSA_direct_mean) <-
  rownames(rRMSE_mSSA_direct_mean_without_covid) <-
  rownames(p_mat_direct_without_covid) <-
  rownames(MSE_oos_mssa_comp_mat) <-
  rownames(MSE_oos_mssa_comp_without_covid_mat) <- paste("Shift=", shift_vec, sep = "")

# =================================================================
# Out-of-Sample Performance Summary
# =================================================================

# HAC-adjusted p-values for the M-SSA component predictor targeting forward-shifted BIP.
# Evaluation covers the out-of-sample period from 'in_out_separator' through January 2025.
p_mat_mssa_components

# Same evaluation excluding the singular COVID-19 pandemic observations.
p_mat_mssa_components_without_covid
# Interpretation: the M-SSA predictor maintains statistically significant predictive
# content for BIP growth across multiple quarters ahead.

# Note on p-values (above) vs. rRMSEs (below):
#   - The p-values reported above are derived from out-of-sample M-SSA
#     outputs, but rely on full-sample regressions for the 'static' level
#     and scale adjustments applied to the M-SSA-BIP predictor. They
#     therefore do not constitute fully out-of-sample evaluations.
#   - The rRMSEs computed below provide a stricter assessment of 'true'
#     forecast accuracy: both the M-SSA extraction and the level and scale
#     alignment are performed in a genuinely out-of-sample fashion, ensuring
#     that no future information leaks into the evaluation.

# a. rRMSE: M-SSA component predictor benchmarked against the naive mean forecast
rRMSE_mSSA_comp_mean

# b. rRMSE: M-SSA component predictor benchmarked against the direct forecast
rRMSE_mSSA_comp_direct

# c. rRMSE: Direct forecast benchmarked against the naive mean forecast
#    Note: columns are identical within each row because the direct forecast's
#    explanatory variables do not depend on the M-SSA component index 'j' —
#    the same model is fitted for all j at a given shift value.
rRMSE_mSSA_direct_mean

# Same three rRMSE comparisons, excluding the COVID-19 pandemic period:
rRMSE_mSSA_comp_mean_without_covid
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_direct_mean_without_covid

# =================================================================
# Summary of Key Findings:
#
# Preliminary Notes:
#   a. The M-SSA filter relies on a VAR model estimated on the in-sample period ending
#      before the 2008 financial crisis; the VAR is fixed and never updated.
#   b. All forecast performances are evaluated out-of-sample over the period 2008–2025.
#
# 1. Forecasting BIP vs. HP-BIP:

#    - HP-BIP is substantially easier to forecast than raw BIP, because the two-sided
#      HP filter removes unpredictable high-frequency noise, leaving a smoother and more
#      systematic cyclical signal for the predictor to track, see exercise 1.0 above.
#    - A statistically significant (HAC-adjusted) predictive relationship is
#      detected up to approximately one year ahead between the M-SSA smoothed
#      predictor and the following target variables:
#       (i)   Forward-shifted HP-BIP: a very strong and robust link that
#             persists across all horizons examined (see Exercise 1.0 above).
#       (ii)  Forward-shifted raw BIP: a weaker but still detectable link,
#             reflecting the additional noise introduced by the unfiltered
#             target series.
#   - The predictive evidence for raw BIP is further reinforced by the
#     following indirect argument: if future BIP is determined, at least
#     in part, by its HP-filtered smooth counterpart, then the demonstrated
#     ability to forecast HP-BIP translates into meaningful forecast evidence
#     for raw BIP as well — even when the direct statistical link appears
#     weaker. Moreover, the alignment between raw BIP and its HP-filtered
#     trend is expected to strengthen particularly during periods of large
#     economic swings, when the cyclical component dominates and idiosyncratic
#     noise plays a comparatively smaller role. Incidentally, these are also the episodes 
#     of main interest to forecasters.
#
# 2. Systematic Horizon-Shift Pattern:

#    - In the performance matrices, rows index forward shifts (target lead time) and columns
#      index forecast horizons h (the horizon for which each M-SSA filter is optimized).
#      For a given row (fixed shift), forecast performance tends to improve as h increases
#      from left to right, peaking when h aligns with — or slightly exceeds — the shift value.
#      This corresponds to the diagonal (and just above) of the performance matrices. 
#
#    - Clarity of the pattern depends on the target series:
#        (i)  HP-BIP (low-noise target): the diagonal pattern is strong, clean, and
#             consistent across all shift values, see exercise 1.0 above.
#        (ii) Raw BIP (high-noise target): the pattern is present but partially obscured
#             by the high-frequency noise in the unfiltered target series.
#
#    - This regularity reflects a coherent and interpretable internal structure:
#      each M-SSA design is effective at (or towards) the horizon for which it was optimized,
#      confirming that the optimization criterion successfully encodes horizon-specific
#      information into the filter design — and that this encoding is empirically
#      recoverable from out-of-sample forecast evaluation (at least in the absence of strong noise).

# 3. Impact of the COVID-19 Pandemic on Out-of-Sample Evaluation:
#    - Including pandemic observations (2020–2021) in the validation sample inflates both
#      p-values and rRMSEs, obscuring the underlying systematic horizon-shift patterns
#      with a small number of extreme outlier-driven observations.
#    - When pandemic observations are included, direct forecasts fail to outperform the
#      naive mean benchmark at forward shifts greater than two quarters — a result driven
#      primarily by the distorting influence of the pandemic episode rather than by a
#      genuine deterioration in predictor quality.
# =================================================================




# =======================================================================
# Exercise 2: Analyze Revisions of the M-SSA Components Predictor
# =======================================================================
#
# Background:
#   - The M-SSA components predictor relies on quarterly re-calibration of the OLS regression weights.
#   - In contrast, the M-SSA filter itself is NOT subject to revisions: the underlying VAR is fixed,
#     estimated on data up to 2008, and is never updated.
#   - This exercise isolates and quantifies the impact of quarterly weight re-calibration on both
#     the predictor series and the regression weights over time.

# Select forecast horizon and forward shift for this analysis
h     <- 6
shift <- 3

# =================================================================
# 2.1 Final vs. Real-Time Predictor: Assessing Revision Magnitude
# =================================================================
# Note: the VAR model, and hence the M-SSA outputs derived from it, are
# kept fixed throughout — estimated on the in-sample span up to 2007.
# The out-of-sample analysis conducted here focuses exclusively on the
# revisions induced by the regression step, in which the M-SSA predictor
# is aligned to the target variable via level and scale adjustments
# (as introduced in Exercise 1.1 above). The stability of the VAR and
# M-SSA components is therefore taken as given, and only the regression-
# based alignment is subject to the out-of-sample evaluation.

# We compare two versions of the predictor:
#   - Final predictor:            regression weights estimated on the full available sample.
#   - Real-time (OOS) predictor:  regression weights re-estimated recursively each quarter,
#                                 using only data available at that point in time.
#
# In the absence of revisions, both series would overlap exactly.
# Discrepancies between them reflect the revision error introduced by recursive weight re-estimation.

#-------------------------
# Plot
par(mfrow = c(1, 1))
mplot <- cbind(
  final_components_preditor_array[shift, h, ],
  oos_components_preditor_array[shift, h, ]
)
colnames(mplot) <- c("Final predictor", "Real-time out-of-sample predictor")
colo       <- c("blue", rainbow(length(select_vec_multi)))
main_title <- "Revisions: final vs. real-time predictor"
plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
# Vertical dashed line marks the start of the out-of-sample evaluation period
abline(v = which(rownames(mplot) <= date_to_fit)[length(which(rownames(mplot) <= date_to_fit))],
       lwd = 2, lty = 2)
axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
     labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
axis(2)
box()
#--------------------------

# Note: This plot uses shift and h as specified above. Analogous plots for other shift/horizon
#      combinations show similar revision behavior.

# Key observations on revision behavior:
#
#   - Early sample (left side of plot): revisions are large due to the short estimation window,
#     causing the real-time predictor to be volatile and poorly calibrated.
#   - Later sample (right side of plot): as the estimation window grows, the real-time predictor
#     converges toward the final (full-sample) estimate, and revisions diminish.
#   - Vertical dashed line: marks the beginning of the out-of-sample evaluation period used
#     for MSE and p-value statistics in Exercise 1.3.5.
#   - Forward outlook: as the sample grows further, revision errors are expected to shrink,
#     contributing to improved out-of-sample forecast performance over time.

# ============================================================================
# 2.2 Regression Weight Stability: Tracking Revisions in OLS Weights Over Time
# ============================================================================
# We now examine how the regression weights themselves evolve as the estimation sample expands.
# Convergence of the weights over time would confirm that the real-time predictor is stabilizing
# toward the final predictor, and that the revision process is well-behaved.

# Note: to mitigate overfitting — which is a material concern given the
# short in-sample estimation span — the regression is restricted to a
# single predictor, namely the BIP-based M-SSA component (M-SSA-BIP).
# Consequently, the model contains only two parameters to estimate:
# an intercept (level adjustment) and a slope coefficient (scale adjustment)
# on M-SSA-BIP.

#------------------------
# Plot
par(mfrow = c(1, 1))
mplot <- track_weights_array[shift, h, , ]
colo       <- c("black", "blue", "red")
main_title <- "Revisions: regression weights over time"

plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
# Vertical dashed line marks the start of the out-of-sample evaluation period
abline(v = which(rownames(mplot) <= date_to_fit)[length(which(rownames(mplot) <= date_to_fit))],
       lwd = 2, lty = 2)
axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
     labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
axis(2)
box()
#------------------------

# Key observations on weight dynamics:
#
#   - Early sample (before 2008 financial crisis) : regression weights are highly volatile, 
#     reflecting estimation uncertainty in short samples and sensitivity to individual observations.
#
#   - Over time: weights progressively stabilize, converging toward apparent fixed points,
#     consistent with a (more or less) stationary weight process.
#     - This convergence of the weights mirrors and explains the convergence of the real-time
#       predictor toward the final full-sample predictor observed in Section 2.1.
#     - The regression weight (blue) and intercept (black) jointly perform the static
#       level and scale calibration of the M-SSA predictor to the target series:
#         (i)   The intercept captures the level offset between the M-SSA output and the
#               target. Since M-SSA is centered (optimized for HP-BIP, which has zero mean (HP applied to standardized BIP)),
#               and the target is standardized BIP growth, the intercept remains close to
#               zero throughout the sample — confirming that no systematic level adjustment
#               is required.
#         (ii)  The regression weight reflects the scale difference between the M-SSA
#               output (a filtered estimate of HP-BIP) and the standardized BIP growth rate
#               target. The consistently large weight value is therefore expected, as it
#               rescales the h-step predictor of HP-BIP to the amplitude of the (standardized) BIP growth target.
#         (iii) When predicting original BIP growth, the regression would account for corresponding 
#               `static' level and scale adjustments, see exercise 6.2. 
#
#   - Anchoring role of the financial crisis (2008-2009):
#     - The sharp BIP contraction and subsequent rebound during the financial crisis constitute
#       a structurally influential event that anchors the statistical dependence between the
#       forward-shifted BIP target and the M-SSA components predictor.
#     - The importance of this event is directly visible in the plot: the regression weight
#       (blue) undergoes a pronounced and persistent upward shift following 2008, reflecting
#       the strong and lasting realignment between the M-SSA predictor and the BIP target
#       that is established by the crisis episode.
#     - In the absence of such pronounced cyclical dynamics, BIP log-differences would behave
#       approximately as white noise around a fixed trend growth rate, leaving little systematic
#       low-frequency (growth/cycle) variation for M-SSA to exploit — and thus providing negligible predictive
#       content at horizons at (or beyond) one quarter.
#
#   - Value of M-SSA during strong cyclical swings:
#     - During episodes of large cyclical swings — such as recession contractions and recovery
#       rebounds — M-SSA effectively tracks the pronounced low- (cycle-) frequency dynamics of BIP,
#       precisely the signal it is designed to extract.
#     - This dynamic tracking is what establishes and reinforces M-SSA as a reliable predictor,
#       particularly during periods of abnormally large and persistent BIP movements that
#       deviate substantially from the mean trend growth path.




# =======================================================================
# Exercise 3: (Skipped)

# =======================================================================



# =======================================================================================
# Exercise 4: Diagnosing the Sources of Forecast Gains in the M-SSA Components Predictor
# =======================================================================================
#
# Motivation:
#   The M-SSA components predictor is a multi-step ('stacked') construction involving:
#     Step 1 — De-noising:  Facilitate prediction by removing unpredictable high-frequency noise. 
#                           Emphasize mid-term dynamics relevant in a one-year forecast perspective:
#                           -Two-sided HP(160) target. 
#                           -One-sided univariate classic HP-C
#                           -See tutorials 7.1-7.3
#     Step 2 — M-SSA:       Multivariate causal (real-time) filtering:
#                           -Now- and forecast the two-sided HP(160): M-SSA.
#                           -Maximize target correlation (equivalently: maximize sign accuracy)
#                           -Subject to a smoothness constraint (holding-time (HT) constraint).
#                           -See tutorials 7.2, 7.3
#     Step 3 — Regression:  Regress the M-SSA components on forward-shifted BIP via OLS.
#                           -Link M-SSA directly with BIP (instead of smoothed HP-BIP)
#                           -Calibrate M-SSA to original level and scale of BIP (log differences)
#                           -This tutorial (7.4)
# 
#
# Prior evidence:
#   - Exercises above confirmed that the M-SSA components predictor outperforms both the naive
#     mean benchmark and the direct forecast (using unfiltered indicators) over a long
#     out-of-sample span including the financial crisis and the COVID-19 pandemic.
#   - However, the specific contribution of M-SSA (Step 2 above) to these forecast gains has not yet
#     been isolated or quantified.
#
# Research questions:
#   1. Why does the M-SSA components predictor outperform classical forecast rules at horizons
#      of two or more quarters?
#   2. Which construction step(s) are driving the performance improvements?
#
# Approach — introducing an intermediary benchmark:
#   To answer these questions, we introduce a new benchmark called the 'direct HP forecast':
#     - Apply the classical univariate HP concurrent filter (HP-C) to each individual indicator.
#     - Regress the HP-C filtered indicators on forward-shifted BIP (analogous to the direct forecast).
#   This benchmark isolates the contribution of filtering alone (Step 1), independently of
#   the multivariate M-SSA optimization (Step 2).
#
# The three predictors now differ only in their choice of explanatory variables:
#
#   | Predictor              | Explanatory Variables                              |
#   |------------------------|----------------------------------------------------|
#   | M-SSA components       | Multivariate (M-SSA) filtered indicators           |
#   | Direct forecast        | Original, unfiltered indicators                    |
#   | Direct HP forecast     | Univariate HP concurrent-filtered indicators       |
#
#   Note: the direct HP forecast is a special case of the M-SSA predictor, where the
#   multivariate filter is replaced by independent univariate HP-C filters.
#
# Interpretation:
#   - Direct vs. Direct HP:   isolates the gain from simple univariate filtering (Step 1) alone.
#   - Direct HP vs. M-SSA:    isolates the additional gain from multivariate filtering (Step 2).
#   - Direct vs. M-SSA:       measures the combined gain.
#
# A structured comparison of these three predictors will reveal the relative contributions
# of each construction step to the overall forecast efficiency of the M-SSA components predictor.


# =======================================================================
# Exercise 4.1: Direct HP Forecast Benchmark
#
# Motivation:
#   - The M-SSA components predictor is benchmarked here against a 'direct HP forecast':
#     a simpler predictor that applies the one-sided (concurrent) HP filter directly
#     to each indicator and uses the filtered outputs as regressors for forward-shifted BIP
#   - This benchmark is more sophisticated than the naive mean or direct OLS forecast,
#     as it incorporates the same smoothing philosophy as M-SSA (removing unpredictable noise) 
#     but without the multivariate optimization and smoothness customization steps
#   - Comparing M-SSA against this benchmark isolates the value added by the
#     M-SSA design beyond simple one-sided HP filtering
# =======================================================================

# =======================================================================
# 4.1.1 Compute the Classic One-Sided (Concurrent) HP Filter
# =======================================================================
# lambda_HP and L are carried over from the M-SSA design in exercise 1
lambda_HP <- lambda_HP
L         <- L

# Compute the HP filter object: extracts the one-sided (concurrent) filter coefficients
HP_obj <- HP_target_mse_modified_gap(L, lambda_HP)

# One-sided HP filter (concurrent): applies only past and current observations
# No future data are used — this is a real-time, causal filter
hp_c <- HP_obj$hp_trend

ts.plot(hp_c,
        main = paste0("One-sided (concurrent) HP(", lambda_HP, ") filter coefficients"),
        xlab = "", ylab = "")

# =================================================================
# 4.1.2 Apply HP-C to All Indicators
# =================================================================
# Filter each indicator series using the one-sided HP filter (causal, side = 1)
hp_c_mat <- NULL
for (i in 1:ncol(x_mat)) {
  hp_c_mat <- cbind(hp_c_mat, filter(x_mat[, i], hp_c, side = 1))
}
colnames(hp_c_mat) <- colnames(x_mat)
rownames(hp_c_mat) <- rownames(x_mat)

# --- Plot Concurrent HP-Filtered Indicators ---
mplot      <- hp_c_mat
colo       <- rainbow(ncol(mplot))
main_title <- paste0("Concurrent HP(", lambda_HP, ") applied to all indicators")
par(mfrow = c(1, 1))
plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
for (j in 1:ncol(mplot)) {
  lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
  mtext(colnames(mplot)[j], col = colo[j], line = -j)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# =================================================================
# 4.1.3 Extend HP-C to Multi-Step Forecasts
# =================================================================
# For each forecast horizon h in h_vec, the one-sided HP filter is adapted to produce
# an h-step-ahead forecast by skipping the first h coefficients (replacing them with zeros).
#
# Rationale:
#   - Skipping the first h coefficients is equivalent to assuming the future h observations
#     are white noise (WN)
#   - Log-returns of the macro indicators are empirically close to WN
#   - The resulting filter is still causal (one-sided) and operationally real-time

h_vec        <- 0:6
hp_c_array   <- array(dim = c(ncol(x_mat), nrow(x_mat), length(h_vec)))

for (j in 1:length(h_vec)) {
  for (i in 1:ncol(x_mat)) {
    # Construct the h-step-ahead HP forecast filter:
    # drop the first h_vec[j] coefficients and append h_vec[j] zeros at the end
    hp_c_forecast       <- c(hp_c[(h_vec[j] + 1):L], rep(0, h_vec[j]))
    hp_c_array[i,, j]  <- filter(x_mat[, i], hp_c_forecast, side = 1)
  }
}

# Assign dimension names for readability
dimnames(hp_c_array)[[1]] <- colnames(x_mat)
dimnames(hp_c_array)[[2]] <- rownames(x_mat)
dimnames(hp_c_array)[[3]] <- paste0("h=", h_vec)

# --- Plot HP Now- and Forecasts for a Selected Indicator ---
# Illustrate the effect of increasing the forecast horizon on the HP filter output
i <- 1
colnames(x_mat)[i]   # confirm the selected indicator (BIP)

# Scale outputs for visual comparability across horizons
mplot      <- scale(hp_c_array[i,,])
colnames(mplot) <- paste0(colnames(x_mat)[i], ": h=", h_vec)
colo       <- rainbow(ncol(mplot))
main_title <- paste0("HP(", lambda_HP, ") now- and forecasts for ", colnames(x_mat)[i])

par(mfrow = c(1, 1))
plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))

for (j in 1:ncol(mplot)) {
  lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
  mtext(colnames(mplot)[j], col = colo[j], line = -j)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()
# As expected: increasing the forecast horizon produces a systematic left-shift (phase advance)
# of the (HP-C) filter output —. This left-shift with increasing h seems to be more pronounced 
# at zero-crossings but less evident at peaks and troughs.

# ===============================================================================
# 4.2: Performance Evaluation — Direct HP Forecast vs. M-SSA Components Predictor
# ===============================================================================
# We compute HAC-adjusted p-values and rRMSEs for all combinations of:
#   - forward-shift of BIP (rows): shift_vec
#   - forecast horizon of HP-C (columns): h_vec
#
# Note: numerical computations may take some time
#   A progress bar pops up
#------------------

shift_vec <- shift_vec

# --- Initialize Performance Matrices ---
# Rows = forward-shifts of BIP; columns = forecast horizons
p_mat_HP_c                         <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)
p_mat_HP_c_without_covid           <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)
MSE_oos_HP_c_without_covid_mat     <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)
rRMSE_mSSA_comp_HP_c               <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)
rRMSE_mSSA_comp_HP_c_without_covid <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)
ht_HP_c_mat                        <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)

# Progress bar
pb <- txtProgressBar(min = min(h_vec), max = max(h_vec) - 1, style = 3)

# --- Double Loop: All Forward-Shifts x Forecast Horizons ---
for (shift in shift_vec)  # outer loop: forward-shift of BIP target
{
  setTxtProgressBar(pb, shift)
  
  for (j in h_vec)  # inner loop: HP-C forecast horizon
  {
    # Map forecast horizon j to array index k
    k <- j + 1
    
    # Construct data matrix for the optimal weighting regression:
    #   - Column 1: forward-shifted BIP (target)
    #   - Column 2+: HP-C univariate filtered outputs for selected indicators at horizon k
    # All indicators are used here (results are robust to this choice provided
    # ifo and ESI are included)
    if (length(sel_vec_pred) > 1) {
      dat <- cbind(
        c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
        t(hp_c_array[sel_vec_pred,, k])
      )
    } else {
      dat <- cbind(
        c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
        hp_c_array[sel_vec_pred,, k]
      )
      colnames(dat)[2] <- sel_vec_pred
    }
    rownames(dat) <- rownames(x_mat)
    dat           <- na.exclude(dat)
    
    # Run the expanding-window optimal weighting function
    perf_obj <- optimal_weight_predictor_func(
      dat, in_out_separator, F, shift, lag_vec, align_sample, reg_type, lambda_reg
    )
    
    # Retrieve and store performance metrics
    # a. Out-of-sample MSE (with and without Pandemic)
    MSE_oos_HP_c                                    <- perf_obj$MSE_oos
    MSE_oos_HP_c_without_covid_mat[shift + 1, k]   <- MSE_oos_HP_c_without_covid <- perf_obj$MSE_oos_without_covid
    
    # b. HAC-adjusted p-values (with and without Pandemic)
    p_mat_HP_c[shift + 1, k]                       <- perf_obj$p_value
    p_mat_HP_c_without_covid[shift + 1, k]         <- perf_obj$p_value_without_covid
    
    # c. rRMSE: M-SSA components predictor vs. direct HP forecast
    #    Values below 1 indicate M-SSA outperforms the direct HP benchmark
    #    MSE of M-SSA components predictor was computed in exercise 1.3.5
    rRMSE_mSSA_comp_HP_c[shift + 1, k]             <- sqrt(MSE_oos_mssa_comp_mat[shift + 1, k] / MSE_oos_HP_c)
    rRMSE_mSSA_comp_HP_c_without_covid[shift + 1, k] <- sqrt(MSE_oos_mssa_comp_without_covid_mat[shift + 1, k] /
                                                               MSE_oos_HP_c_without_covid)
    
    # d. Empirical holding time of the direct HP forecast out-of-sample predictor
    #    (for comparison with M-SSA holding times)
    ht_HP_c_mat[shift + 1, k] <- compute_empirical_ht_func(perf_obj$cal_oos_pred)$empirical_ht
  }
}
close(pb)

# --- Assign Row and Column Names ---
col_names <- paste0("h=",     h_vec)
row_names <- paste0("Shift=", shift_vec)

colnames(p_mat_HP_c) <- colnames(p_mat_HP_c_without_covid) <-
  colnames(MSE_oos_HP_c_without_covid_mat) <- colnames(rRMSE_mSSA_comp_HP_c) <-
  colnames(rRMSE_mSSA_comp_HP_c_without_covid) <- colnames(ht_HP_c_mat) <- col_names

rownames(p_mat_HP_c) <- rownames(p_mat_HP_c_without_covid) <-
  rownames(MSE_oos_HP_c_without_covid_mat) <- rownames(rRMSE_mSSA_comp_HP_c) <-
  rownames(rRMSE_mSSA_comp_HP_c_without_covid) <- rownames(ht_HP_c_mat) <- row_names

# --- Compare HAC-Adjusted p-Values Across All Three Predictors ---
# Three-way comparison: M-SSA components vs. classic direct forecast vs. direct HP forecast
# Systematic differences reveal the marginal contribution of each design feature

# Full sample (including Pandemic)
p_mat_mssa_components   # M-SSA components predictor
p_mat_direct            # Classic direct OLS forecast (raw indicators)
p_mat_HP_c              # Direct HP forecast (HP-filtered indicators)

# Excluding Pandemic observations
# Isolates structural predictability from crisis-driven statistical artifacts
p_mat_mssa_components_without_covid
p_mat_direct_without_covid
p_mat_HP_c_without_covid

# ------------------------------------------------------------------------------------
# Findings: Comparison of Direct, Direct HP, and M-SSA Components Predictors
#
#   - Like the classic direct forecast, the direct HP forecast (based on univariate HP-C filtered
#     indicators) fails to produce statistically significant BIP forecasts at forward shifts
#     greater than one quarter (plus the BIP publication lag).
#   - In contrast, the M-SSA components predictor remains statistically significant for forward
#     shifts of up to four or five quarters (plus the publication lag), demonstrating a clear and
#     systematic advantage at longer forecast horizons.
# ------------------------------------------------------------------------------------

# Relative RMSE Comparison (excluding COVID-19 pandemic period)
#
# Compare forecast accuracy of the M-SSA components predictor against:
#   (a) the classic direct forecast (unfiltered indicators)
#   (b) the direct HP forecast (univariate HP-C filtered indicators)
# rRMSE < 1 indicates that M-SSA outperforms the respective benchmark.
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_comp_HP_c_without_covid

# Absolute MSE Comparison (excluding COVID-19 pandemic period)
#
# The rRMSEs above are the square roots of the ratios of the MSEs below.
# Examining the raw MSEs confirms the magnitude of accuracy differences.
MSE_oos_HP_c_without_covid_mat
MSE_oos_mssa_comp_without_covid_mat

# =================================================================

# Interpretation of results:
#
#   1. Simple univariate filtering does not help:
#      Applying the univariate HP-C filter to the indicators does not improve forecast accuracy
#      (at shift>2) relative to using unfiltered indicators (direct forecast). The direct HP forecast
#      performs comparably to — or no better than — the direct forecast at shift>2.
#
#   2. Simple regression cannot exploit multivariate structure:
#      The outperformance of both direct forecasts (without filtering) and direct HP forecasts 
#      by the M-SSA components predictor implies that OLS regression alone (with or without 
#      HP-C pre-filtering) is insufficient to capture the multivariate dependencies among the indicators.
#
#   3. Joint treatment of longitudinal and cross-sectional structure is key:
#      The evidence suggests that effective multi-horizon BIP forecasting requires the
#      simultaneous exploitation of time-series dynamics (longitudinal) and inter-indicator
#      relationships (cross-sectional), as provided by M-SSA in combination with OLS regression.
#
#   4. Open questions:
#      Why exactly does the M-SSA step contribute? What is the specific mechanism through
#      which the multivariate filter adds forecasting value beyond univariate filtering? And what 
#      is the role of the HT constraint in M-SSA?

# =================================================================



# =================================================================
# 4.3 Diagnosing the M-SSA Contribution: A Targeted Experiment
# =================================================================
#
# To understand why M-SSA outperforms HP-C, we exploit a key asymmetry in the leading/lagging
# relationships between the target variable and the explanatory indicators:
#
#   - Targeting HP-BIP:
#       The explanatory series (ip, ESI, ifo, spread) are all LEADING relative to BIP
#       (accounting for the BIP publication lag). In this configuration, M-SSA can
#       actively leverage the leading information in the indicators, and is therefore
#       expected to outperform the univariate HP-C filter.
#
#   - Targeting HP-spread:
#       The remaining explanatory series (BIP, ip, ifo, ESI) are LAGGING relative to
#       spread. In this configuration, there is little leading information for M-SSA
#       to exploit, and we therefore do NOT expect substantial gains of M-SSA over HP-C.
#
# This asymmetry allows us to test whether the M-SSA advantage is specifically attributable
# to its ability to extract and aggregate leading cross-sectional signals — a capability
# absent in univariate HP-C filtering.
#---------------------


# Four-panel comparison plot:
#   Panel a (top-left):    HP-C filter applied to BIP
#   Panel b (top-right):   M-SSA filter applied to BIP     [M-SSA expected to outperform HP-C]
#   Panel c (bottom-left): HP-C filter applied to spread
#   Panel d (bottom-right):M-SSA filter applied to spread  [M-SSA NOT expected to outperform HP-C]

par(mfrow = c(2, 2))

# --- Panel a: HP-C applied to BIP ---
i <- 1
colnames(x_mat)[i]
# Scale the filtered series for visual comparability across forecast horizons
mplot      <- scale(hp_c_array[i, , ])
colnames(mplot) <- paste(colnames(x_mat)[i], ": h=", h_vec, sep = "")
colo       <- rainbow(ncol(mplot))
main_title <- paste("HP-C targeting HP-", colnames(x_mat)[i], sep = "")

plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (j in 1:ncol(mplot))
{
  lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
  mtext(colnames(mplot)[j], col = colo[j], line = -j)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# --- Panel b: M-SSA applied to BIP ---
# M-SSA leverages leading cross-sectional information; outperformance over HP-C is expected here.
mplot      <- scale(mssa_array[i, , ])
colnames(mplot) <- paste(colnames(x_mat)[i], ": h=", h_vec, sep = "")
colo       <- rainbow(ncol(mplot))
main_title <- paste("M-SSA targeting HP-", colnames(x_mat)[i], sep = "")

plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (j in 1:ncol(mplot))
{
  lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
  mtext(colnames(mplot)[j], col = colo[j], line = -j)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# --- Panel c: HP-C applied to spread ---
i <- 5
colnames(x_mat)[i]
# Scale the filtered series for visual comparability across forecast horizons
mplot      <- scale(hp_c_array[i, , ])
colnames(mplot) <- paste(colnames(x_mat)[i], ": h=", h_vec, sep = "")
colo       <- rainbow(ncol(mplot))
main_title <- paste("HP-C targeting HP-", colnames(x_mat)[i], sep = "")

plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (j in 1:ncol(mplot))
{
  lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
  mtext(colnames(mplot)[j], col = colo[j], line = -j)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# --- Panel d: M-SSA applied to spread ---
# Explanatory series are lagging relative to spread; no substantial M-SSA gain over HP-C expected.
mplot      <- scale(mssa_array[i, , ])
colnames(mplot) <- paste(colnames(x_mat)[i], ": h=", h_vec, sep = "")
colo       <- rainbow(ncol(mplot))
main_title <- paste("M-SSA targeting HP-", colnames(x_mat)[i], sep = "")

plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (j in 1:ncol(mplot))
{
  lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
  mtext(colnames(mplot)[j], col = colo[j], line = -j)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# =================================================================
# Analysis of Exercise 4.3 Results
# =================================================================
# --- Top Two Panels: Targeting HP-BIP ---
#
# The key difference between HP-C (top left) and M-SSA (top right) lies in the
# size and quality of the left-shift (phase advance) as a function of forecast horizon:
#
#   Size:
#     - M-SSA produces a more pronounced left-shift at zero-crossings than HP-C
#     - The advance is larger and more systematic across horizons
#
#   Quality:
#     - HP-C produces a left-shift primarily at zero-crossings
#     - M-SSA generates a more pronounced left-shift at all signal levels: zero-crossings, local peaks,
#       troughs, and intermediate levels — particularly visible at recessions and crises
#         -Compare how troughs and peaks at crises (dotcom, financial crisis, COVID) are  
#         affected by M-SSA vs. HP-C as h increases. 
#     - This richer phase behaviour reflects the multivariate optimization in M-SSA,
#       which exploits leading indicators to advance the filter output more uniformly
#
# --- Bottom Two Panels: Targeting HP-Spread ---
#
#   - M-SSA (bottom right) and HP-C (bottom left) are nearly identical
#   - M-SSA offers no additional left-shift advantage over HP-C in this case
#
#   Explanation:
#     - When targeting HP-spread, the multivariate M-SSA filter degenerates to a
#       univariate design (see tutorial 7.1, exercise 1.5)
#     - All other indicators (BIP, ip, ifo, ESI) lag spread and therefore carry
#       no incremental information for predicting HP-spread
#     - With no leading cross-sectional information to exploit, M-SSA reduces to HP-C
#
# --- Key Inference ---
#
#   - M-SSA outperforms HP-C when targeting HP-BIP (and similarly HP-ip, not shown here)
#   - Within the OLS regression for forward-shifted BIP, the M-SSA BIP and M-SSA ip
#     components are the dominant explanatory variables; remaining components are
#     not statistically significant in the short in-sample span.
#   - M-SSA contributes to forecast outperformance by supplying regressors with a
#     more effective left-shift and an enhanced smoothness (less noisy) than either:
#       (i)  HP-C filtered outputs (used in the direct HP forecast), or
#       (ii) Raw unfiltered indicators (used in the classic direct forecast)

# --- Summary of Findings: HP-C vs. Direct Forecasts ---
#
#   - Applying HP-C does not improve forecast performance over the classic direct forecast
#     (unfiltered indicators): simple one-sided smoothing is insufficient
#   - The BIP forecasting problem requires more advanced signal extraction than
#     univariate HP filtering combined with regression can provide
#   - The gains of M-SSA stem specifically from the multivariate optimization framework,
#     which generates a smoother (larger HT) and more timely predictor (as a function of horizon)

# --- Key Design Elements of the M-SSA Components Predictor ---
#
#   1. Target specification — HP(160):
#      Emphasizes mid-term dynamics relevant in a one-year forecast framework
#      while suppressing unpredictable high-frequency noise; 
#      see tutorial 7.3, exercises 3 and 4 for empirical justification
#
#   2. Multivariate signal extraction — cross-sectional and longitudinal information:
#      - Exploits leading indicators to improve HP-BIP forecasts several quarters ahead
#      - Delivers a smoother, i.e., less noisy predictor (HT constraint)
#      - Delivers a more pronounced and higher-quality left-shift (both in size and
#        uniformity across signal levels) than univariate HP-C filtering
#
#   3. Smoothness control — HT constraint:
#      Controls the rate of zero-crossings of the predictor
#
# Together, these design choices justify the additional complexity of the M-SSA framework:
# the pronounced, horizon-dependent left-shift — the predictor's key operative property —
# is a direct product of the M-SSA optimization and cannot be replicated by simpler methods.

# --- Open Questions for Further Investigation ---
#
#   - Like M-SSA, the classic multivariate M-MSE criterion can exploit both longitudinal
#     and cross-sectional information. We therefore expect M-MSE to achieve a comparable
#     left-shift (timeliness) to M-SSA.
#   - The main anticipated difference between M-SSA and M-MSE is smoothness: M-SSA
#     imposes an explicit HT constraint, whereas M-MSE does not.
#   - Key questions:
#       (i)   Can M-SSA outperform M-MSE in out-of-sample forecast accuracy?
#       (ii)  Can M-SSA provide additional/alternative foresight?
#       (iii) How is this related to enhanced smoothness (larger HT)?
#   - These questions are addressed in the final two exercises.




# =======================================================================
# Exercise 5: M-MSE vs. M-SSA Components Predictor
# =======================================================================
#
# Motivation:
#   - Exercise 4 established that multivariate filtering (M-SSA) outperforms both
#     univariate HP-C filtering and unfiltered direct forecasts, via a more systematic
#     left-shift of the predictor
#   - However, M-SSA has not yet been compared to M-MSE, the classic multivariate
#     MSE-optimal filter (without the HT smoothness constraint)
#   - M-MSE produces more zero-crossings (less smooth) than M-SSA, as the HT
#     constraint is absent — reflecting a different position on the AST trilemma
#   - This exercise evaluates whether the smoothness imposed by M-SSA helps or
#     hurts out-of-sample MSE performance relative to M-MSE
#
# Construction of the M-MSE components predictor:
#   - Identical to the M-SSA components predictor, except that mmse_array replaces
#     mssa_array as the source of regressors in the OLS optimal weighting step
#   - All other settings (regression type, sample alignment, benchmarks) are unchanged
# =======================================================================

# Exercise 5.1: Compute M-MSE Components Predictor Performance
# =======================================================================
shift_vec <- shift_vec

# --- Initialize Performance Matrices ---
# Rows = forward-shifts of BIP; columns = forecast horizons
# Four matrices: M-MSE vs. mean and M-MSE vs. M-SSA, with and without Pandemic
rRMSE_mmse_comp_mean                 <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)
rRMSE_mmse_comp_mean_without_covid   <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)
rRMSE_mmse_comp_mssa                 <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)
rRMSE_mmse_comp_mssa_without_covid   <- matrix(ncol = length(h_vec), nrow = length(h_vec) - 1)

# Progress bar
pb <- txtProgressBar(min = min(h_vec), max = max(h_vec) - 1, style = 3)

# --- Double Loop: All Forward-Shifts x Forecast Horizons ---
for (shift in shift_vec)  # outer loop: forward-shift of BIP target
{
  setTxtProgressBar(pb, shift)
  
  for (j in h_vec)  # inner loop: M-MSE forecast horizon
  {
    # Map forecast horizon j to array index k
    k <- j + 1
    
    # Construct data matrix for the optimal weighting regression:
    #   - Column 1: forward-shifted BIP (target)
    #   - Column 2+: M-MSE filter outputs for selected indicators at horizon k
    #   Key difference from exercise 1.3: mmse_array replaces mssa_array
    if (length(sel_vec_pred) > 1) {
      dat <- cbind(
        c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
        t(mmse_array[sel_vec_pred,, k])
      )
    } else {
      dat <- cbind(
        c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
        mmse_array[sel_vec_pred,, k]
      )
    }
    rownames(dat) <- rownames(x_mat)
    colnames(dat) <- c(colnames(x_mat)[1], sel_vec_pred)
    dat           <- na.exclude(dat)
    
    # Run the expanding-window optimal weighting function
    perf_obj <- optimal_weight_predictor_func(
      dat, in_out_separator, F, shift, lag_vec, align_sample, reg_type, lambda_reg
    )
    
    # Retrieve MSE metrics (with and without Pandemic)
    MSE_oos_mmse                <- perf_obj$MSE_oos
    MSE_oos_mmse_without_covid  <- perf_obj$MSE_oos_without_covid
    MSE_oos_mean                <- perf_obj$MSE_mean_oos
    MSE_oos_mean_without_covid  <- perf_obj$MSE_mean_oos_without_covid
    
    # Compute rRMSEs
    
    # a. M-MSE vs. M-SSA components predictor
    #    Values below 1: M-MSE outperforms M-SSA (smoothness hurts)
    #    Values above 1: M-SSA outperforms M-MSE (smoothness helps)
    #    M-SSA MSEs carried over from exercise 1.3.5
    rRMSE_mmse_comp_mssa[shift + 1, k] <-
      sqrt(MSE_oos_mmse / MSE_oos_mssa_comp_mat[shift + 1, k])
    
    rRMSE_mmse_comp_mssa_without_covid[shift + 1, k] <-
      sqrt(MSE_oos_mmse_without_covid / MSE_oos_mssa_comp_without_covid_mat[shift + 1, k])
    
    # b. M-MSE vs. naive mean benchmark
    #    Values below 1: M-MSE outperforms the mean
    rRMSE_mmse_comp_mean[shift + 1, k] <-
      sqrt(MSE_oos_mmse / MSE_oos_mean)
    
    rRMSE_mmse_comp_mean_without_covid[shift + 1, k] <-
      sqrt(MSE_oos_mmse_without_covid / MSE_oos_mean_without_covid)
  }
}
close(pb)

# --- Assign Row and Column Names ---
col_names <- paste0("h=",     h_vec)
row_names <- paste0("Shift=", shift_vec)

colnames(rRMSE_mmse_comp_mssa) <- colnames(rRMSE_mmse_comp_mssa_without_covid) <-
  colnames(rRMSE_mmse_comp_mean) <- colnames(rRMSE_mmse_comp_mean_without_covid) <- col_names

rownames(rRMSE_mmse_comp_mssa) <- rownames(rRMSE_mmse_comp_mssa_without_covid) <-
  rownames(rRMSE_mmse_comp_mean) <- rownames(rRMSE_mmse_comp_mean_without_covid) <- row_names




# Out-of-Sample MSE Forecast Performance Comparison (excluding COVID-19 period)
# rRMSE < 1 indicates that the numerator predictor outperforms the denominator benchmark.

# M-MSE components vs. naive mean benchmark
rRMSE_mmse_comp_mean_without_covid

# M-SSA components vs. naive mean benchmark
rRMSE_mSSA_comp_mean_without_covid

# M-MSE vs. M-SSA components
# (rRMSE < 1: M-MSE outperforms M-SSA; rRMSE > 1: M-SSA outperforms M-MSE)
rRMSE_mmse_comp_mssa_without_covid

# =======================================================================
# Key finding:
#   Despite its substantially increased smoothness (larger HT, fewer zero-crossings),
#   M-SSA does not incur a significant loss in out-of-sample MSE forecast accuracy
#   relative to M-MSE. This is a notable result: smoothness and forecast accuracy are
#   not (or only marginally) in conflict here, which favors the M-SSA components predictor as the
#   preferred forecasting tool in this application (fewer spurious sign changes).

# But why do we prefer increased smoothness?
# To answer this question let us have a look at regressions of M-SSA and M-MSE on the 
#   target (forward-shifted BIP) where regression coefficients are based on the full sample
#   Note that M-SSA and M-MSE are based on data prior 2008 (this will be addressed in exercise 6 below)

# =======================================================================
# 5.3 Component Predictors: M-MSE vs. M-SSA
# =======================================================================

# =======================================================================
# 5.3.1 Compute M-MSE and M-SSA Component Predictors Based on Full-Sample Regressions
# =======================================================================
# Both predictors are computed using regression weights estimated on the full data sample
# (as opposed to the recursively re-estimated real-time predictors in Exercise 2).

# Select forecast horizon (must not exceed max(h_vec)) and forward shift
# We the practically relevant (and challenging) one-year ahead horizon
h <- 4
if (h > max(h_vec))
  h <- max(h_vec)
shift <- 4

# Data-set for regression: 
#   -we here include the COVID breakout when computing regression estimates
#   -In exercise 6 we work with adjusted data x_mat_wc (wc=without COVID)
#   -Removing the COVID data points is recommended since the outliers are strongly influential and 
#     not representative of the latent dynamics

# --- M-MSE Component Predictor ---
# Construct the data matrix for regression:
#   - Column 1: forward-shifted BIP target (aligned by 'shift' and publication lag)
#   - Remaining columns: selected M-MSE components at forecast horizon h
if (length(sel_vec_pred) > 1)
{
  dat <- cbind(
    c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
    t(mmse_array[sel_vec_pred, , h + 1])
  )
} else
{
  dat <- cbind(
    c(x_mat[(shift + lag_vec[1]+ 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
    mmse_array[sel_vec_pred, , h + 1]
  )
}


# OLS regression of forward-shifted BIP on selected M-MSE components
lm_obj          <- lm(dat[, 1] ~ dat[, 2:ncol(dat)])
summary(lm_obj)
optimal_weights <- lm_obj$coef

# Compute the fitted M-MSE component predictor (intercept + weighted combination of components)
if (length(sel_vec_pred) > 1)
{
  mmse_predictor <- optimal_weights[1] +
    dat[, 2:ncol(dat)] %*% optimal_weights[2:length(optimal_weights)]
} else
{
  mmse_predictor <- optimal_weights[1] +
    dat[, 2:ncol(dat)] * optimal_weights[2:length(optimal_weights)]
}

# --- M-SSA Component Predictor ---
# Analogous construction using M-SSA components in place of M-MSE components.
if (length(sel_vec_pred) > 1)
{
  dat <- cbind(
    c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
    t(mssa_array[sel_vec_pred, , h + 1])
  )
} else
{
  dat <- cbind(
    c(x_mat[(shift + lag_vec[1] + 1):nrow(x_mat), 1], rep(NA, shift + lag_vec[1])),
    mssa_array[sel_vec_pred, , h + 1]
  )
}


# OLS regression of forward-shifted BIP on selected M-SSA components
lm_obj          <- lm(dat[, 1] ~ dat[, 2:ncol(dat)])
summary(lm_obj)
optimal_weights <- lm_obj$coef

# Compute the fitted M-SSA component predictor
if (length(sel_vec_pred) > 1)
{
  mssa_predictor <- optimal_weights[1] +
    dat[, 2:ncol(dat)] %*% optimal_weights[2:length(optimal_weights)]
} else
{
  mssa_predictor <- optimal_weights[1] +
    dat[, 2:ncol(dat)] * optimal_weights[2:length(optimal_weights)]
}

# =======================================================================
# 5.3.2 Smoothness Comparison
# =======================================================================
#
# Holding time (HT) measures the average number of periods between successive zero-crossings
# of a centered series — a higher HT indicates a smoother, less noisy predictor.
#
# By design, M-SSA imposes a holding-time constraint in its optimization criterion,
# targeting an expected HT approximately 50% larger than that of M-MSE.
# We verify this here using empirical HTs computed from the estimated predictors.
#
# Notes:
#   1. The imposed HT constraint in M-SSA targets a 50% increase over the 'natural' M-MSE HT.
#      Ideally, the empirical HT of M-SSA should also be approximately 50% larger than M-MSE.
#   2. M-SSA controls zero-crossings at the mean level; both predictors are therefore
#      centered (scaled) before computing empirical HTs.

compute_empirical_ht_func(scale(mssa_predictor))
compute_empirical_ht_func(scale(mmse_predictor))


# Plot scaled predictors
par(mfrow = c(1, 1))
mplot<-scale(cbind(mssa_predictor, mmse_predictor))
colnames(mplot)<-c("M-SSA","M-MSE")
colo<-c("blue","green")
main_title<-"M-SSA vs. M-MSE Component Predictors"
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     lwd = c(2, rep(1, ncol(data) - 1)),
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
# Overlay all remaining indicators
for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
     labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
axis(2)
box()

#-------------------------------------------------------------------------------------------
# Observation: M-SSA exhibits approximately 33% fewer zero-crossings than M-MSE,
# corresponding to a roughly 50% larger empirical holding time — consistent with the
# imposed HT constraint.
# This result confirms that:
#   (a) the HT hyperparameter effectively controls predictor smoothness, and
#   (b) the empirical HT converges toward the imposed (expected) HT as the sample grows,
#       as demonstrated theoretically in Tutorial 7.1 under a known data-generating process.
#   (c) This convergence occurs if the (VAR) model is not (or only marginally) misspecified
#   (d) If the (VAR) model is misspecified, then sample HT and expected HT differ but M-SSA
#       remains smoother than M-MSE.
# Interestingly, M-SSA is as fast as M-MSE at troughs and peaks: 
#   Increased Smoothness does not harm Timeliness 
# Advantage of increased smoothness: clearer signals (less false alarms) at economic downturns or
#   during longer expansions
# We will explore this signaling ability in exercise 6 where we will compute ROCs and AUCs for 
#   competing predictors

# =======================================================================
# Summary of Findings Exercise 5
#
# Forecast accuracy (rRMSE):
#   - M-SSA and M-MSE component predictors achieve comparable out-of-sample MSE performance
#     when targeting forward-shifted BIP.
#   - Both predictors systematically outperform the naive mean benchmark, the direct forecast,
#     and the direct HP forecast at larger forward shifts (>2 quarters).
#
# Smoothness:
#   - The M-SSA component predictor is substantially smoother than M-MSE (fewer zero-crossings,
#     larger holding time).
#   - Smoothness is directly controllable via the HT hyperparameter in the M-SSA optimization
#     criterion, providing an interpretable and adjustable design lever.
#
# Key insight — smoothness without accuracy loss:
#   - The increased smoothness of M-SSA does not impair out-of-sample MSE forecast accuracy
#     or introduce substantial forecast lag (retardation) relative to M-MSE.
#   - This makes the M-SSA component predictor preferable to M-MSE in this application:
#     it delivers equivalent forecast accuracy with a cleaner, less noisy signal.
#   - Exercise 6 further illustrates and quantifies this advantage by examining the rate of
#     false alarms (spurious zero-crossings): ROC/AUC statistics.
#
# Generality of M-SSA:
#   - The M-MSE component predictor is a special case of the M-SSA component predictor:
#     inserting the M-MSE holding times into the M-SSA HT constraint exactly replicates
#     the M-MSE predictor.
#   - M-SSA is therefore the more general framework, offering explicit control over the
#     shape and smoothness characteristics of the predictor — beyond what MSE minimization
#     alone can achieve.



# =======================================================================
# Exercise 6: Compute Final M-SSA and M-MSE Component Predictors — Alarm Performance Analysis


# Overview:
#   This exercise extends the analysis of Exercises 4 and 5 in three directions:
#
#   1. Model updating:
#      Both the VAR model (used for M-SSA and M-MSE filter computation) and the
#      OLS regression model (used to construct the component predictors) are re-estimated
#      on the full available sample, replacing the fixed 2008 estimation cutoff used in
#      earlier exercises.
#      Notes:
#        - The COVID-19 pandemic period (2020–2021) is excluded throughout to prevent
#          outlier-driven distortion of both the VAR and regression estimates.
#        - The longer estimation sample now available allows the final regression step to
#          include multiple M-SSA components as regressors (rather than restricting to
#          M-SSA-BIP alone, as in earlier exercises). In the shorter in-sample span used
#          previously, this broader regressor set was avoided to guard against overfitting;
#          the extended sample provides sufficient observations to eventually support a richer
#          regression specification without this concern.
#
#   2. Smoothness gains — true vs. false alarms:
#      The exercise quantifies the practical benefit of M-SSA's additional smoothness
#      over M-MSE by analyzing the trade-off between true alarms (correctly signaled
#      turning points) and false alarms (spurious zero-crossings).
#      This is formalized via ROC curves and AUC statistics, providing a forecast quality
#      metric that complements — and goes beyond — the MSE and HAC-adjusted p-value comparisons 
#      of previous Exercises.
#
#   3. Forecast applications:
#      The final M-SSA and M-MSE component predictors are applied to two data variants:
#        (a) Standardized BIP and indicators (as used throughout)  → Exercise 6.2
#        (b) Original (unstandardized) BIP growth rates            → Exercise 6.3
#      Application (b) translates the predictor output to an interpretable economic scale,
#      enabling direct communication of effective BIPP growth forecasts.
# =======================================================================
# 6.1: Full-Sample M-SSA and M-MSE 
# =======================================================================
# Use all available data for VAR estimation (effectively no end-date restriction)
#   Choose any year larger than 2026
date_to_fit <- "3000"

# Remove pandemic years (2020 and 2021) to avoid distortion in VAR and regression estimates
#   Termination _wc: without COVID
x_mat_wc <- x_mat[c(which(rownames(x_mat) < 2020), which(rownames(x_mat) > 2021)), ]

# Plot the pandemic-corrected time series for visual inspection
par(mfrow = c(1, 1))
ts.plot(x_mat_wc)

# Run the M-SSA wrapper function (see Tutorial 7.2):
#   - Computes both M-SSA and M-MSE components for each forecast horizon h in h_vec
#   - Output includes component arrays indexed by selected variables, time, and horizon
final_mssa_indicator_obj <- compute_mssa_BIP_predictors_func(
  x_mat_wc, lambda_HP, L, date_to_fit, p, q,
  ht_mssa_vec, h_vec, f_excess, lag_vec, select_vec_multi
)

# Extract the final M-SSA component array from the wrapper output
final_mssa_array <- final_mssa_indicator_obj$mssa_array

# Extract the final M-MSE component array from the wrapper output
final_mmse_array <- final_mssa_indicator_obj$mmse_array

# M-SSA tracking HP-BIP at horizons h=0 (nowcast),...,h=6 (1.5 years ahead) based on full-sample VAR 
tail(final_mssa_array["BIP",,])
# M-SSA tracking HP-ifo at horizons h=0 (nowcast),...,h=6 (1.5 years ahead) based on full-sample VAR 
tail(final_mssa_array["ifo_c",,])

# =======================================================================
# 6.2 Compute Full-Sample Predictors (Regressions)
# =======================================================================

# For each predictor, we use all five indicators
#   -Common ground for evaluation of predictors
#   -Full sample is sufficiently large (~130 observations in 2025) 
#     to accommodate regression with 5 explanatory variables

# Select data matrix for ROC computation: full sample without pandemic
data_roc<-x_mat_wc

#--------------------------------------------------------
# 6.2.1 Direct forecast
#--------------------------------------------------------
select_direct_indicator<-colnames(x_mat_wc)
direct_forecast_mat<-forward_shifted_BIP_mat<-NULL
for (h in 0:max(h_vec))
{
  # Shift BIP forward by publication lag+forecast horizon
  forward_shifted_BIP<-c(data_roc[(1+lag_vec[1]+h):nrow(data_roc),"BIP"],rep(NA,h+lag_vec[1]))
  forward_shifted_BIP_mat<-cbind(forward_shifted_BIP_mat,forward_shifted_BIP)
  # Regress selected indicators on forward-shifted BIP: start at t=L (same sample as HP-C)
  lm_obj<-lm(forward_shifted_BIP[L:length(forward_shifted_BIP)]~data_roc[L:nrow(data_roc),select_direct_indicator])
  summary(lm_obj)
  # Compute the predictor: one can rely on the generic R-function predict or compute the predictor manually
  direct_forecast<-lm_obj$coef[1]+data_roc[,select_direct_indicator]%*%lm_obj$coef[2:(length(select_direct_indicator)+1)]
  # Note that this is a full-sample predictor (no out-of-sample span)
  
  # We can now plot target and direct forecast: for h>2 the predictor comes close to a flat line centered at zero
  direct_forecast_mat<-cbind(direct_forecast_mat,direct_forecast)
}
colnames(direct_forecast_mat)<-colnames(forward_shifted_BIP_mat)<-paste("h=",h_vec,sep="")
rownames(forward_shifted_BIP_mat)<-rownames(data_roc)
# BIP shifted forward by publication + shift
tail(forward_shifted_BIP_mat)


#------------------------------------------------------
# 6.2.2 Direct HP forecast
#------------------------------------------------------


direct_hp_forecast_mat<-NULL
hp_mat<-NULL
for (i in 1:ncol(data_roc))
  hp_mat<-cbind(hp_mat,filter(data_roc[,i],hp_c,side=1))
colnames(hp_mat)<-select_vec_multi

for (h in 0:max(h_vec))
{
  # Shift BIP forward by publication lag+forecast horizon
  forward_shifted_BIP<-c(data_roc[(1+lag_vec[1]+h):nrow(data_roc),"BIP"],rep(NA,h+lag_vec[1]))
  lm_obj<-lm(forward_shifted_BIP~hp_mat[,select_direct_indicator])
  # Compute the predictor: one can rely on the generic R-function predict or compute the predictor manually
  direct_hp_forecast_mat<-cbind(direct_hp_forecast_mat,lm_obj$coef[1]+hp_mat[,select_direct_indicator]%*%lm_obj$coef[2:(length(select_direct_indicator)+1)])
}
colnames(direct_hp_forecast_mat)<-paste("h=",h_vec,sep="")
tail(direct_hp_forecast_mat)

#---------------------------------------------------------------------
# 6.2.3 M-SSA and M-MSE Component Predictors: 
#   Use All M-SSA Indicators
#---------------------------------------------------------------------
final_mssa_array<-final_mssa_indicator_obj$mssa_array
final_mmse_array<-final_mssa_indicator_obj$mmse_array
mssa_mat<-mmse_mat<-NULL
# Select the same indicators (though smoothed by M-SSA)
sel_vec_pred<-select_direct_indicator
# Note on regression design:
#   The M-SSA and M-MSE component predictors used here are each optimized for forecast
#   horizon h = shift, and are regressed on BIP forward-shifted by the same value of shift.
#   This corresponds to selecting the diagonal entries of the performance matrices computed
#   in Exercise 1, where rows index forward shifts and columns index forecast horizons —
#   ensuring that filter horizon and target horizon are aligned.
for (h in 0:max(h_vec))
{
  # Shift BIP forward by publication lag+forecast horizon
  forward_shifted_BIP<-c(data_roc[(1+lag_vec[1]+h):nrow(data_roc),"BIP"],rep(NA,h+lag_vec[1]))
  lm_obj<-lm(forward_shifted_BIP~t(final_mssa_array[sel_vec_pred,,h+1]))
  # Compute the predictor: one can rely on the generic R-function predict or compute the predictor manually
  mssa_mat<-cbind(mssa_mat,lm_obj$coef[1]+t(final_mssa_array[sel_vec_pred,,h+1])%*%lm_obj$coef[2:(length(select_direct_indicator)+1)])
  lm_obj<-lm(forward_shifted_BIP~t(final_mmse_array[sel_vec_pred,,h+1]))
  mmse_mat<-cbind(mmse_mat,lm_obj$coef[1]+t(final_mmse_array[sel_vec_pred,,h+1])%*%lm_obj$coef[2:(length(select_direct_indicator)+1)])
}
colnames(mssa_mat)<-colnames(mmse_mat)<-paste("shift=",h_vec,sep="")
tail(mssa_mat)
#-------------------------------------------------
# 6.2.4. Forward-Shifted HP-BIP (Smooth Acausal Target)
#-------------------------------------------------

forward_shifted_HP_BIP_mat<-NULL
# Apply two-sided HP to BIP shifted forward by publication lag
hp_bip<-c(rep(NA,lag_vec[1]),filter(data_roc[(1+lag_vec[1]):nrow(data_roc),"BIP"],hp_two,side=2))

for (h in 0:max(h_vec))
{
# Shift HP-BIP forward by forecast horizon
  forward_shifted_HP_BIP_mat<-cbind(forward_shifted_HP_BIP_mat,c(hp_bip[(1+h):length(hp_bip)],rep(NA,h)))
}
rownames(forward_shifted_HP_BIP_mat)<-rownames(data_roc)
colnames(forward_shifted_HP_BIP_mat)<-paste("shift=",h_vec,sep="")
# HP-BIP shifted forward: HP-BIP does not extend to sample end
tail(forward_shifted_HP_BIP_mat[1:(nrow(forward_shifted_HP_BIP_mat)-L+lag_vec[1]),])

#-------------------------------------------------
# 6.2.5: Plot Final M-MSE and M-SSA Component Predictors ---
#-------------------------------------------------
# Panel 1: Standardized forward-shifted BIP alongside both predictors
#   - All three series are scaled for visual comparability
#   - The pandemic exclusion period is noted in the title
par(mfrow = c(2, 1))

shift<-4

# Standardize and combine: forward-shifted BIP, M-SSA predictor, M-MSE predictor
mplot <- scale(cbind(
  c(x_mat_wc[(shift + lag_vec[1] + 1):nrow(x_mat_wc), 1], rep(NA, shift + lag_vec[1])),
  mssa_mat[,shift+1],
  mmse_mat[,shift+1]
))
colnames(mplot) <- c(
  paste("BIP shifted forward by ", shift, " (plus publication lag)", sep = ""),
  "M-SSA component predictor",
  "M-MSE component predictor"
)
colo <- c("black", "blue", "green")

# Plot Panel 1: standardized BIP target and both predictors overlaid
main_title <- paste("Forward-shifted BIP and Predictors: Pandemic episode removed", sep = "")
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (jj in 1:ncol(mplot)) {
  lines(mplot[, jj], col = colo[jj], lwd = 1, lty = 1)
  mtext(colnames(mplot)[jj], col = colo[jj], line = -jj)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# Panel 2: Direct comparison of M-SSA vs. M-MSE predictor (unscaled)
#   - A zero baseline is included for reference
#   - Title reflects the chosen forecast horizon h and forward shift
mplot <- cbind(
  rep(0, nrow(as.matrix(mssa_mat))),
  mssa_mat[,shift+1],
  mmse_mat[,shift+1]
)
colnames(mplot) <- c("", "M-SSA component predictor", "M-MSE component predictor")

main_title <- paste("Predictors: M-SSA component vs. M-MSE component, h=", h,
                    ", shift=", shift, sep = "")
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (jj in 1:ncol(mplot)) {
  lines(mplot[, jj], col = colo[jj], lwd = 1, lty = 1)
  mtext(colnames(mplot)[jj], col = colo[jj], line = -jj)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

#--------------------------------------------------
# Notably, despite its increased smoothness, the M-SSA component predictor (blue) does not
# lag the M-MSE predictor (green): both predictors track the forward-shifted BIP
#--------------------------------------------------

#-------------------------------------------------
# 6.2.6: Smoothness Comparison ---
#-------------------------------------------------
# Visual inspection of Panel 2 suggests that the M-SSA component predictor is smoother
# than its M-MSE counterpart, without introducing any additional phase lag at troughs or peaks.
# The empirical holding times (HTs) are computed below to confirm this observation:
#   - A higher HT indicates a smoother, less oscillatory series.

compute_empirical_ht_func( mssa_mat[,shift+1])   # Expected: higher HT (smoother)
compute_empirical_ht_func( mmse_mat[,shift+1])   # Expected: lower HT (less smooth)

# What is the benefit of a larger HT?

# =======================================================================
# 6.3 ROC and AUC: True vs. False Alarms
# =======================================================================
# Directional Forecast Evaluation: ROC and AUC Analysis
#
# Objective:
#   We assess the ability of each predictor to correctly forecast the sign (direction)
#   of future BIP growth and future HP-BIP — i.e., to distinguish expansion phases
#   from contraction phases at different forecast horizons.
#
# Predictors compared:
#   All predictors are estimated on the full sample:
#     - Direct forecast         (unfiltered indicators)
#     - Direct HP forecast      (univariate HP-C filtered indicators)
#     - M-MSE component predictor
#     - M-SSA component predictor
#
# Evaluation methodology:
#   For each predictor, we compute:
#     - The ROC (Receiver Operating Characteristic) curve, tracing the trade-off between
#       true alarm rate and false alarm rate across all possible decision thresholds.
#     - The AUC (Area Under the ROC Curve) statistic, summarizing overall directional
#       forecast accuracy in a single scalar measure (AUC = 1: perfect; AUC = 0.5: no better than chance).
#   Evaluation is conducted at two forecast horizons:
#     - One quarter ahead  (short horizon)
#     - One year ahead     (long horizon)
#
# Prior expectations:
#   1. Short horizon (one quarter ahead):
#      The direct forecast is expected to perform well — possibly best — at this horizon,
#      since unfiltered indicators retain high-frequency information relevant for
#      near-term directional prediction.
#   2. Long horizon (one year ahead):
#      M-SSA and M-MSE component predictors are expected to outperform the direct and
#      direct HP forecasts, as their multivariate filtering is specifically designed to
#      extract the low-frequency cyclical signal relevant at longer horizons.
#
# Key question:
#   Beyond the MSE comparison of Exercise 5, what is the specific advantage — in terms
#   of directional forecast accuracy and false alarm reduction — of imposing a larger
#   holding-time (HT) constraint in M-SSA relative to M-MSE?


#-------------------------------------------------
# 6.3.1 Target: Forward-Shifted BIP
#-------------------------------------------------

# Select either BIP pr HP-BIP
select_target<-"BIP"

smoothROC<-T
showROC<-T
# Labels: hit rate vs. false alarm rate
lbls = "Hit"
# Size of legend box in plot
lg_cex<-0.5

par(mfrow=c(1,2))

# One quarter ahead
shift<-1

# Select target: shifted BIP or HP-BIP
if (select_target=="BIP")
{
  target<-as.integer(forward_shifted_BIP_mat[,shift+1]>0)
} else
{
  target<-as.integer(forward_shifted_HP_BIP_mat[,shift+1]>0)
}
# Set up data matrix for ROC calculation    
ROC_data<-cbind(target,direct_forecast_mat[,shift+1],direct_hp_forecast_mat[,shift+1],mmse_mat[,shift+1],mssa_mat[,shift+1])
rownames(ROC_data)<-rownames(data_roc)
colnames(ROC_data)<-c("Target","Direct forecast","Direct HP forecast","M-MSE","M-SSA")
ROC_data<-as.data.frame(na.exclude(ROC_data))
# Select plots that will be shown    
showLegend<-T
AUC<-ROCplots(ROC_data, showROC , main = "One quarter ahead", lbls = lbls,
              smoothROC , colours =NULL, lwd = 2,
              showLegend , lg_cex = 1, lg_ncol = 1)
AUC_table<-AUC$AUC

# One year ahead
shift<-4

# Select target: shifted BIP or HP-BIP
if (select_target=="BIP")
{
# Compute a binary indicator based on the sign  
  target<-as.integer(forward_shifted_BIP_mat[,shift+1]>0)
} else
{
# Compute a binary indicator based on the sign  
  target<-as.integer(forward_shifted_HP_BIP_mat[,shift+1]>0)
}
# Set up data matrix for ROC calculation    
ROC_data<-cbind(target,direct_forecast_mat[,shift+1],direct_hp_forecast_mat[,shift+1],mmse_mat[,shift+1],mssa_mat[,shift+1])
rownames(ROC_data)<-rownames(data_roc)
colnames(ROC_data)<-c("Target","Direct forecast","Direct HP forecast","M-MSE","M-SSA")
ROC_data<-as.data.frame(na.exclude(ROC_data))
# Select plots that will be shown    
showLegend<-F
AUC<-ROCplots(ROC_data, showROC , main = "One year ahead", lbls = lbls,
              smoothROC , colours =NULL, lwd = 2,
              showLegend , lg_cex = 1, lg_ncol = 1)

AUC_table<-cbind(AUC_table,AUC$AUC)

#-------------------------------------------------
# 6.3.2 Forward-Shifted HP-BIP
#-------------------------------------------------
select_target<-"HP-BIP"

par(mfrow=c(1,2))

# One quarter ahead
shift<-1

# Select target: shifted BIP or HP-BIP
if (select_target=="BIP")
{
  target<-as.integer(forward_shifted_BIP_mat[,shift+1]>0)
} else
{
  target<-as.integer(forward_shifted_HP_BIP_mat[,shift+1]>0)
}
# Set up data matrix for ROC calculation    
ROC_data<-cbind(target,direct_forecast_mat[,shift+1],direct_hp_forecast_mat[,shift+1],mmse_mat[,shift+1],mssa_mat[,shift+1])
rownames(ROC_data)<-rownames(data_roc)
colnames(ROC_data)<-c("Target","Direct forecast","Direct HP forecast","M-MSE","M-SSA")
ROC_data<-as.data.frame(na.exclude(ROC_data))
# Select plots that will be shown    
showLegend<-T
AUC<-ROCplots(ROC_data, showROC , main = "One quarter ahead", lbls = lbls,
              smoothROC , colours =NULL, lwd = 2,
              showLegend , lg_cex = 1, lg_ncol = 1)
AUC_table<-cbind(AUC_table,AUC$AUC)

# One year ahead
shift<-4

# Select target: shifted BIP or HP-BIP
if (select_target=="BIP")
{
  target<-as.integer(forward_shifted_BIP_mat[,shift+1]>0)
} else
{
  target<-as.integer(forward_shifted_HP_BIP_mat[,shift+1]>0)
}
# Set up data matrix for ROC calculation    
ROC_data<-cbind(target,direct_forecast_mat[,shift+1],direct_hp_forecast_mat[,shift+1],mmse_mat[,shift+1],mssa_mat[,shift+1])
rownames(ROC_data)<-rownames(data_roc)
colnames(ROC_data)<-c("Target","Direct forecast","Direct HP forecast","M-MSE","M-SSA")
ROC_data<-as.data.frame(na.exclude(ROC_data))
# Select plots that will be shown    
showLegend<-F
AUC<-ROCplots(ROC_data, showROC , main = "One year ahead", lbls = lbls,
              smoothROC , colours =NULL, lwd = 2,
              showLegend , lg_cex = 1, lg_ncol = 1)
AUC_table<-cbind(AUC_table,AUC$AUC)

colnames(AUC_table)<-c("BIP 1-Quarter","BIP 1-Year","HP-BIP 1-Quarter","HP-BIP 1-Year")
rownames(AUC_table)<-c("Direct","Direct HP","M-MSE","M-SSA")
AUC_table



# =======================================================================
# Exercise 6.4: Apply Predictor to Original (Unstandardized) BIP Growth 
# =======================================================================
# Purpose: Generate forecasts of the effective (real-scale) BIP growth rate,
#          This yields predictions in interpretable, original units.

# Select practically relevant one-year horizon
h<-4
shift<-h
# Aggressive selection: all indicators
sel_vec_pred<-select_vec_multi
# Inspection of the regression equation below reveals that only M-SSA-BIP and 
# M-SSA-ip are significant. But all (original/unfiltered) indicators (including, 
# ifo,ESI and spread) enter into the VAR-model and hence into M-SSA-BIP and M-SSA-ip

# Note
# The weight on M-SSA-BIP is positive and that on M-SSA-ip is negative.
# The negative weight on ip ensures impressive leads (left-shift) but it 
# is a bit difficult to interpret (other than minimizing MSE).
# Therefore the simpler `conservative' single M-SSA-BIP retains some appeal, 
# in particular when interpretablity imports.

# Conservative selection: M-SSA-BIP only
sel_vec_pred<-"BIP"


# --- 6.4.1: Load and Prepare Original BIP Data ---

# Data file names: monthly indicator data and quarterly BIP data
data_file_name <- c("Data_HWI_2025_02.csv", "gdp_2025_02.csv")

# Load quarterly BIP data; BIP is stored in the first data column
data_quarterly <- read.csv(paste(getwd(), "/Data/", data_file_name[2], sep = ""))
BIP_original <- data_quarterly[, "BIP"]


# --- Plot 1: Original BIP Level ---
par(mfrow = c(2, 1))
mplot <- matrix(BIP_original)
rownames(mplot) <- data_quarterly[, "Date"]
colnames(mplot) <- "Original BIP"
colo <- c("black", "blue", "green")

main_title <- "Original BIP"
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# Compute log-differenced BIP (i.e., approximate quarterly growth rate)
# NA introduced by differencing is removed via na.exclude()
diff_log_BIP <- na.exclude(diff(log(BIP_original)))
names(diff_log_BIP) <- rownames(data)

# --- Plot 2: First Differences of Log-BIP (Growth Rates) ---
mplot <- matrix(diff_log_BIP)
rownames(mplot) <- names(diff_log_BIP)
colnames(mplot) <- "diff-log BIP"
colo <- c("black", "blue", "green")

main_title <- "First Differences of Original Log-BIP (Quarterly Growth Rates)"
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()


# --- Consistency Check: Verify Alignment of x_mat BIP with Scaled diff-log BIP ---
#
# Confirms that x_mat[,"BIP"] (used throughout the analysis) matches the
# standardized diff-log BIP, up to the trimming applied during data preparation.
# Any discrepancy would indicate a mismatch in sample alignment or scaling.
par(mfrow = c(1, 1))
mplot <- cbind(scale(diff_log_BIP), x_mat[, "BIP"])
colnames(mplot) <- c("scaled diff-log BIP", "trimmed and scaled diff-log BIP")
colo <- c("black", "blue", "green")

main_title <- "Consistency Check: Trimmed vs. Untrimmed Scaled diff-log BIP"
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (jj in 1:ncol(mplot)) {
  lines(mplot[, jj], col = colo[jj], lwd = 1, lty = 1)
  mtext(colnames(mplot)[jj], col = colo[jj], line = -jj)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()


# --- Construct Analysis Dataset with Original (Unstandardized) diff-log BIP ---
#
# Replace the standardized BIP column in x_mat with the raw diff-log BIP values.
# All other indicator columns remain unchanged.
x_mat_original_BIP <- x_mat
x_mat_original_BIP[, "BIP"] <- diff_log_BIP

# Remove pandemic years (2020–2021) to prevent distortion of VAR and regression estimates
x_mat_original_BIP_wc <- x_mat_original_BIP[
  c(which(rownames(x_mat_original_BIP) < 2020),
    which(rownames(x_mat_original_BIP) > 2021)), ]




# --- 6.4.2: M-MSE Component Predictor Optimized for Forecast Horizon h ---
#
# Applies OLS regression to predict original
# (unstandardized) BIP growth using selected M-MSE components at horizon h.
# The regression target is BIP shifted forward by 'shift' quarters to account
# for the publication lag.

# Construct the regression dataset:
#   Column 1: forward-shifted original diff-log BIP (target), padded with trailing NAs
#   Remaining columns: selected M-MSE components at horizon h
if (length(sel_vec_pred) > 1) {
  dat <- cbind(
    c(x_mat_original_BIP_wc[(shift + lag_vec[1] + 1):nrow(x_mat_original_BIP_wc), 1],
      rep(NA, shift + lag_vec[1])),
    t(final_mmse_array[sel_vec_pred, , h + 1])
  )
} else {
  dat <- cbind(
    c(x_mat_original_BIP_wc[(shift + lag_vec[1] + 1):nrow(x_mat_original_BIP_wc), 1],
      rep(NA, shift + lag_vec[1])),
    (final_mmse_array[sel_vec_pred, , h + 1])
  )
}


# Run OLS regression of forward-shifted original BIP on selected M-MSE components
lm_obj <- lm(dat[, 1] ~ dat[, 2:ncol(dat)])
summary(lm_obj)

# Store estimated regression coefficients (intercept + component weights)
optimal_weights <- lm_obj$coef

# Compute the M-MSE BIP predictor as a linear combination of selected components
if (length(sel_vec_pred) > 1) {
  final_mmse_bip_predictor <- optimal_weights[1] +
    dat[, 2:ncol(dat)] %*% optimal_weights[2:length(optimal_weights)]
} else {
  final_mmse_bip_predictor <- optimal_weights[1] +
    dat[, 2:ncol(dat)] * optimal_weights[2:length(optimal_weights)]
}


# --- 6.4.3: M-SSA Component Predictor Optimized for Forecast Horizon h ---
#
# Mirrors the M-MSE regression above (Section 6.3.2), but uses M-SSA components
# as regressors instead. The target remains the forward-shifted original diff-log BIP.

# Construct the regression dataset:
#   Column 1: forward-shifted original diff-log BIP (target), padded with trailing NAs
#   Remaining columns: selected M-SSA components at horizon h
if (length(sel_vec_pred) > 1) {
  dat <- cbind(
    c(x_mat_original_BIP_wc[(shift + lag_vec[1] + 1):nrow(x_mat_original_BIP_wc), 1],
      rep(NA, shift + lag_vec[1])),
    t(final_mssa_array[sel_vec_pred, , h + 1])
  )
} else {
  dat <- cbind(
    c(x_mat_original_BIP_wc[(shift + lag_vec[1] + 1):nrow(x_mat_original_BIP_wc), 1],
      rep(NA, shift + lag_vec[1])),
    (final_mssa_array[sel_vec_pred, , h + 1])
  )
}


# Run OLS regression of forward-shifted original BIP on selected M-SSA components
lm_obj <- lm(dat[, 1] ~ dat[, 2:ncol(dat)])
summary(lm_obj)

# Store estimated regression coefficients (intercept + component weights)
optimal_weights <- lm_obj$coef

# Compute the M-SSA BIP predictor as a linear combination of selected components
if (length(sel_vec_pred) > 1) {
  final_mssa_bip_predictor <- optimal_weights[1] +
    dat[, 2:ncol(dat)] %*% optimal_weights[2:length(optimal_weights)]
} else {
  final_mssa_bip_predictor <- optimal_weights[1] +
    dat[, 2:ncol(dat)] * optimal_weights[2:length(optimal_weights)]
}


# --- 6.4.4: Plot Final M-MSE and M-SSA BIP Predictors (Original Scale) ---
#
# Panel 1: Forward-shifted original BIP growth alongside both predictors (unscaled)
#   - Unlike Exercise 6.2, no standardization is applied here, so values are
#     interpretable as actual BIP growth rates.
par(mfrow = c(2, 1))

# Combine target and predictors into a single matrix for plotting
mplot <- cbind(
  c(x_mat_original_BIP_wc[(shift + lag_vec[1] + 1):nrow(x_mat_original_BIP_wc), 1],
    rep(NA, shift + lag_vec[1])),
  final_mssa_bip_predictor,
  final_mmse_bip_predictor
)
colnames(mplot) <- c(
  paste("BIP shifted forward by ", shift, " (plus publication lag)", sep = ""),
  "M-SSA component predictor",
  "M-MSE component predictor"
)
rownames(mplot)<-rownames(x_mat_original_BIP_wc) 
colo <- c("black", "blue", "green")

# Plot Panel 1: original-scale BIP target with M-SSA and M-MSE predictors overlaid
main_title <- "Forward-shifted BIP and Predictors: Pandemic Episode Removed (Original Scale)"
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (jj in 1:ncol(mplot)) {
  lines(mplot[, jj], col = colo[jj], lwd = 1, lty = 1)
  mtext(colnames(mplot)[jj], col = colo[jj], line = -jj)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# Panel 2: Direct comparison of M-SSA vs. M-MSE BIP predictor (original scale)
#   - Zero baseline included for reference
#   - Title reflects the chosen forecast horizon h and forward shift
mplot <- cbind(
  rep(0, nrow(as.matrix(final_mssa_bip_predictor))),
  final_mssa_bip_predictor,
  final_mmse_bip_predictor
)
colnames(mplot) <- c("", "M-SSA component predictor", "M-MSE component predictor")
rownames(mplot)<-rownames(x_mat_original_BIP_wc) 

main_title <- paste("Predictors: M-SSA vs. M-MSE Component (Original BIP Scale), h=",
                    h, ", shift=", shift, sep = "")
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (jj in 1:ncol(mplot)) {
  lines(mplot[, jj], col = colo[jj], lwd = 1, lty = 1)
  mtext(colnames(mplot)[jj], col = colo[jj], line = -jj)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()












