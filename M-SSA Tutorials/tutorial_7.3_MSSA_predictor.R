# ==================================================================
# Tutorial 7.3: M-SSA Predictor Designs for German GDP (BIP)
# ==================================================================
#
# Background:
#   - Tutorial 7.2 introduced the concept of M-SSA predictors for BIP and
#     wrapped the full workflow into a single reusable function.
#   - This tutorial applies that function (and the underlying rationale) 
#     to evaluate and compare multiple M-SSA predictor designs defined by 
#     different hyperparameter choices.
#
# Structure: four exercises
#   Exercise 1: A "fairly adaptive" design targeting HP(160)-filtered BIP.
#     - HP(160) deviates from the standard quarterly specification HP(1600).
#       See Phillips & Jin (2021) for a critique arguing HP(1600) is too smooth.
#       See also the detailed discussion in Tutorial 7.2 and Exercise 5 below.
#       (HP(160) retrieves more interesting signals than HP(1600) for one-year 
#       ahead forecasting).
#     - M-SSA can predict HP-BIP (its explicit optimization target) with
#       statistically significant accuracy multiple quarters ahead.
#     - Predicting raw BIP is harder due to unpredictable high-frequency noise.
#
#   Exercise 2: Apply M-SSA to white-noise data as a falsification test.
#     - Verifies that the proposed performance measures and HAC-adjusted tests
#       correctly reflect unpredictability under the null.
#     - The HAC adjustment does not fully eliminate all data idiosyncrasies,
#       but empirical significance levels do not appear to be strongly biased.
#
#   Exercise 3: A "more adaptive" design targeting HP(16)-filtered BIP.
#
#   Exercise 4: A "more inflexible" design targeting the standard HP(1600).
#
#   Note: Fine-tuning adaptivity through lambda_HP further may yield additional improvements.
#
# Main Purposes of This Tutorial:
#   1. Illustrate M-SSA applied to real macroeconomic data.
#      (Tutorial 7.1 used simulated data; this tutorial uses German GDP and indicators.)
#      - Application: nowcasting and forecasting German GDP (BIP) using a small
#        set of well-known monthly/quarterly indicators.
#
#   2. Demonstrate mid-term forecasting up to 6 quarters (1.5 years) ahead.
#      - Performances of institutional forecasters ("big five" German research institutes)
#        degrade steeply beyond a one-quarter horizon (see Heinisch & Neufing, forthcoming).
#      - We illustrate that consistent BIP predictability beyond half a year is achievable.
#      - Note: Institutional forecasters excel at nowcasting GDP, outperforming M-SSA here,
#        by exploiting a rich cross-section of many series and mixed-frequency (monthly/quarterly)
#        linkages. M-SSA here uses a deliberately small set of key indicators in a purely
#        quarterly framework. High-frequency data tends to be too noisy to be of relevance 
#        multiple quarters ahead.
#      - This tutorial provides novel insights into mid-term GDP (i.e., not only German BIP) forecasting.
#
#   3. Forecast strategy: raw BIP and HP-filtered BIP trend growth (HP-BIP).

#      What is HP-BIP?
#      - HP-BIP is obtained by applying a two-sided HP filter (with lambda_HP<1600) 
#        to log-differenced BIP.
#      - It captures the smooth mid-term trend-growth component of BIP, abstracting
#        from erratic and largely unpredictable short-run fluctuations.
#      - Importantly, we are not targeting the business cycle in the classical sense
#        (i.e., deviations from a long-run trend as in HP(1600)), but rather
#        mid-term growth dynamics, which requires a more adaptive filter (lambda_HP < 1600).
#
#      Why forecast HP-BIP rather than raw BIP directly?
#      - Mid-term trend growth is substantially easier to forecast than raw BIP,
#        because the high-frequency noise component of raw BIP is largely unpredictable.
#      - Tracking HP-BIP is therefore the explicit M-SSA optimization objective here:
#        M-SSA is designed and evaluated primarily against this smoother target.
#
#      What does HP-BIP predictability imply for raw BIP?
#      - If mid-term trend growth (HP-BIP) constitutes a non-trivial share of the
#        variation in raw BIP (i.e., raw BIP is not white noise), then a predictor
#        of HP-BIP is also informative about future raw BIP.
#      - In practice, this means we can partially predict raw BIP by predicting
#        its smooth component, while accepting that the residual high-frequency
#        noise remains unforecastable.
#
#   - FED Chair Jerome Powell, speaking at the University of Chicago Booth
#     School of Business (March 7, 2025), stated:
#       "As we parse the incoming information, we are focused on separating
#        the signal from the noise as the outlook evolves,"
#
#
#   4. Evaluate forecast performance using three complementary metrics:
#      a) Target correlation: correlation between predictor and forward-shifted
#         HP-BIP (or BIP).
#      b) rRMSE: relative root mean-square error benchmarked against classic
#         direct predictors or the simple mean(BIP) benchmark.
#      c) HAC-adjusted p-values: from regressions of predictors on targets,
#         with HAC correction for autocorrelation and heteroscedasticity
#         in regression residuals.
#      (To do: add Diebold-Mariano (DM) and Giacomini-White (GW) tests of
#       unequal predictability, benchmarked against mean(BIP).)
#
# Design Note:
#   - The M-SSA predictor proposed here is optimized to track HP-BIP (the
#     smooth trend-growth component of BIP).
#   - It is primarily intended to detect dynamic turning points (recessions,
#     crises, expansions) in trend growth.
#   - Explicit optimization for raw BIP prediction is not the primary goal here.
#   - Tutorial 7.4 presents a more refined "M-SSA components predictor" that
#     exploits M-SSA components more effectively for direct multi-horizon BIP prediction.
# ==================================================================
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


# Start with a clean workspace
rm(list = ls())


# ==================================================================
# Load Required R Libraries
# ==================================================================

library(mFilter)    # Standard HP and band-pass filter routines
library(MTS)        # Multivariate time series: VARMA modelling for macro indicators
# (used here primarily for VAR simulation and estimation)
library(sandwich)   # HAC-robust variance estimation under autocorrelation
# and heteroscedasticity (Newey-West type)
library(xts)        # Extended time series infrastructure
library(multDM)     # Diebold-Mariano test of equal predictive accuracy
library(fGarch)     # GARCH modelling (for improved regression residual modelling)


# ==================================================================
# Load Custom M-SSA Function Libraries
# ==================================================================

# Core M-SSA filter construction and optimization routines
source(paste(getwd(), "/R/functions_MSSA.r", sep = ""))

# HP filter utilities used in the JBCY paper (relies on mFilter)
source(paste(getwd(), "/R/HP_JBCY_functions.r", sep = ""))

# M-SSA utility functions: data preparation, plotting helpers, wrappers
source(paste(getwd(), "/R/M_SSA_utility_functions.r", sep = ""))

# Performance metrics and tests of unequal predictability (currently inactive)
# source(paste(getwd(), "/R/performance_statistics_functions.r", sep = ""))


# ==================================================================
# Load and Inspect Data
# ==================================================================
# Two vintages are available for comparison of data revisions:
#   - macro_2025: real-time data up to early 2025
#     (used in Heinisch, van Norden & Wildi (2026))
#   - macro_2026: updated data up to early 2026 (one additional year)
# Try anyone and make the real-time comparison test for the last year of data

load(file = paste(getwd(), "/Data/macro_2025", sep = ""))
load(file = paste(getwd(), "/Data/macro_2026", sep = ""))

# Inspect the most recent observations in the loaded data
tail(data)

# Publication lag assumption:
#   - In practice, BIP is published with approximately a 1-quarter delay.
#   - We adopt a conservative 2-quarter delay to build in a safety margin that
#     accounts for data revisions: early BIP releases are subject to potentially
#     substantial revisions, and a better (revised) figure may only be available
#     after two quarters.
#   - Since this tutorial does not explicitly model revision effects, the extra
#     quarter of lag serves as a pragmatic buffer against look-ahead bias.
#   - Concretely: the target series (first column of x_mat) is the BIP series
#     shifted forward by 2 quarters relative to the raw BIP column in data,
#     reflecting what a forecaster can actually observe at each point in time.
#   - All other indicators (ip, ifo_c, ESI, spr_10y_3m) are assumed to be
#     available contemporaneously, i.e., without any publication delay.
lag_vec <- c(2, rep(0, ncol(data) - 1))


# ==================================================================
# Plot All Indicators
# ==================================================================
# The raw BIP series (red) lags the publication-lag-adjusted target (black)
# by lag_vec[1] = 2 quarters, reflecting the assumed release delay.

par(mfrow = c(1, 1))
mplot <- data
colo <- c("black", rainbow(ncol(data) - 1))
main_title <- paste("Quarterly BIP data: target (black) assumes a publication lag of ",
                    lag_vec[1], " quarters", sep = "")

# Base plot: publication-lag-adjusted BIP target in black
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     lwd = c(2, rep(1, ncol(data) - 1)),
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
# Overlay all remaining indicators in distinct colors
for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

abline(h = 0) # Reference line at zero
axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
     labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
axis(2)
box()


# ==================================================================
# Select Indicators for M-SSA
# ==================================================================
# Five key macro-financial indicators are selected:
#   BIP       : German GDP growth (log-differenced, quarterly)
#   ip        : Industrial production growth
#   ifo_c     : ifo Business Climate Index (current conditions)
#   ESI       : Economic Sentiment Indicator (European Commission)
#   spr_10y_3m: Term spread (10-year minus 3-month government bond yield)

select_vec_multi <- c("BIP", "ip", "ifo_c", "ESI", "spr_10y_3m")
x_mat <- data[, select_vec_multi]
rownames(x_mat) <- rownames(data)

# Number of series and total number of observations
n   <- dim(x_mat)[2]
len <- dim(x_mat)[1]

# ==================================================================
# Exercise 1: Multi-Horizon GDP Forecasting with M-SSA (HP(160) Target)
# ==================================================================
#
# Objective:
#   Compute M-SSA predictors for German GDP (BIP) at forecast horizons
#   0 to 6 quarters ahead, using a "fairly adaptive" HP(160) filter as
#   the optimization target.
#
# Motivation for HP(160) as the target:
#   - The standard HP(1600) filter is too smooth for quarterly GDP forecasting:
#     it tends to wash out recession episodes, obscuring the very dynamics
#     we wish to predict at the one-year horizon.
#   - Raw BIP is too noisy to be forecast directly: its high-frequency
#     component is largely unpredictable and dominates the fit.
#   - HP(160) offers a pragmatic compromise: it suppresses unpredictable
#     high-frequency noise sufficiently to allow the predictable mid-term
#     business-cycle signal (i.e., the change in growth) to be identified clearly by M-SSA.
#   - Exercises 3 and 4 explore even more adaptive HP(16),
#     which tracks shorter-run growth dynamics, and classic HP(1600), which 
#     smooths out relevant signals.


# ==================================================================
# 1.1 Apply M-SSA via the Wrapper Function
# ==================================================================
# The wrapper function compute_mssa_BIP_predictors_func was introduced
# and documented in Tutorial 7.2. Here we use it to replicate and extend
# the Tutorial 7.2 results under a unified interface.
head(compute_mssa_BIP_predictors_func)

# ------------------------------------------------------------------
# Hyperparameter Specifications in the call to compute_mssa_BIP_predictors_func()
# ------------------------------------------------------------------

# HP smoothing parameter: the single most important hyperparameter.
# lambda_HP = 160 yields a moderately adaptive trend filter.
# See Tutorial 7.1 for a detailed discussion of lambda_HP's role.
lambda_HP <- 160

# Filter length L (in quarters):
# L = 31 corresponds to nearly 8 years, which is sufficient for lambda_HP = 160
# because the HP filter weights decay fast enough at this smoothing level.
# L must be an odd number to ensure a symmetric filter (see Tutorial 7.1).
L <- 31

# In-sample end date for VAR (i.e., M-SSA) estimation.
# The proposed design is fairly (remarkably) robust to this choice because the VAR is
# kept parsimonious (low p, q), which limits overfitting on short samples. 
# Comparisons with full sample estimation are provided in Heinisch, van Norden & Wildi (2026)
# (one can set date_to_fit<-2200 (any integer number larger than 2026), to replicate these results)
date_to_fit <- "2008"

# VAR model orders: a minimal AR(1) specification (p=1, q=0) is used
# to keep the model simple and avoid overfitting, especially given the
# relatively short in-sample span.
p <- 1
q <- 0

# Holding-time (HT) constraints: control the smoothness of each M-SSA
# predictor by limiting the minimum expected time between zero-crossings.
# - A larger HT value → longer expected time between zero-crossings → smoother predictor.
# - The HT constraint directly governs the smoothness dimension (S) of the AST trilemma.
# - These values were calibrated for the original predictor design in Tutorial 7.2.
# - As established in Tutorial 7.2, the selected HT values are approximately twice
#   as large as the MSE-optimal (unconstrained) benchmark HT values:
#   this means M-SSA predictors produce fewer than half as many zero-crossings
#   as the MSE benchmark, deliberately trading off some accuracy for substantially
#   smoother signals that are easier to interpret in real-time applications.
# - Imposing the additional smoothness in M-SSA narrows the gap to the target HP(160) which 
#   is smoother than its M-MSE predictor.
ht_mssa_vec <- c(6.380160, 6.738270, 7.232453, 7.225927, 7.033768)
names(ht_mssa_vec) <- colnames(x_mat)

# Forecast horizons: M-SSA is separately optimized for each horizon in h_vec.
# h = 0 corresponds to a nowcast; h = 6 to a 6-quarter-ahead forecast.
h_vec <- 0:6

# Forecast excess: adds anticipation beyond the nominal forecast horizon
# to compensate for potential VAR model misspecification.
# A uniform excess of 4 quarters is applied across all indicators.
# See Tutorial 7.2, Exercise 2 for background on forecast excess.
f_excess <- rep(4, length(select_vec_multi))

# ------------------------------------------------------------------
# Run the M-SSA wrapper function
# ------------------------------------------------------------------
mssa_indicator_obj <- compute_mssa_BIP_predictors_func(
  x_mat, lambda_HP, L, date_to_fit, p, q,
  ht_mssa_vec, h_vec, f_excess, lag_vec, select_vec_multi
)

# ------------------------------------------------------------------
# Retrieve outputs from the wrapper function
# ------------------------------------------------------------------

# Forward-shifted HP-BIP targets: one column per forecast horizon in h_vec.
# Each column is the HP-filtered BIP series shifted forward by
# (forecast horizon + publication lag) quarters.
target_shifted_mat <- mssa_indicator_obj$target_shifted_mat
# Note: the symmetric filter is not available towards start and end of the sample (NAs)
tail(target_shifted_mat,50)

# Aggregate M-SSA predictors: one column per forecast horizon in h_vec.
# Each predictor is the equally-weighted average of the standardized M-SSA filtered
# outputs across all selected series (BIP, ip, ifo_c, ESI, spr_10y_3m).
# See Tutorial 7.2, Exercise 3 for the aggregation rationale.
predictor_mssa_mat <- mssa_indicator_obj$predictor_mssa_mat
tail(predictor_mssa_mat)

# M-MSE (minimum mean-square error) predictors: one column per horizon.
# These serve as a reference benchmark based on classic MSE-optimal filtering.
# M-MSE is noisier (less smooth) than M-SSA above.
predictor_mmse_mat <- mssa_indicator_obj$predictor_mmse_mat
tail(predictor_mmse_mat)

# M-SSA component sub-series (3-dimensional array):
#   Dimensions: [series index, time, forecast horizon]
#   - Series index: BIP, ip, ifo_c, ESI, spr_10y_3m (in order of select_vec_multi)
#   - The aggregate M-SSA predictor is the row-wise equally-weighted average
#     of these (standardized) sub-series across the series dimension.
#   - Inspecting individual sub-series can help interpret which indicators
#     drive the aggregate predictor at each forecast horizon.
mssa_array <- mssa_indicator_obj$mssa_array
# The series are not standardized
tail(mssa_array["BIP",,])
# ------------------------------------------------------------------
# Plot: M-SSA predictors across all forecast horizons
# The vertical dashed line marks the end of the in-sample estimation window.
# ------------------------------------------------------------------
mplot <- predictor_mssa_mat
par(mfrow = c(1, 1))
colo <- rainbow(ncol(predictor_mssa_mat))
main_title <- c(
  paste("Standardized M-SSA predictors for forecast horizons ",
        paste(h_vec, collapse = ","), sep = ""),
  "Vertical line delimits in-sample and out-of-sample spans"
)

# Base plot: nowcast predictor (h = 0) as reference series
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay predictors for all other forecast horizons
for (j in 1:ncol(mplot))
{
  lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
  mtext(colnames(mplot)[j], col = colo[j], line = -j)
}

abline(h = 0) # Reference line at zero
# Vertical dashed line: boundary between in-sample and out-of-sample spans
abline(v = which(rownames(mplot) > date_to_fit)[1] - 1, lty = 2)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# Interpretation through the lens of the AST forecast trilemma:
#   - Timeliness (T): M-SSA predictors are progressively left-shifted as the
#     forecast horizon increases, both in- and out-of-sample. This reflects
#     the timeliness component of the AST trilemma: longer horizons require
#     greater anticipation of future target movements.
#     Note: full-sample M-SSA optimization improves the left-shift (lead), see
#     Heinisch, Van Norden & Wildi (2026).
#   - Smoothness (S): the number of zero-crossings is governed by the HT
#     constraint, which directly controls the smoothness component of the trilemma.
#   - Accuracy (A): forecast accuracy relative to benchmarks is examined below
#     in Section 1.2.


# ==================================================================
# 1.2 Compute Forecast Performance Metrics
# ==================================================================

# ------------------------------------------------------------------
# 1.2.1 Define Benchmark Predictors
# ------------------------------------------------------------------
# M-SSA is benchmarked against two alternatives:
#   (a) The mean of BIP: a simple no-change benchmark (hard to beat at
#       longer horizons when BIP has low serial correlation (log differences)).
#   (b) A classic direct forecast: OLS regression of selected macro
#       indicators on forward-shifted BIP.

# Indicator selection for the direct forecast benchmark.
# Note: including too many predictors risks overfitting and degrades
# out-of-sample performance. A parsimonious selection is preferred.
select_direct_indicator <- c("ifo_c", "ESI")
# here we select all indicators
select_direct_indicator <- colnames(x_mat)

# ------------------------------------------------------------------
# Illustrative example: direct h-step-ahead forecast for h = 2
# ------------------------------------------------------------------
h <- 2

# Construct the forward-shifted BIP target:
# shift BIP forward by (publication lag + forecast horizon) quarters,
# padding the end with NAs to preserve the original series length.
forward_shifted_BIP <- c(
  x_mat[(1 + lag_vec[1] + h):nrow(x_mat), "BIP"],
  rep(NA, h + lag_vec[1])
)
# Note the selected publication lag of 2-quarters (safety margin)
# This renders the forecast task more difficult
tail(forward_shifted_BIP)

# Fit the direct forecast regression:
# regress the forward-shifted BIP on the selected contemporaneous indicators.
lm_obj <- lm(forward_shifted_BIP ~ x_mat[, select_direct_indicator])

# Inspect the regression output.
# Note: for h >= 2, regression coefficients are typically not statistically
# significant, reflecting the low signal-to-noise ratio of raw BIP at
# longer horizons.
summary(lm_obj)

# Technical note on inference:
#   - Regression residuals exhibit both heteroscedasticity (driven by crisis
#     episodes) and serial autocorrelation (induced by the forward shift).
#   - Standard OLS standard errors are therefore biased and unreliable.
#   - HAC-adjusted standard errors (via the sandwich package) are used
#     throughout this tutorial to correct for both issues.

# Construct the direct forecast as a fitted value from the regression.
# This is a full-sample (in- and out-of-sample) predictor.
# Equivalent to lm_obj$fitted.values, but computed explicitly here
# to allow for easy out-of-sample subsetting later.
direct_forecast <- lm_obj$coef[1] +
  x_mat[, select_direct_indicator] %*% lm_obj$coef[2:(length(select_direct_indicator) + 1)]

# ------------------------------------------------------------------
# Plot: forward-shifted BIP target vs. direct forecast
# ------------------------------------------------------------------
# - For h >= 2, the direct forecast degenerates toward a noisy flat line
#   near zero, confirming the difficulty of predicting raw BIP at longer
#   horizons: the selected indicators carry insufficient signal to overcome
#   the high-frequency noise in BIP.
#
# - COVID distortions:
#   The COVID episode (2020 Q1-Q2) introduces extreme outliers in BIP.
#   Depending on the forward-shift h, these outliers may be spuriously aligned
#   or misaligned with the predictor, producing artificially large hits or
#   misses in out-of-sample evaluation. These should not be interpreted as
#   genuine forecast skill or failure.
#
# - Effect of increasing h (h = 0, 1, 2, 3, ...):
#   As h increases, the direct forecast progressively loses its ability to
#   track recession dips: the predicted series lags further and further behind
#   the realized BIP target, eventually failing to capture turning points
#   altogether. This is a direct consequence of the low signal-to-noise ratio
#   of raw BIP at longer forecast horizons.
#   In contrast, increasing the forecast horizon in M-SSA produces a deliberate
#   left-shift (advancement) of the predictor: rather than lagging behind the
#   target, M-SSA anticipates future movements by shifting its output earlier
#   in time, which is precisely the timeliness property (T) of the AST trilemma
#   that the direct forecast lacks.
# ------------------------------------------------------------------
ts.plot(cbind(forward_shifted_BIP, direct_forecast),
        main = paste("BIP shifted forward by ", h,
                     " quarters (black) vs. direct forecast (red)"),
        col = c("black", "red"))
abline(h = 0)

#------------
# 1.2.2 Performance Evaluation
#
# The following function computes direct forecasts and evaluates M-SSA against two benchmarks:
#   (i)  The mean of BIP (naive no-change forecast)
#   (ii) A classic direct OLS forecast using a small set of macro-indicators
#
# Performance measures computed:
#
#   1. Target correlations
#      - Correlations between predictors and forward-shifted targets (HP-BIP or raw BIP)
#      - Emphasize dynamic forecast quality; invariant to level and scale differences
#
#   2. Relative Root Mean Square Error (rRMSE)
#      - Ratio of M-SSA RMSE to benchmark RMSE (mean or direct forecast)
#      - Values below 1 indicate M-SSA outperforms the benchmark
#      - Sensitive to level and scale differences
#
#   3. HAC-adjusted p-values
#      - Obtained by regressing the predictor on forward-shifted BIP and extracting
#        the t-statistic of the slope coefficient
#      - Small p-values indicate statistically significant predictive content
#      - HAC (Newey-West) adjustment accounts for heteroscedasticity (e.g., COVID outliers)
#        and autocorrelation in residuals (R package 'sandwich')
#
#   4. Two evaluation spans:
#      - Full sample
#      - Out-of-sample only (post in_out_separator): the entire financial crisis is out-of-sample
#
#   5. Two forecast targets:
#      a. Forward-shifted HP-BIP: the smoothed target for which M-SSA is explicitly optimized
#      b. Forward-shifted raw BIP: includes the unpredictable high-frequency noise component
#
# Note on overfitting:
#   - M-SSA does not target raw BIP directly, so overfitting to BIP is less of a concern

# --- In-sample / Out-of-sample Split ---
# The out-of-sample span is chosen so that the entire financial crisis falls outside the
# estimation window, across all forecast horizons.
#
# Technical note:
#   - Ideally, date_to_fit and in_out_separator would coincide.
#   - However, the MTS package raises an error when date_to_fit = "2007".
#   - Regardless, the VAR is estimated without seeing the financial crisis, which is
#     effectively out-of-sample in both cases.
#   - Setting in_out_separator <- "2007" ensures the financial crisis remains in the
#     out-of-sample evaluation window for all forecast horizons, keeping comparisons
#     consistent and meaningful across horizons.
in_out_separator <- "2007"

# Run the performance evaluation function
perf_obj <- compute_perf_func(
  x_mat, target_shifted_mat, predictor_mssa_mat, predictor_mmse_mat,
  in_out_separator, select_direct_indicator, h_vec
)

# --- Retrieve All Performance Measures ---

# HAC-adjusted p-values and t-statistics: target = HP-BIP
p_value_HAC_HP_BIP_full  <- perf_obj$p_value_HAC_HP_BIP_full   # Full sample
t_HAC_HP_BIP_full        <- perf_obj$t_HAC_HP_BIP_full
p_value_HAC_HP_BIP_oos   <- perf_obj$p_value_HAC_HP_BIP_oos    # Out-of-sample
t_HAC_HP_BIP_oos         <- perf_obj$t_HAC_HP_BIP_oos

# Target correlations: target = HP-BIP
cor_mat_HP_BIP_full <- perf_obj$cor_mat_HP_BIP_full             # Full sample
cor_mat_HP_BIP_oos  <- perf_obj$cor_mat_HP_BIP_oos              # Out-of-sample

# HAC-adjusted p-values and t-statistics: target = raw BIP
p_value_HAC_BIP_full  <- perf_obj$p_value_HAC_BIP_full          # Full sample
t_HAC_BIP_full        <- perf_obj$t_HAC_BIP_full
p_value_HAC_BIP_oos   <- perf_obj$p_value_HAC_BIP_oos           # Out-of-sample
t_HAC_BIP_oos         <- perf_obj$t_HAC_BIP_oos

# Target correlations: target = raw BIP
cor_mat_BIP_full <- perf_obj$cor_mat_BIP_full                   # Full sample
cor_mat_BIP_oos  <- perf_obj$cor_mat_BIP_oos                    # Out-of-sample

# Relative RMSE (rRMSE): target = HP-BIP
rRMSE_MSSA_HP_BIP_direct <- perf_obj$rRMSE_MSSA_HP_BIP_direct  # vs. direct forecast
rRMSE_MSSA_HP_BIP_mean   <- perf_obj$rRMSE_MSSA_HP_BIP_mean    # vs. mean benchmark

# Relative RMSE (rRMSE): target = raw BIP
rRMSE_MSSA_BIP_direct <- perf_obj$rRMSE_MSSA_BIP_direct        # vs. direct forecast
rRMSE_MSSA_BIP_mean   <- perf_obj$rRMSE_MSSA_BIP_mean          # vs. mean benchmark

# Forward-shifted raw BIP matrix (used as target in BIP-based evaluations)
target_BIP_mat <- perf_obj$target_BIP_mat
tail(target_BIP_mat)

#------------
# 1.2.3 Evaluation Results

# --- Target Correlations: M-SSA vs. Forward-Shifted HP-BIP ---

# a. Full sample
cor_mat_HP_BIP_full
# Expected pattern:
#   - Rows correspond to increasing forward-shifts of the target (top to bottom)
#   - Columns correspond to M-SSA designs optimized for increasing forecast horizons (left to right)
#   - The best-performing design for a given forward-shift tends to lie on or near the diagonal,
#     i.e., the M-SSA horizon matches the target forward-shift — as expected from the optimization
#   - M-SSA is progressively leading (left-shifted) in accordance with the target horizon h

# b. Out-of-sample
cor_mat_HP_BIP_oos
# Pattern closely mirrors the full-sample results, suggesting robustness out-of-sample

# --- Target Correlations: M-SSA vs. Forward-Shifted Raw BIP ---
cor_mat_BIP_full
cor_mat_BIP_oos
# Correlations are generally lower due to the high-frequency noise in raw BIP
# The diagonal pattern is still present but more attenuated (signal obscured by noise)
# Correlations negative at horizon h=5

# Note on longer-horizon predictability:
#   - Out-of-sample correlations suggest that M-SSA predictors optimized for h > 4
#     correlate positively with forward-shifted BIP up to 4 quarters ahead
#   - This raises the question: does M-SSA capture genuine, systematic predictability
#     of BIP one year ahead (plus the publication lag)?
#   - HAC-adjusted p-values below provide a formal test of this claim

p_value_HAC_BIP_oos
# The predictor optimized for h = 6 is marginally significant up to a forward-shift of 3 quarters
# Caution: the out-of-sample span is relatively short, limiting the power of these tests

#------------
# Note on MSE-Based Performance (rRMSE)
#
# M-SSA is not explicitly designed to minimize MSE:
#   - The predictor is standardized; level and scale are not calibrated to match BIP
#   - The optimization criterion (target correlation) is scale- and level-invariant
#
# A fully MSE-optimal predictor with explicit level and scale calibration is proposed in Exercise 3.
#
# Despite this, we compute rRMSE here for informational purposes, with the following caveats
# (acknowledged as intentional simplifications):
#   1. Level and scale of M-SSA are calibrated using out-of-sample data (look-ahead bias)
#   2. Direct forecasts are estimated on the full sample (including the out-of-sample span)
#   3. The mean benchmark is computed from out-of-sample BIP observations
#
# These shortcomings are addressed in Exercise 3.
# For now, the rRMSE results are best interpreted as reflecting the dynamic (correlation-based)
# forecast quality, with static calibration parameters treated as secondary.

# rRMSE: M-SSA vs. mean benchmark, target = HP-BIP
# Values below 1 indicate M-SSA outperforms the naive mean forecast
rRMSE_MSSA_HP_BIP_mean

# rRMSE: M-SSA vs. direct forecast, target = HP-BIP
rRMSE_MSSA_HP_BIP_direct

# rRMSE: M-SSA vs. mean benchmark, target = raw BIP
rRMSE_MSSA_BIP_mean

# rRMSE: M-SSA vs. direct forecast, target = raw BIP
rRMSE_MSSA_BIP_direct

# Observed patterns (consistent across all four rRMSE tables):
#   - M-SSA designs optimized for larger horizons (left to right) tend to outperform
#     at larger forward-shifts (top to bottom) — the diagonal pattern reappears
#   - Gains are strongest and most systematic when targeting HP-BIP
#     (the series M-SSA is explicitly optimized for)
#   - For raw BIP, gains are less consistent, as noise dilutes the signal
# Exercise 4 will show that a more adaptive design (smaller lambda_HP) can yield further gains

#------------
# --- HAC-Adjusted p-Values: Formal Tests of Predictive Significance ---
#
# Approach: regress the M-SSA predictor on forward-shifted BIP (or HP-BIP) and test
# whether the slope coefficient is significantly different from zero.
#
# Notes on the HAC adjustment:
#   - In some cases, the HAC variance estimate falls below the standard OLS variance estimate
#     (can occur with certain kernel/bandwidth choices)
#   - We therefore use the maximum of HAC and OLS standard errors to derive p-values
#   - This makes our inference conservative, erring on the side of fewer false positives

# p-values: target = HP-BIP, full sample
p_value_HAC_HP_BIP_full

# p-values: target = HP-BIP, out-of-sample
p_value_HAC_HP_BIP_oos
# The systematic patterns in target correlations and rRMSE are broadly statistically significant
# M-SSA appears capable of predicting forward-shifted HP-BIP several quarters ahead
#
# Important caveat:
#   - HP filtering induces autocorrelation in the target series
#   - Some degree of predictability in HP-BIP is therefore mechanical, not purely informational
#       (even though HAC adjustment should correct for this)
#   - Exercise 3 addresses this by shifting the evaluation target to raw BIP

# p-values: target = raw BIP, full sample
p_value_HAC_BIP_full
# The diagonal pattern is still visible but less sharp — high-frequency noise reduces test power

# p-values: target = raw BIP, out-of-sample
p_value_HAC_BIP_oos

# Concluding remarks:
#   - M-SSA is designed to track business-cycle dynamics (smooth growth fluctuations in HP-BIP)
#   - Part of the observed out-of-sample outperformance is attributable to including the
#     financial crisis in the evaluation span — a period of large, persistent movements
#     that a cycle-tracking filter can partially anticipate
#   - In a tranquil, trend-stationary environment (no recessions or crises), gains over
#     naive benchmarks would likely be smaller or negligible
























 




#-------------------------------------------
# 1.3 Visualize performances: link performance measures to plots of predictors against targets
# 1.3.1 M-SSA predictors (without sub-series)
# Select a forward-shift of target (the k-th entry in h_vec)
k<-4
# This is the forecast horizon 
h_vec[k]
# To obtain the effective forward-shift we have to add the publication lag
shift<-h_vec[k]+lag_vec[1]
# Effective forward shift of BIP
shift
if (k>length(h_vec))
{
  print(paste("k should be smaller equal ",length(h_vec),sep=""))
  k<-length(h_vec)
}  
# Select a M-SSA predictor: optimized for forecast horizon h_vec[j]
j<-k
if (j>length(h_vec))
{
  print(paste("j should be smaller equal ",length(h_vec),sep=""))
  j<-length(h_vec)
}  
# Plot targets (forward-shifted BIP and HP-BIP) and predictor
par(mfrow=c(1,1))
# Scale the data for better visual interpretation of effect of excess forecast on M-SSA (red) vs. previous M-SSA (blue)
mplot<-scale(cbind(target_BIP_mat[,k],target_shifted_mat[,k],predictor_mssa_mat[,j]))
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c(paste("BIP left-shifted by ",h_vec[k]," quarters (plus publication lag)",sep=""),paste("HP-BIP left-shifted by ",h_vec[k]," quarters (plus publication lag)",sep=""),paste("M-SSA predictor optimized for h=",h_vec[j],sep=""))
colo<-c("black","violet","blue")
main_title<-"Standardized forward-shifted BIP and HP-BIP vs. M-SSA predictor"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(x_mat)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()

if (F)
{

  mplot2025<-mplot
  
  mplot2026<-mplot
  
  dim(mplot2025)
  

  mplot<-cbind(c(rep(NA,4),mplot2025[,3]),mplot2026[,3])
  colo<-c("blue","red")
  rownames(mplot)<-c(rep(NA,4),rownames(x_mat))
  colnames(mplot)<-c("One year forecast based on 2025 data: shifted one year to the right","Nowcast based on 2026 data")
  main_title<-"One year forecast Jan 2025 (blue)"
  plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(x_mat)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
  mtext(colnames(mplot)[1],col=colo[1],line=-1)
  for (i in 1:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i],lwd=1,lty=1)
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
  abline(h=0)
  abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
  axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
  axis(2)
  box()
  
  
  ts.plot(cbind(mplot2025[,3],mplot2026[,3]),col=c("blue","red"))
  abline(h=0)
}



#------------------
# 1.3.2 We can now relate the above plot to the performance measures
# Sample correlation: 
cor(na.exclude(mplot))[2,ncol(mplot)]
# This number corresponds to element (k,j) of the matrix cor_mat_HP_BIP_full computed by our function
cor_mat_HP_BIP_full[k,j]

# We can also look at the correlation of the predictor with the forward-shifted BIP (instead of HP-BIP)
cor(na.exclude(mplot))[1,3]
# Note: the (k,j) entry of cor_mat_BIP generally differs from cor(na.exclude(mplot))[1,3]
cor_mat_BIP_full[k,j]
# The reason is simple: by removing NAs (due to inclusion of the two-sided target in mplot) we change the sample-size
# We can easily correct by removing the two-sided target in mplot
mplot_without_two_sided<-scale(cbind(target_BIP_mat[,k],predictor_mssa_mat[,j]))
# This number now matches cor_mat_BIP[k,j]
cor(na.exclude(mplot_without_two_sided))[1,2]

# Assume one selects k=j=5 (h_vec[5]=4: one-year ahead forecast) in the above plot:
# Then the (weak) positive correlation between M-SSA and shifted BIP might suggest a (weak) predictability one year ahead
# Is this (weak) effect statistically significant?
# Let's have a look at the HAC-adjusted p-values
# 1. Full sample 
p_value_HAC_BIP_full[k,j]
# 2. Out-of-sample
p_value_HAC_BIP_oos[k,j]
# Not significant at a one-year ahead horizon when targeting BIP (which is a very noisy series...)
#   -See exercise 3 for further results

# Instead of BIP we might have a look at targeting HP-BIP (also shifted one year ahead)
p_value_HAC_HP_BIP_full[k,j]
p_value_HAC_HP_BIP_oos[k,j]


# ==================================================================
# 1.3 Visualize and Interpret Forecast Performance
# ==================================================================
# This section links the numerical performance measures (correlations,
# HAC-adjusted p-values) to visual inspection of predictors against
# their forward-shifted targets, for both raw BIP and HP-filtered BIP.


# ------------------------------------------------------------------
# 1.3.1 Plot M-SSA predictor against forward-shifted targets
# ------------------------------------------------------------------

# Select the forecast horizon index k (indexes into h_vec).
# k = 4 corresponds to a forecast horizon of h_vec[4] quarters.
k <- 4
# Display the corresponding forecast horizon in quarters
h_vec[k]

# The effective forward-shift of the target equals the forecast horizon
# plus the publication lag (2 quarters): this is the total number of
# quarters by which BIP and HP-BIP are shifted into the future.
shift <- h_vec[k] + lag_vec[1]
shift

# Bounds check: k must not exceed the number of available horizons
if (k > length(h_vec))
{
  print(paste("k should be smaller or equal to ", length(h_vec), sep = ""))
  k <- length(h_vec)
}

# Select the M-SSA predictor optimized for forecast horizon h_vec[j].
# Setting j = k pairs each predictor with its matching target horizon,
# which is the natural comparison for evaluating forecast performance.
j <- k
if (j > length(h_vec))
{
  print(paste("j should be smaller or equal to ", length(h_vec), sep = ""))
  j <- length(h_vec)
}

# ------------------------------------------------------------------
# Construct and plot the standardized comparison:
#   - Black:  forward-shifted raw BIP (noisy target)
#   - Violet: forward-shifted HP-BIP (smooth trend-growth target)
#   - Blue:   M-SSA predictor optimized for horizon h_vec[j]
# All three series are standardized to a common scale for visual comparison.
# ------------------------------------------------------------------
par(mfrow = c(1, 1))
mplot <- scale(cbind(target_BIP_mat[, k],
                     target_shifted_mat[, k],
                     predictor_mssa_mat[, j]))
rownames(mplot) <- rownames(x_mat)
colnames(mplot) <- c(
  paste("BIP left-shifted by ",    h_vec[k], " quarters (plus publication lag)", sep = ""),
  paste("HP-BIP left-shifted by ", h_vec[k], " quarters (plus publication lag)", sep = ""),
  paste("M-SSA predictor optimized for h=", h_vec[j], sep = "")
)
colo <- c("black", "violet", "blue")
main_title <- "Standardized forward-shifted BIP and HP-BIP vs. M-SSA predictor"

# Base plot: forward-shifted raw BIP in black
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     lwd = c(2, rep(1, ncol(x_mat) - 1)),
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay HP-BIP target (violet) and M-SSA predictor (blue)
for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0) # Reference line at zero
# Vertical dashed line: end of in-sample estimation window
abline(v = which(rownames(mplot) <= date_to_fit)[length(which(rownames(mplot) <= date_to_fit))],
       lwd = 2, lty = 2)
axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
     labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
axis(2)
box()

# ------------------------------------------------------------------
# Experimental block: comparison of 2025 and 2026 data vintages
# (currently inactive; set the if(F) condition to if(T) to activate)
# Purpose: overlay the one-year-ahead forecast from the 2025 data vintage
# (shifted one year to the right) against the nowcast from the 2026 vintage,
# to visually assess the accuracy of the one-year-ahead prediction after
# one year of additional data becomes available.
# ------------------------------------------------------------------
if (F)
{
  mplot2025 <- mplot
  mplot2026 <- mplot
  
  # Align 2025 forecast with 2026 nowcast by shifting 4 quarters to the right
  mplot <- cbind(c(rep(NA, 4), mplot2025[, 3]), mplot2026[, 3])
  colo  <- c("blue", "red")
  rownames(mplot) <- c(rep(NA, 4), rownames(x_mat))
  colnames(mplot) <- c(
    "One-year-ahead forecast (2025 vintage, shifted 4 quarters right)",
    "Nowcast based on 2026 vintage"
  )
  main_title <- "One-year-ahead forecast from Jan 2025 (blue) vs. 2026 nowcast (red)"
  
  plot(mplot[, 1], main = main_title, axes = F, type = "l",
       xlab = "", ylab = "", col = colo[1],
       lwd = c(2, rep(1, ncol(x_mat) - 1)),
       ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
  mtext(colnames(mplot)[1], col = colo[1], line = -1)
  for (i in 1:ncol(mplot))
  {
    lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
  abline(h = 0)
  abline(v = which(rownames(mplot) <= date_to_fit)[length(which(rownames(mplot) <= date_to_fit))],
         lwd = 2, lty = 2)
  axis(1, at = c(1, 4 * 1:(nrow(mplot) / 4)),
       labels = rownames(mplot)[c(1, 4 * 1:(nrow(mplot) / 4))])
  axis(2)
  box()
  
  # Direct overlay of 2025 and 2026 M-SSA predictor time series (no shift)
  # Useful for inspecting data revision effects on the predictor
  ts.plot(cbind(mplot2025[, 3], mplot2026[, 3]), col = c("blue", "red"))
  abline(h = 0)
}


# ==================================================================
# 1.3.2 Link the Plot to Quantitative Performance Measures
# ==================================================================

# ------------------------------------------------------------------
# Target correlations
# ------------------------------------------------------------------

# Sample correlation between M-SSA predictor and forward-shifted HP-BIP.
# This corresponds to element (k, j) of cor_mat_HP_BIP_full.
cor(na.exclude(mplot))[2, ncol(mplot)]
# This matches computations from our wrapper
cor_mat_HP_BIP_full[k, j]

# Sample correlation between M-SSA predictor and forward-shifted raw BIP.
cor(na.exclude(mplot))[1, 3]

# Note: the (k, j) entry of cor_mat_BIP_full may differ from the above,
# because including the two-sided HP-BIP target in mplot introduces additional
# NAs (at the filter boundaries), which reduces the effective sample size
# used by na.exclude(). This creates a subtle sample mismatch.
cor_mat_BIP_full[k, j]

# Correction: recompute the BIP correlation without the two-sided HP-BIP target
# in the data matrix, so that no extra NAs are introduced.
# The resulting correlation now correctly matches cor_mat_BIP_full[k, j].
mplot_without_two_sided <- scale(cbind(target_BIP_mat[, k], predictor_mssa_mat[, j]))
cor(na.exclude(mplot_without_two_sided))[1, 2]

# ------------------------------------------------------------------
# Statistical significance: HAC-adjusted p-values
# ------------------------------------------------------------------
# Example: set k = j = 5 (h_vec[5] = 4, i.e., one-year-ahead forecast).
# The positive correlation between M-SSA and forward-shifted BIP hints at
# weak predictability one year ahead. We now assess whether this effect is
# statistically significant after HAC adjustment.
k<-j<-5
# Full-sample HAC-adjusted p-value (raw BIP target):
p_value_HAC_BIP_full[k, j]
# Out-of-sample HAC-adjusted p-value (raw BIP target):
p_value_HAC_BIP_oos[k, j]
# Interpretation: the effect is not statistically significant at a one-year
# horizon when targeting raw BIP. This is expected given the high noise level
# of raw BIP at longer forecast horizons. See Exercise 3 for further results.

# Repeat the significance assessment using HP-BIP as the target instead.
# HP-BIP is a much cleaner signal, so statistical significance should be
# easier to establish.
# Full-sample HAC-adjusted p-value (HP-BIP target):
p_value_HAC_HP_BIP_full[k, j]
# Out-of-sample HAC-adjusted p-value (HP-BIP target):
p_value_HAC_HP_BIP_oos[k, j]

# ==================================================================
# 1.4 Sensitivity Analysis: Effect of Increasing the Holding-Time (HT) Constraint
# ==================================================================
# Context:
#   - Section 1.2 examined the Timeliness dimension of the AST trilemma via h_vec and f_excess
#   - This section examines the Smoothness dimension by doubling the HT constraints
#   - This serves as a robustness check: if the qualitative findings are not overly sensitive to
#     the degree of smoothing imposed, this strengthens confidence in the M-SSA predictor

# --- Double the Holding-Time Constraints ---
# Doubling HT approximately doubles the expected duration between consecutive zero-crossings
# (under the assumption that the VAR is correctly specified; see tutorial 7.1 for theory)
ht_mssa_vec_long <- 2 * ht_mssa_vec
names(ht_mssa_vec_long) <- colnames(x_mat)
ht_mssa_vec_long

# Filter length: stronger smoothing (larger HT) would in principle benefit from a longer filter,
# as the impulse response of the target filter decays more slowly.
# For computational efficiency we keep L_long = L here.
# The impact of this simplification is expected to be minor in practice.
L_long <- L

# --- Run M-SSA with Doubled HT ---
mssa_indicator_obj <- compute_mssa_BIP_predictors_func(
  x_mat, lambda_HP, L_long, date_to_fit, p, q,
  ht_mssa_vec_long, h_vec, f_excess, lag_vec, select_vec_multi
)

# Retrieve outputs (same structure as in section 1.1)
target_shifted_mat <- mssa_indicator_obj$target_shifted_mat   # Forward-shifted HP-BIP target
predictor_mssa_mat <- mssa_indicator_obj$predictor_mssa_mat   # Aggregate M-SSA predictors
predictor_mmse_mat <- mssa_indicator_obj$predictor_mmse_mat   # M-MSE benchmark predictors
mssa_array         <- mssa_indicator_obj$mssa_array           # Component-level M-SSA predictors

# --- Plot M-SSA Predictors (Doubled HT) ---
# Compare visually with the plot from section 1.1 to assess the effect of increased smoothing
mplot <- predictor_mssa_mat
colnames(mplot) <- colnames(predictor_mssa_mat)

par(mfrow = c(1, 1))
colo <- rainbow(ncol(predictor_mssa_mat))

main_title <- c(
  paste("Standardized M-SSA predictors (doubled HT) for forecast horizons ",
        paste(h_vec, collapse = ","), sep = ""),
  "Vertical line delimits in-sample and out-of-sample spans"
)

plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))

for (j in 1:ncol(mplot)) {
  lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
  mtext(colnames(mplot)[j], col = colo[j], line = -j)
}

abline(h = 0)
abline(v = which(rownames(mplot) > date_to_fit)[1] - 1, lty = 2)  # in-sample/out-of-sample boundary
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()



# --- Interpretation ---
#
# 1. Smoothness:
#    - Predictors are visibly smoother, as expected from the doubled HT constraints
#    - Under the assumption that the VAR is correctly specified, the expected duration
#      between consecutive zero-crossings should approximately double relative to section 1.1
#      (see tutorial 7.1 for the theoretical derivation)
#
# 2. Robustness:
#    - The overall shape and directional signals of the predictors near the sample end
#      are qualitatively consistent with the results from section 1.1
#    - This supports the robustness of the M-SSA predictor to the choice of HT
#
# 3. AST trilemma trade-off:
#    - Increased smoothness (S) comes at a cost: the nowcast and short-term forecasts
#      are more conservative (slower to respond) during upswing dynamics
#    - This reflects the inherent Smoothness-Timeliness trade-off in the AST trilemma
#
# 4. Left-shift
#    -Once again, increasing the forecast horizon leads to a systematic left-shift of the M-SSA predictor 
#     (in contrast to direct forecasts, see exercise 1.2.1 above)


# ==================================================================
# Exercise 2: Falsification Test — Applying M-SSA to White Noise Data
# ==================================================================
#
# Objective:
#   Verify that the proposed battery of performance measures and statistical
#   tests correctly reflects unpredictability when the underlying data contain
#   no forecastable signal.
#
# Under the white noise null hypothesis, we expect:
#   - Target correlations close to zero
#   - rRMSEs close to one (no improvement over the mean benchmark)
#   - HAC-adjusted p-values uniformly distributed above 5%
#
# Multiple-testing caveat:
#   - We evaluate 6 × 7 = 42 p-values simultaneously per sample split
#     (full sample and out-of-sample), without any multiple-testing correction
#     (e.g., Bonferroni or Benjamini-Hochberg).
#   - Even under the white noise null, one should expect some p-values below 5%
#     by chance alone: approximately 42 × 0.05 ≈ 2 spuriously significant
#     results per evaluation.
#   - Results on the verge of the 5% threshold should therefore be interpreted
#     with caution; only results below 1% are treated as convincing evidence.


# ------------------------------------------------------------------
# 2.1 Generate Artificial White Noise Data
# ------------------------------------------------------------------
# Several random seeds are tried to assess the empirical false-positive rate
# of the HAC-adjusted tests across independent replications.
#
# Summary of findings across seeds 1–10:
#   - set.seed(1):  several p-values below 5%, none below 1%
#   - set.seed(2):  several p-values below 5%, none below 1%
#   - set.seed(3):  several p-values below 5%, none below 1%
#   - set.seed(4):  none below 1%
#   - set.seed(5):  none below 1%
#   - set.seed(6):  none below 1%
#   - set.seed(7):  one p-value below 1%
#   - set.seed(8):  none below 1%
#   - set.seed(9):  none below 1%
#   - set.seed(10): none below 1%
#
# Overall conclusion:
#   - HAC adjustment does not fully eliminate the dependence induced by
#     filtering and overlapping forecast windows: p-values below 5% occur
#     more frequently than the nominal 5% rate.
#   - However, p-values below 1% are rare: only 1 occurrence out of
#     10 × 6 × 7 × 2 = 840 computed values across all seeds and sample splits.
#   - We therefore treat results below 1% as credible evidence of predictability,
#     while remaining cautious about results in the 1%–5% range.

set.seed(1) # Change seed here to explore alternative white noise realizations

# Generate a multivariate white noise matrix with the same dimensions
# and structure (row/column names) as the real data matrix x_mat
x_mat_white_noise <- NULL
for (i in 1:ncol(x_mat))
  x_mat_white_noise <- cbind(x_mat_white_noise, rnorm(nrow(x_mat)))

# Assign original row and column names: required because the wrapper function
# uses dates (row names) and indicator names (column names) internally
rownames(x_mat_white_noise) <- rownames(x_mat)
colnames(x_mat_white_noise) <- colnames(x_mat)
tail(x_mat_white_noise)

# Confirm white noise structure: ACF should show no significant autocorrelation
acf(x_mat_white_noise)


# ------------------------------------------------------------------
# 2.2 Apply M-SSA to White Noise Data
# ------------------------------------------------------------------
# Hyperparameters are kept identical to Exercise 1 to ensure comparability.
# The only change is p = q = 0 (i.i.d. assumption for the VAR),
# since white noise has no serial dependence to exploit.

# HP smoothing parameter: same as Exercise 1 for comparability
lambda_HP <- 160

# Filter length: same as Exercise 1
L <- 31

# In-sample end date: inherited from Exercise 1
date_to_fit <- date_to_fit

# VAR model orders: p = q = 0 (white noise has no autocorrelation structure)
# Using p > 0 on white noise would introduce spurious VAR dynamics
p <- 0
q <- 0

# HT constraints: same as Exercise 1
# Smoothness properties of the filter are preserved for fair comparison
ht_mssa_vec <- c(6.380160, 6.738270, 7.232453, 7.225927, 7.033768)
names(ht_mssa_vec) <- colnames(x_mat)

# Forecast horizons: same as Exercise 1
h_vec <- 0:6

# Forecast excess: same as Exercise 1
# See Tutorial 7.2, Exercise 2 for background on forecast excess
f_excess <- rep(4, length(select_vec_multi))

# Run the M-SSA wrapper on white noise data
mssa_indicator_obj <- compute_mssa_BIP_predictors_func(
  x_mat_white_noise, lambda_HP, L, date_to_fit, p, q,
  ht_mssa_vec, h_vec, f_excess, lag_vec, select_vec_multi
)


# ------------------------------------------------------------------
# 2.3 Evaluate Performance on White Noise Data
# ------------------------------------------------------------------

# Retrieve forward-shifted HP white noise targets and M-SSA predictors
target_shifted_mat  <- mssa_indicator_obj$target_shifted_mat
predictor_mssa_mat  <- mssa_indicator_obj$predictor_mssa_mat
predictor_mmse_mat  <- mssa_indicator_obj$predictor_mmse_mat

# Direct forecast benchmark: same indicator selection as Exercise 1
# for a consistent comparison of relative performance
select_direct_indicator <- c("ifo_c", "ESI")

# Compute the full suite of performance metrics on white noise data
perf_obj <- compute_perf_func(
  x_mat_white_noise, target_shifted_mat, predictor_mssa_mat,
  predictor_mmse_mat, in_out_separator, select_direct_indicator, h_vec
)

# Extract HAC-adjusted p-values for white noise (WN) target
# (i.e., testing whether M-SSA can predict raw white noise)
p_value_HAC_WN_full <- perf_obj$p_value_HAC_BIP_full
p_value_HAC_WN_oos  <- perf_obj$p_value_HAC_BIP_oos

# Inspect p-value matrices:
# Under the null, these should show no systematic pattern
p_value_HAC_WN_full  # Full-sample HAC-adjusted p-values
p_value_HAC_WN_oos   # Out-of-sample HAC-adjusted p-values

# Count false positives at the 5% significance level
length(which(p_value_HAC_WN_full < 0.05))
length(which(p_value_HAC_WN_oos  < 0.05))

# Count false positives at the 1% significance level
# (should be very rare if HAC adjustment is working reasonably well)
length(which(p_value_HAC_WN_full < 0.01))
length(which(p_value_HAC_WN_oos  < 0.01))


# ------------------------------------------------------------------
# Findings and Discussion
# ------------------------------------------------------------------
#
# 1. HAC adjustment and residual dependence:
#    - HAC-adjusted p-values below 5% occur more frequently than the nominal
#      rate, confirming that the adjustment does not fully account for the
#      serial dependence induced by HP filtering and overlapping forecast windows.
#    - Caution is warranted for results near the 5% boundary.
#    - Results below 1% are rare (≈1 in 840 in this experiment) and can be
#      treated as credible evidence of genuine predictability.
#
# 2. HP-filtered white noise (HP-WN) vs. raw white noise (WN):
#    - Applying HP to white noise produces an autocorrelated series (HP-WN)
#      that can be predicted better than by the mean benchmark: the HP filter
#      introduces serial dependence by construction.
#    - However, predictability of HP-WN does not translate into predictability
#      of the underlying raw white noise series WN.
#    - Translated into the BIP/GDP forecasting context: business cycle turning
#      points and crisis episodes are the primary source of forecastable signal
#      in BIP/GDP. It is precisely during these episodes that BIP deviates most
#      sharply and persistently from white noise behavior, creating the
#      low-frequency dynamics that M-SSA is designed to anticipate.
#      In tranquil periods, where BIP fluctuations are closer to white noise,
#      forecast performance will naturally be weaker and less distinguishable
#      from chance.
#
# 3. Why BIP is not white noise (and HP-BIP predictability matters):
#    - Unlike WN, raw BIP exhibits persistent low-frequency dynamics that
#      contradict the white noise assumption:
#      a) Recessions (business cycles) represent systematic deviations from WN.
#      b) Significant ACF coefficients and a fitted VAR model both confirm
#         serial dependence in BIP.
#      c) The systematic diagonal structure in the performance matrices
#         (increasing horizon → larger forward-shift performs better) is
#         inconsistent with WN and strongly suggests genuine predictability.
#      d) Recent negative BIP readings reflect both exogenous shocks and
#         endogenous structural factors — neither of which is random in the
#         white noise sense.
#    - These non-WN features of BIP are concentrated in its lower-frequency
#      components, which is precisely what HP-BIP captures and what M-SSA
#      is optimized to predict.

#######################################################################################
# Exercise 3: More Adaptive Target Design — HP(16)
#
# Motivation:
#   - Exercise 1 established that M-SSA can predict HP-BIP several quarters ahead
#   - Predicting raw BIP is harder, due to its high-frequency noise component
#   - However, it is possible that HP(160) suppresses genuinely predictable dynamics,
#     i.e., the target may be 'too smooth' relative to the information in the data
#   - To test this conjecture, we replace HP(160) with the more adaptive HP(16) target
#
# Research question:
#   Does a more adaptive target (smaller lambda_HP) allow M-SSA to predict raw BIP
#   more reliably than the HP(160) design from Exercise 1?
#######################################################################################

# 3.1 Run M-SSA with Adaptive Target HP(16)
# ------------------------------------------

# Smaller lambda_HP => less smoothing => target tracks BIP more closely
lambda_HP <- 16

# Forecast horizon and excess adjustments for adaptive designs:
#   - A pronounced left-shift at large horizons can cause phase-reversal (the predictor
#     moves in the opposite direction to the target), which is undesirable in practice
#   - Phase-reversal would be theoretically optimal only if BIP followed the implicit
#     data-generating process assumed by the HP filter — which it does not (see tutorial 2.0)
#   - To guard against phase-reversal, we cap the forecast horizon at 4 and set f_excess = 0
f_excess_adaptive <- rep(0, length(select_vec_multi))
h_vec_adaptive    <- 0:4

# Run the M-SSA wrapper with the adaptive HP(16) design
mssa_indicator_obj <- compute_mssa_BIP_predictors_func(
  x_mat, lambda_HP, L, date_to_fit, p, q,
  ht_mssa_vec, h_vec_adaptive, f_excess_adaptive, lag_vec, select_vec_multi
)

# Retrieve outputs
target_shifted_mat <- mssa_indicator_obj$target_shifted_mat   # Forward-shifted HP(16)-BIP target
predictor_mssa_mat <- mssa_indicator_obj$predictor_mssa_mat   # Aggregate M-SSA predictors
predictor_mmse_mat <- mssa_indicator_obj$predictor_mmse_mat   # M-MSE benchmark predictors

# 3.2 Compute Forecast Performance
# ---------------------------------
# Macro-indicator selection for the direct forecast benchmark
# Parsimonious specification to avoid overfitting in out-of-sample evaluation
select_direct_indicator <- c("ifo_c", "ESI")

perf_obj <- compute_perf_func(
  x_mat, target_shifted_mat, predictor_mssa_mat, predictor_mmse_mat,
  in_out_separator, select_direct_indicator, h_vec_adaptive
)

# Retrieve performance measures (same structure as Exercise 1)

# HAC-adjusted p-values and t-statistics: target = HP(16)-BIP
p_value_HAC_HP_BIP_full <- perf_obj$p_value_HAC_HP_BIP_full
t_HAC_HP_BIP_full       <- perf_obj$t_HAC_HP_BIP_full
p_value_HAC_HP_BIP_oos  <- perf_obj$p_value_HAC_HP_BIP_oos
t_HAC_HP_BIP_oos        <- perf_obj$t_HAC_HP_BIP_oos

# Target correlations: target = HP(16)-BIP
cor_mat_HP_BIP_full <- perf_obj$cor_mat_HP_BIP_full
cor_mat_HP_BIP_oos  <- perf_obj$cor_mat_HP_BIP_oos

# HAC-adjusted p-values and t-statistics: target = raw BIP
p_value_HAC_BIP_full <- perf_obj$p_value_HAC_BIP_full
t_HAC_BIP_full       <- perf_obj$t_HAC_BIP_full
p_value_HAC_BIP_oos  <- perf_obj$p_value_HAC_BIP_oos
t_HAC_BIP_oos        <- perf_obj$t_HAC_BIP_oos

# Target correlations: target = raw BIP
cor_mat_BIP_full <- perf_obj$cor_mat_BIP_full
cor_mat_BIP_oos  <- perf_obj$cor_mat_BIP_oos

# Relative RMSE: target = HP(16)-BIP
rRMSE_MSSA_HP_BIP_direct <- perf_obj$rRMSE_MSSA_HP_BIP_direct
rRMSE_MSSA_HP_BIP_mean   <- perf_obj$rRMSE_MSSA_HP_BIP_mean

# Relative RMSE: target = raw BIP
rRMSE_MSSA_BIP_direct <- perf_obj$rRMSE_MSSA_BIP_direct
rRMSE_MSSA_BIP_mean   <- perf_obj$rRMSE_MSSA_BIP_mean

# Forward-shifted raw BIP (evaluation target)
target_BIP_mat <- perf_obj$target_BIP_mat

# 3.3 Evaluation: Does the Adaptive Design Better Predict Raw BIP?
# ----------------------------------------------------------------
# We focus on raw BIP as the target (HP-BIP predictability was already established in Exercise 1)

# Target correlations: full sample
cor_mat_BIP_full

# Target correlations: out-of-sample
cor_mat_BIP_oos
# Correlations are somewhat larger than in Exercise 1, suggesting the adaptive HP(16) design
# tracks forward-shifted BIP marginally better — consistent with the conjecture that HP(160)
# may suppress some predictable high-frequency dynamics

# Statistical significance: does the improvement hold up formally?
p_value_HAC_BIP_full
p_value_HAC_BIP_oos
# Despite marginally better correlations, there is no systematic reduction in p-values
# relative to Exercise 1 — the improvement in raw BIP predictability is not statistically
# robust, likely because the additional adaptive dynamics are dominated by noise

##################################################################################
# Exercise 4: Overly Smooth Design — Classic HP(1600)
#
# Motivation:
#   - As a counterpoint to Exercise 3, we now evaluate the classic HP(1600) target
#   - This design is standard in the business-cycle literature but has been criticized
#     for being 'too smooth' (Phillips and Jin, 2021)
#   - The goal is to empirically confirm that HP(1600) is suboptimal for short- to
#     medium-term GDP forecasting (1–5 quarters ahead)
##################################################################################

lambda_HP <- 1600

# Filter length options:
#   - HP(1600) weights decay more slowly, in principle favouring a longer filter
#   - However, longer filters increase computational cost
#   - We retain L for consistency and speed
L_long <- 2 * L - 1   # longer option (commented out below)
L_long <- L            # retained for computational efficiency

# Retain the doubled HT constraints from Exercise 1.4
ht_mssa_vec_long <- ht_mssa_vec_long

# Run M-SSA with HP(1600) target
mssa_indicator_obj <- compute_mssa_BIP_predictors_func(
  x_mat, lambda_HP, L_long, date_to_fit, p, q,
  ht_mssa_vec_long, h_vec, f_excess, lag_vec, select_vec_multi
)

# Retrieve outputs
target_shifted_mat <- mssa_indicator_obj$target_shifted_mat
predictor_mssa_mat <- mssa_indicator_obj$predictor_mssa_mat
predictor_mmse_mat <- mssa_indicator_obj$predictor_mmse_mat
mssa_array         <- mssa_indicator_obj$mssa_array

# --- Plot M-SSA Predictors (HP(1600) Design) ---
mplot <- predictor_mssa_mat
colnames(mplot) <- colnames(predictor_mssa_mat)

par(mfrow = c(1, 1))
colo <- rainbow(ncol(predictor_mssa_mat))

main_title <- c(
  paste("Standardized M-SSA predictors [HP(1600)] for forecast horizons ",
        paste(h_vec, collapse = ","), sep = ""),
  "Vertical line delimits in-sample and out-of-sample spans"
)

plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))

for (j in 1:ncol(mplot)) {
  lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
  mtext(colnames(mplot)[j], col = colo[j], line = -j)
}

abline(h = 0)
abline(v = which(rownames(mplot) > date_to_fit)[1] - 1, lty = 2)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# --- Interpretation ---
#
# 1. Over-smoothing effect:
#    - HP(1600) emphasizes very long-term trend dynamics, suppressing the business-cycle
#      fluctuations that are most relevant for 1–5 quarter ahead forecasting
#    - This is consistent with the Phillips and Jin (2021) critique of HP(1600) as
#      insufficiently flexible for business-cycle analysis
#
# 2. Reduced timeliness gain:
#    - Increasing the forecast horizon has only a marginal effect on the phase of the
#      M-SSA predictor — the left-shift is far less pronounced than with HP(160) or HP(16)
#    - This is because the HP(1600) right-tail corresponds to an AR(2) with a long periodicity;
#      advancing the filter by one year produces only a small phase change at such low frequencies
#
# 3. Pandemic sensitivity:
#    - Scrolling back through the plot panel reveals that the two-sided HP(1600) filter
#      behaves anomalously around the financial crisis and especially after the Pandemic
#    - HP(1600) is more sensitive to extreme outliers (Pandemic readings) than the more
#      adaptive HP(160) or HP(16) designs, due to its greater reliance on distant observations

##########################################################################################################
# Summary of Findings Across Exercises 1–4
##########################################################################################################

# A. Classic Direct Forecasts
#    - Direct OLS predictors rarely outperform the naive mean benchmark beyond a 2-quarter forward-shift
#    - They are more sensitive to singular episodes (e.g., the Pandemic) than M-SSA,
#      which benefits from smoothing and multivariate information aggregation
#    - They do not `left-shift' as h increases and therefore they miss turning points

# B. M-SSA: Sensitivity to Target Specification (lambda_HP)
#
#    - HP(1600) [over-smooth]: suppresses recession dynamics; increasing the forecast horizon
#        has only marginal effects on the predictor phase — not well suited for 1–5 quarter forecasting
#
#    - HP(160) [recommended]: achieves a good balance between noise suppression and adaptivity;
#        M-SSA with this target exhibits a logically consistent and statistically significant
#        forecast pattern, outperforming both the mean and direct forecasts out-of-sample
#        when targeting HP-BIP; performance on raw BIP is positive but less conclusive (noise-dominated)
#
#    - HP(16) [over-adaptive]: marginally better raw BIP correlations than HP(160), but the
#        improvement is not statistically robust; risk of phase-reversal at longer horizons
#
#    - Aggregation: the current M-SSA predictor uses equally-weighted averaging of standardized
#        components; tutorial 7.4 introduces an optimally weighted M-SSA components predictor
#        that directly targets BIP in an MSE sense and supports component-level interpretation

# C. Statistical Significance
#    - HAC adjustments (Newey-West) do not fully eliminate finite-sample size distortions,
#        particularly in the presence of crisis-driven heteroscedasticity
#    - Exercise 2 suggests the residual bias is modest: p-values below 1% are rare under
#        simulated white noise
#    - A conservative correction is applied throughout: t-statistics are derived from the
#        maximum of HAC and OLS standard errors, reducing the risk of spurious significance
# See line 8: using max(sd[2], sd_HAC[2]) 
head(HAC_ajusted_p_value_func, 20)

# D. Publication Lag 
#    - All reported forward-shifts are measured relative to the current quarter plus the
#        publication lag (lag_vec[1] = 2 quarters, reflecting a conservative assumption)
#    - The official publication lag for German GDP is one quarter, but BIP is subject to
#        revisions — which are ignored here
#
# E. Data Revisions
#    - M-SSA assigns relatively low weight to BIP (which is subject to strongest revisions)
#    - HP smoothing further mitigates the impact of data revisions on the target






