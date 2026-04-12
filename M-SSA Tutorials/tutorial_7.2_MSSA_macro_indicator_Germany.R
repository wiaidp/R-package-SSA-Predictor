#========================================================================
# Tutorial 7.2: Application of M-SSA to Quarterly Macroeconomic Data (GDP Forecasting)
#========================================================================

# Nomenclature: 
#   The acronym BIP refers to German GDP (Brutto-Inlands-Produkt)

# --- Context and Motivation ---
#
# Tutorial 7.1 demonstrated the asymptotic convergence of M-SSA performance
# metrics to their theoretical (population) values, under the assumption that
# the true data-generating process (DGP) is known.
#   -> Tutorial 7.1 validates the theory under ideal conditions.
#
# In contrast, Tutorial 7.2 operates in a realistic empirical setting:
#   - The available time series are relatively short (in-sample).
#   - The DGP is unknown; model misspecification must be acknowledged and addressed.


# --- Objectives of This Tutorial ---
#
# 1. Apply M-SSA to quarterly German macroeconomic data during a period of
#    economic uncertainty (in-sample span ends just before the onset of the financial crisis).
#      - Tutorials 7.1–7.3 were written in early 2025 and use data up to January 2025.
#      - The HP filter signals a severe and worsening recession at that point:
#        Germany has recorded negative GDP (BIP: Brutto Inlandsprodukt) growth
#        for several consecutive quarters.
#      - Key forward-looking questions (as of January 2025):
#          * Has German GDP reached its cyclical trough?
#          * Is a recovery currently underway?
#          * Can above-trend growth be expected within a foreseeable horizon?
#          * What are the macroeconomic prospects for 2025 and 2026?
#
# 2. Examine multiple M-SSA designs for nowcasting and forecasting German GDP.
#
# 3. Analyze the effects of model misspecification.
#      - A VAR(1) model cannot adequately capture recessionary dynamics.
#
# 4. Identify and evaluate solutions for mitigating model misspecification.
#
# 5. Provide the empirical background and conceptual insights required to
#    understand the forecast designs developed in the next Tutorial 7.3.


# --- Conceptual Background ---
#
# On forecasting GDP "numbers":
#   - As discussed in Tutorial 7.1, this framework does not produce point
#     forecasts of GDP levels ("give me the number").
#   - Any such point forecast would carry a confidence interval wide enough
#     to render the point estimate practically meaningless.
#
# On signal extraction vs. noise reactivity:
#   - The focus here is on sensing future growth dynamics hidden within
#     present-day data: extracting the weak systematic signal while
#     suppressing the dominating noise.
#   - The concept of "signal" is grounded in the notion of the business cycle.
#   - FED Chair Jerome Powell, speaking at the University of Chicago Booth
#     School of Business (March 7, 2025), stated:
#       "As we parse the incoming information, we are focused on separating
#        the signal from the noise as the outlook evolves,"
#     implying that monetary policy should not overreact to noise.
#   - Analogously, a well-designed forecast procedure should not be reactive
#     to high-frequency noise.
#   - Classic direct GDP forecasts targeting the full series (signal + noise)
#     are prone to overfitting: the high-frequency erratic components that
#     dominate the MSE distract OLS optimization from fitting the far weaker
#     but economically meaningful systematic dynamics.
#
# On the M-SSA approach:
#   - M-SSA is fundamentally about the dynamic quality of prediction.
#   - It emphasizes target correlation rather than mean-square error,
#     deliberately setting aside static level and scale adjustments
#     (calibration) that would be needed to produce GDP point forecasts.
#   - Instead of minimizing MSE unilaterally, M-SSA jointly optimizes
#     across the AST trilemma:
#       * Accuracy    – closeness to the target signal
#       * Smoothness  – few false alarms (controlled zero-crossing rate)
#       * Timeliness  – left-shift / lead / advancement of the signal
#   - M-SSA controls the rate of zero-crossings, i.e., how frequently the
#     predictor crosses above or below the long-term average growth rate.
#     This directly governs the number of spurious recession/recovery signals.
#
# On interpreting M-SSA output:
#   - Predictors are standardized: positive readings indicate above-trend
#     growth; negative readings indicate below-trend growth.
#   - GDP level forecasts ("numbers") could be recovered by calibrating
#     standardized M-SSA predictors onto observed GDP via linear regression
#     (optimal static level and scale adjustment). We don't.


# --- Structure of This Tutorial ---
#
# The tutorial is organized into four exercises:
#
# Exercise 1 – Target Specification and Initial M-SSA Application
#   - Discusses key design decisions (target specification).
#   - Applies M-SSA to the data.
#   - Documents the performance degradation caused by VAR(1) misspecification.
#
# Exercise 2 – Diagnosing and Resolving Misspecification
#   - Identifies the primary sources of model misspecification.
#   - Proposes and evaluates remedies to mitigate their adverse effects.
#
# Exercise 3 – Constructing the M-SSA GDP Predictors
#   - Integrates the insights from Exercises 1 and 2.
#   - Provides a step-by-step recipe for building reliable M-SSA BIP predictors.
#
# Exercise 4 – Comparison with the HP Filter
#   - Benchmarks M-SSA output against two-sided and one-sided HP filters.
#   - Highlights divergences between the two approaches.
#   - Notes that M-SSA contradicts the HP filter signal; future data will arbitrate.
#========================================================================


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

#----------------------
# Clear the workspace to ensure a clean environment before starting
rm(list = ls())

#------------------------------------------------------------------------
# Load Required R Libraries

# Standard filter package (used for HP-filter computations)
library(mFilter)

# Multivariate time series package: fits VARMA models to macroeconomic indicators
# Used here primarily for VAR model estimation and simulation
library(MTS)

# HAC (Heteroscedasticity and Autocorrelation Consistent) covariance estimator
# Provides robust standard deviation estimates under autocorrelation and heteroscedasticity
library(sandwich)


#------------------------------------------------------------------------
# Load M-SSA Functionality from Source Files

# Core M-SSA filter design and optimization functions
source(paste(getwd(), "/R/MSSA.r", sep = ""))

# Signal extraction functions developed for the JBCY paper (depends on mFilter)
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))

# Convenience wrapper functions for M-SSA workflows
source(paste(getwd(), "/R utility functions/M_SSA_utility_functions.r", sep = ""))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# There are potential problems when loading SSA together with MSSA.
# Some function names are the same but the underlying functions are different.
# Advice: M-SSA generalizes SSA, therefore there is no need to load the SSA 
# functions in addition to M-SSA. All relevant function for M-SSA are packed 
# in functions_MSSA.r. DO NOT SOURCE simple_sign_accuracy.r when working with 
# M-SSA
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



#========================================================================
# Exercise 1: Apply M-SSA to Quarterly German Macroeconomic Data
#========================================================================

#------------------------------------------------------------------------
# 1.1 Load Data and Select Indicators

# --- 1.1.1 Raw Data Files ---
# These files contain the original (unprocessed) data.
# GDP ("BIP": Brutto Inlandsprodukt) point forecasts would refer to this raw data.
data_file_name <- c("Data_HWI_2025_02.csv", "gdp_2025_02.csv")

# Monthly indicators (note: industrial production (ip) may not be available up
# to the sample end due to its publication lag)
data_monthly <- read.csv(paste(getwd(), "/Data/", data_file_name[1], sep = ""))
tail(data_monthly)

# Quarterly data: BIP is located in the first data column
data_quarterly <- read.csv(paste(getwd(), "/Data/", data_file_name[2], sep = ""))
tail(data_quarterly)


# --- 1.1.2 Data Transformations ---
# We do not work directly with the raw data. Instead, the following sequential
# transformations are applied to each series before use:
#
#   Step 1 – Log-transform:    Stabilizes variance and ensures positivity.
#   Step 2 – First differences (quarterly): Converts levels to growth rates,
#              emphasizing cyclical dynamics over trending behavior.
#   Step 3 – Standardization: Centers each series to zero mean and scales to
#              unit variance, making series comparable across indicators.
#   Step 4 – Trimming:        Clips extreme outliers to ±3 standard deviations.
#              This specifically addresses the singular COVID-19 shock, which
#              would otherwise distort filter and model estimates.
#
# As a result, the plotted values are not directly interpretable as GDP levels
# or industrial production figures. However, these transformations are essential
# for isolating the weak systematic signal from the dominating high-frequency noise.
#
# If point forecasts in original units are needed, the M-SSA output can be
# traced back via straightforward inverse transformations (unscaling, cumulation,
# and exponentiation).

# Load the pre-transformed macro data file (all transformations already applied)
# Data source:
#   - All series were provided by Katja Heinisch (IWH – Halle Institute for
#     Economic Research), co-author of Heinisch, van Norden & Wildi (2026).
#   - For variable definitions, data transformations, and original sources,
#     see the data appendix of the cited paper.
load(file = paste(getwd(), "\\Data\\macro", sep = ""))
tail(data)

# Notes on the structure of the loaded data file:
#
# - All series are standardized (zero mean, unit variance after trimming).
#
# - Column 1 (Target): BIP forward-shifted by the publication lag.
#     This is the series whose two-sided HP-filter output M-SSA aims to predict.
#
# - Column 2 onward (Explanatory variables): Indicators available in real time
#     as of January 2025, used for nowcasting or forecasting the target.
#     BIP appears again in column 2 as an explanatory variable; it is lagged
#     by two quarters relative to the target column due to its publication lag.
#
# - Nowcast (delta = 0): Estimates the two-sided HP-filter applied to the
#     target column at the current period.
#
# - Forecast at horizon h = 4: Shifts the target forward by one additional year
#     relative to the nowcast horizon.


# --- Notes on the Publication Lag ---
#
# BIP has an official publication lag of one quarter. However, in this design
# the target column is shifted one additional quarter forward as a safety margin
# to guard against data revisions (which are otherwise ignored here).
#
# Practical implication: a forecast labeled as "three quarters ahead" in our
# plots may effectively correspond to a full year ahead in real time.
# This conservative shift provides a buffer for late or revised data releases.


# Specify the publication lag vector (in quarters):
#   - BIP (target column): 2 quarters (1 official lag + 1 safety margin)
#   - All other series:    0 quarters (assumed available in real time)
# This vector will be used throughout to determine the effective forward-shift:
#   effective shift = forecast horizon (delta) + lag_vec<a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[1]</a>
lag_vec <- c(2, rep(0, ncol(data) - 1))
lag_vec

# --- Plot the Transformed Data ---
par(mfrow = c(1, 1))
mplot      <- data
colo       <- c("black", rainbow(ncol(data) - 1))
main_title <- paste("Quarterly design BIP: the target (black) assumes a publication lag of ",
                    lag_vec[1], " quarters", sep = "")

plot(mplot[, 1],
     main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     lwd  = c(2, rep(1, ncol(data) - 1)),
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
axis(1, at     = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# Observations from the plot:
#
# Publication lags:
#   - BIP (red): lags the target BIP (black) by lag_vec = 2 quarters.
#   - ip (orange) is lagging when compared to ifo, ESI, or spread.
#   - During crisis episodes, the target column visibly leads other series
#     by roughly one quarter, consistent with the imposed forward shift.
#     This forward shift provides a safety margin for data revisions.
#
# Standardization and trimming:
#   - All series are standardized, making cross-indicator comparisons meaningful.
#   - COVID-19 observations are trimmed to ±3 standard deviations.
#     The trimming also mildly affects the 2008 financial crisis readings,
#     though to a far lesser extent.
#
# Why trimming matters:
#   - Classic approaches (HP filter, VAR models) are sensitive to extreme outliers.
#   - M-SSA's optimization algorithm is inherently robust against singular events,
#     but can be affected indirectly through the VAR model used for MA-inversion,
#     which is itself sensitive to COVID-era outliers.


#------------------------------------------------------------------------
# 1.2 Select Indicators and Construct the Data Matrix

# Five-dimensional multivariate design, selected based on domain expertise:
#   BIP       – German GDP (quarterly, lagged)
#   ip        – Industrial production index
#   ifo_c     – ifo Business Climate Index (current assessment component)
#   ESI       – European Commission Economic Sentiment Indicator
#   spr_10y_3m – Term spread: 10-year minus 3-month government bond yield
select_vec_multi <- c("BIP", "ip", "ifo_c", "ESI", "spr_10y_3m")

# Construct the multivariate data matrix for M-SSA
x_mat <- data[, select_vec_multi]
rownames(x_mat) <- rownames(data)

# Number of series (multivariate dimension)
n <- dim(x_mat)[2]
  
# Total number of observations (sample length)
len <- dim(x_mat)[1]
  
# Note: the target column (column 1 of 'data') is excluded from x_mat.
# It is treated as redundant here because in all comparisons we reconstruct
# the target by shifting the BIP column upward by (lag_vec[1] + delta) periods,
# where delta = 0 for nowcasting and delta > 0 for multi-step forecasting.
tail(x_mat)


#========================================================================
# 1.2 Target Filter Specification
#========================================================================
#
# Motivation:
#   A filter is applied to the target series (BIP shifted forward by
#   lag_vec[1] + delta quarters) in order to suppress the unpredictable
#   high-frequency noise component of GDP growth.
#
# Forecast philosophy:
#   By removing the unpredictable portion (noise), M-SSA can focus on
#   the predictable portion (signal). As demonstrated in Tutorial 7.3,
#   this approach enables statistically significant predictions several
#   quarters ahead.
#
# Key design question:
#   Which target filter is appropriate for multi-step-ahead GDP forecasting?
#   This choice critically influences what M-SSA treats as the "signal" and
#   therefore what it tries to predict. The selection requires careful
#   consideration of:
#     (a) The characteristics of the data.
#     (b) The primary objectives and priorities of the forecast design.
#
# Important note on the role of the target in M-SSA:
#   - Target specification is exogenous to M-SSA.
#   - Once a target is specified, M-SSA derives the optimal predictor for it.
#   - The discussion below is not about M-SSA's internal optimization, but
#     about the upstream design decision that guides what M-SSA looks for.


# --- Why HP(1600) is Unsuitable Here ---
#
# The classic quarterly HP filter uses lambda = 1600 (HP(1600)).
# However, HP(1600) has several drawbacks in this forecasting context:
#
#   1. Excessive smoothness: HP(1600) over-smooths the cycle, obscuring
#      narrow recession dips (e.g., the Euro-Area sovereign debt crisis).
#      Phillips and Jin (2021) specifically criticize HP(1600) as "too smooth"
#      and insufficiently flexible for capturing cyclical dynamics.
#
#   2. Weak dynamic variation: Quarter-to-quarter changes in HP(1600) are
#      very small over a one-year horizon, making it difficult for M-SSA to
#      generate meaningful left-shifts (signal anticipation/lead) as the
#      forecast horizon increases.
#
#   3. Sensitivity to the COVID-19 pandemic: The finite-length truncated
#      HP(1600) filter produces visually distorted output around the pandemic
#      episode.
#
# --- Why HP(160) is Preferred ---
#
# We select HP(160), a more adaptive specification, for the following reasons:
#
#   1. Better recession tracking: HP(160) can follow narrow recession dips
#      more accurately, including the 2011–2012 Euro-Area sovereign debt crisis.
#
#   2. Richer dynamic variation: HP(160) exhibits larger quarter-to-quarter
#      changes over a one-year horizon, enabling M-SSA to generate effective
#      left-shifts as a function of the forecast horizon.
#
#   3. Reduced pandemic sensitivity: HP(160) is less distorted by the COVID-19
#      outlier than HP(1600) when the filter is truncated to a finite length.
#
# Tutorial 7.3 will explore even more adaptive target specifications.


# Regularization parameter for the HP target filter
lambda_HP <- 160

# Filter half-length (approximately 4 years of quarterly data = 16 quarters each side).
# The total filter length L must be odd to allow symmetric mirroring of the right
# tail about the center point when constructing the two-sided target (see Tutorial 7.1).
L <- 31

# Compute the HP target filter coefficients using the mFilter-based wrapper.
# Returns: filter coefficient matrix (gamma_target) and symmetry flag (symmetric_target).
target_obj <- HP_target_sym_T(n, lambda_HP, L)

# Transpose to obtain one row of coefficients per series (n rows, n*L columns)
gamma_target <- t(target_obj$gamma_target)
dim(gamma_target)

# Boolean flag: TRUE indicates the target is symmetric, so the right tail shown
# in the plot below will be mirrored to the left to form the full two-sided filter.
symmetric_target <- target_obj$symmetric_target

# Assign series names to the filter coefficient matrix for readability
colnames(gamma_target) <- select_vec_multi


# --- Plot Target Filter Coefficients ---
#
# Each colored line shows the right tail of the HP target filter for one series.
# In M-SSA, these right tails are mirrored about the center peak to construct
# the full symmetric two-sided HP target (one per series in the multivariate design).
# Vertical lines delimit the coefficient segments for each series.
par(mfrow = c(1, 1))
ts.plot(gamma_target,
        col  = rainbow(n),
        main = "Target filter right tails: mirrored at center peak to form the two-sided HP target")
abline(v = 1 + (1:n) * L)

# Confirm that the symmetric mirroring flag is active (should be TRUE)
symmetric_target
# -> When TRUE, M-SSA uses the full two-sided HP filter as the target.


#========================================================================
# 1.3 VAR Model Estimation
#========================================================================
#
# A VAR model is fitted to the multivariate data to characterize the DGP.
# The in-sample span choice has only a modest effect on the final M-SSA
# predictor because the VAR is sparsely parametrized (order p = 1 with
# regularization via coefficient thresholding).
#
# Three candidate in-sample spans are considered:
#   (a) Full sample up to January 2025
#   (b) Pre-pandemic (up to end of 2019)
#   (c) Pre-financial-crisis (up to end of 2008)  <- selected here
#
# Rationale for selecting the pre-financial-crisis span:
#   Explore performances on a challenging long out-of-sample span (nearly 20 years)

# Option (a): Full sample
date_to_fit <- "2200"
# Option (b): Pre-pandemic
date_to_fit <- "2019"
# Option (c): Pre-financial-crisis (active selection)
date_to_fit <- "2008"

# Subset data to the selected in-sample span, removing any rows with missing values
data_fit <- na.exclude(x_mat[which(rownames(x_mat) < date_to_fit), ])

# Inspect cross-correlations within the in-sample period to guide VAR specification
acf(data_fit)

# Verify the last observations included in the estimation sample
tail(data_fit)


# --- Fit VAR(1) Model ---
p <- 1   # VAR order (autoregressive lags)
q <- 0   # VMA order (moving average lags; zero here -> pure VAR)

set.seed(12)  # Fix random seed for reproducibility
V_obj <- VARMA(data_fit, p = p, q = q)

# Apply coefficient regularization (thresholding) to reduce overfitting.
# Coefficients with t-statistics below the threshold are set to zero.
# See the MTS package vignette for details on the refVARMA procedure.
threshold <- 1.5
V_obj <- refVARMA(V_obj, thres = threshold)

# Optional: inspect model diagnostics (residual autocorrelations, portmanteau tests).
# Results should appear reasonable for the pre-financial-crisis estimation sample.
if (F)
  MTSdiag(V_obj)

# Extract estimated model components:
Sigma <- V_obj$Sigma   # Residual covariance matrix (n x n)
Phi   <- V_obj$Phi     # VAR(1) coefficient matrix (n x n)
Theta <- V_obj$Theta   # VMA coefficient matrix (n x n; zero matrix for pure VAR)


#========================================================================
# 1.4 MA-Inversion of the VAR Model
#========================================================================
#
# M-SSA requires the Wold (MA) representation of the DGP to derive optimal
# filter coefficients. This is obtained by inverting the fitted VAR model.
#
# The MA-inversion produces a set of impulse response matrices (xi), one for
# each lag up to the filter half-length L. These can be interpreted as follows:
#
#   - Weight magnitude: large MA weights for series j in the equation for
#     series i indicate that series j is an important driver of series i.
#
#   - Decay rate: slow decay of MA weights implies long memory in the DGP;
#     rapid decay implies short memory.
#
#   - Lead/lag structure: the peak of the MA weights for the term spread
#     (spr_10y_3m) is typically right-shifted, consistent with its role as
#     a leading indicator.
#
#   - Consistency check: the MA weights should broadly reflect the patterns
#     visible in the earlier cross-correlation (ACF) plot.
#
# A plot of the MA-inversion results will appear in the plot panel automatically.

MA_inv_obj <- MA_inv_VAR_func(Phi, Theta, L, n, T)

# Extract the MA coefficient array (dimensions: L x n x n)
# xi[l, i, j] gives the MA weight at lag l for the effect of series j on series i
xi <- MA_inv_obj$xi

#========================================================================
# 1.5. Call to M-SSA: specify the forecast horizon and the holding-time (HT) constraint(s)
#========================================================================

# Define the forecast horizon: one year ahead (4 quarters) plus the publication lag
#   The target GDP is the explanatory GDP (first column) shifted forward by 6 quarters
delta <- 4 + lag_vec[1]
delta

# Define the HT constraints for M-SSA (one per series).
# These values were chosen to enforce greater smoothness than the classic M-MSE predictor
# (i.e., each imposed HT exceeds the HT of the corresponding M-MSE filter).
# See below for further context.
ht_mssa_vec <- c(6.380160, 6.738270, 7.232453, 7.225927, 7.033768)
names(ht_mssa_vec) <- colnames(x_mat)
ht_mssa_vec

# Apply the M-SSA wrapper introduced in tutorial 7.1.
# Required inputs:
#   - delta            : forecast horizon (in quarters, including publication lag)
#   - ht_mssa_vec      : vector of HT constraints, one per series
#   - xi, Sigma        : MA-inversion coefficients and noise covariance matrix
#                        (characterizing the data-generating process)
#   - gamma_target     : target (e.g., HP filter transfer function)
#   - symmetric_target : Boolean flag indicating whether the target is symmetric
#                        (i.e., whether the left tail mirrors the right tail)
#   - T                : Boolean flag; if TRUE, a plot of the filter coefficients is produced
MSSA_main_obj <- MSSA_main_func(delta, ht_mssa_vec, xi, symmetric_target, gamma_target, Sigma, T)

# Extract the M-SSA filter coefficient matrix
bk_x_mat <- MSSA_main_obj$bk_x_mat

# Retrieve the full M-SSA output object (contains additional diagnostics; see tutorial 7.1)
MSSA_obj <- MSSA_main_obj$MSSA_obj

# Extract the classic M-MSE filter coefficients for comparison.
# M-SSA is designed to track M-MSE while producing smoother output (fewer spurious zero-crossings).
gammak_x_mse <- MSSA_obj$gammak_x_mse
colnames(bk_x_mat) <- colnames(gammak_x_mse) <- select_vec_multi


# Verify that the imposed M-SSA holding times exceed the native HTs of the M-MSE predictor.
# A larger HT corresponds to a smoother signal with fewer zero-crossings.
# M-SSA generates roughly 2-3 times less zero-crossings than classic M-MSE
for (i in 1:ncol(gammak_x_mse))
  print(paste("HT of M-MSE for series '", select_vec_multi[i], "': ",
              compute_holding_time_func(gammak_x_mse[, i])$ht,
              " vs. HT imposed on M-SSA: ", ht_mssa_vec[i], sep = ""))
# The M-MSE benchmark typically produces noisy signals; M-SSA provides explicit control
# over the holding time, making the predictor smoother in a well-defined sense.


#========================================================================
# 1.6. Apply the M-SSA filter to the data
#========================================================================

# Note: the forecast horizon delta incorporates the publication lag,
# so the output of the two-sided (target) filter is automatically left-shifted accordingly.
filt_obj <- filter_func(x_mat, bk_x_mat, gammak_x_mse, gamma_target, symmetric_target, delta)

mssa_mat   <- filt_obj$mssa_mat    # M-SSA filtered output
target_mat <- filt_obj$target_mat  # Two-sided target filter output (reference signal)
mmse_mat   <- filt_obj$mmse_mat    # M-MSE filtered output (benchmark)

# Sanity check: confirm that the two-sided target (HP filter applied to GDP) is shifted
# forward by (publication lag + delta) = 2 + 4 = 6 quarters relative to the raw data.
#   Six NAs in the M-SSA column
cbind(target_mat[, 1], mssa_mat[, 1])[(L - 6):(L + 5), ]


# Plot M-SSA, M-MSE, and the target for each series.
# Colours: black = target, blue = M-SSA, green = M-MSE.
# A vertical dashed line marks the end of the in-sample estimation period.
for (i in n:1)
{
  par(mfrow = c(1, 1))
  mplot <- cbind(target_mat[, i], mssa_mat[, i], mmse_mat[, i])
  colnames(mplot) <- c(
    paste("Target: HP applied to ", select_vec_multi[i],
          ", left-shifted by ", delta - lag_vec[1], " (+publication lag) quarters", sep = ""),
    "M-SSA",
    "M-MSE"
  )
  
  colo       <- c("black", "blue", "green")
  main_title <- paste("M-SSA ", select_vec_multi[i],
                      ": forward-shift = ", delta - lag_vec[1],
                      ", in-sample period ending ", rownames(data_fit)[nrow(data_fit)], sep = "")
  
  plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
       col = colo[1], lwd = c(2, rep(1, ncol(data) - 1)),
       ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
  mtext(colnames(mplot)[1], col = colo[1], line = -1)
  
  for (i in 1:ncol(mplot))
  {
    lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
  
  abline(h = 0)
  # Vertical line indicating the end of the in-sample span
  abline(v = which(rownames(mplot) == rownames(data_fit)[nrow(data_fit)]), lwd = 2, lty = 2)
  axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
       labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
  axis(2)
  box()
}


#========================================================================
# 1.7. Compute performance metrics
#========================================================================

# Sample mean squared error (MSE) of M-SSA relative to the target
#   The estimate is computed on the entire sample (in- and out-of sample spans)
#   The symmetric filter output is missing at sample start and end
apply(na.exclude((target_mat - mssa_mat)^2), 2, mean)

# Sample MSE of M-MSE relative to the target.
# Note: in a correctly specified model with a long sample, M-SSA will have a higher MSE than M-MSE
# because the imposed HT constraint explicitly trades off MSE optimality for smoother output.
# Here we see that M-MSE does not always outperform M-SSA (model misspecification)
apply(na.exclude((target_mat - mmse_mat)^2), 2, mean)

# Sample correlations between the target and M-SSA output.
# In large samples, these converge to the theoretical criterion values (see tutorial 7.1),
# provided the VAR model is correctly specified.
# Warning: the following results may look poor (e.g., negative correlations) due to
# model misspecification or a short estimation sample.
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[, i], mssa_mat[, i])))[1, 2])

# Compare sample correlations to the theoretical (criterion) values optimized by M-SSA.
# The criterion values represent the expected sample correlations under the true model:
#   - By construction (maximization), all criterion values are positive.
#   - A large discrepancy between sample and criterion correlations indicates either
#     VAR(1) misspecification or insufficient sample length (or both).
MSSA_obj$crit_rhoy_target

# These issues will be diagnosed and addressed in Exercise 2 below.

# Holding-time diagnostics:
# M-SSA maximizes target correlation subject to the HT constraint.
# The following compares empirical HTs (from the filtered output) to the imposed values.
# Sample HTs converge to the imposed HTs as the sample length increases.
unlist(apply(mmse_mat, 2, compute_empirical_ht_func))  # Empirical HT of M-MSE
unlist(apply(mssa_mat, 2, compute_empirical_ht_func))  # Empirical HT of M-SSA
ht_mssa_vec                                            # Imposed HT targets for M-SSA

# A ratio below 1 confirms that M-MSE is noisier (shorter holding time) than M-SSA,
# as intended. All ratios should be less than 1 for the constraint to be effective.
unlist(apply(mmse_mat, 2, compute_empirical_ht_func)) /
  unlist(apply(mssa_mat, 2, compute_empirical_ht_func))

# Conclusion: M-SSA successfully produces smoother output than the M-MSE benchmark,
# as measured by the holding-time criterion.
# But sample target correlations are negative.




#========================================================================
# Exercise 2
#========================================================================
# We address the following questions:
#   1. What are the main causes of model misspecification in this setting?
#   2. How can we improve forecast performance, and in particular the sample target correlations
#      (which were unsatisfactory in Exercise 1)?

# The plot of GDP (BIP) below suggests two issues:
#   1. The forecast problem is inherently difficult due to a low signal-to-noise ratio.
#   2. The predictors appear to lag the target (i.e., they are right-shifted / retarded).
#      Note: in this plot, the target has been left-shifted by lag_vec[1] + delta quarters
#      to align it with the real-time predictor output.
par(mfrow = c(1, 1))
i <- 1
mplot <- cbind(target_mat[, i], mssa_mat[, i], mmse_mat[, i])
colnames(mplot) <- c(
  paste("Target: HP applied to ", select_vec_multi[i],
        ", left-shifted by ", delta - lag_vec[1], " (plus publication lag) quarters", sep = ""),
  "M-SSA",
  "M-MSE"
)

colo       <- c("black", "blue", "green")
main_title <- paste("M-SSA ", select_vec_multi[i], ": ",
                    delta - lag_vec[1], "-quarter-ahead forecast, in-sample period ending ",
                    rownames(data_fit)[nrow(data_fit)], sep = "")

plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], lwd = c(2, rep(1, ncol(data) - 1)),
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
abline(v = which(rownames(mplot) == rownames(data_fit)[nrow(data_fit)]), lwd = 2, lty = 2)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()


# Proposed remedy: introduce a 'forecast excess'
# Instead of using the strictly necessary forecast horizon, we deliberately over-specify delta.
# This artificially advances (left-shifts) the M-SSA predictor, compensating for the
# rightward displacement observed above.
f_excess    <- 4
delta_excess <- delta + f_excess  # Augmented forecast horizon passed to M-SSA
delta_excess

# Re-run M-SSA with the augmented forecast horizon; all other settings remain unchanged.
MSSA_main_obj <- MSSA_main_func(delta_excess, ht_mssa_vec, xi, symmetric_target, gamma_target, Sigma, T)

bk_x_mat_excess <- MSSA_main_obj$bk_x_mat   # M-SSA filter coefficients under the excess horizon
MSSA_obj        <- MSSA_main_obj$MSSA_obj    # Full M-SSA output object

# Extract the M-MSE benchmark filter for reference
gammak_x_mse <- MSSA_obj$gammak_x_mse
colnames(bk_x_mat) <- colnames(gammak_x_mse) <- select_vec_multi

# Apply the augmented M-SSA filter to the data.
# Important: the target shift uses the original delta (not delta_excess),
# so the target reference signal remains unchanged.
filt_obj <- filter_func(x_mat, bk_x_mat_excess, gammak_x_mse, gamma_target, symmetric_target, delta)

# Extract the augmented M-SSA output for comparison
mssa_excess_mat <- filt_obj$mssa_mat


# Re-evaluate sample target correlations for the augmented M-SSA.
# Recall: the original M-SSA produced negative sample correlations (Exercise 1).
# With the forecast excess applied, the correlations become positive — a clear improvement.
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[, i], mssa_excess_mat[, i])))[1, 2])

# Interpretation of the improvement:
#   - The excess forecast horizon introduces a commensurate left-shift (lead) in the M-SSA predictor.
#   - This allows the predictor to track recession troughs more promptly.
#   - Timelier trough detection restores a positive sample target correlation.
#   - Root cause: the fitted VAR(1) cannot reproduce the sharp, asymmetric recession dips
#     present in the data — it is structurally misspecified for such episodes.
#   - The forecast excess serves as a pragmatic correction: by requesting M-SSA to be
#     'faster than necessary' under the VAR(1), we compensate for this misspecification.


# Holding-time diagnostics for the augmented M-SSA:
# The HT constraint is enforced independently of the forecast horizon, so empirical HTs
# should remain close to the imposed targets regardless of the excess applied.
unlist(apply(mssa_excess_mat, 2, compute_empirical_ht_func))  # Empirical HT: augmented M-SSA

# Compare with the original M-SSA HTs; any differences reflect random sample variation.
unlist(apply(mssa_mat, 2, compute_empirical_ht_func))         # Empirical HT: original M-SSA

# Confirm that the augmented M-SSA remains substantially smoother than the M-MSE benchmark.
# All ratios should be less than 1 (M-MSE HT < M-SSA HT), indicating M-MSE is noisier.
unlist(apply(mmse_mat, 2, compute_empirical_ht_func)) /
  unlist(apply(mssa_excess_mat, 2, compute_empirical_ht_func))


# Comparative plot: original vs. augmented M-SSA against the target.
# All series are standardized (z-scored) to facilitate visual comparison on a common scale.
par(mfrow = c(1, 1))
mplot <- scale(cbind(target_mat[, 1], mssa_excess_mat[, 1], mssa_mat[, 1]))
colnames(mplot) <- c(
  paste("Target: HP applied to ", select_vec_multi[1],
        ", left-shifted by ", delta - lag_vec[1], " quarters", sep = ""),
  "M-SSA with forecast excess",
  "M-SSA (original)"
)
colo       <- c("black", "red", "blue")
main_title <- "Standardized M-SSA: original (blue) vs. with forecast excess (red)"

plot(mplot[, 1], main = main_title, axes = F, type = "l", xlab = "", ylab = "",
     col = colo[1], lwd = c(2, rep(1, ncol(data) - 1)),
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
# Vertical dashed line marks the end of the in-sample estimation period
abline(v = which(rownames(mplot) == rownames(data_fit)[nrow(data_fit)]), lwd = 2, lty = 2)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# The plot confirms that the forecast excess produces a visible left-shift of M-SSA-excess (red),
# which is the direct cause of the improved sample target correlations.
# Additional remarks:
#   - If the data were truly generated by the fitted VAR(1), marked recession dips would be
#     unlikely, and the left-shift would not yield an improvement (it could even be harmful).
#   - However, in the presence of strong, asymmetric downturns, a 'faster' filter is better
#     positioned to capture the relevant recessionary dynamics in a timely manner.

# Current economic reading (based on the plotted output):
#   - Both predictors reached their trough around late 2023 and are now reverting
#     toward average growth (the zero line).
# Methodological note on growth-rate timing:
#   - The trough of the growth-rate series plotted here typically leads the trough of the
#     GDP level (BIP) by up to several quarters.

# Summary of findings and proposed remedies for VAR(1) misspecification:
#
# When the data exhibit pronounced recession episodes that a VAR(1) cannot reproduce,
# two complementary adjustments can improve real-time forecast performance:
#
#   1. Forecast excess: increase delta beyond the strictly required horizon to introduce
#      a compensatory left-shift (lead) in the predictor.
#
#   2. Amplitude recalibration: the left-shifted predictor may exhibit excessive shrinkage
#      toward zero; a rescaling (calibration) step can restore appropriate signal amplitude.
#
# Both adjustments are applied in the construction of real-time GDP (BIP) predictors below.


#========================================================================
# Exercise 3: Construction of Standardized M-SSA GDP (BIP) Predictors
#========================================================================
# Overview:
#   M-SSA produces five filtered outputs — one for each series: BIP, IP, IFO, ESI, and spread.
#   We construct a composite BIP predictor in three steps:
#
#   A. Equal-weighting (cross-sectional aggregation):
#      Each of the five M-SSA outputs is treated as an equally informative predictor for future BIP.
#      All five series are standardized (z-scored) before aggregation, and then averaged
#      with equal weights (1/5 each).
#
#   B. Forecast excess:
#      As motivated in Exercise 2, we set delta larger than the strictly required forecast horizon.
#      This compensates for the rightward displacement caused by VAR(1) misspecification
#      in the presence of asymmetric recession episodes.
#
#   C. Multiple forecast horizons:
#      We construct M-SSA predictors targeting BIP at horizons h = 0 (nowcast), 1, 2, 4, and 6
#      quarters ahead. We then evaluate all 5 x 5 = 25 combinations of predictor horizon and
#      target shift, and assess statistical significance using HAC-based variance estimates.

# Define the forecast horizons of interest
h_vec <- c(0, 1, 2, 4, 6)

# Set the forecast excess hyperparameter.
# A larger excess produces a stronger left-shift (lead) but at the cost of increased noise.
# Option 1 — Aggressive setting (maximum lead, higher noise risk):
f_excess <- 6
# Option 2 — Moderate setting (recommended starting point):
f_excess <- 4

# Initialize storage matrices for the M-SSA output of each series across all forecast horizons
mssa_bip <- mssa_ip <- mssa_esi <- mssa_ifo <- mssa_spread <- NULL

# Main loop: compute M-SSA predictors for each forecast horizon in h_vec
for (i in 1:length(h_vec))  # i <- 1
{
  # Determine the effective forecast horizon delta passed to M-SSA.
  # delta is a hyperparameter that governs the filter optimization; it may differ from h_vec[i].
  # For BIP, delta incorporates:
  #   - the target forecast horizon h_vec[i]
  #   - the BIP publication lag lag_vec<a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[1]</a> (BIP is released with a delay)
  #   - the forecast excess f_excess (compensates for VAR misspecification)
  delta <- h_vec[i] + lag_vec[1] + f_excess
  
  # --- Step 1: M-SSA for BIP ---
  # Run M-SSA with the BIP-adjusted delta
  MSSA_main_obj <- MSSA_main_func(delta, ht_mssa_vec, xi, symmetric_target, gamma_target, Sigma, T)
  
  bk_x_mat <- MSSA_main_obj$bk_x_mat   # M-SSA filter coefficients
  MSSA_obj  <- MSSA_main_obj$MSSA_obj   # Full M-SSA output object
  colnames(bk_x_mat) <- select_vec_multi
  
  # Apply the M-SSA filter to the data
  filt_obj   <- filter_func(x_mat, bk_x_mat, gammak_x_mse, gamma_target, symmetric_target, delta)
  mssa_mat   <- filt_obj$mssa_mat
  target_mat <- filt_obj$target_mat
  mmse_mat   <- filt_obj$mmse_mat
  colnames(mssa_mat) <- select_vec_multi
  
  # Retain only the BIP column from the M-SSA output; append to storage matrix
  mssa_bip <- cbind(mssa_bip, mssa_mat[, which(colnames(mssa_mat) == select_vec_multi[1])])
  
  # --- Step 2: M-SSA for remaining indicators (IP, IFO, ESI, spread) ---
  # These series are available without a significant publication lag (or with a smaller lag than BIP).
  # Therefore, delta does not include lag_vec[1] — only the forecast horizon and excess are used.
  delta <- h_vec[i] + f_excess
  
  # Re-run M-SSA with the reduced delta for the non-BIP indicators
  MSSA_main_obj <- MSSA_main_func(delta, ht_mssa_vec, xi, symmetric_target, gamma_target, Sigma, T)
  
  bk_x_mat <- MSSA_main_obj$bk_x_mat
  MSSA_obj  <- MSSA_main_obj$MSSA_obj
  colnames(bk_x_mat) <- select_vec_multi
  
  # Apply the filter
  filt_obj   <- filter_func(x_mat, bk_x_mat, gammak_x_mse, gamma_target, symmetric_target, delta)
  mssa_mat   <- filt_obj$mssa_mat
  target_mat <- filt_obj$target_mat
  mmse_mat   <- filt_obj$mmse_mat
  colnames(mssa_mat) <- select_vec_multi
  
  # Retain the relevant column for each indicator and append to the corresponding storage matrix
  mssa_ip     <- cbind(mssa_ip,     mssa_mat[, which(colnames(mssa_mat) == select_vec_multi[2])])
  mssa_ifo    <- cbind(mssa_ifo,    mssa_mat[, which(colnames(mssa_mat) == select_vec_multi[3])])
  mssa_esi    <- cbind(mssa_esi,    mssa_mat[, which(colnames(mssa_mat) == select_vec_multi[4])])
  mssa_spread <- cbind(mssa_spread, mssa_mat[, which(colnames(mssa_mat) == select_vec_multi[5])])
}

# Standardize each series (z-score) and aggregate across all five indicators using equal weights (1/5).
# Equal weighting reflects the assumption that each series contributes equally to BIP prediction.
indicator_mat <- (scale(mssa_bip) + scale(mssa_ip) + scale(mssa_ifo) +
                    scale(mssa_esi) + scale(mssa_spread)) / length(select_vec_multi)

# Assign horizon labels to columns and row dates to all output matrices
colnames(indicator_mat) <-
  colnames(mssa_bip) <- colnames(mssa_ip) <- colnames(mssa_ifo) <-
  colnames(mssa_esi) <- colnames(mssa_spread) <- paste("Horizon=", h_vec, sep = "")
rownames(indicator_mat) <- rownames(x_mat)

# Preview the most recent values of the composite predictor across all forecast horizons
tail(indicator_mat)

# Next: plot the target and predictors, and compute sample target correlations
# for all 5 x 5 combinations of forecast horizon and target forward-shift.
target_shifted_mat <- NULL
cor_mat <- matrix(ncol = length(h_vec), nrow = length(h_vec))

for (i in 1:length(h_vec)) # i <- 1
{
  # --------------------------------------------------------------
  # Compute the total forward-shift: forecast horizon + publication lag
  # (accounts for the fact that data are not yet available at forecast time)
  shift <- h_vec[i] + lag_vec[1]
    
# Apply a two-sided (symmetric) HP filter to BIP and all series,
# then shift the result forward by shift (i.e., forecast horizon + publication lag)
# to align the target with the predictors
  filt_obj <- filter_func(x_mat, bk_x_mat, gammak_x_mse, gamma_target, symmetric_target, shift)
  target_mat <- filt_obj$target_mat
    
# Extract the HP-filtered BIP series (first column) as the target
  target <- target_mat[, "BIP"]
    
# Append the forward-shifted target to the collection matrix
# (each column corresponds to one forecast horizon)
  target_shifted_mat <- cbind(target_shifted_mat, target)
    
# ------------------------------------------------------------------
# Plot: standardize and overlay the shifted target with all indicators
  mplot <- scale(cbind(target, indicator_mat))
  colnames(mplot)[1] <- paste("Target left-shifted by ", shift - lag_vec[1], sep = "")
  par(mfrow = c(1, 1))
  colo <- c("black", rainbow(ncol(indicator_mat)))
  main_title <- paste("Standardized M-SSA predictors for forecast horizons ",
                        paste(h_vec, collapse = ","), sep = "")
    
# Base plot: draw the shifted target in black
  plot(mplot[, 1], main = main_title, axes = F, type = "l",
       xlab = "", ylab = "", col = colo[1],
       lwd = c(2, rep(1, ncol(data) - 1)),
       ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
  mtext(colnames(mplot)[1], col = colo[1], line = -1)
    
# Overlay all M-SSA predictors in distinct colors
  for (j in 1:ncol(mplot))
  {
    lines(mplot[, j], col = colo[j], lwd = 1, lty = 1)
    mtext(colnames(mplot)[j], col = colo[j], line = -j)
  }
    
  abline(h = 0) # Reference line at zero
# Vertical dashed line marking the end of the in-sample estimation window
  abline(v = which(rownames(mplot) == rownames(data_fit)[nrow(data_fit)]),
         lwd = 2, lty = 2)
  axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
       labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
  axis(2)
  box()
    
# ------------------------------------------------------------------
# Compute the sample correlation between each M-SSA predictor and
# the forward-shifted HP-BIP target for the current forecast horizon.
# The resulting cor_mat will contain all (h x h) combinations of
# forecast horizon (rows) and predictor horizon (columns).
  for (j in 1:ncol(indicator_mat))
    cor_mat[i, j] <- cor(na.exclude(cbind(target, indicator_mat[, j])))[1, 2]
}
colnames(target_shifted_mat)<-paste("target h=",h_vec,sep="")
# ------------------------------------------------------------------
# Sanity check on the forward-shift alignment:
#   The target is shifted upward (into the future) by:
#     publication lag (assumed to be 2 quarters) + forecast horizon
#   relative to the predictor series shown in column 1.
cbind(indicator_mat[, 1], target_shifted_mat)[(L - 10):(L + 6), ]

# ------------------------------------------------------------------
# Inspect the full correlation matrix between M-SSA predictors and
# forward-shifted HP-BIP targets across all forecast horizons
# Note: the empirical correlation is based on full sample: in-sample + out-of-sample spans
colnames(cor_mat) <- paste("M-SSA: h=", h_vec, sep = "")
rownames(cor_mat) <- paste("Shift of target: ", h_vec, sep = "")
cor_mat
# Key observations from cor_mat:
#   - M-SSA predictors optimized for longer forecast horizons (left to right)
#     tend to correlate more strongly with correspondingly more forward-shifted
#     targets (top to bottom), producing a pronounced diagonal structure.
#   - For a given forward-shift (row), the largest correlation typically lies
#     on or near the diagonal element of that row.
#   - This systematic pattern suggests that M-SSA predictors carry genuine
#     information about future HP-filtered BIP trend growth.
#   - Since future BIP trend growth reflects the low-frequency component of
#     future BIP, the predictors are also informative about future BIP,
#     provided BIP is not pure white noise.
#   - In practice, (differenced) BIP is noisy, yet the occurrence of recessions
#     confirms it is not white noise. However, assessing forecast accuracy
#     directly for raw BIP is difficult (see Tutorial 7.3 for a more refined
#     out-of-sample analysis).
#   - For HP-filtered BIP, statistical significance of the diagonal pattern
#     can be assessed via regression with HAC-adjusted standard errors,
#     as carried out below.

# ------------------------------------------------------------------
# HAC-adjusted significance test for M-SSA predictors vs. HP-BIP targets
# For each (i, j) combination, regress the i-th forward-shifted HP-BIP target
# on the j-th M-SSA predictor and compute HAC-robust t-statistics and p-values.
t_HAC_mat <- p_value_HAC_mat <- matrix(ncol = length(h_vec), nrow = length(h_vec))

for (i in 1:length(h_vec)) # i <- 1
{
  for (j in 1:length(h_vec)) # j <- 1
  {
    # Regress the i-th forward-shifted HP-BIP target on the j-th M-SSA predictor
    lm_obj <- lm(target_shifted_mat[, i] ~ indicator_mat[, j])
    summary(lm_obj)
    
    # Classic OLS standard errors (for reference; replicates summary() output)
    sd <- sqrt(diag(vcov(lm_obj)))
    
    # HAC-robust standard errors (Newey-West type) via the sandwich package
    sd_HAC <- sqrt(diag(vcovHAC(lm_obj)))
    
    # Alternative way to obtain the same HAC-robust standard errors
    sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
    
    # HAC-adjusted t-statistic for the slope coefficient
    t_HAC_mat[i, j] <- summary(lm_obj)$coef[2, 1] / sd_HAC[2]
      
    # One-sided HAC-adjusted p-value:
    #   We test whether the regression coefficient is *positive*, since an
    #   effective predictor should co-move positively with future BIP trend growth.
    p_value_HAC_mat[i, j] <- pt(t_HAC_mat[i, j], len - length(select_vec_multi),
                                  lower = FALSE)
  }
}

colnames(t_HAC_mat) <- colnames(p_value_HAC_mat) <- paste("M-SSA: h=", h_vec, sep = "")
rownames(t_HAC_mat) <- rownames(p_value_HAC_mat) <- paste("Shift of target: ", h_vec, sep = "")

# Inspect HAC-adjusted p-values:
#   - Small p-values concentrate on (or near) the diagonal, confirming that
#     each M-SSA predictor is most informative for its intended forecast horizon.
#   - Statistical significance is maintained even at longer forecast horizons
#     after HAC adjustment.
#   - As expected, significance diminishes (p-values increase) as the
#     forward-shift grows beyond the predictor's design horizon.
p_value_HAC_mat

# Note:
#   - The analysis above uses the full sample, including the in-sample span.
#   - A more detailed assessment of forecast performance and statistical
#     significance, including out-of-sample results, is provided in Tutorial 7.3.

# ==================================================================
# Extension: Predictability of M-SSA indicators for raw (unfiltered) BIP
# ==================================================================
# The HP-BIP results above suggest predictability of future trend growth.
# We now investigate whether the same predictors are informative about
# future *raw* BIP (i.e., without HP filtering).

t_HAC_mat_BIP <- p_value_HAC_mat_BIP <- matrix(ncol = length(h_vec), nrow = length(h_vec))
BIP_target_mat <- NULL

for (i in 1:length(h_vec)) # i <- 4
{
  # Construct the forward-shifted raw BIP series:
  #   shift values into the future by (forecast horizon + publication lag)
  #   and pad the end with NAs to preserve the original series length
  shift <- h_vec[i] + lag_vec[1]
    BIP_target <- c(x_mat[(1 + shift):nrow(x_mat), "BIP"], rep(NA, shift))
    BIP_target_mat <- cbind(BIP_target_mat, BIP_target)
    
    # Regress each M-SSA predictor on the forward-shifted raw BIP target
    for (j in 1:length(h_vec)) # j <- 5
    {
      lm_obj <- lm(BIP_target ~ indicator_mat[, j])
      summary(lm_obj)
      
      # Classic OLS standard errors (replicates summary() output; for reference)
      sd <- sqrt(diag(vcov(lm_obj)))
      
      # HAC-robust standard errors via the sandwich package
      sd_HAC <- sqrt(diag(vcovHAC(lm_obj)))
      
      # Alternative computation of HAC-robust standard errors (identical result)
      sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
      
      # HAC-adjusted t-statistic for the slope coefficient
      t_HAC_mat_BIP[i, j] <- summary(lm_obj)$coef[2, 1] / sd_HAC[2]
        
        # One-sided HAC-adjusted p-value:
        #   A positive coefficient indicates the predictor correctly anticipates
        #   the direction of future BIP; negative signs are not of interest here.
        p_value_HAC_mat_BIP[i, j] <- pt(t_HAC_mat_BIP[i, j],
                                        len - length(select_vec_multi),
                                        lower = FALSE)
    }
}

colnames(t_HAC_mat_BIP) <- colnames(p_value_HAC_mat_BIP) <- paste("M-SSA: h=", h_vec, sep = "")
rownames(t_HAC_mat_BIP) <- rownames(p_value_HAC_mat_BIP) <- paste("Shift of target: ", h_vec, sep = "")



















for (i in 1:length(h_vec))#i<-1
{
# Forward-shift: forecast horizon plus publication lag  
  shift<-h_vec[i]+lag_vec[1]
# Compute target: two-sided HP applied to BIP and shifted forward by forecast horizon plus publication lag
  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,shift)
  target_mat=filt_obj$target_mat
# Select BIP (first column)  
  target<-target_mat[,"BIP"]
# Collect the forward shifted targets: 
  target_shifted_mat<-cbind(target_shifted_mat,target)
# Plot indicators and shifted target
  mplot<-scale(cbind(target,indicator_mat))
  colnames(mplot)[1]<-paste("Target left-shifted by ",shift-lag_vec[1],sep="")
  par(mfrow=c(1,1))
  colo<-c("black",rainbow(ncol(indicator_mat)))
  main_title<-paste("Standardized M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep="")
  plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(data)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
  mtext(colnames(mplot)[1],col=colo[1],line=-1)
  for (j in 1:ncol(mplot))
  {
    lines(mplot[,j],col=colo[j],lwd=1,lty=1)
    mtext(colnames(mplot)[j],col=colo[j],line=-j)
  }
  abline(h=0)
  abline(v=which(rownames(mplot)==rownames(data_fit)[nrow(data_fit)]),lwd=2,lty=2)
  axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
  axis(2)
  box()

# Compute sample target correlations of all M-SSA predictors with the shifted target: 
# The final matrix will contain all 5*5 combinations of forecast horizon and forward-shift
  for (j in 1:ncol(indicator_mat))
    cor_mat[i,j]<-cor(na.exclude(cbind(target,indicator_mat[,j])))[1,2]

}
# Check the shifts: 
#   The target is shifted upward by publication lag (assumed to be 2 quarters) + forecast horizon relative to the predictor (in the first column) 
cbind(indicator_mat[,1],target_shifted_mat)[(L-10):(L+6),]

# We can now have a look at the target correlations
colnames(cor_mat)<-paste("M-SSA: h=",h_vec,sep="")
rownames(cor_mat)<-paste("Shift of target: ",h_vec,sep="")
cor_mat
# -We can see that M-SSA predictors optimized for larger forecast horizons (from left to right in cor_mat) 
#     correlate more strongly with correspondingly forward-shifted targetS (from top to bottom in cor_mat)
# -For a given forward-shift (row), the largest correlations tend to lie on (or to be close to) the diagonal element of that row in cor_mat


# -We infer from the observed systematic pattern, that the M-SSA predictors tend to be informative about future BIP trend growth
# -Also, since future BIP trend growth tells something about the low-frequency part of future BIP, we may infer that 
#   the M-SSA predictors are informative about future BIP (assuming the latter is not white noise)
# -However, (differenced) BIP is a noisy series (the existence of recessions suggests that it is not white noise)
# -Therefore, it is difficult to assess statistical significance of forecast accuracy with respect to BIP (see tutorial 7.3 for a more refined analysis)
# -But we can assess statistical significance of the effect observed in cor_mat, with respect to HP-BIP (low-frequency part of BIP)
# -For this purpose we regress the predictors on the shifted targets and compute HAC-adjusted p-values of the corresponding regression coefficients
t_HAC_mat<-p_value_HAC_mat<-matrix(ncol=length(h_vec),nrow=length(h_vec))
for (i in 1:length(h_vec))# i<-1
{
  for (j in 1:length(h_vec))# j<-1
  {
# Regress j-th M-SSA predictor on i-th target    
    lm_obj<-lm(target_shifted_mat[,i]~indicator_mat[,j])
    summary(lm_obj)
# This estimate of the variance matrix replicates std in the above summary (classic OLS estimate of variance)
    sd<-sqrt(diag(vcov(lm_obj)))
# Here we use HAC: we rely on the R-package sandwich  
    sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
# This is the same as
    sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
# Compute HAC-adjusted t-statistic
    t_HAC_mat[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
# Compute HAC-adjusted p-value: 
#   -We consider a one-sided test because we expect the regression coefficient (of predictor on target) 
#     to be positive
    p_value_HAC_mat[i,j]<-pt(t_HAC_mat[i,j], len-length(select_vec_multi), lower=FALSE)
  }
}
colnames(t_HAC_mat)<-colnames(p_value_HAC_mat)<-paste("M-SSA: h=",h_vec,sep="")
rownames(t_HAC_mat)<-rownames(p_value_HAC_mat)<-paste("Shift of target: ",h_vec,sep="")
# p-values: small p-values lie on (or close to) the diagonal
# Statistical significance (after HAC-adjustment) is achieved even towards larger forecast horizons
# As expected, the Significance decreases (p-values increase) with increasing forward-shift
p_value_HAC_mat
# Note: 
#   -We here consider the full sample, including the in-sample span.
#   -We shall examine performances and statistical significance in more detail in tutorial 7.3,
#     including out-of-sample results

#--------------------------------------------------
# The above result suggest predictability of M-SSA indicators with respect to future HP-BIP
# What about future BIP?
t_HAC_mat_BIP<-p_value_HAC_mat_BIP<-matrix(ncol=length(h_vec),nrow=length(h_vec))
BIP_target_mat<-NULL
for (i in 1:length(h_vec))# i<-4
{
# Shift BIP  
  shift<-h_vec[i]+lag_vec[1]
  BIP_target<-c(x_mat[(1+shift):nrow(x_mat),"BIP"],rep(NA,shift))
  BIP_target_mat<-cbind(BIP_target_mat,BIP_target)
# Regress predictors on shifted BIP  
  for (j in 1:length(h_vec))# j<-5
  {
    lm_obj<-lm(BIP_target~indicator_mat[,j])
    summary(lm_obj)
# This one replicates std in summary
    sd<-sqrt(diag(vcov(lm_obj)))
# Here we use HAC  
    sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
# This is the same as
    sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
# Compute HAC-adjusted t-statistic    
    t_HAC_mat_BIP[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
# One-sided test: if predictor is effective, then the sign of the coefficient must be positive (we are not interested in negative signs) 
    p_value_HAC_mat_BIP[i,j]<-pt(t_HAC_mat_BIP[i,j], len-length(select_vec_multi), lower=FALSE)
    
  }
}
colnames(t_HAC_mat_BIP)<-colnames(p_value_HAC_mat_BIP)<-paste("M-SSA: h=",h_vec,sep="")
rownames(t_HAC_mat_BIP)<-rownames(p_value_HAC_mat_BIP)<-paste("Shift of target: ",h_vec,sep="")

# --------------------------------------------------------------
# HAC-adjusted p-values for raw BIP predictability
# Key findings:
#   - In contrast to HP-BIP, significance with respect to forward-shifted raw BIP
#     is less conclusive: raw BIP is substantially noisier than its HP-filtered counterpart.
#   - Nevertheless, the systematic diagonal pattern observed earlier is still discernible:
#     M-SSA predictors optimized for longer forecast horizons (left to right in the matrix)
#     tend to perform better for correspondingly larger forward-shifts of BIP (top to bottom).
#   - These results are sensitive to the choice of forecast excess:
#     a more aggressive setting (e.g., f_excess = 6) may reinforce the pattern.
p_value_HAC_mat_BIP

# --------------------------------------------------------------
# Technical Note on HAC Standard Errors:
#   - In some cases, the HAC-adjusted variance estimate from the sandwich package
#     can be smaller than the classic OLS variance estimate, which is theoretically
#     inconsistent (HAC-adjusted errors should be at least as large under serial correlation).
#   - This appears to be a numerical issue with the sandwich package in certain settings.
#   - In Tutorial 7.3, this is handled pragmatically:
#     the maximum of the HAC-adjusted and classic OLS variance estimates is used
#     when computing t-statistics and p-values, ensuring conservatism.

# --------------------------------------------------------------
# Visual comparison: forward-shifted BIP vs. M-SSA predictor
# Select the index into h_vec corresponding to the desired forecast horizon
k <- 4
# Display the corresponding forecast horizon (in quarters)
h_vec[k]

par(mfrow = c(1, 1))

# Standardize the forward-shifted BIP target and the selected M-SSA predictor
# for direct visual comparison on a common scale
mplot <- scale(cbind(BIP_target_mat[, k], indicator_mat[, k]))
colnames(mplot) <- c(paste("BIP left-shifted by ", h_vec[k], " quarters", sep = ""),
                     "M-SSA predictor")
colo <- c("black", "blue")
main_title <- "Standardized forward-shifted BIP vs. M-SSA predictor"

# Base plot: forward-shifted BIP target in black
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     lwd = c(2, rep(1, ncol(data) - 1)),
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay the M-SSA predictor in blue
for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

abline(h = 0) # Reference line at zero
# Vertical dashed line marking the end of the in-sample estimation window
abline(v = which(rownames(mplot) == rownames(data_fit)[nrow(data_fit)]),
       lwd = 2, lty = 2)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# Sample correlation between forward-shifted BIP and M-SSA predictor
cor(na.exclude(mplot))


# ==================================================================
# Exercise 4: Full-Length HP Filter as Alternative Target
# ==================================================================
# Background:
#   - The truncated two-sided HP filter used in
#     previous exercises cannot be computed at the sample boundaries.
#   - The full-length HP filter uses all available observations and does not
#     produce NAs, but becomes increasingly asymmetric (one-sided) near the
#     sample boundaries, degrading filter quality at the ends.
#   - Here we recompute HAC-adjusted t-statistics using the full-length HP filter
#     as the target, to verify robustness of the predictability findings.

# --------------------------------------------------------------
# 4.1 Compute the full-length HP trend filter

len <- nrow(x_mat)

# Obtain the HP filter matrix via mFilter; note: hpfilter() returns the GAP filter
hp_obj <- hpfilter(rnorm(len), type = "lambda", freq = lambda_HP)

# Recover the HP trend filter matrix by subtracting the gap filter from the identity
fmatrix <- diag(rep(1, len)) - hp_obj$fmatrix

# Diagnostic plot: compare filter weights at the start (one-sided/asymmetric),
# the middle (approximately two-sided/symmetric), and the end (one-sided/asymmetric)
ts.plot(fmatrix[, c(1, len / 2, len)],
        col = c("red", "black", "green"),
        main = "HP filter weights: one-sided at start (red), two-sided in middle (black), one-sided at end (green)")

# Apply the full-length HP trend filter to BIP (first column of x_mat)
# Result: smooth trend estimate over the full sample without boundary NAs
target_without_publication_lag <- t(fmatrix) %*% x_mat[, 1]

# Shift the HP trend forward by the publication lag (2 quarters)
# to align the target with what is observable at forecast time
target <- c(target_without_publication_lag[(1 + lag_vec[1]):length(target_without_publication_lag)],
            rep(NA, lag_vec[1]))

# --------------------------------------------------------------
# Plot: full-length HP trend vs. all M-SSA predictors (standardized)
# Note: filter becomes increasingly asymmetric near sample boundaries;
#       interpret the trend estimate at the sample end with caution.
mplot <- scale(cbind(target, indicator_mat))
rownames(mplot) <- rownames(x_mat)
colnames(mplot) <- c("Full-length HP", colnames(indicator_mat))
colo <- c("black", rainbow(ncol(indicator_mat)))
main_title <- "Standardized full-length HP trend vs. M-SSA predictors"

# Base plot: full-length HP trend in black
plot(mplot[, 1], main = main_title, axes = F, type = "l",
     xlab = "", ylab = "", col = colo[1],
     lwd = c(2, rep(1, ncol(data) - 1)),
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay all M-SSA predictors
for (i in 1:ncol(mplot))
{
  lines(mplot[, i], col = colo[i], lwd = 1, lty = 1)
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
axis(1, at = c(1, 12 * 1:(nrow(mplot) / 12)),
     labels = rownames(mplot)[c(1, 12 * 1:(nrow(mplot) / 12))])
axis(2)
box()

# Key observations from the plot:
#   - The full-length HP currently indicates a sharp ongoing decline towards the sample end.
#   - In contrast, the M-SSA predictors suggest a possible recovery over 2025/2026.
#   - The two approaches therefore provide conflicting signals about the near-term outlook.

# Compute contemporaneous correlations between full-length HP target and M-SSA predictors
cor(na.exclude(mplot))
# Note: the full-length HP target here is a nowcast (shift = 0, publication lag already applied).
#       Therefore, correlation is maximized at forecast horizon h = 0 (first column or first row of the matrix).


# --------------------------------------------------------------
# 4.2 Forward-shifted targets based on the full-length HP trend
# Construct a matrix of forward-shifted HP trend targets,
# one column per forecast horizon in h_vec (up to 6 quarters ahead)
target_shifted_mat <- NULL
for (i in 1:length(h_vec))
{
  # Shift the publication-lag-adjusted HP trend forward by h_vec[i] additional quarters
  shift <- h_vec[i]
  target_shifted_mat <- cbind(target_shifted_mat,
                              c(target[(1 + shift):length(target)], rep(NA, shift)))
}

# --------------------------------------------------------------
# Recompute correlations and HAC-adjusted p-values using the full-length HP target
# For each (i, j) pair: regress the i-th forward-shifted HP trend on the j-th M-SSA predictor
cor_mat <- p_value_HAC_mat <- matrix(ncol = length(h_vec), nrow = length(h_vec))

for (i in 1:length(h_vec)) # i <- 1
{
  for (j in 1:length(h_vec)) # j <- 1
  {
    # Sample correlation between i-th shifted HP target and j-th M-SSA predictor
    cor_mat[i, j] <- cor(na.exclude(cbind(target_shifted_mat[, i], indicator_mat[, j])))[1, 2]
    
    # Regression of shifted HP target on M-SSA predictor
    lm_obj <- lm(target_shifted_mat[, i] ~ indicator_mat[, j])
    summary(lm_obj)
    
    # Classic OLS standard errors (for reference; replicates summary() output)
    sd <- sqrt(diag(vcov(lm_obj)))
    
    # HAC-robust standard errors (Newey-West type) via the sandwich package
    sd_HAC <- sqrt(diag(vcovHAC(lm_obj)))
    
    # Alternative computation of identical HAC-robust standard errors
    sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
    
    # HAC-adjusted t-statistic for the slope coefficient
    t_HAC_mat <- summary(lm_obj)$coef[2, 1] / sd_HAC[2]
      
      # One-sided HAC-adjusted p-value:
      #   We test for a *positive* slope, since effective predictors should
      #   co-move positively with future HP-filtered BIP trend growth
      p_value_HAC_mat[i, j] <- pt(t_HAC_mat, len - length(select_vec_multi), lower = FALSE)
  }
}

colnames(cor_mat) <- colnames(p_value_HAC_mat) <- paste("M-SSA: h=", h_vec, sep = "")
rownames(cor_mat) <- rownames(p_value_HAC_mat) <- paste("Shift of target: ", h_vec, sep = "")

# Inspect correlation and significance matrices
cor_mat
p_value_HAC_mat

# Key findings from the full-length HP analysis:
#   - The systematic diagonal pattern (larger h performs better at larger forward-shifts)
#     is confirmed and is, if anything, stronger than with the truncated HP filter.
#   - Correlations tend to be larger and p-values smaller compared to the truncated HP results,
#     suggesting the full-length HP provides a cleaner target signal.


# ==================================================================
# Wrapper Function
# ==================================================================
# The above workflow has been encapsulated in compute_mssa_BIP_predictors_func,
# which will be used in Tutorial 7.3.
head(compute_mssa_BIP_predictors_func)

# Required input arguments:
#   x_mat           : multivariate data matrix (series in columns, time in rows)
#   lambda_HP       : HP smoothing parameter (controls trend adaptivity)
#   L               : filter length for the band-pass/HP approximation
#   date_to_fit     : end date of the in-sample estimation span for the VAR
#   p, q            : VAR lag order (p) and MA order (q)
#   ht_mssa_vec     : hard-thresholding (HT) constraints controlling zero-crossing frequency
#                     (larger values enforce smoother, less oscillatory indicators)
#   h_vec           : vector of forecast horizons (in quarters) for M-SSA optimization
#   f_excess        : forecast excess controlling additional anticipation beyond h_vec
#   lag_vec         : publication lag vector (target shifted forward by horizon + publication lag)
#   select_vec_multi: names of the selected indicator series to include in the VAR


# ==================================================================
# Summary and Main Findings
# ==================================================================
#
# 1. Choice of HP smoothing parameter (lambda_HP):
#    - For forecast horizons of one year or less, sufficient adaptivity is required.
#    - lambda_HP = 160 was selected to allow the HP trend to respond adequately
#      over short time intervals.
#    - Increased adaptivity forces M-SSA predictors to anticipate the target by a
#      left-shift commensurate with the forecast horizon.
#    - Tutorial 7.3 explores even more adaptive (shorter-horizon) designs.
#
# 2. M-SSA design principles and predictive relevance:
#    - Timeliness: the left-shift (anticipation) is directly controlled by the
#      specified forecast horizon parameter.
#    - Smoothness: noise suppression (zero-crossing frequency) is effectively
#      controlled by the hard-thresholding (HT) constraint.
#    - The pertinence of both properties is statistically verified by the diagonal pattern in cor_mat
#      and the corresponding HAC-adjusted p-values.
#
# 3. Handling model misspecification:
#    - Forecast excess (f_excess) allows M-SSA to compensate for VAR misspecification
#      by imposing additional anticipation beyond the nominal forecast horizon.
#
# 4. HP-BIP vs. raw BIP as forecast target:
#    - Predicting HP-filtered BIP (trend component) is substantially easier than
#      predicting raw BIP, because HP-BIP is largely free of erratic high-frequency noise.
#    - The link between forecast horizon and physical forward-shift is statistically
#      and logically consistent in both cases, but cleaner for HP-BIP.
#
# 5. Consistency of forecast horizon hyperparameter:
#    - Increasing the forecast horizon improves predictive performance at larger
#      forward-shifts, and the relationship is commensurate (i.e., not arbitrary).
#
# 6. Predictability of raw BIP:
#    - Results for raw BIP are less conclusive due to unpredictable high-frequency noise.
#    - The systematic diagonal structure remains visible but weaker.
#    - More aggressive forecast excess settings may reinforce these findings,
#      though with diminishing returns beyond a certain point.
#
# 7. Practical implication:
#    - A predictor that tracks the low-frequency component of future HP-BIP is also
#      potentially informative about future raw BIP, provided raw BIP is not white noise.
#      Empirical evidence (e.g., the existence of recessions) supports this assumption.
# ==================================================================










