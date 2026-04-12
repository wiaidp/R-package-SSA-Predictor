# =============================================================================
# Tutorial 7: Multivariate Smooth Sign Accuracy (M-SSA)
# =============================================================================
# Overview:
#   This tutorial series extends the univariate Smooth Sign Accuracy (SSA)
#   framework to a multivariate setting, referred to as M-SSA. The topic is
#   divided into the following subtutorials:
#
#   7.0 - Introduce M-SSA and verify that it replicates SSA in a univariate
#         framework (current file)
#
#   7.1 - Application to a 5-dimensional VAR example:
#           * Illustration of the M-SSA optimization principle
#           * Verification of out-of-sample performances via simulation study
#
#   7.2 - 7.5 - Real-data applications: forecasting German GDP multiple
#               quarters ahead using M-SSA
# =============================================================================


# =============================================================================
# Tutorial 7.0: Introduction to M-SSA
# =============================================================================
# Purpose:
#   - Introduce MSSA_func(), the multivariate extension of SSA_func()
#   - Demonstrate that M-SSA exactly replicates SSA in the univariate case,
#     serving as a correctness check for the multivariate implementation
#   - The HP filter customization setup follows Tutorial 2.1 as a reference
#     example
# =============================================================================

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


# -----------------------------------------------------------------------------
# Environment Setup: clear workspace, load required packages and source files
# -----------------------------------------------------------------------------
rm(list = ls())

# Time series data handling
library(xts)

# HP and BK filter implementations
library(mFilter)

# Visualization: ggplot2 for general plots, gplots for heatmaps
library(ggplot2)
library(gplots)

# Load core SSA utility: simple sign accuracy measure
source(paste(getwd(), "/R/ssa.r", sep = ""))

# Load tau-statistic: quantifies time-shift performance (lead/lag assessment)
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load HP signal extraction functions used in the JBCY paper (requires mFilter)
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))

# Load M-SSA functions (multivariate extension of SSA)
source(paste(getwd(), "/R/MSSA.r", sep = ""))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# There are potential problems when loading SSA together with MSSA.
# Some function names are the same but the underlying functions are different.
# Advice: M-SSA generalizes SSA, therefore there is no need to load the SSA 
# functions in addition to M-SSA. All relevant function for M-SSA are packed 
# in functions_MSSA.r. DO NOT SOURCE simple_sign_accuracy.r when working with 
# M-SSA
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



# =============================================================================
# Example 1: Replicating SSA via M-SSA in a Univariate Framework
# =============================================================================
# As a sanity check, we verify that M-SSA reduces to standard SSA when applied
# to a single time series. This confirms the correctness of the multivariate
# implementation before extending it to higher dimensions.
#
# Setup:
#   - Follows Example 1 from Tutorial 2.1
#   - Target filter: two-sided (symmetric) HP filter
#   - SSA customizes the concurrent HP filter by enforcing a larger holding
#     time (HT), which increases smoothness by reducing zero-crossings
#   - See Tutorial 2.1 for full background on the SSA-HP customization
#
# Goal:
#   - Show that M-SSA produces filter coefficients identical to those of SSA
#     in this univariate setting

# =============================================================================

# -----------------------------------------------------------------------------
# Step 1a: Derive the HP Filter Target
# -----------------------------------------------------------------------------

# Filter length: must be odd so that the two-sided HP filter can be centered
L <- 201

# Safety check: enforce odd filter length
if (L / 2 == as.integer(L / 2)) {
  print("Warning: Filter length should be an odd number.")
  print("An even L prevents adequate centering of the two-sided HP filter.")
  L <- L + 1
}

# Lambda parameter for a monthly frequency design
lambda_monthly <- 14400

# Compute the HP filter target and concurrent filter objects
par(mfrow = c(1, 1))
HP_obj <- HP_target_mse_modified_gap(L, lambda_monthly)

# Bi-infinite (symmetric) two-sided HP filter coefficients
hp_target <- HP_obj$target
ts.plot(hp_target, main = "Symmetric Two-Sided HP Filter")

# Concurrent HP gap filter (applied to series in levels; acts as a high-pass filter)
hp_gap <- HP_obj$hp_gap
ts.plot(hp_gap, main = "Concurrent HP Gap Filter")

# Classic one-sided (concurrent) HP trend filter, assuming an I(2) process
# See McElroy (2006) for theoretical background
hp_trend <- HP_obj$hp_trend
ts.plot(hp_trend, main = "Concurrent HP Trend Filter (I(2) assumption)")

# -----------------------------------------------------------------------------
# Compute lag-one autocorrelation (rho) and holding time (ht) for hp_trend
# Holding time measures the average duration between zero-crossings when the filter
#   is applied to white noise
# -----------------------------------------------------------------------------
htrho_obj <- compute_holding_time_func(hp_trend)
rho_hp    <- htrho_obj$rho_ff1   # Lag-one autocorrelation
ht_hp     <- htrho_obj$ht        # Holding time
ht_hp                            # When applied to white noise, the filter output will cross the zero line all 7.65 time points in the mean

# Compare holding times of the one-sided vs. two-sided HP filters
# Note: the large discrepancy between these is discussed in the JBCY paper
compute_holding_time_func(hp_target)$ht   # Two-sided HP
ht_hp                                     # One-sided (concurrent) HP

# -----------------------------------------------------------------------------
# Step 1.1: MSE-Optimal Concurrent HP Filter (White Noise Assumption)
# -----------------------------------------------------------------------------
# This filter is the right-truncated tail of the symmetric HP filter.
# It is MSE-optimal when the underlying data-generating process is white noise.
# -----------------------------------------------------------------------------

hp_mse <- hp_mse_example7 <- HP_obj$hp_mse
par(mfrow = c(1, 1))
ts.plot(hp_mse, main = "MSE-Optimal Concurrent HP Filter")

# Compute lag-one autocorrelation and holding time for the MSE filter
htrho_obj <- compute_holding_time_func(hp_mse)
rho_hp    <- htrho_obj$rho_ff1
ht_mse    <- htrho_obj$ht

# The white noise MSE filter is smoother than the classic concurrent HP (i.e., larger ht),
# because white noise is "noisier" than an ARIMA(0,2,2) process (implicit model of HP).
# Consequently, hp_mse attenuates high-frequency components more aggressively
# than hp_trend.
ht_mse

# -----------------------------------------------------------------------------
# Step 1.2: Configure the SSA Filter
# -----------------------------------------------------------------------------
# We target a holding time of SSA larger than that of hp_mse to reduce zero-crossings.
# A 50% increase in holding time implies ~33% fewer zero-crossings by SSA.
# -----------------------------------------------------------------------------

ht_mse                              # Reference holding time from MSE filter
ht    <- 1.5 * ht_mse              # Target holding time for SSA (50% increase)
rho1  <- compute_rho_from_ht(ht)   # Convert holding time to lag-one autocorrelation
ht / ht_mse                        # Confirm the relative increase in holding time

# Forecast horizon: nowcast setting (no look-ahead)
forecast_horizon <- 0

# Assume white noise data-generating process (default; no ARMA residual structure)
xi <- NULL

# Target filter: supply the MSE concurrent HP filter, consistent with white noise assumption
# (Alternative: supply hp_trend if assuming an ARIMA(0,2,2) process; see Example 2)
gammak_generic <- hp_mse

# Compute the univariate SSA filter targeting the HP filter
SSA_obj_HP <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# -----------------------------------------------------------------------------
# Step 1.3: Configure and Run M-SSA
# -----------------------------------------------------------------------------
# Use default numerical optimization settings by passing NULL for control parameters.
# In the univariate case, M-SSA should produce results identical to SSA.
# -----------------------------------------------------------------------------

# Numerical optimization controls (NULL = use defaults)
split_grid           <- NULL
grid_size            <- NULL
with_negative_lambda <- NULL

# Run M-SSA
MSSA_obj <- MSSA_func(
  split_grid,
  L,
  forecast_horizon,
  grid_size,
  gammak_generic,
  rho1,
  with_negative_lambda,
  xi
)

# -----------------------------------------------------------------------------
# Plot Comparison: SSA vs. M-SSA Filter Coefficients
# -----------------------------------------------------------------------------
# The two filters should overlap exactly, confirming that M-SSA replicates
# SSA in the univariate case.
# -----------------------------------------------------------------------------
par(mfrow = c(1, 1))
ts.plot(
  cbind(MSSA_obj$bk_mat, SSA_obj_HP$ssa_x),
  col  = c("blue", "black"),
  main = "M-SSA vs. SSA Filter Coefficients (Univariate Case): Filters Overlap Exactly"
)


# -----------------------------------------------------------------------------
# Summary and Outlook
# -----------------------------------------------------------------------------
# The example above confirms that M-SSA exactly replicates SSA in a univariate
# framework, validating the correctness of the multivariate implementation.
#
# The following tutorials extend M-SSA to genuinely multivariate settings.
# Three key themes are explored:
#
#   1. Optimization Principle:
#        Illustrate how M-SSA jointly optimizes filter coefficients across
#        multiple series, and how this differs from applying SSA independently
#        to each series
#
#   2. Convergence of Sample Performances:
#        Demonstrate via simulation that sample-based performance metrics
#        (MSE, sign accuracy, holding time HT) converge to their theoretical
#        counterparts as the sample size increases
#
#   3. Pertinence of the Multivariate Extension:
#        Show that leveraging cross-series information in M-SSA yields
#        measurable improvements over the univariate SSA benchmark,
#        justifying the added complexity of the multivariate framework
# -----------------------------------------------------------------------------






















