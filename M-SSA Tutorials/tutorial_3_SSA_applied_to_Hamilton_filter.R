
# =============================================================================
# Tutorial 3: SSA Applied to Hamilton's Regression Filter (HF)
# =============================================================================
# # James D. Hamilton, 2017. "Why You Should Never Use the Hodrick-Prescott Filter," 
# NBER Working Papers 23429, National Bureau of Economic Research, Inc.

# Overview:
#   This tutorial demonstrates how SSA customization can be applied to
#   Hamilton's regression filter (HF) to improve its real-time business cycle
#   analysis (BCA) properties along two key dimensions: smoothness and timeliness.
#
# Datasets:
#   - Example 1:        Quarterly US GDP
#   - Example 2:        Monthly US non-farm payroll employment (PAYEMS)
#
# Sample coverage:
#   Both long samples (post-WWII for GDP) and shorter samples (post-Great 
#   Moderation, ~1990 onwards for PAYEMS) are analysed. 
#
# Recommendation: to mitigate non-stationarity avoid working on unnecessarily 
# long historical samples when fitting parameters to data.

# =============================================================================
# Summary of Main Findings
# =============================================================================
#
# 1. HF AS A LOW-PASS FILTER:
#      When expressed in terms of FIRST DIFFERENCES (see Exercises 1.4 and 1.9),
#      HF behaves as a low-pass filter. This property confers two important
#      advantages for business cycle analysis (BCA):
#
#        (a) Mitigation of spurious cycles:
#              Low-pass behaviour (in first differences) suppresses the 
#              artificial oscillations that band-pass filters tend to impose 
#              on the data, regardless of whether such oscillations are 
#              present in the underlying series.
#
#        (b) Advantage over classic band-pass designs (in first differences):
#              Filters such as Baxter-King (BK), Christiano-Fitzgerald (CF),
#              and the HP-gap are all band-pass in nature and therefore prone
#              to spurious cycle generation. HF avoids this pathology by design.
#              See Tutorials 2.0 (Example 7), 4, and 5 for detailed evidence.
#
#
# 2. HF ACCOMMODATES ARBITRARY INTEGRATION ORDERS:
#      By increasing the AR lag order p in the regression specification, HF can
#      accommodate series integrated to an arbitrary order. This makes it
#      well-suited for macroeconomic indicators whose growth rates evolve slowly
#      over long historical samples (e.g, post-WWII data).
#
#
# 3. HF IS SAMPLE-DEPENDENT:
#      HF is defined by regression coefficients estimated on a specific sample.
#      As a result, the extracted cycle is inherently sample-specific:
#        - Estimating the regression on the full post-WWII sample versus the
#          post-1990 sub-sample yields substantially different filter coefficients
#          and, consequently, different cycle definitions.
#        - Users should be aware that HF comparisons across studies may reflect
#          differences in estimation samples as much as differences in the data.
#
#
# 4. HF PRODUCES NON-STATIONARY CYCLES OUT-OF-SAMPLE:
#      (Note: this finding does not contradict Finding 2 above.)
#
#      HF is estimated as an unrestricted OLS regression, without imposing
#      any constraint on the unit-root structure of the data. As a consequence:
#        - IN-SAMPLE:  the HF cycle is stationary by construction (OLS residuals
#                      are centred and well-behaved within the estimation window).
#        - OUT-OF-SAMPLE: the HF filter weights do not sum to zero, so the filter
#                      fails to cancel the unit root(s) in the data. The extracted
#                      cycle therefore becomes non-stationary beyond the estimation
#                      window.
#
#      Two remedies are discussed in this tutorial:
#        (i)  Regular re-estimation: updating the regression with new observations
#             restores in-sample stationarity, at the cost of real-time revisions
#             to historical cycle estimates.
#        (ii) Unit-root-constrained regression: imposing a zero-sum constraint on
#             the filter coefficients ensures unit-root cancellation out-of-sample
#             for a single (or more) integrated root. See Exercises 1.2 and 2.3.
#
#
# 5. HF IS NOT ON THE EFFICIENT ATS FRONTIER:
#      The Accuracy-Timeliness-Smoothness (ATS) frontier characterises the set
#      of filters that cannot be improved in one dimension without sacrificing
#      another. HF does not lie on this frontier:
#        - SSA can strictly improve the smoothness (noise suppression) of HF
#          without materially sacrificing accuracy or timeliness.
#          See Examples 2.9 and 2.10 for empirical confirmation.
#        - An analogous result holds for HP-based filters; see Tutorial 2.1.
#        - This inefficiency provides a clear and well-motivated rationale for
#          applying SSA customization to HF.
#
#
# 6. SSA CUSTOMIZATION STRATEGIES:
#      Two complementary SSA customization approaches are demonstrated,
#      targeting different points on the ATS frontier:
#
#        (a) Smoothness improvement only:
#              Target: reduce high-frequency noise while maximizing tracking 
#                      accuracy.
#              Method: impose a holding time 50% larger than HF's empirical
#                      holding time as a constraint within the SSA optimisation.
#              Reference: Exercises 1.6 and 2.7.
#
#        (b) Simultaneous smoothness and timeliness improvement:
#              Target: reduce noise AND advance turning-point detection.
#              Method: impose the holding-time constraint while
#                      simultaneously extending the SSA forecast horizon,
#                      shifting the filter output ahead of real time.
#              Reference: Exercises 1.8 and 2.9.
#
#
# 7. PANDEMIC ROBUSTNESS:
#      The COVID-19 pandemic generated extreme outliers in economic aggregates
#       — observations so large that they approximate impulse inputs to
#      the filters. This provides a natural experiment for studying filter
#      dynamics empirically, without simulation:
#        - The post-pandemic "phantom peak" in each filter's output directly
#          reflects the shape and decay rate of its impulse response function.
#        - Filters with faster-decaying impulse responses recover more quickly
#          from the shock, producing shorter-lived distortions.
#        - SSA forecast filters are shown to recover materially faster than HF,
#          an advantage that is especially relevant in the presence of singular
#          macro events.
#      See Exercise 2.11 for the full analysis.
# =============================================================================

# -----------------------------------------------------------------------------
# Tutorial Structure
# -----------------------------------------------------------------------------
#
# Example 1:  SSA applied to the Hamilton filter using quarterly US GDP,
#             estimated over the full post-WWII sample.
#
# Example 2:  SSA applied to the Hamilton filter using monthly PAYEMS
#             employment data, estimated over the post-1990 sub-sample.
# -----------------------------------------------------------------------------
# Broader Motivation
# -----------------------------------------------------------------------------
# The goal of this tutorial is not to advocate for or against a particular 
# business cycle tool or filter design. Rather, it illustrates a general and 
# broadly applicable principle:
#
#   Any causal linear filter — including HF — can be replicated and
#   systematically improved by SSA with respect to two practically
#   important BCA priorities:
#     1. Smoothness:  suppression of high-frequency noise, reducing false signals.
#     2. Timeliness:  earlier detection of cyclical turning points.
#
# HF serves here as a convenient, well-known baseline platform and a concrete
# showcase for the SSA optimisation principle. Throughout the tutorial, a range
# of quantitative performance measures — including holding times, amplitude
# functions, phase-shift functions, and empirical zero-crossing counts — is
# provided to confirm the practical value of the SSA approach.
# =============================================================================



# ── BACKGROUND ────────────────────────────────────────────────────
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

#   Wildi, M. (2026a) Sign Accuracy, Mean-Squared Error and the Rate
#     of Zero Crossings: a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547


# ─────────────────────────────────────────────────────────────────

# Clear the workspace and load required packages and functions.
rm(list = ls())

library(xts)

# The 'neverhpfilter' package implements the Hamilton filter ("never use the
# HP-filter"). It is used here for background information and plotting only.
library(neverhpfilter)

# Load data from FRED using the quantmod library.
library(quantmod)

# Load data from FRED using the alfred library (no API key required).
install.packages("alfred")
library(alfred)

# Load all relevant SSA functions.
source(paste(getwd(), "/R/ssa.r", sep = ""))

# Load the tau-statistic: quantifies time-shift performance (lead/lag).
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load signal extraction functions used for the JBCY paper (requires mFilter).
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))

# Load the Hamilton regression function with a unit-root constraint.
source(paste(getwd(), "/R utility functions/hamilton_unit_root.r", sep = ""))


# =============================================================================
# Example 1: Quarterly GDP
# =============================================================================
# Overview:
#   This example applies SSA customization to the Hamilton filter (HF) using
#   quarterly U.S. real GDP (GDPC1). We proceed in two stages:
#     (1) Demonstrate HF using the neverhpfilter package to establish a
#         reference decomposition and verify our implementation.
#     (2) Re-implement HF from scratch to enable SSA targeting and
#         customization beyond the original specification.
# ──────────────────────────────────────────────────────────────────────────
# A Note on Sample Length
# ──────────────────────────────────────────────────────────────────────────
#
# The analysis uses the full post-WWII sample, which is deliberately long.
# Two points are worth noting in this regard:
#
#   1. SSA IS SAMPLE-AGNOSTIC IN DESIGN:
#      SSA can customise any target filter regardless of the sample length chosen
#      for the underlying regression.
#
#   2. PRACTICAL CAVEAT FOR THE HAMILTON REGRESSION:
#      In applied work, fitting the Hamilton filter OLS regression over such
#      a long and structurally heterogeneous sample is generally not
#      recommended, as it may mask parameter instability and structural
#      breaks. A shorter, more homogeneous estimation window would typically
#      be preferred.
#
#   3. REPLICATION CONVENTION:
#      Notwithstanding the caveat above, we follow the original
#      implementation in the 'neverhpfilter' R-package, which uses the full
#      available post-WWII sample, in order to ensure reproducibility of
#      the baseline Hamilton cycle estimates.

# -----------------------------------------------------------------------------
# Stage 1: Hamilton Filter via the neverhpfilter Package
# -----------------------------------------------------------------------------
data(GDPC1)

# Decompose log-GDP (scaled by 100) into trend and cycle components.
# Settings follow Hamilton's (2018) recommended quarterly specification:
#   h = 8 quarters (2-year forecast horizon)
#   p = 4 autoregressive lags
gdp_trend <- yth_filter(
  100 * log(GDPC1), h = 8, p = 4, output = c("x", "trend")
)

# Plot log-GDP alongside the estimated Hamilton trend:
plot.xts(
  gdp_trend,
  grid.col   = "white",
  legend.loc = "topleft",
  main       = "Log GDP and Hamilton Trend (Quarterly)"
)

# Extract the cyclical and irregular (random) components:
gdp_cycle <- yth_filter(
  100 * log(GDPC1), h = 8, p = 4, output = c("cycle", "random")
)

# Plot cycle and irregular components:
plot.xts(
  gdp_cycle,
  grid.col   = "white",
  legend.loc = "topleft",
  main       = "Hamilton Cycle and Irregular Component (Quarterly)"
)

# Observations:
#   - The cycle is centred around zero by construction.
#   - HF implicitly removes up to p unit roots, where p is the AR order
#     used in the regression.
#   - In practice, the cycle and irregular components are often strongly
#     correlated, contradicting the standard BCA assumption that these
#     two components are orthogonal (independent).


# -----------------------------------------------------------------------------
# Stage 2: Custom Re-implementation of the Hamilton Filter
# -----------------------------------------------------------------------------
# We abandon neverhpfilter and reconstruct HF from scratch. This is necessary
# to:
#   (1) Define an explicit SSA target based on the Hamilton filter coefficients.
#   (2) Modify filter characteristics (customization) beyond the constraints of
#       the original package implementation.

# --- Load Quarterly Real GDP from FRED ---
# Toggle reload_data = TRUE to download a fresh series from FRED via alfred;
# no API key is required for basic use.
# Toggle reload_data = FALSE to use the locally cached version.
reload_data <- FALSE

if (reload_data) {
  GDPC1 <- get_fred_series("GDPC1", series_name = "GDP")
  GDPC1 <- as.xts(GDPC1)
  save(GDPC1, file = file.path(getwd(), "Data", "GDPC1"))
} else {
  load(file = file.path(getwd(), "Data", "GDPC1"))
}
head(GDPC1)
tail(GDPC1)

# --- Convert to Numeric Vector and Restrict Sample ---
# We convert the xts object to a plain numeric vector for two reasons:
#   (1) xts objects carry hidden metadata and index conventions that make
#       direct application of linear filters error-prone and unpredictable.
#   (2) The sample is truncated at end-2019 to exclude the COVID-19 pandemic.
#       Pandemic-era outliers (early 2020) would severely distort the Hamilton
#       regression coefficients if included in the estimation window.
#       The pandemic's impact on HF and SSA is analysed separately in the
#       final example of this tutorial.
y   <- as.double(log(GDPC1["/2019"]))
len <- length(y)


# =============================================================================
# 1.1 Replication of the Hamilton Filter
# =============================================================================
# Hamilton's recommended settings for quarterly macroeconomic data:
h <- 2 * 4   # Forecast horizon: 8 quarters (2 years ahead)
p <- 4       # Number of autoregressive lags in the regression

# --- Construct the OLS Design Matrix ---
# The Hamilton regression projects y_{t+h} onto its own p most recent lags:
#   y_{t+h} = α + β_1 y_t + β_2 y_{t-1} + ... + β_p y_{t-p+1} + ε_{t+h}
# The first column of 'explanatory' is y_t (contemporaneous relative to
# the forecast origin); subsequent columns contain earlier lags.
explanatory <- y[(p):(len - h)]
for (i in 1:(p - 1))
  explanatory <- cbind(explanatory, y[(p - i):(len - h - i)])

# Define the dependent variable: log-GDP observed h periods ahead:
target <- y[(h + p):len]

# --- Fit the Hamilton Regression ---
# The regression is estimated over the full available sample (post-WWII).
# Note: a shorter sub-sample starting in 1990 is explored in the monthly
# PAYEMS examples below.
lm_obj <- lm(y[(h + p):len] ~ explanatory)
summary(lm_obj)

# Diagnostic observations:
#   (1) Typically, only the first lag coefficient is statistically significant —
#       a common finding when HF is applied to non-stationary macroeconomic
#       series.
#   (2) The sum of the lag coefficients deviates from unity:
#           sum(β_1, ..., β_p) ≠ 1
#       This implies that the filter residual y_{t+h} - ŷ_{t+h} is stationary
#       within the estimation sample but may become non-stationary out-of-sample,
#       since the forecast and the realized future value are not cointegrated
#       beyond the estimation window.

# --- Construct the Hamilton Filter Coefficient Vector ---
# The filter has the following structure:
#   [1, 0, 0, ..., 0, -β_1, -β_2, ..., -β_p]
#    ↑        ↑                ↑
# y_{t+h}  h-1 zeros    negative OLS lag coefficients
#
# The leading 1 selects y_{t+h}; the h-1 zeros cover the intermediate
# periods; the final p entries are the negated OLS coefficients.
hamilton_filter <- c(1, rep(0, h - 1), -lm_obj$coefficients[1 + 1:p])

par(mfrow = c(1, 1))
ts.plot(
  hamilton_filter,xlab="Lag",ylab="",
  main = paste("Hamilton Filter Coefficients: GDP,",
               index(GDPC1["/2019"])[1], "to 2019")
)

# Extract the regression intercept for use in cycle replication.
# Notes on mean-centring:
#   - The cycle is not explicitly mean-centred here.
#   - SSA's target-correlation objective is invariant to level shifts, so
#     the intercept does not affect the optimisation.
#   - The holding-time constraint assumes zero-crossings of the mean level;
#     the intercept shift is therefore inconsequential for HT calculation.
intercept <- lm_obj$coefficients[1]
  
  
# --- Replicate the Hamilton Filter Output ---
# We construct a data matrix to apply the filter directly and verify that
# it reproduces the OLS residuals exactly.
#
# Structure of data_mat:
#   - Columns 1 to h:     filled by repeating 'target' h times. Since the
#                         Hamilton filter coefficients are zero for lags 1
#                         to h-1, these columns do not affect the output;
#                         they serve as placeholders for indexing convenience.
#   - Columns h+1 to h+p: the p lagged regressors from 'explanatory'.
data_mat  <- cbind(matrix(rep(target, h), ncol = h), explanatory)

# Apply the filter and subtract the intercept to recover the cycle:
residuals <- data_mat %*% hamilton_filter - intercept

# --- Verification ---
# The computed residuals must be numerically identical to lm_obj$residuals.
# Both series should overlap perfectly in the plot below.
ts.plot(
  cbind(residuals, lm_obj$residuals),
  main = "Verification: Hamilton Filter vs. OLS Residuals (series should overlap)"
)


# =============================================================================
# 1.2 Unit-Root Adjustment
# =============================================================================

# Diagnostic:  A sum of the OLS lag coefficients materially
# different from 1 signals residual non-stationarity risk out-of-sample.
sum(lm_obj$coefficients[1 + 1:p])

# A zero sum is the algebraic condition for cancelling a single unit root
# (I(1) process), i.e. for the filter to act as a cointegrating combination
# when applied out-of-sample.
sum(hamilton_filter)

# If the sum deviates from zero, applying the unrestricted filter
# out-of-sample produces a non-stationary cycle. To prevent this when
# embedding the Hamilton filter within the SSA framework, we impose a
# unit-root restriction. The resulting adjustment is numerically small and
# has negligible impact on the cycle's dynamic properties.

# --- Unit-Root-Restricted Hamilton Filter ---

# Re-estimate the Hamilton filter under the unit-root restriction (Diff = TRUE
# forces the slope coefficients to sum to exactly 1, which in turn guarantees
# a zero-sum filter).
Diff    <- TRUE
Ham_obj <- HamiltonFilter_Restricted(y, p, h, Diff)

# Confirm that the restricted slope coefficients sum to exactly 1.
sum(Ham_obj$coefficients[-1])

# Build the adjusted Hamilton filter vector:
#   - coefficient 1 corresponds to the current observation (lead h),
#   - h-1 zeros span the forecast horizon gap,
#   - the remaining entries are the negated restricted slope coefficients.
hamilton_filter_adjusted <- c(1, rep(0, h - 1), -Ham_obj$coefficients[-1])

# Confirm that the adjusted filter coefficients sum to exactly zero,
# verifying that the unit-root cancellation condition is satisfied.
sum(hamilton_filter_adjusted)

# --- Visual Comparison: Original vs. Adjusted Hamilton Filter ---

# Overlay the two filter coefficient sequences to assess how much the
# restriction alters the filter shape.
par(mfrow = c(1, 1))
ts.plot(
  cbind(hamilton_filter, hamilton_filter_adjusted),
  col  = c("black", "grey"),xlab="Lag",ylab="",
  main = "Original and Adjusted Hamilton Filter Coefficients"
)
mtext("Original", line = -1)
mtext("Adjusted", col = "grey", line = -2)

# ──────────────────────────────────────────────────────────────────────────
# A Note on the Unit-Root Restriction
# ──────────────────────────────────────────────────────────────────────────
#
# The unit-root restriction is not strictly required
# for the SSA customisation to work. It is adopted for two practical reasons:
#
#   1. COMPUTATIONAL SIMPLIFICATION:
#      Enforcing the zero-sum constraint on the Hamilton filter coefficients
#      simplifies the difference-domain transformation carried out in Section 
#      1.4.
#
#   2. NEGLIGIBLE IMPACT ON CYCLE DYNAMICS:
#      As demonstrated in Section 1.3, the restricted and unrestricted
#      Hamilton cycles share virtually identical dynamic properties. The
#      restriction introduces only a minor, slowly evolving level offset.
#      SSA customisation therefore produces comparable results under either
#      specification, and the choice between them does not materially affect
#      any SSA-based conclusions.

# =============================================================================
# 1.3 Plot and Compare Cycles
# =============================================================================

# Apply the adjusted filter to the data matrix to obtain the adjusted cycle.
# No intercept subtraction is required: the zero-sum constraint ensures that
# any constant in the data is automatically eliminated.
residuals_adjusted <- data_mat %*% hamilton_filter_adjusted

# --- Visual Comparison of Three Cycle Variants ---

# Plot 1: Three cycle estimates overlaid on a single panel.
#   Red    – Original Hamilton cycle (raw OLS forecast residuals).
#   Orange – Original cycle shifted upward by the regression intercept.
#   Blue   – Unit-root-adjusted, uncentred cycle.
# The three series share virtually identical dynamics; they differ mainly
# in their level (vertical offset).
par(mfrow = c(1, 1))
ts.plot(
  cbind(residuals, residuals + intercept, residuals_adjusted),
  col  = c("red", "orange", "blue"),
  main = "Cycle Comparison: Original, Intercept-Shifted, and Adjusted"
)
mtext("Unit-root adjusted uncentred cycle",  col = "blue",   line = -1)
mtext("Hamilton cycle",                      col = "red",    line = -2)
mtext("Hamilton cycle shifted by intercept", col = "orange", line = -3)
abline(h = 0)

# Plot 2: Multi-panel comparison of log-GDP alongside the cycle variants.
par(mfrow = c(2, 2))

# Panel 1: Raw log-GDP series.
ts.plot(y, main = "Log(GDPC1)")

# Panel 2: Original (red) and adjusted (blue) cycles overlaid.
ts.plot(
  cbind(residuals, residuals_adjusted),
  col  = c("red", "blue"),
  main = "Original vs. Adjusted Cycle"
)
mtext("Hamilton cycle",                     col = "red",  line = -1)
mtext("Unit-root adjusted uncentred cycle", col = "blue", line = -2)
abline(h = 0)

# Panel 3: Pointwise difference between the two cycle estimates over time.
# A non-zero, slowly evolving difference confirms a level offset rather
# than a difference in dynamics.
ts.plot(residuals - residuals_adjusted, main = "Cycle Difference (Original – Adjusted)")


# ==========================================================================
# MAIN TAKE-AWAY:
#   - The adjusted and original cycles exhibit NEARLY IDENTICAL DYNAMICS;
#     their primary difference is a slowly evolving level offset visible
#     in the difference plot.
#   - SSA CUSTOMIZATION WILL THEREFORE HAVE COMPARABLE EFFECTS ON EITHER
#     VARIANT, SO THE CHOICE BETWEEN THEM DOES NOT MATERIALLY AFFECT
#     SSA-BASED CONCLUSIONS.
#   - For technical reasons (zero-sum filter constraint ensures the
#     cointegration / unit-root cancellation property), we proceed with
#     the adjusted, uncentred cycle as the basis for SSA customization.
# ==========================================================================


# =============================================================================
# 1.4 Transformation: From Levels to Differences
# =============================================================================
# ──────────────────────────────────────────────────────────────────────────
# Motivation: Difference-Domain Reformulation of the Hamilton Filter
# ──────────────────────────────────────────────────────────────────────────
#
# The Hamilton filter transforms the non-stationary log-GDP series into a
# stationary regression residual (the 'cycle') by projecting future values
# onto lagged levels. While this formulation is natural for estimation, it
# is not directly suitable as an SSA target, because SSA customisation
# requires a filter that operates on stationary input (an extension to 
# non-stationary I-SSA is proposed in tutorial 6).
#
# It is possible, however, to derive an equivalent HF filter representation
# that replicates the same cycle when applied to first differences of
# log-GDP (i.e., the stationary GDP growth rate series), see Wildi (2024).
#
# The difference-domain reformulation serves two purposes:
#
#   1. TARGET SPECIFICATION FOR SSA:
#      The reformulated filter ('ham_diff') is supplied directly as the
#      SSA target filter. SSA then searches for the optimal causal
#      approximation to this target in the difference domain, subject to
#      the imposed smoothness and timeliness constraints.
#
#   2. NUMERICAL VERIFICATION:
#      We confirm that both formulations — the original level-domain Hamilton
#      filter and the derived difference-domain filter 'ham_diff' — produce
#      numerically identical cycle estimates when applied to their respective
#      inputs (log-GDP levels vs. first-differenced log-GDP).


# Set the working filter length L. It must be at least as long as the
# adjusted Hamilton filter; increase L here if a longer filter is needed.
L <- 20
L <- max(length(hamilton_filter_adjusted), L)

# Zero-pad the adjusted Hamilton filter to length L so that the convolution
# below produces a coefficient vector of the desired length.
if (L > length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L <- c(
    hamilton_filter_adjusted,
    rep(0, L - length(hamilton_filter_adjusted))
  )

# Derive the difference-domain filter 'ham_diff' by convolving the zero-padded
# level-domain Hamilton filter with the summation (unit-root) filter (1 - B)^{-1},
# where B denotes the backshift operator. This convolution inverts the
# differencing operation embedded in the zero-sum level filter, yielding an
# equivalent representation that accepts first-differenced log-GDP as input
# and produces an identical cycle estimate, see Wildi (2024).
ham_diff <- conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv


# --- Visual Comparison: Level-Domain vs. Difference-Domain Filter ---
par(mfrow = c(2, 1))
ts.plot(ham_diff,xlab="Lag",ylab="",
        main = "Hamilton Filter: Difference-Domain Representation (Input: x_t-x_{t-1})")
ts.plot(hamilton_filter_adjusted_L,xlab="Lag",ylab="",
        main = "Hamilton Filter: Level-Domain Representation (Input: x_t)")

# --- Numerical Equivalence Check ---
# Both filter representations should yield identical cycle estimates when
# each is applied to its appropriate input (differenced vs. level data).

# Compute first differences of log-GDP (stationary input for 'ham_diff').
x        <- diff(y)
len_diff <- length(x)

# Apply 'ham_diff' to the differenced series to recover the cycle estimate.
residual_diff <- na.exclude(filter(x, ham_diff, side = 1))

# Prepend NAs to the level-domain cycle to align sample lengths: the
# difference-domain output is shorter due to the differencing step and
# filter initialization.
original_hamilton_cycle <- c(rep(NA, length(x) - length(residuals)), residuals)

# Visual check: the blue (difference-domain) and red (level-domain) series
# should overlap closely, confirming numerical equivalence.
par(mfrow = c(1, 1))
ts.plot(residual_diff, col = "blue",
        main = "Equivalence Check: Level- vs. Difference-Domain Filter Output")
lines(residuals_adjusted[(L - p - h + 2):length(residuals)], col = "red")

# Having confirmed equivalence, SSA customization of the Hamilton filter
# can proceed entirely in the difference domain using 'ham_diff' as the
# target filter — without loss of generality.
# Note: the same level-to-difference transformation is applied to the BK
# filter (Tutorial 4) and to the HP-gap filter (Tutorials 2.1 and 5).

# =============================================================================
# 1.5 Holding Times and Dependence
# =============================================================================
#
# Before customizing via SSA, we characterize the zero-crossing frequency
# of the Hamilton filter ham_diff through its theoretical holding time — 
# defined as the expected number of periods between successive zero-crossings 
# under white-noise input (note: diff-log GDP is not white noise).

# Compute the theoretical holding time of 'ham_diff'.
ht_ham_diff_obj <- compute_holding_time_func(ham_diff)

# Result: approximately one zero-crossing every 1.25 years under white noise.
ht_ham_diff_obj$ht

# Compute the empirical holding time directly from the adjusted Hamilton cycle.
compute_empirical_ht_func(residuals_adjusted)

# The empirical holding time is substantially longer than the theoretical one.
# This is expected: the adjusted cycle is uncentred (positive drifting GDP),
# which reduces the frequency of zero-crossings and upward-biases the
# empirical estimate.
par(mfrow = c(1, 1))
ts.plot(residuals_adjusted)
abline(h = 0)

# After mean-centering the cycle, the empirical holding time moves closer
# to the theoretical value (~5). The residual discrepancy is due to 
# model misspecification (centred diff-log GDP is not white noise).
compute_empirical_ht_func(scale(residuals_adjusted))


# Inspect the autocorrelation structure of the first-differenced log-GDP
# series (i.e., the GDP growth rate) to inform the SSA input specification.
acf(x, main = "ACF of First-Differenced Log-GDP")

# The ACF reveals serial dependence in the GDP growth rate.
# For simplicity, and following Wildi (2024), we adopt a white-noise
# assumption for the input process:
#   - Under white noise, the SSA filter coefficients are time-invariant,
#     yielding a revision-free design.
#   - Setting xi = NULL signals to downstream functions that no Wold
#     decomposition is required (i.e., x_t = epsilon_t directly).
# Note: the white noise assumption will be dropped in exercise 2 below.
xi <- NULL


# =============================================================================
# 1.6 Apply SSA to the Hamilton Filter
# =============================================================================

# --------------------------------------------------------------------------
# 1.6.1 SSA Settings
# --------------------------------------------------------------------------

# Display the theoretical holding time of the Hamilton filter as a reference
# (~1.25 years under white-noise input).
ht_ham_diff_obj$ht

# Set the SSA holding-time target 50% above the Hamilton filter's theoretical
# value. A longer holding time forces the SSA filter to produce fewer
# zero-crossings, yielding a smoother cycle estimate than the original HF.
# Note: the smoothness gain is robust to model misspecification — even when
# the white-noise assumption is violated and the theoretical and empirical
# holding times diverge, the SSA filter reliably generates fewer zero-crossings
# than the Hamilton benchmark in finite samples (see results below).
ht <- 1.5 * ht_ham_diff_obj$ht

# Convert the holding-time target to the autocorrelation parameter rho1.
# We can supply either ht or rho1 to SSA (feature introduced 04-14-2026)
rho1 <- compute_rho_from_ht(ht)

# Specify the SSA target filter.
# Because we work in the difference domain, 'ham_diff' (the difference-domain
# representation of the adjusted Hamilton filter) is supplied as the target.
gammak_generic <- ham_diff

# Set the forecast horizon to zero (nowcast).
# At h = 0, SSA does not predict the Hamilton cycle; it approximates it
# subject to the holding-time constraint. If the holding-time target were
# set equal to that of HF and h = 0, SSA would replicate HF exactly.
forecast_horizon <- 0

# --------------------------------------------------------------------------
# 1.6.2 Run SSA Optimisation
# --------------------------------------------------------------------------

# Optimise the SSA filter. When xi is NULL the function internally assumes
# white-noise input (x_t = epsilon_t).
# The HT constraint can be supplied either as lag-one ACF (as done here) 
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1)
# Alternatively we can supply ht instead of rho1: both calls generate identical 
# SSA solutions 
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, ht)

# The optimisation returns two filter representations:
#   ssa_x   – filter applied directly to the observed series x_t.
#   ssa_eps – filter applied to the white-noise innovations epsilon_t
#             (used mainly for holding-time verification).
# Under the white-noise assumption x_t=epsilon_t and both representations are 
# numerically identical.
SSA_filt_ham_diff <- SSA_obj_ham_diff$ssa_x

# Sanity check: confirm that ssa_x and ssa_eps are identical.
# The maximum absolute difference should be zero (or negligible machine
# precision) under the white-noise assumption.
max(abs(SSA_obj_ham_diff$ssa_x - SSA_obj_ham_diff$ssa_eps))

# --------------------------------------------------------------------------
# 1.6.3 Plot Filters
# --------------------------------------------------------------------------

# Overlay the Hamilton difference-domain target filter (black) and the
# optimised SSA filter (blue) to visualise the smoothness trade-off.
par(mfrow = c(1, 1))
mplot <- cbind(ham_diff, SSA_filt_ham_diff)
ts.plot(
  mplot,xlab="Lag",ylab="",
  ylim = c(min(mplot), max(mplot)),
  col  = c("black", "blue"),
  main = "Target Filter vs. SSA Filter"
)
mtext(
  "Hamilton filter (difference-domain representation)",
  col  = "black",
  line = -1
)
mtext(
  paste(
    "SSA: holding time increased by ",
    100 * (ht / ht_ham_diff_obj$ht - 1), "%",
    sep = ""
  ),
  col  = "blue",
  line = -2
)
abline(h = 0)

# --------------------------------------------------------------------------
# 1.6.4 Convergence Check
# --------------------------------------------------------------------------

# Verify that the SSA optimisation has converged: the effective holding time
# of the SSA filter (computed from its coefficients) should match the
# imposed target 'ht' up to rounding error.
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff)

# Effective holding time of the optimised SSA filter:
ht_obj$ht

# Imposed holding-time target:
ht

# Close agreement between the two values confirms convergence to the global
# optimum of the SSA criterion. If greater numerical precision is required,
# the approximation can be tightened by increasing the 'split_grid' parameter
# in the call to SSA_func, which refines the internal frequency grid used
# during optimisation. To inspect the full function signature and all
# available parameters, including 'split_grid' and its default value, run:
head(SSA_func)

# =============================================================================
# 1.7 Filter the Series and Evaluate Performance
# =============================================================================

# Apply the SSA filter to the differenced log-GDP series to obtain the
# SSA-based cycle estimate.
SSA_out <- filter(x, SSA_filt_ham_diff, side = 1)

# --- Holding-Time Comparison: Theoretical vs. Empirical ---

# For SSA: compare the imposed theoretical target with the empirical
# zero-crossing frequency. The two mainly differ because the output series
# is uncentred (positive level offset suppresses zero-crossings).
ht
compute_empirical_ht_func(SSA_out)

# For the Hamilton filter: analogous comparison.
ham_out <- filter(x, ham_diff, side = 1)
ht_ham_diff_obj$ht
compute_empirical_ht_func(ham_out)

# Repeat the comparison on mean-centred (scaled) cycles to remove the
# level-offset bias from the empirical estimates.
# Expected result: the centred SSA cycle exhibits ~30% fewer zero-crossings
# than the centred Hamilton cycle, consistent with the imposed 50% HT
# increase. Residual discrepancies are attributable to model misspecification
# (white-noise assumption, structural breaks, and non-stationarity over the
# post-1947 sample).
compute_empirical_ht_func(scale(SSA_out))
compute_empirical_ht_func(scale(ham_out))

# --- Visual Comparison: Standardized SSA vs. Hamilton Cycle ---
# Cycles are standardized; zero-crossings are marked by colored vertical lines.
# Both cycles track each other closely. SSA (blue) crosses the
# zero line less frequently than the Hamilton output (red),
# reflecting the imposed smoothness constraint.
mplot <- scale(cbind(SSA_out, ham_out))
colo  <- c("blue", "red")
par(mfrow = c(1, 1))
ts.plot(
  mplot[, 1],
  col  = colo[1],
  main = "Standardized SSA vs. Hamilton Cycle"
)
mtext("SSA",      col = colo[1], line = -1)
mtext("Hamilton", col = colo[2], line = -2)
lines(mplot[, 2], col = colo[2])
abline(v=1+which(mplot[2:nrow(mplot),1]*mplot[1:(nrow(mplot)-1),1]<0),col="blue")
abline(v=1+which(mplot[2:nrow(mplot),2]*mplot[1:(nrow(mplot)-1),2]<0),col="red",lty=2)
abline(h = mean(mplot[, 1], na.rm = TRUE), col = "blue", lty = 2)
# Note: although the SSA nowcast imposes a stricter smoothness constraint
# than the Hamilton filter (fewer zero-crossings), this does not translate
# into a materialized lag relative to HF in the above plot. 

# =============================================================================
# 1.8 Forecasting: Gaining Timeliness Without Sacrificing Smoothness
# =============================================================================
#
# We investigate whether increasing the SSA forecast horizon allows the
# SSA cycle to LEAD the Hamilton cycle while preserving the same degree
# of smoothness (i.e., the same holding-time constraint).

# Set a 1-year (4-quarter) forecast horizon; all other SSA settings
# (filter length L, target filter, smoothness parameter rho1, and input
# assumption xi) remain unchanged from Section 1.6.
forecast_horizon     <- 4
SSA_obj_ham_diff     <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)
SSA_filt_ham_diff_forecast <- SSA_obj_ham_diff$ssa_x

# Apply the forecast SSA filter to the differenced log-GDP series.
SSA_out_forecast <- filter(x, SSA_filt_ham_diff_forecast, side = 1)

# --- Holding-Time Comparison ---

# Compare empirical zero-crossing frequencies across the three cycle estimates.
# Expected result: despite the longer forecast horizon, the SSA forecast
# cycle retains fewer zero-crossings than the Hamilton filter, confirming
# that smoothness is preserved.
compute_empirical_ht_func(SSA_out_forecast)
compute_empirical_ht_func(ham_out)

# Repeat on mean-centred (scaled) series to remove the level-offset bias.
# The centred SSA forecast cycle should exhibit approximately 30% fewer
# zero-crossings than the centred Hamilton cycle (subject to sample variation).
compute_empirical_ht_func(scale(SSA_out_forecast))
compute_empirical_ht_func(scale(ham_out))

# --- Visual Comparison: SSA Nowcast, SSA Forecast, and Hamilton Cycle ---

# Standardise all three series before plotting to place them on a common
# scale and facilitate direct visual comparison of timing and smoothness.
mplot <- scale(cbind(SSA_out, SSA_out_forecast, ham_out))
colo  <- c("blue", "darkgreen", "red")

par(mfrow = c(1, 1))
ts.plot(mplot[, 1], col = colo[1],
        ylim  = c(min(mplot, na.rm = TRUE), max(mplot, na.rm = TRUE)),
        main  = "Standardised Cycles: SSA Nowcast, SSA Forecast, and Hamilton Filter",
        ylab  = "Standardised cycle")
lines(mplot[, 2], col = colo[2])
lines(mplot[, 3], col = colo[3])
mtext("SSA nowcast",
      col = colo[1], line = -1)
mtext(paste("SSA forecast: horizon = ", forecast_horizon, " quarters", sep = ""),
      col = colo[2], line = -2)
mtext("Hamilton filter",
      col = colo[3], line = -3)
abline(h = 0)

# Key observations:
#   1. The standardised cycles exhibit a slow downward drift. This could be
#      corrected by accounting for the level drift identified in Section 1.3
#      (omitted here for brevity).
#   2. The SSA forecast cycle is shifted to the LEFT relative to both the
#      SSA nowcast and the Hamilton cycle: it leads the benchmark while
#      maintaining the same smoothness constraint.


# ==============================================================================
# MAIN TAKE-AWAY:
#   SSA outperforms the Hamilton filter on two dimensions simultaneously:
#     - TIMELINESS:  the SSA forecast cycle leads the Hamilton cycle.
#     - SMOOTHNESS:  the SSA cycle generates fewer zero-crossings than HF,
#                    regardless of the forecast horizon.
# ==============================================================================


# =============================================================================
# 1.9 Amplitude and Phase-Shift Functions
# =============================================================================
#
# Frequency-domain diagnostics provide a formal, data-independent complement
# to the time-domain comparisons above. Two key characteristics are examined:
#
#   Amplitude function:
#     Measures the filter gain at each frequency. Values near zero at high
#     frequencies indicate effective suppression of high-frequency noise,
#     which directly reduces spurious zero-crossings in the cycle estimate.
#
#   Phase-shift function:
#     Measures the lag (positive) or lead (negative) of the filter output
#     relative to the true signal at each frequency. A smaller phase-shift
#     in the passband implies a more timely cycle estimate.

# Set the number of equidistant frequency ordinates on [0, pi].
# A larger K yields a finer frequency grid and higher resolution.
K <- 600

# Compute amplitude and phase-shift functions for all three filters.
# All filters are expressed in the difference domain (applied to first-
# differenced log-GDP), enabling direct comparison on the same basis.
amp_obj_SSA_now <- amp_shift_func(K, as.vector(SSA_filt_ham_diff),           FALSE)
amp_obj_SSA_for <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_forecast),  FALSE)
amp_obj_ham     <- amp_shift_func(K, ham_diff,                                FALSE)

par(mfrow = c(2, 1))

# --- Panel 1: Amplitude Functions ---

# Collect amplitude functions into a matrix and normalise each to unity at
# frequency zero. Normalisation at frequency zero is valid for lowpass filters 
# and enables direct visual comparison of relative attenuation across filters.
mplot <- cbind(amp_obj_SSA_now$amp, amp_obj_SSA_for$amp, amp_obj_ham$amp)
mplot[, 1] <- mplot[, 1] / mplot[1, 1]
mplot[, 2] <- mplot[, 2] / mplot[1, 2]
mplot[, 3] <- mplot[, 3] / mplot[1, 3]

colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ", nowcast)", sep = ""),
  paste("SSA(", round(ht, 1), ", h=", forecast_horizon, ")", sep = ""),
  "Hamilton"
)

plot(mplot[, 1], type = "l", axes = FALSE, xlab = "Frequency", ylab = "",
     main = "Amplitude Functions: SSA vs. Hamilton Filter",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# --- Panel 2: Phase-Shift Functions ---

# Phase-shift = phase angle / frequency.
# Interpretation: the number of periods by which the filter output lags
# (positive values) or leads (negative values) the true signal at each
# frequency. Comparisons are most meaningful in the passband (frequencies
# where the amplitude exceeds ~0.5), where the filter transmits rather
# than attenuates the signal.
mplot <- cbind(amp_obj_SSA_now$shift, amp_obj_SSA_for$shift, amp_obj_ham$shift)

colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ", nowcast)", sep = ""),
  paste("SSA(", round(ht, 1), ", h=", forecast_horizon, ")", sep = ""),
  "Hamilton"
)

plot(mplot[, 1], type = "l", axes = FALSE, xlab = "Frequency", ylab = "",
     main = "Phase-Shift Functions: SSA vs. Hamilton Filter",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# =============================================================================
# Discussion of Results
# =============================================================================
#
# Amplitude Functions:
#
#   1. 'ham_diff', applied to first differences, behaves as a LOWPASS filter.
#      (The original Hamilton filter applied to log-levels is a bandpass.)
#
#   2. Both SSA amplitude functions lie BELOW that of 'ham_diff' at higher
#      frequencies, confirming that SSA more aggressively attenuates
#      high-frequency noise. This is the direct mechanism behind the
#      reduction in spurious zero-crossings relative to HF, and is a
#      characteristic property of SSA designs.
#      See Wildi (2026a), Section 4.2, for theoretical background.
#
# Phase-Shift Functions:
#
#   3. The SSA nowcast phase-shift is marginally larger than that of HF in
#      the passband. The difference is negligible in practice: the SSA
#      nowcast and the Hamilton cycle are nearly time-aligned (as illustrated
#      in the filter outputs plot).
#
#   4. The SSA forecast phase-shift is the SMALLEST of the three in the
#      passband, formally confirming the relative lead of the SSA forecast
#      over HF observed in the time-domain comparisons above.



# =============================================================================
# Example 2: Monthly Employment Data
# =============================================================================
#
# This example applies SSA-based customisation of the Hamilton filter to
# monthly non-farm payroll employment (PAYEMS). The analysis proceeds in two
# stages: first, the Hamilton filter is illustrated using the 'neverhpfilter'
# package on quarterly PAYEMS; second, SSA customisation is carried out on
# monthly data over a post-1990 sub-sample.

library(neverhpfilter)

# Load PAYEMS and convert to quarterly log-scaled series (base 100).
data(PAYEMS)
log_Employment <- 100 * log(xts::to.quarterly(PAYEMS["1947/2016-6"], OHLC = FALSE))

# Plot log-employment and its Hamilton trend to visualise the long-run
# trajectory and the trend-fitting behaviour of the Hamilton regression.
employ_trend <- yth_filter(log_Employment, h = 8, p = 4,
                           output = c("x", "trend"), family = gaussian)
plot.xts(employ_trend, grid.col = "white", legend.loc = "topleft",
         main = "Log Employment and Hamilton Trend")

# Decompose into cycle and irregular components.
# Note: in practice the cycle and irregular components are often correlated,
# which contradicts the standard orthogonality assumption underlying most
# trend-cycle decompositions.
employ_cycle <- yth_filter(log_Employment, h = 8, p = 4,
                           output = c("cycle", "random"), family = gaussian)
par(mfrow = c(1, 1))
plot.xts(employ_cycle, grid.col = "white", legend.loc = "topright",
         main = "Hamilton Cycle and Irregular Components")
abline(h = 0)

# Plot monthly log-returns to assess stationarity of the growth rate series.
# The series exhibits visible non-stationarity: a downward drift over time
# and structural breaks at key episodes (pre/post-1960, pre/post-1990,
# Great Moderation, COVID-19).
plot(diff(log(PAYEMS)), main = "Monthly Log-Returns of PAYEMS")
abline(h = 0)


# =============================================================================
# 2.1 Data Selection: Post-1990 Sub-Sample
# =============================================================================
#
# Two methodological novelties relative to Example 1:
#
#   I.  DATA-GENERATING PROCESS (DGP) FOR SSA:
#       First differences of PAYEMS exhibit strong serial autocorrelation.
#       Unlike Example 1, the white-noise assumption is no longer tenable;
#       an explicit ARMA model is fitted and its Wold decomposition (xi) is
#       supplied to the SSA optimisation.
#
#   II. SUB-SAMPLE SELECTION (POST-1990):
#       The full post-WWII sample displays severe non-stationarity and
#       multiple structural breaks. Restricting the sample to the post-1990
#       period (the Great Moderation era) yields a more homogeneous series,
#       improving both the Hamilton regression and the ARMA model fit.
#
# Consequences of sub-sample selection:
#   1. Hamilton filter: regression parameters are re-estimated on post-1990
#      data, potentially yielding a different cycle definition than the
#      full-sample estimate.
#   2. ARMA model: the estimated autocorrelation structure (and hence the
#      Wold decomposition xi) reflects the smoother dynamics of the
#      Great Moderation period.
#
# NOTE: The pandemic period is excluded from the baseline analysis
#       (sample ends December 2019). COVID-19 effects are examined
#       separately at the end of the tutorial.

# --- Data Loading ---

# Set reload_data = TRUE to download the latest vintage from FRED;
# set to FALSE to load the previously saved local copy.
reload_data <- FALSE

if (reload_data) {
  PAYEMS <- get_fred_series("PAYEMS", series_name = "GDP")
  PAYEMS <- as.xts(PAYEMS)
  save(PAYEMS, file = file.path(getwd(), "Data", "PAYEMS"))
} else {
  load(file = file.path(getwd(), "Data", "PAYEMS"))
}

# Inspect the series endpoints to confirm the loaded vintage.
head(PAYEMS)
tail(PAYEMS)

# Extract the post-1990, pre-pandemic sub-sample in log-levels.
y   <- as.double(log(PAYEMS["1990::2019"]))
len <- length(y)
names(y)<-index(PAYEMS["1990::2019"])
plot(y,main = "Log(PAYEMS): 1990–2019",
     type = "l", axes = F,
     xlab = "", ylab = "")
axis(1, at = 1:length(y),
     labels = names(y))
axis(2)
box()


# =============================================================================
# 2.2 Hamilton Filter
# =============================================================================
#
# Hamilton's original parameterisation targets quarterly data:
#   - Forecast horizon h = 8 quarters (2 years).
#   - AR order p = 4 (accommodates up to four potential unit roots).
#
# Adaptation to monthly PAYEMS:
#   - Forecast horizon scaled to h = 24 months (2 years).
#   - AR order p = 4 is retained, as it remains a reasonable choice for
#     capturing short-run persistence at monthly frequency.
h <- 2 * 12   # 24-month forecast horizon
p <- 4        # AR order

# Construct the matrix of lagged regressors and the h-step-ahead forecast target.
explanatory <- cbind(
  y[(p):(len - h)],
  y[(p - 1):(len - h - 1)],
  y[(p - 2):(len - h - 2)],
  y[(p - 3):(len - h - 3)]
)
target <- y[(h + p):len]

# Estimate the Hamilton regression: regress the h-step-ahead log-level on
# the p most recent lags.
lm_obj <- lm(target ~ explanatory)
summary(lm_obj)

# Plot the raw Hamilton cycle (OLS residuals from the above regression).
ts.plot(lm_obj$residuals, main = "Hamilton Cycle Residuals (Post-1990)")

# The dependence structure in the regression residuals reflects the 
# two-year forecast horizon:
acf(lm_obj$residuals)

# --- Construct the Hamilton Filter Coefficient Vector ---
# The filter vector encodes the mapping from current and lagged log-levels
# to the cycle estimate:
#   - The first element (1) corresponds to the h-step-ahead observation.
#   - The next h-1 zeros span the forecast horizon gap.
#   - The remaining p elements are the negated OLS slope coefficients.
hamilton_filter <- c(1, rep(0, h - 1), -lm_obj$coefficients[1 + 1:p])
intercept       <- lm_obj$coefficients[1]
  
# --- Replication Check ---
# Reconstruct the cycle by applying the filter to the data matrix, which
# stacks the forecast target (repeated h times) alongside the regressors.
# Subtracting the intercept should reproduce the OLS residuals exactly.
data_mat  <- cbind(matrix(rep(target, h), ncol = h), explanatory)
residuals <- data_mat %*% hamilton_filter - intercept

# Visual confirmation: the two series should overlap exactly.
par(mfrow = c(1, 1))
mplot<-cbind(residuals,lm_obj$residuals)
colo<-c("blue", "red")
plot(mplot[,1], col = colo[1],
     main = "Replication Check: Hamilton Filter vs. Regression Residuals",
     type = "l", axes = F,
     xlab = "", ylab = "")
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
axis(1, at = 1:nrow(mplot),
     labels = rownames(mplot))
axis(2)
box()



# =============================================================================
# 2.3 Unit-Root Adjusted Hamilton Filter
# =============================================================================

# --- Cointegration Diagnostic ---
# Check whether the unrestricted Hamilton filter coefficients sum to zero.
# A non-zero sum indicates that the filter does not satisfy the unit-root
# cancellation condition and would produce a non-stationary cycle if applied
# out-of-sample.
sum(hamilton_filter)

# --- Unit-Root Restriction ---
# Re-estimate the Hamilton filter subject to the constraint that the slope
# coefficients sum to exactly 1, which forces the filter coefficients to sum
# to zero (cointegration / unit-root cancellation condition).
# See Section 1.2 for a full discussion of the motivation and methodology.
Diff    <- TRUE
Ham_obj <- HamiltonFilter_Restricted(y, p, h, Diff)

# Confirm that the restricted slope coefficients sum to exactly 1.
sum(Ham_obj$coefficients[-1])

# Construct the adjusted Hamilton filter vector.
hamilton_filter_adjusted <- c(1, rep(0, h - 1), -Ham_obj$coefficients[-1])

# Confirm that the adjusted filter coefficients sum to exactly zero,
# verifying that the unit-root cancellation condition is satisfied.
sum(hamilton_filter_adjusted)

# Compute the unit-root-adjusted cycle by applying the adjusted filter to
# the data matrix.
residuals_adjusted <- data_mat %*% hamilton_filter_adjusted

# --- Visual Comparison: Classic vs. Adjusted Cycle ---
# The two cycle variants differ primarily in level; their dynamic shapes
# are nearly identical. See Example 1 for a full discussion.
par(mfrow = c(2, 2))
ts.plot(y,
        main = "Log(PAYEMS): 1990–2019")
ts.plot(cbind(residuals, residuals_adjusted),
        col  = c("red", "blue"),
        main = "Hamilton Cycle: Classic vs. Adjusted")
mtext("Classic cycle (regression residuals)", col = "red",  line = -1)
mtext("Adjusted uncentred cycle",             col = "blue", line = -2)
ts.plot(residuals - residuals_adjusted,
        main = "Difference Between Cycle Definitions")

# ==========================================================================
# MAIN TAKE-AWAY:
#   - The adjusted and original cycles share NEARLY IDENTICAL DYNAMICS;
#     the primary difference is a slowly evolving level offset (bottom plot).
#   - SSA CUSTOMIZATION WILL THEREFORE HAVE COMPARABLE EFFECTS ON EITHER
#     CYCLE VARIANT, SO THE CHOICE BETWEEN THEM DOES NOT MATERIALLY AFFECT
#     SSA-BASED CONCLUSIONS.
#   - For technical reasons (zero-sum filter constraint, cointegration
#     property), we proceed with the adjusted, uncentred cycle as the
#     basis for SSA customization.
# ==========================================================================


# =============================================================================
# 2.4 Transformation: From Levels to First Differences
# =============================================================================
#
# As in Example 1, we derive the difference-domain equivalent filter 'ham_diff'
# by convolving the adjusted Hamilton filter with the summation (unit-root)
# filter. The same approach applies here, but with a larger filter length
# L = 50.
#
# Motivation for the larger L:
#   Post-1990 log-returns exhibit slower ACF decay (longer memory) relative
#   to the full post-WWII sample, consistent with the reduced volatility of
#   the Great Moderation. A larger L ensures that the Wold decomposition
#   coefficients (xi) decay sufficiently close to zero within the filter
#   window, as required for numerical accuracy. See Section 2.3 in 
#   Wildi (2024).

L <- 50

# Ensure L is at least as long as the adjusted Hamilton filter.
L <- max(length(hamilton_filter_adjusted), L)

# Zero-pad the adjusted Hamilton filter to length L if necessary.
if (L > length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L <- c(
    hamilton_filter_adjusted,
    rep(0, L - length(hamilton_filter_adjusted))
  )

# Derive the difference-domain filter 'ham_diff' by convolving the zero-padded
# Hamilton filter with the summation (unit-root) filter (1 - B)^{-1}.
ham_diff <- conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv

# Visualise both filter representations for comparison.
par(mfrow = c(2, 1))
ts.plot(ham_diff,
        main = "Hamilton Filter: Difference-Domain Representation (Input: x_t=y_t-y_{t-1})",
        xlab="Lag",ylab="")
ts.plot(hamilton_filter_adjusted_L,
        main = "Hamilton Filter: Level-Domain Representation (Input: y_t)",
        xlab="Lag",ylab="")
# --- Filter Equivalence Verification ---
# Applying 'ham_diff' to the first-differenced log-PAYEMS series should
# reproduce the same cycle as applying 'hamilton_filter_adjusted_L' to the
# log-level series.
x             <- diff(y)
len_diff      <- length(x)
residual_diff <- na.exclude(filter(x, ham_diff, side = 1))

# Prepend NAs to the level-domain cycle to align sample lengths.
original_hamilton_cycle <- c(rep(NA, length(x) - length(residuals)), residuals)

# Confirmation plot: both series should overlap exactly.
par(mfrow = c(1, 1))
mplot<-cbind(residual_diff,residuals_adjusted[(L - p - h + 2):length(residuals)])
rownames(mplot)<-rownames(residuals_adjusted)[(L - p - h + 2):length(residuals)]
colo<-c("blue", "red")
plot(mplot[,1], col = colo[1],
     main = "Replication Check: Hamilton Filter vs. Regression Residuals",
     type = "l", axes = F,
     xlab = "", ylab = "")
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
axis(1, at = 1:nrow(mplot),
     labels = rownames(mplot))
axis(2)
box()


# Having confirmed equivalence, SSA customisation proceeds entirely in the
# difference domain using 'ham_diff' as the target filter.


# =============================================================================
# 2.5 Holding Time Analysis
# =============================================================================

# Compute the theoretical holding time of 'ham_diff' under the white-noise
# assumption. This serves as the baseline smoothness reference for SSA.
ht_ham_diff_obj <- compute_holding_time_func(ham_diff)
ht_ham_diff_obj$ht

# Compute the empirical holding time directly from the adjusted cycle.
compute_empirical_ht_func(residuals_adjusted)

# The empirical holding time is substantially longer than the theoretical
# value. Two factors contribute to this discrepancy:
#   i.  The adjusted cycle is uncentred (positive level drift), which
#       suppresses zero-crossings and upward-biases the empirical estimate.
#   ii. The log-returns x_t are strongly autocorrelated (not white noise),
#       so the theoretical holding time (derived under white noise) does not
#       apply directly to the filtered output.

# After mean-centring, the empirical holding time moves closer to the
# theoretical value — though the autocorrelation bias remains unaddressed
# at this stage.
compute_empirical_ht_func(scale(residuals_adjusted))

# Visualise the log-returns to confirm the strong serial dependence.
par(mfrow = c(1, 1))
mplot<-matrix(x,ncol=1)
rownames(mplot)<-names(x)
colo<-"black"
plot(mplot[,1], col = colo[1],
     main = "Replication Check: Hamilton Filter vs. Regression Residuals",
     type = "l", axes = F,
     xlab = "", ylab = "")
abline(h = 0)
axis(1, at = 1:nrow(mplot),
     labels = rownames(mplot))
axis(2)
box()

# The ACF of log-returns displays slow, persistent decay — inconsistent
# with white noise and motivating the ARMA model fitted in the next Section 2.6.
acf(x, main = "ACF of Log-Returns: Slowly Decaying (Post-1990)")


# =============================================================================
# 2.6 Autocorrelation Analysis and ARMA Model Fitting
# =============================================================================
#
# The strong autocorrelation in log-returns requires an explicit model of
# the data-generating process for the SSA optimisation. We fit an ARMA(1,1)
# to the log-returns and compute its Wold (MA-infinity) decomposition xi,
# which is then supplied to SSA_func to inform SSA about the data generating 
# process.
#
# OPTIONAL: Set try_out_of_sample = TRUE to estimate the ARMA model on the
# first half of the sample only, as a robustness check against overfitting.
# SSA is largely insensitive to moderate ARMA misspecification: both
# in-sample and out-of-sample parameterisations yield nearly identical
# SSA filter designs. The comments below refer to full-sample estimation
# (try_out_of_sample = FALSE).
try_out_of_sample <- FALSE

if (try_out_of_sample) {
  # Use the first half of x for ARMA estimation.
  in_sample_length <- length(x) / 2
} else {
  # Use the full sample for ARMA estimation.
  in_sample_length <- length(x)
}

# Inspect the ACF of post-1990 log-returns to guide model order selection.
# The slow, persistent decay is consistent with an ARMA(1,1) specification.
acf(x[1:in_sample_length],
    main = "ACF of Post-1990 Log-Returns: Slowly Decaying (Long Memory)")

# Fit an ARMA(1,1) model to the log-returns.
ar_order  <- 1
ma_order  <- 1
estim_obj <- arima(x[1:in_sample_length], order = c(ar_order, 0, ma_order))
estim_obj

# Residual diagnostics: confirm that the ARMA(1,1) adequately whitens the
# log-returns (residual ACF should be negligible at all lags).
tsdiag(estim_obj)

# --- Wold Decomposition (MA-Infinity Representation) ---
# Compute the infinite-order MA coefficients (impulse response weights) of
# the fitted ARMA model. The filter length L = 100 was chosen to ensure that
# the coefficients decay sufficiently close to zero by lag L.
xi <- c(1, ARMAtoMA(
  ar      = estim_obj$coef[1:ar_order],
  ma      = estim_obj$coef[ar_order + 1:ma_order],
  lag.max = L - 1
))

# Visualise xi: the slow decay confirms the longer-memory character of the
# post-1990 log-returns relative to the full post-WWII sample.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition xi: Slowly Decaying Impulse Response (Post-1990)")
# The plot illustrates that filter length L=50 is required

# --- Holding-Time Correction for Autocorrelation ---
# Convolve xi with 'ham_diff' to obtain the composite filter applied to
# the white-noise innovations epsilon_t. The holding time of this convolved
# filter accounts for the autocorrelation structure of the log-returns and
# provides a corrected theoretical benchmark for comparison with the
# empirical holding time.
ham_conv        <- conv_two_filt_func(xi, ham_diff)$conv
ht_ham_conv_obj <- compute_holding_time_func(ham_conv)

# Compare the autocorrelation-corrected theoretical holding time with the
# empirical holding time of the centred adjusted cycle.
ht_ham_conv_obj$ht
compute_empirical_ht_func(scale(residuals_adjusted))

# After correcting for both the uncentred mean (via scaling) and the
# autocorrelation structure (via the convolved filter), the theoretical
# and empirical holding times are in agreement (up to finite sample variation).


# =============================================================================
# 2.7 Apply SSA (Post-1990, ARMA-Informed)
# =============================================================================

# -----------------------------------------------------------------------------
# 2.7.1 SSA Settings
# -----------------------------------------------------------------------------

# Display the autocorrelation-corrected theoretical holding time of the
# Hamilton filter as the baseline smoothness reference.
# The value is slightly under one year, reflecting the relatively high
# zero-crossing frequency of the Hamilton cycle at monthly frequency.
ht_ham_conv_obj$ht

# Set the SSA holding-time target 50% above the Hamilton baseline.
# Under correct model specification and stationarity, this corresponds to
# approximately 30% fewer zero-crossings in the SSA output relative to HF.
ht <- 1.5 * ht_ham_conv_obj$ht

# We can supply the lag-one ACF rho1 or ht to SSA
rho1 <- compute_rho_from_ht(ht)

# Supply 'ham_diff' as the SSA target filter: SSA seeks the optimal causal
# approximation to this difference-domain Hamilton filter subject to the
# imposed smoothness and timeliness constraints.
gammak_generic <- ham_diff

# Set the forecast horizon to zero (nowcast).
# Forecasting variants are explored in Section 2.9.
forecast_horizon <- 0

# -----------------------------------------------------------------------------
# 2.7.2 Run SSA Optimisation
# -----------------------------------------------------------------------------

# Run SSA with the ARMA(1,1) Wold decomposition xi explicitly supplied.
# Providing xi ensures that the optimisation correctly accounts for the
# autocorrelation structure of the log-returns x_t, rather than defaulting
# to the white-noise assumption used in Example 1.
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract the two filter representations returned by SSA_func:
#
#   ssa_eps – filter in the innovation (epsilon_t) domain.
#             Used primarily to verify convergence of the optimisation,
#             since the holding-time constraint is defined in this domain.
#
#   ssa_x   – filter in the observable data (x_t) domain.
#             The operationally relevant filter for real-time cycle extraction:
#             apply this directly to the observed log-returns.
SSA_filt_ham_diff_eps <- SSA_obj_ham_diff$ssa_eps
SSA_filt_ham_diff_x   <- SSA_obj_ham_diff$ssa_x

# -----------------------------------------------------------------------------
# 2.7.3 Plot Filters
# -----------------------------------------------------------------------------

# --- Comparison in the x_t Domain ---
# 'ham_diff' and 'ssa_x' are both applied to the observed log-returns x_t,
# enabling a direct apples-to-apples comparison of the filter shapes.
mplot <- cbind(ham_diff, SSA_filt_ham_diff_x)
par(mfrow = c(1, 1))
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"),
        main = "Filter Coefficients in the Observable Domain (Applied to x_t)",
        xlab="Lag",ylab="")
mtext("Hamilton filter (ham_diff)", col = "black", line = -1)
mtext("SSA filter (ssa_x)",         col = "blue",  line = -2)

# --- Comparison in the Innovation Domain ---
# 'ham_conv' (Hamilton filter convolved with xi) and 'ssa_eps' are both
# expressed in terms of the white-noise innovations epsilon_t — the natural
# comparison domain within the Wold representation. Differences here reflect
# the SSA smoothness constraint operating on the innovation process.
mplot <- cbind(ham_conv, SSA_obj_ham_diff$ssa_eps)
par(mfrow = c(1, 1))
ts.plot(mplot, ylim = c(min(mplot), max(mplot)), col = c("black", "blue"),
        main = "Filter Coefficients in the Innovation Domain (Applied to epsilon_t)",
        xlab="Lag",ylab="")
mtext("Hamilton filter convolved with xi (ham_conv)", col = "black", line = -1)
mtext("SSA filter (ssa_eps)",                         col = "blue",  line = -2)

# -----------------------------------------------------------------------------
# 2.7.4 Convergence Check
# -----------------------------------------------------------------------------

# Verify convergence of the SSA optimisation: the effective holding time of
# 'ssa_eps' should match the imposed target 'ht' up to numerical rounding.
# Close agreement between the two values confirms convergence to the global
# optimum of the SSA criterion.
ht_obj <- compute_holding_time_func(SSA_filt_ham_diff_eps)

# Effective holding time of the optimised SSA filter (innovation domain):
ht_obj$ht

# Imposed holding-time target:
ht


# =============================================================================
# 2.8 Filter the Series and Evaluate Performance
# =============================================================================

# --- 2.8.1 SSA Filter Output ---

# Apply the SSA nowcast filter (observable domain, ssa_x) to the log-returns
# using one-sided (causal) filtering.
SSA_out <- filter(x, SSA_filt_ham_diff_x, side = 1)
names(SSA_out)<-names(x)

# Compare the empirical holding time of the centred SSA output with the target.
# The estimate matches up to finite sample variation.
compute_empirical_ht_func(scale(SSA_out))
ht   # Imposed holding-time target for reference

# --- 2.8.2 Hamilton Filter Output ---

# Apply the Hamilton difference-domain filter to the log-returns.
ham_out <- filter(x, ham_diff, side = 1)
names(ham_out)<-names(x)

# Empirical holding time of the centred Hamilton output.
# Empirical estimate matches up to sampling error. 
compute_empirical_ht_func(scale(ham_out))
ht_ham_conv_obj$ht   # Autocorrelation-corrected theoretical benchmark

# --- Visual Comparison of SSA and Hamilton Cycle Estimates ---
# Standardized cycles with zero-crossings marked by colored vertical lines.
# SSA generates less crossings than HF
mplot <- scale(na.exclude(cbind(SSA_out, ham_out)))
rownames(mplot)<-names(na.exclude(SSA_out))
colo<-c("blue", "red")
plot(mplot[,1], col = colo[1],
     main = "SSA vs. Hamilton Cycle: Centred and Standardised",
     type = "l", axes = F,
     xlab = "", ylab = "")
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h = 0)
abline(v=1+which(mplot[2:nrow(mplot),1]*mplot[1:(nrow(mplot)-1),1]<0),col=colo[1])
abline(v=1+which(mplot[2:nrow(mplot),2]*mplot[1:(nrow(mplot)-1),2]<0),col=colo[2],lty=2,lwd=1)
axis(1, at = 1:nrow(mplot),
     labels = rownames(mplot))
axis(2)
box()

# --- Holding-Time Summary ---

# SSA achieves approximately 50% longer holding time (30% fewer zero-crossings)
# relative to the Hamilton filter, consistent with the imposed target.
# Residual discrepancies are within the range of sampling variability.
compute_empirical_ht_func(scale(SSA_out))
compute_empirical_ht_func(scale(ham_out))


# ==========================================================================
# 2.9 Forecasting: Timeliness and Lead/Lag Analysis
# ==========================================================================
# =============================================================================
# OBJECTIVE
# =============================================================================
# Goal:
#   Construct faster (more timely) concurrent SSA filters by augmenting the
#   forecast horizon. 
#
# Design strategy:
#   - The holding-time constraint is held fixed throughout, ensuring that the
#     noise-suppression properties of the filter are preserved regardless of
#     the chosen forecast horizon.
#   - Extending the forecast horizon improves timeliness (reduces phase lag)
#     while maintaining smoothness — achieving both desirable properties
#     simultaneously.
#
# Trade-off:
#   - The dual gain in smoothness and timeliness comes at a cost: accuracy
#     (as measured by target correlation) deteriorates disproportionately
#     as the forecast horizon is extended.
#   - In other words, the filter becomes faster and smoother, but at the
#     expense of tracking the target signal less faithfully.
#
# Note:
#   - The forecast horizon addresses timeliness indirectly. The novel 
#     look-ahead DFP and PCS are Pareto optimal and trace the accuracy-
#     timeliness frontier (tutorial in preparation).
#
# =============================================================================
# 2.9.1 Forecast Horizon Specification
# =============================================================================
# We define two forecast horizons to compare against the concurrent (nowcast)
# filter (delta = 0):
#   - delta =  6: six months ahead  (half-year horizon)
#   - delta = 12: twelve months ahead (full-year horizon)
#
# Note: SSA_func accepts a vector of horizons and returns a filter matrix
#       with one column per forecast horizon.
forecast_horizon <- c(6, 12)


# =============================================================================
# 2.9.2 SSA Filter Estimation: Forecast Horizons
# =============================================================================
# Fit SSA filters for both forecast horizons using:
#   - Filter length L
#   - ARMA(1,1) Wold decomposition coefficients xi
#   - Autocovariance sequence gammak_generic
#   - HT constraint parameter rho1
#   - xi Wold decomposition of ARMA(1,1)
#
# The returned object contains filters expressed in both the x_t domain
# and the epsilon_t (innovations) domain.
SSA_obj_ham_diff <- SSA_func(L, forecast_horizon, gammak_generic, rho1, xi)

# Extract the estimated filter coefficients in the x_t domain
# (one column per forecast horizon):
SSA_filt_ham_diff_x_forecast <- SSA_obj_ham_diff$ssa_x


# =============================================================================
# 2.9.3 Visualization: Filter Coefficient Morphing Across Horizons
# =============================================================================
# As the forecast horizon increases, the filter shifts progressively more weight
# toward recent observations and away from the remote past — a morphing toward
# a leading (anticipating) shape.
#
# Crucially, the holding-time constraint is held fixed across all horizons,
# so smoothness (noise suppression) is preserved as timeliness improves.
par(mfrow = c(1, 1))
ts.plot(SSA_filt_ham_diff_x_forecast,
        col  = c("orange", "darkgreen"),
        main = "SSA Filter Coefficients: Nowcast vs. Forecast Horizons",
        ylim = c(min(SSA_filt_ham_diff_x), max(SSA_filt_ham_diff_x_forecast)),
        xlab="Lag",ylab="")
lines(SSA_filt_ham_diff_x, col = "blue")
mtext(paste("SSA forecast: delta =", forecast_horizon[1]), col = "orange",    line = -1)
mtext(paste("SSA forecast: delta =", forecast_horizon[2]), col = "darkgreen", line = -2)
mtext("SSA nowcast: delta = 0",                            col = "blue",      line = -3)


# =============================================================================
# 2.9.4 Convergence Verification: Holding Times Across Forecast Horizons
# =============================================================================
# We apply compute_holding_time_func to each column of ssa_eps (the filter
# expressed in the epsilon_t innovations domain).
#
# If the holding times match the target ht across all forecast horizons,
# this confirms that the optimizer converged to the global optimum
# for each horizon.
apply(SSA_obj_ham_diff$ssa_eps, 2, compute_holding_time_func)
ht   # Target holding time (for reference)


# =============================================================================
# 2.9.5 Apply Forecast Filters to the First-Differenced Series
# =============================================================================
# Compute the filter output for each forecast horizon by convolving the
# estimated filter coefficients with the first-differenced input series x.

# Six-month-ahead SSA forecast filter output:
SSA_out_forecast_6  <- filter(x, SSA_filt_ham_diff_x_forecast[, 1], side = 1)
names(SSA_out_forecast_6)<-names(x)
# Twelve-month-ahead SSA forecast filter output:
SSA_out_forecast_12 <- filter(x, SSA_filt_ham_diff_x_forecast[, 2], side = 1)
names(SSA_out_forecast_12)<-names(x)


# =============================================================================
# Visualization: All Filter Outputs
# =============================================================================
# We overlay all filter outputs (nowcast + two forecast horizons + Hamilton)
# to assess the timeliness-smoothness trade-off visually.
#
# Expected pattern:
#   - SSA forecast outputs are left-shifted (leading) relative to the Hamilton
#     filter, with the degree of lead increasing with the forecast horizon.
#   - Smoothness (cycle duration) is approximately equal across all SSA variants,
#     reflecting the binding holding-time constraint.
mplot <- scale(na.exclude(cbind(SSA_out, SSA_out_forecast_6, SSA_out_forecast_12, ham_out)))
rownames(mplot)<-names(na.exclude(SSA_out))
colo  <- c("blue", "orange", "darkgreen", "red")
plot(mplot[,1], col = colo[1],ylim = c(min(mplot), max(mplot)),
     main = "SSA vs. Hamilton Cycles:  Standardised",
     type = "l", axes = F,
     xlab = "", ylab = "")
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h = 0)
#abline(v=1+which(mplot[2:nrow(mplot),1]*mplot[1:(nrow(mplot)-1),1]<0),col=colo[1])
#abline(v=1+which(mplot[2:nrow(mplot),2]*mplot[1:(nrow(mplot)-1),2]<0),col=colo[2],lty=2,lwd=1)
axis(1, at = 1:nrow(mplot),
     labels = rownames(mplot))
axis(2)
box()

# =============================================================================
# 2.9.6 Empirical Holding Times: SSA Variants vs. Hamilton Benchmark
# =============================================================================
# We compute empirical holding times (average duration between zero-crossings)
# for all filter outputs.
#
# Expected result:
#   - All SSA variants should achieve longer holding times than Hamilton,
#     confirming that the holding-time constraint enforces
#     stronger noise suppression across all forecast horizons.
#   - Holding times should be approximately equal (up to random sampling 
#     error) across SSA variants, since all share the same constraint.
compute_empirical_ht_func(scale(SSA_out))             # SSA nowcast  (delta = 0)
compute_empirical_ht_func(scale(SSA_out_forecast_6))  # SSA forecast (delta = 6)
compute_empirical_ht_func(scale(SSA_out_forecast_12)) # SSA forecast (delta = 12)
compute_empirical_ht_func(scale(ham_out))             # Hamilton benchmark


# =============================================================================
# 2.10 Amplitude and Phase-Shift Functions
# =============================================================================
# PURPOSE:
#   Provide formal frequency-domain diagnostics that confirm the empirical
#   findings of Section 2.7. Two complementary functions are examined:
#
#   Amplitude function:
#     - Quantifies the filter's gain at each frequency.
#     - High-frequency amplitude close to zero indicates strong noise suppression,
#       which corresponds directly to longer holding times (smoother output).
#
#   Phase-shift function:
#     - Quantifies timing distortion introduced by the filter at each frequency.
#     - A more negative (smaller) phase shift in the passband (low frequencies)
#       indicates a relative lead — i.e., the filter output anticipates
#       turning points in the target signal.
#
# Together, these diagnostics characterise the smoothness-timeliness trade-off
# across the SSA nowcast, SSA forecast variants, and the Hamilton benchmark.


# -----------------------------------------------------------------------------
# Setup: Frequency Grid and Color Palette
# -----------------------------------------------------------------------------
# Number of equidistant frequency ordinates over [0, pi]:
K <- 600
# Colors assigned consistently across all plots:
#   Column 1 — SSA nowcast  (delta = 0)
#   Column 2 — SSA forecast (delta = 6 months)
#   Column 3 — SSA forecast (delta = 12 months)
#   Column 4 — Hamilton benchmark
colo <- c("black", "blue", "darkgreen", "red")
# -----------------------------------------------------------------------------
# Compute Amplitude and Phase-Shift Objects for All Four Filters
# -----------------------------------------------------------------------------
# All filters are expressed in the x_t domain (i.e., applied to log-return series).
# amp_shift_func returns both the amplitude and phase-shift at each frequency ordinate.
amp_obj_SSA_now    <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x),               F)
amp_obj_SSA_for_6  <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 1]), F)
amp_obj_SSA_for_12 <- amp_shift_func(K, as.vector(SSA_filt_ham_diff_x_forecast[, 2]), F)
amp_obj_ham        <- amp_shift_func(K, ham_diff,                                      F)
# -----------------------------------------------------------------------------
# Panel Layout: Amplitude (top panel) and Phase-Shift (bottom panel)
# -----------------------------------------------------------------------------
par(mfrow = c(2, 1))
# PLOT 1: Amplitude Functions
# =============================================================================
# Assemble amplitude matrix (one column per filter):
mplot <- cbind(
  amp_obj_SSA_now$amp,
  amp_obj_SSA_for_6$amp,
  amp_obj_SSA_for_12$amp,
  amp_obj_ham$amp
)
# Normalize each amplitude curve to unity at frequency zero (DC component).
# This enables a scale-free comparison of noise-suppression profiles across
# filters with potentially different overall gains.
mplot[, 1] <- mplot[, 1] / mplot[1, 1]
mplot[, 2] <- mplot[, 2] / mplot[1, 2]
mplot[, 3] <- mplot[, 3] / mplot[1, 3]
mplot[, 4] <- mplot[, 4] / mplot[1, 4]
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                   ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1],  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2],  ")", sep = ""),
  "Hamilton"
)
# Plot all normalized amplitude functions.
# Interpretation: a smaller amplitude at high frequencies implies stronger
# attenuation of noise, consistent with a longer empirical holding time.
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Amplitude Functions: Hamilton vs. SSA Variants (Post-1990)",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
# Overlay amplitude curves for remaining filters:
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()
# PLOT 2: Phase-Shift Functions
# =============================================================================
# Assemble phase-shift matrix (one column per filter):
mplot <- cbind(
  amp_obj_SSA_now$shift,
  amp_obj_SSA_for_6$shift,
  amp_obj_SSA_for_12$shift,
  amp_obj_ham$shift
)
colnames(mplot) <- c(
  paste("SSA(", round(ht, 1), ",", 0,                   ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[1],  ")", sep = ""),
  paste("SSA(", round(ht, 1), ",", forecast_horizon[2],  ")", sep = ""),
  "Hamilton"
)
# Plot all phase-shift functions.
# Interpretation: a smaller phase shift at low (passband) frequencies
# indicates that the filter output leads the target signal — i.e., turning
# points are anticipated. A larger shift implies a delay in signal detection.
plot(mplot[, 1], type = "l", axes = F,
     xlab = "Frequency", ylab = "",
     main = "Phase-Shift Functions: Hamilton vs. SSA Variants (Post-1990)",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
# Overlay phase-shift curves for remaining filters:
if (ncol(mplot) > 1) {
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i], col = colo[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

#
# =============================================================================
# DISCUSSION: Frequency-Domain Diagnostics — Amplitude and Phase-Shift
# =============================================================================
#
# The two plots above reveal a clear and consistent pattern across three
# evaluation criteria: smoothness, timeliness, and accuracy.
#
# -----------------------------------------------------------------------------
# Smoothness:
# -----------------------------------------------------------------------------
#   - In the top panel (amplitude functions), all SSA designs exhibit smaller
#     amplitudes than the Hamilton filter (HF) at high frequencies.
#   - Stronger attenuation of high-frequency components translates directly
#     into fewer zero-crossings and smoother filter output — confirming that
#     the holding-time constraint is binding and effective across all SSA variants.
#
# -----------------------------------------------------------------------------
# Timeliness:
# -----------------------------------------------------------------------------
#   - In the bottom panel (phase-shift functions), increasing the forecast
#     horizon progressively reduces the phase shift at low (passband) frequencies.
#   - A smaller phase shift implies that the filter output leads
#     the target HF signal — i.e., turning points are detected earlier as the
#     forecast horizon grows.
#
# -----------------------------------------------------------------------------
# Accuracy:
# -----------------------------------------------------------------------------
#   - As the forecast horizon increases, the SSA amplitude functions deviate
#     more strongly from the Hamilton filter target at low frequencies,
#     indicating growing passband distortion and a loss of accuracy.
#   - Zero-shrinkage pulls SSA filter coefficients toward zero, further
#     attenuating the passband gain. This effect is not visible in the plots
#     due to the unit normalization applied at frequency zero.
#   - Notably, the SSA nowcast (black curve) is nearly indistinguishable
#     from the Hamilton filter (red) across the entire passband:
#       * Amplitude functions are virtually identical at low frequencies.
#       * Phase-shift functions are virtually identical at low frequencies,
#         implying equivalent timeliness between the two.
#     The SSA nowcast thus matches Hamilton in both accuracy and timeliness,
#     while strictly dominating it in smoothness (stronger high-frequency
#     attenuation).
#
# -----------------------------------------------------------------------------
# Key Implication:
# -----------------------------------------------------------------------------
#   - The fact that SSA nowcast (black) outperforms HF in smoothness 
#     (amplitude in stop band) without sacrificing accuracy (amplitude in 
#     passband) or timeliness (phase-shift in passband) demonstrates that HF is
#     suboptimal: it does not lie on the efficient Accuracy-Timeliness-
#     Smoothness (ATS) frontier.
#   - Put differently, a better (SSA-nowcast) filter exists — one that achieves 
#     the same signal-tracking (amplitude in passband) and timing performance 
#     (time-shift in passband) as HF while attenuating noise (amplitude in 
#     stop band) more effectively.
#   - Extending the forecast horizon beyond the nowcast trades accuracy for
#     timeliness (for fixed HT smoothness) , tracing out a frontier of 
#     Pareto-improving designs relative to Hamilton along the smoothness-
#     timeliness dimension.
# Note: 
#   - Similar findings applied to HP (tutorial 2.1) where SSA could replicate
#     HP smoothness (HT) while improving target correlation or MSE substantially.



# =============================================================================
# 2.11 Out-of-Sample Application Including the Pandemic Period
# =============================================================================
# PURPOSE:
#   Apply all filters estimated on the 1990–2019 subsample — without any
#   re-estimation — to the extended series that includes the pandemic period
#   (2020 onwards). This out-of-sample exercise serves two purposes:
#     (1) Assess whether filter properties established in-sample hold up when
#         exposed to extreme, unprecedented observations.
#     (2) Exploit the pandemic shock as a natural impulse experiment
#         to visualise each filter's impulse response function directly 
#         in the data.
#
# KEY INSIGHT:
#   The extreme pandemic outliers appear as impulses in the log-differenced
#   series. By linearity, each filter responds to such an impulse with an output
#   proportional to its impulse response function. 


# -----------------------------------------------------------------------------
# Load Extended Sample: 1990 to Present (Including Pandemic)
# -----------------------------------------------------------------------------
y <- (log(PAYEMS["1990/"]))
x <- diff(y)
par(mfrow = c(1, 1))
plot(x, main = "Log-Returns of PAYEMS: 1990–Present (Including Pandemic)")
abline(h = 0)
x<-as.double(x)
names(x)<-index(diff(y))

# Visualise log-returns: the pandemic shock (early 2020) manifests as extreme
# outliers — a large negative return followed by a large positive rebound.


# -----------------------------------------------------------------------------
# Apply All Pre-Estimated Filters Out-of-Sample (No Re-Estimation)
# -----------------------------------------------------------------------------
# Filter coefficients are held fixed at their 1990–2019 in-sample estimates.
# Any differences in output across filters reflect differences in filter design,
# not differences in the estimation sample.

# 1. SSA nowcast filter (delta = 0):
SSA_out <- filter(x, SSA_filt_ham_diff_x, side = 1)
names(SSA_out)<-names(x)

# 2. Hamilton filter (benchmark):
ham_out <- filter(x, ham_diff, side = 1)
names(ham_out)<-names(x)

# 3. SSA 6-month-ahead forecast filter (delta = 6):
SSA_out_forecast_6 <- filter(x, SSA_filt_ham_diff_x_forecast[, 1], side = 1)
names(SSA_out_forecast_6)<-names(x)
# 4. SSA 12-month-ahead forecast filter (delta = 12):
SSA_out_forecast_12 <- filter(x, SSA_filt_ham_diff_x_forecast[, 2], side = 1)
names(SSA_out_forecast_12)<-names(x)


# -----------------------------------------------------------------------------
# Visualise All Filter Outputs Over the Extended Sample
# -----------------------------------------------------------------------------
# Expected pattern:
#   The large negative pandemic impulse (early 2020) is followed by a secondary
#   positive "phantom" peak in each filter output. This phantom peak is a filter
#   artifact — a direct consequence of the negative lobe in each filter's impulse
#   response — and does not reflect any real economic signal.
#   Its magnitude and timing vary across filter designs, reflecting differences
#   in coefficient shape and rate of decay.

mplot <- scale(na.exclude(cbind(SSA_out, SSA_out_forecast_6, SSA_out_forecast_12, ham_out)))
rownames(mplot)<-names(na.exclude(SSA_out))
colo  <- c("blue", "orange", "darkgreen", "red")
plot(mplot[,1], col = colo[1],ylim = c(min(mplot), max(mplot)),
     main = "SSA vs. Hamilton Cycles:  Standardised",
     type = "l", axes = F,
     xlab = "", ylab = "")
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h = 0)
#abline(v=1+which(mplot[2:nrow(mplot),1]*mplot[1:(nrow(mplot)-1),1]<0),col=colo[1])
#abline(v=1+which(mplot[2:nrow(mplot),2]*mplot[1:(nrow(mplot)-1),2]<0),col=colo[2],lty=2,lwd=1)
axis(1, at = 1:nrow(mplot),
     labels = rownames(mplot))
axis(2)
box()


# -----------------------------------------------------------------------------
# Revisit Filter Coefficients to Interpret the Pandemic Response
# -----------------------------------------------------------------------------
# The shape of the phantom post-pandemic peak in each filtered output is 
# determined by the corresponding filter's coefficient sequence (impulse response).
# Plotting the coefficients side by side explains:
#   (a) The relative magnitude of the phantom peak across filter designs.
#   (b) The temporal location of the phantom peak, which shifts with the
#       forecast horizon due to the morphing of coefficient weights.
par(mfrow = c(1, 1))
ts.plot(SSA_filt_ham_diff_x_forecast,
        col  = c("orange", "darkgreen"),xlab="Lag",ylab="",
        main = "SSA Filter Coefficients: Forecast Horizons vs. Nowcast",
        ylim = c(min(SSA_filt_ham_diff_x), max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x, col = "blue")
mtext(paste("Forecast horizon delta =", forecast_horizon[1]), col = "orange",    line = -1)
mtext(paste("Forecast horizon delta =", forecast_horizon[2]), col = "darkgreen", line = -2)
mtext("Nowcast (delta = 0)",                                   col = "blue",      line = -3)


# -----------------------------------------------------------------------------
# Interpretation: Forecast Filters Recover More Rapidly from Extreme Outliers
# -----------------------------------------------------------------------------
# The phantom post-pandemic peak is a filter artifact whose duration is governed
# by the rate at which the filter's impulse response decays to zero.
#
# Key finding — advantage of longer forecast horizons in the presence of outliers:
#   - Forecast filters (delta > 0) assign less weight to the remote past and
#     consequently decay to zero faster than the nowcast or Hamilton filter.
#   - As a result, they "forget" extreme singular observations — such as the
#     pandemic shock — more quickly, limiting the duration of filter-induced
#     distortions in the output.
#   - Specifically: the 12-month-ahead SSA forecast filter clears the pandemic
#     outlier approximately 1.5 years after its occurrence.
#   - By contrast, the SSA nowcast and Hamilton filter require an additional
#     10–11 months to fully dissipate the impulse response, sustaining the
#     phantom peak for a materially longer period.
#




# =============================================================================
# SUMMARY
# =============================================================================
#
# PROPOSED FRAMEWORK:
#   A unit-root-adjusted variant of the Hamilton filter (HF) was derived and
#   transformed into an equivalent difference-domain representation ('ham_diff').
#   This reformulation allows SSA customisation to be applied directly to the
#   stationary first-differenced series, without altering the cycle dynamics.
#   While not strictly necessary, the transformation simplifies the SSA
#   customization process. 
#
# SSA NOWCAST PERFORMANCE:
#   - SSA nowcast filters consistently outperform the Hamilton benchmark in
#     terms of smoothness: the imposed holding-time constraint reliably
#     reduces the frequency of zero-crossings in the cycle estimate.
#   - Empirical improvements are broadly commensurate with theoretical
#     specifications. Discrepancies are attributable to sample variation or 
#     model misspecification. The latter are more pronounced in long historical 
#     samples spanning multiple structural breaks.
#   - In Example 2 (monthly PAYEMS, post-1990), the additional noise reduction
#     achieved by SSA comes at no measurable cost in timeliness or accuracy 
#     relative to the Hamilton filter (HF is not on the efficient ATS frontier).
#
# SSA FORECAST PERFORMANCE:
#   - Increasing the SSA forecast horizon shifts the cycle estimate to the
#     LEFT relative to the Hamilton benchmark (improved timeliness), while
#     preserving the same smoothness constraint (identical holding time).
#   - This timeliness gain is achieved without sacrificing noise suppression:
#     the holding-time target remains unchanged across nowcast and forecast
#     variants.
#   - However, simultaneous improvement across all three dimensions — speed
#     (timeliness), smoothness, and MSE accuracy — is impossible. This
#     fundamental constraint is the FILTER DESIGN TRILEMMA:
#     => See Tutorial 0.1 for a formal exposition of the trilemma.
#   - SSA resolves the trilemma optimally: for any given combination of
#     forecast horizon and holding-time target, SSA delivers the lowest
#     achievable MSE. No other causal filter of the same length can do better.
#
# OUTLOOK:
#   - A more refined and operationally effective treatment of timeliness —
#     allowing finer control over the lead structure at zero-crossings — is
#     provided by the novel Look-Ahead DFP/PCS predictors, currently in
#     preparation as a dedicated tutorial.
# =============================================================================

###################################################################################################
















