# This tutorial is under construction

# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 8: M-SSA SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# SMOOTHING VS. PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
# M-SSA can target either acausal or causal objectives (which do not rely on 
# future unknown observations), subject to a holding-time (HT) constraint:
#
#   • Acausal target → M-SSA acts as a PREDICTOR
#                      (the target lies in the future relative to t)
#   • Causal target  → M-SSA acts as a SMOOTHER
#                      (the target lies at or before t)
#
# Previous tutorials emphasised prediction; this tutorial focuses on smoothing.

# ──────────────────────────────────────────────────────────────────────────────
# SMOOTHING TARGET
# ──────────────────────────────────────────────────────────────────────────────
# Let x_t be a stationary time series representing the data.
#   • In macroeconomic applications, x_t may correspond to the first differences
#     of a non-stationary level series I_t: x_t=I_t-I_{t-1}.
#   • First differences of economic data are typically noisy, making smoothing 
#     a compelling optimization problem.
#   • This is especially true because noise in these series obscures the 
#     underlying growth dynamics that are central to decision-making by economic 
#     actors, investors, and institutions alike.
#
# The smoothing target in this tutorial is x_{t + δ}, the value of the series
# at lag −δ relative to the current time t, where −T ≤ δ ≤ 0.
#
#   • δ = 0  (nowcast)   : M-SSA produces a real-time estimate of x_t at t = T,
#                          subject to the specified HT constraint.
#   • δ < 0  (backcast)  : M-SSA produces a retrospective estimate of x_{t+δ},
#                          exploiting observations up to t = T.
#
# Contrast with other estimation problems addressed in previous tutorials:
#   • Forecasting        : δ > 0  — the target lies in the future.
#   • Signal extraction  : the target is a filtered version of x_t
#                          (e.g., the HP trend or an ideal low-pass trend),
#                          rather than x_t itself.
#
# In principle, smoothing arises for any causal target specification in M-SSA.
#   • When the target is the causal MSE predictor, M-SSA serves as its smoother 
#     (so-called customization).
#
# Selecting the identity target x_t (or x_{t+δ} with δ ≤ 0) in this tutorial 
# isolates and reveals the INTRINSIC SMOOTHING PROPERTIES of M-SSA, unconfounded 
# by any pre-filtering introduced through an extraneous target specification 
# (e.g., HP or ideal trend).
#
# ──────────────────────────────────────────────────────────────────────────────
# M-SSA SMOOTHING CONCEPT
# ──────────────────────────────────────────────────────────────────────────────
# M-SSA maximises tracking of a target — via maximum target correlation or
# maximum sign accuracy — subject to a HT constraint that specifies the mean
# duration between consecutive mean-crossings (equivalently, zero-crossings
# for a zero-mean series).
#
# Smoothing the original series x_t via M-SSA means:
#
#   • The smoothed output tracks x_t as closely as possible,
#     subject to the HT constraint.
#
#   • If HT > HT(x_t): the smoothed series crosses its mean less frequently
#     than x_t — i.e., it is smoother than the original.
#
#   • If HT < HT(x_t): the smoothed series crosses its mean more frequently
#     than x_t — i.e., it is rougher than the original; this case is
#     generally not of practical interest.
#
# What distinguishes M-SSA smoothing from classical approaches?
#
# ──────────────────────────────────────────────────────────────────────────────
# DATA-GENERATING PROCESS
# ──────────────────────────────────────────────────────────────────────────────
# To apply the HT concept, we assume x_t to be stationary, eventually after 
# differences (for an integrated process neither the mean nor the crossings are 
# properly defined).
#
#   • M-SSA smoothing can be applied to non-stationary series, see example ???
#   • But the HT constraint must be specified in stationary differences.
#
# The Wold decomposition of the (stationary) data x_t enters M-SSA via the
# function argument ξ (the variable "xi") and plays a central role in filter 
# or smoother design. Incorporating the correct ξ into the M-SSA optimisation 
# ensures:
#
#   a) Optimality      : M-SSA maximises tracking of x_{t + delta} (in terms
#                        of target correlation or sign accuracy) for a
#                        given HT constraint.
#
#   b) Interpretability: the HT parameter is directly interpretable as the
#                        mean duration between consecutive mean-crossings
#                        of the smoothed output. 
#
# Under misspecification of ξ, both optimality and interpretability are
# compromised.
#
# Unlike classical smoothing procedures — which typically operate without an
# explicit statistical model of the data — M-SSA incorporates a parametric
# model specified through ξ (the Wold decomposition).
#
# A model-free view analogous to classical smoothing is recovered in M-SSA
# by omitting an explicit ξ specification. In this case, the implicit model
# defaults to white noise, and M-SSA retains its optimality and interpretability
# whenever the data are well-approximated by white noise — for example, when
# x_t represents the first differences of a "typical" non-stationary economic
# series (cf. Granger (1966): The Typical Spectral Shape of an Economic Variable. 
# Econometrica).
#
# If, however, the white-noise assumption is violated and ξ is left unspecified,
# increasing the HT still enforces greater smoothness, but only in a relative
# sense: the HT no longer carries a direct interpretation as the mean duration
# between consecutive mean-crossings of the output, and tracking of x_t can be 
# compromised. 
#
# ──────────────────────────────────────────────────────────────────────────────
# SMOOTHNESS VIA THE HT CONSTRAINT
# ──────────────────────────────────────────────────────────────────────────────
# The holding-time (HT) constraint governs the smoothness of the M-SSA output:
#
#   • HT > HT(x_t) : M-SSA yields a smoother estimate of x_{t+δ};
#                    mean-crossings occur less frequently than in the raw x_t.
#
#   • HT < HT(x_t) : M-SSA yields a noisier estimate of x_{t+δ};
#                    this regime is generally not of practical interest.
#
# The HT retains its natural interpretation as the mean duration between
# consecutive mean-crossings of the output, provided that the model expressed 
# in ξ is correctly specified. 
#
# ──────────────────────────────────────────────────────────────────────────────
# CENTRAL QUESTIONS
# ──────────────────────────────────────────────────────────────────────────────
# The central questions addressed in this tutorial are:
#
#   • What does a series smoothed by imposing a HT constraint look like,
#     and how does it compare visually to classically smoothed series?
#
#   • Does HT-based smoothing differ from classical smoothing approaches?
#
#   • If yes, what are the main differences? Specifically:
#
#       – Which smoothness concept is more natural and interpretable
#         in the context of sign-based decision-making?
#
#       – Which approach avoids imposing its own structural signature
#         on the data — i.e., which one is free of spurious smoothing
#         artefacts introduced by the filter itself rather than by the
#         underlying signal?
#
# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK IN THIS TUTORIAL (8)
# ══════════════════════════════════════════════════════════════════════════════
# Smoothing refers to a data-processing technique that removes noise or
# short-term fluctuations from a series, typically via an optimisation approach
# that balances fidelity to the observed data against smoothness of the output
# (e.g., by penalising second-order differences).
#
# Classical Whittaker–Henderson (WH) graduation is based on the following
# optimisation problem:
#
#   min_{y_t} sum_{t=1}^{T} (x_t - y_t)^2
#              + lambda * sum_{t=d+1}^{T} ((1-B)^d * y_t)^2
#
# For d = 2, this yields the Hodrick–Prescott (HP) filter.
# Throughout Tutorial 8, M-SSA is benchmarked against WH/HP (d = 2).
#
# While many different smoothing approaches exist, a common feature is that they
# generally do not appeal to an explicit model of the data-generating process —
# smoothness is instead enforced through a penalty term or a structural
# constraint imposed directly on the output.
#
# ── Key question ──────────────────────────────────────────────────────────────
# Does imposing such an extraneous constraint — one that is not derived from
# the data-generating process itself — introduce undesirable artefacts or 
# spurious structure into the smoothed output?
#
# Alternative smoothing approaches — polynomial splines, exponential smoothing,
# LOESS, moving averages, or more modern methods such as P-splines, GAMs,
# Kriging, total variation, or trend filtering — either assume an idealised
# functional shape or impose a particular (implicit) model structure, both of which
# imprint extraneous structure on the output that may conflict with the
# data-generating process.
#
# M-SSA, by contrast, is amorphous: smoothing is achieved by optimally tracking
# x_{t+delta} subject to a constraint on the lag-1 autocorrelation (equivalently,
# the HT). A large lag-1 ACF enforces smoothness through memory alone — it does
# not prescribe any particular shape (linear, polynomial, or cyclical) for the
# smoothed series, nor does it impose any structure on higher-order dependence
# (ACF at lags greater than one). Smoothness and the data-generating process
# are therefore not in conflict; zero-crossings are derived from optimal tracking,
# ensuring both logical consistency and statistical efficiency.
#
# ──────────────────────────────────────────────────────────────────────────────
# Classical Smoothing: Whittaker–Henderson (WH) Graduation / HP Filter
# ──────────────────────────────────────────────────────────────────────────────
#   • The HP filter is the solution to the WH optimisation problem (WH graduation
#     of order two), where the penalty term targets squared second-order 
#     differences (d=2).
#       → Smoothness criterion: CURVATURE
#   • HP maximizes tracking of x_t subject to a curvature constraint.
#
# ──────────────────────────────────────────────────────────────────────────────
# Smoothing: M-SSA vs. HP
# ──────────────────────────────────────────────────────────────────────────────
#
#   • (M-)SSA controls smoothness in x_t via the holding-time (HT) constraint
#     rather than curvature.
#
#   • When ξ is correctly specified, the HT carries a direct interpretation as
#     the mean duration between consecutive passages of the output above or
#     below its mean — equivalently, between consecutive sign changes when
#     the series is zero-mean.
#
#   • When x_t = (1−B)I_t (first differences of a non-stationary level series),
#     controlling the HT of x_t via M-SSA directly addresses the frequency of
#     transitions between above- and below-average growth in I_t. If the mean of 
#     x_t vanishes (or is small/negligible), M-SSA smoothing addresses turning 
#     points of I_t.
#
#   • Differences of typical economic series are close to white noise, the 
#     default assumption in M-SSA when ξ is not explicitly specified. 
#       
# ══════════════════════════════════════════════════════════════════════════════
# REFERENCES
# ──────────────────────
#
# Wildi, M. (2024). Business Cycle Analysis and Zero-Crossings of Time Series:
#    A Generalized Forecast Approach.  Journal of Business Cycle Research,
#   https://doi.org/10.1007/s41549-024-00097-5

# Wildi, M. (2026a). Sign Accuracy, Mean-Squared Error and the Rate of Zero Crossings:
#     a Generalized Forecast Approach, https://doi.org/10.48550/arXiv.2601.06547
#
# Wildi, M. (2026b). The Accuracy-Smoothness Dilemma in Prediction:
#   A Novel Multivariate M-SSA Forecast Approach.
#   Journal of Time Series Analysis, http://doi.org/10.1111/jtsa.70058 
#   arXiv: https://doi.org/10.48550/arXiv.2602.13722
#
# Parts in this  tutorial are  based on Wildi (2026b), Section 4.2. Additional 
#   applications are given in Wildi (2024). Extensions to non-stationary series (I-SSA) 
#   are presented in Wildi (2026a).

# ══════════════════════════════════════════════════════════════════════════════


# Clear the workspace to ensure a clean environment
rm(list = ls())


# LOAD REQUIRED LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# HP filter and other standard time-series filters
library(mFilter)


# LOAD CUSTOM M-SSA FUNCTION LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# Core M-SSA filter construction and optimisation routines
source(paste(getwd(), "/R/functions_MSSA.r", sep = ""))

# HP-filter utilities used in the JBCY paper (depends on mFilter)
source(paste(getwd(), "/R/HP_JBCY_functions.r", sep = ""))

# M-SSA utility functions: data preparation, plotting helpers, and wrappers
source(paste(getwd(), "/R/M_SSA_utility_functions.r", sep = ""))


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: UNIVARIATE SYMMETRIC SSA SMOOTHERS
# GOAL
# ════
# Benchmark (univariate) SSA smoothing against Whittaker–Henderson (WH) graduation,
# which generalises the HP filter (assuming second-order differences, d = 2).
#
# BACKGROUND  
# ═══════════
# Wildi 2026b, Section 4.2

# COMPARISON STRATEGY 
# ════════════════════════════════════
# Two complementary SSA designs are evaluated against WH (or HP):
#
#   Exercise 1.1 — Match holding time of HP:
#     Design SSA  so that its HT equals that of HP.
#     Then compare the MSE tracking and curvature of both smoothers.
#     Expected result: SSA  achieves better MSE tracking (closer to x_t)
#     for the same HT but larger curvature.

#   Exercise 1.2 — Match MSE tracking of x_t:
#     Design SSA  so that its MSE tracking of x_t equals that of HP.
#     Then compare the HT and curvature of both smoothers.
#     Expected result:  SSA achieves a longer HT (less zero-crossings)
#     for the same tracking accuracy, but larger curvature.



# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1.1: SSA vs. HP — Match Holding TimeS
# ══════════════════════════════════════════════════════════════════════════════
# Design SSA with the same HT as the symmetric HP filter.

# ─────────────────────────────────────────────────────────────────────────────
# 1.1.1  Specify the HP Filter
# ─────────────────────────────────────────────────────────────────────────────

lambda_HP <- 14400   # Monthly HP smoothing parameter (standard business-cycle value)
L         <- 201     # Filter length (number of coefficients)

# Compute the bi-infinite HP target filter and its MSE-optimal finite approximation
HP_obj    <- HP_target_mse_modified_gap(L, lambda_HP)
hp_target <- HP_obj$target    # Bi-infinite HP filter coefficients (length L)
hp_trend  <- HP_obj$hp_trend  # Associated HP trend estimate

par(mfrow=c(1,1))
ts.plot(hp_target,main="Target Smoother: Two-Sided HP")

# ─────────────────────────────────────────────────────────────────────────────
# 1.1.2  Specify SSA Design Settings
# ─────────────────────────────────────────────────────────────────────────────

# Sigma: variance–covariance matrix of the multivariate innovation process.
#        NULL → univariate design; 
Sigma <- NULL

# xi: spectral density of the input process.
#     NULL → white-noise assumption (xi = Identity).
xi <- NULL

# symmetric_target: FALSE → causal (one-sided) target filter.
symmetric_target <- FALSE

# --- Target filter ---
# gamma_target = 1 → allpass (identity) target: SSA tracks the data x_t itself.
# This contrasts with signal-extraction or nowcasting settings where the
# target is a non-trivial filtered version of x_t (e.g., the HP trend or an ideal trend).
# MSSA_func zero-pads gamma_target to length L automatically if needed.
gamma_target <- 1

# --- Lag parameter delta ---
# The SSA filter targets x_{t + delta}, i.e., it tracks the series value
# delta steps ahead of (or behind) the current observation t.
#   delta < 0  → smoother  (target lies in the past; causal design)
#   delta = 0  → nowcaster (real-time estimate of x_t)
#   delta > 0  → predictor (target lies in the future)
#
# Setting delta = -(L-1)/2 places the target at the centre of the filter
# support, yielding a symmetric (two-sided) smoothing filter — directly
# comparable to the symmetric HP filter used as benchmark.
delta <- -(L - 1) / 2

# --- Bisection optimiser resolution ---
# The optimiser halves the search interval at each step, so effective
# resolution scales as 2^split_grid (much faster than brute-force search).
#   • split_grid = 20 provides a good balance of speed and precision.
#   • Increase if convergence diagnostics indicate insufficient resolution.
split_grid <- 20


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.3  Specify the HT Constraint — Match HP 
# ─────────────────────────────────────────────────────────────────────────────
# Impose the same HT as the two-sided HP filter.
# Both filters are symmetric (delta = -(L-1)/2), so a like-for-like
# comparison is valid.
# Under an identical HT constraint, SSA is guaranteed to outperform HP 
# in target correlation and hence sign accuracy / MSE (as applied to noise).

# Extract the HT and the corresponding rho from the bi-infinite HP filter
rho1 <- compute_holding_time_func(hp_target)$rho_ff1  # rho implied by HP's HT
ht1  <- compute_holding_time_func(hp_target)$ht       # Holding time of HP filter

# Interpretation of ht1:
#   The symmetric HP filter applied to white noise produces sign changes with
#   a mean inter-crossing duration of approximately 60 time steps (see performances below).
#   Imposing this HT on SSA ensures both smoothers operate at the same
#   level of smoothness in terms of zero-crossing rate; any gain in tracking accuracy 
#   then reflects the optimality of SSA within the HT-constrained class.

ht1   # Inspect the HP holding time


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.4  Compute the SSA Filter
# ─────────────────────────────────────────────────────────────────────────────
# Design the SSA filter subject to the HP-matched HT constraint.
# Note: xi = NULL assumes white-noise input; MSSA_func will issue a warning
#       confirming that gamma_target has been zero-padded to length L.

SSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1)

# Extract the optimised (univariate) filter coefficients
bk_mat <- SSA_obj$bk_mat

# --- Visual inspection ---
# The filter should be symmetric, consistent with delta = -(L-1)/2
par(mfrow = c(1, 1))
ts.plot(bk_mat, main = "SSA Smoother (Matches HT of HP)")

# --- Optimisation diagnostics ---
# crit_rhoy_target: maximised target correlation achieved by SSA.
#     This should exceed the HP target correlation (see performances below).
# Interpretation: the correlation between SSA and x_{t+delta} is ~0.23 
SSA_obj$crit_rhoy_target

# Collect both filters into filter_mat 
filter_mat<-cbind(hp_target,bk_mat)
colnames(filter_mat)<-c("HP","SSA")


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.5  Results: Tracking Accuracy on Simulated White Noise
# ─────────────────────────────────────────────────────────────────────────────
# Performance is evaluated on three complementary criteria:
#   a) Mean Squared Error (MSE)
#   b) Target correlation
#   c) Sign accuracy (proportion of correct sign agreements)
# ─────────────────────────────────────────────────────────────────────────────

# --- Simulate a long white-noise series ---
# This ensures alignment of sample performances and expected values 
lenq <- 100000
set.seed(86)
x <- rnorm(lenq)

# --- Apply both symmetric (two-sided) filters to x ---
# sides = 2 specifies an acausal two-sided (symmetric) convolution,
# consistent with the symmetric filter design (delta = -(L-1)/2).
y_ssa    <- filter(x, bk_mat,    sides = 2)  # SSA smoother output
y_hp_two <- filter(x, hp_target, sides = 2)  # HP smoother output

# Target: since both filters are symmetric and acausal, the target is
# the unshifted series x (no lead or lag adjustment required because the 
# smoothers are forward-shifted by abs(delta)).
target <- x


# ── a. MSE ───────────────────────────────────────────────────────────────────
# Compute MSE of each smoother relative to the target series.

mse_ssa_smooth <- mean((target - y_ssa)^2,    na.rm = TRUE)
mse_hp_smooth  <- mean((target - y_hp_two)^2, na.rm = TRUE)

# Expected result: SSA achieves a smaller MSE than HP.
mse_ssa_smooth   # SSA MSE
mse_hp_smooth    # HP MSE


# ── b. Target Correlation ─────────────────────────────────────────────────────
# Compute the correlation matrix of the target and both filter outputs.

output_mat <- na.exclude(cbind(target, y_ssa, y_hp_two))
colnames(output_mat)<-c("x","SSA","HP")
cor_mat    <- cor(output_mat)
cor_mat    # Full correlation matrix (first row: correlations with target)

# Compare SSA and HP target correlations:
# Expected result: SSA correlates more strongly with the target than HP.
cor_mat[1, 2]   # SSA target correlation
cor_mat[1, 3]   # HP target correlation

# --- Internal convergence check ---
# The sample correlation cor(x, y_ssa) should equal crit_rhoy_target
# (up to Monte Carlo sampling error), confirming correct optimisation.
cor_mat[1, 2]             # Empirical correlation: target vs. SSA output
SSA_obj$crit_rhoy_target  # Optimised objective (true target correlation)
# The sample correlation should asymptotically converge to the true target correlation if the model is correctly specified (here: white noise)
# Hence maximizing the true target correlation within SSA is a meaningful criterion

# ── c. Sign Accuracy ──────────────────────────────────────────────────────────
# Sign accuracy: proportion of time steps where the filter output and the
# target share the same sign. A higher value indicates fewer false alarms.
# Expected result: SSA achieves higher sign accuracy than HP.

sign_acc_ssa <- sum((target * y_ssa)    > 0, na.rm = TRUE) /
  length(na.exclude(target * y_ssa))

sign_acc_hp  <- sum((target * y_hp_two) > 0, na.rm = TRUE) /
  length(na.exclude(target * y_hp_two))

sign_acc_ssa   # SSA sign accuracy
sign_acc_hp    # HP sign accuracy


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.6  Results: Smoothness
# ─────────────────────────────────────────────────────────────────────────────
# Smoothness is evaluated on two complementary criteria:
#   a) Holding time (HT): mean duration between consecutive sign changes.
#   b) Curvature: root mean squared second-order differences (RMSD2).
# ─────────────────────────────────────────────────────────────────────────────

# ── a. Holding Time ───────────────────────────────────────────────────────────
# The HT constraint imposed on SSA is set equal to that of the HP filter.
# The empirical HTs of both smoothers should therefore agree closely.

ht1                              # Target HT imposed on SSA (= HP holding time)
compute_empirical_ht_func(y_ssa)      # Empirical HT of SSA output
compute_empirical_ht_func(y_hp_two)   # Empirical HT of HP output

# Expected result: both empirical HTs match each other, i.e., SSA matches the target ht1,
# confirming that the HT constraint is binding and correctly enforced by the optimiser.


# ── b. Curvature (Root Mean Squared Second-Order Differences) ─────────────────
# Curvature is measured as the root mean squared second-order difference
# (RMSD2) of each smoother's output — the natural smoothness criterion
# minimised by HP under the WH framework.
# Expected result: HP achieves the smallest RMSD2 by construction (WH
# optimality), while SSA incurs a larger curvature as the cost of
# superior tracking accuracy under the same HT constraint.

sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif

# Note: the first element of sq_se_dif corresponds to the raw target x_t
# (standardised white noise). Applying the second-order difference operator
# (1 - B)^2 to white noise gives:
#
#   (1 - B)^2 x_t  =  x_t - 2 x_{t-1} + x_{t-2}
#
# The variance of this expression is:
#
#   Var[(1-B)^2 x_t]  =  1^2 + (-2)^2 + 1^2  =  1 + 4 + 1  =  6
#
# so the expected RMSD2 of the raw target is sqrt(6) ≈ 2.449.
# The first element of sq_se_dif converges asymptotically to this theoretical value,
# providing a useful sanity check on the curvature calculations.



# ─────────────────────────────────────────────────────────────────────────────
# 1.1.7  Plot Series: 1,000-observation window
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 1))
colo <- c("blue", "violet")

ts.plot(na.exclude(output_mat)[1000:2000, 2:3], col = colo)
abline(h = 0)
# Add colour-coded series labels as margin text
for (i in 1:ncol(output_mat[ , 2:3]))
  mtext(colnames(output_mat[ , 2:3])[i], col = colo[i], line = -i)

# HP (violet) appears visually "rounder" — i.e., it exhibits lower curvature
# and longer monotonicity intervals. Before investigating this stylised fact
# further, we compute additional performance measures.

# ─────────────────────────────────────────────────────────────────────────────
# 1.1.8  Distribution of Zero-Crossings: Mean and Standard Deviation
# ─────────────────────────────────────────────────────────────────────────────

# Identify time points at which the SSA output changes sign
ssa_zc <- which(output_mat[1:(nrow(output_mat) - 1), "SSA"] *
                  output_mat[2:(nrow(output_mat)),     "SSA"] < 0)

# Identify time points at which the HP output changes sign
hp_zc  <- which(output_mat[1:(nrow(output_mat) - 1), "HP"] *
                  output_mat[2:(nrow(output_mat)),      "HP"] < 0)

# ── Mean durations between consecutive zero-crossings (empirical HTs) ────────
mean(diff(ssa_zc))   # SSA empirical HT
mean(diff(hp_zc))    # HP  empirical HT
# Both means should match the imposed HT target: ht1
ht1

# ── Standard deviations of inter-crossing durations ──────────────────────────
sd(diff(ssa_zc), na.rm = TRUE)   # SSA: variability in zero-crossing spacing
sd(diff(hp_zc),  na.rm = TRUE)   # HP:  variability in zero-crossing spacing

# SSA exhibits a larger standard deviation than HP, indicating that its
# zero-crossings are less regularly spaced — a pattern also visible in the
# time-series plot above.
#
# Key question: is a smaller standard deviation (i.e., more regularly spaced
# zero-crossings, as produced by HP) actually desirable?
#
#   • More regular zero-crossings imply a more periodic, clock-like output —
#     a structural regularity not present in the original data x_t, noise, 
#     which is unstructured, irregular and aperiodic by nature.
#   • This regularity is a signature imposed by the HP filter itself, not a
#     feature of the underlying signal: it constitutes spurious smoothing.
#   • SSA, by contrast, inherits the irregular spacing of zero-crossings
#     from the data, producing a more faithful representation of the
#     underlying growth dynamics, see below.
#
#   Consider a business-cycle context, such as tracking recessions:
#
#   • US recessions are irregularly spaced: the double-dip recessions of the
#     early 1980s were separated by roughly one year, whereas the subsequent
#     expansion during the Great Moderation (1990s) lasted approximately a
#     decade.
#   • Imposing a quasi-regular cyclical pattern — as HP's curvature constraint
#     tends to do — conflicts with the inherent irregularity of business cycles.
#   • In such a context, M-SSA imprints less extraneous structure on the output:
#     it accommodates large gaps between consecutive TPs whenever the underlying
#     growth dynamics (which M-SSA tracks optimally) do not indicate an
#     imminent reversal.
#
# ─────────────────────────────────────────────────────────────────────────────
# 1.1.9 Turning Points in SSA
# ─────────────────────────────────────────────────────────────────────────────
# Identify turning points (local maxima/minima) of the SSA output:
# a turning point occurs where consecutive first differences change sign
ssa_tp <- which(diff(output_mat[1:(nrow(output_mat) - 1), "SSA"]) *
                  diff(output_mat[2:(nrow(output_mat)),     "SSA"]) < 0)

# Identify turning points (local maxima/minima) of the HP output
hp_tp  <- which(diff(output_mat[1:(nrow(output_mat) - 1), "HP"]) *
                  diff(output_mat[2:(nrow(output_mat)),     "HP"]) < 0)

# ── Mean duration between consecutive turning points ──────────────────────────
# Expected result: HP exhibits a longer mean duration between turning points
# (i.e., longer monotonicity intervals) than SSA, reflecting its lower
# curvature — consistent with HP appearing visually "rounder".
nrow(output_mat) / length(ssa_tp)   # SSA: mean monotonicity interval length
nrow(output_mat) / length(hp_tp)    # HP:  mean monotonicity interval length


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.10  Dependence Structure: Autocorrelation Functions
# ─────────────────────────────────────────────────────────────────────────────
# The autocorrelation function (ACF) provides an alternative perspective on
# the structural differences between the SSA and HP smoothing outputs.
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(2, 1))

# SSA output: slowly and monotonically decaying ACF, reflecting long memory
# and an acyclical (non-oscillatory) dependence structure.
acf(na.exclude(output_mat)[, 2], lag.max = 100, main = "SSA")

# HP output: ACF decays more rapidly and exhibits a cyclical (oscillatory)
# pattern. The half-period of the oscillation (≈ 57 lags) is consistent
# with the imposed HP holding time.
acf(na.exclude(output_mat)[, 3], lag.max = 100, main = "HP")

# ── Key structural differences between SSA and HP ACFs ─────────────────────
#
#   a) Shape : SSA produces an acyclical, monotonically decaying ACF,
#              whereas HP exhibits a cyclical (oscillatory) ACF pattern.
#              This explains the visual impression of HP being "rounder":
#              its oscillatory ACF imposes a pseudo-periodic structure on
#              the smoothed output that is absent from the original data.
#
#   b) Memory: the SSA ACF decays more slowly than that of HP, indicating
#              stronger and more persistent serial dependence in the SSA
#              smoothed output.
#
# ── Technical note: HT and the first-order ACF ───────────────────────────────
# The HT is linked bijectively to the lag-1 autocorrelation. Since SSA and
# HP are designed to achieve the same HT, their lag-1 ACF values must be
# identical — a consistency check visible in the plots above.

# ─────────────────────────────────────────────────────────────────────────────
# Discussion
# ─────────────────────────────────────────────────────────────────────────────
# For an identical rate of zero-crossings, the following differences emerge:
#
#   • Tracking performance: SSA outperforms HP — it achieves larger target
#     correlation and larger sign accuracy.
#
#   • Curvature: HP exhibits smaller curvature, manifest as:
#       – a visually rounder plot,
#       – fewer turning points, and
#       – more evenly spaced zero-crossings.
#
# ── Interpretation via the Level Series I_t ───────────────────────────────────
# Let x_t = I_t − I_{t−1}, where I_t is the relevant non-stationary level series.
#
#   • Zero-crossings of x_t correspond to turning points (TPs) in I_t.
#   • M-SSA and HP produce different TP datings for I_t.
#   • Key question: which TP dating is more compatible with the
#     data-generating process — and therefore more intrinsic?
#
# ── Data-Driven vs. Externally Imposed Structure ──────────────────────────────
#   • SSA derives TP datings in I_t through optimal tracking of the first
#     differences x_t. Turning points are determined by systematic changes in
#     growth — they emerge intrinsically from the data, based on an optimal 
#     (smooth) local growth tracker.
#
#   • HP derives TP datings by imposing minimal curvature on the output. The
#     resulting turning points reflect a structural shape (low curvature) that
#     is not an inherent property of the data, but is imposed externally —
#     a form of spurious smoothing. This structure is consistent with certain
#     data-generating processes (e.g., HP is the optimal trend estimator under
#     the ARIMA(0,2,2) model; see Tutorial 2), but it is unclear how closely
#     typical economic data conform to such models. By contrast, white noise
#     is a well-established proxy for the first differences of typical economic
#     series — and first differences directly capture the growth dynamics that
#     are of primary practical relevance.
#
# ── Interpretability ──────────────────────────────────────────────────────────
#   • SSA: TP datings of I_t follow directly from optimal tracking of growth
#     in x_t — an intuitively natural and operationally meaningful criterion.
#
#   • HP: TP datings follow from controlling curvature in x_t, which corresponds
#     to penalising third-order differences (1−B)³I_t — a quantity that is
#     difficult to interpret in economic or practical terms. The interpretive
#     difficulty extends to the TPs themselves: what predictive or informational
#     content can be assigned to a turning point identified by a smoother whose
#     curvature has been minimised by construction, rather than by reference to
#     the data? For M-SSA, the answer is clear: each TP follows directly from
#     optimal tracking of growth in x_t — a criterion that is both
#     operationally meaningful and data-driven.
#
# ── Penalty Term: Imposed Structure vs. Data-Driven Behaviour ─────────────────
#   • HP penalty: forces the smoother towards a straight line — the idealised
#     series in which curvature vanishes entirely. This imposes a strong
#     structural shape on the output, regardless of the data.
#
#   • SSA penalty: forces the smoother towards a series with a lag-1
#     autocorrelation of 1 (HT and lag-1 ACF are linked bijectively). This is a 
#     far weaker structural constraint than a straight line, allowing the output 
#     to remain more faithful to the data-generating process even when imposing 
#     strong regularization (large HT).
#
# ── Cosmetics and Apparent Advantages of HP ───────────────────────────────────
#   • The rounder shape of HP is visually appealing.
#   • HP has relatively few turning points (TPs) between consecutive
#     zero-crossings — often only a single TP separates two crossings.
#   • Consequently, each TP in HP serves as a reliable early warning of an
#     imminent zero-crossing (direction reversal in I_t).
#   • By contrast, M-SSA exhibits many TPs between consecutive zero-crossings,
#     generating frequent false alarms: a TP in M-SSA is therefore a less
#     reliable signal of an impending zero-crossing (TP in I_t). This reflects 
#     the absence of extraneous structure imposed on the data.
#
# ── Qualification ─────────────────────────────────────────────────────────────
# While the above observations concerning predictability of TPs for HP are 
# accurate, it is not clear that they constitute a genuine advantage for HP. 
# The apparent anticipativity of HP's
# TPs — i.e., their tendency to reliably precede zero-crossings — is largely
# an artefact of the curvature constraint: HP's TPs are, to a significant
# degree, artificially generated by the externally imposed low-curvature
# structure, rather than reflecting data-immanent growth dynamics. This shows-up 
# in a smaller standard deviation, signifying a more regular distribution over 
# time. Their seemingly superior predictive value is therefore mainly due to a 
# structural feature of the filter (spurious), not a signal extracted from the 
# data.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways from Exercise 1.1
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Optimality of SSA Smoothing
#    For a given HT constraint, SSA maximises target correlation and sign
#    accuracy with respect to the target x_{t+δ} — by construction.
#
# 2. Dominance over HP (WH) smoothing
#    Imposing the HP (WH) holding time as the HT constraint guarantees that SSA
#    outperforms HP (WH) in tracking x_{t+δ}, for an identical zero-crossing rate.
#
# 3. Cost of Superior Tracking: Increased Curvature
#    The gain in tracking accuracy of SSA comes at the cost of greater
#    curvature relative to HP. As a consequence, HP outputs:
#      – appear visually rounder,
#      – exhibit fewer turning points, and
#      – have more regularly (evenly) spaced zero-crossings, and
#      – display a faster-decaying, damped oscillatory ACF structure.
#
# 4. SSA Imposes Less External Structure on the Data
#    SSA smoothing is more faithful to the data-generating process than HP:
#      – Zero-crossings emerge from optimal tracking of the data (purely
#        data-driven), rather than from an externally imposed shape constraint.
#      – The HT constraint affects only the lag-1 ACF; it does not prescribe
#        or prefer any particular shape for the smoothed series.
#      – The resulting ACF is monotonically decreasing, free of the artificial
#        oscillatory structure introduced by HP's curvature penalty.
#
# 5. Generalisation to Autocorrelated Processes
#    All of the above results extend from white noise to arbitrary stationary
#    autocorrelated processes by supplying the appropriate ξ (Wold decomposition) 
#    to the SSA design criterion. Non-stationary integrated processes are 
#    discussed in example ???
#
# ─────────────────────────────────────────────────────────────────────────────



# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1.2: SSA vs. HP — Matching Target Correlation
# ══════════════════════════════════════════════════════════════════════════════
# Here we invert the problem formulation of Exercise 1.1: rather than fixing
# the HT and comparing tracking accuracy, we fix the tracking accuracy
# (target correlation) and compare the resulting HT.
#
# Expected result: for equal tracking accuracy, SSA achieves a larger HT
# than HP — i.e., it produces a smoother output (fewer mean-crossings) for
# the same degree of fidelity to the target x_{t+δ}.
#
# This is the dual of the result established in Exercise 1.1:
#   Exercise 1.1 (primal) : fix HT            → SSA maximises target correlation.
#   Exercise 1.2 (dual)   : fix target corr.  → SSA maximises HT.
#
# Together, these results confirm that SSA defines the efficient frontier of
# the smoothness–accuracy trade-off: no linear smoother or predictor can
# achieve a larger HT for a given level of tracking accuracy, or equivalently,
# a higher tracking accuracy for a given HT — provided that the model
# ξ (Wold decomposition) is correctly specified.
# ─────────────────────────────────────────────────────────────────────────────
# 1.2.1  Compute SSA with Correlation-Matched HT
# ─────────────────────────────────────────────────────────────────────────────
# Goal: find the HT such that SSA achieves the same target correlation as HP.
# The value ht1_1 = 75 was determined empirically (see calibration below).

ht1_1  <- 75                              # HT chosen to match HP's target correlation
rho1_1 <- compute_rho_from_ht(ht1_1)$rho # Corresponding lag-1 ACF parameter

# Design the SSA filter under the correlation-matched HT constraint
SSA_obj_1 <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1_1)

# Extract filter coefficients
bk_mat_1 <- SSA_obj_1$bk_mat
par(mfrow=c(1,1))
ts.plot(bk_mat_1,main=paste("SSA smoother: HT=",round(ht1_1),sep=""))

# ── Verify that the SSA target correlation matches that of HP ───────────────
# If the values below do not match, increase or decrease ht1_1 manually until 
# the target correlations agree.
SSA_obj_1$crit_rhoy_target   # SSA population target correlation
cor(output_mat)[1, 3]        # HP empirical target correlation (sample estimate)

# ── Conclusion ────────────────────────────────────────────────────────────────
# Since both target correlations match, we conclude that SSA generates a
# larger HT (ht1_1 = 75) than HP (ht1) to achieve the same tracking accuracy —
# confirming the dual result of Exercise 1.2.
ht1   # HP holding time (for comparison)

# ─────────────────────────────────────────────────────────────────────────────
# 1.2.2  Empirical Tracking Performances: 
# ─────────────────────────────────────────────────────────────────────────────
# Performance is evaluated on three complementary criteria:
#   a) Mean Squared Error (MSE)
#   b) Target correlation
#   c) Sign accuracy (proportion of correct sign agreements)
# ─────────────────────────────────────────────────────────────────────────────

# --- Simulate a long white-noise series ---
# This ensures alignment of sample performances and expected values 
lenq <- 1000000
set.seed(861)
x <- rnorm(lenq)

# --- Apply both symmetric (two-sided) filters to x ---
# sides = 2 specifies an acausal two-sided (symmetric) convolution,
# consistent with the symmetric filter design (delta = -(L-1)/2).
y_ssa_1    <- filter(x, bk_mat_1,    sides = 2)  # SSA smoother output
y_hp_two <- filter(x, hp_target, sides = 2)  # HP smoother output

# Target: since both filters are symmetric and acausal, the target is
# the unshifted series x (no lead or lag adjustment required because the 
# smoothers are forward-shifted by abs(delta)).
target <- x


# ── a. MSE ───────────────────────────────────────────────────────────────────
# Compute MSE of each smoother relative to the target series.

mse_ssa_smooth <- mean((target - y_ssa_1)^2,    na.rm = TRUE)
mse_hp_smooth  <- mean((target - y_hp_two)^2, na.rm = TRUE)

# Expected result: Both numbers should match
mse_ssa_smooth   # SSA MSE
mse_hp_smooth    # HP MSE


# ── b. Target Correlation ─────────────────────────────────────────────────────
# Compute the correlation matrix of the target and both filter outputs.

output_mat <- na.exclude(cbind(target, y_ssa_1, y_hp_two))
colnames(output_mat)<-c("x","SSA","HP")
cor_mat    <- cor(output_mat)
cor_mat    # Full correlation matrix (first row: correlations with target)

# Compare SSA and HP target correlations:
# Expected result: SSA and HP match.
cor_mat[1, 2]   # SSA target correlation
cor_mat[1, 3]   # HP target correlation

# --- Internal convergence check ---
# The sample correlation cor(x, y_ssa_1) should equal crit_rhoy_target
# (up to Monte Carlo sampling error), confirming correct optimisation.
cor_mat[1, 2]             # Empirical correlation: target vs. SSA output
SSA_obj_1$crit_rhoy_target  # Optimised objective (true target correlation)
# The sample correlation should asymptotically converge to the true target correlation if the model is correctly specified (here: white noise)
# Hence maximizing the true target correlation within SSA is a meaningful criterion

# ── c. Sign Accuracy ──────────────────────────────────────────────────────────
# Sign accuracy: proportion of time steps where the filter output and the
# target share the same sign. A higher value indicates fewer false alarms.
# Expected result: SSA and HP match.

sign_acc_ssa <- sum((target * y_ssa_1)    > 0, na.rm = TRUE) /
  length(na.exclude(target * y_ssa_1))

sign_acc_hp  <- sum((target * y_hp_two) > 0, na.rm = TRUE) /
  length(na.exclude(target * y_hp_two))

sign_acc_ssa   # SSA sign accuracy
sign_acc_hp    # HP sign accuracy


# ─────────────────────────────────────────────────────────────────────────────
# 1.2.3  Results: Smoothness
# ─────────────────────────────────────────────────────────────────────────────
# Smoothness is evaluated on two complementary criteria:
#   a) Holding time (HT): mean duration between consecutive sign changes.
#   b) Curvature: root mean squared second-order differences (RMSD2).
# ─────────────────────────────────────────────────────────────────────────────

# ── a. Holding Time ───────────────────────────────────────────────────────────
# The HT constraint imposed on SSA is larger than HP.

ht1_1                              # Target HT imposed on SSA (= HP holding time)
compute_empirical_ht_func(y_ssa_1)      # Empirical HT of SSA output
compute_empirical_ht_func(y_hp_two)   # Empirical HT of HP output

# SSA matches the target ht1_1, confirming that the HT constraint is binding and 
# correctly enforced by the optimiser. SSA generates less zero-crossings than HP


# ── b. Curvature (Root Mean Squared Second-Order Differences) ─────────────────

sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif

# Despite stronger smoothness (larger HT), SSA's curvature is larger than HP

# ─────────────────────────────────────────────────────────────────────────────
# 1.2.4  Plot Series: 1,000-observation window
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 1))
colo <- c("blue", "violet")

ts.plot(na.exclude(output_mat)[1000:2000, 2:3], col = colo)
abline(h = 0)
# Add colour-coded series labels as margin text
for (i in 1:ncol(output_mat[ , 2:3]))
  mtext(colnames(output_mat[ , 2:3])[i], col = colo[i], line = -i)

# HP (violet) appears visually "rounder" — i.e., it exhibits lower curvature
# and longer monotonicity intervals. 

# ─────────────────────────────────────────────────────────────────────────────
# 1.2.5  Distribution of Zero-Crossings: Mean and Standard Deviation
# ─────────────────────────────────────────────────────────────────────────────
# This is skipped: comparing standard deviations is not informative when 
# the means differ.

# ─────────────────────────────────────────────────────────────────────────────
# 1.2.6 Turning Points in SSA
# ─────────────────────────────────────────────────────────────────────────────
# Identify turning points (local maxima/minima) of the SSA output:
# a turning point occurs where consecutive first differences change sign
ssa_tp <- which(diff(output_mat[1:(nrow(output_mat) - 1), "SSA"]) *
                  diff(output_mat[2:(nrow(output_mat)),     "SSA"]) < 0)

# Identify turning points (local maxima/minima) of the HP output
hp_tp  <- which(diff(output_mat[1:(nrow(output_mat) - 1), "HP"]) *
                  diff(output_mat[2:(nrow(output_mat)),     "HP"]) < 0)

# ── Mean duration between consecutive turning points ──────────────────────────
# Lower curvature of HP implies fewer TPs
nrow(output_mat) / length(ssa_tp)   # SSA: mean monotonicity interval length
nrow(output_mat) / length(hp_tp)    # HP:  mean monotonicity interval length

# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways from Exercise 1.2
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Dual Result Verified
#    For equal tracking accuracy (matched target correlation), SSA achieves
#    a larger HT than HP — i.e., it produces fewer zero-crossings for the same
#    degree of fidelity to the target x_{t+δ}. This confirms the dual of the
#    result established in Exercise 1.1.
#
# 2. HP Retains Its Curvature Advantage
#    Despite its smaller HT, HP still exhibits lower curvature than SSA —
#    producing a visually rounder output with fewer turning points and more
#    regularly spaced zero-crossings.
#
# 3. Conclusions from Exercise 1.1 Carry Over
#    Enforcing smoothness through curvature (as HP does) imposes extraneous
#    structure on the data that need not reflect the underlying data-generating
#    process. The resulting turning points are, to a significant degree,
#    artefacts of the curvature penalty rather than genuine features of the
#    signal — a form of spurious smoothing.
#
# ─────────────────────────────────────────────────────────────────────────────




# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1.3: Replicating HP Smoothing via SSA Smoothing
# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 2 demonstrated that SSA can replicate the HP filter by specifying
# the target as the HP-filtered series. In the present tutorial, however, the
# target is the identity — i.e., the raw series x_t itself — since the focus
# here is on the native smoothing properties of SSA and the HT constraint,
# unconfounded by any pre-filtering of the target.

#
# The question addressed in this exercise is:
#   Can SSA smoothing replicate HP smoothing, when the target is x_t?
#   If so, under what conditions does this replication hold?

# ─────────────────────────────────────────────────────────────────────────────
# 1.3.1  SSA Design Settings
# ─────────────────────────────────────────────────────────────────────────────

Sigma            <- NULL   # Univariate design (no cross-sectional structure)
symmetric_target <- FALSE  # Causal (one-sided) filter
gamma_target     <- 1      # Identity target: SSA tracks x_t directly

# Lag parameter: delta = 0 corresponds to a nowcast (real-time estimate of x_T)
# In contrast to Exercise 1.1, which used delta = -(L-1)/2 (fully symmetric),
# we here set delta = 0.
delta <- 0   

# Numerical optimisation settings (default values)
split_grid           <- 20
with_negative_lambda <- FALSE

# HT constraint: use the rho implied by the HP holding time
rho1 <- compute_holding_time_func(hp_target)$rho_ff1

# ─────────────────────────────────────────────────────────────────────────────
# 1.3.2  Model Specification: ξ Identified with the HP Filter Weights
# ─────────────────────────────────────────────────────────────────────────────
# Key novelty relative to Exercises 1.1 and 1.2:
# Rather than assuming white noise (ξ = NULL, the default), we here supply the
# two-sided HP filter weights as the Wold decomposition ξ of the data-generating
# process. This identifies the implicit model with the weights of HP —
# an unconventional assumption, as noted below.
xi <- hp_target

# ─────────────────────────────────────────────────────────────────────────────
# 1.3.3  Compute the SSA Filter/Smoother
# ─────────────────────────────────────────────────────────────────────────────
# Supplying ξ explicitly; omitting ξ would default to the white-noise assumption.
SSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1,
                     with_negative_lambda, xi)

# ── Extract filter coefficients ───────────────────────────────────────────────
# Coefficients applied to ε_t (the innovation / noise term in the Wold decomposition)
bk_mat_eps <- SSA_obj$bk_mat

# Coefficients applied to x_t (the observed series)
bk_mat <- SSA_obj$bk_x_mat

# ── Inspect the filter applied to x_t ────────────────────────────────────────
# Expected result: an identity filter (nowcasting x_t from x_t is trivial).
# Note: minor deviations towards larger lags reflect incomplete numerical
# convergence; increasing split_grid to 30 or higher resolves this.
ts.plot(bk_mat)

# ── Inspect the filter applied to ε_t ────────────────────────────────────────
# Expected result: the filter replicates the HP smoother weights.
par(mfrow = c(1, 1))
ts.plot(cbind(bk_mat_eps, hp_target),
        main = "SSA and HP Smoother Coefficients: both overlap")

# ── Optimisation diagnostic ───────────────────────────────────────────────────
# Target correlation should equal 1, since x_t is replicated exactly.
SSA_obj$crit_rhoy_target

# ─────────────────────────────────────────────────────────────────────────────
# Answer to the Opening Question
# ─────────────────────────────────────────────────────────────────────────────
# SSA can replicate HP smoothing when the target is the identity x_t,
# provided that the Wold decomposition ξ of the data-generating process is
# identified with the HP filter weights. Under this assumption, SSA recovers
# the HP smoother exactly — confirming the replication result.
#
# This is, however, an unconventional model assumption: identifying the
# data-generating process with the HP filter weights is not a natural or
# well-motivated choice in most applications. The result is therefore of
# theoretical interest — demonstrating the flexibility of the SSA framework
# in smoothing — rather than a practically recommended specification.

# Hint: setting ξ = NULL (xi<-NULL: white-noise assumption) and re-running the 
# design yields the optimal nowcast (δ = 0) smoother of x_t (noise) subject to 
# the imposed HT constraint.
# ─────────────────────────────────────────────────────────────────────────────



# ════════════════════════════════════════════════════════════════════════════════
# Exercise 2: TARGET HP-MONOTONICITY IN SSA
# ════════════════════════════════════════════════════════════════════════════════
# This exercise is deliberately extreme and, in practice, unrealistic. It is
# designed to explore the characteristics of SSA smoothing when pushed to
# its limits — a setting that, while practically irrelevant, is instructive
# for understanding the fundamentals of the SSA smoothing by probing its
# boundaries. 

# ─────────────────────────────────────────────────────────────────────────────
# Motivation
# ─────────────────────────────────────────────────────────────────────────────
# Exercise 1.1 showed that SSA exhibits substantially more turning points
# (TPs) than HP for an equal HT, since HP minimises curvature and thereby
# suppresses TPs. 
# 
# Problem formulation: design an SSA smoother that replicates not the HT of
# HP, but rather its rate of turning points (TPs) — a strictly stronger
# smoothness requirement, since matching the TP rate of HP demands a
# substantially larger HT in SSA. Since HP evolves monotonically between TPs
# this problem equates to replicating HP-monotonicity by the SSA-smoother.

# ─────────────────────────────────────────────────────────────────────────────
# Problem Translation
# ─────────────────────────────────────────────────────────────────────────────
# The TP-matching objective is translated into a tractable HT constraint
# as follows:
#
#   • TPs of HP correspond to zero-crossings of its first differences.
#   • To match the TP rate of HP, we compute the empirical HT of HP in
#     first differences.
#   • We then impose this HT (in first differences) as the SSA constraint.
# ─────────────────────────────────────────────────────────────────────────────

# We begin by replicating Exercise 1.1 within a more general framework that
# accommodates a smooth transition from white noise, x_t = ε_t (the setting
# of Exercise 1.1), to differenced white noise, x_t = ε_t − ε_{t−1} — the
# process most relevant for deriving the SSA smoother when matching the TP rate 
# of HP.
# Specifically, we explore the MA(1) model x_t = ε_t + b1 * ε_{t-1} and examine
# how SSA changes as b1 transitions from b1 = 0 (white noise process)
# to b1 = -1 (first-difference process). A key technical challenge arises at
# b1 = -1, where the MA(1) process becomes non-invertible. To handle this
# boundary case rigorously, we derive and implement an exact SSA closed-form
# solution that remains valid even under non-invertibility.

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Specify M-SSA Design Settings: b1=0 
# ─────────────────────────────────────────────────────────────────────────────
Sigma            <- NULL   # Univariate design; identity covariance assumed
symmetric_target <- FALSE  # Causal (one-sided) target filter
gamma_target     <- 1      # Identity target: M-SSA tracks x_t = ε_t + b1*ε_{t-1}

# MA(1) parameter: x_t = ε_t + b1*ε_{t-1}
#   b1 =  0  → white noise       (replicates Exercise 1.1)
#   b1 = -1  → first differences (the limit case of primary interest)
# We begin with white noise.
b1 <- 0

# Wold decomposition of the MA(1) process
xi <- c(1, b1)

# δ = -(L-1)/2 places the target at the centre of the filter window
# (fully symmetric, two-sided smoother)
delta <- -(L - 1) / 2

# Default numerical optimisation settings
split_grid           <- 20
with_negative_lambda <- FALSE

# ─────────────────────────────────────────────────────────────────────────────
# 2.2  Specify the HT Constraint — Match HP
# ─────────────────────────────────────────────────────────────────────────────
# Compute the effective HP filter when applied to the MA(1) process x_t.
# The composite filter hp_target_diff, when applied to ε_t, replicates the
# output of HP applied to x_t = ε_t + b1*ε_{t-1}.
# When b1 = 0, x_t = ε_t and hp_target_diff reduces to the original HP filter.
hp_target_diff <- c(hp_target, 0) + b1 * c(0, hp_target)

rho1 <- compute_holding_time_func(hp_target_diff)$rho_ff1  # ρ implied by HP's HT
ht1  <- compute_holding_time_func(hp_target_diff)$ht       # HP holding time

par(mfrow = c(1, 1))
ts.plot(hp_target_diff,
        main = paste("Two-sided HP, b1 =", b1, ", HT =", round(ht1, 1)))

# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Compute the M-SSA Filter (b1 = 0: white noise)
# ─────────────────────────────────────────────────────────────────────────────
# Notes:
#   1. hp_target_diff enters only through rho1 (the HT constraint).
#   2. When b1 ≠ 0, ξ must be supplied; otherwise M-SSA defaults to white noise.
#      When b1 = 0, supplying ξ is optional (white noise is the correct model).
SSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1,
                     with_negative_lambda, xi)

# Filter coefficients applied to x_t = ε_t + b1*ε_{t-1}
bk_mat_diff_x <- SSA_obj$bk_x_mat
# Filter coefficients applied to ε_t directly
# (comparable to hp_target_diff, which is also expressed in terms of ε_t)
bk_mat_diff <- SSA_obj$bk_mat

# ── Visual inspection ─────────────────────────────────────────────────────────
# When b1 = 0, M-SSA replicates Exercise 1.1: both HP and M-SSA are symmetric.
par(mfrow = c(2, 1))
ts.plot(bk_mat_diff,
        main = "M-SSA Filter Coefficients (HT matched to HP)")
ts.plot(hp_target_diff,
        main = paste("Two-sided HP, b1 =", b1, ", HT =", round(ht1, 1)))

# When b1 = 0, x_t = ε_t, so the filters applied to x_t and ε_t coincide.
par(mfrow=c(1,1))
ts.plot(cbind(bk_mat_diff_x, bk_mat_diff), col = c("blue", "red"),main=
"SSA applied to x equals SSA applied to ε (both smoothers overlap)")
mtext("M-SSA applied to x",  col = "blue", line = -1)
mtext("M-SSA applied to ε",  col = "red",  line = -2)

# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Set b1 = -0.5
# ─────────────────────────────────────────────────────────────────────────────
b1  <- -0.5
xi  <- c(1, b1)   # Updated Wold decomposition

# Recompute the effective HP filter and its HT for the updated MA(1) process
hp_target_diff <- c(hp_target, 0) + b1 * c(0, hp_target)
rho1 <- compute_holding_time_func(hp_target_diff)$rho_ff1
ht1  <- compute_holding_time_func(hp_target_diff)$ht

# The two-sided HP filter is no longer exactly symmetric and its HT decreases.
par(mfrow = c(1, 1))
ts.plot(hp_target_diff,
        main = paste("Two-sided HP, b1 =", b1, ", HT =", round(ht1, 1)))

# Design the M-SSA filter (same notes as Section 2.3 apply)
SSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1,
                     with_negative_lambda, xi)

bk_mat_diff_x <- SSA_obj$bk_x_mat   # Filter applied to x_t
bk_mat_diff   <- SSA_obj$bk_mat     # Filter applied to ε_t

# ── Visual inspection ─────────────────────────────────────────────────────────
# For b1 = -0.5, both the M-SSA and HP filters are no longer symmetric.
par(mfrow = c(2, 1))
ts.plot(bk_mat_diff,
        main = "M-SSA Filter Coefficients (HT matched to HP applied to MA(1) x_t)")
ts.plot(hp_target_diff,
        main = paste("Two-sided HP, b1 =", b1, ", HT =", round(ht1, 1)))

# When b1 ≠ 0, x_t ≠ ε_t, so the two filters no longer coincide.
par(mfrow=c(1,1))
ts.plot(cbind(bk_mat_diff_x, bk_mat_diff), col = c("blue", "red"),main=
          "SSA applied to x differs from SSA applied to ε")
mtext("M-SSA applied to x_t",  col = "blue", line = -1)
mtext("M-SSA applied to ε_t",  col = "red",  line = -2)

# ─────────────────────────────────────────────────────────────────────────────
# 2.5  Set b1 = -0.9
# ─────────────────────────────────────────────────────────────────────────────
b1  <- -0.9
xi  <- c(1, b1)   # Updated Wold decomposition

# Recompute the effective HP filter and its HT for the updated MA(1) process
hp_target_diff <- c(hp_target, 0) + b1 * c(0, hp_target)
rho1 <- compute_holding_time_func(hp_target_diff)$rho_ff1
ht1  <- compute_holding_time_func(hp_target_diff)$ht

# As b1 → -1, the HP filter becomes increasingly asymmetric and its HT shrinks.
par(mfrow = c(1, 1))
ts.plot(hp_target_diff,
        main = paste("Two-sided HP, b1 =", b1, ", HT =", round(ht1, 1)))

# Design the M-SSA filter (same notes as Section 2.3 apply)
SSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1,
                     with_negative_lambda, xi)

bk_mat_diff_x <- SSA_obj$bk_x_mat   # Filter applied to x_t
bk_mat_diff   <- SSA_obj$bk_mat     # Filter applied to ε_t

# ── Visual inspection ─────────────────────────────────────────────────────────
# For b1 = -0.9, both filters exhibit more pronounced asymmetry.
par(mfrow = c(2, 1))
ts.plot(bk_mat_diff,
        main = "M-SSA Filter Coefficients (HT matched to HP applied to MA(1) x_t)")
ts.plot(hp_target_diff,
        main = paste("Two-sided HP, b1 =", b1, ", HT =", round(ht1, 1)))

# When b1 ≠ 0, x_t ≠ ε_t, so the two filters do not coincide.
par(mfrow=c(1,1))
ts.plot(cbind(bk_mat_diff_x, bk_mat_diff), col = c("blue", "red"),main=
          "SSA applied to x differs from SSA applied to ε")
mtext("M-SSA applied to x_t",  col = "blue", line = -1)
mtext("M-SSA applied to ε_t",  col = "red",  line = -2)

# Note: as b1 → -1, the right tail of the M-SSA filter decays progressively
# more slowly, reflecting the increasing persistence induced by the MA(1)
# structure of x_t.


# ─────────────────────────────────────────────────────────────────────────────
# 2.6  Simulate x_t and ε_t and Verify MA Inversion
# ─────────────────────────────────────────────────────────────────────────────
# We verify that applying bk_mat_diff_x to x_t yields the same smoother output
# as applying bk_mat_diff to ε_t. Agreement holds to the extent that the
# MA(1)-to-AR(L) inversion embedded in bk_mat_diff_x is sufficiently accurate.

# --- Simulate a white-noise series ---
lenq <- 2000
set.seed(86)
eps <- rnorm(lenq)

# --- Generate the MA(1) process x_t = ε_t + b1 * ε_{t-1} ---
# The first observation is initialised to zero (no lagged innovation available).
x <- c(0, eps[2:lenq] + b1 * eps[1:(lenq - 1)])

# --- Apply both filters and compare outputs ---
# Ideally the two outputs coincide; any discrepancy reflects inversion error.
# The filter applied to x
y_ssa_diff_x   <- filter(x,   bk_mat_diff_x, sides = 2)  # SSA smoother applied to x_t
# The filter applied to ε
y_ssa_diff_eps <- filter(eps, bk_mat_diff,   sides = 2)  # SSA smoother applied to ε_t

par(mfrow = c(1, 1))
ts.plot(cbind(y_ssa_diff_x, y_ssa_diff_eps)[500:1000, ], col = c("blue", "red"))

# The two series nearly overlap but do not coincide exactly.
# Reason: with b1 close to -1, the MA(1) is near non-invertible, so the AR(L)
# approximation embedded in bk_mat_diff_x requires a very long filter to converge.
# This problem is further compounded by the strong smoothness constraint imposed
# on the SSA filter, which slows the decay of the filter coefficients even further;
# as a result, the coefficients of bk_mat_diff_x have not decayed to zero within
# the available filter length L.
#
# Remedies:
#   1. Increase the filter length L to improve the AR approximation.
#   2. Use bk_mat_diff (applied to ε_t) instead, which bypasses inversion
#      entirely and is therefore exact. We adopt this approach in exercise 2.7,
#      where b1 = -1 renders the MA(1) strictly non-invertible.


# ─────────────────────────────────────────────────────────────────────────────
# 2.7  Set b1 = -1: First Differences (Non-Invertible MA(1))
# ─────────────────────────────────────────────────────────────────────────────
b1 <- -1

# Wold coefficients of the MA(1): xi = (1, b1).
# Special cases: b1 = 0 gives white noise; b1 = -1 gives first differences.
xi <- c(1, b1)

# --- Derive the target filter for ε_t ---
# hp_target_diff is the filter applied to ε_t that replicates the output of
# the two-sided HP filter applied to x_t = ε_t + b1 * ε_{t-1}.
hp_target_diff <- c(hp_target, 0) + b1 * c(0, hp_target)

# --- Extract holding-time (HT) characteristics of the HP target filter ---
rho1 <- compute_holding_time_func(hp_target_diff)$rho_ff1  # Autocorrelation implied by HP's HT
ht1  <- compute_holding_time_func(hp_target_diff)$ht       # Holding time of the HP filter

# --- Plot the HP target filter for ε_t ---
# With b1 = -1 the filter becomes anti-symmetric, reflecting the differencing operation.
par(mfrow = c(1, 1))
ts.plot(hp_target_diff,
        main = paste("Two-sided HP, b1 =", b1,
                     ", HT =", round(ht1, 1),
                     " (anti-symmetric cyclical filter)"))

# --- Design the SSA filter matching the HP holding-time constraint ---
# Notes:
#   1. hp_target_diff enters only through rho1 (the HT constraint); the full
#      target shape is not imposed directly.
#   2. When b1 ≠ 0, xi must be supplied so that SSA accounts for the MA(1)
#      structure of x_t. When b1 = 0 (white noise), xi may be omitted (it
#      defaults to white noise internally), but supplying it is also valid.
SSA_obj <- MSSA_func(split_grid, L, delta, grid_size,
                     gamma_target, rho1, with_negative_lambda, xi)

# --- Extract optimised filter coefficients ---
# bk_mat_diff_x: filter to be applied directly to x_t = ε_t + b1 * ε_{t-1}.
# bk_mat_diff:   filter to be applied to ε_t; comparable to hp_target_diff.
bk_mat_diff_x <- SSA_obj$bk_x_mat
bk_mat_diff   <- SSA_obj$bk_mat

# --- Visual inspection of filter shapes ---
# As b1 decreases from 0 toward -1, the SSA filter applied to ε_t (bk_mat_diff)
# becomes increasingly asymmetric: rather than inducing a spurious cycle (as HP
# does), its left and right tails are separated by a sharp, discontinuous step.
# This unconventional filter shape may appear counterintuitive at first glance;
# verifying that it nonetheless delivers superior tracking performance relative
# to HP will therefore be an instructive empirical exercise in example 2.8.
# Note: HP artificially imprints a cyclical structure on the filtered output
# that has no counterpart in the true data-generating process and therefore
# represents a spurious artifact of the filter design rather than a genuine
# feature of the underlying series. 
# In contrast, the SSA smoother is deliberately amorphous: it imposes no
# pre-specified functional form on the filtered output and its shape aims  
# at tracking $x_{t+delta}$ optimally (given the HT constraint). 

par(mfrow = c(2, 1))
ts.plot(bk_mat_diff,
        main = "SSA Filter Coefficients (HT matched to HP applied to MA(1) x_t)")
ts.plot(hp_target_diff,
        main = paste("Two-sided HP, b1 =", b1, ", HT =", round(ht1, 1)))

# --- Compare the two SSA filter representations ---
# bk_mat_diff_x (blue) and bk_mat_diff (red) differ because x_t ≠ ε_t when b1 ≠ 0.
# As b1 approaches -1, the right tail of bk_mat_diff_x fails to decay at all lags,
# signalling exact non-invertibility when b1=-1.
par(mfrow=c(1,1))
ts.plot(cbind(bk_mat_diff_x, bk_mat_diff), col = c("blue", "red"),main=
          "SSA applied to x differs strongly from SSA applied to ε")
mtext("SSA applied to x_t",  col = "blue", line = -1)
mtext("SSA applied to ε_t",  col = "red",  line = -2)

# Conclusion: because b1 = -1 renders the MA(1) strictly non-invertible,
# bk_mat_diff_x cannot be used reliably. All subsequent empirical evaluations
# therefore use bk_mat_diff (the filter applied to ε_t), which is exact.


# ─────────────────────────────────────────────────────────────────────────────
# 2.8  Verify Performance: Tracking Accuracy on Simulated Data
# ─────────────────────────────────────────────────────────────────────────────
# Filter performance is assessed on three complementary criteria:
#   a) Mean Squared Error (MSE)        — measures average squared deviation from target
#   b) Target correlation              — measures linear co-movement with the target
#   c) Sign accuracy                   — proportion of time steps where filter output
#                                        and target share the same sign (fewer false alarms)
# ─────────────────────────────────────────────────────────────────────────────

# --- Simulate a large sample for reliable Monte Carlo estimates ---
lenq <- 1000000
set.seed(86)
eps <- rnorm(lenq)

# Generate the MA(1) process (first observation initialised to zero)
x <- c(0, eps[2:lenq] + b1 * eps[1:(lenq - 1)])

# --- Apply filters ---
# Non-invertibility (b1 = -1) requires applying bk_mat_diff to ε_t rather than
# applying bk_mat_diff_x to x_t, as the latter is not well-defined in this case.
y_ssa_diff_eps <- filter(eps, bk_mat_diff,   sides = 2)  # SSA smoother output (exact)
y_hp_two_diff  <- filter(eps, hp_target_diff, sides = 2)  # HP smoother output

# --- Define the target series ---
# Using sides = 2 (in the above filter call) centres the filter output at t, so 
# the output already incorporates the lag of -delta periods. The target is 
# therefore x_t (not x_{t + delta}).
target <- x


# ── a. Mean Squared Error (MSE) ───────────────────────────────────────────────
# MSE quantifies how closely each smoother tracks the target on average.
mse_ssa_smooth <- mean((target - y_ssa_diff_eps)^2, na.rm = TRUE)
mse_hp_smooth  <- mean((target - y_hp_two_diff)^2,  na.rm = TRUE)

# Expected result: SSA achieves a lower MSE than HP.
# The improvement may be modest because the design (b1 = -1) is an extreme case;
# nonetheless SSA almost always outperforms HP across different random seeds.
# Alternatively, outperformance could be verified on even longer samples.
mse_ssa_smooth  # SSA MSE
mse_hp_smooth   # HP  MSE


# ── b. Target Correlation ─────────────────────────────────────────────────────
# Pearson correlation between each smoother output and the target series.
output_mat_diff <- na.exclude(cbind(target, y_ssa_diff_eps, y_hp_two_diff))

# Prepend a row of zeros to align lengths with previous designs
output_mat_diff <- rbind(rep(0, ncol(output_mat_diff)), output_mat_diff)
colnames(output_mat_diff) <- c("x-diff", "SSA-diff", "HP-diff")

cor_mat <- cor(output_mat_diff)
cor_mat  # Full correlation matrix (first row: correlations with target)

# Expected result: SSA correlates more strongly with the target than HP.
cor_mat[1, 2]  # Target–SSA correlation
cor_mat[1, 3]  # Target–HP  correlation

# --- Internal convergence check ---
# The sample correlation cor(target, y_ssa_diff_eps) should converge to
# SSA_obj$crit_rhoy_target as the sample size grows, confirming that the
# optimisation correctly maximises the true target correlation criterion.
cor_mat[1, 2]             # Empirical target–SSA correlation
SSA_obj$crit_rhoy_target  # Optimised (theoretical) target correlation
# Agreement between these two quantities validates that maximising the true
# target correlation within SSA is both well-defined and empirically meaningful.


# ── c. Sign Accuracy ──────────────────────────────────────────────────────────
# Proportion of time steps at which the smoother output and the target agree
# in sign. Higher sign accuracy implies fewer false directional signals.
# Expected result: SSA achieves higher sign accuracy than HP.
sign_acc_ssa <- sum((target * y_ssa_diff_eps) > 0, na.rm = TRUE) /
  length(na.exclude(target * y_ssa_diff_eps))

sign_acc_hp  <- sum((target * y_hp_two_diff) > 0, na.rm = TRUE) /
  length(na.exclude(target * y_hp_two_diff))

sign_acc_ssa  # SSA sign accuracy
sign_acc_hp   # HP  sign accuracy

# Note that in this extreme case (study), both smoothers perform only marginally
# above pure chance (50%), reflecting the fact that the target series is highly
# noisy: applying the first-difference filter amplifies high-frequency content,
# which makes accurate sign prediction inherently difficult regardless of the
# smoother used.


# ─────────────────────────────────────────────────────────────────────────────
# 2.9  Results: Smoothness
# ─────────────────────────────────────────────────────────────────────────────
# Smoothness is evaluated on two complementary criteria:
#   a) Holding time (HT): mean duration between consecutive sign changes.
#   b) Curvature: root mean squared second-order differences (RMSD2).
# ─────────────────────────────────────────────────────────────────────────────

# ── a. Holding Time ───────────────────────────────────────────────────────────
# The HT constraint is imposed on SSA applied to first differences x_t.
# The empirical HTs of both smoothers should agree closely in first differences,
# which ensures that SSA replicates the rate of turning points of HP in levels
# (eps_t). This confirms that the problem specified at the start of the exercise
# has been solved.
ht1                                        # Target HT imposed on SSA (= HP holding time)
compute_empirical_ht_func(y_ssa_diff_eps)  # Empirical HT of SSA output
compute_empirical_ht_func(y_hp_two_diff)   # Empirical HT of HP output

# ── b. Curvature (Root Mean Squared Second-Order Differences) ─────────────────
# HP has a much smaller curvature than SSA by construction (WH optimality).
sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat_diff), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif

# Note: the first element of sq_se_dif corresponds to the raw target
# x_t = eps_t - eps_{t-1} (first differences of standardised white noise).
# Applying the second-order difference operator (1 - B)^2 to these first
# differences gives:
#
#   (1-B)^2 (eps_t - eps_{t-1})
#     = eps_t - 3*eps_{t-1} + 3*eps_{t-2} - eps_{t-3}
#     = (1-B)^3 eps_t
#
# The variance of this expression is:
#
#   Var[(1-B)^3 eps_t] = 1^2 + 3*((-3)^2) + 3*(3^2) + (-1)^2
#                      = 1 + 9 + 9 + 1 = 20
#
# so the expected RMSD2 of the raw target is sqrt(20) ≈ 4.472.
# The first element of sq_se_dif converges asymptotically to this theoretical
# value, providing a useful sanity check on the curvature calculations.

# ─────────────────────────────────────────────────────────────────────────────
# 2.10  Plot Series: 1,000-observation window
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 1))
colo <- c("blue", "violet")

ts.plot(na.exclude(output_mat_diff)[1000:2000, 2:3], col = colo)
abline(h = 0)

# Add colour-coded series labels as margin text
for (i in 1:ncol(output_mat_diff[ , 2:3]))
  mtext(colnames(output_mat_diff[ , 2:3])[i], col = colo[i], line = -i)

# This extreme example highlights the main differences between the two smoothing
# concepts:
#   • Both smoothers share the same HT (rate of zero-crossings).
#   • They differ markedly, however, in shape, scale, and dynamics.
#
# SSA maximizes tracking of x_{t+delta} = eps_{t+delta} - eps_{t+delta-1}
# (target correlation, sign accuracy, and MSE), whereas HP minimizes curvature.
#
# SSA controls the rate of zero-crossings by generating large, noisy cycles
# that drift away from the zero line for extended intervals.
#   • Once near the zero line, multiple crossings may occur due to noise.
#   • Crossings are irregularly distributed and tend to cluster.
#
# HP controls HT by constraining curvature instead.
#   • This imposes extraneous (cyclical) structure on the data that is absent
#     from the data-generating process.
#   • As a result, crossings are much more regularly spaced, a regularity that
#     conflicts with the noisy data-generating process.
#
# Neither solution is practically relevant here because it is applied to
# x_t = (1-B)eps_t, a non-invertible MA process that emphasizes high-frequency
# oscillations. In economic applications, this case corresponds to second-order
# differencing of a non-stationary indicator (i.e., overdifferencing), which
# emphasizes acceleration rather than growth. Since growth is typically the more
# relevant quantity for decision-making, this setting serves primarily as an
# instructive theoretical case rather than a practical one.
#
# The example nevertheless illustrates the modus operandi of both smoothness
# concepts by exposing their characteristic traits (amorphous vs. shaping) in 
# a revelatory setting.

# ─────────────────────────────────────────────────────────────────────────────
# 2.11  Distribution of Crossings: Mean and Standard Deviation
# ─────────────────────────────────────────────────────────────────────────────

# Identify time points where the SSA output changes sign
ssa_zc <- which(output_mat_diff[1:(nrow(output_mat_diff) - 1), "SSA-diff"] *
                  output_mat_diff[2:(nrow(output_mat_diff)),     "SSA-diff"] < 0)

# Identify time points where the HP output changes sign
hp_zc  <- which(output_mat_diff[1:(nrow(output_mat_diff) - 1), "HP-diff"] *
                  output_mat_diff[2:(nrow(output_mat_diff)),      "HP-diff"] < 0)

# ── Mean durations between consecutive zero-crossings (empirical HTs) ────────
mean(diff(ssa_zc))   # SSA empirical HT
mean(diff(hp_zc))    # HP  empirical HT
# Both means should match the imposed HT target ht1
ht1

# ── Standard deviations of inter-crossing durations ──────────────────────────
sd(diff(ssa_zc), na.rm = TRUE)   # SSA: variability in zero-crossing spacing
sd(diff(hp_zc),  na.rm = TRUE)   # HP:  variability in zero-crossing spacing

# Zero-crossings of SSA are irregularly spaced and clustered, reflected in its
# much higher standard deviation relative to HP.
#   • The amorphous SSA smoother does not imprint any particular regularity
#     pattern on the data-generating process.

# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways
# ─────────────────────────────────────────────────────────────────────────────

# Exercise 2 highlights, more clearly than Exercise 1, the amorphous nature of
# SSA smoothing by contrasting it with WH/HP in an extreme - and therefore more 
# revelatory - case study.
#
# 1. For a given HT constraint, SSA maximizes target correlation and hence
#    sign accuracy (equivalently, minimizes MSE after suitable calibration).
#
# 2. This provides a natural and efficient framework for defining zero-crossings
#    based on optimal tracking performance. The approach does not imprint any
#    unnatural or artificial structure on the data since optimal tracking is
#    fully data-driven. The HT constraint operates through the first-order
#    autocorrelation of the filtered output, which is a minimal and unintrusive
#    restriction that neither conflicts with nor distorts the structure of the
#    data-generating process.
#
# 3. The HT constraint is appealing in situations where crossings of the 
#    smoother of the data x_t can be interpreted as TPs of a relevant 
#    non-stationary series I_t in levels, where x_t=I_t-I_{t-1} represents a 
#    noisy growth estimate. Defining TPs in I_t as zero-crossings of an optimal 
#    growth tracker is both logically consistent and statistically efficient. 
#
# 4. Imposing the HP holding time on SSA in first differences guarantees that:
#    i)  SSA replicates the rate of turning points (zero-crossings) of HP,
#        which was the original problem posed at the beginning of Exercise 2.
#        Achieving this replication requires much stronger smoothing by SSA.
#    ii) SSA outperforms HP in tracking x_{t+delta}, though gains are
#        marginal in this extreme setting because the target has a very strong
#        high-frequency content (which inherently conflicts with `smoothing').
#
# 5. Smoothed series may appear very different even when their tracking 
#    performances are similar. 
#     • HP imposes a smooth, regular cyclical structure with evenly spaced
#       crossings; 
#     • SSA instead produces larger, more irregular dynamics in order to
#       track the target optimally. Consequently, zero-crossings are less 
#       regularly distributed and exhibit greater variance in their durations.
#
# 6. These results extend straightforwardly to arbitrary stationary-dependent
#    processes by specifying the appropriate xi (Wold decomposition).
# ─────────────────────────────────────────────────────────────────────────────

################################################################################
# Exercises 1 and 2 considered smoothers at delta = -(L-1)/2. We now explore
# what happens when delta is varied from delta = -(L-1)/2 to delta = 0
# (nowcasting). To increase the challenge, we impose a fixed HT for all delta. 
# In this case, the nowcast will be as smooth as the symmetric backcast in terms 
# of zero-crossing rate (in contrast to HP whose nowcast is much less smooth).
################################################################################




# ════════════════════════════════════════════════════════════════════════════════
# Exercise 3: COMPUTE THE FULL SSA SMOOTHER FAMILY (delta SWEEP) KEEPING HT FIXED
# ════════════════════════════════════════════════════════════════════════════════
# Sweep delta from the fully symmetric end (-(L-1)/2) to the nowcast (delta = 0),
# designing one filter per lag shift.
# Assumption: x_t = eps_t is white noise (approximating a differenced economic series).

# ─────────────────────────────────────────────────────────────────────────────
# 3.1  HT OF HP NOWCAST
# ─────────────────────────────────────────────────────────────────────────────

ht_one_sided <- compute_holding_time_func(hp_trend)$ht    # HT of one-sided HP filter
ht_two_sided <- compute_holding_time_func(hp_target)$ht   # HT of two-sided HP filter

# The HT of the one-sided HP is much shorter than that of the two-sided HP:
# the one-sided filter generates approximately 8 times more zero-crossings.
ht_one_sided
ht_two_sided
# We here construct SSA-smoothers with a fixed HT, irrespective of the backcast 
# lag

# ─────────────────────────────────────────────────────────────────────────────
# 3.2  IMPOSE FIXED HT OF TWO-SIDED HP ON SSA
# ─────────────────────────────────────────────────────────────────────────────
# Re-extract the HP holding-time parameters for use in the delta sweep.
rho1 <- compute_holding_time_func(hp_target)$rho_ff1
ht1  <- compute_holding_time_func(hp_target)$ht

gamma_target <- 1   # Allpass target
filt_mat <- acf1 <- target_cor <- NULL

# Design one filter for each lag delta, keeping HT (rho1) fixed.
# xi is not specified in the SSA call: x_t is assumed to be white noise.
for (delta in (-(L - 1) / 2):0)
{
  print(paste("Progress: ", ((L - 1) / 2 + delta) * 100 / ((L - 1) / 2), "%", sep = ""))
  SSA_obj    <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1)
  filt_mat   <- cbind(filt_mat, SSA_obj$bk_mat)
  acf1       <- c(acf1, SSA_obj$crit_rhoyy)
  target_cor <- c(target_cor, SSA_obj$crit_rhoy_target)
}

# Plot target correlation as a function of the backcast lag.
# A monotonically increasing shape indicates that the target becomes
# progressively easier to track as the backcast lag increases.
ts.plot(target_cor[length(target_cor):1], main = "Target correlation as a function of lag")

# Plot the first-order ACF (HT constraint).
# Ideally this is a flat line matching the imposed rho1.
# Small deviations reflect numerical imprecision and can be reduced by
# increasing the split_grid parameter in the SSA function call.
ts.plot(acf1)
var(acf1)   # Variance is negligible

# ─────────────────────────────────────────────────────────────────────────────
# 3.3  PLOT THE FULL SSA SMOOTHER FAMILY
# ─────────────────────────────────────────────────────────────────────────────

mplot <- filt_mat

# Plot all smoother coefficient profiles, coloured by delta
plot(
  mplot[, 1],
  col  = rainbow(ncol(filt_mat))[1],
  main = "SSA Smoothers",
  axes = FALSE, type = "l",
  ylab = "", xlab = expression(delta)
)
for (i in 2:ncol(filt_mat))
  lines(mplot[, i], col = rainbow(ncol(mplot))[i])

# Mark the filter centre (lag-0 position).
# At delta = -(L-1)/2 the design replicates the solution from Exercise 1.1.
abline(v = (L - 1) / 2 + 1)

axis(1,
     at     = 1 + c(1, 50 * 1:(nrow(filt_mat) / 50)),
     labels = c(0,    50 * 1:(nrow(filt_mat) / 50)))
axis(2)
box()

# ─────────────────────────────────────────────────────────────────────────────
# 3.4  Apply One-Sided and Symmetric SSA Smoothers and Compare Performances
# ─────────────────────────────────────────────────────────────────────────────

# Simulate a long white-noise series to ensure reliable sample estimates
lenq <- 1000000
set.seed(86)
x <- rnorm(lenq)

# Apply the one-sided (nowcast) and two-sided (symmetric) SSA smoothers to x.
# sides = 1: causal (one-sided) convolution, corresponding to delta = 0.
# sides = 2: acausal (two-sided) convolution, corresponding to delta = -(L-1)/2.
y_ssa_one_sided <- filter(x, filt_mat[, ncol(filt_mat)], sides = 1)
y_ssa_two_sided <- filter(x, filt_mat[, 1],              sides = 2)

# Target: the unshifted series x, since both filters are designed to track
# x_t directly (the forward shift by abs(delta) is absorbed into the two-sided 
# filter).
target <- x

# ── a. MSE ───────────────────────────────────────────────────────────────────
# Compute MSE of each smoother relative to the target series.
mse_ssa_one_sided <- mean((target - y_ssa_one_sided)^2, na.rm = TRUE)
mse_ssa_two_sided <- mean((target - y_ssa_two_sided)^2, na.rm = TRUE)

mse_ssa_one_sided
mse_ssa_two_sided

# ── b. Target Correlation ─────────────────────────────────────────────────────
# Compute the correlation matrix of the target and both filter outputs.
output_mat <- na.exclude(cbind(target, y_ssa_one_sided, y_ssa_two_sided))
colnames(output_mat) <- c("x", "One-Sided", "Two-Sided")
cor_mat <- cor(output_mat)
cor_mat   # Full correlation matrix (first row: correlations with target)

cor_mat[1, 2]   # Target correlation: one-sided SSA
cor_mat[1, 3]   # Target correlation: two-sided SSA

# Internal convergence check: the sample correlation cor(x, y_ssa) should
# match crit_rhoy_target up to Monte Carlo sampling error, confirming
# that the optimisation has converged correctly.
cor_mat[1, 2]              # Empirical correlation: one-sided SSA
target_cor[length(target_cor)]  # Expected target correlation: one-sided SSA
cor_mat[1, 3]              # Empirical correlation: two-sided SSA
target_cor[1]              # Expected target correlation: two-sided SSA
  
# ── c. Sign Accuracy ──────────────────────────────────────────────────────────
# Proportion of time steps at which the filter output and the target share
# the same sign.
sign_acc_one_sided <- sum((target * y_ssa_one_sided) > 0, na.rm = TRUE) /
length(na.exclude(target * y_ssa_one_sided))

sign_acc_two_sided <- sum((target * y_ssa_two_sided) > 0, na.rm = TRUE) /
length(na.exclude(target * y_ssa_two_sided))

sign_acc_one_sided
sign_acc_two_sided

# The two-sided filter tracks the target better, as expected. Performances are 
# mitigated by the fact that the series is white noise and the smoothing is 
# strong.

# ─────────────────────────────────────────────────────────────────────────────
# 3.5  Smoothness
# ─────────────────────────────────────────────────────────────────────────────
# Smoothness is evaluated on two complementary criteria:
#   a) Holding time (HT): mean duration between consecutive sign changes.
#   b) Curvature: root mean squared second-order differences (RMSD2).
# ─────────────────────────────────────────────────────────────────────────────

# ── a. Holding Time ───────────────────────────────────────────────────────────
# The HT constraint imposed on SSA is set equal to that of the two-sided HP
# filter. The empirical HTs of both SSA smoothers should agree
# closely with ht1.
ht1                                       # Target HT (= two-sided HP holding time)
compute_empirical_ht_func(y_ssa_one_sided)  # Empirical HT of one-sided SSA
compute_empirical_ht_func(y_ssa_two_sided)  # Empirical HT of two-sided SSA

# Notably, the one-sided SSA maintains the imposed HT,
# in contrast to the one-sided HP filter (see ht_one_sided above).

# ── b. Curvature (Root Mean Squared Second-Order Differences) ─────────────────
# Curvature is measured as the RMSD2 of each smoother's output — the natural
# smoothness criterion minimised by HP under the WH framework.
# HP achieves the smallest RMSD2 by construction (WH optimality); SSA incurs
# larger curvature as the cost of superior tracking accuracy under the same
# HT constraint.
output_mat <- cbind(y_ssa_one_sided, y_ssa_two_sided)
sq_se_dif <- sqrt(apply(
  apply(apply(na.exclude(output_mat), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif

# Unexpectedly, the one-sided SSA exhibits smaller curvature than the two-sided
# smoother. This reflects the more gradual, regular weight profile of the
# one-sided filter, which lacks the sharp central peak of the two-sided design.

# this result once more 



# ══════════════════════════════════════════════════════════════════════════════
# Exercise 4: I-SSA Smoothing on non-stationary levels
# ══════════════════════════════════════════════════════════════════════════════
# Similar to tutorial 6 but we target I_t instead of HP(I_t)

# Exercise 4.1 Create exercise (random-walk???)



# Exercise 4.2 Replicate TP-frequency of HP by SSA
# This is once again unusal because we use I-SSA for series that are stationary.
# 1. Define HP in diffs: HT in diffs = TP rate on level
# 2. Specify I-SSA that targets cumsum(x_t)=eps_t on levels and imposes 
#     HT on differences x_t=eps_t-eps_{t-1}
# In contrast to exercise 2, the resulting I-SSA replicates TP-rate without 
#   additional cumsum, is stationary and tracks eps_t optimally.

# ══════════════════════════════════════════════════════════════════════════════
# Exercise 5: M-SSA Smoothing
# ══════════════════════════════════════════════════════════════════════════════







