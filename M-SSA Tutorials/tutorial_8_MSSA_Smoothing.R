# This tutorial is under construction

# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 8: M-SSA SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# SMOOTHING VS. PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
# M-SSA can target either acausal or causal objectives, subject to a
# holding-time (HT) constraint:
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
#     of a non-stationary level series I_t.
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
# by any pre-filtering introduced through a non-trivial target specification.
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
# ──────────────────────────────────────────────────────────────────────────────
# DATA-GENERATING PROCESS
# ──────────────────────────────────────────────────────────────────────────────
# To apply the HT concept, we assume x_t to be stationary, eventually after 
# differences.
#   • M-SSA smoothing can be applied to non-stationary series, see example ???
#   • But the HT constraint must be specified in stationary differences.
#
# The Wold decomposition of the (stationary) data x_t enters M-SSA via the
# function argument ξ (the variable "xi") and plays a central role in filter 
# design. Incorporating the correct ξ into the M-SSA optimisation ensures:
#
#   a) Optimality      : M-SSA maximises tracking of x_{t + delta} (in terms
#                        of target correlation, sign accuracy, or MSE) for a
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
# BENCHMARK IN TUTORIAL 8
# ══════════════════════════════════════════════════════════════════════════════
# Smoothing Refers to a data processing technique that removes noise or 
# short-term fluctuations from a series, typically based on an optimization  
# approach that balances fidelity to the observed data and smoothness of the 
# resulting curve (e.g., penalizing second differences)
#
# Classical Whittaker–Henderson (WH) graduation is based on the following 
# optimization problem:
#
#   min_{y_t} sum_{t=1}^{T} (x_t - y_t)^2
#              + lambda * sum_{t=d+1}^{T} ((1-B)^d * y_t)^2
#
# For d=2 the Hodrick Prescott (HP) filter is obtained. 
# We here benchmark M-SSA against WH and to simplify exposition we assume d=2 (HP).
#
# While many different smoothing approaches exist, a common feature is that they
# generally do not appeal to an explicit model of the data-generating
# process — smoothness is instead enforced through a penalty term or
# a structural constraint imposed directly on the output.
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
#     transitions between above- and below-average growth in I_t.
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
#     a structural regularity not present in the original data x_t, which is
#     well-approximated by white noise (aperiodic by nature).
#   • This regularity is a signature imposed by the HP filter itself, not a
#     feature of the underlying signal: it constitutes spurious smoothing.
#   • SSA, by contrast, inherits the irregular spacing of zero-crossings
#     from the data, producing a more faithful representation of the
#     underlying growth dynamics.

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
#   • SSA and HP produce different TP datings for I_t.
#   • Key question: which TP dating is more credible?
#
# ── Data-Driven vs. Externally Imposed Structure ──────────────────────────────
#   • SSA derives TP datings in I_t through optimal tracking of the first
#     differences x_t. Turning points are determined by systematic changes in
#     growth — they emerge intrinsically from the data.
#
#   • HP derives TP datings by imposing minimal curvature on the output. Turning
#     points reflect a structural shape (low curvature) that is not an inherent
#     property of the data but is imposed from outside — a form of spurious
#     smoothing.
#
# ── Interpretability ──────────────────────────────────────────────────────────
#   • SSA: TP datings of I_t follow directly from optimal tracking of growth
#     in x_t — an intuitively natural and operationally meaningful criterion.
#
#   • HP: TP datings follow from controlling curvature in x_t, which corresponds
#     to penalising third-order differences (1−B)³I_t in I_t — a quantity that
#     is difficult to interpret in economic or practical terms.
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
#     reliable signal of an impending zero-crossing (TP in I_t).
#
# ── Qualification ─────────────────────────────────────────────────────────────
# While the above observations are accurate, it is not clear that they
# constitute a genuine advantage for HP. The apparent anticipativity of HP's
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
# 1. Optimality of SSA
#    For a given HT constraint, SSA maximises target correlation and sign
#    accuracy with respect to the target x_{t+δ} — by construction.
#
# 2. Dominance over HP
#    Imposing the HP holding time as the HT constraint guarantees that SSA
#    outperforms HP in tracking x_{t+δ}, for an identical zero-crossing rate.
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
#    to the SSA design criterion.
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways from Exercise 1.1
# ─────────────────────────────────────────────────────────────────────────────
# 1. For a given HT constraint, SSA maximises target correlation and hence
#    sign accuracy with respect to the target x_{t+delta}.
#
# 2. Imposing the HP holding time on SSA guarantees that SSA outperforms
#    HP in tracking x_{t + delta} for identical zero crossing rate.
#
# 3. The gain in tracking accuracy of SSA comes at the cost of larger curvature
#    relative to HP
#       →HP smoothers appear rounder
#       →with less TPs and 
#       →with more regularly distributed zero crossings.
#       →faster decaying damped cycle ACF structure
#
# 4. SSA smoothing imposes less structure on the data than HP
#       →zero crossings are derived from optimally tracking the data (purely 
#         data based)
#       →the HT constraint affects the lag-one ACF (does not impose or prefer a 
#         particular shape of the series)
#       →The ACF pattern is monotonically decreasing, without artificial 
#         structure (cycle)    
#
# 5. These results extend from white noise to autocorrelated processes by 
#     specifying the appropriate ξ (Wold decomposition).
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
ts.plot(bk_mat_1)

# ── Verify that the SSA target correlation matches that of HP ───────────────
# If the values below do not match, revise ht1_1 manually until they agree.
SSA_obj_1$crit_rhoy_target   # SSA population target correlation
cor(output_mat)[1, 3]        # HP empirical target correlation (sample estimate)

# ── Conclusion ────────────────────────────────────────────────────────────────
# Since both target correlations match, we conclude that SSA generates a
# larger HT (ht1_1 = 75) than HP (ht1) to achieve the same tracking accuracy —
# confirming the dual result of Exercise 1.2.
ht1   # HP holding time (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 1.2.2  Results: Tracking Accuracy on Simulated White Noise
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
# Exercise 2: TARGET MONOTONICITY IN SSA
# ════════════════════════════════════════════════════════════════════════════════
# This exercise is deliberately extreme and, in practice, unrealistic. It is
# designed to explore the characteristics of SSA smoothing when pushed to
# its limits — a setting that, while practically irrelevant, is instructive
# for understanding the fundamentals of the SSA framework by probing its
# boundaries.

# ─────────────────────────────────────────────────────────────────────────────
# Motivation
# ─────────────────────────────────────────────────────────────────────────────
# Exercise 1.1 showed that SSA exhibits substantially more turning points
# (TPs) than HP for an equal HT, since HP minimises curvature and thereby
# suppresses TPs. 
# 
# Problem formulation: design an M-SSA smoother that replicates not the HT of
# HP, but rather its rate of turning points (TPs) — a strictly stronger
# smoothness requirement, since matching the TP rate of HP demands a
# substantially larger HT in SSA.

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


# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Specify SSA Design Settings
# ─────────────────────────────────────────────────────────────────────────────

# Sigma: variance–covariance matrix of the innovation process.
#        NULL → univariate design; identity matrix assumed.
Sigma <- NULL

# xi: spectral density of first differences of the input (white noise) process.
# x_t=eps_t-eps_{t-1}: this not invertible and therefore the SSA filter 
# applied to x_t is not optimal (to be optimal, the MA must be invertible)
xi <- c(1,-1)

# symmetric_target: FALSE → causal (one-sided) target filter.
symmetric_target <- FALSE

# --- Target filter ---
# Target x_{t+delta}=eps_{t+delta}-eps_{t+delta-1}
gamma_target <- 1

# Setting delta = -(L-1)/2 places the target at the centre of the filter
delta <- -(L - 1) / 2

# Default numerical optimization settings
split_grid <- 20
with_negative_lambda<-F

# ─────────────────────────────────────────────────────────────────────────────
# 2.2  Specify the HT Constraint — Match HP 
# ─────────────────────────────────────────────────────────────────────────────
# Impose the same HT as the two-sided HP filter.
# The filter applied to eps_t: this generates the same output as HP applied to x_t=eps_t-eps_{t-1}
# We need this filter (applied to noise eps_t) to compute the HT only (it has no other meaning)
hp_target_diff<-c(hp_target,0)-c(0,hp_target)

par(mfrow=c(1,1))
ts.plot(hp_target_diff,main="Anti-symmetric two-sided HP in first differences")

rho1 <- compute_holding_time_func(hp_target_diff)$rho_ff1  # rho implied by HP's HT
ht1  <- compute_holding_time_func(hp_target_diff)$ht       # Holding time of HP filter

# Interpretation of ht1:
#   This reflects the mean duration between consecutive zero-crossings of first
#   differences of HP: the mean length of monotonous cycle phases
# This duration corresponds to the mean duration between consecutive TPs in levels of HP, see exercise 1.1.9
# This duration is shorter than the HT in levels in exercise 1
ht1   # Inspect the HP holding time: 


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Compute the SSA Filter
# ─────────────────────────────────────────────────────────────────────────────
# Design the SSA filter subject to the HP-matched HT constraint.
# Note: we must supply xi (difference filter), since otherwise SSA assumes white noise

SSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1,with_negative_lambda,xi)

# Extract the optimised (univariate) filter coefficients
# This is the filter applied to x_t=eps_t-eps_{t-1}
# This is not optimal because the MA is not invertible
bk_mat_diff_x <- SSA_obj$bk_x_mat
# This is the filter applied to eps_t
# it can be compared to hp_target_diff which is also applied to eps_t
bk_mat_diff <- SSA_obj$bk_mat

# --- Visual inspection ---
# The filter should be symmetric, consistent with delta = -(L-1)/2
par(mfrow = c(2, 1))
ts.plot(bk_mat_diff, main = "SSA Filter Coefficients (Match HT of HP in first differences)")
ts.plot(hp_target_diff,main="Anti-symmetric two-sided HP in first differences")

# --- Optimisation diagnostics ---
# Check first-order ACF: if the numerical optimisation converged, then this should match rho1
#   Otherwise: increase split_grid
SSA_obj$crit_rhoyy
rho1

# Target correlation: this is maximized by SSA
# It should be larger than HP, see below
SSA_obj$crit_rhoy_target
# The traget correlation is very small (nearly vanishing) because we ask SSA (and HP)
# to accomplish a very difficult smoothing task, since the data x_t=eps_t-eps_{t-1} is 
# very noisy (very strong high-frequency content)


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Results: Tracking Accuracy on Simulated White Noise
# ─────────────────────────────────────────────────────────────────────────────
# Performance is evaluated on three complementary criteria:
#   a) Mean Squared Error (MSE)
#   b) Target correlation
#   c) Sign accuracy (proportion of correct sign agreements)
# ─────────────────────────────────────────────────────────────────────────────

# --- Simulate a long white-noise series ---
lenq <- 1000000
set.seed(86)
eps <- rnorm(lenq)
x<-c(0,diff(eps))

#????????? Explain why we apply to xt and not xt-xt-1
# --- Apply both symmetric (two-sided) filters to x ---
# sides = 2 specifies an acausal two-sided (symmetric) convolution,
# consistent with the symmetric filter design (delta = -(L-1)/2).
y_ssa_diff    <- filter(x, bk_mat_diff_x,    sides = 2)  # SSA smoother output
y_ssa_diff    <- filter(eps, bk_mat_diff,    sides = 2)  # SSA smoother output
y_hp_two_diff <- filter(eps, hp_target_diff, sides = 2)  # HP smoother output

# Target: since both filters are symmetric and acausal, the target is
# the unshifted series x (no lead or lag adjustment required).
target <- x


# ── a. MSE ───────────────────────────────────────────────────────────────────
# Compute MSE of each smoother relative to the target series.

mse_ssa_smooth <- mean((target - y_ssa_diff)^2,    na.rm = TRUE)
mse_hp_smooth  <- mean((target - y_hp_two_diff)^2, na.rm = TRUE)

# Expected result: SSA achieves a smaller MSE than HP.
mse_ssa_smooth   # SSA MSE
mse_hp_smooth    # HP MSE


# ── b. Target Correlation ─────────────────────────────────────────────────────
# Compute the correlation matrix of the target and both filter outputs.

output_mat_diff <- na.exclude(cbind(target, y_ssa_diff, y_hp_two_diff))
# Append a zero to match the length of the series in exercise 1
output_mat_diff<-rbind(rep(0,ncol(output_mat_diff)),output_mat_diff)
colnames(output_mat_diff)<-c("x-diff","SSA-diff","HP-diff")

cor_mat    <- cor(output_mat_diff)
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

sign_acc_ssa <- sum((target * y_ssa_diff)    > 0, na.rm = TRUE) /
  length(na.exclude(target * y_ssa_diff))

sign_acc_hp  <- sum((target * y_hp_two_diff) > 0, na.rm = TRUE) /
  length(na.exclude(target * y_hp_two_diff))

sign_acc_ssa   # SSA sign accuracy
sign_acc_hp    # HP sign accuracy


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  Results: Smoothness
# ─────────────────────────────────────────────────────────────────────────────
# Smoothness is evaluated on two complementary criteria:
#   a) Holding time (HT): mean duration between consecutive sign changes.
#   b) Curvature: root mean squared second-order differences (RMSD2).
# ─────────────────────────────────────────────────────────────────────────────

# ── a. Holding Time ───────────────────────────────────────────────────────────
# The HT constraint imposed on SSA is set equal to that of the HP filter.
# The empirical HTs of both smoothers should therefore agree closely.

ht1                              # Target HT imposed on SSA (= HP holding time)
compute_empirical_ht_func(y_ssa_diff)      # Empirical HT of SSA output
compute_empirical_ht_func(y_hp_two_diff)   # Empirical HT of HP output

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
  apply(apply(na.exclude(output_mat_diff), 2, diff), 2, diff)^2,
  2, mean
))
sq_se_dif

# Note: the first element of sq_se_dif corresponds to the raw target x_t-x_{t-1}
# (first differences of standardised white noise). Applying the second-order difference operator
# (1 - B)^2 to these differences gives:
#
#   (1 - B)^2 x_t-x_{t-1}  =  x_t - 2 x_{t-1} + x_{t-2}-(x_{t-1}-2x_{t-2}+x_{t-3})
#   =x_t-3x_{t-1}+3x_{t-2}-x_{t-3}
#
# The variance of this expression is:
#
#   Var[(1-B)^3 x_t]  =  1^2 + 2*(3)^2 + 1^2  =  1 + 18 + 1  =  20
#
# so the expected RMSD2 of the raw target is sqrt(20) ≈ 4.472.
# The first element of sq_se_dif converges asymptotically to this theoretical value,
# providing a useful sanity check on the curvature calculations.


# ─────────────────────────────────────────────────────────────────────────────
# 2.6  Plot Series: 1,000-observation window
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 1))
colo <- c("blue", "violet")

ts.plot(na.exclude(output_mat_diff)[1000:2000, 2:3], col = colo)
abline(h = 0)
# Add colour-coded series labels as margin text
for (i in 1:ncol(output_mat_diff[ , 2:3]))
  mtext(colnames(output_mat_diff[ , 2:3])[i], col = colo[i], line = -i)

# HP (violet) appears visually "rounder" — i.e., it exhibits lower curvature
# and longer monotonicity intervals. Before investigating this stylised fact
# further, we compute additional performance measures.

# Both smoothers have the same HT (rate of zero-crossings)
# But they differ strongly in shape, scale and dynamics
# The SSA profile maximizes tracking of x_{t+delta}=eps_{t+delta}-eps_{t+delta-1} 
#   (target correlation, sign accuracy)
# In contrast, HP minimizes curvature
# SSA controles the rate of zero-crossings by generating large noisy cycles that 
# drift away from the zero line for longer time intervals. Once at the zero-line, 
# multiple crossings may occur due to noisiness.

# HP, on the other hand, controls HT by constraining curvature. This imposes 
# extraneous (cyclical) structure upon the data that is not part of the data generating process. 

# Neither solution is practically relevant because it is applied to (1-B)epsilon_t 
# a non-invertible MA which emphasizes high-frequency oscillations.

# But the example illustrates the modus operandi of both smoothness concepts by 
# highlighting their specific traits through an extreme (unrealistic) case study.

# ─────────────────────────────────────────────────────────────────────────────
# 2.7  Distribution of Crossings: Mean and Standard Deviation
# ─────────────────────────────────────────────────────────────────────────────





# Identify time points at which the SSA output changes sign
ssa_zc <- which(output_mat_diff[1:(nrow(output_mat_diff) - 1), "SSA-diff"] *
                  output_mat_diff[2:(nrow(output_mat_diff)),     "SSA-diff"] < 0)

# Identify time points at which the HP output changes sign
hp_zc  <- which(output_mat_diff[1:(nrow(output_mat_diff) - 1), "HP-diff"] *
                  output_mat_diff[2:(nrow(output_mat_diff)),      "HP-diff"] < 0)

# ── Mean durations between consecutive zero-crossings (empirical HTs) ────────
mean(diff(ssa_zc))   # SSA empirical HT
mean(diff(hp_zc))    # HP  empirical HT
# Both means should match the imposed HT target: ht1
ht1

# ── Standard deviations of inter-crossing durations ──────────────────────────
sd(diff(ssa_zc), na.rm = TRUE)   # SSA: variability in zero-crossing spacing
sd(diff(hp_zc),  na.rm = TRUE)   # HP:  variability in zero-crossing spacing

# Zero-crossings of SSA tend to be clustered and the standard deviation is much larger than HP



# ─────────────────────────────────────────────────────────────────────────────
# 2.8 Verify TP-rate on Cumsum
# ─────────────────────────────────────────────────────────────────────────────
# Compare HP(xt) with cumsum(ssa) optimized for xt-x_{t-1}
mat<-cbind(output_mat[,"HP"],cumsum(output_mat_diff[,"SSA-diff"]))

par(mfrow=c(1,1))
colo <- c("blue", "violet")
mplot<-scale(mat[1000:100000,])
colnames(mplot)<-c("HP in levels","Cumsum SSA-diff")
ts.plot(mplot, col = colo)
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)



# Check: mean length between consecutive TPs of cumsum-ssa 
lenq/length(which(diff(mat[1:(nrow(mat)-1),1])*diff(mat[2:(nrow(mat)),1])<0))
# mean length between consecutive TPs of original HP of exercise 1. 
lenq/length(which(diff(mat[1:(nrow(mat)-1),2])*diff(mat[2:(nrow(mat)),2])<0))
# Imposed HT in this exercise
ht1

# All three numbers match: we have succeeded in designing a SSA smoother 
# that replicates the rate of TPs of HP in exercise 1

#------------------------------------------------------------
# Compute ht_i when integrating SSA from x_t=eps_t-eps_{t-1} to eps_t

# The convolution with the integration operator gives bk_mat_diff_x
# The difference vanishes
conv_with_unitroot_func(bk_mat_diff)$conv-bk_mat_diff_x

# Compute the HT of this filter
compute_holding_time_func(conv_with_unitroot_func(bk_mat_diff_x)$conv)


# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways
# ─────────────────────────────────────────────────────────────────────────────
# 1. For a given HT constraint, SSA maximises target correlation and hence
#    sign accuracy (equivalently, minimises MSE after suitable calibration).
#
# 2. Imposing the HP holding time on SSA guarantees that SSA outperforms
#    HP in tracking x_{t + delta}.
#
# 3. The gain in tracking accuracy comes at the cost of larger curvature
#    (greater mean squared second-order differences) relative to HP, which
#    minimises curvature by construction (WH optimality).
#
# 4. These results extend straightforwardly from white noise to arbitrary
#    stationary dependent processes by specifying the appropriate xi (Wold decomposition).
# ─────────────────────────────────────────────────────────────────────────────






















# ════════════════════════════════════════════════════════════════════════════════
# Exercise 3:  COMPUTE THE FULL SSA SMOOTHER FAMILY (delta SWEEP) KEEPING HT FIXED
# ════════════════════════════════════════════════════════════════════════════════
# Sweep delta from the fully causal end (-(L-1)/2) to nowcast (0),
# designing one  filter per lag shift. This may be time-consuming;
# set recompute_calculations = TRUE only when a fresh run is needed.

# 3.1 HP cannot intrinsivcally incapable of maintining fixed smoothing twoarfs sample end


# 3.2
# Re-extract HP holding-time parameters for the sweep
rho1 <- compute_holding_time_func(hp_target)$rho_ff1
ht1  <- compute_holding_time_func(hp_target)$ht
  
gamma_target <- 1   # Allpass target
filt_mat     <- NULL
  
# Design one  filter for each lag delta
for (delta in (-(L - 1) / 2):0)
{
  print(delta)
  SSA_obj  <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1)
  filt_mat <- cbind(filt_mat, SSA_obj$bk_mat)
}
  


# ══════════════════════════════════════════════════════════════════════════════
# 3.3 PLOT THE FULL SSA SMOOTHER FAMILY
# ══════════════════════════════════════════════════════════════════════════════

# --- Scale all filters to unit variance for a fair visual comparison ---
filt_mat <- scale(filt_mat, center = FALSE, scale = TRUE) / sqrt(L - 1)

# Verify unit L2 norm after scaling (each column sum of squares ≈ 1)
apply(filt_mat^2, 2, sum)

# --- Plot all smoother coefficient profiles, coloured by delta ---
plot(
  filt_mat[, 1],
  col  = rainbow(ncol(filt_mat))[1],
  main = "SSA Smoothers",
  axes = FALSE, type = "l",
  ylab = "", xlab = expression(delta)
)
for (i in 2:ncol(filt_mat))
  lines(filt_mat[, i], col = rainbow(ncol(filt_mat))[i])

# Mark the filter centre (lag 0 position)
abline(v = (L - 1) / 2 + 1)

axis(1,
     at     = 1 + c(1, 50 * 1:(nrow(filt_mat) / 50)),
     labels = c(0,    50 * 1:(nrow(filt_mat) / 50)))
axis(2)
box()


# ══════════════════════════════════════════════════════════════════════════════
# 3.4 OVERLAY PLOT — SYMMETRIC SSA VS HP FILTER COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════════

coloh <- c("blue", "violet", "black")

plot(
  mplot[, 1],
  main  = "Symmetric SSA and HP Smoothers",
  axes  = FALSE, type = "l",
  xlab  = "", ylab = "",
  ylim  = c(min(na.exclude(mplot)), max(na.exclude(mplot))),
  col   = coloh[1], lwd = 2, lty = 1
)

# Annotate each filter in its own colour
mtext(paste("SSA(", round(ht1,   2), ",", delta, ")   ", sep = ""),
      col = coloh[1], line = -1)
mtext(paste("SSA(", round(ht1_1, 2), ",", delta, ")   ", sep = ""),
      col = coloh[2], line = -2)

lines(mplot[, 2], col = coloh[2], lwd = 2, lty = 1)
lines(mplot[, 3], col = coloh[3], lwd = 2, lty = 1)

mtext(paste("HP(", lambda_HP, ")   ", sep = ""),
      col = coloh[3], line = -3)

axis(1,
     at     = c(1, 50 * 1:(nrow(mplot) / 50)),
     labels = c(0,    50 * 1:(nrow(mplot) / 50)))
axis(2)
box()







# --- Compute the spectral radius of M-tilde (see Wildi (2026b)---
# rho_max is the largest eigenvalue of M_tilde; it defines the upper bound
# for the HT constraint parameter rho for a filter of length L.
M_obj    <- M_func(L, Sigma)
M_tilde  <- M_obj$M_tilde
rho_max  <- max(eigen(M_tilde)$values)   # Maximum achievable rho
v1       <- eigen(M_tilde)$vectors[, 1]  # Leading eigenvector of M_tilde (to the max eigenvalue rho_max)




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







