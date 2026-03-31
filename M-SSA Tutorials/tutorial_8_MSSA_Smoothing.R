# This tutorial is under construction

# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 8: M-SSA SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# OVERVIEW
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
# In principle, smoothing arises for any causal target specification.
#   • When the target is the causal MSE predictor, M-SSA serves as its smoother 
#     (so-called customization).
#
# Selecting the identity target x_t (or x_{t+δ} with δ ≤ 0) in this tutorial 
# isolates and reveals the INTRINSIC SMOOTHING PROPERTIES of M-SSA, unconfounded 
# by any pre-filtering introduced through a non-trivial target specification.
#
# ──────────────────────────────────────────────────────────────────────────────
# DATA-GENERATING PROCESS
# ──────────────────────────────────────────────────────────────────────────────
# The Wold decomposition of the input process — applied to the differenced
# series x_t when the original I_t is non-stationary — enters M-SSA via the
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
# series (cf. the typical spectral shape).
#
# If, however, the white-noise assumption is violated and ξ is left unspecified,
# increasing the HT still enforces greater smoothness, but only in a relative
# sense: the HT no longer carries a direct interpretation as the mean duration
# between consecutive mean-crossings of the output, and smoothness cannot be
# anchored to any operationally meaningful phenomenon.
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
#
# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK IN TUTORIAL 8
# ══════════════════════════════════════════════════════════════════════════════
# M-SSA smoothing is benchmarked against classical Whittaker–Henderson (WH)
# graduation, which generalises the HP filter. The comparison focuses on
# smoothing performance and tracking ability.
#
# ──────────────────────────────────────────────────────────────────────────────
# Classical Smoothing: Whittaker–Henderson (WH) Graduation / HP Filter
# ──────────────────────────────────────────────────────────────────────────────
#   • The HP filter is the solution to the WH optimisation problem (WH graduation
#     of order two), where the penalty term targets squared second-order differences.
#       → Smoothness criterion: CURVATURE
#   • HP maximizes tracking of x_t subject to a curvature constraint.
#   • Penalising curvature in x_t (first differences) addresses the occurrence 
#     of inflection points in I_t (levels).
#
# ──────────────────────────────────────────────────────────────────────────────
# (M-)SSA Smoothing: Core Concept
# ──────────────────────────────────────────────────────────────────────────────
#   • (M-)SSA controls smoothness via the holding-time (HT) constraint rather
#     than curvature, addressing passages above or below the series mean — 
#     equivalently, between sign changes when the series is zero-mean.
#
#   • Controlling the HT via M-SSA governs the frequency of transitions between
#     above- and below-average growth on levels (I_t).
#   
#       
# ══════════════════════════════════════════════════════════════════════════════
# THEORETICAL REFERENCES
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
# EXERCISE 1: Univariate Symmetric Smoothers
# GOAL
# ════
# Benchmark (univariate) SSA smoothing against Whittaker–Henderson (WH) graduation,
# which generalises the HP filter (assuming second-order differences, d = 2).
#
# BACKGROUND  (Wildi 2026b, Section 4.2)
# ═══════════

# COMPARISON STRATEGY 
# ════════════════════════════════════
# Two complementary designs are evaluated against WH (or HP):
#
#   Exercise 1.1 — Match holding time of HP:
#     Design SSA  so that its HT equals that of HP.
#     Then compare the MSE tracking and curvature of both smoothers.
#     Expected result: SSA  achieves better MSE tracking (closer to x_t)
#     for the same HT but larger curvature.

#   Exercise 1.2 — Match MSE tracking of x_t:
#     Design SSA  so that its MSE tracking of x_t equals that of HP.
#     Then compare the HT and curvature of both smoothers.
#     Expected result:  SSA achieves a longer HT (lss zero-crossings)
#     for the same tracking accuracy, but larger curvature.



# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1.1: SSA vs. HP — Match Holding TimeS
# ══════════════════════════════════════════════════════════════════════════════
# Design SSA with the same HT as the symmetric HP filter, then compare
# tracking accuracy (target correlation) and curvature of both smoothers.
# Expected outcome: SSA outperforms HP in target correlation (and hence
# sign accuracy / MSE) for the same degree of smoothness.
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.1  Specify the HP Filter
# ─────────────────────────────────────────────────────────────────────────────

lambda_HP <- 14400   # Monthly HP smoothing parameter (standard business-cycle value)
L         <- 201     # Filter length (number of coefficients)

# Compute the bi-infinite HP target filter and its MSE-optimal finite approximation
HP_obj    <- HP_target_mse_modified_gap(L, lambda_HP)
hp_target <- HP_obj$target    # Bi-infinite HP filter coefficients (length L)
hp_trend  <- HP_obj$hp_trend  # Associated HP trend estimate


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.2  Specify SSA Design Settings
# ─────────────────────────────────────────────────────────────────────────────

# Sigma: variance–covariance matrix of the innovation process.
#        NULL → univariate design; identity matrix assumed.
Sigma <- NULL

# xi: spectral density of the input process.
#     NULL → white-noise assumption (xi = Identity).
xi <- NULL

# symmetric_target: FALSE → causal (one-sided) target filter.
symmetric_target <- FALSE

# --- Target filter ---
# gamma_target = 1 → allpass (identity) target: SSA tracks x_t itself.
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
# in target correlation (and hence sign accuracy / MSE).

# Extract the HT and the corresponding rho from the bi-infinite HP filter
rho1 <- compute_holding_time_func(hp_target)$rho_ff1  # rho implied by HP's HT
ht1  <- compute_holding_time_func(hp_target)$ht       # Holding time of HP filter

# Interpretation of ht1:
#   The symmetric HP filter applied to white noise produces sign changes with
#   a mean inter-crossing duration of approximately 60 time steps.
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
ts.plot(bk_mat, main = "SSA Filter Coefficients (Match HT of HP)")

# --- Optimisation diagnostics ---
# crit_rhoy_target: maximised target correlation achieved by SSA.
#                   This should exceed the HP target correlation.
SSA_obj$crit_rhoy_target

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
lenq <- 100000
set.seed(86)
x <- rnorm(lenq)

# --- Apply both symmetric (two-sided) filters to x ---
# sides = 2 specifies an acausal two-sided (symmetric) convolution,
# consistent with the symmetric filter design (delta = -(L-1)/2).
y_ssa    <- filter(x, bk_mat,    sides = 2)  # M-SSA smoother output
y_hp_two <- filter(x, hp_target, sides = 2)  # HP smoother output

# Target: since both filters are symmetric and acausal, the target is
# the unshifted series x (no lead or lag adjustment required).
target <- x


# ── a. MSE ───────────────────────────────────────────────────────────────────
# Compute MSE of each smoother relative to the target series.

mse_ssa_smooth <- mean((target - y_ssa)^2,    na.rm = TRUE)
mse_hp_smooth  <- mean((target - y_hp_two)^2, na.rm = TRUE)

# Expected result: M-SSA achieves a smaller MSE than HP.
mse_ssa_smooth   # M-SSA MSE
mse_hp_smooth    # HP MSE


# ── b. Target Correlation ─────────────────────────────────────────────────────
# Compute the correlation matrix of the target and both filter outputs.

output_mat <- na.exclude(cbind(target, y_ssa, y_hp_two))
colnames(output_mat)<-c("x","SSA","HP")
cor_mat    <- cor(output_mat)
cor_mat    # Full correlation matrix (first row: correlations with target)

# Compare M-SSA and HP target correlations:
# Expected result: M-SSA correlates more strongly with the target than HP.
cor_mat[1, 2]   # M-SSA target correlation
cor_mat[1, 3]   # HP target correlation

# --- Internal convergence check ---
# The sample correlation cor(x, y_ssa) should equal crit_rhoy_target
# (up to Monte Carlo sampling error), confirming correct optimisation.
cor_mat[1, 2]             # Empirical correlation: target vs. M-SSA output
SSA_obj$crit_rhoy_target  # Optimised objective (true target correlation)
# The sample correlation should asymptotically converge to the true target correlation if the model is correctly specified (here: white noise)
# Hence maximizing the true target correlation within SSA is a meaningful criterion

# ── c. Sign Accuracy ──────────────────────────────────────────────────────────
# Sign accuracy: proportion of time steps where the filter output and the
# target share the same sign. A higher value indicates fewer false alarms.
# Expected result: M-SSA achieves higher sign accuracy than HP.

sign_acc_ssa <- sum((target * y_ssa)    > 0, na.rm = TRUE) /
  length(na.exclude(target * y_ssa))

sign_acc_hp  <- sum((target * y_hp_two) > 0, na.rm = TRUE) /
  length(na.exclude(target * y_hp_two))

sign_acc_ssa   # M-SSA sign accuracy
sign_acc_hp    # HP sign accuracy


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.6  Results: Smoothness
# ─────────────────────────────────────────────────────────────────────────────
# Smoothness is evaluated on two complementary criteria:
#   a) Holding time (HT): mean duration between consecutive sign changes.
#   b) Curvature: root mean squared second-order differences (RMSD2).
# ─────────────────────────────────────────────────────────────────────────────

# ── a. Holding Time ───────────────────────────────────────────────────────────
# The HT constraint imposed on M-SSA is set equal to that of the HP filter.
# The empirical HTs of both smoothers should therefore agree closely.

ht1                              # Target HT imposed on M-SSA (= HP holding time)
compute_empirical_ht_func(y_ssa)      # Empirical HT of M-SSA output
compute_empirical_ht_func(y_hp_two)   # Empirical HT of HP output

# Expected result: both empirical HTs match each other, i.e., SSA matches the target ht1,
# confirming that the HT constraint is binding and correctly enforced by the optimiser.


# ── b. Curvature (Root Mean Squared Second-Order Differences) ─────────────────
# Curvature is measured as the root mean squared second-order difference
# (RMSD2) of each smoother's output — the natural smoothness criterion
# minimised by HP under the WH framework.
# Expected result: HP achieves the smallest RMSD2 by construction (WH
# optimality), while M-SSA incurs a larger curvature as the cost of
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
# 1.1.7  Plot Series
# ─────────────────────────────────────────────────────────────────────────────

# --- Time-series plot of a 1 000-observation window ---
par(mfrow=c(1,1))
colo <- c("blue", "violet")
ts.plot(na.exclude(output_mat)[1000:2000, 2:3], col = colo)
abline(h = 0)
for (i in 1:ncol(output_mat[,2:3]))
  mtext(colnames(output_mat[,2:3])[i],col=colo[i],line=-i)


# The smaller curvature of HP suggests a visually smoother output than M-SSA.
# However, both smoothers exhibit the same rate of mean-crossings (identical
# empirical HT), so they are equally smooth in the sense that matters most
# for sign-based decision-making: the mean distance between consecutive
# turning points in the levels series.
#
# The apparent visual roughness of the SSA output — reflected in more
# frequent sign changes of the slope (shorter monotonicity intervals in
# the smoothed growth series) — is not a deficiency but a necessary feature.
# It is precisely this additional variation that allows M-SSA to track the
# target x_t (growth of the original levels series) more closely than HP,
# as measured by MSE, target correlation, and sign accuracy.
#
# In summary: for a given HT constraint — equivalently, a prescribed mean
# duration between consecutive turning points in the levels series — SSA
# achieves superior tracking of level dynamics (in first differences) or growth dynamics (on levels)
# relative to HP. The trade-off is a larger curvature (more inflection
# points in levels), but the number of inflection points is operationally
# inconsequential as long as the number of turning points is held fixed.
#
# Stated differently: the curvature-based smoothness concept enforced by
# WH/HP does not explicitly control the distance between turning points in
# levels — the quantity of primary interest in sign-based applications.
# Consequently, the cost of not controlling curvature, as in SSA, is
# mitigated in settings where turning-point control is the
# primary objective. 

# From a visual standpoint, HP appears smoother — an impression
# driven by its lower curvature and oscillatory ACF structure. However, this visual
# impression should not distract from the effective optimisation objective:
# when turning-point control is the primary operational priority, the HT
# constraint is the more relevant smoothness criterion, and SSA is the
# more efficient filter by construction.










# ─────────────────────────────────────────────────────────────────────────────
# 1.1.8  Monotonicity
# ─────────────────────────────────────────────────────────────────────────────
# Definitions:
# A turning point is where a graph changes direction (from increasing to decreasing, or vice versa), 
#   acting as a local maximum or minimum. 
# An inflection point is where the graph's curvature 
#   (concavity) changes, often where the slope changes from bending downward to bending upward, 
# not necessarily changing direction. 
# ─────────────────────────────────────────────────────────────────────────────
# We here compute the mean duration between consecutive turning points (in levels)
# A turning point is obtained when growth (first differences of the filtered series) changes sign
# ─────────────────────────────────────────────────────────────────────────────

# Locations at which first differences of SSA change sign: turning points
ssa_tp<-which(diff(output_mat[1:(nrow(output_mat)-1),"SSA"])*diff(output_mat[2:(nrow(output_mat)),"SSA"])<0)
hp_tp<-which(diff(output_mat[1:(nrow(output_mat)-1),"HP"])*diff(output_mat[2:(nrow(output_mat)),"HP"])<0)

# Mean duration between turning points
nrow(output_mat)/length(ssa_tp)
nrow(output_mat)/length(hp_tp)



# ─────────────────────────────────────────────────────────────────────────────
# NOTE ON SMOOTHNESS CRITERIA
# ─────────────────────────────────────────────────────────────────────────────
# The preceding discussion is not specifically about the HP filter per se,
# but about the broader concept of smoothness and how it is formalised.
#
# In the WH framework, smoothness is enforced by penalising "unsmooth"
# behaviour through a regularisation term — typically squared differences
# of order d. HP is a special case of WH with d = 2, penalising curvature
# (squared second-order differences).
#
# The key question is therefore which notion of smoothness is most
# appropriate for a given application — curvature-based (WH/HP) or
# mean-crossing-based (M-SSA) — rather than a comparison between two
# specific filters.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# INTERPRETATION: TURNING POINTS, INFLECTION POINTS, AND SMOOTHNESS CRITERIA
# ─────────────────────────────────────────────────────────────────────────────
#
# Contextual mapping to macroeconomic / financial data
# ─────────────────────────────────────────────────────
# In practice, the simulated white-noise series x_t represents first
# differences (growth rates) of a non-stationary financial or macroeconomic
# series. Under this interpretation:
#
#   • Zero-crossings of x_t (first differences)
#       ↔ turning points of the original series in levels.
#   • Turning points of x_t (first differences)
#       ↔ inflection points of the original series in levels.
#
# The relevant smoothness question
# ─────────────────────────────────
# The appropriate smoothness criterion depends on the operational objective:
#
#   • If the primary interest is in turning points in levels
#     (zero-crossings in first differences), then controlling the HT —
#     rather than curvature — is the more relevant smoothness criterion.
#     For a given HT, tracking x_t (growth) more closely is a worthwhile
#     objective (SSA).
#
#   • If the primary interest is in controlling the rate of inflection points in levels
#     (turning points in first differences), then curvature-based criteria
#     such as WH/HP are more appropriate.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE PRIMAL PERSPECTIVE: FIX HT, MAXIMISE GROWTH TRACKING
# ─────────────────────────────────────────────────────────────────────────────
# The series x_t (first differences) is an natural unbiased but noisy estimate of
# growth in the underlying levels series. For a prescribed mean distance
# between consecutive turning points in levels (i.e., a fixed HT in differences), SSA
# maximises tracking of x_t — the natural, unbiased growth signal.
#
# This constitutes a compelling and operationally meaningful criterion:
# it simultaneously controls the business-cycle frequency (via HT) and
# minimises noise in the growth estimate (via target correlation / MSE).
#
# Example — Business-cycle analysis:
#   Business cycles are conventionally defined over durations of 2–8 years,
#   with a typical mean cycle length of approximately 5 years. Imposing an
#   HT of 5 years in the SSA design would yield the closest possible
#   tracking of the unbiased (but noisy) growth estimate x_t, while
#   ensuring that the smoothed output generates turning-point signals at
#   the prescribed frequency.
# ─────────────────────────────────────────────────────────────────────────────

#
#
# The dual perspective: fixing MSE, maximising HT
# ─────────────────────────────────────────────────
# The argument can be reversed: suppose the tracking ability (MSE, target
# correlation, or sign accuracy) of x_t is fixed a priori. A natural
# complementary smoothness objective is then to maximise the HT subject to
# this tracking constraint — directly controlling the distance between
# consecutive turning points in levels. This combines:
#
#   • An MSE criterion on growth (first differences), and
#   • An explicit turning-point control on levels.
#
# Exercise 1.2 below explores exactly this dual formulation.
#
# Why M-SSA smoothness differs from HP smoothness
# ─────────────────────────────────────────────────
# Once the filtered series is clearly away from zero, noisy ripples
# (turning points in first differences, i.e., inflection points in levels)
# are operationally harmless — they do not generate false turning-point
# signals. The visual roughness of M-SSA at non-zero levels is therefore
# inconsequential in sign-based (turning points in levels) applications.
#
# What matters is the behaviour near zero: spurious zero-crossings at this
# boundary generate noisy turning-point signals. M-SSA controls precisely
# this rate — the frequency of zero-crossings — via the HT constraint.
#
# HP, by contrast, controls curvature (turning-point rate) uniformly across
# all levels, including regions far from zero where such control is
# operationally unnecessary. This makes HP's smoothness criterion less
# targeted in sign-based (turning points in levels) decision-making applications.
# ─────────────────────────────────────────────────────────────────────────────











# ─────────────────────────────────────────────────────────────────────────────
# 1.1.9  Dependence Structure: Autocorrelation Functions
# ─────────────────────────────────────────────────────────────────────────────
# The autocorrelation function (ACF) provides an alternative perspective on
# the structural differences between M-SSA and HP smoothing outputs.
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(2, 1))

# M-SSA output: slowly and monotonically decaying ACF, indicating long memory
# in the smoothed series and an acyclical dependence structure.
acf(na.exclude(output_mat)[, 2], lag.max = 100, main = "M-SSA")

# HP output: faster-decaying ACF with a cyclical pattern.
# The half-period of the oscillation (≈ 57 lags) is consistent with
# the HP holding time.
acf(na.exclude(output_mat)[, 3], lag.max = 100, main = "HP")

# Key differences between M-SSA and HP autocorrelation structures:
#   a) Shape : M-SSA produces an acyclical, monotonically decaying ACF,
#              whereas HP exhibits a cyclical (oscillatory) ACF pattern.
#   b) Memory: M-SSA decays more slowly than HP, indicating stronger and
#              more persistent serial dependence in the smoothed output.



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




# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1.2: SSA vs. HP — Match target Correlation (MSE)
# ══════════════════════════════════════════════════════════════════════════════

# 1.2.1 Compute SSA
# Strategy 2: find the HT such that SSA achieves the same target correlation
#             as HP. Under identical correlation,  SSA will outperform HP in HT.

ht1_1  <- 75                                    # Target HT to match HP correlation
rho1_1 <- compute_rho_from_ht(ht1_1)$rho        # Corresponding rho value

# Design  filter with the correlation-matched HT constraint
SSA_obj_1 <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1_1)

# Extract filter coefficients and verify correlation
bk_mat_1 <- SSA_obj_1$bk_mat
ts.plot(bk_mat_1)
SSA_obj_1$crit_rhoy_target   # Should approximately equal cor(output_mat)[1, 3]
# Check:  replicates (sample) target correlation of HP
cor(output_mat)[1, 3]

# ══════════════════════════════════════════════════════════════════════════════
# 1.2.2 Compute Curvatures
# ══════════════════════════════════════════════════════════════════════════════
# All three filters are scaled to unit variance before comparison.
# HP minimises mean squared second differences by construction (WH criterion).
#???????????????why scaling??????????????
# Scale all filters to unit variance
scaled_filters <- scale(cbind(bk_mat, bk_mat_1, hp_target),
                        center = FALSE, scale = TRUE) / sqrt(L - 1)
yhat_mat <- NULL
for (i in 1:ncol(filter_mat))
  yhat_mat <- cbind(yhat_mat, filter(x, scaled_filters[, i], sides = 2))

filter_mat <- cbind(bk_mat, bk_mat_1, hp_target)

# Apply each  filter to x
yhat_mat <- NULL
for (i in 1:ncol(filter_mat))
  yhat_mat <- cbind(yhat_mat, filter(x, filter_mat[, i], sides = 2))

# Assign descriptive column names
colnames(scaled_filters)<-colnames(yhat_mat) <- c(
  paste("SSA(", round(ht1,   2), ",", delta, ")", sep = ""),
  paste("SSA(", round(ht1_1, 2), ",", delta, ")", sep = ""),
  "HP"
)

# Compute mean squared second-order differences for each smoother
# HP is expected to achieve the smallest value by the WH optimality criterion
sq_se_dif <- apply(
  apply(apply(na.exclude(yhat_mat), 2, diff), 2, diff)^2,
  2, mean
)
sq_se_dif

# Outcome: as expected, HP minimizes the curvature among all three designs


# ══════════════════════════════════════════════════════════════════════════════
# 1.2.3 Plot and Sample Performances
# ══════════════════════════════════════════════════════════════════════════════

# --- Time-series plot of a 1 000-observation window ---
colo <- c("blue", "violet", "black")
ts.plot(na.exclude(yhat_mat)[1000:2000, ], col = colo)
abline(h = 0)
for (i in 1:ncol(yhat_mat))
  mtext(colnames(yhat_mat)[i],col=colo[i],line=-i)

# --- Autocorrelation functions ---
#  filters: slowly decaying ACF (long memory in the smooth output)
acf(na.exclude(yhat_mat)[, 1], lag.max = 100)
acf(na.exclude(yhat_mat)[, 2], lag.max = 100)

# HP filter: faster decay with cyclical structure; half-period ≈ 57 lags,
# consistent with its holding time
acf(na.exclude(yhat_mat)[, 3], lag.max = 100)

# Diagnostics: comparison of sample performances

# Empirical holding times:  should equal or exceed HP's HT
apply(na.exclude(yhat_mat), 2, compute_empirical_ht_func)

# Tracking ability (correlation with x):  should equal or exceed HP
cor(na.exclude(cbind(x, yhat_mat)))[1, ]


# ════════════════════════════════════════════════════════════════════════════════
# Exercise 1.3: Replicate Symmetric HP
# ════════════════════════════════════════════════════════════════════════════════
# Tutorial 2 illustrated how SSA can replicate HP by specifying the target accordingly
# However, the target in this tutorial is x_t
# Can SSA replicate HP when targeting x_t and under which conditions?
# Answer: xi=hp_two




# ════════════════════════════════════════════════════════════════════════════════
# Exercise 2: TARGET MONOTONICITY IN SSA
# ════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Specify SSA Design Settings
# ─────────────────────────────────────────────────────────────────────────────

# Sigma: variance–covariance matrix of the innovation process.
#        NULL → univariate design; identity matrix assumed.
Sigma <- NULL

# xi: spectral density of first differences of the input (white noise) process.
xi <- c(1,-1)

# symmetric_target: FALSE → causal (one-sided) target filter.
symmetric_target <- FALSE

# --- Target filter ---
# gamma_target = 1 → allpass (identity) target: SSA tracks x_t itself.
# This contrasts with signal-extraction or nowcasting settings where the
# target is a non-trivial filtered version of x_t (e.g., the HP trend or an ideal trend).
# MSSA_func zero-pads gamma_target to length L automatically if needed.
gamma_target <- c(1,-1)

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
# 2.2  Specify the HT Constraint — Match HP 
# ─────────────────────────────────────────────────────────────────────────────
# Impose the same HT as the two-sided HP filter.
# Both filters are symmetric (delta = -(L-1)/2), so a like-for-like
# comparison is valid.
# Under an identical HT constraint, SSA is guaranteed to outperform HP
# in target correlation (and hence sign accuracy / MSE).

hp_target_diff<-c(hp_target,0)-c(0,hp_target)

ts.plot(hp_target_diff,main="Anti-symmetric two-sided HP in first differences")

rho1 <- compute_holding_time_func(hp_target_diff)$rho_ff1  # rho implied by HP's HT
ht1  <- compute_holding_time_func(hp_target_diff)$ht       # Holding time of HP filter

# Interpretation of ht1:
#   This reflects the mean duration between consecutive zero-crossings of first
#   differences of HP: the mean length of monotonous cycle phases
# This duration is shorter than the HT in levels in exercise 1

ht1   # Inspect the HP holding time


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Compute the SSA Filter
# ─────────────────────────────────────────────────────────────────────────────
# Design the SSA filter subject to the HP-matched HT constraint.
# Note: xi = NULL assumes white-noise input; MSSA_func will issue a warning
#       confirming that gamma_target has been zero-padded to length L.

SSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho1,xi)

# Extract the optimised (univariate) filter coefficients
bk_mat_diff <- SSA_obj$bk_mat

# --- Visual inspection ---
# The filter should be symmetric, consistent with delta = -(L-1)/2
par(mfrow = c(2, 1))
ts.plot(bk_mat_diff, main = "SSA Filter Coefficients (Match HT of HP)")
ts.plot(hp_target_diff,main="Anti-symmetric two-sided HP in first differences")


SSA_obj$crit_rhoyy
rho1

# --- Optimisation diagnostics ---
# crit_rhoy_target: maximised target correlation achieved by SSA.
#                   This should exceed the HP target correlation.
SSA_obj$crit_rhoy_target




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
x <- rnorm(lenq)
x_diff<-diff(x)

#????????? Explain why we apply to xt and not xt-xt-1
# --- Apply both symmetric (two-sided) filters to x ---
# sides = 2 specifies an acausal two-sided (symmetric) convolution,
# consistent with the symmetric filter design (delta = -(L-1)/2).
y_ssa_diff    <- filter(x, bk_mat_diff,    sides = 2)  # M-SSA smoother output
y_hp_two_diff <- filter(x, hp_target_diff, sides = 2)  # HP smoother output

ts.plot(cbind(y_ssa_diff,y_hp_two_diff)[1000:1500,],col=c("blue","violet"))
# Target: since both filters are symmetric and acausal, the target is
# the unshifted series x (no lead or lag adjustment required).
target <- c(0,x_diff)


# ── a. MSE ───────────────────────────────────────────────────────────────────
# Compute MSE of each smoother relative to the target series.

mse_ssa_smooth <- mean((target - y_ssa_diff)^2,    na.rm = TRUE)
mse_hp_smooth  <- mean((target - y_hp_two_diff)^2, na.rm = TRUE)

# Expected result: M-SSA achieves a smaller MSE than HP.
mse_ssa_smooth   # M-SSA MSE
mse_hp_smooth    # HP MSE


# ── b. Target Correlation ─────────────────────────────────────────────────────
# Compute the correlation matrix of the target and both filter outputs.

output_mat_diff <- na.exclude(cbind(target, y_ssa_diff, y_hp_two_diff))
# Append a zero to match the length of the series in exercise 1
output_mat_diff<-rbind(rep(0,ncol(output_mat_diff)),output_mat_diff)
colnames(output_mat_diff)<-c("x-diff","SSA-diff","HP-diff")

cor_mat    <- cor(output_mat_diff)
cor_mat    # Full correlation matrix (first row: correlations with target)

# Compare M-SSA and HP target correlations:
# Expected result: M-SSA correlates more strongly with the target than HP.
cor_mat[1, 2]   # M-SSA target correlation
cor_mat[1, 3]   # HP target correlation

# --- Internal convergence check ---
# The sample correlation cor(x, y_ssa) should equal crit_rhoy_target
# (up to Monte Carlo sampling error), confirming correct optimisation.
cor_mat[1, 2]             # Empirical correlation: target vs. M-SSA output
SSA_obj$crit_rhoy_target  # Optimised objective (true target correlation)
# The sample correlation should asymptotically converge to the true target correlation if the model is correctly specified (here: white noise)
# Hence maximizing the true target correlation within SSA is a meaningful criterion

# ── c. Sign Accuracy ──────────────────────────────────────────────────────────
# Sign accuracy: proportion of time steps where the filter output and the
# target share the same sign. A higher value indicates fewer false alarms.
# Expected result: M-SSA achieves higher sign accuracy than HP.

sign_acc_ssa <- sum((target * y_ssa_diff)    > 0, na.rm = TRUE) /
  length(na.exclude(target * y_ssa_diff))

sign_acc_hp  <- sum((target * y_hp_two_diff) > 0, na.rm = TRUE) /
  length(na.exclude(target * y_hp_two_diff))

sign_acc_ssa   # M-SSA sign accuracy
sign_acc_hp    # HP sign accuracy


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  Results: Smoothness
# ─────────────────────────────────────────────────────────────────────────────
# Smoothness is evaluated on two complementary criteria:
#   a) Holding time (HT): mean duration between consecutive sign changes.
#   b) Curvature: root mean squared second-order differences (RMSD2).
# ─────────────────────────────────────────────────────────────────────────────

# ── a. Holding Time ───────────────────────────────────────────────────────────
# The HT constraint imposed on M-SSA is set equal to that of the HP filter.
# The empirical HTs of both smoothers should therefore agree closely.

ht1                              # Target HT imposed on M-SSA (= HP holding time)
compute_empirical_ht_func(y_ssa_diff)      # Empirical HT of M-SSA output
compute_empirical_ht_func(y_hp_two_diff)   # Empirical HT of HP output

# Expected result: both empirical HTs match each other, i.e., SSA matches the target ht1,
# confirming that the HT constraint is binding and correctly enforced by the optimiser.


# ── b. Curvature (Root Mean Squared Second-Order Differences) ─────────────────
# Curvature is measured as the root mean squared second-order difference
# (RMSD2) of each smoother's output — the natural smoothness criterion
# minimised by HP under the WH framework.
# Expected result: HP achieves the smallest RMSD2 by construction (WH
# optimality), while M-SSA incurs a larger curvature as the cost of
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
# 2.6  Plot Series: very interesting!!!!!!!!!!!!!!!!????????????
# SSA is much noisier but makes much higher/longer cycles away from zero
# Zero-crossings tend to cluster much more than for HP
# ─────────────────────────────────────────────────────────────────────────────

# --- Time-series plot of a 1 000-observation window ---
par(mfrow=c(1,1))
colo <- c("blue", "violet")
ts.plot(na.exclude(output_mat_diff)[1000:2000, 2:3], col = colo)
abline(h = 0)
for (i in 1:ncol(output_mat_diff[,2:3]))
  mtext(colnames(output_mat_diff[,2:3])[i],col=colo[i],line=-i)


# The smaller curvature of HP suggests a visually smoother output than SSA.
# However, both smoothers have been verified to exhibit the same rate of
# mean-crossings (identical empirical HT), so they are equally smooth in the
# sense that matters most for sign-based decision-making.
#
# The apparent visual roughness of the SSA output — reflected in more
# frequent sign changes in the growth rate (first differences) — is not a
# deficiency but a necessary feature: it is precisely this additional
# high-frequency variation that allows SSA to track the target x_t more
# closely than HP in terms of MSE, target correlation, and sign accuracy.
#
# In summary: for a given HT constraint, M-SSA achieves superior tracking
# efficiency relative to HP. The trade-off is a larger curvature, but since
# curvature does not directly govern sign-based performance, this cost is
# operationally inconsequential in applications driven by mean-crossings.


#---------------------------------------------------------
# Cumsum

# Compare HP(xt) with cumsum(ssa) optimized for xt-x_{t-1}
mat<-cbind(output_mat[,"HP"],cumsum(output_mat_diff[,"SSA-diff"]))

par(mfrow=c(1,1))
colo <- c("blue", "violet")
mplot<-scale(mat[1000:2000,])
colnames(mplot)<-c("HP in levels","Cumsum SSA-diff")
ts.plot(mplot, col = colo)
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
# Check length between inflection points of cumsum ssa matches HP and ht1
lenq/length(which(diff(mat[1:(nrow(mat)-1),1])*diff(mat[2:(nrow(mat)),1])<0))
lenq/length(which(diff(mat[1:(nrow(mat)-1),2])*diff(mat[2:(nrow(mat)),2])<0))
ht1

# ─────────────────────────────────────────────────────────────────────────────
# 1.1.8  Dependence Structure: Autocorrelation Functions
# ─────────────────────────────────────────────────────────────────────────────
# The autocorrelation function (ACF) provides an alternative perspective on
# the structural differences between M-SSA and HP smoothing outputs.
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(2, 1))

# M-SSA output: slowly and monotonically decaying ACF, indicating long memory
# in the smoothed series and an acyclical dependence structure.
acf(na.exclude(output_mat_diff)[, 2], lag.max = 100, main = "M-SSA")

# HP output: faster-decaying ACF with a cyclical pattern.
# The half-period of the oscillation (≈ 57 lags) is consistent with
# the HP holding time.
acf(na.exclude(output_mat_diff)[, 3], lag.max = 100, main = "HP")

# Key differences between M-SSA and HP autocorrelation structures:
#   a) Shape : M-SSA produces an acyclical, monotonically decaying ACF,
#              whereas HP exhibits a cyclical (oscillatory) ACF pattern.
#   b) Memory: M-SSA decays more slowly than HP, indicating stronger and
#              more persistent serial dependence in the smoothed output.



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
s



# ══════════════════════════════════════════════════════════════════════════════
# Exercise 5: M-SSA Smoothing
# ══════════════════════════════════════════════════════════════════════════════







