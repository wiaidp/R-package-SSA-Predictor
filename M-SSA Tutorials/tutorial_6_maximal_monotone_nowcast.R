# ========================================================================
# Tutorial 7: Extension of SSA to non-stationary (integrated) processes 
#   Introduce I-SSA
# ========================================================================


# SSA typically assumes stationary series

# Optimization objective:
#   Maximize sign accuracy (or target correlation) subject to a holding-time (HT) constraint
#   HT = mean duration between consecutive mean crossings of the predictor
#   For zero-mean processes, HT corresponds to the mean duration between sign changes

# ── BACKGROUND ────────────────────────────────────────────────────
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

# Theoretical background:
# Wildi, M. (2026a)
#   Sign Accuracy, Mean-Squared Error and the Rate of Zero Crossings:
#   a Generalized Forecast Approach
#   https://doi.org/10.48550/arXiv.2601.06547

# ─────────────────────────────────────────────────────────────────


# Tutorials 1–4 are based on Wildi (2024)
# This tutorial builds specifically on Wildi (2026a), which provides:
#   - Formal/theoretical foundations of Wildi (2024)
#   - Introduction of an I-SSA predictor for non-stationary integrated series
# Theory is discussed in sections 5.3 and 5.4
# The example in this tutorial replicates section 5.5


# Non-stationary (integrated) processes are not mean-reverting
# ⇒ The concept of mean duration between mean crossings is not defined in levels

# However:
#   - For I(1) processes, the concept applies to stationary first differences
#   - Moreover, a dual SSA result states that no linear predictor outperforms SSA in HT
#     for a given tracking accuracy (Wildi, 2026a)

# Extension to non-stationary series:
#   - In first differences: SSA maximizes HT among linear predictors
#   - In levels: I-SSA is maximal monotone (Wildi, 2026a)
#   - For I(2): I-SSA minimizes curvature among linear predictors


# Focus of this tutorial:
# Maximal monotone predictors for non-stationary I(1) time series


# I-SSA framework:
#   - Produces predictors that optimally track the target 
#     (max sign accuracy, max target correlation or min MSE)
#   - Predictor and target are cointegrated
#   - Predictor controls zero-crossing rate of first differences (stationary component)

# Interpretation:
#   - Minimize MSE (in levels) subject to a holding-time (HT) constraint on first differences
#       Equivalently: maximize sign-accuarcy or target correlation subject 
#       to unit root constraint (cointegration) and a HT constraint
#   - Dual: minimize mean-crossings (in differences) for a given MSE (in levels)
#   - Equivalent: predictor in levels is MAXIMALLY MONOTONE 


# Application in this tutorial (see Wildi 2026a):
#   - Construct maximal monotone SSA nowcast of two-sided HP(14400)
#     for monthly US industrial production (INDPRO)

# Benchmarks:
#   1) MSE-optimal nowcast using ARIMA(1,1,0)
#      → Typically much noisier than real-time HP filters

#   2) One-sided concurrent HP filter (HP-C)
#      → Standard in business-cycle analysis

# Strategy:
#   - Match SSA holding-time (smoothness) to HP-C
#   - Optimize SSA for best tracking (MSE) under this constraint

# Expected outcome:
#   - SSA ≈ HP-C in smoothness (rate of mean crossings in differences or monotonicity in levels)
#   - SSA outperforms HP-C in MSE (or sign accuracy or target correlation)

# Key question:
#   - How much MSE performance is sacrificed by enforcing smoothness?
#   - SSA minimizes this loss given the holding-time constraint


# ════════════════════════════════════════════════════════════════
# Main Ideas
# ════════════════════════════════════════════════════════════════

# ── I. Tracking Recessions and Expansions ───────────────────────
#
# We use the classic one-sided (concurrent) HP trend — applied to levels,
# not the gap — as a real-time nowcast of the underlying trend.
#
# First differences of this trend nowcast are then interpreted as follows:
#   - Negative readings : trend growth is negative → slowdown / recession
#   - Positive readings : trend growth is positive → recovery / expansion
#
# Zero-crossings of the differenced trend signal the onset or end of
# recession and expansion episodes (see plots below).

# ── II. Alternatives to the Classic One-Sided HP ────────────────
#
# Within this framework, two alternatives to the classic one-sided HP
# trend nowcast are considered:
#   (i)  MSE-optimal nowcast  : minimises mean squared error against the
#                               two-sided HP trend (the target in levels)
#   (ii) I-SSA nowcast        : optimally constrained real-time filter
#
# The MSE-optimal nowcast is highly noisy in first differences, generating
# a continuous stream of false recession/expansion signals and rendering it
# unsuitable for zero-crossing analysis.
#
# I-SSA is therefore designed to satisfy two objectives simultaneously:
#   (a) Smoothness parity : replicate the holding-time (HT) of the classic
#                           one-sided HP filter in first differences
#   (b) Level accuracy    : among all linear predictors sharing the same HT,
#                           I-SSA is the closest to the two-sided HP trend
#                           in levels (MSE-optimal under the HT constraint)

# ── III. The Double-Stroke: Level Nowcasting and Recession Signaling
#
# Pursuing both objectives within a single filter design is non-trivial,
# as they impose conflicting requirements:
#
#   • Nowcasting the trend level demands timeliness:
#       The MSE-optimal filter tracks the level most accurately but is
#       very noisy — smoothness is sacrificed for proximity to the target.
#
#   • Signaling recessions and expansions demands smoothness:
#       Avoiding false signals during sustained expansion or recession
#       episodes requires a filter whose first differences change sign
#       infrequently and meaningfully.
#
# I-SSA reconciles these competing demands through a constrained optimisation:
#   → Track the two-sided HP trend in levels as closely as possible (MSE),
#   → subject to generating sparse, reliable sign changes in first differences
#      that faithfully mark the onset and end of recession/expansion episodes.
#
# This is the "double-stroke": level accuracy and recession signaling,
# achieved within a single, coherent filter design.



# ────────────────────────────────────────────────────────────────
# Initialize session: clear workspace and load required packages
# ────────────────────────────────────────────────────────────────
rm(list = ls())

library(xts)        # Time series handling
library(mFilter)    # HP and BK filters
library(quantmod)   # Data access (FRED)


# ────────────────────────────────────────────────────────────────
# Load custom SSA and signal extraction functions
# ────────────────────────────────────────────────────────────────
source(file.path(getwd(), "R", "simple_sign_accuracy.r"))
source(file.path(getwd(), "R", "HP_JBCY_functions.r"))


# ────────────────────────────────────────────────────────────────
# Load data (option to refresh from FRED)
# ────────────────────────────────────────────────────────────────
reload_data <- FALSE

if (reload_data) {
  getSymbols("INDPRO", src = "FRED")
  save(INDPRO, file = file.path(getwd(), "Data", "INDPRO"))
} else {
  load(file = file.path(getwd(), "Data", "INDPRO"))
}

tail(INDPRO)


# ────────────────────────────────────────────────────────────────
# Sample selection and transformations
# ────────────────────────────────────────────────────────────────
start_year <- 1982
end_year   <- 2024

y      <- as.double(log(INDPRO[paste0(start_year, "/", end_year)]))
y_xts  <- log(INDPRO[paste0(start_year, "/", end_year)])


# ────────────────────────────────────────────────────────────────
# Plot raw data, log-levels, and first differences
# ────────────────────────────────────────────────────────────────
par(mfrow = c(2, 2))

plot(as.double(INDPRO), main = "INDPRO", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = "black", lwd = 1)
axis(1, at = 1:length(INDPRO), labels = index(INDPRO))
axis(2); box()

plot(as.double(y_xts), main = "Log-INDPRO", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = "black", lwd = 1)
axis(1, at = 1:length(y_xts), labels = index(y_xts))
axis(2); box()

plot(as.double(diff(y_xts)), main = "Diff-log", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = "black", lwd = 1)
abline(h = 0)
axis(1, at = 1:length(diff(y_xts)), labels = index(diff(y_xts)))
axis(2); box()

# ACF suggests AR(1) behavior in first differences
acf(na.exclude(diff(y_xts)), main = "ACF of diff-log")

len     <- length(y)
x_tilde <- as.double(y_xts)

# Note:
# Earlier work (Wildi, 2024) focuses on SSA applied to log-differences,
# whereas here we work with I-SSA and log-levels.
# I-SSA must track the two-sided HP filter applied to levels.


# ────────────────────────────────────────────────────────────────
# Model setup for maximal monotone predictor
# ────────────────────────────────────────────────────────────────
# Cointegration constraint ensures finite MSE for integrated processes

# Assume ARIMA(1,1,0) for log-INDPRO (based on ACF)
# Parameters fixed to avoid instability from extreme observations
a1 <- 0.3
b1 <- 0

# Filter design parameters
L         <- 101
lambda_hp <- 14400

# Compute system matrices and filters
filter_obj <- compute_system_filters_func(L, lambda_hp, a1, b1)

B            <- filter_obj$B            # Cointegration matrix, see cited literature
M            <- filter_obj$M            # Autocovariance generator
gamma_tilde  <- filter_obj$gamma_tilde  # Transformed (HP) target
gamma_mse    <- filter_obj$gamma_mse    # MSE-optimal filter (one-sided estimate of HP target)
Xi_tilde     <- filter_obj$Xi_tilde     # Convolution operator (see section 5.3: Wold-decomposition of first differences convolved with integration operator)
Sigma        <- filter_obj$Sigma        # Integration operator (see section 5.3)
Delta        <- filter_obj$Delta        # Differencing operator
Xi           <- filter_obj$Xi           # Wold MA representation in matrix notation, see equation 22 in Wildi 2026a
hp_two       <- filter_obj$hp_two       # Two-sided HP target
hp_trend     <- filter_obj$hp_trend     # Classic one-sided HP (HP-C): benchmark for I-SSA customization

ts.plot(cbind(gamma_tilde,gamma_mse),col=c("black","brown"))
mtext("Optimal MSE filter applied to data in levels",line=-1,col="brown")
mtext("Serves as target in constrained optimization",line=-2)

# Plot Wold decomposition of differenced process: first column in matrix Xi
ts.plot(Xi[,1],main="Wold decomposition of ARMA(1,1)")




# ────────────────────────────────────────────────────────────────
# Holding-Time (HT) Constraint Calibration
# ────────────────────────────────────────────────────────────────
# The HT constraint is defined on first differences (see eq. 29, Wildi 2026a)


# Derivation of the HT of the first-differenced HP predictor:
#
#   Xi       : convolution matrix (eq. 22, Wildi 2026a)
#   Xi_tilde : Xi composed with Sigma (integration operator), see section 5.3
#
#   Xi_tilde %*% hp_trend provides a finite MA representation in levels.
#   Although not the predictor itself (which is non-stationary), this stationary representation is
#   suitable for differencing, as finite and infinite MA filters behave
#   equivalently under first differencing.
#
#   Applying the differencing operator Delta:
#       Delta %*% Xi_tilde %*% hp_trend = Xi %*% hp_trend (check: both expressions are identical)
#
#   Hence, Xi %*% hp_trend is used to compute the holding-time of the
#   differenced HP (the classic trend-nowcast in levels).

HT_HP_obj<-compute_holding_time_func(Xi %*% hp_trend)

# HT
HT_HP_obj$ht
# First-order ACF
HT_HP_obj$rho_ff1

# HT and first-order ACF are in one-to-one correspondence
# (see eq. 18, Wildi 2026a).
# We use rho1 to impose the HT constraint in I-SSA.

rho1 <- rho_hp_concurrent<-HT_HP_obj$rho_ff1

# Interpretation:
# I-SSA is constrained to replicate the holding-time (first-order ACF rho1) of the
# one-sided HP filter in levels. By optimality, the resulting trend
# nowcast is expected to improve upon HP in terms of MSE.


# ────────────────────────────────────────────────────────────────
# Reference: HT of differenced MSE-optimal predictor (MSE-optimal in levels)
# ────────────────────────────────────────────────────────────────
# Using the same derivation logic:
#   Xi %*% gamma_mse characterizes the differenced representation
#   of the MSE-optimal predictor in levels.

rho_mse <- compute_holding_time_func(Xi %*% gamma_mse)$rho_ff1

# Typically, rho_mse is small → frequent zero-crossings
# This reflects the higher noise of MSE-optimal predictors in levels.
# MSE optimality trades off smoothness for timeliness:
# such predictors are generally more reactive but noisier.
rho_mse


# ────────────────────────────────────────────────────────────────
# Motivation for I-SSA
# ────────────────────────────────────────────────────────────────
# - MSE-optimal level predictors are efficient but very noisy
# - They generate excessive sign changes in first differences
#   → unreliable signals of slowdowns/accelerations
#
# - One-sided HP (applied to levels) is smoother:
#   → fewer crossings in differences
#   → better tracking of business-cycle turning points
#   → but MSE performances of the trend nowcast are inferior 
#   
#
# - I-SSA combines both advantages:
#   → replicates HP smoothness (via HT constraint): generate fewer spurious false alarms in first differences
#   → improves tracking of the two-sided HP in levels (lower MSE)
#
# Result:
# A predictor that closely tracks the level while producing
# sparse and informative sign changes in first differences.


# ────────────────────────────────────────────────────────────────
# Compute I-SSA solution (Wildi, 2026a, Sections 5.3–5.4)
# ────────────────────────────────────────────────────────────────
# Use numerical optimization (optim) to determine the optimal
# Lagrange multiplier λ ensuring compliance with the HT constraint.
# Initialization at λ = 0 corresponds to the MSE benchmark.

lambda <- 0

opt_obj <- optim(
  lambda,
  b_optim,
  lambda,
  gamma_mse,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  gamma_tilde,
  rho1
)

# Optimal Lagrange multiplier
lambda_opt <- opt_obj$par

# Compute I(1) cointegrated SSA solution
bk_obj <- bk_int_func(
  lambda_opt,
  gamma_mse,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  gamma_tilde
)

# Diagnostics
bk_obj$rho_yy   # matches rho1 (HT constraint): this verifies convergence of the optimization (if not: increase the number of iterations)
rho1
bk_obj$rho_yz   # correlation with target
bk_obj$mse_yz   # MSE vs MSE-optimal predictor

# Extract filters
b_x   <- bk_obj$b_x     # applied to data
b_eps <- bk_obj$b_eps   # applied to innovations

# If Xi = I, then b_x = b_eps
par(mfrow = c(1, 2))
ts.plot(b_eps, main = "b applied to epsilon")
ts.plot(b_x,   main = "b applied to INDPRO")

# Constraint checks
sum(b_x) - sum(gamma_mse)     # cointegration (≈ 0)
bk_obj$rho_yy - rho1          # HT constraint (≈ 0)


# ────────────────────────────────────────────────────────────────
# Plot filters
# ────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "green", "blue", "red")

mplot <- cbind(
  hp_two,
  c(gamma_mse, rep(0, L - 1)),
  c(b_x,       rep(0, L - 1)),
  c(hp_trend,  rep(0, L - 1))
)
colnames(mplot) <- c("HP-two", "MSE", "SSA", "HP-C")

plot(mplot[, 1], main = "Trend filters", axes = FALSE, type = "l",
     ylab = "", xlab = "Lags", col = colo[1], lwd = 1,
     ylim = range(mplot))
abline(h = 0)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}

axis(1, at = 1:nrow(mplot), labels = 0:(nrow(mplot) - 1))
axis(2); box()

# Zoom on first 30 lags
mplot <- mplot[1:30, ]

plot(mplot[, 1], axes = FALSE, type = "l", col = colo[1], lwd = 1,
     ylim = c(min(mplot[, "HP-C"]), max(mplot[, "SSA"])))
abline(h = 0)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}

axis(1, at = 1:nrow(mplot), labels = 0:(nrow(mplot) - 1))
axis(2); box()


# ────────────────────────────────────────────────────────────────
# Filter data
# ────────────────────────────────────────────────────────────────
y_ssa            <- filter(x_tilde, b_x,       side = 1)
y_hp_concurrent  <- filter(x_tilde, hp_trend,  side = 1)
y_mse            <- filter(x_tilde, gamma_mse, side = 1)
y_target         <- filter(x_tilde, hp_two,    side = 2)

colo <- c("black", "violet", "green", "blue", "red")

# Plot: levels
par(mfrow = c(1, 1))
anf <- L + 100
enf <- length(x_tilde)

mplot <- cbind(x_tilde, y_target, y_mse, y_ssa, y_hp_concurrent)[anf:enf, ]
colnames(mplot) <- c("Data", "Target: HP-two", "MSE: HP-one", "SSA", "HP-C")

plot(mplot[, 1], main = "Data and trends", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = colo[1], lwd = 1)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}

axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[anf:length(y_xts)])
axis(2); box()


# Plot: first differences
anf <- L + 100
enf <- length(x_tilde) - 1

mplot <- apply(
  cbind(x_tilde, y_target, y_mse, y_ssa, y_hp_concurrent),
  2,
  diff
)[anf:enf, ]

colnames(mplot) <- c("Diff-Data", "Target: HP-two", "MSE: HP-one", "SSA", "HP-C")

par(mfrow = c(3, 1))

# MSE vs target
select_vec <- c(2, 3)
plot(mplot[, select_vec[1]], main = "Zero Crossings MSE",
     axes = FALSE, type = "l", col = colo[select_vec[1]], lwd = 1,
     ylim = c(-0.013, 0.006))
abline(v = 1 + which(sign(mplot[-1, select_vec[2]]) != sign(mplot[-nrow(mplot), select_vec[2]])),
       col = colo[select_vec[2]], lty = 2)
abline(h = 0)

for (i in seq_along(select_vec)) {
  lines(mplot[, select_vec[i]], col = colo[select_vec[i]])
  mtext(colnames(mplot)[select_vec[i]], line = -i, col = colo[select_vec[i]])
}

axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[(anf + 1):length(y_xts)])
axis(2); box()

# HP-C vs target
select_vec <- c(2, 5)
plot(mplot[, select_vec[1]], main = "Zero Crossings HP-C",
     axes = FALSE, type = "l", col = colo[select_vec[1]], lwd = 1,
     ylim = c(-0.013, 0.006))
abline(v = 1 + which(sign(mplot[-1, select_vec[2]]) != sign(mplot[-nrow(mplot), select_vec[2]])),
       col = colo[select_vec[2]], lty = 2)
abline(h = 0)

for (i in seq_along(select_vec)) {
  lines(mplot[, select_vec[i]], col = colo[select_vec[i]])
  mtext(colnames(mplot)[select_vec[i]], line = -i, col = colo[select_vec[i]])
}

axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[(anf + 1):length(y_xts)])
axis(2); box()

# SSA vs target
select_vec <- c(2, 4)
plot(mplot[, select_vec[1]], main = "Zero Crossings SSA",
     axes = FALSE, type = "l", col = colo[select_vec[1]], lwd = 1,
     ylim = c(-0.013, 0.006))
abline(v = 1 + which(sign(mplot[-1, select_vec[2]]) != sign(mplot[-nrow(mplot), select_vec[2]])),
       col = colo[select_vec[2]], lty = 2)
abline(h = 0)

for (i in seq_along(select_vec)) {
  lines(mplot[, select_vec[i]], col = colo[select_vec[i]])
  mtext(colnames(mplot)[select_vec[i]], line = -i, col = colo[select_vec[i]])
}

axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[(anf + 1):length(y_xts)])
axis(2); box()


# ────────────────────────────────────────────────────────────────
# Sample performance evaluation
# ────────────────────────────────────────────────────────────────

# MSE vs two-sided target
mean((y_target - y_mse)^2, na.rm = TRUE)
mean((y_target - y_ssa)^2, na.rm = TRUE)
mean((y_target - y_hp_concurrent)^2, na.rm = TRUE)

# MSE vs MSE-optimal predictor
mean((y_mse - y_ssa)^2, na.rm = TRUE)
bk_obj$mse_yz

# Empirical vs theoretical holding times (on differences)
compute_empirical_ht_func(scale(diff(y_mse)[anf:enf]))$empirical_ht
compute_holding_time_from_rho_func(rho_mse)$ht

compute_empirical_ht_func(scale(diff(y_ssa)[anf:enf]))$empirical_ht
compute_holding_time_from_rho_func(bk_obj$rho_yy)$ht

compute_empirical_ht_func(scale(diff(y_hp_concurrent)[anf:enf]))$empirical_ht
compute_holding_time_from_rho_func(rho_hp_concurrent)$ht

# Target
compute_empirical_ht_func(scale(diff(y_target)))$empirical_ht


# Summary table
mat_perf <- matrix(nrow = 2, ncol = 3)

mat_perf[1, ] <- c(
  mean((y_target - y_mse)^2, na.rm = TRUE),
  mean((y_target - y_ssa)^2, na.rm = TRUE),
  mean((y_target - y_hp_concurrent)^2, na.rm = TRUE)
)

mat_perf[2, ] <- c(
  compute_empirical_ht_func(diff(y_mse)[anf:enf])$empirical_ht,
  compute_empirical_ht_func(diff(y_ssa)[anf:enf])$empirical_ht,
  compute_empirical_ht_func(diff(y_hp_concurrent)[anf:enf])$empirical_ht
)

colnames(mat_perf) <- c("MSE nowcast", "SSA", "HP-C")
rownames(mat_perf) <- c("Sample mean square error", "Sample holding time")

mat_perf


# Findings:
# - MSE nowcast minimizes MSE but is highly noisy (low HT)
# - HP-C is much smoother but has substantially higher MSE
# - SSA matches HP-C smoothness while improving MSE
# - SSA achieves an efficient trade-off: large gains in smoothness
#   with moderate loss relative to the MSE-optimal benchmark












