# This tutorial is under revision/construction

# ========================================================================
# Tutorial 6: I-SSA
# Extension of SSA to non-stationary (I-ntegrated) processes
# ========================================================================
# Topics:
#   - Introduction of I-SSA
#   - Joint unified treatment of level nowcasting and growth-sign signaling
#   - Maximal monotone predictors
# ========================================================================

# ────────────────────────────────────────────────────────────────
# Context
# ────────────────────────────────────────────────────────────────
# SSA is typically formulated for stationary time series.
#
# SSA optimization principle:
#   - Maximize sign accuracy (or target correlation)
#   - Subject to a holding-time (HT) constraint
#
# Holding-time (HT):
#   - Mean duration between consecutive mean crossings of the predictor
#   - For zero-mean processes: mean duration between sign changes
#
# Issue:
#   - Integrated (non-stationary) processes are not mean-reverting
#   - ⇒ HT is undefined for non-stationary levels
#   - ⇒ Neither target correlation nor sign accuracy are well-defined
#       for non-stationary levels

# ────────────────────────────────────────────────────────────────
# Extension to Non-Stationary Series
# ────────────────────────────────────────────────────────────────
# Notation:
#   - x_t       : observed data; assumed I(1), i.e. Δx_t = x_t - x_{t-1} = u_t
#                 is stationary. Extensions to I(2) are covered in Wildi (2026a).
#   - z_{t+δ}   : acausal (two-sided) linear target relying on future values
#                 x_{t+k}, k > 0. Here: output of an acausal trend filter.
#   - y_t       : I-SSA predictor (causal nowcast of z_{t+δ}).
#
# Working in stationary first differences:
#   - HT is defined on Δy_t = y_t - y_{t-1}  (stationary)
#   - A pseudo target correlation and pseudo sign accuracy can be defined
#     via finite-length MA inversions of y_t (details omitted here)
#
# Primal and dual formulations of I-SSA:
#   Primal : For a given HT in first differences, y_t maximises tracking of
#            z_t in non-stationary levels (minimises level MSE).
#   Dual   : For a given level-tracking accuracy (e.g. MSE), I-SSA maximises
#            HT in first differences Δy_t.
#
# Maximal monotone property:
#   - If Δy_t is zero-mean, zero-crossings of Δy_t are turning points (TPs) of
#     y_t which is maximally monotone: no other linear predictor of z_t has
#     a longer mean duration between consecutive TPs than I-SSA.
#   - If E[Δy_t] = μ ≠ 0, HT regulates the mean duration between above-average
#     (Δy_t > μ) and below-average (Δy_t < μ) growth episodes (crossover
#     points). I-SSA is then maximally monotone about the linear trend T_t
#     with slope μ, i.e. the average duration of crossings of T_t is maximal.

# ────────────────────────────────────────────────────────────────
# References
# ────────────────────────────────────────────────────────────────
# Application to business-cycle analysis (stationary first differences):
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5
#
# Application to business-cycle analysis in original (non-stationary) levels:
#   Wildi, M. (2026a)
#     Sign Accuracy, Mean-Squared Error and the Rate of Zero Crossings:
#     a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547

# ────────────────────────────────────────────────────────────────
# This tutorial is based on Section 5 of Wildi (2026a)
# ────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────
# Initialize session: clear workspace and load required packages
# ────────────────────────────────────────────────────────────────
rm(list = ls())

library(xts)        # Extended time-series objects and utilities
library(mFilter)    # HP and BK trend/cycle filters
library(quantmod)   # Data retrieval (e.g. from FRED)


# ────────────────────────────────────────────────────────────────
# Load custom I-SSA and signal-extraction functions
# ────────────────────────────────────────────────────────────────
source(file.path(getwd(), "R", "simple_sign_accuracy.r"))   # Core SSA routines
source(file.path(getwd(), "R", "ISSA_functions.r"))         # Core I-SSA routines
source(file.path(getwd(), "R", "HP_JBCY_functions.r"))      # HP-filter helpers and JBCY utilities


# ========================================================================
# Exercise 1: Nowcasting via the "Double-Stroke" Principle
# ========================================================================
#
# Objective:
#   Design a nowcast of the HP trend that simultaneously achieves:
#     • Accurate trend tracking in LEVELS  (fidelity / low MSE)
#     • Smooth growth tracking in DIFFERENCES  (regularity / high HT)
#
# The "double-stroke" principle refers to satisfying both goals with a
# single, consistent filter design — rather than applying two separate,
# potentially conflicting approaches.
#
# Practical outputs of such a nowcast:
#   (a) An estimate of the current trend level
#   (b) A signal for the current economic state (above/below average growth,
#       or recession/expansion phases)
#
# ────────────────────────────────────────────────────────────────
# Specific challenge:
#   Construct an I-SSA nowcast of the two-sided HP(14400) trend for monthly
#   US industrial production (INDPRO) such that the nowcast:
#     (i)  Matches the holding-time (HT) of the classic one-sided HP filter
#          when evaluated in first differences, and
#     (ii) Achieves a lower MSE (better level tracking) than the classic
#          one-sided HP nowcast.
#
# ────────────────────────────────────────────────────────────────
# Benchmarks:
#
#   1) MSE-optimal predictor (based on an ARIMA(1,1,0) model)
#      • Minimises level MSE unconstrainedly
#      • Typically very noisy (low HT) — unsuitable for phase signaling
#
#   2) One-sided concurrent HP filter (HP-C)
#      • Standard business-cycle monitoring tool
#      • Adequate smoothness for phase signaling, but suboptimal level MSE
#
#   I-SSA target: match HP-C smoothness (same HT in differences) while
#   improving level-tracking accuracy (lower MSE than HP-C).
#
# ────────────────────────────────────────────────────────────────
# Key evaluation questions:
#   Q1: How much MSE does I-SSA sacrifice relative to the unconstrained
#       MSE-optimal predictor (cost of imposing the HT constraint)?
#   Q2: How much MSE does I-SSA gain relative to HP-C for the same
#       degree of smoothness (benefit of I-SSA optimisation)?
# ========================================================================





# ────────────────────────────────────────────────────────────────
# 1.1 INDPRO
# ────────────────────────────────────────────────────────────────
# 1.1.1 Load data (option to refresh from FRED)
# ────────────────────────────────────────────────────────────────
reload_data <- FALSE

if (reload_data) {
  getSymbols("INDPRO", src = "FRED")
  save(INDPRO, file = file.path(getwd(), "Data", "INDPRO"))
} else {
  load(file = file.path(getwd(), "Data", "INDPRO"))
}

tail(INDPRO)


# 1.1.2 Sample selection and transformations
# ────────────────────────────────────────────────────────────────
start_year <- 1982
end_year   <- 2024

y      <- as.double(log(INDPRO[paste0(start_year, "/", end_year)]))
y_xts  <- log(INDPRO[paste0(start_year, "/", end_year)])


# 1.1.3 Plot raw data, log-levels, and first differences
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
# I-SSA must track the two-sided HP filter applied to non-stationary levels.


# ────────────────────────────────────────────────────────────────
# 1.2 Specify HP Target
# ────────────────────────────────────────────────────────────────

# Filter design parameters
L<-101
# Classical HP setting for monthly data:
lambda_hp <- 14400



# Compute HP
HP_obj<-HP_target_mse_modified_gap(L,lambda_hp)
# Concurrent MSE estimate of bi-infinite HP assuming white noise (truncate symmetric filter)
hp_mse<-gamma<-HP_obj$hp_mse
# Reconstruct two-sided HP by mirroring hp_mse at center point
hp_two<-c(hp_mse[L:2],hp_mse)
# Classic one-sided concurrent HP (HP-C)
hp_c<-HP_obj$hp_trend

ts.plot(cbind(hp_two,c(hp_mse,rep(0,L-1)),c(hp_c,rep(0,L-1))),xlab="Lags",col=rainbow(3))
mtext("Two-sided HP shifted to the right",col=rainbow(3)[1],line=-1)
mtext("Classic one-sided (concurrent) HP-C",col=rainbow(3)[2],line=-2)
mtext("MSE optimal one-sided HP when data is white noise",col=rainbow(3)[3],line=-3)


# ────────────────────────────────────────────────────────────────
# 1.3 Model Setup for I-SSA
# ────────────────────────────────────────────────────────────────

# Model: Assume ARIMA(1,1,0) for log-INDPRO (based on ACF)
# Parameters fixed (to avoid instability from extreme (COVID) observations)
a1 <- 0.3
b1 <- 0
# This alignes with pre-pandemic estimate of AR(1)
p=1;q=0
arima_obj<-arima(diff(y_xts)["/2020"],order=c(p,0,q))
arima_obj
# Diagnostics are not perfect but ACF of residuals are close to zero and model is simple
# contain overfitting)
tsdiag(arima_obj)
# Need vola estimate for later calibration
sigma_ip<-sqrt(arima_obj$sigma2)

# ────────────────────────────────────────────────────────────────
# 1.4 HT Constraint
# ────────────────────────────────────────────────────────────────

# The HT constraint is defined on first differences (see eq. 29, Wildi 2026a).
# On first differences the process is AR(1)
# We compute the Wold decomposition
xi<-c(1,ARMAtoMA(ar=a1,ma=b1,lag.max=L-1))
# Compute convolution of target with xi: this filter replicates HP-C when applied to innovations 
# (MA inversion) of AR(1)
hp_c_convolved_with_xi<-conv_two_filt_func(xi,hp_c)$conv
# Compute HT: HT computation assumes data to be white noise
# therefore we supply hp_c_convolved_with_xi
HT_HP_obj<-compute_holding_time_func(hp_c_convolved_with_xi)
# HT: expected duration (in months) between consecutive mean-crossings of the
# one-sided HP, assuming HP is applied to the AR(1) process.
ht_constraint<-HT_HP_obj$ht
ht_constraint
# For comparison: the HT when the same HP filter is applied to white noise.
# This HT is shorter because white noise is less regular (more volatile) 
# than an AR(1) process with positive autocorrelation, resulting in more 
# frequent mean-crossings.
compute_holding_time_func(hp_c)$ht




# ────────────────────────────────────────────────────────────────
# 1.5 I-SSA 
# ────────────────────────────────────────────────────────────────
# Nowcast
delta<-0

# Target for I-SSA
# Two possibilities
# 1. Possibility
# Supply the right half of the two-sided HP
gamma_target<-hp_mse
# Specify that this should be transformed into symmetric target (mirrored at center point)
symmetric_target<-T

if (F)
{
# 2. Possibility
# Supply the right-shifted (causal) two-sided HP
  gamma_target<-hp_two
# Specify that this should not be transformed into symmetric target 
  symmetric_target<-F
# Specify backcast located at center point of causal HP: add (length(gamma_target)-1)/2 to delta
  delta<-delta+(length(gamma_target)-1)/2
}

# Initial value of the Lagrangian multiplier, see Wildi (2026a), equation 30.
# This Lagrange multiplier is not related to lambda_hp
lambda_start<-0

ISSA_obj<-ISSA_func(ht_constraint,L,delta,gamma_target,symmetric_target,a1,b1,lambda_start)
 
bk_obj<-ISSA_obj$bk_obj
gamma_mse<-ISSA_obj$gamma_mse
b_x<-ISSA_obj$b_x 


# Diagnostics: If the numerical optimization converges the HT of the optimized I-SSA
# should match the imposed ht: the difference should be negligible. This is generally 
# the case because the optimization relies on the R function optim which is an efficient optimizer.
# If the difference is large: increase the number of iterations in optim or provide 
# a better initial value lambda_start (lambda_start=0 corresponds to the MSE benchmark) 
abs(bk_obj$ht_issa-ht_constraint)

# The above function also computes the pseudo target correlation. Note that:
#   - The target for I-SSA is the two-sided HP trend
#   - Since the target is non-stationary, a conventional correlation is not 
#     well-defined
#   - A meaningful pseudo target correlation suitable for optimisation is 
#     derived in Wildi (2026a), equation 29 (left-hand side)
bk_obj$rho_yz   # pseudo correlation with target

# The above function also computes the expected (theoretical) MSE of the 
# optimal I-SSA nowcast, referenced against the MSE-optimal 
# predictor of the two-sided HP under the posited AR(1) model (note: MSE 
# is referenced against the optimal predictor, not the acausal HP itself).
bk_obj$mse_yz * sigma_ip^2 # MSE rescaled by the residual variance of the fitted AR(1) model



# ────────────────────────────────────────────────────────────────
# Plot filters
# ────────────────────────────────────────────────────────────────
par(mfrow = c(1, 2))
colo <- c("violet", "green", "blue", "red")

mplot <- cbind(
  hp_two,
  c(gamma_mse, rep(0, L - 1)),
  c(b_x,       rep(0, L - 1)),
  c(hp_c,  rep(0, L - 1))
)
colnames(mplot) <- c("HP-two", "MSE", "I-SSA", "HP-C")

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
     ylim = c(min(mplot[, "HP-C"]), max(mplot[, "I-SSA"])))
abline(h = 0)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}

axis(1, at = 1:nrow(mplot), labels = 0:(nrow(mplot) - 1))
axis(2); box()

# The MSE filter assigns most weight to the most recent data point — a 
# consequence of the data following a random walk. Consequently, its HT 
# in first differences is very small, as verified above.
# The decay profile of I-SSA is more complex than that of HP, and 
# convergence to zero is faster. The characteristic nose-shaped profile 
# near lag 0 is a consequence of the implicit zero-boundary constraint of 
# (I-)SSA — see Theorem 1 in Wildi (2026a).

# ────────────────────────────────────────────────────────────────
# Filter data
# ────────────────────────────────────────────────────────────────
y_ssa            <- filter(x_tilde, b_x,       side = 1)
y_hp_concurrent  <- filter(x_tilde, hp_c,  side = 1)
y_mse            <- filter(x_tilde, gamma_mse, side = 1)
y_target         <- filter(x_tilde, hp_two,    side = 2)

# ────────────────────────────────────────────────────────────────
# Plot Nowcasts and Target: in Levels
# ────────────────────────────────────────────────────────────────


colo <- c("black", "violet", "green", "blue", "red")

# Plot: levels
par(mfrow = c(1, 1))
anf <- L + 100
enf <- length(x_tilde)

mplot <- cbind(x_tilde, y_target, y_mse, y_ssa, y_hp_concurrent)[anf:enf, ]
colnames(mplot) <- c("Data", "Target: HP-two", "MSE: HP-one", "I-SSA", "HP-C")

plot(mplot[, 1], main = "Data and trends", axes = FALSE, type = "l",
     xlab = "", ylab = "", col = colo[1], lwd = 1)

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
}

axis(1, at = 1:nrow(mplot),
     labels = index(y_xts)[anf:length(y_xts)])
axis(2); box()

# We observe that the I-SSA trend tends to be more timely than the classical 
# HP nowcast around economic peaks and troughs. At the same time, I-SSA 
# appears as smooth as HP — as desired by design through the imposed HT 
# constraint.
 


# ────────────────────────────────────────────────────────────────
# Plot Nowcasts and Target: in First Differences
# ────────────────────────────────────────────────────────────────


# Plot: first differences
anf <- L + 100
enf <- length(x_tilde) - 1

mplot <- apply(
  cbind(x_tilde, y_target, y_mse, y_ssa, y_hp_concurrent),
  2,
  diff
)[anf:enf, ]

colnames(mplot) <- c("Diff-Data", "Target: HP-two", "MSE: HP-one", "I-SSA", "HP-C")

par(mfrow = c(3, 1))

# MSE vs target
select_vec <- c(2, 3)
plot(mplot[, select_vec[1]], main = "Zero Crossings MSE in First Differences",
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
plot(mplot[, select_vec[1]], main = "Zero Crossings HP-C in First Differences",
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

# I-SSA vs target
select_vec <- c(2, 4)
plot(mplot[, select_vec[1]], main = "Zero Crossings I-SSA in First Differences",
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

# We observe that the MSE-optimal nowcast (green, top panel) is very noisy 
# in first differences. In contrast, HP-C and I-SSA are both smooth in 
# first differences, and both trends drop below the zero-growth line at 
# recessions — with occasional false signals during prolonged expansions.


# ────────────────────────────────────────────────────────────────
# Sample performance evaluation
# ────────────────────────────────────────────────────────────────

# Compare MSE of each predictor against the two-sided HP filter (the target)
# Expected ranking: classic MSE predictor < I-SSA < HP-C (concurrent HP)
# Key question: I-SSA outperforms HP-C in MSE terms, but does it match HP-C in smoothness?
mean((y_target - y_mse)^2, na.rm = TRUE)
mean((y_target - y_ssa)^2, na.rm = TRUE)
mean((y_target - y_hp_concurrent)^2, na.rm = TRUE)

# Verify that theoretical expectations match sample estimates
# Good agreement is expected when:
#   - The sample is large (law of large numbers)
#   - The assumed model (AR(1)) is correctly specified

# Sample MSE between the classic MSE predictor and I-SSA
mean((y_mse - y_ssa)^2, na.rm = TRUE)
# Theoretical counterpart: expected MSE under a true AR(1) model
# Good agreement with the sample estimate confirms model adequacy
bk_obj$mse_yz*sigma_ip^2

# Sample holding time of the classic MSE predictor (computed on first differences)
# Note: the data are mean-centred to emphasise mean-crossings, which are 
# directly governed by the HT constraint.
compute_empirical_ht_func(scale(diff(y_mse)[anf:enf]))$empirical_ht
# Theoretical HT under a true AR(1) model: good agreement with sample estimate
ht_mse
# Sample holding time of the I-SSA predictor (computed on first differences)
# Data are scaled before computing zero-crossings to ensure comparability
compute_empirical_ht_func(scale(diff(y_ssa)[anf:enf]))$empirical_ht
# Theoretical HT: slightly smaller than the sample estimate,
# though the discrepancy is within the range of expected sampling variation
compute_holding_time_from_rho_func(bk_obj$rho_yy)$ht

# Sample holding time of the concurrent HP filter (computed on first differences)
compute_empirical_ht_func(scale(diff(y_hp_concurrent)[anf:enf]))$empirical_ht
# Theoretical HT under a true AR(1) model: good agreement with sample estimate
ht_constraint

# Sample holding time of the two-sided HP target (computed on first differences)
# Substantially larger than all one-sided predictors, reflecting the greater
# smoothness of the two-sided HP filter
compute_empirical_ht_func(scale(diff(y_target)))$empirical_ht


# Summary table: MSE and sample holding time for each predictor
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

colnames(mat_perf) <- c("Classical MSE optimal nowcast", "I-SSA", "HP-C")
rownames(mat_perf) <- c("Sample MSE", "Sample holding time")

mat_perf


# Findings from the summary table:
#   - Classic MSE predictor: minimises MSE but is highly noisy (low HT),
#     making it impractical for real-time recession monitoring.
#   - HP-C (concurrent HP): much smoother (higher HT), but MSE is approximately
#     100% larger than the MSE-optimal benchmark.
#   - I-SSA: matches or slightly exceeds HP-C in smoothness, while incurring
#     only approximately 50% larger MSE than the MSE-optimal benchmark.
#   - Overall: I-SSA achieves an efficient trade-off on the smoothness-accuracy
#     frontier
#     -delivering large gains in smoothness at a moderate MSE cost
#     relative to the classic MSE-optimal predictor.
#     -delivering substantial gains in MSE for identical (or slightly better) 
#     smoothness  relative to the classic one-sided HP nowcast.


# MSE of predictors vs. two-sided HP (target)
# Classic MSE predictor dominates, as expected. It is followed by I-SSA and HP-C.
# So I-SSA outperforms HP-C in terms of MSE. But is it as smooth as HP-C?
mean((y_target - y_mse)^2, na.rm = TRUE)
mean((y_target - y_ssa)^2, na.rm = TRUE)
mean((y_target - y_hp_concurrent)^2, na.rm = TRUE)

# First we check if expected numbers match sample estimates
# They match when:
# -The sample is large
# -The model is true


# Sample MSE 
mean((y_mse - y_ssa)^2, na.rm = TRUE)
# Good agreement with expected value (assuming a true AR(1) model for I-SSA)
bk_obj$mse_yz*sigma_ip^2 

# Sample holding time of classic MSE predictor (on differences)
compute_empirical_ht_func(scale(diff(y_mse)[anf:enf]))$empirical_ht
#  Good agreement with theoretical holding time assuming true AR(1) model for MSE predictor 
ht_mse
# Sample holding time of I-SSA predictor (on differences)
# Scale data to address zero-crossings
compute_empirical_ht_func(scale(diff(y_ssa)[anf:enf]))$empirical_ht
#  Theoretical holding time is a bit smaller (though compliant with sample fluctuation)  
compute_holding_time_from_rho_func(bk_obj$rho_yy)$ht

# Sample holding time of HP (on differences)
compute_empirical_ht_func(scale(diff(y_hp_concurrent)[anf:enf]))$empirical_ht
# Good agreement with expected value 
ht_constraint
# Sample HT of two-sided HP (target): much larger (two-sided HP is too smooth)
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

colnames(mat_perf) <- c("Classical MSE optimal nowcast", "I-SSA", "HP-C")
rownames(mat_perf) <- c("Sample MSE", "Sample holding time")

mat_perf


# Findings from summary table:
# - MSE nowcast minimizes MSE but is highly noisy (low HT)
# - HP-C is much smoother but 100% larger MSE
# - I-SSA matches HP-C smoothness (even slightly smoother) but only 50% larger sample MSE
# - I-SSA achieves an efficient trade-off: large gains in smoothness 
#   with moderate MSE loss relative to the MSE-optimal benchmark












