# ========================================================================
# Tutorial 6: I-SSA 
# Extension of SSA to non-stationary (I-ntegrated) processes
# ========================================================================
# Topics:
#   - Introduction of I-SSA
#   - Joint treatment of level nowcasting and growth-sign signaling
#   - Maximal monotone predictors
# ========================================================================

# ────────────────────────────────────────────────────────────────
# Context
# ────────────────────────────────────────────────────────────────
# SSA is typically formulated for stationary time series.

# SSA optimization principle:
#   - Maximize sign accuracy (or target correlation)
#   - Subject to a holding-time (HT) constraint
#
# Holding-time (HT):
#   - Mean duration between consecutive mean crossings of the predictor
#   - For zero-mean processes: mean duration between sign changes

# Issue:
#   - Integrated (non-stationary) processes are not mean-reverting
#   - ⇒ HT is not defined in (non-stationary) levels
#   - ⇒ neither the target correlation nor the sign accuracy are properly 
#       defined in (non-stationary) levels
#


# ────────────────────────────────────────────────────────────────
# Extension to non-stationary series
# ────────────────────────────────────────────────────────────────
#   - In first differences (stationary): I-SSA maximizes HT
#   - In levels: I-SSA yields a maximal monotone predictor
#
# See: Wildi (2026a), sections 5.3-5.5


# ────────────────────────────────────────────────────────────────
# Background
# ────────────────────────────────────────────────────────────────

# Application to business-cycle analysis (stationary first differences)
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

# Application to business-cycle analysis in original (non-stationary) levels
#   Wildi, M. (2026a)
#     Sign Accuracy, Mean-Squared Error and the Rate of Zero Crossings:
#     a Generalized Forecast Approach
#     https://doi.org/10.48550/arXiv.2601.06547

# ────────────────────────────────────────────────────────────────
# I-SSA Optimization Principle
# ────────────────────────────────────────────────────────────────
#   - Tracking:
#       Optimal trend nowcast in LEVELS (minimize MSE or maximize
#       pseudo target correlation or  pseudo sign accuracy)
#
#   - Smoothness:
#       Control mean-crossing rate in FIRST DIFFERENCES (HT constraint)
#
#   - Cointegration constraint:
#       Ensures finite MSE between predictor and target
#
# Dual interpretation:
#   - Maximize monotonicity for given MSE (in levels)
#   - For a fixed MSE specification, the level predictor is
#       maximally monotone among linear predictors


# ────────────────────────────────────────────────────────────────
# This Tutorial is Based on Section 5, Wildi 2026a)
# ────────────────────────────────────────────────────────────────
# Challenge:
#   Construct a I-SSA nowcast of the two-sided HP(14400) for monthly US 
#   industrial production (INDPRO). The I-SSA nowcast must 
#   i)   replicate the HT of the classic one-sided HP in first differences, and
#   ii)  improve MSE performances when compared to the classic HP nowcast.

# Benchmarks against which I-SSA will be compared:
#
#   1) MSE-optimal nowcast (ARIMA(1,1,0))
#      → Typically very noisy
#
#   2) One-sided concurrent HP (HP-C)
#      → Standard business-cycle tool

# Key questions:
#   - How much MSE is lost relative to the unconstrained MSE predictor?
#   - How much MSE is gained relative to HP-C for the same smoothness?



# ────────────────────────────────────────────────────────────────
# Objective: The “Double-Stroke” Principle 
# ────────────────────────────────────────────────────────────────
#
# Design a novel nowcast that jointly achieves:
#   • Accurate trend tracking in level (accuracy)
#   • Reliable recession signaling in differences (smoothness)
#
# These objectives conflict:
#   - Accuracy on levels highlights Timeliness (coincidence)
#   - Zero-crossing analysis in differences highlights Smoothness (few false alarms)
#   - ATS-trilemma: Accuracy, Timeliness and Smoothness are competing requirements
#
# I-SSA resolves this multi-objective prediction problem:
#   → Tracks the two-sided HP trend in levels (Accuracy)
#   → Produces sparse, meaningful sign changes in differences (signal quality)
#
# Result:
#   A unified filter delivering both level nowcasts and business-cycle turning 
#   point signals.
# ========================================================================




# ────────────────────────────────────────────────────────────────
# Initialize session: clear workspace and load required packages
# ────────────────────────────────────────────────────────────────
rm(list = ls())

library(xts)        # Time series handling
library(mFilter)    # HP and BK filters
library(quantmod)   # Data access (FRED)


# ────────────────────────────────────────────────────────────────
# Load custom I-SSA and signal extraction functions
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
# I-SSA must track the two-sided HP filter applied to non-stationary levels.


# ────────────────────────────────────────────────────────────────
# Model setup for maximal monotone predictor
# ────────────────────────────────────────────────────────────────
# Cointegration constraint ensures finite MSE for integrated processes

# Assume ARIMA(1,1,0) for log-INDPRO (based on ACF)
# Parameters fixed to avoid instability from extreme (COVID) observations
a1 <- 0.3
b1 <- 0
# This alignes with pre-pandemic estimate of AR(1)
p=1;q=0
arima_obj<-arima(diff(y_xts)["/2020"],order=c(p,0,q))
arima_obj
# Diagnostics are not perfect but ACF of residuals are close to zero
tsdiag(arima_obj)
# Need vola estimate for later calibration
sigma_ip<-sqrt(arima_obj$sigma2)

# Filter design parameters
L         <- 101
# Classical HP setting for monthly data:
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

# Specifically:
#   gamma_mse   : the MSE-optimal nowcast of the two-sided HP trend under 
#                 the AR(1) model assumption: no other nowcast outperfroms
#                 gamma_mse in terms of mean-squared filter error.
#   gamma_tilde : the finite-length (truncated) MA-inversion of gamma_mse;
#                 see Wildi 2026a, sections 5.3–5.5
#
# Technical note:
#   gamma_tilde is used for optimisation only — not for filtering. Imposing 
#   the cointegration constraint ensures that the filter error is stationary. 
#   Under stationarity, MA-inversions of the filter error decay to zero, so 
#   that infinite-length inversions can be replaced by finite-length 
#   approximations. Consequently, a finite MA-inversion gamma_tilde suffices 
#   for optimisation — even though the MA coefficients of the level filter 
#   themselves do not decay to zero. 

# The I-SSA concept is non-trivial. The theory is developed in Wildi 2026a,
# section 5. Even if the individual steps are not 
# immediately transparent, the sample results presented below will 
# demonstrate the validity and pertinence of the approach from an empirical 
# point of view.

par(mfrow=c(1,1))
ts.plot(cbind(gamma_tilde,gamma_mse),col=c("black","brown"))
mtext("Optimal MSE filter applied to data in levels",line=-1,col="brown")
mtext("Finite MA-Inversion of MSE used for optimization",line=-2)

# Plot Wold decomposition of differenced process: first column in matrix Xi
ts.plot(Xi[,1],main="Wold decomposition of AR(1)")




# ────────────────────────────────────────────────────────────────
# Holding-Time (HT) Constraint Calibration
# ────────────────────────────────────────────────────────────────
# The HT constraint is defined on first differences (see eq. 29, Wildi 2026a).

# Derivation of the HT of the first-differenced HP predictor (not trivial):
#
#   Xi       : convolution matrix (eq. 22, Wildi 2026a)
#   Xi_tilde : Xi composed with Sigma (the integration operator);
#              see section 5.3 of Wildi 2026a
#
#   Xi_tilde %*% hp_trend yields a finite MA representation of the HP 
#   predictor in levels. Although this representation is not the predictor 
#   itself (which is non-stationary), it is suitable for optimisation for 
#   the following reasons:
#     - Optimisation addresses the filter error
#     - Under the cointegration constraint, the filter error is stationary
#       despite the series being non-stationary
#     - Under stationarity, finite and infinite MA representations are
#       equivalent (the MA inversions converge to zero)
#
#   Applying the differencing operator Delta yields:
#       Delta %*% Xi_tilde %*% hp_trend = Xi %*% hp_trend
#   (both expressions are algebraically identical, see Wildi 2026a)
#
#   Therefore, Xi %*% hp_trend is the appropriate input for computing the
#   HT of the differenced HP trend (i.e., the classical concurrent trend 
#   nowcast in levels).

# This reasoning is non-trivial. Even if the individual steps are not 
# immediately transparent, the sample results presented below will 
# demonstrate the validity and pertinence of the approach from an empirical 
# point of view.

HT_HP_obj<-compute_holding_time_func(Xi %*% hp_trend)

# HT: expected duration (in months) between consecutive mean-crossings of the
# one-sided HP, assuming HP is applied to the AR(1) process.
HT_HP_obj$ht

# For comparison: the HT when the same HP filter is applied to white noise.
# This HT is shorter because white noise is less regular (more volatile) 
# than an AR(1) process with positive autocorrelation, resulting in more 
# frequent mean-crossings.
compute_holding_time_func(hp_trend)$ht

# First-order autocorrelation (rho1): stands in a one-to-one (bijective) correspondence
#   with the HT above; see eq. 18, Wildi (2026a) for the analytical relationship
HT_HP_obj$rho_ff1

# Since HT and rho1 are in bijective correspondence, imposing rho1 is equivalent
#   to imposing the HT constraint. We therefore use rho1 to enforce the HT
#   constraint in I-SSA (see eq. 18, Wildi 2026a).
rho1 <- as.double(rho_hp_concurrent<-HT_HP_obj$rho_ff1)

# Interpretation of the I-SSA constraint:
#   I-SSA is constrained to match the holding time of the one-sided (causal) HP filter,
#   as measured in stationary first differences. Concretely:
#     1. Apply the causal HP filter to the data in levels.
#     2. Compute the first differences of the filtered series.
#     3. Compute the mean duration between consecutive sign changes of these differences.
#   This mean duration is the target HT (stored in HT_HP_obj$ht), and rho1 is its
#   bijective counterpart used to impose the constraint in I-SSA.
#   Two remarks on HT_HP_obj$ht:
#     - It is derived from a theoretical formula and is therefore exact only if
#       the assumed model (Xi) is the true data-generating process.
#     - We will compare this theoretical value against the empirical estimate
#       obtained by applying the filters directly to the data, as a model diagnostic.
#
#   By the optimality properties of I-SSA, a trend nowcast that replicates the HT of HP
#   (via the rho1 constraint) is expected to outperform the classic HP filter in terms
#   of MSE on non-stationary levels. Moreover, no other linear predictor subject to the
#   same HT constraint can improve upon I-SSA if the model (Xi) is correctly specified.

# The following code verifies optimality empirically and quantifies the MSE gain.



# ────────────────────────────────────────────────────────────────
# Reference: HT of the Differenced MSE-Optimal Predictor
# ────────────────────────────────────────────────────────────────
# For reference, we compute the HT of the MSE-optimal nowcast of the 
# two-sided HP trend. To clarify the role of each competitor:
#
#   - Target        : the acausal two-sided HP trend (to be nowcasted)
#   - Benchmark 1   : the classical one-sided HP — a standard nowcast of
#                     the two-sided HP, serving as a benchmark for I-SSA
#   - Benchmark 2   : the classical MSE-optimal nowcast of the two-sided HP —
#                     by construction, no other nowcast can achieve a lower MSE
#   - I-SSA         : replicates the HT of the one-sided HP (Benchmark 1)
#
# Key questions:
#   - How much MSE does I-SSA sacrifice relative to the MSE-optimal nowcast
#     (Benchmark 2) due to imposing the HT constraint?
#   - How much MSE does I-SSA gain relative to the classical one-sided HP 
#     (Benchmark 1) for identical HT, due to optimality of I-SSA?

# Using the same derivation logic:
#   Xi %*% gamma_mse characterizes the differenced representation
#   of the MSE-optimal predictor in levels.

mse_ht_obj<-compute_holding_time_func(Xi %*% gamma_mse)
# The MSE optimal predictor has a much smaller HT than HP
mse_ht_obj$ht
HT_HP_obj$ht

# This reflects the higher noise of MSE-optimal predictors in levels.
# MSE optimality often trades off smoothness for timeliness:
# such predictors are generally more reactive but noisier.
# The plots below will illustrate the smoothness differences and their impact 
#   on recession signaling

# Note: rather than supplying the HT directly, we pass the lag-one ACF to I-SSA, 
# from which the HT constraint is derived internally.
rho_mse <- as.double(mse_ht_obj$rho_ff1)



# ────────────────────────────────────────────────────────────────
# Motivation for I-SSA
# ────────────────────────────────────────────────────────────────
# - MSE-optimal level predictors are efficient but often very noisy
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
#   → replicates HP smoothness (via HT constraint): generate fewer spurious 
#     false alarms in first differences
#   → improves tracking of the two-sided HP in levels (lower MSE)
#
# Result:
# A predictor that closely tracks the level while producing
# sparse and informative sign changes in first differences: a DOUBLE-STROKE.


# ────────────────────────────────────────────────────────────────
# Compute I-SSA solution (Wildi, 2026a, Sections 5.3–5.4)
# ────────────────────────────────────────────────────────────────
# Use numerical optimization (optim) to determine the optimal
# Lagrange multiplier λ ensuring compliance with the HT constraint.
# Initialization at λ = 0 corresponds to the MSE benchmark.

# Do not confuse this lambda with lambda_hp (the lambda regularization 
# parameter of HP)
lambda <- 0

# Classical numerical optimization procedure optim in R
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
# Negative values of the Lagrange multiplier indicate stronger smoothing of 
# I-SSA compared to the MSE benchmark.
lambda_opt

# Compute I(1) cointegrated I-SSA solution based on lambda_opt
bk_obj <- bk_int_func(
  lambda_opt,
  gamma_mse,
  Xi,
  Sigma,
  Xi_tilde,
  M,
  B,
  gamma_tilde,
  rho1
)

# Diagnostics: bk_obj$rho_yy is the lag-one ACF of the optimised I-SSA 
# trend in first differences. When numerical optimisation has succeeded, 
# this value should match the imposed rho1 up to negligible error. In 
# practice, the R function optim is sufficiently powerful that convergence 
# is reliably achieved.
bk_obj$rho_yy   # should match rho1 (HT constraint)
rho1

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

#-------------------------------------------------------------------------------
# Note:
# The above MSE vanishes if I-SSA exactly replicates the MSE-optimal nowcast 
# (this happens when the imposed HT equals the HT of the MSE nowcast).
#
# Verifications: 
# i) either set lambda_opt <- 0 above and confirm that bk_obj$mse_yz = 0, or
# ii) set the HT in I-SSA to match the MSE nowcast and confirm that 
# bk_obj$mse_yz = 0.
#
# Background on the I-SSA optimisation principle:
#   - The I-SSA solution obtained by targeting the acausal two-sided HP 
#     directly is identical to the solution obtained by targeting its causal 
#     MSE-optimal predictor — see Wildi (2026a), section 2.
#   - Hence, we can match the MSE-optimal predictor/nowcast (of the two-sided 
#     HP) as closely as possible subject to the HT constraint.

# Below, we verify that the sample MSE between the I-SSA trend and the 
# MSE-optimal nowcast matches bk_obj$mse_yz * sigma_ip^2.
#-------------------------------------------------------------------------------

# Extract filter
b_x   <- bk_obj$b_x     

# Constraint checks
sum(b_x) - sum(gamma_mse)     # cointegration (≈ 0): ensures a finite MSE on non-stationary levels
bk_obj$rho_yy - rho1          # HT constraint (≈ 0). If this is not small, the numerical optimization did not converge


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
y_hp_concurrent  <- filter(x_tilde, hp_trend,  side = 1)
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
compute_holding_time_from_rho_func(rho_mse)$ht

# Sample holding time of the I-SSA predictor (computed on first differences)
# Data are scaled before computing zero-crossings to ensure comparability
compute_empirical_ht_func(scale(diff(y_ssa)[anf:enf]))$empirical_ht
# Theoretical HT: slightly smaller than the sample estimate,
# though the discrepancy is within the range of expected sampling variation
compute_holding_time_from_rho_func(bk_obj$rho_yy)$ht

# Sample holding time of the concurrent HP filter (computed on first differences)
compute_empirical_ht_func(scale(diff(y_hp_concurrent)[anf:enf]))$empirical_ht
# Theoretical HT under a true AR(1) model: good agreement with sample estimate
compute_holding_time_from_rho_func(rho_hp_concurrent)$ht

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
compute_holding_time_from_rho_func(rho_mse)$ht

# Sample holding time of I-SSA predictor (on differences)
# Scale data to address zero-crossings
compute_empirical_ht_func(scale(diff(y_ssa)[anf:enf]))$empirical_ht
#  Theoretical holding time is a bit smaller (though compliant with sample fluctuation)  
compute_holding_time_from_rho_func(bk_obj$rho_yy)$ht

# Sample holding time of HP (on differences)
compute_empirical_ht_func(scale(diff(y_hp_concurrent)[anf:enf]))$empirical_ht
# Good agreement with expected value 
compute_holding_time_from_rho_func(rho_hp_concurrent)$ht

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












