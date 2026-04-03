

# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 9: I-SSA and M-SSA SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════


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




# ────────────────────────────────────────────────────────────────
# Load custom I-SSA and signal extraction functions
# ────────────────────────────────────────────────────────────────
source(file.path(getwd(), "R", "simple_sign_accuracy.r"))
source(file.path(getwd(), "R", "HP_JBCY_functions.r"))




# ══════════════════════════════════════════════════════════════════════════════
# Exercise 4: I-SSA Smoothing on non-stationary levels
# ══════════════════════════════════════════════════════════════════════════════
# Similar to tutorial 6 but we target I_t instead of HP(I_t)

# ─────────────────────────────────────────────────────────────────────────────
# Exercise 4.1 Create exercise (random-walk???)
# ─────────────────────────────────────────────────────────────────────────────

# We assume white noise after first differences: random-walk
a1 <- 0.
b1 <- 0
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

par(mfrow=c(1,1))
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
#   Xi_tilde : Xi composed with Sigma (the integration operator); see section 5.3
#
#   Xi_tilde %*% hp_trend provides a finite MA representation of the HP predictor in levels.
#   Although this representation is not the predictor itself (which is non-stationary),
#   it is suitable for differencing: finite and infinite MA filters behave equivalently
#   under first differencing, so the stationary differences are well-defined.
#
#   Applying the differencing operator Delta yields:
#       Delta %*% Xi_tilde %*% hp_trend = Xi %*% hp_trend
#   (both expressions are algebraically identical)
#
#   Therefore, Xi %*% hp_trend is the appropriate input for computing the holding time
#   of the differenced HP trend (i.e., the classic concurrent trend nowcast in levels).

HT_HP_obj<-compute_holding_time_func(Xi %*% hp_trend)

# HT: expected duration (in months) between consecutive mean-crossings of the
#   filtered process, where the filter is described by the MA inversion Xi
#   (in this application: an AR(1) process)
HT_HP_obj$ht

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
# Reference: HT of differenced MSE-optimal predictor (MSE-optimal in levels)
# ────────────────────────────────────────────────────────────────
# Using the same derivation logic:
#   Xi %*% gamma_mse characterizes the differenced representation
#   of the MSE-optimal predictor in levels.

rho_mse <- as.double(compute_holding_time_func(Xi %*% gamma_mse)$rho_ff1)

# Typically, rho_mse is small → frequent zero-crossings
# This reflects the higher noise of MSE-optimal predictors in levels.
# MSE optimality trades off smoothness for timeliness:
# such predictors are generally more reactive but noisier.
rho_mse
# Compare to HP: the latter is much smoother
rho1
# The plots below will illustrate the smoothness differences and their impact 
#   on recession signaling


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

# Do not confuse this lambda with lambda_hp (the lambda regularization parameter of HP)
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

# Compute I(1) cointegrated I-SSA solution based on lambda_opt
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
bk_obj$rho_yy   # should match rho1 (HT constraint): this verifies convergence of the optimization (if not: increase the number of iterations)
rho1
bk_obj$rho_yz   # correlation with target
bk_obj$mse_yz*sigma_ip^2   # MSE with respect to MSE-optimal predictor (rescaled with residual variance from AR(1) model)
# This MSE vanishes if I-SSA replicates exactly the MSE predictor 
# (set lambda_opt<-0 and verify that bk_obj$mse_yz=0)
# I-SSA optimization principle:
# Match the classic MSE predictor as close as possible under the HT constraint.


# Extract filters
b_x   <- bk_obj$b_x     # applied to data
b_eps <- bk_obj$b_eps   # applied to innovations

# If Xi = I, then b_x = b_eps
par(mfrow = c(1, 2))
ts.plot(b_eps, main = "b applied to epsilon")
ts.plot(b_x,   main = "b applied to INDPRO")

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





# ─────────────────────────────────────────────────────────────────────────────
# Exercise 4.2 Replicate TP-frequency of HP by SSA
# ─────────────────────────────────────────────────────────────────────────────
# This is once again unusal because we use I-SSA for series that are stationary.
# 1. Define HP in diffs: HT in diffs = TP rate on level
# 2. Specify I-SSA that targets cumsum(x_t)=eps_t on levels and imposes 
#     HT on differences x_t=eps_t-eps_{t-1}
# In contrast to exercise 2, the resulting I-SSA replicates TP-rate without 
#   additional cumsum, is stationary and tracks eps_t optimally.










# ══════════════════════════════════════════════════════════════════════════════
# Exercise 5: M-SSA Smoothing
# ══════════════════════════════════════════════════════════════════════════════

