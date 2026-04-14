# ════════════════════════════════════════════════════════════════════
# TUTORIAL 0.2 — SSA: THE OPTIMIZATION CRITERION
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────
# This tutorial explains the SSA optimization criterion through a
# simple, illustrative case study.
#
# A key motivation is to resolve an apparent paradox in the approach:
#
#   • SSA is designed to predict signs and control zero-crossings
#       → Yet the criterion relies solely on cross-correlations
#         and first-order autocorrelations — with no explicit reference
#         to signs
#
# Exercise 1 demonstrates that, despite this apparent disconnect,
# SSA effectively achieves its stated objectives.

# ── COMPARISON WITH ALTERNATIVE APPROACHES ────────────────────────
# Alternative methods exist for directly predicting the sign of a
# target series. This tutorial argues that SSA is more statistically
# efficient than such alternatives if the full (continuous) information 
# (not only the signs) is available.
#
# The key reason:
#   → SSA exploits the full (continuous) information set of the time series,
#     whereas sign-based approaches (such as logit models) reduce
#     the data to a binary sequence, discarding valuable magnitude
#     information in the process.
#
#   • Exercise 2 benchmarks SSA against a classical logit model,
#     providing direct empirical evidence for this efficiency claim.
#
#   • Exercise 3 extends exercise 2 to a comprehensive Monte Carlo 
#     simulation experiment, providing more refined and detailed 
#     out-of-sample comparisons and statistical significance tests.
# ─────────────────────────────────────────────────────────────────

# ── BACKGROUND ────────────────────────────────────────────────────
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

# Theoretical background:
#   Wildi, M. (2026a) Sign Accuracy, Mean-Squared Error and the Rate
#     of Zero Crossings: a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547

# ─────────────────────────────────────────────────────────────────


rm(list=ls())

# Load SSA-related functions
source(paste(getwd(),"/R/SSA.r",sep=""))
# Load tau-statistic (quantifies lead/lag performance)
source(paste(getwd(),"/R utility functions/Tau_statistic.r",sep=""))
# Load signal extraction functions (used in JBCY paper; depends on mFilter)
source(paste(getwd(),"/R utility functions/HP_JBCY_functions.r",sep=""))


#───────────────────────────────────────────────────────────────────────────
# Exercise 1: EQUIVALENCE OF SIGNS AND CORRELATIONS IN SSA 
#───────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────
# Exercise 1.1: EMPIRICAL FRAMEWORK
#───────────────────────────────────────────────────────────────────────────

# 1.1.1 Target
#------------------------------
# Define a symmetric target filter
# (symmetry is not required in general; see Tutorials 2–5)
gamma <- c(0.25,0.5,0.75,1,0.75,0.5,0.25)

# Plot the symmetric (two-sided) target filter
plot(gamma, axes=F, type="l",
     xlab="Lag structure", ylab="Filter coefficients",
     main="Simple signal extraction (smoothing) filter")
axis(1, at=1:length(gamma),
     labels=(-(length(gamma)+1)/2)+1:length(gamma))
axis(2)
box()
# Interpretation:
# The filter is two-sided (acausal), assigning weights to both
# past and future observations.

# 1.1.2 Data
#----------------------------
# Simulate white noise input (x_t = ε_t)

set.seed(231)
len <- 120
sigma <- 1
epsilon <- sigma * rnorm(len)
x <- epsilon

# Verify absence of autocorrelation
acf(x)

# 1.1.3 Apply filters
#-----------------------------
# side = 2 → symmetric (two-sided, acausal)
# side = 1 → one-sided (causal, real-time)
y_sym <- filter(x, gamma, side=2)
y_one_sided <- filter(x, gamma, side=1)

tail(cbind(y_sym,y_one_sided))

# The symmetric filter is centered but undefined near the sample end (NAs).
# The one-sided filter is available in real time but delayed (right-shifted).
ts.plot(cbind(y_sym,y_one_sided),
        col=c("black","black"), lty=1:2,
        main="Two-sided vs one-sided filter")


#───────────────────────────────────────────────────────────────────────────
# Exercise 1.2 NOWCASTING AT THE SAMPLE END
#───────────────────────────────────────────────────────────────────────────
# In practice, the symmetric output y_sym is often required at
# the sample boundary.

# Definition:
#   • Nowcast: estimate of y_sym at t = len
#   • Forecast: estimate at t = len + δ (δ > 0)
#   • Backcast: estimate at t = len − δ (δ > 0)

# Problem:
#   • The symmetric filter at t = len depends on future values
#     x_{len+1}, x_{len+2}, …
#   • These must be replaced by forecasts


# MSE principle:
#   • Substitute unknown future values with their MSE-optimal forecasts
#   • This yields the MSE-optimal nowcast
#   • The MSE optimal nowcast is better than the one-sided filter in 
#     previous plot

# Special case (white noise):
#   • Optimal forecasts of future values are zero
#   • ⇒ The MSE-optimal nowcast reduces to a truncated one-sided filter
# ───────────────────────────────────────────────────────────────
# Truncated (MSE nowcast) filter
b_MSE <- gamma[((length(gamma)+1)/2):length(gamma)]

plot(b_MSE, axes=F, type="l",
     xlab="Lag structure", ylab="Filter coefficients",
     main="MSE nowcast filter (white noise assumption)")
axis(1, at=1:((length(gamma)+1)/2),
     labels=-1+1:((length(gamma)+1)/2))
axis(2)
box()

# Apply MSE filter and compare with target
y_mse <- filter(x, b_MSE, side=1)
y_sym <- filter(x, gamma, side=2)

ts.plot(cbind(y_sym,y_mse),
        col=c("black","green"), lty=1:2,
        main="Target (black) vs MSE predictor (green)")
abline(h=0)

#───────────────────────────────────────────────────────────────────────────
# Exercise 1.3 HT AND LAG-ONE ACF
#───────────────────────────────────────────────────────────────────────────


# ── HOLDING TIME (ht): EMPIRICAL VS THEORETICAL ────────────────
# Holding time = average duration between consecutive zero-crossings
# (see Tutorial 0.1)

# 1. Empirical ht
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)

# 2. Theoretical ht (white noise case; see Wildi, 2024, 2026a)
# The above sample estimates converge to these values for long samples.
HT_true_gamma<-compute_holding_time_func(gamma)$ht
HT_true_b<-compute_holding_time_func(b_MSE)$ht
HT_true_gamma
HT_true_b

# 3. The theoretical HT is linked bijectively to the lag-one ACF, see 
#     Wildi 2024 and 2026a
# a. Lag-one ACF (assuming white noise)
rho_ff1<-b_MSE[1:(length(b_MSE)-1)]%*%b_MSE[2:length(b_MSE)]/sum(b_MSE^2)
# b. Bijective link via arccos function
# True/expected holding-time of MSE nowcast
pi/acos(rho_ff1)
# The same as
HT_true_b

#--------------------------------------------------------------------
# Empirical HTs converge to theoretical values as len increases, see 
# exercise 1.5 below
#--------------------------------------------------------------------

#───────────────────────────────────────────────────────────────────────────
# Exercise 1.4 SA AND TARGET CORRELATION
#───────────────────────────────────────────────────────────────────────────

# ── SIGN ACCURACY (SA): EMPIRICAL ESTIMATE ─────────────────────
# SA = probability that the predictor correctly matches the sign
#      of the target

SA_empirical <- sum((sign(y_mse * y_sym) + 1)/2, na.rm=T) /
  length(na.exclude(y_sym))
SA_empirical

# ── SIGN ACCURACY: LINK TO CORRELATION ─────────────────────────
# Under Gaussianity, SA is a deterministic function of correlation
# (Wildi, 2024; 2026a)

# a. Construct filter representation
filter_mat <- cbind(gamma,
                    c(rep(0, length(gamma)-length(b_MSE)), b_MSE))
colnames(filter_mat)[2] <- "predictor"
# Target (two-sided) and nowcast (one-sided) filters
filter_mat

# b. True correlation between target and predictor, assuming white noise
rho_yz <- filter_mat[,1] %*% filter_mat[,2] /
  sqrt(filter_mat[,1] %*% filter_mat[,1] *
         filter_mat[,2] %*% filter_mat[,2])
rho_yz

# c. Transform correlation into theoretical SA (see Wildi 2024, proof of 
#    proposition 1)
SA_true <- asin(rho_yz)/pi + 0.5
SA_true
# Compare with empirical SA from above
SA_empirical

#-----------------------------------------------------------
# Empirical SA converges to SA_true as sample size increases, see 
# exercise 1.5 below
#-----------------------------------------------------------

# ── ALTERNATIVE EMPIRICAL SA ESTIMATE ──────────────────────────
# Compute SA via empirical correlation: 
#   Replace above rho_yz by sample estimate.
asin(cor(na.exclude(cbind(y_sym,y_mse)))[1,2])/pi + 0.5

# Compare with true (expected) empirical SA
SA_empirical

#───────────────────────────────────────────────────────────────────────────
# Exercise 1.5 SAMPLE ESTIMATES CONVERGE TO TRUE (EXPECTED) VALUES FOR LARGE SAMPLE SIZES
#───────────────────────────────────────────────────────────────────────────
# Increase sample size to illustrate convergence: 1 million observations
set.seed(65)
len <- 1000000
epsilon <- sigma * rnorm(len)
x <- epsilon

acf(x)

# Apply filters
y_sym <- filter(x,gamma,side=2)
y_mse <- filter(x,b_MSE,side=1)

# Holding time: empirical estimates match true (expected) values 
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)
compute_holding_time_func(gamma)$ht
compute_holding_time_func(b_MSE)$ht

# Sign accuracy: empirical estimates match true (expected) values 
sum((sign(y_mse*y_sym)+1)/2,na.rm=T)/length(na.exclude(y_sym))
SA_true
asin(cor(na.exclude(cbind(y_sym,y_mse)))[1,2])/pi+0.5


#============================================================
# MAIN TAKE AWAYS
#============================================================

# ── DISCUSSION: SSA PRINCIPLE ──────────────────────────────────
# - The closed-form expressions SA_true, HT_true link filter weights directly
#     to sign accuracy and HT, enabling explicit optimization.
#
# - SSA (Wildi, 2024; 2026a) maximizes SA subject to a constraint
#     on holding time (smoothness).
#
# - Equivalent formulation (used in R-code):
#     maximize target correlation rho_yz subject to a constraint on the
#     lag-one ACF of the predictor.
#     → Under Gaussianity, both approaches yield identical solutions.
#     → The link between (SA_true, HT_true) and the filter weights is
#         robust to departures from Gaussianity (Wildi, 2024; 2026a; 2026b).
#     → Regardless of distributional assumptions, the formulation remains
#         interpretable: the objective (target correlation) and the
#         constraint (lag-one ACF) are both meaningful and intuitive.
#
# Bijective Relationships:
# - SA and rho_yz are bijectively related via the arcsin transform:
#     maximizing one is equivalent to maximizing the other.
# - HT and the lag-one ACF are bijectively related via the arccos transform:
#     constraining one is equivalent to constraining the other.
#
# Invariance:
# - SA, rho_yz, ht, and lag-one ACF are invariant to affine scaling.
# - Hence, the SSA solution is defined only up to a scaling factor s.
# - The level is mu=0: for mu!=0 we consider centered data xt-mu 

# Normalization:
# - The scaling factor can be chosen to optimize MSE subject to the
#     imposed ht constraint.
# - If the ht constraint equals the holding time of the MSE-optimal
#     filter (ht_mse), SSA reproduces the MSE solution exactly
#     (see Tutorial 0.3). In this sense, SSA generalizes the MSE approach.
#============================================================





#───────────────────────────────────────────────────────────────────────────
# EXERCISE 2: SSA EFFICIENCY vs LOGISTIC REGRESSION
#───────────────────────────────────────────────────────────────────────────
# SSA is designed to align the signs of a predictor with those 
# of a target series.
#
# While alternative sign-based methods exist, none — to our
# knowledge — supports an explicit holding-time (ht) constraint,
# i.e., direct control over the smoothness of sign changes.
#
# EFFICIENCY ARGUMENT FOR SSA:
#   - SSA maximizes the cross-correlation ρ_yz between target
#     and predictor, exploiting the full continuous information
#     in the data.
#   - Purely sign-based methods (e.g., logit) apply a binary
#     transformation to the target, discarding magnitude
#     information and thereby reducing statistical efficiency.
#   - Under the assumptions considered here, the SSA predictor
#     coincides with the maximum likelihood estimator and is
#     therefore statistically efficient.
#
# COMPARISON SETUP:
#   - SSA is compared against classical logistic regression.
#   - Since logit cannot impose an ht constraint, we set SSA
#     to its unconstrained form: the MSE predictor.
#       → In this setting, SSA reduces to linear regression.
#   - A direct like-for-like comparison between SSA (regression) 
#     and logit is therefore valid.
#───────────────────────────────────────────────────────────────────────────


#------------------------------------------------------------
# SIMULATION SETUP
#------------------------------------------------------------
# Sample length: 120 observations ≈ 10 years of monthly data
# (representative of a typical business-cycle analysis window)
len <- 120

# Target filter: symmetric equally weighted MA of length L = 11
L     <- 11
gamma <- rep(1/L, L)

set.seed(23)

# Simulate white noise input
x <- rnorm(len)

# Apply symmetric (acausal, two-sided) target filter
# NAs appear at both ends due to missing boundary values
z <- filter(x, gamma, side = 2)

ts.plot(cbind(x, z),
        col  = c("black", "red"),
        main = "White noise input (black) and smoothed target signal (red)")


#------------------------------------------------------------
# FORECAST HORIZON & TARGET ALIGNMENT
#------------------------------------------------------------
# delta = 0 → nowcast: estimate z at the current sample end
# Shift z forward by delta so that x_t aligns with z_{t+delta}
delta <- 0
delta <- abs(delta)   # Ensure non-negative horizon

# Align target by advancing it delta steps
z <- c(z[(1 + delta):len], rep(NA, delta))


#───────────────────────────────────────────────────────────────────────────
# Exercise 2.1 SIGN PREDICTION MODELS
#───────────────────────────────────────────────────────────────────────────

# Exercise 2.1.1 LOGISTIC REGRESSION (LOGIT) MODEL
#───────────────────────────────────────────────────────────────────────────

#------------------------------------------------------------
# STEP 1: BINARY TARGET CONSTRUCTION
#------------------------------------------------------------
# Map the sign of the (shifted) target z to {0, 1}:
#   sign(z) = +1 → target = 1  (positive regime)
#   sign(z) = -1 → target = 0  (negative regime)
target <- (1 + sign(z)) / 2
ts.plot(target,
        main = "Binary target: sign of smoothed signal (0/1)")


#------------------------------------------------------------
# STEP 2: LAGGED EXPLANATORY VARIABLES
#------------------------------------------------------------
# Construct a matrix of lagged values of x aligned with the
# shifted target. The number of lags matches the one-sided
# length of the MSE filter for the given delta.
explanatory <- c(x[((L+1)/2 + delta):len],
                 rep(NA, ((L+1)/2 + delta) - 1))

if (((L+1)/2 + delta) < L) {
  for (i in 1:(L - ((L+1)/2 + delta))) {
    explanatory <- cbind(
      explanatory,
      c(x[((L+1)/2 + delta - i):len],
        rep(NA, ((L+1)/2 + delta - i) - 1))
    )
  }
}

colnames(explanatory) <- paste("Lag", 0:(ncol(explanatory) - 1))
tail(explanatory)


#------------------------------------------------------------
# STEP 3: FIT LOGIT MODEL
#------------------------------------------------------------
# Logistic regression of the binary target on lagged x values.
# The model estimates P(sign(z_{t+delta}) = +1 | x_t, x_{t-1}, ...)
sample      <- data.frame(cbind(target, explanatory))
colnames(sample)<-c("target",colnames(explanatory))
logit_model <- glm(target ~ .,
                   family = binomial(link = "logit"),
                   data   = sample)
summary(logit_model)
# Early lags tend to be more informative (larger t-values)
# This is because the weights assigned by the two-sided filter decay,
# assigning less importance to observations away from the center point.


# Exercise 2.1.2: MSE PREDICTOR (LINEAR REGRESSION)
#───────────────────────────────────────────────────────────────────────────
# Same lagged regressors as the logit model, but fitted to
# the CONTINUOUS target z instead of its binary sign.
# No intercept: the MSE filter is constrained to pass through
# the origin (consistent with symmetric filter structure).
mse_model <- lm(z ~ explanatory - 1)
summary(mse_model)

# IN-SAMPLE COMPARISON — MSE vs LOGIT:
#   - MSE predictor: smaller coefficient standard errors
#     (larger t-statistics) → higher estimation precision
#   - Logit model: binary transformation of z discards
#     magnitude information → lower efficiency
#   - Conclusion: continuous target (MSE) uses strictly more
#     information than binary target (logit)



#───────────────────────────────────────────────────────────────────────────
# EXERCISE 2.2: OUT-OF-SAMPLE SIGN ACCURACY COMPARISON
#───────────────────────────────────────────────────────────────────────────
# This section evaluates and compares the out-of-sample sign
# accuracy (SA) of three predictors:
#   1. MSE predictor       (sign of linear regression output)
#   2. Naive logit         (sign of causal logit-filter output)
#   3. Classic logit       (threshold-based P(sign=+1|x) > 0.5)
#
# A large synthetic sample (n = 10,000,000) is used to closely
# approximate true population (expected) SA values.
#
# Theoretical SA is then derived analytically via the arcsin
# transform of the population cross-correlation ρ_yz, and
# compared against the large-sample empirical estimates.

# Note: this experiment is based on a single realization of 
#   length 120.
# Exercise 3 presents a Monte Carlo simulation based on 
#   multiple realizations. 
#============================================================


#------------------------------------------------------------
# 2.2.1 GENERATE LARGE OUT-OF-SAMPLE DATASET
#------------------------------------------------------------
# n = 10,000,000 observations of white noise.
# Sample size is chosen so that empirical SA ≈ population SA.
set.seed(104)
len <- 10000000
x   <- rnorm(len)
# Apply symmetric (acausal, two-sided) target filter
# NAs appear at both ends due to missing boundary values
z <- filter(x, gamma, side = 2)



#------------------------------------------------------------
# 2.2.2 EXTRACT IN-SAMPLE FILTER COEFFICIENTS
#------------------------------------------------------------
# Retrieve estimated filter weights from in-sample models.
#
# MSE predictor: all coefficients are retained.
# Naive logit predictor: the intercept is excluded.
#   Rationale: The intercept is insignificant and out-of-sample 
#   performances degrade
b_mse   <- mse_model$coef
b_logit <- logit_model$coef[-1]   # Intercept dropped

# Inspect L2 norms: scale differs between models, but SA is
# scale-invariant so this does not affect sign predictions
sum(b_mse^2)
sum(b_logit^2)


#------------------------------------------------------------
# 2.2.3 APPLY FILTERS OUT-OF-SAMPLE
#------------------------------------------------------------
# All predictors are applied causally (side = 1): they use
# only current and past values of x, as required in real time.
# The target z is reconstructed using the acausal filter (side = 2).

# --- MSE predictor ---
y_mse <- filter(x, b_mse, side = 1)

# --- Naive logit predictor ---
# Apply the estimated logit coefficient vector directly as a
# causal linear filter. The sign of the output is used as the
# predicted sign of the target — no threshold required.
y_logit_naive <- filter(x, b_logit, side = 1)

# --- Classic logit predictor ---
# Reconstruct the lagged explanatory variable matrix for the
# out-of-sample data using the same structure as in-sample.
explanatory <- c(x[((L+1)/2 + delta):len],
                 rep(NA, ((L+1)/2 + delta) - 1))

if (((L+1)/2 + delta) < L) {
  for (i in 1:(L - ((L+1)/2 + delta))) {
    explanatory <- cbind(
      explanatory,
      c(x[((L+1)/2 + delta - i):len],
        rep(NA, ((L+1)/2 + delta - i) - 1))
    )
  }
}
colnames(explanatory) <- paste("Lag", 0:(ncol(explanatory) - 1))
tail(explanatory)
explanatory <- as.data.frame(explanatory)

# Generate predicted probabilities P(sign(z) = +1 | x)
y_logit_pred <- predict(logit_model,
                        newdata = as.data.frame(explanatory),
                        type    = "response")
# Need a decision rule for the sign forecast:
#   y_logit_pred > 50% corresponds to "plus" forecast (otherwise "minus"). 
threshold_logit    <- 0.5
mat<-cbind(y_logit_pred,sign(y_logit_pred-threshold_logit))
colnames(mat)<-c("Logit predictor",paste("Sign predictor for threshold ",threshold_logit,sep=""))
tail(mat,10)
# Centre predictions around the classification threshold (0.5):
#   > 0  → predicted positive regime
#   < 0  → predicted negative regime
# Note: threshold_logit = 0.54 yields near-perfect concordance with
#       naive logit (not pursued here; 0.5 is the standard choice)
y_logit_centered   <- y_logit_pred - threshold_logit

# Shift predictions forward by (L-1)/2 to match the nowcasting
# alignment used by the MSE and naive logit predictors
y_logit <- c(rep(NA, (L-1)/2),
             y_logit_centered[1:(len - (L-1)/2)])

# Diagnostic: verify sign concordance between naive and classic logit
# (should be near 1.0 — discrepancies reflect threshold sensitivity)
tail(sign(cbind(y_logit_naive, y_logit)))

sign_concordance_logit <- sum((sign(y_logit * y_logit_naive) + 1)/2,
                              na.rm = TRUE) / (len-(L-1)/2)
sign_concordance_logit
# → Values close to 1 confirm near-identical sign predictions
# → Setting threshold_logit = 0.54 leads to nearly 100% agreement


#------------------------------------------------------------
# OPTIONAL: LOGIT INTERCEPT INCLUSION 
#------------------------------------------------------------
# Including the intercept in the naive filter shifts the output and typically
# degrades sign accuracy, as the intercept is not statistically significant.
# Disabled by default.
if (F)
  y_logit_naive <- y_logit_naive + logit_model$coef[1]
  
# Note: Including the intercept in the naive forecast replicates the classic
# logit sign predictor based on a threshold of 0.5 (since the logit applies
# a monotonic transformation, the sign of the output is preserved).
# The following verifies this assertion:
sign_concordance_logit <- sum((sign(y_logit * 
                              (y_logit_naive + logit_model$coef[1])) + 1)/2,
                              na.rm = TRUE) / (len-(L-1)/2)
sign_concordance_logit
# The naive logit sign predictor differs from the classic logit predictor only
# by excluding the statistically insignificant intercept, which in turn
# improves sign accuracy (SA).


#------------------------------------------------------------
# 2.2.4 EMPIRICAL SIGN ACCURACY (LARGE-SAMPLE)
#------------------------------------------------------------
# Sign accuracy (SA): proportion of periods where the
# predictor's sign agrees with the target's sign.
#
# Formula:
#   SA = Σ [ (sign(ŷ_t · z_t) + 1) / 2 ] / n_valid
#
# where n_valid excludes NA values at sample boundaries.

SA_emp_mse <- sum((sign(y_mse         * z) + 1)/2, na.rm = TRUE) /
  length(na.exclude(z))

SA_emp_logit_naive <- sum((sign(y_logit_naive * z) + 1)/2, na.rm = TRUE) /
  length(na.exclude(z))

SA_emp_logit <- sum((sign(y_logit       * z) + 1)/2, na.rm = TRUE) /
  length(na.exclude(z))

# Display empirical SA for all three predictors
SA_emp_mse           # MSE predictor
SA_emp_logit_naive   # Naive logit predictor
SA_emp_logit         # Classic logit predictor (threshold = 0.5)

# Preliminary conclusions:
#   → Naive and classic logit yield near-identical SA at threshold = 0.5
#     ∴ The naive formulation suffices for analytical SA derivation
#   → MSE predictor achieves strictly higher SA than both logit variants
#     ∴ Continuous target information confers a measurable efficiency gain

# We now verify these claims based on true SAs


#───────────────────────────────────────────────────────────────────────────
# EXERCISE 2.3: TRUE (POPULATION-LEVEL) SIGN ACCURACY OUT OF SAMPLE
#───────────────────────────────────────────────────────────────────────────
# The true Sign Accuracy (SA) is derived analytically using the population
# cross-correlation between the target filter γ and the predictor filter b,
# under the assumption of white noise input x_t ~ WN(0, σ²).
#
# Closed-form formula (Wildi 2024, Proposition 1):
#
#   SA_true = arcsin(ρ_yz) / π + 0.5
#
# where ρ_yz = (γ · b) / (‖γ‖ · ‖b‖)
#
#   - γ        : target filter coefficients (ideal/benchmark filter)
#   - b        : predictor filter coefficients (estimated filter)
#   - ρ_yz     : True (target) correlation between y_t and z_t under white noise
#   - arcsin(·): inverse sine function (in radians), mapping ρ_yz ∈ [-1,1]
#                to SA_true ∈ [0, 1]
#
# Notes:
# 1. Since the naive logit and classic logit models produce virtually
#    identical sign predictions, we use the naive logit coefficients
#    (b_logit) for deriving true SA in this analytical derivation.
#
# 2. Applying this closed-form expression to in-sample filter coefficients
#    is equivalent to evaluating out-of-sample performance over an
#    infinitely long out-of-sample span — i.e., it reflects the true
#    asymptotic (population-level) SA rather than a finite-sample estimate.
#───────────────────────────────────────────────────────────────────────────

#------------------------------------------------------------
# 2.3.1 TRUE SA: MSE PREDICTOR
#------------------------------------------------------------
# Zero-pad b_mse to match the length of gamma, then stack
# both filter vectors into a matrix for inner product computation
filter_mat_mse <- cbind(
  gamma,
  c(rep(0, length(gamma) - length(b_mse)), b_mse)
)

# Verify filter alignment (each row = one lag index)
filter_mat_mse

# Population cross-correlation ρ_yz (white noise assumption)
rho_yz_mse <- filter_mat_mse[,1] %*% filter_mat_mse[,2] /
  sqrt((filter_mat_mse[,1] %*% filter_mat_mse[,1]) *
         (filter_mat_mse[,2] %*% filter_mat_mse[,2]))

# Theoretical SA via arcsin transform
SA_true_mse <- asin(rho_yz_mse) / pi + 0.5
SA_true_mse

# Compare with large-sample empirical estimate
SA_emp_mse
# → Close agreement confirms analytical formula is correct


#------------------------------------------------------------
# 2.3.2 TRUE SA: NAIVE LOGIT PREDICTOR
#------------------------------------------------------------
# Same procedure applied to the naive logit coefficient vector.
# Since SA_emp_logit ≈ SA_emp_logit_naive, this derivation is
# representative of both logit variants.
filter_mat_logit <- cbind(
  gamma,
  c(rep(0, length(gamma) - length(b_logit)), b_logit)
)

# Population cross-correlation ρ_yz for logit predictor
rho_yz_logit <- filter_mat_logit[,1] %*% filter_mat_logit[,2] /
  sqrt((filter_mat_logit[,1] %*% filter_mat_logit[,1]) *
         (filter_mat_logit[,2] %*% filter_mat_logit[,2]))

# Theoretical SA via arcsin transform
SA_true_logit_naive <- asin(rho_yz_logit) / pi + 0.5
SA_true_logit_naive

# Compare with large-sample empirical estimate
SA_emp_logit_naive
# → Close agreement confirms consistency of naive logit SA estimate


#============================================================
# MAIN TAKE AWAYS
#============================================================
#
#   1. EQUIVALENCE OF LOGIT VARIANTS:
#      SA_emp_logit_naive ≈ SA_emp_logit
#      → Naive and classic logit yield near-identical sign predictions
#        at threshold = 0.5; naive approach used for analysis.
#
#   2. MSE DOMINANCE (POPULATION):
#      SA_true_mse > SA_true_logit_naive
#      → In expectation, the MSE predictor outperforms logit.
#        The gap reflects the information lost by binarising z.
#
#   3. MSE DOMINANCE (LARGE SAMPLE):
#      SA_emp_mse > SA_emp_logit_naive ≈ SA_emp_logit
#      → Population dominance is confirmed empirically.
#
#   4. CONSISTENCY OF ESTIMATORS:
#      SA_emp ≈ SA_true for both predictors
#      → Large-sample empirical SA converges to the analytical value,
#        validating the closed-form arcsin formula.
#
# CONCLUSION:
#   The MSE predictor (SSA without ht constraint) is strictly more
#   efficient than logistic regression for sign prediction under
#   white noise input. The efficiency gain arises because the MSE
#   approach uses the full continuous information in z, whereas
#   logistic regression operates on its binary projection —
#   discarding magnitude information and reducing statistical power.

# But we used a single realization only.
#============================================================





#───────────────────────────────────────────────────────────────────────────
# EXERCISE 3: MONTE CARLO STUDY — MSE vs LOGIT SIGN ACCURACY
#───────────────────────────────────────────────────────────────────────────
# This Monte Carlo study extends Exercise 2 by evaluating the relative
# out-of-sample Sign Accuracy (SA) performance of the MSE predictor and
# the logistic regression (logit) model across a large number of repeated
# independent samples.
#
# Motivation & Structure:
#  - Exercise 2 was based on a single realization of the data-generating
#    process, meaning its results are subject to sampling variability and
#    may not generalize reliably.
#  - Exercise 3 addresses this limitation by generating multiple independent
#    realizations of the same experiment, enabling a proper Monte Carlo
#    assessment of estimator performance.
#
# By aggregating results across replications, we can compute the following
# distributional moments of the out-of-sample SA performances:
#
#   · Probabilities : e.g., P(SA_logit < SA_MSE) — the proportion of
#                     replications in which one model outperforms the other
#   · Means         : the expected (average) SA for each model across
#                     replications, reflecting typical performance
#   · Variances     : the dispersion of SA across replications, reflecting
#                     the stability and reliability of each model
#
# DESIGN:
#   - Number of replications : anzsim = 1,000
#   - Sample length per run  : len    = 120 (≈ 10 years monthly)
#   - For each replication:
#       1. Simulate white noise input x_t ~ WN(0,1)
#       2. Construct target z_t via symmetric filter γ
#       3. Estimate MSE (linear regression) and logit models
#       4. Compute TRUE SA for each model via the closed-form
#          arcsin formula (see Exercise 2)
# Note: we rely on the naive logit prediction for deriving the true SA.       
#
# RATIONALE FOR USING TRUE SA:
#   - Exercise 2 confirmed that empirical SA (large sample)
#     converges to true SA derived from estimated coefficients.
#   - Using true SA avoids the need to simulate very long
#     out-of-sample series in each replication, substantially
#     reducing computational cost without loss of validity.
#     Nonetheless, we include sample estimates for completeness.
#   - True SA based on estimated model parameters is therefore
#     equivalent to out-of-sample empirical SA on a very long
#     series — it reflects the population SA implied by the
#     in-sample estimated filter.
#
#───────────────────────────────────────────────────────────────────────────
# RESEARCH QUESTIONS
#───────────────────────────────────────────────────────────────────────────
# The Monte Carlo study is designed to address the following three research
# questions regarding the out-of-sample Sign Accuracy (SA) of the MSE
# predictor relative to the logistic regression (logit) model:
#
#  Q1 · MEAN PERFORMANCE
#       Does the MSE predictor achieve higher sign accuracy than logit
#       on average across replications?
#
#         H0: mean(SA_MSE) ≤ mean(SA_logit)
#         H1: mean(SA_MSE) >  mean(SA_logit)
#
#       This question addresses whether MSE is systematically the better
#       model in terms of expected predictive accuracy.
#
#  Q2 · ESTIMATION STABILITY
#       Is the MSE predictor more stable (less variable) across repeated
#       samples than the logit model?
#
#         H0: Var(SA_MSE) ≥ Var(SA_logit)
#         H1: Var(SA_MSE) <  Var(SA_logit)
#
#       This question addresses whether MSE produces more consistent and
#       reliable sign predictions across different realizations of the data.
#
#  Q3 · OUTPERFORMANCE PROBABILITY
#       How likely is it that the MSE predictor outperforms logit in any
#       given out-of-sample replication?
#
#         P(SA_MSE > SA_logit) = ?
#
#       This question provides a replication-level view of dominance,
#       complementing the mean and variance comparisons above.
#───────────────────────────────────────────────────────────────────────────

set.seed(243)
anzsim  <- 1000   # Number of Monte Carlo replications
len     <- 120    # Sample length per replication
mat_perf <- NULL  # Storage matrix for SA results (anzsim × 2)
delta<-0          # Nowcast

#------------------------------------------------------------
# MONTE CARLO LOOP
#------------------------------------------------------------
for (sim in 1:anzsim) {
  
  if (as.integer(sim/100)==sim/100)
    print(paste(100*sim/anzsim,"%",sep=""))
  #----------------------------------------------------------
  # STEP 1: SIMULATE WHITE NOISE INPUT
  #----------------------------------------------------------
  x <- rnorm(len)
  
  #----------------------------------------------------------
  # STEP 2: CONSTRUCT TARGET SIGNAL
  #----------------------------------------------------------
  # Apply symmetric (acausal) target filter γ
  z <- filter(x, gamma, side = 2)
  
  # Shift z forward by delta to align with nowcast/forecast
  # horizon (delta = 0 for nowcast)
  z <- c(z[(1 + delta):len], rep(NA, delta))
  
  #----------------------------------------------------------
  # STEP 3: CONSTRUCT BINARY TARGET FOR LOGIT
  #----------------------------------------------------------
  # Map sign of z to {0, 1}:
  #   sign(z) = +1 → target = 1  (positive regime)
  #   sign(z) = -1 → target = 0  (negative regime)
  target <- (1 + sign(z)) / 2
  
  #----------------------------------------------------------
  # STEP 4: BUILD LAGGED EXPLANATORY VARIABLE MATRIX
  #----------------------------------------------------------
  # Construct causal regressors: lagged values of x aligned
  # with the shifted target z. Structure mirrors the one-sided
  # MSE filter for the given forecast horizon delta.
  explanatory <- c(x[((L+1)/2 + delta):len],
                   rep(NA, ((L+1)/2 + delta) - 1))
  
  if (((L+1)/2 + delta) < L) {
    for (j in 1:(L - ((L+1)/2 + delta))) {
      explanatory <- cbind(
        explanatory,
        c(x[((L+1)/2 + delta - j):len],
          rep(NA, ((L+1)/2 + delta - j) - 1))
      )
    }
  }
  colnames(explanatory) <- paste("Lag", 0:(ncol(explanatory) - 1))
  
  # Combine into a data frame for model fitting
  sample <- data.frame(cbind(target, explanatory))
  colnames(sample)<-c("target",colnames(explanatory))
  
  #----------------------------------------------------------
  # STEP 5: ESTIMATE MODELS
  #----------------------------------------------------------
  # Logit model: binary target ~ lagged regressors
  logit_model <- glm(target ~ .,
                     family = binomial(link = "logit"),
                     data   = sample)
  
  # MSE model: continuous target ~ lagged regressors (no intercept)
  # No intercept: consistent with zero-mean white noise assumption
  mse_model <- lm(z ~ explanatory - 1)
  
  # Extract estimated filter coefficients
  b_mse   <- mse_model$coef
  b_logit <- logit_model$coef[-1]   # Drop intercept (see above comments)
  
  
  #----------------------------------------------------------
  # STEP 6: COMPUTE TRUE SA — MSE PREDICTOR
  #----------------------------------------------------------
  # These correspond to population out-of-sample SA (see exercise 2)
  # Zero-pad b_mse to match the length of γ, then compute
  # population cross-correlation ρ_yz (white noise assumption)
  filter_mat_mse <- cbind(
    gamma,
    c(rep(0, length(gamma) - length(b_mse)), b_mse)
  )
  
  rho_yz_mse <- filter_mat_mse[,1] %*% filter_mat_mse[,2] /
    sqrt((filter_mat_mse[,1] %*% filter_mat_mse[,1]) *
           (filter_mat_mse[,2] %*% filter_mat_mse[,2]))
  
  # True SA via arcsin transform (Wildi 2024, Proposition 1)
  SA_true_mse <- asin(rho_yz_mse) / pi + 0.5
  
  #----------------------------------------------------------
  # STEP 7: COMPUTE TRUE SA — LOGIT PREDICTOR
  #----------------------------------------------------------
  # Same procedure applied to the naive logit coefficient vector
  # (intercept excluded; SA is invariant to affine scaling)
  filter_mat_logit <- cbind(
    gamma,
    c(rep(0, length(gamma) - length(b_logit)), b_logit)
  )
  
  rho_yz_logit <- filter_mat_logit[,1] %*% filter_mat_logit[,2] /
    sqrt((filter_mat_logit[,1] %*% filter_mat_logit[,1]) *
           (filter_mat_logit[,2] %*% filter_mat_logit[,2]))
  
  # True SA via arcsin transform
  SA_true_logit <- asin(rho_yz_logit) / pi + 0.5
  
  # For completeness we also compute sample estimates
  #----------------------------------------------------------
  # STEP 8: COMPUTE OUT-OF-SAMPLE SA 
  # (not strictly necessary but provided for completeness)
  #----------------------------------------------------------
  # Out of sample data  
  x <- rnorm(len)
  # target  
  z <- filter(x, gamma, side = 2)
  # MSE
  y_mse <- filter(x, b_mse, side = 1)
  # --- Naive logit predictor ---
  y_logit_naive <- filter(x, b_logit, side = 1)
  # --- Classic logit predictor ---
  # Reconstruct the lagged explanatory variable matrix for the
  # out-of-sample data using the same structure as in-sample.
  explanatory <- c(x[((L+1)/2 + delta):len],
                   rep(NA, ((L+1)/2 + delta) - 1))
  if (((L+1)/2 + delta) < L) {
    for (i in 1:(L - ((L+1)/2 + delta))) {
      explanatory <- cbind(
        explanatory,
        c(x[((L+1)/2 + delta - i):len],
          rep(NA, ((L+1)/2 + delta - i) - 1))
      )
    }
  }
  colnames(explanatory) <- paste("Lag", 0:(ncol(explanatory) - 1))
  tail(explanatory)
  explanatory <- as.data.frame(explanatory)
  # Generate predicted probabilities P(sign(z) = +1 | x)
  y_logit_pred <- predict(logit_model,
                          newdata = as.data.frame(explanatory),
                          type    = "response")
  threshold_logit    <- 0.5
  y_logit_centered   <- y_logit_pred - threshold_logit
  # Shift predictions forward by (L-1)/2 to match the nowcasting
  # alignment used by the MSE and naive logit predictors
  y_logit <- c(rep(NA, (L-1)/2),
               y_logit_centered[1:(len - (L-1)/2)])
# Empirical SAs  
  SA_emp_mse <- sum((sign(y_mse         * z) + 1)/2, na.rm = TRUE) /
    length(na.exclude(z))
  SA_emp_logit_naive <- sum((sign(y_logit_naive * z) + 1)/2, na.rm = TRUE) /
    length(na.exclude(z))
  SA_emp_logit <- sum((sign(y_logit       * z) + 1)/2, na.rm = TRUE) /
    length(na.exclude(z))

  #----------------------------------------------------------
  # STEP 8: STORE RESULTS
  #----------------------------------------------------------
  mat_perf <- rbind(mat_perf, c(SA_true_mse, SA_true_logit,SA_emp_mse,
                                SA_emp_logit_naive,SA_emp_logit))
}

# Label columns for clarity
colnames(mat_perf) <- c("SA_true_MSE", "SA_true_Logit","SA_emp_MSE",
                        "SA_emp_naive_logit","SA_emp_logit")


#------------------------------------------------------------
# MONTE CARLO RESULTS: SUMMARY STATISTICS
#------------------------------------------------------------

# Answer to Q1: MEAN SA: MSE predictor achieves higher average SA
#    → Consistent with efficiency advantage over logit
cat("\n--- Mean Sign Accuracy ---\n")
print(apply(mat_perf, 2, mean))

# Answer to Q2: STANDARD DEVIATION: MSE predictor is more stably estimated
#    → Logit's binary transformation inflates sampling variability
cat("\n--- Standard Deviation of Sign Accuracy ---\n")
print(apply(mat_perf, 2, sd))

# Note: Although the sample estimates (last three entries) are considerably 
# noisier, the difference in standard deviation between SA_emp_logit and 
# SA_emp_MSE remains roughly consistent with the difference observed between 
# SA_true_logit and SA_true_MSE.

#------------------------------------------------------------
# STATISTICAL TEST: DIFFERENCE IN MEAN SA
#------------------------------------------------------------
# 1. Testing differences between true SA estimates:
# One-sided Welch t-test:
#   H0: mean(SA_true_mse) ≤ mean(SA_true_Logit)
#   H1: mean(SA_true_MSE) >  mean(SA_true_Logit)  ← expected direction
#
# Welch variant (var.equal = FALSE) used because MSE and logit
# SA distributions have materially different variances.
cat("\n--- Welch t-test: H1: mean(SA_MSE) > mean(SA_Logit) ---\n")
print(t.test(mat_perf[,"SA_true_MSE"], mat_perf[,"SA_true_Logit"],
             paired      = T,
             alternative = "greater",
             var.equal   = FALSE))
# Outcome: statistically significant differences 

# 2. Testing differences between sample SA estimates:
print(t.test(mat_perf[,"SA_emp_MSE"], mat_perf[,"SA_emp_logit"],
             paired      = T,
             alternative = "greater",
             var.equal   = FALSE))
# Outcome: statistically significant differences 

#------------------------------------------------------------
# Answer to Q3: WIN RATE: FREQUENCY OF MSE OUTPERFORMING LOGIT
#------------------------------------------------------------
# Proportion of replications in which SA_MSE > SA_Logit
win_rate_logit <- length(which(mat_perf[,1] < mat_perf[,2])) / anzsim

cat("\n--- Frequency: Logit outperforms MSE ---\n")
cat("Logit wins in", round(win_rate_logit * 100, 1),
    "% of replications\n")
cat("MSE wins in  ", round((1 - win_rate_logit) * 100, 1),
    "% of replications\n")
# → MSE dominates in approximately 90% of all replications

# Test of significance
binom.test(length(which(mat_perf[,1] < mat_perf[,2])), anzsim, p = 0.5,
           alternative ="less",conf.level = 0.95)


#------------------------------------------------------------
# CONSOLIDATED FINDINGS & CONCLUSIONS
#------------------------------------------------------------
# 1. MEAN SA ADVANTAGE:
#    E[SA_MSE] > E[SA_Logit]
#    → The MSE predictor achieves higher average sign accuracy
#      across all Monte Carlo replications.
#
# 2. VARIANCE ADVANTAGE:
#    Var[SA_MSE] < Var[SA_Logit]
#    → MSE coefficient estimates are more stable across samples.
#      Logit's binary transformation amplifies sampling noise,
#      leading to greater variability in the estimated filter
#      and hence in the resulting SA.
#
# 3. WIN RATE:
#    P(SA_MSE > SA_Logit) ≈ 0.90
#    → The MSE predictor outperforms logit in ~90% of samples,
#      confirming robust and consistent dominance.
#
# 4. SOURCE OF EFFICIENCY GAIN:
#    The logit model maps the continuous target z to a binary
#    {0,1} variable before estimation. This discards magnitude
#    information, reduces the effective signal-to-noise ratio,
#    and leads to both lower mean SA and higher SA variability.
#    The MSE predictor avoids this information loss entirely.
#
# CONCLUSION:
#    Under white noise input and a symmetric target filter, the
#    MSE predictor (SSA without ht constraint) strictly dominates
#    logistic regression in terms of sign accuracy — both in
#    expectation and with high probability across samples.
#    This illustrates the core efficiency argument for SSA.
#============================================================


