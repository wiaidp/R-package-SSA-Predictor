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
# The case study demonstrates that, despite this apparent disconnect,
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


#----------------------------------------------------------------
# Exercise 1: White noise input (x_t = ε_t)

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

# Simulate white noise input
set.seed(231)
len <- 120
sigma <- 1
epsilon <- sigma * rnorm(len)
x <- epsilon

# Verify absence of autocorrelation
acf(x)

# Apply filters:
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

# ── NOWCASTING AT THE SAMPLE END ────────────────────────────────
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

# ── HOLDING TIME (ht): EMPIRICAL VS THEORETICAL ────────────────
# Holding time = average duration between consecutive zero-crossings
# (see Tutorial 0.1)

# 1. Empirical ht
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)

# 2. Theoretical ht (white noise case; see Wildi, 2024, 2026a)
HT_true_gamma<-compute_holding_time_func(gamma)$ht
HT_true_b<-compute_holding_time_func(b_MSE)$ht
HT_true_gamma
HT_true_b

# 3. The theoretical HT is linked bijectively to the lag-one ACF, see 
#     Wildi 2024 and 2026a
rho_ff1<-b_MSE[1:(length(b_MSE)-1)]%*%b_MSE[2:length(b_MSE)]/sum(b_MSE^2)
# True/expected holding-time of MSE nowcast
pi/acos(rho_ff1)
# The same as
HT_true_b


#--------------------------------------------------------------------
# Empirical HTs converge to theoretical values as len increases
#--------------------------------------------------------------------

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
# Empirical SA converges to SA_true as sample size increases
#-----------------------------------------------------------

# ── ALTERNATIVE EMPIRICAL SA ESTIMATE ──────────────────────────
# Compute SA via empirical correlation: 
#   Replace above rho_yz by sample estimate.
asin(cor(na.exclude(cbind(y_sym,y_mse)))[1,2])/pi + 0.5

# Compare with true (expected) empirical SA
SA_empirical

#===========================================================================================
# VERIFICATION: SAMPLE ESTIMATES CONVERGE TO TRUE (EXPECTED) VALUES FOR LARGE SAMPLE SIZES
#===========================================================================================
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



#############################################################################################################
# Exercise 2
# SSA is designed to align the signs of a predictor with those of a target 
#   series.
# While alternative sign-based methods exist, none— to our knowledge—
#   allows for an explicit constraint on holding time (smoothness in terms of 
#   frequency of sign changes).
#
# Beyond this additional flexibility, SSA is expected to be more efficient:
#   - SSA maximizes the cross-correlation rho_yz between target and predictor,
#       thereby exploiting the full (continuous) information in the data.
#   - In contrast, purely sign-based methods rely on binary transformations,
#       discarding magnitude information and thus reducing efficiency.
#   - Under the assumptions considered here, the SSA predictor coincides with
#       the maximum likelihood estimator and is therefore statistically 
#       efficient.
#
# We illustrate this by comparing SSA to a classical logistic regression 
#   (logit) model.
# For simplicity, we consider the case where SSA reduces to the MSE predictor
#   (i.e., no holding-time constraint is imposed).
# In this setting, SSA is equivalent to linear regression.

# Sample length: 120 observations correspond to 10 years of monthly data 
#   (typical business-cycle framework)
len <- 120

# Target filter: simple equally weighted design
L <- 11
gamma <- rep(1/L, L)

set.seed(23)

# Simulated data
x <- rnorm(len)

# Target signal (two-sided / acausal filter)
z <- filter(x, gamma, side=2)

ts.plot(cbind(x,z), col=c("black","red"),
        main="Data (black) and target (red)")

# Forecast horizon: nowcast
delta <- 0
# Ensure positiveness
delta<-abs(delta)

# Shift z forward by delta
z<-c(z[(1+delta):len],rep(NA,delta))

#-------------------------------
# Logit model

# 1. Construct binary target (map signs to {0,1}): shift to the left by delta
target <- (1 + sign(z)) / 2
ts.plot(target)

# 2. Build matrix of lagged explanatory variables
explanatory <- c(x[((L+1)/2+delta):len],
                 rep(NA,((L+1)/2+delta)-1))

if (((L+1)/2+delta) < L)
{
  for (i in 1:(L-((L+1)/2+delta)))
  {
    explanatory <- cbind(
      explanatory,
      c(x[((L+1)/2+delta-i):len],
        rep(NA,((L+1)/2+delta-i)-1))
    )
  }
}
colnames(explanatory)<-paste("Lag ",0:(ncol(explanatory)-1))
tail(explanatory)

# Need a data frame
sample <- data.frame(cbind(target, explanatory))

# 3. Estimate logit model
logit_model <- glm(target ~ ., family=binomial(link='logit'),
                   data=sample)
summary(logit_model)

#-----------------------------
# MSE predictor (linear regression)
# Same regressors, but using the continuous target z
mse_model <- lm(z ~ explanatory - 1)
summary(mse_model)

# Comparison:
#   - The MSE predictor yields smaller coefficient variances
#       (larger t-statistics), indicating higher estimation precision.
#   - It exploits the full continuous information in z,
#       whereas the logit model uses only sign information.
#   - This loss of information in the logit model leads to lower efficiency.

#-------------------------------------------------
# Out-of-sample comparison

set.seed(104)

# Very long (out-of-sample) series to approximate population (true) SA
len <- 10000000
x <- rnorm(len)

# Extract estimated coefficients from above in-sample experiment (exclude intercept for logit)
b_mse <- mse_model$coef
b_logit <- logit_model$coef[-1]

# Note: the scale of the logit predictor is arbitrary;
# sign accuracy is invariant to scaling.
sum(b_mse^2)
sum(b_logit^2)

# Apply filters to new data (out-of-sample)
# Apply causal filters: side=1
y_mse <- filter(x, b_mse,side=1)
y_logit <- filter(x, b_logit,side=1)

# Optional: include intercept (worsens performance slightly)
if (F)
  y_logit <- y_logit + logit_model$coef[1]
  
# Target: symmetric filter (side=2)
z <- filter(x, gamma,side=2)
# Empirical sign accuracy
sum((sign(y_mse*z)+1)/2, na.rm=T) / length(na.exclude(z))
sum((sign(y_logit*z)+1)/2, na.rm=T) / length(na.exclude(z))
# → MSE predictor outperforms logit

#-------------------------------------------------
# We now compute true (expected) sign accuracy

# 1. MSE predictor
filter_mat <- cbind(gamma,
                    c(rep(0,length(gamma)-length(b_mse)), b_mse))
# Check: filters are aligned correctly
filter_mat
# Compute true target correlation (between MSE predictor and target z)
# The formula assumes white noise 
rho_yz <- filter_mat[,1] %*% filter_mat[,2] /
  sqrt(filter_mat[,1] %*% filter_mat[,1] *
         filter_mat[,2] %*% filter_mat[,2])
rho_yz
# Compute true sign accuracy based on correlation (see Wildi 2024, proof of 
# proposition 1)
SA_true_mse <- asin(rho_yz)/pi + 0.5
SA_true_mse

# 2. Logit predictor
filter_mat <- cbind(gamma,
                    c(rep(0,length(gamma)-length(b_logit)), b_logit))
rho_yz <- filter_mat[,1] %*% filter_mat[,2] /
  sqrt(filter_mat[,1] %*% filter_mat[,1] *
         filter_mat[,2] %*% filter_mat[,2])
SA_true_logit <- asin(rho_yz)/pi + 0.5

# True SA comparison: 
SA_true_mse
SA_true_logit
# Compare with empirical estimates (the latter converge to the former):
sum((sign(y_mse*z)+1)/2, na.rm=T) / length(na.exclude(z))
sum((sign(y_logit*z)+1)/2, na.rm=T) / length(na.exclude(z))
# → Confirms MSE dominance


#############################################################################################################
# Exercise 3: Monte Carlo study 
# The following Monte Carlo study is based on exercise 2: 
#   -Simulate multiple series (anzsim<-1000)
#   -For each series, compute SA based on estimated model parameters
#     (based on linear regression and logit models)
#     Note: -We have verified in exercise 2 that empirical SAs converge to 
#             true SAs based on model parameters
#           -Accordingly, we can avoid simulating long time series and rely on 
#             true SAs instead (based on empirical models).
#   -True SAs based on empirical models are equivalent to out-of-sample 
#       empirical SAs based on very long series.
# Idea: Compare empirical means and variances of true SAs for both models
# Outcome: 
#   -Mean SA of regression is larger (regression outperforms logit)
#   -Variance of SA is smaller (regression is more precise)
#   -Regression outperforms logit with probability ~0.9 (in 90% of cases).
set.seed(43)

anzsim <- 1000
len <- 120
mat_perf <- NULL

for (i in 1:anzsim)
{
  # Simulate data
  x <- rnorm(len)
  
  # Target regression: two-sided filter
  z <- filter(x, gamma, side=2)
  # Shift forward by delta
  z<-c(z[(1+delta):len],rep(NA,delta))
  
  # Target logit
  target <- (1 + sign(z)) / 2
  
  # Build regressors
  explanatory <- c(x[((L+1)/2+delta):len],
                   rep(NA,((L+1)/2+delta)-1))
  
  if (((L+1)/2+delta) < L)
  {
    for (i in 1:(L-((L+1)/2+delta)))
    {
      explanatory <- cbind(
        explanatory,
        c(x[((L+1)/2+delta-i):len],
          rep(NA,((L+1)/2+delta-i)-1))
      )
    }
  }
  colnames(explanatory)<-paste("Lag ",0:(ncol(explanatory)-1))
  
  
  sample <- data.frame(cbind(target, explanatory))
  
  # Estimate models
  logit_model <- glm(target ~ ., family=binomial(link='logit'),
                     data=sample)
  mse_model <- lm(z ~ explanatory - 1)
  
  b_mse <- mse_model$coef
  b_logit <- logit_model$coef[-1]
  
  # Compute true SA for MSE predictor
  filter_mat <- cbind(gamma,
                      c(rep(0,length(gamma)-length(b_mse)), b_mse))
  rho_yz <- filter_mat[,1] %*% filter_mat[,2] /
    sqrt(filter_mat[,1] %*% filter_mat[,1] *
           filter_mat[,2] %*% filter_mat[,2])
  SA_true_mse <- asin(rho_yz)/pi + 0.5
# Compute true SA for logit predictor
  
  filter_mat <- cbind(gamma,
                      c(rep(0,length(gamma)-length(b_logit)), b_logit))
  rho_yz <- filter_mat[,1] %*% filter_mat[,2] /
    sqrt(filter_mat[,1] %*% filter_mat[,1] *
           filter_mat[,2] %*% filter_mat[,2])
  SA_true_logit <- asin(rho_yz)/pi + 0.5
  
  mat_perf <- rbind(mat_perf, c(SA_true_mse, SA_true_logit))
}

colnames(mat_perf) <- c("SA MSE","SA Logit")

# Summary statistics
# 1. Mean: regression outperforms logit (mean SA larger)
apply(mat_perf, 2, mean)
# 2. Standard error: regression has markedly smaller variance 
apply(mat_perf, 2, sd)

# Test difference in means
t.test(mat_perf[,1], mat_perf[,2],
       paired=F, alternative="greater", var.equal=F)

# Frequency with which logit outperforms MSE: only in 10% of all cases
length(which(mat_perf[,1] < mat_perf[,2])) / anzsim

# Findings:
#   - The MSE predictor achieves higher average sign accuracy (SA).
#   - Its sampling variability is smaller, reflecting more precise estimation.
#   - The efficiency loss of the logit model stems from discarding magnitude information.
#   - The MSE predictor outperforms logit in the vast majority of samples (~90%).


























