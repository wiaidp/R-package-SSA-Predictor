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
# efficient than such alternatives.
#
# The key reason:
#   → SSA exploits the full information set of the time series,
#     whereas sign-based approaches (such as logit models) reduce
#     the data to a binary sequence, discarding valuable magnitude
#     information in the process.
#
#   • Example 2 benchmarks SSA against a classical logit model,
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
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic (quantifies lead/lag performance)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))
# Load signal extraction functions (used in JBCY paper; depends on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


#----------------------------------------------------------------
# Example 1: White noise input (x_t = ε_t)

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
compute_holding_time_func(gamma)$ht
compute_holding_time_func(b_MSE)$ht

# Empirical estimates converge to theoretical values as len increases

# ── SIGN ACCURACY (SA): EMPIRICAL ESTIMATE ─────────────────────
# SA = probability that the predictor correctly matches the sign
#      of the target

SA_empirical <- sum((sign(y_mse * y_sym) + 1)/2, na.rm=T) /
  length(na.exclude(y_sym))
SA_empirical

# ── SIGN ACCURACY: LINK TO CORRELATION ─────────────────────────
# Under Gaussianity, SA is a deterministic function of correlation
# (Wildi, 2024; 2026a)

# Construct filter representation
filter_mat <- cbind(gamma,
                    c(rep(0, length(gamma)-length(b_MSE)), b_MSE))
colnames(filter_mat)[2] <- "predictor"

# True correlation between target and predictor
rho_yz <- filter_mat[,1] %*% filter_mat[,2] /
  sqrt(filter_mat[,1] %*% filter_mat[,1] *
         filter_mat[,2] %*% filter_mat[,2])
rho_yz

# Transform correlation into theoretical SA (see Wildi 2024 and 2026a)
SA_true <- asin(rho_yz)/pi + 0.5
SA_true

# Empirical SA converges to SA_true as sample size increases

# ── ALTERNATIVE EMPIRICAL SA ESTIMATE ──────────────────────────
# Compute SA via empirical correlation
asin(cor(na.exclude(cbind(y_sym,y_mse)))[1,2])/pi + 0.5

# Compare with direct empirical SA
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
# - The closed-form expression SA_true links filter weights directly
#     to sign accuracy, enabling explicit optimization.
#
# - SSA (Wildi, 2024; 2026a) maximizes SA subject to a constraint
#     on holding time (smoothness).
#
# - Equivalent formulation (used in practice):
#     maximize correlation rho_yz subject to a constraint on the
#     lag-one ACF of the predictor.
#     → Under Gaussianity, both approaches yield identical solutions.
#
# - SA and correlation are bijectively related via the arcsin transform
#     (up to an affine transformation), so maximizing one implies
#     maximizing the other.

# Invariance:
# - SA, rho_yz, ht, and lag-one ACF are invariant to affine scaling.
# - Hence, the SSA solution is defined only up to a scaling factor s.
# - The level is mu=0: for mu!=0 we consider centered data xt-mu 

# Normalization:
# - The scaling factor can be chosen to optimize MSE subject to the
#     imposed ht constraint.
# - If the ht constraint equals the holding time of the MSE-optimal
#     filter (ht_mse), SSA reproduces the MSE solution exactly
#     (see Tutorial 0.3).



#############################################################################################################
# Example 2
# SSA is designed to align the signs of a predictor with those of a target series.
# While alternative sign-based methods exist, none— to our knowledge—
#   allows for an explicit constraint on holding time (smoothness in terms of frequency of sign changes).
#
# Beyond this additional flexibility, SSA is expected to be more efficient:
#   - SSA maximizes the cross-correlation rho_yz between target and predictor,
#       thereby exploiting the full (continuous) information in the data.
#   - In contrast, purely sign-based methods rely on binary transformations,
#       discarding magnitude information and thus reducing efficiency.
#   - Under the assumptions considered here, the SSA predictor coincides with
#       the maximum likelihood estimator and is therefore statistically efficient.
#
# We illustrate this by comparing SSA to a classical logistic regression (logit) model.
# For simplicity, we consider the case where SSA reduces to the MSE predictor
#   (i.e., no holding-time constraint is imposed).
#   In this setting, SSA is equivalent to linear regression.

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

#-------------------------------
# Logit model

# 1. Construct binary target (map signs to {0,1})
target <- (1 + sign(z)[(1+2*delta-1):len]) / 2
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

dim(explanatory)
tail(explanatory)

# Arrange data so that the first column corresponds to the most recent observation
sample <- data.frame(cbind(target, explanatory[,ncol(explanatory):1]))

# 3. Estimate logit model
logit_model <- glm(target ~ ., family=binomial(link='logit'),
                   data=sample)
summary(logit_model)

#-----------------------------
# MSE predictor (linear regression)
# Same regressors, but using the continuous target z
mse_model <- lm(z ~ explanatory[,ncol(explanatory):1] - 1)
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

# Very long series to approximate population (true) SA
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
y_mse <- filter(x, b_mse)
y_logit <- filter(x, b_logit)

# Optional: include intercept (worsens performance slightly)
if (F)
  y_logit <- y_logit + logit_model$coef[1]
  
# Target
z <- filter(x, gamma)

# Empirical sign accuracy
sum((sign(y_mse*z)+1)/2, na.rm=T) / length(na.exclude(z))
sum((sign(y_logit*z)+1)/2, na.rm=T) / length(na.exclude(z))
# → MSE predictor outperforms logit

#-------------------------------------------------
# We now compare true (expected) sign accuracy

# 1. MSE predictor
filter_mat <- cbind(gamma,
                    c(rep(0,length(gamma)-length(b_mse)), b_mse))
rho_yz <- filter_mat[,1] %*% filter_mat[,2] /
  sqrt(filter_mat[,1] %*% filter_mat[,1] *
         filter_mat[,2] %*% filter_mat[,2])
SA_true_mse <- asin(rho_yz)/pi + 0.5

# 2. Logit predictor
filter_mat <- cbind(gamma,
                    c(rep(0,length(gamma)-length(b_logit)), b_logit))
rho_yz <- filter_mat[,1] %*% filter_mat[,2] /
  sqrt(filter_mat[,1] %*% filter_mat[,1] *
         filter_mat[,2] %*% filter_mat[,2])
SA_true_logit <- asin(rho_yz)/pi + 0.5

# True SA comparison: empirical SA above converge to true values
SA_true_mse
SA_true_logit
# → Confirms MSE dominance

#-------------------------------------------
# Monte Carlo study: 
#   -simulate multiple series
#   -For each series, compute SA based on estimated model parameters
#     (based on linear regression and logit models)
#   -Compare empirical means and variances of SAs for both models
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
  
  # Target
  z <- filter(x, gamma, side=2)
  target <- (1 + sign(z)[(1+2*delta-1):len]) / 2
  
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
  
  sample <- data.frame(cbind(target,
                             explanatory[,ncol(explanatory):1]))
  
  # Estimate models
  logit_model <- glm(target ~ ., family=binomial(link='logit'),
                     data=sample)
  mse_model <- lm(z ~ explanatory[,ncol(explanatory):1] - 1)
  
  b_mse <- mse_model$coef
  b_logit <- logit_model$coef[-1]
  
  # Compute true SA
  filter_mat <- cbind(gamma,
                      c(rep(0,length(gamma)-length(b_mse)), b_mse))
  rho_yz <- filter_mat[,1] %*% filter_mat[,2] /
    sqrt(filter_mat[,1] %*% filter_mat[,1] *
           filter_mat[,2] %*% filter_mat[,2])
  SA_true_mse <- asin(rho_yz)/pi + 0.5
  
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
apply(mat_perf, 2, mean)
apply(mat_perf, 2, sd)

# Test difference in means
t.test(mat_perf[,1], mat_perf[,2],
       paired=F, alternative="greater", var.equal=F)

# Frequency with which logit outperforms MSE
length(which(mat_perf[,1] < mat_perf[,2])) / anzsim

# Findings:
#   - The MSE predictor achieves higher average sign accuracy (SA).
#   - Its sampling variability is smaller, reflecting more precise estimation.
#   - The efficiency loss of the logit model stems from discarding magnitude information.
#   - The MSE predictor outperforms logit in the vast majority of samples (~90%).










































#############################################################################################################
# Example 2
# SSA is designed to match the signs of a target series with a predictor.
# While alternative sign-matching approaches exist, none - to our knowledge -
#   supports an explicit holding-time constraint.
#
# Beyond this added flexibility, we argue that SSA is inherently more efficient.
# The reasoning is as follows:
#   - SSA maximizes the cross-correlation rho_yz between target and predictor,
#       which leverages the full distributional information of the data.
#   - In contrast, methods that operate on signs only discard the magnitude
#       information, leading to efficiency losses.
#   - Under the assumptions posited here, the SSA predictor maximizes the
#       likelihood, and is therefore statistically efficient.
#
# We examine this claim by benchmarking SSA against a classical logistic
#   regression (logit) model.
# For illustration, we assume that SSA is the MSE predictor, i.e., we skip the ht constraint
#   In this case SSA=linear regression

# Sample length
len<-120
# Target: simple equally weighted design
L<-11
gamma<-rep(1/L,L)

set.seed(23)
# Data
x<-rnorm(len)
# Target signal: apply two-sided (acausal) filter
z<-filter(x,gamma,side=2)

ts.plot(cbind(x,z),col=c("black","red"),main="Data (black) and target (red)")

# Forecast horizon: nowcast
delta<-0

#-------------------------------
# Apply Logit-Model
# 1. Compute signs of target and map to 0,1 for logit-fit
target<-(1+sign(z)[(1+2*delta-1):len])/2
ts.plot(target)

# 2. Compute matrix of explanatory series
explanatory<-c(x[((L+1)/2+delta):len],rep(NA,((L+1)/2+delta)-1))
if (((L+1)/2+delta)<L)
{
  for (i in 1:(L-((L+1)/2+delta)))#i<-1
  {
    explanatory<-cbind(explanatory,c(x[((L+1)/2+delta-i):len],rep(NA,((L+1)/2+delta-i)-1)))
  }
}
dim(explanatory)
# Stacked and shifted explanatory data for regression
tail(explanatory)

# data set: we invert column ordering such that first column corresponds to most recent data (important when filtering series below)
sample<-data.frame(cbind(target,explanatory[,ncol(explanatory):1]))

# 3. Fit logit-model
logit_model <- glm(target ~.,family=binomial(link='logit'),data=sample)
summary(logit_model)
#-----------------------------
# Fit classic regression: MSE predictor
# We invert column ordering such that first column corresponds to most recent data (important when filtering series below)
mse_model<-lm(z~explanatory[,ncol(explanatory):1]-1)
summary(mse_model)

# Advantages of the MSE predictor over the logit model:
#   - The MSE predictor yields smaller sampling variances and, equivalently,
#       larger t-statistics, indicating more precise coefficient estimates.
#   - The MSE predictor exploits the full interval-scaled data, whereas the
#       logit model operates on binary sign indicators only, discarding
#       magnitude information and reducing statistical efficiency.

#-------------------------------------------------
# Let us now apply both predictors out-of-sample

set.seed(104)
# Generate a very long series for empirical SA to converge to expected/true SA
len<-10000000
x<-rnorm(len)
# Extract the predictor weights from the estimated objects: we skip the intercept
b_mse<-mse_model$coef
b_logit<-logit_model$coef[-1]
# Note that the scale of the logit-model predictor is `arbitrary': but we here consider SA only (which is indifferent to scaling)
sum(b_mse^2)
sum(b_logit^2)
# Apply empirical MSE and logit-filters to data
y_mse<-filter(x,b_mse)
y_logit<-filter(x,b_logit)
# If desired, one can add the intercept (performances are slightly worse with the intercept)
if (F)
  y_logit<-y_logit+logit_model$coef[1]
# Target
z<-filter(x,gamma)

# Empirical SA of MSE-predictor
sum((sign(y_mse*z)+1)/2,na.rm=T)/length(na.exclude(z))
# Empirical SA of logit-predictor
sum((sign(y_logit*z)+1)/2,na.rm=T)/length(na.exclude(z))
# MSE outperforms logit!

# We can also compute the true or expected SA of both predictors, 
#   The empirical out-of-sample estimates converge to the true SA, for increasing out-of-sample span
# 1. MSE
filter_mat<-cbind(gamma,c(rep(0,length(gamma)-length(b_mse)),b_mse))
filter_mat
rho_yz<-filter_mat[,1]%*%filter_mat[,2]/sqrt(filter_mat[,1]%*%filter_mat[,1]*filter_mat[,2]%*%filter_mat[,2])
rho_yz
SA_true_mse<-asin(rho_yz)/pi+0.5
# 2. Logit
filter_mat<-cbind(gamma,c(rep(0,length(gamma)-length(b_logit)),b_logit))
filter_mat
rho_yz<-filter_mat[,1]%*%filter_mat[,2]/sqrt(filter_mat[,1]%*%filter_mat[,1]*filter_mat[,2]%*%filter_mat[,2])
rho_yz
SA_true_logit<-asin(rho_yz)/pi+0.5
# True SA:
SA_true_mse
SA_true_logit
# As expected, MSE outperforms logit. Also, the previous empirical SA converge to these true numbers

#-------------------------------------------
# The above results were based on a single long sample of xt
#   We now perform a simulation over multiple `normal-size' samples and look at the sample distribution of SA: mean, sd

set.seed(43)
# Number of simulation runs
anzsim<-1000
# Length cor
len<-120
mat_perf<-NULL
for (i in 1:anzsim)
{ 
# Compute data
  x<-rnorm(len)
# Target: 
  z<-filter(x,gamma,side=2)
  target<-(1+sign(z)[(1+2*delta-1):len])/2

  # Compute matrix of explanatory series
  explanatory<-c(x[((L+1)/2+delta):len],rep(NA,((L+1)/2+delta)-1))
  if (((L+1)/2+delta)<L)
  {
    for (i in 1:(L-((L+1)/2+delta)))#i<-1
    {
      explanatory<-cbind(explanatory,c(x[((L+1)/2+delta-i):len],rep(NA,((L+1)/2+delta-i)-1)))
    }
  }
# data set: we invert column ordering such that first column is most recent data
  sample<-data.frame(cbind(target,explanatory[,ncol(explanatory):1]))
# Fit models
  logit_model <- glm(target ~.,family=binomial(link='logit'),data=sample)
#  We invert column ordering such that first column is most recent data  
  mse_model<-lm(z~explanatory[,ncol(explanatory):1]-1)
  b_mse<-mse_model$coef
  b_logit<-logit_model$coef[-1]
# Compute true SA of both predictors: expected or true out-of-sample performances
  filter_mat<-cbind(gamma,c(rep(0,length(gamma)-length(b_mse)),b_mse))
  filter_mat
  rho_yz<-filter_mat[,1]%*%filter_mat[,2]/sqrt(filter_mat[,1]%*%filter_mat[,1]*filter_mat[,2]%*%filter_mat[,2])
  rho_yz
  SA_true_mse<-asin(rho_yz)/pi+0.5
  filter_mat<-cbind(gamma,c(rep(0,length(gamma)-length(b_logit)),b_logit))
  filter_mat
  rho_yz<-filter_mat[,1]%*%filter_mat[,2]/sqrt(filter_mat[,1]%*%filter_mat[,1]*filter_mat[,2]%*%filter_mat[,2])
  rho_yz
  SA_true_logit<-asin(rho_yz)/pi+0.5
# Collect SA: MSE in first column, logit in second column  
  mat_perf<-rbind(mat_perf,c(SA_true_mse,SA_true_logit))
  
}
colnames(mat_perf)<-c("SA MSE","SA Logit")
# Mean of sample SA for MSE (first column) and logit (second column)
apply(mat_perf,2,mean)
# Standard errors: differences of means are significant 
apply(mat_perf,2,sd)
# Mean differences are significant
t.test(mat_perf[,1], mat_perf[,2], paired = F, alternative = "greater",var.equal=F)
# Proportion of cases where MSE is outperformed by logit
length(which(mat_perf[,1]<mat_perf[,2]))/anzsim

# Findings:
#   - The MSE predictor achieves higher Sign Accuracy (SA) on average across samples.
#   - The sampling variance of SA is substantially smaller for the MSE predictor,
#       reflecting the more precise coefficient estimates noted above.
#       This gain in precision stems directly from the logit model's reliance on
#       binary sign indicators, which discards the magnitude information present
#       in the data.
#   - Consequently, the MSE predictor outperforms the logit model in approximately
#       90% of simulated samples.




