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
# ─────────────────────────────────────────────────────────────────

rm(list=ls())

# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


#----------------------------------------------------------------
# Example 1: xt=epsilont white noise
# Assume the following symmetric target filter (gamma does not have to be symmetric, see tutorials 2-5)
gamma<-c(0.25,0.5,0.75,1,0.75,0.5,0.25)

# Symmetric target filter
plot(gamma,axes=F,type="l",xlab="Lag-structure",ylab="filter-coefficients",main="Simple signal extraction (smoothing) filter")
axis(1,at=1:length(gamma),labels=(-(length(gamma)+1)/2)+1:length(gamma))
axis(2)
box()
# Note that the above plot indicates that gamma is meant as a two-sided (acausal) filter with weights assigned to both past and future data

# We can apply the filter to white noise: xt=epsilont
set.seed(231)
len<-120
# Scaling
sigma<-1
epsilon<-sigma*rnorm(len)
x<-epsilon
# No autocorrelation
acf(x)


# We can filter the data: either by assuming a two-sided acausal design (side=2) or a causal one-sided design (side=1)
y_sym<-filter(x,gamma,side=2)
y_one_sided<-filter(x,gamma,side=1)

tail(cbind(y_sym,y_one_sided))

# When the filter is two-sided (y_sym) the series is left-shifted and we do not observe the filter output 
# towards the sample end (NAs). In contrast, we observe the one-sided filter `till the sample end, but it is right-shifted (delayed)

ts.plot(cbind(y_sym,y_one_sided),col=c("black","black"),lty=1:2,main="One-sided vs. two-sided")

# ── NOWCASTING AT THE SAMPLE END ──────────────────────────────────
# In practice, estimates of the symmetric filter output (y_sym)
# are often required at or near the sample boundary.
#
# Definition:
#   • An estimate of y_sym at the sample end (t = len) is called
#     a nowcast
#
# The nowcasting problem:
#   • At t = len, the symmetric filter requires future observations
#     x_{len+1}, x_{len+2}, x_{len+3} that are not yet available
#   • These missing values must be replaced by forecasts
#   • Substituting MSE-optimal forecasts yields the MSE-optimal
#     estimate of y_sym at the sample end
#
# Special case — white noise input:
#   • MSE forecasts of future white noise observations are zero
#   • The optimal causal MSE nowcast therefore reduces to the
#     one-sided truncated filter
# ─────────────────────────────────────────────────────────────────

b_MSE<-gamma[((length(gamma)+1)/2):length(gamma)]
plot(b_MSE,axes=F,type="l",xlab="Lag-structure",ylab="filter-coefficients",main="MSE-nowcast filter, assuming white noise data")
axis(1,at=1:((length(gamma)+1)/2),labels=-1+1:((length(gamma)+1)/2))
axis(2)
box()

# We can now filter xt with this filter to obtain yt and compare the estimate yt and the target zt
# The filter is one-sided: side=1
y_mse<-filter(x,b_MSE,side=1)

y_sym<-filter(x,gamma,side=2)

ts.plot(cbind(y_sym,y_mse),col=c("black","green"),lty=1:2,main="Target (black) vs MSE (green)")
abline(h=0)

# ── HOLDING-TIME: EMPIRICAL VS. EXPECTED ──────────────────────────
# Holding-time (ht): mean duration between consecutive zero-crossings
# Introduced in Tutorial 0.1.

# 1. Empirical ht — two-sided and one-sided filters
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)

# 2. True (expected) ht — see Wildi, M. (2024) and (2026a), Section 2
compute_holding_time_func(gamma)$ht
compute_holding_time_func(b_MSE)$ht

# Note: empirical ht converges to expected ht as sample size (len)
# increases.

# ── SIGN ACCURACY: EMPIRICAL ESTIMATE ─────────────────────────────
# The SSA criterion targets sign accuracy (SA):
#   → Maximize the probability that the predictor yt correctly
#     forecasts the sign of the target z_{t+delta}

# Step 1 — Compute the empirical sign accuracy
SA_empirical <- sum((sign(y_mse * y_sym) + 1) / 2, na.rm = T) /
  length(na.exclude(y_sym))
SA_empirical

# ── SIGN ACCURACY: THEORETICAL LINK TO CORRELATION ────────────────
# Under Gaussianity, sign accuracy and target-predictor correlation
# are analytically linked — see Wildi, M. (2024), (2026a).

# Step 2 — Compute the true (expected) correlation between
#           target and predictor (assuming white noise)
filter_mat <- cbind(gamma,
                    c(rep(0, length(gamma) - length(b_MSE)), b_MSE))
colnames(filter_mat)[2]<-"predictor"

rho_yz <- filter_mat[, 1] %*% filter_mat[, 2] /
  sqrt(filter_mat[, 1] %*% filter_mat[, 1] *
         filter_mat[, 2] %*% filter_mat[, 2])
rho_yz

# Step 3 — Derive true (expected) SA from rho_yz
# The following non-linear transform of rho_yz yields the
# true/expected SA — see Wildi, M. (2024), (2026a)
SA_true <- asin(rho_yz) / pi + 0.5
SA_true

# Empirical SA (step 1) converges to theoretical SA (step 3) for large sample sizes len

# ── SIGN ACCURACY: ALTERNATIVE EMPIRICAL ESTIMATE ─────────────────
# An alternative empirical SA estimate based on applying the asin function to the empirical target correlation
asin(cor(na.exclude(cbind(y_sym,y_mse)))[1,2])/pi+0.5
# Compare with original empirical estimate
SA_empirical
# Notice how both empirical SA estimates approach the true SA value (SA_true) as the sample size increases.
# To confirm this, we generate a considerably larger time series, allowing the empirical estimates to converge to their 
#   theoretical (true) values.
set.seed(65)
len<-120000
# Scaling
sigma<-1
epsilon<-sigma*rnorm(len)
x<-epsilon
# No autocorrelation
acf(x)


# Apply filters
y_sym<-filter(x,gamma,side=2)
y_mse<-filter(x,b_MSE,side=1)

# Holding times were emphasized in tutorial 0
# 1: empirical ht
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)
# 2. true or expected ht
compute_holding_time_func(gamma)$ht
compute_holding_time_func(b_MSE)$ht


# Sign accuracy: SA
# 1. Empirical SA
sum((sign(y_mse*y_sym)+1)/2,na.rm=T)/length(na.exclude(y_sym))
SA_true
asin(cor(na.exclude(cbind(y_sym,y_mse)))[1,2])/pi+0.5


# Discussion:
# -By relying on the closed-form `true' expression SA_true for SA, we can link the predictor to Sign Accuracy and maximize accuracy as 
#     a function of filter weights: optimality
# -SSA criterion, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5 
#   -Maximize SA under a constraint for ht
# -Alternative (effective) criterion as implemented in R-code
#   -Maximize rho_yz under a constraint for lag-one ACF: the solution is the same!
# -SA and correlations are linked bijectively by the monotonous arc-sin function + an affine transformation
#   -Therefore maximizing either expression must generate the same optimal SSA-predictor

# Note: rho_yz, SA, ht or the lag-one ACF are indifferent to affine transformations
#   -The solution of SSA is determined up to an arbitrary scaling constant s: any s is allowed 
#   -We solve this undeterminacy by computing that particular scaling s which maximizes MSE-performances under the posited ht-constraint
#   -Therefore, if we insert the holding-time ht_mse of the optimal MSE predictor, then SSA replicates the latter,
#     see tutorial 0.3

# Discussion:
# - The closed-form expression SA_true links the predictor directly to Sign Accuracy (SA),
#     enabling explicit maximization of SA as a function of the filter weights. This defines
#     the notion of optimality in the SSA framework.
#
# - The SSA criterion (Wildi, M. (2024), (2026a), section 2) maximizes SA subject to a holding-time constraint ht.
#
# - An equivalent, computationally efficient formulation - as implemented here - maximizes
#     the cross-correlation rho_yz subject to a constraint on the lag-one ACF of the
#     predictor. Both formulations share the same optimal solution under Gaussianity.
#     In the absence of Gaussianity, both criteria are generally close (see the discussion in the cited literature)
#
# - Under Gaussianity, SA and rho_yz are linked bijectively via the monotone arc-sine function (up to an affine
#     transformation). Consequently, maximizing either expression yields the same optimal
#     SSA predictor.
#
# Note: rho_yz, SA, ht, and the lag-one ACF are all invariant to affine transformations.
#   - The SSA solution is determined only up to an arbitrary scaling constant s; any
#       value of s yields a valid solution with respect to the SA and ht criteria.
#   - This indeterminacy is resolved by selecting the scaling s that maximizes MSE
#       performance subject to the imposed ht constraint.
#   - Therefore, when the ht constraint is set to match the holding time ht_mse of the
#       optimal MSE predictor, SSA exactly replicates the MSE solution (see tutorial 0.3).


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




