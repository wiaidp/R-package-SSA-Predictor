# In this tutorial we explain the SSA criterion based on a simple case study
#   -This background is needed for understanding some of the counter-intuitive `idiosyncrasies' of our approach
#   -For example: the predictor is claimed to predict signs and to control zero-crossings
#   -But the criterion relies on cross-correlations and lag-one autocorrelations only: nothing about signs!  
#   -The example illustrates that SSA effectively `does the job`
# There exist alternative approaches for predicting signs of a target
#   -We argue that SSA is more efficient than these!
#   -In particular, we benchmark SSA to a classic logit model in example 2 of this tutorial
# For background, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5. 


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
# Note that the above plot indicates that gamma is meant as a two-sided (acausal) filter

# We can apply the filter to white noise: xt=epsilont
set.seed(231)
len<-120
# Scaling
sigma<-1
epsilon<-sigma*rnorm(len)
x<-epsilon
# No autocorrelation
acf(x)


# We can filter the data: either by assuming a two-sided a causal design (side=2) or a causal one-sided design (side=1)
y_sym<-filter(x,gamma,side=2)
y_one_sided<-filter(x,gamma,side=1)

tail(cbind(y_sym,y_one_sided))

# When the filter is two-sided (y_sym) the series is left-shifted and we do not observe the filter output 
# towards the sample end (NAs). In contrast, we observe the one-sided filter `till the sample end, but it is right-shifted (delayed)

ts.plot(cbind(y_sym,y_one_sided),col=c("black","black"),lty=1:2,main="One-sided vs. two-sided")

# -In applications one is often interested in obtaining estimates of y_sym towards the sample end
# -An estimate of y_sym at the sample end t=len is called a nowcast: in the above example, we have to compute 
#   forecasts for the future data x_{len+1}, x_{len+2},x_{len+3} missing in the symmetric filter at the sample end. 
#   If we compute MSE-forecasts, then we obtain an MSE estimate of zt at t=len.
# -In our case (white noise) the MSE forecasts of the future data are zero and we obtain the one-sided truncated 
#   filter as our optimal causal MSE nowcast

b_MSE<-gamma[((length(gamma)+1)/2):length(gamma)]
plot(b_MSE,axes=F,type="l",xlab="Lag-structure",ylab="filter-coefficients",main="MSE-nowcast filter")
axis(1,at=1:((length(gamma)+1)/2),labels=-1+1:((length(gamma)+1)/2))
axis(2)
box()

# We can now filter xt with this filter to obtain yt and compare the estimate yt and the target zt
# The filter is one-sided: side=1
y_mse<-filter(x,b_MSE,side=1)

y_sym<-filter(x,gamma,side=2)

ts.plot(cbind(y_sym,y_mse),col=c("black","green"),lty=1:2,main="Target (black) vs MSE (green)")
abline(h=0)

# Holding times were introduced in tutorial 0.1 (mean duration between consecutive zero-crossings)
# 1: empirical ht
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)
# 2. true or expected ht, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
compute_holding_time_func(gamma)$ht
compute_holding_time_func(b_MSE)$ht
# Empirical ht converge to expected ht for increasing sample size len


# The SSA criterion emphasizes zero-crossings or sign accuracy: it aims at maximizing the probability that the 
#   predictor yt forecasts the sign of the target z_{t+delta} correctly

# We first compute the empirical sign-accuracy
SA_empirical<-sum((sign(y_mse*y_sym)+1)/2,na.rm=T)/length(na.exclude(y_sym))
SA_empirical
# If xt and thus yt and zt are Gaussian, then sign-accuracy and target correlation can be linked, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# For this purpose we compute the (true or expected) correlation between target and predictor
filter_mat<-cbind(gamma,c(rep(0,length(gamma)-length(b_MSE)),b_MSE))
colnames(filter_mat)[2]<-"predictor"
filter_mat
# Compute (true) correlation assuming white noise
rho_yz<-filter_mat[,1]%*%filter_mat[,2]/sqrt(filter_mat[,1]%*%filter_mat[,1]*filter_mat[,2]%*%filter_mat[,2])
rho_yz
# The following non-linear expression of rho_yz corresponds to the true/expected SA, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
SA_true<-asin(rho_yz)/pi+0.5
SA_true
# We can also compute a second empirical estimate
asin(cor(na.exclude(cbind(y_sym,y_mse)))[1,2])/pi+0.5
# Here our original estimate
SA_empirical
# One can verify that both empirical estimates of SA converge towards SA_true for large sample sizes



# We here verify this claim: generate a much longer series in order for empirical estimates to converge towards expected (true) values

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

#############################################################################################################
# Example 2
# As claimed, SSA is designed to match signs of a target by a predictor
# There are alternative tools for matching signs
#   -But none (as far as we know) allows to impose a holding-time constraint
# Also, we here argue that SSA is more efficient (irrespective of the added bonus of a ht-constraint) 
# Why?
#   -The SSA criterion emphasizes sign accuracy by maximizing the correlation rho_yz between the target and the predictor
#   -The correlation rho_yz uses the `full' information: it does not restrict attention to signs only
#   -Using the full data information affects efficiency: SSA is more efficient than methods relying on signs only 
#   -Under the posited assumptions, the predictor maximizes the likelihood (efficient)
# Let us check this claim by comparing SSA to a classic logit-model

# Sample length
len<-120
# Target: simple equally weighted design
L<-11
gamma<-rep(1/L,L)

set.seed(23)
# Data
x<-rnorm(len)
# Target signal 
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

# Advantage classic MSE predictor over logit-model: 
#   -The MSE predictor has smaller sample variances or, equivalently, larger t-values
#   -The MSE predictor has the full data as disposal vs. the logit-model which `sees' signs only

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
# As expected, MSE outperforms logit

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
# -MSE has a higher sign accuracy, in the mean over all samples
# -The sample variance of SA is substantially smaller (this result reflects the higher sample variance of coefficient estimates)
#   -Using the signs only in the logit-model discards information in the data
# -MSE outperforms logit in a majority of cases (~90%)



