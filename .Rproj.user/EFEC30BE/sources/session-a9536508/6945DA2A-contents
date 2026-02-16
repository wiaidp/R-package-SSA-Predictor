# We here present and discuss a fundamental forecast trilemma
# See Wildi, M. (2024) Business Cycle Analysis and Zero-Crossings of Time Series: a Generalized Forecast Approach: https://doi.org/10.1007/s41549-024-00097-5.


rm(list=ls())

# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

#------------------------------------------------------------
# Example 0: Forecast trilemma and the MSE approach

# Let xt=epsilont be white noise and zt=Gamma(xt) be the output of a simple two-sided equally-weighted MA filter
# Target
L<-11
gamma<-rep(1/L,L)

# Two-sided target filter
plot(gamma,axes=F,type="l",xlab="Lag-structure",ylab="filter-coefficients",main="Simple equally-weighted (smoothing) filter")
axis(1,at=1:length(gamma),labels=(-(length(gamma)+1)/2)+1:length(gamma))
axis(2)
box()

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


# Suppose we want to estimate y_sym at the end point t=len, where the above output is NA
# This is  called a nowcast
#   Estimating y_sym at t=len-delta, delta>0, is a backcast
#   Estimating y_sym at t=len+delta, delta>0, is a forecast
#   For a nowcast delta=0
delta<-0

# The optimal mean-square error (MSE) predictor is obtained by replacing unknown future x_{len+j} by their 
#   MSE-forecasts, see tutorial 0.3. For white noise the MSE-forecast is the mean i.e. 0.
# Therefore the MSE-filter or predictor is just the one-sided truncated gamma
b_mse<-gamma[((L+1)/2+delta):L]

# Apply the filters to data
y_mse<-filter(x,b_mse,side=1)
y_sym<-filter(x,gamma,side=2)

short_sample<-cbind(y_sym,y_mse)
# Plot: zero crossings or sign changes are marked by vertical lines
ts.plot(short_sample,col=c("black","green"),lty=1:2,main="Target (black) vs MSE (green) with sign changes")
abline(h=0)
abline(v=1+which(sign(y_mse[2:len])!=sign(y_mse[1:(len-1)])),col="green")
abline(v=1+which(sign(y_sym[2:len])!=sign(y_sym[1:(len-1)])),col="black")

# The predictor appears right-shifted (lag) and noisier
#   Noisy in this context means: more crossings. 
# We can measure noisiness or smoothness of predictor and target
#   Simple measure: compute the mean distance between consecutive zero-crossings
#   For this purpose we generate a longer series in order to obtain reliable empirical estimates
set.seed(16)
len<-12000
# Scaling
sigma<-1
epsilon<-sigma*rnorm(len)
x<-epsilon
y_mse<-filter(x,b_mse,side=1)
y_sym<-filter(x,gamma,side=2)

# We now compute the empirical mean distance between consecutive zero-crossings: this is called holding-time (ht)
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)
# The empirical holding-time is larger for the two-sided target: it has less crossings
# The predictor generates roughly 30% more `noisy' crossings
compute_empirical_ht_func(y_sym)$empirical_ht/compute_empirical_ht_func(y_mse)$empirical_ht


# We can also compute the true (expected) ht, if the data is white noise, see  Wildi, M. (2024)
#   This number is connected to the lag-one autocorrelation function (ACF)
compute_holding_time_func(gamma)$ht
compute_holding_time_func(b_mse)$ht
# Empirical estimates will converge to the expected ht when increasing the length of the time series

# We can also compute the relative lag of the predictor when measured against the target
# For this purpose, we set-up a matrix with the competing filters
#   The first column contains the filter with the reference crossings: in our case the two-sided target
#   The second column contains the filter whose crossings will be measured against the reference: the mse predictor
filter_mat<-cbind(y_sym,y_mse)

# The following function shifts the series in the second column against the series in the first column for various leads and lags
# For each shift it computes the distances between (closest) zero-crossings of both series and sums these distances
# The shift at which the sum is smallest is a measure for the lead (left shift) or lag (right shift) of the series 
#   in the first column,  see Wildi, M. (2024) for background
# The following plot suggest a lead of the target (a lag of the MSE) of three time-units: the minimum of 
#   the curve is at -3
compute_min_tau_func(filter_mat)

# Let's shift the predictor accordingly in the short sample

short_sample_shifted<-cbind(short_sample[1:(nrow(short_sample)-3),1],short_sample[4:nrow(short_sample),2])

# Plot shifted series: this seems about right in the mean
#   Sometimes earlier, sometimes later...
ts.plot(short_sample_shifted,col=c("black","green"),lty=1:2,main="Shift predictor (green line) by 3 units to the left")
abline(h=0)

# We can infer the lag or right-shift directly from the transfer function of the filter or, more precisely, 
#   from the phase-shift
K<-600
# Compute amplitude of gap_diff
shift<-amp_shift_func(K,b_mse,F)$shift

# Plot shift function
plot(shift,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-Shift MSE predictor",sep=""))
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# The plot suggests that low-frequency components (in the passband of the predictor) are lagged by 2.5 time units
# This is an exact measure for the shift vs. the empirical measures provided above




# From the above we infer that the predictor generates ~30% more crossings and that crossings are lagged by roughly 3 time units

# Finally, we can compute the mean-square error
mean((y_sym-y_mse)^2,na.rm=T)
# The `true' MSE (expected squared error) can be obtained easily by realizing that we substituted 0-forecasts for the future xt
#   Therefore the error is the cummulated weighted sum of future xt
#   Since xt is white noise the `true' MSE must be
sum(sigma^2*gamma[1:((L-1)/2+delta)]^2)
# The empirical MSE converges to the expected squared error for longer time series

# We can now think of the above numbers in terms of forecast performances
# A `good' predictor should have a small MSE, a small lag and nearly the same holding-time (ht) 
#   as the target 
# These three terms constitute a forecast trilemma: one cannot improve all three of them at once (unless the original design is severely misspecified)
# Intuitively, these numbers depend on the forecast horizon delta
# In order to verify this conjectur we compute performance numbers for a selection of forecast horizons

# A selection of backcast, nowcast and forecast horizons
delta_vec<--((L-1)/2):((L-1)/2)
delta_vec
ht<-mse<-shift<-NULL
# Loop through all forecast horizons and compute corresponding performances of predictors
for (i in 1:length(delta_vec))# i<-1
{
  delta<-delta_vec[i]
# Compute predictor for assigned forecast horizon delta  
  b_mse<-gamma[((L+1)/2+delta):L]
# Performances  
# 1. Compute true ht  
  ht<-c(ht,compute_holding_time_func(b_mse)$ht)
# 2. Compute shift: lead/lag  
  shift<-c(shift,amp_shift_func(K,b_mse,F)$shift[1])
# 3. Compute MSE  
  if (((L-1)/2+delta)>0)
  {  
    mse<-c(mse,sum(sigma^2*gamma[1:((L-1)/2+delta)]^2))
  } else
  {
# first backcast horizon is such that the target can be replicated perfectly (no future data involved)
    mse<-c(mse,0)
  }
}
table<-rbind(ht,shift,mse)
rownames(table)<-c("holding-time","shift","mse")
colnames(table)<-paste("delta=",delta_vec,sep="")
# Let's look at the outcome
table

# Discussion
# -The MSE predictor emphasizes MSE performances, only/exclusively
#   MSE does not allow for explicit controls of smoothness (ht) or timeliness (shift)
# -MSE performances and ht worsen with increasing delta (forecast horizon)
#   The predictor becomes noisier and it lies farther away from the target
# -The lag decreases with increasing delta 
#   -Note that the lag here always refers to a nowcast (estimate of y_sym at the sample end)
# -The table illustrates a forecast trilemma: improving one or two performance measures always 
#   affects negatively (at least one of) the remaining one(s)

# Purpose of SSA
# -Modify characteristics of the predictor in terms of smoothness (ht) and timeliness (shift)
#   -At costs of MSE
# -Optimality concept: 
#   -For a given ht (smoothness) the SSA-predictor outperforms all competing designs in terms of MSE
# -Timeliness: for a given/fixed ht, determine delta such that 
#   1. SSA(ht,delta) is sufficiently `fast' (leading) 
#   2. MSE-performances are adequate (not unduly compromised)
# Trilemma (Part I): 
# -Smoothness is obtained by imposing ht
# -Timeliness is obtained by selecting delta
# -MSE is obtained by computing the best MSE filter, conditional on ht and delta

# Trilemma (Part II): 
# -Since SSA optimizes MSE subject to ht and delta, SSA effectively addresses a trilemma
# -Increasing ht or delta worsens MSE
# -MSE can be enhanced by reducing ht or delta
# -For given ht: delta (lead) can be traded against MSE
# -For given shift: ht can be traded against MSE
# -Improving shift and ht simultaneously affects MSE disproportionately

# See tutorials 1-5 for applications and example 8 in tutorial 2 for a comprehensive example based on the HP filter 

# Final notes
# 1. SSA currently addresses timeliness by changing delta (the forecast horizon)
# -This proceeding `works' and it is intuitively appealing
# -However, timeliness can be addressed more directly, more fundamentally and more effectively: 
#   A corresponding extension is on the way 
# 2. McElroy and Wildi (2018) address a forecast trilemma, too: AST-trilemma (use your favorite search engine)
# -However, their so called ATS-trilemma does not optimize MSE conditional on ht and shift
# -SSA improves on the ATS-trilemma mainly in terms of interpretability
