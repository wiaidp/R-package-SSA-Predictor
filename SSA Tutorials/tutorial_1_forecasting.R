# In this tutorial we propose applications of SSA mainly to classic forecasting (not signal extraction/nowcasting). 
# -All examples emphasize the holding-time (ht) concept as a method and concept for controlling the smoothness 
#   of a (forecast or signal extraction) filter in a systematic and predictable way. 
# -We relate the theoretical or expected ht to the effective (empirical or measured) ht, see example 1.
# -We discuss feasibility, see example 2. 
# -We improve smoothness of a simple one-step ahead predictor, see example 3.
# -We replicate the MSE predictor by SSA, see example 4.
# -We `play' with the flexible interface and interchange role of data-generating process and target filter, see example 5.
# -We propose an `unsmoothing' exercise whereby SSA is asked to generate more zero-crossings 
#   than the benchmark predictor, see example 6
# -We analyze simple cases of model misspecification, for which expected and empirical hts differ, and we 
#   show how to resolve the mismatch or misspecification by simple adjustments, see example 7.  
# -Finally, in the last example we replicate the HP filter designs, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5


rm(list=ls())

library(xts)
# Load the library mFilter
# HP and BK filters
library(mFilter)

# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

#---------------------------------------------------------
# Example 1
# Illustrate the holding-time, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5


# Let xt be a realization of length 12000 of an AR(1)-process
# We need a long series in order to obtain an accurate empirical holding-time estimate
len<-12000
a1<-0.8
set.seed(1)
x<-arima.sim(n = len, list(ar = a1))

ts.plot(x)
# Typical patterns for acf and pacf
acf(x)
acf(x,type="partial")
# Estimation
ar_obj<-arima(x,order=c(1,0,0))
# Diagnostics are OK
tsdiag(ar_obj)



# Holding time
# Lets have a look at the holding time of the AR(1) process
# We first compute the empirical holding-time i.e. the mean-duration between consecutive zero-crossings of the data
empirical_ht<-len/length(which(sign(x[2:len])!=sign(x[1:(len-1)])))
empirical_ht
# We have implemented the formula in a function
empirical_ht<-compute_empirical_ht_func(x)
empirical_ht

# We now rely on the exact holding-time expression, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# For that purpose we need the MA-inversion of the process: we can use the true a1 or the MSE estimate for that purpose
# The function ARMAtoMA can invert arbitrary stationary ARMA-specifications
xi<-c(1,ARMAtoMA(ar=a1,lag.max=len-1))
# Note: we can interpret the AR(1)-process as MA(infty)-filter applied to epsilont
# We can then plug the filter (MA-inversion) into the function compute_holding_time_func
ht_obj<-compute_holding_time_func(xi)
# This is the 'true' or expected holding time of the AR(1)-filter (or process): compare with the empirical one above
ht_obj$ht
# The function also computes the lag-one acf which is of course a1 in this case
ht_obj$rho_ff1

# If the lag-one acf is known, then we can compute the holding-time ht with the function compute_holding_time_from_rho_func
rho<-a1
compute_holding_time_from_rho_func(rho)

# If the holding-time ht is known, then we can compute the lag-one acf with the function compute_rho_from_ht
ht<-ht_obj$ht
compute_rho_from_ht(ht)


# We can analyze finite sample issues
# For that purpose consider a much shorter sample of the above process
len<-100
a1<-0.8
set.seed(1)
x<-arima.sim(n = len, list(ar = a1))

ts.plot(x)
# Typical patterns for acf and pacf
acf(x)
acf(x,type="partial")
# Estimation
ar_obj<-arima(x,order=c(1,0,0))
# Diagnostics OK
tsdiag(ar_obj)

ahat<-ar_obj$coef["ar1"]

rho<-ahat
# Compare with true/expected holding-time above
compute_holding_time_from_rho_func(rho)
# See also tables 3 and 4 in Wildi 2024 and the comments referring to the sampling error
#   -The sampling error is mostly irrelevant in applications because it cancels in relative terms
#   -SSA is mainly about relative performances, against a benchmark


#----------------------------------------------------------------------------------------------------
# Example 2
# Inconsistent settings: 
#   A MA-filter of length L cannot exceed an upper limit for the holding-time
#   We here briefly illustrate such a case

# We assume that the data is the above AR(1)-process or filter
len<-120
xi<-c(1,ARMAtoMA(ar=a1,lag.max=len-1))
# Filter length: proposition 3 suggests that L depends on ht in the sense that ht should not exceed the maximal holding-time of a MA(L)  
# We here purposely select a (too) small L
L<-5
# Target: identity (we shall explain this setting below)
gammak_generic<-1
# Forecast horizon: one-step ahead
forecast_horizon<-1
# Here is the problem: ht is too large for a filter of length 5
ht<-7
# Note that we need to supply rho1 (instead of ht) to SSA_func below
rho1<-compute_rho_from_ht(ht)
rho1
# This function computes the maximal lag-one acf for a filter of length L, see proposition 3: the maximum is smaller than rho1 above 
#  In this case there does not exist a SSA-solution (L is too small)
rhomax_func(L)
# Estimation function: 
# There are two optimization routines: brute-force grid-search (for 'exotic' cases) and fast triangulation (for most practically relevant applications)
#   Fast triangulation can handle all cases such that the imposed holding-time does not exceed some limit which depends on L (see examples further down)
# The function generates an error message: ht is too large (rho1>rhomax)
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# Note also that we do not provide the data sample xt to SSA_func!!!!!
# The whole information about the data is summarized in xi: if xi==NULL then we assume white noise, otherwise xi 
#  are the weights of the MA-inversion (Wold decomposition)

# Back to the above example: we can now either increase L or decrease ht
# Let's first increase L
L<-15
# Target: identity (we shall explain this setting below)
gammak_generic<-1
# Forecast horizon: one-step ahead
forecast_horizon<-1
# Same ht as above
ht<-7
# Note that we need to supply rho1 (instead of ht) to SSA_func below
rho1<-compute_rho_from_ht(ht)
rho1
# This function computes the maximal lag-one acf, see proposition 3: since rhomax>rho1 we can obtain a solution
rhomax_func(L)

SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# See tutorial 0.3 for reference of the return: what is ssa_x? 
# This is the optimal SSA-filter: of length L
SSA_obj$ssa_x


# Let's now decrease ht (with the short filter length)
L<-5
# Now ht is OK
ht<-4
# Note that we need to supply rho1 (instead of ht) to SSA_func below
rho1<-compute_rho_from_ht(ht)
rho1
rhomax_func(L)

SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_obj$ssa_x

# General remark: if rho1 in the holding-time constraint is close to rhomax, then the solution of the (SSA-) 
#   optimization problem can look `weird'. We recommend to set L sufficiently large, as a function of rho1, 
#   so that this problem could be avoided. As a general rule, L>=2*ht.  





#---------------------------------------------------------------------------------------
# Example 3
# Forecasting:  
# Perform one-step ahead forecasting for the above AR(1) filter or process

a1<-0.8
# Compute the Wold-decomposition (use true or empirical AR-estimate)
len<-100
# Reminder: we do not supply data to the SSA estimation function; the optimization relies on xi (and on target) only
# One could substitute the finite sample estimate of a1
xi<-c(1,ARMAtoMA(ar=a1,lag.max=len-1))

# In general we want the SSA-filter to be smoother (less zero-crossings)
# For that purpose we select ht larger than the holding-time of the AR(1)-filter, see exercise 1 above
ht<-6
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# Filter length
# Should be sufficiently large (see exercise 2 above) but smaller than sample length
# If filter parameters decay sufficiently fast to zero then the selected L is OK
# Larger L do not lead to overfitting of the SSA-filter unless xi (the MA-inversion) is overfitted
L<-20
# Target: in our case this is the identity since we want to forecast the original data xt
# In the signal extraction example further down the target is typically a non-trivial filter applied to xt (not the identity)
# Summary: for forecasting set gammak_generic=1
gammak_generic<-1
# Forecast horizon: one-step ahead
forecast_horizon<-1
# We retain the previous settings for the numerical optimization
# Note that the function checks if the length of the target matches L: if not, a warning is printed meaning that the target is artificially extended with zeroes in order to match the length L
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

ssa_x<-SSA_obj$ssa_x
# Plot optimized filter
ts.plot(ssa_x)

# Various checks:
# 1. Verify holding-time: compute empirical holding-time based on a very long time series
len<-100000
set.seed(1)
x<-arima.sim(n = len, list(ar = a1))

# Compute filter output
yhat<-filter(x,ssa_x,sides = 1)
# Compute empirical holding-time
empirical_ht<-compute_empirical_ht_func(yhat)
empirical_ht
# compare with imposed constraint: seems to work!
ht

# 2. Compare lag-one acf of optimized design with ht: 
#   If both numbers are identical (up to rounding errors) then the optimization converged to the global maximum
#   If there is a substantial difference: increase split_grid (number of iterations: default is 20 which should be fine for nearly all applications)
# In our example both numbers match almost perfectly (they invariably due in `non-exotic' applications)
SSA_obj$crit_rhoyy
rho1

# 3. SSA criterion values: two different criteria, see proposition 4 in JBCY paper
# Both criteria lead to the same solution, but they measure different performances
# 3.1 Correlation with one-step ahead forecast (MSE estimate)
SSA_obj$crit_rhoyz
# Compare with empirical correlation
# Compute one-step ahead (MSE-) predictor
MSE_forecast<-a1*x
cor(yhat,MSE_forecast,use='pairwise.complete.obs')

# 3.2 Correlation with effective target i.e. series shifted forward by forecast_horizon
SSA_obj$crit_rhoy_target
# Compare with empirical correlation: target=original series shifted forward by forecast_horizon
cor(yhat,c(x[(1+forecast_horizon):len],rep(NA,forecast_horizon)),use='pairwise.complete.obs')
# The sample estimate converges to the criterion value for large sample size
# Both targets are equivalent, see proposition 4 in JBCY paper: both lead to the same SSA-filter 
# -The correlation with the effective target (second criterion) is smaller because the latter assumes knowledge of future observations (in this case: the one-step ahead observation) 
# -The correlation with the MSE target is larger because the latter is one-sided (it does not assume knowledge of future data)
# -If the holding-time imposed to SSA coincides with the holding-time of MSE, then SSA just replicates MSE, see tutorial 0.3 and the next example 4 below

#---------------------------------------------------------------------
# Example 4
# Replicate MSE solution: this is the same as example 3 except that rho1=a1 in the holding-time constraint, i.e., 
#   we replicate the lag-one ACF of the MSE-filter (or its holding-time) by SSA
#   -Setting rho1=a1 in the holding-time constraint replicates the lag-one ACF of MSE
#   -Therefore SSA also replicates MSE

a1<-0.8
# Compute the Wold-decomposition (use true or empirical AR-estimate)
len<-100
# Reminder: we do not supply data to the SSA estimation function; the optimization relies on xi (and on target) only
xi<-c(1,ARMAtoMA(ar=a1,lag.max=len-1))
# Replicate lag-one-acf of one-step ahead (MSE) forecast filter: this is just a1
rho1<-a1
# Filter length
L<-20
# Target: in our case this is the identity since we want to forecast the original data xt
gammak_generic<-1
# Forecast horizon: one-step ahead
forecast_horizon<-1
# We retain the previous settings for the numerical optimization
# The two warnings prevent that the target filter gammak_generic is shorter than L: this is OK (target is identity shifted forward)
# The second warning informs that the SSA-solution is very close to MSE after optimization
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

ssa_x<-SSA_obj$ssa_x
# Plot optimized filter: it is the MSE predictor
ts.plot(ssa_x)

# This is the filter applied to epsilont (before deconvolution): it follows the exponential decay specified by a1
ssa_eps<-SSA_obj$ssa_eps
ts.plot(ssa_eps)
# The decay pattern replicates a1 
ssa_eps[2:L]/ssa_eps[1:(L-1)]




#--------------------------------------------------------------------
# Example 5
# Exchange roles of xi and gammak_target in the previous examples
# Background: 
# In the above examples we assumed that xt = AR(1) and that zt=xt identity (z_{t+delta} is the target, see , see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# However, we could look at the forecast problem alternatively, by setting xt=epsilont and zt=AR(1)-filter applied to xt
# We now show how to implement the latter design


a1<-0.8
# Since xt=epsilont the Wold-decomposition is an identity: we then set xi<-NULL
xi<-NULL
# Replicate lag-one ACF of one-step ahead (MSE) forecast filter: this is just a1
rho1<-0.8
# Filter length
L<-20
# Target: we now supply the AR(1)-filter (its MA-inversion): in the previous example gamma_k_generic was the identity
gammak_generic<-c(1,ARMAtoMA(ar=a1,lag.max=len-1))
# Forecast horizon: one-step ahead
forecast_horizon<-1
# We retain the previous settings for the numerical optimization
# Note that the function checks if the length of the target matches L: if not, a warning is printed meaning that the target is artificially extended with zeroes in order to match the length L
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)
# This is the same as the following call (omitting xi in the function call assumes default xi=NULL i.e. white noise)
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1)

ssa_x<-SSA_obj$ssa_x
# Plot optimized filter: in contrast to ssa_x in previous exercise (which was a1*identity), we now have the 
#   finite-sample MA-forecast filter 
ts.plot(ssa_x)
# Exponential decay 
ssa_x[2:L]/ssa_x[1:(L-1)]

# This is the filter applied to epsilont 
# It is now identical with ssa_x since xt=epsilont
ssa_eps<-SSA_obj$ssa_eps
# The decay pattern replicates a1
ssa_eps[2:L]/ssa_eps[1:(L-1)]
# In the previous example ssa_eps differed from ssa_x





#--------------------------------------------------------------------
# Example 6
# Unsmoothing: we replicate example 3 but we specify a smaller holding-time constraint; in this case the SSA-filter 
#   must generate additional zero-crossings 

a1<-0.8
# Compute the Wold-decomposition (use true or empirical AR-estimate)
len<-100
# Reminder: we do not supply data to the SSA estimation function; the optimization relies on xi (and on target) only
xi<-c(1,ARMAtoMA(ar=a1,lag.max=len-1))

# In general we want the SSA-filter to be smoother (less zero-crossings)
# However here we impose a smaller holding time
ht<-3
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# Filter length
L<-20
# Target: in our case this is the identity since we want to forecast the original data xt
gammak_generic<-1
# Forecast horizon: one-step ahead
forecast_horizon<-1
# We retain the previous settings for the numerical optimization

SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

ssa_x<-SSA_obj$ssa_x
# Plot optimized filter: the alternating pattern of the filter generates additional 'noisy' crossings
ts.plot(ssa_x)

# Various checks:
# 1. Verify holding-time: compute empirical holding-time based on a very long time series
len<-100000
set.seed(1)
x<-arima.sim(n = len, list(ar = a1))

# Compute filter output
yhat<-filter(x,ssa_x,side=1)
# Compute empirical holding-time
empirical_ht<-compute_empirical_ht_func(yhat)
empirical_ht
# compare with imposed constraint: they match
ht

# 2. Compare lag-one ACF of optimized design with imposed constraint: successful optimization means that both numbers should be close
# In our example both numbers match nearly perfectly
SSA_obj$crit_rhoyy
rho1

# 3. Criterion values: correlations with two different targets, see proposition 4 in JBCY paper
#   a. Target = MSE predictor (idea: approximate the MSE-predictor as well as possible, subject to the holding-time constraint)
#   b. Target = effective target (here: data shifted forward by one forecast step)
# 3.a `True' (expected) correlation with one-step ahead forecast (MSE estimate)
SSA_obj$crit_rhoyz
# Compare with empirical correlation
# Compute one-step ahead (MSE-) predictor
MSE_forecast<-a1*x
# Empirical correlation: matches 'true' correlation above
cor(yhat,MSE_forecast,use='pairwise.complete.obs')

# 3.b Correlation with effective target i.e. series shifted by one time unit
SSA_obj$crit_rhoy_target
# Compare with empirical correlation
cor(yhat,c(x[(2):len],x[len]),use='pairwise.complete.obs')


#-----------------------------------------------------------------------------
# Example 7
# Holding-time: strict interpretation, misspecification, smoothing hyperparameter

# 7.1 Correct model xt=epsilont white noise. If the model is not misspecified, then expected ht = empirical ht
len<-12000
# Apply an equally-weighted MA of length L to the data
L<-10
b<-rep(1/L,L)
set.seed(65)  
x<-rnorm(len)
yhat<-filter(x,b,side=1)
yhat<-na.exclude(yhat)

# Compare expected and empirical holding-times: they match up to sampling error
compute_holding_time_func(b)$ht
compute_empirical_ht_func(yhat)
  
#----------------------------------------------------------------
# 7.2 Misspecified model xt=ARMA: expected ht does not match empirical ht because our expression for ht assumes white noise
L<-10
b<-rep(1/L,L)
a1<-0.4
b1<-0.3
set.seed(65)  
x<-arima.sim(n=len,model=list(ar=a1,ma=b1))
yhat<-filter(x,b,side=1)
yhat<-na.exclude(yhat)

# Compare expected and empirical holding-times: they differ
compute_holding_time_func(b)$ht
compute_empirical_ht_func(yhat)

# In order to reconcile both numbers we have to specify the correct model or, more precisely, the correct filter
# Correct here means: the filter whose input is epsilont (instead of xt) and whose output is the same as b when applied to xt 
# The correct filter is thus the convolution of b with MA-inversion (Wold decomposition) of xt

# Step 1 MA-inversion
xi<-c(1,ARMAtoMA(ar=a1,ma=b1,lag.max=len-1))
# Step 2 Covolution
ssa_eps<-conv_two_filt_func(xi,b)$conv
# Compare ssa_eps and b
ts.plot(ssa_eps[1:30],col="red",main="Original filter (black) vs. convolved filter (red)")
lines(b,col="black")
# Compute expected holding time of convolved filter: once corrected, the filter matches the empirical ht of yhat above
compute_holding_time_func(ssa_eps)$ht

# Filter data
set.seed<-43
x<-eps<-rnorm(len)
for (i in 2:len)
  x[i]<-a1*x[i-1]+eps[i]+b1*eps[i-1]

# Filter x with b and eps with ssa_eps and compare filter outputs
yhat_x<-filter(x,b,side=1)
yhat_eps<-filter(eps,ssa_eps[1:30],side=1)
mplot<-na.exclude(cbind(yhat_x,yhat_eps))
# Both series are identical
ts.plot(mplot[1:1000,],lty=1:2,main="b applied to xt generates the same output as convolved filter applied to epsilont")

#---------------------------------------------
# 7.3 Misspecified model xt=ARMA+mu: we here shift additionally the series by mu!=0
L<-10
b<-rep(1/L,L)
a1<-0.4
b1<-0.3
mu<-1
set.seed(655)  
x<-mu+arima.sim(n=len,model=list(ar=a1,ma=b1))
yhat<-filter(x,b,side=1)
yhat<-na.exclude(yhat)

# Compare expected and empirical holding-times: they differ even more than previously (when mu=0)
compute_holding_time_func(b)$ht
compute_empirical_ht_func(yhat)

# Let's see what happened
ts.plot(x[1:100])
abline(h=0)
abline(h=mu)
# By shifting the series upward by mu, the number of zero crossings decreased (for mu>4 we would not observe zero-crossings anymore)

# In order to reconcile expected and empirical hts we have to specify the correct model or, more precisely, the correct filter
# Correct here means: 
#   -The filter whose input is epsilont (instead of xt) and whose output is the same as b when applied to 
#     xt-mu (we need to center the data in order to reconcile holding times) 
# The correct filter is thus the convolution of b with MA-inversion (Wold decomposition) of xt, shifted upwards by mu

# Step 1 MA-inversion
xi<-c(1,ARMAtoMA(ar=a1,ma=b1,lag.max=len-1))
# Step 2 Covolution
ssa_eps<-conv_two_filt_func(xi,b)$conv
# Compare ssa_eps and b
ts.plot(ssa_eps[1:30],col="red",main="Original filter (black) vs. convolved filter (red)")
lines(b,col="black")
# Compute expected holding time: it does not yet match the empirical ht of yhat above because xt has not been centered
compute_holding_time_func(ssa_eps)$ht

# Filter data
set.seed<-43
y<-eps<-rnorm(len)
for (i in 2:len)
  y[i]<-a1*y[i-1]+eps[i]+b1*eps[i-1]
x<-y+mu
# Step 2: center data
# Filter x-mean(x) (centered data) with b; filter eps with ssa_eps and compare filter outputs
yhat_x<-filter(x-mean(x),b,side=1)
yhat_eps<-filter(eps,ssa_eps[1:30],side=1)
mplot<-na.exclude(cbind(yhat_x,yhat_eps))
# Both series are virtually identical: up to finite sample estimation error of mu by mean(x)
ts.plot(mplot[1:1000,],lty=1:2)

# Now the empirical ht matches the expected number (up to sampling error)
compute_empirical_ht_func(yhat_x)
compute_holding_time_func(ssa_eps)$ht

#--------------------------------------------
# Conclusions
# -The expected ht of a filter b assumes that b is applied to white noise: no autocorrelation, zero mean
# -If the series xt is autocorrelated, then one needs to correct the filter b for the dependence structure
#   -Compute the convolution of b with the MA-inversion xi of xt
# -If the series has a non-vanishing mean mu, then ht addresses crossings at the mu-line (not at the zero-line anymore) 
# -If the model is misspecified, then ht cannot be interpreted literally as `mean duration between consecutive zero-crossings'
# -But increasing ht in the holding-time constraint of SSA will always lead to a smoother filter (irrespective of misspecification)
#   -The smoothing effect (at the mu-line) operates also on any other level-line
#   -But ht is biased
# -In this sense, ht can be interpreted as a hyperparameter which triggers smoothness and controls noise suppression,
#     even if the model is misspecified (not white noise)
# -Tutorials 2,3 and 4 illustrate these effects

#-----------------------------------------------------------------
# Example 8
# We here replicate the SSA-filters in the business-cycle analysis, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# SSA-filters assume white noise i.e. xi=NULL: we do not `mine' the data by fitting a model

# HP and hyperparameter
L<-201
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)

# Two sided filter
target=HP_obj$target
# One-sided gap filter: observation minus HP-trend
hp_gap=HP_obj$hp_gap
# One sided gap filter when applied to first differences of the data
#  modified_hp_gap applied to first differences replicates the output of the original gap applied to the original data  
modified_hp_gap=HP_obj$modified_hp_gap
# Classic one-sided (concurrent) HP-trend: this is an optimal (MSE) estimate of the two-sided filter if the data follows an ARIMA(0,2,2) specification 
hp_trend=HP_obj$hp_trend
# Alternative one-sided HP-trend based on truncating the two-sided filter
#   It is an optimal (MSE) estimate of the two-sided filter if the data is white noise (which is consistent with xi=NULL)
hp_mse=HP_obj$hp_mse
# We can use any of the above HP-designs as targets for SSA, see tutorials 2 and 5
#---------------------------
# SSA and hyperparameters
# The classic one-sided HP-trend hp_trend is subject to some undesirable noise-leakage: the filter generates 
#   unwanted noisy crossings
# Its holding time is:
compute_holding_time_func(hp_trend)$ht
# We therefore ask SSA to target hp_trend while simultaneously improving noise-suppression (less noisy alarms)  
# For this purpose we impose a larger holding-time, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
ht<-12
# The resulting SSA filter will generate roughly 40% less (noisy) crossings:
ht/compute_holding_time_func(hp_trend)$ht
rho1<-compute_rho_from_ht(ht)
# We compute a nowcast of the classic concurrent HP as well as a 18-months ahead forecast, both subject to ht
# We can supply a vector with the desired forecast horizons: SSA will compute optimal filters for each forecast horizon
forecast_horizon_vec<-c(0,18)
# We here want SSA to approximate the one-sided MSE HP-trend
gammak_generic<-hp_mse

# By omitting xi in the call we assume the data to be white noise
SSA_obj<-SSA_func(L,forecast_horizon_vec,gammak_generic,rho1)

# In this case ssa_x and ssa_eps are the same filters
ssa_eps<-SSA_obj$ssa_eps
colnames(ssa_eps)<-paste("SSA(",round(ht,2),",",forecast_horizon_vec,")",sep="")


# We also compute a concurrent SSA filter which replicates ht of the classic one-sided HP and which forecasts 
#  it 18-months ahead
ht_short<-compute_holding_time_func(hp_trend)$ht
rho1_short<-compute_rho_from_ht(ht_short)
forecast_short<-forecast_horizon_vec[2]

# Compute fast SSA forecast filter with same holding time as HP-trend
SSA_obj<-SSA_func(L,forecast_short,gammak_generic,rho1_short)

ssa_eps1<-SSA_obj$ssa_eps
colnames(ssa_eps1)<-paste("SSA(",round(ht_short,2),",",forecast_horizon_vec[2],")",sep="")

ssa_eps<-cbind(ssa_eps,ssa_eps1)

#--------------------------------------
# Plot filters: this plot replicates fig.4 in JBCY paper
colo_hp_all<-c("brown","red")
colo_SSA<-c("orange","blue","violet")
colo_all<-c(colo_hp_all,colo_SSA)


par(mfrow=c(1,2))
mplot<-scale(cbind(hp_trend,target,hp_gap,modified_hp_gap),center=F,scale=T)
colnames(mplot)<-c("HP trend","Target symmetric","HP-gap (original)","HP-gap (modified)")
colo<-c(colo_hp_all[1],"black","darkgreen",colo_hp_all[2])
plot(mplot[,1],main="",axes=F,type="l",xlab="Lag-structure",ylab="filter-coefficients",ylim=c(min(mplot),max(mplot)),col=colo[1])
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()
# Select forecast horizons 0 and 18
select_vec<-1:3
mplot<-scale(ssa_eps[,select_vec],center=F,scale=T)
plot(mplot[,1],main="",axes=F,type="l",xlab="",ylab="",ylim=c(min(mplot),max(mplot)),col=colo_SSA[1])
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo_SSA[i])
  mtext(colnames(mplot)[i],col=colo_SSA[i],line=-i)
}  
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# We employ these filters in tutorial 5, when applied to the monthly INDPRO series
# The above plot replicates results in Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5





