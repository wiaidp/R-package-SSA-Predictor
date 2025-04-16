# This tutorial is not related to SSA. It addresses the topic of business-cycle analysis (BCA-), i.e., the particular 
# design on which SSA is plugged

# This tutorial is about a particular target, as proposed by Hodrick and Prescott (HP)

# In our applications of SSA we are mainly interested in `prediction'
# Therefore we emphasize one-sided filters (also called, `causal' or `concurrent' or `real-time' filters)
# Two-sided designs will be considered, too, because the one-sided/causal filters are derived from the former

# We address the following points:

# 1.Background: where it starts 
#     Whittaker-Henderson smoothing and signal extraction 

# 2.One- and two-sided HP-trend filters

# 3.One- and two-sided HP-gap filters (typically used in BCA applications)

# 4.Frequency-domain analysis: amplitude and time-shifts 

# 5.Spectral density of the extracted cycle: HP-trend applied to differences

# 6.Spectral density of the extracted cycle: original HP-gap applied to levels

# 7.Comparison of original HP-gap (typical BCA-application) and HP-trend applied to differences
#   -The latter design has some important advantages (and some minor drawbacks) when compared to the original HP-gap
#   -We then plug SSA on HP-trend in tutorial 2.1

# 8. Summary


#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions
rm(list=ls())

library(xts)
# Load the library mFilter
# Standard R-package for HP and other filters 
library(mFilter)
# McElroys package for HP
source(paste(getwd(),"/R/hpFilt.r",sep=""))
# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))


# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

# Load data from FRED with library quantmod
library(quantmod)
# Download Non-farm payroll and INDPRO
getSymbols('PAYEMS',src='FRED')
getSymbols('INDPRO',src='FRED')

# We now develop points 1-8 listed above 
#########################################################################################################
#########################################################################################################
# 1. Background
# 1.1 Derivation of HP based on Whittaker-Henderson smoothing
# For given lambda, find yt such that

# sum_{t=1}^T (xt-yt)^2+lambda \sum_{t=d+1}^T((1-B)^d yt)^2 \to minimum

# In words: find yt such that yt is close to (the data) xt while being smooth
# For d=2 the HP-trend filter solves the above optimization problem

# We use the R-package mFilter for computing HP 
#   Specify filter length: should be an odd number since otherwise the two-sided HP filter could not be adequately centered 
L<-201
# Should be an odd number
if (L/2==as.integer(L/2))
{
  print("Filter length should be an odd number")
  print("If L is even then HP cannot be adequately centered")
  L<-L+1
}  
# Specify lambda: monthly design
lambda_monthly<-14400
par(mfrow=c(1,1))
# This function relies on mFilter and it derives additional HP-designs to be discussed further down
HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)
# Bi-infinite two-sided (symmetric) HP
hp_target<-HP_obj$target
# This is a finite version of the symmetric HP-trend filter
#   It is a causal (one-sided) filter
#   In applications, the filter is centered at t: it is acausal (expands equally in the past and in the future) 
ts.plot(hp_target,main=paste("Two-sided HP trend, lambda=",lambda_monthly,sep=""))
# Filter coefficients add to 1: it is a lowpass (see amplitude functions further down)
sum(hp_target)

#------------------------------------
# 1.2 Alternative derivation of HP
#   -HP can be interpreted as an optimal MSE-signal extraction filter for the trend in the smooth trend model, see Harvey (1989).
# Let us reproduce the implicit or latent model here: 
#   -It is an ARIMA(0,2,0)-trend T_t plus a scaled white noise: x_t=T_t+sqrt(lambda_monthly)*I_t where T_t=ARIMA(0,2,0) and I_t=noise
#   -Note that the `cycle', i.e., the difference x_t-T_t is white noise I_t (one of several flaws)
set.seed(23)
len<-12000
# Let's simulate such a series with the above lamba_monthly=14400 parameter
x<-cumsum(cumsum(rnorm(len)))+rnorm(len)*sqrt(lambda_monthly)
ts.plot(x)
# First differences: slowly drifting away
ts.plot(diff(x))
# Look at acfs: after lag-one, the acf indicates a weak but permanent positive pattern (differences are non-stationary)
# The negative lag-one acf is due to the differenced noise component
acf(diff(x))
# After second order differences the acf at higher lags vanishes: the first two lags correspond to double differences of the noise component
acf(diff(diff(x)))
# Let's fit a MA(2) model as suggested by the acf
arima(x,order=c(0,2,2))
# The implicit data-generating process follows an ARIMA(0,2,2)-specification whose MA-coefficients are determined by 
# lambda, see  McElroy (2006).
# We can use McElroy's package to compute the MA-parameters exactly for lambda_monthly
q<-1/lambda_monthly
hp_filt_obj<-hpFilt(q,L)
# HP-filter (left tail only)
hp_filt_obj$filter_coef
# MA-parameters of ARIMA(0,2,2)-model
# The first parameter is a normalizing constant, the last two are the MA-parameters
hp_filt_obj$ma_model
ma<-hp_filt_obj$ma_model[2:3]
ma
# Compare with the estimated MA parameters: estimates are within the 95% interval
arima(x,order=c(0,2,2))

#-------------------------------
# 1.3 We can now simulate time series corresponding to this implicit model
# If economic series look similar then we can conclude that HP is an optimal filter for extracting the trend out of them
#   -But this would imply that the cycle is just noise
set.seed(1)
# Monthly US industrial production index (INDPRO) (https://fred.stlouisfed.org/series/INDPRO) starts in 1920
# Roughly 100*12=1200 observations
# Assume a similar length for our simulation, initializing the series 100 years back with zero (initialization is necessary because an integrated process has infinite memory: initialization has a permanent effect)
len<-1200
x<-cumsum(cumsum(rnorm(len)))+rnorm(len)*sqrt(lambda_monthly)

# Check model: parameters are fine
arima(x,order=c(0,2,2))
# Diagnostics are fine, too
tsdiag(arima(x,order=c(0,2,2)))
# Have a look at the data
ts.plot(x,main="Single realization of ARIMA(0,2,2)")
# Does not look like a typical economic time series...
# Let's generate a couple of them for comparison
anzsim<-5
mat_sim<-NULL
set.seed(96)
for (i in 1:anzsim)
{
  x<-cumsum(cumsum(rnorm(len)))+rnorm(len)*sqrt(lambda_monthly)
  
  # Compute ARIMA(0,2,2)
  mat_sim<-cbind(mat_sim,x)
}
ts.plot(mat_sim,col=rainbow(anzsim),main="Realizations of ARIMA(0,2,2)")

# The series are `flexing' and they are a bit `noisy': the `cycle' is just noise...
# Have a look at first differences
ts.plot(apply(mat_sim,2,diff),col=rainbow(anzsim),main="First differences of ARIMA(0,2,2)")
abline(h=0)
# The series are slowly drifting away (non-stationary), as expected
# First differences are noisy: the differenced cycle (differenced white noise) dominates the dynamics 

# Integrated processes have an infinite memory: therefore they never forget initialization
#   -The double difference means that x_1=2*x_{0}-x_{-1}
#   -We initialized the difference equation with x_0=x_{-1}=0 for the above realization
# What happens if we select x_{0}=1000 and x_{-1}=0?
set.seed(1)
x<-cumsum(1000+cumsum(rnorm(len)))+rnorm(len)*sqrt(lambda_monthly)
# Plot
ts.plot(x,main="Single realization with different initialization")
# Looks like a straight line
# And first differences?
ts.plot(diff(x),main="First differences")
# They are just shifted upwards by the amount of our initialization (but they are slowly drifting away from that level)


# We now compare the simulated data (with 0-initialization) with two important monthly macro-indicators
# Original (un-transformed) data
par(mfrow=c(2,2))
ts.plot(mat_sim,xlab="",main="Artificial/simulated",col=rainbow(anzsim))
plot(PAYEMS,main="Non-farm payroll")
plot(INDPRO,main="Indpro")
# The artificial data has less `structure' (no recession dips), the series appear noisier, growth-sign is random and 
#     the drift can become arbitrarily large in absolute value over time 
# We recommend a log-transform in order to stabilize the variances of the macro-indicators 
#   (the variance is changing as levels increase whereas the noise variance in the simulated data is level-independent)
par(mfrow=c(2,2))
ts.plot(mat_sim,xlab="",col=rainbow(anzsim),main="Artificial/simulated")
plot(log(PAYEMS),main="Log Non-farm payroll")
plot(log(INDPRO),main="Log Indpro")
# Comparison: see comments above. The growth of the log-transformed series is much more regular than the simulated data
#   We could account for this discrepancy by changing initial values but this would affect smoothness: whatever we do, simulated data won't match real data 
# We now compare series in first differences
par(mfrow=c(2,2))
ts.plot(apply(mat_sim,2,diff),xlab="",col=rainbow(anzsim),main="Artificial/simulated")
abline(h=0)
plot(diff(log(PAYEMS)),main="Returns Non-farm payroll")
plot(diff(log(INDPRO)),main="Returns Indpro")
# Differences of simulated series are much noisier and more `regular' (less non-stationary)  

# We now remove the pandemic, start in 1990 and use ts.plot (same graphic), shorten simulated sample accordingly
par(mfrow=c(2,2))
ts.plot(apply(mat_sim,2,diff)[(len-29*12):(len-1),],xlab="",col=rainbow(anzsim),main="Artificial/simulated")
abline(h=0)
ts.plot(diff(log(PAYEMS["1990/2019"])),main="Non-farm payroll",xlab="",ylab="")
abline(h=0)
ts.plot(diff(log(INDPRO)["1990/2019"]),main="Indpro",xlab="",ylab="")
abline(h=0)
# The simulated data has no structure and it looks much noisier 

# Consider zero crossings of first differences
compute_empirical_ht_func(diff(mat_sim[(len-29*12):(len-1),1]))
compute_empirical_ht_func(as.double(diff(PAYEMS["1990/2019"])))
compute_empirical_ht_func(as.double(diff(INDPRO["1990/2019"])))
# As expected, the artificial data is subject to more crossings 
#   White noise has a holding-time of 2: both macro-indicators are above 2 (they are positively autocorrelated) and the simulated data is below 2 (negatively autocorrelated) 
# Also, non-farm payroll seems to be much smoother: we will need to fit models to the data in order to capture these differences (see tutorials 2.1, 3 and 4)



####################################################################################################################
####################################################################################################################
# 2. HP-trend: two-sided and one-sided designs 
# Recall, from the above, that HP aims at estimating the trend from the trend+noise model.
# The above results suggest that 
#   a. The artificial data has less (no) `structure', when compared with macro-data (recessions, expansions)
#   b. The artificial data is noisier
# Therefore, if we assume the artificial data to be the truth, then it would make sense to apply a filter, i.e. HP-trend, which damps effectively (strongly) the noise
#   a. The filter does not have to `care' about additional structure (like recession dips)
#   b. Its only purpose is to erase the noise
# When applied to Macro-series, strong smoothing can attenuate or under-estimate the relevant structure, by washing-out 
#   critical (more or less sharp) recession dips, see Phillips and Jin (2021)

# We briefly compute the holding-time of the two-sided HP-trend: this is the mean duration between consecutive zero-crossings when the filter is applied to white noise
compute_holding_time_func(hp_target)$ht
# Big number! The filter has a very strong smoothing effect.  
# The above (finite-length) two-sided filter cannot be applied towards the sample end
# For this purpose one can use the optimal one-sided HP-trend
hp_trend<-HP_obj$hp_trend
par(mfrow=c(1,1))
ts.plot(hp_trend,main=paste("Optimal one-side HP-trend assuming an ARIMA(0,2,2) DGP with lambda=",lambda_monthly,sep=""))
# Filter coefficients add to 1: it is a lowpass (see amplitude functions further down)
sum(hp_trend)
# Holding-time
compute_holding_time_func(hp_trend)$ht
# The holding-time is considerably shorter: the causal/real-time/concurrent filter is more permeable (noise leakage) 

# Let's now apply the filter to simulated data
set.seed(18)
len<-1200
x<-cumsum(cumsum(rnorm(len)))+rnorm(len)*sqrt(lambda_monthly)

# Compute filter output of SSA-HP filter
y_hp_concurrent<-filter(x,hp_trend,side=1)
y_hp_symmetric<-filter(x,hp_target,side=2)

ts.plot(y_hp_concurrent,main="HP: two-sided vs one-sided filter",col="blue")
lines(y_hp_symmetric)
abline(h=0)
mtext("Two-sided HP",col="black",line=-1)
mtext("One-sided HP",col="blue",line=-2)
# Both lines seem to overlap (resolution is weak): the one-sided HP extends to the sample end

# Look at the filter approximation error 
error<-y_hp_symmetric-y_hp_concurrent
ts.plot(error,main="Filter approximation error")
# The error looks stationary or `cyclical': by optimality of the one-sided filter the error is smallest possible and both trend series are cointegrated
# MSE 
MSE_error<-mean((error)^2,na.rm=T)
MSE_error
# Any other one-sided filter has a larger MSE when the DGP is the above ARIMA(0,2,2), specified by lambda_monthly

# We could also look at the gap: the difference between data and trend (Identity-HP_trend)
gap<-x-y_hp_symmetric
ts.plot(gap,main="Gap")
# MSE: pretty close to our choice for lambda_monthly... It will converge for longer samples.
mean((gap)^2,na.rm=T)
# Ideally, this error should be white noise
acf(na.exclude(gap),main="ACF of gap")

####################################################################################################################
####################################################################################################################
# 3. HP-gap 
# In BCA applications, typically, the filter for extracting the cycle is HP-gap: the identity minus HP-trend 
#   -In principle, this should be (close to) white noise since x_t=T_t+sqrt(lambda_monthly)*I_t, i.e., x_t-T_t is noise  
#   -Positive/negative gap: the current data point lies above/below trend. Additional stabilization (anti-cyclical policy) would be unnecessary if the gap happened to be white noise

hp_gap_sym<-c(rep(0,(L-1)/2),1,rep(0,(L-1)/2))-hp_target
ts.plot(hp_gap_sym,main="HP-gap two-sided")

# We can apply this filter to the simulated data
y_hp_gap_symmetric<-filter(x,hp_gap_sym,side=2)
# Looks noisy:
ts.plot(y_hp_gap_symmetric,main="Output of two-sided HP-gap (should be close to white noise)")
# Acf suggest noise
acf(na.exclude(y_hp_gap_symmetric),main="Acf two-sided HP-gap output")

# HP-gap has cancelled the unit-roots of the ARIMA(0,2,2): the filter output is stationary (nearly white noise)
# The one-sided filter does similarly
hp_gap<-c(1,rep(0,L-1))-hp_trend
ts.plot(hp_gap,main="HP-gap one-sided")

# We can apply this filter to the simulated data, too
y_hp_gap_concurrent<-filter(x,hp_gap,side=1)
# Looks noisy:
ts.plot(y_hp_gap_concurrent,main="Output of one-sided HP-gap (should be close to white noise)")
# Acf suggest noise
acf(na.exclude(y_hp_gap_concurrent),main="Acf one-sided HP-gap output")

# MSE: this is of course the same as MSE_error above (the identity in both gap-filters cancels) 
mean((y_hp_gap_concurrent-y_hp_gap_symmetric)^2,na.rm=T)
# Any other one-sided filter has a larger MSE when the DGP is the above ARIMA(0,2,2), specified by lambda_monthly

#####################################################################################################
######################################################################################################
# 4. Frequency-domain analysis: Amplitude functions of filters
#   We now compute amplitude and time-shift functions of the above filters

# Specify the number of equidistant frequency ordinates in [0,pi]
K<-600
# Compute transfer, amplitude and shift functions (shift=phase divided by frequency)
amp_obj_hp_trend_concurrent<-amp_shift_func(K,hp_trend,F)
amp_obj_hp_trend_sym<-amp_shift_func(K,hp_target,F)
amp_obj_hp_gap_sym<-amp_shift_func(K,hp_gap_sym,F)
amp_obj_hp_gap_concurrent<-amp_shift_func(K,hp_gap,F)

# Plot amplitude functions
par(mfrow=c(1,1))
mplot<-cbind(amp_obj_hp_trend_concurrent$amp,amp_obj_hp_trend_sym$amp,amp_obj_hp_gap_sym$amp,amp_obj_hp_gap_concurrent$amp)
colnames(mplot)<-c("HP-trend one-sided","hp-trend two-sided","Hp-gap two-sided","HP-gap one-sided")
colo<-c("blue",rainbow(ncol(mplot)))
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
abline(v=which(mplot[,1]==max(mplot[,1])),col=colo[1])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()

# Outcome
# -Both HP-trend filters are lowpass, HP-gaps are highpass 
# -The two-sided HP-trend has a very steep and fast decaying amplitude: 
#   -it eliminates high-frequency components very strongly
#   -strong smoothness
#   -large holding-time
# -The one-sided HP-trend has more noise leakage 
#   -but it still damps high-frequency components effectively
#   -It's peak amplitude (marked by a vertical line) is obtained at a periodicity of 85 months or nearly 7 years:
2*(K-1)/(which(mplot[,1]==max(mplot[,1]))-1)
# This matches roughly the length of business-cycles!


# Time shift functions
mplot<-cbind(amp_obj_hp_trend_concurrent$shift,amp_obj_hp_trend_sym$shift,amp_obj_hp_gap_sym$shift,amp_obj_hp_gap_concurrent$shift)
colnames(mplot)<-c("HP-trend one-sided","hp-trend two-sided","Hp-gap two-sided","HP-gap one-sided")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()
# Note that all filters are causal: the lags (shifts) of the two-sided designs correspond to half their length
#   The effective two-sided acausal filters, used in applications, are not shifted: their phase vanishes
# Also, the one-sided gap has a negative shift: it is anticipative (but the filter has no noise suppression)
# Let's then focus on the one-sided HP-trend only
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot[,1]),max(mplot[,1])),col=colo[1])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()

# We note that the time-shift (phase divided by frequency) of the one-sided HP-trend vanishes at frequency zero
#   -Quite unusual for a lowpass design: a small shift means a timely detection of peaks and troughs of a time series
#   -The shift must vanish at frequency zero because otherwise the filter error (between two-sided and one-sided trends) would not be stationary anymore, at least for an I(2)-process

###################################################################################################################
#################################################################################################################
# 5. Spectral densities HP-trend
# -We want to compare the spectral densities of the extracted cycles
# -For that purpose we compute cycles based on two- and one-sided HP-trend filters, as applied to differences of the data
# -This topic requires 
#   5.1. modelling of the DGP
#   5.2. derivation of the spectral density


# 5.1 Let us assume, that typical (eventually log-transformed) economic time series are compatible with an I(1)-DGP
# We try to motivate this I(1)-assumption
# a. Let's look at returns:
par(mfrow=c(2,2))
plot(diff(log(INDPRO)["1960/2019"]),xlab="",ylab="",main="Returns Indpro")
plot(diff(log(PAYEMS)["1960/2019"]),xlab="",ylab="",main="Returns non-farm payroll")
acf(na.exclude(diff(log(INDPRO)["1960/2019"])),main="ACF Indpro")                
acf(na.exclude(diff(log(PAYEMS)["1960/2019"])),main="ACF non-farm payroll")                
# Returns do not seem to be subject to pronounced changes of their levels (but data prior 1990 looks different: great moderation)

# b. Also, simple stationary ARMA-models seem to fit the data 
model_indpro<-arima(diff(log(INDPRO)["1960/2019"]),order=c(2,0,1))
model_payems<-arima(diff(log(PAYEMS)["1960/2019"]),order=c(2,0,1))
tsdiag(model_indpro)
tsdiag(model_payems)

# c. Finally, economic theory suggests that a broad range of economic series, such as  stock prices, futures prices, 
#   long-term interest rates, oil prices, consumption spending, inflation, tax rates, or money supply growth rates  
#   should follow (near) martingales, see Fama (1965), Samuelson (1965), Sargent (1976), Hamilton (2009), Hall (1978) 
#   and Mankiw (1987)

# Let us therefore assume that the data is I(1), in contradiction to the ARIMA(0,2,2)-hypothesis of the implicit DGP justifying optimality of HP
# In principle, we could apply HP-trend to returns: the above amplitude functions suggest that 
#   a. the one-sided filter would not overtly conflict with an extraction of the (differenced) business-cycle from the data (its peak amplitude corresponds to a periodicity of seven years)
#   b. the two-sided filter would damp cycle-frequencies too heavily (over-smoothing), see also Phillips and Jin (2021)

# Let us now compute the spectral density of the components (`cycles`) extracted by both filters when applied to differenced macro-data.
# For this purpose we need to compute the convolution of HP and xi, the Wold-decomposition of the differenced data
# We can compute the weights of the Wold-decomposition of the above models
ar<-model_indpro$coef[1:2]
ma<-model_indpro$coef[3]
xi_indpro<-c(1,ARMAtoMA(lag.max=L-1,ar=ar,ma=ma))
ar<-model_payems$coef[1:2]
ma<-model_payems$coef[3]
xi_payems<-c(1,ARMAtoMA(lag.max=L-1,ar=ar,ma=ma))
par(mfrow=c(1,1))
ts.plot(cbind(xi_indpro,xi_payems),main="MA-inversions (Wold decomposition) for Macro-Indicators",col=c("violet","orange"))
mtext("Wold-decomposition Indpro",col="violet",line=-1)
mtext("Wold-decomposition PAYEMS",col="orange",line=-2)

# We can now compute the convolution of HP and Wold-decompositions
hp_one_sided_conv_indpro<-conv_two_filt_func(xi_indpro,hp_trend)$conv
hp_two_sided_conv_indpro<-conv_two_filt_func(xi_indpro,hp_target)$conv
hp_one_sided_conv_payems<-conv_two_filt_func(xi_payems,hp_trend)$conv
hp_two_sided_conv_payems<-conv_two_filt_func(xi_payems,hp_target)$conv


# Interpretation: hp_one_sided_conv applied to model-residuals generates the same output as hp_trend applied to returns
# We briefly check this assertion for Indpro (check for PAYEMS is similar)
#   Note that we must shift by the mean ("intercept") since residuals are centered at zero
y_hp_conv<-filter(model_indpro$residuals,hp_one_sided_conv_indpro,side=1)+model_indpro$coef["intercept"]
y_hp<-filter(diff(log(INDPRO)["1960/2019"]),hp_trend,side=1)
# Both series overlap, as expected.
ts.plot(cbind(y_hp_conv,y_hp),main="Convolution applied to residuals vs. hp applied to returns: both series overlap")

#-------------------
# 5.2 Spectral densities
# The spectral density of the `cycle` extracted by hp_trend corresponds to the squared amplitude of hp_one_sided_conv
#   -Recall that hp_one_sided_conv is applied to model residuals and that the spectral density of white noise (residuals) is flat
#   -We ignore the scaling by sigma^2/(2*pi) corresponding to the residuals' spectral density
#   -We scale the spectral densities in order to simplify comparisons 
amp_obj_one_sided_hp_conv_indpro<-amp_shift_func(K,hp_one_sided_conv_indpro,F)
amp_obj_one_sided_hp_conv_payems<-amp_shift_func(K,hp_one_sided_conv_payems,F)
par(mfrow=c(1,1))
# Let's scale the spectral densities for easier comparison
#   -We also add two vertical bars corresponding to business-cycle frequencies: 2-10 years periodicities
plot(scale(amp_obj_one_sided_hp_conv_indpro$amp^2,center=F,scale=T),type="l",col="violet",axes=F,xlab="Frequency",ylab="",main=paste("Scaled spectral density of cycle extracted by one-sided HP trend",sep=""))
lines(scale(amp_obj_one_sided_hp_conv_payems$amp^2,center=F,scale=T),col="orange")
abline(v=K/12)
abline(v=K/60)
mtext("Scaled spectral density of one-sided HP-trend applied to differences of Indpro",col="violet",line=-1)
mtext("Scaled spectral density of one-sided HP-trend applied to differences of Payems",col="orange",line=-2)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# Outcome:
# We see that the spectral density, trivially, depends on the DGP (this evidence is frequently lacking, omitted or neglected in applications)
#   -The Indpro-cycle spectral density is not overtly conflicting with a BCA-perspective 
#   -The PAYEMS-cycle is likely a bit `too smooth' (the peak is left-shifted towards lower frequencies)
# Note: we here apply filters to differenced data. Accordingly, the above filters are supposed to `extract' the differenced cycle
#   -We don't want the spectral density of the differenced cycle to be flat in the business-cycle band: in this sense the trend-filters seem to perform well.
#   -But maybe we'd like the density to vanish towards frequency zero? That depends... (see below)

# We now look at the spectral densities of two-sided trend filters
amp_obj_two_sided_hp_conv_indpro<-amp_shift_func(K,hp_two_sided_conv_indpro,F)
amp_obj_two_sided_hp_conv_payems<-amp_shift_func(K,hp_two_sided_conv_payems,F)
par(mfrow=c(1,1))
# Let's scale the spectral densities for easier comparison
#   -We also add two vertical bars corresponding to business-cycle frequencies: 2-10 years periodicities
plot(scale(amp_obj_two_sided_hp_conv_indpro$amp^2,center=F,scale=T),type="l",col="violet",axes=F,xlab="Frequency",ylab="",main=paste("Scaled spectral density of cycle extracted by two-sided HP trend",sep=""))
lines(scale(amp_obj_two_sided_hp_conv_payems$amp^2,center=F,scale=T),col="orange")
abline(v=K/12)
abline(v=K/60)
mtext("Scaled spectral density of two-sided HP-trend applied to differences of Indpro",col="violet",line=-1)
mtext("Scaled spectral density of two-sided HP-trend applied to differences of Payems",col="orange",line=-2)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# The spectral density are left-shifted: the two-sided filters smooth out business-cycle frequencies, see Phillips and Jin (2021)

##################################################################################################
################################################################################################
# 6. Spectral densities of original HP-gap
# -We want to compare the spectral densities of the extracted cycles
# -For that purpose we compute cycles based on two- and one-sided gaps, applied to data in levels (HP-trend was applied to differences)
# -This topic requires 
#   6.1. modelling of the DGP (see 5.1 above)
#   6.2  stationarity (gaps are applied to non-stationary data in levels)
#   6.3. derivation of the spectral density

# 6.1: see 5.1 above

# 6.2 Transformation: from levels to differences
#   -The gap is applied to levels (not differences)
#   -We here transform the gap-filter such that one can work with differenced data
#  For that purpose we can rely on conv_with_unitroot_func, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
hp_gap_diff<-conv_with_unitroot_func(hp_gap)$conv

# Compare both gap filters
ts.plot(cbind(hp_gap,hp_gap_diff),col=c("red","darkred"),main=paste("HP-gap: original and transformed filters, lambda=",lambda_monthly,sep=""))
mtext("Original HP-gap as applied to levels",col="red",line=-1)  
mtext("Transformed HP-gap as applied to differences",col="darkred",line=-2)  

# After scaling, we can note a degree of familiarity between the transformed gap and the trend-filter (both are applied to differences)
ts.plot(scale(cbind(hp_trend,hp_gap_diff),scale=T,center=F),col=c("blue","darkred"),main=paste("Transformed HP-gap vs. HP-trend, lambda=",lambda_monthly,sep=""))
mtext("HP-trend as applied to differences",col="blue",line=-1)  
mtext("Transformed HP-gap as applied to differences",col="darkred",line=-2)  

# We check that the output of hp_gap_diff as applied to differences replicates the original gap when applied to levels 
y_hp_gap<-filter((log(INDPRO)["1960/2019"]),hp_gap,side=1)
y_hp_gap_diff<-filter(diff(log(INDPRO)["1960/2019"]),hp_gap_diff,side=1)
# Both series overlap
#   -Therefore the corresponding spectral densities must overlap, too
ts.plot(cbind(y_hp_gap,y_hp_gap_diff),main="Original HP-gap applied to levels vs. modified gap applied to returns: both series overlap")

# We are now closer to our goal: computing the spectral density of the original HP-gap cycle
# We just have to convolve hp_gap_diff with the Wold-decomposition of the series
hp_gap_diff_one_sided_conv_indpro<-conv_two_filt_func(xi_indpro,hp_gap_diff)$conv
hp_gap_diff_one_sided_conv_payems<-conv_two_filt_func(xi_payems,hp_gap_diff)$conv

# The spectral densities are obtained by squaring the amplitude functions of the convolved transformed filters (up to an arbitrary scaling)
amp_obj_hp_gap_diff_conv_indpro<-amp_shift_func(K,hp_gap_diff_one_sided_conv_indpro,F)
amp_obj_hp_gap_diff_conv_payems<-amp_shift_func(K,hp_gap_diff_one_sided_conv_payems,F)
par(mfrow=c(1,1))
plot(scale(amp_obj_hp_gap_diff_conv_indpro$amp^2,center=F,scale=T),type="l",axes=F,xlab="Frequency",col="darkred",ylab="",main=paste("Spectral density of cycle extracted by one-sided HP filters",sep=""))
lines(scale(amp_obj_hp_gap_diff_conv_payems$amp^2,center=F,scale=T),col="brown")
mtext("Spectral density original HP-gap Indpro",line=-1,col="darkred")
mtext("Spectral density original HP-gap Payems",line=-2,col="brown")
abline(v=K/12)
abline(v=K/60)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# HP-gap applied to Payems generates a smoother cycle 
#   -The corresponding Wold-decomposition un-whitens innovations in the Wold-decomposition more effectively
#   -The ARMA-model of Payems is a more effective lowpass than the ARMA-model of Indpro
#   -In simpler terms: PAYEMS looks (and is) smoother than Indpro
#   -Therefore, applying the same filter to different series will generate a smoother cycle for the smoother series

# Let us compare both cycles:
y_hp_gap_payems<-filter((log(PAYEMS)["1960/2019"]),hp_gap,side=1)
mplot<-scale(cbind(y_hp_gap,y_hp_gap_payems),center=F,scale=T)
plot(mplot[,1],main="Cycles of HP-gap applied to Indpro and PAYEMS",col="darkred",axes=F,xlab="",ylab="",type="l")
lines(mplot[,2],col="green")
mtext("HP-gap applied to Indpro",line=-1,col="darkred")
mtext("HP-gap applied to PAYEMS",line=-2,col="green")
abline(h=0)
axis(1,at=12*(1:(nrow(mplot)/12)),labels=index(diff(log(INDPRO)["1960/2019"]))[12*(1:(nrow(mplot)/12))])
axis(2)
box()
# Differences: 
#   -the cycle based on non-farm payroll is substantially smoother, 
#   -it transforms the twin-recessions in the early 80s into a single `double-dip' recession, 
#   -it is systematically lagging behind the cycle extracted from Indpro 
#   -it behaves differently after the financial crisis (more or less monotonous decay after the recession-rebound)
# Applying the same HP-filter (based on lambda=14400) to qualitatively different series is not recommended (sadly, this is not our topic of interest)

#################################################################################################
#################################################################################################
# 7. Comparison: HP-trend applied to differences vs. original HP-gap (applied to levels)
# 7.1 Compare spectral densities 
# We now compare the spectral densities of HP-trend applied to differences with the original HP-gap applied to levels
# We do this for Indpro (a similar outcome would be obtained for Payems)
par(mfrow=c(1,1))
plot(scale(amp_obj_hp_gap_diff_conv_indpro$amp^2,center=F,scale=T),type="l",axes=F,xlab="Frequency",col="darkred",ylab="",main=paste("Spectral density of cycles extracted by one-sided HP-gap and HP-trend, lambda=",lambda_monthly,sep=""))
lines(scale(amp_obj_one_sided_hp_conv_indpro$amp^2,center=F,scale=T),col="violet")
mtext("Spectral density original HP-gap applied to levels of Indpro",line=-1,col="darkred")
mtext("Spectral density HP-trend applied to differences of Indpro",line=-2,col="violet")
abline(v=K/12)
abline(v=K/60)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# Quite similar...
# -The main difference: the density of HP-trend does not vanish at frequency zero
# -The cycle extracted by HP-trend is also a bit smoother (emphasizes more heavily lower frequencies): 
#   -This particular effect seems comparable to the difference between gap-filters applied to Payems and Indpro in a previous plot above

# Which `cycle' is `better'? The answer depends on the purpose of the analysis
# -The cycle extracted by HP-trend in first differences can track long expansion episodes better, because level-information (at frequency zero) passes through
#   -In contrast, the gap-filter will generate (more false) alarms in between consecutive recessions, at least if the expansion separating them is longer than usual
#   -But the gap-filter will track very short expansion episodes better (the twin recessions in the early eighties)
# -In general, the shift or lag of the gap filter is smaller (the shift-function computed above indicated anticipation)
#   -Pros: the gap generally anticipates zero-crossings of the trend
#   -Cons: sometimes the anticipation can be  `too much`: we expect the gap to drop below the zero-line up to several years before the effective recession starts
#------------------------
# 7.2 Compare filter outputs and zero-crossings
# Let's look at the data: compare outputs of one-sided HP-trend and HP-gap

mplot<-scale(cbind(y_hp_gap,y_hp),center=F,scale=T)
plot(mplot[,1],main="One-sided HP-trend vs. HP-gap applied to Indpro",col="darkred",axes=F,xlab="",ylab="",type="l")
lines(mplot[,2],col="violet")
mtext("HP-gap applied to levels",line=-1,col="darkred")
mtext("HP-trend applied to returns",line=-2,col="violet")
abline(h=0)
axis(1,at=12*(1:(nrow(mplot)/12)),labels=index(diff(log(INDPRO)["1960/2019"]))[12*(1:(nrow(mplot)/12))])
axis(2)
box()

# The plot confirms the above conjectures based on frequency-domain analysis of the filters
# -The gap-cycle generates numerous zero-crossings along the expansions before the dotcom-bubble and the financial crisis  
# -In both cases early zero-crossings of the gap-cycle anticipate the recessions by several years
# -In contrast, the trend-cycle is much more conservative: it tracks longer expansions better but it generally lags behind the gap-cycle at start and end of recessions
# -Surprisingly, the trend filter is able to resolve/separate the twin-recessions in the early 80s.
# -Both filters indicate a trough around 2016, at a time  when the price for crude oil declined sharply, hence affecting petrol extraction as well as collateral industrial activity in the US.
#   -PAYEMS (or GDP) is less affected by this singular event
# -HP-trend applied to differences is a growth-tracker: 
#   -its mean-level, the drift of the original series, is not zero (unless the original series is flat) 
#   -it is not automatically reverting to the zero-line
#   -it is not even stationary: if the drift or slope of the original series changes, then the level of the `cycle` will change accordingly
# -HP-gap, on the other hand, has all requested cycle-characteristics (zero-mean, stationary, mean-reverting)
# We argue that this fundamental characteristics of a cycle, and in particular its `automatic' mean-reversion, are spurious
#   -If a particular phenomenon does not have an explicit built-in mean-reverting cycle mechanism, or if important 
#     actors work succesfully (counter-cyclically) against its appearance, or if the mean-reverting mechanism is weak, 
#     then a lowpass design, letting frequency zero pass through, is likely a more suitable design for analysing 
#     growth-phases.   


# Consider zero-crossings: empirical holding-time
compute_empirical_ht_func(y_hp)
compute_empirical_ht_func(y_hp_gap)
# The trend filter is slightly smoother 
# Compare with expected holding times, assuming the model of the data is correct: we use the convolved filters which are applied to model residuals (which is supposed to be white noise)
compute_holding_time_func(hp_one_sided_conv_indpro)$ht
compute_holding_time_func(hp_gap_diff_one_sided_conv_indpro)$ht
# The discrepancy between empirical and expected holding-times suggests 
#   a. random-sample errors (of empirical ht) 
#   b. model misspecification (for example non-vanishing means i.e. off-centered cycles)
# Probably both... But the numbers confirm that the cycle extracted by HP-trend is smoother (on the long run we expect ~30% less zero-crossings)

#-------------------------------------
# 7.3 Compare filter coefficients
# 7.3.1 Both filters as applied to returns
# Let us also compare hp_gap_diff and hp_trend, both applied to differences
# Since the scales differ we normalize the filters
ts.plot(scale(cbind(hp_trend,hp_gap_diff),center=F,scale=T),col=c("blue","darkred"),main="HP-trend and transformed HP-gap: both applied to differences")
mtext("HP-trend",col="blue",line=-1)
mtext("Transformed HP-gap",col="darkred",line=-2)
# We see a degree of familiarity. 
#   -The gap-coefficients drop below zero because their sum must vanish (vanishing amplitude at frequency zero: the original concurrent gap must cancel a second-order unit-root)
sum(hp_gap_diff)

# 7.3.2 Both filters as applied to levels
# We could also transform hp_trend (applied to differences) into a filter that replicates its cycle when applied to levels
# The transformation is very simple: apply first differences to filter coefficients:
hp_trend_sum<-c(hp_trend,0)-c(0,hp_trend)

# We can check that the transformed filter replicates the original one
y_hp<-filter(diff(log(INDPRO)["1960/2019"]),hp_trend,side=1)
y_hp_sum<-filter((log(INDPRO)["1960/2019"]),hp_trend_sum,side=1)
# Both series overlap
ts.plot(cbind(y_hp,y_hp_sum),main="Transformed HP-trend applied to levels replicates original HP-trend applied to returns")

# We can now compare the original HP-gap (as applied to levels) with the transformed HP-trend (also applied to levels, too)
# We rescale coefficients for easier visual comparison
ts.plot(scale(cbind(hp_gap,hp_trend_sum[1:L]),center=F,scale=T),col=c("red","darkblue"),main="Original HP-gap vs. transformed HP-trend: both applied to levels")
mtext("Transformed HP-trend",col="darkblue",line=-1)
mtext("Original HP-gap",col="red",line=-2)
abline(h=0)
# Once again: a degree of familiarity is apparent 
# Note that the coefficients of the transformed HP-trend now add to zero: the filter must cancel a single unit-root of the data in levels
sum(hp_trend_sum)

# We can also compare amplitude functions of original HP-gap and transformed HP-trend (as applied to levels)
amp_obj_gap<-amp_shift_func(K,hp_gap,F)
amp_obj_trend_sum<-amp_shift_func(K,hp_trend_sum,F)
par(mfrow=c(1,1))
# Let's scale the spectral densities for easier comparison
#   -We also add two vertical bars corresponding to business-cycle frequencies: 2-10 years periodicities
plot(scale(amp_obj_gap$amp,center=F,scale=T),type="l",col="red",axes=F,xlab="Frequency",ylab="",main=paste("Scaled amplitude of original gap and transformed trend",sep=""))
lines(scale(amp_obj_trend_sum$amp,center=F,scale=T),col="darkblue")
abline(v=K/12)
abline(v=K/60)
mtext("Amplitude HP-gap original (applied to levels)",col="red",line=-1)
mtext("Amplitude transformed trend (applied to levels)",col="darkblue",line=-2)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# Once again, the degree of familiarity is patent
#   -Both filters are highpass
#   -Original HP-gap approaches zero more smoothly (second order zero to cancel a double unit-root)
#   -Transformed HP-trend approaches zero more linearly (first order zero: cancels a single unit-root only)
#   -Transformed HP-trend is a hair smoother (the amplitude is a bit larger at business-cycle frequencies, where it peaks)

#-----------------------------------------
# Spurious cycles, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# -The following simulation experiment applies hp_trend and hp_gap_diff to an artificial time series
# -The time series in differences corresponds to a deterministic changing level (signal) overlaid with noise
# -The series in levels is a non-stationary series with changing growth-rate


set.seed(35)
len<-1000
# Deterministic changing level 
mu1<-c(rep(-1,200),seq(-1,1,by=0.05),rep(1,100),seq(1,0,by=-0.005),rep(1,198))
mu<-c(mu1,mu1)[1:len]
# Noise+level
eps<-rnorm(len)+mu

# Apply hp_trend and hp_gap_diff to the data 
x_trende<-filter(eps,scale(hp_trend,center=F,scale=T)/sqrt(L-1),sides=1)
x_gape<-filter(eps,scale(hp_gap_diff,center=F,scale=T)/sqrt(L-1),sides=1)


par(mfrow=c(2,2))
# Plot data together with level: representative of economic data in first differences
mplot<-na.exclude(cbind(eps[(L+1):len]))
plot(mplot[,1],main="Series in differences",axes=F,type="l",xlab="",ylab="",ylim=c(min(mplot),max(mplot)),col="grey")
#lines(c(rep(3.28,200),rep(NA,200)),lty=2)
#lines(c(rep(NA,400),rep(3.28,200)),lty=2)
lines(mu[L:len],lty=2,lwd=2)
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Plot integrated data: representative of economic indicator in levels
mplot<-na.exclude(cbind(eps[(L+1):len]))
plot(cumsum(mplot[,1]),main="Series in levels",axes=F,type="l",xlab="",ylab="",col="grey")
#lines(c(rep(3.28,200),rep(NA,200)),lty=2)
#lines(c(rep(NA,400),rep(3.28,200)),lty=2)
#lines(mu[L:800],lty=2)
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Plot filter outputs
mplot<-na.exclude(cbind(x_trende,x_gape))
plot(mplot[,1],main="Filter outputs",axes=F,type="l",xlab="",ylab="",ylim=c(min(mplot),max(mplot)),col="brown")
lines(mplot[,2],col="red")
mtext("Original HP-trend",col="brown",line=-1)
mtext("Transformed HP-gap",col="red",line=-2)
#lines(c(rep(3.28,200),rep(NA,200)),lty=2)
#lines(c(rep(NA,400),rep(3.28,200)),lty=2)
lines(3.28*mu[L:len],lty=2,lwd=2)
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


# Outcomes:
# -hp_gap_diff (red line) cancels the signal (changing level) and stays centered at the zero line
# -hp_trend (brown line) tracks the changing level (changing growth rate of data in levels)
# -The output of hp_gap_diff is a spurious cycle whose duration is determined by the frequency at which its amplitude peaks.
#   -the corresponding `cycle' is an artifact of the bandpass characteristics of the filter, as determined by lambda.
#   -the corresponding cycle is not related to a salient feature of the data.
# -In contrast, the output of hp_trend `extracts` a salient feature of the data, namely the changing level


# The undesirable `cyclical` movements of hp_trend are just `noise' which is due to high-frequency leakage of the filter
#   -The level shifts of the data are the proper `signal'
# In contrast, the `cyclical' gap of hp_gap_diff is the proper signal and the level-shifts of the data are a nuisance.

# Questions:
# Which filter output is `less spurious'? 
# Which filter output is more likely to extract relevant (salient) features from economic data?






###################################################################################################
###################################################################################################
# 8. Summary
# -The theoretical time series model underlying HP does not comply with a (business-) cycle
# -The two-sided HP filters (gap and trend) are 'too smooth' 
#   -The classic values of lambda, proposed in the literature, are eventually too large, see Phillips and Jin (2021) for background 
# -The classic one-sided HP, on the other hand, is less smooth: therefore it can better track short and severe recession dips 
#   -The twin recessions in the early 80s could be resolved
#   -The peak amplitude matches business-cycle frequencies
#   -The vanishing time-shift means that the filter is a tough benchmark in real-time applications
#     -The filter is typically faster than Hamilton's regression filter, see tutorial 3
# -The spectral density of the extracted cycle depends on the DGP (Wold decomposition)
#   -Applying the same HP-filter to Indpro and to Payems generates qualitatively different cycles (the latter is smoother than the former)
#   -This unresolved problem is not addressed in applications
# -HP-trend applied to differences shares many of the characteristics of the original HP-gap (applied to levels)
#   -Strong degree of familiarity when comparing filters for differences or levels: similar shapes, similar amplitude functions
#   -Advantages HP-trend: the cycle is smoother, it tracks long expansion-episodes better and it does not generate `spurious` cycle alarms in midst of an expansion
#   -Disadvantage: the cycle is systematically lagging at start and end of recessions
# -HP-gap is very `fast': the shift indicates an anticipative filter. Sometimes the anticipation is `too much' and the cycle systematically drops below the zero-line in midst of longer expansions 
# -A direct comparison of filter coefficients (in levels or differences) suggests a close affiliation
#   -The gap is based on identity minus trend 
#   -The transformed trend is based on trend_t minus trend_{t-1}
#   -These are similar transformations (but the former cancels unit-roots up to order two whereas the latter can cancel a single unit root only)
# In summary: HP-trend applied to differences is a more conservative design which tracks expansions and recessions more accurately than the original HP-gap 
# -Tutorial 2.1 will emphasize HP-trend filters accordingly: we then plug SSA on HP-trend in order to address smoothness/timeliness

