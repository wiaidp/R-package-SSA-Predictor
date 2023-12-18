# This tutorial is not related to SSA. It addresses the question of the business-cycle analysis (BCA-)design on which 
#     SSA is plugged in the JBCY-paper
# This tutorial is about a particular target, as proposed by Hodrick and Prescott (HP)






# 1.Background
# 1. Where it starts: Whittaker-Henderson smoothing
# Consider the smoothing problem
# \sum_{t=1}^T(x_t-y_t)^2+lambda \sum_{t=d}^T ((1-B)^d y_t)^2
# For d=2 the HP filter is obtained

# ARIMA-model: Tucker's package for MA
# Simulations of ARIMA with MA corresponding to lambda
# HP-gap vs. HP-trend

# 2. Are economic series compatible with I(2)?
# Links in JBCY paper
# Simulations
# Which BCA-tool is compatible with I(1): HP-trend applied to returns/diffs
#   -explain why

# 3. Derivation of HP
# Amplitude and shift of: trend one and two-sided, gap one and two-sided
# Transform HP-trend in diffs to new gap in levels: compare with original gap


# 4. Conclusions: HP-trend applied to returns
#   -Compatible with I(1)
#   -No spurious cycle
#   -Peak amplitude OK with business-cycle spec.
#   -Shift vanishing


#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions
rm(list=ls())

library(xts)
# Load the library mFilter
# Standard R-package for HP and other filters 
library(mFilter)
# Load data from FRED with library quantmod
library(quantmod)
# McElroy's package for HP
source(paste(getwd(),"/R/hpFilt.r",sep=""))
# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))


# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

# This tutorial is not related to SSA. It addresses the question of the design on which SSA is plugged in JBCY-paper
# Typical BCA-topic

# 1. Background
# 1.1 Derivation of HP based on Whittaker-Henderson smoothing
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
# This is a finite version the symmetric HP-trend filter
#   It is a causal (one-sided) filter
#   In applications, the filter is centered at t: it is acausal (expands equally in the past as in the future) 
ts.plot(hp_target)
# Filter coefficients add to 1: it is a lowpass (see amplitude functions further down)
sum(hp_target)

# HP can be interpreted as an optimal MSE-signal extraction filter for the trend in the smooth trend model, see Harvey (1989).
# Let us reproduce the implicit or latent model here: 
#   -It is an ARIMA(0,2,0)-trend plus a scaled white noise: y_t=T_t+sqrt(lambda_monthly)*I_t where T_t=ARIMA(0,2,0) and I_t=noise
set.seed(23)
len<-12000
x<-cumsum(cumsum(rnorm(len)))+rnorm(len)*sqrt(lambda_monthly)
ts.plot(x)
# First differences: slowly drifting away
ts.plot(diff(x))
# Look at acfs: acf of first differences indicates a weak but permanent positive pattern (the integration order is two)
acf(diff(x))
# After second order differences the acf looks fine
acf(diff(diff(x)))
# Let's fit a MA(2) model as suggested by the acf
arima(x,order=c(0,2,2))
# The implicit data-generating process follows an ARIMA(0,2,2)-specification whose MA-coefficients are determined by 
# lambda, see  McElroy (2006).
# We can use McElroy's package to compute the MA-parameters
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
# We can now simulate time series corresponding to this implicit model
# If economic series look similar then we can conclude that HP is an optimal filter for extracting the trend of them
set.seed(1)
# Monthly US industrial production index (INDPRO) (https://fred.stlouisfed.org/series/INDPRO) starts in 1920
# Roughly 100*12=1200 observations
# Assume a similar length for our simulation, initializing the series 100 years back (without initialization the series drift away rapidly with asymptotically unbounded slope)
len<-1200
x<-cumsum(cumsum(rnorm(len)))+rnorm(len)*sqrt(lambda_monthly)

# Check model: parameters are fine
arima(x,order=c(0,2,2))
# Diagnostics are fine, too
tsdiag(arima(x,order=c(0,2,2)))
# Have a look at the data
ts.plot(x)
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
ts.plot(mat_sim,col=rainbow(anzsim))

# The series are `flexing' and they are a bit `noisy': the `cycle' is just noise...
# Have a look at first differences
ts.plot(apply(mat_sim,2,diff),col=rainbow(anzsim))
abline(h=0)
# The series are slowly drifting away (non-stationary), as expected
# First differences are noisy: the differenced cycle dominates the dynamics 

# Let's compare with monthly indicators: Non-farm payroll and INDPRO
getSymbols('PAYEMS',src='FRED')
getSymbols('INDPRO',src='FRED')
# Original data
par(mfrow=c(2,2))
ts.plot(mat_sim,xlab="",main="Artificial/simulated",col=rainbow(anzsim))
plot(PAYEMS,main="Non-farm payroll")
plot(INDPRO,main="Indpro")
# The artificial data has less `structure' (the cycle is white noise...) 
# We recommend a log-transform in order to stabilize the variances which are changing with the level
par(mfrow=c(2,2))
ts.plot(mat_sim,xlab="",col=rainbow(anzsim),main="Artificial/simulated")
plot(log(PAYEMS),main="Log Non-farm payroll")
plot(log(INDPRO),main="Log Indpro")

# Compare series in first differences
par(mfrow=c(2,2))
ts.plot(apply(mat_sim,2,diff),xlab="",col=rainbow(anzsim),main="Artificial/simulated")
abline(h=0)
plot(diff(log(PAYEMS)),main="Returns Non-farm payroll")
plot(diff(log(INDPRO)),main="Returns Indpro")

# Remove pandemic, start in 1990 and use ts.plot (same graphic), shorten simulated sample accordingly
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
# As expected, the artificial data is subject to more crossings (the differenced `cycle' is differenced white noise)

####################################################################################################################
####################################################################################################################
# 2. HP-trend: two-sided and one-sided designs 
# Recall, from the above, that HP aims at estimating the trend from the trend+noise model.
# The above results suggest that 
#   a. The artificial data has less (no) `structure', when compared with macro-data (recessions, expansions)
#   b. The artificial data is noisier
# Therefore, if we assume the artificial data to be the truth, then it would make sense to apply a filter, i.e. HP-trend, which damps effectively (strongly) the noise
#   a. The filter does not have to `care' about additional structure 
#   b. Its only purpose is to erase the noise
# When applied to Macro-series, strong smoothing can attenuate or under-estimate the relevant structure, by washing-out 
#   critical (more or less sharp) recession dips

# We briefly compute the holding-time of the two-sided HP-trend
compute_holding_time_func(hp_target)$ht
# Big number! The filter has a very strong smoothing effect.  
# The above (finite-length) two-sided filter cannot be applied towards the sample end
# For this purpose one can use the optimal one-sided HP-trend
hp_trend<-HP_obj$hp_trend
par(mfrow=c(1,1))
ts.plot(hp_trend,main="Optimal one-side HP-trend assuming an ARIMA(0,2,2) DGP")
# Filter coefficients add to 1: it is a lowpass (see amplitude functions further down)
sum(hp_trend)
# Holding-time
compute_holding_time_func(hp_trend)$ht
# The holding-time is much shorter than for the two-sided filter

# Let's now apply the filter to simulated data
set.seed(18)
len<-1200
x<-cumsum(cumsum(rnorm(len)))+rnorm(len)*sqrt(lambda_monthly)

# Compute filter output of SSA-HP filter
y_hp_concurrent<-filter(x,hp_trend,side=1)
y_hp_symmetric<-filter(x,hp_target,side=2)

ts.plot(y_hp_concurrent,main="HP: two-sided vs one-sided filter",col="red")
lines(y_hp_symmetric)
abline(h=0)
mtext("Two-sided HP",col="black",line=-1)
mtext("One-sided HP",col="red",line=-2)

# Look at the filter approximation error
ts.plot(y_hp_symmetric-y_hp_concurrent,main="Filter error")
# The error looks stationary: it is, indeed, by optimality of the one-sided filter (both trend series are cointegrated)
# MSE 
mean((y_hp_concurrent-y_hp_symmetric)^2,na.rm=T)
# Any other one-sided filter has a larger MSE when the DGP is the above ARIMA(0,2,2), specified by lambda_monthly


####################################################################################################################
####################################################################################################################
# 3. In BCA applications, typically, the filter for extracting the cycle is HP-gap: the identity minus HP-trend
#   -In principle, this should be (close to) white noise since y_t=T_t+sqrt(lambda_monthly)*I_t  

hp_gap_sym<-c(rep(0,(L-1)/2),1,rep(0,(L-1)/2))-hp_target
ts.plot(hp_gap_sym,main="HP-gap two-sided")

# We can apply this filter to the data
y_hp_gap_symmetric<-filter(x,hp_gap_sym,side=2)
# Looks noisy:
ts.plot(y_hp_gap_symmetric)
# Acf suggest noise
acf(na.exclude(y_hp_gap_symmetric))

# HP-gap has cancelled the unit-roots of the ARIMA(0,2,2): the filter output is stationary (nearly white noise)
# The one-sided filter does similarly
hp_gap<-c(1,rep(0,L-1))-hp_trend
ts.plot(hp_gap,main="HP-gap one-sided")

# We can apply this filter to the data
y_hp_gap_concurrent<-filter(x,hp_gap,side=1)
# Looks noisy:
ts.plot(y_hp_gap_concurrent)
# Acf suggest noise
acf(na.exclude(y_hp_gap_concurrent))

# MSE: this is of course the same as MSE of trend approximation above (the identity cancels) 
mean((y_hp_gap_concurrent-y_hp_gap_symmetric)^2,na.rm=T)
# Any other one-sided filter has a larger MSE when the DGP is the above ARIMA(0,2,2), specified by lambda_monthly

#####################################################################################################
######################################################################################################
# 4. Frequency-domain analysis
# We now compute amplitude and time-shift function of the above filters

K<-600
amp_obj_hp_trend_concurrent<-amp_shift_func(K,hp_trend,F)
amp_obj_hp_trend_sym<-amp_shift_func(K,hp_target,F)
amp_obj_hp_gap_sym<-amp_shift_func(K,hp_gap_sym,F)
amp_obj_hp_gap_concurrent<-amp_shift_func(K,hp_gap,F)

# Amplitude functions
par(mfrow=c(1,1))
mplot<-cbind(amp_obj_hp_trend_concurrent$amp,amp_obj_hp_trend_sym$amp,amp_obj_hp_gap_sym$amp,amp_obj_hp_gap_concurrent$amp)
colnames(mplot)<-c("HP-trend one-sided","hp-trend two-sided","Hp-gap two-sided","HP-gap one-sided")
colo<-rainbow(ncol(mplot))
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
# Note that all filters are causal: therefore the lag of the two-sided designs correspond to half their length
#   The two-sided acausal filters are not shifted: their phase vanishes
# Also, the one-sided Gap has a negative shift: it is anticipative (but the filter let's seep through all high-frequency noise components)
# Let's then focus on the one-sided HP-trend only
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot[,1]),max(mplot[,1])),col=colo[1])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()

# We note that the time-shift (phase divided by frequency) of the one-sided HP-trend vanishes at frequency zero
#   -Quite unusual for a lowpass design: a small shift means a timely detection of peaks and troughs of a time series
#   -The shift must vanish because otherwise the filter error (between two-sided and one-sided trends) would not be stationary anymore in the case of an I(2)-process

###################################################################################################################
#################################################################################################################
# 5. Transformation: from levels to differences
# Let us assume, that typical (eventually log-transformed) economic time series are compatible with an I(1)-DGP

par(mfrow=c(2,2))
plot(diff(log(INDPRO)["1960/2019"]),xlab="",ylab="",main="Returns Indpro")
plot(diff(log(PAYEMS)["1960/2019"]),xlab="",ylab="",main="Returns non-farm payroll")
acf(na.exclude(diff(log(INDPRO)["1960/2019"])),main="ACF Indpro")                
acf(na.exclude(diff(log(PAYEMS)["1960/2019"])),main="ACF non-farm payroll")                
# Returns do not seem to be subject to changing levels 

# Also, simple stationary ARMA-models seem to fit the data well
model_indpro<-arima(diff(log(INDPRO)["1960/2019"]),order=c(2,0,1))
model_payems<-arima(diff(log(PAYEMS)["1960/2019"]),order=c(2,0,1))
tsdiag(model_indpro)
tsdiag(model_payems)

# Finally, economic theory suggests that a broad range of economic series, such as  stock prices, futures prices, 
#   long-term interest rates, oil prices, consumption spending, inflation, tax rates, or money supply growth rates  
#   should follow (near) martingales, see Fama (1965), Samuelson (1965), Sargent (1976), Hamilton (2009), Hall (1978) 
#   and Mankiw (1987)

# Let us therefore assume that the data is I(1), in contradiction to the ARIMA(0,2,2)-hypothesis of the implicit DGP justifying optimality of HP
# In this case, we could apply HP-trend to returns: the above amplitude functions suggest that 
#   a. the one-sided filter would be compatible with an extraction of the (differenced) business-cycle from the data 
#   b. the two-sided filter would damp cycle-frequencies too heavily (over-smoothing)

# Let us now compute the spectral density of the components (`cycles`) extracted by both filters.
# For this purpose we need to compute the convolution of HP and xi, the Wold-decomposition of the differenced data
# We can compute the weights of the Wold-decomposition of the above models: let's do so for Indpro (PAYEMS is similar)
ar<-model_indpro$coef[1:2]
ma<-model_indpro$coef[3]
xi_indpro<-c(1,ARMAtoMA(lag.max=L-1,ar=ar,ma=ma))
par(mfrow=c(1,1))
ts.plot(xi_indpro,main="MA-inversion (Wold decomposition) for Indpro")

hp_one_sided_conv<-conv_two_filt_func(xi_indpro,hp_trend)$conv
hp_two_sided_conv<-conv_two_filt_func(xi_indpro,hp_target)$conv

# Interpretation: hp_one_sided_conv applied to model-residuals generates the same output as hp_trend applied to returns
# We briefly check this assertion
y_hp_conv<-filter(model_indpro$residuals,hp_one_sided_conv,side=1)+model_indpro$coef["intercept"]
y_hp<-filter(diff(log(INDPRO)["1960/2019"]),hp_trend,side=1)
# Both series are overlapping
ts.plot(cbind(y_hp_conv,y_hp),main="Convolution applied to residuals vs. hp applied to returns: both series overlap")

# The spectral density of the `cycle` extracted by hp_trend is then the squared amplitude of hp_one_sided_conv
amp_obj_one_sided_hp_conv<-amp_shift_func(K,hp_one_sided_conv,F)
par(mfrow=c(1,1))
plot(amp_obj_one_sided_hp_conv$amp^2,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Spectral density of cycle extracted by one-sided HP trend",sep=""),col="blue")
abline(v=K/12)
abline(v=K/60)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# The vertical lines in the plot correspond to typical business-cycles periodicities: between two years and 10 years
#   -Pretty good match!

amp_obj_two_sided_hp_conv<-amp_shift_func(K,hp_two_sided_conv,F)
par(mfrow=c(1,1))
plot(amp_obj_two_sided_hp_conv$amp^2,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Spectral density of cycle extracted by two-sided HP trend",sep=""))
abline(v=K/12)
abline(v=K/60)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# The spectral density is right-shifted and does not match business-cycle frequencies
#   The two-sided is too-smooth, confirming our previous analysis

# Next we want to derive the spectral density of the cycle extracted by the gap-filter
# Note that the gap is typically applied to levels (not differences)
# We then transform the filter such that it can be applied to differences: our function computes this filter 
hp_gap_diff<-HP_obj$modified_hp_gap
# We check that the output of hp_gap_diff as applied to differences replicates the original gap when applied to levels 
y_hp_gap<-filter((log(INDPRO)["1960/2019"]),hp_gap,side=1)
y_hp_gap_diff<-filter(diff(log(INDPRO)["1960/2019"]),hp_gap_diff,side=1)
# Both series are overlapping
ts.plot(cbind(y_hp_gap,y_hp_gap_diff),main="Original HP-gap applied to levels vs. modified gap applied to returns: both series overlap")

# We can then convolve hp_gap_diff with the Wold-decomposition
hp_gap_diff_one_sided_conv<-conv_two_filt_func(xi_indpro,hp_gap_diff)$conv
# Why did we all of this?
# -The input of the original gap (hp_gap) is a non-stationary series: therefore the spectral density does not exist
# -The input of hp_gap_diff_one_sided_conv are the model residuals from the ARMA-model of the differencd series
#   -Both filter outputs are identical
#   -But now we can compute the spectral density of the 'cycle' (all series are stationary)
# The following plot compares spectral densities of the original HP-gap cycle (applied to levels) and HP-trend (applied to returns)
amp_obj_hp_gap_diff_conv<-amp_shift_func(K,hp_gap_diff_one_sided_conv,F)
par(mfrow=c(1,1))
plot(scale(amp_obj_hp_gap_diff_conv$amp^2,center=F,scale=T),type="l",axes=F,xlab="Frequency",col="red",ylab="",main=paste("Spectral density of cycle extracted by one-sided HP filters",sep=""))
lines(scale(amp_obj_one_sided_hp_conv$amp^2,center=F,scale=T),col="blue")
mtext("Spectral density original HP-gap applied to levels",line=-1,col="red")
mtext("Spectral density HP-trend applied to returns",line=-2,col="blue")
abline(v=K/12)
abline(v=K/60)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# The peaks of both spectral densities emphasize are left-shifted: the one-sided filters are damping the higher cycle-frequencies  
# HP-trend is slightly more left-shifted (stronger smoothing) and it also let's low-frequency trend components seep through
#   -Both properties are potentially interesting when trying to track recessions



# Compare one-sided HP-trend and HP-gap
ts.plot(scale(cbind(y_hp_gap,y_hp),center=F,scale=T),main="One-sided HP-trend vs. HP-gap applied to Indpro",col=c("blue","red"))
mtext("HP-gap applied to levels",line=-1,col="red")
mtext("HP-trend applied to returns",line=-2,col="blue")
abline(h=0)

compute_empirical_ht_func(y_hp)
compute_empirical_ht_func(y_hp_gap)


compute_holding_time_func(hp_one_sided_conv)$ht
compute_holding_time_func(hp_gap_diff_one_sided_conv)$ht


# Concurrent gap: as applied to series in levels: this is a high pass filter
hp_gap=HP_obj$hp_gap
ts.plot(hp_gap)
# Concurrent HP assuming I(2)-process 
# This is the Classic concurrent or one-sided low pass HP, see e.g. McElroy (2006)
hp_trend=HP_obj$hp_trend
ts.plot(hp_trend)

# Compute lag one acf and holding time of HP concurrent
htrho_obj<-compute_holding_time_func(hp_trend)
rho_hp<-htrho_obj$rho_ff1
ht_hp<-htrho_obj$ht
ht_hp

# Compare hts of one- and two-sided filters
compute_holding_time_func(hp_target)$ht
ht_hp
# The large (atypical) discrepancy between holding-times of two- and one-sided filters is discussed in the JBCY paper

#------------------------------
# b. Classic concurrent HP trend
# Let us briefly have a look at the above discrepancy and some of its effects: for this purpose we compute and compare filter outputs
len<-L+1000
set.seed(67)
# Data: white noise (in the JBCY paper we apply the filter to log-returns of INDPRO which are close to noise)
a1<-0
x<-arima.sim(n = len, list(ar = a1))
# Compute filter output of SSA-HP filter
y_hp_concurrent<-filter(x,hp_trend,side=1)
y_hp_symmetric<-filter(x,hp_target,side=2)

ts.plot(y_hp_concurrent,main="HP: two-sided vs one-sided filter",col="red")
lines(y_hp_symmetric)
abline(h=0)
mtext("Two-sided HP",col="black",line=-1)
mtext("One-sided HP",col="red",line=-2)

# Let us compute empirical holding times of both filters:
compute_empirical_ht_func(y_hp_concurrent)
compute_empirical_ht_func(y_hp_symmetric)
# The difference of empirical hts is large, as expected (these numbers would converge to the above `true' hts for very long samples)


# -The plot of the time series suggests that the two-sided filter can stay away from the zero-line over long time episodes.
# -This characteristic of the two-sided HP can lead to over-smoothing, whereby critical recession dips are 
#   washed-out by the filter: the resulting overdamped cycle underestimates the felt impact of crises, see Phillips and Jin (2021) for background.
# -Interestingly, the one-sided HP can track recession dips better, due to its weaker noise-suppression (smaller ht) 
#   -We can gather more information by looking at frequency domain characteristics

# Let us compute and plot amplitude and time-shifts of the classic concurrent HP
K<-600
amp_obj<-amp_shift_func(K,as.vector(hp_trend),F)
par(mfrow=c(1,2))
# Amplitude
plot(amp_obj$amp,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP",sep=""))
mtext("Amplitude classic concurrent HP trend",line=-1)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()
# Shift
plot(amp_obj$shift,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Time-shift HP",sep=""))
mtext("Shift  classic concurrent HP trend",line=-1)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()

# The peak of the amplitude function corresponds roughly to a periodicity of 7 years, which is in accordance with the concept of a `business-cycle'
# The time-shift is small, in particular towards frequency zero
#   -Recall that the filter output must track the level of an I(2) series: a non-vanishing shift would generate a non-stationary (integrated) error with respect to the two-sided target output

#-----------------------
# c. Summary
# -The two-sided HP is not necessarily a worthwhile target for BCA: it is possibly `too smooth' 
#   -The classic values of lambda, proposed in the literature, are eventually too large, see Phillips and Jin (2021) for background 
# -The classic one-sided HP, on the other hand, is less smooth: therefore it can better track short but severe recession dips
#   -The peak amplitude matches business-cycle frequencies
#   -The vanishing time-shift means that the filter is a tough benchmark
#     -The filter is typically faster than Hamilton's regression filter in real-time applications
# -We therefore propose to target two-sided  (examples 4 and 6)  as well as one-sided designs by SSA (examples 1,2,3,5 and 8) 
# -Example 1 addresses specifically the one-sided hp_mse, assuming the data to be white noise
#   -Example 4 illustrates that this particular target is equivalent to the two-sided HP when the data is white noise, thus confirming proposition 5 in the JBCY paper
