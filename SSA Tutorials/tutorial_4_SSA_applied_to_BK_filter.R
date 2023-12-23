# In this tutorial we discuss an application of SSA to the Baxter and King (BK) ideal bandpass filter.

# Main outcomes: 
#   1.The analysis suggests that BK is not well suited for BCA (business cycle analysis) 
#         because the filter generates a `spurious' cycle (see our analysis of the amplitude functions below). 
#       -The periodicity of the nearly sinusoidal cycle is determined entirely by the lower bandpass specification (artifact)
#   2.The BK filter output is very smooth (sinusoidal like) and therefore an application of SSA is not necessary 
#       or useful in this case. 
#       -Increasing smoothness even more, by imposing a larger holding-time in SSA, leads to atypical filter-characteristics in this particular setting  
# Recommendations: 
#   -Be cautious when applying bandpass designs for the purpose of BCA (HP-gap is subject to similar shortfalls)
#   -Compute amplitude functions to assess filter characteristics (detect spurious cycles). 
#   -Verify that SSA does the right job, namely: lessen the number of noisy-crossings attributable to undesirable leakage of high-frequency components by the benchmark  
# In short: this tutorial is an illustrative counter example with some unexpected (but statistically sound) outcomes 
#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions
rm(list=ls())

library(xts)
# HP and BK filters
library(mFilter)
# Load data from FRED
library(quantmod)


# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


#----------------------------------------------------------------------
# Source data directly from FRED
getSymbols('PAYEMS',src='FRED')

# Discard Pandemic: extreme outliers affect regression of Hamilton filter
# Make double: xts objects are subject to lots of automatic/hidden assumptions which make an application of SSA 
# more cumbersome, unpredictable and hazardous
y<-as.double(log(PAYEMS["/2019"]))
len<-length(y)
#--------------------------------------
# Example 1: apply BK to levels (classic usage: example 2 below applies BK to returns)
# 1.1. BK filter: rely on package mFilter
# Need to provide a series in order to obtain filter coefficients: length must be an odd integer to obtain a correctly centered two-sided filter   

len_f<-1201
x<-1:len_f
x<-ts(x,frequency=96)
# Settings for bandpass specification for monthly data

# Bandpass: in their paper B-K select pl=6, pu=32 for quarterly data: we adapt these settings for monthly data here
pl=3*6;pu=3*32
filt_obj<-mFilter(x,filter="BK",pl=pl,pu=pu,drift=T)
# BK-filter matrix
filt<-filt_obj$fmatrix

dim(filt)

# Filter length: select an odd number such that the symmetric filter is indeed symmetric
L<-201

# Symmetric filter: see plot further down
bk_target<-filt[((len_f+1)/2-(L+1)/2+1):((len_f+1)/2+(L+1)/2-1),(len_f+1)/2]
# bandpass: the bi-infinte filter adds to zero; the finite-length filter does not
# If the coefficients do not add to zero, then the filter cannot remove unit-roots 
sum(bk_target)
# Concurrent filter assuming white noise (truncate symmetric filter)
fil_c<-filt[((len_f+1)/2):((len_f+1)/2+L),(len_f+1)/2]
# Coefficients do not add zero: filter cannot remove unit-roots 
sum(fil_c)
bk_wn<-fil_c[1:L]

# Concurrent filter assuming a random-walk: weight at lag zero is the sum of entire left tail of the two-sided filter
#   -The best forecast of a random-walk is the last observation x_T 
#   -Inserting this forecast for the unknown future observations means that X_T receives weight: bk_0+bk_{-1}+bk_{-2}+....
bk_rw_finite<-bk_wn
bk_rw_finite[1]<-sum(bk_rw_finite)
# Coefficients do not add to zero because of the finite length of all filters: however, a zero is required here
#   because the filter is applied to non-stationary data: it must cancel a unit-root (the trend)
sum(bk_rw_finite)
# We then adjust the filter such that its coefficients add to zero
#   -Just distribute the error evenly over all lags>0 (the alternative would be to assign the zero-error to the last lag only: backcast of series into past is just first observation)
#   -Whichever solution we use does not affect our results below
bk_rw<-bk_rw_finite
bk_rw[2:L]<-bk_rw_finite[2:L]-sum(bk_rw_finite)/(L-1)
# Now the concurrent BK filter weights add to zero: the filter can remove a single unit-root in the data
# As a result, the cycle or filter output will be stationary (at least if the integration order of the data generating process is one)
sum(bk_rw)

# Finally, we can plot all filters
par(mfrow=c(2,2))
ts.plot(bk_target,main="Two-sided BK")
ts.plot(bk_wn,main="One-sided BK assuming the data to be white noise")
ts.plot(bk_rw_finite,main="One-sided BK assuming the data to be a RW")
ts.plot(bk_rw,main="Adjusted one-sided BK assuming the data to be a RW")


# Let's check the amplitude functions and compare the passband with our specifications, pl and pu (vertical black lines in plot)
K<-600
par(mfrow=c(1,1))
plot(amp_shift_func(K,bk_target,F)$amp,type="l",axes=F,xlab="Frequency",ylab="",col="blue",main="Amplitude of 
     two- and one-sided BK with theoretical bandpass specifications")
lines(amp_shift_func(K,bk_rw,F)$amp,col="red")
mtext("Two sided BK",col="blue",line=-1)
mtext("One sided adjusted BK",col="red",line=-2)
abline(v=2*K/18+1)
abline(v=2*K/96+1)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# The filters let pass components of duration between 18 and 96 months, as requested
#   Ripples are a manifestation of the Gibbs phenomenon (finite-length truncation)
# Since the amplitude of the (finite-length) two-sided filter does not vanish at frequency zero we infer that this 
#   filter would not cancel unit-roots (trend)
# However, the adjusted one-sided filter cancels a (single) unit-root: its amplitude vanishes at frequency zero, as desired




#---------------------------------------------------
# 1.2. Transformation: from levels to differences (see section 2.3 in JBCY paper)
# SSA must be applied to stationary data: we then transform the BK filter such that it can be applied to differences
# This proceeding is necessary for all bandpass designs, when applied to non-stationary data
# A similar proceeding was applied to the Hamilton filter in tutorial 3 as well as to the HP-gap in section 4 of JBCY paper
# See section 2.3 and proposition 4 in JBCY paper for background.

# Convolution with summation filter (unit-root assumption)
bk_diff<-conv_with_unitroot_func(bk_rw)$conv
par(mfrow=c(2,1))
ts.plot(bk_diff,main="BK filter as applied to first differences")
ts.plot(bk_rw,main="BK filter as applied to level")

# Check that both filters generate the same output/cycle
x<-diff(y)
len_diff<-length(x)
cycle<-na.exclude(filter(y,bk_rw,side=1))
cycle_diff<-na.exclude(filter(x,bk_diff,side=1))

# Check: bk_diff, as applied to first differences, generates the same cycle as original bk_rw filter applied to levels
# With this transformation in place we can now work with SSA
par(mfrow=c(1,1))
ts.plot(cycle_diff,col="blue",main="bk_diff applied to differences replicates BK applied to levels")
lines(cycle[2:length(cycle)],col="red")

#-------------------------------------------------------------------------------
# 1.3. Holding-times and 'understanding the data'
# We first compute the holding-time of the BK-filter (as applied to noise i.e. differenced data)

ht_bk_diff_obj<-compute_holding_time_func(bk_diff)
# Large holding-time: the filter is much smoother than Hamilton-Filter (tutorial 3) or concurrent HP (tutorial 2.0)
#   -The original HP-gap or the transformed HP-trend are high-pass filters: in contrast to BK they do not damp high-frequency components (of data in levels)
ht_bk_diff_obj$ht 
# But the empirical holding time is even longer
compute_empirical_ht_func(cycle_diff)
# Even after centering the empirical ht is still substantially larger
compute_empirical_ht_func(cycle_diff-mean(cycle_diff))

# Explanation: xt (log-returns PAYEMS) is not white noise
par(mfrow=c(2,1))
ts.plot(x,ylim=c(-0.05,0.05),main="Log-returns PAYEMS")
abline(h=0)
acf(x,main="ACF")

#------------------------------------------------------------------------
# 1.4. Autocorrelation
# Fit a simple model to the data: in most cases ARMA(1,1) is well suited to fit a weak but slowly decaying ACF pattern
ar_order<-1
ma_order<-1
estim_obj<-arima(x,order=c(ar_order,0,ma_order))
# We have the typical cancelling AR and MA-roots which can fit a weak but long lasting ACF 
estim_obj
tsdiag(estim_obj)

# Compute the MA-inversion of the ARMA: Wold-decomposition
xi<-c(1,ARMAtoMA(ar=estim_obj$coef[1:ar_order],ma=estim_obj$coef[ar_order+1:ma_order],lag.max=L-1))
par(mfrow=c(1,1))
ts.plot(xi)

# Convolve xi and bk_diff: this filter would be applied to epsilont: it us used for determining the holding-time
bk_conv<-conv_two_filt_func(xi,bk_diff)$conv
ht_bk_conv_obj<-compute_holding_time_func(bk_conv)
# Now the expected holding time matches the empirical one (up to finite sample error)
ht_bk_conv_obj$ht 
# Empirical holding time (need to center data in order to comply with white noise assumption underlying ht)
compute_empirical_ht_func(cycle_diff-mean(cycle_diff))

# We are now in a position to plug SSA on BK
#   -We can work with stationary series thanks to our transformation from bk_rw to bk_diff
#   -We can work with the Wold decomposition given by xi

#------------------------------------------------------------
# 1.5. Diagnostics
# Before proceeding to SSA, let's first have a closer look at the BK filter:
# We now compute squared amplitude functions of the filters bk_rw (original concurrent BK as applied to data in levels), 
#   bk_diff (same output as bk_rew but when applied to first differences) and bk_conv (same output as bk_diff or bk_rw 
#   but when applied to white noise epsilont)  
# All filters generate the same output or cycle, but based on different 'input' series
K<-600
amp_obj_rw<-amp_shift_func(K,bk_rw,F)
amp_obj_x<-amp_shift_func(K,bk_diff,F)
amp_obj_eps<-amp_shift_func(K,bk_conv,F)

par(mfrow=c(1,1))
mplot<-scale(cbind(amp_obj_rw$amp^2,amp_obj_x$amp^2,amp_obj_eps$amp^2),scale=T,center=F)
colnames(mplot)<-c("BK-level","BK-diff xt","BK-diff epsilont")
colo=c("blue","darkgreen","red")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Squared amplitude one-sided BK",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
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
axis(2)
box()

# Findings:
# BK-level looks like a bandpass: it is applied to the non-stationary (log-transformed) data in level
# BK-diff has the same output as BK_level, but it is applied to xt, i.e. log-returns(PAYEMS), which is an ARMA process
# BK_diff epsilont has the same output as BK_level and BK-diff, but it is applied to epsilont, the innovations of the Wold-decomposition of xt
# Since epsilont is (nearly) white noise, its spectral density is flat. Therefore the squared amplitude of the last filter 
#   (red line in above plot) corresponds to the spectral density of the filter-output (common to all three filters) 
# We infer that the output of the classic BK, as applied to levels, is a spurious cycle whose peak frequency 
#     corresponds to the frequency of the lower bandpass cutoff (specified by pu)
#   -This phenomenon explains the remarkable smoothness of the BK-cycle. In particular BK does not require any additional 
#     enhancement (in terms of larger ht) by SSA
# To be clear: this 'cycle' is not a feature of the data. Rather, it is an artifact which is determined by the 
#   lower cutoff of the BK-filter specification
# Another interesting characteristic can be derived from the amplitude functions at frequency zero
# -BK-level vanishes by our modification above: the zero at frequency zero removes a unit-root of the series in levels
# -The amplitudes of the other BK-filers (applied to returns) do not vanish at frequency zero (the zero in  the original filter is cancelled by the unit-root and the limiting amplitude value is shown in the above plots)
# -As a consequence, content at frequency zero (of the returns) will seep through the filter
# -We know that the level of the returns is non-stationary (the drift of the log-transformed data is changing over time)
# -This changing level (non-stationarity) will thus 'leak' through (all) the filters
# -We may claim, alternatively, that the above stationary ARMA(1,1)-model is a misspecification for the (drifting) returns
# -This misspecification will have consequences on the forecasts and, in particular, on long-term (4-years ahead) forecasts, see corresponding results below
# We will talk more about these topics below, in particular when looking at forecasts: the 'strange' looking forecasts are a 
#    consequence of the observed artifacts here


# Note: 
#   1. The Hamilton filter applied to levels (see tutorial 3) is a lowpass and does not generate spurious cycles; 
#   2. Similarly for HP lowpass applied to returns, see tutorial 5 and JBCY paper
#   3. On the other hand, HP-gap, applied to levels, generates also a spurious cycle, see tutorials 2 and 5: it is not recommended for BCA

# Given these findings we could stop here. But the application of SSA to BK is instructive by and of itself (skip and forget the BCA framework).
# From now on, we are NOT looking at improving a BCA-tool (which the BK is not) but rather at some interesting implications 
#   of engrafting SSA onto BK. The outcome is not necessarily what we're looking for with SSA: mainly, because BK is 
#   sufficiently smooth at the start (very large holding-time, few `noisy' crossings) and does not require additional 
#   smoothness by SSA. 
# What happens in such a case?
#--------------------------------------
# 1.6. Apply SSA: 
# The above holding-time was approximately 2.5 years
ht_bk_conv_obj$ht 
# We can lengthen by 20% (on an already very large ht of BK)
#  The large ht requires a long filter: L is larger than in other tutorials
ht<-1.2*ht_bk_conv_obj$ht
# This is the corresponding rho1 for the holding time constraint
rho1<-compute_rho_from_ht(ht)
# Since we apply the filter to xt=log-returns, our target is bk_diff 
#   If we applied the filter to epsilont in the Wold decomposition then the correct target would be bk_conv
gammak_generic<-bk_diff
# Forecast horizon: nowcast
forecast_horizon<-0
# We want to apply the filter to xt (not to epsilont): therefore we supply xi (Wold decomposition)

SSA_obj_bk_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# ssa_eps is the filter which is applied to epsilont
SSA_filt_bk_diff_eps<-SSA_obj_bk_diff$ssa_eps
# ssa_x is the filter which is applied to xt
SSA_filt_bk_diff_x<-SSA_obj_bk_diff$ssa_x

# Compare compare bk_diff with ssa_x: both filters are applied to xt 
mplot<-cbind(bk_diff,SSA_filt_bk_diff_x)
# Compare target and SSA: 
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"))
mtext("BK (as applied to differences)",col="black",line=-1)
mtext("SSA", col="blue",line=-2)

# We can also compare bk_conv with ssa_eps: both filters are applied to epsilont 
mplot<-cbind(bk_conv,SSA_filt_bk_diff_eps)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"))
mtext("BK (as applied to differences)",col="black",line=-1)
mtext("SSA", col="blue",line=-2)



# Check holding time constraint
ht_obj<-compute_holding_time_func(SSA_filt_bk_diff_x)
# Does not match our target: this is because xt is not white noise 
ht_obj$ht 
ht
# In order to check ht we must supply the 'other' filter, ssa_eps, which is (or could be) applied to epsilont
ht_obj<-compute_holding_time_func(SSA_filt_bk_diff_eps)
# Now everything accords
ht_obj$ht 
ht

# We can verify that the convolution of ssa_x and xi replicates ssa_eps 
ts.plot(scale(cbind(conv_two_filt_func(SSA_filt_bk_diff_x,xi)$conv[1:L],SSA_filt_bk_diff_eps),center=F,scale=T),col=c("blue","red"),main="Convolution of ssa_x and xi replicates ssa_eps")

#--------------------------------------------------
# 1.7. Filter series
# Apply SSA
SSA_out<-filter(x,SSA_filt_bk_diff_x,side=1)
# Quite larger than targeted ht...
compute_empirical_ht_func(SSA_out)
ht  
# Apply BK
bk_out<-filter(x,bk_diff,side=1)
compute_empirical_ht_func(bk_out)
# Quite larger than expected ht
ht_bk_conv_obj$ht 
# Surprisingly, empirical holding times of BK and SSA match exactly 
#   The large holding-times would require a much longer sample in order to obtain reliable empirical estimates
#   Also, we required a 20% increase of ht only (in contrast to previous tutorials where we imposed 50%)


# We can see why the empirical holding time is larger: the series has a non-vanishing (positive) mean
#   The computation of the expected holding-time assumes white noise, i.e., a zero mean 
# The plot also confirms that the BK-cycle is already extremely smooth and does not require additional treatment by SSA
mplot<-na.exclude(cbind(SSA_out,bk_out))
colo=c("blue","red")
ts.plot(mplot[,1],col=colo[1])
lines(mplot[,2],col=colo[2])
abline(h=0)
mtext("BK ",col="red",line=-1)
mtext("SSA", col="blue",line=-2)


# In order to verify our conjecture, let's apply the filter to the mean-centered series instead
SSA_out_center<-filter(x-mean(x),SSA_filt_bk_diff_x,side=1)
# Now the empirical holding time accords better with the expected ht (but the series is still non-stationary: the model is misspecified) 
compute_empirical_ht_func(SSA_out_center)
ht 


#-----------------------------------------------
# 1.8. Timeliness
# Let us now address timeliness or lead/lags: the last plot suggested that SSA is slightly right shifted (lagging)
# We augment the forecast horizon (delta in the JBCY paper) to obtain a faster SSA filter without giving-up smoothness
forecast_horizon<-12
# SSA of BK-diff
SSA_obj_bk_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_bk_diff_x_forecast<-SSA_obj_bk_diff$ssa_x

# Filter series
SSA_out_forecast<-filter(x,SSA_filt_bk_diff_x_forecast,side=1)
bk_out<-filter(x,bk_diff,side=1)

# Plot series: 
par(mfrow=c(1,1))
# The SSA-forecast seems odd...
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast,bk_out))
colo=c("blue","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon,sep=""),col=colo[2],line=-2)
mtext("BK",col=colo[3],line=-3)
abline(h=0)

# The forecast does not seem right...

# Let's shift the forecast forward by one year: seems better now
par(mfrow=c(1,1))
# That's better now
mplot_shift<-na.exclude(cbind(SSA_out[(1+forecast_horizon):length(SSA_out_forecast)],SSA_out_forecast[1:(length(SSA_out)-forecast_horizon)],bk_out[(1+forecast_horizon):length(SSA_out_forecast)]))
colo=c("blue","darkgreen","red")
ts.plot(mplot_shift[,1],col=colo[1],ylim=c(min(mplot_shift),max(mplot_shift)),main="All series with forecast shifted forward by one year")
lines(mplot_shift[,2],col=colo[2])
lines(mplot_shift[,3],col=colo[3])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon,sep=""),col=colo[2],line=-2)
mtext("BK",col=colo[3],line=-3)
abline(h=0)




# Let us also compare the SSA-criterion value (expected correlation with shifted target) and the empirical correlation  
#   of SSA-forecast and target (shifted BK)
# 1. Theoretical correlation: this is computed in closed-form in SSA-function, see section 2 in JBCY-paper for details
SSA_obj_bk_diff$crit_rhoyz
# 2. Empirical correlation of forecast and shifted BK
cor(mplot[(1+forecast_horizon):nrow(mplot),3],mplot[1:(nrow(mplot)-forecast_horizon),2])
# Not too bad: numbers are pretty close.
# Part of the difference is due to sample error and to the fact that the cycle is non-stationary
#   (changing drift of the data, model misspecification)

#-----------------------------------------------
# 1.9. Long forecast horizon
# Let's now try a much longer forecast horizon corresponding to half the cycle-length at the left passband
#  limit specified by pu: we will understand further down the purpose of this particular forecast horizon 
# We compute a very long 4-year ahead forecast!
forecast_horizon<-pu/2
# SSA of BK-diff
SSA_obj_bk_diff_long<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_bk_diff_x_forecast_long<-SSA_obj_bk_diff_long$ssa_x

# Filter series
SSA_out_forecast_long<-filter(x,SSA_filt_bk_diff_x_forecast_long,side=1)

# Plot series: the scale of the forecast is smaller (zero-shrinkage: the estimation problem is more difficult)
par(mfrow=c(1,1))
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_long,bk_out))
colo=c("blue","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon,sep=""),col=colo[2],line=-2)
mtext("BK",col=colo[3],line=-3)
abline(h=0)

# The SSA-forecast seems to mirror the target (BK): it is out of phase.
# This conjectuire is confirmed by the filter coefficients
ts.plot(SSA_filt_bk_diff_x_forecast_long,col="darkgreen",ylim=c(min(SSA_filt_bk_diff_x_forecast_long),max(SSA_filt_bk_diff_x)),main="SSA nowcast (blue) vs forecast (green)")
lines(SSA_filt_bk_diff_x,col="blue")

# How can we explain and interpret this forecast filter?

# We first compare the SSA-criterion value (expected correlation with shifted target) and the empirical correlation
# 1. Theoretical correlation: the number is a bit optimistic (see below for an explanation)
SSA_obj_bk_diff$crit_rhoyz
# 2. Empirical correlation of forecast and shifted BK
cor(mplot[(1+forecast_horizon):nrow(mplot),3],mplot[1:(nrow(mplot)-forecast_horizon),2])
# The empirical correlation is smaller but overall the discrepancy is not too bad given the large 4-year ahead 
#   forecast horizon and the misspecification (non-stationarity : slowly drifting returns)
# Conjecture: the very long forecast horizon magnifies misspecification (slow drift) 

#-----------------------------------------------
# To confirm our conjecture we can try a smaller forecast horizon corresponding to half a year 
forecast_horizon<-6
# SSA of BK-diff
SSA_obj_bk_diff_long<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_bk_diff_x_forecast_long<-SSA_obj_bk_diff_long$ssa_x

# Filter series
SSA_out_forecast_long<-filter(x,SSA_filt_bk_diff_x_forecast_long,side=1)

# Plot series
par(mfrow=c(1,1))
# The SSA-forecast seems to mirror the target (BK)...
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_long,bk_out))
colo=c("blue","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon,sep=""),col=colo[2],line=-2)
mtext("BK",col=colo[3],line=-3)
abline(h=0)

# Compare the SSA-criterion value (expected correlation with shifted target) and the empirical correlation
# 1. Theoretical correlation: well, the number is a bit optimistic... we shall give an explanation further down
SSA_obj_bk_diff$crit_rhoyz
# 2. Empirical correlation of forecast and shifted BK
cor(mplot[(1+forecast_horizon):nrow(mplot),3],mplot[1:(nrow(mplot)-forecast_horizon),2])
# Quite close: the non-stationarity (slow drift) of the cycle is less problematic at the (much) shorter forecast horizon


# In order to obtain a better understanding of the `strange' long-term out-of-phase forecast we have to look at the 
#   amplitude functions
#----------------------------------------
# 1.10. Compute amplitude and phase-shift functions

K<-600
amp_obj_SSA_now<-amp_shift_func(K,as.vector(SSA_filt_bk_diff_x),F)
amp_obj_SSA_for<-amp_shift_func(K,as.vector(SSA_filt_bk_diff_x_forecast),F)
amp_obj_bk<-amp_shift_func(K,bk_diff,F)

par(mfrow=c(2,1))
mplot<-cbind(amp_obj_SSA_now$amp,amp_obj_SSA_for$amp,amp_obj_bk$amp)
# We here scale all amplitudes so that their peak values are identical 
mplot[,2]<-mplot[,2]/max(mplot[,2])*max(mplot[,1])
mplot[,3]<-mplot[,3]/max(mplot[,3])*max(mplot[,1])
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"BK")

# The amplitude functions suggest that the BK-design is not well-suited for BCA because the (one-sided) 
#   filter has a strong and narrow peak at the lower limit of the passband: essentially, the filter generates
#   a spurious cycle with frequency 2*pi/pu
colo=c("blue","darkgreen","red")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude BK",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
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
axis(2)
box()

mplot<-cbind(amp_obj_SSA_now$shift,amp_obj_SSA_for$shift,amp_obj_bk$shift)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"BK")
# The phase-shift is quite large at the peak of the amplitude function so that the series will be shifted to the right (lags)
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Shift (phase-lag) ",sep=""),ylim=c(-6,20),col=colo[1])
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

# 1. The amplitude functions of SSA are interesting and to some extent atypical and unexpected 
#   -The original BK is already very smooth 
#     -BK does not require additional noise suppression, unlike HP (tutorials 2 and 5) or Hamilton-filter (tutorial 3)
#     -The amplitude function of BK is very close to zero towards higher frequencies: strong smoothing
#     -Therefore, SSA cannot reduce the number of crossings by damping high-frequencies even more: BK does already a nearly perfect `job'!
#     -Quite the contrary: the amplitude of SSA is (marginally) farther away from zero at higher frequencies. 
#       -This outcome is very atypical and unexpected...
#   -So how does SSA manage to generate less crossings?
# 2. In order to reduce the number of crossings, SSA damps lower frequencies in the passband of (the original) BK  
#     -The amplitude functions of SSA drop below BK to the left and to the right of the upper bandpass boundary 2*pi/18
#       -Once again: this is atypical and unexpected
#       -But statistically pertinent: it is the most effective way for the filter to reduce the number of crossings further
#     -We infer that the `noise suppression' feature of SSA in this particular BK-context is unneccesary and, to some extent, 
#       detrimental because the passband is affected in an undesirable way.
# We now try to interpret the somehow odd forecasts in the above plots, in particular at the (very long) forecast horizon pu/2
#   1. Due to the narrow amplitude peak at 2*pi/pu, the BK filter output is not too far away from a regular (sinusoidal like)  
#         spurious cycle with frequency 2*pi/pu
#       -This kind of signal is very smooth 
#       -Also, this kind of regular pattern is pretty easy to forecast: just add a phase corresponding to the forecast horizon
#       -The long forecast_horizon=pu/2 corresponds to half the spurious cycle-length: therefore the corresponding 
#         4-year ahead forecast is out-of-phase
#   2. Unfortunately, the series to be filtered is non-stationary (slowly drifting): 
#       -It is not an ARMA(1,1) as assumed by our model (from which xi is derived)
#       -Moreover, the amplitude functions of the filters do not vanish at frequency zero. 
#         -Therefore the drift will seep through and affect filter outputs
#       -The empirical forecast error is larger than projected by the criterion value (if the model were true): 
#         the expected correlation (in the absence of misspecification) is larger than the empirical correlation 
#       -Moreover, the discrepancy between theoretical and empirical numbers increases with the forecast horizon
#         -Because the seeping-through slow changing level affects long-term forecasts more heavily

# BK: an instructive counter-intuitive counterexample! 

##################################################################################################
###################################################################################################
# What would happen if we applied BK to differences or returns? See example 2 below!
#   -The spectral content of such a design should be more evenly distributed 
#     -no isolated single peak as in example 1 above
#     -does that mean no spurious cycle?
#     -Would the filter be suitable for BCA?
#   -Another advantage: the filter would remove the remaining non-stationarity (slowly changing drift of the data) 
#     so that the resulting cycle would be centered at the zero-line.
# -Note: as a potential alternative, one could also consider HP-gap, as applied to returns (instead of levels) 
#   -But the resulting high-pass (no suppression of high-frequency content) would lead to full (100%) noise-leakage 
#   -In such a case, SSA could be used to emphasize smoothness, increasing ht strongly over the high-pass benchmark

####################################################################################################
######################################################################################################
# Example 2: Apply BK to returns
# Run the code lines 1.1-1.4 in the above example 1 if not done yet

# We can mostly rely on example 1 with three modifications.
# Modifications:  
# 1. We apply bk_rw to x (instead of bk_diff to x)
# 2. We consider a convolution of xi with bk_rw (instead of xi with bk_diff)
# 3. The new target for SSA is bk_rw (instead of bk_diff)

# Modification 1
cycle_new<-na.exclude(filter(x,bk_rw,side=1))
# Compare previous cycle with new cycle
par(mfrow=c(1,1))
ts.plot(cycle_diff,col="blue",main="Classic and new BK cycles")
lines(cycle_new,col="red")
mtext("Classic BK-cycle",col="blue",line=-1)
mtext("New BK-cycle",col="red",line=-2)
# We can scale the data for better visual comparisons
par(mfrow=c(1,1))
ts.plot(scale(cycle_diff,center=F,scale=T),col="blue",main="Classic and new BK cycles")
lines(scale(cycle_new,center=F,scale=T),col="red")
mtext("Classic BK-cycle",col="blue",line=-1)
mtext("New BK-cycle",col="red",line=-2)
abline(h=0)
# The new cycle, obtained when applying bk_rw to returns, is
#   a. centered at the zero-line
#   b. left-shifted 
#   c. and zero-crossings are more numerous
# By removing low-frequency content, the bandpass generates a shorter (spurious?) cycle
# Does not seem right! 
# If we were interested in a business-cycle tool then we could stop here: bandpass designs don't work.
# But out of curiosity we can apply SSA: we then need to apply the remaining two modifications.

# Modification 2: convolve xi and bk_rw (this filter would be applied to epsilont: it us used for determining the holding-time)
bk_conv<-conv_two_filt_func(xi,bk_rw)$conv
ht_bk_conv_obj<-compute_holding_time_func(bk_conv)
# The expected holding time is close to the empirical one (up to finite sample error)
ht_bk_conv_obj$ht 
# Empirical holding time 
compute_empirical_ht_func(cycle_new)

# Modification 3: our new target is bk_rw
gammak_generic<-bk_rw
ts.plot(gammak_generic)

# We are now in a position to plug SSA on BK
#   -We can work with stationary series thanks to our transformation from bk_rw to bk_diff
#   -We can work with the Wold decomposition given by xi
#--------------------------------------
# 6. Apply SSA: 
# The above holding-time was approximately 1 year
ht_bk_conv_obj$ht 
# We can lengthen by 50% 
ht<-1.5*ht_bk_conv_obj$ht
# This is the corresponding rho1 for the holding time constraint
rho1<-compute_rho_from_ht(ht)
# Forecast horizon: nowcast
forecast_horizon<-0
# We want to apply the filter to xt (not to epsilont): therefore we supply xi (Wold decomposition)

SSA_obj_bk_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# ssa_eps is the filter which is applied to epsilont
SSA_filt_bk_diff_eps<-SSA_obj_bk_diff$ssa_eps
# ssa_x is the filter which is applied to xt
SSA_filt_bk_diff_x<-SSA_obj_bk_diff$ssa_x

# Compare compare bk_rw with ssa_x: both filters are applied to xt 
mplot<-cbind(bk_rw,SSA_filt_bk_diff_x)
# Compare target and SSA
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"))
mtext("BK",col="black",line=-1)
mtext("SSA", col="blue",line=-2)

# We can also compare the new bk_conv with ssa_eps: both filters are applied to epsilont 
mplot<-cbind(bk_conv,SSA_filt_bk_diff_eps)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"))
mtext("BK",col="black",line=-1)
mtext("SSA", col="blue",line=-2)



# In order to check ht we must supply the 'other' filter, ssa_eps, which is (or could be) applied to epsilont
ht_obj<-compute_holding_time_func(SSA_filt_bk_diff_eps)
# Since ht of the optimized SSA matches the imposed ht we conclude that the optimization converged to the global maximum, as desired
ht_obj$ht 
ht


#--------------------------------------------------
# Filter series
# Apply SSA
SSA_out<-filter(x,SSA_filt_bk_diff_x,side=1)
# Quite larger than targeted ht...
compute_empirical_ht_func(SSA_out)
# Apply BK: bk_rw (instead of bk_diff)
bk_out<-filter(x,bk_rw,side=1)
compute_empirical_ht_func(cycle_new)
# In contrast to example 1 above, SSA is now smoother than BK


# SSA is smoother, as desired
mplot<-na.exclude(cbind(SSA_out,bk_out))
colo=c("blue","red")
ts.plot(mplot[,1],col=colo[1])
lines(mplot[,2],col=colo[2])
abline(h=0)
mtext("BK ",col="red",line=-1)
mtext("SSA", col="blue",line=-2)




#-----------------------------------------------
# We can also address timeliness or lead/lags: the last plot suggested that SSA is slightly right shifted (lagging)
# We augment the forecast horizon (delta in the JBCY paper) to obtain a faster SSA filter without giving-up smoothness
forecast_horizon<-12
# SSA of BK-diff
SSA_obj_bk_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_bk_diff_x_forecast<-SSA_obj_bk_diff$ssa_x

# Filter series: apply bk_rw to data (not bk_diff)
SSA_out_forecast<-filter(x,SSA_filt_bk_diff_x_forecast,side=1)
bk_out<-filter(x,bk_rw,side=1)

# Plot series: 
par(mfrow=c(1,1))
# The SSA-forecast seems odd...
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast,bk_out))
colo=c("blue","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon,sep=""),col=colo[2],line=-2)
mtext("BK",col=colo[3],line=-3)
abline(h=0)


# Let's shift the forecast forward by one year to verify pertinence of the optimization
par(mfrow=c(1,1))
# Looks fine
mplot_shift<-na.exclude(cbind(SSA_out[(1+forecast_horizon):length(SSA_out_forecast)],SSA_out_forecast[1:(length(SSA_out)-forecast_horizon)],bk_out[(1+forecast_horizon):length(SSA_out_forecast)]))
colo=c("blue","darkgreen","red")
ts.plot(mplot_shift[,1],col=colo[1],ylim=c(min(mplot_shift),max(mplot_shift)),main="All series with forecast shifted forward by one year")
lines(mplot_shift[,2],col=colo[2])
lines(mplot_shift[,3],col=colo[3])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon,sep=""),col=colo[2],line=-2)
mtext("BK",col=colo[3],line=-3)
abline(h=0)


