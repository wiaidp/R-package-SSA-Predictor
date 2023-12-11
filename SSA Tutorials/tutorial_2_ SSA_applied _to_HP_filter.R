# Application of SSA to HP filter 
# Complete case-study illustrating all features of the current SSA 
# SSA: Simple Sign Accuracy
# `Simple' means: univariate design without formal timeliness approach (timeliness is addressed indirectly, by means of the forecast horizon)
# Simplifications
#   -We discard the singular pandemic data 
#   -some of our results assume log-returns of the data to be (nearly) stationary 
# These assumptions do not affect our findings when engrafting SSA onto HP  

# We emphasize a business-cycle perspective
#   -We discard the classic HP-gap filter (1-trend) because the resulting bandpass generates spurious cycles 
#       See example 7 further down, section 4 in JBCY paper and tutorial 5 (the latter two apply the filter to monthly INDPRO)
#   -Instead we here consider the HP-trend or lowpass filter as applied to differenced (stationary) data
#   -All examples in this tutorial rely on simulated stationary series 
#     -An application to the monthly industrial production index (INDPRO) is provided in tutorial_5 (replicates 
#       section 4, JBCY paper)

# We apply SSA to different targets 
# Note: we do not discuss pertinence of the targets; instead we want to illustrate how SSA can address and 
#       improve some of the important characteristics of a particular approach. The way this is done is `generic',i.e.,
#       independent of HP (see tutorials 3 and 4 for alternative specifications)
# Targets in this tutorial:
#   a.The one-sided MSE HP, assuming the data to be white noise, see example 1
#   b.The classic one-sided HP, assuming the data to be white noise, see example 2
#     (the classic HP-concurrent has some interesting properties and therefore it makes sense to consider the filter as applied to noise)
#   c.The classic one-sided HP assuming the true or estimated (stationary ARMA) model, see example 5
#   d.The one-sided MSE HP assuming the true or estimated (stationary ARMA) model, see example 5 (just change the target specification) 
#   e.The symmetric HP assuming the true or estimated (stationary ARMA) model, see example 6


# Main outcomes: 
#   1.HP-trend (applied to first differences of a non-stationary series) does not generate spurious cycles. 
#     In contrast, classic bandpass designs, such as  BK (see tutorial 4), CF or HP-gap (see tutorial 5) generate spurious cycles
#   2.HP-trend applied to returns is similar to the Hamilton filter, see tutorial 3
#       Both designs are lowpass; neither generate spurious cycles
#   3.The classic concurrent (one-sided) HP assumes a particular implicit (ARIMA(0,2,2)) model. 
#     Consequences: 
#       -In applications, typically, HP-concurrent is not an optimal (MSE) nowcast of the symmetric HP, see example 6
#       -The holding-time of HP-concurrent is rather small i.e. the filter is subject to noise-leakage 
#         (noisy zero crossings).
#   4.SSA can be applied to control and to tame the number of noisy crossings of the proposed targets
#       -We typically impose 50% larger ht or, equivalently, 33% less (noisy) crossings
#       -Our empirical results confirm pertinence of the approach
#   5.Besides nowcasts (delta=0) we also consider 12-steps ahead forecasts (one year for monthly data)
#       -SSA-forecasts adopt the same stringent holding-time constraint: 33% less noisy crossings than (one-sided) HP targets (in the long run)
#       -SSA-forecasts are left-shifted: they generally have a lead when referenced against (one-sided) HP targets
#       -SSA real-time (concurrent) designs can be smoother (stronger noise suppression) and leading 

#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions
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



#-------------------------------------------------------------------------------
# Example 1
# Signal extraction with HP and BK: replicate example in section 2.2 of JBCY paper

# 1.1 Compute HP and compare concurrent MSE and SSA designs
# Filter length: should be an odd number since otherwise the (truncated bi-infinite) HP filter cannot be symmetric
L<-201
# Should be an odd number: otherwise HP is not symmetric
if (L/2==as.integer(L/2))
{
  print("Filter length should be an odd number")
  print("If L is even then HP cannot be symmetric")
  L<-L+1
}  
# HP monthly design
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)
# Bi-infinite HP
hp_target<-HP_obj$target
ts.plot(hp_target)
# Concurrent gap: as applied to series in levels: this is a high pass filter
hp_gap=HP_obj$hp_gap
ts.plot(hp_gap)
# Concurrent gap: as applied to series in differences (this is a band pass filter)
modified_hp_gap=HP_obj$modified_hp_gap
ts.plot(modified_hp_gap)
# Concurrent HP assuming I(2)-process 
# This is the Classic concurrent or one-sided low pass HP, see e.g. McElroy (2006)
hp_trend=HP_obj$hp_trend
ts.plot(hp_trend)
# Concurrent MSE estimate of bi-infinite HP assuming white noise
# This is just the truncate right tail of the symmetric filter
# This one is optimal is the data is white noise
# The previous one (classic concurrent HP above) is optimal if the data is an ARIMA(0,2,2)
hp_mse=hp_mse_example7=HP_obj$hp_mse
ts.plot(hp_mse)

# Compute lag one acf and holding time of HP concurrent
htrho_obj<-compute_holding_time_func(hp_trend)
rho_hp<-htrho_obj$rho_ff1
ht_hp<-htrho_obj$ht
# This is the holding-time of the classic concurrent HP, see tutorial 5 and JBCY paper
# This number matches the empirical holding-time if the data is white noise
# Log-returns of INDPRO in section 4 of JBCY paper are not white noise: therefore ht is biased and the empirical 
#  holding-time is much larger, see tutorial 5
ht_hp

# Same but for hp_mse
htrho_obj<-compute_holding_time_func(hp_mse)
rho_hp<-htrho_obj$rho_ff1
ht_mse<-htrho_obj$ht
# MSE filter is smoother than classic HP concurrent (larger ht)
ht_mse



#-----------------------------------------------------------------------------------
# 1.2. SSA and hyperparameters
# Holding time: we want SSA to lessen the number of zero-crossings when compared to HP 
ht_mse
# Therefore we select a ht which is larger than the above number
ht<-16
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# Our selection here means that SSA will have approximately 33% less crossings:
ht/ht_mse
# Forecast horizon: nowcast i.e. delta=0
forecast_horizon<-0
# We assume the data to be white noise which is the default setting (xi=NULL)
xi<-NULL
# Target: we supply the MSE concurrent filter which is in accordance with the white noise assumption
# Note: we could supply the classic concurrent HP instead (assuming an ARIMA(0,2,2)), see example 8 below
gammak_generic<-hp_mse


# SSA of HP-target
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)
# This is the same as the simpler call (omitting xi in the function call assumes xi=NULL, xt=epsilont white noise)
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1)


# Since xt is white noise (xi=NULL) the two SSA filters ssa_eps and ssa_x are identical (deconvolution in section 2.3 of JBCY paper is an identity)
ssa_x<-SSA_obj_HP$ssa_x
SSA_filt_HP<-SSA_example1<-ssa_eps<-SSA_obj_HP$ssa_eps



#----------------------
# 1.3 Plot
colo<-c("black","brown","blue")
par(mfrow=c(1,2))
mplot<-cbind(hp_target,hp_mse)
colnames(mplot)<-c("Symmetric","Concurrent")

plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=2)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

mplot<-cbind(SSA_filt_HP,hp_mse)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP")

plot(mplot[,1],main=paste("Concurrent",sep=""),axes=F,type="l",xlab="",ylab="",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[3],lwd=2)
mtext(colnames(mplot)[1],col=colo[3],line=-1)
lines(mplot[,2],col=colo[2],lwd=2)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


#------------------------
# 1.4 Checks

len<-100000
set.seed(16)
# White noise
a1<-0
x<-arima.sim(n = len, list(ar = a1))
# Compute filter output of SSA-HP filter
yhat<-filter(x,SSA_filt_HP,side=1)

# 1.4.1. Compare empirical and expected holding-times
# Compute empirical holding-time
empirical_ht<-compute_empirical_ht_func(yhat)
empirical_ht
#  compare with imposed constraint: matches up to sampling error
ht

# 1.4.2. Compare lag-one acf of optimized design with imposed constraint: successful optimization means that both numbers should be close
#  If there is a substantial difference: increase split_grid (number of iterations): default 20 should match most applications
# In our example both numbers match perfectly
SSA_obj_HP$crit_rhoyy
rho1

# 1.4.3. Criterion values: we here check the correlation with the MSE nowcast which is our effective target
# Correlations with the symmetric target are obtained in examples 10 and 12 below 
#   -crit_rhoyz is computed in the SSA-function: it is the true (expected) correlation of SSA with target 
#     if all assumptions are met (correct model for xt)
crit_example1<-SSA_obj_HP$crit_rhoyz
crit_example1
# We now compute the corresponding empirical correlation: 
# First derive the MSE nowcast filter output
MSE_nowcast<-filter(x,hp_mse,side=1)
# Compute empirical correlation and compare with true or expected number crit_example1
cor(yhat,MSE_nowcast,use='pairwise.complete.obs')


##############################################################################################################
###############################################################################################################
# Example 2
# Same as example 1 except that our target is the classic concurrent HP (instead of MSE)

# 2.1 Compute HP, see example 1.1 above 
#-----------------------------------------------------------------------------------
# 2.2. SSA and hyperparameters
ht_hp
# Holding time: we want SSA to lessen the number of zero-crossings when compared to HP
ht<-12
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# Our selection here means that SSA will have approximately 33% less crossings:
ht/ht_hp
# Forecast horizon: nowcast i.e. delta=0
forecast_horizon<-0
# Target: in contrast to example 7 above we here supply the classic HP-concurrent
gammak_generic<-hp_trend
# We assume the data to be white noise which is the default setting (xi=NULL)
xi<-NULL

# SSA of HP-target
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# Since xt is white noise the two SSA filters ssa_eps and ssa_x are identical (deconvolution is an identity)
ssa_x<-SSA_obj_HP$ssa_x
SSA_filt_HP<-ssa_eps<-SSA_obj_HP$ssa_eps

# Plot and compare filters: 
mplot<-cbind(ssa_x,hp_trend)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
colo<-c("blue","green")

# The typical noise-shape or tip of SSA is due to an implicit boundary constraint which states that the coefficient at lag -1 vanishes, see theorem 1 in JBCY paper
par(mfrow=c(1,1))
plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

#--------------------------------------------
# 2.3 Filter series and compare classic concurrent HP with SSA

len<-100000
set.seed(1)
# White noise
a1<-0
x<-arima.sim(n = len, list(ar = a1))
# Compute filter output of SSA-HP filter
yhat<-filter(x,SSA_filt_HP,side=1)
# Compare expected and empirical holding-times
ht
compute_empirical_ht_func(yhat)

# Compute concurrent HP-mse
HP_concurrent<-filter(x,hp_trend,side=1)
# Compare expected and empirical holding-times
ht_hp
compute_empirical_ht_func(HP_concurrent)

# Plot both series: we re-scale both series to unit variance for better comparison (note that SSA-scaling is arbitrary)
mplot<-na.exclude(cbind(yhat,HP_concurrent))
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# Plot a short sample of the series
par(mfrow=c(1,1))
anf<-500
enf<-1000
anf<-1000
enf<-1500

# SSA generates ~33% less crossings
# The additional crossings of HP are typically clustered at time points where the filter output seems to hover at the zero line
# There, the SSA filter maintains a better control of noisy crossings (see amplitude function below for a formal explanation of these properties)
# At up- and downswings, away from the zero line, SSA tracks the target well, due to optimality (SSA optimization principle)
ts.plot(mplot[anf:enf,],col=colo)
mtext(paste("SSA(",ht,",",forecast_horizon,")",sep=""),col=colo[1],line=-1)
mtext("HP-concurrent",col=colo[2],line=-2)
abline(h=0)

#------------------------------------------
# 2.4 Look at shift
# The above plot also suggests that the concurrent HP is slightly left-shifted (small lead): see also phase-lag plot below for formal background

# Here we compute the shift at zero-crossings
# The slight asymmetry suggests that SSA is slightly lagging HP
shift_tau_obj<-compute_min_tau_func(mplot)

#-------------------------------------------
# 2.5 Look at amplitude and phase-shift
K<-600
amp_obj_SSA<-amp_shift_func(K,as.vector(SSA_filt_HP),F)
amp_obj_HP<-amp_shift_func(K,hp_trend,F)

par(mfrow=c(2,1))
mplot<-scale(cbind(amp_obj_SSA$amp,amp_obj_HP$amp),scale=T,center=F)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")

# The HP filter is a lowpass applied to differences (returns)
# The amplitude function of SSA is much closer to 0 in stop band: stronger smoothing and less noisy zero-crossings
# This typical property of SSA-smoothing designs has coined the term 'noisy crossings' 
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
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

mplot<-cbind(amp_obj_SSA$shift,amp_obj_HP$shift)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# The larger phase-lag of SSA implies a slight lag relative to HP-concurrent: phase-shift is a bit larger 
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


# Findings: SSA lags slightly behind HP-concurrent (by approximately half a time-unit) and it generates ~33% less noisy crossings 

###########################################################################################################
############################################################################################################
# Example 3: 
# We can address lags (time shifts) by selecting a different forecast_horizon (delta in JBCY paper) 
#   in the above example 2, see sections 3.2 and 4.2 in JBCY paper
# We here look specifically at a one-year forecast
# Note, however, that we keep the holding-time fixed: 
#   -The resulting SSA-filter will retain smoothness (noise suppression) 
#   -But it will be slightly faster (leading instead of lagging as in example 2)
#   -Classic MSE approaches generally trade lead for smoothness: this is not the case here
# SSA here keeps smoothness fixed and trades lead against MSE-performances: see tutorial 0.1 on trilemma

# 3.1. SSA and hyperparameters
# Forecast horizon: one-year forecast
forecast_horizon<-12
# Holding time: see example 2
ht<-12
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# Our selection here means that SSA will have approximately 33% less crossings:
ht/ht_hp
# Target: like example 2 (and in contrast to example 1) we here supply the classic HP-concurrent
gammak_generic<-hp_trend
# We assume the data to be white noise 
# SSA of HP-target
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1)

# Since xt is white noise the two SSA filters ssa_eps and ssa_x are identical (deconvolution is an identity)
ssa_x<-SSA_obj_HP$ssa_x
SSA_filt_HP<-ssa_eps<-SSA_obj_HP$ssa_eps

# Plot and compare filters: 
mplot<-cbind(ssa_x,hp_trend)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
colo<-c("blue","green")

# Note the different scales of SSA and MSE
#   -As noted above, asking for a lead as well as for improved smoothness will affect MSE-performances
#   -The smaller scale of SSA results from this effect: SSA is shrunken towards zero
#   -One could re-scale and inflate the SSA-filter but MSE performances would be even worse (we here see the optimal scaling)
# Note also that zero-crossings or correlations (SSA-criterion, lag-one ACF in ht-constraint) are indifferent to scalings
par(mfrow=c(1,1))
# The typical noise-shape or tip of SSA is due to an implicit boundary constraint which states that the coefficient at lag -1 vanishes, see theorem 1 in JBCY paper
plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

#--------------------------------------------
# 3.2 Filter series and compare classic concurrent HP with SSA

len<-100000
set.seed(1)
# White noise
a1<-0
x<-arima.sim(n = len, list(ar = a1))
# Compute filter output of SSA-HP filter
yhat<-filter(x,SSA_filt_HP,side=1)
# Compare expected and empirical holding-times
ht
compute_empirical_ht_func(yhat)

# Compute concurrent HP (classic HP nowcast)
HP_concurrent<-filter(x,hp_trend,side=1)
# Compare expected and empirical holding-times
ht_hp
compute_empirical_ht_func(HP_concurrent)

# Plot both series: we re-scale both series to unit variance for easier visual comparison 
mplot<-na.exclude(scale(cbind(yhat,HP_concurrent),scale=T,center=F))
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# Plot a short sample of the series
par(mfrow=c(1,1))
anf<-500
enf<-1000
anf<-1000
enf<-1500

# SSA generates ~33% less crossings
# The additional crossings of HP are typically clustered at time points where the filter output seems to hover at the zero line
# There, the SSA filter maintains a better control of noisy crossings
# At up- and downswings, away from the zero line, SSA tracks the target well, due to optimality (SSA optimization principle)
# In contrast to example 8 above, SSA now seems to lead HP-concurrent
# This lead is to some extent remarkable because the classic HP-concurrent filter is known for being 'pretty fast'
ts.plot(mplot[anf:enf,],col=colo)
mtext(paste("SSA(",ht,",",forecast_horizon,")",sep=""),col=colo[1],line=-1)
mtext("HP-concurrent",col=colo[2],line=-2)
abline(h=0)

#------------------------------------------
# 3.3 Look at shift
# We can see a lead of SSA-forecast: the minimum of the curve is left-shifted 

shift_tau_obj<-compute_min_tau_func(mplot)

#-------------------------------------------
# 3.4 Look at amplitude and phase-shift
K<-600
amp_obj_SSA<-amp_shift_func(K,as.vector(SSA_filt_HP),F)
amp_obj_HP<-amp_shift_func(K,hp_trend,F)

par(mfrow=c(2,1))
mplot<-scale(cbind(amp_obj_SSA$amp,amp_obj_HP$amp),scale=T,center=F)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# Once again, the amplitude function of SSA is closer to zero in stop band: stronger noise suppression, less noisy crossings
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
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
# Phase-shift of SSA is now smaller than HP-concurrent over most of the passband
mplot<-cbind(amp_obj_SSA$shift,amp_obj_HP$shift)
# Set a floor to large (absolute) numbers for easier visual inspection/comparison
mplot[which(mplot[,1]<(-5)),1]<--5
mplot[1,1]<-0
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")

plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase shift",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
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


########################################################################################################
##########################################################################################################
# Example 4
# Play with target and forecast horizon (=delta in JBCY paper)
# We rely on example 1 but we specify the symmetric two-sided HP filter as our target
#   Since hp_target is a causal one-sided filter (instead of a two-sided) we'll have to shift the target 
#   by adjusting the forecast horizon suitably

# 4.1 Compute HP and compare concurrent MSE and SSA designs
# Filter length: must be twice (L=401) of example 7 (L=201) since otherwise the left tail of the symmetric filter will not match one-sided MSE in example 7
L_sym<-401
# Should be an odd number: otherwise HP is not symmetric
if (L_sym/2==as.integer(L_sym/2))
{
  print("Filter length should be an odd number")
  print("If L_sym is even then HP cannot be symmetric")
  L_sym<-L_sym+1
}  
# HP monthly design
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L_sym,lambda_monthly)
# Bi-infinite HP: this is our new target (instead of one-sided MSE in example 7)
hp_target=HP_obj$target
# The filter is causal
ts.plot(hp_target)
#-----------------------------------------------------------------------------------
# 4.2. SSA and hyperparameters
# Holding time: we use the same as in example 1
ht<-16
# Recall that we provide the lag-one acf to SSA_func below: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# We assume the data to be white noise which is the default setting in the SSA function-call (xi=NULL)
xi<-NULL
# Target: we supply the symmetric filter
gammak_generic<-hp_target
# Forecast horizon: the symmetric filter is non-causal
# The center is at (L+1)/2: therefore our new  forecast horizon is (L-1)/2 
forecast_horizon<-(L_sym-1)/2
# We use the same filter length as in example 1
L<-201
# SSA of HP-target: the function call here is shorter because we use the default settings of the missing parameters
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1)

SSA_filt_HP<-SSA_example4<-SSA_obj_HP$ssa_x

#----------------------
# 4.3 Plot: the filters in examples 1 and 4 are identical: 
#   -This is because hp_mse in example 1 is the MSE predictor of hp_target here, at least if xt=epsilont (white noise)
#   -Proposition 4 in the JBCY paper then implies that the SSA-solution must be the same
colo<-c("black","brown","blue")
par(mfrow=c(1,1))
mplot<-cbind(SSA_example1,SSA_example4)
colnames(mplot)<-c("Example 1","Example 4")

plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

#------------------------------
# 4.4 Criterion values
# SSA computes two different criterion values
# The first one is the correlation of SSA with (one-sided) MSE-filter : it is the same as in example 1
SSA_obj_HP$crit_rhoyz
crit_example1
# The second one is the correlation of SSA with the effective target (in our case the symmetric filter) 
#  The correlation is smaller because the target is the output of a non-causal (symmetric) filter
SSA_obj_HP$crit_rhoy_target

# We now compute the empirical correlations
# Generate very long series in order to obtain accurate empirical estimates
len<-1000000
set.seed(14)
x<-arima.sim(n = len, list(ar = a1))
# Compute filter output of SSA-HP filter
yhat<-filter(x,SSA_filt_HP,side=1)
# Compute symmetric HP output: we now have to set side=2 (symmetry)
HP_symmetric<-filter(x,hp_target,side=2)
# Compute MSE HP output: we have to set side=1 (one-sided)
HP_concurrent<-filter(x,hp_mse,side=1)

# Compute empirical correlations
mplot<-na.exclude(scale(cbind(yhat,HP_concurrent,HP_symmetric),scale=T,center=F))
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP-MSE","HP-symmetric (effective target")
cor(mplot)
# The empirical correlations (see first row in above correlation matrix) match crit_rhoyz and 
#   crit_rhoy_target
# SSA maximizes the theoretical criterion values which correspond to empirical correlations 
# This result justifies the optimality concept of SSA 
# Note that maximizing the correlation (between estimate and target) is equivalent to MSE, up to an arbitrary scaling
# SSA computes the optimal scaling constant: any alternative scaling would worsen MSE-performances of SSA

######################################################################################################################
######################################################################################################################
# Example 5
# Working with autocorrelated data (instead of noise)
# We here engraft SSA onto the classic HP-concurrent and apply the filter(s) to autocorrelated data
# 5.1 Generate data 
set.seed(4)
len<-1200
a1<-0.3
b1<-0.2
# Generate series
x<-arima.sim(n = len, list(ar = a1, ma = b1))
# Estimate coefficients
estim_obj<-arima(x,order=c(1,0,1))
# Check diagnostics: OK
tsdiag(estim_obj)
# Filter data: apply HP-concurrent
y_hp<-na.exclude(filter(x,hp_trend,side=1))
ts.plot(y_hp)

#-----------------
# 5.2 Holding time: we use the classic HP-concurrent (hp_trend)
ts.plot(hp_trend)
ht_hp_trend_obj<-compute_holding_time_func(hp_trend)
ht_hp<-ht_hp_trend_obj$ht 
ht_hp
# Compare with empirical holding time: the latter is larger
compute_empirical_ht_func(y_hp)
# This is because xt is not white noise!
# However, the computation of the holding-time assumes white noise
# Therefore, we have to decompose xt in a white noise sequence: Wold decomposition, see section 2 in JBCY paper
# Once decomposed, the holding-time is calculated properly

# Step 1: Compute the MA-inversion of the ARMA (Wold-decomposition or MA-inversion)
# One can insert true or estimated parameters 
xi_data<-c(1,ARMAtoMA(ar=a1,ma=b1,lag.max=L-1))
# Have a look at xi: if xi has not decayed to zero then one should increase the filter-length L
# Note:
#   -L should be sufficiently large so that finite-length convolutions and deconvolutions are good proxies (of infinite length transformations), see section 2 in JBCY paper
#   -For integrated processes xt, the MA-inversion xi does not decay to zero: this case will be analyzed in tutorials 3 (Hamilton filter), 4 (Baxter King filter) and 5 (HP-gap filter)
par(mfrow=c(1,1))
ts.plot(xi_data)
# Step 2: Convolve xi_data and hp, see section 2 of JBCY paper: 
#   The resulting convolved filter is applied to epsilont (innovation in Wold decomposition)
#   Therefore the expected holding-time (of the convolved filter) is 
hp_conv<-conv_two_filt_func(xi_data,hp_trend)$conv
ht_hp_conv_obj<-compute_holding_time_func(hp_conv)
ht_hp_conv_obj$ht
# Now the expected holding time (above) matches the empirical one (below), at least up to finite sample error
compute_empirical_ht_func(y_hp)
#------------------
# 5.3 SSA and hyperparameters
# Holding time: we want SSA to lessen the number of zero-crossings when compared to HP 
# We here increase ht by 50% when compared to HP
# Note: since the data xt is not white noise, we do not provide the original holding time of HP (ht_hp), 
#   which assumes xt=epsilont is white noise. Instead we here use the effective and corrected holding-time 
#   ht_hp_conv_obj$ht, see the previous discussion and analysis above 
ht<-1.5*ht_hp_conv_obj$ht
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# Our selection here means that SSA will have 33% less crossings:
ht/ht_hp_conv_obj$ht
# Forecast horizon: nowcast i.e. delta=0
forecast_horizon<-0
# Target: in contrast to example 1 above we here supply the classic HP-concurrent
gammak_generic<-hp_trend
# Autocorrelated data: we provide the Wold decomposition to the SSA function
xi<-xi_data

# SSA of HP-target: we must include xi in the function call (otherwise it is assumed that xt=epsilont is white noise, by default)
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# Since xt is not white noise, the two SSA filters ssa_eps and ssa_x are not the same: 
# ssa_x is the filter which is applied to xt
SSA_filt_HP<-ssa_x<-SSA_example5<-SSA_obj_HP$ssa_x
# ssa_eps is applied to epsilont: it is the convolution of ssa_x and Wold-decomposition xi_data
ssa_eps<-SSA_obj_HP$ssa_eps

# Plot and compare filters: 
mplot<-cbind(SSA_filt_HP,hp_trend)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP-concurrent")
colo<-c("blue","green")

# The typical noise-shape or tip of SSA is due to an implicit boundary constraint which states that the coefficient at lag -1 vanishes, see theorem 1 in JBCY paper
plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

#--------------------------------------------
# 5.4 Filter series and compare classic concurrent HP with SSA

len<-100000
set.seed(1)
x<-arima.sim(n = len, list(ar = a1,ma=b1))
# Compute filter output of SSA-HP filter
yhat<-filter(x,SSA_filt_HP,side=1)
# Compare expected and empirical holding-times: they match, as desired
ht
compute_empirical_ht_func(yhat)

# Compute concurrent HP (classic HP nowcast)
HP_concurrent<-filter(x,hp_trend,side=1)
# Compare expected and empirical holding-times: the empirical holding time of SSA is (approximately) 50% larger, 
#   as desired
ht_hp_conv_obj$ht
compute_empirical_ht_func(HP_concurrent)

# Plot both series: we re-scale both series to unit variance for better comparison (note that SSA-scaling is arbitrary)
mplot<-na.exclude(scale(cbind(yhat,HP_concurrent),scale=T,center=F))
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# Plot a short sample of the series
par(mfrow=c(1,1))
anf<-500
enf<-1000
anf<-1000
enf<-1500

# SSA generates ~50% less crossings
# The additional crossings of HP are typically clustered at time points where the filter output seems to hover at the zero line
ts.plot(mplot[anf:enf,],col=colo)
mtext(paste("SSA(",ht,",",forecast_horizon,")",sep=""),col=colo[1],line=-1)
mtext("HP-concurrent",col=colo[2],line=-2)
abline(h=0)

# We could address timeliness (lead/lags) as in example 4 above but we leave this as an exercise 


#------------------------------------------------------------------
# Example 6
# Same as example 5 but we now target specifically the output of the symmetric HP-filter
# As a result, we will obtain a one-sided filter which tracks the output of the symmetric filter better than HP-concurrent or HP-MSE above (the latter is assuming white noise)
# Recall that the classic HP-concurrent is optimal if the series is an ARIMA(0,2,2), where the two MA-parameters are determined by lambda
# This example is a bit tricky because everything is merged and entangled: most complex design
#   Besides 'true' optimal designs, We also present some 'bad good ideas', illustrating that intuition can conflict with logic

# Let us emphasize that:
# -This exercise does not refer to the majority of applications with HP: 
#   -Most practitioners rely on the classic concurrent HP to approximate the two-sided filter
#   -We addressed that problem in the previous exercise 5 by targeting the one-sided HP
# -This exercise addresses specifically an optimal one-sided filter for the two-sided HP, assuming the true 
#   (or the empirical) model of the data (which is not an ARIMA(0,2,2))
# -This exercise is meant for analysts mainly interested in tracking the symmetric filter by an optimal concurrent design
# -Besides the optimal MSE we also present SSA extensions which are smoother or/and faster 

# 6.1 Specify symmetric filter as target
L_sym<-401
# Should be an odd number: otherwise HP is not symmetric
if (L_sym/2==as.integer(L_sym/2))
{
  print("Filter length should be an odd number")
  print("If L_sym is even then HP cannot be symmetric")
  L_sym<-L_sym+1
}  
# HP monthly design
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L_sym,lambda_monthly)
# Symmetric HP: this is not yet our new target, because this filter is one-sided causal
# But we want the two-sided acausal as target for SSA (instead of one-sided MSE in example 1)
hp_target=HP_obj$target
ts.plot(hp_target)

# Forecast horizon: here things become a bit tricky
# The variable h is now our effective forecast horizon
# For a nowcast of the symmetric filter we set h<-0 (h>0 for forecast, h<0 for backcast)
h<-0
# Here is the tricky part:
# The symmetric filter hp_target is causal: its center is at lag (L_sym+1)/2, see brown line in plot below
# For a nowcast of the two-sided acausal filter we have to shift the causal filter to the left, see violet line
#   The peak of the violet line is now shifted to lag h=0 (nowcast)
# Therefore our effective  forecast horizon is (L_sym-1)/2+h 
forecast_horizon<-(L_sym-1)/2+h
causal_sym<-c(rep(0 ,(L_sym-1)/2 ),hp_target)
acausal_sym<-c(rep(0,max(0,(L_sym+1)/2-forecast_horizon-1)),hp_target[max(1,1+(forecast_horizon-(L_sym-1)/2)):L_sym],rep(0 ,forecast_horizon ))
acausal_sym<-acausal_sym[1:length(causal_sym)]
plot(causal_sym,col="brown",axes=F,type="l",xlab="lead                                                                  lag",ylab="",main="Acausal vs. causal HP: the target is acausal")
lines(acausal_sym,col="violet")
abline(v=(L_sym+1)/2,col="violet")
abline(v=L_sym,col="brown")
mtext(paste("Target acausal filter: center is at lag 0. For SSA we shift the causal HP (brown line) to the left by forecast_horizon=",forecast_horizon,sep=""), line=-1,col="violet")
mtext(paste("                                                   Causal symmetric HP as calculated by mFilter: center is at lag ",(L_sym+1)/2,sep=""), line=-3,col="brown")
axis(1,at=1:length(acausal_sym),labels=-forecast_horizon +1:length(acausal_sym))
axis(2)
box()

# Since the target is the symmetric filter, the natural benchmark for SSA is the MSE estimate of the target 
#  (this is also computed by our SSA function, see below. We here replicate these calculations, for illustration)
# Proceeding for obtaining the MSE-filter (recall that hp_mse is not optimal here because xt is not white noise):
#   a. Compute the convolution of symmetric filter hp_trend and Wold-decomposition xi: this way, HP is applied to epsilont 
#   b. Truncate the convolution at lag forecast_horizon+1 (which is lag 0 in un-shifted data) because forecasts of future epsilont are zero
# To a: convolve xi_data and symmetric hp_target, see section 2 of JBCY paper: 
hp_conv_mse_d<-conv_two_filt_func(xi_data,hp_target)$conv
# To b: truncate at lag 0 corresponding to forecast_horizon
hp_conv_mse<-hp_conv_mse_d[(forecast_horizon+1):L_sym]
# We can now compute the holding-time of the optimal MSE filter: this is a natural benchmark for SSA 
# -Idea: SSA should improve smoothness of best (MSE) nowcast
# -Of course, the holding-time could be set differently (based on a priori knowledge or on particular priorities). 
# -But having a natural (optimal) benchmark at disposal allows more stringent comparisons
ht_hp_conv_mse_obj<-compute_holding_time_func(hp_conv_mse)
ht_hp_conv_mse<-ht_hp_conv_mse_obj$ht
# ht of MSE: it is larger than ht of hp_mse above
ht_hp_conv_mse
#-----------------------------------------------
# 6.2 Apply SSA: 
# SSA: augment ht_hp_conv_mse of optimal MSE benchmark by 50% 
ht<-1.5*ht_hp_conv_mse
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# In contrast to example 5 we now specify the symmetric target filter
gammak_generic<-hp_target
# Forecast horizon was discussed above 
forecast_horizon<-forecast_horizon
# We use the same filter length as in example 5
L<-201
# In contrast to example 4, we now supply the Wold-decomposition (MA-inversion) of the data generating process
xi<-xi_data
# SSA of HP-target
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# SSA_func also computes the MSE estimate which we already computed above
#   This filter is supposed to be applied to epsilont, not xt (convolution of target with xi)
mse_eps<-SSA_obj_HP$mse_eps
ts.plot(cbind(hp_conv_mse[1:L],mse_eps),main="MSE estimate: SSA_func vs. own calculation (both filters overlap)")

# retrieve the SSA-filter applied to xt (we here ignore ssa_eps as applied to epsilont since we shall filter xt)
SSA_filt_HP<-SSA_obj_HP$ssa_x

# Plot and compare with SSA designs of previous examples 4 and 5
colo<-c("black","brown","blue")
par(mfrow=c(1,1))
mplot<-cbind(SSA_example4,SSA_example5,SSA_filt_HP)
colnames(mplot)<-c("Example 4","Example 5","Example 6")

# The filters of examples 4 and 6 look similar. Why is that?
#   -Example 4 relied on the symmetric target too, but it assumed xt=epsilont white noise (xi=NULL) and a smaller holding time ht=12
#   -Here, in example 6, we assume an ARMA-process and a larger holding-time of approximately 17
#   -As it happens (fortuitously) the solution of example 4 when applied to the ARMA-process has a holding-time very close to 17
#   -Therefore, both designs are pretty close to each other (up to some differences towards lag 0)
#   -More significant differences could be observed by changing the above specifications (the ARMA-process and/or the holding-time constraint)
# The scale of example 5 is quite different
#   -This is because the estimation problem of example 5 is much simpler since we target the one-sided filter hp_trend
#   -In contrast, examples 4 and 6 nowcast the acausal two-sided HP
#   -The smaller scaling (zero-shrinkage) in the latter examples reflects the increased uncertainty
plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
lines(mplot[,3],col=colo[3],lwd=2,lty=1)
mtext(colnames(mplot)[3],col=colo[3],line=-3)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

#-----------------------------------
# 6.3 Compute benchmark MSE filter as applied to xt: this is once again a bit tricky 
#   -mse_eps (or hp_conv_mse: both are identical) is the correct (optimal MSE) filters when applied to epsilont
#     -This is not hp_mse (the latter assumes xt=epsilont)
#   -we now compute the correct or optimal MSE filter when applied to xt 
# Two possibilities
# A. First possibility (direct computation): obtain the MSE-filter by deconvolution of mse_eps by xi
#   -Once again, this is not hp_mse, see plot below 
deconv_obj<-deconvolute_func(as.vector(mse_eps),xi)
hp_deconv_mse<-deconv_obj$dec_filt 

# The correct MSE assigns more weight to the first observation (lag 0) because the series xt is smoother than epsilont 
#   The SSA filter applied to xt is looking 'unsmoother' than hp_mse (the latter ebing optimal for white noise xt=epsilont)
ts.plot(scale(cbind(hp_mse[1:L],hp_deconv_mse)),col=c("darkgreen","green"),main="MSE vs. hp_mse")
mtext("Hp_mse: assumes that xt=epsilont is white noise",col="darkgreen",line=-1)
mtext("Effective MSE: obtained by deconvolution",col="green",line=-2)

# B. Second possibility
# Two steps:  1. replicate MSE in SSA by setting a corresponding holding-time constraint and 
#             2. retrieve the MSE filter (as applied to xt) from SSA_func
# Step 1: set holding time to replicate MSE (MSE as applied to epsilont)
ht_mse<-ht_hp_conv_mse
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1_mse<-compute_rho_from_ht(ht_mse)
# SSA replicating MSE
SSA_obj_HP_mse<-SSA_func(L,forecast_horizon,gammak_generic,rho1_mse,xi)

# Verify replication:
ts.plot(scale(cbind(SSA_obj_HP_mse$ssa_eps,mse_eps)),main="Replicate MSE as applied to epsilont: both filters overlap")

# Step 2. ssa_x is the optimal MSE (applied to xt)
HP_MSE_x<-SSA_obj_HP_mse$ssa_x

# Verify that HP_MSE_x and hp_deconv_mse are identical: we can use either one in our simulations below
ts.plot(scale(cbind(HP_MSE_x,hp_deconv_mse)),col=c("darkgreen","green"),main="MSE filters applied to xt: direct vs. SSA computation (both filters overlap)")
mtext("MSE based on SSA-function",col="blue",line=-1)
mtext("MSE obtained by direct deconvolution",col="green",line=-2)

# Now that the relevant filters have been sorted out and computed, we can  filter the series, compare 
#   performances and proceed to our checks of empirical and expected numbers (holding times, correlations)
#-------------------------------------------
# 6.4 Filter series 
# Generate very long series in order to obtain accurate empirical estimates
len<-1000000
set.seed(14)
x<-arima.sim(n = len, list(ar = a1,ma=b1))
# Compute filter output of SSA-HP filter: optimal as applied to xt
yhat<-filter(x,SSA_filt_HP,side=1)
# Compare expected and empirical holding-times: they match, as desired
ht
compute_empirical_ht_func(yhat)

# Compute filter output of correct MSE computed above 
HP_mse<-filter(x,HP_MSE_x,side=1)
# Compare expected and empirical holding-times: both match
ht_hp_conv_mse
compute_empirical_ht_func(HP_mse)
# The empirical holding time of SSA is (approximately) 50% larger, as desired

# Compute output of target, i.e., symmetric HP: we now have to set side=2 (symmetry)
HP_symmetric<-filter(x,hp_target,side=2)

# Out of curiosity we also compute the output of the classic HP-concurrent: side=1 since the filter is one-sided
HP_concurrent<-filter(x,hp_trend,side=1)
# The classic HP concurrent generates more crossings than SSA
compute_empirical_ht_func(HP_concurrent)
# Interestingly, as we shall see below, the classic HP concurrent looses also in terms of correlation with the target
#   -The underlying implicit ARIMA(0,2,2) assumption is a misspecification!

# Plot all filter outputs
colo<-c( "blue","green","black","brown" )
# Plot both series: 
mplot<-na.exclude(cbind(yhat,HP_mse,HP_symmetric,HP_concurrent))
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP MSE","HP-symmetric (effective target)","Classic HP concurrent")
# Plot a short sample of the series
par(mfrow=c(1,1))
anf<-500
enf<-1000
anf<-1000
enf<-1500
# SSA generates ~30% less crossings than MSE or HP concurrent
ts.plot(mplot[anf:enf,],col=colo)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
if (ncol(mplot)>1)
  for (i in 2:ncol(mplot))
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
abline(h=0)

# Discussion:
#   -The scale of the classic HP-concurrent is off the mark: the filter assumes xt to be an ARIMA(0,2,2)-process
#     -Severe misspecification
# One possibility to look at scalings (of filter outputs) is to look at the sum of filter coefficients
#   -The sum of filter coefficients corresponds to the amplitude function of a filter at frequency zero (eventually up to sign)  
#   -Therefore, the sum is informative about the scaling of low-frequency components by the filter
# Two-sided target: the coefficients sum to one
sum(hp_target)
# Classic concurrent: sum is one
sum(hp_trend)
# SSA: sum is markedly smaller than one
sum(SSA_filt_HP)
# hp_mse: sum is markedly smaller than one
sum(hp_mse)
# We infer that hp_trend matches hp_target, at least in terms of amplitude at frequency zero: 
#   -this is because the implicit model (ARMA(0,2,2)), justifying hp_trend, assumes the data to be strongly trending
#   -in order to track the two-sided filter (MSE-optimality), the one-sided hp_trend must let the dominating 
#     (low-frequency) trend pass without changing scales: the filter coefficients sum to one.
# However, if the data is not strongly trending, i.e. if the ARMA(0,2,2) is severely misspecified, then there is 
#     no necessity to track the (missing) trend: the filter coefficients do not have to add to one anymore
#   -In particular, in the presence of noise (strong high-frequency components), MSE-performances generally improve
#     by shrinking the predictor (zero-shrinkage): 
#       -the coefficients of SSA sum to ~0.5 (instead of one)
#       -hp_mse assumes the data to be white noise and its coefficients sum to ~0.5
# The `odd' scaling of hp_trend in the above plot is due to severe misspecification. 
#   -Part of this misspecification could be addressed by simply re-scaling the filter output (say by 0.5)
# Our (to some extent original) view on this problem, in the context of SSA, is: we do not look at scales
#   -We do not emphasize `correct' scalings because sign accuracy and holding-times focus on zero-crossings
#   -Therefore, the SSA-criterion emphasizes correlations and our SSA-function returns correlation measures
#     -Correlation of the predictor with the target
#     -Correlation of the predictor with the MSE-benchmark
#     -Lag-one ACF of the predictor: holding-time
#   -Obviously, we can also provide the optimal MSE-scaling, which is very easy to obtain, see JBCY paper
#     -However, in typical applications we generally scale all competing filter outputs to unit-variance in order
#       to facilitate visual inspection of the relevant aspects, namely: 
#       -shifts at crossings and at local peaks/troughs (lag/lead)
#       -smoothness. 
# In this context, it would be more consistent to look at performances of the classic concurrent HP in terms of 
#     correlations, too, and thus ignore the `odd' scaling (equivalently: re-scale all filter-outputs to unit-variance)
#   -We already looked at its lag-one ACF above (in terms of empirical holding-time)
#   -Let's then have a look at its correlation with the two-sided target hp_target

#------------------------------------------
# 6.5 Compute empirical and true (expected) performance numbers and compare all one-sided designs
# Compute empirical cross correlations of all designs: 
cor_mat<-cor(mplot)
cor_mat
# Let's first look at correlations of the various one-sided designs with the target: 
#  Numbers in the third row of the correlation matrix
cor_mat[3,]
# -MSE outperforms SSA (because MSE is best possible; but SSA is smoother: 50% larger empirical holding-times in above simulation):
cor_mat[3,2]
cor_mat[3,1]
# -Of course, MSE also outperforms the classic one-sided HP
cor_mat[3,2]
cor_mat[3,4]
# -Interestingly, SSA outperforms HP-concurrent both in terms of smoothness (fewer crossings) as well 
#   as in terms of correlation with target 
# As stated: the classic HP-concurrent filter is generally not optimal for applications to typical economic data
cor_mat[3,1]
cor_mat[3,4]

# Let's now look at empirical correlations of SSA with target and MSE-benchmark: 
#  Numbers in first row of the above correlation matrix
cor_mat[1,]
# -These numbers correspond to the criterion values of SSA: 
#   They should match crit_rhoy_target (correlation with target) and crit_rhoyz (correlation with MSE: see proposition 4 in JBCY paper)
# A. crit_rhoy_target
cor_mat[1,3]
SSA_obj_HP$crit_rhoy_target
# B. crit_rhoyz
cor_mat[1,2]
SSA_obj_HP$crit_rhoyz


#--------------------------------------------------------------
# Example 7
# This is more of a counter-example and it has nothing to do with SSA
# We here analyze the classic HP-gap filter
# Based on its characteristics we strongly advise against its use in BCA, in agreement with Hamilton: "never use the HP (gap)", see tutorial 3

# HP gap
# 7.1 Compute HP-gap: we use the same length as in example 6 above
L_sym<-401
# Should be an odd number: otherwise HP is not symmetric
if (L_sym/2==as.integer(L_sym/2))
{
  print("Filter length should be an odd number")
  print("If L_sym is even then HP cannot be symmetric")
  L_sym<-L_sym+1
}  
# HP monthly design
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L_sym,lambda_monthly)
# Bi-infinite HP: this is our new target (instead of one-sided MSE in example 1)
hp_target=HP_obj$target
ts.plot(hp_target)
# Gap
hp_gap<-HP_obj$hp_gap
ts.plot(hp_gap)
# Classic concurrent
hp_trend<-HP_obj$hp_trend
ts.plot(hp_trend)
# Gap is 1-hp_trend
# Difference vanishes
(c(1,rep(0,L_sym-1))-hp_trend)-hp_gap

#---------------------------------------------
# 7.2 Transformation: from levels to first differences
# The gap filter is applied to data in levels: this renders a direct analysis cumbersome and difficult
# Instead, we here derive a filter whose output is the same as HP-gap when applied to first differences: this
#  will ease our analysis
gap_diff<-conv_with_unitroot_func(hp_gap)$conv
par(mfrow=c(2,1))
# Note that the coefficients of the new filter ham_diff vanish for lags larger than length of Hamilton_filter_adjusted and therefore we could set L=length(hamilton_filter_adjusted)
ts.plot(gap_diff,main="HP-gap as applied to first differences")
ts.plot(hp_gap,main="HP-gap as applied to level")

# We now verify that the outputs of both filters are identical
set.seed(252)
len<-L+2000
# Generate random-walk: data in levels will be fed to hp_gap
y<-cumsum(rnorm(len))
# Difference data: this will be fed to gap_diff: lengthen series with a zero to match y
x<-c(0,diff(y))
len_diff<-length(x)
# Compute new cycle based on new filter ham_diff applied to returns
yhat_diff<-filter(x,gap_diff,side=1)
yhat_gap<-filter(y,hp_gap,side=1)

# Check: both series are identical
par(mfrow=c(1,1))
ts.plot(yhat_diff,col="blue")
lines(yhat_gap,col="red")
#-------------------------------------------
# 7.3 We can now compute the amplitude  of HP-gap or, better, gap_diff as applied to first differences
# First differences of typical economic data are generally close to white noise (Typical spectral shape by Granger)
# Therefore the squared amplitude of gap_diff is close to the spectral density of the filter output (convolution theorem)
#   This is the main reason for considering gap_diff: an interpretation of filter characteristics is much easier
# Select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude of gap_diff
amp_gap_diff<-amp_shift_func(K,as.vector(gap_diff),F)$amp

# Plot amplitude function
plot(amp_gap_diff,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP-gap as applied to differences",sep=""))
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# Plot squared amplitude function: this is the true spectral density of the HP-gap `cycle', when applied to a random-walk
# This is a good proxy of the true spectral density of the HP-gap `cycle' when applied to typical (non-stationary) economic data in levels
par(mfrow=c(1,1))
plot(amp_gap_diff^2,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Squared amplitude HP-gap as applied to differences",sep=""))
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# Outcomes:
#   1. HP-gap is a bandpass when applied to levels; but it is also a bandpass when applied to first differences (the above amplitude vanishes at zero)
#     The reason is that (the one-sided) HP assumes a second order unit-root: after first differences, gap_diff must still remove an additional unit-root
#   2. HP-gap generates a spurious cycle whose frequency corresponds to the peak of the above amplitude functions

# Let's briefly check our second claim
# This is the frequency of the peak-amplitude
omega_gap<-pi*(which(amp_gap_diff==max(amp_gap_diff))-1)/K
# This is its periodicity: approximately 6 years
# A periodicity of 6 years for a completed cycle is too short for longer expansions and therefore HP-gap will generate spurious alarms in midst of some expansions, see tutorial 5
2*pi/omega_gap
# The periodogram of the filter output confirms our findings: low and high frequencies are damped, 
#   thus generating a spurious cycle at the peak frequency
per_obj<-per(na.exclude(yhat_gap),T)
# The acf confirms these findings
acf(na.exclude(yhat_gap))
# Finally: here's the cyle
par(mfrow=c(1,1))
ts.plot(yhat_gap)
abline(h=0)
# There are lots of noisy zero-crossings, too, which would hamper a real-time assessment of the cycle, see tutorial 5
# As we shall see BK is subject to a similar shortcoming, see tutorial 4
# These findings confirm and reinforce Hamilton's statement: never use the HP (-gap) filter for BCA
# But we could add: use the HP-concurrent lowpass instead, as applied to differences, see tutorial 5