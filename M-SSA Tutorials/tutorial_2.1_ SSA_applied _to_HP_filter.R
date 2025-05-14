# In this tutorial we consider an application of SSA to the Hodrick-Prescott (HP) filter 
# Tutorial 2.0 provided background:
#   -For understanding HP; 
#   -For justifying and motivating some of our decisions when working with HP (which differ from classic applications of HP)
# In particular we prefer HP-trend applied to differenced (stationary) data to the original HP-gap (applied to levels)

# Accordingly, we here consider the HP-trend or lowpass filter, applied to stationary data, resembling differenced economic data/series
#   -This design is used in Wildi (2024) https://doi.org/10.1007/s41549-024-00097-5: HP-trend applied to log-returns of US INDPRO (does not generate spurious cycle)
#   -All examples in this tutorial rely on artificial (simulated) stationary series: knowing the true model allows for verification of theoretical results 
#     -Tutorial 5 applies HP and SSA to US-INDPRO
#     -Tutorial 7 presents an application of the more general multivariate SSA (M-SSA) to German macro-data
# We apply SSA to different targets 
#   a.The one-sided MSE HP (optimal if data is white noise), applied to white noise, see example 1
#   b.The classic one-sided HP (optimal if data is an ARIMA(0,2,2)), applied to white noise, too, see example 2
#   c.The classic one-sided HP, applied to autocorrelated data, see example 5
#   d.The one-sided MSE HP, applied to autocorrelated data, too, see example 5 (just change the target specification) 
#   e.The two-sided HP applied to autocorrelated data, see example 6

# Note: a look at tutorial 2.0 suggests that the two-sided target is less relevant in a BCA-context: 
#   -it is too smooth: recession dips are washed-out and may eventually vanish or merge
#   -the classic one-sided HP-trend filter is relevant (it has desirable frequency-domain and time-domain characteristics)


# Main outcomes: 
#   1.The classic HP-gap (as applied to non-stationary data in levels) is not suited for BCA, see example 7 below: "never use HP-gap". 
#       -Therefore, we here emphasize the trend filter(s) only: as applied to stationary data (first differences of economic time series) 
#   2.The classic concurrent (one-sided) HP-trend can be derived from a particular (implicit) ARIMA(0,2,2) model for the data, see tutorial 2.0.
#     -Typically, economic data does not conform to such a model, see tutorial 2.0
#     -Consequences: 
#       -In contrast to usual assumptions, HP-concurrent is not an optimal (MSE) nowcast of the two-sided trend, see example 6
#       -The holding-time of HP-concurrent is rather small i.e. the filter is subject to noise-leakage 
#         (noisy zero crossings, see also tutorial 2.0).
#   3.In this scenario, SSA can be applied to control and to tame the number of noisy crossings of HP
#       -We typically impose 50% larger ht or, equivalently, 33% less (noisy) crossings
#   4.Besides nowcasts (delta=0) we also consider 12-steps ahead forecasts (one year for monthly data)
#       -SSA-forecasts adopt the same stringent holding-time constraint: 33% less noisy crossings than (one-sided) HP targets (in the long run)
#       -SSA-forecasts are left-shifted (relative advancement): they generally have a lead when referenced against the concurrent benchmarks
#       -SSA real-time (concurrent) designs can be smoother as well as leading, when compared to the concurrent benchmarks 
#   5. The forecast trilemma (see tutorial 0.1) is visualized in example 8 for a SSA-design targeting HP-MSE

# Note: 
# -our intention is not to push a particular BCA-tool (HP filter). 
# -Rather, we strive at illustrating that a particular predictor or BCA-filter (any one as long 
#       as it's linear in the data) can be replicated and modified by SSA in view of addressing 
#   1. smoothness (noise suppression) and 
#   2. timeliness (advancement)
# -In this perspective, HP is considered as a basic platform and a vitrine for showcasing SSA
#   -We offer a number of compelling performance measures, confirming pertinence of a simple novel optimization principle  
# -Applications of SSA to Hamilton's filter (proposed as an alternative to HP) and to the Baxter-King filter 
#   are proposed in tutorials 3 and 4, respectively

#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions
rm(list=ls())

library(xts)
# Load the library mFilter
# HP and BK filters
library(mFilter)
# Plot for heat map of Trilemma
library(ggplot2)
library("gplots")

# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


##########################################################################################################
##########################################################################################################
# Introduction, see also tutorial 2.0
# a.Derivation of HP 
# b.Brief analysis of the classic one-sided HP concurrent trend filter
# c. Summary
#--------------------------
# a. Derivation
# We use the R-package mFilter for computing HP 
# Specify filter length: should be an odd number since otherwise the two-sided HP filter could not be adequately centered 
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
HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)
# Bi-infinite (here truncated) two-sided (symmetric) HP
hp_target<-HP_obj$target
ts.plot(hp_target,main=paste("HP(",lambda_monthly,") two-sided target",sep=""))
# Concurrent gap: as applied to series in levels: this is a high pass filter
hp_gap=HP_obj$hp_gap
ts.plot(hp_gap)
# Concurrent HP assuming I(2)-process 
# This is the Classic concurrent or one-sided low pass HP, see e.g. McElroy (2006)
hp_trend=HP_obj$hp_trend
ts.plot(hp_trend,main="One-sided HP")

# Compute lag one acf and holding time of HP concurrent
htrho_obj<-compute_holding_time_func(hp_trend)
rho_hp<-htrho_obj$rho_ff1
ht_hp<-htrho_obj$ht
ht_hp

# Compare holding-times (ht) of one- and two-sided filters
compute_holding_time_func(hp_target)$ht
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

# The one-sided filter (red) is much noisier, with much more zero-crossings (marked by vertical lines in plot)
ts.plot(y_hp_concurrent,main="HP: two-sided vs one-sided filter. Vertical lines indicate zero-crossings of one-sided design",col="red")
lines(y_hp_symmetric)
abline(h=0)
abline(v=which(y_hp_concurrent[2:len]*y_hp_concurrent[1:(len-1)]<0),col="red",lty=3)
mtext("Two-sided HP",col="black",line=-1)
mtext("One-sided HP",col="red",line=-2)

# Let us compute empirical holding times of both filters:
compute_empirical_ht_func(y_hp_concurrent)
compute_empirical_ht_func(y_hp_symmetric)
# The difference of empirical hts is large, as expected 
# Note: for very long samples, these estimates converge to the `true' hts
ht_hp
compute_holding_time_func(hp_target)$ht



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
plot(amp_obj$amp,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP",sep=""),ylim=c(0,max(amp_obj$amp)))
mtext("Amplitude classic concurrent HP trend",line=-1)
abline(h=0)
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
#   -Alternatively, we may claim: the model is overtly misspecified, see tutorial 2.0
# -The classic one-sided HP, on the other hand, is less smooth: therefore it can better track short but severe recession dips
#   -The peak amplitude matches business-cycle frequencies
#   -The vanishing time-shift means that the filter is a tough benchmark
#     -The filter is typically faster than Hamilton's regression filter in real-time applications, see tutorial 3
# -We therefore propose to target two-sided  (examples 4 and 6)  as well as one-sided designs by SSA (examples 1,2,3,5 and 8) 
# -Example 1 addresses specifically the one-sided hp_mse, assuming the data to be white noise
#   -Example 4 illustrates that this particular target is equivalent to the two-sided HP when the data is white noise, thus confirming proposition 5 in the JBCY paper


######################################################################################################################
######################################################################################################################
# Example 1: 
# Target HP-MSE


# 1.1 Concurrent MSE estimate of bi-infinite HP assuming white noise
# This is just the truncate right tail of the symmetric filter
# This one is an optimal MSE estimate of the two-sided filter if the data is white noise
hp_mse=hp_mse_example7=HP_obj$hp_mse
par(mfrow=c(1,1))
ts.plot(hp_mse)
# Compute lag-one acf and ht for hp_mse
htrho_obj<-compute_holding_time_func(hp_mse)
rho_hp<-htrho_obj$rho_ff1
ht_mse<-htrho_obj$ht
# MSE filter is smoother than classic HP concurrent (larger ht) because white noise is `noisier' than ARIMA(0,2,2)
#   As a result, hp_mse must damp high-frequency components more strongly than hp_trend
ht_mse

#-----------------------------------------------------------------------------------
# 1.2. Setting-up SSA
# Holding time: we typically want SSA to lessen the number of zero-crossings when compared to hp_mse 
ht_mse
# Therefore we select a ht which is larger than the above number
ht<-1.5*ht_mse
# Recall that we provide the lag-one acf: therefore we have to compute rho1 (corresponding to ht) for SSA
rho1<-compute_rho_from_ht(ht)
# Our selection here means that SSA will have 33% less crossings on average:
ht/ht_mse
# Forecast horizon: nowcast i.e. delta=0
forecast_horizon<-0
# We assume the data to be white noise which is the default setting (xi=NULL)
xi<-NULL
# Target: we supply the MSE concurrent filter which is in accordance with the white noise assumption
# Note: we could supply the classic concurrent HP instead (assuming an ARIMA(0,2,2)), see example 2 below
gammak_generic<-hp_mse


# SSA of HP-target
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)
# This is the same as the simpler call (omitting xi in the function call assumes xi=NULL, xt=epsilont white noise)
SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1)


# Since xt is white noise (xi=NULL) the two SSA filters ssa_eps and ssa_x are identical, see tutorial 1
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
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP")

plot(mplot[,1],main=paste("Concurrent",sep=""),axes=F,type="l",xlab="",ylab="",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[3],lwd=2)
mtext(colnames(mplot)[1],col=colo[3],line=-1)
lines(mplot[,2],col=colo[2],lwd=2)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


#------------------------
# 1.4 Checks: we check convergence of sample estimates to expectations
# -Thereby we check that the SSA optimization principle is easily interpretable and practically relevant 

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
# -compare with imposed constraint: both numbers match up to sampling error
# -SSA controls the ht, as claimed
ht

# 1.4.2. Compare lag-one acf of optimized design with imposed constraint: successful optimization means that both numbers should be close
# In our example both numbers match perfectly: the optimization converged to the global optimum 
SSA_obj_HP$crit_rhoyy
rho1

# 1.4.3. Criterion values: we here check the correlation of SSA with the MSE nowcast which is our effective target
#   -Correlations with the two-sided target are obtained in examples 4 and 6 below 
#   -crit_rhoyz is computed in the SSA-function: it is the true (expected) correlation of SSA with the target 
#     if all assumptions are met (correct model for xt)
crit_example1<-SSA_obj_HP$crit_rhoyz
crit_example1
# We now compute the corresponding sample correlation in two steps: 
# a. First derive the MSE nowcast filter output
MSE_nowcast<-filter(x,hp_mse,side=1)
# b. Second, compute the sample correlation and compare with true or expected number crit_example1
cor(yhat,MSE_nowcast,use='pairwise.complete.obs')
# Both numbers match: the sample correlation converges to the criterion value with increasing sample length

# Summary:
# -We verified that SSA maximizes the correlation of the predictor with the target, subject to the ht constraint


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
# Target: in contrast to example 1 above we here supply the classic HP-concurrent
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

# Compute concurrent HP
HP_concurrent<-filter(x,hp_trend,side=1)
# Compare expected and empirical holding-times
ht_hp
compute_empirical_ht_func(HP_concurrent)

# As expected, SSA generates ~33% less zero-crossings

# Plot both series: we re-scale both series to unit variance for better comparison (note that SSA-scaling is arbitrary)
mplot<-na.exclude(cbind(yhat,HP_concurrent))
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# Plot a short sample of the series
par(mfrow=c(1,1))
anf<-500
enf<-1000
anf<-1000
enf<-1500

# Both series look similar but SSA generates ~33% less crossings
# The additional crossings of HP are typically clustered at time points where the filter output hovers over the zero line
# There, the SSA filter maintains a better control of noisy crossings (see amplitude function below for a formal explanation of these properties)
# At up- and downswings, away from the zero line, SSA tracks the target well, due to optimality (SSA optimization principle)
ts.plot(mplot[anf:enf,],col=colo)
mtext(paste("SSA(",ht,",",forecast_horizon,")",sep=""),col=colo[1],line=-1)
mtext("HP-concurrent",col=colo[2],line=-2)
abline(h=0)

#------------------------------------------
# 2.4 Look at shift
# The above plot also suggests that the concurrent HP is slightly left-shifted (small lead): see also phase-lag plot below for formal background

# Here we compute the shift at zero-crossings: the tau-statistic is proposed in Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5 and explained in previous tutorials
#   -The minimum value (trough) indicates a relative lead (left of zero) or lag (right of zero) of the series in the first column of the matrix (here: SSA)
# The slight asymmetry suggests that SSA is marginally lagging HP-concurrent (by roughly half a time-unit)
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
# The amplitude function of SSA is closer to zero in stop band: stronger smoothing and less noisy zero-crossings
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
# The larger phase-lag of SSA implies a slight lag relative to HP-concurrent: roughly half a time-unit (confirming the above tau-statistic in the time-domain)
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


# Findings: SSA lags slightly behind HP-concurrent (by roughly half a time-unit) and it generates ~33% less noisy crossings 

###########################################################################################################
############################################################################################################
# Example 3: 
# We can address lags (time shifts) by selecting a different forecast_horizon, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5 
# We here look specifically at a one-year forecast
# Note, however, that we keep the holding-time fixed: 
#   -The resulting SSA-filter will retain smoothness (noise suppression) 
#   -But it will be slightly faster (leading instead of lagging as in example 2)
# SSA keeps `smoothness' fixed and trades (exchanges) `lead' against MSE-performances, see tutorial 0.1 and example 8 below (trilemma)

# 3.1. SSA and hyperparameters
# Forecast horizon: one-year forecast
forecast_horizon<-12
# Holding time is kept fixed, see example 2
ht<-12
# Recall that we provide the lag-one acf: therefore we have to compute rho1 corresponding to ht
rho1<-compute_rho_from_ht(ht)
# Our selection here means that SSA will have approximately 33% less crossings:
ht/ht_hp
# Target: like example 2 we here supply the classic HP-concurrent
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
#   -As an effect, the scale of SSA is smaller: zero-shrinkage
#   -One could re-scale and inflate the SSA-filter but MSE performances would be even worse (we here see the optimal scaling) 
# Note also that zero-crossings or correlations are indifferent to scalings
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
# SSA is smoother, as expected

# Plot both series: we re-scale both series to unit variance for easier visual comparison 
mplot<-na.exclude(scale(cbind(yhat,HP_concurrent),scale=T,center=F))
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon,")",sep=""),"HP-concurrent")
# Plot a short sample of the series
par(mfrow=c(1,1))
anf<-1000
anf<-1500
anf<-3000
enf<-3500

# SSA generates ~33% less crossings
# The additional crossings of HP are typically clustered at time points where the filter output seems to hover at the zero line
# There, the SSA filter maintains a better control of noisy crossings
# At up- and downswings, away from the zero line, SSA tracks the target well, due to optimality (SSA optimization principle)
# In contrast to example 2 above, SSA now seems to lead HP-concurrent
# This lead is to some extent remarkable because the classic HP-concurrent filter is known for being 'pretty fast'
ts.plot(mplot[anf:enf,],col=colo)
mtext(paste("SSA(",ht,",",forecast_horizon,")",sep=""),col=colo[1],line=-1)
mtext("HP-concurrent",col=colo[2],line=-2)
abline(h=0)

#------------------------------------------
# 3.3 Look at shift
# We can see a lead of SSA-forecast: the minimum of the tau-curve is left-shifted 

shift_tau_obj<-compute_min_tau_func(mplot)

#-------------------------------------------
# 3.4 Look at amplitude and phase-shift
K<-600
amp_obj_SSA<-amp_shift_func(K,as.vector(SSA_filt_HP),F)
amp_obj_HP<-amp_shift_func(K,hp_trend,F)

par(mfrow=c(2,1))
# We scale amplitude functions for ease of visual inspection
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

# The effects of imposing a lead-time are
# 1. Scaling (zero-shrinkage)
# 2. The amplitude morphs into a bandpass-like design
# 3. The shift becomes negative towards lower frequencies 
# The effect of imposing a larger holding-time is:
# 1. The amplitude function is closer to zero in the stopband
# 2. The stronger noise suppression affects the shift (for minimum-phase filters, both amplitude and shift functions are linked bijectively)

########################################################################################################
##########################################################################################################
# Example 4
# Play with target and forecast horizon 
# We rely on example 1 but we specify the symmetric two-sided HP filter as our target
# Problem: 
#   -hp_target provided by the R-package mFilter is a causal one-sided filter (it should be two-sided) 
#   -Therefore we'll have to shift the target `indirectly', by specifying a suitable/corresponding forecast horizon

# 4.1 Compute HP and compare concurrent MSE and SSA designs
# Filter length: we select twice the length (L=401) of examples 1,2 above (L=201) since we split the filter into left and right halves
L_sym<-401
# Should be an odd number: otherwise HP is not centered correctly
if (L_sym/2==as.integer(L_sym/2))
{
  print("Filter length should be an odd number")
  print("If L_sym is even then HP cannot be centered correctly")
  L_sym<-L_sym+1
}  
# HP monthly design
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L_sym,lambda_monthly)
# Bi-infinite HP: this is our new target (instead of one-sided MSE in example 7)
hp_target=HP_obj$target
# The filter is causal
par(mfrow=c(1,1))
ts.plot(hp_target)
#-----------------------------------------------------------------------------------
# 4.2. SSA and hyperparameters
# Holding time: we use the same as in example 1
ht<-1.5*ht_mse
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
# 4.3 Plot: 
#   -The filters in examples 1 and 4 (here) are identical! 
#   -This is because hp_mse in example 1 is the MSE predictor of hp_target here, at least if xt=epsilont (white noise)
#   -Therefore the SSA-solution must be the same, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5 (we can replace the two-sided target by its MSE predictor without affecting optimization)
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
#   -The correlation is smaller because the target is the output of a non-causal (symmetric) filter
#   -SSA cannot track the acausal filter as well
SSA_obj_HP$crit_rhoy_target

# We now compute the empirical correlations
# a. Generate very long series in order to obtain accurate empirical estimates
len<-1000000
set.seed(14)
x<-arima.sim(n = len, list(ar = a1))
# b. Compute filter outputs
# of SSA-HP filter
yhat<-filter(x,SSA_filt_HP,side=1)
# of symmetric HP output: we now have to set side=2 (symmetry)
HP_symmetric<-filter(x,hp_target,side=2)
# of MSE HP output: we have to set side=1 (one-sided)
HP_concurrent<-filter(x,hp_mse,side=1)
# c. Compute empirical correlations
mplot<-na.exclude(scale(cbind(yhat,HP_concurrent,HP_symmetric),scale=T,center=F))
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP-MSE","HP-symmetric (effective target")
cor(mplot)
# The empirical correlations (see first row in the empirical correlation matrix) match crit_rhoyz and crit_rhoy_target
# The MSE filter has a slightly larger target correlation with the symmetric HP, by design (it maximizes the target correlation)

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
# Therefore, we have to decompose xt in a white noise sequence: Wold decomposition, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# Once decomposed, the holding-time is calculated properly

# Step 1: Compute the MA-inversion of the ARMA (Wold-decomposition or MA-inversion)
# One can insert true or estimated parameters 
xi_data<-c(1,ARMAtoMA(ar=a1,ma=b1,lag.max=L-1))
# Have a look at xi: if xi has not decayed to zero then one should increase the filter-length L
# Note:
#   -L should be sufficiently large so that finite-length convolutions and deconvolutions are good proxies (of infinite length transformations), see section 2 in JBCY paper
#   -For integrated processes xt, the MA-inversion xi does not decay to zero: this case will be analyzed in tutorials 3 (Hamilton filter), 4 (Baxter King filter) and 5 (HP-gap filter): see proposition 4 in JBCY paper for background 
par(mfrow=c(1,1))
ts.plot(xi_data)
# Step 2: Convolve xi_data and hp, see section 2 of JBCY paper: 
#   The resulting convolved filter is applied to epsilont (innovation in Wold decomposition)
#   Therefore the expected holding-time (of the convolved filter) is 
hp_conv<-conv_two_filt_func(xi_data,hp_trend)$conv
ht_hp_conv_obj<-compute_holding_time_func(hp_conv)
ht_hp_conv_obj$ht
# Now the expected holding time (above) matches the empirical one (below), at least up to finite sample error (the empirical number corresponds to the expected ht for increasing sample length)
compute_empirical_ht_func(y_hp)

# We stressed in tutorial 2.0 that an application of the same fixed HP-design to macro-indicators with different 
#   autocorrelation structure leads to qualitatively different cycles (INDPRO and non-farm payroll)

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
# Target: classic HP-concurrent
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
par(mfrow=c(2,1))
# Filters applied to xt (these are mostly relevant in applications)
mplot<-cbind(SSA_filt_HP,hp_trend)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP-concurrent")
colo<-c("blue","green")
# The typical noise-shape or tip of SSA is due to an implicit boundary constraint which states that the coefficient at lag -1 vanishes, see theorem 1 in JBCY paper
plot(mplot[,1],main=paste("HP(",lambda_monthly,"): filters as applied to xt",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()
# We can also compare the filters which are applied to epsilont (model residuals)
mplot<-cbind(ssa_eps,hp_conv)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"HP-concurrent")
colo<-c("blue","green")
# The typical noise-shape or tip of SSA is due to an implicit boundary constraint which states that the coefficient at lag -1 vanishes, see theorem 1 in JBCY paper
plot(mplot[,1],main=paste("HP(",lambda_monthly,"): filters as applied to epsilont",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
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

# Plot both series:
mplot<-na.exclude(cbind(yhat,HP_concurrent))
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

# We could also address timeliness (lead/lags) as in example 4 above but we leave this as an exercise 


##########################################################################################################
##########################################################################################################
# Example 6
# Same as example 5 but we now target specifically the output of the symmetric HP-filter
# As a result, we will obtain a one-sided filter which tracks the output of the symmetric filter better than the classic HP-concurrent or HP-MSE (the latter is assuming white noise)
# Recall that the classic HP-concurrent is optimal if the series is an ARIMA(0,2,2), where the two MA-parameters are determined by lambda, see tutorial 2.0
# This example is a bit tricky because everything is merged and entangled!

# Let us emphasize that:
# -This exercise does not refer to the majority of use-cases of HP: 
#   -Most (nearly all) practitioners assume that the classic concurrent HP is the optimal estimate of the two-sided filter at the sample end (real-time business BCA)
#   -This assumption is wrong because economic data does not conform to an ARMA(0,2,2)-process, see tutorial 2.0
#   -Therefore, by applying the classic HP-concurrent, users are not effectively tracking the two-sided target (at least not optimally)
# -We advocated in tutorial 2.0 that the classic HP-concurrent has some desirable characteristics for BCA
#   -Therefore, we plugged SSA on the classic HP-concurrent in the previous exercises 2,3 and 5
#   -But, clearly, its tracking ability (of the two-sided target) is not optimal in typical applications 
#   -When applying HP-concurrent, practitioners use a pertinent BCA-tool; but the tool is not doing what it is claimed to do (and what the analyst think it does)
# -Exercise 6, here, addresses specifically an optimal one-sided filter for tracking the two-sided HP, assuming the true 
#   (or the empirical) model of the data (which is not an ARIMA(0,2,2))
# -This exercise is meant for analysts mainly interested in tracking the two-sided filter by an optimal concurrent design
# -Besides the optimal MSE we also present SSA extensions which are smoother and/or faster 

# 6.1 Specify symmetric filter as target
L_sym<-401
# Should be an odd number: otherwise HP is not centered correctly
if (L_sym/2==as.integer(L_sym/2))
{
  print("Filter length should be an odd number")
  print("If L_sym is even then HP cannot be correctly centered")
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
# Step a: convolve xi_data and symmetric hp_target
hp_conv_mse_d<-conv_two_filt_func(xi_data,hp_target)$conv
# Step b: truncate at lag 0 corresponding to forecast_horizon
hp_conv_mse<-hp_conv_mse_d[(forecast_horizon+1):L_sym]
# We can now compute the holding-time of the optimal MSE filter: this is a natural benchmark for SSA 
# -Idea: SSA should improve smoothness of best (MSE) nowcast
# -Of course, the holding-time could be set differently (based on a priori knowledge or on particular priorities). 
# -But having a natural (optimal) benchmark at disposal allows for more stringent/interesting comparisons
ht_hp_conv_mse_obj<-compute_holding_time_func(hp_conv_mse)
ht_hp_conv_mse<-ht_hp_conv_mse_obj$ht
# ht of optimal MSE (assuming the data is not white noise): 
ht_hp_conv_mse
# We can compare with the holding-time ht_mse of hp_mse above (assuming white noise): this is slightly smaller because the data is weakly positively autocorrelated
ht_mse
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
# Retrieve benchmark MSE filter: this is automatically calculated by SSA 
HP_MSE_x<-SSA_obj_HP$mse_x


# Plot and compare with SSA designs of previous examples 4 and 5
colo<-c("black","brown","blue")
par(mfrow=c(1,1))
mplot<-cbind(SSA_example4,SSA_example5,SSA_filt_HP)
colnames(mplot)<-c("Example 4","Example 5","Example 6")

# The filters of examples 4 and 6 look similar. Why is that?
#   -Example 4 relied on the symmetric target too, but it assumed xt=epsilont white noise (xi=NULL) and a smaller holding time ht=12
#   -Here, in example 6, we assume an ARMA-process and a larger holding-time of approximately 17
#   -As it happens (fortuitously) the solution of example 4 when applied to the ARMA-process has a holding-time very close to 17
#   -Therefore, both designs are close to each other (up to some differences towards lag 0)
#   -More significant differences could be observed by changing the above specifications (the ARMA-process and/or the holding-time constraint)
# The scale of example 5 is quite different
#   -This is because the estimation problem of example 5 is much simpler since we target the one-sided filter hp_trend
#   -In contrast, examples 4 and 6 nowcast the acausal two-sided HP
#   -The smaller scaling (zero-shrinkage) in the latter examples reflects the increased uncertainty in this case
plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
lines(mplot[,3],col=colo[3],lwd=2,lty=1)
mtext(colnames(mplot)[3],col=colo[3],line=-3)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Now that the relevant filters have been sorted out and computed, we can  filter the series, compare 
#   performances and proceed to our checks of empirical and expected numbers (holding times, correlations)
#-------------------------------------------
# 6.3 Filter series 
# Generate very long series in order to obtain accurate empirical estimates
len<-1000000
set.seed(14)
x<-arima.sim(n = len, list(ar = a1,ma=b1))
# Compute filter output of SSA-HP filter: optimal as applied to xt
yhat<-filter(x,SSA_filt_HP,side=1)
# Compare expected and empirical holding-times: they match, as desired
ht
compute_empirical_ht_func(yhat)

# Compute filter output of  MSE 
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
# Interestingly, as we shall see below, the classic HP concurrent looses not only in terms of smoothness but also in terms of correlation with the target
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
# Classic concurrent: sum is one, too
sum(hp_trend)
# SSA: sum is markedly smaller than one
sum(SSA_filt_HP)
# hp_mse: sum is markedly smaller than one
sum(hp_mse)
# We infer that hp_trend (classic concurrent HP) matches hp_target, at least in terms of amplitude at frequency zero: 
#   -this is because the implicit model (ARIMA(0,2,2)) assumes the data to be strongly trending
#   -in order to track the two-sided filter, the one-sided hp_trend must pass the dominating 
#     (low-frequency) trend, without changing scales: the filter coefficients sum to one.
# However, if the data is not strongly trending, i.e. if the ARIMA(0,2,2) is severely misspecified, then there is 
#     no necessity to track the (missing) trend: the filter coefficients do not have to add to one anymore
#   -In particular, in the presence of noise (strong high-frequency components), MSE-performances generally improve
#     by shrinking the predictor (zero-shrinkage): 
#       -the coefficients of SSA sum to ~0.5 (instead of one)
#       -hp_mse assumes the data to be white noise and its coefficients sum to ~0.5
# The `odd' scaling of hp_trend in the above plot is due to severe misspecification. 
#   -Part of this misspecification could be addressed by simply re-scaling the filter output (say by 0.5)
#   -But even after re-scaling the classic HP-concurrent is not optimal, see below

#------------------------------------------
# 6.4 Compute empirical and true (expected) performance numbers and compare all one-sided designs
# Compute empirical cross correlations of all designs: 
cor_mat<-cor(mplot)
cor_mat
# Let's first look at correlations of the various one-sided designs with the target: 
#  Numbers in the third row of the correlation matrix
cor_mat[3,]
# -MSE marginally outperforms SSA (MSE is best possible; but SSA is smoother: 50% larger empirical holding-times in above simulation):
cor_mat[3,2]
cor_mat[3,1]
# -Of course, MSE also outperforms the classic one-sided HP
cor_mat[3,2]
cor_mat[3,4]
# -Interestingly, SSA outperforms HP-concurrent both in terms of smoothness (fewer crossings) as well 
#   as in terms of correlation with target 
# As stated: the classic HP-concurrent filter is generally not optimal for applications: typical economic data
#   (in levels or in differences) is less smooth than assumed by hp_trend
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


##########################################################################################################
##########################################################################################################
# Example 7
# This is more of a counter-example and it is not related to SSA.
# We here analyze the classic HP-gap filter, see also tutorial 2.0.
# Based on its characteristics we advise against its use in BCA, in agreement with Hamilton: "never use the HP (gap)...", see tutorial 3

# HP gap
# 7.1 Compute HP-gap: we use the same length as in example 6 above
L_sym<-401
# Should be an odd number: otherwise HP is not centered correctly
if (L_sym/2==as.integer(L_sym/2))
{
  print("Filter length should be an odd number")
  print("If L_sym is even then HP cannot be correctly centered")
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
# Check: the difference of the following filters should vanish everywhere
max(abs((c(1,rep(0,L_sym-1))-hp_trend)-hp_gap))

#---------------------------------------------
# 7.2 Transformation: from levels to first differences
# The gap filter is applied to data in levels: this renders a direct analysis cumbersome and difficult
# Instead, we here derive a filter whose output is the same as HP-gap when applied to first differences, see tutorial 2.0
# This transformation will simplify our analysis, see proposition 4 in JBCY paper for background.
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
ts.plot(yhat_diff,col="blue",main="Transformed HP-gap applied to differences replicates original HP-gap applied to levels")
lines(yhat_gap,col="red")
#-------------------------------------------
# 7.3 We can now compute the amplitude  of HP-gap or, better, gap_diff as applied to first differences
# First differences of typical economic data are generally close to white noise (typical spectral shape, see Granger (1966))
# Therefore the squared amplitude of gap_diff is close to the spectral density of the filter output (convolution theorem)
#   This is the main reason for considering gap_diff: we can derive the spectral density of the extracted cycle

# We now select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude of gap_diff
amp_gap_diff<-amp_shift_func(K,as.vector(gap_diff),F)$amp

# Plot amplitude function
plot(amp_gap_diff,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP-gap as applied to differences",sep=""))
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# Plot squared amplitude function: this is the true spectral density of the original HP-gap `cycle', when applied to a random-walk
# This is a good proxy of the true spectral density of the HP-gap `cycle' when applied to typical (non-stationary) economic data in levels
par(mfrow=c(1,1))
plot(amp_gap_diff^2,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Squared amplitude HP-gap as applied to differences",sep=""))
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

# Outcomes:
#   1. The original HP-gap is a highpass when applied to levels, see tutorial 2.0; but it is a bandpass when applied to first differences (the above amplitude vanishes at frequency zero)
#     The one-sided HP-gap assumes a second order unit-root: after first differences, gap_diff must still remove a remaining unit-root
#   2. HP-gap tends to generate a spurious cycle whose periodicity corresponds to the frequency of the peak-amplitude in the above plot

# Let's briefly check the second claim
# This is the frequency of the peak-amplitude
omega_gap<-pi*(which(amp_gap_diff==max(amp_gap_diff))-1)/K
# The periodicity in months: 
2*pi/omega_gap
# Approximately 6 years: a bit short for a completed cycle when compared against the duration of longer expansions (great moderation) 
#   Therefore HP-gap tends to generate spurious alarms in midst of expansions, see tutorials 2.0 and 5

# The periodogram of the filter output confirms our findings: low and high frequencies are damped, 
#   thus generating a spurious cycle at the peak frequency
per_obj<-per(na.exclude(yhat_gap),T)
# The acf confirms these findings
acf(na.exclude(yhat_gap))
# Finally: here's the cyle
par(mfrow=c(1,1))
ts.plot(yhat_gap)
abline(h=0)
# There are lots of noisy zero-crossings, too, which would hamper a real-time assessment of the cycle, see tutorials 2.0 and 5
#   As we shall see, the Baxter and King filter is subject to similar issues, see tutorial 4
# These findings confirm and reinforce Hamilton's statement: never use the HP (-gap) filter for BCA
# But we may add: try the HP-concurrent lowpass instead, as applied to differences, see tutorials 2.0 and 5
# Tutorial 2.0 shows that HP-trend applied to differences is a more conservative design (than the original HP-gap)
#   -It tracks expansions and recessions well (neither too smooth nor too noisy: a smart compromise); 
#   -It does not generate spurious alarms years ahead of (or past) effective recessions; 
#   -It is slightly lagging behind HP-gap at start and end of recessions;
#   -Its real-time characteristics can be modified by SSA in order to match specific priorities, see the above examples


##########################################################################################################
##########################################################################################################
# Example 8
# Visualization of Prediction  Trilemma for HP
# Background:
# -MSE, timeliness and smoothness define a forecast trilemma, see tutorial 0.1
# -We measure all three terms as follows in this exercise:
# a. MSE: we compute the SSA-criteria crit_rhoyz and crit_rhoy_target 
#   -The criteria measure the correlations of the SSA-predictor with causal and acausal targets, see examples above
#   -Large correlations mean: small MSE 
# b. Timeliness: forecast horizon
# c. Smoothness: ht in holding-time constraint

# For visualization, we then represent or plot all three terms of the trilemma in a heat-map

# For this study we consider the classic concurrent trend estimate hp_trend as a target of SSA. 
# We then compute crit_rhoy_target and crit_rhoyz for a range of ht and forecast horizons.
# The forecast trilemma is visualized by a heat map of the criterion value(s) as a function of ht and 
#   forecast horizon.

#----------------------------------------
# 8.1 Setting-up SSA

# The following computations are a bit lengthy (~5 Min. on 3GHz single core)
#   One can skip the loop and load a file with the results

compute_length_loop<-F
# Specify target
gammak_generic<-hp_target

if (compute_length_loop)
{  
# Holding time of hp_mse 
  ht_mse
# Compute SSA and MSE for a selection of ht
  ht_vec<-seq(max(2,ht_mse/4), 2*ht_mse, by = 0.1)
# Compute SSA and MSE for a selection of forecasts horizons
# Note: we must shift the causal symmetric HP by (L_sym-1)/2 to the left in order to obtain the acausal two-sided target
  delta_vec<-0:24+(L_sym-1)/2
  
  pb = txtProgressBar(min = 0, max = length(ht_vec), initial = 0,style=3) 
  
  MSE_mat<-target_mat<-matrix(ncol=length(delta_vec),nrow=length(ht_vec))
# Loop through all combinations of ht and forecast horizon: compute the SSA filter and collect 
#   crit_rhoy_target (correlation of SSA with effective target) as well as crit_rhoyz (correlation with causal MSE benchmark)  
  for (i in 1:length(ht_vec))
  {
    setTxtProgressBar(pb,i)
    for (j in 1:length(delta_vec))
    {  
      rho1<-rho1<-compute_rho_from_ht(ht_vec[i])
      forecast_horizon<-delta_vec[j]
# Skip xi: we assume white noise    
      SSA_obj_HP<-SSA_func(L,forecast_horizon,gammak_generic,rho1)
# Correlation with (caual) MSE predictor: this is the preferred measure here because we can benchmark SSA
#   directly against MSE 
      MSE_mat[i,j]<-SSA_obj_HP$crit_rhoyz
# Or correlation with (acausal) target 
      target_mat[i,j]<-SSA_obj_HP$crit_rhoy_target
    }
  }
  close(pb)
# Row-names correspond to holding-times; column-names are forecast horizons  
  rownames(MSE_mat)<-rownames(target_mat)<-round(ht_vec,2)
# Forecast horizon: we remove the artificial shift (L_sym-1)/2 
  colnames(MSE_mat)<-colnames(target_mat)<-delta_vec-(L_sym-1)/2
# Save results
  save(MSE_mat,file=paste(getwd(),"/Data/Trilemma_mse_heat_map",sep=""))
  save(target_mat,file=paste(getwd(),"/Data/Trilemma_target_heat_map",sep=""))
} else
{  
# Load pre-computed results  
  load(file=paste(getwd(),"/Data/Trilemma_mse_heat_map",sep=""))
  load(file=paste(getwd(),"/Data/Trilemma_target_heat_map",sep=""))
}

# 1. MSE_mat collects the correlations crit_rhoyz of SSA with the causal MSE benchmark predictor of the target
#   Row names correspond to ht (holding-time constraint)
#   Column names correspond to the forecast horizon: from a nowcast up to 24-steps ahead
head(MSE_mat)
tail(MSE_mat)
# 2. target_mat collects the correlations crit_rhoy_target of SSA with the effective (acausal two-sided) target 
#     -In our case: the two-sided filter shifted by 0,1,...,24
#   Naturally, these correlations are smaller than in MSE_mat 
head(target_mat)
tail(target_mat)
#---------------------------------------
# 8.2 Heat maps
# We can now represent and visualize the trilemma in a heat map
# 8.2.0 Specify color scheme
lcol<-100
coloh<-rainbow(lcol)[1:(10*lcol/11)]
colo<-coloh[length(coloh):1]
# 8.2.1. Heat-map of correlations with acausal (effective) target
heatmap.2(target_mat[nrow(target_mat):1,], dendrogram="none",scale = "none", col = colo,trace = "none", density.info = "none",Rowv = F, Colv = F,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Trilemma: Correlation with Effective Acausal Target")

# Interpretations of heat-map
# -For fixed ht, the correlations decrease with increasing forecast horizon
#   -Enhancing timeliness (larger forecast horizon) for fixed smoothness (ht) means an increase of MSE (decrease of correlation)
# -For fixed forecast horizon, the correlations peak somewhere: 
#   -The peak value corresponds to the classic MSE-predictor of the target for that forecast horizon and the corresponding ht is the holding-time of the classic MSE predictor
#   -Above and below the peak-value, the holding-time constraint in SSA is `activated': SSA maximizes the correlation subject to ht
#   -Enhancing smoothness, relative to the MSE-benchmark and for fixed forecast horizon, means a decrease of correlation or increase of MSE
# -Increasing simultaneously ht and the forecast horizon (along a diagonal) affects correlations (MSE) disproportionately
#   -See SSA-forecasts vs. SSA-nowcasts in the above examples 
# -Improving correlations (smaller MSE) and timeliness (larger forecast horizon) affects smoothness (ht) disproportionately

# 8.2.2 The resolution in the above plot can be improved by `squeezing' the correlations with the non-linear log-transform
heatmap.2(log(target_mat[nrow(target_mat):1,]), dendrogram="none",scale = "none", col = colo,trace = "none", density.info = "none",Rowv = F, Colv = F,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Trilemma: Log-Correlation with Effective Acausal Target")
# We can now observe additional structure

# 8.2.3 We can plot slices (selected columns) of the above heat-map 
select_vec<-22:24
mplot<-target_mat[,select_vec]
coli<-rainbow(length(select_vec))
par(mfrow=c(1,1))
plot(mplot[,1],col=colo,ylim=c(min(mplot),max(mplot)),axes=F,type="l",xlab="Holding time",ylab="Correlation",main="Trilemma for HP: Correlations as a Function of ht and a Selection of Forecast Horizons")
mtext(paste("Forecast horizon ",colnames(MSE_mat)[select_vec[1]],sep=""),col=coli[1],line=-1)
if (length(select_vec)>1)
  for (i in 2:length(select_vec))
  {
    lines(mplot[,i],col=coli[i])
    mtext(paste("Forecast horizon ",colnames(MSE_mat)[select_vec[i]],sep=""),col=coli[i],line=-i)
  }
cor_val=0.105
abline(h=cor_val)
for (i in 1:ncol(mplot))
  abline(v=which(mplot[1:(nrow(mplot)-1),i]>cor_val&mplot[2:nrow(mplot),i]<cor_val),col=coli[i])
axis(1,at=1:nrow(MSE_mat),labels=rownames(MSE_mat))
axis(2)
box()
# For a given forecast horizon, the peak of the correlation value corresponds to the MSE-predictor 
# To the left and to the right of the peak-value, SSA maximizes the correlation subject to ht
# SSA is smoother (than the classic MSE predictor) if ht is to the right of the peak; otherwise it is `unsmoother'
# For a given fixed correlation, one can trade smoothness (holding-time) for timeliness (forecast horizon)
#   -As an example, a fixed correlation value of 0.105 (horizontal black line in graph) intersects the correlation 
#     curves corresponding to forecast horizons 21, 22 and 23 at the holding-times ht~20, ht~16 and ht~9, in decreasing size

# In our applications of SSA to HP in example 3, we traded timeliness (larger forecast horizon) against MSE (smaller correlation) for fixed ht, see also tutorial 5

# 8.2.4 It is also possible to draw a heat-map for the correlations with the causal MSE-predictor
#   -This heat-map does not emphasize prediction (the target is causal); it is about smoothing (see section 2.4 in JBCY paper)
heatmap.2(MSE_mat[nrow(MSE_mat):1,], dendrogram="none",scale = "none", col = colo,trace = "none", density.info = "none",Rowv = F, Colv = F,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Correlation with causal MSE-Predictor")


# 8.2.5 We can emphasize more specifically the effect of ht on correlations (MSE) for a fixed forecast horizon
# For this purpose, we can scale the data in the column direction of the heat map 
#   -Scaling along the column means that the absolute effect of the forecast horizon is diminished
#   -As above, the MSE-benchmark predictor corresponds to the peak criterion value for each forecast criterion (darkest ridge in plot)
heatmap(target_mat,col=colo,scale="column",Rowv = NA, Colv = NA,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Trilemma: Correlation with  Effective Acausal Target (scaled along ht)")

# 8.2.6 We can apply the same scaling to the correlations with the causal MSE predictor
heatmap(MSE_mat, scale = "column", col = colo,Rowv = NA, Colv = NA,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Trilemma: Correlation with Causal MSE Predictor (scaled along ht)")
# Same plot because for given forecast horizon the two criteria differ only by a scaling, see proposition 5 in JBCY paper
#   Therefore, scaling along columns will cancel the difference between both criteria






if (F)
{  
# We can also rely on ggplot for drawing a heat-map: the code is set-up in a function heat_map_func
# We can apply the same scaling as above: to bring forward the ht-effect better
  scale_column<-T
# We can select MSE-predictor or effective target
  select_acausal_target<-T

  heat_map_func(scale_column,select_acausal_target,MSE_mat,target_mat)
}


