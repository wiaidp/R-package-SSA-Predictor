# We here introduce to the M-SSA. see paper provide link???
# Work in progress...

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
# Example 1: we replicate SSA by M-SSA
# We first compute a univariate SSA design
# For this purpose we rely on example 1 of tutorial 2.1
# This example relies on targeting a two-sided HP filter
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
# Bi-infinite two-sided (symmetric) HP
hp_target<-HP_obj$target
ts.plot(hp_target)
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

# Compare holding-times (ht) of one- and two-sided filters
compute_holding_time_func(hp_target)$ht
ht_hp
# The large (atypical) discrepancy between holding-times of two- and one-sided filters is discussed in the JBCY paper

#------------------------------
# Target HP-MSE


# 1.1 Concurrent MSE estimate of bi-infinite HP assuming white noise
# This is just the truncate right tail of the symmetric filter
# This one is optimal if the data is white noise
hp_mse=hp_mse_example7=HP_obj$hp_mse
par(mfrow=c(1,1))
ts.plot(hp_mse)
# Compute lag-one acf and ht for hp_mse
htrho_obj<-compute_holding_time_func(hp_mse)
rho_hp<-htrho_obj$rho_ff1
ht_mse<-htrho_obj$ht
# MSE filter is smoother than classic HP concurrent (larger ht) because white noise is, well, `noisier' than ARIMA(0,2,2)
#   Therefore hp_mse must damp high-frequency components more strongly than hp_trend
ht_mse

#-----------------------------------------------------------------------------------
# 1.2. Setting-up SSA
# Holding time: we typically want SSA to lessen the number of zero-crossings when compared to hp_mse 
ht_mse
# Therefore we select a ht which is larger than the above number
ht<-1.5*ht_mse
# Recall that we provide the lag-one acf: therefore we have to compute rho1 (corresponding to ht) for SSA
rho1<-compute_rho_from_ht(ht)
# Our selection here means that SSA will have 33% less crossings:
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








# We can replicate univariate models by specifying `diagonal' xi and Sigma as below
if (F)
{
  xi<-xi_mat_uni_bi
  Sigma<-Sigma_uni_bi
  rho0<-rho0_uni_vec
  ht_ssa_vec<-compute_holding_time_from_rho_func(rho0)$ht
  
}




# Load  M-SSA functions
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))






# Example 2: we illustrate a bivariate M-SSA based on a `bogus' design 
#  -In principle the same as example 1 but we add an irrelevant (cross-sectionally uncorrelated white noise) series 



