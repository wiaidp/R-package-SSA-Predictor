# Tutorial on maximal monotone predictors for non-stationary (I(1)) time series

# Previous tutorials 1-4 are based on Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# This (new) tutorial 5 is based on Wildi (2025). The new paper proposes: 
#   -Technical/formal background to Wildi (2024)
#   -Introduction of a new SSA predictor for non-stationary time series

# For non-stationary time series, an extension of SSA is proposed which tracks the target optimally in a MSE sense
#   (predictor and target are cointegrated) and which controls the rate of zero-crossings of (stationary) FIRST DIFFERENCES of the predictor.
# The predictor tracks the target as closely as possible subject to a holding-time constraint imposed to first differences.
#   -In its dual form: the predictor generates the least number of zero-crossings in first differences for a given track accuracy (MSE).
#   -Equivalently: the original predictor (in levels) is maximal monotone

# Application/prediction problem:
# -We compute a maximal monotone SSA nowcast for the two-sided (bi-infinite) HP(14400) applied to monthly US-INDPRO.
# -We compute the MSE nowcast of HP based on an ARIMA(1,1,0)-model for INDPRO: benchmark 1
#   -This MSE nowcast is much noisier than the classic HP concurrent (real-time) filter, widely used in business-cycle applications
# -Therefore, we consider the classic concurrent one-sided HP (HP-C): benchmark 2
#   -We replicate the holding-time (smoothness) of HP-C by SSA
# -By optimization we then expect SSA to equal HP-C in terms of smoothness and to outperform HP-C in terms of MSE (tracking accuracy)
# -Since SSA is much smoother than the MSE nowcast we are interested in finding out how much SSA looses in terms of MSE performances.
#   -The SSA criterion minimizes MSE losses for given smoothness (holding-time)

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
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

#----------------------------------------------------------------------------
# Load data: setting reload_data<-T downloads latest release
reload_data<-F
if (reload_data)
{  
  getSymbols('INDPRO',src='FRED')
  save(INDPRO,file=paste(getwd(),"/Data/INDPRO",sep=""))
  
} else
{
  load(file=paste(getwd(),"/Data/INDPRO",sep=""))
}
tail(INDPRO)

end_year<-2024
start_year<-1982

y<-as.double(log(INDPRO[paste(start_year,"/",end_year,sep="")]))
y_xts<-log(INDPRO[paste(start_year,"/",end_year,sep="")])

#-----------------------------------------------------------
# Plot data
par(mfrow=c(2,2))

plot(as.double(INDPRO),main="INDPRO",axes=F,type="l",xlab="",ylab="",col="black",lwd=1)
axis(1,at=1:length(INDPRO),labels=index(INDPRO))
axis(2)
box()
plot(as.double(y_xts),main="Log-INDPRO",axes=F,type="l",xlab="",ylab="",col="black",lwd=1)
axis(1,at=1:length(y_xts),labels=index(y_xts))
axis(2)
box()
plot(as.double(diff(y_xts)),main="Diff-log",axes=F,type="l",xlab="",ylab="",col="black",lwd=1)
abline(h=0)
axis(1,at=1:length(diff(y_xts)),labels=index(diff(y_xts)))
axis(2)
box()

# ACF suggests an AR(1) process for differenced INDPRO
acf(na.exclude(diff(y_xts)),main="ACF of diff-log")
len<-length(y)
x_tilde<-as.double(y_xts)

# In contrast to Wildi (2024), who emphasizes log-differences of INDPRO, Wildi (2025) considers log-INDPRO (in levels)
#   SSA must track the two-sided HP as applied to levels (not first differences)

####################################################################
# The novel maximal monotone predictor relies on a cointegration constraint such that the MSE between predictor and target 
#   is finite, see Wildi 2025 for background.
# For this purpose we need to compute particular system matrices
# We assume (log-) INDPRO to follow an ARIMA(1,1,0) specification, see ACF above.
# We do not fit a model to avoid data-mining (the singular observations during great lockdown are likely to corrupt parameter estimates)
a1<-0.3
b1<-0

# Filter settings: length and HP-lambda
L<-101
lambda_hp<-14400

# Compute all relevant system matrices: HP two- and one-sided and all relevant SSA design filters
filter_obj<-compute_system_filters_func(L,lambda_hp,a1,b1)

# Cointegration matrix: ensures finite MSE in the case of I(2) processes
B=filter_obj$B
# Autocovariance generating matrix, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
M=filter_obj$M
# Transformed target, see section 5.3 Wildi (2025)
gamma_tilde=filter_obj$gamma_tilde
# MSE filter applied to original (non-stationary) data: derived from target filter and MA inversion
gamma_mse=filter_obj$gamma_mse
# Convolution of integration and Wold decomposition, see section 5.3 Wildi (2025)
Xi_tilde=filter_obj$Xi_tilde
# Integration (sum), see section 5.3 Wildi (2025)
Sigma=filter_obj$Sigma
# Inverse of integration, see section 5.3 Wildi (2025)
Delta=filter_obj$Delta
# Wold decomposition (MA-inversion of AR(1) with parameter a1=0.3): depends on a1
Xi=filter_obj$Xi
# Two-sided target: used for computing sample MSEs below
hp_two=filter_obj$hp_two
# Classic concurrent HP: we want to replicate the latter's holding-time by SSA
hp_trend=filter_obj$hp_trend

# Specify rho1 in HT constraint, see Wildi (2025) section 5.4
# 1. Compute rho_mse the lag-one ACF of the MSE nowcast
#   The holding time constraint is expressed in (stationary) first differences of the I(1) process
#   The (differenced) data is an AR(1) and therefore we have to compute the convolution of filters and MA-inversion, see tutorial 2
rho_mse<-compute_holding_time_func(Xi%*%gamma_mse)$rho_ff1
# We also use the MA-inversion Xi%*%hp_trend for deriving the lag-one ACF of HP-C
rho_hp_concurrent<-as.double(compute_holding_time_func(Xi%*%hp_trend)$rho_ff1)

# The lag-one ACF of the MSE nowcast is small: the filter generates many noisy zero-crossings (see last plot below)
rho_mse
# We set rho1=rho_hp_concurrent in the HT constraint
# Research question/assumption: impose same smoothness as HP-C; should outperform HP-C in terms of MSE; without loosing too much vs. gamma_mse 
# Increasing smoothness leads to worse MSE performances (Accuracy-Smoothness dilemma)
rho1<-rho_hp_concurrent

if (F)
{
# We can impose a holding time 50% larger than HP-C to equal MSE performances: 
#   SSA will replicate MSE performances of HP-C with less noisy zero-crossings  
  ht<-1.5*as.double(compute_holding_time_func(Xi%*%hp_trend)$ht)
  rho1<-compute_rho_from_ht(ht)$rho
} 
#----------------------------
# Compute SSA solution, seee Wildi (2025) sections 5.3 and 5.4
#   Use optim to determine optimal Lagrangian multiplier lambda numerically
#     The optimal Lagrangian ensures compliance with the HT constraint
#   Initialize lambda with 0: MSE benchmark (numerical optimization must improve upon MSE-initialization)
lambda<-0
opt_obj<-optim(lambda,b_optim,lambda,gamma_mse,Xi,Sigma,Xi_tilde,M,B,gamma_tilde,rho1)

# Optimized lambda
lambda_opt<-opt_obj$par

# Compute cointegrated I(1) SSA solution for the optimal Lagrangian parameter lambda_opt
bk_obj<-bk_int_func(lambda_opt,gamma_mse,Xi,Sigma,Xi_tilde,M,B,gamma_tilde)

# Lag-one ACF: matches HT constraint rho1 as desired
bk_obj$rho_yy
rho1
# Correlation with target (based on synthetic stationary series, see Wildi (2025), section 5.3)
bk_obj$rho_yz
# MSE: with respect to causal MSE predictor
bk_obj$mse_yz
# SSA filter: b applied to x, i.e., INDPRO
b_x<-bk_obj$b_x
# b applied to eps in synthetic series
b_eps<-bk_obj$b_eps

# If Xi=Identity then b_x=b_eps
par(mfrow=c(1,2))
ts.plot(b_eps,main="B as applied to epsilon")
ts.plot(b_x,main="b as applied to INDPRO")

# Check cointegration constraint: difference should vanish, see Wildi (2025) section 5.3
sum(b_x)-sum(gamma_mse)
# Check HT constraint: difference should vanish for optimal Lagrangian lambda_opt (up to rounding error)
bk_obj$rho_yy-rho1

# Plot the filters
par(mfrow=c(1,2))
colo<-c("violet","green","blue","red")

mplot<-cbind(hp_two,c(gamma_mse,rep(0,L-1)),c(b_x,rep(0,L-1)),c(hp_trend,rep(0,L-1)))
colnames(mplot)<-c("HP-two","MSE","SSA","HP-C")
plot(mplot[,1],main="Trend filters",axes=F,type="l",ylab="",xlab="Lags",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
abline(h=0)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],line=-i,col=colo[i])
}
axis(1,at=1:nrow(mplot),labels=0:(nrow(mplot)-1))
axis(2)
box()


mplot<-cbind(hp_two,c(gamma_mse,rep(0,L-1)),c(b_x,rep(0,L-1)),c(hp_trend,rep(0,L-1)))[1:30,]
colnames(mplot)<-c("HP-two","MSE","SSA","HP-C")
plot(mplot[,1],main="",axes=F,type="l",ylab="",xlab="",col=colo[1],lwd=1,ylim=c(min(mplot[,"HP-C"]),max(mplot[,"SSA"])))
abline(h=0)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],line=-i,col=colo[i])
}
axis(1,at=1:nrow(mplot),labels=0:(nrow(mplot)-1))
axis(2)
box()


#----------------------------------------
# Filter data

y_ssa<-filter(x_tilde,b_x,side=1)
y_hp_concurrent<-filter(x_tilde,hp_trend,side=1)
y_mse<-filter(x_tilde,gamma_mse,side=1)
y_target<-filter(x_tilde,hp_two,side=2)
colo<-c("black","violet","green","blue","red")

# Plots
# First plot: data in levels
#   -Classic HP (HP-C) over/under-shoots at peaks and dips; it also systematically lags (right shifted); 
#   -MSE is closest to target, by design; SSA is as smooth as HP-C but closer to target
par(mfrow=c(1,1))
anf<-L+100
enf<-length(x_tilde)
mplot<-cbind(x_tilde,y_target,y_mse,y_ssa,y_hp_concurrent)[anf:enf,]
colnames(mplot)<-c("Data","Target: HP-two","MSE: HP-one","SSA","HP-C")
plot(mplot[,1],main="Data and trends",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],line=-i,col=colo[i])
}
axis(1,at=1:nrow(mplot),labels=index(y_xts)[(anf):length(y_xts)])
axis(2)
box()


# Second plot: data in differences
#   -MSE (top blot) is very noisy and generates lots of zero-crossings
#   -HP-C and SSA are much smoother and both track last three recessions quite well with few false alarms during longer expansions
anf<-L+100
enf<-length(x_tilde)-1
mplot<-apply(cbind(x_tilde,y_target,y_mse,y_ssa,y_hp_concurrent),2,diff)[anf:enf,]
colnames(mplot)<-c("Diff-Data","Target: HP-two","MSE: HP-one","SSA","HP-C")

par(mfrow=c(3,1))
# Select target and MSE
select_vec<-c(2,3)
plot(mplot[,select_vec[1]],main="Zero Crossings MSE",axes=F,type="l",xlab="",ylab="",col=colo[select_vec[1]],lwd=1,ylim=c(-0.013,0.006))
abline(v=1+which(sign(mplot[2:nrow(mplot),select_vec[2]])!=sign(mplot[1:(nrow(mplot)-1),select_vec[2]])),col=colo[select_vec[2]],lwd=1,lty=2)
abline(v=1+which(sign(mplot[2:nrow(mplot),select_vec[2]])!=sign(mplot[1:(nrow(mplot)-1),select_vec[2]])),col="black",lwd=1,lty=2)
abline(h=0)
for (i in 1:length(select_vec))
{
  lines(mplot[,select_vec[i]],col=colo[select_vec[i]])
  mtext(colnames(mplot)[select_vec[i]],line=-i,col=colo[select_vec[i]])
}
axis(1,at=1:nrow(mplot),labels=index(y_xts)[(anf+1):length(y_xts)])
axis(2)
box()

# Select target and HP-C
select_vec<-c(2,5)
plot(mplot[,select_vec[1]],main="Zero Crossings SSA",axes=F,type="l",xlab="",ylab="",col=colo[select_vec[1]],lwd=1,ylim=c(-0.013,0.006))
abline(v=1+which(sign(mplot[2:nrow(mplot),select_vec[2]])!=sign(mplot[1:(nrow(mplot)-1),select_vec[2]])),col=colo[select_vec[2]],lwd=1,lty=2)
abline(v=1+which(sign(mplot[2:nrow(mplot),select_vec[2]])!=sign(mplot[1:(nrow(mplot)-1),select_vec[2]])),col="black",lwd=1,lty=2)
abline(h=0)
for (i in 1:length(select_vec))
{
  lines(mplot[,select_vec[i]],col=colo[select_vec[i]])
  mtext(colnames(mplot)[select_vec[i]],line=-i,col=colo[select_vec[i]])
}
axis(1,at=1:nrow(mplot),labels=index(y_xts)[(anf+1):length(y_xts)])
axis(2)
box()

# Select target and SSA
select_vec<-c(2,4)
plot(mplot[,select_vec[1]],main="Zero Crossings SSA",axes=F,type="l",xlab="",ylab="",col=colo[select_vec[1]],lwd=1,ylim=c(-0.013,0.006))
abline(v=1+which(sign(mplot[2:nrow(mplot),select_vec[2]])!=sign(mplot[1:(nrow(mplot)-1),select_vec[2]])),col=colo[select_vec[2]],lwd=1,lty=2)
abline(v=1+which(sign(mplot[2:nrow(mplot),select_vec[2]])!=sign(mplot[1:(nrow(mplot)-1),select_vec[2]])),col="black",lwd=1,lty=2)
abline(h=0)
for (i in 1:length(select_vec))
{
  lines(mplot[,select_vec[i]],col=colo[select_vec[i]])
  mtext(colnames(mplot)[select_vec[i]],line=-i,col=colo[select_vec[i]])
}
axis(1,at=1:nrow(mplot),labels=index(y_xts)[(anf+1):length(y_xts)])
axis(2)
box()


#----------------------------------------
# Sample performances: we expect SSA to equal HP-C in terms of smoothness and to outperform in terms of MSE performances; 

# MSE with respect to two-sided target
mean((y_target-y_mse)^2,na.rm=T)
mean((y_target-y_ssa)^2,na.rm=T)
mean((y_target-y_hp_concurrent)^2,na.rm=T)
# MSE with of SSA with respect to one-sided MSE predcitor (this should match bk_obj$mse_yz)
mean((y_mse-y_ssa)^2,na.rm=T)
bk_obj$mse_yz
# Compute and compare empirical and theoretical holding times
# The holding times concern first differences of the series
compute_empirical_ht_func(diff(y_mse)[anf:enf])$empirical_ht
compute_holding_time_from_rho_func(rho_mse)$ht
compute_empirical_ht_func(diff(y_ssa)[anf:enf])$empirical_ht
compute_holding_time_from_rho_func(bk_obj$rho_yy)$ht
compute_empirical_ht_func(diff(y_hp_concurrent)[anf:enf])$empirical_ht
compute_holding_time_from_rho_func(rho_hp_concurrent)$ht
# Target
compute_empirical_ht_func(diff(y_target))$empirical_ht

mat_perf<-matrix(nrow=2,ncol=3)
mat_perf[1,]<-c(mean((y_target-y_mse)^2,na.rm=T),mean((y_target-y_ssa)^2,na.rm=T),mean((y_target-y_hp_concurrent)^2,na.rm=T))
mat_perf[2,]<-c(compute_empirical_ht_func(diff(y_mse)[anf:enf])$empirical_ht,compute_empirical_ht_func(diff(y_ssa)[anf:enf])$empirical_ht,compute_empirical_ht_func(diff(y_hp_concurrent)[anf:enf])$empirical_ht)
colnames(mat_perf)<-c("MSE nowcast","SSA","HP-C")
rownames(mat_perf)<-c("Sample mean square error","Sample holding time")

# All performances at a glance:
mat_perf

# Summary/findings:
# -MSE-nowcast has smallest MSE, by design; but it is very noisy (small holding-time), see previous plot, top panel
# -The classic one-sided HP-C is much smoother but its MSE is twice as large as MSE-nowcast
# -SSA is as smooth as HP-C and its MSE is halfway between MSE-nowcast and HP-C
# -We conclude that SSA can gain substantially in terms of smoothness (ten times larger holding time) over MSE without loosing 
#   excessively in terms of MSE: optimal tradeoff ensured by SSA criterion 

