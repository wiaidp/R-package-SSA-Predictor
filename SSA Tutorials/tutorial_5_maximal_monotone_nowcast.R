# Tutorial on maximal monotone predictors for non-stationary (I(1)) time series

# Previous tutorials 1-4 are based on Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# This (new) tutorial 5 is based on Wildi (2025). The new paper proposes: 
#   -Technical/formal background to Wildi (2024)
#   -Introduction of a new SSA predictor for non-stationary time series

# For non-stationary time series, an extension of SSA is proposed which matches the target optimally in a MSE sense
#   (predictor and target are cointegrated) and which controls the rate of zero-crossings of (stationary) 
#   FIRST DIFFERENCES of the predictor.
# The predictor tracks the target as closely as possible subject to a holding-time constraint imposed to its first differences.
#   -In its dual form: the predictor generates the least number of zero-crossings in first differences.
#   -Equivalently: the original predictor (in levels) is maximal monotone

# Application: we compute a maximal monotone SSA nowcast for the two-sided (bi-infinite) HP(14400) applied to monthly US-INDPRO.


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

L<-101
lambda_hp<-14400
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

####################################################################
# The novel maximal monotone predictor relies on a cointegration constraint such that the MSE between predictor and target 
#   is finite, see Wildi 2025 for background.
# For this purpose we need to compute particular system matrices
# We assume the differenced data (INDPRO) to be an AR(1): see ACF above.
a1<-0.3

filter_obj<-compute_system_filters_func(L,lambda_hp,a1)

# Cointegration matrix
B=filter_obj$B
# Autocovariance generating matrix
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
# Wold decomposition: depends on a1
Xi=filter_obj$Xi
# Two-sided target: used for computing sample MSEs below
hp_two=filter_obj$hp_two
# Classic concurrent HP
hp_trend=filter_obj$hp_trend

# Specify rho1 in HT constraint, see Wildi (2025) section 5.4
# 1. Compute rho_mse
#   Use MA-inversion, i.e., gamma_mse=Xi%*%gamma: this is applied to stationary first differences 
rho_mse<-compute_holding_time_func(Xi%*%gamma_mse)$rho_ff1
# We also use the MA-inversion Xi%*%hp_trend for deriving the lag-one ACF of HP-C
rho_hp_concurrent<-as.double(compute_holding_time_func(Xi%*%hp_trend)$rho_ff1)

rho_mse
# Set rho1=rho_hp_concurrent in HT constraint
# Research question/assumption: impose same smoothness as HP-C; should outperform HP-C in terms of MSE; without loosing too much vs. gamma_mse 
rho1<-rho_hp_concurrent


#----------------------------
# Compute SSA solution
#   Use optim to determine optimal lambda numerically
#   Initialize lambda with 0: MSE benchmark (numerical optimization must improve upon MSE-initialization)
lambda<-0
opt_obj<-optim(lambda,b_optim,lambda,gamma_mse,Xi,Sigma,Xi_tilde,M,B,gamma_tilde)

lambda_opt<-opt_obj$par

# Compute cointegrated I(1) SSA solution for the optimal Lagrangian parameter lambda_opt
bk_obj<-bk_int_func(lambda_opt,gamma_mse,Xi,Sigma,Xi_tilde,M,B,gamma_tilde)

# Lag-one ACF: should match HT constraint
bk_obj$rho_yy
# Correlation with target (based on synthetic stationary series)
bk_obj$rho_yz
# MSE: with respect to causal MSE predictor
bk_obj$mse_yz
# b applied to x
b_x<-bk_obj$b_x
# b applied to eps in synthetic series
b_eps<-bk_obj$b_eps

# If Xi=Identity then b_x=b_eps
ts.plot(b_eps)
ts.plot(b_x)

# Check cointegration constraint: difference should vanish
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



if (F)
{
  mplot<-cbind(hp_two,c(gamma_mse,rep(0,L-1)),c(b_x,rep(0,L-1)),c(hp_trend,rep(0,L-1)))
  as_obj<-amp_shift_func(600,mplot[,4],F)
}


y_ssa<-filter(x_tilde,b_x,side=1)
y_hp_concurrent<-filter(x_tilde,hp_trend,side=1)
y_mse<-filter(x_tilde,gamma_mse,side=1)
y_target<-filter(x_tilde,hp_two,side=2)
colo<-c("black","violet","green","blue","red")

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
colnames(mat_perf)<-c("MSE","SSA","HP-C")
rownames(mat_perf)<-c("Sample mean square error","Sample holding time")

mat_perf
