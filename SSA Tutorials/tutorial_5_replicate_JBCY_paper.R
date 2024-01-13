
# The present tutorial replicates Figures and Tables in the JBCY-paper

#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions
rm(list=ls())

library(tis)

library(xts)
# Load the library mFilter
# HP and BK filters
library(mFilter)

library(ggplot2)
library("gplots")



# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

# Additional functions used in JBCY paper
# Some of them apply filters to data (can work with xts data!)
# Others are setting-up specialized plots used in section 4: SSA and HP applied to monthly INDPRO 
#   (for example plots with shadings of NBER-recessions)
source(paste(getwd(),"/R/additional_filter_plot_functions_for_JBCY_paper.r",sep=""))
#-------------------------------------------------------
###################################################
### Replicate section figure 1 in 2.2
###################################################
# Compute HP and B-K filters and compare concurrent MSE and SSA designs
# Filter length
L<-200
# 1. HP
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)
# Bi-infinite HP
hp_target=HP_obj$target
ts.plot(hp_target)
# Concurrent gap: as applied to series in levels
hp_gap=HP_obj$hp_gap
ts.plot(hp_gap)
# Concurrent gap: as applied to series in differences
modified_hp_gap=HP_obj$modified_hp_gap
ts.plot(modified_hp_gap)
# Concurrent HP assuming I(2)-process (Classic concurrent, see McElroy)
hp_trend=HP_obj$hp_trend
# Concurrent MSE estimate of bi-infinite HP assuming white noise (truncate symmetric filter)
hp_mse=HP_obj$hp_mse

# Compute lag one acf and holding time of HP concurrent
htrho_obj<-compute_holding_time_func(hp_trend)
rho_hp<-htrho_obj$rho_ff1
ht_hp<-htrho_obj$ht
ht_hp_two<-compute_holding_time_func(hp_target)$ht



#---------------------
# 2. BK-filter
# Need to provide a fake series in order to obtain filter coefficients
len<-1200
x<-1:len
x<-ts(x,frequency=96)
# Bandpass: between 6 and 10 years i.e. pl=72, pu=120
pl=24;pu=120
filt_obj<-mFilter(x,filter="BK",pl=pl,pu=pu,drift=T)
# BK-filter matrix
filt<-filt_obj$fmatrix

filt[(len/2-L/2):(len/2+L/2-1),len/2]

# Symmetric filter
bk_target<-filt[(len/2-L/2):(len/2+L/2-1),len/2]
# bandpass: sums to zero
sum(bk_target)
# Concurrent filter assuming white noise (truncate symmetric filter)
fil_c<-filt[(len/2):(len/2+L),len/2]
sum(fil_c)
bk_mse<-fil_c[1:L]

# Compute lag one acf and holding time of BK concurrent
htrho_obj<-compute_holding_time_func(bk_mse)
rho_bk<-htrho_obj$rho_ff1
ht_bk<-htrho_obj$ht



#---------------------------
# 3. SSA and hyperparameters
# Holding time
ht<-12# This is the corresponding lag-one ACF
rho1<-compute_rho_from_ht(ht)

# Forecast horizon
forecast_horizon_vec<-0
# Size of discret grid for computing nu
grid_size<-200
# This is used for fast numerical computations: the interval is iteratively splitted into halves 
# The precision will be 1/2^split_grid i.e. for split_grid>10 estimates are virtually identical 
# It is very fast because there are only split_grid numerical steps/iterations
# However it assumes uniqueness of nu as a function of ht (or rho), see technical JTSE paper for background
split_grid<-20
# SSA of HP-target
SSA_obj<-SSA_func(L,forecast_horizon_vec,hp_mse,rho1)
SSA_filt_HP<-ssa_eps<-SSA_obj$ssa_eps
# SSA of BK target
SSA_obj<-SSA_func(L,forecast_horizon_vec,bk_mse,rho1)
SSA_filt_BK<-ssa_eps<-SSA_obj$ssa_eps


#################################################
# Figure 1
##################################################

colo<-c("black","brown","blue")
par(mfrow=c(2,2))
mplot<-scale(cbind(hp_target,hp_trend),center=F,scale=T)
colnames(mplot)<-c("Symmetric","Concurrent")

plot(mplot[,1],main=paste("HP(",lambda_monthly,")",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=2)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

mplot<-scale(cbind(SSA_filt_HP,hp_trend),center=F,scale=T)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon_vec,")",sep=""),"HP")

plot(mplot[,1],main=paste("Concurrent",sep=""),axes=F,type="l",xlab="",ylab="",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[3],lwd=2)
mtext(colnames(mplot)[1],col=colo[3],line=-1)
lines(mplot[,2],col=colo[2],lwd=2)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()



mplot<-cbind(bk_target,bk_mse)
colnames(mplot)<-c("Symmetric","Concurrent-MSE")

plot(mplot[,1],main=paste("BK(",pl,",",pu,")",sep=""),axes=F,type="l",xlab="",ylab="",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=2)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


mplot<-scale(cbind(SSA_filt_BK,bk_mse),center=F,scale=T)
colnames(mplot)<-c(paste("SSA(",ht,",",forecast_horizon_vec,")",sep=""),"BK")

plot(mplot[,1],main=paste("Concurrrent",sep=""),axes=F,type="l",xlab="",ylab="",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[3],lwd=2)
mtext(colnames(mplot)[1],col=colo[3],line=-1)
lines(mplot[,2],col=colo[2],lwd=2)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()





###################################################
### Replicate figure 2 in section 2.3
###################################################
# Compute HP and B-K filters and compare concurrent MSE and SSA designs
# Filter length
L<-200
# 1. HP
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)
# Bi-infinite HP
hp_target=HP_obj$target
ts.plot(hp_target)
# Concurrent gap: as applied to series in levels
hp_gap=HP_obj$hp_gap
ts.plot(hp_gap)
# Concurrent gap: as applied to series in differences
modified_hp_gap=HP_obj$modified_hp_gap
ts.plot(modified_hp_gap)
# Concurrent HP assuming I(2)-process (Classic concurrent, see McElroy)
hp_trend=HP_obj$hp_trend
ts.plot(hp_trend)
# Concurrent MSE estimate of bi-infinite HP assuming white noise (truncate symmetric filter)
hp_mse=HP_obj$hp_mse
ts.plot(hp_mse)

# Compute lag one acf and holding time of HP concurrent
htrho_obj<-compute_holding_time_func(hp_trend)
rho_hp<-htrho_obj$rho_ff1
ht_hp<-htrho_obj$ht

#----------------------------
# Generate data: AR(1)
len<-12000
a1<-0.6
set.seed(2)
x<-arima.sim(list(ar=a1),n=len)
acf(x,type="partial")

#------------------------------------
# Convolution target and xi
# Use true ar1 or estimated one
use_empirical_a1<-F

a1f<-ifelse(use_empirical_a1,arima(x,order=c(1,0,0),include.mean=F)$coef,a1)

xi<-a1f^(1:L)
gamma_target<-hp_trend

conv<-xi
for (i in 1:L)
{
  conv[i]<-xi[1:i]%*%gamma_target[i:1]
}  

ts.plot(conv)
ht_conv_pos<-compute_holding_time_func(conv)$ht
ht_target<-compute_holding_time_func(gamma_target)$ht

#---------------------
# 2. SSA-design: positive a1
# Holding time
ht<-12
rho1<-compute_rho_from_ht(ht)$rho
# Forecast horizon: nowcast
delta<-0
# Account for AR(1)-DGP: if xi is omitted then it is assumed that data is white noise
xi<-xi

SSA_obj<-SSA_func(L,delta,gamma_target,rho1,xi)

ssa_eps<-SSA_obj$ssa_eps
ssa_x<-SSA_obj$ssa_x

if (F)
{  
  # Compare theoretical (expected) and empirical holding times based on effective zero-crossings of filter outputs
  # Expected ht
  ht
  # Compute filter output
  filt_obj<-SSA_filter_func(ssa_x,L,x)
  yhat<-filt_obj$y_mat
  # Compute mean holding time: should match ht_conv above (subject to sampling error...)
  length(yhat)/length(which(sign(yhat[1:(length(yhat)-1)]*yhat[2:(length(yhat))])<0))
}
#-----------------------------------------------------------
# 3. SSA-design: negative a1
a1f<--ifelse(use_empirical_a1,arima(x,order=c(1,0,0),include.mean=F)$coef,a1)
xi<--(a1f^(1:L))
conv<-xi
for (i in 1:L)
{
  conv[i]<-xi[1:i]%*%gamma_target[i:1]
}  

ts.plot(conv)
ht_conv_neg<-compute_holding_time_func(conv)$ht

SSA_obj<-SSA_func(L,delta,gamma_target,rho1,xi)


ssa_eps_neg<-SSA_obj$ssa_eps
ssa_x_neg<-SSA_obj$ssa_x



#############################################
# Figure 2
#############################################

par(mfrow=c(2,2))

colo<-c("blue","red","darkgreen")

mplot<-scale(cbind(SSA_filt_HP,ssa_x,ssa_x_neg),center=F,scale=T)
colnames(mplot)<-c(paste("White noise",sep=""),paste("AR(1): a1=",round(abs(a1f),2),sep=""),paste("AR(1): a1=",round((a1f),2),sep=""))

plot(mplot[,1],main=paste("SSA(",ht,",",delta,"): white noise vs. AR(1) ",sep=""),axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
lines(mplot[,3],col=colo[3],lwd=2,lty=1)
mtext(colnames(mplot)[3],col=colo[3],line=-3)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()
# Magnify early lags
mplot_short<-mplot[1:20,]
plot(mplot_short[,1],main="Early lags",axes=F,type="l",xlab="",ylab="",ylim=c(min(na.exclude(mplot_short)),max(na.exclude(mplot_short))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
lines(mplot_short[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot)[2],col=colo[2],line=-2)
lines(mplot_short[,3],col=colo[3],lwd=2,lty=1)
mtext(colnames(mplot)[3],col=colo[3],line=-3)
axis(1,at=1:nrow(mplot_short),labels=-1+1:nrow(mplot_short))
axis(2)
box()
# Amplitude functions
K<-600
mplot_amp<-cbind(amp_shift_func(K,mplot[,1],F)$amp,amp_shift_func(K,mplot[,2],F)$amp,amp_shift_func(K,mplot[,3],F)$amp)
colnames(mplot_amp)<-c(paste("White noise",sep=""),paste("AR(1): a1=",round(abs(a1f),2),sep=""),paste("AR(1): a1=",round(a1f,2),sep=""))
plot(mplot_amp[,1],main="Amplitude functions",axes=F,type="l",xlab="Frequency",ylab="Amplitude",ylim=c(min(na.exclude(mplot_amp)),max(na.exclude(mplot_amp))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot_amp)[1],col=colo[1],line=-1)
lines(mplot_amp[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot_amp)[2],col=colo[2],line=-2)
lines(mplot_amp[,3],col=colo[3],lwd=2,lty=1)
mtext(colnames(mplot_amp)[3],col=colo[3],line=-3)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# Magnify high-frequencies
mplot_amp_short<-mplot_amp[1+as.integer(5*K/6):K,]
plot(mplot_amp_short[,1],main="High frequencies",axes=F,type="l",xlab="",ylab="",ylim=c(0,max(na.exclude(mplot_amp_short))),col=colo[1],lwd=2,lty=1)
mtext(colnames(mplot_amp_short)[1],col=colo[1],line=-1)
lines(mplot_amp_short[,2],col=colo[2],lwd=2,lty=1)
mtext(colnames(mplot_amp_short)[2],col=colo[2],line=-2)
lines(mplot_amp_short[,3],col=colo[3],lwd=2,lty=1)
mtext(colnames(mplot_amp_short)[3],col=colo[3],line=-3)
axis(1,at=c(1,K/6),labels=expression(5*pi/6,pi))
axis(2)
box()





###################################################
### Replicate tables 2, 3 and 4
###################################################
# Compute finite sample estimates of a1 and compare difference of (theoretical) holding times based on true/estimated parameters

# Computationally intensive because the number of simution runs is high (anzsim<-100000): 
# if recompute_results==T then the computations are made

recompute_results<-F
a1_vec<-c(-0.9,-0.5,-0.2,0.0001,0.2,0.5,0.9)

if (recompute_results)
{  
  #----------------------------
  # Generate data: AR(1)
  len<-120
  anzsim<-100000
  set.seed(2)

  ht_emp_vec<-1:anzsim
  mean_vec_HP<-sd_vec_HP<-ht_vec_true_HP<-mean_vec_SSA<-sd_vec_SSA<-ht_vec_true_SSA<-a1_vec
  
  for (k in 1:length(a1_vec))#k<-1
  { 
    print(k)
  
    a1<-a1_vec[k]
  # Compute (theoretical) ht of true AR(1) for HP
    xi<-a1^(1:(L))
    conv_HP<-xi
    for (i in 1:L)
    {
      conv_HP[i]<-xi[1:i]%*%gamma_target[i:1]
    }  
    ht_vec_true_HP[k]<-compute_holding_time_func(conv_HP)$ht
  # Compute (theoretical) ht of true AR(1) for SSA(12,0)
    conv_SSA<-xi
    for (i in 1:L)
    {
      conv_SSA[i]<-xi[1:i]%*%SSA_filt_HP[i:1]
    }  
    ht_vec_true_SSA[k]<-compute_holding_time_func(conv_SSA)$ht
    ht_emp_vec_HP<-ht_emp_vec_SSA<-rep(NA,L)
    for (j in 1:anzsim)
    {  
      # Simulate data  
      x<-arima.sim(list(ar=a1),n=len)
      # Estimate a1  
      a1e<-arima(x,order=c(1,0,0),include.mean=F)$coef
      xi<-a1e^(1:(L))
    # Compute holding time of HP based on estimated a1
      conv_HP<-xi
      for (i in 1:L)
      {
        conv_HP[i]<-xi[1:i]%*%gamma_target[i:1]
      }  
      ht_emp_vec_HP[j]<-compute_holding_time_func(conv_HP)$ht
  # Compute ht of SSA(12,0) based on estimated a1
      conv_SSA<-xi
      for (i in 1:L)
      {
        conv_SSA[i]<-xi[1:i]%*%SSA_filt_HP[i:1]
      }  
      ht_emp_vec_SSA[j]<-compute_holding_time_func(conv_SSA)$ht
  
    }
      
    mean_vec_HP[k]<-mean(ht_emp_vec_HP)
    sd_vec_HP[k]<-sd(ht_emp_vec_HP)
    mean_vec_SSA[k]<-mean(ht_emp_vec_SSA)
    sd_vec_SSA[k]<-sd(ht_emp_vec_SSA)
  }
  list_ht<-list(ht_vec_true_HP=ht_vec_true_HP,mean_vec_HP=mean_vec_HP,sd_vec_HP=sd_vec_HP,
                  ht_vec_true_SSA=ht_vec_true_SSA,mean_vec_SSA=mean_vec_SSA,sd_vec_SSA=sd_vec_SSA)
  save(list_ht,file=paste(getwd(),"/Data/list_ht",sep=""))
  
} else
{
  load(file=paste(getwd(),"/Data/list_ht",sep=""))
  ht_vec_true_HP=list_ht$ht_vec_true_HP
  mean_vec_HP=list_ht$mean_vec_HP
  sd_vec_HP=list_ht$sd_vec_HP
  ht_vec_true_SSA=list_ht$ht_vec_true_SSA
  mean_vec_SSA=list_ht$mean_vec_SSA
  sd_vec_SSA=list_ht$sd_vec_SSA

}

mat<-rbind(round(ht_vec_true_HP,2),round(mean_vec_HP,2),round(sd_vec_HP,2),
           round(ht_vec_true_SSA,2),round(mean_vec_SSA,2),round(sd_vec_SSA,2))
colnames(mat)<-paste("a1=",round(a1_vec,2),sep="")
rownames(mat)<-c("ht HP true a1", "mean empirical ht HP", "sd empirical ht HP",
                 "ht SSA(12,0) true a1", "mean empirical SSA(12,0)", "sd empirical SSA(12,0)")





rho_fixed<-compute_rho_from_ht(12)

recompute_results<-F
if (recompute_results)
{  
  #----------------------------
  # Generate data: AR(1)
  len<-120
  anzsim<-1000
  a1_vec<-c(-0.9,-0.5,-0.2,0.0001,0.2,0.5,0.9)
  set.seed(2)

  ht_emp_vec<-1:anzsim
  mean_vec_SSA<-sd_vec_SSA<-ht_emp_vec_SSA<-a1_vec
  
  for (k in 1:length(a1_vec))#k<-1
  { 
    print(k)
    
    a1<-a1_vec[k]
    for (j in 1:anzsim)
    { 
#      print(j)
      # Simulate data  
      x<-arima.sim(list(ar=a1),n=len)
      # Estimate a1  
      a1e<-arima(x,order=c(1,0,0),include.mean=F)$coef
      xi<-a1e^(1:(L))
      # Compute holding time of HP based on estimated a1
      SSA_obj<-SSA_func(L,forecast_horizon_vec,gamma_target,rho_fixed,xi)
      
# Compute ht of empirical SSA(12,0): convolve ssa_x with true DGP based on a1
      SSA_conv<-conv_two_filt_func(a1^(1:(L)),SSA_obj$ssa_x)$conv
      ht_emp_vec_SSA[j]<-compute_holding_time_func(SSA_conv)$ht
      
    }
    
    mean_vec_SSA[k]<-mean(ht_emp_vec_SSA)
    sd_vec_SSA[k]<-sd(ht_emp_vec_SSA)
    sd_vec_SSA/sqrt(anzsim)
    round(mean_vec_SSA,1)
  }
  list_ht<-list(mean_vec_SSA=mean_vec_SSA,sd_vec_SSA=sd_vec_SSA,ht_emp_vec_SSA=ht_emp_vec_SSA)
  
  save(list_ht,file=paste(getwd(),"/Data/list_ht_empirical_ssa",sep=""))
  
} else
{
  load(file=paste(getwd(),"/Data/list_ht_empirical_ssa",sep=""))
  mean_vec_SSA=list_ht$mean_vec_SSA
  sd_vec_SSA=list_ht$sd_vec_SSA
# This is the realization for the last a1 in a1_vec only  
  ht_emp_vec_SSA=list_ht$ht_emp_vec_SSA
  
}
mat<-rbind(round(mean_vec_SSA,2),round(sd_vec_SSA,2))
colnames(mat)<-c(paste("AR(1)=",round(a1_vec[1],2),sep=""),round(a1_vec,2)[2:length(a1_vec)])
rownames(mat)<-c("mean empirical SSA(12,0)", "sd empirical SSA(12,0)")

mat







#####################################################
# Replicate Fig.3 in section 2.4
##################################################

# Optimal regularization: SSA vs HP (Whittaker Henderson smoothing assuming d=2)

L<-401
HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)
# Bi-infinite HP
hp_target=HP_obj$target

rho1<-compute_holding_time_func(hp_target)$rho_ff1
ht1<-compute_holding_time_func(hp_target)$ht
gamma_target<-1
L<-401
delta<--(L-1)/2
SSA_obj<-SSA_func(L,delta,gamma_target,rho1)

bk_mat<-SSA_obj$ssa_x
par(mfrow=c(1,1))
ts.plot(SSA_obj$ssa_x)
SSA_obj$crit_rhoy_target
SSA_obj$crit_rhoyz


lenq<-100000
set.seed(86)
x<-rnorm(lenq)

# Apply both two-sided filters to x
y_ssa<-filter(x,SSA_obj$ssa_x,side=2)
y_hp_two<-filter(x,hp_target,side=2)
MSE_scale<-as.double(bk_mat[(L+1)/2]/t(bk_mat)%*%bk_mat)

mse_ssa_smmoth<-mean((x-MSE_scale*y_ssa)^2,na.rm=T)
mse_hp_smmoth<-mean((x-y_hp_two)^2,na.rm=T)

filter_mat<-na.exclude(cbind(x,y_ssa,y_hp_two))
cor(filter_mat)

# Find ht such that tracking ability is the same as HP
ht1_1<-75
rho1_1<-compute_rho_from_ht(ht1_1)$rho

SSA_obj_1<-SSA_func(L,delta,gamma_target,rho1_1,xi)

bk_mat_1<-SSA_obj_1$ssa_x
ts.plot(bk_mat_1)
SSA_obj_1$crit_rhoy_target
# This is nearly the same as 
cor(filter_mat)[1,3]

# Compute squared second order differences: all filters scaled to unit variance
mplot<-scale(cbind(bk_mat,bk_mat_1,hp_target),center=F,scale=T)
yhat_mat<-NULL
for (i in 1:ncol(mplot))
  yhat_mat<-cbind(yhat_mat,filter(x,mplot[,i],side=2))
tail(yhat_mat)
colnames(yhat_mat)<-c(paste("SSA(",round(ht1,2),",",delta,")",sep=""),paste("SSA(",round(ht1_1,2),",",delta,")",sep=""),"HP")
# Compute mean squared second order differences: HP minimizes this smoothness measure
apply(apply(apply(na.exclude(yhat_mat),2,diff),2,diff)^2,2,mean)
colo<-c("blue","violet","black")
ts.plot(na.exclude(yhat_mat)[1000:2000,],col=colo)
abline(h=0)

# Compare autocorrelation functions: SSA slowly decaying
acf(na.exclude(yhat_mat)[,1],lag.max = 100)
acf(na.exclude(yhat_mat)[,2],lag.max = 100)
# HP decays faster, it is cyclical with a half-periodicity of ~57 corresponding to its holding-time
acf(na.exclude(yhat_mat)[,3],lag.max = 100)


# Verify empirical holding-times: SSA equals or outperforms HP, as claimed
apply(na.exclude(yhat_mat),2,compute_empirical_ht_func)
# Verify tracking ability: SSA equals or outperforms HP, as claimed
cor(na.exclude(cbind(x,yhat_mat)))[1,]  

# Figure 3
# Compare filter coefficients
par(mfrow=c(1,1))
coloh<-c("blue","Violet","black")
plot(mplot[,1],main=paste("Two-sided SSA and HP Smoothers",sep=""),axes=F,type="l",xlab="",ylab="",ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),col=coloh[1],lwd=2,lty=1)
mtext(paste("SSA(",round(ht1,2),",",delta,")                  ",sep=""),col=coloh[1],line=-1)
mtext(paste("SSA(",round(ht1_1,2),",",delta,")                    ",sep=""),col=coloh[2],line=-2)
lines(mplot[,2],col=coloh[2],lwd=2,lty=1)
lines(mplot[,3],col=coloh[3],lwd=2,lty=1)
mtext(paste("HP(",lambda_monthly,")                       ",sep=""),col=coloh[3],line=-3)
axis(1,at=c(1,50*1:(nrow(mplot)/50)),labels=-((L+1)/2)+1+c(0,50*1:(nrow(mplot)/50)))
axis(2)
box()










######################################################################################################
### Replicate Figs. 4&5 and Tables 5&6 in section 3.2: 
####################################################################################################

ht1<-round((acos(2/3)/pi)^{-1},3)
ht2<-round((acos(1/3)/pi)^{-1},3)
L<-L_short<-20
L_long<-50
ht_large<-10
rho_tt1<-rho_tt1_1<-2/3
# Mean holding time MA(1): this will be larger/smaller than 2 depending on sign of ma1-coeff (if MA(1) is used)
ht_short<-1/(2*(0.25-asin(rho_tt1)/(2*pi)))



target<-rep(1,3)
gamma_mse<-gammak_generic<-rep(1,2)
forecast_horizon<-1
L_short<-20
L_long<-50
rho1<-as.double(compute_holding_time_func(target)$rho_ff1)
ht<-ht_first<-as.double(compute_holding_time_func(target)$ht)
with_negative_lambda<-F
grid_size<-1000

# We can specify either target with forecast horizon 1 or mse with forecast horizon 0
delta<-0
gamma_target<-gamma_mse

SSA_obj<-SSA_func(L_short,delta,gamma_target,rho1)


ssa_eps<-SSA_obj$ssa_eps
colnames(ssa_eps)<-paste("SSA(",round(ht,2),",",forecast_horizon,")",sep="")

ht<-ht_second<-10
rho1<-compute_rho_from_ht(ht)$rho

SSA_obj<-SSA_func(L_short,delta,gamma_target,rho1)

ssa_eps<-cbind(ssa_eps,SSA_obj$ssa_eps)
colnames(ssa_eps)[2]<-paste("SSA(",round(ht,2),",",forecast_horizon,")",sep="")

# Check holding times
compute_holding_time_func(ssa_eps[,1])$ht
compute_holding_time_func(ssa_eps[,2])$ht
# Criterion values (correlations)
t(ssa_eps)%*%c(gamma_mse,rep(0,L_short-2))/sqrt(apply(ssa_eps^2,2,sum)*sum(target^2))

ssa_eps<-rbind(ssa_eps,matrix(rep(0,(L_long-L_short)*ncol(ssa_eps)),ncol=ncol(ssa_eps)))

L_long<-50

SSA_obj<-SSA_func(L_long,delta,gamma_target,rho1)

ssa_eps=cbind(ssa_eps,SSA_obj$ssa_eps)
colnames(ssa_eps)[3]<-paste("SSA(",round(ht,2),",",forecast_horizon,")",sep="")

mplot<-cbind(ssa_eps,c(target,rep(0,L_long-3)),c(gamma_mse,rep(0,L_long-2)))
colnames(mplot)[4:5]<-c("Lag-by-one","MSE")
mplot[,1]<-mplot[,1]*1.441641/mplot[1,1]
mplot[,2]<-mplot[,2]*1.6178706/mplot[1,2]
mplot[,3]<-mplot[,3]*2.264349010/mplot[1,3]

# Figure 4

#mplot<-scale(mplot,scale=T,center=F)
colo<-c("blue","red","violet","black","green")
par(mfrow=c(1,2))
plot(mplot[,1],main="",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights",ylim=c(0,max(na.exclude(mplot))),col=colo[1],lwd=2)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
#  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=2)
  lines(mplot[,i],col=colo[i],lwd=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

mplot<-mplot[1:10,]
plot(mplot[,1],main="",axes=F,type="l",xlab="",ylab="",ylim=c(0,max(na.exclude(mplot))),col=colo[1],lwd=2)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
#  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=2)
  lines(mplot[,i],col=colo[i],lwd=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()



###################################################
### Replicate table 5
###################################################
library(Hmisc)
require(xtable)
# Correlations with z_{t+1}
cor_vec<-ht_vec<-proba_vec<-NULL
for (i in 1:ncol(mplot))
{
  cor_vec<-c(cor_vec,  (mplot[1,i]+mplot[2,i])/(sqrt(3)*sqrt(sum(mplot[,i]^2,na.rm=T))))
  ht_vec<-c(ht_vec,compute_holding_time_func(mplot[,i])$ht)
  proba_vec<-c(proba_vec,1-(2*(0.25-asin(cor_vec[length(cor_vec)])/(2*pi))))
}



mat_re<-rbind(cor_vec,ht_vec,proba_vec)
rownames(mat_re)<-c("Correlation with target","Empirical holding times","Empirical sign accuracy")
colnames(mat_re)[4]<-"Lag-by-one"
colnames(mat_re)[3]<-"SSA(10,1)-long"
mat1<-round(mat_re,3)
mat1
#latex(cor_vec, dec = 1, , caption = "Example of using latex to create table",
#center = "centering", file = "", floating = FALSE)

###################################################
### Replicate table 6
###################################################

ht_vec<-c((8:12)/2,7:10)
ssa_eps<-NULL

for (i in 1:length(ht_vec))
{  
  rho1<-compute_rho_from_ht(ht_vec[i])$rho


  SSA_obj<-SSA_func(L_long,delta,gamma_target,rho1)
  ssa_eps=cbind(ssa_eps,SSA_obj$ssa_eps)
}  
colnames(ssa_eps)<-paste("ht=",ht_vec,sep="")

mplot<-ssa_eps

# Correlations with z_{t+1}
cor_vec<-ht_vec<-proba_vec<-NULL
for (i in 1:ncol(mplot))
{
  cor_vec<-c(cor_vec,  (mplot[1,i]+mplot[2,i])/(sqrt(3)*sqrt(sum(mplot[,i]^2,na.rm=T))))
  ht_vec<-c(ht_vec,compute_holding_time_func(mplot[,i])$ht)
  proba_vec<-c(proba_vec,1-(2*(0.25-asin(cor_vec[length(cor_vec)])/(2*pi))))
}



mat_re<-rbind(cor_vec,ht_vec,proba_vec)
rownames(mat_re)<-c("Correlation","Emp. ht","Sign accuracy")
colnames(mat_re)<-colnames(ssa_eps)
mat1<-round(mat_re,2)

mat1

###################################################
### Computations for fig.5 
###################################################

# Specify target
gammak_generic<-rep(1/3,3)

ht<-compute_holding_time_func(gammak_generic)$ht

# Compute SSA and MSE for a selection of ht
ht_vec<-seq(max(2,ht/4), 5*ht, by = 0.1)
# Compute SSA and MSE for a selection of forecasts horizons
# Note: we must shift the causal symmetric HP by (L_sym-1)/2 to the left in order to obtain the acausal two-sided target
delta_vec<-0:2
  
pb = txtProgressBar(min = 0, max = length(ht_vec), initial = 0,style=3) 
  
MSE_mat<-target_mat<-rho1_mat<-lambda_mat<-nu_mat<-matrix(ncol=length(delta_vec),nrow=length(ht_vec))
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
# Lag-one acf of optimum: could have been computed directly from ht_vec 
    rho1_mat[i,j]<-SSA_obj_HP$crit_rhoyy
    lambda_mat[i,j]<-SSA_obj_HP$lambda_opt
    nu_mat[i,j]<-lambda_mat[i,j]+1/lambda_mat[i,j]
  }
}
close(pb)
# Row-names correspond to holding-times; column-names are forecast horizons  
rownames(MSE_mat)<-rownames(target_mat)<-rownames(rho1_mat)<-round(ht_vec,2)
# Forecast horizon: we remove the artificial shift (L_sym-1)/2 
colnames(MSE_mat)<-colnames(target_mat)<-colnames(rho1_mat)<-delta_vec
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
# 3. Lag-one acf (this could have been computed directly from ht_vec)
head(rho1_mat)
tail(rho1_mat)
#---------------------------------------

# Heat-map of correlations with acausal (effective) target
heatmap.2(target_mat[nrow(target_mat):1,], dendrogram="none",scale = "none", col = colo,trace = "none", density.info = "none",Rowv = F, Colv = F,ylab="Smoothness: holding time",xlab="Timeliness: forecast horizon",main="Prediction Trilemma")

###################################################
### Replicate fig. 5
###################################################

# Figure 5
select_vec<-1:3
mplot<-target_mat[,select_vec]
coli<-rainbow(length(select_vec))
par(mfrow=c(1,1))
plot(mplot[,1],col=coli[1],ylim=c(min(mplot),max(mplot)),axes=F,type="l",xlab="Holding time",ylab="Correlation",main="Prediction Trilemma")
mtext(paste("Forecast horizon ",colnames(MSE_mat)[select_vec[1]],sep=""),col=coli[1],line=-1)
if (length(select_vec)>1)
  for (i in 2:length(select_vec))
  {
    lines(mplot[,i],col=coli[i])
    mtext(paste("Forecast horizon ",colnames(MSE_mat)[select_vec[i]],sep=""),col=coli[i],line=-i)
  }
cor_val=0.5
abline(h=cor_val)
gt<-NULL
for (i in 1:ncol(mplot))
{  
  ww<-which(mplot[1:(nrow(mplot)-1),i]>cor_val&mplot[2:nrow(mplot),i]<cor_val)
  abline(v=ww,col=coli[i])
  gt<-c(gt,rownames(mplot)[ww])
}
axis(1,at=1:nrow(MSE_mat),labels=rownames(MSE_mat))
axis(2)
box()






###########################################################################################
##########################################################################################
# Replicate section 4: application of SSA to HP
###########################################################################################
##########################################################################################
# Part 1. Load and select data
data_obj<-data_load_func(getwd())
# Data from FRED
# Remove fourth series (INDPRO not seasonally adjusted: all other series are adjusted)
indpro<-data_obj$indpro[,-4]
indpro_level<-data_obj$indpro_level
# Second data: from OECD, seasonally adjusted 
# Data is shorter than FRED: for countries appearing in both data sets the longer series in FRED are selected 
indpro_euh<-data_obj$indpro_eu
# Remove Spain and Japan which are contained in indpro (longer series there)
remove_series<-which(colnames(indpro_euh)%in%c("Japan","Spain"))
indpro_eu<-indpro_euh[,-remove_series]

# Select INDPRO and make xts object
select_series<-"US"
series_level<-indpro_level[,select_series]
series<-indpro[,select_series]
plot(series)





# Part 2. HP and hyperparameter
L<-200
lambda_monthly<-14400

HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)

target=HP_obj$target
hp_gap=HP_obj$hp_gap
modified_hp_gap=HP_obj$modified_hp_gap
# Concurrent HP assuming I(2)-process
hp_trend=HP_obj$hp_trend
# MSE estimate of bi-infinite HP assuming white noise
hp_mse=HP_obj$hp_mse
#---------------------------
# Part 3. SSA and hyperparameters
# Holding time
ht<-12
rho1<-compute_rho_from_ht(ht)
forecast_horizon_vec<-c(0,18)
# White noise assumption: MA1_adjustment<-F (MA1_adjustment<-T is not used and should be checked) 
MA1_adjustment<-F
# Size of discret grid for computing nu
grid_size<-200

# Computations are done if recompute_calculations==T (takes a couple seconds). Otherwise saved coefficients are loaded from path.result

SSA_obj<-SSA_func(L,forecast_horizon_vec,hp_mse,rho1)

ssa_eps<-SSA_obj$ssa_eps
colnames(ssa_eps)<-paste("SSA(",round(ht,2),",",forecast_horizon_vec,")",sep="")


#mse_forecast<-c(hp_mse[(1+forecast_horizon_vec[2]):L],rep(0,forecast_horizon_vec[2]))
ht_short<-compute_holding_time_func(hp_trend)$ht
rho1_short<-compute_rho_from_ht(ht_short)


# Compute fast SSA-filter with same holding time as HP-trend
SSA_obj<-SSA_func(L,forecast_horizon_vec[2],hp_mse,rho1_short)

ssa_eps1<-SSA_obj$ssa_eps
colnames(ssa_eps1)<-paste("SSA(",round(ht_short,2),",",forecast_horizon_vec[2],")",sep="")

ssa_eps<-cbind(ssa_eps,ssa_eps1)

#####################################################################################
# Now SSA is done: everything else concerns application of HP and SSA to INDPRO!!!!!!
# -The three above SSA filters are applied to US INDPRO as well as to industrial production series of a 
#   selection of countries with sufficiently long and consistent histories
# -The above SSA designs are 'data-free': they assume the data to be white noise
#   -No data-mining: the same filters are applied `as is' to all countries
#   -The white noise assumption is a misspecification: we could improve the SSA-designs by fitting models to the data
#   -But we ignore that feature: performance comparisons are conservative/fair 
#####################################################################################

#----------------------------
# Part 4. Filter and plot series
# 4.1 Filter
# Start date for plots
start_date<-"1970-01-01"
end_date<-NULL
colo_hp_all<-c("brown","red")
colo_SSA<-c("orange","blue","violet")
colo_all<-c(colo_hp_all,colo_SSA)


# 4.2 Specify filters: HP concurrent and SSA filters
filter_mat<-cbind(hp_trend,modified_hp_gap,ssa_eps)
colnames(filter_mat)<-c("HP trend","Modified gap",colnames(ssa_eps))
ts.plot(scale(filter_mat,center=F,scale=T),col=colo_all)

#   Filter data
filter_obj<-SSA_filter_func(filter_mat,L,series)

y_mat=filter_obj$y_mat
ts.plot(scale(y_mat,center=F,scale=T))

# number of crossings
number_cross_all_filters<-rep(NA,ncol(filter_mat))
names(number_cross_all_filters)<-colnames(filter_mat)
for (i in 1:ncol(y_mat))
{
  if (is.xts(y_mat))
  {  
    number_cross_all_filters[i]<-length(which(sign(y_mat[,i])!=sign(lag(y_mat[,i]))))
  } else
  {
    number_cross_all_filters[i]<-length(which(sign(y_mat[1:(nrow(y_mat)-1),i])!=sign(lag(y_mat[2:nrow(y_mat),i]))))
  }
}
number_cross_all_filters
#-------------
# 4.3 Plot 


plot_obj<-plot_paper(y_mat,start_date,end_date,colo_all)
  
q_gap=plot_obj$q_gap
q_trend=plot_obj$q_trend
x_trend=plot_obj$x_trend
y_trend=plot_obj$y_trend
x_gap=plot_obj$x_gap
y_gap=plot_obj$y_gap
# Plot great-lockdown (Pandemy)
start_date_covid<-"2019-01-01"
end_date_covid<-"2021-06-01"
plot_obj<-plot_paper(y_mat,start_date_covid,end_date_covid,colo_all)
  
q_trend_covid=plot_obj$q_trend
x_trend_covid=plot_obj$x_trend
y_trend_covid=plot_obj$y_trend
q_trend_covid
polygon(x_trend_covid, y_trend_covid, xpd = T, col = "grey",density=10)#
q_SSA_covid=plot_obj$q_SSA
x_SSA_covid=plot_obj$x_trend
y_SSA_covid=plot_obj$y_trend
q_SSA_covid
polygon(x_SSA_covid, y_SSA_covid, xpd = T, col = "grey",density=10)#
  
 
start_date_moderation_financial_1<-"1990-01-01"
end_date_moderation_financial_1<-"2002-01-01"
plot_obj<-plot_paper(y_mat,start_date_moderation_financial_1,end_date_moderation_financial_1,colo_all)
  
q_gap_great_moderation_1=plot_obj$q_gap
x_gap_great_moderation_1=plot_obj$x_trend
y_gap_great_moderation_1=plot_obj$y_trend

par(mfrow=c(1,1))
q_gap_great_moderation_1
polygon(x_gap_great_moderation_1, y_gap_great_moderation_1, xpd = T, col = "grey",density=10)#
  
start_date_moderation_financial_2<-"2001-01-01"
end_date_moderation_financial_2<-"2010-01-01"
plot_obj<-plot_paper(y_mat,start_date_moderation_financial_2,end_date_moderation_financial_2,colo_all)
  
q_gap_great_moderation_2=plot_obj$q_gap
x_gap_great_moderation_2=plot_obj$x_trend
y_gap_great_moderation_2=plot_obj$y_trend
par(mfrow=c(1,1))
q_gap_great_moderation_2
polygon(x_gap_great_moderation_2, y_gap_great_moderation_2, xpd = T, col = "grey",density=10)#

# Number recessions occurring after start date

number_recessions<-length(which(nberDates()>as.double(substr(start_date,1,4))*10000))


#------------------------------------
# Part 5. Diagnostics real data: number of Crossings, peak correlation and shift (tau statistic)

# 5.1 SSA(12,18) vs HP trend
# Select competing series: reference filter/series first
#   Reference series determines crossings, see description of tau-statistic in paper (should be smoother)
#   Reference series determines sign of lead/lags of peak-correlation
# Note: should not be an xts-object!!!!
mplot<-as.matrix(y_mat[,c(4,1)])
# Max lead for peak-correlation
max_lead<-41
# Select closest crossings of contender to reference crossings (conservative measure of shift: corresponds to tau-statistic in paper)
last_crossing_or_closest_crossing<-F
# Skip shifts larger than outlier_limit in absolute value: useful when reference filter has additional crossings which do not correspond to contender
outlier_limit<-10

timeliness_obj<-compute_timeliness_func(mplot,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)

cor_peak=timeliness_obj$cor_peak
tau_vec=timeliness_obj$tau_vec
tau_vec_adjusted=timeliness_obj$tau_vec_adjusted
tau=timeliness_obj$tau
tau_adjusted=timeliness_obj$tau_adjusted
t_test=timeliness_obj$t_test
t_test_adjusted=timeliness_obj$t_test_adjusted
number_cross=timeliness_obj$number_cross

tau
tau_adjusted
t_test
number_cross

# 5.2 SSA(12,18) vs HP gap
mplot<-as.matrix(y_mat[,c(4,2)])
# Max lead for peak-correlation
max_lead<-41
# Select closest crossings of contender to reference crossings (conservative measure of shift: corresponds to tau-statistic in paper)
last_crossing_or_closest_crossing<-F

timeliness_obj<-compute_timeliness_func(mplot,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)

tau_gap=timeliness_obj$tau
tau_gap

# 5.3 SSA(12,18) vs SSA(12,0)
mplot<-as.matrix(y_mat[,c(4,3)])
# Max lead for peak-correlation
max_lead<-41
# Select closest crossings of contender to reference crossings (conservative measure of shift: corresponds to tau-statistic in paper)
last_crossing_or_closest_crossing<-F

timeliness_obj<-compute_timeliness_func(mplot,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)

tau_slow=timeliness_obj$tau
tau_slow


# 5.4 SSA(12,18) vs SSA(7.66,18)
mplot<-as.matrix(y_mat[,c(4,5)])
# Max lead for peak-correlation
max_lead<-41
# Select closest crossings of contender to reference crossings (conservative measure of shift: corresponds to tau-statistic in paper)
last_crossing_or_closest_crossing<-F

timeliness_obj<-compute_timeliness_func(mplot,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)

tau_fast=timeliness_obj$tau
tau_fast



#------------------------------------
# Part 6. Diagnostics Gaussian noise: number of Crossings, peak correlation and shift (tau statistic)
len<-100000
set.seed<-(46)
series_Gauss<-rnorm(len)

filter_obj<-SSA_filter_func(filter_mat,L,series_Gauss)

y_mat_Gauss=filter_obj$y_mat

# 6.1 SSA(12,0) vs HP trend
mplot_Gauss<-as.matrix(y_mat_Gauss[,c(3,1)])
# Max lead for peak-correlation
max_lead<-41
# Select closest crossings of contender to reference crossings (conservative measure of shift: corresponds to tau-statistic in paper)
last_crossing_or_closest_crossing<-F

timeliness_obj<-compute_timeliness_func(mplot_Gauss,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)

tau_Gauss_slow=timeliness_obj$tau

tau_Gauss_slow

# 6.2 SSA(12,18) vs HP trend
mplot_Gauss<-as.matrix(y_mat_Gauss[,c(4,1)])
# Max lead for peak-correlation
max_lead<-41
# Select closest crossings of contender to reference crossings (conservative measure of shift: corresponds to tau-statistic in paper)
last_crossing_or_closest_crossing<-F

timeliness_obj<-compute_timeliness_func(mplot_Gauss,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)

tau_Gauss_middle=timeliness_obj$tau

tau_Gauss_middle

# 6.3 SSA(7.66,18) vs HP trend
mplot_Gauss<-as.matrix(y_mat_Gauss[,c(5,1)])
# Max lead for peak-correlation
max_lead<-41
# Select closest crossings of contender to reference crossings (conservative measure of shift: corresponds to tau-statistic in paper)
last_crossing_or_closest_crossing<-F

timeliness_obj<-compute_timeliness_func(mplot_Gauss,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)

tau_Gauss_fast=timeliness_obj$tau

tau_Gauss_fast


ht_trend<-round(compute_holding_time_func(hp_trend)$ht,2)

###########################################################################
# All computations are done
###########################################################################

# We can now replicate figures and tables in section 4



###################################################
### Replicate fig. 6
###################################################

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




###################################################
### Replicate fig. 7
###################################################


mat_coef_hp<-scale(cbind(hp_trend,modified_hp_gap),center=F,scale=T)
mat_coef_SSA<-scale(ssa_eps[,select_vec],center=F,scale=T)

mat_amp_hp<-mat_shift_hp<-mat_amp_SSA<-mat_shift_SSA<-NULL
K<-600
for (i in 1:ncol(mat_coef_hp))
{
  tr_obj_hp<-amp_shift_func(K,mat_coef_hp[,i],F)
  mat_amp_hp<-cbind(mat_amp_hp,tr_obj_hp$amp)  
  mat_shift_hp<-cbind(mat_shift_hp,tr_obj_hp$shift)  
} 
colnames(mat_amp_hp)<-colnames(mat_shift_hp)<-colnames(mat_coef_hp)
for (i in 1:ncol(mat_coef_SSA))
{
  tr_obj_SSA<-amp_shift_func(K,mat_coef_SSA[,i],F)
  mat_amp_SSA<-cbind(mat_amp_SSA,tr_obj_SSA$amp)  
  mat_shift_SSA<-cbind(mat_shift_SSA,tr_obj_SSA$shift)  
} 
colnames(mat_amp_SSA)<-colnames(mat_shift_SSA)<-colnames(mat_coef_SSA)
colo_hp<-colo_hp_all[c(1,4)]

par(mfrow=c(2,2))
mplot<-scale(mat_amp_hp,center=F,scale=T)
#colnames(mplot)<-c("HP trend","HP-gap (modified)","MSE-forecast")

plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP",sep=""),ylim=c(min(mplot),max(mplot)),col=colo_hp[1])
mtext(colnames(mplot)[1],col=colo_hp_all[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo_hp_all[i])
  mtext(colnames(mplot)[i],col=colo_hp_all[i],line=-i)
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()

mplot<-scale(mat_amp_SSA,center=F,scale=T)

plot(mplot[,1],type="l",axes=F,xlab="",ylab="",main=paste("Amplitude SSA",sep=""),ylim=c(min(mplot),max(mplot)),col=colo_SSA[1])
mtext(colnames(mplot)[1],col=colo_SSA[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo_SSA[i])
  mtext(colnames(mplot)[i],col=colo_SSA[i],line=-i)
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()


mplot<-(mat_shift_hp)
#colnames(mplot)<-c("HP trend","Trend MSE","HP-gap (modified)","MSE-forecast")

mplot[1,]<-NA
# skip extreme values (larger than max of other shifts)
#ex_val<-max(na.exclude(mplot[,1:ncol(mplot)]))
#mplot[which(abs(mplot[,1])>ex_val),1]<-NA
plot(mplot[,1],type="l",axes=F,xlab="",ylab="",main=paste("Phase-lag HP",sep=""),
     ylim=c(-1,max(na.exclude(mplot))),col=colo_hp_all[1])
mtext(colnames(mplot)[1],col=colo_hp_all[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo_hp_all[i])
  mtext(colnames(mplot)[i],col=colo_hp_all[i],line=-i)
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

mplot<-(mat_shift_SSA)

mplot[1,]<-NA
# skip extreme values (larger than max of other shifts)
#ex_val<-max(na.exclude(mplot[,2:ncol(mplot)]))
#mplot[which(abs(mplot[,1])>ex_val),1]<-NA
plot(mplot[,1],type="l",axes=F,xlab="",ylab="",main=paste("Phase-lag SSA",sep=""),
     ylim=c(-1,max(na.exclude(mplot))),col=colo_SSA[1])
mtext(colnames(mplot)[1],col=colo_SSA[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo_SSA[i])
  mtext(colnames(mplot)[i],col=colo_SSA[i],line=-i)
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()






###################################################
### Replicate fig.8
###################################################
par(mfrow=c(1,1))
mplot<-cbind(mat_shift_hp[,1],mat_shift_SSA)
colnames(mplot)[1]<-colnames(mat_shift_hp)[1]
mplot[1,]<-NA
# skip extreme values (larger than max of other shifts)
ex_val<-max(na.exclude(mplot[,2:ncol(mplot)]))
mplot[which(abs(mplot[,1])>ex_val),1]<-NA
plot(mplot[,1]-mplot[,2],type="l",axes=F,xlab="Frequency",ylab="Lag                              Lead            ",main=paste("",sep=""),col=colo_SSA[1],ylim=c(-2,6))
for (i in 2:ncol(mplot))
{  
lines(mplot[,1]-mplot[,i],col=colo_SSA[i-1])
mtext(paste("Relative lead/lag of ",colnames(mplot)[1], " over ",colnames(mplot)[i],sep=""),col=colo_SSA[i-1],line=-i)
}
x<-c(as.integer(nrow(mplot)/(10*6)),as.integer(nrow(mplot)/(10*6)),as.integer(nrow(mplot)/(2*6)),as.integer(nrow(mplot)/(2*6)))
#y<-c(min(na.exclude(mplot[,2]-mplot[,3])),max(na.exclude(mplot[,2]-mplot[,3])),max(na.exclude(mplot[,2]-mplot[,3])),min(na.exclude(mplot[,2]-mplot[,3])))
y<-c(-2,6,6,-2)
polygon(x, y, xpd = T, col = "grey",density=10)#, lty = 2, lwd = 2, border = "red")
#mtext("Trend-cycle frequencies",col="grey",line=-3)
abline(h=0,lty=2)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()






###################################################
### Replicate table 7
###################################################
# Compute empirical mean shift: lead/lag
mat_re<-matrix(rbind(as.integer(c(0,round(tau,0),round(tau_gap,0),round(tau_slow,0),round(tau_fast,0))),as.integer(c(number_cross_all_filters[c("SSA(12,18)","HP trend","Modified gap","SSA(12,0)","SSA(7.66,18)")]))),ncol=5)
colnames(mat_re)<-c("SSA(12,18)","HP trend","Modified gap","SSA(12,0)","SSA(7.66,18)")
rownames(mat_re)<-c("Mean-shift","Number of crossings")
#save(mat_re_gap_trend_zcc,file=paste(path.result,"mat_re_gap_trend_zcc"))
index(indpro)[1]
index(indpro)[nrow(indpro)]
mat_sh<-matrix(c(round(tau_Gauss_slow,0),round(tau_Gauss_middle,0),round(tau_Gauss_fast,0)),nrow=1)
colnames(mat_sh)<-colnames(y_mat_Gauss)[c(3,4,5)]
rownames(mat_sh)<-"Mean-shift"

mat_sh



###################################################
### Replicate fig. 9
###################################################
x<-nber_dates_polygon(start_date,series_level)$x
y<-nber_dates_polygon(start_date,series_level)$y
# Adjust rectangles of shaded recession episodes to minimal series value
y[which(y==min(y))]<-min(series_level[paste(start_date,"/",sep="")])
#x<-c(start_date[33],start_date[33],end_date[33],end_date[33])
#y<-c(min(level),max(level),max(level),min(level))
plot((series_level[paste(start_date,"/",sep="")]),#ylim=c(min(series_level[paste(start_date,"/",sep="")]),max(series_level[paste(start_date,"/",sep="")])),
     plot.type='s',col="black",ylab="",main="US industrial production index ")
polygon(x, y, xpd = T, col = "grey",density=10)#, lty = 2, lwd = 2, border = "red")

# Log-returns follow a MA(1)
x<-na.exclude(diff(log(series_level[paste(start_date,"/",sep="")])))
acf(x)
a_obj<-arima(x,order=c(0,0,1))

tsdiag(a_obj)



###################################################
### Replicate fig. 10
###################################################
# Load full data-set
par(mfrow=c(2,1))
q_gap
polygon(x_gap, y_gap, xpd = T, col = "grey",density=10)#
q_trend
polygon(x_trend, y_trend, xpd = T, col = "grey",density=10)#



###################################################
### Replicate fig. 11
###################################################
# Load full data-set
par(mfrow=c(2,2))
q_gap_great_moderation_1
polygon(x_gap_great_moderation_1, y_gap_great_moderation_1, xpd = T, col = "grey",density=10)#
q_trend_covid
polygon(x_trend_covid, y_trend_covid, xpd = T, col = "grey",density=10)#
q_gap_great_moderation_2
polygon(x_gap_great_moderation_2, y_gap_great_moderation_2, xpd = T, col = "grey",density=10)#
q_SSA_covid
polygon(x_SSA_covid, y_SSA_covid, xpd = T, col = "grey",density=10)#



###################################################
### Replicate table 8
###################################################

# Compute empirical mean shift: lead/lag
mat_re<-matrix(rbind(as.integer(c(0,round(tau,0),round(tau_gap,0),round(tau_slow,0),round(tau_fast,0))),as.integer(c(number_cross_all_filters[c("SSA(12,18)","HP trend","Modified gap","SSA(12,0)","SSA(7.66,18)")]))),ncol=5)
colnames(mat_re)<-c("SSA(12,18)","HP trend","Modified gap","SSA(12,0)","SSA(7.66,18)")
rownames(mat_re)<-c("Mean-shift","Number of crossings")
#save(mat_re_gap_trend_zcc,file=paste(path.result,"mat_re_gap_trend_zcc"))
index(indpro)[1]
index(indpro)[nrow(indpro)]

mat_re


############################################################################################
# The remaining computations concern section 4.3: extension of SSA to additional countries
# The SSA filter (used for INDPRO in section 4.2) is left unchanged: it assumes log-returns to be white noise 
# White noise is a misspecification but we ignore data-fitting (-mining) deliberately
# We now replicate tables 9 and 10
##################################################################################################

# Compute timeliness smoothness for all time series

# 1. First data set 
# We initialize table with results for US above i.e. we skip US from indpro: indpro[,-3]
time_cross_mat<-mat_re[,c(1,2,5)]
series_mat<-indpro[,-3]
time_cross_mat<-NULL
number_cross_mid<-number_cross_fast<-matrix(rep(0,2),nrow=1)
series_mat<-indpro
tau_vec_long_mid<-tau_vec_long_fast<-NULL
for (ijk in 1:ncol(series_mat))#ijk<-1
{  
# Select INDPRO and make xts object
  select_series<-ijk
# Remove NAs and fix correct type  
  series<-as.vector(na.exclude(series_mat[,select_series]))
  ts.plot(series)

#   Filter data

  filter_obj<-SSA_filter_func(filter_mat,L,series)

  y_mat=filter_obj$y_mat
  
  ts.plot(scale(y_mat,center=F,scale=T)[,c("SSA(12,18)","HP trend")],col=c("blue","brown"))
  abline(h=0)

#------------------------------------
# Diagnostics real data: number of Crossings, peak correlation and shift (tau statistic)

# 5.1 SSA-mid vs HP trend
# Select competing series: reference filter/series first
#   Reference series determines crossings, see description of tau-statistic in paper (should be smoother)
#   Reference series determines sign of lead/lags of peak-correlation
# Note: should not be an xts-object!!!!
  mplot<-as.matrix(y_mat[,c("SSA(12,18)","HP trend")])
# Max lead for peak-correlation
  max_lead<-41
# Select closest crossings of contender to reference crossings (conservative measure of shift: corresponds to tau-statistic in paper)
  last_crossing_or_closest_crossing<-F
  
  timeliness_obj<-compute_timeliness_func(mplot,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)
  
  cor_peak=timeliness_obj$cor_peak
  tau_vec=timeliness_obj$tau_vec
  tau_vec_adjusted=timeliness_obj$tau_vec_adjusted
# Collect tau summands over all series for statistical test  
  tau_vec_long_mid<-c(tau_vec_long_mid,tau_vec_adjusted)
  tau_trend=timeliness_obj$tau
  tau_adjusted_trend=timeliness_obj$tau_adjusted
  t_test=timeliness_obj$t_test
  t_test_adjusted=timeliness_obj$t_test_adjusted
  number_cross_trend=timeliness_obj$number_cross
  number_cross_mid<-number_cross_mid+number_cross_trend
  tau_trend
  tau_adjusted_trend
  number_cross_trend
  
# 5.2 SSA-mid vs. SSA-fast
  mplot<-as.matrix(y_mat[,c(4,5)])

  timeliness_obj<-compute_timeliness_func(mplot,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)
  
  tau_fast_trend=timeliness_obj$tau
  tau_adjusted_fast_trend=timeliness_obj$tau_adjusted
  tau_vec_fast_trend_adjusted=timeliness_obj$tau_vec_adjusted
# Collect tau summands over all series for statistical test  
  tau_vec_long_fast<-c(tau_vec_long_fast,tau_vec_fast_trend_adjusted)
  number_cross_fast_trend=timeliness_obj$number_cross
  number_cross_fast<-number_cross_fast+number_cross_fast_trend

  tau_fast_trend
  tau_adjusted_fast_trend
  number_cross_fast_trend

  time_cross_mat<-rbind(time_cross_mat,rbind(c(0,as.integer(round(tau_adjusted_trend,0)),as.integer(round(tau_adjusted_fast_trend,0)))),c(as.integer(number_cross_trend[2:1]),as.integer(number_cross_fast_trend[1])))

}

colnames(time_cross_mat)<-colnames(y_mat)[c(4,1,5)]
#rowname<-c("US shift","US number crossings")
rowname<-NULL#c("US shift","US number crossings")
for (i in 1:ncol(series_mat))
  rowname<-c(rowname,paste(colnames(series_mat)[i],"shift"),paste(colnames(series_mat)[i],"number crossings"))
rownames(time_cross_mat)<-rowname
time_cross_mat

time_cross_mat_first_data_set<-time_cross_mat

#-----------------------------------------------------
# 2. Second data set 
colnames(indpro_eu)
series_mat<-indpro_eu

time_cross_mat<-NULL

for (ijk in 1:ncol(indpro_eu))#ijk<-7
{  
# Select INDPRO and make xts object
  select_series<-ijk
# Remove NAs and fix correct type  
  series<-as.vector(na.exclude(indpro_eu[,select_series]))
  ts.plot(series)


  filter_obj<-SSA_filter_func(filter_mat,L,series)

  y_mat=filter_obj$y_mat
  
  ts.plot(scale(y_mat,center=F,scale=T)[,c("SSA(12,18)","HP trend")],col=c("blue","brown"))
  abline(h=0)

#------------------------------------
# Diagnostics real data: number of Crossings, peak correlation and shift (tau statistic)

# 5.1 SSA-mid vs HP trend
# Select competing series: reference filter/series first
#   Reference series determines crossings, see description of tau-statistic in paper (should be smoother)
#   Reference series determines sign of lead/lags of peak-correlation
# Note: should not be an xts-object!!!!
  mplot<-as.matrix(y_mat[,c("SSA(12,18)","HP trend")])
# Max lead for peak-correlation
  max_lead<-41
# Select closest crossings of contender to reference crossings (conservative measure of shift: corresponds to tau-statistic in paper)
  last_crossing_or_closest_crossing<-F
  
  timeliness_obj<-compute_timeliness_func(mplot,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)
  
  cor_peak=timeliness_obj$cor_peak
  tau_vec=timeliness_obj$tau_vec
  tau_vec_adjusted=timeliness_obj$tau_vec_adjusted
# Collect tau summands over all series for statistical test  
  tau_vec_long_mid<-c(tau_vec_long_mid,tau_vec_adjusted)
  tau_trend=timeliness_obj$tau
  tau_adjusted_trend=timeliness_obj$tau_adjusted
  t_test=timeliness_obj$t_test
  t_test_adjusted=timeliness_obj$t_test_adjusted
  number_cross_trend=timeliness_obj$number_cross
  number_cross_mid<-number_cross_mid+number_cross_trend

  tau_trend
  tau_adjusted_trend
  number_cross_trend
  
# 5.2 SSA-mid vs. SSA-fast
  mplot<-as.matrix(y_mat[,c(4,5)])

  timeliness_obj<-compute_timeliness_func(mplot,max_lead,ht,last_crossing_or_closest_crossing,outlier_limit)
  
  tau_fast_trend=timeliness_obj$tau
  tau_adjusted_fast_trend=timeliness_obj$tau_adjusted
  tau_vec_fast_trend_adjusted=timeliness_obj$tau_vec_adjusted
# Collect tau summands over all series for statistical test  
  tau_vec_long_fast<-c(tau_vec_long_fast,tau_vec_fast_trend_adjusted)
  number_cross_fast_trend=timeliness_obj$number_cross
  number_cross_fast<-number_cross_fast+number_cross_fast_trend

  tau_fast_trend
  tau_adjusted_fast_trend
  number_cross_fast_trend

  time_cross_mat<-rbind(time_cross_mat,rbind(c(0,as.integer(round(tau_adjusted_trend,0)),as.integer(round(tau_adjusted_fast_trend,0)))),c(as.integer(number_cross_trend[2:1]),as.integer(number_cross_fast_trend[1])))


}



t_test_mid<-t.test(tau_vec_long_mid,  alternative = "two.sided")$statistic
t_test_fast<-t.test(tau_vec_long_fast,  alternative = "two.sided")$statistic
ts.plot(cumsum(tau_vec_long_mid))
mean(tau_vec_long_mid)
ts.plot(cumsum(tau_vec_long_fast))
mean(tau_vec_long_fast)


colnames(time_cross_mat)<-colnames(y_mat)[c(4,1,5)]
#rowname<-c("US shift","US number crossings")
rowname<-NULL
for (i in 1:ncol(series_mat))
  rowname<-c(rowname,paste(colnames(series_mat)[i],"shift"),paste(colnames(series_mat)[i],"number crossings"))
rownames(time_cross_mat)<-rowname
time_cross_mat

# Concatenate both country sets and perform aggregate mean performances over all countries
time_cross_mat_final<-rbind(time_cross_mat_first_data_set,time_cross_mat)
aggregate_time_cross_mat_final<-rbind(round(c(0,mean(tau_vec_long_mid),mean(tau_vec_long_fast)),2),
                                      c(0,t_test_mid,t_test_fast),
                                     c( as.integer(number_cross_mid[2:1]),as.integer(number_cross_fast[1])))
  
colnames(aggregate_time_cross_mat_final)<-colnames(time_cross_mat)
  

rownames(aggregate_time_cross_mat_final)<-c("Mean-shift over countries","t-test for time-shift","Total number of crossings")
colnames(aggregate_time_cross_mat_final)<-colnames(aggregate_time_cross_mat_final)


###################################################
### Replicate table 9
###################################################
time_cross_mat_final


###################################################
### Replicate table 10
###################################################
aggregate_time_cross_mat_final

###################################################
### Replicate fig. 12
###################################################
par(mfrow=c(1,2))
plot(cumsum(tau_vec_long_mid),type="l",axes=F,xlab="Number of crossings",ylab="Cumulated shift",main=paste(" HP-trend vs. ",colnames(filter_mat)[4],sep=""),col=colo_SSA[2])
#mtext(paste("Relative lead/lag of ",colnames(mplot)[1], " over ",colnames(mplot)[i],sep=""),col=colo_SSA[i-1],line=-i)
axis(1,at=1:length(tau_vec_long_mid),labels=1:length(tau_vec_long_mid))
axis(2)
box()
plot(cumsum(tau_vec_long_fast),type="l",axes=F,xlab="",ylab="",main=paste(colnames(filter_mat)[5]," vs. ",colnames(filter_mat)[4],sep=""),col=colo_SSA[3])
#mtext(paste("Relative lead/lag of ",colnames(mplot)[1], " over ",colnames(mplot)[i],sep=""),col=colo_SSA[i-1],line=-i)
axis(1,at=1:length(tau_vec_long_mid),labels=1:length(tau_vec_long_mid))
axis(2)
box()


