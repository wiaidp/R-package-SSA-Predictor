# M-SSA: extension of univariate to multivariate SSA
# Work in progress

# -The first M-SSA tutorial, this one, is based on a simulation example derived from an application of M-SSA to 
#   predicting German GDP (or BIP)
# -The data generating process here relies on the VAR fitted to German data
# -We here show that M-SSA is optimal (if the data generating is the true process) and that the most relevant 
#   sample performances converge to their expected numbers for sufficiently long samples of (artificial) data

# Clean sheet
rm(list=ls())

# Let's start by loading the required R-libraries

# Standard filter package
library(mFilter)
# Multivariate time series: VARMA model for macro indicators: used here for simulation purposes only
library(MTS)
# HAC estimate of standard deviations in the presence of autocorrelation and heteroscedasticity
library(sandwich)


# Load the relevant M-SSA functionalities
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))
# Utility functions for M-SSA, see tutorial 
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))


#------------------------------------------------------------------------
# Let's apply M-SSA to quarterly German Macro-data

# 1. Load data and select indicators
load(file="C:\\Users\\marca\\OneDrive\\2025\\R-package-SSA-Predictor\\Data\\macro")
# BIP(GDP) has a publication lag of a quarter
# The target column corresponds to a one-step ahead forecast: BIP has a publication lag of one quarter but we 
#   shifted the data in the target columnone additional quarter upwards to be on the safe-side (for example to account for revisions)
# Columns 2-8 are the data available in Jan-2025 for nowcasting the target
# All indicators where log-transformed (except spread), differenced and standardized
#   -Calibration of true levels and variances can be obtained afterwards, by simple linear regression
# Extreme (singular) observations during Pandemic (2019-2020) where trimmed at 3 standard deviations 
#   After discussion this trimming was deemed acceptable (and transparent,reproducible) to avoid overly strong impact of singular data
tail(data)
# We see that the data in the target column (BIP) is shifted upwards two quarters: 
# We here want to nowcast/predict this series given the data in columns 2-7 (explanatory variables)
# Specify the publication lag
lag_vec<-c(2,rep(0,ncol(data)-1))

# Plot the data
# The real-time BIP (red) is lagging the target by one quarter (publication lag)
par(mfrow=c(1,1))
mplot<-data
colo<-c("black",rainbow(ncol(data)-1))
main_title<-paste("Quarterly design BIP: the target (black) assumes a publication lag of ",lag_vec[1]," Quarters",sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(data)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Select macro indicators
select_vec_multi<-c("BIP","ip","ifo_c","ESI","spr_10y_3m")

x_mat<-data[,select_vec_multi] 
rownames(x_mat)<-rownames(data)
n<-dim(x_mat)[2]
# Number of observations
len<-dim(x_mat)[1]
#------------------------------
# 2. Target filter: the two-sided HP will be applied to target column (or equivalently: BIP shifted upward by lag_vec[1] quarters)
lambda_HP<-160
# Filter length: roughly 4 years. The length should be an odd number in order to have a symmetric HP 
#   with a peak in the middle (for even numbers the peak is truncated)
L<-31

target_obj<-HP_target_sym_T(n,lambda_HP,L)

gamma_target=t(target_obj$gamma_target)
symmetric_target=target_obj$symmetric_target 
colnames(gamma_target)<-select_vec_multi


# The targets are one-sided
par(mfrow=c(1,1))
ts.plot(gamma_target,col=rainbow(n))

# But we tell M-SSA to mirror the target filter at its peak value
symmetric_target


#-------------------------
# 3. Fit the VAR
# Select any in-sample span: the effect on the final M-SSA predictor is remarkably weak
#   -The VAR is sparsely parametrized  (p=1 and regularization)
# Set in-sample span: full set
date_to_fit<-"2200"
# Set in-sample span: prior Pandemic
date_to_fit<-"2019"
# Set in-sample span: prior financial crisis
date_to_fit<-"2008"
data_fit<-na.exclude(x_mat[which(rownames(x_mat)<date_to_fit),])#date_to_fit<-"2019-01-01"
# Have a look at cross correlation: in-sample span
acf(data_fit)
# Check span
tail(data_fit)



# VARMA modelling
p<-1
q<-0
set.seed(12)
V_obj<-VARMA(data_fit,p=p,q=q)
# Apply regularization: see vignette MTS package
threshold<-1.5
V_obj<-refVARMA(V_obj, thres = threshold)

# Have a look at diagnostics
if (F)
  MTSdiag(V_obj)
# Sigma
Sigma<-V_obj$Sigma
Phi<-V_obj$Phi
Theta<-V_obj$Theta

#---------------------------------------
# 4. MA inversion

MA_inv_obj<-MA_inv_VAR_func(Phi,Theta,L,n,T)

xi<-MA_inv_obj$xi

#-----------------------------------
# 5. M-SSA function
# One year ahead forecast: 4 quarters + publication lag
delta<-4+lag_vec[1]
# Specify HT constraint: 
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Compute corresponding lag-one ACF in HT constraint: see previous tutorials on the link between HT and lag-one ACF  
rho0<-compute_rho_from_ht(ht_mssa_vec)$rho

MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)

bk_x_mat=MSSA_main_obj$bk_x_mat
MSSA_obj=MSSA_main_obj$MSSA_obj 
# Benchmark MSE predictor
gammak_x_mse<-MSSA_obj$gammak_x_mse

colnames(bk_x_mat)<-colnames(gammak_x_mse)<-select_vec_multi

#-----------------------
# 6. Filter: apply M-SSA filter to data

# Note that delta accounts for publication lag so that output of two-sided filter is leaft shifted accordingly
#   4 quarters+lag_vec[1]
filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)


mssa_mat=filt_obj$mssa_mat
target_mat=filt_obj$target_mat
mmse_mat<-filt_obj$mmse_mat

# Plots: in-sample span is marked by vertical line
for (i in 1:n)
{
  par(mfrow=c(1,1))
  mplot<-cbind(target_mat[,i],mssa_mat[,i],mmse_mat[,i])
  colnames(mplot)<-c(paste("Target: HP applied to ",select_vec_multi[i],", left-shifted by ",delta-lag_vec[1]," quarters",sep=""),"M-SSA","M-MSE")

  colo<-c("black","blue","green")
  main_title<-paste("M-SSA ",select_vec_multi[i],": delta=",delta,", in-sample span ending in ",rownames(data_fit)[nrow(data_fit)],sep="")
  plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(data)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
  mtext(colnames(mplot)[1],col=colo[1],line=-1)
  for (i in 1:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i],lwd=1,lty=1)
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
  abline(h=0)
  abline(v=which(rownames(mplot)==rownames(data_fit)[nrow(data_fit)]),lwd=2,lty=2)
  axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
  axis(2)
  box()
}



#------------------------
# Mean-square errors M-SSA
apply(na.exclude((target_mat-mssa_mat)^2),2,mean)
# Mean-square errors M-MSE: for true models and long samples MSE of M-SSA is larger than MSE of M-MSE benchmark
apply(na.exclude((target_mat-mmse_mat)^2),2,mean)

# Correlations between target and M-SSA: sample estimates converge to criterion value for increasing sample size len, assuming the VAR is true
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[,i],mssa_mat[,i])))[1,2])
# This is the criterion value of M-SSA: the target correlation is maximized under the HT constraint
# Large discrepancy to sample statistics: suggests that VAR(1) is misspecified
MSSA_obj$crit_rhoy_target

# M-SSA optimizes target correlation under holding time constraint:
# Compare empirical and theoretical (imposed) HTs: sample HT converges to imposed HT for increasing sample size len
unlist(apply(mmse_mat,2,compute_empirical_ht_func))
unlist(apply(mssa_mat,2,compute_empirical_ht_func))
ht_mssa_vec
# M-SSA smoother than M-MSE: the following ratios should be smaller for sufficiently long samples if imposed HT of M-SSA is larger than HT of M-MSE benchmark
unlist(apply(mmse_mat,2,compute_empirical_ht_func))/unlist(apply(mssa_mat,2,compute_empirical_ht_func))


################################################################################################################
# We now replicate designs used in macro-design
# -We consider each of the above M-SSA outputs as an equally valid/informative predictor for future BIP
#   -Cross-sectional aggregation: equal weighting
# -However, BIP and ip are lagging, ESI and ifo are coincident and spread is leading
# -Therefore, we select larger delta (forecast excess) for the first two: BIP and ip

# These are the interesting forecast horizons
# We compute a BIP indicator for each of these forecast horizons and we evaluate its performances based on various performance metrics
h_vec<-c(0,1,2,4,6)
# Forecast excesses: we demonstrate a `mildly aggressive' design
f_excess<-c(4,2)
mssa_bip<-mssa_ip<-mssa_esi<-mssa_ifo<-mssa_spread<-NULL
for (i in 1:length(h_vec))#i<-1
{
# BIP and ip require a larger forecast excess. We also add the publication lag
  delta<-h_vec[i]+lag_vec[1]+f_excess[1]
  
  MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)
  
  bk_x_mat=MSSA_main_obj$bk_x_mat
  MSSA_obj=MSSA_main_obj$MSSA_obj 
  # Benchmark MSE predictor
  gammak_x_mse<-MSSA_obj$gammak_x_mse
  colnames(bk_x_mat)<-colnames(gammak_x_mse)<-select_vec_multi
  
  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
  
  mssa_mat=filt_obj$mssa_mat
  target_mat=filt_obj$target_mat
  mmse_mat<-filt_obj$mmse_mat
  colnames(mssa_mat)<-select_vec_multi
  
  
  mssa_bip<-cbind(mssa_bip,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[1])])
  mssa_ip<-cbind(mssa_ip,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[2])])
  
  # ESI, ifo and spread do not require a forecast excess
  delta<-h_vec[i]+lag_vec[1]+f_excess[2]
  
  MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)
  
  bk_x_mat=MSSA_main_obj$bk_x_mat
  MSSA_obj=MSSA_main_obj$MSSA_obj 
  # Benchmark MSE predictor
  gammak_x_mse<-MSSA_obj$gammak_x_mse
  colnames(bk_x_mat)<-colnames(gammak_x_mse)<-select_vec_multi

  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
  
  mssa_mat=filt_obj$mssa_mat
  target_mat=filt_obj$target_mat
  mmse_mat<-filt_obj$mmse_mat
  colnames(mssa_mat)<-select_vec_multi
  
  
  mssa_ifo<-cbind(mssa_ifo,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[3])])
  mssa_esi<-cbind(mssa_esi,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[4])])
  mssa_spread<-cbind(mssa_spread,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[5])])
  
}

indicator_mat<-(scale(mssa_bip)+scale(mssa_ip)+scale(mssa_ifo)+scale(mssa_esi)+scale(mssa_spread))/length(select_vec_multi)

colnames(indicator_mat)<-colnames(mssa_bip)<-colnames(mssa_ip)<-colnames(mssa_ifo)<-colnames(mssa_esi)<-colnames(mssa_spread)<-paste("Horizon=",h_vec,sep="")
rownames(indicator_mat)<-rownames(x_mat)

target_shifted_mat<-NULL
cor_mat<-matrix(ncol=length(h_vec),nrow=length(h_vec))

for (i in 1:length(h_vec))#i<-1
{
  shift<-h_vec[i]+lag_vec[1]
# Compute target shifted forward by shift  
  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,shift)
  target_mat=filt_obj$target_mat
  target<-target_mat[,1]
# Collect the forward shifted targets: at i=1 shift=h_vec[1]+lag_vec[1]=1 corresponds to the publication lag of BIP  
  target_shifted_mat<-cbind(target_shifted_mat,target)

  mplot<-cbind(target,indicator_mat)
  colnames(mplot)[1]<-paste("Target left-shifted by ",shift-lag_vec[1],sep="")
  par(mfrow=c(1,1))
  colo<-c("black",rainbow(ncol(indicator_mat)))
  main_title<-paste("M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep="")
  plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(data)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
  mtext(colnames(mplot)[1],col=colo[1],line=-1)
  for (j in 1:ncol(mplot))
  {
    lines(mplot[,j],col=colo[j],lwd=1,lty=1)
    mtext(colnames(mplot)[j],col=colo[j],line=-j)
  }
  abline(h=0)
  abline(v=which(rownames(mplot)==rownames(data_fit)[nrow(data_fit)]),lwd=2,lty=2)
  axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
  axis(2)
  box()

# Compute target correlations of M-SSA predictors with shifted target
  for (j in 1:ncol(indicator_mat))
    cor_mat[i,j]<-cor(na.exclude(cbind(target,indicator_mat[,j])))[1,2]

}

colnames(cor_mat)<-paste("M-SSA: h=",h_vec,sep="")
rownames(cor_mat)<-paste("Shift of target: ",h_vec,sep="")
# -We can see that M-SSA predictors optimized for larger forecast horizons correlate more strongly 
#   with correspondingly forward-shifted target
# -The largest correlations tend to lie on (or to be close to) the diagonal of cor_mat
cor_mat

# We infer that the M-SSA predictors are informative about future BIP trend growth
# Since future BIP trend growth tells something about the low-frequency part of future BIP, we infer that 
#   the M-SSA predictors are also informative about future BIP
# However, (differenced) BIP is a very noisy series
# Therefore it is difficult to assess statistical significance of forecast accuracy with respect to BIP
# But we can assess statistical significance of the effect observed in cor_mat
# For this purpose we regress the predictors on the shifted targets and compute HAC-adjusted p-values of the corresponding regression coefficients
t_HAC_mat<-p_value_HAC_mat<-matrix(ncol=length(h_vec),nrow=length(h_vec))
for (i in 1:length(h_vec))# i<-1
{
  for (j in 1:length(h_vec))# j<-1
  {
    lm_obj<-lm(target_shifted_mat[,i]~indicator_mat[,j])
    summary(lm_obj)
    # This one replicates std in summary
    sd<-sqrt(diag(vcov(lm_obj)))
    # Here we use HAC  
    sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
    # This is the same as
    sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
    t_HAC_mat[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
    p_value_HAC_mat[i,j]<-2*pt(t_HAC_mat[i,j], len-length(select_vec_multi), lower=FALSE)
    
  }
}


colnames(t_HAC_mat)<-colnames(p_value_HAC_mat)<-paste("M-SSA: h=",h_vec,sep="")
rownames(t_HAC_mat)<-rownames(p_value_HAC_mat)<-paste("Shift of target: ",h_vec,sep="")
# p-values: small p-values lie on (or close to) the diagonal
# Statistical significance after HAC-correction reaches up to max(h_vec)
# Significance decreases with increasing forward-shift
p_value_HAC_mat





