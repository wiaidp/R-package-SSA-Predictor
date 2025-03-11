# M-SSA: extension of univariate to multivariate SSA
# Work in progress
# To dos: 
#   -use full HP-filtering facility, up to sample end
#   -compute rRMSE
#   -insert Diebold Mariano and Giacomini White tests of equal predictive ability

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
# BIP(GDP) has a publication lag of one quarter
# The target column in the data-file below corresponds to a one-step ahead forecast: 
#   -BIP has a publication lag of one quarter but we shifted the data (BIP) in the target column one additional 
#     quarter upwards (forward) to be on the safe-side (for example to account for revisions)
#   -In summary: a nowcast in our designs assumes that the two-sided target filter is applied to the target column in the data file below
# Columns 2-8 are the data available in Jan-2025 for nowcasting the target column (we can also rely on BIP as an explanatory variable)
# All indicators except spread are log-transformed. Then all indicators are differenced (quarterly differences) and standardized
#   -Calibration of true levels and variances can be obtained afterwards, by simple linear regression
# Extreme (singular) observations during Pandemic (2019-2020) where trimmed at 3 standard deviations 
#   After discussion this trimming was deemed acceptable (and transparent,reproducible) to avoid overly strong impact of singular data
tail(data)
# We see that the data in the target column (BIP) is shifted upwards two quarters: 
# We here want to nowcast/predict this series given the data in columns 2-7 (explanatory variables)
# Specify the publication lag
lag_vec<-c(2,rep(0,ncol(data)-1))

# Plot the data
# The real-time BIP (red) is lagging the target by lag_vec[1] quarters (publication lag)
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
#   -The classic quarterly setting lambda_HP=1600 leads to a design which tends to smooth out recessions
#   -Dynamics are too weak to be `useful' as a forecast tool (dynamic changes over a one-year horizon are weak)
#     -Therefore, imposing a one-year ahead forecast horizon does not significantly affect the M-SSA predictor
#   -Most important: the left-shift of the M-SSA predictor as a function of the forecast horizon 
#      is either inexistent or weak when selecting lambda_HP=1600 
#     -As stated, HP(1600) is so smooth that the series does not vary/change substantially over a one-year horizon
#     -Therefore, predictors are affected only marginally by the forecast horizon over such a `short' time span 
#   -Finally, the classic design (lambda_HP=1600) is (more) sensitive to Pandemic: the finite-length truncated filter looks `bad'
# To summarize
# -lambda_HP=160 is adapting more rapidly to changes of BIP, so as to be able track dynamic shifts within a one-year horizon
#   -Accordingly, M-SSA predictors are reactive to the forecast horizon
#   -Predictors are increasingly left-shifted (anticipative) as a function of the forecast horizon
# In the next tutorial we shall present an even more reactive design, which tracks future BIP slightly better 

lambda_HP<-160
# Filter length: roughly 4 years. 
#   The length should be an odd number in order to have a center point of the filter (where left and right tails are mirrored)
L<-31

target_obj<-HP_target_sym_T(n,lambda_HP,L)

gamma_target=t(target_obj$gamma_target)
symmetric_target=target_obj$symmetric_target 
colnames(gamma_target)<-select_vec_multi


# The targets of each series are one-sided
par(mfrow=c(1,1))
ts.plot(gamma_target,col=rainbow(n))

# But we tell M-SSA to mirror the right tail at the center peak: symmetric_target==T
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

# We can verify that target (two-sided HP applied to BIP) is shifted upward by publication-lag+delta=2+4
cbind(target_mat[,1],mssa_mat[,1])[(L-6):(L+5),]


# Plots: in-sample span is marked by vertical line
for (i in n:1)
{
  par(mfrow=c(1,1))
  mplot<-cbind(target_mat[,i],mssa_mat[,i],mmse_mat[,i])
  colnames(mplot)<-c(paste("Target: HP applied to ",select_vec_multi[i],", left-shifted by ",delta-lag_vec[1],"+publag quarters",sep=""),"M-SSA","M-MSE")

  colo<-c("black","blue","green")
  main_title<-paste("M-SSA ",select_vec_multi[i],": delta-publag=",delta-lag_vec[1],", in-sample span ending in ",rownames(data_fit)[nrow(data_fit)],sep="")
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

# Correlations between target and M-SSA: sample estimates converge to criterion value for increasing 
#   sample size len, assuming the VAR to be true, see tutorial 7.1
# The following results look bad: negative sample correlations...
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[,i],mssa_mat[,i])))[1,2])
# We can compare to the criterion value of M-SSA: the target correlation is maximized under the HT constraint
#   -By maximization the true correlations must be positive
#   -A large discrepancy to the above sample correlations suggests that the VAR(1) is misspecified
MSSA_obj$crit_rhoy_target

# M-SSA optimizes target correlation under holding time constraint:
# Compare empirical and theoretical (imposed) HTs: sample HT converges to imposed HT for increasing sample size len
unlist(apply(mmse_mat,2,compute_empirical_ht_func))
unlist(apply(mssa_mat,2,compute_empirical_ht_func))
ht_mssa_vec
# In contrast to correlations above, the sample HTs look OK, given the rather short sample (random fluctuation)
# In any case, we'd like M-SSA to be smoother smoother than M-MSE: 
#   The following ratios should be smaller and this looks fine! 
unlist(apply(mmse_mat,2,compute_empirical_ht_func))/unlist(apply(mssa_mat,2,compute_empirical_ht_func))


#############################################################################################################
# How can we improve performances, in parrticular target correlations?

# The following plot for BIP suggests that
# 1. The forecast problem is rather difficult
# 2. The predictors are too much right shifted (retarded)
par(mfrow=c(1,1))
i<-1
mplot<-cbind(target_mat[,i],mssa_mat[,i],mmse_mat[,i])
colnames(mplot)<-c(paste("Target: HP applied to ",select_vec_multi[i],", left-shifted by ",delta-lag_vec[1],"+publag quarters",sep=""),"M-SSA","M-MSE")

colo<-c("black","blue","green")
main_title<-paste("M-SSA ",select_vec_multi[i],": delta-publag=",delta-lag_vec[1],", in-sample span ending in ",rownames(data_fit)[nrow(data_fit)],sep="")
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


# Let's try a larger forecast horizon than necessary: we call this a forecast excess
f_excess<-4
# Increase artificially delta
delta_excess<-delta+f_excess
# All other settings remain the same: we now call M-SSA with that larger forecast horizon

MSSA_main_obj<-MSSA_main_func(delta_excess,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)

bk_x_mat_excess=MSSA_main_obj$bk_x_mat
MSSA_obj=MSSA_main_obj$MSSA_obj 
# Benchmark MSE predictor
gammak_x_mse<-MSSA_obj$gammak_x_mse

colnames(bk_x_mat)<-colnames(gammak_x_mse)<-select_vec_multi

# Filter: use the new bk_x_mat_excess but the previous delta (not delta_excess)

filt_obj<-filter_func(x_mat,bk_x_mat_excess,gammak_x_mse,gamma_target,symmetric_target,delta)

# We just need the new M-SSA for comparison
mssa_excess_mat=filt_obj$mssa_mat

# We can verify that target (two-sided HP applied to BIP) is shifted upward by publication-lag+delta=2+4
#   We do not rely on delta_excess
cbind(target_mat[,1],mssa_excess_mat[,1])[(L-6):(L+5),]

# Compute sample correlations: they are now positive!
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[,i],mssa_excess_mat[,i])))[1,2])

# Compute sample HTs: M-SSA keeps the expected HT fixed (independent of forecast horizon)
unlist(apply(mssa_excess_mat,2,compute_empirical_ht_func))
# Compare with previous M-SSA: differences are mainly due to random fluctuations
unlist(apply(mssa_mat,2,compute_empirical_ht_func))
# New M-SSA is still markedly smoother than classic MSE predictor, as desired
unlist(apply(mmse_mat,2,compute_empirical_ht_func))/unlist(apply(mssa_excess_mat,2,compute_empirical_ht_func))



par(mfrow=c(1,1))
mplot<-(cbind(target_mat[,1],mssa_excess_mat[,1],mssa_mat[,1]))
colnames(mplot)<-c(paste("Target as above",""),"M-SSA excess","M-SSA")
colo<-c("black","red","blue")
main_title<-"M-SSA: without forecast excess (blue) and with forecast excess (red)"
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

# M-SSA-excess is subject to stronger zero-shrinkage since it is looking further into the future
#   -M-SSA effectively computes the predictor with the smallest mean-square error subject to the HT constraint
#   -Under the assumption that the model is `true'
# Therefore let us standardize all series for better visual inspection
par(mfrow=c(1,1))
# Scale the data for better visual interpretation of effect of excess forecast on M-SSA (red) vs. previous M-SSA (blue)
mplot<-scale(cbind(target_mat[,1],mssa_excess_mat[,1],mssa_mat[,1]))
colnames(mplot)<-c(paste("Target: HP applied to ",select_vec_multi[1],", left-shifted by ",delta-lag_vec[1]," quarters",sep=""),"M-SSA excess","M-SSA")
colo<-c("black","red","blue")
main_title<-"Standardized M-SSA: without forecast excess (blue) and with forecast excess (red)"
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

# We see that the concept of `excess forecast' leads to a commensurate left-shift of M-SSA-excess (red)
# This left-shift is responsible for the improvement seen in the sample target correlations
#   -If the true process were the fitted VAR(1), then we would probably not observe the marked recession episodes
#   -Therefore, the obtained left-shift would not improve performances (quite the opposite in fact)
#   -But in the presence of strong asymmetric down-turns, a `faster' filter can track the relevant dynamics better 

#####################
# Findings:
# -We can address misspecification of the VAR(1) (for data with marked recession episodes) by:
#   1. allowing for a stronger left-shift of the predictor (excess forecast) and by
#   2. allowing for a re-scaling (calibration) of the left-shifted predictor, which is typically subject 
#         to excessive zero-shrinkage
# We now apply these findings towards the construction of predictors for German GDP (BIP)



################################################################################################################
# M-SSA generates five outputs: for BIP, ip, ifo, ESI and spread
# A. Equal-weighting
# -We consider each of the five M-SSA outputs as an equally valid and informative predictor for future BIP
# -Therefore we aggregate all five predictors equally, assuming that each one was previously standardized
#   -Cross-sectional aggregation: equal weighting
# B. Forecast excess
# -BIP and ip tend to lag behind ESI and ifo (mainly because of publication lag) and spread is leading overall
# -We select a larger delta for BIP and ip: forecast excess detailed above 
# C. Forecast horizons:
# -We compute M-SSA predictors based on A. and B. above, targeting BIP at horizons 0 (nowcast), 1, 2, 4 (one year) and 6 quarters ahead
#   -This results in 5 predictors 
#   -We then compute performances of each of the 5 predictors relative to BIP shifted by 0,1,2,4,6 quarters: 
#       -We consider all 25 combinations
#   -We also consider statistical significance, by relying on different statistics (relying on HAC estimates of variances)

# Let's start
# These are the interesting forecast horizons indicated above
h_vec<-c(0,1,2,4,6)
# Forecast excesses: 
#   -The first number in f_excess is the excess applied to M-SSA-BIP and M-SSA-ip
#   -The second number in f_excess is the excess applied to M-SSA-ifo, -ESI and -spread, see loop below
# This design corresponds to a `mildly aggressive' design: predictors will be left-shifted but not too heavily
f_excess<-c(4,2)
mssa_bip<-mssa_ip<-mssa_esi<-mssa_ifo<-mssa_spread<-NULL
# Compute M-SSA predictors for each forecast horizon
for (i in 1:length(h_vec))#i<-1
{
# For each forecast horizon h_vec[i] we compute M-SSA for BIP and ip first, based on the proposed forecast excess
#   BIP and ip require a larger forecast excess f_excess[1]. We also add the publication lag
  delta<-h_vec[i]+lag_vec[1]+f_excess[1]
  
# M-SSA  
  MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)
  
  bk_x_mat=MSSA_main_obj$bk_x_mat
  MSSA_obj=MSSA_main_obj$MSSA_obj 
  colnames(bk_x_mat)<-select_vec_multi
  
# Filter
  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
  
  mssa_mat=filt_obj$mssa_mat
  target_mat=filt_obj$target_mat
  mmse_mat<-filt_obj$mmse_mat
  colnames(mssa_mat)<-select_vec_multi
# Select M-SSA BIP and ip  
  mssa_bip<-cbind(mssa_bip,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[1])])
  mssa_ip<-cbind(mssa_ip,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[2])])
  
# Now compute M-SSA for the remaining ifo, ESI and spread  
# These series require a smaller forecast excess f_excess[2] 
  delta<-h_vec[i]+lag_vec[1]+f_excess[2]
  
  MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)
  
  bk_x_mat=MSSA_main_obj$bk_x_mat
  MSSA_obj=MSSA_main_obj$MSSA_obj 
  colnames(bk_x_mat)<-select_vec_multi

  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
  
  mssa_mat=filt_obj$mssa_mat
  target_mat=filt_obj$target_mat
  mmse_mat<-filt_obj$mmse_mat
  colnames(mssa_mat)<-select_vec_multi
  
# Select M-SSA-ifo, -ESI and -spread  
  mssa_ifo<-cbind(mssa_ifo,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[3])])
  mssa_esi<-cbind(mssa_esi,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[4])])
  mssa_spread<-cbind(mssa_spread,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[5])])
  
}

# Standardize and aggregate: equal weighting
indicator_mat<-(scale(mssa_bip)+scale(mssa_ip)+scale(mssa_ifo)+scale(mssa_esi)+scale(mssa_spread))/length(select_vec_multi)

colnames(indicator_mat)<-colnames(mssa_bip)<-colnames(mssa_ip)<-colnames(mssa_ifo)<-colnames(mssa_esi)<-colnames(mssa_spread)<-paste("Horizon=",h_vec,sep="")
rownames(indicator_mat)<-rownames(x_mat)

# The five M-SSA predictors
tail(indicator_mat)

# Compute sample target correlations: all 5*5 combinations
target_shifted_mat<-NULL
cor_mat<-matrix(ncol=length(h_vec),nrow=length(h_vec))

for (i in 1:length(h_vec))#i<-1
{
  shift<-h_vec[i]+lag_vec[1]
# Compute target: two-sided HP applied to BIP and shifted forward by forecast horizon plus publication lag
  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,shift)
  target_mat=filt_obj$target_mat
# Select BIP (first column)  
  target<-target_mat[,"BIP"]
# Collect the forward shifted targets: 
#   For the first loop-run, i=1 and shift=h_vec[1]+lag_vec[1]=2 corresponds to the publication lag of BIP (note that we selected a slightly larger publication lag, as discussed at the top of the this tutorial)  
  target_shifted_mat<-cbind(target_shifted_mat,target)
# Plot indicators and shifting target
  mplot<-scale(cbind(target,indicator_mat))
  colnames(mplot)[1]<-paste("Target left-shifted by ",shift-lag_vec[1],sep="")
  par(mfrow=c(1,1))
  colo<-c("black",rainbow(ncol(indicator_mat)))
  main_title<-paste("Standardized M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep="")
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

# Compute target correlations of all M-SSA predictors with shifted target: all combinations
  for (j in 1:ncol(indicator_mat))
    cor_mat[i,j]<-cor(na.exclude(cbind(target,indicator_mat[,j])))[1,2]

}
# Observe the shifts: 
#   The target is shifted upward by publication lag (assumed to be 2 quarters) + forecast horizon relative to the predictor (in the first column) 
cbind(indicator_mat[,1],target_shifted_mat)[(L-10):(L+6),]

colnames(cor_mat)<-paste("M-SSA: h=",h_vec,sep="")
rownames(cor_mat)<-paste("Shift of target: ",h_vec,sep="")
# -We can see that M-SSA predictors optimized for larger forecast horizons (from left to right in cor_mat) 
#     correlate more strongly with correspondingly forward-shifted target (from top to bottom in cor_mat)
# -The largest correlations tend to lie on (or to be close to) the diagonal of cor_mat
cor_mat

# We infer from the observed pattern, that the M-SSA predictors tend to be informative about future BIP trend growth
# Since future BIP trend growth tells something about the low-frequency part of future BIP, we may infer that 
#   the M-SSA predictors are also informative about future BIP
# However, (differenced) BIP is a very noisy series
# Therefore, it is difficult to assess statistical significance of forecast accuracy with respect to BIP
# But we can assess statistical significance of the effect observed in cor_mat, with respect to HP-BIP (low-frequency part of BIP)
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
# Statistical significance (after HAC-correction) is still achieved towards larger forecast horizons
# As expected, the Significance decreases (p-values increase) with increasing forward-shift
p_value_HAC_mat

#--------------------------------------------------
# The above result suggest predictability of M-SSA indicators with respect to future HP-BIP
# What about future BIP?
t_HAC_mat_BIP<-p_value_HAC_mat_BIP<-matrix(ncol=length(h_vec),nrow=length(h_vec))
BIP_target_mat<-NULL
for (i in 1:length(h_vec))# i<-1
{
# Shift BIP  
  shift<-h_vec[i]+lag_vec[1]
  BIP_target<-c(x_mat[(1+shift):nrow(x_mat),"BIP"],rep(NA,shift))
  BIP_target_mat<-cbind(BIP_target_mat,BIP_target)
# Rgress predictors on shifted BIP  
  for (j in 1:length(h_vec))# j<-1
  {
    lm_obj<-lm(BIP_target~indicator_mat[,j])
    summary(lm_obj)
    # This one replicates std in summary
    sd<-sqrt(diag(vcov(lm_obj)))
    # Here we use HAC  
    sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
    # This is the same as
    sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
    t_HAC_mat_BIP[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
    p_value_HAC_mat_BIP[i,j]<-2*pt(t_HAC_mat_BIP[i,j], len-length(select_vec_multi), lower=FALSE)
    
  }
}
colnames(t_HAC_mat_BIP)<-colnames(p_value_HAC_mat_BIP)<-paste("M-SSA: h=",h_vec,sep="")
rownames(t_HAC_mat_BIP)<-rownames(p_value_HAC_mat_BIP)<-paste("Shift of target: ",h_vec,sep="")
# p-values: 
# In contrast to HP-BIP, significance with respect to BIP is less conclusive: BIP is much noisier
# However, we still find that for increasing forward-shift of BIP (from top to bottom) 
#   the M-SSA indicators optimized for larger forecast horizon (from left to right) tend to perform better
# These results could be altered by modifying the forecast excesses: 
#   -Selecting more aggressive designs (larger excesses) may lead to stronger significance at larger shifts, up to a point 
#   -You may try f_excess<-c(6,4): a strong result at a one-year ahead forecast horizon (plus publication lag) is achievable
p_value_HAC_mat_BIP


# Select a forecast horizon 
k<-4
h_vec[k]
par(mfrow=c(1,1))
# Scale the data for better visual interpretation of effect of excess forecast on M-SSA (red) vs. previous M-SSA (blue)
mplot<-scale(cbind(BIP_target_mat[,k],indicator_mat[,k]))
colnames(mplot)<-c(paste("BIP left-shifted by ",h_vec[k]," quarters",sep=""),"M-SSA predictor")
colo<-c("black","blue")
main_title<-"Standardized forward-shifted BIP vs. predictor"
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

# Sample correlation
cor(na.exclude(mplot))


#--------------------------------------------------------------------------------
# Full-length HP
# -In the above experiment we relied on finite length (L=31) truncated version of the two-sided HP filter
# -Instead, we could rely on the full-length HP filter and recompute HAC-adjusted t-statistics to verify 
#   statistical significance (predictability) of M-SSA predictors

# 1. Compute full-length HP
len<-nrow(x_mat)
hp_obj<-hpfilter(rnorm(len),type="lambda", freq=lambda_HP)
# Specify trend filters: the above function returns HP-gap
fmatrix<-diag(rep(1,len))-hp_obj$fmatrix
# Check: plot one-sided trend at start, two-sided in middle and one-sided at end
ts.plot(fmatrix[,c(1,len/2,len)])

# 2. Compute full-length HP trend output
#   -Relies on full-length filter
#   -Does not have NAs at start and end
target_without_publication_lag<-t(fmatrix)%*%x_mat[,1]
# Shift forward by publication lag (2 quarters)
target<-c(target_without_publication_lag[(1+lag_vec[1]):length(target_without_publication_lag)],rep(NA,lag_vec[1]))


# Plot
#   -Note that full-length HP becomes increasingly asymmetric towards the sample boundaries
#   -The quality towards the sample boundaries degrades
# We here discard the first and last year
target[length(target):(length(target)-3)]<-target[1:4]<-NA

mplot<-scale(cbind(target,indicator_mat))
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c("Full-length HP",colnames(indicator_mat))
colo<-c("black",rainbow(ncol(indicator_mat)))
main_title<-"Full-length HP"
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

# Compute correlations: note that the target corresponds to a nowcast
cor(na.exclude(mplot))

# 3. In addition to a nowcast we also analyze forward-shifts of the target
target_shifted_mat<-NULL
for (i in 1:length(h_vec))
{
  shift<-h_vec[i]
  target_shifted_mat<-cbind(target_shifted_mat,c(target[(1+shift):length(target)],rep(NA,shift)))
}

# 4. Recompute correlations and HAC-adjusted t-statistics of regression of M-SSA indicators on shifted full-sample HP trend
cor_mat<-p_value_HAC_mat<-matrix(ncol=length(h_vec),nrow=length(h_vec))
for (i in 1:length(h_vec))# i<-1
{
  for (j in 1:length(h_vec))# j<-1
  {
    cor_mat[i,j]<-cor(na.exclude(cbind(target_shifted_mat[,i],indicator_mat[,j])))[1,2]
    lm_obj<-lm(target_shifted_mat[,i]~indicator_mat[,j])
    summary(lm_obj)
    # This one replicates std in summary
    sd<-sqrt(diag(vcov(lm_obj)))
    # Here we use HAC  
    sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
    # This is the same as
    sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
    t_HAC_mat<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
    p_value_HAC_mat[i,j]<-2*pt(t_HAC_mat, len-length(select_vec_multi), lower=FALSE)
    
  }
}
colnames(cor_mat)<-colnames(p_value_HAC_mat)<-paste("M-SSA: h=",h_vec,sep="")
rownames(cor_mat)<-rownames(p_value_HAC_mat)<-paste("Shift of target: ",h_vec,sep="")
# The full-length HP results confirm earlier findings 
cor_mat
p_value_HAC_mat



#################################################################
# Findings
# A. When targeting forecast horizons of a year or less, we need to concentrate on signals (HP-trends) with sufficiently 
#   strong dynamic changes over such a time interval
#   -For this purpose we selected lambda_HP=160 (more adaptive than the classic lambda_HP=1600 setting)
#   -The increased adaptivity forces predictors to react to the forecast horizon by a commensurate left-shift (anticipation)
#   -In the next tutorial we shall look at even more adaptive designs
# -Assuming a suitable choice for lambda_HP, the main construction principles behind M-SSA indicators leads to 
#   forecast designs with predictive relevance
#   -Left-shift controlled by forecast horizon
#   -Smoothness (nois-suppression) controlled by HT
# -Predicting HP-BIP (the trend component) seems easier than predicting BIP
#   -HP-BIP is mostly exempted from erratic (unpredictable) high-frequency components of BIP
# -The effect of the forecast horizon (hyperparameter) in M-SSA is statistically as well as logically consistent: 
#   -Increasing the forecast horizon leads to improved performances at larger forward-shifts
#   -Forecast horizon is commensurate to `physical' forward shift of target 
# -Performances with respect to BIP (instead of HP-BIP) are less conclusive, due in part to unpredictable high-frequency noise
#   -However, the link between forecast horizon and physical-shift is still recognizable
#   -More aggressive settings for the forecast excess may reinforce these findings (up to a point)
# -Finally, a predictor of the low-frequency component of (future) HP-BIP is intrinsically informative about 
#     (future) BIP, even if statistical significance is obstructed by noise. 

#---------------------------------------------------------------------------------------------------


############################################################
# Coming next: rRMSE (relative root mean-square error) and HAC-adjusted DM/GW statistics of predictive outperformance

