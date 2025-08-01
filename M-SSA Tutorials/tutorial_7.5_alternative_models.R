# Tutorial 7.5
# Almost the same as tutorial 7.4 but we propose a different multivariate model
#   -BVAR(3) instead of VAR(1)
#   -The BVAR allows for richer dynamics (it also assigns more weight to ESI, i.e., the economic sentiment indicator)
# M-SSA based on BVAR(3) performs better (than VAR(1) in tutorial 7.4)
#   -Smaller p-values (stronger link to future GDP) and smaller rRMSEs (stronger outperformance of mean and direct forecasts benchmark predictors)
#   -More consistent patterns in shift/h matrices: for given forward-shift of GDP, the designs optimized for h=shift (diagonal) are close to optimal   

#----------------------------------------------
# Start with a clean sheet
rm(list=ls())


# Load the required R-libraries
# Standard filter package
library(mFilter)
# Multivariate time series: VARMA model for macro indicators: used here for simulation purposes only
library(MTS)
# HAC estimate of standard deviations in the presence of autocorrelation and heteroscedasticity
library(sandwich)
# Extended time series
library(xts)
# Library for Diebold-Mariano test of equal forecast performance
library(multDM)
# GARCH model: for improving regression estimates
library(fGarch)
# Packages for Ridge and LASSO
library(MASS)
library(glmnet) 




# Load the relevant M-SSA functionalities
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))
# Utility functions for M-SSA, see tutorial 
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))


#------------------------------------------------------------------------
# Load the data and select the relevant indicators: see tutorials 7.2 and 7.3 for background
load(file=paste(getwd(),"\\Data\\macro",sep=""))

# Publication lag: we assume a lag of one quarter for BIP 
lag_vec<-c(1,rep(0,ncol(data)-1))

# Plot the data
# The real-time BIP (red) is lagging the target (black) by lag_vec[1] quarters (publication lag)
par(mfrow=c(1,1))
mplot<-data[,-1]
colo<-c(rainbow(ncol(data)-1))
main_title<-paste("Quarterly indicators",sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(data)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()
# -The explanatory variables BIP (red line) and ip (orange) are right shifted 
#   -Publication lags: BIP one quarter and ip two months (see data_monthly for the 2-month lag of ip)


# Select macro indicators for M-SSA 
select_vec_multi<-c("BIP","ip","ifo_c","ESI","spr_10y_3m")
x_mat<-data[,select_vec_multi] 
rownames(x_mat)<-rownames(data)
n<-dim(x_mat)[2]
# Number of observations
len<-dim(x_mat)[1]




##################################################################################
# Exercise 1 
# Exercise 1.1: Define and apply different multivariate models for M-SSA (and M-MSE)

# We can select the type of VAR model
# Previous tutorials relied on a VAR(1)
VAR_type="VAR"
# BVAR are interesting because they allow to incorporate more lags (richer dynamics, better lead/lag fitting).
#   -This is achieved by shrinking the model to a plausible benchmark
VAR_type="BVAR"
if (VAR_type=="BVAR")
{
# Simon's settings for BVAR(3)  
  lambda_BVAR <- 0.001
  p<-3
  q<-0
}
if (VAR_type=="VAR")
{
# Marc's settings for VAR  
  p<-1
  q<-0
}
# We use pre pandemic data for model fitting
date_to_fit<-"2020"


#----------
# Filter settings
# Target filter: lambda_HP is the single most important hyperparameter, see tutorial 7.1 for a discussion
lambda_HP<-160
# Filter length: nearly 8 years is fine for the selected lambda_HP (filter weights decay sufficiently fast)
#   The length should be an odd number (see tutorial 7.1)
L<-31

# HT constraint: same as in tutorial 7.2 (roughly 50% larger than classic M-MSE predictor)
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Forecast horizons: M-SSA is optimized for each forecast horizon in h_vec 
h_vec<-0:6
# Forecast excesses: see tutorial 7.2, exercise 2 for background
f_excess<-c(5,rep(4,length(select_vec_multi)-1))

# Run the M-SSA wrapper, see tutorial 7.2
#   -The function computes M-SSA and M-MSE for each forecast horizon h in h_vec
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi,VAR_type,lambda_BVAR)

# M-SSA components
mssa_array<-mssa_indicator_obj$mssa_array
# M-MSE components
# -Same as mssa_array but without HT imposed, i.e., classic multivariate mean-square error signal extraction
mmse_array<-mssa_indicator_obj$mmse_array

#----------------
# 1.2 Compute performances of M-SSA component predictor for all combinations of forward-shift and 
#   forecast horizon (6*7 matrix of performance metrics)

# Select M-SSA components: use M-SSA output tracking BIP 
#   -Natural candidate for forecasting BIP; 
#   -Simple regression equation (single explanatory); 
#   -Small revisions (due to quarterly up-dating of equations)
sel_vec_mssa_comp<-c("BIP")
#sel_vec_mssa_comp<-select_vec_multi
# Select indicators for direct forecast: all indicators lead to overfitting (worse out-of-sample performances)
# Best design out-of-sample is based on ifo_c and ESI
sel_vec_direct_forecast<-c("ifo_c","ESI")
# M-SSA looses L observations at start when compared to direct forecast or mean benchmarks: 
#   -therefore the samples are slightly different
# If align_sample==T then the first L observations of mean and direct forecasts are removed to align with M-SSA
#   -Avoid random differences due to unequal samples.
align_sample<-T
# Can define type of regression
reg_type<-"OLS"
reg_type<-"LASSO"
reg_type<-"Ridge"
# Penalty weight in regularization
lambda_reg<-10
# Use GARCH to account for heteroscedasticity (bad idea because recession episodes are underweighted)
use_garch<-F
# Shift BIP forward by shift_vec + publication lag
shift_vec<-0:5
# Out-of-sample period for regression equations starts before financial crisis 
in_out_separator<-"2007"

# Select indicators for direct forecast: use all indicators
sel_indicator_out_sample<-select_vec_multi
# Initialize performance matrices
MSE_oos_mssa_comp_without_covid_mat<-MSE_oos_mssa_comp_mat<-p_mat_mssa<-p_mat_mssa_components<-p_mat_mssa_components_without_covid<-p_mat_direct<-rRMSE_mSSA_comp_direct<-rRMSE_mSSA_comp_mean<-rRMSE_mSSA_comp_direct_without_covid<-rRMSE_mSSA_comp_mean_without_covid<-rRMSE_mSSA_direct_mean_without_covid<-rRMSE_mSSA_direct_mean<-p_mat_direct_without_covid<-matrix(nrow=length(shift_vec),ncol=length(h_vec))
final_components_preditor_array<-oos_components_preditor_array<-array(dim=c(length(shift_vec),length(h_vec),nrow(x_mat)))
dimnames(final_components_preditor_array)<-dimnames(oos_components_preditor_array)<-list(paste("shift=",shift_vec,sep=""), paste("h=",h_vec,sep=""),rownames(x_mat))
  
dim(p_mat_mssa_components)
# Initialize arrays collecting final predictors, real-time predictors and regression weights
#   -These will be used when analyzing revisions
track_weights_array<-array(dim=c(length(shift_vec),length(h_vec),nrow(x_mat),length(sel_vec_mssa_comp)+1))
dimnames(track_weights_array)<-list(paste("shift=",shift_vec,sep=""),
                                      paste("h=",h_vec,sep=""),rownames(x_mat),c("Intercept",sel_vec_mssa_comp))
  
# Set-up progress bar: indicates progress in R-console
pb <- txtProgressBar(min=min(h_vec),max=max(h_vec)-1,style=3)
  
# The following double loop computes all combinations of forward-shifts (of BIP) and forecast horizons (of M-SSA)
for (shift in shift_vec)#shift<-1
{
# Progress bar: see R-console
  setTxtProgressBar(pb, shift)
  for (j in h_vec)#j<-1
  {
# Horizon j corresponds to k=j+1-th entry of array    
    k<-j+1
# A. M-SSA component predictor
# Specify data matrix for WLS regression
    if (length(sel_vec_mssa_comp)>1)
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_mssa_comp,,k]))
    } else
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mssa_array[sel_vec_mssa_comp,,k]))
    }
    rownames(dat)<-rownames(x_mat)
    colnames(dat)<-c(colnames(x_mat)[1],sel_vec_mssa_comp)
    nrow(na.exclude(dat))
      
# Apply the previous function: compute GARCH, WLS regression, out-of-sample MSEs and p-values    
    perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec,align_sample,reg_type,lambda_reg)
# Retrieve out-of-sample performances 
# a. p-values with/without Pandemic    
    p_mat_mssa_components[shift+1,k]<-perf_obj$p_value
    p_mat_mssa_components_without_covid[shift+1,k]<-perf_obj$p_value_without_covid
# b. MSE forecast error out-of-sample
#   -M-SSA components with/without Pandemic    
    MSE_oos_mssa_comp_mat[shift+1,k]<-MSE_oos_mssa_comp<-perf_obj$MSE_oos
    MSE_oos_mssa_comp_without_covid_mat[shift+1,k]<-MSE_oos_mssa_comp_without_covid<-perf_obj$MSE_oos_without_covid
#   -mean-benchmark with/without Pandemic    
    MSE_mean_oos<-perf_obj$MSE_mean_oos
    MSE_mean_oos_without_covid<-perf_obj$MSE_mean_oos_without_covid
# Here we retrieve the final in-sample predictor (based on the full-sample WLS regression) as well as the 
#    real-time out-of-sample predictor (re-adjusted to new data at each time point)
# We can plot both predictors to illustrate revisions (due to WLS estimation at each time point), see below
    final_in_sample_preditor<-perf_obj$final_in_sample_preditor
    final_components_preditor_array[shift+1,j+1,(nrow(x_mat)-length(final_in_sample_preditor)+1):nrow(x_mat)]<-final_in_sample_preditor
    cal_oos_pred<-perf_obj$cal_oos_pred
    oos_components_preditor_array[shift+1,j+1,(nrow(x_mat)-length(cal_oos_pred)+1):nrow(x_mat)]<-cal_oos_pred
# We can also obtain the regression weights to track changes (systematic vs. noisy revisions) over time
# Note: the variable will be overwritten, i.e., we keep only the last run through the double loop, 
#   corresponding to maximal shift and maximal forecast horizon, see exercise 2.2 below 
    track_weights<-perf_obj$track_weights
    track_weights_array[shift+1,j+1,(nrow(x_mat)-nrow(track_weights)+1):nrow(x_mat),]<-track_weights
      
#----------------      
# B. Direct forecasts
# -The main difference to M-SSA above is the specification of the explanatory variables in the data 
#     matrix dat: we here use x_mat instead of mssa_array. 
#   -We select all indicators (one could easily change this setting but results are only marginally effected as long as ifo and ESi are included)
#   -Note that the data matrix here does not depend on j, in contrast  to the M-SSA components above    
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),x_mat[,sel_vec_direct_forecast])
    rownames(dat)<-rownames(x_mat)
# Two variants for data set:  
    if (align_sample)
    {
# 1. Same sample as M-SSA (due to filter initialization the first L observations are lost) 
      dat<-dat[L:nrow(dat),]
# Same length as M-SSA      
      nrow(na.exclude(dat))
    } else
    {
# 2. Full data, including the first L observations (makes sense since the data is available) 
      dat<-dat[1:nrow(dat),]
    }
    perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec,align_sample,reg_type,lambda_reg)
# Retrieve out-of-sample performances: p-values and forecast MSE, with/without Pandemic 
    p_mat_direct[shift+1,k]<-perf_obj$p_value 
    p_mat_direct_without_covid[shift+1,k]<-perf_obj$p_value_without_covid 
    MSE_oos_direct<-perf_obj$MSE_oos
    MSE_oos_direct_without_covid<-perf_obj$MSE_oos_without_covid
      
# Compute rRMSEs
# a. M-SSA Components vs. direct forecast    
    rRMSE_mSSA_comp_direct[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_oos_direct)
# b. M-SSA Components vs. mean benchmark    
    rRMSE_mSSA_comp_mean[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_mean_oos)
# c. Direct forecast vs. mean benchmark    
    rRMSE_mSSA_direct_mean[shift+1,k]<-sqrt(MSE_oos_direct/MSE_mean_oos)
# Same as a, b, c but without Pandemic
    rRMSE_mSSA_comp_direct_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_oos_direct_without_covid)
    rRMSE_mSSA_comp_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_mean_oos_without_covid)
    rRMSE_mSSA_direct_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_direct_without_covid/MSE_mean_oos_without_covid)
  }
}
# Close progress bar
close(pb)
# Note: possible warnings issued by the GARCH estimation routine during computations can be ignored
  
# Assign column and rownames
colnames(p_mat_mssa_components)<-colnames(p_mat_direct)<-colnames(p_mat_mssa_components_without_covid)<-
colnames(rRMSE_mSSA_comp_direct)<-colnames(rRMSE_mSSA_comp_mean)<-
colnames(rRMSE_mSSA_comp_direct_without_covid)<-colnames(rRMSE_mSSA_comp_mean_without_covid)<-
colnames(rRMSE_mSSA_direct_mean)<-colnames(rRMSE_mSSA_direct_mean_without_covid)<-
colnames(p_mat_direct_without_covid)<-colnames(MSE_oos_mssa_comp_mat)<-
colnames(MSE_oos_mssa_comp_without_covid_mat)<-paste("h=",h_vec,sep="")
rownames(p_mat_mssa_components)<-rownames(p_mat_direct)<-rownames(p_mat_mssa_components_without_covid)<-
rownames(rRMSE_mSSA_comp_direct)<-rownames(rRMSE_mSSA_comp_mean)<-
rownames(rRMSE_mSSA_comp_direct_without_covid)<-rownames(rRMSE_mSSA_comp_mean_without_covid)<-
rownames(rRMSE_mSSA_direct_mean)<-rownames(rRMSE_mSSA_direct_mean_without_covid)<-
rownames(p_mat_direct_without_covid)<-rownames(MSE_oos_mssa_comp_mat)<-
rownames(MSE_oos_mssa_comp_without_covid_mat)<-paste("Shift=",shift_vec,sep="")
# Define list for saving all matrices  

#-----------------------
# Examine out-of-sample performances
# HAC-adjusted p-values of out-of-sample (M-SSA) components predictor when targeting forward-shifted BIP
#   -Evaluation based on out-of-sample span starting at in_out_separator and ending on Jan-2025
p_mat_mssa_components
# Same but without singular Pandemic
p_mat_mssa_components_without_covid
# The link between the new predictor and future BIP is statistically significant up to multiple quarters ahead 


# -The above p-values are based on regressions (of out-of-sample predictors on BIP) thereby ignoring 
#   `static' level and scale parameters (which are automatically adjusted by the regressions)
# -The following rRMSEs account (also) for scale and level of the new predictor
#   -All results out-of-sample (starting just before the financial crisis)
# a. rRMSE of M-SSA components when benchmarked against mean, out-of-sample
rRMSE_mSSA_comp_mean
# b. rRMSE of M-SSA components when benchmarked against direct forecast, out-of-sample
rRMSE_mSSA_comp_direct
# c. rRMSE of direct forecasts when benchmarked against mean benchmark, out-of-sample
#   Note: the columns are all the same because for a given shift, the explanatory variables of the 
#     direct forecast do not depend on j in the above loop  
rRMSE_mSSA_direct_mean

# Same but without Pandemic
rRMSE_mSSA_comp_mean_without_covid
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_direct_mean_without_covid

# Comments:
# -Systematic pattern: for larger forward-shifts (from top to bottom), designs optimized for larger 
#     forecasts horizons (from left to right) tend to perform better
# -Singular Pandemic data affects evaluation 
#   -p-values and rRMSEs increase; systematic patterns are cluttered by noise
#   -Direct forecasts barely outperform the simple mean benchmark, see rRMSE_mSSA_direct_mean


################################################################################################################
# Exercise 2 Analyze revisions of M-SSA components predictor
# -The new predictor relies on quarterly up-dating of the (WLS-) regression weights
#   -Note that M-SSA is not subject to revisions because the VAR is fixed (based on data up to 2008: no up-dating)
# -We here analyze the impact of the quarterly up-dating on the predictor as well as on the regression weights

# Select h and shift
h<-4
shift<-4


# 2.1 Compare the final predictor (full sample regression) with the out-of-sample sequence of 
#   continuously re-calibrated predictors: ideally (in the absence of revisions), both series would overlap
# -Differences illustrate revisions due to re-estimating regression weights each quarter
par(mfrow=c(1,1))
mplot<-cbind(final_components_preditor_array[shift,h,],oos_components_preditor_array[shift,h,])
colnames(mplot)<-c("Final predictor","Real-time out-of-sample predictor")
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-"Revisions: final vs. real-time predictor"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()
# Notes: 
# 1. The above plot compares real-time and final predictors for maximal shift and maximal forecast horizon
#     -Last run in the double-loop of exercise 1.3.5 above 
#     -Similar plots could be obtained for all combinations of forward-shift and forecast horizon (with no systematic change/difference)
# 2. The last data point in the above plot does not correspond to Jan-2025 (because the target is forward-shifted and NAs are removed)
tail(mplot)
#     -The purpose of the above plot is to illustrate revisions 
#     -It does not show the current forecast at the series end (Jan-2025)

# Comments (revisions): 
# -To the left of the plot, the real-time predictor is volatile because the sample is short (revisions are large)
# -With increasing sample size (from left to right), the real-time predictor approaches the final estimate 
# -The vertical black line in the plot indicates the start of the evaluation period, relevant for 
#     out-of-sample MSEs and p-value statistics, see exercise 1.3.5 above
# -As time progresses, we expect better out-of-sample forecast performances because the part imputable to the 
#   above revision error will decrease

# 2.2 We now examine the effect of the revisions on the regression weights  
par(mfrow=c(1,1))
mplot<-track_weights_array[shift,h,,]
#colnames(mplot)[2:ncol(track_weights)]<-paste("Weight of M-SSA component ",colnames(track_weights)[2:ncol(track_weights)])
colo<-c("black","blue","red")
main_title<-"Revisions: regression weights over time"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()

# Comments (revisions)
# -Regression weights are quite volatile at the start
# -However, progressively over time the weights appear to converge to some fix-points (stationarity)
#   -The real-time predictor converges to the final predictor


################################################################################################################
# Exercises 3 and 4 have been skipped 

#############################################################################################
# Exercise 5: M-MSE vs. M-SSA
# -We illustrated in exercise 4 that a multivariate filter approach outperforms univariate (HP-C) filtering 
#   (or no filtering at all) by generating a more systematic left-shift of the M-SSA component predictor
# -However, we have not compared M-SSA to M-MSE, the classic multivariate mean-square error filter 
#   -The M-MSE filter does not impose a HT constraint: it is less smooth (more zero-crossings) 
# -We here briefly derive forecast performances of the `M-MSE component predictor' and compare results to the 
#   simple mean as well as to the M-SSA component predictor
# -Construction of the M-MSE component predictor is straightforward: 
#   -Instead of M-SSA components (filter outputs) we regress M-MSE components (filter outputs) on 
#     forward-shifted BIP
#   -In the code snippet below we substitute mmse_array for mssa_array in the data-matrix dat

# Exercise 5.1: compute M-MSE-component predictor
shift_vec<-shift_vec
sel_vec_pred<-sel_vec_mssa_comp
# Initialize performance matrices
rRMSE_mmse_comp_mean<-rRMSE_mmse_comp_mean_without_covid<-rRMSE_mmse_comp_mssa<-rRMSE_mmse_comp_mssa_without_covid<-matrix(ncol=length(h_vec),nrow=length(h_vec)-1)

# The following double loop computes all combinations of forward-shifts (of BIP) and forecast horizons 
# -We compute HAC-adjusted p-values (significance of out-of-sample predictor) and 
#  out-of-sample rRMSEs (relative root mean-square forecast errors) for the direct HP forecast
  
# Set-up progress bar: indicates progress in R-console
pb <- txtProgressBar(min=min(h_vec),max=max(h_vec)-1,style=3)
  
for (shift in shift_vec)#shift<-2
{
# Progress bar: see R-console
  setTxtProgressBar(pb, shift)
  for (j in h_vec)#j<-5
  {
# Horizon j corresponds to k=j+1-th entry of array    
    k<-j+1
      
# M-MSE component predictor
# Specify data matrix for WLS regression: we insert mmse_array instead of mssa_array for the explanatory variables
    if (length(sel_vec_pred)>1)
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mmse_array[sel_vec_pred,,k]))
    } else
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mmse_array[sel_vec_pred,,k]))
    }
    rownames(dat)<-rownames(x_mat)
    colnames(dat)<-c(colnames(x_mat)[1],sel_vec_pred)
    dat<-na.exclude(dat)
      
    perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec,align_sample,reg_type,lambda_reg)
# Out-of-sample performances: p-values and forecast MSE, with/without Pandemic 
    MSE_oos_mmse<-perf_obj$MSE_oos
    MSE_oos_mmse_without_covid<-perf_obj$MSE_oos_without_covid
    MSE_oos_mean<-perf_obj$MSE_mean_oos
    MSE_oos_mean_without_covid<-perf_obj$MSE_mean_oos_without_covid
# Compute rRMSEs
# a. M-MSE vs- M-SSA
#   Note that MSEs of M-SSA predictor were computed in exercise 1.3.5
    rRMSE_mmse_comp_mssa[shift+1,k]<-sqrt(MSE_oos_mmse/MSE_oos_mssa_comp_mat[shift+1,k])
# Same but without Pandemic
    rRMSE_mmse_comp_mssa_without_covid[shift+1,k]<-sqrt(MSE_oos_mmse_without_covid/MSE_oos_mssa_comp_without_covid_mat[shift+1,k])
# b. M-MSE vs. mean
    rRMSE_mmse_comp_mean[shift+1,k]<-sqrt(MSE_oos_mmse/MSE_oos_mean)
# Same but without Pandemic
    rRMSE_mmse_comp_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_mmse_without_covid/MSE_oos_mean_without_covid)
  }
}
close(pb)
# Note: possible warnings issued by the GARCH estimation routine during computations can be ignored
  
# Assign column and rownames
colnames(rRMSE_mmse_comp_mssa)<-colnames(rRMSE_mmse_comp_mssa_without_covid)<-
colnames(rRMSE_mmse_comp_mean)<-colnames(rRMSE_mmse_comp_mean_without_covid)<-paste("h=",h_vec,sep="")
rownames(rRMSE_mmse_comp_mssa)<-rownames(rRMSE_mmse_comp_mssa_without_covid)<-
rownames(rRMSE_mmse_comp_mean)<-rownames(rRMSE_mmse_comp_mean_without_covid)<-paste("Shift=",shift_vec,sep="")

#-------------------
# Exercise 5.2: evaluate out-of-sample performances of M-MSE component predictor
#   -We here emphasize a four quarters ahead forecast (challenging forecast problem)

# 5.2.1 Compute Final M-MSE and M-SSA component predictors (whose regression relies on 
#   the full data sample)

# Select h and shift (h should be be smaller than max(h_vec))
h<-4
if (h>max(h_vec))
  h=max(h_vec)
# Select forward-shift
shift<-4

# Compute the final M-MSE component predictor optimized for forecast horizon h
# Note: for simplicity we here compute an OLS regression (WLS looks nearly the same)
if (length(sel_vec_pred)>1)
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mmse_array[sel_vec_pred,,h+1]))
} else
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mmse_array[sel_vec_pred,,h+1]))
}

if (F)
{
  # Weights for WLS regression: inverse prop. to GARCH-vola
  #   Warnings are generated by garch-package and can be ignored
  weight<-garch_vola_func(dat,shift,lag_vec)$weight
  ts.plot(1/weight,main="GARCH(1,1) conditional variance")
  # Regression  
  lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)],weight=weight)
}
# Regression  
lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)])
summary(lm_obj)

optimal_weights<-lm_obj$coef
# Compute predictor for each forward-shift  
if (length(sel_vec_pred)>1)
{  
  mmse_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]%*%optimal_weights[2:length(optimal_weights)]
} else
{
  mmse_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]*optimal_weights[2:length(optimal_weights)]
}
# Compute the final M-SSA component predictor optimized for forecast horizon h
if (length(sel_vec_pred)>1)
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,h+1]))
} else
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mssa_array[sel_vec_pred,,h+1]))
}
# Weights for WLS regression: inverse prop. to GARCH-vola
#   Warnings are generated by garch-package and can be ignored
if (F)
{
  weight<-garch_vola_func(dat,shift,lag_vec)$weight
  # Regression  
  lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)],weight=weight)
}
lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)])
summary(lm_obj)

optimal_weights<-lm_obj$coef
# Compute predictor for each forward-shift  
if (length(sel_vec_pred)>1)
{  
  mssa_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]%*%optimal_weights[2:length(optimal_weights)]
} else
{  
  mssa_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]*optimal_weights[2:length(optimal_weights)]
} 

# 5.2.2 Holding times
# Let's measure smoothness in terms of empirical holding-times
# -M-SSA imposes a larger HT (than the `natural' HT of M-MSE) and therefore it should be smoother
# -Notes: 
#   1. We imposed a 50% larger expected (true) HT than M-MSE in the HT constraint of the optimization criterion
#     -Ideally, the empirical HT of M-SSA should be (roughly) 50% larger than M-MSE, too.
#   2. M-SSA controls crossings at the mean-level. 
#     -Therefore we center the predictors when computing the empirical HT
compute_empirical_ht_func(scale(mssa_predictor))
compute_empirical_ht_func(scale(mmse_predictor))

par(mfrow=c(1,1))
ts.plot(scale(cbind(mssa_predictor,mmse_predictor)),col=c("blue","green"))
abline(h=0)
# M-SSA has approximately 33% less crossings (roughly 50% larger empirical HT)
#   -This feature of the predictor can be controlled by the HT hyperparameter
#   -The sample estimate is close to the expected (true) number 
#     -As shown in tutorial 7.1, the sample HT converges to the imposed (expected) HT for sufficiently 
#     long samples if the data generating process is known


# 5.2.3 MSE forecast performances
# M-MSE vs. mean
rRMSE_mmse_comp_mean_without_covid
# M-SSA components vs. mean
rRMSE_mSSA_comp_mean_without_covid
# M-MSE vs. M-SSA: rRMSE<1 signifies that M-MSE perform better
rRMSE_mmse_comp_mssa_without_covid

# Findings
# -M-SSA does not perform markedly worse than M-MSE in terms of forecast MSE, 
#     despite increased smoothness (larger HT, fewer zero-crossings). 
#   -This outcome is quite remarkable and pleads in favor of the M-SSA components predictor (as a 
#     potentially more interesting/relevant predictor than M-MSE).


# Summary
# -M-SSA and M-MSE component predictors perform similarly in terms of out-of-sample MSE forecast 
#     performances when targeting forward-shifted BIP
#   -Both predictors outperform the mean, the direct forecast, the direct HP forecast and the original 
#     M-SSA predictor (tutorial 7.3), specifically at larger forward shifts (shifts>=1 quarter).
# -The M-SSA component predictor is smoother (less noisy, fewer zero-crossings) 
#   -The smoothness of M-SSA can be controlled by the HT hyperparameter
# -Interestingly, increased smoothness of M-SSA does not impair (out-of-sample) MSE forecast performances or 
#     lag (retardation) when references against M-MSE
#   -Therefore we may prefer the M-SSA componnet predictor (over M-MSE) in this application.
# -The M-MSE component predictor can be replicated by the M-SSA component predictor by inserting the 
#     former's HTs into the constraint
#   -The M-SSA component predictor is more general 
#   -The optimization principle offers more control on important characteristics (`shape') of the predictor 


