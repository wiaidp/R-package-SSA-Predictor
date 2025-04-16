# The M-SSA components forecast of tutorial 7.3 is a rather complex design, involving multiple steps 
#     in the derivation of the BIP predictor
# Construction steps:
#   -Filtering: remove undesirable high frequency noise 
#     -Filter based on HP(160) target to emphasizes mid-term dynamics relevant in a mid-term (2-4 quarters) 
#       forecast perspective
#   -Optimization criterion: maximize the target correlation under a HT (holding time) constraint
#   -Target HP-BIP (two-sided HP applied to BIP) or BIP (WLS regression of M-SSA components on BIP)
# -In order to check pertinence and relevance of this construction principle, we verified outperformance
#     of the M-SSA predictor over the simple mean (mean of BIP) and the direct forecasts (regressing indicators 
#     on future BIP)
# -However, at this stage, we are still unable to assess the importance of M-SSA (the multivariate extension 
#     of SSA), as based on the VAR-model of the DGP (data generating process).
# -Question: why does the M-SSA components predictor perform better? Which steps (in ist construction) are relevant? 

# -To answer this question we here consider a simple intermediary step (simplifications of M-SSA) as additional 
#   benchmark for the M-SSA predictor: 
# -Specifically, we consider an extension of the direct forecast, called direct HP forecast:
#   -We apply the classic univariate HP concurrent filter (HP-C) to each indicator
#   -The direct HP forecast is obtained by regressing the HP-C filtered indicators on future BIP
# -M-SSA component, direct forecast and direct HP forecast differ only in the explanatory variables regressed on future BIP
#   -M-SSA components rely on multivariate filters 
#   -direct forecasts does not rely on filters
#   -direct HP forecast relies on univariate HP-C 
# -A comparison of these predictors will hint at the cause and origin of efficiency gains by the M-SSA predictor



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



# Load the relevant M-SSA functionalities
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))
# Utility functions for M-SSA, see tutorial 
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))
# Set of performance metrics and tests of unequal predictability
#source(paste(getwd(),"/R/performance_statistics_functions.r",sep=""))


#------------------------------------------------------------------------
# Load data and select indicators: see tutorial 7.2 for background
load(file=paste(getwd(),"\\Data\\macro",sep=""))
tail(data)
lag_vec<-c(2,rep(0,ncol(data)-1))
# Note: we assume a publication lag of two quarters for BIP, see the discussion in tutorial 7.2

# Plot the data
# The real-time BIP (red) is lagging the target (black) by lag_vec[1] quarters (publication lag)
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
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()

# Select macro indicators for M-SSA 
select_vec_multi<-c("BIP","ip","ifo_c","ESI","spr_10y_3m")
x_mat<-data[,select_vec_multi] 
rownames(x_mat)<-rownames(data)
n<-dim(x_mat)[2]
# Number of observations
len<-dim(x_mat)[1]

###############################################################################################
# Exercise 1: Compute the new benchmark (direct HP forecast)
# 1.1 HP filter

lambda_HP<-160
L<-31

HP_obj<-HP_target_mse_modified_gap(L,lambda_HP)
# Classic concurrent (one-sided) HP filter
hp_c<-HP_obj$hp_trend
# This is a finite version of the symmetric HP-trend filter
#   It is a causal (one-sided) filter
#   In applications, the filter is centered at t: it is acausal (expands equally in the past and in the future) 
ts.plot(hp_c,main=paste("Concurrent HP, lambda=",lambda_HP,sep=""))
# Filter coefficients add to 1: it is a lowpass 
# Specify the number of equidistant frequency ordinates in [0,pi]
K<-600
# Compute transfer, amplitude and shift functions (shift=phase divided by frequency)
amp_obj_hp_c<-amp_shift_func(K,hp_c,F)

# Plot amplitude function
par(mfrow=c(1,1))
mplot<-matrix(amp_obj_hp_c$amp,ncol=1)
colnames(mplot)<-paste("Concurrent HP, lambda=",lambda_HP,sep="")
colo<-c("blue",rainbow(ncol(mplot)))
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP, lambda=",lambda_HP,sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
if (ncol(mplot)>1)
{
  lines(mplot[,2],col=colo[2])
  abline(v=which(mplot[,1]==max(mplot[,1])),col=colo[1])
  mtext(colnames(mplot)[1],line=-1,col=colo[1])
  
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:4*K/4,labels=expression(0, pi/4, 2*pi/4,3*pi/4,pi))
axis(2)
box()

# 1.2 Filter indicators
hp_c_mat<-NULL
for (i in 1:ncol(x_mat))
{
  hp_c_mat<-cbind(hp_c_mat,filter(x_mat[,i],hp_c,side=1))
}
colnames(hp_c_mat)<-colnames(x_mat)
rownames(hp_c_mat)<-rownames(x_mat)

# Plot
mplot<-hp_c_mat
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("Concurrent HP(",lambda_HP,") applied to indicators",sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# 1.3 Compute filter outputs for forecast horizons 0:6, assuming WN data
#   -If the data is WN, then the best forecast is zero and the resulting HP forecast is obtained 
#     by skipping the first filter coefficients

# Forecast horizons
h_vec<-0:6
hp_c_array<-array(dim=c(ncol(x_mat),nrow(x_mat),length(h_vec)))
for (j in 1:length(h_vec))
{
  for (i in 1:ncol(x_mat))
  {
# For forecast horizon h_vec[j], the firdt h_vec[j] filter coefficients are skipped (and the filter is fowrd-shifted)   
    hp_c_forecast<-c(hp_c[(h_vec[j]+1):L],rep(0,h_vec[j]))
    hp_c_array[i,,j]<-filter(x_mat[,i],hp_c_forecast,side=1)
  }
}
dimnames(hp_c_array)[[1]]<-colnames(x_mat)
dimnames(hp_c_array)[[2]]<-rownames(x_mat)
dimnames(hp_c_array)[[3]]<-paste("h=",h_vec,sep="")

# Plot
# Select an indicator
i<-1
colnames(x_mat)[i]
# Plot HP nowcast and forecasts for that indicator: we scale the series for better visual inspection
mplot<-scale(hp_c_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("HP(",lambda_HP,") now- and forecasts applied to ",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()
# For increasing forecast horizon, the filter outputs tend to be left-shifted



###############################################################################################
# Exercise 2: Compute M-SSA-components predictor together with direct forecast and direct HP forecast
#   -Parts of the following code are cut and paste from tutorial 7.3, exercises 7.1 and 7.3
#   -But we add the new direct HP forecast

lambda_HP<-lambda_HP
# Filter length: nearly 8 years is fine for the selected lambda_HP (filter weights decay sufficiently fast)
#   The length should be an odd number (see tutorial 7.1)
L<-L
# In-sample span for VAR, i.e., M-SSA (the proposed design is quite insensitive to this specification because the VAR is parsimoniously parameterized
date_to_fit<-"2008"
# VARMA model orders: keep the model simple in particular for short/tight in-sample spans
p<-1
q<-0
# Holding-times (HT): controls smoothness of M-SSA (the following numbers are pasted from the original predictor)
# Increasing these numbers leads to predictors with less zero-crossings (smoother), see tutorial 7.1
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Forecast horizons: M-SSA is optimized for each forecast horizon in h_vec 
h_vec<-0:6
# Forecast excesses: see tutorial 7.2, exercise 2 for background
f_excess<-rep(4,length(select_vec_multi))

# Run the wrapper  
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)

# Retrieve predictors and targets from the above function-call
# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA predictors: for each forecast horizon h( the columns of predictor_mssa_mat), the M-SSA predictor is 
#   obtained as equally-weighted average of all series outputs: BIP, ip,..., see exercise 3 of tutorial 7.2
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE predictors
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat



# Initialize performance matrices
p_mat_mssa<-p_mat_mssa_components<-p_mat_mssa_components_without_covid<-p_mat_direct<-rRMSE_mSSA_comp_direct<-
rRMSE_mSSA_comp_mean<-rRMSE_mSSA_comp_direct_without_covid<-rRMSE_mSSA_comp_mean_without_covid<-
rRMSE_mSSA_direct_mean_without_covid<-rRMSE_mSSA_direct_mean<-p_mat_direct_without_covid<-
p_mat_HP_c<-p_mat_HP_c_without_covid<-MSE_oos_HP_c<-MSE_oos_HP_c_without_covid<-
rRMSE_mSSA_comp_HP_c<-rRMSE_mSSA_comp_HP_c_without_covid<-matrix(ncol=length(h_vec),nrow=length(h_vec)-1)
# Use WLS
use_garch<-T
# Set-up progress bar: indicates progress in R-console
pb <- txtProgressBar(min=min(h_vec),max=max(h_vec)-1,style=3)

# The following double loop computes all combinations of forward-shifts (of BIP) and forecast horizons (of M-SSA)
# -We compute:
#   A.The M-SSA components predictor
#   B.The direct forecast
#   C.The direct HP forecast
# -For each predictor we compute HAC-adjusted p-values (significance of out-of-sample predictor) and 
#  out-of-sample rRMSEs (relative root mean-square forecast errors)

for (shift in 0:5)#shift<-2
{
# Progress bar: see R-console
  setTxtProgressBar(pb, shift)
  for (j in h_vec)#j<-5
  {
# Horizon j corresponds to k=j+1-th entry of array    
    k<-j+1
# A. M-SSA components
# For a single predictor (vector) one does not have to rely on the transposition t(mssa_array[sel_vec_pred,,k])   
    if (length(sel_vec_pred)>1)
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,k]))
    } else
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mssa_array[sel_vec_pred,,k]))
    }
    rownames(dat)<-rownames(x_mat)
    dat<-na.exclude(dat)
# Apply the previous function    
    perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift)
# Retrieve out-of-sample performances 
# a. p-values with/without Pandemic    
    p_mat_mssa_components[shift+1,k]<-perf_obj$p_value
    p_mat_mssa_components_without_covid[shift+1,k]<-perf_obj$p_value_without_covid
# b. MSE forecast error out-of-sample
#   -M-SSA components with/without Pandemic    
    MSE_oos_mssa_comp<-perf_obj$MSE_oos
    MSE_oos_mssa_comp_without_covid<-perf_obj$MSE_oos_without_covid
#   -mean-benchmark with/without Pandemic    
    MSE_mean_oos<-perf_obj$MSE_mean_oos
    MSE_mean_oos_without_covid<-perf_obj$MSE_mean_oos_without_covid
# Here we retrieve the final in-sample predictor (based on full-sample WLS regression) and the 
#    out-of-sample predictor (re-adjusted at each time point)
# We can plot both predictors to illustrate revisions (due to WLS estimation at each time point)
# Note: the variables will be overwritten, i.e., we keep only the last one corresponding to maximal shift
#   and maximal forecast horizon 
    final_components_preditor<-perf_obj$final_in_sample_preditor
    oos_components_preditor<-perf_obj$cal_oos_pred
# We can also obtain the regression weights to track changes (systematic vs. noisy revisions) over time
# Note: the variable will be overwritten, i.e., we keep only the last one corresponding to maximal shift
#   and maximal forecast horizon 
    track_weights<-perf_obj$track_weights
    
# B. Direct forecasts
# -The main difference to M-SSA above is the specification of the explanatory variables in the data 
#     matrix dat: here x_mat 
#   -We select all indicators (one could easily change this setting but results are only marginally effected as long as ifo and ESi are included)
#   -Note that the data matrix here does not depend on j, in contrast  to the M-SSA components above    
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),x_mat)
# Same but only ifo and ESI    
#    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),x_mat[,c("ifo_c","ESI")])
    rownames(dat)<-rownames(x_mat)
    dat<-na.exclude(dat)
    
    perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift)
    # Retrieve out-of-sample performances: p-values and forecast MSE, with/without Pandemic 
    p_mat_direct[shift+1,k]<-perf_obj$p_value 
    p_mat_direct_without_covid[shift+1,k]<-perf_obj$p_value_without_covid 
    MSE_oos_direct<-perf_obj$MSE_oos
    MSE_oos_direct_without_covid<-perf_obj$MSE_oos_without_covid
    
# C. HP concurrent
# -The main difference to M-SSA and direct forecasts is the specification of the explanatory variables in the data 
#     matrix dat: here hp_c_array[,,k] 
#   -We select all indicators (one could easily change this setting but results are only marginally effected as long as ifo and ESi are included)
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(hp_c_array[,,k]))
    rownames(dat)<-rownames(x_mat)
    dat<-na.exclude(dat)

    perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift)
# Retrieve out-of-sample performances: p-values and forecast MSE, with/without Pandemic 
    p_mat_HP_c[shift+1,k]<-perf_obj$p_value 
    p_mat_HP_c_without_covid[shift+1,k]<-perf_obj$p_value_without_covid 
    MSE_oos_HP_c<-perf_obj$MSE_oos
    MSE_oos_HP_c_without_covid<-perf_obj$MSE_oos_without_covid
    
    
# Compute rRMSEs
# a. M-SSA Components vs. direct forecast    
    rRMSE_mSSA_comp_direct[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_oos_direct)
# b. M-SSA Components vs. mean benchmark    
    rRMSE_mSSA_comp_mean[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_mean_oos)
# c. Direct forecast vs. mean benchmark    
    rRMSE_mSSA_direct_mean[shift+1,k]<-sqrt(MSE_oos_direct/MSE_mean_oos)
# d. M-SSA Components vs. HP concurrent 
    rRMSE_mSSA_comp_HP_c[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_oos_HP_c)
# Same as a, b, c, d but without Pandemic
    rRMSE_mSSA_comp_direct_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_oos_direct_without_covid)
    rRMSE_mSSA_comp_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_mean_oos_without_covid)
    rRMSE_mSSA_direct_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_direct_without_covid/MSE_mean_oos_without_covid)
    rRMSE_mSSA_comp_HP_c_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_oos_HP_c_without_covid)
  }
}
close(pb)
# Note: possible warnings issued by the GARCH estimation routine during computations can be ignored

# Assign column and rownames
colnames(p_mat_mssa_components)<-colnames(p_mat_direct)<-colnames(p_mat_mssa_components_without_covid)<-
colnames(rRMSE_mSSA_comp_direct)<-colnames(rRMSE_mSSA_comp_mean)<-
colnames(rRMSE_mSSA_comp_direct_without_covid)<-colnames(rRMSE_mSSA_comp_mean_without_covid)<-
colnames(rRMSE_mSSA_direct_mean)<-colnames(rRMSE_mSSA_direct_mean_without_covid)<-
colnames(p_mat_direct_without_covid)<-
colnames(p_mat_HP_c)<-colnames(p_mat_HP_c_without_covid)<-
colnames(rRMSE_mSSA_comp_HP_c)<-colnames(rRMSE_mSSA_comp_HP_c_without_covid)<-paste("h=",h_vec,sep="")

rownames(p_mat_mssa_components)<-rownames(p_mat_direct)<-rownames(p_mat_mssa_components_without_covid)<-
rownames(rRMSE_mSSA_comp_direct)<-rownames(rRMSE_mSSA_comp_mean)<-
rownames(rRMSE_mSSA_comp_direct_without_covid)<-rownames(rRMSE_mSSA_comp_mean_without_covid)<-
rownames(rRMSE_mSSA_direct_mean)<-rownames(rRMSE_mSSA_direct_mean_without_covid)<-
rownames(p_mat_direct_without_covid)<-
rownames(p_mat_HP_c)<-rownames(p_mat_HP_c_without_covid)<-rownames(rRMSE_mSSA_comp_HP_c)<-rownames(rRMSE_mSSA_comp_HP_c_without_covid)<-
paste("Shift=",0:5,sep="")

# HAC-adjusted p-values of out-of-sample (M-SSA) components predictor when targeting forward-shifted BIP
#   -Evaluation based on out-of-sample span starting at in_out_separator and ending on Jan-2025
# We here compare M-SSA components predictor with direct forecast and classic concurrent HP
p_mat_mssa_components
p_mat_direct
p_mat_HP_c

# Same but without singular Pandemic
p_mat_mssa_components_without_covid
p_mat_direct_without_covid
p_mat_HP_c_without_covid


# Findings:
# -Like the classic direct forecast, the new direct HP forecast is unable to forecast BIP at shifts larger than a quarter (+ publication lag)
#   -p-values are larger 0.05 for shift>=2
#   -For shift<=1 the direct HP forecast seems slightly better (smaller p-values) than the direct forecasts 
#     on data including the Pandemic (more robust?). 
# -In contrast, the M-SSA components predictor remains significant for shifts up to four quarters (plus publication lag)
 


# We can compare the M-SSA componsts predictor to the direct forecast and HP-c in terms of rRMSEs out-of-sample
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_comp_HP_c_without_covid

# -The above results suggest that applying the classic (univariate) HP-c to the data does not improve performances
#   when compared to direct forecasts
# -Therefore, outperformance by the M-SSA components predictor cannot be replicated/explained by a `simple` 
#   univariate (HP-) filtering of the indicators
# -On the other hand, outperformance of the direct forecasts (or HP) by M-SSA suggests that the multivariate  
#   aspect cannot be efficiently handled by (WLS) regression, eventually combined with univariate filtering.
# -We may infer that the BIP forecast problem requires a simultaneous treatment of longitudinal and 
#   cross-sectional aspects, such as provided by M-SSA in combination with (WLS-) regression 
# -To back-up the last claim we compare classic HP-c filtered indicators with M-SSA components
#   -Both procedures share the same common target, namely the two-sided HP applied to an indicator
#   -The classic HP-C relies on the classic univariate concurrent HP, deemed relevant in business-cycle analysis
#   -The M-SSA relies on multiple time series for each target, emphasizing the target correlation (in the 
#     objective function) and the HT (in the constraint)

# In order to understand the contribution of the multivariate framework (over a univariate filter) we here 
#   consider BIP (lagging) and spread (leading) targets
# Background:
# -We expect the multivariate design to extract relevant information from leading series when targeting a 
#   lagging series, i.e., we expect most efficiency gains of M-SSA over HP-C when targeting HP-BIP (the two-sided HP applied to BIP) 
# -In contrast, when targeting a leading series, we expect gains of the multivariate filter to be less significant, 
#   i.e., we expect least efficiency gains of M-SSA over HP-C when targeting HP-spread (two-sided HP applied to spread)
# -In this comparison, M-SSA and HP-C share the same target (two-sided HP applied to a series) but M-SSA can rely
#   on multiple series (explanatory variables) and a more sophisticated optimization framework

# To verify the above conjecture we now generate a main plot with 4 sub-panels: 
# -panel a: HP-C aplied to BIP
# -panel b: M-SSA applied to BIP (in this case we expect M-SSA to outperform HP-C) 
# -panel c: HP_c applied to spread
# -panel d: M-SSA applied to spread (in this case we do not expect M-SSA to outperform HP-C substantially, if at all) 

# a. HP-C appliued to BIP
i<-1
colnames(x_mat)[i]
# Plot HP-C for that indicator: we scale the series for better visual inspection
mplot<-scale(hp_c_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
par(mfrow=c(2,2))
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("HP-C targeting ",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()
# b. M-SSA applied to BIP 
mplot<-scale(mssa_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("M-SSA targeting ",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# c. HP_C applied to spread
i<-5
colnames(x_mat)[i]
# Plot HP-C for that indicator: we scale the series for better visual inspection
mplot<-scale(hp_c_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("HP-C targeting ",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()
# d. M-SSA applied to spread 
mplot<-scale(mssa_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("M-SSA targeting ",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Findings:
# -Let us first focus attention on the top two panels: targeting HP-BIP
#   -The main difference between the classic HP-C (top left) and M-SSA (top right) is the size and the 
#     `quality' of the left-shift as a function of the forecast horizon:
#   -Size: the left-shift at the zero-crossings is stronger with M-SSA
#   -Quality: 
#     -In contrast to HP-C, M-SSA also leads to a (more pronounced) left-shift of dips and peaks, 
#       in particular at recessions/crises
#     -The left-shift operates at all levels: not only at zero-crossings but also at local peaks and dips and
#       at any level in between
# -Let us now look at the bottom two panels: targeting HP-spread
#   -In this case, M-SSA and HP-C are nearly identical (no outperformance)
#   -The explanation is pretty simple: 
#     -The multivariate M-SSA filter targeting HP-spread is nearly identical to the univariate filter,  
#       see tutorial 7.1, exercise 1.5
#     -This is because the lagging explanatory series are (nearly) irrelevant when  
#       targeting the leading (in relative terms) HP-spread

# Summary:
# Applying a classic one-sided HP-C does not improve forecast performances over the classic direct predictor: 
#   -the forecast problem is more complex 
#   -predicting BIP takes advantage of a more effective treatment/extraction of the available information

# Key elements of the design of the M-SSA component predictor are
#   -Emphasis of the relevant mid-term components (HP(160) target)
#     -This explains why M-SSA outperforms classic direct forecast (which do not damp undesirable high-frequency noise)
#   -Efficient exploitation of longitudinal and cross-sectional information when targeting HP-BIP
#     -Exploit leading (in relative terms) indicators to (significantly) improve forecasts of HP-BIP, up to multiple quarters ahead
#     -Address the size and the quality of the left-shift of the multivariate predictor (with increasing forecast horizon)
#   -Explains why M-SSA designs outperform direct forecast as well as direct HP forecast (based on univariate HP)

# These findings justify the  rather complex forecast design, wherein the novel (M-SSA) optimization framework 
#   ensures an effective information retrieval along the time axis and across series, when tracking BIP.