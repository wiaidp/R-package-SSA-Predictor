# Tutorial 7.3: we propose various M-SSA BIP (GDP Germany) predictor designs
# -The concept of M-SSA predictors for BIP was introduced in tutorial 7.2
# -We wrapped this proceeding into a single function to be able to analyze various M-SSA BIP predictor designs (hyperparameters)
# -In exercise 1, we propose a `fairly adaptive' predictor. A `more adaptive' one is analyzed in exercise 4 and a `more inflexible' one in exercise 5
#   -One might be able to find better hyperparameters by fine-tuning adaptivity further

# Main purposes of this tutorial
# -Illustrate M-SSA as applied to real data (in contrast to tutorial 7.1, based on simulated data)
#   -The application considers nowcasting and forecasting of German GDP (BIP) based on a set of well-known indicators
# -For the sake of interest, we shall consider forecast horizons of up to 5 quarters (one and a half year)
#   -Performances of institutional forecasters (`big five' German forecast institutes) degrade steeply 
#     beyond a one quarter forecast horizon, see up-coming publication by Heinisch and Neufing (currently working paper)
#   -We here illustrate that BIP can possibly be predicted consistently beyond half a year ahead
#     -We emphasize mid-term predictability: up to 5 quarters ahead
#   -Institutional forecasters are very good at nowcasting GDP: indeed, quite better than M-SSA presented here
#     -For this purpose they rely on a rich cross-section (many series) and mixed-frequency approaches, linking monthly and quarterly data
#     -In contrast, M-SSA proposed in this tutorial considers few (important) indicators, within a purely quarterly scheme 
#   -Tutorial 7.3 could eventually provide additional insights into the important prospect of mid-term GDP/BIP forecasting
# -In addition to forecasting BIP, we also consider nowcasting and forecasting of the trend-growth of BIP
#   -For this purpose we apply a HP-filter to the differenced (and log-transformed) BIP
#     -This trend-growth component is termed HP-BIP
# -Forecast performance measures:
#   -We shall consider forecast performances of M-SSA against forward-shifted HP-BIP and BIP based on
#     -The target correlation: correlation of predictor with forward-shifted BIP or HP-BIP
#     -The rRMSE: relative root mean-square error when benchmarked against classic direct predictors or mean(BIP) (simple benchmark)
#     -HAC-adjusted p-values of (t-statistics of) regressions of predictors on targets (HAC adjustment can account for autocorrelation and heteroscedasticity of regression residuals)
# To do: provide additional Diebold-Mariano (DM) and Giacomini-White (GW) tests of unequal predictability (benchmarked against mean(BIP))

# The tutorial is structured into five exercises
# Exercise 1: apply a fairly adaptive design based on targeting a HP(160) filter by M-SSA
#   -HP(160) deviates from the standard HP(1600) specification typically recommended for quarterly data
#     -See a critic by Phillips and Jin (2021), suggesting that HP(1600) is `too smooth' (insufficiently flexible)
#     -See also the lengthy discussion in tutorial 7.2 
#     -See also exercise 5 below
#   -We shall see that M-SSA can predict HP-BIP (for which it is explicitly optimized) consistently 
#      multiple quarters ahead (statistical significance)
#   -It is more difficult to predict BIP, though: the noisy high-frequency components of BIP are unpredictable 
# Exercise 2: apply M-SSA to white noise data to verify that the proposed performance measures (and tests) confirm unpredictability
#   -We shall see that the proposed HAC-adjustment cannot fully account for all data idiosyncrasies
#   -However, empirical significance levels do not appear to be strongly biased
# Exercise 3: the proposed M-SSA predictor emphasizes dynamic changes of the trend growth-rate of BIP. It is 
#   not designed explicitly for mean-square forecast performances when targeting BIP. Therefore, exercise 3 proposes
#   to rely on the (sub-) components of the M-SSA predictor when targeting BIP (instead of HP-BIP). We shall see that the 
#   corresponding predictor is able to outperform the classic mean-benchmark as well as the direct forecasts in 
#   terms of out-sample MSE performances up to multiple quarters ahead.
# Exercise 4: analyze a more adaptive M-SSA design based on targeting HP(16) by M-SSA
# Finally, exercise 5 briefly analyzes the classic HP(1600) as a target for M-SSA


#-------------------------------------
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
# Exercise 1: Compute forecasts for German GDP up to 5 quarters ahead, based on the above selection of 
#   macro-indicators
# -Rely on a `fairly adaptive' HP(160) as the target, based on lambda_HP=160, 
#   -The standard HP(1600) is too smooth (smooths out recessions)
#   -But BIP (German GDP) is too noisy to be forecasted `directly'
#   -Finding a target that emphasizes the relevant dynamics while damping the unpredictable 
#     high-frequency noise is challenging
#   -HP(160) is possibly a good compromise: it damps high-frequency noise sufficiently strongly 
#     to let the `signal` (i.e. the predictable business-cycle part of BIP) appear more clearly to 
#     the forecast fitting-tool: M-SSA
# -In exercise 3 we shall rely on an even more adaptive design based on HP(16)

# 1.1 Apply M-SSA
# Here's the wrapper proposed in tutorial 7.2: 
head(compute_mssa_BIP_predictors_func)

# The head of the function needs the following specifications:
# -x_mat: data 
# -lambda_HP: HP parameter
# -L: filter length
# -date_to_fit: in-sample span for the VAR
# -p,q: model orders of the VAR
# -ht_mssa_vec: HT constraints (larger means less zero-crossings)
# -h_vec: (vector of) forecast horizon(s) for M-SSA
# -f_excess: forecast excesses, see tutorial 7.2, exercise 2 
# -lag_vec: publication lag (target is forward shifted by forecast horizon plus publication lag)
# -select_vec_multi: names of selected indicators

# We can supply various hyperparameters (designs) and the function returns M-SSA predictors as 
#     specified in tutorial 7.2
# -The main hyperparameter is lambda_HP: a smaller lambda_HP means increased adaptivity
# -Below, we shall also consider various settings for the forecast horizon (Timeliness) and the HT (smoothness) 

# We now first replicate tutorial 7.2 with the above wrapper

# Target filter: lambda_HP is the single most important hyperparameter, see tutorial 7.1 for a discussion
lambda_HP<-160
# Filter length: nearly 8 years is fine for the selected lambda_HP (filter weights decay sufficiently fast)
#   The length should be an odd number (see tutorial 7.1)
L<-31
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
# Sub-series of M-SSA predictors: for each forecast horizon h, the M-SSA predictor is obtained as equally-weighted 
#   average of all series outputs: BIP, ip,...
# The components or sub-series of the aggregate predictor are contained in mssa_array 
# This is a 3-dim array with dimensions: sub-series (BIP, ip,...), time, forecast horizon h in h_vec
#   -We shall see below that the sub-series can be important when interpreting the M-SSA predictors 
mssa_array<-mssa_indicator_obj$mssa_array

# Plot M-SSA: the vertical line indicates the end of the in-sample span
mplot<-predictor_mssa_mat
colnames(mplot)<-colnames(predictor_mssa_mat)
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(predictor_mssa_mat)))
main_title<-c(paste("Standardized M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep=""),"Vertical line delimites in-sample and out-of-sample spans")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
abline(v=which(rownames(mplot)>date_to_fit)[1]-1,lty=2)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# How can we relate the above plot to the AST forecast trilemma?
# -We can see that M-SSA predictors are increasingly left-shifted with increasing forecast horizon, 
#     both in- as well as out-of-sample
#   -Timeliness aspect of the AST trilemma
# -Also, the number of zero-crossings is controlled by the holding-time (HT constraint)
#   -Smoothness aspect of the AST trilemma
# -The remaining A (for Accuracy) is examined below 

#----------------------------------------------------------
# 1.2. Compute performances
# 1.2.1 Specify the benchmarks against which M-SSA will be compared
#   -We compare M-SSA against the mean (of BIP) and against a classic direct forecast
#   -The direct forecast is based on regressing a selection of macro indicators on forward-shifted BIP

# We can specify the selection of macro-indicators for the direct forecast 
select_direct_indicator<-c("ifo_c","ESI")
# Note: too complex designs (too many indicators) lead to overfitting and thus worse out-of-sample performances
# To illustrate the direct predictor consider the following example of a h-step ahead direct forecast:
h<-2
# Shift BIP forward by publication lag+forecast horizon
forward_shifted_BIP<-c(x_mat[(1+lag_vec[1]+h):nrow(x_mat),"BIP"],rep(NA,h+lag_vec[1]))
# Regress selected indicators on forward-shifted BIP
lm_obj<-lm(forward_shifted_BIP~x_mat[,select_direct_indicator])
# You will probably not find statistically significant regressors for h>2: BIP is a noisy series
summary(lm_obj)
# Technical note: 
# -Residuals are subject to heteroscedasticity (crises) and autocorrelation
# -Therefore classic OLS tests for statistical significance are biased
# -We shall rely on HAC-adjusted p-values further down (R-package sandwich)

# Compute the predictor: one can rely on the generic R-function predict or compute the predictor manually
direct_forecast<-lm_obj$coef[1]+x_mat[,select_direct_indicator]%*%lm_obj$coef[2:(length(select_direct_indicator)+1)]
# Note that this is a full-sample predictor (no out-of-sample span)

# We can now plot target and direct forecast: for h>2 the predictor comes close to a flat line centered at zero 
ts.plot(cbind(forward_shifted_BIP,direct_forecast),main=paste("BIP shifted forward by ",h," quarters (black) vs. direct forecast (red)"),col=c("black","red"))
abline(h=0)

#------------
# 1.2.2 Performances
# The following function computes direct forecasts and evaluates M-SSA against mean(BIP) and direct forecasts
# -Performance measures:
#   -Target correlations (correlations of predictors with targets)
#     -Correlations emphasize the dynamic aspects of forecasts: they ignore static level and scale adjustments
#   -Relative root mean-square errors: potential gains of M-SSA over mean(BIP) and direct forecasts
#   -HAC-adjusted p-values: 
#     -we regress the predictor on forward-shifted BIP and compute the t-statistic 
#       of the regression coefficient
#     -Small p-values indicate statistical significance of the predictor
#     -Since target and predictor are subject to heteroscedasticity (COVID-outliers...) and autocorrelation
#       we here rely on HAC-adjusted p-values (R-package sandwich)
#   -We compute full-sample and out-of-sample results: the entire financial crisis is out-of-sample
#   -We consider two different targets
#     a. Forward-shifted HP applied to BIP: HP_BIP. This is the target for which M-SSA has been optimized
#     b. Forward-shifted BIP, i.e., we include the (unpredictable) noisy high-frequency part of BIP
# Note:
#   -M-SSA does not target BIP directly; overfitting is not (less) of concern

# We select the out-of-sample span so that the entire financial crisis remains out-of-sample for all forecast horizons. 
# Technical note: 
#   -In principle date_to_fit and in_out_separator should be the same. 
#   -But the MTS package stubbornly refuses to fit the VAR when date_to_fit=2007 (error message). 
#   -In any case, the VAR does not `see` the financial crisis, which is effectively out-of-sample. 
#   -Setting in_out_separator<-"2007" ensures that the financial crisis remains in the out-of-sample evaluation 
#     span even for larger forecast horizons (so that comparisons remain meaningful for all forecast horizons)
in_out_separator<-"2007"
perf_obj<-compute_perf_func(x_mat,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,in_out_separator,select_direct_indicator,h_vec) 

# Retrieve all performance measures
p_value_HAC_HP_BIP_full=perf_obj$p_value_HAC_HP_BIP_full
t_HAC_HP_BIP_full=perf_obj$t_HAC_HP_BIP_full
cor_mat_HP_BIP_full=perf_obj$cor_mat_HP_BIP_full
p_value_HAC_HP_BIP_oos=perf_obj$p_value_HAC_HP_BIP_oos
t_HAC_HP_BIP_oos=perf_obj$t_HAC_HP_BIP_oos
cor_mat_HP_BIP_oos=perf_obj$cor_mat_HP_BIP_oos
p_value_HAC_BIP_full=perf_obj$p_value_HAC_BIP_full
t_HAC_BIP_full=perf_obj$t_HAC_BIP_full
cor_mat_BIP_full=perf_obj$cor_mat_BIP_full
p_value_HAC_BIP_oos=perf_obj$p_value_HAC_BIP_oos
t_HAC_BIP_oos=perf_obj$t_HAC_BIP_oos
cor_mat_BIP_oos=perf_obj$cor_mat_BIP_oos
rRMSE_MSSA_HP_BIP_direct=perf_obj$rRMSE_MSSA_HP_BIP_direct
rRMSE_MSSA_HP_BIP_mean=perf_obj$rRMSE_MSSA_HP_BIP_mean
rRMSE_MSSA_BIP_direct=perf_obj$rRMSE_MSSA_BIP_direct
rRMSE_MSSA_BIP_mean=perf_obj$rRMSE_MSSA_BIP_mean
target_BIP_mat=perf_obj$target_BIP_mat

#------------
# 1.2.3 Evaluation
# We first look at the target correlations between M-SSA predictors and forward-shifted HP-BIP (including the publication lag)
# a. Full sample
cor_mat_HP_BIP_full
# We can recognize a systematic pattern: 
#   -For increasing forward-shift (from top to bottom in the above matrix) M-SSA designs optimized for 
#     larger forecast horizons h (from left to right) tend to perform better (larger correlations), 
#     until h is too large
#   -For a given row (forward-shift of target) the maximum correlations stays at (or close to) the 
#     diagonal element in this row

# b. Out-of-sample 
cor_mat_HP_BIP_oos
# Similar to full-sample

# We now look at the target correlation when the target is forward-shifted BIP (instead of HP-BIP) 
cor_mat_BIP_full
cor_mat_BIP_oos
# Target correlations tend to be smaller (due to noise), but the previous systematic pattern is still recognizable
#   but attenuated (cluttered by noise) 


# Note: 
# -Out-of-sample correlations suggest that M-SSA predictors optimized for forecast horizons h>4 (the last two columns) 
#     correlate positively with forward-shifted BIP up to shifts of 4 quarters 
# -Are these results suggesting a systematic predictability of BIP one year ahead (plus the publication lag lag_vec[1])?
# -For answering this question we may look at HAC-adjusted p-values of regressions of out-of-sample predictors on 
#   forward-shifted BIP (see further down for details)
p_value_HAC_BIP_oos
# The last predictor, optimized for h=6, is on the edge of statistical significance up to a forward-shift=3 
#   -Note that the out-of-sample span is quite short here 


# We may also look at mean-square (MSE) forecast performances.
# However, the M-SSA predictor is not explicitly designed for this purpose:
#   -It is standardized (neither level nor scale are calibrated to match BIP).
#   -The optimization criterion (the target correlation) ignores static level and scale parameters.
# Therefore, we shall propose an alternative predictor in exercise 3 below
#   -An extension based on explicit calibration of level and scale to match future BIP by minimizing MSE

# However, for the sake of curiosity, we here compute relative root MSE (rRMSE) of the M-SSA 
#   predictor against the simple mean benchmark as well as against the classic direct forecast (the latter as described in exercise 1.2.1 above
# Note that we are willingly cheating at this stage: 
# 1. We calibrate level and scale of the M-SSA predictor based on out-of-sample data
# 2. Direct forecasts are based on the full sample (including the out-of-sample span)
# 3. The mean benchmark (mean of BIP) is based on out-of-sample BIP data
# These shortcomings will be addressed in exercise 3 below. 
# For now, our `cheating' rRMSE metrics reflect dynamic aspects of the forecast problem, 
#   wherein static level and scale parameters are deemed less relevant 

# The first rRMSE emphasizes M-SSA vs. the mean benchmark (of BIP), both targeting HP-BIP: 
#   -Numbers smaller one signify an outperformance of M-SSA against the mean-benchmark when targeting HP-BIP
# The evaluation sample is to the right (after) in_out_separator 
rRMSE_MSSA_HP_BIP_mean
# We next look at M-SSA vs. direct predictors based on indicators selected  in select_direct_indicator: targeting HP-BIP
rRMSE_MSSA_HP_BIP_direct
# Next: M-SSA vs. mean (of BIP) when targeting BIP
rRMSE_MSSA_BIP_mean
# Finally: M-SSA vs. direct predictors based on indicators selected  in select_direct_indicator: targeting BIP
rRMSE_MSSA_BIP_direct
# We see similar systematic patterns as for the previous target correlations: 
#   -Designs optimized for larger forecast horizons (from left to right) tend to outperform at larger forward-shifts (from top to bottom) 
#   -The systematic patterns are strong when targeting forward-shifted HP-BIP (for which M-SSA is explicitly optimized)
#   -For forward-shifted BIP the results are less conclusive (cluttered by noise)
# But we shall see below that a more adaptive design can improve performances slightly, see exercise 4


# The above correlations and rRMSE do not test for statistical significance (of predictability)
# The following HAC-adjusted p-values provide a way to infer statistical significance 
# We look at HAC-adjusted p-values of regressions of M-SSA on forward-shifted targets
# Remarks:
#   -In some cases the HAC standard error (of the regression coefficient) seems `suspicious' 
#     -Sometimes, the HAC estimate of the variance is substantially smaller than the ordinary OLS (unadjusted) estimate
#   -We therefore compute both types of standard errors and we rely on the maximum for a derivation of p-values
#   -In this sense our p-values may be claimed to be `conservative'
# We first consider forward-shifted HP-BIP and full sample p-values
p_value_HAC_HP_BIP_full
# Out-of-sample: 
p_value_HAC_HP_BIP_oos
# We infer that the systematic patterns observed in target correlations and rRMSEs above are statistically significant
#   -M-SSA seems capable of predicting forward-shifted HP-BIP several quarters ahead
# Remark:
# -Applying HP to BIP results in an autocorrelated time series
# -Therefore, HP-BIP must be `predictable', as confirmed by the above metrics
# -Exercise 3 below then analyzes predictability when targeting BIP explicitly    




# Same as above but now targeting BIP
# -Full sample
p_value_HAC_BIP_full
# We still see a systematic pattern from top to bottom and left to right but the overall picture 
#   is cluttered by noise 

# -Out-of-sample: 
p_value_HAC_BIP_oos

# Note: 
# -The M-SSA predictor is designed to track dynamic changes of the growth rate (HP-BIP)
# -Part of the outperformance of M-SSA is due to inclusion of the financial crisis in the 
#   evaluation span.
# -In the absence of crises (business-cycle), gains should be null (or negative)  

#-------------------------------------------
# 1.3 Visualize performances: link performance measures to plots of predictors against targets
# 1.3.1 M-SSA predictors (without sub-series)
# Select a forward-shift of target (the k-th entry in h_vec)
k<-5
# This is the forecast horizon 
h_vec[k]
# To obtain the effective forward-shift we have to add the publication lag
shift<-h_vec[k]+lag_vec[1]
# Effective forward shift of BIP
shift
if (k>length(h_vec))
{
  print(paste("k should be smaller equal ",length(h_vec),sep=""))
  k<-length(h_vec)
}  
# Select a M-SSA predictor: optimized for forecast horizon h_vec[j]
j<-k
if (j>length(h_vec))
{
  print(paste("j should be smaller equal ",length(h_vec),sep=""))
  j<-length(h_vec)
}  
# Plot targets (forward-shifted BIP and HP-BIP) and predictor
par(mfrow=c(1,1))
# Scale the data for better visual interpretation of effect of excess forecast on M-SSA (red) vs. previous M-SSA (blue)
mplot<-scale(cbind(target_BIP_mat[,k],target_shifted_mat[,k],predictor_mssa_mat[,j]))
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c(paste("BIP left-shifted by ",h_vec[k]," quarters (plus publication lag)",sep=""),paste("HP-BIP left-shifted by ",h_vec[k]," quarters (plus publication lag)",sep=""),paste("M-SSA predictor optimized for h=",h_vec[j],sep=""))
colo<-c("black","violet","blue")
main_title<-"Standardized forward-shifted BIP and HP-BIP vs. M-SSA predictor"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(x_mat)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
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


#------------------
# 1.3.2 We can now relate the above plot to the performance measures
# Sample correlation: 
cor(na.exclude(mplot))[2,ncol(mplot)]
# This number corresponds to element (k,j) of the matrix cor_mat_HP_BIP_full computed by our function
cor_mat_HP_BIP_full[k,j]

# We can also look at the correlation of the predictor with the forward-shifted BIP (instead of HP-BIP)
cor(na.exclude(mplot))[1,3]
# Note: the (k,j) entry of cor_mat_BIP generally differs from cor(na.exclude(mplot))[1,3]
cor_mat_BIP_full[k,j]
# The reason is simple: by removing NAs (due to inclusion of the two-sided target in mplot) we change the sample-size
# We can easily correct by removing the two-sided target in mplot
mplot_without_two_sided<-scale(cbind(target_BIP_mat[,k],predictor_mssa_mat[,j]))
# This number now matches cor_mat_BIP[k,j]
cor(na.exclude(mplot_without_two_sided))[1,2]

# Assume one selects k=j=5 (h_vec[5]=4: one-year ahead forecast) in the above plot:
# Then the (weak) positive correlation between M-SSA and shifted BIP might suggest a (weak) predictability one year ahead
# Is this (weak) effect statistically significant?
# Let's have a look at the HAC-adjusted p-values
# 1. Full sample 
p_value_HAC_BIP_full[k,j]
# 2. Out-of-sample
p_value_HAC_BIP_oos[k,j]
# Not significant at a one-year ahead horizon when targeting BIP (which is a very noisy series...)
#   -See exercise 3 for further results

# Instead of BIP we might have a look at targeting HP-BIP (also shifted one year ahead)
p_value_HAC_HP_BIP_full[k,j]
p_value_HAC_HP_BIP_oos[k,j]



#---------------
# 1.4 Changing the HT
# -In the above exercises we addressed timeliness by the forecast horizon h_vec and the forecast excess f_excess
# -Here we briefly address smoothness by increasing the HT in the HT-constraint
# -This exercise can be viewed as a further validity or robustness check for the M-SSA predictor

# We impose twice as large HTs
ht_mssa_vec_long<-2*ht_mssa_vec
names(ht_mssa_vec_long)<-colnames(x_mat)
# Stronger smoothing might require longer filters 
#   -We here ignore this effect and set L_long=L (advantage of shorter filters: faster numerical computation)
L_long<-L

# Run the wrapper  
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L_long,date_to_fit,p,q,ht_mssa_vec_long,h_vec,f_excess,lag_vec,select_vec_multi)

target_shifted_mat=mssa_indicator_obj$target_shifted_mat
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat
mssa_array<-mssa_indicator_obj$mssa_array

# Plot 
mplot<-predictor_mssa_mat
colnames(mplot)<-colnames(predictor_mssa_mat)
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(predictor_mssa_mat)))
main_title<-c(paste("Standardized M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep=""),"Vertical line delimites in-sample and out-of-sample spans")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
abline(v=which(rownames(mplot)>date_to_fit)[1]-1,lty=2)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# -We can observe increased smoothness of the predictors, as expected 
#   -The expected duration between consecutive zero-crossings should double (when compared to the previous results) 
#     on sufficiently long spans, assuming the VAR to be `true', see tutorial 7.1 for background
# -Overall, the general pattern of the predictors towards the sample end confirms earlier findings (based on smaller HTs)
# -But nowcast and short-term forecasts are more prudent with respect to up-turn dynamics


################################################################################################################
# Exercise 2
# Let's check what happens when we apply the above battery of tests and performance measures to white noise
#   -Target correlations should be small, rRMSEs should be close to one and p-values should be above 5%
# Note: 
# -We're looking into a multiple-test problem, since we consider 6*7=42 tests (p-values)
#   -We do not account/adjust for this problem. 
#   -But one should expect to see randomly significant results in the simultaneous 42 tests 
#     even if the data is white noise

# 2.1
# Generate artificial white noise data
# One can try multiple set.seed
# This selection does not generate p-values below 5%
set.seed(1)
# This one will generate several p-values below 5% in the full sample span but none below 1% 
set.seed(2)
# Multiple below 5% but none below 1%, 
set.seed(3)
# None below 1%
set.seed(4)
# None below 1%
set.seed(5)
# None below 1%
set.seed(6)
# One single p-value below 1%
set.seed(7)
# None below 1%
set.seed(8)
# None below 1%
set.seed(9)
# None below 1%
set.seed(10)

# The outcome suggests that HAC-adjustments are unable to correct fully for data-dependence
#   -We observe p-values below 5% more often than in 5% of all cases
#   -Therefore, some care is needed when evaluating results on the verge of statistical significance
# However, our results also suggest that p-values below 1% are `rare` 
#   -We found only one p-value below 1% for the above set.seeds, out of 10*6*7*2=840 computed values

x_mat_white_noise<-NULL
for (i in 1:ncol(x_mat))
  x_mat_white_noise<-cbind(x_mat_white_noise,rnorm(nrow(x_mat)))

# Provide colnames and rownames from x_mat: necessary because the function relies on dates and column names of selected indicators
rownames(x_mat_white_noise)<-rownames(x_mat)
colnames(x_mat_white_noise)<-colnames(x_mat)
tail(x_mat_white_noise)

# Check ACF
acf(x_mat_white_noise)


# 2.2 Apply M-SSA
# Target filter: lambda_HP is the single most important hyperparameter, see tutorial 7.1 for a discussion
# Briefly: we avoid the classic quarterly setting lambda_HP=1600 because the resulting filter would be too smooth
# Too smooth means: the forecast horizon would have nearly no effect on the M-SSA predictor (almost no left-shift, no anticipation)
lambda_HP<-160
# Filter length: nearly 8 years is fine for the selected lambda_HP (filter weights decay sufficiently fast)
L<-31
# In-sample span for VAR, i.e., M-SSA (the proposed design is quite insensitive to this specification because the VAR is parsimoniously parameterized)
date_to_fit<-date_to_fit
# VARMA model orders: keep the model simple in particular for short/tight in-sample spans
p<-0
q<-0
# Holding-times (HT): controls smoothness of M-SSA (the following numbers are pasted from the original predictor)
# Increasing these numbers leads to predictors with less zero-crossings (smoother)
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Forecast horizons: M-SSA is optimized for each forecast horizon in h_vec 
h_vec<-0:6
# Forecast excesses: see tutorial 7.1 for background
f_excess<-rep(4,length(select_vec_multi))

# Run the function packing and implementing our previous findings (tutorial 7.2) 
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat_white_noise,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)


# 2.3 Check performances: 
# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA indicators
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat

# For the direct predictor we can specify the macro-indicators in the expanding-window regressions
#   -Note: too complex designs lead to overfitting and thus worse out-of-sample performances
select_direct_indicator<-c("ifo_c","ESI")

perf_obj<-compute_perf_func(x_mat_white_noise,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,in_out_separator,select_direct_indicator,h_vec) 

p_value_HAC_WN_full=perf_obj$p_value_HAC_BIP_full
p_value_HAC_WN_oos=perf_obj$p_value_HAC_BIP_oos


# We need to check whether we can forecast the white noise data WN (not HP-WN)  based on the predictors
# Full sample
p_value_HAC_WN_full
# Out-of-sample: 
p_value_HAC_WN_oos

# Check number of occurrences below 5%
length(which(p_value_HAC_WN_full<0.05))
length(which(p_value_HAC_WN_oos<0.05))
# Check number of occurrences below 1%
length(which(p_value_HAC_WN_full<0.01))
length(which(p_value_HAC_WN_oos<0.01))

# Findings:
# -We did not account for the multiple-test problem
# -The above simulation experiment suggests that HAC-adjustments are unable to correct fully for the dependence
#   -Therefore, some care is needed when evaluating results on the verge of statistical significance
# -However, strongly significant results, such as found for HP-BIP, seem convincing, in a statistical sense, 
#   since occurrences with p-values below 1% are `rare' (one out of ~1000 observations) in the above experiment

# Discussion
# -HP applied to white noise (HP-WN) leads to an autocorrelated series which can be predicted (better than by the mean benchmark)
# -However, a predictor of HP-WN is not helpful in predicting WN 
# -In contrast, we expect HP-BIP in exercise 1 (and 3 below) to be informative about future BIP because the 
#     latter is not WN:
#   -The occurrence of recessions (business-cycle) contradicts the WN assumption
#   -The acf and the VAR-model contradict the WN assumption
#   -The systematic structure in the performance matrices (top-down/left-right) suggests predictability
#   -The recent negative BIP-readings are not `random` events: they are due in part to exogenous shock-waves and 
#     also to endogenous decisions whose underlying(s) did not realize as wished for 
#   -These deviations of BIP from WN concern lower frequency components of the spectral decomposition, as 
#     emphasized by HP-BIP
# -In the following exercise we will emphasize specifically BIP and MSE performances

##################################################################################
# Exercise 3 Working with (M-SSA) BIP predictor (sub-)components

# Purposes of exercise 3: 
# 1. Interpretability: M-SSA predictor components can be used for gauging the M-SSA predictor (of exercise 1)
# 2.  M-SSA predictor components can also be used to address explicitly MSE-forecast performances when targeting 
#       forward-shifted BIP
# -Explanation to 2: the M-SSA predictor in exercise 1 is designed to track HP-BIP; in doing so, it also 
#     tracks BIP `somehow`; but the optimization criterion does not address BIP explicitly; the design 
#     proposed here addresses MSE performances when targeting future BIP 

# Overview: 
# -The M-SSA predictor (the matrix predictor_mssa_mat) is constructed from components (contained in array mssa_array)
#   -We here briefly review, illustrate and confirm the construction principle: see exercise 3.1. 
# -Subsequently, we suggest that predictor components support additional information that can be exploited:
#   -for interpretability, see exercise 3.2
#   -and to address BIP-MSE forecast performances, see exercise 3.3

# To start, let us initialize some important settings (based on exercise 1 above)
#   -It is assumed that exercise 1 has been run beforehand
lambda_HP<-160
L<-31
date_to_fit<-date_to_fit
p<-1
q<-0
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
h_vec<-0:6
f_excess<-rep(4,length(select_vec_multi))

# Run the wrapper  
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)

target_shifted_mat=mssa_indicator_obj$target_shifted_mat
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat
mssa_array<-mssa_indicator_obj$mssa_array

#-----------
# 3.1 What are BIP predictor (sub-)components?
# -The M-SSA BIP predictor was introduced in tutorial 7.2, exercise 3
#   -The M-SSA BIP predictor is obtained as the equally-weighted aggregate of standardized M-SSA outputs of all indicators (BIP, ip, ifo, ESI, spread)
#   -For illustration, we here briefly replicate the predictor, as based on its (equally-weighted) components. 
#   -For this purpose, we select the M-SSA nowcast
j_now<-1
# This is the forecast horizon (nowcast)
h_vec[j_now]
# For forecast horizon h_vec[j_now], the sub-series of the M-SSA predictor are:  
tail(t(mssa_array[,,j_now]))
# These sub-series correspond to the outputs of the multivariate M-SSA optimized for horizon h_vec[j_now]
#   -For each series of the multivariate design, the target is the two-sided HP applied to this series and 
#     shifted forward by the forecast horizon (plus the publication lag in case of BIP)
#   -For each of these targets, the explanatory variables are all series of the multivariate design (BIP, ip, ifo, ESI and spread) 
# We can check that the M-SSA predictor is the cross-sectional mean of these standardized sub-series: 
agg_std_comp<-apply(scale(t(mssa_array[,,j_now])),1,mean)
# Plot the above aggregate and the M-SSA predictor
mplot<-cbind(agg_std_comp, predictor_mssa_mat[,j_now])
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c("Cross-sectional mean of standardized predictor components","M-SSA predictor")
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-"Replication of M-SSA predictor, based on its components"
par(mfrow=c(1,1))
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=2,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()
# Both series are identical (overlap)
# Alternative check: the maximal error/deviation should be `small' (zero up to numerical precision)
max(abs(apply(scale(t(mssa_array[,,j_now])),1,mean)-predictor_mssa_mat[,j_now]),na.rm=T)

# Remarks:
# -Equal-weighting of the M-SSA components, as done above, indicates that we assume each M-SSA series to be equally
#   important for tracking dynamic changes of the BIP growth-rate by the resulting M-SSA predictor. This 
#   `naive' assumption might be questioned. But the rule (equal-weighting) is robust and simple.
# -Instead, we could think about a more sophisticated weighting scheme: for example, by regressing the components 
#     on forward-shifted BIP. 
#   -This way, we'd explicitly emphasize BIP-MSE forecast performances by the resulting (new) M-SSA `components` predictor
# -The corresponding `component predictor' will be derived and analyzed in exercise 3.3 below.
# -But first we consider an alternative usage of the components, namely interpretability (of the M-SSA predictor)  
#---------------
# 3.2 We now exploit the M-SSA components in view of a better interpretation (explanation/understanding) of the M-SSA predictor.
#   -We can examine which sub-series is (are) more/less likely to trigger a dynamic change of the M-SSA predictor
#   -For illustration, we select the M-SSA nowcast

# Plot M-SSA nowcast and components
par(mfrow=c(1,1))
# Scale the data 
mplot<-scale(cbind(predictor_mssa_mat[,j_now],scale(t(mssa_array[,,j_now]))))
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c(paste("M-SSA predictor optimized for h=",h_vec[j_now],sep=""),
                   paste("Subseries ",select_vec_multi,sep=""))
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-c(paste("M-SSA predictor for h=",h_vec[j_now]," (solid blue) and sub-series (dashed lines)",sep=""),"In-sample span up to black vertical (dashed) line")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=2,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()
# Discussion:
# -All sub-series date the trough of the growth rate of the German economy in late 2023 
# -Currently (Jan-2025), the strongest positive dynamics are supported by the (leading) spread sub-series (violet dashed line)
# Notes:
# -The trough (minimum) of the grow-rate in the previous figure anticipates the effective trough of BIP (in levels) 
#     by up to several quarters
# -The timing of the BIP-trough (in levels) is sandwiched between the trough of the growth-rate and the next 
#     zero-crossing (of the growth-rate in the above graph). 
#   -Recall, in this context, that the zero-line in the above plot corresponds to (long-term) average growth 
#     (crossing `average growth' signifies positive growth). 
# -Given that the nowcast just passed the zero-line in the above plot, we infer that the 
#     trough of BIP might be behind, based on Jan-2025 data.
# -However, not all components (sub-series) would support this claim
#   -The strongest up-turn signal is supported by the (leading) spread (which has been subjected to critic in is function as a leading indicator)
#   -ifo and ESI are barely above the zero-line 
#   -ip and BIP are `waiting' for further evidence and confirmation. 
# -We infer that we can gauge the M-SSA predictor by looking at its components; which component(s) trigger(s) a change in dynamics (explainability)? 
# -Faint/fragile signals are sensitive to announced and/or unexpected disorders (tariffs, geopolitical contentions)
#   which are not yet `priced-in' (as of Jan-2025).

#-------------------------------------
# Exercise 3.3 Addressing BIP-MSE performances
# -As stated above, the M-SSA predictor emphasizes dynamic changes (recessions/expansions); MSE performances are deemed less relevant
#   -In particular, the predictor is standardized: neither its level nor its scale are calibrated to BIP
# -In order to track BIP more explicitly, we may rely on the predictor (sub-)components in the previous figure
# -For this purpose, we regress the components on forward-shifted BIP (instead of applying an equal-weighting scheme)
# -This is the same proceeding as for the direct forecasts, as illustrated in exercise 1.2.1, except that we rely  
#   on the M-SSA components as regressors (instead of the original indicators)

# 3.3.1 Selection
# -We can select the M-SSA components which are deemed relevant for MSE performances when targeting BIP
#   -ESI, ifo and spread M-SSA components are mainly relevant in a dynamic context (for the M-SSA predictor)
#   -In contrast, BIP and ip M-SSA components are natural candidates in a MSE perspective (which is emphasized here).
#   -Note, however, that the original indicators (data) ESI, ifo and spread are important determinants 
#     of the selected (BIP- and ip-) M-SSA components
sel_vec_pred<-select_vec_multi[c(1,2)]
# Selected M-SSA components
sel_vec_pred
# We can select the forward shift of BIP: for illustration we assume a 2 quarters ahead forward-shift 
#   (below we analyze all shifts, from zero to five quarters ahead)
shift<-2
# We can select the forecast horizon of the M-SSA components: we select a 4 quarters ahead horizon 
#   (below, we shall look all all combinations of shift and forecast horizon)
k<-5
# Check: k=5 corresponds to h_vec[k]=4, a one-year ahead horizon
h_vec[k]
# Define the data matrix for the regression
dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,k]))
rownames(dat)<-rownames(x_mat)
colnames(dat)<-c(paste("BIP shifted forward by lag_vec+shift=",shift+lag_vec[1],sep=""),
                    paste("Predictor component ",colnames(t(mssa_array[sel_vec_pred,,k])),": h=",h_vec[k],sep=""))
tail(dat)
# We target BIP shifted forward by shift+publication lag (first column) based on M-SSA components BIP and ip
#   -As stated above, ifo, ESI and spread (original data) are determinants of the two M-SSA components 
# For estimation purposes we can remove all NAs
dat<-na.exclude(dat)

#-----------------
# 3.3.2 Regression
# We now regress forward-shifted BIP (first column) on the components
#   -Specify an arbitrary in-sample span for illustration (below we shall use an expanding window starting in Q1-2007)
i_time<-which(rownames(dat)>2011)[1]
# In-sample span: 
tail(dat[1:i_time,])
# Regression
lm_obj<-lm(dat[1:i_time,1]~dat[1:i_time,2:ncol(dat)])
summary(lm_obj)
# The M-SSA components seem to be significant (HAC-adjustment wouldn't contradict this statement)

# Compute an out-of-sample prediction for time point i_time+shift
oos_pred<-(lm_obj$coef[1]+lm_obj$coef[2:ncol(dat)]%*%dat[i_time+shift,2:ncol(dat)]) 
# Compute the out-of-sample forecast error
oos_error<-dat[i_time+shift,1]-oos_pred
# This is the out-of-sample error that will be observed shift=2 quarters ahead
oos_error

#---------------
# 3.3.3 Better regression: we can improve the weighting of the M-SSA components further.
# -Given that BIP is subject to heteroscedasticity we may apply a GARCH(1,1) to obtain an estimate of its variance
y.garch_11<-garchFit(~garch(1,1),data=dat[1:i_time,1],include.mean=T,trace=F)
summary(y.garch_11)
# sigmat could be retrieved from GARCH-object
sigmat<-y.garch_11@sigma.t
# But this is lagged by one period
# Therefore we recompute the vola based on the estimated GARCH-parameters
eps<-y.garch_11@residuals
d<-y.garch_11@fit$matcoef["omega",1]
alpha<-y.garch_11@fit$matcoef["alpha1",1]
beta<-y.garch_11@fit$matcoef["beta1",1]
sigmat_own<-sigmat
for (i in 2:length(sigmat))#i<-2
  sigmat_own[i]<-sqrt(d+beta*sigmat_own[i-1]^2+alpha*eps[i]^2)
# This is now correct (not lagging anymore)
sigmat<-sigmat_own


# Plot BIP and its vola
par(mfrow=c(1,1))
# Scale the data 
mplot<-cbind(sigmat,dat[1:i_time,1])
rownames(mplot)<-names(dat[1:i_time,1])
colnames(mplot)<-c("GARCH-vola","BIP")
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-"BIP and GARCH(1,1)-vola"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=2,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()

# We can now apply weighted least-squares (WLS) instead of OLS, using the (inverse of the) GARCH-vola for the weights 
weight<-1/sigmat^2
# Apply WLS instead of OLS
lm_obj<-lm(dat[1:i_time,1]~dat[1:i_time,2:ncol(dat)],weight=weight)
summary(lm_obj)
# The regression coefficients are slightly different (when compared to OLS above)

# Compute out-of-sample prediction for time point i+shift: note that the GARCH is irrelevant when computing the predictor
oos_pred_wls<-as.double(lm_obj$coef[1]+lm_obj$coef[2:ncol(dat)]%*%dat[i_time+shift,2:ncol(dat)]) 
# Compute out-of-sample forecast error
oos_error_wls<-dat[i_time+shift,1]-oos_pred_wls
# This is the out-of-sample error we observe in shift=2 quarters later
oos_error_wls
# Compare to out-of-sample error based on OLS: 
oos_error
# Depending on the selected time point, WLS performs better or worse.
# But on average, over many (out-of-sample) time points, WLS tends to outperform slightly OLS (see exercise 3.3.4 below for confirmation). 
#   -Therefore we now apply WLS when deriving weights of M-SSA components
#   -For comparison and benchmarking, we also derive a new direct forecast based on WLS (in contrast to 
#     exercise 1.2.1 above which is based on classic OLS)

# In addition to regressions, we can also compute the simple mean-benchmark 
mean_bench<-mean(dat[1:i_time,1])
# Its out-of-sample forecast error is
oos_error_mean<-dat[i_time+shift,1]-mean_bench
# The rRMSE of the WLS (M-SSA) component predictor referenced against the mean-benchmark when targeting 
#   BIP shifted forward by shift (+publication lag) is:
rRMSE_mSSA_comp_mean<-sqrt(mean(oos_error_wls^2)/mean(oos_error_mean^2))
rRMSE_mSSA_comp_mean
# Depending on the selected time point, the rRMSE is larger or smaller one
# On average, over a longer out-of-span, we expect the more sophisticated predictor(s) to outperform the simple mean benchmark

# We now apply the above proceeding to a longer out-of-sample span and compute average performances
#------------
# 3.3.4 Average performances: apply the above findings to all data points after 2007, including the 
#   entire financial crisis for out-of-sample evaluations

# Start point for out-of-sample evaluation: 2007
in_out_separator<-in_out_separator
# Note that the in-sample span is rather short at the start (due in part to filter initialization)

# Next: use WLS based on GARCH model when regressing explanatory on forward-shifted BIP (setting the Boolean to F amounts to OLS)
use_garch<-T
# The following function applies the above WLS-regression for all time points (expanding window)
#   -For each time point, a GARCH is fitted and the WLS-regression is computed based on the up-dated GARCH-weights
# Note: we can ignore `warnings' which are generated by the GARCH estimation routine
perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift)

# Here we have the out-of-sample forecast errors of the M-SSA components predictor
tail(perf_obj$epsilon_oos)
# We can see that the function replicates the proceeding in 3.3.3 above, i.e., oos_error_wls is obtained as one of the entries of the longer out-of-sample vector
which(perf_obj$epsilon_oos==oos_error_wls)
# We also obtain the out-of-sample forecast errors of the mean benchmark predictor
tail(perf_obj$epsilon_mean_oos)
# One again, the corresponding out-of-sample error (in 3.3.3) is replicated as an entry of the longer vector
which(perf_obj$epsilon_mean_oos==oos_error_mean)
# More importantly, the function computes HAC-adjusted p-values of the regression of the out-of-sample predictor (oos_pred_wls obtained in exercise 3.3.3 above)
#   on forward-shifted BIP
perf_obj$p_value
# The same but without singular Pandemic readings
perf_obj$p_value_without_covid
# We infer that there exists a statistically significant link, out-of-sample, between the new predictor and forward-shifted BIP
#   -The singular pandemic data affects (negatively) the strength of this link
# In addition, we also obtain the out-of-sample MSE of the M-SSA component predictor (the mean of oos_error_wls^2, where oos_error_wls was obtained in exercise 3.3.3 above)
perf_obj$MSE_oos
# The same but without Pandemic: we can s(e)ize the impact of the crisis on the MSE metric!
perf_obj$MSE_oos_without_covid
# The function also computes the out-of-sample MSE of the simple mean benchmark predictor (expanding window): 
#   -we expect the mean-benchmark to be slightly worse (larger MSE) than the 
#     M-SSA components, at least for smaller forward-shifts of BIP (in the long run, the mean is difficult to outperform)
perf_obj$MSE_mean_oos
perf_obj$MSE_mean_oos_without_covid

# We can compute a classic OLS regression and compare with WLS above. For this purpose we set use_garch<-F
use_garch<-F
perf_obj_OLS<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift)

perf_obj_OLS$MSE_oos
# Compare with WLS
perf_obj$MSE_oos
# The same but without Pandemic: we can s(e)ize the impact of the crisis on the MSE metric!
perf_obj_OLS$MSE_oos_without_covid
# Compare with WLS
perf_obj$MSE_oos_without_covid
# As claimed, WLS outperforms OLS out-of-sample, on average

# In the next step, we compute the above performance metrics for all combinations of forward-shift (of BIP) and 
#   forecast horizons (of M-SSA components)

#----------------
# 3.3.5 Compute performances of M-SSA components as predictors for forward-shifted BIP
# -We compute all combinations of forward-shift and forecast horizon (6*7 matrix of performance metrics)

# -Depending on the CPU, computations may last up to several minutes (regressions and GARCH-models are recomputed for each time point and for all combinations of shift and forecast horizon))
#   -A progress bar pops-up in the R-console

# Initialize performance matrices
p_mat_mssa<-p_mat_mssa_components<-p_mat_mssa_components_without_covid<-p_mat_direct<-rRMSE_mSSA_comp_direct<-rRMSE_mSSA_comp_mean<-rRMSE_mSSA_comp_direct_without_covid<-rRMSE_mSSA_comp_mean_without_covid<-rRMSE_mSSA_direct_mean_without_covid<-rRMSE_mSSA_direct_mean<-p_mat_direct_without_covid<-matrix(ncol=length(h_vec),nrow=length(h_vec)-1)
# Use WLS
use_garch<-T
# Set-up progress bar: indicates progress in R-console
pb <- txtProgressBar(min=min(h_vec),max=max(h_vec)-1,style=3)

# The following double loop computes all combinations of forward-shifts (of BIP) and forecast horizons (of M-SSA)
for (shift in 0:5)
{
# Progress bar: see R-console
  setTxtProgressBar(pb, shift)
  for (j in h_vec)
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
  colnames(p_mat_direct_without_covid)<-paste("h=",h_vec,sep="")
rownames(p_mat_mssa_components)<-rownames(p_mat_direct)<-rownames(p_mat_mssa_components_without_covid)<-
  rownames(rRMSE_mSSA_comp_direct)<-rownames(rRMSE_mSSA_comp_mean)<-
  rownames(rRMSE_mSSA_comp_direct_without_covid)<-rownames(rRMSE_mSSA_comp_mean_without_covid)<-
  rownames(rRMSE_mSSA_direct_mean)<-rownames(rRMSE_mSSA_direct_mean_without_covid)<-
  rownames(p_mat_direct_without_covid)<-paste("Shift=",0:5,sep="")

# HAC-adjusted p-values of out-of-sample (M-SSA) components predictor when targeting forward-shifted BIP
#   -Evaluation based on out-of-sample span starting at in_out_separator and ending on Jan-2025
p_mat_mssa_components
# Same but without singular Pandemic
p_mat_mssa_components_without_covid
# The link between the new predictor and future BIP is statistically significant up to multiple quarters ahead 
#   -Designs optimized for larger forecast horizons (h>=4) seem to perform significantly up to one year ahead

# We can compare new component and original M-SSA predictors
# a. New M-SSA components predictor
p_mat_mssa_components
# b. Original M-SSA predictor (based on naive equal-weighting of the components)
p_value_HAC_BIP_oos
# Findings: the new M-SSA components predictors tend to track BIP better than the M-SSA predictors  
#   at larger forward-shifts (3 and 4 quarters ahead) 

# -The above p-values are based on regressions (of predictors on BIP) thereby ignoring `static' level and scale parameters
# -The following rRMSEs account (also) for scale and level of the new predictor
#   -In contrast to exercise 1.2.3, all computations are now effectively out-of-sample
# a. rRMSE of M-SSA components when benchmarked against mean, out-of-sample, all time points from in_out_separator=2007 onwards
rRMSE_mSSA_comp_mean
# b. rRMSE of M-SSA components when benchmarked against direct forecast, out-of-sample, all time points from in_out_separator=2007 onwards
rRMSE_mSSA_comp_direct
# c. rRMSE of direct forecasts when benchmarked against mean benchmark, out-of-sample, all time points from in_out_separator=2007 onwards
#   Note: the columns are identical since for given shift, the direct forecast does not depend on j in the above loop  
rRMSE_mSSA_direct_mean

# Same but without Pandemic
rRMSE_mSSA_comp_mean_without_covid
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_direct_mean_without_covid

# Comments:
# -New M-SSA components predictor addresses future BIP and MSE performances explicitly 
#   -Weights of components rely on WLS regression of components on future BIP (instead of equal-weighting in original M-SSA predictor, exercise 1)
# -New predictor optimized for larger forecast horizons (h>=4) outperforms original M-SSA predictor when 
#     targeting BIP at larger forward shifts (3 and 4 quarters ahead), see p_mat_mssa_components vs. p_value_HAC_BIP_oos
# -Systematic pattern: for larger forward-shifts (from top to bottom), designs optimized for larger 
#     forecasts horizons tend to perform better
# -Singular Pandemic data affects evaluation 
#   -p-values and rRMSEs increase; systematic patterns are cluttered by noise

# -Analysis on data excluding the Pandemic:
#   -New predictor vs. mean: (see rRMSE_mSSA_comp_mean_without_covid)
#     -rRMSEs are below 90% for shifts up to 4 quarters and for M-SSA designs optimized for larger forecast horizons 
#   -New predictor vs. direct forecasts: (see rRMSE_mSSA_comp_direct_without_covid) 
#     -rRMSEs are below 90% for shifts 2<=shift<=4 and for M-SSA designs optimized for larger forecast horizons 
#     -Outperformance of M-SSA is less strong for shifts<=1: direct forecasts are informative at short forecast horizons
#   -Direct forecast vs. mean benchmark: (see rRMSE_mSSA_direct_mean_without_covid)
#     -rRMSEs are below  (or close to) 90% for small shifts (shift<=1)
#     -Confirmation: direct forecasts are informative at short forecast horizons

#---------------
# 3.3.6 Analyze revisions of components predictor

# A. Compare the final predictor (calibrated over full sample) with the out-of-sample sequence of 
#   continuously re-calibrated predictors: ideally, both series overlap
# -Differences illustrate revisions due to re-estimating regression weights at each time point
par(mfrow=c(1,1))
mplot<-cbind(final_components_preditor,oos_components_preditor)
colnames(mplot)<-c("Final predictor","Real-time out-of-sample predictor")
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-"Revisions: final vs. real-time predictor"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=2,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()
# Remarks: 
#   -To the left of the plot, the real-time predictor is volatile because the sample is short
#   -With increasing sample size, the real-time predictor approaches the final estimate (even during Pandemic)
#   -The vertical black line in the plot indicates the start of the evaluation period, determining 
#     out-of-sample MSE and p-value statistics, see exercise 3.3.5 above


# B. Track regression weights over time 

par(mfrow=c(1,1))
mplot<-track_weights

colo<-rainbow(ncol(mplot))
main_title<-"Revisions: regression weights over time"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=2,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()







#######################################################################################
# Exercise 4: More adaptive design
# -Exercise 1 illustrates that HP-BIP can be forecasted several quarters ahead by the M-SSA predictor
# -But predicting BIP is more challenging
#   -For this purpose exercise 3 proposed an analysis of M-SSA components, based on a more sophisticated 
#     weighting scheme (than equal-weighting)
# -However, we cannot exclude (a priori) that the target specification, HP(160) is `too smooth'
#   -Maybe relevant information is being suppressed by the filter...
# -To verify this conjecture, we now analyze a more adaptive design by selecting lambda_HP=16 `small`

# Question: does the more flexible design allow to predict BIP (not HP-BIP) more reliably?

# 4.1 Run more adaptive M-SSA design
lambda_HP<-16
# Notes: 
# -For adaptive designs, a pronounced left-shift might lead to phase-reversal which is undesirable
# -Therefore we use forecast horizons up to 4 quarters and no forecast excess
#   -Phase-reversal would be fine (optimal) if the data were in agreement with the implicit assumptions 
#     underlying the HP filter (which is not the case, see tutorial 2.0)
f_excess_adaptive<-rep(0,length(select_vec_multi))
h_vec_adaptive<-0:4

# Run the M-SSA predictor function
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec_adaptive,f_excess_adaptive,lag_vec,select_vec_multi)

# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA indicators
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat

# 4.2 Compute performances
# For the direct predictor we can specify the macro-indicators in the expanding-window regressions
#   -Note: too complex designs lead to overfitting and thus worse out-of-sample performances
select_direct_indicator<-c("ifo_c","ESI")

perf_obj<-compute_perf_func(x_mat,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,in_out_separator,select_direct_indicator,h_vec_adaptive) 

p_value_HAC_HP_BIP_full=perf_obj$p_value_HAC_HP_BIP_full
t_HAC_HP_BIP_full=perf_obj$t_HAC_HP_BIP_full
cor_mat_HP_BIP_full=perf_obj$cor_mat_HP_BIP_full
p_value_HAC_HP_BIP_oos=perf_obj$p_value_HAC_HP_BIP_oos
t_HAC_HP_BIP_oos=perf_obj$t_HAC_HP_BIP_oos
cor_mat_HP_BIP_oos=perf_obj$cor_mat_HP_BIP_oos
p_value_HAC_BIP_full=perf_obj$p_value_HAC_BIP_full
t_HAC_BIP_full=perf_obj$t_HAC_BIP_full
cor_mat_BIP_full=perf_obj$cor_mat_BIP_full
p_value_HAC_BIP_oos=perf_obj$p_value_HAC_BIP_oos
t_HAC_BIP_oos=perf_obj$t_HAC_BIP_oos
cor_mat_BIP_oos=perf_obj$cor_mat_BIP_oos
rRMSE_MSSA_HP_BIP_direct=perf_obj$rRMSE_MSSA_HP_BIP_direct
rRMSE_MSSA_HP_BIP_mean=perf_obj$rRMSE_MSSA_HP_BIP_mean
rRMSE_MSSA_BIP_direct=perf_obj$rRMSE_MSSA_BIP_direct
rRMSE_MSSA_BIP_mean=perf_obj$rRMSE_MSSA_BIP_mean
target_BIP_mat=perf_obj$target_BIP_mat

# We now examine performances when targeting specifically BIP (we already know that HP-BIP can be predicted)
# 1. Full sample
cor_mat_BIP_full
# 2. Out-of-sample 
cor_mat_BIP_oos
# These numbers are larger than in exercise 1, suggesting that the more adaptive design here is better able to
#   track forward-shifted BIP

# Are the results significant?
p_value_HAC_BIP_full
p_value_HAC_BIP_oos

# We cannot see systematic improvements (smaller p-values) when using a more adaptive target HP(16) for M-SSA

##################################################################################
# Exercise 5
# As a complement to exercise 4 above, we now briefly analyze the rather inflexible classic
#   quarterly HP(1600) design as a target for M-SSA, to back-up our previous discussion 
#   with empirical facts

lambda_HP<-1600
# Given a slower decay (stronger smoothing), we may consider longer filters
L_long<-2*L-1
# Or just keep it fixed (faster numerical optimization if filters are shorter)
L_long<-L
# We also use the larger HTs of exercise 1.5
ht_mssa_vec_long<-ht_mssa_vec_long
# Run the wrapper  
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L_long,date_to_fit,p,q,ht_mssa_vec_long,h_vec,f_excess,lag_vec,select_vec_multi)

target_shifted_mat=mssa_indicator_obj$target_shifted_mat
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat
mssa_array<-mssa_indicator_obj$mssa_array

# Plot 
mplot<-predictor_mssa_mat
colnames(mplot)<-colnames(predictor_mssa_mat)
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(predictor_mssa_mat)))
main_title<-c(paste("Standardized M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep=""),"Vertical line delimites in-sample and out-of-sample spans")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
abline(v=which(rownames(mplot)>date_to_fit)[1]-1,lty=2)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Discussion:
# -The stronger smoothing by HP(1600) emphasizes long-term dynamics which are less relevant in a one-year forecast perspective
#   -See a critic by Phillips and Jin (2021), suggesting that HP(1600) is `too smooth' (insufficiently flexible)
# -As a result, increasing the forecast horizon has only marginal effects on the M-SSA predictor
#   -The left-shift of the M-SSA predictors is less pronounced
# -Stated otherwise: increasing the forecast horizon has only a marginal effect on the `phase' of the filter
#   -the right tail of the HP(1600) corresponds to an AR(2) with a comparatively long periodicity
#   -advancing the filter by a full year has only small/marginal effects on the phase (when compared to HP(160) or HP(16)) 

# Further comments:
# -One can scroll back the plots generated in the plot-panel
# -Doing so would show that the two-sided filter (black-line in previous plots) behaves strangely after the 
#     financial crisis
#   -To a good extent, some of the anomalies are due to the singular Pandemic readings
#   -HP(1600) is more sensitive to the Pandemic than the previous (more adaptive) designs

##########################################################################################################
# Findings overall:

# A. Classic direct predictors:
#   -Classic direct predictors often do not perform better (out-of-sample) than the simple mean benchmark at 
#     forward-shifts of 2 quarters or more
#   -Classic direct predictors tend to be more sensitive (than M-SSA) to singular episodes (Pandemic)

# B. M-SSA
#   -Classic business-cycle designs (lambda_HP=1600) smooth out recessions and hide  
#     dynamics potentially relevant in a short- to mid-term forecast exercise (1-5 quarters ahead)
#   -Fairly adaptive designs (lambda_HP=160) show a (logically and) statistically consistent forecast pattern, 
#       suggesting that M-SSA outperforms both the mean and the direct forecasts out-of-sample when targeting HP-BIP
#     -This result suggests that M-SSA is informative about forward-shifted BIP too, although corresponding 
#       performance statistics are less conclusive, see exercise 1 (cluttered by noise)
#   -More adaptive (lambda_HP=16) or less adaptive designs (lambda_HP=1600) do not track forward-shifted BIP 
#       systematically better, see exercises 4 and 5
#   -Sub-components of the M-SSA predictor are potentially useful for interpretation purposes (see exercise 3.2) 
#     and for addressing MSE performances explicitly, see exercise 3.3
#     -Weighting M-SSA components `optimally' (instead of equal-weighting in M-SSA predictor) leads to 
#       performance gains when targeting BIP multiple quarters ahead
#     -One can observe smaller p-values (stronger link between M-SSA component predictor and future BIP) as well as 
#       better MSE performances (smaller rRMSE) out-of-sample 
#     -Gains seem to take a hit for horizons larger than a year.

# C. Statistical significance
#   -HAC-adjustments (of test-statistics) seem unable to account fully for the observed data-idiosyncrasies
#   -However, exercise 2 suggests that biases are relatively small
#     -p-values smaller than one percent are rare in the case of simulated white noise
#   -The original bias has been reduced further by a simple trick
#     -Compute standard errors (of regressors) based on OLS (classic) and HAC (R-package sandwich)
#     -Compute the max of both standard errors
#     -Derive t-statistics based on the max (conservative setting)
#     -See our R-code: we use sd_max when computing the t-statistic in the following function
head(HAC_ajusted_p_value_func,20)

# Final notes on the publication lag and data revisions
#   -All results relate to forward-shifts augmented by the publication lag
#   -According to feedback, our setting for the publication lag, i.e., lag_vec[1]=2, is too large (reflecting some prudence)
#     -The official/effective publication lag of BIP is one quarter
#     -But BIP is revised and we here ignore revisions
#     -On the other hand, the weight M-SSA assigns to BIP is rather weak (other, timely indicators are more important); 
#       moreover, data revisions affect mainly the so-called `direct forecasts'; 
#       finally, smoothing by HP mitigates the effect of data revisions
# In summary: we expect that performances at an indicated forward-shift of k quarters (in all the above evaluations) 
#   are likely to be representative of performances at effectively k+1 quarters ahead. 

