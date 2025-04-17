# Tutorial 7.3: we propose various M-SSA BIP (GDP Germany) predictor designs
# -The concept of M-SSA predictors for BIP was introduced in tutorial 7.2
# -We wrapped this proceeding into a single function to be able to analyze various M-SSA BIP predictor designs (hyperparameters)
# -In exercise 1, we propose a `fairly adaptive' predictor. A `more adaptive' one is analyzed in exercise 3 and a `more inflexible' one in exercise 4
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

# The tutorial is structured into four exercises
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
# Exercise 3: analyze a more adaptive M-SSA design based on targeting HP(16) by M-SSA
# Finally, exercise 4 briefly analyzes the classic HP(1600) as a target for M-SSA

# Note: the M-SSA predictor proposed in this tutorial is designed to track HP-BIP (two-sided HP applied to BIP). 
#   -It is mainly intended for tracking dynamic changes in the trend growth-rate (recessions/crises/expansions).  
#   -The design of the proposed M-SSA predictor does not emphasize BIP explicitly
#   -Tutorial 7.4 presents a slightly more refined `M-SSA components predictor' which exploits M-SSA 
#     components more effectively in order to track explicitly BIP up to multiple quarters ahead.
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
# -In contrast, we expect HP-BIP in exercise 1 to be informative about future BIP (log-differences) because the 
#     latter is not WN:
#   -The occurrence of recessions (business-cycle) contradicts the WN assumption
#   -The acf and the VAR-model contradict the WN assumption 
#   -The systematic structure in the performance matrices (top-down/left-right) suggests predictability
#   -The recent negative BIP-readings are not `random` events: they are due in part to exogenous shock-waves and 
#     also to endogenous decisions whose underlying(s) did not realize as wished for 
#   -These deviations of BIP from WN concern lower frequency components of the spectral decomposition, as 
#     emphasized by HP-BIP


#######################################################################################
# Exercise 3: More adaptive design
# -Exercise 1 above illustrates that HP-BIP can be forecasted several quarters ahead by the M-SSA predictor
# -But predicting BIP is more challenging
# -However, we cannot exclude (a priori) that the target specification, HP(160) is `too smooth'
#   -Maybe relevant information is being suppressed by the filter.
# -To verify this conjecture, we now analyze a more adaptive design by selecting lambda_HP=16 `small`

# Question: does the more flexible design allow to predict BIP (not HP-BIP) more reliably?

# 3.1 Run more adaptive M-SSA design
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

# 3.2 Compute performances
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
# A. Full sample
cor_mat_BIP_full
# B. Out-of-sample 
cor_mat_BIP_oos
# These numbers are larger than in exercise 1, suggesting that the more adaptive design here is better able to
#   track forward-shifted BIP

# Are the results significant?
p_value_HAC_BIP_full
p_value_HAC_BIP_oos

# We cannot see systematic improvements (smaller p-values) when using a more adaptive target HP(16) for M-SSA

##################################################################################
# Exercise 4
# As a complement to exercise 3 above, we now briefly analyze the rather inflexible classic
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
#   -The proposed M-SSA predictor relies on a simple equally-weighted aggregation of (standardized) 
#       M-SSA components
#     -A better optimal weighting of these components is proposed in tutorial 7.4: M-SSA components predictor
#     -The new M-SSA components predictor can address MSE forecast performances when targeting BIP, explicitly
#     -Moreover, the components can be used for interpretation purposes, see also tutorial 7.4 


# C. Statistical significance
#   -HAC-adjustments (of test-statistics) seem unable to account fully for the observed data-idiosyncrasies
#   -However, exercise 2 suggests that biases are relatively small
#     -p-values smaller than one percent are rare in the case of simulated white noise
#   -The original (finite sample) bias has been reduced further by a simple trick
#     -Compute standard errors (of regressors) based on OLS (classic) and HAC (R-package sandwich)
#     -Compute the max of both standard errors
#     -Derive t-statistics based on the max standard error (conservative setting)
#     -See our R-code in the following function: we use sd_max when computing the t-statistic in the following function
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

