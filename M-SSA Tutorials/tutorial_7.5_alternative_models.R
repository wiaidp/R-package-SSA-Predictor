# ============================================================
# Tutorial 7.5
# ============================================================

# Nomenclature:
#   BIP = Brutto-Inlands-Produkt (German Gross Domestic Product, GDP)

# Overview:
#   This tutorial builds on Tutorial 7.4 with one key modification:
#   instead of the standard VAR(1) model used previously, a richer Bayesian
#   Vector Autoregression of order 3 (BVAR(3)) is used to capture
#   the dependence structure in the data exploited by M-SSA.
#
# Motivation for switching to BVAR(3):
#   - Higher lag order (3 vs. 1) captures richer temporal dynamics in the data.
#   - The BVAR places greater weight on the Economic
#     Sentiment Indicator (ESI), improving its role as a leading indicator 
#     (leading BIP after accounting for the latter's publication lag).
#
# Key improvements of M-SSA based on BVAR(3) over VAR(1) (Tutorial 7.4):
#   - Lower p-values: indicating a statistically stronger predictive link
#     to future BIP growth.
#   - Smaller relative Root Mean Squared Errors (rRMSEs): confirming
#     greater outperformance relative to two benchmark predictors,
#     namely the mean forecast and the direct forecast.
#   - More consistent patterns in the shift/h matrices: for a given
#     forward shift of BIP, forecast designs optimized at horizon h = shift
#     (i.e., the diagonal entries) are close to globally optimal,
#     suggesting well-calibrated and coherent forecast horizons.
# ============================================================
# References
# ============================================================
#
# Wildi, M. (2026b). The Accuracy-Smoothness Dilemma in Prediction:
#   A Novel Multivariate M-SSA Forecast Approach.
#   Journal of Time Series Analysis, http://doi.org/10.1111/jtsa.70058 
#   arXiv: https://doi.org/10.48550/arXiv.2602.13722
#
# Heinisch, K., Van Norden, S., and Wildi, M. (2026).
#   Smooth and Persistent Forecasts of German GDP:
#   Balancing Accuracy and Stability.
#   IWH Discussion Papers, 1/2026.
#   Halle Institute for Economic Research.
#   https://doi.org/10.18717/dp99kr-7336
#
# ============================================================

# Clear the workspace to ensure a clean environment
rm(list=ls())


# Load required R libraries
# Standard filter package (HP filter and related methods)
library(mFilter)
# Multivariate time series: VARMA models for macroeconomic indicators
# (used here for simulation purposes only)
library(MTS)
# HAC-consistent standard error estimation
# (robust to both autocorrelation and heteroscedasticity)
library(sandwich)
# Extended time series objects with date/time indexing
library(xts)
# Diebold-Mariano test for equal predictive accuracy
library(multDM)
# GARCH models for volatility modelling and improving regression estimates
library(fGarch)
# Ridge regression (MASS) and LASSO / elastic net regularisation (glmnet)
library(MASS)
library(glmnet)


# Load M-SSA source files
# Core M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Signal extraction functions from the JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))
# Utility functions supporting M-SSA workflows (see tutorials for details)
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))


# ============================================================
# Load data and select relevant indicators
# See Tutorials 7.2 and 7.3 for data description and variable selection background
load(file=paste(getwd(),"\\Data\\macro",sep=""))

# Publication lag vector: BIP is published with a one-quarter delay;
# all other indicators are assumed to be available contemporaneously
lag_vec<-c(1,rep(0,ncol(data)-1))

# Plot all indicators to verify the data and visualise publication lags
# Note: the real-time GDP/BIP series (red) lags the target (black) by lag_vec[1] quarters
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
# Note: BIP (red) and ip (orange) are visibly right-shifted relative to other series,
#   reflecting their respective publication lags:
#   - BIP: one quarter
#   - ip:  two months (see data_monthly for the two-month lag of ip)


# Select macroeconomic indicators for M-SSA
# Variables: BIP (GDP), industrial production (ip), ifo climate index (ifo_c),
#            Economic Sentiment Indicator (ESI), and 10y-3m yield spread (spr_10y_3m)
select_vec_multi<-c("BIP","ip","ifo_c","ESI","spr_10y_3m")
x_mat<-data[,select_vec_multi]
rownames(x_mat)<-rownames(data)
# Number of indicators (columns)
n<-dim(x_mat)[2]
# Number of time periods (rows)
len<-dim(x_mat)[1]
  
  
# ====================================================================================
# Exercise 1: Perform M-SSA multi-Quarters Ahead BIP Forecasts Based on the BVAR
# ====================================================================================
# Exercise 1.1: Specify Models for M-SSA (and M-MSE)
# ====================================================================================


# Select the type of VAR model to use
# Option "VAR":  standard VAR(1), as used in Tutorial 7.4
VAR_type="VAR"
# Option "BVAR": Bayesian VAR, which supports higher lag orders by shrinking
#   coefficients toward a simple benchmark (Minnesota or Litterman prior).
#   This regularisation enables richer lead/lag dynamics without overfitting.
VAR_type="BVAR"

if (VAR_type=="BVAR")
{
  # BVAR(3) settings: lag order p=3, shrinkage controlled by lambda_BVAR
  # A small lambda_BVAR applies strong shrinkage toward the Random Walk benchmark
  lambda_BVAR <- 0.001
  p<-3
  q<-0
}
if (VAR_type=="VAR")
{
  # Standard VAR(1) settings: lag order p=1, no shrinkage 
  # This corresponds to tutorial 7.4 
  p<-1
  q<-0
}

# Training sample end date: pre-pandemic data only, to avoid COVID-19 distortions
# Note: Tutorial 7.4 used pre-financial-crisis data (up to 2008), but that sample
#       is too short to reliably estimate a BVAR(3); extending to 2020 resolves this
date_to_fit<-"2020"


#----------
# Filter settings

# HP smoothing parameter: the single most important hyperparameter governing
# the target trend; see Tutorial 7.1 for a detailed discussion
#   - lambda=1600 removes too much signal when targeting BIP one year ahead
#   - lambda=16   retains too much noise
#   - lambda=160  provides a good balance for the one-year-ahead forecast exercise
lambda_HP<-160

# Filter length: approximately 8 years (31 quarters), chosen so that HP filter
# weights decay sufficiently before the truncation point;
# must be an odd number (see Tutorial 7.1)
L<-31

# Holding Time (HT) constraints for each indicator,
# calibrated to be roughly 50% larger than the classic M-MSE predictor
# (same values as in Tutorial 7.2): M-SSA generates 33% less crossings mean crossings than
# the classic M-MSE asymptotically, if the model is not misspecified (see tutorial 7.1)
ht_mssa_vec<-c(6.380160, 6.738270, 7.232453, 7.225927, 7.033768)
names(ht_mssa_vec)<-colnames(x_mat)

# Forecast horizons: M-SSA filter is optimised separately for each h in h_vec
h_vec<-0:6

# Forecast excess parameters: VAR-type models tend to underfit sharp recession dips
# due to inherent model misspecification. Allowing a forecast excess larger than
# the nominal horizon h partially compensates for this by enabling better tracking
# of cyclical swings in the data; see Tutorial 7.2, Exercise 2 for full background
f_excess<-c(5,rep(4,length(select_vec_multi)-1))

# Compute M-SSA and M-MSE predictors for all forecast horizons in h_vec
# See Tutorial 7.2 for a full description of the wrapper function
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi,VAR_type,lambda_BVAR)

# Extract M-SSA output array:
# Dimensions: [indicator, time, forecast horizon h]
# For each indicator, columns correspond to optimal predictors at h=0 (nowcast) through h=6
# M-SSA is smoother than M-MSE because it additionally imposes the HT constraint,
# which controls the mean duration between consecutive zero-crossings
mssa_array<-mssa_indicator_obj$mssa_array
tail(mssa_array["BIP",,])
tail(mssa_array["ESI",,])

# Extract M-MSE output array:
# Same structure as mssa_array but without the HT constraint,
# i.e., classic multivariate minimum mean-square error signal extraction
mmse_array<-mssa_indicator_obj$mmse_array
tail(mmse_array["BIP",,])
tail(mmse_array["ESI",,])


# ====================================================================================
# Exercise 1.2: Out-of-Sample Forecast Performances 
# ====================================================================================

# We compute out-of-sample forecast performances of the M-SSA component predictor
#   for all combinations of forward shift (of BIP) and forecast horizon (of M-SSA),
#   yielding a (length(shift_vec) x length(h_vec)) matrix of performance metrics;
#   see Tutorial 7.4 for the analogous VAR(1)-based results.
# For reference, we also compute performances for the classic direct forecast, which regresses
#   original (unfiltered) indicators on forward-shifted BIP

# Select M-SSA components to use as predictors for BIP
#   - The BIP M-SSA component is the natural candidate: it directly targets BIP dynamics.
#     All indicators enter the M-SSA computation, so M-SSA-BIP implicitly aggregates
#     the information content of the full multivariate system into a single series.
#   - Using M-SSA-BIP as the sole explanatory variable keeps the regression parsimonious:
#     the regression serves to calibrate the level and scale of M-SSA to match actual BIP.
#   - Single-variable regressions also produce smaller revisions when
#     regression equations are re-estimated each quarter as new data arrive.
#
# Findings from Tutorial 7.4 (Exercise 6) motivating this choice:
#   - Among all M-SSA components, only M-SSA-BIP and M-SSA-ip are statistically
#     significant predictors of future BIP.
#   - However, the estimated coefficient on M-SSA-ip is negative, which is
#     counterintuitive and difficult to interpret economically.
#   - Despite its negative sign, including M-SSA-ip produces a faster-responding
#     predictor that better tracks forward-shifted BIP during strong economic swings.
#
# Summary of the trade-off:
#   - M-SSA-BIP alone:          interpretable, but responds more slowly to turning points
#   - M-SSA-BIP + M-SSA-ip:     faster and more responsive, but less intuitive
#                                due to the negative coefficient on ip

# Selection: M-SSA-BIP and M-SSA-ip
sel_vec_mssa_comp<-c("BIP","ip")
# Selection: M-SSA-BIP only
sel_vec_mssa_comp<-c("BIP")

# Select indicators for the direct forecast benchmark
# Note: using all indicators leads to overfitting and worse out-of-sample performance;
#       ifo_c and ESI provide the best out-of-sample design
sel_vec_direct_forecast<-c("ifo_c","ESI")

# Sample alignment flag:
#   M-SSA loses the first L observations due to filter initialisation,
#   so its effective sample is shorter than that of the direct forecast or mean benchmarks.
#   If align_sample=TRUE, the first L observations are removed from all benchmarks
#   to ensure comparisons are made on identical samples (avoiding spurious differences)
align_sample<-T

# Regression type for calibrating predictors to the forecast target (the same as in tutorial 7.4)
reg_type<-"OLS"
reg_type<-"LASSO"
reg_type<-"Ridge"

# Penalty weight for regularised regression (LASSO or Ridge)
lambda_reg<-10

# GARCH weighting flag: if TRUE, uses GARCH-based weights to account for heteroscedasticity.
# Set to FALSE here because GARCH downweights recession episodes, which are
# precisely the periods of greatest forecasting interest
use_garch<-F

# Forward-shift values: BIP is shifted forward by (shift + publication lag) quarters,
# so that M-SSA at time t predicts BIP at time t + shift + lag_vec[1] (including the publication lag)
shift_vec<-0:5

# Start of the out-of-sample evaluation period for the regression: set before the 2008 financial crisis
#   so that the financial crisis episode falls within the out-of-sample window
# Notes:
#   - The BVAR(3) is estimated on data up to 2020 (date_to_fit): a longer sample
#     is required because the pre-2008 sample alone is too short to reliably fit a BVAR(3).
#   - However, the regression of future BIP on M-SSA uses the same in-sample span
#     as in Tutorial 7.4 (in-sample up to 2007), ensuring that out-of-sample
#     performance comparisons across the two tutorials remain meaningful.
#   - In other words, the BVAR estimation sample and the regression estimation sample
#     are decoupled here.
#   - This decoupling is admittedly a limitation, but evidence from Tutorial 7.4 suggests
#     that M-SSA forecasts are not heavily sensitive to the length of the model estimation
#     sample, since both the VAR(1) and BVAR(3) specifications are parsimonious and
#     therefore relatively stable across different sample sizes.

in_out_separator<-"2007"

# Initialise performance matrices (rows = forward shifts, columns = forecast horizons)
MSE_oos_mssa_comp_without_covid_mat<-MSE_oos_mssa_comp_mat<-
  p_mat_mssa<-p_mat_mssa_components<-p_mat_mssa_components_without_covid<-
  p_mat_direct<-rRMSE_mSSA_comp_direct<-rRMSE_mSSA_comp_mean<-
  rRMSE_mSSA_comp_direct_without_covid<-rRMSE_mSSA_comp_mean_without_covid<-
  rRMSE_mSSA_direct_mean_without_covid<-rRMSE_mSSA_direct_mean<-
  p_mat_direct_without_covid<-matrix(nrow=length(shift_vec),ncol=length(h_vec))

# Initialise arrays to store final (in-sample) and real-time (out-of-sample) predictors
# Dimensions: [forward shift, forecast horizon, time]
# These will be used to analyse forecast revisions in Exercise 2.1
final_components_preditor_array<-oos_components_preditor_array<-array(dim=c(length(shift_vec),length(h_vec),nrow(x_mat)))
dimnames(final_components_preditor_array)<-dimnames(oos_components_preditor_array)<-list(paste("shift=",shift_vec,sep=""),
                                                                                         paste("h=",h_vec,sep=""),
                                                                                         rownames(x_mat))
dim(p_mat_mssa_components)

# Initialise array to track regression weights over time
# Dimensions: [forward shift, forecast horizon, time, regression coefficients]
# Used to distinguish systematic from noisy revisions in Exercise 2.2
track_weights_array<-array(dim=c(length(shift_vec),length(h_vec),nrow(x_mat),length(sel_vec_mssa_comp)+1))
dimnames(track_weights_array)<-list(paste("shift=",shift_vec,sep=""),
                                    paste("h=",h_vec,sep=""),
                                    rownames(x_mat),
                                    c("Intercept",sel_vec_mssa_comp))

# Initialise progress bar to monitor loop execution in the R console
pb <- txtProgressBar(min=min(h_vec),max=max(h_vec)-1,style=3)

# Double loop over all combinations of forward shifts and forecast horizons
for (shift in shift_vec)  # shift<-1
{
  # Update progress bar
  setTxtProgressBar(pb, shift)
  
  for (j in h_vec)  # j<-1
  {
    # Horizon index: h=j corresponds to the (j+1)-th column of the M-SSA array
    k<-j+1
    
    #------------------------------------------------------------
    # A. M-SSA component predictor
    #------------------------------------------------------------
    # Construct the regression data matrix:
    #   - First column (target):    BIP shifted forward by (shift + publication lag),
    #                               so that the predictor at time t targets BIP at t + shift + lag_vec[1]
    #   - Remaining columns (predictors): M-SSA components for the selected indicators
    #
    # Rationale: M-SSA/M-MSE is optimised to track the HP-filtered BIP trend (two-sided HP),
    #   which is the denoised target. Regressing future BIP on the M-SSA output
    #   calibrates its level and scale to match actual BIP (see tutorial 7.4 for background)
    if (length(sel_vec_mssa_comp)>1)
    {
# R handles the case with multiple (M-SSA) predictors differently: have to transpose the matrix t(mssa_array)     
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_mssa_comp,,k]))
    } else
    {
# Univariate case: single predictor      
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mssa_array[sel_vec_mssa_comp,,k]))
    }
    rownames(dat)<-rownames(x_mat)
    colnames(dat)<-c(colnames(x_mat)[1],sel_vec_mssa_comp)
    nrow(na.exclude(dat))
    
# Fit regression and compute out-of-sample MSEs and HAC-adjusted p-values
# p-values measure the statistical significance of M-SSA's ability
# to predict future BIP movements out-of-sample
# HAC adjustement accounts for serial correlation and heteroscedasticity of regression residuals
    perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec,align_sample,reg_type,lambda_reg)
    
# Store p-values: full out-of-sample sample and excluding the COVID-19 pandemic period
    p_mat_mssa_components[shift+1,k]<-perf_obj$p_value
    p_mat_mssa_components_without_covid[shift+1,k]<-perf_obj$p_value_without_covid
    
# Store out-of-sample MSEs: full out-of-sample sample and excluding the COVID-19 pandemic period
    MSE_oos_mssa_comp_mat[shift+1,k]<-MSE_oos_mssa_comp<-perf_obj$MSE_oos
    MSE_oos_mssa_comp_without_covid_mat[shift+1,k]<-MSE_oos_mssa_comp_without_covid<-perf_obj$MSE_oos_without_covid
    
# Store mean benchmark MSEs: full out-of-sample sample and excluding the COVID-19 pandemic period
    MSE_mean_oos<-perf_obj$MSE_mean_oos
    MSE_mean_oos_without_covid<-perf_obj$MSE_mean_oos_without_covid
    
# Retrieve final full-sample predictor (fitted on the full sample) and
# real-time out-of-sample predictor (re-estimated at each time step).
# Both are stored for revision analysis in Exercise 2.1
    final_in_sample_preditor<-perf_obj$final_in_sample_preditor
    final_components_preditor_array[shift+1,j+1,(nrow(x_mat)-length(final_in_sample_preditor)+1):nrow(x_mat)]<-final_in_sample_preditor
    cal_oos_pred<-perf_obj$cal_oos_pred
    oos_components_preditor_array[shift+1,j+1,(nrow(x_mat)-length(cal_oos_pred)+1):nrow(x_mat)]<-cal_oos_pred
    
# Store time-varying regression weights for revision analysis in Exercise 2.2
# Note: track_weights_array is overwritten on each iteration; only the last
#       iteration (maximum shift and maximum horizon) is retained after the loop
    track_weights<-perf_obj$track_weights
    track_weights_array[shift+1,j+1,(nrow(x_mat)-nrow(track_weights)+1):nrow(x_mat),]<-track_weights
    
#------------------------------------------------------------
# B. Direct forecast benchmark
#------------------------------------------------------------
# The setup mirrors Part A, with one key difference:
#   the predictors are the raw (unfiltered) indicators from x_mat,
#   rather than M-SSA-filtered components from mssa_array.
# Note: the direct forecast data matrix does not depend on j (the forecast horizon),
#       unlike the M-SSA component matrix above
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),x_mat[,sel_vec_direct_forecast])
    rownames(dat)<-rownames(x_mat)
    
    if (align_sample)
    {
# Aligned sample: drop the first L observations to match the M-SSA sample length
      dat<-dat[L:nrow(dat),]
      nrow(na.exclude(dat))
    } else
    {
# Full sample: retain all observations including the filter initialisation period
      dat<-dat[1:nrow(dat),]
    }
    
    perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec,align_sample,reg_type,lambda_reg)
    
# Store direct forecast out-of-sample p-values and MSEs: full out-of-sample sample and excluding COVID-19
    p_mat_direct[shift+1,k]<-perf_obj$p_value
    p_mat_direct_without_covid[shift+1,k]<-perf_obj$p_value_without_covid
    MSE_oos_direct<-perf_obj$MSE_oos
    MSE_oos_direct_without_covid<-perf_obj$MSE_oos_without_covid
    
#------------------------------------------------------------
# Compute relative Root Mean Squared Errors (rRMSEs)
# Values below 1 indicate that the numerator model outperforms the denominator model
#------------------------------------------------------------
# a. M-SSA components vs. direct forecast
    rRMSE_mSSA_comp_direct[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_oos_direct)
# b. M-SSA components vs. mean benchmark
    rRMSE_mSSA_comp_mean[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_mean_oos)
# c. Direct forecast vs. mean benchmark
    rRMSE_mSSA_direct_mean[shift+1,k]<-sqrt(MSE_oos_direct/MSE_mean_oos)
# d-f. Same comparisons as a-c, excluding the COVID-19 pandemic period
    rRMSE_mSSA_comp_direct_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_oos_direct_without_covid)
    rRMSE_mSSA_comp_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_mean_oos_without_covid)
    rRMSE_mSSA_direct_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_direct_without_covid/MSE_mean_oos_without_covid)
  }
}

# Close the progress bar
close(pb)

# Assign row and column names to all performance matrices
# Columns correspond to forecast horizons; rows correspond to forward shifts
colnames(p_mat_mssa_components)<-colnames(p_mat_direct)<-
  colnames(p_mat_mssa_components_without_covid)<-
  colnames(rRMSE_mSSA_comp_direct)<-colnames(rRMSE_mSSA_comp_mean)<-
  colnames(rRMSE_mSSA_comp_direct_without_covid)<-colnames(rRMSE_mSSA_comp_mean_without_covid)<-
  colnames(rRMSE_mSSA_direct_mean)<-colnames(rRMSE_mSSA_direct_mean_without_covid)<-
  colnames(p_mat_direct_without_covid)<-colnames(MSE_oos_mssa_comp_mat)<-
  colnames(MSE_oos_mssa_comp_without_covid_mat)<-paste("h=",h_vec,sep="")

rownames(p_mat_mssa_components)<-rownames(p_mat_direct)<-
  rownames(p_mat_mssa_components_without_covid)<-
  rownames(rRMSE_mSSA_comp_direct)<-rownames(rRMSE_mSSA_comp_mean)<-
  rownames(rRMSE_mSSA_comp_direct_without_covid)<-rownames(rRMSE_mSSA_comp_mean_without_covid)<-
  rownames(rRMSE_mSSA_direct_mean)<-rownames(rRMSE_mSSA_direct_mean_without_covid)<-
  rownames(p_mat_direct_without_covid)<-rownames(MSE_oos_mssa_comp_mat)<-
  rownames(MSE_oos_mssa_comp_without_covid_mat)<-paste("Shift=",shift_vec,sep="")

#-----------------------
# HAC-adjusted p-values for the M-SSA component predictor targeting forward-shifted BIP.
# Evaluation covers the out-of-sample period from 'in_out_separator' through January 2025.
p_mat_mssa_components

# Same evaluation excluding the singular COVID-19 pandemic observations.
p_mat_mssa_components_without_covid
# Interpretation: the M-SSA predictor maintains statistically significant predictive
# content for BIP growth across multiple quarters ahead.


# Note on p-values (above) vs. rRMSEs (below):
#   - The p-values reported above are derived from out-of-sample M-SSA
#     outputs, but rely on full-sample regressions for the 'static' level
#     and scale adjustments applied to the M-SSA-BIP predictor. They
#     therefore do not constitute fully out-of-sample evaluations.
#   - The rRMSEs computed below provide a stricter assessment of 'true'
#     forecast accuracy: both the M-SSA extraction and the level and scale
#     alignment are performed in a genuinely out-of-sample fashion, ensuring
#     that no future information leaks into the evaluation.


# a. rRMSE: M-SSA component predictor benchmarked against the naive mean forecast
rRMSE_mSSA_comp_mean

# b. rRMSE: M-SSA component predictor benchmarked against the direct forecast
rRMSE_mSSA_comp_direct

# c. rRMSE: Direct forecast benchmarked against the naive mean forecast
#    Note: columns are identical within each row because the direct forecast's
#    explanatory variables do not depend on the M-SSA component index 'j' —
#    the same model is fitted for all j at a given shift value.
rRMSE_mSSA_direct_mean

# Same three rRMSE comparisons, excluding the COVID-19 pandemic period:
rRMSE_mSSA_comp_mean_without_covid
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_direct_mean_without_covid

# =================================================================
# Summary of Key Findings:
#
# Preliminary Notes:
#   a. The M-SSA filter relies on a BVAR model estimated on the in-sample period ending
#      in 2020. A longer sample is required because the pre-2008 sample alone is too short to reliably fit a BVAR(3).
#      In contrast, Tutorial 7.4 is based on a VAR(1) estimated on a shorter (2008-) in-sample span.
#   b. The regression equation is re-estimated each quarter from 2008 onwards.
#
# In contrast to tutorial 7.4: BVAR(3) against VAR(1)
#   - The main difference to tutorial 7.4 (exercise 1.3) is that the p-values as well as the rRMSEs 
#     are substantially smaller
#   - The diagonal pattern (smallest p-vales, rRMSEs are obtained when h~shift) applies more clearly here

# For completeness, we here paste the previous findings after exercise 1.3 in tutorial 7.4

# 1. Forecasting BIP vs. HP-BIP:

#    - HP-BIP is substantially easier to forecast than raw BIP, because the two-sided
#      HP filter removes unpredictable high-frequency noise, leaving a smoother and more
#      systematic cyclical signal for the predictor to track, see tutorial 7.4, exercise 1.0.
#    - A statistically significant (HAC-adjusted) predictive relationship is
#      detected up to approximately one year ahead between the M-SSA smoothed
#      predictor and the following target variables:
#       (i)   Forward-shifted HP-BIP: a very strong and robust link that
#             persists across all horizons examined (see tutorial 7.4, Exercise 1.0).
#       (ii)  Forward-shifted raw BIP (the above results): a weaker but still detectable link,
#             reflecting the additional noise introduced by the unfiltered
#             target series.
#   - The predictive evidence for raw BIP is further reinforced by the
#     following indirect argument: if future BIP is determined, at least
#     in part, by its HP-filtered smooth counterpart, then the demonstrated
#     ability to forecast HP-BIP translates into meaningful forecast evidence
#     for raw BIP as well — even when the direct statistical link appears
#     weaker. Moreover, the alignment between raw BIP and its HP-filtered
#     trend is expected to strengthen particularly during periods of large
#     economic swings, when the cyclical component dominates and idiosyncratic
#     noise plays a comparatively smaller role. Incidentally, these are also the episodes 
#     of main interest to forecasters.
#   - The main difference to tutorial 7.4 (exercise 1.3) is that the p-values as well as the rRMSEs 
#     are smaller
#
# 2. Systematic Horizon-Shift Pattern:

#    - In the performance matrices, rows index forward shifts (target lead time) and columns
#      index forecast horizons h (the horizon for which each M-SSA filter is optimized).
#      For a given row (fixed shift), forecast performance tends to improve as h increases
#      from left to right, peaking when h aligns with — or slightly exceeds — the shift value.
#      This corresponds to the diagonal (and just above) of the performance matrices. 
#
#    - Clarity of the pattern depends on the target series:
#        (i)  HP-BIP (low-noise target): the diagonal pattern is strong, clean, and
#             consistent across all shift values, see exercise 1.0 above.
#        (ii) Raw BIP (high-noise target): the pattern is present but partially obscured
#             by the high-frequency noise in the unfiltered target series.
#
#    - This regularity reflects a coherent and interpretable internal structure:
#      each M-SSA design is effective at (or towards) the horizon for which it was optimized,
#      confirming that the optimization criterion successfully encodes horizon-specific
#      information into the filter design — and that this encoding is empirically
#      recoverable from out-of-sample forecast evaluation (at least in the absence of strong noise).

# 3. Impact of the COVID-19 Pandemic on Out-of-Sample Evaluation:
#    - Including pandemic observations (2020–2021) in the validation sample inflates both
#      p-values and rRMSEs, obscuring the underlying systematic horizon-shift patterns
#      with a small number of extreme outlier-driven observations.
#    - When pandemic observations are included, direct forecasts fail to outperform the
#      naive mean benchmark at forward shifts greater than two quarters — a result driven
#      primarily by the distorting influence of the pandemic episode rather than by a
#      genuine deterioration in predictor quality.
# =================================================================


################################################################################################################
# Exercise 2 Analyze revisions of M-SSA components predictor
# -The new predictor relies on quarterly up-dating of the (OLS-) regression weights
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
# Specify data matrix for OLS regression: we insert mmse_array instead of mssa_array for the explanatory variables
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
# Note: for simplicity we here compute an OLS regression 
if (length(sel_vec_pred)>1)
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mmse_array[sel_vec_pred,,h+1]))
} else
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mmse_array[sel_vec_pred,,h+1]))
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


