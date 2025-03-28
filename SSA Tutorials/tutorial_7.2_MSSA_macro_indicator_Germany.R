# Tutorial 7.2: application of the M-SSA to quarterly macro-data

# The previous tutorial 7.1 demonstrates asymptotic convergence of the relevant performance numbers to 
#   expected values (as derived in the M-SSA paper), assuming knowledge of the data generating process (DGP)
# -Tutorial 7.1 checks the theory

# In contrast, we here have relatively short (in-sample) spans and we (have to) allow for (and address) 
#   model misspecification

# Purposes of this tutorial:
# -Apply the M-SSA to quarterly (German) macro-data during insecure times 
#   -Tutorials 7.1-7.3 were written early 2025 and rely on data up to Jan-2025
#   -The HP-filter signals a severe on-going (and worsening) recession, see exercise 4 below
#     -Germany endures a recession: we observe negative BIP-growth since several quarters
#   -Can we use M-SSA to predict future BIP dynamics: what are the prospects for 2025 and 2026?
# -Analyze various designs for nowcasting and forecasting German GDP (BIP:=Brutto Inland Produkt)
# -Analyze effects of model misspecifications (the VAR(1) cannot render recessions properly)
# -Infer possible solutions for eluding model misspecification issues and analyze their efficacity
# -Provide an empirical background and basic insights to understand the various forecast designs 
#   proposed in tutorial 7.3 

# Some background information:
# -As discussed in tutorial 7.1, we do not deliver `GDP numbers' ("give me the number")
#   -Such a `number' would be subject to a forecast interval whose width would invariably invalidate its 
#     relevance. 
#   -Forecasting GDP `numbers' is almost surely (with probability one) a futile exercise, 
#     at least in a multi-step  ahead perspective (several quarters ahead)
# -We here focus on looking ahead (sensing) the future growth dynamics as contained (but masked/hidden) 
#     in present-day data: we try to `extract the minute signal` and `skip the dominating noise'
#   -The reliance in a `signal' may be anchored in the concept of `business cycle'
#   -In his talk at the University of Chicago’s Booth School of Business (March 07, 2025), FED-chair Jerome Powell 
#       said: "As we parse the incoming information, we are focused on separating the signal from the noise 
#       as the outlook evolves", suggesting in his talk that the FED should not (over)react to noise.
#   -Similarly, a forecast procedure should not be reactive to `noise'
#   -However, when targeting BIP in its entirety, including its noisy part, classic direct forecasts are 
#     subject to overfitting, mainly because the components that dominate the MSE (the erratic 
#     high-frequency pulses) distract the OLS optimization from fitting the (much weaker) systematic dynamics  
# -M-SSA in this application is about dynamic aspects of prediction: 
#     -M-SSA emphasizes the target correlation, thereby ignoring `static' level and scale adjustments (calibration), needed to generate GDP `numbers' 
#     -We derive predictors which do not emphasize unilaterally a mean-square error metric (we already know the outcome of doing so)
#       -Instead, M-SSA emphasizes ALSO left-shift/lead/advancement and smoothness (few false alarms): AST trilemma
# -We try to address questions like: 
#   -Did we reach the bottom of the current recession in Germany (based on data up to Jan-2025)? 
#     -The HP filter suggests that we did not, see exercise 4 below
#   -Is the economy currently recovering (Jan 2025)?
#   -Can we expect to reach above long-term growth in foreseeable time? 
# -To be clear: don't ask for a `number'
#   -In our plots, predictors are standardized and positive/negative readings suggest above/below 
#     long-term growth
#   -M-SSA controls the rate of zero-crossings, i.e., the rate of crossings of the predictors above or 
#     below long-term average growth
#   -Therefore we can control for the number of noisy (false) alarms (signaling below or above long-term growth)
# -BIP `numbers' could be obtained by calibration of the standardized M-SSA predictors on BIP
#   -Determine optimal static level and scale adjustments by linear regression

# The tutorial is structured into exercises
# -Exercise 1 discusses important design decisions (target specification) and applies M-SSA to the data
#   -Unfortunately, the VAR(1) model is subject to misspecification resulting in poor performances
# -Exercise 2 will analyze the (main) causes of misspecification and propose solutions to overcome undesirable consequences
# -Exercise 3 combines all ingredients to propose a recipe for constructing the M-SSA BIP predictors
# -Exercise 4 compares our results with the HP-filter
#   -M-SSA contradicts HP (and the future will tell)


#----------------------
# Start with a clean sheet
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
# Useful M-SSA wrappers 
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))


#------------------------------------------------------------------------
# Exercise 1: apply M-SSA to quarterly German Macro-data

# 1.1. Load data and select indicators
# 1.1.1 We first look at the original files: BIP `numbers` refer to this data
data_file_name<-c("Data_HWI_2025_02.csv","gdp_2025_02.csv")
# Monthly data
data_monthly<-read.csv(paste(getwd(),"/Data/",data_file_name[1],sep=""))
tail(data_monthly)
# Quarterly data: BIP in the first data column
data_quarterly<-read.csv(paste(getwd(),"/Data/",data_file_name[2],sep=""))
tail(data_quarterly)

# 1.1.2 We do not work with original (unprocessed) data
# -Instead we apply the following transformation steps
#   -Log-transform (to positive series) 
#   -Quarterly differences (to emphasize growth)
#   -Standardization: series are zero-centered and scaled to unit variance 
#   -Trimming: the singular Covid-data is trimmed (we trim to 3 sigma, as can be observed in the following plot) 
# -In summary, the `numbers' that will be plotted in the following figure seemingly do not relate to the original data
#   -After transformation, the numbers are not interpretable in terms of GDP or industrial production
# -But the transformations help in extracting the minute signal from the overwhelming noise by M-SSA
# -If explicitly needed, forecasts could be traced-back to original `numbers' by straightforward inverse transformations

# We proceeded to the above sequence of transformations and selected a couple of series deemed (very) important
#   -The transformed data is provided in the following macro data-file
load(file=paste(getwd(),"\\Data\\macro",sep=""))
tail(data)
# Remarks on the format of the data file:
# -Series are standardized
# -The target column (first one) in the above file refers to the series to which the target filter will be applied
#   -BIP forward-shifted by the publication lag
# -All subsequent columns correspond to the explanatory data available in January 2025 for nowcasting or 
#     forecasting the target
#   -BIP in the second column is an explanatory variable: it is lagged by two quarters relative to the target 
#     column (BIP is subject to a publication lag)
# -A nowcast of BIP means that we compute an estimate of the output of the two-sided HP-filter when 
#   applied to the first (target) column. 
# -A forecast at horizon h=4 (one year) means that we shift forward by one additional year the target
   

# Remarks on the publication lag
# -BIP has a publication lag of one quarter actually 
#   -But we shifted the data in the target column one additional quarter upwards (forward) to be on the 
#     safe-side (for example to account for data revisions, which are ignored here)
# -Keep in mind this feature of our design: performances at a forecast horizon of three quarters in our plots 
#   might be indicative of performances a full year ahead, as well


# Let's now specify the publication lag: 2 quarters for BIP in the target column
#   -This lag will be used extensively when shifting the target forward
#   -We shall always add lag_vec[1] to the forecast horizon to obtain the effective forward-shift
lag_vec<-c(2,rep(0,ncol(data)-1))

# Plot the data:
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

# Comments:
# -The explanatory variable BIP (red) is lagging the target BIP (black) by lag_vec[1]=2 quarters 
# -The figure suggests that our selection of the publication lag might be too large since the target column
#     anticipates peaks and dips of the other series by one quarter during crises (the target is left-shifted)
# -This excessive shift gives us some safety-margin regarding data revisions (which are ignored here)
# -The series are standardized to account for the different scales of the data
# -Pandemic is trimmed to 3 sigma (the trimming affects also the financial crisis but to a much lesser extent)
# Remarks:
# -We use trimming because classic approaches (HP, VAR-model) are sensitive to the singular COVID readings
#   -M-SSA by itself (the optimization algorithm) would be quite robust against the singularity
#   -But M-SSA could be affected indirectly, by the VAR-modeling which is sensitive to the Pandemic outliers 


# Select the macro indicators for predicting the target by M-SSA: 
#   Five dimensional design in accordance with expert feedback
select_vec_multi<-c("BIP","ip","ifo_c","ESI","spr_10y_3m")

# Compute the relevant data matrix x_mat
x_mat<-data[,select_vec_multi] 
rownames(x_mat)<-rownames(data)
n<-dim(x_mat)[2]
# Number of observations
len<-dim(x_mat)[1]

# Note: the target column is ignored (it is redundant)
#   -In our comparisons we will simply shift the BIP-column upwards by lag_vec[1]+shift, where shift=0 (nowcast) or shift>0 (forecast)
tail(x_mat)
#------------------------------
# 1.2. Target filter: 
# -We apply a filter to the target series (BIP shifted upward by lag_vec[1]+shift quarters) in order to
#   damp noise (the unpredictable high-frequency components of BIP)
# -Main idea (forecast `philosophy'):  removing/damping the unpredictable part (noise) helps in 
#   addressing the predictable portion (signal) of BIP 
#   -As we shall see in tutorial 7.3, statistical significance of predictors can be verified multiple 
#     quarters ahead in this framework
# -Question: which target are we aiming for? 
#   -This choice should support our prospect to forecast BIP multiple quarters ahead
# -The target specification is a difficult problem that requires a comprehensive analysis of the prediction problem:
#   -What is the data looking like?
#   -What are the main purposes (priorities) of the forecast design?

# Note: 
# -The target specification is exogenous to M-SSA 
# -M-SSA assumes that a target has been specified: once done, optimal predictors can be derived
# -The discussion here is not about M-SSA directly, but about an important guide-line, telling M-SSA 
#   what signal to look for

# Back to the selection of an appropriate target:
# -The HP-filter has some interesting (not well-known) properties, see tutorial 2 
# -The classic quarterly design assumes an HP(1600) with lambda=1600
# -However HP(1600) alters relevant information: for example by smoothing out narrow recession dips
#   -See a corresponding critic by Phillips and Jin (2021), suggesting that HP(1600) is `too smooth' (insufficiently flexible)
#   -Dynamic changes as obtained by HP(1600) are too weak (too smooth) to be `useful' in our forecast context (dynamic changes of HP(1600) over a one-year horizon are weak)
#   -As a result, left-shifts (anticipations) of the M-SSA predictor as a function of the forecast horizon 
#        cannot be obtained effectively 
#   -Also, HP(1600) seems (more) sensitive to the Pandemic: the finite-length truncated filter looks `terrible'
# -To summarize
#   -We here select a more adaptive HP(160) design (tutorial 7.3 will explore even more adaptive settings)
#   -HP(160) is able to track narrow recession dips `better': it is also able to track the Euro-Area (sovereign debt) crisis 
#   -HP(160) is able to track dynamic shifts occurring within a one-year horizon better than HP(1600)
# -Overall, we prefer HP(160) as a target specification for this application

lambda_HP<-160
# Filter length: roughly 4 years. 
#   The length should be an odd number, see tutorial 7.1 for background (mirroring of right tail to obtain two-sided target in M-SSA)
L<-31
# Compute the filter: we rely on the mFilter R-package in the following wrapper
target_obj<-HP_target_sym_T(n,lambda_HP,L)

gamma_target=t(target_obj$gamma_target)
symmetric_target=target_obj$symmetric_target 
colnames(gamma_target)<-select_vec_multi


# We plot the right tails of the target filter (the left tails will be obtained by `mirroring', see tutorial 7.1)
#   -To each series of the multivariate design, we assign a target, namely the two-sided HP applied to this series, see tutorial 7.1
par(mfrow=c(1,1))
ts.plot(gamma_target,col=rainbow(n),main="Target filters: the right tails will be mirrored to obtain the two-sided target")
abline(v=1+(1:n)*L)

# We tell M-SSA to mirror the right tail of the HP (as shown in the above figure) at the center peak: 
#   symmetric_target==T (Boolean is true)
symmetric_target
# This means that the effective target will by the two-sided filter


#-------------------------
# 1.3. Fit the VAR
# Select any in-sample span: the effect on the final M-SSA predictor is modest
#   -The VAR is sparsely parametrized  (p=1 and regularization)
# Set in-sample span: full set
date_to_fit<-"2200"
# Set in-sample span: prior Pandemic
date_to_fit<-"2019"
# Set in-sample span: prior financial crisis: this is the one we select here
date_to_fit<-"2008"
data_fit<-na.exclude(x_mat[which(rownames(x_mat)<date_to_fit),])#date_to_fit<-"2019-01-01"
# Have a look at cross correlations: in-sample span
acf(data_fit)
# Check in-sample specification
tail(data_fit)

# VARMA modelling
p<-1
q<-0
set.seed(12)
V_obj<-VARMA(data_fit,p=p,q=q)
# Apply regularization: see vignette to MTS package
threshold<-1.5
V_obj<-refVARMA(V_obj, thres = threshold)

# If interested, have a look at diagnostics (they should look fine for data prior the financial crisis)
if (F)
  MTSdiag(V_obj)
# Sigma
Sigma<-V_obj$Sigma
# AR(1)
Phi<-V_obj$Phi
Theta<-V_obj$Theta

#---------------------------------------
# 1.4. MA inversion: as in the (univariate) SSA, we need to specify the data generating process (DGP) to M-SSA 
# -For this purpose we rely on the Wold decomposition of the DGP: MA-inversion of the above VAR
# -A plot should pop-up in the plot-panel, after running the next code line, illustrating the MA-inversion
#   -An MA-inversion is obtained for each of the five series used in the multivariate design
# -The MA-inversion can be used to interpret the VAR: 
#   -Large weights of series j for series i mean that series j is an important explanatory variable for series i 
#   -The rate of decay of the weights is indicative about the memory of the data generating process
#   -The plot also discloses leads or lags (the peak of spread is generally right-shifted)
#   -In principle, the MA-weights should reflect the previous acf-plot `somehow'
MA_inv_obj<-MA_inv_VAR_func(Phi,Theta,L,n,T)

xi<-MA_inv_obj$xi

#-----------------------------------
# 1.5. Call to M-SSA: specify the forecast horizon and the HT constraint(s)
# One year ahead forecast: 4 quarters + publication lag
delta<-4+lag_vec[1]
delta
# Specify HT constraint: these numbers were derived by imposing a larger HT than the classic MSE-predictor, see below for context
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)

# We can now apply the wrapper to M-SSA proposed in tutorial 7.1
# -The wrapper requires the following information/instructions
#   -Forecast horizon delta
#   -HT constraint ht_mssa_vec
#   -The data-generating process: MA-inversion xi and Sigma
#   -The target: gamma_target together with the Boolean symmetric_target signifying mirroring (of the right tail of the one-sided target) 
#   -The last boolean T triggers a plot of the filter coefficients (in the plot panel)
MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)

# Extract M-SSA filter
bk_x_mat=MSSA_main_obj$bk_x_mat
# We can also retrieve the complete M-SSA object (which contains more information, see tutorial 7.1)
MSSA_obj=MSSA_main_obj$MSSA_obj 
# M-SSA also computes the classic M-MSE predictor: M-SSA tries to track M-MSE while being smoother (less noisy zero-crossings) 
gammak_x_mse<-MSSA_obj$gammak_x_mse
colnames(bk_x_mat)<-colnames(gammak_x_mse)<-select_vec_multi


# Note that we can compute the HT of the MSE predictor and verify that our constraints impose larger HTs for M-SSA
for (i in 1:ncol(gammak_x_mse))
  print(paste("HT M-MSE of series ",select_vec_multi[i],": ",compute_holding_time_func(gammak_x_mse[,i])$ht, " vs. imposed HT of ",ht_mssa_vec[i]," by M-SSA",sep=""))
# Typically, the M-MSE benchmark tends to be noisy (many crossings) and M-SSA allows for an explicit control
#   of this practically relevant property of a predictor


#-----------------------
# 1.6. Filter: apply the M-SSA filter to the data
# Note that delta accounts for the publication lag so that the output of the two-sided filter is left-shifted accordingly

filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)

mssa_mat=filt_obj$mssa_mat
target_mat=filt_obj$target_mat
mmse_mat<-filt_obj$mmse_mat

# We can verify that the target (two-sided HP applied to BIP) is shifted upward (forward) by the publication-lag+delta=2+4
cbind(target_mat[,1],mssa_mat[,1])[(L-6):(L+5),]


# Plots: for each series we plot the target (black) and the M-SSA (blue) and M-MSE (green) predictors 
# The in-sample span is marked by a vertical line
for (i in n:1)
{
  par(mfrow=c(1,1))
  mplot<-cbind(target_mat[,i],mssa_mat[,i],mmse_mat[,i])
  colnames(mplot)<-c(paste("Target: HP applied to ",select_vec_multi[i],", left-shifted by ",delta-lag_vec[1],"(+publication lag) quarters",sep=""),"M-SSA","M-MSE")

  colo<-c("black","blue","green")
  main_title<-paste("M-SSA ",select_vec_multi[i],": forward-shift=",delta-lag_vec[1],", in-sample span ending in ",rownames(data_fit)[nrow(data_fit)],sep="")
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
# 1.7. Compute performance metrics
# Sample mean-square errors of M-SSA
apply(na.exclude((target_mat-mssa_mat)^2),2,mean)
# Sample mean-square errors M-MSE: 
#   -For true models  and long samples (in a simulation context) the MSE of M-SSA is larger than the MSE 
#     of M-MSE because we impose stronger smoothness (a larger HT) to M-SSA
apply(na.exclude((target_mat-mmse_mat)^2),2,mean)

# Correlations between target and M-SSA: sample estimates converge to criterion values for increasing 
#   sample size, assuming the VAR to be true, see tutorial 7.1
# The following results look bad: we obtain negative sample correlations...
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[,i],mssa_mat[,i])))[1,2])
# We can compare these sample estimates to the criterion value achieved by M-SSA for each series: 
#   -The latter are the expected values of the former if the model is not misspecified
#   -Due to maximization, the `true' correlations are always positive
#   -A large discrepancy between true and sample correlations suggests that the VAR(1) might be misspecified 
#     or that the sample is too short (random fluctuation). Likely, both apply here...
MSSA_obj$crit_rhoy_target

# We shall analyze the problem and propose solutions in exercise 2 below.

# M-SSA optimizes the target correlation under the holding time constraint:
# We can now compare empirical and theoretical (imposed) HTs: sample HT converges to imposed HT for increasing sample size len
unlist(apply(mmse_mat,2,compute_empirical_ht_func))
unlist(apply(mssa_mat,2,compute_empirical_ht_func))
ht_mssa_vec
# In contrast to the sample correlations above, the sample HTs look fine, given the rather short sample (random fluctuation)
# In any case, we'd like M-SSA to be smoother than M-MSE: 
#   -In this case, the following ratios of HTs should all be smaller one and this looks fine!
unlist(apply(mmse_mat,2,compute_empirical_ht_func))/unlist(apply(mssa_mat,2,compute_empirical_ht_func))

# M-SSA is effectively smoother than the classic M-MSE benchmark as requested!


#############################################################################################################
# Exercise 2
# We address the following questions:
# -What are the (main) causes of model misspecification
# -How can we improve performances and in particular the (sample) target correlations (which were terrible, see exercise 1 above)?

# The following plot of BIP suggests that
# 1. The forecast problem is rather difficult (due to noise)
# 2. The predictors are too much right-shifted (retarded): 
#   Note: the target in this plot is left-shifted by lag_vec[1]+delta quarters
par(mfrow=c(1,1))
i<-1
mplot<-cbind(target_mat[,i],mssa_mat[,i],mmse_mat[,i])
colnames(mplot)<-c(paste("Target: HP applied to ",select_vec_multi[i],", left-shifted by ",delta-lag_vec[1],"(plus publication lag) quarters",sep=""),"M-SSA","M-MSE")

colo<-c("black","blue","green")
main_title<-paste("M-SSA ",select_vec_multi[i],": ",delta-lag_vec[1]," quarters ahead forecast horizon, in-sample span ending in ",rownames(data_fit)[nrow(data_fit)],sep="")
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


# Let's try a larger forecast horizon than strictly necessary: we call this a `forecast excess'
f_excess<-4
# Increase artificially delta in the call to M-SSA (delta is the forecast horizon)
delta_excess<-delta+f_excess
# All other settings remain the same: we now call M-SSA with that larger forecast horizon

MSSA_main_obj<-MSSA_main_func(delta_excess,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)

bk_x_mat_excess=MSSA_main_obj$bk_x_mat
MSSA_obj=MSSA_main_obj$MSSA_obj 
# Benchmark MSE predictor
gammak_x_mse<-MSSA_obj$gammak_x_mse

colnames(bk_x_mat)<-colnames(gammak_x_mse)<-select_vec_multi

# Filter: use the new bk_x_mat_excess but apply the original delta (not delta_excess) for shifting the target

filt_obj<-filter_func(x_mat,bk_x_mat_excess,gammak_x_mse,gamma_target,symmetric_target,delta)

# We just need the new M-SSA for comparison
mssa_excess_mat=filt_obj$mssa_mat


# Compute the new sample target correlations: 
# -Recall that the previous sample target correlations were negative
# -But now they have turned positive! This looks good, indeed. 
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[,i],mssa_excess_mat[,i])))[1,2])

# What happened? 
#   -The excess forecast horizon leads to a corresponding left-shift (lead/advancement) of the M-SSA predictor
#   -Therefore, the `doped' M-SSA is able to track the recession dips more timely
#   -The more timely tracking of the recession dips leads to a positive sample target correlation
#   -The VAR(1) is unable to `explain' (or generate) recessions dips: it is misspecified
#   -We can account for this model misspecification by asking M-SSA to be faster than necessary (if the VAR were the true data generating process)

# Compute sample HTs: M-SSA keeps the expected HT fixed as desired (independent of forecast horizon)
unlist(apply(mssa_excess_mat,2,compute_empirical_ht_func))
# Compare with previous M-SSA: differences are mainly due to random fluctuations
unlist(apply(mssa_mat,2,compute_empirical_ht_func))
# New M-SSA is still markedly smoother than classic MSE predictor, as desired
unlist(apply(mmse_mat,2,compute_empirical_ht_func))/unlist(apply(mssa_excess_mat,2,compute_empirical_ht_func))


# Plot: 
#   -Let us standardize all series for better visual inspection
par(mfrow=c(1,1))
# Scale the data 
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
#   -If the true process were the fitted VAR(1), then we would probably not observe the marked recession dips
#   -Therefore, the obtained left-shift would not improve performances (quite the opposite in fact)
#   -But in the presence of strong asymmetric down-turns, a `faster' filter can track the relevant dynamics better 

# By the way: we see that both predictors bottomed-out around late 2023 and are now reverting 
#   towards average growth (zero-line)
# Notes:
# -The trough (minimum) of the grow-rate (plotted above) anticipates the trough of BIP by up to several quarters
# -The timing of the BIP-trough is sandwiched between the trough and the next zero-crossing of the growth-rate 


# Findings:
# -We can address misspecification of the VAR(1) (for data with marked recession episodes) by:
#   1. allowing for a stronger left-shift of the predictor (excess forecast) and by
#   2. allowing for a re-scaling (calibration) of the left-shifted predictor, which is typically subject 
#         to excessive zero-shrinkage
# We now apply these findings towards the construction of predictors for German GDP (BIP)



################################################################################################################
# Exercise 3: construction of (standardized) M-SSA BIP predictors

# -M-SSA generates five outputs: for BIP, ip, ifo, ESI and spread
# -We now propose a construction in three steps for the M-SSA BIP predictors
# A. Equal-weighting
# -We consider each of the five M-SSA outputs as an equally valid and informative predictor for future BIP
# -Therefore we aggregate all five predictors equally, assuming that each one was previously standardized
#   -Cross-sectional aggregation: equal weighting
# B. Forecast excess
# -We select a larger delta than strictly necessary if the VAR were true (forecast-excess, see exercise 2)  
# C. Forecast horizons:
# -We compute M-SSA predictors based on A. and B. above, targeting BIP at horizons 0 (nowcast), 1, 2, 4 (one year) and 6 quarters ahead
#   -We then compute performances of each of the five predictors relative to BIP shifted by 0,1,2,4,6 quarters: 
#       -We consider all 5*5=25 combinations
#   -We also consider statistical significance, by relying on different statistics (using HAC estimates of variances)

# Start with the interesting forecast horizons, as indicated above
h_vec<-c(0,1,2,4,6)
# Forecast excesses: 
#   A stronger look-ahead could be obtained by imposing larger forecast excesses (at the detriment of noise)
# 1. Aggressive setting (with all drawbacks)
f_excess<-6
# 2. Less aggressive `OK'-setting: try anyone
f_excess<-4
mssa_bip<-mssa_ip<-mssa_esi<-mssa_ifo<-mssa_spread<-NULL
# Compute M-SSA predictors according to steps A,B,C 
for (i in 1:length(h_vec))#i<-1
{
# -delta is the forecast horizon imposed to M-SSA: M-SSA optimizes filters for a forecast horizon of delta
# -delta can deviate from the effective forecast horizon h_vec[i]: it is a hyperparameter that conditions the optimization process  
# -For each forecast horizon h_vec[i] we determine delta as the sum of h_vec[i], publication lag and forecast excess
# -We now compute delta for BIP, based on the publication lag of BIP (lag_vec[1]) and the selected forecast excess  
  delta<-h_vec[i]+lag_vec[1]+f_excess

# Apply M-SSA
  
  MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)
  
  bk_x_mat=MSSA_main_obj$bk_x_mat
  MSSA_obj=MSSA_main_obj$MSSA_obj 
  colnames(bk_x_mat)<-select_vec_multi
  
# Filter the data
  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
  
  mssa_mat=filt_obj$mssa_mat
  target_mat=filt_obj$target_mat
  mmse_mat<-filt_obj$mmse_mat
  colnames(mssa_mat)<-select_vec_multi
# Select M-SSA BIP (we retain the output for BIP only and skip all other M-SSA outputs)  
  mssa_bip<-cbind(mssa_bip,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[1])])
  
# Next, we compute M-SSA for the remaining ip, ifo, ESI and spread series  
# These series are not subject to publication lags (or at least lags are smaller)
# Therefore delta is smaller than for BIP above
  delta<-h_vec[i]+f_excess
  
  MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)
  
  bk_x_mat=MSSA_main_obj$bk_x_mat
  MSSA_obj=MSSA_main_obj$MSSA_obj 
  colnames(bk_x_mat)<-select_vec_multi

  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
  
  mssa_mat=filt_obj$mssa_mat
  target_mat=filt_obj$target_mat
  mmse_mat<-filt_obj$mmse_mat
  colnames(mssa_mat)<-select_vec_multi
  
# Select M-SSA-ip, -ifo, -ESI and -spread  
  mssa_ip<-cbind(mssa_ip,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[2])])
  mssa_ifo<-cbind(mssa_ifo,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[3])])
  mssa_esi<-cbind(mssa_esi,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[4])])
  mssa_spread<-cbind(mssa_spread,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[5])])
  
}

# Standardize and aggregate all five predictors: equal weighting
indicator_mat<-(scale(mssa_bip)+scale(mssa_ip)+scale(mssa_ifo)+scale(mssa_esi)+scale(mssa_spread))/length(select_vec_multi)

colnames(indicator_mat)<-colnames(mssa_bip)<-colnames(mssa_ip)<-colnames(mssa_ifo)<-colnames(mssa_esi)<-colnames(mssa_spread)<-paste("Horizon=",h_vec,sep="")
rownames(indicator_mat)<-rownames(x_mat)

# The five M-SSA predictors: one for each forecast horizon
tail(indicator_mat)

# Plot target and predictors and compute sample target correlations: compute all 5x5 combinations of forecast horizon and forward-shift
target_shifted_mat<-NULL
cor_mat<-matrix(ncol=length(h_vec),nrow=length(h_vec))

for (i in 1:length(h_vec))#i<-1
{
# Forward-shift: forecast horizon plus publication lag  
  shift<-h_vec[i]+lag_vec[1]
# Compute target: two-sided HP applied to BIP and shifted forward by forecast horizon plus publication lag
  filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,shift)
  target_mat=filt_obj$target_mat
# Select BIP (first column)  
  target<-target_mat[,"BIP"]
# Collect the forward shifted targets: 
  target_shifted_mat<-cbind(target_shifted_mat,target)
# Plot indicators and shifted target
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

# Compute sample target correlations of all M-SSA predictors with the shifted target: 
# The final matrix will contain all 5*5 combinations of forecast horizon and forward-shift
  for (j in 1:ncol(indicator_mat))
    cor_mat[i,j]<-cor(na.exclude(cbind(target,indicator_mat[,j])))[1,2]

}
# Check the shifts: 
#   The target is shifted upward by publication lag (assumed to be 2 quarters) + forecast horizon relative to the predictor (in the first column) 
cbind(indicator_mat[,1],target_shifted_mat)[(L-10):(L+6),]

# We can now have a look at the target correlations
colnames(cor_mat)<-paste("M-SSA: h=",h_vec,sep="")
rownames(cor_mat)<-paste("Shift of target: ",h_vec,sep="")
cor_mat
# -We can see that M-SSA predictors optimized for larger forecast horizons (from left to right in cor_mat) 
#     correlate more strongly with correspondingly forward-shifted targetS (from top to bottom in cor_mat)
# -For a given forward-shift (row), the largest correlations tend to lie on (or to be close to) the diagonal element of that row in cor_mat


# -We infer from the observed systematic pattern, that the M-SSA predictors tend to be informative about future BIP trend growth
# -Also, since future BIP trend growth tells something about the low-frequency part of future BIP, we may infer that 
#   the M-SSA predictors are informative about future BIP (assuming the latter is not white noise)
# -However, (differenced) BIP is a noisy series (the existence of recessions suggests that it is not white noise)
# -Therefore, it is difficult to assess statistical significance of forecast accuracy with respect to BIP (see tutorial 7.3 for a more refined analysis)
# -But we can assess statistical significance of the effect observed in cor_mat, with respect to HP-BIP (low-frequency part of BIP)
# -For this purpose we regress the predictors on the shifted targets and compute HAC-adjusted p-values of the corresponding regression coefficients
t_HAC_mat<-p_value_HAC_mat<-matrix(ncol=length(h_vec),nrow=length(h_vec))
for (i in 1:length(h_vec))# i<-1
{
  for (j in 1:length(h_vec))# j<-1
  {
# Regress j-th M-SSA predictor on i-th target    
    lm_obj<-lm(target_shifted_mat[,i]~indicator_mat[,j])
    summary(lm_obj)
# This estimate of the variance matrix replicates std in the above summary (classic OLS estimate of variance)
    sd<-sqrt(diag(vcov(lm_obj)))
# Here we use HAC: we rely on the R-package sandwich  
    sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
# This is the same as
    sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
# Compute HAC-adjusted t-statistic
    t_HAC_mat[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
# Compute HAC-adjusted p-value: 
#   -We consider a one-sided test because we expect the regression coefficient (of predictor on target) 
#     to be positive
    p_value_HAC_mat[i,j]<-pt(t_HAC_mat[i,j], len-length(select_vec_multi), lower=FALSE)
  }
}
colnames(t_HAC_mat)<-colnames(p_value_HAC_mat)<-paste("M-SSA: h=",h_vec,sep="")
rownames(t_HAC_mat)<-rownames(p_value_HAC_mat)<-paste("Shift of target: ",h_vec,sep="")
# p-values: small p-values lie on (or close to) the diagonal
# Statistical significance (after HAC-adjustment) is achieved even towards larger forecast horizons
# As expected, the Significance decreases (p-values increase) with increasing forward-shift
p_value_HAC_mat
# Note: 
#   -We here consider the full sample, including the in-sample span.
#   -We shall examine performances and statistical significance in more detail in tutorial 7.3,
#     including out-of-sample results

#--------------------------------------------------
# The above result suggest predictability of M-SSA indicators with respect to future HP-BIP
# What about future BIP?
t_HAC_mat_BIP<-p_value_HAC_mat_BIP<-matrix(ncol=length(h_vec),nrow=length(h_vec))
BIP_target_mat<-NULL
for (i in 1:length(h_vec))# i<-4
{
# Shift BIP  
  shift<-h_vec[i]+lag_vec[1]
  BIP_target<-c(x_mat[(1+shift):nrow(x_mat),"BIP"],rep(NA,shift))
  BIP_target_mat<-cbind(BIP_target_mat,BIP_target)
# Regress predictors on shifted BIP  
  for (j in 1:length(h_vec))# j<-5
  {
    lm_obj<-lm(BIP_target~indicator_mat[,j])
    summary(lm_obj)
# This one replicates std in summary
    sd<-sqrt(diag(vcov(lm_obj)))
# Here we use HAC  
    sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
# This is the same as
    sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
# Compute HAC-adjusted t-statistic    
    t_HAC_mat_BIP[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
# One-sided test: if predictor is effective, then the sign of the coefficient must be positive (we are not interested in negative signs) 
    p_value_HAC_mat_BIP[i,j]<-pt(t_HAC_mat_BIP[i,j], len-length(select_vec_multi), lower=FALSE)
    
  }
}
colnames(t_HAC_mat_BIP)<-colnames(p_value_HAC_mat_BIP)<-paste("M-SSA: h=",h_vec,sep="")
rownames(t_HAC_mat_BIP)<-rownames(p_value_HAC_mat_BIP)<-paste("Shift of target: ",h_vec,sep="")
# p-values: 
# -In contrast to HP-BIP, significance with respect to forward-shifted BIP is less conclusive: BIP is much noisier
# -However, we still find evidence of the previously observed systematic pattern in the new correlation matrix
#   -For increasing forward-shift of BIP (from top to bottom), the M-SSA indicators optimized for 
#     larger forecast horizon (from left to right) tend to perform better
# -These results could be altered by modifying the forecast excess 
#   -f_excess=6 is a more aggressive setting
p_value_HAC_mat_BIP

# Technical Note: 
# -Sometimes the HAC-adjustment seems to deliver inconsistent results (might be a problem in the R-package sandwich)
#   -In particular, in some cases the adjusted variance is substantially smaller than the classic OLS estimate
# -In tutorial 7.3 we account in a `pragmatic' way for this problem:
#   -We compute the HAC-adjusted variance as well as the standard/classic OLS variance
#   -We select the larger of the two when computing t-statistics and p-values

# We can now `visualize' the above target correlations by plotting and comparing predictors and forward-shifted targets
# Select an entry of h_vec 
k<-4
# This is the corresponding horizon
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
# Exercise 4: consider the full-length HP
# -In the above exercises we relied on the truncated version of the two-sided HP filter: this filter cannot be obtained at the sample end
# -Instead, we could rely on the common (full-length) HP filter and recompute HAC-adjusted t-statistics to verify 
#   statistical significance (predictability) of M-SSA predictors

# 4.1. Compute full-length HP
len<-nrow(x_mat)
hp_obj<-hpfilter(rnorm(len),type="lambda", freq=lambda_HP)
# Specify trend filters: the above function returns HP-gap
fmatrix<-diag(rep(1,len))-hp_obj$fmatrix
# Check: plot one-sided trend at start, two-sided in middle and one-sided at end
ts.plot(fmatrix[,c(1,len/2,len)],col=c("red","black","green"),main="One-sided at start (red), two-sided (black) and one-sided at end (green)")

# Compute full-length HP trend output
#   -Relies on full-length filter
#   -Does not have NAs at start and end
target_without_publication_lag<-t(fmatrix)%*%x_mat[,1]
# Shift forward by publication lag (2 quarters)
target<-c(target_without_publication_lag[(1+lag_vec[1]):length(target_without_publication_lag)],rep(NA,lag_vec[1]))


# Plot:
#   -Note that full-length HP becomes increasingly asymmetric towards the sample boundaries
#   -The quality towards the sample boundaries degrades
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
# Outcome:
# -The HP is currently indicating an ongoing sharp decline towards the sample end
# -In contrast, the M-SSA predictors envisage a possible recovery over 2025/2026
# -M-SSA and HP are conflicting in terms of future outlooks


# Compute target correlations: 
cor(na.exclude(mplot))
# -Note that the target (full-length HP applied to BIP) corresponds to a nowcast (shift=0)
# -Accordingly, the correlation is maximized at horizon 0 by M-SSA (first row in the above matrix) 


# 4.2 In addition to a nowcast we can also analyze forward-shifts of the target
target_shifted_mat<-NULL
for (i in 1:length(h_vec))
{
# Forward shifts are specified by h_vec: up to 6 quarters ahead  
  shift<-h_vec[i]
  target_shifted_mat<-cbind(target_shifted_mat,c(target[(1+shift):length(target)],rep(NA,shift)))
}

# Recompute correlations and HAC-adjusted t-statistics of regression of M-SSA indicators on shifted full-sample HP trend
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
# One-sided test    
    p_value_HAC_mat[i,j]<-pt(t_HAC_mat, len-length(select_vec_multi), lower=FALSE)
    
  }
}
colnames(cor_mat)<-colnames(p_value_HAC_mat)<-paste("M-SSA: h=",h_vec,sep="")
rownames(cor_mat)<-rownames(p_value_HAC_mat)<-paste("Shift of target: ",h_vec,sep="")

cor_mat
p_value_HAC_mat

# The full-length HP results confirm earlier findings
#   -Confirmation of the systematic effect (left-right vs- top-bottom)
#   -Correlations tend to be larger
#   -p-values tend to be smaller (stronger effect)

# The above findings have been wrapped into a single function called compute_mssa_BIP_predictors_func
head(compute_mssa_BIP_predictors_func)

# This function will be used in tutorial 7.3
# The head of the function needs the following specifications:
# x_mat: data 
# lambda_HP: HP parameter
# L: filter length
# date_to_fit: in-sample span for the VAR
# p,q: model orders of the VAR
# ht_mssa_vec: HT constraints (larger means less zero-crossings)
# h_vec: (vector of) forecast horizon(s) for M-SSA
# f_excess: forecast excesses, see exercises 2 and 3 above
# lag_vec: publication lag (target is forward shifted by forecast horizon plus publication lag)
# select_vec_multi: names of selected indicators


#################################################################
# Summary and main findings
# -When targeting forecast horizons of a year or less, we need to focus on signals (HP-trends) 
#   which allow for sufficient adaptivity (sufficiently strong dynamics over such a time interval) 
#   -For this purpose we selected lambda_HP=160 
#   -The increased adaptivity forces predictors to react to the forecast horizon by a commensurate left-shift (anticipation)
#   -In tutorial 7.3 we shall look at even more adaptive designs
# -Assuming a suitable choice for lambda_HP, the main construction principles behind M-SSA indicators leads to 
#     forecast designs with predictive relevance
#   -Timeliness: The left-shift can be controlled by the forecast horizon
#   -Smoothness: noise-suppression (zero-crossings)) can be controlled effectively by the HT constraint
# -Model misspecification (of the VAR) can be addressed by imposing a forecast `excess' to M-SSA
# -Predicting HP-BIP (the trend component) seems easier than predicting BIP
#   -HP-BIP is fairly exempted from erratic (unpredictable) high-frequency components of BIP
# -The effect of the forecast horizon (hyperparameter) is statistically (and logically) consistent: 
#   -Increasing the forecast horizon leads to improved performances at larger forward-shifts
#   -The forecast horizon is commensurate to the observed `physical' forward-shift of the target 
# -Performances with respect to BIP (instead of HP-BIP) are less conclusive, due in part to unpredictable high-frequency noise
#   -However, the link between the forecast horizon and the physical-shift is still recognizable
#   -More aggressive settings for the forecast excess may reinforce these findings (up to a point)
# -Finally, a predictor of the low-frequency component of (future) HP-BIP is potentially informative about 
#     future BIP (if the latter is not white noise).
#---------------------------------------------------------------------------------------------------



