# Tutorial 7.1 proposes an introduction to M-SSA: 
#   -An extension of the univariate SSA analyzed in the previous tutorials to a multivariate framework

# -This tutorial is built around a simulation exercise: the model specification is derived from an 
#       application of M-SSA to forecasting German GDP (BIP: Brutto Inland Produkt): data available in tutorial 7.3
#   -The data generating process (DGP) relies on a simple VAR(1) fitted to German macro-data, see 
#     tutorials 7.2. and 7.3 for details

# Purpose: demonstrate that M-SSA is `doing the job' when assuming the DGP to be known 
# -Exercise 1 illustrates that sample performances are converging to theoretical expectations
#   -Thereby, we can confirm that theory/formula as derived in the M-SSA paper are pertinent 
# -Exercise 2 transcribes formula of the paper in R-code and provides additional performance checks (nearly asymptotic results) 
# -Exercise 3 wraps the proposed code into compact R-functions which will be used in tutorials 7.2 and 7.3

# -As discussed at a 2025-conference (Conference on Real-Time Data Analysis, Methods, and Applications, Prague, 2025)
#   SSA and M-SSA are about `empirical forecasting`: the method addresses problems which might be relevant 
#   to practitioners by extending the classic mean-square error (MSE) paradigm to address smoothness and
#   timeliness, in addition to MSE: 
#     -Smoothness concerns the number/rate/frequency of zero-crossings of a (centered) predictor/nowcast/forecast
#     -Timeliness addresses retardation/lag/right-shift of a predictor
# -M-SSA extends these concepts to a multivariate design, wherein multiple series can be used for predicting
#   a specific target
# -In such a case, controlling zero-crossings of the predictor is more complex because multiple time series
#   are aggregated into a single predictor, whereby each individual series can 
#   contribute its `own' crossings
# -However, the mathematical trick for addressing this `seemingly complex` problem is neat, simple and effective: 
#   -We know how to derive the lag-one ACF for a multivariate design, see, e.g., the M-SSA paper
#   -The rate of zero-crossings can be related to the lag-one ACF, see SSA- and M-SSA papers
#   -Exercise 1 verifies pertinence of this mathematical trick

#-----------------------
# Start with a clean sheet
rm(list=ls())

# Load the required R-libraries

# Standard filter package
library(mFilter)
# Multivariate time series: VARMA model for macro indicators: used here for simulation purposes only
library(MTS)


# Load the relevant M-SSA functionalities
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

#-----------------------------------------------
# Exercise 1
# 1.1 Specify the data generating process (DGP): 
# -The following model is obtained by fitting a simple VAR(1) to quarterly German macro data, see tutorials 7.2 and 7.3
#   -Since the estimation span can be adjusted, the particular VAR may deviate from tutorials 7.2 and 7.3  
#   -But we can rely on any multivariate model to demonstrate M-SSA
# -Tutorials 7.2 and 7.3 consider a 5-dimensional design comprising BIP (i.e. GDP), industrial production, economic sentiment, spread and an ifo indicator
#   -All series are log-transformed (except spread), differenced (no cointegration) and standardized (to account for grossly different scales)
# -Data span: the quarterly series start at the introduction of the EURO and the in-sample span determined in
#     tutorials 7.2 and 7.3 ends in Dec-2007, leaving the financial crisis entirely out-of-sample
#   -As a consequence, the data span is relatively short (less than 20 years of quarterly data)
#   -Therefore, a verification of (asymptotic) results remains challenging
#     -Which justifies the simulation experiment herein
#   -Overfitting is of concern, too.
#   -We select a sparsely parametrized simple VAR(1) specification, based on library MTS (VARMA, R.Tsay),
#     see tutorials 7.2 and 7.3
#   -Note that M-SSA is fairly insensitive to overfitting and/or to the specification of the in-sample span

# Dimension of the multivariate design: 
n<-5
# AR order: we exclude MA-terms because of invertibility issues (the VARMA(1,1) would not be always invertible)
p<-1
# AR(1) coefficient (we could just plug in any numbers, assuming stationarity)
Phi<-matrix(rbind(c( 0.0000000,  0.00000000, 0.4481816,    0, 0.0000000),
               c(0.2387036, -0.33015450, 0.5487510,    0, 0.0000000),
               c(0.0000000,  0.00000000, 0.4546929,    0, 0.3371898),
               c(0.0000000,  0.07804158, 0.4470288,    0, 0.3276132),
               c(0.0000000,  0.00000000, 0.0000000,    0, 0.3583553)),nrow=n)
colnames(Phi)<-rownames(Phi)<-c("BIP", "ip", "ifo_c","ESI", "spr_10y_3m") 
Phi
# According to Phi, BIP (first row) is related to lagged ifo_c; ip (indust-prod, second row) is related to lagged BIP, ip and ifo; and so on...
#   -But additional dependencies are determined by Sigma below (sample cross correlation matrix of model residuals).
#   -Therefore, a better understanding of the model can be obtained by looking at the impulse response function (MA-inversion) further down
# In any case, M-SSA assumes this model to be true and derives optimal predictors under this assumption
# A better model of the data (than the VAR(1)) could be plugged into M-SSA to eventually obtain even better predictors
#   -To be clear: we do not discuss pertinence of the VAR(1) and accept this result `as is'
# The sparse parametrization is reflected by multiple zeroes in Phi: see MTS-package for details

# Covariance matrix (of noise terms)
# Note: if Phi and Sigma are diagonal, then the VAR reduces to n univariate designs running in parallel
Sigma<-matrix(rbind(c(0.755535544,  0.49500481, 0.11051024, 0.007546104, -0.16687913),
  c(0.495004806,  0.65832962, 0.07810020, 0.025101191, -0.25578971),
  c(0.110510236,  0.07810020, 0.66385111, 0.502140497,  0.08539719),
  c(0.007546104,  0.02510119, 0.50214050, 0.639843288,  0.05908741),
c(-0.166879134, -0.25578971, 0.08539719, 0.059087406,  0.84463448)),nrow=n)

# Simulate a series corresponding to 25 years of quarterly data    
len<-100
set.seed(31)
x_mat=VARMAsim(len,arlags=c(p),phi=Phi,sigma=Sigma)$series

# Use the original column names: GDP (BIP), industrial production, ifo, sentiment and spread
colnames(x_mat)<-c("BIP", "ip", "ifo_c","ESI", "spr_10y_3m") 
# Have a look at the data
#   -We expect to see mutually (cross-section) correlated and (longitudinal) autocorrelated series
#   -Can you glimpse `crises`, i.e., phases characterized by `striking' negative growth?
par(mfrow=c(1,1))
ts.plot(x_mat[(len-99):len,])

# Generate a very long series for our simulation experiment: to verify convergence of sample estimates to expectations    
len<-100000
set.seed(871)
x_mat=VARMAsim(len,arlags=c(p),phi=Phi,sigma=Sigma)$series

# Use the original column names: GDP (BIP), industrial production, ifo, sentiment and spread
colnames(x_mat)<-c("BIP", "ip", "ifo_c","ESI", "spr_10y_3m") 

#-------------------------------------------------------------------
# 1.2 Specify the acausal target: 
# -We want to nowcast and to predict the output of a (two-sided) Hodrick Prescott (HP) filter
#   -We do not want to suggest that HP is the single optimal design for this application (German macro data)
#     see tutorials 2.0 and 2.1
#   -Consequently, we could use alternative signals: see tutorials 3, 4 and 6
#   -The point here is: we could plug any target into M-SSA and obtain optimal predictors by M-SSA
#   -Given the above VAR(1) and given a target, M-SSA solves the corresponding prediction problem. 
# -The sampling frequency of our data is quarterly and the classic HP-specification would be HP(1600) with lambda=1600 
# -However, we here select a smaller lambda=160, see detailed discussions in tutorials 7.2 and 7.3
#   -Briefly: the classic HP(1600) is too smooth (it removes too much relevant information) for the purpose of 
#     predicting BIP
#   -See also a corresponding critic by Phillips and Jin (2021)
lambda_HP<-160
# Filter length: roughly 4 years. 
# Technical side-note: 
#   -We provide only the right half of the two-sided filter to M-SSA: the left tail is obtained by `mirroring'
#   -Therefore, the length of the filter (L=31) should be an odd number to obtain an exact replication of the HP 
#     after `mirroring' (i.e., a symmetric design with a single peak at its center) 
L<-31
# HP filter: we use the R-package mFilter to compute HP in the following wrapper
HP_obj<-HP_target_mse_modified_gap(L,lambda_HP)

hp_symmetric=HP_obj$target
hp_classic_concurrent=HP_obj$hp_trend
hp_one_sided<-HP_obj$hp_mse

# We can see in the following plot that the truncation of the symmetric filter of length L=31 leaves room 
#     for improvement (L is obviously too small)
ts.plot(hp_symmetric,main=paste("Symmetric filter of length ",L,sep=""))
# But we supply the right half of the two-sided filter to M-SSA: the two-sided filter is reconstructed internally by `mirroring'`
#   -The right half is also of length L=31 and therefore the two-sided filter will be of 
#     total length 2*L-1, after reconstruction (`mirroring`) 
#   -Therefore, truncation is not a problem anymore (from a practical point of view at least)
ts.plot(hp_one_sided,main=paste("Right tail of length ",L," of the two-sided filter of length ",2*L-1,sep=""),ylab="")

# Specifically, the two-sided filter will be reconstructed as follows in M-SSA
#   This step suggests that L should be an odd number
reconstructed_two_sided_hp<-c(hp_one_sided[L:2],hp_one_sided[1:L])
ts.plot(reconstructed_two_sided_hp,main=paste("Reconstructed two-sided filter of length ",2*L-1,sep=""),ylab="")
#-------------------------------------------------------------------
# 1.3 MA inversion of VAR(1)
#   -The M-SSA optimization criterion relies on the Wold-decomposition of a stationary process (see M-SSA paper)
#   -We thus compute the MA-inversion of the VAR, based on package MTS
xi_psi<-PSIwgt(Phi = Phi, Theta = NULL, lag = L, plot = F, output = F)
xi_p<-xi_psi$psi.weight
# We now transform the MA-inversion xi_p in a readable format (for M-SSA)
#   -The first L entries, from left to right, are the weights of the first white noise (WN) series
#   -The following L entries are the weights corresponding to the second WN series 
xi<-matrix(nrow=n,ncol=n*L)
for (i in 1:n)
{
  for (j in 1:L)
    xi[,(i-1)*L+j]<-xi_p[,i+(j-1)*n]
}
# Plot the obtained MA inversion
par(mfrow=c(1,n))
for (i in 1:n)#i<-1
{
  mplot<-xi[i,1:min(10,L)]
  
  for (j in 2:n)
  {
    mplot<-cbind(mplot,xi[i,(j-1)*L+1:min(10,L)])
    
  }
  colnames(mplot)<-colnames(x_mat)
  
  colo<-rainbow(ncol(mplot))
  ts.plot(mplot,col=colo,main=paste("MA inversion ",colnames(x_mat)[i],sep=""))
  for (i in 1:ncol(mplot))
  {
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
# The MA-inversion can be useful for understanding the VAR (explainability) and for narrative purposes 
#   -As an example, the MA-inversion of spread (rightmost plot) relies only on lagged spread (spread is leading and the other series are not informative)
#   -In general we see that lagging series rely on leading (in relative terms) series as explanatory variables

#-----------------------------------------------------------------
# 1.4 Target for M-SSA: any acausal (prediction) or causal (smoothing) filter 
# -A target must be specified for each one of the 5 series of the VAR
#   -The target for each one of the five series is the two-sided HP applied to this series, 
#       possibly shifted forward (prediction) or unshifted (nowcast) or shifted backward (backcast)
# -M-SSA filters are structured/organized in vectors and matrices
#   -The filter matrix has dimension n cross (n*L): n rows and n*L 
#   -The i-th row of the filter matrix collects the filters for the i-th target
#   -Each row consists of n filters: a series specific filter is assigned to each of the n explanatory variables
#   -Therefore the number of columns is n*L (each sub-filter has length(L))
# We now specify the target filter accordingly: dimension n cross (n*L)
# We start by the target for the first series
gamma_target<-c(hp_one_sided,rep(0,(n-1)*L))
# For the first series, the target is HP applied to the first series and then zeroes
#   We delimit series by vertical lines in the plot
#   Each series receives L filter coefficients
par(mfrow=c(1,1))
ts.plot(gamma_target)
abline(v=(1:n)*L)
# Note that the filter is one-sided yet and the right tail will be mirrored to the left, see below for further details
# We now proceed to specify the targets of the remaining n-1 series
for (i in 2:n)
  gamma_target<-rbind(gamma_target,c(rep(0,(i-1)*L),hp_one_sided,rep(0,(n-i)*L)))
# The target of each series is just HP applied to this series (still one-sided yet)
par(mfrow=c(1,1))
ts.plot(t(gamma_target),col=rainbow(n))
abline(v=(1:n)*L+1)

# In the above plots, the target filters where one-sided
# We now tell M-SSA that it has to mirror the above filters at their center points to obtain two-sided targets
symmetric_target<-T
# When symmetric_target==T, M-SSA is advised to compute a symmetric target by mirroring 
#   the right tail to the left 
# This is obtained as illustrated in the following plot: the reconstructed filter is of length 2L-1
ts.plot(c(hp_one_sided[L:2],hp_one_sided[1:L]),main=paste("Reconstructed two-sided filter of length ",2*L-1,sep=""),ylab="")
# Note that this procedure (mirroring) assumes L to be an odd number (see previous comments above)
# Further down we will see another equivalent target specification based on supplying the two-sided filter directly

#--------------------------------------------------------------------------------
# 1.5 Forecast horizon
# delta=0 means a nowcast
# delta>0 means a forecast
# delta<0 means a backcast: clearly, this is not our main application case
#   Note that for backcasts the filters may look weird because the HT in the constraint is too small (M-SSA will unsmooth the data)

# Nowcast
delta<-0
# Or one year ahead forecast for quarterly data (try either one or anyone)
delta<-4


# -Specify the holding times (HT) in the M-SSA criterion, see previous tutorials
#   -The HT is the mean duration between consecutive zero-crossings of the (centered) filter
#   -A larger HT means less zero-crossings and consequently a smoother filter that generates less noisy alarms
#   -But increasing HT affects the lag/retardation/right-shift of the filter: AST trilemma, see previous tutorials
# -We have to specify a HT for each series
# -The following numbers are derived from the Macro-application: 
#   -We impose mean durations between 1.5 and 2 years between consecutive zero-crossings of the predictor
#   -Imposing larger numbers would render M-SSA smoother (less zero-crossings)
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Compute corresponding lag-one ACF in HT constraint: see previous tutorials on the link between HT and lag-one ACF  
rho0<-compute_rho_from_ht(ht_mssa_vec)$rho

# The following settings are default values controlling the numerical optimization algorithm
#   -with_negative_lambda==T allows to extend the search to un-smoothing (generate more zero-crossings than benchmark): 
#   -Default value is FALSE (smoothing only)
with_negative_lambda<-F
# Default setting for numerical optimization
lower_limit_nu<-"rhomax"
# Optimization with half-way triangulation: effective resolution is 2^split_grid. Much faster than brute-force grid-search.
# 20 is a good value: fast and strong convergence in most applications
# WE can check if this number is large enough, see further down for details
split_grid<-20

# Now we can apply M-SSA
MSSA_obj<-MSSA_func(split_grid,L,delta,grid_size,gamma_target,rho0,with_negative_lambda,xi,lower_limit_nu,Sigma,symmetric_target)

# In principle we could retrieve filters, apply to data and check performances
#   -But M-SSA delivers a much richer output, containing different filters and useful evaluation metrics
#   -These will be analyzed further down, see exercise 2
# So let's pick out the M-SSA filter
bk_x_mat<-MSSA_obj$bk_x_mat
colnames(bk_x_mat)<-colnames(x_mat)
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-bk_x_mat[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,bk_x_mat[(j-1)*L+1:L,i])
  }
  colnames(mplot)<-colnames(x_mat)
  colo<-rainbow(n)
  ts.plot(mplot,main=paste("MSSA applied to x ",colnames(x_mat)[i],sep=""),col=colo)
  for (i in 1:n)
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
# The above outcome is intuitively appealing:
#   -The M-SSA filters of all five series rely on spread (leading indicator)
#   -The M-SSA filter of spread does not depend on other series
#   -The weight distribution (longitudinal and cross sectional) reflects the dependency as modeled by the VAR(1)
#   -Increased smoothness (larger HTs) is reflected by the (slower) decay of filter weights
#----------------------------------------------
# 1.6 We now apply M-SSA to the simulated data and verify important performance metrics
#   -Select any of the series as your m-th target: m=1,...,n 
m<-m_check<-3
if (m>n)
  print(paste("Warning: m should be smaller than n=",n,sep=""))
bk<-NULL
# Extract coefficients applied to m-th series    
for (j in 1:n)#j<-2
  bk<-cbind(bk,bk_x_mat[((j-1)*L+1):(j*L),m])
colnames(bk)<-colnames(x_mat)
# Plot
par(mfrow=c(1,1))
colo<-rainbow(ncol(bk))
ts.plot((bk),main=paste("M-SSA filter coefficients applied to ",colnames(x_mat)[m],sep=""),col=colo)
for (i in 1:ncol(bk))
  mtext(colnames(bk)[i],col=colo[i],line=-i)
# Apply the j-th component of filter m to the j-th series and aggregate 
#   -This is the predictor of the m-th target
y<-rep(NA,len)
for (j in L:len)#j<-L
{
  y[j]<-sum(apply(bk*(x_mat[j:(j-L+1),]),2,sum))
}

# Apply acausal target filter to the m-th-series
gammak<-hp_one_sided[1:L]
par(mfrow=c(1,1))
ts.plot(gammak,main="Right tail of two-sided target")
z<-rep(NA,len)
# Here the right tail of the filter is mirrored to obtain the acausal two-sided target filter
for (j in L:(len-L))
  z[j]<-gammak%*%x_mat[j:(j-L+1),m]+gammak[-1]%*%x_mat[(j+1):(j+L-1),m]
# Shift z by delta    
if (delta>0)
{  
  zdelta<-c(z[(delta+1):len],rep(NA,delta))
} else
{
  if (delta<0)
  {
    zdelta<-c(rep(NA,abs(delta)),z[1:(len-abs(delta))])
  } else
  {
    zdelta<-z
  }
}
names(zdelta)<-names(y)<-rownames(x_mat)

# Plot last 200 observations of the long simulated span: 
#   -Recall that the experimental design corresponds to quarterly economic data
#   -We plot the last 200 observations of target and M-SSA predictor (delta>0), nowcast (delta=0) or backcast (delta<0)
#   -200 observations correspond to 50 years of quarterly data 
mplot<-cbind(zdelta,y)[(len-200):len,]
ts.plot(mplot,col=c("black","blue"),main="Target (black) and M-SSA (blue): zero-crossings marked by blue vertical lines")
abline(h=0)
abline(v=1+which(sign(mplot[2:nrow(mplot),2]*mplot[1:(nrow(mplot)-1),2])<0),lty=3,col="blue")
# Remarks:
#   -We can recognize four artificial `recessions' (dips or local minima of target in black) over a time span 
#     of roughly 150 observations
#     -On average, a recession every 150/(4*4)~10 years (HP(160) seems quite right here!)
#     -In general, recession durations are shorter than expansions: 
#       -The VAR cannot render this stylized fact properly (Hamilton regime-switching model can map asymmetry)
#   -The two-sided filter cannot reach the sample end
#   -We marked zero-crossings by M-SSA (blue) by vertical dashed lines: M-SSA controls the expected duration 
#     between consecutive zero-crossings, i.e. the holding time (HT)


# We now compute the mean duration between consecutive zero-crossings: this is the sample holding time (HT)
#   -M-SSA controls the expected (theoretical) HT to which the sample HT converges in very long samples, assuming a true model
compute_empirical_ht_func(mplot[,2])
# Compare with the imposed constraint (pretty good match on this short sample) 
ht_mssa_vec[m]


# We now compute performance measures for the entire sample
# 1. Mean-square forecast error
# Technical note: 
#   -M-SSA maximizes the target correlation between zdelta and y
#   -This is equivalent to minimizing MSE up to static scale and level parameters (the latter are ignored by the correlation)
#   -Ex post calibration of static level and scale parameters can be obtained easily by linear regression 
#     -Fit y on zdelta
#   -M-SSA also computes the best static scale adjustment internally, assuming the model is correctly specified
#     -The blue line in the previous figure is optimally scaled
#   -However, M-SSA does not provide any level adjustment, since the data is assumed to be zero-centered (zero-crossings)
#   -The static level-adjustment must be provided explicitly, if desired
# Background (philosophy): M-SSA emphasizes dynamic aspects of forecasting (zero-crossings, lead/advancement, growth).
#   -Static (ex post) adjustments (of level/scale) are deemed less relevant
# Some of the discussants at a recent conference (Conference on Real-Time Data Analysis, Methods, and Applications, Prague, 2025),
#     asked for a GDP-number ("Give me a number"). We fear the correct answer should be: "M-SSA provides a tendency"
# Going a step further: M-SSA as implemented in tutorials 7.2 and 7.3 is designed to provide (new) added value
#     in a multi-step ahead forecast exercise. 
#   -In a multi-step ahead perspective, the tendency is more relevant than a single `number' (assorted with an uninformative wide forecast interval)
# Back to the sample MSE performance metric:
mean((zdelta-y)^2,na.rm=T)

# 2. Target correlation: correlation between target and M-SSA: 
#   -The sample estimate of the target correlation should converge to the true target correlation for 
#       sufficiently long samples (assuming the model is not misspecified)
#   -The true target correlation corresponds to the objective function of the M-SSA criterion
#   -M-SSA maximizes the (true) target correlation under the HT constraint (noise suppression)
#   -Maximizing the target correlation is equivalent to minimizing the mean square error between target and 
#     M-SSA outputs, up to an affine transformation (static level and scale adjustments)
#   -M-SSA determines the optimal scale of the affine transformation, i.e., the blue line in the previous figure is 
#     also the optimal MSE-estimate under the imposed HT constraint (assuming the true level of the data is zero)
#   -The sample correlation is the (1,2) element of the following correlation matrix 
cor(na.exclude(cbind(zdelta,y)))[1,2]
# The following MSSA_obj$crit_rhoy_target is a vector of criterion values (true target correlations)
#   -One criterion value per series
# The above sample target correlation converges to the corresponding criterion value for sufficiently long samples (large len)
MSSA_obj$crit_rhoy_target[m]

# Smoothness (number of zero-crossings): 
#   -M-SSA optimizes the target correlation under the holding time constraint
# We now compare empirical and theoretical (imposed) HTs
compute_empirical_ht_func(y)
ht_mssa_vec[m]
# Findings: the sample HT converges to the imposed HT for increasing sample size len


#####################################################################################################################################
# Summary: The above simulation experiment confirms that M-SSA maximizes the target correlation 
#   subject to the HT constraint
# -Both aspects, target correlation and HT, are deemed relevant in applications (`empirical' forecasting)
# -Timeliness (lead/left-shift/advancement) could be obtained simply by increasing the forecast horizon, see
#   e.g. tutorials 2.1 or 3 (AST trilemma)
############################################################################################################################## 

# Exercise 2
# -Advanced results (intended after a dive into the M-SSA paper, which proposes expressions for expected performance measures)
# -We also look in more detail at some of the additional outputs generated by M-SSA

# Some validation checks: 
# -compute  theoretical (expected) performances and check HT constraint  
# -Target correlation: instead of maximizing the correlation with the acausal target, 
#     the objective function of (M-)SSA considers the correlation with the causal (classic) MSE benchmark
#   -Both `targets` (acausal and causal) are equivalent in the sense that the same (M-) SSA solution is obtained, see papers
#   -When targeting the causal MSE-predictor (instead of the effective target), (M-)SSA attempts to replicate 
#       the latter's optimal tracking (of the effective target) while being smoother
#     -This is a proper smoothing problem, not a prediction problem (see papers)
#   -However, when targeting the acausal target, for example the two-sided HP, (M-) SSA is addressing a 
#     proper prediction problem, combined with a HT constraint imposing additional smoothness 
#   -Both problems (acausal vs. causal M-MSE targets) are qualitatively different, but the outcome (the M-SSA solution) is the same
# M-SSA replicates the classic MSE estimate when the HT in the constraint matches the HT of the MSE benchmark
#   -However, the MSE benchmark is often `noisy' and we want M-SSA to be smoother (stronger noise suppression)`
#   -Therefore the correlation with the MSE benchmark is typically smaller one
#   -But the correlation with the MSE-benchmark is almost surely larger than the target correlation with the 
#     effective (acausal) target, because the latter assumes additional knowledge of future observations
# Let's check these claims. First we compute the correlation of M-SSA with the classic M-MSE predictor: we obtain a 
#   number for each series
MSSA_obj$crit_rhoyz
# Note that the correlations are large, indicating that M-SSA is pretty close to M-MSE

# Next we compute the target correlation with the effective acausal target (here: two-sided HP) 
#   These numbers are substantially smaller since the acausal target `knows` the future
MSSA_obj$crit_rhoy_target
# Below, we shall link these theoretical expressions with sample estimates, for a 
#   better interpretation of their meaning

# Next, have a look at the lag one ACF: it should match the imposed lag-one ACF (note: the HT constraint 
#     can be expressed alternatively in terms of lag-one ACF)
#   -This is a quick check to verify successful numerical optimization 
#   -Convergence: the optimal solution should match the imposed constraint (sufficiently closely)
# If the discrepancy is `too large', the number of iteration steps specified by the hyperparameter 
#   split_grid (see above function call) should be increased
# First: the lag-one ACF the M-SSA solution
MSSA_obj$crit_rhoyy
# Compare with the imposed constraint (note that we can express the constraint either in terms of HT or lag-one ACF, see papers)
rho0
# We see that the M-SSA solution matches the constraint up to negligible rounding erros: 
#   -The numerical optimization converged successfully (the solution is unique due to strict monotonicity, see M-SSA paper)

# Equivalently, the HT of the optimized M-SSA should match the imposed HT
#   -Lag-one ACF and HT can be linked bijectively, see (M-)SSA papers
compute_holding_time_from_rho_func(MSSA_obj$crit_rhoyy)
compute_holding_time_from_rho_func(rho0)

# Optimized nu in discrete difference equation, see paper for interpretation: values above 2 indicate smoothing (less zero-crossings than classic MSE-benchmark) 
MSSA_obj$nu_opt
# M-SSA also computes optimal filters when applied to residuals of the VAR (after MA-inversion)
#   -These filters can be useful for visualization of the smoothing property (impulse response, for `expert' users)
#   -Applying these filters to VAR residuals replicates the ordinary M-SSA output (time series y in 
#     above simulation exercise) if L is sufficiently large (or MA-inversion of VAR decays rapidly to zero)
#       -The approximation (MA-inversion) improves with increasing filter length L
# In a first step, M-SSA computes bk_mat: the filter as applied to the residuals (MA-inversion)
# In a second step, the `relevant' M-SSA filter (bk_x_mat, see the above simulation exercise) is 
#     obtained from deconvolution, see below for details. We now look at bk_mat (as applied to MA-inversion)
bk_mat<-MSSA_obj$bk_mat
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-bk_mat[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,bk_mat[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("SSA applied to epsilon ",colnames(x_mat)[i],sep=""),col=rainbow(n))
}
# We can compare to bk_x_mat (the filter effectively applied to the data)
bk_x_mat<-MSSA_obj$bk_x_mat
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-bk_x_mat[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,bk_x_mat[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("MSSA applied to x ",colnames(x_mat)[i],sep=""),col=rainbow(n))
}
# The weights of bk_mat (applied to residuals) decay more slowly in this example, because the white noise 
#   (WN) sequence of the MA-inversion is noisier than the original VAR(1) (the original data)

# We can now check the deconvolution claim: deconvolution of the MA-inversion xi from bk_mat gives bk_x_mat  
deconv_M<-t(M_deconvolute_func(t(bk_mat),xi)$deconv)
# Check: both filters match perfectly (absolute error is zero)
max(abs(deconv_M-bk_x_mat))
# To summarize
# -bk_x_mat is the relevant filter in applications (the filter which is applied to the data)
# -bk_mat (as applied to the MA-inversion) is the proper solution to the M-SSA criterion: 
#   -bk_x_mat is derived from bk_mat by deconvolution of the MA-inversion 
# -bk_mat can provide some useful visual clues (impulse response) and helps when 
#   trying to understand/interpret the M-SSA solution (explainability) 

# M-SSA also computes the classic MSE filter called M-MSE: 
#   -This is the design obtained by classic signal extraction (Wiener-Kolmogorov)
# First we look at the MSE-filter as applied to the VAR residuals (after MA-inversion of the VAR)
gammak_mse<-MSSA_obj$gammak_mse
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-gammak_mse[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,gammak_mse[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("M-MSE applied to eps ",colnames(x_mat)[i],sep=""),col=rainbow(n))
}
# Second we look at the MSE filter as applied to the original data (x_mat)
#   -If the HT constraint of M-SSA matches the HT of this filter, then M-SSA replicates M-MSE
#   -M-SSA is a generalization of classic signal extraction algorithms, allowing for a control of the HT (smoothness or noise suppression)
gammak_x_mse<-MSSA_obj$gammak_x_mse
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-gammak_x_mse[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,gammak_x_mse[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("M-MSE applied to x ",colnames(x_mat)[i],sep=""),col=rainbow(n))
}
# If the imposed HT of M-SSA is larger than the intrinsic HT  of M-MSE, then the M-SSA filter will typically 
#   decay more slowly towards zero than the M-MSE benchmark (stronger smoothing)



# We now look at the acausal target
par(mfrow=c(1,1))
ts.plot(t(gamma_target),col=rainbow(n),main=c("Target as applied to original data", "Right tail is mirrored to the left to obtain two-sided filter","For each series, the target assigns weight to that series only"))
abline(v=(1:n*(nrow(t(gamma_target))/n)))

# But we can also look at the acausal target as applied to the MA-inversion of the VAR
# The following plot displays the target (two-sided HP filter) as applied to VAR-residuals 
#   (convolution of above target and MA-inversion xi)
gamma_target_long<-MSSA_obj$gammak_target
par(mfrow=c(1,1))
ts.plot(gamma_target_long,col=rainbow(n),main=c("Targets as assigned to MA-inversion of original data","The originally symmetric two-sided target appears less symmetric after convolution with MA-inversion","Weight can be assigned to multiple series if VAR is not diagonal"))
abline(v=(1:n*(nrow(gamma_target_long)/n)))
# Notes 
# 1. We show both sides of the filter (after `mirroring')
# 2. For a particular series (say the first one, i=1), the convolution of the original target and 
#     the MA-inversion xi may assign weight to (VAR-) residuals of other series in the presence of 
#     cross-correlation (non-diagonal VAR) 


# M-SSA also computes the (true) variance of the two-sided HP output (assuming no model misspecification)
var_target<-MSSA_obj$var_target
# We can compare variances on the diagonal of the matrix var_target with variances of M-MSE (below): 
#   -The latter are smaller (because the causal MSE predictor is missing future epsilons)
# For this purpose, we need the system matrices M_tilde and I_tilde, see M-SSA paper for background
#   -The system matrices are defined as Kronecker-products of Sigma (the variance covariance matrix of VAR residuals) 
#     with either the identity (giving I_tilde) or the lag-one autocovariance generating matrix M (giving M_tilde)
M_obj<-M_func(L,Sigma)
M_tilde<-M_obj$M_tilde
I_tilde<-M_obj$I_tilde

# Variance of target series: computed by M-SSA
diag(var_target)
# Variance of classic MSE predictors: formula based on M-SSA paper
diag(t(gammak_mse)%*%I_tilde%*%gammak_mse)
# Variance of M-SSA: in general variances of M-SSA <= variances MSE <= variances target (due to zero-shrinkage)
diag(t(bk_mat)%*%I_tilde%*%bk_mat)
# Compare true and sample estimates: the following two numbers match (sample converges to true for increasing sample length)
#   The m-th entry of the above vector should match the sample variance of the M-SSA output of the m-th series, see simulation above
diag(t(bk_mat)%*%I_tilde%*%bk_mat)[m]
var(na.exclude(y))
# Similarly, the m-th variance of the target should match the sample variance of zdelta, see above simulation experiment
diag(var_target)[m]
var(na.exclude(zdelta))
# Summary: we verified (some of) the theoretical expressions in the M-SSA paper: 
#   -Sample estimates of variances converge to expected values


# We can now compute the true lag one ACFs of the classic MSE nowcast (look at the M-SSA paper for technical details)
rho_mse<-gammak_mse[,1]%*%M_tilde%*%gammak_mse[,1]/gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]
for (i in 2:n)
  rho_mse<-c(rho_mse,gammak_mse[,i]%*%M_tilde%*%gammak_mse[,i]/gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i])
# Similarly, we can compute the true lag-one ACFs of M-SSA, see M-SSA paper
rho_ssa<-bk_mat[,1]%*%M_tilde%*%bk_mat[,1]/bk_mat[,1]%*%I_tilde%*%bk_mat[,1]
for (i in 2:n)
  rho_ssa<-c(rho_ssa,bk_mat[,i]%*%M_tilde%*%bk_mat[,i]/bk_mat[,i]%*%I_tilde%*%bk_mat[,i])
# We can also derive the HTs based on the above ACFs: HT and lag-one ACFs are linked bijectively (at least for Gaussian processes)
ht_comp<-apply(matrix(rho_ssa,nrow=1),1,compute_holding_time_from_rho_func)[[1]]$ht
ht_comp
# We have verified in the above simulation experiment that sample HTs match these `true' numbers`

# We can once again verify successful numerical optimization 
#  -If successful, then ht_comp (based on optimized filters) should match the imposed HTs ht_mssa_vec
#  -Increasing the size of split_grid (the number of iterations) tightens the fit between ht_comp above and ht_mssa_vec below
abs(ht_mssa_vec-ht_comp)
# We can also compute HTs of the classic M-MSE benchmark: 
#   -In general M-SSA is designed to be smoother (stronger noise suppression), i.e., ht_mssa_vec is larger than the below HTs of MSE design
#   -We can of course change the HT in the constraint as specified in the call to M-SSA
apply(matrix(rho_mse,nrow=1),1,compute_holding_time_from_rho_func)[[1]]$ht

# Next we can compute the target correlations
#   -As explained above, we here compute the correlation of M-SSA with M-MSE benchmark (instead of acausal target) 
#     -Targeting the two-sided filter is formally equivalent to targeting the classic M-MSE
#   -If HT of M-SSA matches HT of M-MSE, then M-SSA replicates the latter (in this case, the `target correlation' would be one)
#   -If the imposed HT of M-SSA is larger than HT of M-MSE, then the target correlation is smaller one 
#     -M-SSA maximizes this correlation subject to the HT constraint
# -See formula in M-SSA paper. In R-code we obtain:
crit_mse<-gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]/gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]
for (i in 2:n)
  crit_mse<-c(crit_mse,gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i]/gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i])
# The correlation of the MSE-benchmark with itself is trivially one
crit_mse
# M-SSA tries to maximize this correlation  subject to the HT constraint
crit_ssa<-gammak_mse[,1]%*%I_tilde%*%bk_mat[,1]/(sqrt(bk_mat[,1]%*%I_tilde%*%bk_mat[,1])*sqrt(gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]))
for (i in 2:n)
  crit_ssa<-c(crit_ssa,gammak_mse[,i]%*%I_tilde%*%bk_mat[,i]/(sqrt(bk_mat[,i]%*%I_tilde%*%bk_mat[,i])*sqrt(gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i])))
crit_ssa

criterion_mat<-rbind(crit_mse,crit_ssa)
colnames(criterion_mat)<-c(paste("Series ",1:n,paste=""))
rownames(criterion_mat)<-c("MSE","SSA")
# Correlations with classic MSE predictor
# M-SSA maximizes these numbers (equivalent objective function)
criterion_mat
# Compare the second row of this matrix with MSSA_obj$crit_rhoyz computed by M-SSA 
MSSA_obj$crit_rhoyz
#  crit_rhoyz is the objective function of the optimization criterion and is maximized by M-SSA

# We verified convergence of sample estimates in exercise 1 and we cross-checked some of the expressions in the M-SSA paper
#   -We shall complete this experiment in exercise 3 below
#   -In particular, we shall see that sample estimates of the (target-) correlations will converge 
#     towards the above numbers when targeting either the acausal filter or the causal M-MSE



##########################################################################################
# Summary:
# -M-SSA has a rich output with additional filters (including M-MSE) and additional performance metrics
# -Theoretical expressions (expected values: see M-SSA paper) match sample estimates (for sufficiently long samples)
# -M-SSA optimization concept: 
#   -We verified that M-SSA maximizes the target correlation conditional on the HT constraint
#   -The target correlation can be defined with respect to the effective acausal target or M-MSE: the M-SSA solution will be the same
#   -The target correlation ignores static level and scale adjustments but is otherwise equivalent to minimum MSE
# -M-SSA replicates classic M-MSE signal extraction filters (up to static level adjustment)  by 
#   inserting the HT of the latter into the M-SSA constraint
# -M-SSA can address backcasting (delta<0), nowcasting (delta=0) and forecasting (delta>0)
# -The target specification is generic: in the above experiment we relied on the two-sided HP
#   -Classic h-step ahead forecasting can be obtained by replacing the HP-filter by the identity (see univariate SSA tutorials on the topic)
#   -We could insert Hamilton or Baxter-King or Beveridge-Nelson specifications, see earlier tutorials
# -The data generating process (DGP) is assumed to be stationary (could be generalized); otherwise the specification is general
#   -In our applications we typically consider growth-dynamics, i.e., data in first differences (differenced data is close to stationarity)
#   -M-SSA relies on the Wold-decomposition of the (stationary) DGP which is straightforward to obtain for a VARMA process (MA-inversion)
# -A convergence of sample performances towards expected numbers assumes the model to be `true'
#   -We shall see that the application to German macro data (tutorials 7.2 and 7.3) is remarkably robust 
#     -against singular Pandemic data (outliers)
#     -against in-sample span for VAR: pre-financial crisis M-SSA (data up Jan-2007) performs nearly as well as full sample M-SSA
#     -against VARMA specification (as long as heavy overfitting is avoided)   
# -Tutorial 7.2 will demonstrate that the VAR(1) is (most likely) misspecified, as we might already suspect
#   -We then provide a simple and effective trick to overcome the misspecification in the context of this 
#     macro-application
###########################################################################################
# Exercise 3: to conclude this tutorial we wrap the above code into functions and we verify some additional 
#   convergence of sample estimates to theoretical performance numbers

# Wrappers: let's pack the above code into functions with distinct tasks
# 1. Target function
HP_target_sym_T<-function(n,lambda_HP,L)
{
  HP_obj<-HP_target_mse_modified_gap(L,lambda_HP)
  
  hp_symmetric=HP_obj$target
  hp_classic_concurrent=HP_obj$hp_trend
  hp_one_sided<-HP_obj$hp_mse
  # Target first series  
  gamma_target<-c(hp_one_sided,rep(0,(n-1)*L))
  # We now proceed to specifying the targets of the remaining n-1 series
  for (i in 2:n)
    gamma_target<-rbind(gamma_target,c(rep(0,(i-1)*L),hp_one_sided,rep(0,(n-i)*L)))
  # The above target filters are one-sided (right half of two-sided filter)
  # We now tell M-SSA that it has to mirror the above filters at their center points to obtain two-sided targets
  symmetric_target<-T
  return(list(gamma_target=gamma_target,symmetric_target=symmetric_target))
}  



# 2. MA-inversion as based on VAR model
MA_inv_VAR_func<-function(Phi,Theta,L,n,Plot=F)
{
  # MA inversion of VAR
  # MA inversion is used because the M-SSA optimization criterion relies an white noise
  #   For autocorrelated data, we thus require the MA-inversion of the DGP
  xi_psi<-PSIwgt(Phi = Phi, Theta = NULL, lag = L, plot = F, output = F)
  xi_p<-xi_psi$psi.weight
  # Transform Xi_p into Xi as structured/organized for M-SSA
  #   First L entries, from left to right, are weights of first explanatory series, next L entries are weights of second WN 
  xi<-matrix(nrow=n,ncol=n*L)
  for (i in 1:n)
  {
    for (j in 1:L)
      xi[,(i-1)*L+j]<-xi_p[,i+(j-1)*n]
  }
  if (Plot)
  {
    # Plot MA inversions  
    par(mfrow=c(1,n))
    for (i in 1:n)#i<-1
    {
      mplot<-xi[i,1:min(10,L)]
      
      for (j in 2:n)
      {
        mplot<-cbind(mplot,xi[i,(j-1)*L+1:min(10,L)])
        
      }
      ts.plot(mplot,col=rainbow(ncol(mplot)),main=paste("MA inversion ",colnames(x_mat)[i],sep=""))
    }
  }
  return(list(xi=xi))
}

# M-SSA
MSSA_main_func<-function(delta,ht_vec,xi,symmetric_target,gamma_target,Sigma,Plot=F)
{
  # Compute lag-one ACF corresponding to HT in M-SSA constraint: see previous tutorials on the link between HT and lag-one ACF  
  rho0<-compute_rho_from_ht(ht_vec)$rho
  
  # Some default settings for numerical optimization
  # with_negative_lambda==T allows the extend the search to unsmoothing (generate more zero-crossings than benchmark): 
  #   Default value is FALSE (smoothing only)
  with_negative_lambda<-F
  # Default setting for numerical optimization
  lower_limit_nu<-"rhomax"
  # Optimization with half-way triangulation: effective resolution is 2^split_grid. Much faster than brute-force grid-search.
  # 20 is a good value: fast and strong convergence in most applications
  split_grid<-20
  # M-SSA wants the target with rows=target-series and columns=lags: for this purpose we here transpose the filter  
  gamma_target<-t(gamma_target)
  
  # Now we can apply M-SSA
  MSSA_obj<-MSSA_func(split_grid,L,delta,grid_size,gamma_target,rho0,with_negative_lambda,xi,lower_limit_nu,Sigma,symmetric_target)
  
  # In principle we could retrieve filters, apply to data and check performances
  # But M-SSA delivers a much richer output, containing different filters and useful evaluation metrics
  # These will be analyzed further down
  # So let's pick out the real-time filter
  bk_x_mat<-MSSA_obj$bk_x_mat
  if (Plot)
  {
    par(mfrow=c(1,n))
    for (i in 1:n)# i<-1
    {
      mplot<-bk_x_mat[1:L,i]
      for (j in 2:n)
      {
        mplot<-cbind(mplot,bk_x_mat[(j-1)*L+1:L,i])
      }
      ts.plot(mplot,main=paste("MSSA applied to x ",colnames(x_mat)[i],sep=""),col=rainbow(n))
    }
  }
  # We return the M-SSA filter as well as the whole M-SSA object which hides additional useful objects  
  return(list(bk_x_mat=bk_x_mat,MSSA_obj=MSSA_obj))
}


# 4. Filter function: apply M-SSA filter to data and compute target
filter_func<-function(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
{
  len<-nrow(x_mat)
  n<-dim(bk_x_mat)[2]
  # Compute M-SSA filter output 
  mssa_mat<-mmse_mat<-target_mat<-NULL
  for (m in 1:n)
  {
    bk<-NULL
    # Extract coefficients applied to m-th series    
    for (j in 1:n)#j<-2
      bk<-cbind(bk,bk_x_mat[((j-1)*L+1):(j*L),m])
    y<-rep(NA,len)
    for (j in L:len)#j<-L
    {
      y[j]<-sum(apply(bk*(x_mat[j:(j-L+1),]),2,sum))
    }
    mssa_mat<-cbind(mssa_mat,y)
  }  
  # Compute M-MSE: classic MSE signal extraction design 
  for (m in 1:n)
  {
    gamma_mse<-NULL
    # Extract coefficients applied to m-th series    
    for (j in 1:n)#j<-2
      gamma_mse<-cbind(gamma_mse,gammak_x_mse[((j-1)*L+1):(j*L),m])
    ymse<-rep(NA,len)
    for (j in L:len)#j<-L
    {
      ymse[j]<-sum(apply(gamma_mse*(x_mat[j:(j-L+1),]),2,sum))
    }
    mmse_mat<-cbind(mmse_mat,ymse)
  }  
  # Apply target to m-th-series
  target_mat<-NULL
  for (m in 1:n)#
  {
    # In general, m-th target is based on j=1,...,n filters applied to explanatory variables j=1,...,n
    gammak<-NULL
    for (j in 1:n)
    {
      # Retrieve j-th filter for m-th target       
      gammak<-cbind(gammak,gamma_target[(j-1)*L+1:L,m])
    }
    z<-rep(NA,len)
    if (symmetric_target)
    {
      # Here the right half of the filter is mirrored to the left at its peak
      # Moreover, the data is shifted by delta
      for (j in (L-delta):(len-L-delta+1))#j<-L-delta
        z[j]<-sum(apply(gammak*x_mat[delta+j:(j-L+1),],2,sum))+sum(apply(gammak[-1,]*x_mat[delta+(j+1):(j+L-1),],2,sum))
    } else
    {
      # Data shifted by delta: we do not mirror filter weights      
      for (j in (L-delta):(len-delta))
      {
        z[j]<-sum(apply(gammak*(x_mat[delta+j:(j-L+1),]),2,sum))
      }
    }
    
    names(z)<-names(y)<-rownames(x_mat)
    target_mat<-cbind(target_mat,z)
  } 
  colnames(mssa_mat)<-colnames(mmse_mat)<-colnames(target_mat)<-colnames(x_mat)
  return(list(mssa_mat=mssa_mat,target_mat=target_mat,mmse_mat=mmse_mat))
}

#------------------------------------------------------------------------
# Let's now apply the above functions to the previous simulation experiment

# 1. Target
lambda_HP<-160
# Filter length: roughly 4 years. The length should be an odd number, see above comments (mirroring)
L<-31

target_obj<-HP_target_sym_T(n,lambda_HP,L)

gamma_target=t(target_obj$gamma_target)
symmetric_target=target_obj$symmetric_target 

# Target as applied to original data (not MA-inversion)
# To obtain the two-sided filter, the right tail will be mirrored to the left, about the center point
par(mfrow=c(1,1))
ts.plot(gamma_target,col=rainbow(n),main="Target as applied to original data: right tail will be mirrored to the left to obtain two-sided HP")
abline(v=1+(1:n*(nrow((gamma_target))/n)))

# If the boolean symmetric_target is true, the right tail of the (one-sided) target will be mirrored to 
#   the left of the center point to obtain the two-sided design
symmetric_target

# 2. MA-inversion as based on VAR model

MA_inv_obj<-MA_inv_VAR_func(Phi,Theta,L,n,T)

xi<-MA_inv_obj$xi

# 3. M-SSA function
# Nowcast
delta<-0
# One year ahead forecast for quarterly data
delta<-4

MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)

MSSA_main_obj$bk_x_mat=bk_x_mat
MSSA_obj=MSSA_main_obj$MSSA_obj 
# Benchmark MSE predictor
gammak_x_mse<-MSSA_obj$gammak_x_mse

# 4. Filter function: apply M-SSA filter to data
# For long samples the execution may require some patience (because of the for-loops which take time to process in R)

filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)

# retrieve M-SSA and acausal target
mssa_mat=filt_obj$mssa_mat
target_mat=filt_obj$target_mat
mmse_mat<-filt_obj$mmse_mat

#------------------------
# Checks: the obtained output should be identical to previous y and zdelta for series m_check: differences should vanish
max(abs(y-mssa_mat[,m_check]),na.rm=T)
max(abs(zdelta-target_mat[,m_check]),na.rm=T)

# Mean-square errors
apply(na.exclude((target_mat-mssa_mat)^2),2,mean)

# We can verify that the sample target correlations between effective acausal target and M-SSA converge towards 
#   expectations for increasing sample length
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[,i],mssa_mat[,i])))[1,2])
# Sample estimates should be close to true values:
MSSA_obj$crit_rhoy_target


# Similarly, we can verify that the sample correlations between M-MSE and M-SSA converge to criterion values
#   for increasing sample size len
for (i in 1:n)
  print(cor(na.exclude(cbind(mmse_mat[,i],mssa_mat[,i])))[1,2])
# Sample estimates should be close to true values (objective function of M-SSA):
MSSA_obj$crit_rhoyz



# M-SSA optimizes the target correlation (between M-SSA and M-MSE) under the holding time constraint:
# We can compare sample and expected (imposed) HTs:
apply(mssa_mat,2,compute_empirical_ht_func)
ht_mssa_vec



# The above functions can also be sourced
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))
# We shall rely on these functions in tutorials 7.2 and 7.3


