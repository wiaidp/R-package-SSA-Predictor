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


# Load the relevant M-SSA functionalities
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

#-----------------------------------------------
# Specify data generating process: this model is obtained by fitting a VAR(1) to quarterly German macro data
# The VAR is based on data up to Jan 2007 to demonstrate out-of-sample prioperties of the M-SSA predictor on 
#   a long out-of-sample span including the financial crisis as well as all subsequent `never ending' sequence of crises
# It is a 5-dimensional design comprising BIP (i.e. GDP), industrial production, economic sentiment, spread and an ifo indicator
#   -All series are log-transformed (except spread) and differenced (no cointegration)
# Since the series are relatively short (introduction of EURO up to Jan-2007) the VAR is sparsely parametrized
n<-5
# AR order
p_mult<-1
# AR(1) coefficient
Phi<-matrix(c( 0.0000000,  0.00000000, 0.4481816,    0, 0.0000000,
               0.2387036, -0.33015450, 0.5487510,    0, 0.0000000,
               0.0000000,  0.00000000, 0.4546929,    0, 0.3371898,
               0.0000000,  0.07804158, 0.4470288,    0, 0.3276132,
               0.0000000,  0.00000000, 0.0000000,    0, 0.3583553),nrow=n)

# covariance matrix
Sigma<-matrix(c(0.755535544,  0.49500481, 0.11051024, 0.007546104, -0.16687913,
  0.495004806,  0.65832962, 0.07810020, 0.025101191, -0.25578971,
  0.110510236,  0.07810020, 0.66385111, 0.502140497,  0.08539719,
  0.007546104,  0.02510119, 0.50214050, 0.639843288,  0.05908741,
 -0.166879134, -0.25578971, 0.08539719, 0.059087406,  0.84463448),nrow=n)

# Simulate a very long series    
len<-101
set.seed(31)
x_mat=VARMAsim(len,arlags=c(p_mult),phi=Phi,sigma=Sigma)$series
# Have a look at the data
# We expect to see mutually (cross-section) correlated and (longitudinal) autocorrelated series
# Can you glimpse `crises`, i.e., phases with strong negative growth?
ts.plot(x_mat[(len-100):100,])

#-------------------------------------------------------------------


