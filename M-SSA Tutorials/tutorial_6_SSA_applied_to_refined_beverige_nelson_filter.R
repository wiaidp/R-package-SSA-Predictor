# We here apply SSA to the refined Beveridge Nelson filter proposed in
# Gunes Kamber & James Morley & Benjamin Wong, 2024. "Trend-Cycle Decomposition in the Presence of Large Shocks," CAMA Working Papers 2024-24, Centre for Applied Macroeconomic Analysis, Crawford School of Public Policy, The Australian National University, revised Aug 2024.

# For this purpose we rely on their R-code at https://drive.google.com/file/d/15P2OOV7aPl8qcwAsvSltaO2QfmuxID_N/view?pli=1
# Alternative web-side for codes: https://bnfiltering.com/

# Here is the description of the R-code by the authors:


#This code calculates the BN filter output gap with the automatic signal-to-noise selection criteria described in Kamber, Morley and Wong (2018) (KMW2018 hereafter), "Intuitive and Reliable Estimates of the Output Gap from a Beveridge-Nelson Filter," Review of Economics and Statistics 100 (3), 550-566 https://doi.org/10.1162/rest_a_00691

#It has also been updated to allow refinements of the original BN filter, as described in Kamber, Morley and Wong (2024) (KMW2024 hereafter), "Trend-Cycle Decomposition in the Presence of Large Shocks" (https://ideas.repec.org/p/een/camaaa/2024-24.html) and in further detail below.

#bnf_run.R is the main file, with data input and various choices about estimation to be made in this file. bnf_fcn.R contains all of the functions called from the main file.

#To allow for possible structural breaks in the long-run drift of a time series, we include options to implement dynamic demeaning as described in KMW2018 or to enter breakdate(s) informed by a test such as Bai-Perron.

#We also allow the possibility of imposing no drift in levels, such as might be case for variables like the unemployment rate or inflation.

#To implement the BN filter, one needs four inputs: The data in first differences of the series being detrended, the lag order of the restricted AR model used in estimation, an indicator of whether or not iterative backcasting is employed, and a signal-to-noise ratio delta.

#The code was modified in October 2021 to allow calculation of error bands according to the formula in the online appendix of BMW2018 (the online appendix is also available at https://doi.org/10.1162/rest_a_00691).

#As proposed in KMW2024, we have modified the original code to allow four refinements relative to KMW2018:
#  1) an alternative automatic selection of delta based on the local minimum of the variance of trend shocks rather than local maximum of the amplitude-to-noise ratio
#2) iterative dynamic mean adjustment that uses estimates of trend instead of overall growth to avoid undue influence of outlier cyclical observations
#3) dynamic estimation of BN cycle variance using the same window as dynamic demeaning for purposes of constructing more accurate 95% error bands
#4) an option of iterative backcasting (until parameter estimates/backcasts converge) that uses the reversibility of the restricted AR process to backcast output growth prior to the initial observation, allowing for calculation of the BN filter cycle for the first observation in levels instead of from the second observation in levels  

#We are not responsible for any loss you may incur, financial or otherwise, by using our code
#If you use the code, please cite and acknowledge the paper.

#Gunes Kamber
#James Morley
#Benjamin Wong
#April 2024

#------------------------------------------------------------
# Let's start with the main function in the provided R-code

####################################################################################################
# Example using US data to illustrate how to use the various 'bnf' methods 
####################################################################################################

# Clear the workspace -- comment out if you have loaded other packages already
rm(list = ls(all = TRUE))
gc()

# Source required functions
source(paste(getwd(),"/R/bnf_fcns.R",sep=""))

# Read in US centric data to use for the demonstration
#usdata <- read.csv(file = 'us_data.csv', header = TRUE, stringsAsFactors = FALSE)
usdata <- read.csv(file = paste(getwd(),'/Data/USGDP_updated.csv',sep=""), header = FALSE, stringsAsFactors = FALSE)

# Make the series GDP a 'ts' object
# type 'help(ts)' in the R console for more information
#gdp <- ts(data = usdata$GDPC1, end = c(2016, 2), frequency = 4)
gdp <- ts(data = usdata, end = c(2023, 2), frequency = 4)

#Take logs and multiply by 100
y <- transform_series(y = gdp, take_log = TRUE, pcode = "p1") # same as: log(raw_y) * 100.0

# Example: Automatically determined delta and full sample mean demeaning method
cat("Example: log US real GDP\n\n")

p <- 12					#Default lag order for AR(p) model. Set to large value to allow for low value of delta

ib <- 1                        #Set to 0 if no iterative backcasting as in KMW2018 (just unconditional mean), set to 1 if iterative backcasting
iterative <- 100			   #Set to >1 for max number of iterations for iterative dynamic demeaning
window <- 40			   #Rolling window length for dynamic demeaning and/or dynamic error bands (e.g., 40 is 10 years for quarterly data)

delta_select <- 2		#set to 0 if use fixed delta, 1 if max amp-to-noise, 2 if min var(trend shocks)
fixed_delta <- 0.01 #set a fixed delta to be used if delta_select=0
d0 = 0.005			#lowest value considered
dt = 0.0005			#increments for grid of delta

demean <- "nd"			#assume no drift, applies if iterative set to <0
dynamic_bands <- 0		#fixed standard error bands

if (iterative == 0) {	
  demean <- "sm"				#use sample mean for drift			
} else if (iterative > 0){
  demean <- "dm"				#dynamic demeaning
  dynamic_bands <- 1			#dynamic error bands		
}	

#uncomment next two lines to allow for structural breaks in mean at set breakdates, with example dates on second line
#demean <- "pm"
#breaks <- c(50, 75, 100, 125, 150)
if (demean == "pm"){iterative <- 0}		#override choice to dynamically demean if structural breaks allowed for

#uncomment next line to override choice of error bands based on dynamic demeaning
#dynamic_bands <- 1			#set to 1 for dynamic error bands, 0 for fixed standard error bands		

bnf <- bnf(y)

plot(bnf, main = "US Output Gap", col = "red", plot_ci = TRUE)
cat("\nPrinting out cycle data...\n")
print(bnf) # comment this command to stop the cycle data being printed to the console
cat('\n')

##############################################################################################
#------------------------------------------------------------------------------------------
# From now on we work with the above code
# 1. First, we want to specify and replicate the filter of the refined procedure proposed by the authors
# The composite filter is based on modifications of the original BN-filter and additional refinements
# While it is possible to replicate the filter based on the cited literature, an easier proceeding consists in obtaining the filter weights directly from the above filter output, by inversion 

# 1.1: combine original and filtered data
mat<-cbind(y,bnf$cycle)
tail(mat)

# 1.2 Define `regression' matrix: lagged observations in columns
len<-length(y)
L<-as.integer(len/2)
reg_mat<-y[len:(len-L+1)]
for (i in 1:(L-1))
  reg_mat<-cbind(reg_mat,y[-i+len:(len-L+1)])
dim(reg_mat)

# 1.3 Obtain filter weights by applying the inverted regression matrix to the filtered output
# The acronym rbn refers to refined Beveridge-Nelson
rbn<-as.double(solve(reg_mat)%*%bnf$cycle[len:(len-L+1)])

ts.plot(rbn,main="Refined BN filter: default settings of 2024 CAMA paper")

# 1.4 Check: apply obtained filter to data and verify replication of original filter output
y_check<-rep(NA,len)
for (i in L:len)
  y_check[i]<-rbn%*%y[i:(i-L+1)]

ts.plot(cbind(y_check,bnf$cycle),main="Replication successful: both series overlap")

# Replication successful
tail(cbind(y_check,bnf$cycle))

#----------------------------------------------------------------------------
# 2. Analyze the obtained filter in the frequency-domain: compute amplitude and shift

amp_shift_func<-function(K,b,plot_T)
{
  #  if (sum(b)<0)
  #  {
  #    print("Sign of coefficients has been changed")
  #    b<-b*sign(sum(b))
  #  }
  omega_k<-(0:K)*pi/K
  trffkt<-0:K
  for (i in 0:K)
  {
    trffkt[i+1]<-b%*%exp(1.i*omega_k[i+1]*(0:(length(b)-1)))
  }
  amp<-abs(trffkt)
  shift<-Arg(trffkt)/omega_k
  shift[1]<-sum((0:(length(b)-1))*b)/sum(b)
  if (plot_T)
  {
    par(mfrow=c(2,1))
    plot(amp,type="l",axes=F,xlab="Frequency",ylab="Amplitude",main="Amplitude")
    axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
    axis(2)
    box()
    plot(shift,type="l",axes=F,xlab="Frequency",ylab="Shift",main="Shift",ylim=c(min(min(shift,na.rm=T),0),max(shift,na.rm=T)))
    axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
    axis(2)
    box()
  }  
  return(list(trffkt=trffkt,amp=amp,shift=shift))
}

# 2.1 Analyze original refined BN
K<-600

amp_shift_obj<-amp_shift_func(K,rbn,F)

amp<-amp_shift_obj$amp
shift<-amp_shift_obj$shift

par(mfrow=c(1,2))
plot(amp,type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude Default Settings BN 2024 paper")
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()
plot(shift,type="l",axes=F,xlab="Frequency",ylab="Shift",main="Shift",ylim=c(-3,5))
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
abline(h=0)
box()

# 2.2 Transform filter in first differences: the transformed filter replicates the original one when applied to differenced data


conv_with_unitroot_func<-function(filt)
{
  conv<-filt
  L<-length(filt)
  for (i in 1:L)
  {
    conv[i]<-sum(filt[1:i])
  }  
  return(list(conv=conv))
}

rbn_d<-conv_with_unitroot_func(rbn)$conv

# Check: verify that output of transformed and original filters match
# New filter_d is applied to differenced data
y_d<-rep(NA,len)
for (i in (L+1):len)
  y_d[i]<-rbn_d%*%diff(y)[-1+i:(i-L+1)]

# Check
ts.plot(cbind(y_d,bnf$cycle),main="Both outputs overlap")

# 2.3 Compute amplitude and shift of transformed filter

amp_shift_obj<-amp_shift_func(K,rbn_d,F)

amp<-amp_shift_obj$amp
shift<-amp_shift_obj$shift

par(mfrow=c(1,2))

plot(amp,type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude filter as applied to diff")
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()
plot(shift,type="l",axes=F,xlab="Frequency",ylab="Shift",main="Shift",ylim=c(-3,5))
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
abline(h=0)
box()

# Peak of amplitude
which(amp==max(amp))
# Periodicity of corresponding frequency: roughly 17 years (data quarterly)
2*K/(which(amp==max(amp))-1)

#----------------------------------------------------------
# 3 Apply the  filter to a random-walk

# 3.1 Original filter (in level)
set.seed(123)
len_rw<-1200
x<-cumsum(rnorm(len_rw))
par(mfrow=c(1,1))
ts.plot(x,main="Random-walk")

# Apply filter
output<-rep(NA,len_rw)
for (i in L:len_rw)
  output[i]<-rbn%*%x[i:(i-L+1)]

# The filter generates a spurious cycle
ts.plot(output,main="Spurious cycle")


# 3.2 Apply differenced filter to noise

set.seed(123)
len_rw<-1200
x<-(rnorm(len_rw))
par(mfrow=c(1,1))
ts.plot(x,main="Noise")

# Apply filter: output is the same as above (up to negligible finite sample convolution error)
output<-rep(NA,len_rw)
for (i in L:len_rw)
  output[i]<-rbn_d%*%x[i:(i-L+1)]


# The filter generates a spurious cycle
ts.plot(output,main="Spurious cycle")


# Add slowly changing deterministic level to noise
omega<-2*pi/len_rw
level<-cos((1:len_rw)*omega)

ts.plot(x+level)

# Filter data
output<-rep(NA,len_rw)
for (i in L:len_rw)
  output[i]<-rbn_d%*%(x+level)[i:(i-L+1)]


# The filter cannot track salient feature (changing level: recessions/expansions)
ts.plot(output,main="Bandpass cannot track changing level")


#------------------------------------------------------------------------
# 4. Define an equivalent trend filter:
# 4.1 rbn is a `gap` filter much like HP_gap
# To obtain the equivalent trend filter we just use 1-rbn

rbn_trend<-c(1,rep(0,length(rbn)-1))-rbn

# Coefficients add to one
sum(rbn_trend)

# Looks `strange' (overfitting?)
ts.plot(rbn_trend,main="rbn trend filter")

#---------------
# 4.2 Analyze in frequency-domain
amp_shift_obj<-amp_shift_func(K,rbn_trend,F)

amp<-amp_shift_obj$amp
shift<-amp_shift_obj$shift

par(mfrow=c(1,2))
# Amplitude is unsmooth: overfitting?
# Peak amplitude matches business-cycle frequencies
plot(amp,type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude filter rbn trend")
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()
# Like classic concurrent HP the shift vanishes at frequency zero (rbn cancels a double unit-root)
plot(shift,type="l",axes=F,xlab="Frequency",ylab="Shift",main="Shift",ylim=c(-3,5))
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
abline(h=0)
box()

#---------------------------------------------------------------------
# 5. Compare rbn with HP-gap and rbn_trend with HP_trend

library(mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))
# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))

# 5.1 Compute HP(1600): quarterly data
# Same length as rbn
L<-length(rbn)
# Should be an odd number
if (L/2==as.integer(L/2))
{
  print("Filter length should be an odd number")
  print("If L is even then HP cannot be adequately centered")
  L<-L+1
}  
# Specify lambda: querterly design to match rbn filter
lambda_monthly<-1600
par(mfrow=c(1,1))
# This function relies on mFilter and it derives additional HP-designs to be discussed further down
HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)

hp_trend<-HP_obj$hp_trend
hp_gap<-HP_obj$hp_gap

colo<-c("black","red")
par(mfrow=c(1,2))
# Gap filters look similar
ts.plot(cbind(rbn,hp_gap),main="Original gap filters",col=colo)
# Trend filters are a bit different: strange negative lob of rbn_trend towards lag 40 (10 years)
ts.plot(cbind(rbn_trend,hp_trend),main="Trend filters",col=colo)

#------------------------
# 5.2 Compare amplitude functions of trend filters

amp_shift_obj<-amp_shift_func(K,rbn_trend,F)

amp_rbn<-amp_shift_obj$amp
shift_rbn<-amp_shift_obj$shift

amp_shift_obj<-amp_shift_func(K,hp_trend,F)

amp_hp<-amp_shift_obj$amp
shift_hp<-amp_shift_obj$shift


par(mfrow=c(1,2))
# Amplitude rbn_trend is unsmooth (overfitting) and stronger noise suppression than hp_trend
plot(amp_rbn,col=colo[1],type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude trend filters",ylim=c(0,1.3))
lines(amp_hp,col=colo[2])
abline(h=0)
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()
# Larger shift of rbn_trend due to stronger noise suppression: HP is faster but noisier
plot(shift_rbn,col=colo[1],type="l",axes=F,xlab="Frequency",ylab="",main="Shift trend filters")
lines(shift_hp,col=colo[2])
abline(h=0)
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()

