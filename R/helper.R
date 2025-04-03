
compute_calibrated_out_of_sample_predictors_func<-function(dat,date_to_fit,use_garch,shift)
{
  
  len<-dim(dat)[1]
# First column is target, i.e. dimension is dim(dat)[2]-1
  n<-dim(dat)[2]-1
# Compute calibrated out-of-sample predictor, based on expanding window
#   -Use data up i for fitting the regression
#   -Compute a prediction with explanatory data in i+1
  cal_oos_pred<-cal_oos_mean_pred<-rep(NA,len)
  for (i in (n+2):(len-shift)) #i<-n+2
  {
# If use_garch==T then the regression relies on weighted least-squares, whereby the weights are based 
#   on volatility obtained from a GARCH(1,1) model fitted to target (first column of dat)
    if (use_garch)
    {
      y.garch_11<-garchFit(~garch(1,1),data=dat[1:i,1],include.mean=T,trace=F)
      sigmat<-y.garch_11@sigma.t
# Weights are proportional to 1/sigmat^2      
      weight<-1/sigmat^2
    } else
    {
# Fixed weight      
      weight<-rep(1,i)
    }
# 1. Use predictors in columns 2,..., of dat    
# Fit model with data up to time point i; weighted least-squares relying on weight as defined above  
    lm_obj<-lm(dat[1:i,1]~dat[1:i,2:(n+1)],weight=weight)
    summary(lm_obj)
# Compute out-of-sample prediction for time point i+shift
    if (n==1)
    {
# Only one predictor      
#   Classic regression prediction        
      cal_oos_pred[i+shift]<-(lm_obj$coef[1]+lm_obj$coef[2]*dat[i+shift,2])
    } else
    {
# Multiple predictors      
#  We use %*% instead of * above      
      cal_oos_pred[i+shift]<-(lm_obj$coef[1]+lm_obj$coef[2:(n+1)]%*%dat[i+shift,2:(n+1)]) 
    }
# 2. Use mean as predictor (simplest benchmark)
    cal_oos_mean_pred[i+shift]<-mean(dat[1:i,1])
  }
# Once the predictors are computed we can obtain the out-of-sample prediction errors
  epsilon_oos<-dat[,1]-cal_oos_pred
  index_oos<-which(rownames(dat)>date_to_fit)
# And we can compute the HAC-adjusted p-values of the regression of the predictor on the target, out-of-sample  
  lm_oos<-lm(dat[index_oos,1]~cal_oos_pred[index_oos])
  ts.plot(cbind(dat[index_oos,1],cal_oos_pred[index_oos]),main=paste("shift=",shift,sep=""))
  summary(lm_oos)
  sd_HAC<-sqrt(diag(vcovHAC(lm_oos)))
  sd_ols<-sqrt(diag(vcov(lm_oos)))
# Compute max of both vola estimates: HAC-adjustment is not 100% reliable (maybe issue with R-package sandwich)  
  sd_max<-max(sd_ols[2],sd_HAC[2])
  sd_max<-sd_HAC[2]
  t_HAC<-summary(lm_oos)$coef[2,1]/sd_max
# One-sided test: if predictor is effective, then the sign of the coefficient must be positive (ngetaive signs can be ignored) 
  p_value<-pt(t_HAC, nrow(dat)-2, lower=FALSE)
# Out-of-sample MSE of predictor  
  MSE_oos<-mean(epsilon_oos[index_oos]^2)
# Same but for benchmark mean predictor  
  epsilon_mean_oos<-dat[,1]-cal_oos_mean_pred
  MSE_mean_oos<-mean(epsilon_mean_oos[index_oos]^2)
  
  return(list(cal_oos_pred=cal_oos_pred,epsilon_oos=epsilon_oos,p_value=p_value,MSE_oos=MSE_oos,MSE_mean_oos=MSE_mean_oos))
}


library(fGarch)

select_vec_multi
sel_vec_pred<-select_vec_multi[2]
sel_vec_pred<-select_vec_multi
sel_vec_pred<-select_vec_multi[c(1,2)]
date_to_fit<-"2007"
use_garch<-T

p_mat_mssa<-p_mat_mssa_components<-p_mat_direct<-rRMSE_mSSA_comp_direct<-rRMSE_mSSA_comp_mean<-matrix(ncol=length(h_vec),nrow=length(h_vec))
for (shift in h_vec)#shift<-3
{
  print(shift)
  
  for (j in h_vec)
  {
  
    k<-j+1

# M-SSA  
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),predictor_mssa_mat[,k])
    rownames(dat)<-rownames(x_mat)
    dat<-na.exclude(dat)
  
    perf_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit,use_garch,shift)
  
    p_mat_mssa[shift+1,k]<-perf_obj$p_value 
    MSE_oos_mssa<-perf_obj$MSE_oos
  
# M-SSA components
    if (length(sel_vec_pred)>1)
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,k]))
    } else
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mssa_array[sel_vec_pred,,k]))
    }
  
    rownames(dat)<-rownames(x_mat)
    dat<-na.exclude(dat)
  
    perf_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit,use_garch,shift)
  
    p_mat_mssa_components[shift+1,k]<-perf_obj$p_value 
    MSE_oos_mssa_comp<-perf_obj$MSE_oos
    MSE_mean_oos<-perf_obj$MSE_mean_oos
  
# Direct
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),x_mat[,sel_vec_pred])
    rownames(dat)<-rownames(x_mat)
    dat<-na.exclude(dat)
    
    perf_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit,use_garch,shift)
    
    p_mat_direct[shift+1,k]<-perf_obj$p_value 
    MSE_oos_direct<-perf_obj$MSE_oos
    
    rRMSE_mSSA_comp_direct[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_oos_direct)
    rRMSE_mSSA_comp_mean[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_mean_oos)
  }
}

colnames(p_mat_mssa)<-colnames(p_mat_mssa_components)<-colnames(p_mat_direct)<-
  colnames(rRMSE_mSSA_comp_direct)<-colnames(rRMSE_mSSA_comp_mean)<-paste("h=",h_vec,sep="")
rownames(p_mat_mssa)<-rownames(p_mat_mssa_components)<-rownames(p_mat_direct)<-
  rownames(rRMSE_mSSA_comp_direct)<-rownames(rRMSE_mSSA_comp_mean)<-paste("Shift=",h_vec,sep="")

p_mat_mssa
p_mat_mssa_components
p_mat_direct
rRMSE_mSSA_comp_mean
rRMSE_mSSA_comp_direct

ts.plot(scale(dat),col=c("black",rainbow(ncol(dat)-1)))


