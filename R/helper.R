
compute_calibrated_out_of_sample_predictors_func<-function(dat,date_to_fit,use_garch,shift)
{
  
  len<-dim(dat)[1]
  # First column is target, i.e. dimension is dim(dat)[2]-1
  n<-dim(dat)[2]-1
  # Compute calibrated out-of-sample predictor, based on expanding window
  #   -Use data up i for fitting the regression
  #   -Compute a prediction with explanatory data in i+1
  cal_oos_pred<-rep(NA,len)
  for (i in (n+2):(len-shift)) #i<-n+2
  {
    if (use_garch)
    {
      y.garch_11<-garchFit(~garch(1,1),data=dat[1:i,1],include.mean=T,trace=F)
      sigma<-y.garch_11@sigma.t
      weight<-1/sigma^2
    } else
    {
      weight<-rep(1,i)
    }
    # Fit model with data up to time point i; weighted least-squares using GARCH vola   
    lm_obj<-lm(dat[1:i,1]~dat[1:i,2:(n+1)],weight=weight)
    summary(lm_obj)
    # Compute out-of-sample prediction for i+1
    # Distinguish only one from multiple explanatory variables (R-code different...)    
    if (n==1)
    {
      # Classic regression prediction        
      cal_oos_pred[i+shift]<-lm_obj$coef[1]+lm_obj$coef[2]*dat[i+shift,2] 
    } else
    {
      # Classic regression prediction though we use %*% instead of * above      
      cal_oos_pred[i+shift]<-lm_obj$coef[1]+lm_obj$coef[2:(n+1)]%*%dat[i+shift,2:(n+1)] 
    }
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
  if (F)
  {
    # Classic OLS p-values
    sd_ols<-sqrt(diag(vcov(lm_oos)))
    t_ols<-summary(lm_oos)$coef[2,1]/sd_ols[2]
    OLS_p_value<-pt(t_ols, nrow(dat)-2, lower=FALSE)
  }
  MSE_oos<-mean(epsilon_oos[index_oos]^2)
  
  return(list(cal_oos_pred=cal_oos_pred,epsilon_oos=epsilon_oos,p_value=p_value,MSE_oos=MSE_oos))
}

library(fGarch)

select_vec_multi
sel_vec_pred<-select_vec_multi[1:5]
sel_vec_pred<-select_vec_multi[c(1,2)]
date_to_fit<-"2007"
use_garch<-T

rRMSE_vec<-NULL
p_mat<-matrix(ncol=3,nrow=6)
for (shift in 1:6)#shift<-4
{
  print(shift)
  k<-shift+1

# M-SSA  
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),predictor_mssa_mat[,k])
  rownames(dat)<-rownames(x_mat)
  dat<-na.exclude(dat)
  
  perf_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit,use_garch,shift)
  
  p_mat[shift,1]<-perf_obj$p_value 
  MSE_oos_mssa<-perf_obj$MSE_oos
  
# M-SSA components
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,k]))
  rownames(dat)<-rownames(x_mat)
  dat<-na.exclude(dat)
  
  perf_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit,use_garch,shift)
  
  p_mat[shift,2]<-perf_obj$p_value 
  MSE_oos_mssa_comp<-perf_obj$MSE_oos
  
  
# Direct
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),x_mat[,sel_vec_pred])
  rownames(dat)<-rownames(x_mat)
  dat<-na.exclude(dat)
  
  perf_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit,use_garch,shift)
  
  p_mat[shift,3]<-perf_obj$p_value 
  MSE_oos_direct<-perf_obj$MSE_oos
  
  rRMSE_vec<-c(rRMSE_vec,sqrt(MSE_oos_mssa_comp/MSE_oos_direct))
}

colnames(p_mat)<-c("M-SSA","M-SSA components","Direct")
rownames(p_mat)<-paste("shift=",1:6,sep="")

p_mat
rRMSE_vec


