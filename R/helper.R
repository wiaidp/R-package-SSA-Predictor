
compute_calibrated_out_of_sample_predictors_func<-function(dat,start_fit,use_garch,shift)
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
  index_oos<-which(rownames(dat)>start_fit)
# And we can compute the HAC-adjusted p-values of the regression of the predictor on the target, out-of-sample  
  lm_oos<-lm(dat[index_oos,1]~cal_oos_pred[index_oos])
  ts.plot(cbind(dat[index_oos,1],cal_oos_pred[index_oos]),main=paste("shift=",shift,sep=""))
  summary(lm_oos)
  sd_HAC<-sqrt(diag(vcovHAC(lm_oos)))
  t_HAC<-summary(lm_oos)$coef[2,1]/sd_HAC[2]
# One-sided test: if predictor is effective, then the sign of the coefficient must be positive (negative signs can be ignored) 
  p_value<-pt(t_HAC, nrow(dat)-2, lower=FALSE)
# Out-of-sample MSE of predictor  
  MSE_oos<-mean(epsilon_oos[index_oos]^2)
# Same but for benchmark mean predictor  
  epsilon_mean_oos<-dat[,1]-cal_oos_mean_pred
  MSE_mean_oos<-mean(epsilon_mean_oos[index_oos]^2)
# The same as above but without Pandemic: check that Pandemic is within data span
  if ((sum(rownames(dat)>2019)>0)&(sum(rownames(dat)<2019)>0))
  {
# Specify out-of-sample span without COVID    
    index_oos_without_covid<-index_oos[rownames(dat)[index_oos]<2019]
# Check
    rownames(dat)[index_oos_without_covid]
# Compute HAC adjusted p-value    
    lm_oos<-lm(dat[index_oos_without_covid,1]~cal_oos_pred[index_oos_without_covid])
    summary(lm_oos)
    sd_HAC<-sqrt(diag(vcovHAC(lm_oos)))
    t_HAC<-summary(lm_oos)$coef[2,1]/sd_HAC[2]
# One-sided test: if predictor is effective, then the sign of the coefficient must be positive (negtaive signs can be ignored) 
    p_value_without_covid<-pt(t_HAC, nrow(dat)-2, lower=FALSE)
# Compute MSE: predictor and mean   
    MSE_oos_without_covid<-mean(epsilon_oos[index_oos_without_covid]^2)
    MSE_mean_oos_without_covid<-mean(epsilon_mean_oos[index_oos_without_covid]^2)
  }
  
  return(list(cal_oos_pred=cal_oos_pred,epsilon_oos=epsilon_oos,p_value=p_value,MSE_oos=MSE_oos,MSE_mean_oos=MSE_mean_oos,MSE_mean_oos_without_covid=MSE_mean_oos_without_covid,MSE_oos_without_covid=MSE_oos_without_covid,p_value_without_covid=p_value_without_covid))
}


library(fGarch)

select_vec_multi
sel_vec_pred<-select_vec_multi[3:5]
sel_vec_pred<-select_vec_multi
sel_vec_pred<-select_vec_multi[c(1,2)]
date_to_fit<-"2007"
use_garch<-T

p_mat_mssa<-p_mat_mssa_components<-p_mat_mssa_components_without_covid<-p_mat_direct<-rRMSE_mSSA_comp_direct<-rRMSE_mSSA_comp_mean<-rRMSE_mSSA_comp_direct_without_covid<-rRMSE_mSSA_comp_mean_without_covid<-matrix(ncol=length(h_vec),nrow=length(h_vec))
for (shift in h_vec)#shift<-1
{
  print(shift)
  
  for (j in h_vec)#j<-1
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
  
    perf_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit,use_garch,shift)
  
    p_mat_mssa_components[shift+1,k]<-perf_obj$p_value
    p_mat_mssa_components_without_covid[shift+1,k]<-perf_obj$p_value_without_covid
    MSE_oos_mssa_comp<-perf_obj$MSE_oos
    MSE_oos_mssa_comp_without_covid<-perf_obj$MSE_oos_without_covid
    MSE_mean_oos<-perf_obj$MSE_mean_oos
    MSE_mean_oos_without_covid<-perf_obj$MSE_mean_oos_without_covid
    
# Direct
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),x_mat[,sel_vec_pred])
    rownames(dat)<-rownames(x_mat)
    dat<-na.exclude(dat)
    
    perf_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit,use_garch,shift)
    
    p_mat_direct[shift+1,k]<-perf_obj$p_value 
    MSE_oos_direct<-perf_obj$MSE_oos
    MSE_oos_direct_without_covid<-perf_obj$MSE_oos_without_covid
    
    rRMSE_mSSA_comp_direct[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_oos_direct)
    rRMSE_mSSA_comp_mean[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_mean_oos)
    rRMSE_mSSA_comp_direct_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_oos_direct_without_covid)
    rRMSE_mSSA_comp_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_mean_oos_without_covid)
  }
}

colnames(p_mat_mssa)<-colnames(p_mat_mssa_components)<-colnames(p_mat_direct)<-colnames(p_mat_mssa_components_without_covid)<-
  colnames(rRMSE_mSSA_comp_direct)<-colnames(rRMSE_mSSA_comp_mean)<-
colnames(rRMSE_mSSA_comp_direct_without_covid)<-colnames(rRMSE_mSSA_comp_mean_without_covid)<-paste("h=",h_vec,sep="")
rownames(p_mat_mssa)<-rownames(p_mat_mssa_components)<-rownames(p_mat_direct)<-rownames(p_mat_mssa_components_without_covid)<-
  rownames(rRMSE_mSSA_comp_direct)<-rownames(rRMSE_mSSA_comp_mean)<-
rownames(rRMSE_mSSA_comp_direct_without_covid)<-rownames(rRMSE_mSSA_comp_mean_without_covid)<-paste("Shift=",h_vec,sep="")

p_mat_mssa
p_mat_mssa_components
p_mat_mssa_components_without_covid
p_mat_direct
rRMSE_mSSA_comp_mean
rRMSE_mSSA_comp_direct
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_comp_mean_without_covid

ts.plot(scale(dat),col=c("black",rainbow(ncol(dat)-1)))


