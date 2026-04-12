compute_calibrated_out_of_sample_predictors_func<-function(dat)
{
  len<-dim(dat)[1]
# First column is target, i.e. dimension is dim(dat)[2]-1
  n<-dim(dat)[2]-1
# Compute calibrated out-of-sample predictor, based on expanding window
#   -Use data up i for fitting the regression
#   -Compute a prediction with explanatory data in i+1
  cal_oos_pred<-rep(NA,len)
  for (i in (n+2):(len-1)) #i<-n+2
  {
# Fit model with data up to time point i    
    lm_obj<-lm(dat[1:i,1]~dat[1:i,2:(n+1)])
# Compute out-of-sample prediction for i+1
# Distinguish only one from multiple explanatory variables (R-code different...)    
    if (n==1)
    {
# Explanatory variable is the constant one (mean benchmark)      
      if (abs(sum(dat[,2]-1))<1.e-10)
      {
# In this case the regression coefficient by lm is NA (redundant column)
# Therefore we just use lm_obj$coef[1] which is the expanding mean
        cal_oos_pred[i+1]<-lm_obj$coef[1]
      } else
      {
# Otherwise we use the classic regression prediction        
        cal_oos_pred[i+1]<-lm_obj$coef[1]+lm_obj$coef[2]*dat[i+1,2] 
      } 
    } else
    {
# Classic regression prediction though we use %*% instead of * above      
      cal_oos_pred[i+1]<-lm_obj$coef[1]+lm_obj$coef[2:(n+1)]%*%dat[i+1,2:(n+1)] 
    }
  }
# Once the predictors are computed we can obtain the out-of-sample prediction errors
  epsilon_oos<-dat[,1]-cal_oos_pred
# And we can compute the HAC-adjusted p-values of the regression of the predictor on the target, out-of-sample  
  lm_oos<-lm(dat[,1]~cal_oos_pred)
  ts.plot(cbind(dat[,1],cal_oos_pred))
  summary(lm_oos)
  sd_HAC<-sqrt(diag(vcovHAC(lm_oos)))
  t_HAC<-summary(lm_oos)$coef[2,1]/sd_HAC[2]
  HAC_p_value<-pt(t_HAC, nrow(dat)-2, lower=FALSE)
# One-sided test: if predictor is effective, then the sign of the coefficient must be positive  
  p_value<-summary(lm_oos)$coef[2,4]
  
  return(list(cal_oos_pred=cal_oos_pred,epsilon_oos=epsilon_oos,p_value=p_value,HAC_p_value=HAC_p_value))
}







compute_perf_mean_func<-function(dat)
{
  len<-dim(dat)[1]
  # First column is target, i.e. dimension is dim(dat)[2]-1
  n<-dim(dat)[2]-1
  # Compute calibrated out-of-sample predictor, based on expanding window
  #   -Use data up i for fitting the regression
  #   -Compute a prediction with explanatory data in i+1
  cal_oos_pred<-dat[,2]
  # Once the predictors are computed we can obtain the out-of-sample prediction errors
  epsilon_oos<-dat[,1]-cal_oos_pred
  # And we can compute the HAC-adjusted p-values of the regression of the predictor on the target, out-of-sample  
  lm_oos<-lm(dat[,1]~cal_oos_pred)
  ts.plot(cbind(dat[,1],cal_oos_pred))
  summary(lm_oos)
  sd_HAC<-sqrt(diag(vcovHAC(lm_oos)))
  t_HAC<-summary(lm_oos)$coef[2,1]/sd_HAC[2]
  HAC_p_value<-pt(t_HAC, nrow(dat)-2, lower=FALSE)
  # One-sided test: if predictor is effective, then the sign of the coefficient must be positive  
  p_value<-summary(lm_oos)$coef[2,4]
  
  return(list(cal_oos_pred=cal_oos_pred,epsilon_oos=epsilon_oos,p_value=p_value,HAC_p_value=HAC_p_value))
}











oos_perf_func<-function(BIP_target,h_vec,data,indicator_mat,date_to_fit,lag_vec,target_shifted_mat,select_direct_indicator)
{
  # Initialize all matrices  
  dm_mat<-gw_mat<-HAC_p_value_mssa<-p_value_mssa<-rRMSE_mssa_mean<-rRMSE_mssa_direct<-rRMSE_direct_mean<-matrix(ncol=length(h_vec),nrow=length(h_vec))
  
  # i runs on the forecast horizon for which M-SSA has been optimized    
  for (i in 1:length(h_vec))#i<-1
  {
    print(paste("Shift=",h_vec[i]," out of max ",max(h_vec),sep=""))
    # j runs on the forward-shifts of the target BIP    
    for (j in 1:length(h_vec))#j<-1
    {
      shift<-h_vec[j]
# 1. Compute out-of-sample performances for MSSA
# Select forecast horizon for which M-SSA has been optimized: i-th column of indicator_mat
      h_vec[i]
      if (BIP_target)
      {
# For the target we can use either 
#   -data[(lag_vec[1]+shift+1):nrow(data),2]: this is the second column of data, i.e. BIP aligned at sample end
#   -or data[(shift+1):nrow(data),1]: this is the first column of data, i.e., BIP up-shifted by lag_vec[1]
        dat<-na.exclude(cbind(data[(shift+1):nrow(data),1],indicator_mat[1:(nrow(data)-shift),i]))
# The same as      
        dat<-na.exclude(cbind(data[(lag_vec[1]+shift+1):nrow(data),2],indicator_mat[1:(nrow(data)-shift-lag_vec[1]),i]))
      } else
      {
# Compute matrix with forward-shifted HP-BIP and M-SSA indicator
# We do not need to shift the series explicitly since the i-th column target_shifted_mat[,i] is already shifted
# Remove NAs since otherwise lm (regression) breaks down
        dat<-na.exclude(cbind(target_shifted_mat[,j],indicator_mat[,i]))
      }
      nrow(dat)
# Compute out-of-sample calibrated predictor
      oos_pred_obj<-compute_calibrated_out_of_sample_predictors_func(dat)
      
# Calibrated M-SSA Predictor    
      oos_mssa_pred<-oos_pred_obj$cal_oos_pred

# Out of sample forecast error of calibrated predictor    
      epsilon_oos_msa=oos_pred_obj$epsilon_oos
# HAC adjusted p-value of regression of regression of out-of-sample predictor on target    
      HAC_p_value=oos_pred_obj$HAC_p_value
      HAC_p_value_mssa[j,i]<-HAC_p_value
      p_value=oos_pred_obj$p_value
      p_value_mssa[j,i]<-p_value
      
# Add NA's at start to match full length
      oos_mssa_pred<-c(rep(NA,nrow(data)-length(oos_mssa_pred)),oos_mssa_pred)
      epsilon_oos_msa<-c(rep(NA,nrow(data)-length(epsilon_oos_msa)),epsilon_oos_msa)
      
#-------------------------------      
# 2. Same as above but for direct forecast
# Select indicators
      if (BIP_target)
      {
# Compute matrix with forward-shifted BIP and macro-indicators
# Remove NAs since otherwise lm breaks down
# We also add M-SSA in last column to obtain the same time span after removing NAs
        dat<-na.exclude(cbind(data[(shift+1+lag_vec[1]):nrow(data),2],data[1:(nrow(data)-shift-lag_vec[1]),select_direct_indicator],indicator_mat[1:(nrow(data)-shift-lag_vec[1]),j]))
      } else
      {
# Compute matrix with forward-shifted HP-BIP and macro-indicators
# We do not need to shift the series explicitly since the i-th column target_shifted_mat[,i] is already shifted
# We also add M-SSA in last column to obtain the same time span after removing NAs
        dat<-na.exclude(cbind(target_shifted_mat[,j],data[,select_direct_indicator],indicator_mat[,i]))
      }
# We now remove M-SSA in last column: M-SSA was just used to ensure that the samples are comparable after removal of NAs
      dat<-dat[,-ncol(dat)]
      nrow(dat)
      # Compute out-of-sample calibrated predictor
      oos_pred_obj<-compute_calibrated_out_of_sample_predictors_func(dat)
      
      oos_direct_pred<-oos_pred_obj$cal_oos_pred
      epsilon_oos_direct<-oos_pred_obj$epsilon_oos
      # Add NA's at start to match full length
      oos_direct_pred<-c(rep(NA,nrow(data)-length(oos_direct_pred)),oos_direct_pred)
      epsilon_oos_direct<-c(rep(NA,nrow(data)-length(epsilon_oos_direct)),epsilon_oos_direct)
      
#---------------------------      
# 3. Same but mean benchmark (the latter is based on expanding window)
# The mean of BIP based on the expanding window relies on the second column of data:     
      oos_mean_pred<-cumsum(data[,2])/1:nrow(data)
      if (BIP_target)
      {
# For consistency we use the same function as above, adding a column of ones for the explanatory variable in this case     
# We also add M-SSA in last column in order to have the same data span after removing NAs        
        dat<-na.exclude(cbind(data[(shift+1+lag_vec[1]):nrow(data),2],oos_mean_pred[(shift+1+lag_vec[1]):nrow(data)],indicator_mat[(shift+1+lag_vec[1]):nrow(data),j]))
      } else
      {
# Compute matrix with forward-shifted HP-BIP and macro-indicators
# We do not need to shift the series explicitly since the i-th column target_shifted_mat[,i] is already shifted
# We also add M-SSA in last column in order to have the same data span after removing NAs        
        dat<-na.exclude(cbind(target_shifted_mat[,j],oos_mean_pred,indicator_mat[,i]))
      }
# Remove M-SSA in last column      
      dat<-dat[,-ncol(dat)]
      ts.plot(dat)
      
      oos_pred_obj<-compute_perf_mean_func(dat)
        
      
      # Compute out-of-sample calibrated predictor
      oos_pred_obj<-compute_calibrated_out_of_sample_predictors_func(dat)
      
      epsilon_oos_mean<-oos_pred_obj$epsilon_oos
      # Add NA's at start to match full length
      oos_mean_pred<-c(rep(NA,nrow(data)-length(oos_mean_pred)),oos_mean_pred)
      epsilon_oos_mean<-c(rep(NA,nrow(data)-length(epsilon_oos_mean)),epsilon_oos_mean)
      
      # Bind all out-of-sample errors together    
      eps_mat<-cbind(epsilon_oos_mean,epsilon_oos_direct,epsilon_oos_msa)
      colnames(eps_mat)<-c("Mean","Direct",paste("MSSA, h=",h_vec[i],sep=""))
      rownames(eps_mat)<-rownames(data)
      
      # Select out-of-sample span of M-SSA
      eps_mat_oos<-eps_mat[which(rownames(eps_mat)>date_to_fit),]
      # One could also remove Pandemic    
      #      eps_mat_oos<-eps_mat[which(rownames(eps_mat)>date_to_fit&rownames(eps_mat)<2019),]
      
      # Compute out-of-sample root mean-square errors        
      RMSE<-sqrt(apply(na.exclude(eps_mat_oos)^2,2,mean))
      # Compute relative root mean-square errors        
      rRMSE_mssa_mean[j,i]<-RMSE[paste("MSSA, h=",h_vec[i],sep="")]/RMSE["Mean"]
      rRMSE_mssa_direct[j,i]<-RMSE[paste("MSSA, h=",h_vec[i],sep="")]/RMSE["Direct"]
      rRMSE_direct_mean[j,i]<-RMSE["Direct"]/RMSE["Mean"]
      if (T)
      {

        # Estimator of variance   
        method_vec<-c("HAC","NeweyWest","Andrews","LumleyHeagerty")
        method = method_vec[1]
        target<-c(rep(NA,nrow(data)-length(dat[,1])),dat[,1])
        length(target)
        length(oos_mean_pred)
        length(oos_mssa_pred)
        test_mat<-na.exclude(cbind(target,oos_mean_pred,oos_mssa_pred))
        # 2. Diebold Mariano    
        loss.type="SE" 
        # One-sided tests (of course specification of alternative is just opposite... )    
        H1<-"more"
        # c=T in call means finite/small sample correction
        dm_obj<-DM.test(f1=test_mat[,3],f2=test_mat[,2],y=test_mat[,1],loss=loss.type,h=shift,c=T,H1=H1)
        
        dm_mat[j,i]<-dm_obj$p.value
        
        summary(lm(test_mat[,1]~test_mat[,2]))
        summary(lm(test_mat[,1]~test_mat[,3]))
        
        par(mfrow=c(2,1))
        ts.plot(test_mat,col=c("black","red","blue"))
        ts.plot(cbind(target_shifted_mat[,j],indicator_mat[,i]))
        
        alternative<-"less"
        method = "HAC"
        tau<-shift
        
        gw_obj<-gw.test(x = test_mat[,3], y = test_mat[,2], p =test_mat[,1], method = method, alternative = alternative)
        gw_mat[j,i]<-gw_obj$p.value
        
        
      }
      
    }
  }
  colnames(rRMSE_mssa_mean)<-colnames(rRMSE_mssa_direct)<-colnames(rRMSE_direct_mean)<-colnames(p_value_mssa)<-colnames(HAC_p_value_mssa)<-paste("h=",h_vec,sep="")
  rownames(rRMSE_mssa_mean)<-rownames(rRMSE_mssa_direct)<-rownames(rRMSE_direct_mean)<-rownames(p_value_mssa)<-rownames(HAC_p_value_mssa)<-paste("Shift=",h_vec,sep="")
  # Since direct forecasts do not depend on h all columns are identical: we then just keep the first
  rRMSE_direct_mean<-matrix(rRMSE_direct_mean[,1],ncol=1)
  rownames(rRMSE_direct_mean)<-rownames(rRMSE_mssa_mean)
  colnames(rRMSE_direct_mean)<-"rRMSE Direct"
  return(list(rRMSE_mssa_mean=rRMSE_mssa_mean,rRMSE_mssa_direct=rRMSE_mssa_direct,rRMSE_direct_mean=rRMSE_direct_mean,p_value_mssa=p_value_mssa,HAC_p_value_mssa=HAC_p_value_mssa,dm_mat=dm_mat,gw_mat=gw_mat))
}





oos_perf_func<-function(BIP_target,h_vec,data,indicator_mat,date_to_fit,lag_vec,target_shifted_mat,select_direct_indicator)
{
  # We first compute out-of-sample direct forecasts for each forward step
  direct_mat<-NULL
  for (j in 1:length(h_vec))
  {
    shift<-h_vec[j]
    # Distinguish BIP and HP-BIP targets    
    if (BIP_target)
    {
      # Compute matrix with forward-shifted BIP and selected macro-indicators
      # We can use either data[(shift+1+lag_vec[1]):nrow(data),2] or data[(1+lag_vec[1]):nrow(data),1]
      #   The first column is BIP shifted upwards by lag_vec[1]      
      # Remove NAs since otherwise lm breaks down
      dat<-na.exclude(cbind(data[(shift+1+lag_vec[1]):nrow(data),2],data[1:(nrow(data)-shift-lag_vec[1]),select_direct_indicator]))
    } else
    {
      # Compute matrix with forward-shifted HP-BIP and selected macro-indicators
      # We do not need to shift the series explicitly as above because the i-th column target_shifted_mat[,i] 
      #   is already forward-shifted shifted
      dat<-na.exclude(cbind(target_shifted_mat[,j],data[,select_direct_indicator]))
    }
    # Compute out-of-sample calibrated predictor based on expanding-window
    oos_pred_obj<-compute_calibrated_out_of_sample_predictors_func(dat)
    
    oos_direct_pred<-oos_pred_obj$cal_oos_pred
    
    direct_mat<-cbind(direct_mat,oos_direct_pred)
  }
  
  
  # Initialize all matrices  
  HAC_p_value_mssa<-p_value_mssa<-rRMSE_mssa_mean<-rRMSE_mssa_direct<-rRMSE_direct_mean<-matrix(ncol=length(h_vec),nrow=length(h_vec))
  
  # i runs on the forecast horizon for which M-SSA has been optimized    
  for (i in 1:length(h_vec))#i<-7
  {
    print(paste("Shift=",h_vec[i]," out of max ",max(h_vec),sep=""))
    # j runs on the forward-shifts of the target BIP    
    for (j in 1:length(h_vec))#j<-1
    {
      shift<-h_vec[j]
      # 1. Compute out-of-sample performances for MSSA
      # Select forecast horizon for which M-SSA has been optimized: i-th column of indicator_mat
      h_vec[i]
      if (BIP_target)
      {
        # For the target we can use either 
        #   -data[(lag_vec[1]+shift+1):nrow(data),2]: this is the second column of data, i.e. BIP aligned at sample end
        #   -or data[(shift+1):nrow(data),1]: this is the first column of data, i.e., BIP up-shifted by lag_vec[1]
        dat<-na.exclude(cbind(data[(shift+1):nrow(data),1],indicator_mat[1:(nrow(data)-shift),i]))
        # The same as      
        dat<-na.exclude(cbind(data[(lag_vec[1]+shift+1):nrow(data),2],indicator_mat[1:(nrow(data)-shift-lag_vec[1]),i]))
      } else
      {
        # Compute matrix with forward-shifted HP-BIP and M-SSA indicator
        # We do not need to shift the series explicitly since the i-th column target_shifted_mat[,i] is already shifted
        # Remove NAs since otherwise lm (regression) breaks down
        dat<-na.exclude(cbind(target_shifted_mat[,j],indicator_mat[,i]))
      }
      # Compute out-of-sample calibrated predictor
      oos_pred_obj<-compute_calibrated_out_of_sample_predictors_func(dat)
      
      # Calibrated M-SSA Predictor    
      oos_mssa_pred<-oos_pred_obj$cal_oos_pred
      # Out of sample forecast error of calibrated predictor    
      epsilon_oos_msa=oos_pred_obj$epsilon_oos
      # HAC adjusted p-value of regression of regression of out-of-sample predictor on target    
      HAC_p_value=oos_pred_obj$HAC_p_value
      HAC_p_value_mssa[j,i]<-HAC_p_value
      p_value=oos_pred_obj$p_value
      p_value_mssa[j,i]<-p_value
      
      # Add NA's at start to match full length
      oos_mssa_pred<-c(rep(NA,nrow(data)-length(oos_mssa_pred)),oos_mssa_pred)
      epsilon_oos_msa<-c(rep(NA,nrow(data)-length(epsilon_oos_msa)),epsilon_oos_msa)
      
      #-----------------------      
      # 2. Same as above but for direct forecast
      # Direct (out-of-sample) predictors have been computed above 
      if (BIP_target)
      {
        # For the target we can use either 
        #   -data[(lag_vec[1]+shift+1):nrow(data),2]: this is the second column of data, i.e. BIP aligned at sample end
        #   -or data[(shift+1):nrow(data),1]: this is the first column of data, i.e., BIP up-shifted by lag_vec[1]
        dat<-na.exclude(cbind(data[(shift+1):nrow(data),1],direct_mat[1:(nrow(data)-shift),i]))
        # The same as      
        dat<-na.exclude(cbind(data[(lag_vec[1]+shift+1):nrow(data),2],direct_mat[1:(nrow(data)-shift-lag_vec[1]),i]))
      } else
      {
        # Compute matrix with forward-shifted HP-BIP and M-SSA indicator
        # We do not need to shift the series explicitly since the i-th column target_shifted_mat[,i] is already shifted
        # Remove NAs since otherwise lm (regression) breaks down
        dat<-na.exclude(cbind(target_shifted_mat[,j],direct_mat[,i]))
      }
      
      # Compute out-of-sample calibrated predictor
      oos_pred_obj<-compute_calibrated_out_of_sample_predictors_func(dat)
      
      oos_direct_pred<-oos_pred_obj$cal_oos_pred
      epsilon_oos_direct<-oos_pred_obj$epsilon_oos
      # Add NA's at start to match full length
      oos_direct_pred<-c(rep(NA,nrow(data)-length(oos_direct_pred)),oos_direct_pred)
      epsilon_oos_direct<-c(rep(NA,nrow(data)-length(epsilon_oos_direct)),epsilon_oos_direct)
      
      
      # 3. Mean benchmark
      # We could use something like     
      cumsum(data[,1])/1:nrow(data)
      if (BIP_target)
      {
        # But for consistency we use the same function as above, adding a column of ones for the explanatory variable in this case     
        dat<-na.exclude(cbind(data[(shift+1+lag_vec[1]):nrow(data),2],rep(1,nrow(data)-shift-lag_vec[1])))
      } else
      {
        # Compute matrix with forward-shifted HP-BIP and macro-indicators
        # We do not need to shift the series explicitly since the i-th column target_shifted_mat[,i] is already shifted
        dat<-na.exclude(cbind(target_shifted_mat[,j],rep(1,nrow(target_shifted_mat))))
      }
      
      # Compute out-of-sample calibrated predictor
      oos_pred_obj<-compute_calibrated_out_of_sample_predictors_func(dat)
      
      oos_mean_pred<-oos_pred_obj$cal_oos_pred
      epsilon_oos_mean<-oos_pred_obj$epsilon_oos
      # Add NA's at start to match full length
      oos_mean_pred<-c(rep(NA,nrow(data)-length(oos_direct_pred)),oos_mean_pred)
      epsilon_oos_mean<-c(rep(NA,nrow(data)-length(epsilon_oos_mean)),epsilon_oos_mean)
      
      # Bind all out-of-sample errors together    
      eps_mat<-cbind(epsilon_oos_mean,epsilon_oos_direct,epsilon_oos_msa)
      colnames(eps_mat)<-c("Mean","Direct",paste("MSSA, h=",h_vec[i],sep=""))
      rownames(eps_mat)<-rownames(data)
      
      # Select out-of-sample span of M-SSA
      eps_mat_oos<-eps_mat[which(rownames(eps_mat)>date_to_fit),]
      # One could also remove Pandemic    
      #      eps_mat_oos<-eps_mat[which(rownames(eps_mat)>date_to_fit&rownames(eps_mat)<2019),]
      
      # Compute out-of-sample root mean-square errors        
      RMSE<-sqrt(apply(na.exclude(eps_mat_oos)^2,2,mean))
      # Compute relative root mean-square errors        
      rRMSE_mssa_mean[j,i]<-RMSE[paste("MSSA, h=",h_vec[i],sep="")]/RMSE["Mean"]
      rRMSE_mssa_direct[j,i]<-RMSE[paste("MSSA, h=",h_vec[i],sep="")]/RMSE["Direct"]
      rRMSE_direct_mean[j,i]<-RMSE["Direct"]/RMSE["Mean"]
      
    }
  }
  colnames(rRMSE_mssa_mean)<-colnames(rRMSE_mssa_direct)<-colnames(rRMSE_direct_mean)<-colnames(p_value_mssa)<-colnames(HAC_p_value_mssa)<-paste("h=",h_vec,sep="")
  rownames(rRMSE_mssa_mean)<-rownames(rRMSE_mssa_direct)<-rownames(rRMSE_direct_mean)<-rownames(p_value_mssa)<-rownames(HAC_p_value_mssa)<-paste("Shift=",h_vec,sep="")
  return(list(rRMSE_mssa_mean=rRMSE_mssa_mean,rRMSE_mssa_direct=rRMSE_mssa_direct,rRMSE_direct_mean=rRMSE_direct_mean,p_value_mssa=p_value_mssa,HAC_p_value_mssa=HAC_p_value_mssa))
}










##################################################################################
# Old code

# The function returns rMSE, d t-statistics of regressions of predictors on shifted BIP and DM/GW statistics
#   -tstatistics are based on OLS and GLS (assuming AR(1) process for residuals)
#   -rRMSE is based on OLS residuals, references against the mean (of BIP) as benchmark
compute_all_perf_func<-function(indicator_cal,data,lag_vec,h_vec,h,select_direct_indicator,L,lambda_HP,date_to_fit)
{
  # A. M-SSA predictor against shifts of BIP
  # A.1. Compute data matrix    
  # combine UNSHIFTED target in SECOND column of data with indicator_cal
  #   -The lengths differ but we align both of them at the end of the series
  #   -We write NAs forward with na.locf (at sample end) 
  dat_mhh<-na.locf(cbind(data[(nrow(data)-length(indicator_cal)+1):nrow(data),2],indicator_cal))
  # We remove all NAs at start: due to filter initialization: na.locf above must be done first  
  dat_mhh<-na.exclude((dat_mhh))
  tail(dat_mhh)
  head(dat_mhh)
  # We now shift target column (first column) by its publication lag
  dat_mh<-cbind(c(dat_mhh[(1+lag_vec[1]):nrow(dat_mhh),1],rep(dat_mhh[nrow(dat_mhh),1],lag_vec[1])),dat_mhh[,2:ncol(dat_mhh)])
  # Supply correct dates  
  rownames(dat_mh)<-rownames(dat_mhh)
  tail(dat_mh)
  nrow(dat_mh)
  # Remove rows with redundant BIP at end (due to publication lag): we lose lag_vec[1] observations at end 
  dat_mh<-dat_mh[-((nrow(dat_mh)-lag_vec[1]+1):nrow(dat_mh)),]
  tail(dat_mh)
  nrow(dat_mh)
  
  
  # A.2 M-SSA predictor against all shifts of BIP
  #   -Compute rRMSE, t- and F-statistics of OLS regression of predictor on shifted target  
  #   -Compute also t- and F-statistics of GLS regression of predictor on shifted target 
  p_dm_vec<-p_gw_vec<-p_dm_vec_short<-p_gw_vec_short<-p_dm_vec_out<-p_gw_vec_out<-NULL
  for (j in 1:length(h_vec))#j<-1
  {
    # Forward-shift of target    
    i<-h_vec[j]
    # Target (first column) is shifted by i
    dat_m<-cbind(dat_mh[(1+i):nrow(dat_mh),1],dat_mh[1:(nrow(dat_mh)-i),2])
    tail(dat_m)
    # Compute t-tests and rRMSE
    comp_obj<-comp_perf_func(dat_m)
    
    if (j==1)
    {
      mat_all<-comp_obj$mat_all
    } else
    {
      mat_all<-rbind(mat_all,comp_obj$mat_all)
    }
    
    # Compute DM and GW tests    
    DM_GW_obj<-pcompute_DM_GW_statistics(dat_m,i)
    
    p_dm=DM_GW_obj$p_dm
    p_gw=DM_GW_obj$p_gw
    p_dm_vec<-c(p_dm_vec,p_dm)
    p_gw_vec<-c(p_gw_vec,p_gw)
    #---------------------------------------    
    # Same but without Pandemic
    ind<-which(rownames(dat_m)<=2019)
    enfh<-ind[length(ind)]
    dat_m_short<-dat_m[1:enfh,]
    tail(dat_m_short)
    # Same but out-of-sample
    ind<-which(rownames(dat_m)>date_to_fit)
    dat_m_out<-dat_m[ind,]
    head(dat_m_out)
    tail(dat_m_out)

    comp_obj_short<-comp_perf_func(dat_m_short)
    
    comp_obj_out<-comp_perf_func(dat_m_out)
    
    if (j==1)
    {
      mat_short<-comp_obj_short$mat_all
      mat_out<-comp_obj_out$mat_all
    } else
    {
      mat_short<-rbind(mat_short,comp_obj_short$mat_all)
      mat_out<-rbind(mat_out,comp_obj_out$mat_all)
    }
    
    # Compute DM and GW tests    
    DM_GW_obj_short<-pcompute_DM_GW_statistics(dat_m_short,i)
    DM_GW_obj_out<-pcompute_DM_GW_statistics(dat_m_out,i)
    
    p_dm=DM_GW_obj_short$p_dm
    p_gw=DM_GW_obj_short$p_gw
    p_dm_vec_short<-c(p_dm_vec_short,p_dm)
    p_gw_vec_short<-c(p_gw_vec_short,p_gw)
    p_dm=DM_GW_obj_out$p_dm
    p_gw=DM_GW_obj_out$p_gw
    p_dm_vec_out<-c(p_dm_vec_out,p_dm)
    p_gw_vec_out<-c(p_gw_vec_out,p_gw)
  }
  
  names(p_dm_vec)<-names(p_gw_vec)<-names(p_dm_vec_short)<-names(p_gw_vec_short)<-names(p_dm_vec_out)<-names(p_gw_vec_out)<-paste("shift=",h_vec,sep="")
  
  mat_all<-matrix(mat_all,nrow=length(h_vec))
  mat_short<-matrix(mat_short,nrow=length(h_vec))
  mat_out<-matrix(mat_out,nrow=length(h_vec))
  colnames(mat_all)<-colnames(mat_short)<-colnames(mat_out)<-c("rRMSE","HAC t-stat OLS")
  rownames(mat_all)<-rownames(mat_short)<-rownames(mat_out)<-paste("M-SSA optimized for h=",h,": shift ",h_vec,sep="")
  
  mat_all
  mat_short 
  p_dm_vec_short
  p_gw_vec_short
  p_dm_vec
  p_gw_vec
  p_dm_vec_out
  p_gw_vec_out
  
  #------------------------------------------------------------------------------
  # B. M-SSA predictor against shifts of HP-BIP
  # B.1. Compute data matrix    
  # combine UNSHIFTED target in SECOND column of data with indicator_cal
  #   -The lengths differ but we align both of them at the end of the series
  #   -We write NAs forward with na.locf (at sample end) 
  dat_mhh<-na.locf(cbind(data[(nrow(data)-length(indicator_cal)+1):nrow(data),2],indicator_cal))
  tail(dat_mhh)
  head(dat_mhh)
  # We now shift target column (first column) by its publication lag
  dat_mh<-cbind(c(dat_mhh[(1+lag_vec[1]):nrow(dat_mhh),1],rep(dat_mhh[nrow(dat_mhh),1],lag_vec[1])),dat_mhh[,2:ncol(dat_mhh)])
  # Supply correct dates  
  rownames(dat_mh)<-rownames(dat_mhh)
  tail(dat_mh)
  nrow(dat_mh)
  # Remove rows with redundant BIP at end (due to publication lag): we lose lag_vec[1] observations at end 
  dat_mh<-dat_mh[-((nrow(dat_mh)-lag_vec[1]+1):nrow(dat_mh)),]
  tail(dat_mh)
  nrow(dat_mh)
  # B.2 Apply symmetric HP
  HP_obj<-HP_target_mse_modified_gap(2*(L-1)+1,lambda_HP)
  hp_symmetric<-HP_obj$target
  hp_bip<-as.double(filter(dat_mh[,1],hp_symmetric,side=2))
  names(hp_bip)<-rownames(dat_mh)
  dat_mh<-cbind(hp_bip,dat_mh[,2:ncol(dat_mh)])
  tail(dat_mh)
  dat_mh<-na.exclude(dat_mh)
  tail(dat_mh)
  
  # B.2 M-SSA predictor against all shifts of BIP
  #   -Compute rRMSE, t- and F-statistics of OLS regression of predictor on shifted target  
  #   -Compute also t- and F-statistics of GLS regression of predictor on shifted target 
  p_dm_vec_HP<-p_gw_vec_HP<-p_dm_vec_short_HP<-p_gw_vec_short_HP<-p_dm_vec_out_HP<-p_gw_vec_out_HP<-NULL
  for (j in 1:length(h_vec))#i<-0
  {
    # Forward shift    
    i<-h_vec[j]
    # Target (first column) is shifted by i
    dat_m<-cbind(dat_mh[(1+i):nrow(dat_mh),1],dat_mh[1:(nrow(dat_mh)-i),2])
    tail(dat_m)
    # Compute t-tests and rRMSE
    
    comp_obj<-comp_perf_func(dat_m)
    

    if (j==1)
    {
      mat_all_HP<-comp_obj$mat_all
    } else
    {
      mat_all_HP<-rbind(mat_all_HP,comp_obj$mat_all)
    }
    
    # Compute DM and GW tests    
    DM_GW_obj<-pcompute_DM_GW_statistics(dat_m,i)
    
    p_dm=DM_GW_obj$p_dm
    p_gw=DM_GW_obj$p_gw
    p_dm_vec_HP<-c(p_dm_vec_HP,p_dm)
    p_gw_vec_HP<-c(p_gw_vec_HP,p_gw)

    #---------------------------------------    
    # Same but without Pandemic
    ind<-which(rownames(dat_m)<=2019)
    enfh<-ind[length(ind)]
    dat_m_short<-dat_m[1:enfh,]
    tail(dat_m_short)
    # Same but out-of-sample
    ind<-which(rownames(dat_m)>date_to_fit)
    dat_m_out<-dat_m[ind,]
    head(dat_m_out)
    tail(dat_m_out)
    
    comp_obj_short<-comp_perf_func(dat_m_short)
    
    comp_obj_out<-comp_perf_func(dat_m_out)
    
    if (j==1)
    {
      mat_short_HP<-comp_obj_short$mat_all
      mat_out_HP<-comp_obj_out$mat_all
    } else
    {
      mat_short_HP<-rbind(mat_short_HP,comp_obj_short$mat_all)
      mat_out_HP<-rbind(mat_out_HP,comp_obj_out$mat_all)
    }
    
    # Compute DM and GW tests    
    DM_GW_obj_short_HP<-pcompute_DM_GW_statistics(dat_m_short,i)
    DM_GW_obj_out_HP<-pcompute_DM_GW_statistics(dat_m_out,i)
    
    p_dm=DM_GW_obj_short_HP$p_dm
    p_gw=DM_GW_obj_short_HP$p_gw
    p_dm_vec_short_HP<-c(p_dm_vec_short_HP,p_dm)
    p_gw_vec_short_HP<-c(p_gw_vec_short_HP,p_gw)
    p_dm=DM_GW_obj_out_HP$p_dm
    p_gw=DM_GW_obj_out_HP$p_gw
    p_dm_vec_out_HP<-c(p_dm_vec_out_HP,p_dm)
    p_gw_vec_out_HP<-c(p_gw_vec_out_HP,p_gw)
    
  }
  
  names(p_dm_vec_HP)<-names(p_gw_vec_HP)<-names(p_dm_vec_short_HP)<-names(p_gw_vec_out_HP)<-names(p_dm_vec_out_HP)<-names(p_gw_vec_short_HP)<-paste("shift=",h_vec,sep="")
  
  p_dm_vec_short_HP
  p_gw_vec_short_HP
  # Note that short and full sample performances might be identical because two-sided HP does not reach sample end
  # If Post Pnademic is in left tail then short and full sample perfs are identical
  p_dm_vec_short_HP
  p_dm_vec_HP
  
  
  #-----------------------------------------------------------------------------------------------  
  # C. Direct predictor against BIP
  # Direct predictor is based on indicators specified by select_direct_indicator
  # Take care that sample is the same as for M-SSA above (the latter relies on filter initialization)  
  # C.1 Combine data with indicator, both series aligned at end (in order to have the same sample for evaluation as indicator)  
  dat_mhh<-na.locf(cbind(data[(nrow(data)-length(indicator_cal)+1):nrow(data),],indicator_cal))
  # We remove all NAs at start: due to filter initialization: na.locf above must be done first  
  dat_mhh<-na.exclude((dat_mhh))
  tail(dat_mhh)
  # Select BIP (UNSHIFTED) and indicators
  dat_mh<-na.exclude(dat_mhh[,c("BIP",select_direct_indicator)])
  tail(dat_mh)
  # Shift BIP by publication lag
  dat_mh<-cbind(c(dat_mh[(1+lag_vec[1]):nrow(dat_mh),1],rep(dat_mh[nrow(dat_mh),1],lag_vec[1])),dat_mh[,2:ncol(dat_mh)])
  rownames(dat_mh)<-rownames(dat_mhh)
  tail(dat_mh)
  nrow(dat_mh)
  # Remove rows with redundant BIP at end (due to publication lag): we lose lag_vec[1] observations at end 
  dat_mh<-dat_mh[-((nrow(dat_mh)-lag_vec[1]+1):nrow(dat_mh)),]
  tail(dat_mh)
  # Same sample (length) as above, for M-SSA  
  nrow(dat_mh)
  
  
  direct_pred_mat<-p_dm_vec_direct<-p_gw_vec_direct<-p_dm_vec_direct_short<-p_gw_vec_direct_short<-p_dm_vec_direct_out<-p_gw_vec_direct_out<-NULL
  
  # C.3 Compute direct predictor up to sample end and derive performances for all shifts
  for (j in 1:length(h_vec))#j<-1
  {
    # Forward shift
    i<-h_vec[j]
    # Target (first column) is shifted by i
    dat_m<-cbind(dat_mh[(1+i):nrow(dat_mh),1],dat_mh[1:(nrow(dat_mh)-i),2:ncol(dat_mh)])
    tail(dat_m)
    # We apply regression to full sample to obtain predictor values up to the sample end    
    dat_apply_reg<-dat_mh
    
    # Compute direct forecast, t-tests and rRMSE
    comp_obj<-comp_perf_func(dat_m,dat_apply_reg)
    
    if (j==1)
    {
      mat_all_direct<-comp_obj$mat_all
      direct_pred_mat=comp_obj$direct_pred
    } else
    {
      mat_all_direct<-rbind(mat_all_direct,comp_obj$mat_all)
      direct_pred_mat=cbind(direct_pred_mat,comp_obj$direct_pred)
    }
    
    # Compute DM and GW tests    
    DM_GW_obj<-pcompute_DM_GW_statistics(dat_m,i)
    
    
    p_dm=DM_GW_obj$p_dm
    p_gw=DM_GW_obj$p_gw
    p_dm_vec_direct<-c(p_dm_vec_direct,p_dm)
    p_gw_vec_direct<-c(p_gw_vec_direct,p_gw)
    #-----------------------    
    # Same but data prior Pandemic 
    
    ind<-which(rownames(dat_m)<=2019)
    enfh<-ind[length(ind)]
    dat_m_short<-dat_m[1:enfh,]
    tail(dat_m_short)
    # Same but out-of-sample
    ind<-which(rownames(dat_m)<date_to_fit)
    dat_m_out<-dat_m[ind,]
    head(dat_m_out)
    tail(dat_m_out)
# We fit data to shorter samples but we apply regression to full sample to obtain predictor values up to the sample end    
    dat_apply_reg<-dat_mh

    comp_obj_short<-comp_perf_func(dat_m_short,dat_apply_reg)
    
# We fit data to insample span but we apply regression to out-of-sample span   
    ind<-which(rownames(dat_m)>date_to_fit)
    
    dat_apply_reg<-dat_mh[ind,]
    head(dat_apply_reg)
    tail(dat_apply_reg)
    
    
    comp_obj_out<-comp_perf_func(dat_m_out,dat_apply_reg)

    if (j==1)
    {
      mat_short_direct<-comp_obj_short$mat_all
      mat_out_direct<-comp_obj_out$mat_all
      direct_pred_mat_short=comp_obj_short$direct_pred
      direct_pred_mat_out=comp_obj_out$direct_pred
    } else
    {
      mat_short_direct<-rbind(mat_short_direct,comp_obj_short$mat_all)
      direct_pred_mat_short=cbind(direct_pred_mat_short,comp_obj_short$direct_pred)
      mat_out_direct<-rbind(mat_out_direct,comp_obj_out$mat_all)
      direct_pred_mat_out=cbind(direct_pred_mat_out,comp_obj_out$direct_pred)
    }
    
    # Compute DM and GW tests    
    DM_GW_obj_short<-pcompute_DM_GW_statistics(dat_m_short,i)
    DM_GW_obj_out<-pcompute_DM_GW_statistics(dat_m_out,i)
    
    p_dm=DM_GW_obj_short$p_dm
    p_gw=DM_GW_obj_short$p_gw
    p_dm_vec_direct_short<-c(p_dm_vec_direct_short,p_dm)
    p_gw_vec_direct_short<-c(p_gw_vec_direct_short,p_gw)
    p_dm=DM_GW_obj_out$p_dm
    p_gw=DM_GW_obj_out$p_gw
    p_dm_vec_direct_out<-c(p_dm_vec_direct_out,p_dm)
    p_gw_vec_direct_out<-c(p_gw_vec_direct_out,p_gw)
    
  }
  names(p_dm_vec_direct)<-names(p_gw_vec_direct)<-names(p_dm_vec_direct_short)<-names(p_gw_vec_direct_short)<-names(p_dm_vec_direct_out)<-names(p_gw_vec_direct_out)<-paste("shift=",h_vec,sep="")

  colnames(direct_pred_mat)<-paste("Direct AR predictor: h=",h_vec,sep="")
  rownames(direct_pred_mat)<-rownames(dat_mh)[(nrow(dat_mh)-nrow(direct_pred_mat)+1):nrow(dat_mh)]
  mat_all_direct<-matrix(mat_all_direct,nrow=length(h_vec))
  mat_short_direct<-matrix(mat_short_direct,nrow=length(h_vec))
  mat_out_direct<-matrix(mat_out_direct,nrow=length(h_vec))
  colnames(mat_all_direct)<-colnames(mat_short_direct)<-colnames(mat_out_direct)<-c("rRMSE","HAC t-stat OLS")
  rownames(mat_all_direct)<-rownames(mat_short_direct)<-rownames(mat_out_direct)<-paste("Shift=",h_vec,sep="")
  
  mat_all_direct
  mat_short_direct 
  p_dm_vec_direct_short
  p_gw_vec_direct_short
  
  #----------------------------------------------------------------------------
  # D. M-SSA predictor against direct predictor: both targeting shifted BIP
  # D.1. Compute data matrix: Combine data with indicator, both series aligned at end (in order to have the same sample for evaluation as indicator)  
  dat_mhh<-na.locf(cbind(data[(nrow(data)-length(indicator_cal)+1):nrow(data),],indicator_cal))
  # We remove all NAs at start: due to filter initialization: na.locf above must be done first  
  dat_mhh<-na.exclude((dat_mhh))
  tail(dat_mhh)
  # Select BIP (UNSHIFTED) and indicators
  dat_mh<-na.exclude(dat_mhh[,c("BIP",select_direct_indicator,"indicator_cal")])
  tail(dat_mh)
  # Shift BIP by publication lag
  dat_mh<-cbind(c(dat_mh[(1+lag_vec[1]):nrow(dat_mh),1],rep(dat_mh[nrow(dat_mh),1],lag_vec[1])),dat_mh[,2:ncol(dat_mh)])
  rownames(dat_mh)<-rownames(dat_mhh)
  tail(dat_mh)
  nrow(dat_mh)
  # Remove rows with redundant BIP at end (due to publication lag): we lose lag_vec[1] observations at end 
  dat_mh<-dat_mh[-((nrow(dat_mh)-lag_vec[1]+1):nrow(dat_mh)),]
  tail(dat_mh)
  # Same sample (length) as above, for M-SSA  
  nrow(dat_mh)
  
  # D.2 M-SSA predictor against all shifts of BIP
  #   -Compute rRMSE, t- and F-statistics of OLS regression of predictor on shifted target  
  #   -Compute also t- and F-statistics of GLS regression of predictor on shifted target 
  p_dm_vec_mssa_direct<-p_gw_vec_mssa_direct<-p_dm_vec_short_mssa_direct<-p_gw_vec_short_mssa_direct<-p_dm_vec_out_mssa_direct<-p_gw_vec_out_mssa_direct<-NULL
  for (j in 1:length(h_vec))#i<-0
  {
    # Forward shift
    i<-h_vec[j]
    # Target (first column) is shifted by i, we also append M-SSA shifted as the explanatory indicators
    dat_all<-cbind(dat_mh[(1+i):nrow(dat_mh),1],dat_mh[1:(nrow(dat_mh)-i),2:ncol(dat_mh)])
    # Remove M-SSA    
    dat_m<-dat_all[,1:(ncol(dat_all)-1)]
    # M-SSA    
    mssa<-dat_all[,ncol(dat_all)]
    
    tail(dat_m)
    # Compute DM and GW tests    
    DM_GW_obj<-pcompute_DM_GW_statistics_MSSA_against_Direct(dat_m,mssa,i)#end_date<-2019
    
    
    p_dm=DM_GW_obj$p_dm
    p_gw=DM_GW_obj$p_gw
    p_dm_vec_mssa_direct<-c(p_dm_vec_mssa_direct,p_dm)
    p_gw_vec_mssa_direct<-c(p_gw_vec_mssa_direct,p_gw)
    #---------------------------------------    
    # Same but data prior Pandemic 
    ind<-which(rownames(dat_m)<=2019)
    enfh<-ind[length(ind)]
    dat_all_short<-dat_all[1:enfh,]
    # Remove M-SSA    
    dat_m_short<-dat_all_short[,1:(ncol(dat_all_short)-1)]
    # M-SSA    
    mssa_short<-dat_all_short[,ncol(dat_all_short)]
    
    tail(dat_m_short)
    
    # Same but out-of-sample
    ind<-which(rownames(dat_m)>date_to_fit)
    dat_all_out<-dat_all[ind,]
    
    # Remove M-SSA    
    dat_m_out<-dat_all_out[,1:(ncol(dat_all_out)-1)]
    # M-SSA    
    mssa_out<-dat_all_out[,ncol(dat_all_out)]
    
    head(dat_m_out)
    tail(dat_m_out)
    
    
    # Compute DM and GW tests    
    # Compute DM and GW tests    
    DM_GW_obj_short<-pcompute_DM_GW_statistics_MSSA_against_Direct(dat_m_short,mssa_short,i)#end_date<-2019
    DM_GW_obj_out<-pcompute_DM_GW_statistics_MSSA_against_Direct(dat_m_out,mssa_out,i)#end_date<-2019
    
    p_dm=DM_GW_obj_short$p_dm
    p_gw=DM_GW_obj_short$p_gw
    p_dm_vec_short_mssa_direct<-c(p_dm_vec_short_mssa_direct,p_dm)
    p_gw_vec_short_mssa_direct<-c(p_gw_vec_short_mssa_direct,p_gw)
    p_dm=DM_GW_obj_out$p_dm
    p_gw=DM_GW_obj_out$p_gw
    p_dm_vec_out_mssa_direct<-c(p_dm_vec_out_mssa_direct,p_dm)
    p_gw_vec_out_mssa_direct<-c(p_gw_vec_out_mssa_direct,p_gw)
  }
  
  names(p_dm_vec_mssa_direct)<-names(p_gw_vec_mssa_direct)<-names(p_dm_vec_short_mssa_direct)<-names(p_gw_vec_short_mssa_direct)<-names(p_dm_vec_out_mssa_direct)<-names(p_gw_vec_out_mssa_direct)<-paste("shift=",h_vec,sep="")
  
  p_dm_vec_short_mssa_direct
  p_gw_vec_short_mssa_direct
  
  #---------------------------
  # Collect DM and GW statistics in matrices
  gw_dm_short_mat<-cbind(p_dm_vec_short,p_gw_vec_short,p_dm_vec_HP,p_gw_vec_short_HP,p_dm_vec_direct_short,p_gw_vec_direct_short,p_dm_vec_short_mssa_direct,p_gw_vec_short_mssa_direct)
  gw_dm_out_mat<-cbind(p_dm_vec_out,p_gw_vec_out,p_dm_vec_HP,p_gw_vec_out_HP,p_dm_vec_direct_out,p_gw_vec_direct_out,p_dm_vec_out_mssa_direct,p_gw_vec_out_mssa_direct)
  gw_dm_all_mat<-cbind(p_dm_vec,p_gw_vec,p_dm_vec_HP,p_gw_vec_HP,p_dm_vec_direct,p_gw_vec_direct,p_dm_vec_mssa_direct,p_gw_vec_mssa_direct)
  colnames(gw_dm_short_mat)<-colnames(gw_dm_out_mat)<-colnames(gw_dm_all_mat)<-c("DM M-SSA/BIP","GW M-SSA/BIP","DM M-SSA/HP-BIP","GW M-SSA/HP-BIP","DM direct","GW direct","DM M-SSA vs. direct","GW M-SSA vs. direct")
  
  
  
  
  return(list(mat_all=mat_all,mat_short=mat_short,mat_out=mat_out,indicator_cal=indicator_cal,mat_all_direct=mat_all_direct,mat_short_direct=mat_short_direct,mat_out_direct=mat_out_direct,direct_pred_mat=direct_pred_mat,
              direct_pred_mat_short=direct_pred_mat_short,direct_pred_mat_out=direct_pred_mat_out,gw_dm_all_mat=gw_dm_all_mat,gw_dm_short_mat=gw_dm_short_mat,gw_dm_out_mat=gw_dm_out_mat))
  
}






# Compute rRMSE and t-statistics OLS and GLS, full sample and without Pandemic
comp_perf_func<-function(dat_m,dat_apply_reg=NULL)
{
  # rRMSE and t-statistics full data: OLS and GLS  
  lm_obj<-lm(dat_m[,1]~dat_m[,2:ncol(dat_m)])
  summary(lm_obj)
  # This one replicates std in summary
  sd<-sqrt(diag(vcov(lm_obj)))
  # Here we use HAC  
  sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
  # This is the same as
  sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
  
  sumfm1 <- summary(lm_obj)
  # Compute direct predictor: apply to full data set dat_apply_reg to obtain predictor up to sample end 
  if (!is.null(dat_apply_reg))
  {
    if (length(sumfm1$coef[2:ncol(dat_m)])==1)
    {
# Explanatory is in second column      
      direct_pred<-sumfm1$coef[1]+dat_apply_reg[,2]*sumfm1$coef[2:nrow(sumfm1$coef),1]
    } else
    {
# Explanatories in columns 2,3,...      
      direct_pred<-sumfm1$coef[1]+dat_apply_reg[,2:ncol(dat_apply_reg)]%*%sumfm1$coef[2:nrow(sumfm1$coef),1]
    }
# Compute residuals on full data set or on out-of-sample data set    
    res<-dat_apply_reg[,1]-direct_pred
  } else
  {
    direct_pred<-NULL
# Use in-sample residuals    
    res<-lm_obj$res
  }
  # Extract maximum t-value of explanatory variables: if maximum is insignificant then regression is weak  
  max_t_ols<-(abs(summary(lm_obj)$coef[1+1:(ncol(dat_m)-1),4]))
  # Use HAC  
  t_HAC<-summary(lm_obj)$coef[1+1:(ncol(dat_m)-1),1]/sd_HAC[2:length(sd)]
  # p-value: take minimum  
  min_t_ols<-min(pt(t_HAC, nrow(dat_m)-ncol(dat_m), lower=FALSE))
  mat_all<-c(sqrt(mean(res^2))/sd(dat_m[,1]),min_t_ols)
  
  
  return(list(mat_all=mat_all,direct_pred=direct_pred))
  
  
}



pcompute_DM_GW_statistics<-function(dat_m,i)#end_date<-3000
{
  #mat<-dat_m_short
  mat<-dat_m
  tail(mat)
  shift<-i
  # Calibration: compute direct forecast    
  lm_obj<-lm(mat[,1]~mat[,2:ncol(mat)])
  summary(lm_obj)
  # Calibrate indicator: compute direct predictor
  coef<-lm(mat[,1]~mat[,2:ncol(mat)])$coef
  # Some R finickery...    
  if (ncol(mat)==2)
  {
    cal_ind<-coef[1]+(mat[,2:ncol(mat)])*coef[2:length(coef)]
  } else
  {
    cal_ind<-coef[1]+(mat[,2:ncol(mat)])%*%coef[2:length(coef)]
  }
  # predictor
  P1<-cal_ind
  # Benchmark against which P1 is evaluated: mean of first column
  P2<-rep(mean(mat[,1]),length(P1))
  P<-cbind(P1,P2)
  
  # Target: forward-shifted observations
  y.real<-mat[,1]
  # Plot; animated figure when looping through shifts    
  ts.plot((cbind(y.real,P1)))
  abline(h=mean(y.real))
  summary(lm(y.real~P1))
  # Tests
  # 1. Giacomini White 2006
  # One-sided alternative (mean performs worse)
  alternative<-"less"
  # Estimator of variance   
  method_vec<-c("HAC","NeweyWest","Andrews","LumleyHeagerty")
  method = method_vec[1]
  
  gw_obj<-gw.test(x = P[,1], y = P[,2], p = y.real,  method = method, alternative = alternative)
  
  statistic<-gw_obj$statistic
  # Empirical significance level 
  pnorm(statistic)
  
  p_gw<-gw_obj$p.value
  # Check    
  if (F)
  {
    statistic
    mean((P1-y.real)^2-(P2-y.real)^2)/gw_obj$ds
    mean((P1-y.real)^2-(P2-y.real)^2)
    gw_obj$ds
    sd((P1-y.real)^2)
    
    sqrt(mean((P1-y.real)^2)/mean((P2-y.real)^2))
    sqrt(mean((P1-y.real)^2))/sd(y.real)
  }  
  
  
  # 2. Diebold Mariano    
  loss.type="SE" 
  # One-sided tests (of course specification of alternative is just opposite... )    
  H1<-"more"
  # c=T in call means finite/small sample correction
  dm_obj<-DM.test(f1=P[,1],f2=P[,2],y=y.real,loss=loss.type,h=shift,c=T,H1=H1)
  
  p_dm<-dm_obj$p.value
  return(list(p_dm=p_dm,p_gw=p_gw))
}  




gw.test <-function(x,y,p,method=c("HAC","NeweyWest","Andrews","LumleyHeagerty"), alternative=c("two.sided","less","greater"))
{
  #x<-P1, y<-P2, p<-y.real
  
  if (is.matrix(x) && ncol(x) > 2) 
    stop("multivariate time series not allowed")
  if (is.matrix(y) && ncol(y) > 2) 
    stop("multivariate time series not allowed")
  if (is.matrix(p) && ncol(p) > 2) 
    stop("multivariate time series not allowed")
  
  # x: predictions model 1
  # y: predictions model 2
  # p: observations
  # T: sample total size
  # tau: forecast horizon
  # method: if tau=1, method=NA. if tau>1, methods
  # alternative: "two.sided","less","greater"
  
  if(NCOL(x) > 1) stop("x is not a vector or univariate time series")     
  if(length(x) != length(y)) stop("size of x and y difier")
  
  alternative <- match.arg(alternative)     
  DNAME <- deparse(substitute(x)) 
  
  l1=abs(x-p)^2
  l2=abs(y-p)^2
  dif=l1-l2
  q=length(dif)
  delta=mean(dif)
  mod <- lm(dif~rep(1,q))
  # Exclude this case    
  if (F)
  {
    if(tau==1){
      re=summary(mod)
      STATISTIC = re$coefficients[1,3]
      if (alternative == "two.sided") PVAL <- 2 * pnorm(-abs(STATISTIC))
      else if (alternative == "less") PVAL <- round(pnorm(STATISTIC),4)
      else if (alternative == "greater") PVAL <- round(pnorm(STATISTIC, lower.tail = FALSE),4)     
      names(STATISTIC) <- "Normal Standad"
      METHOD <- "Standard Statistic Simple Regression Estimator" 
    }
  }
  
  #    if(tau>1){
  # Always use HAC      
  if(method=="HAC"){ 
    METHOD <- "HAC Covariance matrix Estimation"
    ds=sqrt(vcovHAC(mod)[1,1])
  }
  if(method=="NeweyWest"){ 
    METHOD <- "Newey-West HAC Covariance matrix Estimation"
    ds=sqrt(NeweyWest(mod,tau)[1,1])
  }
  if(method=="LumleyHeagerty"){ 
    METHOD <- "Lumley HAC Covariance matrix Estimation"
    ds=sqrt(weave(mod)[1,1])
  }
  if(method=="Andrews"){ 
    METHOD <- "kernel-based HAC Covariance matrix Estimator"
    ds=sqrt(kernHAC(mod)[1,1])
  }
  #STATISTIC = sqrt(n)*delta/ds
  STATISTIC = delta/ds
  if (alternative == "two.sided") PVAL <- 2 * pnorm(-abs(STATISTIC))     
  else if (alternative == "less") PVAL <- pnorm(STATISTIC)
  else if (alternative == "greater") PVAL <- pnorm(STATISTIC, lower.tail = FALSE)     
  names(STATISTIC) <- "Normal Standard"
  #    }
  structure(list(statistic = STATISTIC, alternative = alternative,ds=ds,p.value = PVAL, method = METHOD, data.name = DNAME))
}






# Same as above but we test M-SSA against direct predictor (instead of mean(BIP))
# Small p-values indicate outperformance of M-SSA
pcompute_DM_GW_statistics_MSSA_against_Direct<-function(dat_m,mssa,i)#end_date<-2019
{
  # Compute direct predictors: the predictors in direct_pred or direct_pred_short were not computed for all forecast horizons
  # Therefore we recompute the direct predictor for each forecast horizon in shift_vec
  # Set-up data for direct predictors 
  #  mssa<-mssa_short
  #  mat<-dat_m_short
  mat<-dat_m
  tail(mat)
  shift<-i
  # Calibrate   
  lm_obj<-lm(mat[,1]~mat[,2:ncol(mat)])
  summary(lm_obj)
  # Calibrate direct predictor
  coef<-lm_obj$coef
  if (length(coef)==1)
  {
    cal_dir<-coef[1]+coef[2]*mat[,2]
  } else
  {
    cal_dir<-coef[1]+mat[,2:ncol(mat)]%*%coef[2:length(coef)]
  }
  length(cal_dir)
  mean((mat[,1]-cal_dir)^2)
  mse_mssa<-mean(lm_obj$res^2)
  
  
  # 2. M-SSA 
  mat<-cbind(mat[,1],mssa)
  tail(mat)
  head(mat)
  nrow(mat)
  lm_obj<-lm(mat[,1]~mat[,2])
  summary(lm_obj)
  # Calibrate M-SSA 
  coef<-lm_obj$coef
  cal_mssa<-coef[1]+coef[2]*mat[,2]
  
  
  # M-SSA predictor
  P1<-cal_mssa
  P2<-cal_dir
  # Compare M-SSA against Direct: small p-values indicate outperformnce of M-SSA
  P<-na.exclude(cbind(P1,P2))
  
  y.real<-mat[,1]
  length(y.real)
  length(P1)
  
  ts.plot((cbind(y.real,P[,1])))
  abline(h=mean(y.real))
  summary(lm(y.real~P[,1]))
  
  # H0: P1 is less good than mean
  alternative<-"less"
  method = "HAC"
  tau<-shift
  
  gw_obj<-gw.test(x = P[,1], y = P[,2], p = y.real, method = method, alternative = alternative)
  p_gw_vec<-gw_obj$p.value
  
  if (F)
  {
    gw_obj$statistic
    mean((P1-y.real)^2-(P2-y.real)^2)/gw_obj$ds
    gw_obj$ds
    sd((P1-y.real)^2)
    
    
  }  
  
  
  loss.type="SE" 
  #DM.test(f1=P[,1],f2=P[,2],y=y.real,loss=loss.type,h=1,c=FALSE,H1="more")
  # c=T means finite/small sample correction
  dm_obj<-DM.test(f1=P[,1],f2=P[,2],y=y.real,loss=loss.type,h=shift,c=T,H1="more")
  #    dm_obj<-DM.test(f1=P[,1],f2=P[,2],y=y.real,loss=loss.type,h=1,c=T,H1="more")
  p_dm_vec<-dm_obj$p.value
  
  return(list(p_dm_vec=p_dm_vec,p_gw_vec=p_gw_vec))
}  



