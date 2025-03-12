

# The function returns rMSE, d t-statistics of regressions of predictors on shifted BIP and DM/GW statistics
#   -tstatistics are based on OLS and GLS (assuming AR(1) process for residuals)
#   -rRMSE is based on OLS residuals, references against the mean (of BIP) as benchmark
compute_all_perf_func<-function(indicator_cal,data,lag_vec,h_vec,h,select_direct_indicator,L,lambda_HP)
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
  p_dm_vec<-p_gw_vec<-p_dm_vec_short<-p_gw_vec_short<-NULL
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
    # Same but data prior Pandemic 
    ind<-which(rownames(dat_m)<=2019)
    enfh<-ind[length(ind)]
    dat_m_short<-dat_m[1:enfh,]
    tail(dat_m_short)
    
    comp_obj<-comp_perf_func(dat_m_short)
    
    if (j==1)
    {
      mat_short<-comp_obj$mat_all
    } else
    {
      mat_short<-rbind(mat_short,comp_obj$mat_all)
    }
    
    # Compute DM and GW tests    
    DM_GW_obj<-pcompute_DM_GW_statistics(dat_m_short,i)
    
    p_dm=DM_GW_obj$p_dm
    p_gw=DM_GW_obj$p_gw
    p_dm_vec_short<-c(p_dm_vec_short,p_dm)
    p_gw_vec_short<-c(p_gw_vec_short,p_gw)
  }
  
  names(p_dm_vec)<-names(p_gw_vec)<-names(p_dm_vec_short)<-names(p_gw_vec_short)<-paste("shift=",h_vec,sep="")
  
  mat_all<-matrix(mat_all,nrow=length(h_vec))
  mat_short<-matrix(mat_short,nrow=length(h_vec))
  colnames(mat_all)<-colnames(mat_short)<-c("rRMSE","t-stat OLS")
  rownames(mat_all)<-rownames(mat_short)<-paste("Agg-BIP h=",h,": shift ",h_vec,sep="")
  
  mat_all
  mat_short 
  p_dm_vec_short
  p_gw_vec_short
  p_dm_vec
  p_gw_vec
  
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
  p_dm_vec_HP<-p_gw_vec_HP<-p_dm_vec_short_HP<-p_gw_vec_short_HP<-NULL
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
    # Same but data prior Pandemic 
    ind<-which(rownames(dat_m)<=2019)
    enfh<-ind[length(ind)]
    dat_m_short<-dat_m[1:enfh,]
    tail(dat_m_short)
    
    comp_obj<-comp_perf_func(dat_m_short)
    
    if (j==1)
    {
      mat_short_HP<-comp_obj$mat_all
    } else
    {
      mat_short_HP<-rbind(mat_short_HP,comp_obj$mat_all)
    }
    
    # Compute DM and GW tests    
    DM_GW_obj<-pcompute_DM_GW_statistics(dat_m_short,i)
    
    p_dm=DM_GW_obj$p_dm
    p_gw=DM_GW_obj$p_gw
    p_dm_vec_short_HP<-c(p_dm_vec_short_HP,p_dm)
    p_gw_vec_short_HP<-c(p_gw_vec_short_HP,p_gw)
  }
  
  names(p_dm_vec_HP)<-names(p_gw_vec_HP)<-names(p_dm_vec_short_HP)<-names(p_gw_vec_short_HP)<-paste("shift=",h_vec,sep="")
  
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
  
  
  direct_pred_mat<-p_dm_vec_direct<-p_gw_vec_direct<-p_dm_vec_direct_short<-p_gw_vec_direct_short<-NULL
  
  # C.3 Compute direct predictor up to sample end and derive performances for all shifts
  for (j in 1:length(h_vec))#i<-0
  {
    # Forward shift
    i<-h_vec[j]
    # Target (first column) is shifted by i
    dat_m<-cbind(dat_mh[(1+i):nrow(dat_mh),1],dat_mh[1:(nrow(dat_mh)-i),2:ncol(dat_mh)])
    tail(dat_m)
    # We apply regression to full sample to obtain predictor values up to the sample end    
    dat_apply_reg<-dat_mh[,2:ncol(dat_mh)]
    
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
    
    # We apply regression to full sample to obtain predictor values up to the sample end    
    dat_apply_reg<-dat_mh[,2:ncol(dat_mh)]
    
    # Compute direct forecast, t-tests and rRMSE
    comp_obj<-comp_perf_func(dat_m_short,dat_apply_reg)
    
    if (j==1)
    {
      mat_short_direct<-comp_obj$mat_all
      direct_pred_mat_short=comp_obj$direct_pred
    } else
    {
      mat_short_direct<-rbind(mat_short_direct,comp_obj$mat_all)
      direct_pred_mat_short=cbind(direct_pred_mat_short,comp_obj$direct_pred)
    }
    
    # Compute DM and GW tests    
    DM_GW_obj<-pcompute_DM_GW_statistics(dat_m_short,i)
    
    p_dm=DM_GW_obj$p_dm
    p_gw=DM_GW_obj$p_gw
    p_dm_vec_direct_short<-c(p_dm_vec_direct_short,p_dm)
    p_gw_vec_direct_short<-c(p_gw_vec_direct_short,p_gw)
    
  }
  names(p_dm_vec_direct)<-names(p_gw_vec_direct)<-names(p_dm_vec_direct_short)<-names(p_gw_vec_direct_short)<-paste("shift=",h_vec,sep="")
  
  colnames(direct_pred_mat)<-paste("Direct AR predictor: h=",h_vec,sep="")
  rownames(direct_pred_mat)<-rownames(dat_mh)[(nrow(dat_mh)-nrow(direct_pred_mat)+1):nrow(dat_mh)]
  mat_all_direct<-matrix(mat_all_direct,nrow=length(h_vec))
  mat_short_direct<-matrix(mat_short_direct,nrow=length(h_vec))
  colnames(mat_all_direct)<-colnames(mat_short_direct)<-c("rRMSE","t-stat OLS")
  rownames(mat_all_direct)<-rownames(mat_short_direct)<-paste("Shift=",h_vec,sep="")
  
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
  p_dm_vec_mssa_direct<-p_gw_vec_mssa_direct<-p_dm_vec_short_mssa_direct<-p_gw_vec_short_mssa_direct<-NULL
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
    
    # Compute DM and GW tests    
    # Compute DM and GW tests    
    DM_GW_obj<-pcompute_DM_GW_statistics_MSSA_against_Direct(dat_m_short,mssa_short,i)#end_date<-2019
    
    p_dm=DM_GW_obj$p_dm
    p_gw=DM_GW_obj$p_gw
    p_dm_vec_short_mssa_direct<-c(p_dm_vec_short_mssa_direct,p_dm)
    p_gw_vec_short_mssa_direct<-c(p_gw_vec_short_mssa_direct,p_gw)
  }
  
  names(p_dm_vec_mssa_direct)<-names(p_gw_vec_mssa_direct)<-names(p_dm_vec_short_mssa_direct)<-names(p_gw_vec_short_mssa_direct)<-paste("shift=",h_vec,sep="")
  
  p_dm_vec_short_mssa_direct
  p_gw_vec_short_mssa_direct
  
  #---------------------------
  # Collect DM and GW statistics in matrices
  gw_dm_short_mat<-cbind(p_dm_vec_short,p_gw_vec_short,p_dm_vec_HP,p_gw_vec_short_HP,p_dm_vec_direct_short,p_gw_vec_direct_short,p_dm_vec_short_mssa_direct,p_gw_vec_short_mssa_direct)
  gw_dm_all_mat<-cbind(p_dm_vec,p_gw_vec,p_dm_vec_HP,p_gw_vec_HP,p_dm_vec_direct,p_gw_vec_direct,p_dm_vec_mssa_direct,p_gw_vec_mssa_direct)
  colnames(gw_dm_short_mat)<-colnames(gw_dm_all_mat)<-c("DM M-SSA/BIP","GW M-SSA/BIP","DM M-SSA/HP-BIP","GW M-SSA/HP-BIP","DM direct","GW direct","DM M-SSA vs. direct","GW M-SSA vs. direct")
  
  
  
  
  return(list(mat_all=mat_all,mat_short=mat_short,indicator_cal=indicator_cal,mat_all_direct=mat_all_direct,mat_short_direct=mat_short_direct,direct_pred_mat=direct_pred_mat,direct_pred_mat_short=direct_pred_mat_short,gw_dm_all_mat=gw_dm_all_mat,gw_dm_short_mat=gw_dm_short_mat))
  
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
      direct_pred<-sumfm1$coef[1]+dat_apply_reg*sumfm1$coef[2:nrow(sumfm1$coef),1]
    } else
    {
      direct_pred<-sumfm1$coef[1]+dat_apply_reg%*%sumfm1$coef[2:nrow(sumfm1$coef),1]
    }
  } else
  {
    direct_pred<-NULL
  }
  # Extract maximum t-value of explanatory variables: if maximum is insignificant then regression is weak  
  max_t_ols<-(abs(summary(lm_obj)$coef[1+1:(ncol(dat_m)-1),4]))
  # Use HAC  
  t_HAC<-summary(lm_obj)$coef[1+1:(ncol(dat_m)-1),1]/sd_HAC[2:length(sd)]
  # p-value: take minimum  
  min_t_ols<-min(2*pt(t_HAC, nrow(dat_m)-ncol(dat_m), lower=FALSE))
  mat_all<-c(sqrt(mean(lm_obj$res^2))/sd(dat_m[,1]),min_t_ols)
  
  
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



