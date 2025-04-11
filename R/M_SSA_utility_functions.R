
# Compute target: two-sided HP
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



# Compute MA-inversion as based on VAR model
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

# Compute M-SSA (and accessory M-MSE)
MSSA_main_func<-function(delta,ht_vec,xi,symmetric_target,gamma_target,Sigma,Plot=F)
{
  L<-dim(gamma_target)[1]/dim(Sigma)[1]
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


# Filter function: apply filters to data
# data: x_mat
# M-SSA: bk_x_mat
# M-MSE: gammak_x_mse
# Acausal target: gamma_target
# Forward-shift of acausal target: delta
# Mirror left half of Two-sided filter to the right to obtain symmetric two-sided filter: symmetric_target==T
# Returns:
# M-SSA output: mssa_mat
# M-MSE output: mmse_mat
# Target filter output: target_mat
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




# This function operationalizes the M-SSA concept for predicting quarterly (German) GDP
# It relies on hyperparameters specifying the design: lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess
# It returns M-SSA and M-MSE predictors as well as forward-shifted HP-BIP (two-sided HP applied to BIP)
compute_mssa_BIP_predictors_func<-function(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)
{
# 1. Compute target
  
  target_obj<-HP_target_sym_T(n,lambda_HP,L)
  
  gamma_target=t(target_obj$gamma_target)
  symmetric_target=target_obj$symmetric_target 
  colnames(gamma_target)<-select_vec_multi
#-------------------------
# 2. Fit  VAR on specified in-sample span
  data_fit<-na.exclude(x_mat[which(rownames(x_mat)<date_to_fit),])#date_to_fit<-"2019-01-01"
  set.seed(12)
  V_obj<-VARMA(data_fit,p=p,q=q)
# Apply regularization: see vignette MTS package
  threshold<-1.5
  V_obj<-refVARMA(V_obj, thres = threshold)
  
  Sigma<-V_obj$Sigma
  Phi<-V_obj$Phi
  Theta<-V_obj$Theta
  
#---------------------------------------
# 3. MA inversion: M-SSA relies on MA-inversion of VAR
  
  MA_inv_obj<-MA_inv_VAR_func(Phi,Theta,L,n,T)
  
  xi<-MA_inv_obj$xi

#-----------------------
# 4. Compute M-SSA for the specified forecast horizons in h_vec
# Initialize array: first dimension=target series, second dimension=time, third dimension=forecast horizon  
  mssa_array<-mmse_array<-array(dim=c(length(select_vec_multi),dim(x_mat)[1],length(h_vec)),
                                dimnames=list(select_vec_multi,rownames(x_mat),paste("h=",h_vec,sep="")))
# Loop over all explanatory variables 
#   -We need to differenciate the series because lag_vec and or f_excess can vary depending of the series 
#   -If lag_vec and f_excess were both fixed, then we compute the M-SSA predictors in a single run
  for (ijk in 1:length(select_vec_multi))#ijk<-1
  {
# Loop over forecast horizons    
    for (i in 1:length(h_vec))#i<-1
    {
# For each series ijk, the forecast horizon delta applöied by M-SSA is the sum of h_vec[i], 
#       publication lag and forecast excess
      delta<-h_vec[i]+lag_vec[ijk]+f_excess[ijk]
# M-SSA  
      MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)
      
      bk_x_mat=MSSA_main_obj$bk_x_mat
      MSSA_obj=MSSA_main_obj$MSSA_obj 
      gammak_x_mse=MSSA_obj$gammak_x_mse
      colnames(bk_x_mat)<-select_vec_multi
      
# Apply filters to data
      filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
# We extract series ijk only from the filter object and assign to element ijk,,i of the array
#   ijk (first dimension) is the series and i (3-rd dimension) is the forecast horizon of the M-SSA predictor
# The forecast horizon is h_vec[i] (looking for BIP h_vec[i] steps away) but we generally impose delta>h_vec[i], 
#   because of publication lag and/or forecast excess (the latter is justified in exercise 2 of tutorial 7.2: VAR misspecification)  
      mssa_array[ijk,,i]=filt_obj$mssa_mat[,ijk]
      target_mat=filt_obj$target_mat
      mmse_array[ijk,,i]<-filt_obj$mmse_mat[,ijk]
    }
  }
  
# 5. Compute M-SSA predictors
  
  predictor_mssa_mat<-predictor_mmse_mat<-0*mssa_array[1,,]
  for (i in 1:length(select_vec_multi))#i<-1
  {
#   Standardize and aggregate (sum over all series): equal weighting
    predictor_mssa_mat<-predictor_mssa_mat+scale(mssa_array[i,,])
    predictor_mmse_mat<-predictor_mmse_mat+scale(mmse_array[i,,])
  }
  predictor_mssa_mat<-predictor_mssa_mat/length(select_vec_multi)
  predictor_mmse_mat<-predictor_mmse_mat/length(select_vec_multi)
  colnames(predictor_mssa_mat)<-colnames(predictor_mmse_mat)<-dimnames(mssa_array)[[3]]
  rownames(predictor_mssa_mat)<-rownames(predictor_mmse_mat)<-dimnames(mssa_array)[[2]]
  
#-----------------------------
# 6. Compute plots
  target_shifted_mat<-NULL
  cor_mat_HP_BIP_full_sample<-cor_mat_HP_BIP_out_of_sample<-matrix(nrow=length(h_vec),ncol=length(h_vec))
  for (i in 1:length(h_vec))#i<-1
  {
    shift<-h_vec[i]+lag_vec[1]
# Compute target: two-sided HP applied to BIP and shifted forward by forecast horizon plus publication lag
    filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,shift)
    target_mat=filt_obj$target_mat
# Select HP-BIP (first column)  
    target<-target_mat[,"BIP"]
# Collect the forward shifted targets: 
    target_shifted_mat<-cbind(target_shifted_mat,target)
# Plot indicators and shifting target
    mplot<-scale(cbind(target,predictor_mssa_mat))
    colnames(mplot)[1]<-paste("Target left-shifted by ",shift-lag_vec[1],sep="")
    par(mfrow=c(1,1))
    colo<-c("black",rainbow(ncol(predictor_mssa_mat)))
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
    
  }
  return(list(target_shifted_mat=target_shifted_mat,predictor_mssa_mat=predictor_mssa_mat,predictor_mmse_mat=predictor_mmse_mat,mssa_array=mssa_array,mmse_array=mmse_array))
}  
  




# Compute sample performances:
# Correlations with forward-shifted HP-BIP and BIP
# HAC adjusted p-Values of regressions of predictors on forward-shifted HP-BIP and BIP
# Full sample and out-of-sample: out-of-sample is based on date_to_fit (in-sample span for estimating VAR of M-SSA)
compute_perf_func<-function(x_mat,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,date_to_fit,select_direct_indicator,h_vec) 
{
# 1. Compute Correlations and HAC-adjusted p-value of-one-sided test when regressing predictor on target
# 1.1 Target is forward-shifted HP-BIP
# Check forward-shifts: HP-BIP cannot reach sample-end since filter is two-sided
  tail(target_shifted_mat,40)
  t_HAC_HP_BIP_full<-p_value_HAC_HP_BIP_full<-t_HAC_HP_BIP_oos<-p_value_HAC_HP_BIP_oos<-cor_mat_HP_BIP_full<-cor_mat_HP_BIP_oos<-matrix(ncol=length(h_vec),nrow=length(h_vec)-1)
  for (i in 1:(length(h_vec)-1))# i<-1
  {
    for (j in 1:length(h_vec))# j<-1  j<-3
    {
# Remove NAs      
      da<-na.exclude(cbind(target_shifted_mat[,i],predictor_mssa_mat[,j]))
# Compute HAC-adjusted p-value of-one-sided test when regressing column 2 (predictor) on column 1 (target) of da
      p_obj<-HAC_ajusted_p_value_func(da)
      p_value_HAC_HP_BIP_full[i,j]<-p_obj$p_value
# The corresponding t-statistic      
      t_HAC_HP_BIP_full[i,j]=p_obj$t_stat
      
# Compute target correlations 
      cor_mat_HP_BIP_full[i,j]<-cor(da)[1,2]
      
# Same but out of sample 
      oos_index<-which(rownames(predictor_mssa_mat)>date_to_fit)
      da<-na.exclude(cbind(target_shifted_mat[,i],predictor_mssa_mat[,j])[oos_index,])
      
      # Compute HAC-adjusted p-value of-one-sided test when regressing column 2 (predictor) on column 1 (target) of da
      p_obj<-HAC_ajusted_p_value_func(da)
      
      p_value_HAC_HP_BIP_oos[i,j]<-p_obj$p_value
      t_HAC_HP_BIP_oos[i,j]=p_obj$t_stat
      
      # Compute target correlations 
      cor_mat_HP_BIP_oos[i,j]<-cor(da)[1,2]
      
    }
  }
  colnames(p_value_HAC_HP_BIP_full)<-colnames(t_HAC_HP_BIP_full)<-colnames(cor_mat_HP_BIP_full)<-
  colnames(p_value_HAC_HP_BIP_oos)<-colnames(t_HAC_HP_BIP_oos)<-colnames(cor_mat_HP_BIP_oos)<-paste("M-SSA: h=",h_vec,sep="")
  rownames(p_value_HAC_HP_BIP_full)<-rownames(t_HAC_HP_BIP_full)<-rownames(cor_mat_HP_BIP_full)<-
  rownames(p_value_HAC_HP_BIP_oos)<-rownames(t_HAC_HP_BIP_oos)<-rownames(cor_mat_HP_BIP_oos)<-paste("Shift of target: ",h_vec[-length(h_vec)],sep="")
  
# 1.2 Target is forward-shifted BIP
  t_HAC_BIP_full<-p_value_HAC_BIP_full<-t_HAC_BIP_oos<-p_value_HAC_BIP_oos<-cor_mat_BIP_full<-cor_mat_BIP_oos<-matrix(ncol=length(h_vec),nrow=length(h_vec)-1)
  BIP_target_mat<-NULL
  for (i in 1:(length(h_vec)-1))# i<-1
  {
# Shift BIP forward by publication lag and forecast horizon  
    shift<-h_vec[i]+lag_vec[1]
    BIP_target<-c(x_mat[(1+shift):nrow(x_mat),"BIP"],rep(NA,shift))
# Collect all forward shifted series    
    BIP_target_mat<-cbind(BIP_target_mat,BIP_target)
    rownames(BIP_target_mat)<-rownames(x_mat)
# Check: shifts of target starting with publication-lag to the left and increasing upward-shifts to the right    
    tail(BIP_target_mat)
# Regress predictors on shifted BIP  
    for (j in 1:length(h_vec))# j<-1
    {
# Remove NAs      
      da<-na.exclude(cbind(BIP_target,predictor_mssa_mat[,j]))
# Compute HAC-adjusted p-value of-one-sided test when regressing column 2 (predictor) on column 1 (target) of da
      p_obj<-HAC_ajusted_p_value_func(da)
      
      p_value_HAC_BIP_full[i,j]<-p_obj$p_value
      t_HAC_BIP_full[i,j]=p_obj$t_stat
      
# Compute target correlations 
      cor_mat_BIP_full[i,j]<-cor(da)[1,2]
# Same but out of sample 
      oos_index<-which(rownames(predictor_mssa_mat)>date_to_fit)
      da<-na.exclude(cbind(BIP_target,predictor_mssa_mat[,j])[oos_index,])
      # Compute HAC-adjusted p-value of-one-sided test when regressing column 2 (predictor) on column 1 (target) of da
      p_obj<-HAC_ajusted_p_value_func(da)
      
      p_value_HAC_BIP_oos[i,j]<-p_obj$p_value
      t_HAC_BIP_oos[i,j]=p_obj$t_stat
      
      # Compute target correlations 
      cor_mat_BIP_oos[i,j]<-cor(da)[1,2]
      
      
    }
  }
  colnames(p_value_HAC_BIP_full)<-colnames(t_HAC_BIP_full)<-colnames(cor_mat_BIP_full)<-
  colnames(p_value_HAC_BIP_oos)<-colnames(t_HAC_BIP_oos)<-colnames(cor_mat_BIP_oos)<-paste("M-SSA: h=",h_vec,sep="")
  rownames(p_value_HAC_BIP_full)<-rownames(t_HAC_BIP_full)<-rownames(cor_mat_BIP_full)<-
  rownames(p_value_HAC_BIP_oos)<-rownames(t_HAC_BIP_oos)<-rownames(cor_mat_BIP_oos)<-paste("Shift of target: ",h_vec[-length(h_vec)],sep="")
  
# 2. Compute Direct predictors
# We use full sample direct predictors
# Idea: calibration
#   -Static level and scale adjustments are ignored in M-SSA (objective function relies on target correlation)
#     -Emphasize dynamic aspects of prediction problem
#   -We proceed similarly for rRMSE
# -Direct predictors: 
#   -Regressing the macro-indicators to full-sample target means an implicit calibration of level and scale
# -M-SSA:
#   -Per construction (equal-weighting) the M-SSA predictors are standardized
#   -MSE-performances: calibrate level and scale to full-sample target (similar to direct predictors)  

# Direct predictor: natural target is forward-shifted BIP  
  p_val_direct_mat<-matrix(nrow=length(h_vec),ncol=1)
  direct_pred_mat<-NULL
  for (i in 1:(length(h_vec)))#i<-1
  { 
    shift<-h_vec[i]
# Target: first column is forward-shifted BIP    
    dat<-cbind(c(x_mat[(lag_vec[1]+1+shift):nrow(x_mat),1],rep(NA,lag_vec[1]+shift)),x_mat[,select_direct_indicator])
    tail(dat,9)
    dat<-na.exclude(dat)
    n<-dim(dat)[2]-1
# Compute calibrated out-of-sample predictor, based on expanding window
#   -Use data up i for fitting the regression
#   -Compute a prediction with explanatory data in i+1
# Fit model    
    lm_obj<-lm(dat[,1]~dat[,2:(n+1)])
    if (n==1)
    {
# Classic regression prediction        
      direct_pred<-c(rep(NA,shift+lag_vec[1]),lm_obj$coef[1]+lm_obj$coef[2]*dat[,2])
    } else
    {
# Classic regression prediction though we use %*% instead of * above  
      direct_pred<-c(rep(NA,shift+lag_vec[1]),lm_obj$coef[1]+lm_obj$coef[2]%*%dat[,2])
    }
    direct_pred_mat<-cbind(direct_pred_mat,direct_pred)
    
# Compute the HAC-adjusted p-values of the regression of the predictor on the target, out-of-sample  
    lm_obj<-lm(dat[,1]~na.exclude(direct_pred))
    
#    ts.plot(cbind(dat[,1],direct_pred))
    summary(lm_obj)
# Compute classic and HAC-adjusted standard errors    
    sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
    sd_ols<-sqrt(diag(vcov(lm_obj)))
# Compute max of both: we rely on this estimate because HAC-adjustment is sometimes inconsistent (too small: maybe issue with R-package sandwich)    
    sd_max<-max(sd_HAC[2],sd_ols[2])
# Rely on max vola    
    t_HAC<-summary(lm_obj)$coef[2,1]/sd_max
# One-sided test: if predictor is effective, then the sign of the coefficient must be positive  
    HAC_p_value<-pt(t_HAC, nrow(dat)-2, lower=FALSE)
    if (F)
    {
# Classic OLS p-values      
      sd_ols<-sqrt(diag(vcov(lm_obj)))
      t_ols<-summary(lm_obj)$coef[2,1]/sd_ols[2]
      OLS_p_value<-pt(t_ols, nrow(dat)-2, lower=FALSE)
    }
    p_val_direct_mat[i,1]<-HAC_p_value
  }
  rownames(p_val_direct_mat)<-paste("shift=",h_vec,sep="")
  
# 3. Compute rRMSEs: 
# 3.1 Target is forward-shifted HP-BIP
  rRMSE_MSSA_HP_BIP_direct<-rRMSE_MSSA_HP_BIP_mean<-matrix(nrow=length(h_vec)-1,ncol=length(h_vec))
  for (i in 1:(length(h_vec)-1))#i<-1
  { 
    shift<-h_vec[i]
# Target: first column is forward-shifted HP-BIP  
    for (j in 1:length(h_vec))
    {
      index_oos<-which(rownames(predictor_mssa_mat)>date_to_fit)
      target_HP_BIP<-target_shifted_mat[,i]
      dat<-cbind(target_HP_BIP,predictor_mssa_mat[,j])[index_oos,]
      lm_obj<-lm(dat[,1]~dat[,2])
      RMSE_MSSA<-summary(lm_obj)$sigma
      dat<-cbind(target_HP_BIP,direct_pred_mat[,j])[index_oos,]
      lm_obj<-lm(dat[,1]~dat[,2])
      RMSE_direct<-summary(lm_obj)$sigma
      RMSE_mean<-sd(target_HP_BIP[index_oos],na.rm=T)
      rRMSE_MSSA_HP_BIP_direct[i,j]<-RMSE_MSSA/RMSE_direct  
      rRMSE_MSSA_HP_BIP_mean[i,j]<-RMSE_MSSA/RMSE_mean  
    }
  }
  
# 3.2 Target is forward-shifted BIP
  rRMSE_MSSA_BIP_direct<-rRMSE_MSSA_BIP_mean<-matrix(nrow=length(h_vec)-1,ncol=length(h_vec))
  target_BIP_mat<-NULL
  for (i in 1:(length(h_vec)-1))#i<-1
  { 
    shift<-h_vec[i]
    index_oos<-which(rownames(predictor_mssa_mat)>date_to_fit)
    target_BIP<-c(x_mat[(lag_vec[1]+1+shift):nrow(x_mat),1],rep(NA,lag_vec[1]+shift))
# Chexk forwrad-shift    
    tail(target_BIP,9)
    
    target_BIP_mat<-cbind(target_BIP_mat,target_BIP)
    
    # Target: first column is forward-shifted HP-BIP  
    for (j in 1:length(h_vec))
    {
      dat<-cbind(target_BIP,predictor_mssa_mat[,j])[index_oos,]
      lm_obj<-lm(dat[,1]~dat[,2])
      RMSE_MSSA<-summary(lm_obj)$sigma
      dat<-cbind(target_BIP,direct_pred_mat[,j])[index_oos,]
      lm_obj<-lm(dat[,1]~dat[,2])
      RMSE_direct<-summary(lm_obj)$sigma
      RMSE_mean<-sd(target_BIP[index_oos],na.rm=T)
      rRMSE_MSSA_BIP_direct[i,j]<-RMSE_MSSA/RMSE_direct  
      rRMSE_MSSA_BIP_mean[i,j]<-RMSE_MSSA/RMSE_mean  
    }
  }
  colnames(target_BIP_mat)<-paste("shift=",h_vec[-length(h_vec)],sep="")
  rownames(rRMSE_MSSA_HP_BIP_direct)<-rownames(rRMSE_MSSA_HP_BIP_mean)<-rownames(rRMSE_MSSA_BIP_direct)<-rownames(rRMSE_MSSA_BIP_mean)<-paste("shift=",h_vec[-length(h_vec)],sep="")
  colnames(rRMSE_MSSA_HP_BIP_direct)<-colnames(rRMSE_MSSA_HP_BIP_mean)<-colnames(rRMSE_MSSA_BIP_direct)<-colnames(rRMSE_MSSA_BIP_mean)<-paste("h=",h_vec,sep="")
    

# Older measures without calibration  
  if (F)
  {
    p_val_mat<-matrix(nrow=length(h_vec),ncol=length(h_vec))
    date_to_fit<-"2008"
    for (i in 1:length(h_vec))
    { 
      for (j in 1:length(h_vec))#j<-1
      {
        shift<-h_vec[i]
        dat<-cbind(c(x_mat[(lag_vec[1]+1+shift):nrow(x_mat),1],rep(NA,lag_vec[1]+shift)),predictor_mssa_mat[,j])
        tail(dat,9)
        dat<-na.exclude(dat)
        ts.plot(dat)
        oos_pred_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit)
        
        p_val_mat[i,j]<-oos_pred_obj$HAC_p_value
        
      }
      colnames(p_val_mat)<-paste("h=",h_vec,sep="")
      rownames(p_val_mat)<-paste("shift=",h_vec,sep="")
      
    }
    
    p_val_mat
    
    p_val_mat<-matrix(nrow=length(h_vec),ncol=1)
    for (i in 1:length(h_vec))
    { 
      shift<-h_vec[i]
      dat<-cbind(c(x_mat[(lag_vec[1]+1+shift):nrow(x_mat),1],rep(NA,lag_vec[1]+shift)),x_mat[,select_direct_indicator])
      tail(dat,9)
      dat<-na.exclude(dat)
      ts.plot(dat)
      oos_pred_obj<-compute_calibrated_out_of_sample_predictors_func(dat,date_to_fit)
        
      p_val_mat[i,1]<-oos_pred_obj$HAC_p_value
        
      
    }
    rownames(p_val_mat)<-paste("shift=",h_vec,sep="")
    
    
    p_val_mat
    
    
    
  }
  
  return(list(p_value_HAC_HP_BIP_full=p_value_HAC_HP_BIP_full,t_HAC_HP_BIP_full=t_HAC_HP_BIP_full,
              cor_mat_HP_BIP_full=cor_mat_HP_BIP_full,p_value_HAC_HP_BIP_oos=p_value_HAC_HP_BIP_oos,
              t_HAC_HP_BIP_oos=t_HAC_HP_BIP_oos,cor_mat_HP_BIP_oos=cor_mat_HP_BIP_oos,
              p_value_HAC_BIP_full=p_value_HAC_BIP_full,t_HAC_BIP_full=t_HAC_BIP_full,
              cor_mat_BIP_full=cor_mat_BIP_full,p_value_HAC_BIP_oos=p_value_HAC_BIP_oos,
              t_HAC_BIP_oos=t_HAC_BIP_oos,cor_mat_BIP_oos=cor_mat_BIP_oos,
              rRMSE_MSSA_HP_BIP_direct=rRMSE_MSSA_HP_BIP_direct,rRMSE_MSSA_HP_BIP_mean=rRMSE_MSSA_HP_BIP_mean,
              rRMSE_MSSA_BIP_direct=rRMSE_MSSA_BIP_direct,rRMSE_MSSA_BIP_mean=rRMSE_MSSA_BIP_mean,
              target_BIP_mat=target_BIP_mat))
}



# Compute HAC-adjusted p-value of-one-sided test when regressing column 2 on column 1 of da
HAC_ajusted_p_value_func<-function(da)
{
  
  lm_obj<-lm(da[,1]~da[,2])
  summary(lm_obj)
# This one replicates Std. Error in summary
  sd<-sqrt(diag(vcov(lm_obj)))
# Here we use HAC  
  sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
# This is the same as
  sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
# In some cases the HAC-adjustment is suspect (too small): we select the max of HAC-adjusted and OLS standard errors  
  sd_max<-max(sd[2],sd_HAC[2])
# Classic OLS
  t_stat<-summary(lm_obj)$coef[2,1]/sd[2]
# HAC adjusted  
  t_stat<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
# We noted that the HAC adjustment does not always lead to consistent results (maybe a problem with R-package sandwich)
# In any case we here try to adopt a pragmatic proceeding by using sd_max, the larger of the two variance estimates 
  t_stat<-summary(lm_obj)$coef[2,1]/sd_max
# One-sided test: if regressor is effective, then coefficient must be positive (we are not interested in testing negative readings)     
  p_value<-pt(t_stat, df=nrow(na.exclude(da))-ncol(da), lower=FALSE)
  return(list(p_value=p_value,t_stat=t_stat))
}



compute_calibrated_out_of_sample_predictors_func<-function(dat,date_to_fit)
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
# Classic regression prediction        
        cal_oos_pred[i+1]<-lm_obj$coef[1]+lm_obj$coef[2]*dat[i+1,2] 
    } else
    {
# Classic regression prediction though we use %*% instead of * above      
      cal_oos_pred[i+1]<-lm_obj$coef[1]+lm_obj$coef[2:(n+1)]%*%dat[i+1,2:(n+1)] 
    }
  }
# Once the predictors are computed we can obtain the out-of-sample prediction errors
  epsilon_oos<-dat[,1]-cal_oos_pred
  index_oos<-which(rownames(dat)>date_to_fit)
# And we can compute the HAC-adjusted p-values of the regression of the predictor on the target, out-of-sample  
  lm_oos<-lm(dat[index_oos,1]~cal_oos_pred[index_oos])
  ts.plot(cbind(dat[,1],cal_oos_pred))
  summary(lm_oos)
  sd_HAC<-sqrt(diag(vcovHAC(lm_oos)))
  sd_ols<-sqrt(diag(vcov(lm_oos)))
# Compute max of both vola estimates: HAC-adjustment is not 100% reliable (maybe issue with R-package sandwich)  
  sd_max<-max(sd_ols[2],sd_HAC[2])
  t_HAC<-summary(lm_oos)$coef[2,1]/sd_max
# One-sided test: if predictor is effective, then the sign of the coefficient must be positive (ngetaive signs can be ignored) 
  HAC_p_value<-pt(t_HAC, nrow(dat)-2, lower=FALSE)
  if (F)
  {
# Classic OLS p-values
    sd_ols<-sqrt(diag(vcov(lm_obj)))
    t_ols<-summary(lm_obj)$coef[2,1]/sd_ols[2]
    OLS_p_value<-pt(t_ols, nrow(dat)-2, lower=FALSE)
  }
  p_value<-HAC_p_value
  
  
  return(list(cal_oos_pred=cal_oos_pred,epsilon_oos=epsilon_oos,p_value=p_value,HAC_p_value=HAC_p_value))
}



compute_component_predictors_func<-function(dat,start_fit,use_garch,shift)
{
  
  len<-dim(dat)[1]
# First column is target, i.e. dimension is dim(dat)[2]-1
  n<-dim(dat)[2]-1
# Compute calibrated out-of-sample predictor, based on expanding window
  cal_oos_pred<-cal_oos_mean_pred<-rep(NA,len)
  for (i in (n+2):(len-shift)) #i<-n+2
  {
# If use_garch==T then the regression relies on weighted least-squares, whereby the weights are based 
#   on volatility obtained from a GARCH(1,1) model fitted to target (first column of dat)
    if (use_garch)
    {
      y.garch_11<-garchFit(~garch(1,1),data=dat[1:i,1],include.mean=T,trace=F)
# sigmat could be retrieved from GARCH-object
      sigmat<-y.garch_11@sigma.t
# But this is lagged by one period
# Therefore we recompute the vola based on the estimated GARCH-parameters
      eps<-y.garch_11@residuals
      d<-y.garch_11@fit$matcoef["omega",1]
      alpha<-y.garch_11@fit$matcoef["alpha1",1]
      beta<-y.garch_11@fit$matcoef["beta1",1]
      sigmat_own<-sigmat
      for (i in 2:length(sigmat))#i<-2
        sigmat_own[i]<-sqrt(d+beta*sigmat_own[i-1]^2+alpha*eps[i]^2)
# This is now correct (not lagging anymore)
      sigmat<-sigmat_own
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
# We can compute the HAC-adjusted p-values of the regression of the predictor on the target, out-of-sample  
  lm_oos<-lm(dat[index_oos,1]~cal_oos_pred[index_oos])
# Plot target and out-of-sample predictor  
  ts.plot(cbind(dat[index_oos,1],cal_oos_pred[index_oos]),main=paste("shift=",shift,sep=""))
  summary(lm_oos)
# OLS standard error  
  sd_OLS<-sqrt(diag(vcov(lm_oos)))
# HAC adjustment (R-package sandwich)  
  sd_HAC<-sqrt(diag(vcovHAC(lm_oos)))
  t_HAC<-summary(lm_oos)$coef[2,1]/max(sd_HAC[2],sd_OLS[2])
#  t_HAC<-summary(lm_oos)$coef[2,1]/sd_HAC[2]
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
  
  return(list(cal_oos_pred=cal_oos_pred,epsilon_oos=epsilon_oos,epsilon_mean_oos=epsilon_mean_oos,p_value=p_value,MSE_oos=MSE_oos,MSE_mean_oos=MSE_mean_oos,MSE_mean_oos_without_covid=MSE_mean_oos_without_covid,MSE_oos_without_covid=MSE_oos_without_covid,p_value_without_covid=p_value_without_covid))
}



##################################################################################
# Old code

# This function operationalizes the M-SSA concept for predicting quarterly (German) GDP
# It relies on hyperparameters specifying the design: lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess
# It returns M-SSA and M-MSE predictors as well as forward-shifted HP-BIP (two-sided HP applied to BIP)
compute_mssa_BIP_predictors_func_old<-function(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)
{
  # 1. Compute target
  n<-ncol(x_mat)
  target_obj<-HP_target_sym_T(n,lambda_HP,L)
  
  gamma_target=t(target_obj$gamma_target)
  symmetric_target=target_obj$symmetric_target 
  colnames(gamma_target)<-select_vec_multi
  #-------------------------
  # 2. Fit  VAR on specified in-sample span   date_to_fit<-"2007-12-01"
  data_fit<-na.exclude(x_mat[which(rownames(x_mat)<=date_to_fit),])#date_to_fit<-"2019-01-01"
  set.seed(12)
  V_obj<-VARMA(data_fit,p=p,q=q)
  # Apply regularization: see vignette MTS package
  threshold<-1.5
  V_obj<-refVARMA(V_obj, thres = threshold)
  
  Sigma<-V_obj$Sigma
  Phi<-V_obj$Phi
  Theta<-V_obj$Theta
  
  #---------------------------------------
  # 3. MA inversion: M-SSA relies on MA-inversion of VAR
  
  MA_inv_obj<-MA_inv_VAR_func(Phi,Theta,L,n,T)
  
  xi<-MA_inv_obj$xi
  
  #-----------------------
  # 4. Compute M-SSA for the specified forecast horizons in h_vec
  
  mssa_bip<-mssa_ip<-mssa_esi<-mssa_ifo<-mssa_spread<-mmse_bip<-mmse_ip<-mmse_esi<-mmse_ifo<-mmse_spread<-NULL
  for (i in 1:length(h_vec))#i<-1
  {
    # For each forecast horizon h_vec[i] we compute M-SSA for BIP and ip first, based on the proposed forecast excess
    #   -BIP and ip require a larger forecast excess f_excess[1]. We also add the publication lag
    delta<-h_vec[i]+lag_vec[1]+f_excess[1]
    
    # M-SSA  
    MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)
    
    bk_x_mat=MSSA_main_obj$bk_x_mat
    MSSA_obj=MSSA_main_obj$MSSA_obj 
    gammak_x_mse=MSSA_obj$gammak_x_mse
    colnames(bk_x_mat)<-select_vec_multi
    
    # Filter
    filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
    
    mssa_mat=filt_obj$mssa_mat
    target_mat=filt_obj$target_mat
    mmse_mat<-filt_obj$mmse_mat
    colnames(mssa_mat)<-select_vec_multi
    # Select M-SSA BIP and ip  
    mssa_bip<-cbind(mssa_bip,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[1])])
    mssa_ip<-cbind(mssa_ip,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[2])])
    # M-MSE
    mmse_bip<-cbind(mmse_bip,mmse_mat[,which(colnames(mmse_mat)==select_vec_multi[1])])
    mmse_ip<-cbind(mmse_ip,mmse_mat[,which(colnames(mmse_mat)==select_vec_multi[2])])
    
    # Now compute M-SSA for the remaining ifo, ESI and spread  
    #   -These series require a smaller forecast excess f_excess[2] because of their smaller publication lag
    delta<-h_vec[i]+lag_vec[1]+f_excess[2]
    
    MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,Sigma,T)
    
    bk_x_mat=MSSA_main_obj$bk_x_mat
    MSSA_obj=MSSA_main_obj$MSSA_obj 
    colnames(bk_x_mat)<-select_vec_multi
    
    filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,delta)
    
    mssa_mat=filt_obj$mssa_mat
    target_mat=filt_obj$target_mat
    mmse_mat<-filt_obj$mmse_mat
    colnames(mssa_mat)<-select_vec_multi
    
    # Select M-SSA-ifo, -ESI and -spread  
    mssa_ifo<-cbind(mssa_ifo,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[3])])
    mssa_esi<-cbind(mssa_esi,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[4])])
    mssa_spread<-cbind(mssa_spread,mssa_mat[,which(colnames(mssa_mat)==select_vec_multi[5])])
    # Select M-MSE-ifo, -ESI and -spread  
    mmse_ifo<-cbind(mmse_ifo,mmse_mat[,which(colnames(mmse_mat)==select_vec_multi[3])])
    mmse_esi<-cbind(mmse_esi,mmse_mat[,which(colnames(mmse_mat)==select_vec_multi[4])])
    mmse_spread<-cbind(mmse_spread,mmse_mat[,which(colnames(mmse_mat)==select_vec_multi[5])])
    
  }
  #------------------------  
  # 5. Compute M-SSA predictors
  #   Standardize and aggregate: equal weighting
  predictor_mssa_mat<-(scale(mssa_bip)+scale(mssa_ip)+scale(mssa_ifo)+scale(mssa_esi)+scale(mssa_spread))/length(select_vec_multi)
  colnames(predictor_mssa_mat)<-colnames(mssa_bip)<-colnames(mssa_ip)<-colnames(mssa_ifo)<-colnames(mssa_esi)<-colnames(mssa_spread)<-paste("Horizon=",h_vec,sep="")
  rownames(predictor_mssa_mat)<-rownames(x_mat)
  
  predictor_mmse_mat<-(scale(mmse_bip)+scale(mmse_ip)+scale(mmse_ifo)+scale(mmse_esi)+scale(mmse_spread))/length(select_vec_multi)
  colnames(predictor_mmse_mat)<-colnames(mmse_bip)<-colnames(mmse_ip)<-colnames(mmse_ifo)<-colnames(mmse_esi)<-colnames(mmse_spread)<-paste("Horizon=",h_vec,sep="")
  rownames(predictor_mmse_mat)<-rownames(x_mat)
  
  #-----------------------------
  # 6. Compute plots
  target_shifted_mat<-NULL
  cor_mat_HP_BIP_full_sample<-cor_mat_HP_BIP_out_of_sample<-matrix(nrow=length(h_vec),ncol=length(h_vec))
  for (i in 1:length(h_vec))#i<-1
  {
    shift<-h_vec[i]+lag_vec[1]
    # Compute target: two-sided HP applied to BIP and shifted forward by forecast horizon plus publication lag
    filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,shift)
    target_mat=filt_obj$target_mat
    # Select HP-BIP (first column)  
    target<-target_mat[,"BIP"]
    # Collect the forward shifted targets: 
    target_shifted_mat<-cbind(target_shifted_mat,target)
    
    # Plot indicators and shifting target
    mplot<-scale(cbind(target,predictor_mssa_mat))
    colnames(mplot)[1]<-paste("Target left-shifted by ",shift-lag_vec[1],sep="")
    par(mfrow=c(1,1))
    colo<-c("black",rainbow(ncol(predictor_mssa_mat)))
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
    
  }
  return(list(target_shifted_mat=target_shifted_mat,predictor_mssa_mat=predictor_mssa_mat,predictor_mmse_mat=predictor_mmse_mat))
}  






