
# Densify code: let's pack the above code into functions with distinct tasks
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


# 4. Filter function: apply M-SSA filter to data

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
  colnames(mssa_mat)<-colnames(target_mat)<-colnames(x_mat)
  return(list(mssa_mat=mssa_mat,target_mat=target_mat,mmse_mat=mmse_mat))
}




# This function operationalizes the M-SSA concept for predicting quarterly (German) GDP
# It relies on hyperparameters specifying the design
# It returns 
compute_mssa_BIP_predictors_func<-function(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess)
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
  # 3. MA inversion
  
  MA_inv_obj<-MA_inv_VAR_func(Phi,Theta,L,n,T)
  
  xi<-MA_inv_obj$xi
  
  
  #-----------------------
  # 4. Compute M-SSA for the specified forecast horizons in h_vec
  
  mssa_bip<-mssa_ip<-mssa_esi<-mssa_ifo<-mssa_spread<-NULL
  for (i in 1:length(h_vec))#i<-1
  {
    # For each forecast horizon h_vec[i] we compute M-SSA for BIP and ip first, based on the proposed forecast excess
    #   BIP and ip require a larger forecast excess f_excess[1]. We also add the publication lag
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
    
    # Now compute M-SSA for the remaining ifo, ESI and spread  
    # These series require a smaller forecast excess f_excess[2] 
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
    
  }
  #------------------------  
  # 5. Compute M-SSA predictors
  #   Standardize and aggregate: equal weighting
  indicator_mat<-(scale(mssa_bip)+scale(mssa_ip)+scale(mssa_ifo)+scale(mssa_esi)+scale(mssa_spread))/length(select_vec_multi)
  colnames(indicator_mat)<-colnames(mssa_bip)<-colnames(mssa_ip)<-colnames(mssa_ifo)<-colnames(mssa_esi)<-colnames(mssa_spread)<-paste("Horizon=",h_vec,sep="")
  rownames(indicator_mat)<-rownames(x_mat)
  #-----------------------------
  # 6. Compute performance measures
  # 6.1 sample target correlations: all combinations
  target_shifted_mat<-NULL
  cor_mat<-matrix(ncol=length(h_vec),nrow=length(h_vec))
  
  for (i in 1:length(h_vec))#i<-1
  {
    shift<-h_vec[i]+lag_vec[1]
    # Compute target: two-sided HP applied to BIP and shifted forward by forecast horizon plus publication lag
    filt_obj<-filter_func(x_mat,bk_x_mat,gammak_x_mse,gamma_target,symmetric_target,shift)
    target_mat=filt_obj$target_mat
    # Select BIP (first column)  
    target<-target_mat[,"BIP"]
    # Collect the forward shifted targets: 
    #   For the first loop-run, i=1 and shift=h_vec[1]+lag_vec[1]=2 corresponds to the publication lag of BIP (note that we selected a slightly larger publication lag, as discussed at the top of the this tutorial)  
    target_shifted_mat<-cbind(target_shifted_mat,target)
    # Plot indicators and shifting target
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
    
    # Compute target correlations of all M-SSA predictors with shifted target: all combinations
    for (j in 1:ncol(indicator_mat))
      cor_mat[i,j]<-cor(na.exclude(cbind(target,indicator_mat[,j])))[1,2]
    
  }
  
  colnames(cor_mat)<-paste("M-SSA: h=",h_vec,sep="")
  rownames(cor_mat)<-paste("Shift of target: ",h_vec,sep="")
  
  # 6.2 HAC-adjusted p-values of t-statistics when targeting shifted HP-BIP
  t_HAC_mat<-p_value_HAC_mat<-matrix(ncol=length(h_vec),nrow=length(h_vec))
  for (i in 1:length(h_vec))# i<-1
  {
    for (j in 1:length(h_vec))# j<-1
    {
# Remove NAs      
      da<-na.exclude(cbind(target_shifted_mat[,i],indicator_mat[,j]))
      lm_obj<-lm(da[,1]~da[,2])
      summary(lm_obj)
      # This one replicates std in summary
      sd<-sqrt(diag(vcov(lm_obj)))
      # Here we use HAC  
      sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
      # This is the same as
      sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
      t_HAC_mat[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
      p_value_HAC_mat[i,j]<-2*pt(t_HAC_mat[i,j], nrow(da)-length(select_vec_multi), lower=FALSE)
      
    }
  }
  colnames(t_HAC_mat)<-colnames(p_value_HAC_mat)<-paste("M-SSA: h=",h_vec,sep="")
  rownames(t_HAC_mat)<-rownames(p_value_HAC_mat)<-paste("Shift of target: ",h_vec,sep="")
  
  # 6.3 Same as 6.2 but targeting shifted BIP
  t_HAC_mat_BIP<-p_value_HAC_mat_BIP<-cor_mat_BIP<-matrix(ncol=length(h_vec),nrow=length(h_vec))
  BIP_target_mat<-NULL
  for (i in 1:length(h_vec))# i<-1
  {
    # Shift BIP  
    shift<-h_vec[i]+lag_vec[1]
    BIP_target<-c(x_mat[(1+shift):nrow(x_mat),"BIP"],rep(NA,shift))
    BIP_target_mat<-cbind(BIP_target_mat,BIP_target)
    # Rgress predictors on shifted BIP  
    for (j in 1:length(h_vec))# j<-1
    {
# Remove NAs      
      da<-na.exclude(cbind(BIP_target,indicator_mat[,j]))
      cor_mat_BIP[i,j]<-cor(da)[1,2]
      lm_obj<-lm(da[,1]~da[,2])
      summary(lm_obj)
      # This one replicates std in summary
      sd<-sqrt(diag(vcov(lm_obj)))
      # Here we use HAC  
      sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
      # This is the same as
      sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
      t_HAC_mat_BIP[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
      p_value_HAC_mat_BIP[i,j]<-2*pt(t_HAC_mat_BIP[i,j], nrow(da)-length(select_vec_multi), lower=FALSE)
      
    }
  }
  colnames(t_HAC_mat_BIP)<-colnames(p_value_HAC_mat_BIP)<-colnames(cor_mat_BIP)<-paste("M-SSA: h=",h_vec,sep="")
  rownames(t_HAC_mat_BIP)<-rownames(p_value_HAC_mat_BIP)<-rownames(cor_mat_BIP)<-paste("Shift of target: ",h_vec,sep="")
  # p-values: 
  # In contrast to HP-BIP, significance with respect to BIP is less conclusive: BIP is much noisier
  # However, we still find that for increasing forward-shift of BIP (from top to bottom) 
  #   the M-SSA indicators optimized for larger forecast horizon (from left to right) tend to perform better
  # These results could be altered by modifying the forecast excesses: 
  #   -Selecting more aggressive designs (larger excesses) may lead to stronger significance at larger shifts, up to a point 
  #   -You may try f_excess<-c(6,4): a strong result at a one-year ahead forecast horizon (plus publication lag) is achievable
  p_value_HAC_mat_BIP
  return(list(indicator_mat=indicator_mat,cor_mat=cor_mat,p_value_HAC_mat_BIP=p_value_HAC_mat_BIP,p_value_HAC_mat=p_value_HAC_mat,cor_mat_BIP=cor_mat_BIP,BIP_target_mat=BIP_target_mat,target_shifted_mat=target_shifted_mat))
}







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



