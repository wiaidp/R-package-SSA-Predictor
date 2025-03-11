
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
      lm_obj<-lm(target_shifted_mat[,i]~indicator_mat[,j])
      summary(lm_obj)
      # This one replicates std in summary
      sd<-sqrt(diag(vcov(lm_obj)))
      # Here we use HAC  
      sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
      # This is the same as
      sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
      t_HAC_mat[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
      p_value_HAC_mat[i,j]<-2*pt(t_HAC_mat[i,j], len-length(select_vec_multi), lower=FALSE)
      
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
      cor_mat_BIP[i,j]<-cor(na.exclude(cbind(BIP_target,indicator_mat[,j])))[1,2]
      lm_obj<-lm(BIP_target~indicator_mat[,j])
      summary(lm_obj)
      # This one replicates std in summary
      sd<-sqrt(diag(vcov(lm_obj)))
      # Here we use HAC  
      sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
      # This is the same as
      sqrt(diag(sandwich(lm_obj, meat. = meatHAC)))
      t_HAC_mat_BIP[i,j]<-summary(lm_obj)$coef[2,1]/sd_HAC[2]
      p_value_HAC_mat_BIP[i,j]<-2*pt(t_HAC_mat_BIP[i,j], len-length(select_vec_multi), lower=FALSE)
      
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