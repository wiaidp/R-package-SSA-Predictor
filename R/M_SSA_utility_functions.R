
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
MSSA_main_func<-function(delta,ht_vec,xi,symmetric_target,gamma_target,Plot=F)
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

filter_func<-function(x_mat,bk_x_mat,gamma_target,symmetric_target,delta)
{
  len<-nrow(x_mat)
  n<-dim(bk_x_mat)[2]
  # Compute M-SSA filter output 
  mssa_mat<-target_mat<-NULL
  for (m in 1:n)
  {
    bk<-NULL
    # Extract coefficients applied to m-th series    
    for (j in 1:n)#j<-2
      bk<-cbind(bk,bk_x_mat[((j-1)*L+1):(j*L),m])
    ts.plot((bk))
    y<-rep(NA,len)
    for (j in L:len)#j<-L
    {
      y[j]<-sum(apply(bk*(x_mat[j:(j-L+1),]),2,sum))
    }
    mssa_mat<-cbind(mssa_mat,y)
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
      gammak<-cbind(gammak,gamma_target[m,(j-1)*L+1:L])
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
    
    names(zdelta)<-names(y)<-rownames(x_mat)
    target_mat<-cbind(target_mat,z)
  } 
  colnames(mssa_mat)<-colnames(target_mat)<-colnames(x_mat)
  return(list(mssa_mat=mssa_mat,target_mat=target_mat))
}