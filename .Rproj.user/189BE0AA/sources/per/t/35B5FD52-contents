#----------------------------------------------------------------------------
# Timeliness function: compute peak correlation and tau-statistic, see appendix in paper
# It relies on new_lead_at_crossing_func for computing the time-shift
# Idea: compute zero-crossings of two filters and compare timings of zero-crossings of the filters
#   -filter_mat is matrix with outputs of filter1 and filter2
#   -filter1 serves as reference: we look at all crossings of filter1 and determine the nearest crossing of filter2
#   -one then sums the distances (of nearest crossings) over all crossings of filter1
#   -a negative sum indicates a lead of reference filter: a positive sum means a lag
# Assumption: 
#   -filter1 is the reference design: it should have less crossings, i.e., be smoother. 
#   -We typically assume filter1 is the (smoother) SSA-design.
#   -exchanging filter1 and filter2 generates different ersults because the reference crossings change
# The statistic (sum of nearest crossing distances) is biased against the smoother reference filter because the 
#   the noisier filter2 can generate random crossings preceding the reference filter purely by chance
# A better measure of the time-shift at zero-crossings is proposed by the function compute_min_tau_func below
compute_timeliness_func<-function(filter_mat,max_lead=6,vicinity=4,last_crossing_or_closest_crossing=F,outlier_limit=10)
{  
  
  # Peak correlation
  
  cor_peak<-NULL
  for (i in 1:max_lead)
  {
    cor_peak<-c(cor_peak,cor(filter_mat[i:(nrow(filter_mat)),2],filter_mat[1:(nrow(filter_mat)-i+1),1]))
  }
  # Invert time ordering
  cor_peak<-cor_peak[max_lead:1]
  # Compute other tail
  for (i in 1:(max_lead-1))
  {
    cor_peak<-c(cor_peak,cor(filter_mat[(i+1):(nrow(filter_mat)),1],filter_mat[1:(nrow(filter_mat)-i),2]))
  }
  
  
  plot(cor_peak,col="blue",main="Peak correlations",axes=F,type="l", xlab="Lead/lag",ylab="Correlation")
  abline(v=which(cor_peak==max(cor_peak)),col="blue")
  at_vec<-c(1,max_lead/2,max_lead,3*max_lead/2,2*max_lead-1)
  axis(1,at=at_vec,labels=at_vec-max_lead)
  axis(2)
  box()
  
  peak_cor_plot<-recordPlot()
  
  
  #------------------------------------------------------------
  # Empirical lead/lag at zero-crossings
  # Skip all crossings with lead/lag>outlier_limit
  skip_larger<-outlier_limit
  # Index of series with more crossings: this is measured against the crossings of the reference series
  con_ind<-2
  # Index of reference series: this one has less crossings and shift is measured with reference to thse crossings only
  ref_ind<-1

  
  lead_lag_cross_obj<-new_lead_at_crossing_func(ref_ind,con_ind,filter_mat,last_crossing_or_closest_crossing,vicinity)
  
  number_cross<-lead_lag_cross_obj$number_crossings_per_sample
  colnames(number_cross)<-colnames(filter_mat)[c(con_ind,ref_ind)]
  # Summands of Tau statistic in paper  
  tau_vec<-c(lead_lag_cross_obj$cum_ref_con[1],diff(lead_lag_cross_obj$cum_ref_con))
  names(tau_vec)<-lead_lag_cross_obj$ref_cross
  remove_tp<-which(abs(tau_vec)>skip_larger)
  if (length(remove_tp)>0)
  {  
    tau_vec_adjusted<-tau_vec[-remove_tp]
  } else
  {
    print("No outlier adjustment for tau-statistic")    
    tau_vec_adjusted<-tau_vec
  }
  # Positive drift i.e. lead of SSA filter
  ts.plot(cumsum(tau_vec_adjusted))
  ts.plot(cumsum(tau_vec))
  # Tau-statistic: mean lead (positive) or lag (negative) of reference filter: with outlier removal
  tau_adjusted<-mean(tau_vec_adjusted)
  tau_adjusted
  # Shift without outlier removal
  tau<-lead_lag_cross_obj$mean_lead_ref_con
  # Test for significance of shift
  t_conf_level<-t.test(tau_vec_adjusted,  alternative = "two.sided")$p.value
  # Strongly significant lead
  t_conf_level
  t_test_adjusted<-t.test(tau_vec_adjusted,  alternative = "two.sided")$statistic
  t_test<-t.test(tau_vec,  alternative = "two.sided")$statistic
  

  return(list(cor_peak=cor_peak,tau_vec=tau_vec,tau_vec_adjusted=tau_vec_adjusted,tau=tau,tau_adjusted=tau_adjusted,t_test=t_test,t_test_adjusted=t_test_adjusted,number_cross=number_cross,peak_cor_plot=peak_cor_plot))
  
}

# This function shifts filter2 against filter1: for each lead or lag the function computes the sum of distances between crossings
# The function plots this sum as a function of lead/lag
# The minimum of the fuction (sum of timing-distances) indicates the lad or lag of filter2 relative to filter1 
compute_min_tau_func<-function(filter_mat,max_lead=6,vicinity=4,last_crossing_or_closest_crossing=F,outlier_limit=10)
{  
  
  #------------------------------------------------------------
  # Empirical lead/lag at zero-crossings
  # Skip all crossings with lead/lag>outlier_limit
  skip_larger<-outlier_limit
  # Index of series with more crossings: this is measured against the crossings of the reference series
  con_ind<-2
  # Index of reference series: this one has less crossings and shift is measured with reference to thse crossings only
  ref_ind<-1
  mean_shift_vec<-mean_shift_adjusted_vec<-NULL
  for (i in 1:max_lead)
  {
    shift_series<-cbind(filter_mat[i:nrow(filter_mat),1],filter_mat[1:(nrow(filter_mat)-i+1),2])
  
  
    lead_lag_cross_obj<-new_lead_at_crossing_func(ref_ind,con_ind,shift_series,last_crossing_or_closest_crossing,vicinity)
  
    tau_vec<-c(lead_lag_cross_obj$cum_ref_con[1],diff(lead_lag_cross_obj$cum_ref_con))
    remove_tp<-which(abs(tau_vec)>skip_larger)
    if (length(remove_tp)>0)
    {  
      tau_vec_adjusted<-tau_vec[-remove_tp]
    } else
    {
      print("No outlier adjustment for tau-statistic")    
      tau_vec_adjusted<-tau_vec
    }
    tau<-lead_lag_cross_obj$mean_lead_ref_con
    tau_adjusted<-mean(tau_vec_adjusted)
    mean_shift_vec<-c(mean_shift_vec,tau)
    mean_shift_adjusted_vec<-c(mean_shift_adjusted_vec,tau_adjusted)
  }
# revert ordering  
  mean_shift_vec<-mean_shift_vec[max_lead:1]
  mean_shift_adjusted_vec<-mean_shift_adjusted_vec[max_lead:1]
# Shift the other series  
  for (i in 2:max_lead)# i<-3
  {
    shift_series<-cbind(filter_mat[1:(nrow(filter_mat)-i+1),1],filter_mat[i:nrow(filter_mat),2])
    
    
    lead_lag_cross_obj<-new_lead_at_crossing_func(ref_ind,con_ind,shift_series,last_crossing_or_closest_crossing,vicinity)
    
    tau_vec<-c(lead_lag_cross_obj$cum_ref_con[1],diff(lead_lag_cross_obj$cum_ref_con))
    remove_tp<-which(abs(tau_vec)>skip_larger)
    if (length(remove_tp)>0)
    {  
      tau_vec_adjusted<-tau_vec[-remove_tp]
    } else
    {
      print("no outlier adjustment necessary in tau-statistic")    
      tau_vec_adjusted<-tau_vec
    }
    tau<-lead_lag_cross_obj$mean_lead_ref_con
    tau_adjusted<-mean(tau_vec_adjusted)
    mean_shift_vec<-c(mean_shift_vec,tau)
    mean_shift_adjusted_vec<-c(mean_shift_adjusted_vec,tau_adjusted)
  }
# Revert time once again to conform with peak-cor plot: leads of reference filter (first column) correspond to troughs to the left  
  mean_shift_vec<-mean_shift_vec[length(mean_shift_vec):1]
  mean_shift_adjusted_vec<-mean_shift_adjusted_vec[length(mean_shift_adjusted_vec):1]
  if (F)
  {  
    par(mfrow=c(2,1))
    plot(abs(mean_shift_vec),col="blue",main="Min-tau shift",axes=F,type="l", xlab="Lead/lag",ylab="")
    abline(v=max_lead)
    at_vec<-c(1,max_lead/2,max_lead,3*max_lead/2,2*max_lead-1)
    axis(1,at=at_vec,labels=at_vec-max_lead)
    axis(2)
    box()
    plot(abs(mean_shift_adjusted_vec),col="blue",main="Min-tau adjusted shift",axes=F,type="l", xlab="Lead/lag",ylab="")
    abline(v=max_lead)
    at_vec<-c(1,max_lead/2,max_lead,3*max_lead/2,2*max_lead-1)
    axis(1,at=at_vec,labels=at_vec-max_lead)
    axis(2)
    box()
    
    
  }
  
  par(mfrow=c(1,1))
  plot(abs(mean_shift_adjusted_vec),col="blue",main="Min-tau adjusted shift",axes=F,type="l", xlab="Lead/lag",ylab="")
  abline(v=max_lead)
  at_vec<-c(1,max_lead/2,max_lead,3*max_lead/2,2*max_lead-1)
  axis(1,at=at_vec,labels=at_vec-max_lead)
  axis(2)
  box()
  
  
  min_tau_plot<-recordPlot()
  
  
  
  
  
  
  
  return(list(mean_shift_vec=mean_shift_vec,mean_shift_adjusted_vec=mean_shift_adjusted_vec,min_tau_plot=min_tau_plot))
  
}


# This function implements the mean-shift statistic in paper
#   It accounts for sign of crossings (up-swing/down-swing)
# There is a new additional feature when compared with earlier version above
#   1. If last_crossing_or_closest_crossing==F then it replicates the function used and described in paper
#   2. If last_crossing_or_closest_crossing==T then for each up- or down-turn of the reference filter 
#     one selects the last up- or down-turn of contender in a vicinity of crossing (otherwise the closest will be selected) 
# The setting last_crossing_or_closest_crossing==T is more realistic but still a bit optimistic because 
#   the user generally does not know that a particular crossing will be the last one.
# Results depend on vicinity which is the size of the neighborhood +/-vicinity about crossing
#   Can be chosen with a link to holding time: if holding time is large, then vicinity could be larger, too
# In general last_crossing_or_closest_crossing==F will lead to smaller lead-times (closest crossing) than
#   last_crossing_or_closest_crossing==T (latest crossing in vicinity)
# In the paper we use last_crossing_or_closest_crossing==F exclusively (closest crossings)
new_lead_at_crossing_func<-function(ref_ind,con_ind,filter_mat,last_crossing_or_closest_crossing,vicinity)
{
  ref_cross<-which(sign(filter_mat[1:(nrow(filter_mat)-1),ref_ind])!=sign(filter_mat[2:(nrow(filter_mat)),ref_ind]))
  con_cross<-which(sign(filter_mat[1:(nrow(filter_mat)-1),con_ind])!=sign(filter_mat[2:(nrow(filter_mat)),con_ind]))
  if (filter_mat[ref_cross[1],ref_ind]<0)
  {
    ref_cross_sign<-(-1)^(0:(length(ref_cross)-1))*ref_cross  
  } else
  {
    ref_cross_sign<--1*((-1)^(0:(length(ref_cross)-1))*ref_cross)
  }
  if (filter_mat[con_cross[1],con_ind]<0)
  {
    con_cross_sign<-(-1)^(0:(length(con_cross)-1))*con_cross  
  } else
  {
    con_cross_sign<--1*((-1)^(0:(length(con_cross)-1))*con_cross)
  }
  # Crossings from above and from below  
  con_cross_sign_plus<-con_cross_sign[con_cross_sign>0]
  con_cross_sign_negative<-con_cross_sign[con_cross_sign<0]
  
  disc_vec<-NULL
  # Check if there are crossings    
  if (length(ref_cross)>0&length(con_cross)>0)
  {
    
    
    ref_len<-length(ref_cross)#length(con_cross)
    
    # For the j-th zero-crossing of reference
    for (j in 1:ref_len)#  j=1
    {
      if (ref_cross_sign[j]>0)
      {
        # Up-turns      
        if (!last_crossing_or_closest_crossing)
        {    
          # For upturns of reference: select nearest upturn of contender        
          min_i<-min(abs(abs(con_cross_sign_plus)-abs(ref_cross[j])))
          which_min<-which(abs(abs(con_cross_sign_plus)-abs(ref_cross[j]))==min_i)
          disc_vec<-c(disc_vec,max(abs(con_cross_sign_plus[which_min]))-abs(ref_cross[j]))
#          disc_vec<-c(disc_vec,min_i)
        } else
        {
          # For upturns of reference: select latest upturn of contender in vicinity of reference up-turn
          # This is closer to real-time application though it is still optimistic because the user doesn't know 
          #   that this will be the last up-turn in practice      
          # 1 All upturns in vicinity of reference up-turn        
          rert<-which(abs(abs(con_cross_sign_plus)-abs(ref_cross[j]))<vicinity)
          # 2. Select latest one or border of vicinity
          if (length(rert)>0)
          {  
            # If there is a TP in vicinity: select last one            
            max_rert<-max(rert)
            last_tp<-abs(con_cross_sign_plus[max_rert])
          } else
          {
            # Otherwise select border of vicinity (one again a bit optimistic)            
            last_tp<-abs(ref_cross[j])+vicinity
            last_tp<-abs(ref_cross[j])
          }
          disc_vec<-c(disc_vec,last_tp-abs(ref_cross[j]))
          
        }  
      } else
      {
        # For downturns      
        if (!last_crossing_or_closest_crossing)
        {  
          # For downturns of reference: select nearest downturn of contender        
          min_i<-min(abs(abs(con_cross_sign_negative)-abs(ref_cross[j])))
          which_min<-which(abs(abs(con_cross_sign_negative)-abs(ref_cross[j]))==min_i)
          # Time-difference at down-turn: 
          #   in case of two possible nearest crossings we select the max (because the contrary direction is on at the time point of the crossing of the reference filter)
          disc_vec<-c(disc_vec,max(abs(con_cross_sign_negative[which_min]))-abs(ref_cross[j]))
#          disc_vec<-c(disc_vec,min_i)
          
        } else
        {
          # For down-turns of reference: select latest down-turn of contender in vicinity of reference down-turn
          # This is closer to real-time application though it is still optimistic because the user doesn't know 
          #   that this will be the last down-turn in practice      
          # 1 All down-pturns in vicinity of reference down-turn        
          rert<-which(abs(abs(con_cross_sign_negative)-abs(ref_cross[j]))<vicinity)
          # 2. Select latest one or border of vicinity
          if (length(rert)>0)
          {  
            # If there is a TP in vicinity: select last one            
            max_rert<-max(rert)
            last_tp<-abs(con_cross_sign_negative[max_rert])
          } else
          {
            # Otherwise select border of vicinity (one again a bit optimistic)            
            last_tp<-abs(ref_cross[j])+vicinity
            last_tp<-abs(ref_cross[j])
          }
          disc_vec<-c(disc_vec,last_tp-abs(ref_cross[j]))
        }  
      }
    }
    # Total number of crossings of contender and of reference
    number_crossings_per_sample<-c(length(con_cross),length(ref_cross))
  } else
  {
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("no zero-crossings observed")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    number_crossings_per_sample<-rbind(number_crossings_per_sample,c(0,0))
  }
  if (length(names(ref_cross))>0)
    names(disc_vec)<-names(ref_cross)
  number_crossings_per_sample<-t(as.matrix(number_crossings_per_sample,nrow=1))
  cum_ref_con<-cumsum(na.exclude(disc_vec))
  #  ts.plot(cum_ref_con)
  mean_lead_ref_con<-cum_ref_con[length(cum_ref_con)]/length(cum_ref_con)
  mean_lead_ref_con
  tail(cum_ref_con)
  return(list(cum_ref_con=cum_ref_con,mean_lead_ref_con=mean_lead_ref_con,number_crossings_per_sample=number_crossings_per_sample,ref_cross=ref_cross))
}

