
# Two filter functions 

# This function applies filters to data
# The function filt_func can deal with double, matrix or xts objects
SSA_filter_func<-function(filter_mat,L,x)#x<-series
{
  # filter_mat<-bk_mat  
  y_mat_indicatorh<-NULL
  for (j in 1:ncol(filter_mat))#j<-2
  {
    b<-filter_mat[1:L,j]#/sqrt(var(filter_mat_select[1:L_gdp,j]))
    
    filt_obj<-filt_func(x,b)
    #  if (j==1)
    #    filt_obj$yhat<-filt_obj$yhat+0.01*sqrt(var(filter_mat_select[1:L,j]))
    
    y_mat_indicatorh<-cbind(y_mat_indicatorh,filt_obj$yhat)
    
  }
  if (ncol(filter_mat)==1)
  {
    y_mat<-y_mat_indicatorh[L:length(y_mat_indicatorh)]
    
  } else
  {
    y_mat<-y_mat_indicatorh[L:nrow(y_mat_indicatorh),]
    colnames(y_mat)<-colnames(filter_mat)
  }
  
  return(list(y_mat=y_mat))
} 


# Filter function: applies a filter b to a series x which can be xts or double
#   If x is xts then time ordering of b is reversed
filt_func<-function(x,b)
{
  L<-length(b)
  yhat<-x
  if (is.matrix(x))
  {  
    length_time_series<-nrow(x)
  } else
  {
    if (is.vector(x)|is.double(x))
    {
      length_time_series<-length(x)
    } else
    {
      print("Error: x is neither a matrix nor a vector!!!!")
    }
  }
  for (i in L:length_time_series)
  {
    # If x is an xts object then we cannot reorder x in desceding time i.e. x[i:(i-L+1)] is the same as  x[(i-L+1):i]
    #   Therefore, in this case, we have to revert the ordering of the b coefficients.    
    if (is.xts(x))
    {
      yhat[i]<-as.double(b[L:1]%*%x[i:(i-L+1)])#tail(x) x[(i-L+1):i]
    } else
    {
      yhat[i]<-as.double(b%*%x[i:(i-L+1)])#tail(x) x[(i-L+1):i]
    }
  }
  #  names(yhat)<-index(x)#index(yhat)  index(x)
  #  yhat<-as.xts(yhat,tz="GMT")
  return(list(yhat=yhat))
}



# Need some additional functions from paper

# Data load
data_load_func<-function(path.main) 
{  
  
  indpro_mat<-read.csv(paste(path.main,"/Data/indpro.csv",sep=""),sep=",",header=T,na.strings="NA",dec=".",row.names=1)
  
  indpro<-indpro_level<-NULL
  for (i in 1:ncol(indpro_mat))
  {
    indpro<-cbind(indpro,diff(log(indpro_mat[,i])))
    indpro_level<-cbind(indpro_level,indpro_mat[,i])
  }
  typeof(indpro)
  colnames(indpro)<-colnames(indpro_level)<-colnames(indpro_mat)
  rownames(indpro)<-rownames(indpro_mat)[2:nrow(indpro_mat)]
  rownames(indpro_level)<-rownames(indpro_mat)
  mean(indpro_mat[,1],na.rm=T)
  
  indpro_mat_eu<-read.csv(paste(path.main,"/Data/indpro_eu_sa.csv",sep=""),sep=",",header=T,na.strings="NA",dec=".",row.names=1)
  
  indpro_eu<-NULL
  for (i in 1:ncol(indpro_mat_eu))
    indpro_eu<-cbind(indpro_eu,diff(log(as.double(indpro_mat_eu[,i]))))
  typeof(indpro_eu)
  tail(indpro_eu)
  colnames(indpro_eu)<-colnames(indpro_mat_eu)
  rownames(indpro_eu)<-rownames(indpro_mat_eu)[2:nrow(indpro_mat_eu)]
  
  indpro<-as.xts(indpro,order.by=as.Date(rownames(indpro),"%d/%m/%Y"))
  indpro_level<-as.xts(indpro_level,order.by=as.Date(rownames(indpro_level),"%d/%m/%Y"))
  return(list(indpro=indpro,indpro_level=indpro_level,indpro_eu=indpro_eu))
} 


#------------------------------------------------------------------------
# Plots for BCA-section in paper

plot_paper<-function(y_mat,start_date,end_date,colo_all)
{  
  # start_date<-start_date_moderation_financial_1
  # end_date<-end_date_moderation_financial_1
  if (is.null(end_date))
  { 
    # Last data point    
    end_date<-format(Sys.time(), "%Y-%m-%d")
  } 
  if (is.null(start_date))
  { 
    # Last data point    
    start_date<-index(y_mat)[1]
  } 
  select_output<-c(2,4)
  mplot<-scale(y_mat[paste(start_date,"/",end_date,sep=""),select_output],center=F,scale=T)
  coli<-colo_all[select_output]
  q_gap<-plot(mplot,main="",col=coli)
  #  q_gap<-plot(mplot,main=paste(colnames(y_mat)[select_output[1]], " vs. ",colnames(y_mat)[select_output[2]],sep=""),col=coli)
  #p<-mtext("HP-gap",col=coli[1],line=-3)
  sel_tp<-c(1,2)
  for (i in 1:length(sel_tp))
  {
    ret<-mplot[,sel_tp[i]]
    tp_last<-index(mplot)[which(sign(ret)!=lag(sign(ret)))]
    events<-xts(rep("",length(tp_last)),tp_last)
    q_gap<-addEventLines(events, srt=90, pos=2,col=coli[sel_tp[i]])
  }
  q_gap
  x_gap<-nber_dates_polygon(start_date,mplot)$x
  y_gap<-nber_dates_polygon(start_date,mplot)$y
  q_gap
  polygon(x_gap, y_gap, xpd = T, col = "grey",density=10)#
  
  
  select_output<-c(1,4)
  mplot<-scale(y_mat[paste(start_date,"/",end_date,sep=""),select_output],center=F,scale=T)
  coli<-colo_all[select_output]
  #  q_trend<-plot(mplot,main=paste(colnames(y_mat)[select_output[1]], " vs. ",colnames(y_mat)[select_output[2]],sep=""),col=coli)
  q_trend<-plot(mplot,main="",col=coli)
  #p<-mtext("HP-gap",col=coli[1],line=-3)
  sel_tp<-c(1,2)
  for (i in 1:length(sel_tp))
  {
    ret<-mplot[,sel_tp[i]]
    tp_last<-index(mplot)[which(sign(ret)!=lag(sign(ret)))]
    events<-xts(rep("",length(tp_last)),tp_last)
    q_trend<-addEventLines(events, srt=90, pos=2,col=coli[sel_tp[i]])
  }
  x_trend<-nber_dates_polygon(start_date,mplot)$x
  y_trend<-nber_dates_polygon(start_date,mplot)$y
  q_trend
  polygon(x_trend, y_trend, xpd = T, col = "grey",density=10)#
  
  select_output<-c(4,3)
  mplot<-scale(y_mat[paste(start_date,"/",end_date,sep=""),select_output],center=F,scale=T)
  coli<-colo_all[select_output]
  #  q_SSA<-plot(mplot,main=paste(colnames(y_mat)[select_output[1]], " vs. ",colnames(y_mat)[select_output[2]],sep=""),col=coli)
  q_SSA<-plot(mplot,main="",col=coli)
  #p<-mtext("HP-gap",col=coli[1],line=-3)
  sel_tp<-c(1,2)
  for (i in 1:length(sel_tp))
  {
    ret<-mplot[,sel_tp[i]]
    tp_last<-index(mplot)[which(sign(ret)!=lag(sign(ret)))]
    events<-xts(rep("",length(tp_last)),tp_last)
    q_SSA<-addEventLines(events, srt=90, pos=2,col=coli[sel_tp[i]])
  }
  q_SSA
  x_SSA<-nber_dates_polygon(start_date,mplot)$x
  y_SSA<-nber_dates_polygon(start_date,mplot)$y
  q_SSA
  polygon(x_SSA, y_SSA, xpd = T, col = "grey",density=10)#
  
  
  
  return(list(q_gap=q_gap,q_trend=q_trend,q_SSA=q_SSA,x_trend=x_trend,y_trend=y_trend,x_gap=x_gap,y_gap=y_gap,x_SSA=x_SSA,y_SSA=y_SSA))
}


# Adds shaded areas corresponding to NBER-recession datings
nber_dates_polygon<-function(start_date,mat)
{
  dat<-NULL
  for (i in 1:nrow(nberDates()))
  {
    dat<-c(dat,paste(substr(nberDates()[i,1],1,4),"-",substr(nberDates()[i,1],5,6),"-",substr(nberDates()[i,1],7,8),sep=""))
  }
  
  
  starting_date<-as.POSIXct(strptime(dat, "%Y-%m-%d"),tz="UTC")
  
  starting_date<-starting_date[which(starting_date>start_date)]
  
  dat<-NULL
  for (i in 1:nrow(nberDates()))
  {
    dat<-c(dat,paste(substr(nberDates()[i,2],1,4),"-",substr(nberDates()[i,2],5,6),"-",substr(nberDates()[i,2],7,8),sep=""))
  }
  
  ending_date<-as.POSIXct(strptime(dat, "%Y-%m-%d"),tz="UTC")
  ending_date<-ending_date[which(ending_date>start_date)]
  
  x<-y<-NULL
  for (i in 1:length(starting_date))
  {
    x<-c(x,starting_date[i],starting_date[i],ending_date[i],ending_date[i])
    y<-c(y,min(na.exclude(mat)),max(na.exclude(mat)),max(na.exclude(mat)),min(na.exclude(mat)))
  }
  return(list(x=x,y=y))
}



