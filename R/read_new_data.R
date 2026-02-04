
path.main<-getwd()

path.pgm<-paste(path.main,"/R/",sep="")
path.out<-paste(path.main,"/Latex/",sep="")
path.sweave<-paste(path.main,"/Sweave/",sep="")
path.data<-paste(path.main,"/Data/",sep="")
# Savd results from an empirical analysis of S&P500
path.result<-paste(path.main,"/Results/",sep="")
fig_size<-4

#------------------------------
# Section 2.1: data
recompute_results<-T




# Load data
data_file_name<-c("Data_HWI_2025_02.csv","gdp_2025_02.csv")
data_file_name<-c("Data_HWI_2026_02.csv","gdp_2026_02.csv")

select_vec_multi<-c("BIP","ip","ifo_c","ifo_exp","ifo_l","ESI","spr_10y_3m")

if (F)
{
# Original (un-transformed) indicators
  data_quarterly<-read.csv(paste(getwd(),"/Data/",data_file_name[2],sep=""))
  tail(data_quarterly)
  BIP_original<-data_quarterly[,"BIP"]
  
  
  
  # Transformed indicators: differences, trimmed, standardized
  load(file=paste(getwd(),"\\Data\\macro",sep=""))
  
  tail(data,20)
  
  x_mat<-data[,select_vec_multi]
  head(x_mat)
  tail(x_mat)
  rownames(x_mat)<-rownames(data)
  n<-dim(x_mat)[2]
  # Number of observations
  len<-dim(x_mat)[1]
}

# Read original data 
h0<-1
trim_threshold<-3
na_rw_lin<-T
select_vec<-select_vec_multi
diff_lag<-3



read.obj<-read_data_func_quarterly(path.data,select_vec,h0,trim_threshold,na_rw_lin,data_file_name,diff_lag)

data<-read.obj$data
tail(data)


data[(nrow(data)-1):nrow(data),1]<-NA
tail(data)

save(data,file=paste(getwd(),"\\Data\\macro_2026",sep=""))

load(file=paste(getwd(),"\\Data\\macro_2026",sep=""))

tail(data,20)





