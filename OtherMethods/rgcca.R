library(RGCCA)

args <- commandArgs(trailingOnly = TRUE)

i <- as.numeric(args[1])
mode <- as.numeric(args[2])
n <- as.numeric(args[3])
v <- as.numeric(args[4])
p <- as.numeric(args[5])

cor_matrix <- matrix(c(1, 0.7, 0.7,
                       0.7, 1, 0,
                       0.7, 0, 1), 
                     nrow = 3, 
                     ncol = 3, 
                     byrow = TRUE)
RGCCA_u1 <- array(NA, dim = c(1, p))
RGCCA_u2 <- array(NA, dim = c(1, p))
RGCCA_u3 <- array(NA, dim = c(1, p))
SGCCA_u1 <- array(NA, dim = c(1, p))
SGCCA_u2 <- array(NA, dim = c(1, p))
SGCCA_u3 <- array(NA, dim = c(1, p))
print(i)
#t_rgcca <- rep(0,r)
#t_sgcca <- rep(0,r)
root <- paste0('/scratch/rw2867/projects/SNGCCA/SNGCCA/Data/Linear/',n,'_',p,'_',v)
print(root)
  X = read.csv(paste0(root,'/data1_',i-1,'.csv'),header = F)
  Y = read.csv(paste0(root,'/data2_',i-1,'.csv'),header = F)
  Z = read.csv(paste0(root,'/data3_',i-1,'.csv'),header = F)
  Data1 <- list(X,Y,Z)

    cv_tau <- rgcca_cv(Data1, response = 3, method = "rgcca",
                       par_type = "tau",
                       par_length = 5,
                       prediction_model = "glmnet", #caret::modelLookup()
                       metric = "RMSE",
                       k=3, n_run = 3,
                       verbose = TRUE)
    cv_s <- rgcca_cv(Data1, response = 3, ncomp = 1,
                     prediction_model = "glmnet", lambda = .001,
                     par_type = "sparsity",
                     par_value = c(.7, .2, 1),
                     metric = "RMSE",
                     n_cores = 2,
    )

  ts <- Sys.time()
  res1 = rgcca(Data1, method = "rgcca", tau = cv_tau$call$tau, scheme = 'horst')
  te <- Sys.time()
  RGCCA_u1 = res1$a[[1]]
  RGCCA_u2 = res1$a[[2]]
  RGCCA_u3 = res1$a[[3]]
  t_rgcca <- te-ts
  
  ts <- Sys.time()
  res2 = rgcca(Data1, connection = 1 - cor_matrix, tau = c(1,1,1), sparsity = cv_s$best_params, method = "sgcca")
  te <- Sys.time()
  t_sgcca <- te-ts
  SGCCA_u1 = res2$a[[1]]
  SGCCA_u2 = res2$a[[2]]
  SGCCA_u3 = res2$a[[3]]
  

write.csv(t_rgcca,paste0("/scratch/rw2867/projects/SNGCCA/OtherMethods/hpcres/rgcca/rgcca_t",i,".csv"), row.names = FALSE)
write.csv(t_sgcca,paste0("/scratch/rw2867/projects/SNGCCA/OtherMethods/hpcres/sgcca/sgcca_t",i,".csv"), row.names = FALSE)
write.csv(RGCCA_u1,paste0("/scratch/rw2867/projects/SNGCCA/OtherMethods/hpcres/rgcca/rgcca_u1",i,".csv"), row.names = FALSE)
write.csv(RGCCA_u2,paste0("/scratch/rw2867/projects/SNGCCA/OtherMethods/hpcres/rgcca/rgcca_u2",i,".csv"), row.names = FALSE)
write.csv(RGCCA_u3,paste0("/scratch/rw2867/projects/SNGCCA/OtherMethods/hpcres/rgcca/rgcca_u3",i,".csv"), row.names = FALSE)
write.csv(SGCCA_u1,paste0("/scratch/rw2867/projects/SNGCCA/OtherMethods/hpcres/sgcca/sgcca_u1",i,".csv"), row.names = FALSE)
write.csv(SGCCA_u2,paste0("/scratch/rw2867/projects/SNGCCA/OtherMethods/hpcres/sgcca/sgcca_u2",i,".csv"), row.names = FALSE)
write.csv(SGCCA_u3,paste0("/scratch/rw2867/projects/SNGCCA/OtherMethods/hpcres/sgcca/sgcca_u3",i,".csv"), row.names = FALSE)