library(cdetools)
library(FlexCoDE)
library(ggplot2)
library(qqplotr)
library(comprehenr)

setwd(file.path("codes", "fit"))
wd <- getwd()
setwd(file.path("..", "..", "_survey", "crossvalidation"))
crossvalidation_path <- getwd()
setwd(file.path("..", "results", "model"))
model_path <- getwd() 
setwd(wd)

# Reading data (same training, validation and test sets from Python files)
trainf0 <- read.csv(file.path(crossvalidation_path,
"trainf0.csv"), encoding = "utf8")
trainf1 <- read.csv(file.path(crossvalidation_path,
"trainf1.csv"), encoding = "utf8")
trainf2 <- read.csv(file.path(crossvalidation_path,
"trainf2.csv"), encoding = "utf8")
trainf3 <- read.csv(file.path(crossvalidation_path,
"trainf3.csv"), encoding = "utf8")
trainf4 <- read.csv(file.path(crossvalidation_path,
"trainf4.csv"), encoding = "utf8")

validf0 <- read.csv(file.path(crossvalidation_path,
"validf0.csv"), encoding = "utf8")
validf1 <- read.csv(file.path(crossvalidation_path,
"validf1.csv"), encoding = "utf8")
validf2 <- read.csv(file.path(crossvalidation_path,
"validf2.csv"), encoding = "utf8")
validf3 <- read.csv(file.path(crossvalidation_path,
"validf3.csv"), encoding = "utf8")
validf4 <- read.csv(file.path(crossvalidation_path,
"validf4.csv"), encoding = "utf8")

test <- read.csv(file.path(crossvalidation_path, "test.csv"), encoding = "utf8", header = TRUE)


# Feature spaces
colors_broad <-  c("u_PStotal.r_PStotal",
               "g_PStotal.r_PStotal",
               "r_PStotal.i_PStotal",
               "r_PStotal.z_PStotal")

colors_wise <- c("r_PStotal.W1_MAG", "r_PStotal.W2_MAG")

colors_galex <- c("FUVmag.r_PStotal", "NUVmag.r_PStotal")

colors_all <- c(colors_broad, colors_wise, colors_galex)


# Fit FlexCoDE without narrow bands

# Fold 1
set.seed(47)
start.time <- Sys.time()
fit1=fitFlexCoDE(xTrain = trainf0[c(colors_broad,colors_galex,colors_wise)],
                 zTrain = trainf0['Z'],
                 xValidation = validf0[c(colors_broad,colors_galex,colors_wise)],
                 zValidation = validf0['Z'],
                 xTest = test[c(colors_broad,colors_galex,colors_wise)],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores = 5, ntree = 100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit1,"fit1.rds") #save model

# Fold 2
set.seed(47)
start.time <- Sys.time()
fit1.1=fitFlexCoDE(xTrain = trainf1[c(colors_broad,colors_galex,colors_wise)],
                 zTrain = trainf1['Z'],
                 xValidation = validf1[c(colors_broad,colors_galex,colors_wise)],
                 zValidation = validf1['Z'],
                 xTest = test[c(colors_broad,colors_galex,colors_wise)],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores = 5, ntree = 100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit1.1,"fit1_1.rds")

# Fold 3
set.seed(47)
start.time <- Sys.time()
fit1.2=fitFlexCoDE(xTrain = trainf2[c(colors_broad,colors_galex,colors_wise)],
                 zTrain = trainf2['Z'],
                 xValidation = validf2[c(colors_broad,colors_galex,colors_wise)],
                 zValidation = validf2['Z'],
                 xTest = test[c(colors_broad,colors_galex,colors_wise)],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores = 5, ntree = 100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit1.2,"fit1_2.rds")

# Fold 4
set.seed(47)
start.time <- Sys.time()
fit1.3=fitFlexCoDE(xTrain = trainf3[c(colors_broad,colors_galex,colors_wise)],
                 zTrain = trainf3['Z'],
                 xValidation = validf3[c(colors_broad,colors_galex,colors_wise)],
                 zValidation = validf3['Z'],
                 xTest = test[c(colors_broad,colors_galex,colors_wise)],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores = 5, ntree = 100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit1.3,"fit1_3.rds")

# Fold 5
set.seed(47)
start.time <- Sys.time()
fit1.4=fitFlexCoDE(xTrain = trainf4[c(colors_broad,colors_galex,colors_wise)],
                 zTrain = trainf4['Z'],
                 xValidation = validf4[c(colors_broad,colors_galex,colors_wise)],
                 zValidation = validf4['Z'],
                 xTest = test[c(colors_broad,colors_galex,colors_wise)],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores = 5, ntree = 100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit1.4,"fit1_4.rds")

# Fit FlexCoDE with narrow bands

# Fold 1
set.seed(47)
start.time <- Sys.time()
fit1=fitFlexCoDE(xTrain = trainf0[c(colors_all)],
                 zTrain = trainf0['Z'],
                 xValidation = validf0[c(colors_all)],
                 zValidation = validf0['Z'],
                 xTest = test[c(colors_all)],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=5, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2,"fit2.rds") #save model

# Fold 2
set.seed(47)
start.time <- Sys.time()
fit2.1=fitFlexCoDE(xTrain=trainf1[c(colors_all)],
                 zTrain=trainf1['Z'],
                 xValidation=validf1[c(colors_all)],
                 zValidation=validf1['Z'],
                 xTest=test[c(colors_all)],
                 zTest=test['Z'],
                 nIMax = 45,
                 system="Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=5, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2.1,"fit2_1.rds")

# Fold 3
set.seed(47)
start.time <- Sys.time()
fit2.2=fitFlexCoDE(xTrain=trainf2[c(colors_all)],
                 zTrain=trainf2['Z'],
                 xValidation=validf2[c(colors_all)],
                 zValidation=validf2['Z'],
                 xTest=test[c(colors_all)],
                 zTest=test['Z'],
                 nIMax = 45,
                 system="Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=5, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2.2,"fit2_2.rds")

# Fold 4
set.seed(47)
start.time <- Sys.time()
fit2.3=fitFlexCoDE(xTrain=trainf3[c(colors_all)],
                 zTrain=trainf3['Z'],
                 xValidation=validf3[c(colors_all)],
                 zValidation=validf3['Z'],
                 xTest=test[c(colors_all)],
                 zTest=test['Z'],
                 nIMax = 45,
                 system="Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=5, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2.3,"fit2_3.rds")

# Fold 5
set.seed(47)
start.time <- Sys.time()
fit2.4=fitFlexCoDE(xTrain=trainf4[c(colors_all)],
                 zTrain=trainf4['Z'],
                 xValidation=validf4[c(colors_all)],
                 zValidation=validf4['Z'],
                 xTest=test[c(colors_all)],
                 zTest=test['Z'],
                 nIMax = 45,
                 system="Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=5, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2.4,"fit2_4.rds")