# install.packages("devtools")
library(devtools)

# Installing cdetools
# devtools::install_github("tpospisi/cdetools/r")

# Installing FlexCoDE
# devtools::install_github("rizbicki/FlexCoDE")

library(cdetools)
library(FlexCoDE)
library(ggplot2)
library(qqplotr)
library(comprehenr)

# setwd(file.path("codes", "validation"))
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
wd <- getwd()
setwd(file.path("..", "..", "data", "crossvalidation"))
crossvalidation_path <- getwd()
setwd(file.path("..", "..", "results", "validation", "flexcode"))
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
"valf0.csv"), encoding = "utf8")
validf1 <- read.csv(file.path(crossvalidation_path,
"valf1.csv"), encoding = "utf8")
validf2 <- read.csv(file.path(crossvalidation_path,
"valf2.csv"), encoding = "utf8")
validf3 <- read.csv(file.path(crossvalidation_path,
"valf3.csv"), encoding = "utf8")
validf4 <- read.csv(file.path(crossvalidation_path,
"valf4.csv"), encoding = "utf8")

test <- read.csv(file.path(crossvalidation_path, "test.csv"), encoding = "utf8", header = TRUE)


# Feature spaces

colors_broad <-  c("u_PStotal.r_PStotal",
               "g_PStotal.r_PStotal",
               "r_PStotal.i_PStotal",
               "r_PStotal.z_PStotal")

colors_wise <- c("r_PStotal.W1", "r_PStotal.W2")

colors_galex <- c("FUVmag.r_PStotal", "NUVmag.r_PStotal")

color_narrow <- c("J0378_PStotal.r_PStotal",
                  "J0395_PStotal.r_PStotal",
                  "J0410_PStotal.r_PStotal",
                  "J0430_PStotal.r_PStotal",
                  "J0515_PStotal.r_PStotal",
                  "r_PStotal.J0660_PStotal",
                  "r_PStotal.J0861_PStotal")

# flags <- c("flag_GALEX", "flag_WISE")

feat_broad <- c(colors_broad,colors_galex,colors_wise)
feat_all <- c(color_narrow, colors_broad, colors_wise, colors_galex)

# Fit FlexCoDE without narrow bands

# Fold 1
set.seed(47)
start.time <- Sys.time()
fit1=fitFlexCoDE(xTrain = trainf0[feat_broad],
                 zTrain = trainf0['Z'],
                 xValidation = validf0[feat_broad],
                 zValidation = validf0['Z'],
                 xTest = test[feat_broad],
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
saveRDS(fit1,file.path(model_path,"broad+GALEX+WISE_0.rds")) #save model
rm(fit1)

# Fold 2
set.seed(47)
start.time <- Sys.time()
fit1.1=fitFlexCoDE(xTrain = trainf1[feat_broad],
                 zTrain = trainf1['Z'],
                 xValidation = validf1[feat_broad],
                 zValidation = validf1['Z'],
                 xTest = test[feat_broad],
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
saveRDS(fit1.1,file.path(model_path,"broad+GALEX+WISE_1.rds"))
rm(fit1.1)

# Fold 3
set.seed(47)
start.time <- Sys.time()
fit1.2=fitFlexCoDE(xTrain = trainf2[feat_broad],
                 zTrain = trainf2['Z'],
                 xValidation = validf2[feat_broad],
                 zValidation = validf2['Z'],
                 xTest = test[feat_broad],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores = 10, ntree = 100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit1.2,file.path(model_path,"broad+GALEX+WISE_2.rds"))
rm(fit1.2)

# Fold 4
set.seed(47)
start.time <- Sys.time()
fit1.3=fitFlexCoDE(xTrain = trainf3[feat_broad],
                 zTrain = trainf3['Z'],
                 xValidation = validf3[feat_broad],
                 zValidation = validf3['Z'],
                 xTest = test[feat_broad],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores = 10, ntree = 100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit1.3,file.path(model_path,"broad+GALEX+WISE_3.rds"))
rm(fit1.3)


# Fold 5
set.seed(47)
start.time <- Sys.time()
fit1.4=fitFlexCoDE(xTrain = trainf4[feat_broad],
                 zTrain = trainf4['Z'],
                 xValidation = validf4[feat_broad],
                 zValidation = validf4['Z'],
                 xTest = test[feat_broad],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores = 10, ntree = 100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit1.4,file.path(model_path,"broad+GALEX+WISE_4.rds"))
rm(fit1.4)



# Fit FlexCoDE with narrow bands

# Fold 1
set.seed(47)
start.time <- Sys.time()
fit2=fitFlexCoDE(xTrain = trainf0[c(feat_all)],
                 zTrain = trainf0['Z'],
                 xValidation = validf0[c(feat_all)],
                 zValidation = validf0['Z'],
                 xTest = test[c(feat_all)],
                 zTest = test['Z'],
                 nIMax = 45,
                 system = "Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=10, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2,file.path(model_path,"broad+GALEX+WISE+narrow_0.rds")) #save model
rm(fit2)

# Fold 2
set.seed(47)
start.time <- Sys.time()
fit2.1=fitFlexCoDE(xTrain=trainf1[c(feat_all)],
                 zTrain=trainf1['Z'],
                 xValidation=validf1[c(feat_all)],
                 zValidation=validf1['Z'],
                 xTest=test[c(feat_all)],
                 zTest=test['Z'],
                 nIMax = 45,
                 system="Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=10, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2.1,file.path(model_path,"broad+GALEX+WISE+narrow_1.rds"))
rm(fit2.1)

# Fold 3
set.seed(47)
start.time <- Sys.time()
fit2.2=fitFlexCoDE(xTrain=trainf2[c(feat_all)],
                 zTrain=trainf2['Z'],
                 xValidation=validf2[c(feat_all)],
                 zValidation=validf2['Z'],
                 xTest=test[c(feat_all)],
                 zTest=test['Z'],
                 nIMax = 45,
                 system="Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=10, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2.2,file.path(model_path,"broad+GALEX+WISE+narrow_2.rds"))
rm(fit2.2)

# Fold 4
set.seed(47)
start.time <- Sys.time()
fit2.3=fitFlexCoDE(xTrain=trainf3[c(feat_all)],
                 zTrain=trainf3['Z'],
                 xValidation=validf3[c(feat_all)],
                 zValidation=validf3['Z'],
                 xTest=test[c(feat_all)],
                 zTest=test['Z'],
                 nIMax = 45,
                 system="Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=10, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2.3,file.path(model_path,"broad+GALEX+WISE+narrow_3.rds"))
rm(fit2.3)

# Fold 5
set.seed(47)
start.time <- Sys.time()
fit2.4=fitFlexCoDE(xTrain=trainf4[c(feat_all)],
                 zTrain=trainf4['Z'],
                 xValidation=validf4[c(feat_all)],
                 zValidation=validf4['Z'],
                 xTest=test[c(feat_all)],
                 zTest=test['Z'],
                 nIMax = 45,
                 system="Fourier",
                 regressionFunction = regressionFunction.Forest,
                 regressionFunction.extra = list(nCores=10, ntree=100),
                 chooseDelta = TRUE,
                 chooseSharpen = TRUE,
                 verbose = TRUE)

end.time <- Sys.time()
time.taken1 <- end.time - start.time
saveRDS(fit2.4,file.path(model_path,"broad+GALEX+WISE+narrow_4.rds"))
rm(fit2.4)



