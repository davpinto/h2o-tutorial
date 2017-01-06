## Load required packages
library("readr")
library("xgboost")
library("Matrix")
library("magrittr")

options(scipen = 999)

## Read training data
tr.data <- readr::read_csv("./data/train.csv.zip")
te.data <- readr::read_csv("./data/test.csv.zip")
y <- factor(tr.data$target, levels = paste("Class", 1:9, sep = "_"))
x.tr <- Matrix(as.matrix(tr.data[,-c(1,95)]), sparse = TRUE)
x.te <- Matrix(as.matrix(te.data[,-1]), sparse = TRUE)

## Format data
dtrain <- xgb.DMatrix(
   data = x.tr,
   label = as.integer(y) - 1
)
dtest <- xgb.DMatrix(
   data = x.te
)

### 5-Fold CV
load('./data/cv_folds.rda')
xgb.params <- list(
   "booster" = "gbtree",
   "eta" = 0.1,
   "max_depth" = 10,
   "subsample" = 0.6,
   "colsample_bytree" = 0.8,
   "colsample_bylevel" = 0.6,
   "num_class" = nlevels(y),
   "objective" = "multi:softprob",
   "eval_metric" = "mlogloss",
   "num_parallel_tree" = 10,
   "silent" = 1,
   "nthread" = 12
)
cv.out <- xgb.cv(params = xgb.params, data = dtrain, nrounds = 5e2,
                 folds = folds.list, prediction = TRUE, 
                 verbose = TRUE, showsd = FALSE, print_every_n = 10, 
                 early_stopping_rounds = 10, maximize = FALSE)
best.tr.loss <- min(cv.out$evaluation_log$train_mlogloss_mean)
best.te.loss <- min(cv.out$evaluation_log$test_mlogloss_mean)
best.iter <- which.min(cv.out$evaluation_log$test_mlogloss_mean)

### Save level-one data for training
var.names <- paste("xgb", 1:nlevels(y), sep = "_")
xgb.tr <- cbind.data.frame(round(cv.out$pred, 6), y)
names(xgb.tr) <- c(var.names, "target")
write.csv(
   xgb.tr, 
   file = gzfile('./xgb/xgb_levone_train.csv.gz'),
   row.names = FALSE
)

### Get predictions for test set
xgb.model <- xgb.train(data = dtrain, params = xgb.params, nrounds = round(1.1*best.iter))
xgb.out <- matrix(predict(xgb.model, dtest), ncol = nlevels(y), byrow = TRUE)
xgb.out <- cbind.data.frame(te.data$id, round(xgb.out, 6))
dt <- as.data.frame(xgb.out)
names(dt) <- c("id", levels(y))
write.csv(
   dt,
   file = gzfile('./xgb/xgb_output.csv.gz'), 
   row.names = FALSE
)

### Save level-one for test
xgb.te <- xgb.out[,-1]
names(xgb.te) <- var.names
write.csv(
   xgb.te, 
   file = gzfile('./xgb/xgb_levone_test.csv.gz'), 
   row.names = FALSE
)
