## Load required packages
library("readr")
library("caret")
library("h2o")
library("magrittr")

options(scipen = 999)

## Read training data
tr.data <- readr::read_csv("./data/train.csv.zip")
y <- factor(tr.data$target, levels = paste("Class", 1:9, sep = "_"))

## Create stratified data folds
nfolds <- 5
set.seed(2020)
folds.id <- caret::createFolds(y, k = nfolds, list = FALSE)
set.seed(2020)
folds.list <- caret::createFolds(y, k = nfolds, list = TRUE)
save("folds.id", "folds.list", file = "./data/cv_folds.rda", compress = "bzip2")
rm(list = ls())
gc(verbose = FALSE)

## Instantiate H2O cluster
h2o.init(max_mem_size = '8G', nthreads = 6)
h2o.removeAll()

## Load training and test data
label.name <- 'target'
train.hex <- h2o.importFile(
   path = normalizePath("./data/train.csv.zip"),
   destination_frame = 'train_hex'
)
train.hex[,label.name] <- h2o.asfactor(train.hex[,label.name])
test.hex <- h2o.importFile(
   path = normalizePath("./data/test.csv.zip"), 
   destination_frame = 'test_hex'
)
input.names <- h2o.colnames(train.hex) %>% setdiff(c('id', label.name))

## Assign data folds
load('./data/cv_folds.rda')
train.hex <- h2o.cbind(train.hex, as.h2o(data.frame('cv' = folds.id), destination_frame = 'fold_idx'))
h2o.colnames(train.hex)

## Close H2O instance
# h2o.shutdown(prompt = FALSE)
