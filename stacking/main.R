## Import level-one data
# GBM
gbm.tr.hex <- h2o.importFile(
   path = normalizePath("./gbm/gbm_levone_train.csv.gz"),
   destination_frame = 'gbm_train_hex'
)
gbm.tr.hex[,label.name] <- NULL
gbm.te.hex <- h2o.importFile(
   path = normalizePath("./gbm/gbm_levone_test.csv.gz"),
   destination_frame = 'gbm_test_hex'
)
# Deeplearning
dl.tr.hex <- h2o.importFile(
   path = normalizePath("./dl/dl_levone_train.csv.gz"),
   destination_frame = 'dl_train_hex'
)
dl.tr.hex[,label.name] <- NULL
dl.te.hex <- h2o.importFile(
   path = normalizePath("./dl/dl_levone_test.csv.gz"),
   destination_frame = 'dl_test_hex'
)
# GLM
glm.tr.hex <- h2o.importFile(
   path = normalizePath("./glm/glm_levone_train.csv.gz"),
   destination_frame = 'glm_train_hex'
)
glm.tr.hex[,label.name] <- NULL
glm.te.hex <- h2o.importFile(
   path = normalizePath("./glm/glm_levone_test.csv.gz"),
   destination_frame = 'glm_test_hex'
)
# RandomForest
rf.tr.hex <- h2o.importFile(
   path = normalizePath("./rf/rf_levone_train.csv.gz"),
   destination_frame = 'rf_train_hex'
)
rf.tr.hex[,label.name] <- NULL
rf.te.hex <- h2o.importFile(
   path = normalizePath("./rf/rf_levone_test.csv.gz"),
   destination_frame = 'rf_test_hex'
)
# NaiveBayes
gnb.tr.hex <- h2o.importFile(
   path = normalizePath("./gnb/gnb_levone_train.csv.gz"),
   destination_frame = 'gnb_train_hex'
)
gnb.tr.hex[,label.name] <- NULL
gnb.te.hex <- h2o.importFile(
   path = normalizePath("./gnb/gnb_levone_test.csv.gz"),
   destination_frame = 'gnb_test_hex'
)

## Merge level-one data
new.train.hex <- h2o.cbind(gbm.tr.hex, rf.tr.hex, dl.tr.hex, glm.tr.hex, gnb.tr.hex)
new.names <- h2o.colnames(new.train.hex)
new.train.hex <- h2o.cbind(new.train.hex, train.hex[,label.name])
new.test.hex <- h2o.cbind(gbm.te.hex, rf.te.hex, dl.te.hex, glm.te.hex, gnb.te.hex)

## Train metalearner
gbm.model <- h2o.gbm(x = new.names, y = label.name, training_frame = new.train.hex, 
                     nfolds = 5, seed = 2020, distribution = "multinomial", 
                     ntrees = 150, score_each_iteration = FALSE,
                     score_tree_interval = 5, stopping_rounds = 2, 
                     stopping_metric = "logloss", stopping_tolerance = 1e-4)
h2o.logloss(gbm.model@model$cross_validation_metrics)
gbm.model@parameters$ntrees

## Get predictions for the test set
gbm.pred.hex <- predict(gbm.model, new.test.hex)
gbm.pred.hex[,"predict"] <- NULL
gbm.pred.hex <- h2o.round(gbm.pred.hex, 6)
gbm.pred.hex <- h2o.cbind(test.hex[,"id"], gbm.pred.hex)
write.csv(
   as.data.frame(gbm.pred.hex), 
   file = gzfile('./stacking/stack_output.csv.gz'), 
   row.names = FALSE
)

## Variable ranking
var.rank <- h2o.varimp(gbm.model)
save(var.rank, file = "./stacking/var_rank.rda", compress = "bzip2")
