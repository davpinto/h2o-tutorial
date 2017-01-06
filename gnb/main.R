## Transform Variables
var_transform <- function(x) {
   log(x + 1)
}
train.hex[, input.names] <- apply(train.hex[, input.names], 2, var_transform)
test.hex[, input.names] <- apply(test.hex[, input.names], 2, var_transform)

## Feature extraction with PCA
pca.model <- h2o.prcomp(training_frame = h2o.rbind(train.hex[,input.names], 
                                                   test.hex[,input.names]),
                        x = input.names, k = 5, transform = "DEMEAN", 
                        pca_method = "GLRM", use_all_factor_levels = TRUE)
new.tr.hex <- h2o.predict(pca.model, train.hex[, input.names])
new.names <- h2o.colnames(new.tr.hex)
new.tr.hex <- h2o.cbind(new.tr.hex, train.hex[,label.name], train.hex[,"cv"])
new.te.hex <- h2o.predict(pca.model, test.hex[, input.names])

## Fit Gaussian Naive-Bayes
gnb.model <- h2o.naiveBayes(x = new.names, y = label.name, model_id = "best_model",
                            training_frame = new.tr.hex, fold_column = "cv",
                            keep_cross_validation_predictions = TRUE)
h2o.logloss(gnb.model@model$cross_validation_metrics)
h2o.saveModel(gnb.model, path = "./gnb", force = TRUE)

## Get predictions for the training cv folds
var.names <- paste("gnb", 1:h2o.nlevels(train.hex[,label.name]), sep = "_")
gnb.train.hex <- h2o.getFrame(gnb.model@model$cross_validation_holdout_predictions_frame_id$name)
gnb.train.hex[,"predict"] <- NULL
colnames(gnb.train.hex) <- var.names
gnb.train.hex <- h2o.round(gnb.train.hex, 6)
gnb.train.hex <- h2o.cbind(gnb.train.hex, train.hex[,label.name])
write.csv(
   as.data.frame(gnb.train.hex), 
   file = gzfile('./gnb/gnb_levone_train.csv.gz'), 
   row.names = FALSE
)

## Get predictions for the test set
gnb.test.hex <- predict(gnb.model, new.te.hex)
gnb.test.hex[,"predict"] <- NULL
gnb.test.hex <- h2o.round(gnb.test.hex, 6)
dt <- as.data.frame(gnb.test.hex)
names(dt) <- var.names
write.csv(
   dt, 
   file = gzfile('./gnb/gnb_levone_test.csv.gz'), 
   row.names = FALSE
)

## Save output for the test set
gnb.out.hex <- h2o.cbind(test.hex[,"id"], gnb.test.hex)
write.csv(
   as.data.frame(gnb.out.hex), 
   file = gzfile('./gnb/gnb_output.csv.gz'), 
   row.names = FALSE
)
