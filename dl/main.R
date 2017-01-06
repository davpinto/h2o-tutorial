## Transform Variables
var_transform <- function(x) {
   sqrt(x + 3/8)
}
train.hex[, input.names] <- apply(train.hex[, input.names], 2, var_transform)
test.hex[, input.names] <- apply(test.hex[, input.names], 2, var_transform)

## Blending
# *** References:
# 1. http://pt.slideshare.net/0xdata/h2o-world-top-10-deep-learning-tips-tricks-arno-candel
# 2. https://github.com/h2oai/h2o-2/blob/master/R/examples/Kaggle/MeetupKaggleAfricaSoil.R
# 3. https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14134/deep-learning-h2o-0-44
ensemble.size <- 15 
for (i in 1:ensemble.size) {
   ## Fit model using 5-fold CV
   dl.model <- h2o.deeplearning(x = input.names, y = label.name, training_frame = train.hex,
                                fold_column = "cv", model_id = paste0("dl_model_", i),
                                loss = "CrossEntropy", distribution = "multinomial",
                                epochs = 50, stopping_rounds = 0,
                                score_interval = 30, score_training_samples = 1e4, 
                                max_runtime_secs = 30 * 60, keep_cross_validation_predictions = TRUE, 
                                seed = i, activation = "RectifierWithDropout",
                                hidden = c(1024, 512), hidden_dropout_ratios = rep(0.3, 2),
                                input_dropout_ratio = 0.15, l1 = 1e-5, l2 = 1e-5,
                                rho = 0.99, epsilon = 1e-8, 
                                train_samples_per_iteration = 2e3, max_w2 = 10)
   
   ## Get predictions for the training cv folds
   train.pred.hex <- h2o.getFrame(dl.model@model$cross_validation_holdout_predictions_frame_id$name)
   train.pred.hex[,"predict"] <- NULL
   train.pred.hex <- h2o.round(train.pred.hex, 6)
   
   ## Get predictions for the test set
   test.pred.hex <- predict(dl.model, test.hex)
   test.pred.hex[,"predict"] <- NULL
   test.pred.hex <- h2o.round(test.pred.hex, 6)
   
   ## Print performance
   print(paste(
      "Model", i, "|", "logloss = ", 
      round(h2o.logloss(dl.model@model$cross_validation_metrics), 4)
   ))
   
   ## Sum results
   if (i == 1) {
      dl.train.hex <- train.pred.hex
      dl.test.hex <- test.pred.hex
   } else {
      dl.train.hex <- dl.train.hex + train.pred.hex
      dl.test.hex <- dl.test.hex + test.pred.hex
   }
}

## Average results
dl.train.hex <- dl.train.hex / ensemble.size
dl.test.hex <- dl.test.hex / ensemble.size

## Save level-one data
var.names <- paste("dl", 1:h2o.nlevels(train.hex[,label.name]), sep = "_")
# Train
dl.train.hex <- h2o.cbind(dl.train.hex, train.hex[,label.name])
dt <- as.data.frame(dl.train.hex)
names(dt) <- c(var.names, label.name)
write.csv(
   dt,
   file = gzfile('./dl/dl_levone_train.csv.gz'),
   row.names = FALSE
)
# Test
dt <- as.data.frame(dl.test.hex)
names(dt) <- var.names
write.csv(
   dt, 
   file = gzfile('./dl/dl_levone_test.csv.gz'), 
   row.names = FALSE
)

## Save output for the test set
dl.out.hex <- h2o.cbind(test.hex[,"id"], dl.test.hex)
write.csv(
   as.data.frame(dl.out.hex), 
   file = gzfile('./dl/dl_output.csv.gz'), 
   row.names = FALSE
)
