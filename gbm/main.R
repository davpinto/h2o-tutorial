## Random search for parameter tuning
## *** References: 
## 1. https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
## 2. http://blog.h2o.ai/2016/06/h2o-gbm-tuning-tutorial-for-r/
gbm.params <- list(
   max_depth = seq(2, 24, by = 2),
   min_rows = seq(10, 150, by = 10),                 # minimum observations required in a terminal node or leaf
   sample_rate = seq(0.1, 1, by = 0.1),              # row sample rate per tree (boostrap = 0.632)
   col_sample_rate = seq(0.1, 1, by = 0.1),          # column sample rate per split
   col_sample_rate_per_tree = seq(0.1, 1, by = 0.1),
   nbins = round(2 ^ seq(2, 6, length = 15)),        # number of levels for numerical features discretization
   histogram_type = c("UniformAdaptive", "Random", "QuantilesGlobal", "RoundRobin")
)
gbm.grid <- h2o.grid(
   algorithm = "gbm", grid_id = "gbm_grid",
   x = input.names, y = label.name, training_frame = train.hex,
   fold_column = "cv", distribution = "multinomial", ntrees = 500, 
   learn_rate = 0.1, learn_rate_annealing = 0.995, 
   stopping_rounds = 2, stopping_metric = 'logloss', stopping_tolerance = 1e-5,
   score_each_iteration = FALSE, score_tree_interval = 10,
   keep_cross_validation_predictions = TRUE, 
   seed = 2020, max_runtime_secs = 30 * 60,
   search_criteria = list(
      strategy = "RandomDiscrete", max_models = 25, 
      max_runtime_secs = 12 * 60 * 60, seed = 2020
   ),
   hyper_params = gbm.params
)

## Get best model
grid.table <- h2o.getGrid("gbm_grid", sort_by = "logloss", decreasing = FALSE)@summary_table
save(grid.table, file = "./gbm/grid_table.rda", compress = "bzip2")
best.gbm <- h2o.getModel(grid.table$model_ids[1])
h2o.logloss(best.gbm@model$cross_validation_metrics)
h2o.saveModel(best.gbm, path = "./gbm", force = TRUE)
file.rename(from = paste("gbm", grid.table$model_ids[1], sep = "/"), to = "gbm/best_model")
best.params <- best.gbm@allparameters
save(best.params, file = "./gbm/best_params.rda", compress = "bzip2")

## Get predictions for the training cv folds
var.names <- paste("gbm", 1:h2o.nlevels(train.hex[,label.name]), sep = "_")
gbm.train.hex <- h2o.getFrame(best.gbm@model$cross_validation_holdout_predictions_frame_id$name)
gbm.train.hex[,"predict"] <- NULL
colnames(gbm.train.hex) <- var.names
gbm.train.hex <- h2o.round(gbm.train.hex, 6)
gbm.train.hex <- h2o.cbind(gbm.train.hex, train.hex[,label.name])
write.csv(
   as.data.frame(gbm.train.hex), 
   file = gzfile('./gbm/gbm_levone_train.csv.gz'), 
   row.names = FALSE
)
# h2o.exportFile(gbm.train.hex, path = "./gbm/gbm_levone_train.csv", force = TRUE)

## Get predictions for the test set
gbm.test.hex <- predict(best.gbm, test.hex)
gbm.test.hex[,"predict"] <- NULL
gbm.test.hex <- h2o.round(gbm.test.hex, 6)
write.csv(
   as.data.frame(gbm.test.hex), 
   file = gzfile('./gbm/gbm_levone_test.csv.gz'), 
   col.names = var.names,
   row.names = FALSE
)

## Save output for the test set
gbm.out.hex <- h2o.cbind(test.hex[,"id"], gbm.test.hex)
write.csv(
   as.data.frame(gbm.out.hex), 
   file = gzfile('./gbm/gbm_output.csv.gz'), 
   row.names = FALSE
)

## Feature importance
feat.rank <- h2o.varimp(best.gbm)
save(feat.rank, file = "./gbm/feature_rank.rda", compress = "bzip2")
