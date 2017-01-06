## Random search for parameter tuning
rf.params <- list(
   max_depth = seq(2, 24, by = 2),
   min_rows = seq(10, 150, by = 10),                 # minimum observations required in a terminal node or leaf
   sample_rate = seq(0.1, 1, by = 0.1),              # row sample rate per tree (boostrap = 0.632)
   mtries = round(seq(0.1, 1, by = 0.1) * 93),       # column sample rate per split
   col_sample_rate_per_tree = seq(0.1, 1, by = 0.1),
   nbins = round(2 ^ seq(2, 6, length = 15)),        # number of levels for numerical features discretization
   histogram_type = c("UniformAdaptive", "Random", "QuantilesGlobal", "RoundRobin")
)
rf.grid <- h2o.grid(
   algorithm = "randomForest", grid_id = "rf_grid",
   x = input.names, y = label.name, training_frame = train.hex,
   fold_column = "cv", ntrees = 150, 
   stopping_rounds = 2, stopping_metric = 'logloss', stopping_tolerance = 1e-5,
   score_each_iteration = FALSE, score_tree_interval = 15,
   keep_cross_validation_predictions = TRUE, 
   seed = 2020, max_runtime_secs = 15 * 60,
   search_criteria = list(
      strategy = "RandomDiscrete", max_models = 50, 
      max_runtime_secs = 6 * 60 * 60, seed = 2020
   ),
   hyper_params = rf.params
)

## Get best parameters
grid.table <- h2o.getGrid("rf_grid", sort_by = "logloss", decreasing = FALSE)@summary_table
save(grid.table, file = "./rf/grid_table.rda", compress = "bzip2")
best.rf <- h2o.getModel(grid.table$model_ids[1])
h2o.logloss(best.rf@model$cross_validation_metrics)
h2o.saveModel(best.rf, path = "./rf", force = TRUE)
file.rename(from = paste("rf", best.rf@model_id, sep = "/"), to = "rf/best_model")
best.params <- best.rf@allparameters
save(best.params, file = "./rf/best_params.rda", compress = "bzip2")

## Train best model with more iterations
load("./rf/best_params.rda")
best.rf <- h2o.randomForest(x = input.names, y = label.name, training_frame = train.hex,
                            model_id = "best_model",
                            fold_column = "cv", ntrees = 500,
                            stopping_rounds = 2, stopping_metric = 'logloss', stopping_tolerance = 1e-5,
                            score_each_iteration = FALSE, score_tree_interval = 15,
                            keep_cross_validation_predictions = TRUE, seed = 2020, 
                            max_depth = best.params$max_depth,
                            min_rows = best.params$min_rows,
                            sample_rate = best.params$sample_rate,
                            mtries = best.params$mtries,
                            col_sample_rate_per_tree = best.params$col_sample_rate_per_tree,
                            nbins = best.params$nbins,
                            histogram_type = best.params$histogram_type)
h2o.logloss(best.rf@model$cross_validation_metrics)
best.rf@parameters$ntrees
h2o.saveModel(best.rf, path = "./rf", force = TRUE)

## Get predictions for the training cv folds
var.names <- paste("rf", 1:h2o.nlevels(train.hex[,label.name]), sep = "_")
rf.train.hex <- h2o.getFrame(best.rf@model$cross_validation_holdout_predictions_frame_id$name)
rf.train.hex[,"predict"] <- NULL
colnames(rf.train.hex) <- var.names
rf.train.hex <- h2o.round(rf.train.hex, 6)
rf.train.hex <- h2o.cbind(rf.train.hex, train.hex[,label.name])
write.csv(
   as.data.frame(rf.train.hex), 
   file = gzfile('./rf/rf_levone_train.csv.gz'), 
   row.names = FALSE
)
# h2o.exportFile(rf.train.hex, path = "./rf/rf_levone_train.csv", force = TRUE)

## Get predictions for the test set
rf.test.hex <- predict(best.rf, test.hex)
rf.test.hex[,"predict"] <- NULL
rf.test.hex <- h2o.round(rf.test.hex, 6)
write.csv(
   as.data.frame(rf.test.hex), 
   file = gzfile('./rf/rf_levone_test.csv.gz'), 
   col.names = var.names,
   row.names = FALSE
)

## Save output for the test set
rf.out.hex <- h2o.cbind(test.hex[,"id"], rf.test.hex)
write.csv(
   as.data.frame(rf.out.hex), 
   file = gzfile('./rf/rf_output.csv.gz'), 
   row.names = FALSE
)

## Feature importance
feat.rank <- h2o.varimp(best.rf)
save(feat.rank, file = "./rf/feature_rank.rda", compress = "bzip2")
