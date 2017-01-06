## Random search for parameter tuning
glm.params <- list(
   alpha = seq(0.1, 0.9, length = 10),
   lambda = 2 ^ seq(15, -15, length = 50)
)
glm.grid <- h2o.grid(
   algorithm = "glm", grid_id = "glm_grid", family = "multinomial",
   x = input.names, y = label.name, training_frame = train.hex,
   fold_column = "cv", solver = "IRLSM", max_iterations = 5e2,
   standardize = TRUE, lambda_search = FALSE, 
   early_stopping = TRUE, intercept = TRUE,
   keep_cross_validation_predictions = TRUE,
   search_criteria = list(
      strategy = "RandomDiscrete", max_models = 50, 
      max_runtime_secs = 6 * 60 * 60, seed = 2020
   ),
   hyper_params = glm.params
)

## Get best parameters
grid.table <- h2o.getGrid("glm_grid", sort_by = "logloss", decreasing = FALSE)@summary_table
save(grid.table, file = "./glm/grid_table.rda", compress = "bzip2")
best.glm <- h2o.getModel(grid.table$model_ids[1])
h2o.logloss(best.glm@model$cross_validation_metrics)
h2o.saveModel(best.glm, path = "./glm", force = TRUE)
file.rename(from = paste("glm", best.glm@model_id, sep = "/"), to = "glm/best_model")
best.params <- best.glm@allparameters
save(best.params, file = "./glm/best_params.rda", compress = "bzip2")

## Get predictions for the training cv folds
var.names <- paste("glm", 1:h2o.nlevels(train.hex[,label.name]), sep = "_")
glm.train.hex <- h2o.getFrame(best.glm@model$cross_validation_holdout_predictions_frame_id$name)
glm.train.hex[,"predict"] <- NULL
colnames(glm.train.hex) <- var.names
glm.train.hex <- h2o.round(glm.train.hex, 6)
glm.train.hex <- h2o.cbind(glm.train.hex, train.hex[,label.name])
write.csv(
   as.data.frame(glm.train.hex), 
   file = gzfile('./glm/glm_levone_train.csv.gz'), 
   row.names = FALSE
)
# h2o.exportFile(glm.train.hex, path = "./glm/glm_levone_train.csv", force = TRUE)

## Get predictions for the test set
glm.test.hex <- predict(best.glm, test.hex)
glm.test.hex[,"predict"] <- NULL
glm.test.hex <- h2o.round(glm.test.hex, 6)
write.csv(
   as.data.frame(glm.test.hex), 
   file = gzfile('./glm/glm_levone_test.csv.gz'), 
   col.names = var.names,
   row.names = FALSE
)

## Save output for the test set
glm.out.hex <- h2o.cbind(test.hex[,"id"], glm.test.hex)
write.csv(
   as.data.frame(glm.out.hex), 
   file = gzfile('./glm/glm_output.csv.gz'), 
   row.names = FALSE
)

## Feature importance
feat.rank <- h2o.varimp(best.glm)
save(feat.rank, file = "./glm/feature_rank.rda", compress = "bzip2")
