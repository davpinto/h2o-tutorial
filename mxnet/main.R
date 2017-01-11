## Load required packages
library("readr")
library("mxnet")
library("caret")
library("magrittr")
library("ggplot2")

options(scipen = 999)

## Read training data
tr.data <- readr::read_csv("./data/train.csv.zip")
te.data <- readr::read_csv("./data/test.csv.zip")
y <- factor(tr.data$target, levels = paste("Class", 1:9, sep = "_"))
x <- as.matrix(tr.data[,-c(1,95)])
x.te <- as.matrix(te.data[,-1])

## Transform data
x <- sqrt( x + 3/8 )
x.te <- sqrt( x.te + 3/8 )

## Split data
set.seed(2048)
val.idx <- createDataPartition(y, p = 0.3, list = FALSE)
y.num <- as.integer(y) - 1
x.val <- x[val.idx,]
x.tr <- x[-val.idx,]
y.val <- y.num[val.idx]
y.tr <- y.num[-val.idx]

## Multiclass Log-Loss
demo.metric.mlogloss <- mx.metric.custom("mlogloss", function(true.label, pred.prob) {
   pred.prob  <- t(pred.prob)
   true.label <- factor(true.label, levels = 0:max(true.label))
   
   ## Avoid extreme (0 and 1) probabilities
   eps <- 1e-15
   pred.prob <- pmin(pmax(pred.prob, eps), 1-eps)
   
   ## Transform labels into a binary matrix
   y.mat <- as.data.frame.matrix(
      table(1:length(true.label), true.label)
   )
   
   ## Compute log-loss
   log.loss <- (-1/nrow(y.mat)) * sum(y.mat*log(pred.prob))
   
   return(log.loss)
})

### BatchNorm Deep Model
# Deep NN Structure
input.dropout <- 0.05
hidden <- c(1024, 512, 256)
hidden.dropout <- c(0.5, 0.5, 0.5)
softmax <- mx.symbol.Variable("data") %>% 
   mx.symbol.Dropout(name='dp1', p = input.dropout) %>% 
   # Hidden layer 1
   mx.symbol.FullyConnected(name="fc1", num_hidden=hidden[1]) %>% 
   mx.symbol.Activation(name="relu1", act_type="relu") %>% 
   mx.symbol.BatchNorm(name="bn1") %>%
   mx.symbol.Dropout(name='dp2', p = hidden.dropout[1]) %>% 
   # Hidden layer 2
   mx.symbol.FullyConnected(name="fc2", num_hidden=hidden[2]) %>% 
   mx.symbol.Activation(name="relu2", act_type="relu") %>% 
   mx.symbol.BatchNorm(name="bn2") %>%
   mx.symbol.Dropout(name='dp3', p = hidden.dropout[2]) %>% 
   # Hidden layer 3
   mx.symbol.FullyConnected(name="fc3", num_hidden=hidden[3]) %>%
   mx.symbol.Activation(name="relu3", act_type="relu") %>%
   mx.symbol.BatchNorm(name="bn3") %>%
   mx.symbol.Dropout(name='dp4', p = hidden.dropout[3]) %>%
   # Output layer
   mx.symbol.FullyConnected(name="fc4", num_hidden=nlevels(y)) %>% 
   mx.symbol.SoftmaxOutput(name="sm")

## Model training
metric.logger <- mx.metric.logger$new()
mx.set.seed(2048)
mx.model <- mx.model.FeedForward.create(
   softmax, X=x.tr, y=y.tr, ctx=mx.cpu(), num.round=200, 
   array.layout="rowmajor", array.batch.size=4096, optimizer="adam", 
   # learning.rate=0.1, momentum = 0.9, wd = 1e-6,
   eval.metric=demo.metric.mlogloss, initializer=mx.init.uniform(0.05),
   eval.data=list(data=x.val, label=y.val), 
   epoch.end.callback = mx.callback.log.train.metric(1, metric.logger)
)

## Plot convergence
data.frame(
   epoch = 1:length(metric.logger$train),
   logloss = c(metric.logger$train, metric.logger$eval),
   class = rep(c("train", "valid"), each = length(metric.logger$train))
) %>% 
   dplyr::filter(epoch >= 50) %>% 
   ggplot(aes(x = epoch, y = logloss, color = class)) +
   geom_line(size = 0.5, alpha = 0.6) +
   geom_point(size = 1, alpha = 0.8) +
   geom_vline(xintercept = which.min(metric.logger$eval), linetype = "dashed", 
              size = 0.5, color = "black") +
   geom_hline(yintercept = min(metric.logger$eval), linetype = "dashed", 
              size = 0.5, color = "black") +
   labs(x = "Training Epoch", y = "Performance (LogLoss)") +
   scale_color_manual(name = "Data", values = c("red", "dodgerblue"))

### Model prediction for test data
mx.out <- predict(mx.model, x.te, array.layout = 'rowmajor')
mx.out <- round(t(mx.out), 6)
colnames(mx.out) <- levels(y)
mx.out <- cbind.data.frame(id = te.data$id, mx.out)
write.csv(
   mx.out,
   file = gzfile('./mxnet/mxnet_output.csv.gz'), 
   row.names = FALSE
)

### Blending
ensemble.size <- 15
pb <- txtProgressBar(0, ensemble.size, style = 3)
for (i in seq(ensemble.size)) {
   ## Train model
   mx.set.seed(i)
   mx.model <- mx.model.FeedForward.create(
      softmax, X=x, y=y.num, ctx=mx.cpu(), num.round=120, 
      array.layout="rowmajor", array.batch.size=4096, optimizer="adam", 
      initializer=mx.init.uniform(0.05), verbose = FALSE
   )
   
   setTxtProgressBar(pb, value = i)
   Sys.sleep(1)
   
   ## Predict probabilities
   mx.out <- predict(mx.model, x.te, array.layout = 'rowmajor')
   mx.out <- round(t(mx.out), 6)
   if (i == 1) {
      pred.prob <- mx.out
   } else {
      pred.prob <- pred.prob + mx.out
   }
}
close(pb)
pred.prob <- pred.prob / ensemble.size
pred.prob <- round(pred.prob, 6)
colnames(pred.prob) <- levels(y)
pred.prob <- cbind.data.frame(id = te.data$id, pred.prob)
write.csv(
   pred.prob,
   file = gzfile('./mxnet/mx_blend_output.csv.gz'), 
   row.names = FALSE
)
