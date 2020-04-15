# Neural Networks:
# Regression Problems and Functional API
# install the core Keras Libraries + Tensorflow
library(keras)
install_keras()
# loading the in-built dataset
boston_housing <- dataset_boston_housing()

c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels)   %<-% boston_housing$test

# Test data is not used when computing the mean and std
# Normalizing the data
train_data <- scale(train_data)

# use the means and stdevs from the traiing set to normalize test set
col_means_train <- attr(train_data, "scaled:center")
col_stdevs_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center = col_means_train,
                   scale = col_stdevs_train)

# Functional API has two parts: Inputs and Outputs
# Input Layer
inputs <- layer_input(shape = dim(train_data)[2])

# Outputs = Input + Dense Layers
predictions <- inputs %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)

# create and compile the model
model <- keras_model(inputs = inputs, outputs = predictions)
model %>% compile(
  optimizer = "rmsprop",
  loss      = "mse",
  metrics   = list("mean_absolute_error")
)

# train the model with the training data
model %>% fit(train_data, train_labels, epochs=30, batch_size=100)

# Test Performance
score <- model %>% evaluate(test_data, test_labels)
cat("Test loss:", score$loss, "\n")
cat("Test Absolute Error:", score$mean_absolute_error, "\n")

# Complex Architectures using Functional API
# Input Layer
inputs_func <- layer_input(shape = dim(train_data)[2])

#Output Layer = Input Features after third hidden layer
predictions_func <- inputs_func %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu")

# Re-use the input featres after the third hidden layer
main_output <- layer_concatenate(c(predictions_func, inputs_func)) %>%
  layer_dense(units = 1)

# create and compile the model
model_func <- keras_model(inputs = inputs_func, 
                          outputs = main_output)
model_func %>% compile(
  optimizer = "rmsprop",
  loss      = "mse",
  metrics   = list("mean_absolute_error")
)
summary(model_func)
# train the model with the training data
model %>% fit(train_data, train_labels, epochs=30, batch_size=100)

# Test Performance
score_func <- model_func %>% evaluate(test_data, test_labels)
cat("Test loss:", score_func$loss, "\n")
cat("Test Absolute Error:", score_func$mean_absolute_error, "\n")
