

# Step 1: Load the data
# Load the necessary library
library(neuralnet)
library(dplyr)
# Step 1: Load and prepare the data
data <- read.csv("Olivine.xlsx - Data (1).csv")

ppm_columns <- data[, grep("ppm$", names(data))]
data <-ppm_columns # only ppms
head(data)
data <- data %>%
  mutate_all(as.numeric)


data<- na.omit(data)
str(data)
# Step 2: Normalize the data (using min-max scaling)
normalized_data <- as.data.frame(scale(data))
print(normalized_data)

# Step 3: Split the data into training and test sets (70% training, 30% test)
set.seed(42)  # For reproducibility
train_index <- sample(1:nrow(normalized_data), 0.7 * nrow(normalized_data))
train_data <- normalized_data[train_index, ]
test_data <- normalized_data[-train_index, ]

# Step 4: Define the neural network architecture
# Replace 'response_var' with the name of your response variable
response_var <- "U238_ppm_2SE.int."
predictor_vars <- setdiff(names(normalized_data), response_var)

# One-hot encode the response variable
train_data$response_var_onehot <- ifelse(train_data$response_var == 0, 0, 1)
train_data$response_var <- NULL

formula <- as.formula(paste("response_var_onehot", "~", paste(predictor_vars, collapse = "+")))

# Set up the neural network
nn <- neuralnet(formula, data = train_data, hidden = c(10, 5), act.fct = "logistic")

# Step 5: Train the neural network
# To train the neural network, simply run the neuralnet function, as it performs training by default.

# Step 6: Evaluate the neural network
# Use the predict function to make predictions on the test data
test_data$response_var_onehot <- ifelse(test_data$response_var == 0, 0, 1)
test_data$response_var <- NULL

predictions <- predict(nn, test_data)

# Calculate the mean squared error (MSE) or any other appropriate evaluation metric
mse <- mean((test_data$response_var_onehot - predictions)^2)
cat("Mean Squared Error (MSE) for the neural network:", mse, "\n")

data <- read.csv("new.csv")

# Step 2: Load the necessary library (typically already loaded by default)
library(stats)

# Step 3: Fit the GLM model
response_var <- "U238_ppm_2SE.int."  # Replace with the name of your response variable
predictor_vars <- c("Si29_ppm_2SE.int.", "S34_ppm_2SE.int.", "Ti47_ppm_2SE.int.")  # Replace with the names of your predictor variables

# Create the formula for the model
formula <- as.formula(paste(response_var, "~", paste(predictor_vars, collapse = "+")))

# Fit the GLM model
glm_model <- glm(formula, data = data, family = gaussian)

# Print the summary of the model
summary(glm_model)


library(foreign)
library(dplyr)
library(ggplot2)
# Replace "your_file_path.arff" with the actual path to your .arff file
dat <- read.arff("AP_Colon_Kidney.arff")


# inspect data set
dim(dat)  # count rows and colums
head(dat) # show first few policies
tail(dat) # show last few policies
colnames(dat)
library(rpart) # "Recursive partitioning for classification, regression and survival trees". Default metric: Gini impurity
tree <- rpart(cbind(data["1438_at","1405_i_at") ~ "1552365_at"+"1552365_at" + "1552519_at" + "1552519_at"+ "1552628_a_at" +"1553117_a_at" +"1553155_x_at"+ "1554152_a_at"+ "1554899_s_at", dat, method="poisson", control=rpart.control(maxdepth=3,cp=0.001))
#            complexity-parameter cp is used to control the number of splits

library(repr) # Scale plot size:
options(repr.plot.width=12, repr.plot.height = 7)

library(rpart.plot)
rpart.plot(tree) # display decision tree

# View distribution of all claims
table(dat$"201240_s_at")

dat %>%
group_by("121_at") %>%
summarise_at(vars("1438_at","1405_i_at"), list(~sum(.), ~mean(.)))
# for later use (Ch.5) we create five 20%-subsamples ("folds") and take the last fold as the holdout data set
k <- 5

# Install and load necessary libraries

library(MASS)
library(neuralnet)

# Function to simulate high-dimensional data
simulate_high_dimensional_data <- function(num_samples, num_features) {
X <- matrix(rnorm(num_samples * num_features), nrow = num_samples, ncol = num_features)
true_coefficients <- rnorm(num_features)
noise <- 0.1 * rnorm(num_samples)
y <- X %*% true_coefficients + noise
return(list(X = X, y = y))
}


num_samples <- 1000
num_features <- 100
data <- simulate_high_dimensional_data(num_samples, num_features)

set.seed(42)
train_indices <- sample(1:num_samples, size = 0.8 * num_samples, replace = FALSE)
X_train <- data$X[train_indices, ]
X_test <- data$X[-train_indices, ]
y_train <- data$y[train_indices]
y_test <- data$y[-train_indices]

# Train a Generalized Linear Model (GLM)
glm_model <- lm(y_train ~ X_train)


glm_predictions <- predict(glm_model, newdata = as.data.frame(X_test))

# Calculate the mean squared error for the GLM model
glm_mse <- mean((y_test - glm_predictions)^2)
print(paste("GLM Mean Squared Error:", round(glm_mse, 4)))

# Train a Neural Network
nn_model <- neuralnet(y_train ~ X_train, hidden = c(64, 32), linear.output = TRUE)

#
nn_predictions <- compute(nn_model, as.data.frame(X_test))$net.result


nn_mse <- mean((y_test - nn_predictions)^2)
print(paste("Neural Network Mean Squared Error:", round(nn_mse, 4)))

library(tidyverse)
library(keras)
library(caret)
library(MASS)

simulate_high_dimensional_data <- function(num_samples, num_features) {
X <- matrix(rnorm(num_samples * num_features), ncol = num_features)
true_coefficients <- rnorm(num_features)
noise <- 0.1 * rnorm(num_samples)
y <- X %*% true_coefficients + noise
list(X = X, y = y)
}

# Simulate high-dimensional data with 1000 samples and 100 features
num_samples <- 1000
num_features <- 100
data <- simulate_high_dimensional_data(num_samples, num_features)


set.seed(42)
index <- createDataPartition(data$y, p = 0.8, list = FALSE)
train_data <- data$X[index, ]
test_data <- data$X[-index, ]
y_train <- data$y[index]
y_test <- data$y[-index]


glm_model <- lm(y_train ~ ., data = as.data.frame(cbind(y_train, train_data)))

glm_predictions <- predict(glm_model, newdata = as.data.frame(test_data))


glm_mse <- mean((glm_predictions - y_test)^2)
cat(paste("GLM Mean Squared Error:", round(glm_mse, 4), "\n"))


nn_model <- keras_model_sequential() %>%
layer_dense(units = 64, input_shape = num_features, activation = 'relu') %>%
layer_dense(units = 32, activation = 'relu') %>%
layer_dense(units = 1, activation = 'linear')

compile(nn_model, loss = 'mse', optimizer = optimizer_adam(lr = 0.001))
history <- fit(nn_model, train_data, y_train, epochs = 50, batch_size = 32, verbose = 1)


nn_predictions <- predict(nn_model, test_data)

nn_mse <- mean((nn_predictions - y_test)^2)
cat(paste("Neural Network Mean Squared Error:", round(nn_mse, 4), "\n"))
