
library("neuralnet")
library("caret")
library("glmnet")
library("ncpen")
library("abess")
library("flare")
mutual_fund <- read_csv("mutual fund.csv")
# The function of prediction error
Prediction_Error <- function(y,yhat){mean((y-yhat)^2)}

set.seed(2500)

#Load fund data
fund_y = as.data.frame(mutual_fund[,1])
fund_x = as.data.frame(mutual_fund[,-1])

# Split data into train and test sets
n <- nrow(fund_y)  # Number of observations in the dataset
train_size <- round(0.8 * n)  # Number of observations for the training set
train_ind <- sample(1:n, train_size)  # Randomly select indices for the training set
test_ind <- setdiff(1:n, train_ind)  # Get the remaining indices for the test set

y_train <- fund_y$y[train_ind]
X_train <- fund_x[train_ind, ]
X_train_mean <- apply(X_train, 2, mean)
X_train_scale <- scale(X_train, X_train_mean, F)
y_train_mean <- mean(y_train)
y_train_scale <- scale(y_train, y_train_mean, F)
train_scale_df <- data.frame(y_train_scale, X_train_scale)
y_test <- fund_y$y[test_ind]
X_test <- fund_x[test_ind, ]
X_test_scale <- scale(X_test, X_train_mean, F)
y_test_scale <- scale(y_test, y_train_mean, F)

colnames(train_scale_df)[-1] <- paste("X", 1:(ncol(train_scale_df)-1), sep = "")
colnames(X_train_scale) <- paste0("X", 1:ncol(X_train_scale))

colnames(X_test_scale) <- paste0("X", 1:ncol(X_test_scale))


p=ncol(X_train_scale)


###Use ENET
# Convert y_train to a numeric vector (train function wants it)
y_train_scale_numeric <- as.numeric(y_train_scale)

#ENET using Cross Validation
#Find the optimal parameters using validation data
optimal_param_enet <- train(X_train_scale, y_train_scale_numeric,
                            method= "glmnet",
                            tuneGrid=expand.grid(alpha=seq(0.1,0.9,length=10),
                                                 lambda = seq(0.001, 0.02, length=20)),
                            metric = "RMSE",
                            trControl=trainControl(method = "cv", number = 10))
# Get the lambda and alpha combination that minimizes RMSE
best_lambda_enet <- optimal_param_enet$bestTune$lambda
best_alpha_enet <- optimal_param_enet$bestTune$alpha
# Re-fit the model with the selected lambda and alpha
mod_enet <- glmnet(X_train_scale, y_train_scale, alpha = best_alpha_enet, lambda = best_lambda_enet)
b.enet = as.vector(coef(mod_enet))[-1]


###Use ALASSO
#LASSO using Cross Validation
# Fit LASSO model using the training and validation data and select tuning parameters
lasso_cv <- cv.glmnet(X_train_scale, y_train_scale, type.measure = "mse", alpha = 1, nfolds = 10)
# Find the optimal lambda value chosen by cross-validation
optimal_lambda_lasso <- lasso_cv$lambda.min
# Fit the LASSO model using the optimal lambda and the full training data
mod_lasso <- glmnet(X_train_scale, y_train_scale, alpha = 1, lambda = optimal_lambda_lasso)
b.lasso = as.vector(coef(mod_lasso))[-1]

#Adaptive LASSO with 10-fold CV
weight <- b.lasso
weight <- ifelse(weight == 0, 0.00001, weight)
alasso_cv <- cv.glmnet(X_train_scale, y_train_scale, type.measure = "mse", alpha = 1, nfolds = 10,
                       penalty.factor = 1 / abs(weight), keep = TRUE)
optimal_lambda_alasso = alasso_cv$lambda.min
best_alasso_coef1 <- coef(alasso_cv, s = alasso_cv$lambda.min)
b.alasso = as.vector(best_alasso_coef1)[-1]

# Formula of the enet model
# Get the column names of your predictor variables
predictor_names <- colnames(X_train_scale)
# Get the indices of the non-zero coefficients
non_zero_indices_FM <- which(b.enet != 0)
# Get the corresponding predictor names for non-zero coefficients
selected_predictors_FM <- predictor_names[non_zero_indices_FM]

xcount.FM <- c(0,paste(selected_predictors_FM, sep=""))
Formula_enet <- as.formula(paste("y_train_scale~", paste(xcount.FM, collapse= "+")))

#Formula of the alasso model
# Get the indices of the non-zero coefficients
non_zero_indices_SM <- which(b.alasso != 0)
# Get the corresponding predictor names for non-zero coefficients
selected_predictors_SM <- predictor_names[non_zero_indices_SM]

xcount.SM <- c(0,paste(selected_predictors_SM, sep=""))
Formula_alasso <- as.formula(paste("y_train_scale~", paste(xcount.SM, collapse= "+")))

###Neural Network

#formula of the model without variable selection
xcount <- c(0,paste("X", 1:p, sep=""))
Formula_full <- as.formula(paste("y_train_scale~", paste(xcount, collapse= "+")))

#formula of the model with enet (already found in the previous chunk)
#formula of the model with alasso (already found in the previous chunk)

#fitting the neural networks
nn_mutualfunds_1 = neuralnet(Formula_full, data=train_scale_df, hidden=1, linear.output = TRUE)
nn_mutualfunds_2 = neuralnet(Formula_enet, data=train_scale_df, hidden=1, linear.output = TRUE)
nn_mutualfunds_3 = neuralnet(Formula_alasso, data=train_scale_df, hidden=1, linear.output = TRUE)

# Find the column indices of selected predictor variables for each model
indices_selected_enet <- which(colnames(X_test_scale) %in% selected_predictors_FM)
indices_selected_alasso <- which(colnames(X_test_scale) %in% selected_predictors_SM)

# Create data frames for input data with selected predictors
X_test_selected_enet <- X_test_scale[, indices_selected_enet]
X_test_selected_alasso <- X_test_scale[, indices_selected_alasso]

#prediction using test set
y_hat_full = compute(nn_mutualfunds_1,X_test_scale)$net.result
y_hat_enet = compute(nn_mutualfunds_2,X_test_selected_enet)$net.result
y_hat_alasso = compute(nn_mutualfunds_3,X_test_selected_alasso)$net.result

pred_error_full = Prediction_Error(y_test_scale, y_hat_full)
pred_error_enet = Prediction_Error(y_test_scale, y_hat_enet)
pred_error_alasso = Prediction_Error(y_test_scale, y_hat_alasso)
