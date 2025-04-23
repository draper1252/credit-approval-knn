# STEP 1: Load dataset
credit <- read.csv("crx.data", header = FALSE, na.strings = "?")

# Update column names to replace ambiguous symbols with descriptive names
# The original dataset's attribute names were anonymized for confidentiality purposes.
colnames(credit) <- c("Gender", "Age", "Debt", "Marital_Status", "Credit_History", 
                      "Education_Level", "Residential_Type", "Years_Employed", 
                      "Prior_Default", "Is_Employed", "Credit_Score", 
                      "Has_Drivers_License", "Citizen_Status", "Region", 
                      "Income", "Approval_Status")

# Display dataset information
names(credit)
str(credit)
summary(credit)


# STEP 2: Convert relevant columns to factors
cat_cols <- c("Gender", "Marital_Status", "Credit_History", "Education_Level",
              "Residential_Type", "Prior_Default", "Is_Employed",
              "Has_Drivers_License", "Citizen_Status", "Approval_Status")

credit[cat_cols] <- lapply(credit[cat_cols], as.factor)

str(credit)
summary(credit)


# STEP 3: Handle missing data using mice

# install.packages("mice")  # Uncomment if not already installed
library(mice)

# Perform imputation (m = 1 for one completed dataset)
imputed_data <- mice(credit, m = 1, method = 'pmm', seed = 123)

# Extract completed dataset
credit_complete <- complete(imputed_data, 1)

# Reapply factor conversion in case mice changed them
credit_complete[cat_cols] <- lapply(credit_complete[cat_cols], as.factor)

# Check structure and summary after imputation
str(credit_complete)
summary(credit_complete)


# STEP 4: Select and normalize numeric predictors

# Select only numeric columns for KNN
numeric_cols <- c("Age", "Debt", "Years_Employed", "Credit_Score", "Region", "Income")
credit_numeric <- credit_complete[, numeric_cols]

# Define normalize() function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization to numeric columns
credit_normalized <- as.data.frame(lapply(credit_numeric, normalize))


# STEP 5: Extract target data

# Extract target labels
credit_labels <- credit_complete$Approval_Status

# Check results
summary(credit_normalized)
str(credit_normalized)
table(credit_labels)


# STEP 6: Split data into training and test sets

set.seed(1)
rows <- sample(nrow(credit_normalized))
train.X <- credit_normalized[rows[1:483], ]
test.X  <- credit_normalized[rows[484:690], ]
train.Y <- credit_labels[rows[1:483]]
test.Y  <- credit_labels[rows[484:690]]

# STEP 7: Train KNN and find best k

# install.packages("class")  # Uncomment if not installed
library(class)

# Initialize vector to store test error rates for each k
err_values <- numeric(21)

# Loop over k values from 1 to 21
for (k in 1:21) {
  predicted <- knn(train.X, test.X, train.Y, k = k)
  err_values[k] <- mean(predicted != test.Y)
}

# Find the best k (lowest error)
best_k <- which.min(err_values)
cat("Optimal K:", best_k, "with error rate:", err_values[best_k], "\n")


# STEP 8: Visualize error across k values

# Plot 1: Error vs. 1/k (flexibility plot)
plot(1/(1:21), err_values,
     type = "b", col = "blue", pch = 19,
     main = "Test Error vs 1/K (Model Flexibility)",
     xlab = "1/K",
     ylab = "Test Error")

# Plot 2: Error vs. K with best K highlighted
plot(1:21, err_values,
     type = "b", col = "darkgreen", pch = 19,
     main = "Test Error vs K Value",
     xlab = "K",
     ylab = "Test Error")

# Highlight the best K on the plot
points(best_k, err_values[best_k],
       col = "red", pch = 19, cex = 1.5)

legend("bottomright",
       legend = "Best K",
       col = "red",
       pch = 19,
       pt.cex = 1.5)

# STEP 9: Evaluate final KNN model

final_pred <- knn(train.X, test.X, train.Y, k = best_k)

# Confusion matrix
table(Predicted = final_pred, Actual = test.Y)

# Overall accuracy
accuracy <- mean(final_pred == test.Y)
cat("Final model accuracy with K =", best_k, ":", accuracy, "\n")

