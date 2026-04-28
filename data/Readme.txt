This project uses the UCI Heart Disease dataset to build a neural network from scratch using only NumPy, focusing on understanding core ANN mechanics and data preprocessing decisions.
Type:- Tabular
Task:- Binary Classification(disease or not)
Samples:- 920
Features:- 16(before preprocessing) and 18(after preprocessing)

MISSING VALUES HANDLING
-Columns with more than 50% missing values were dropped as imputing them would introduce significant noise and reduce model reliability.
-Columns with<50% were filled accordingly i.e. For numerical values mean was taken and for binary and categorical we used mode

ENCODING STRATEGY
-Binary columns were converted to integer type for better performance with ANN
-Categorical columns were converted to Binary using one hot encoding
-Drop first strategy was used to avoid multicollinearity

FEATURE SCALING
-Standardization applied (zero mean, unit standard deviation)
-Ensures stable gradient descent and prevents feature dominance(as the Dataset contains values over different ranges for diff features)

FEATURE TARGET SPLIT
-Features(X): All inputs
-Target(y): 'num'column(converted to binary)

SHAPE TRANSFORMATION
-Input reshaped to (features, samples) using transpose
-Required for matrix-based ANN implementation

