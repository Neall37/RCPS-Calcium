Sure! Here's how you could structure your functions documentation in Markdown format for a GitHub README or similar documentation file:

```markdown
# Function Documentation

## Process Calcium Imaging Data

### Description

Processes calcium imaging data and outputs a JSON file including identified neurons and their corresponding pixel coordinates.

### Usage

```python
process_calcium_imaging_data(movie, min_area, n_components=200, activity_threshold=0.9, mu=0.5)
```

### Arguments

- `movie`: numpy array converted from TIFF or other file formats.
- `min_area`: the minimum area in pixels below which objects are not recognized as neurons.
- `n_components`: number of principal components for dimensionality reduction. Default is 200.
- `activity_threshold`: threshold for activity recognition. Default is 0.9.
- `mu`: the weight given to the spatial component relative to the temporal component in the ICA algorithm. Increasing `mu` makes spatial information more important.

### Output

- A JSON file containing the details of every identified neuron and their corresponding pixel coordinates.

---

## Load and Preprocess Data

### Description

Loads and preprocesses the training and testing data from pandas dataframes.

### Usage

```python
load_and_preprocess(df, df_test)
```

### Arguments

- `df`: Training pandas dataframe containing all metrics and required information.
- `df_test`: Testing pandas dataframe containing all metrics and required information.

### Output

- `X_train`: Preprocessed training data features.
- `X_test`: Preprocessed testing data features.
- `y_train`: Training data labels.
- `y_test`: Testing data labels.

---

## SVM Model

### Description

Trains and evaluates a Support Vector Machine (SVM) model using the provided training and testing data.

### Usage

```python
svm_model(X_train, X_test, y_train, y_test)
```

### Arguments

- `X_train`: Training data features.
- `X_test`: Testing data features.
- `y_train`: Training data labels.
- `y_test`: Testing data labels.

### Output

- Prints the training and testing accuracy of the model.

---

## Logistic Regression Model

### Description

Trains and evaluates a Logistic Regression model using the provided training and testing data.

### Usage

```python
logistic_regression(X_train, X_test, y_train, y_test)
```

### Arguments

- `X_train`: Training data features.
- `X_test`: Testing data features.
- `y_train`: Training data labels.
- `y_test`: Testing data labels.

### Output

- Prints the best hyperparameters, training, and testing accuracy of the model.

---

## KNN Model

### Description

Trains and evaluates a K-Nearest Neighbors (KNN) model using the provided training and testing data.

### Usage

```python
knn_model(X_train, X_test, y_train, y_test)
```

### Arguments

- `X_train`: Training data features.
- `X_test`: Testing data features.
- `y_train`: Training data labels.
- `y_test`: Testing data labels.

### Output

- Prints the best hyperparameters and the testing accuracy of the model.

---
```

This format includes sections for each function, with clear descriptions, usage instructions, parameter explanations, and expected outputs. Adjustments to the specific implementation details can be made based on the actual code and requirements.
