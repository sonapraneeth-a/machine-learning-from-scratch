# Machine learning codes from scratch

**Author:** [Sona Praneeth Akula](https://sonapraneeth-a.github.io/)

[![Build Status](https://travis-ci.org/sonapraneeth-a/machine-learning-from-scratch.svg?branch=master)](https://travis-ci.org/sonapraneeth-a/machine-learning-from-scratch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Linear Regression

### Formulation

### Parameters

The following parameters have given the best accuracy on kaggle
- *Gradient Descent Method :* Batch
- *Learning rate :* 0.9
- *Regularization constant :* 0.009
- *Maximum number of iterations :* 52000
- *Norm :* L<sub>2</sub> norm
- *Initial weights initialization:* Uniform
- *Cross validation :* k-fold with k = 5. Gave best accuracy on 5<sup>th</sup> fold (For estimating the best hyper-parameter lambda. That code is not submitted as it takes long time to iterate among all posssible values to find the best lambda)
- *Feature Enginering :* Performed square of all normalized features and appended these features to the original training features
- *Outliers :* = [369, 372, 373, 413]

## Running Instructions

```bash
$ python3 regression_test.py
```

## Steps of Implementation

- Removing of outliers corresponding to indexes = [369, 372, 373, 413]
- Split the data into train and validation set with k fold cross validation of k = 5
- Perform normalization of data
- Square each feature and append to the original set of features increasing the feature space
- Started with uniform weight initializations
- Perform Linear regression (fit the data) and estimate the weights (bias added)
- Measure the accuracy and other errors
- Make predictions on the test data set
- Write the predictions to csv file


### Feature Engineering

I've done the following feature engineering:
- Performed log of square of all features and appended these features to the original training vector. Got an RMSE of 4.19966 using batch gradient descent with alpha = 0.9, lambda = 0.009, 52000 iterations
- Performed square of all features and appended these features to the original training vector. Got an RMSE of 3.56075 using batch gradient descent with alpha = 0.9, lambda = 0.009, 52000 iterations
- Performed square of all features and appended these features to the original training vector. Got an RMSE of 3.51231 after removing outliers using batch gradient descent with alpha = 0.9, lambda = 0.009, 52000 iterations
- Performed the above two in combination using the same parameters to get RMSE of 3.63349
- Performed raised to the power 4 of all features and appended these features to the original training vector. Got an RMSE of 4.28459 using batch gradient descent with alpha = 0.9, lambda = 0.000009, 52000 iterations

### Observations

- Gradient descent with L<sub>p</sub> norm, for 3 different values of p in (1,2]
    - *p:* 1.2. Output in ./output_p1.csv
    - *p:* 1.5. Output in ./output_p2.csv
    - *p:* 1.8. Output in ./output_p3.csv
- Contrast difference between performance of linear regression L<sub>p</sub> norm and L<sub>2</sub> norm for these 3 different values. 
    - All the L<sub>p</sub> norm regression equally work good with a very minor difference between them
    - RMSE for various L<sub>p</sub> 
    - L<sub>1.2</sub>: 3.5441
    - L<sub>1.5</sub>: 3.5441
    - L<sub>1.8</sub>: 3.5441
- Batch gradient descent works faster than stochastic gradient descent for a fixed number of iterations
- The RMSE with least square solution using gradient descent with L<sub>2</sub> norm and regularization (RMSE: ) is slightly better than what we get with least squares solution matrix inversion (RMSE: 5.04965) (see the table below)
- Better results were obtained at 5-fold cross validation


### Functions implemented

All the functions are available in folder named *library*

- Data utilities (data.py)
    - ```read_data_from_csv()```
        - Utility for reading data from given csv file 
    - ```train_validate_split()```
    - ```write_data_to_csv()```
    - ```normalize_data()```
        - Function for normalizing the complete data 
    - ```feature_normalize()```
        - Function for normalizing a particular feature 
    - ```cross_validate_split()```
        - Function for k-fold cross validation 
- Linear Regression Class (linear_regression.py)
    - Least squares solution with L<sub>2</sub> regularization
    - Batch Gradient Descent with L<sub>p</sub> regularization
    - Stochastic Gradient Descent with L<sub>p</sub> regularization
- Feature Reduction (feature_selection.py)
    - PCA
- Error Metrics (metrics.py)
    - RMSE - Root Mean Square error
    - Mean absolute error
    - Median absolute error
- Score metrics (metrics.py)
    - R<sup>2</sup> score
    - Explained variance score


## Neural Network

### Running Instructions

**NOTE:** Please copy the library directory to the place from where the program is run.

```bash
$ python3 train_net.py
$ python3 test_net.py
```


### Steps of Implementation

- Read the dataset from train.csv into pandas Dataframe
- Impute the '?' missing values using the most frequent strategy
    - Replace the '?' with the most occurring value in the attribute column
- Encode the categorical values using one-hot encoding
    - Use one hot encoding for attribute value representation leading to
      increase in feature space from 14 to 105
- Apply Standard Scaling transformation to the modified data
    - Subtract the mean and divide by standard deviation for every feature
- Run the neural network with a specified configuration and learn the parameters
    - See the configuration above
    - Used momentum stochastic gradient descent
    - Used L<sub>2</sub> regularization
    - Used cross entropy loss function
- Predict the output for test dataset and write to file

### Implemented features

- One hot encoding for categorical attributes
- Numerical encoding for categorical attributes
- Most frequent strategy for imputing missing values in attribute column
- Activation functions: Tanh, Sigmoid, ReLU
- Cross entropy loss with one hot encoding of output
- Neural network with flexible layer structure
- Backpropagation with momentum stochastic gradient descent with L<sub>2</sub> regularization
- Neural network training in batches
- Feature normalization strategies: StandardScaler, MinMaxScaler
- Accuracy metrics: Precision, Recall, F<sub>beta</sub> score, Confusion matrix, Classification report
- Plotting scores and variance
