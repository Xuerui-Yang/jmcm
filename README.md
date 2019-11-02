# jmcm
[![Build Status](https://www.travis-ci.org/Xuerui-Yang/jmcm.svg?branch=master
)](https://www.travis-ci.org/Xuerui-Yang/jmcm)

## Description
jmcm is an open-source Python package for fitting the joint mean-covariance models for longitudinal data. 

It provides:
* function to estimate parameters for the mean, innovation variance, and generalised auto-regressive coefficient
* function to do the Wald hypothesis tests to check the significance of the parameters
* model selection procedures
* bootstrap method to plot the curves for the mean, innovation variance, and generalised auto-regressive coefficient

## Source code
https://github.com/Xuerui-Yang/jmcm

## Installation
```
pip install jmcm
```

## Usage
The following command computes the estimates of parameters for a joint mean-covariance model.
```python
from jmcm import JointModel
JM=JointModel(df, formula, poly_orders = (), optim_meth = 'default', model_select = False)
```

### Arguments
- **df**: The dataset of interest. It should be a 'pandas' DataFrame        
- **formula**: A formula showing the response, subject id, design matrices. It is a string following the rules defined in R, for example, 'y|id|t~x1+x2|x1+x3'.
   - On the left hand side of '~', there are 3 headers of the data, partitioned by '|':
   
         - y: The response vector for all the subjects
         - id: The ID number which identifying different subjects
         - t: The vector of time, which constructs the polynomials for modelling the means, innovation variances, and generalised auto regressive parameters.
    - On the right hand side of '~', there are two parts partitioned by '|':
    
          - x1+x2: '+' joints two headers, which are the covariates for the mean. There can be more headers and operators. Package 'patsy' is used to achieve that.
          - x1+x3: Similar to the left part, except that they are for the innovation variances.
- **poly_orders**: A tuple of length 3. If the format is correct, it specifies the polynomial orders of time for the mean, innovation variance, and generalised auto regressive parameters. Otherwise the model selection procedures would be used.
- **optim_meth**: The optimisation algorithm used. There are 2 options:
   1. 'default': use profile likelihood estimation to update the 3 types of parameters.
   2. 'BFGS': use BFGS method to update all 3 parameters together. 
If not specified, 'default' would be used
- **model_select**: True or False. To do model selection or not. If it is True, there are two situations according to poly_orders:
   1. poly_orderes is assigned with a tuple of length 3. Then a traverse under the given triple would be done to find the the triple with the smallest BIC values. And the model would be fitted based on the selected poly_orders
   2. poly_orders is not assigned or assigned in a incorrect format. Then the a profile based search would be done. And the model would be fitted based on the selected poly_orders.

The following commands print the values of MLEs, BICs, test statistics, p-values, and figures of curves. 
```python
JM.summary()
JM.wald_test()
JM.boot_curve(num_boot)
```

### Arguments
- **num_boot**: The number of bootstrap samples. Note that a large number may cost much time to run
