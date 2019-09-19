from math import *
import numpy as np
from numpy.linalg import *
import pandas as pd
from collections import Counter
import patsy


class ReadData():
    """
    Provide functions to read the data and change it so that it can be
    used to build the model.
    """

    def __init__(self, df, formula, poly_orders):
        """
        Build a super class for model_fit. The data would be preprocessed to be
        a standard form for analysis.

        Parameters:
        - df: The dataset of interest. It should be a 'pandas' DataFrame
        - formula: A formula showing the response, subject id, design
             matrices. It is a string following the rules defined in R,
             for example, 'y|id|t~x1+x2|x1+x3'.
             [*] On the left hand side of '~', there are 3 headers of the
             data, partitioned by '|':
             - y: The response vector for all the subjects
             - id: The ID number which identifying different subjects
             - t: The vector of time, which constructs the polynomials
                 for modelling the means, innovation variances, and
                 generalised auto regressive parameters.
             [*] On the right hand side of '~', there are two parts
             partitioned by '|':
             - x1+x2: '+' joints two headers, which are the covariates
                 for the mean. There can be more headers and operators.
                 Package 'patsy' is used to achieve that.
             - x1+x3: Similar to the left part, except that they are
                 for the innovation variances.
        - poly_orders: A tuple of length 3 or length 0. If the length is 3,
             it specifies the polynomial orders of time for the mean, innovation
             variance, and generalised auto regressive parameters. If the length
             is 0, then the model selection procedures might be used.
        """

        # Extract information from the formula
        y_header, id_header, t_header, mean_part, inno_part = self._read_formula(formula)

        # Sort the data by its ID numbers, begin at the smallest
        self.df = self._sort_by_id(df, id_header, t_header)

        # Compute the number of measurements in each subject vec_n,
        # and the number of subjects m
        vec_id = self.df.loc[:, id_header].values
        self.vec_n = self._get_num(vec_id)

        # Take out the the response vector and the vector of time
        self.vec_y = np.array(self.df.loc[:, y_header])
        self.vec_t = np.array(self.df.loc[:, t_header])

        # Build the design matrices X, Z, W
        self.mat_X,self.mat_Z,self.mat_W=self._get_design(mean_part, inno_part, poly_orders)


    def _read_formula(self,formula):
        """
        Take the matrices from the data, guiding by the formula with R
        style.
        """
        lhs, rhs = formula.split('~')
        y_header, id_header, t_header = lhs.split('|')
        mean_part, inno_part = rhs.split('|')
        return y_header, id_header, t_header, mean_part, inno_part

    def _sort_by_id(self, df, id_header, t_header):
        """
        Sort the data by its id number (1st) and time (2nd)
        """
        return df.sort_values(by=[id_header, t_header])

    def _get_num(self, vec_id):
        """
        Compute the number of measurements for each subject
        """
        # Package 'collection' is used to count the numbers
        n = [b for a, b in sorted(dict(Counter(vec_id)).items())]
        # Return the number of measruements and the number of subjects
        return np.array(n)

    def _build_dmatrix(self, formula, num_col):
        """
        Build the design matrix which contains polynomial terms of time
        and the other terms of covariates.
        """
        # Generate the polynomial part of the design matrix
        lhs = np.ones((len(self.vec_t), num_col))
        for i in range(1, num_col):
            lhs[:, i] = np.power(self.vec_t, i)
        rhs = np.array(patsy.dmatrix(formula + '-1', self.df))
        return np.hstack((lhs, rhs))

    def _build_mat_W(self, num_col):
        """
        Build the design matrix W which contains polynomial terms of time.
        All the Wi's are concatenated by row-wise.
        """
        # Initialise W with q+1 columns
        mat = np.empty((0, num_col))
        last=0
        # Search within subjects
        for i in range(len(self.vec_n)):
            # No rows for the subjects with only 1 measurement
            ni = self.vec_n[i]
            first=last
            last+=ni

            if ni != 1:
                # Get a sub-vector of time for the i-th subject
                ti = self.vec_t[first:last]
                # Generate rows for W
                for j in range(1, ni):
                    for k in range(j):
                        wijk = [(ti[j] - ti[k])**l for l in range(num_col)]
                        mat = np.vstack((mat, wijk))
        return mat

    def _get_design(self, mean_part, inno_part, poly_orders):
        """
        Get the design matrices X, Z, and W
        """
        if len(poly_orders) == 3:
            mat_X = self._build_dmatrix(mean_part, poly_orders[0] + 1)
            mat_Z = self._build_dmatrix(inno_part, poly_orders[1] + 1)
            mat_W = self._build_mat_W(poly_orders[2] + 1)
        else:
            mat_X = np.array(patsy.dmatrix(mean_part + '-1', self.df))
            mat_Z = np.array(patsy.dmatrix(inno_part + '-1', self.df))
            mat_W = np.empty((0, 0))
        return mat_X,mat_Z,mat_W


